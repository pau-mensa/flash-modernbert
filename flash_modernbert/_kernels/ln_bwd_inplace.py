"""M-dynamic CuteDSL LayerNorm backward (no β term, mmBERT-style).

M (total tokens) is a runtime grid dim, so one compiled artifact serves every
sequence length (no per-seqlen re-JIT). This is the variant wired into the
fused-tail encoder's LayerNorm.

Inputs: grad_ln_x [M,K] bf16 (grad wrt LN output), x [M,K] bf16 (saved input),
mean/rstd [M] fp32 (saved per-row stats), γ [K] bf16. Outputs grad_x [M,K] bf16
and grad_γ [K] bf16 (summed from per-CTA fp32 partials).

With `inplace=True`, grad_x is written over grad_ln_x's storage (per-thread
read-then-write at the same cell, race-free), saving an [M,K] tensor — the
bwd-peak win that unblocks large training batches.

Math (no-bias variant):
    z[i,k]      = (x[i,k] - μ[i]) · rstd[i]
    grad_z[i,k] = grad_ln_x[i,k] · γ[k]
    c1[i]       = mean_k(grad_z[i,k]) ;  c2[i] = mean_k(grad_z[i,k] · z[i,k])
    grad_x[i,k] = (grad_z[i,k] - c1[i] - c2[i]·z[i,k]) · rstd[i]
    grad_γ[k]   = Σ_i grad_ln_x[i,k] · z[i,k]

Layout: NUM_WARPS=8, one warp per row, each lane handles K/32 cols. Persistent
CTAs (grid = SM count) stride over row-bands; each writes its [K] fp32 grad_γ
partial, reduced in Python.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from flash_modernbert._kernels._compile_cache import current_cute_stream, get_compiled

WARP_SIZE = 32
NUM_WARPS = 8
NUM_THREADS = NUM_WARPS * WARP_SIZE  # 256
BM_BAND = NUM_WARPS  # one warp per row in the band


def _build_dyn(k: int, n_sm: int):
    assert k % WARP_SIZE == 0, f"K={k} must be a multiple of warp size {WARP_SIZE}"
    assert k % NUM_THREADS == 0, (
        f"K={k} must be a multiple of NUM_THREADS={NUM_THREADS} for the "
        f"final per-CTA cross-warp reduction"
    )
    elems_per_thread = k // WARP_SIZE
    cols_per_thread_final = k // NUM_THREADS
    inv_k = 1.0 / float(k)

    @cute.kernel
    def kernel(
        g_grad_in: cute.Tensor,  # [M, K] bf16  grad wrt LN output
        g_grad_x: cute.Tensor,  # [M, K] bf16  output (may alias g_grad_in)
        g_x: cute.Tensor,  # [M, K] bf16
        g_mean: cute.Tensor,  # [M]    fp32
        g_rstd: cute.Tensor,  # [M]    fp32
        g_gamma: cute.Tensor,  # [K]    bf16
        g_grad_gamma_partials: cute.Tensor,  # [n_sm, K] fp32
        m: Int32,
    ):
        cta_idx, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        tidx, _, _ = cute.arch.thread_idx()

        warp_id = tidx // Int32(WARP_SIZE)
        lane_id = tidx % Int32(WARP_SIZE)

        n_bands = (m + Int32(BM_BAND - 1)) // Int32(BM_BAND)

        grad_gamma_acc = cute.make_rmem_tensor((elems_per_thread,), Float32)
        for ki in cutlass.range_constexpr(elems_per_thread):
            grad_gamma_acc[ki] = Float32(0.0)

        # Dynamic SMEM (sized by the `smem=` launch arg below), NOT alloc_smem
        smem_part_ptr = cute.arch.get_dyn_smem(Float32, alignment=16)
        smem_part = cute.make_tensor(
            smem_part_ptr,
            cute.make_layout((NUM_WARPS, k), stride=(k, 1)),
        )

        band_iter = Int32(cta_idx)
        while band_iter < n_bands:
            m_row = band_iter * Int32(BM_BAND) + warp_id

            if m_row < m:
                mean_val = g_mean[m_row]
                rstd_val = g_rstd[m_row]

                grad_z_local = cute.make_rmem_tensor((elems_per_thread,), Float32)
                z_local = cute.make_rmem_tensor((elems_per_thread,), Float32)

                local_c1 = Float32(0.0)
                local_c2 = Float32(0.0)

                for ki in cutlass.range_constexpr(elems_per_thread):
                    col = lane_id + Int32(ki * WARP_SIZE)
                    gly = Float32(g_grad_in[m_row, col])
                    xv = Float32(g_x[m_row, col])
                    gv = Float32(g_gamma[col])
                    z = (xv - mean_val) * rstd_val
                    gz = gly * gv
                    grad_z_local[ki] = gz
                    z_local[ki] = z
                    local_c1 = local_c1 + gz
                    local_c2 = local_c2 + gz * z
                    grad_gamma_acc[ki] = grad_gamma_acc[ki] + gly * z

                for delta in (16, 8, 4, 2, 1):
                    local_c1 = local_c1 + cute.arch.shuffle_sync_bfly(
                        local_c1,
                        offset=delta,
                        mask=-1,
                        mask_and_clamp=WARP_SIZE - 1,
                    )
                    local_c2 = local_c2 + cute.arch.shuffle_sync_bfly(
                        local_c2,
                        offset=delta,
                        mask=-1,
                        mask_and_clamp=WARP_SIZE - 1,
                    )

                c1 = local_c1 * Float32(inv_k)
                c2 = local_c2 * Float32(inv_k)

                for ki in cutlass.range_constexpr(elems_per_thread):
                    col = lane_id + Int32(ki * WARP_SIZE)
                    gx = (grad_z_local[ki] - c1 - c2 * z_local[ki]) * rstd_val
                    g_grad_x[m_row, col] = g_grad_x.element_type(gx)

            band_iter = band_iter + grid_x

        for ki in cutlass.range_constexpr(elems_per_thread):
            col = lane_id + Int32(ki * WARP_SIZE)
            smem_part[warp_id, col] = grad_gamma_acc[ki]

        cute.arch.sync_threads()

        for ki in cutlass.range_constexpr(cols_per_thread_final):
            col = tidx + Int32(ki * NUM_THREADS)
            total = Float32(0.0)
            for w in cutlass.range_constexpr(NUM_WARPS):
                total = total + smem_part[w, col]
            g_grad_gamma_partials[Int32(cta_idx), col] = total

    @cute.jit
    def launch(
        GradIn: cute.Tensor,
        GradX: cute.Tensor,
        X: cute.Tensor,
        Mean: cute.Tensor,
        Rstd: cute.Tensor,
        Gamma: cute.Tensor,
        GradGammaPartials: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        m = cute.size(X, mode=[0])
        kernel(GradIn, GradX, X, Mean, Rstd, Gamma, GradGammaPartials, m).launch(
            grid=(n_sm, 1, 1),
            block=(NUM_THREADS, 1, 1),
            smem=(NUM_WARPS * k * 4 + 1024) & ~127,
            stream=stream,
        )

    return launch


_compiled_dyn: dict[tuple[int, int], object] = {}


def _get_jit_dyn(k: int, n_sm: int):
    key = (k, n_sm)
    if key not in _compiled_dyn:
        _compiled_dyn[key] = _build_dyn(k, n_sm)
    return _compiled_dyn[key]


def ln_bwd(
    grad_ln_x: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    gamma: torch.Tensor,
    *,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LayerNorm backward (no β). See the module docstring for the math and layout.

    grad_ln_x / x: [..., K] bf16; mean, rstd: flat [M] fp32; gamma: [K] bf16. With
    `inplace=True`, grad_x is written over grad_ln_x's storage (caller must own it).
    Returns `(grad_x, grad_gamma)`; grad_gamma is freshly allocated bf16.
    """
    assert grad_ln_x.is_cuda and x.is_cuda and mean.is_cuda and rstd.is_cuda
    assert gamma.is_cuda
    assert grad_ln_x.dtype == x.dtype == gamma.dtype == torch.bfloat16
    assert mean.dtype == torch.float32 and rstd.dtype == torch.float32

    K = gamma.shape[0]
    flat_grad = grad_ln_x.reshape(-1, K)
    flat_x = x.reshape(-1, K)
    assert flat_grad.is_contiguous() and flat_x.is_contiguous()
    assert flat_grad.shape == flat_x.shape
    M = flat_grad.shape[0]
    assert mean.shape == (M,) and rstd.shape == (M,)

    flat_out = flat_grad if inplace else torch.empty_like(flat_grad)

    n_sm = torch.cuda.get_device_properties(grad_ln_x.device).multi_processor_count
    grad_gamma_partials = torch.empty(
        (n_sm, K), dtype=torch.float32, device=grad_ln_x.device
    )

    launcher = _get_jit_dyn(K, n_sm)
    args = (
        from_dlpack(flat_grad, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(flat_out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(flat_x, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(mean, assumed_align=16).mark_layout_dynamic(),
        from_dlpack(rstd, assumed_align=16).mark_layout_dynamic(),
        from_dlpack(gamma, assumed_align=16),
        from_dlpack(grad_gamma_partials, assumed_align=16),
        current_cute_stream(),
    )
    compiled = get_compiled(launcher, args, key=("ln_bwd_dyn", K, n_sm))
    compiled(*args)

    grad_gamma = grad_gamma_partials.sum(dim=0).to(torch.bfloat16)
    grad_x = grad_ln_x if inplace else flat_out.view_as(x)
    return grad_x, grad_gamma
