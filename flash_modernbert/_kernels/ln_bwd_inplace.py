"""In-place CuteDSL LayerNorm backward (no β term, mmBERT-style).

Companion to the ln_linear_ws_persistent FORWARD fusion. Inputs:

    grad_ln_x [M, K]  bf16  — gradient w.r.t. ln_x (= the GEMM input);
                              also the in-place output buffer (becomes grad_x)
    x         [M, K]  bf16  — saved fwd input
    mean      [M]     fp32  — saved per-row mean
    rstd      [M]     fp32  — saved per-row rstd
    γ         [K]     bf16  — LN scale

Outputs:

    grad_x        [M, K]  bf16  — written IN-PLACE over grad_ln_x's storage
    grad_γ        [K]     bf16  — produced via a per-CTA partial buffer +
                                  Python sum (this kernel writes the partials)

The point of the in-place output is the bwd-peak win: in the standard
chain (`grad_y @ W → grad_ln_x` then `aten.native_layer_norm_backward`),
both grad_ln_x AND grad_x are simultaneously alive during the LN-bwd
call — that's an extra ~700 MB at training shape (M ≈ 458k, K=768) that
costs us B=3584-GC on a 32 GB card. By writing grad_x into grad_ln_x's
buffer (per-thread reads-then-writes the same cell, no aliasing race),
peak drops to one [M, K] tensor.

Math (standard, no-bias variant):
    z[i, k]      = (x[i, k] - μ[i]) · rstd[i]
    grad_z[i, k] = grad_ln_x[i, k] · γ[k]
    c1[i]        = mean_k(grad_z[i, k])
    c2[i]        = mean_k(grad_z[i, k] · z[i, k])
    grad_x[i, k] = (grad_z[i, k] - c1[i] - c2[i] · z[i, k]) · rstd[i]
    grad_γ[k]    = Σ_i grad_ln_x[i, k] · z[i, k]

Layout:
    NUM_WARPS = 8, BM_BAND = 8 rows per band (one warp per row).
    Each lane handles K/32 = 24 cols (for K=768).
    Persistent CTA: launch n_sm CTAs, loop over row-bands.
    grad_γ partial: each CTA writes its [K] fp32 contribution to
        grad_gamma_partials[cta_idx, :] after the persistent loop;
        Python reduces partials → grad_γ in bf16.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.runtime import from_dlpack

from flash_modernbert._kernels._compile_cache import current_cute_stream, get_compiled

WARP_SIZE = 32
NUM_WARPS = 8
NUM_THREADS = NUM_WARPS * WARP_SIZE   # 256
BM_BAND = NUM_WARPS                    # one warp per row in the band


def _build_for_mk(m: int, k: int, n_sm: int):
    assert k % WARP_SIZE == 0, f"K={k} must be a multiple of warp size {WARP_SIZE}"
    assert k % NUM_THREADS == 0, (
        f"K={k} must be a multiple of NUM_THREADS={NUM_THREADS} for the "
        f"final per-CTA cross-warp reduction"
    )
    elems_per_thread = k // WARP_SIZE
    cols_per_thread_final = k // NUM_THREADS
    n_bands = (m + BM_BAND - 1) // BM_BAND
    inv_k = 1.0 / float(k)

    @cute.kernel
    def kernel(
        g_grad_ln_x: cute.Tensor,             # [M, K] bf16  (in-place output)
        g_x: cute.Tensor,                     # [M, K] bf16
        g_mean: cute.Tensor,                  # [M]    fp32
        g_rstd: cute.Tensor,                  # [M]    fp32
        g_gamma: cute.Tensor,                 # [K]    bf16
        g_grad_gamma_partials: cute.Tensor,   # [grid_x, K] fp32
    ):
        cta_idx, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        tidx, _, _ = cute.arch.thread_idx()

        warp_id = tidx // Int32(WARP_SIZE)    # 0..7
        lane_id = tidx % Int32(WARP_SIZE)     # 0..31

        # Per-thread cumulative grad_γ accumulator (over all rows this thread
        # touches via the persistent loop). 24 fp32 per thread at K=768.
        grad_gamma_acc = cute.make_rmem_tensor((elems_per_thread,), Float32)
        for ki in cutlass.range_constexpr(elems_per_thread):
            grad_gamma_acc[ki] = Float32(0.0)

        # SMEM for cross-warp grad_γ reduction at the end.
        smem_part_ptr = cute.arch.alloc_smem(
            Float32, NUM_WARPS * k, alignment=16
        )
        smem_part = cute.make_tensor(
            smem_part_ptr,
            cute.make_layout((NUM_WARPS, k), stride=(k, 1)),
        )

        # ---- Persistent loop over row-bands ----
        band_iter = Int32(cta_idx)
        while band_iter < Int32(n_bands):
            m_row = band_iter * Int32(BM_BAND) + warp_id

            if m_row < Int32(m):
                mean_val = g_mean[m_row]
                rstd_val = g_rstd[m_row]

                grad_z_local = cute.make_rmem_tensor(
                    (elems_per_thread,), Float32
                )
                z_local = cute.make_rmem_tensor(
                    (elems_per_thread,), Float32
                )

                local_c1 = Float32(0.0)
                local_c2 = Float32(0.0)

                # Pass 1: read grad_ln_x and x, compute z and grad_z; reduce
                # c1, c2 partials; accumulate grad_γ.
                for ki in cutlass.range_constexpr(elems_per_thread):
                    col = lane_id + Int32(ki * WARP_SIZE)
                    gly = Float32(g_grad_ln_x[m_row, col])
                    xv = Float32(g_x[m_row, col])
                    gv = Float32(g_gamma[col])
                    z = (xv - mean_val) * rstd_val
                    gz = gly * gv
                    grad_z_local[ki] = gz
                    z_local[ki] = z
                    local_c1 = local_c1 + gz
                    local_c2 = local_c2 + gz * z
                    grad_gamma_acc[ki] = grad_gamma_acc[ki] + gly * z

                # Warp-reduce c1, c2 across 32 lanes.
                for delta in (16, 8, 4, 2, 1):
                    local_c1 = local_c1 + cute.arch.shuffle_sync_bfly(
                        local_c1, offset=delta, mask=-1,
                        mask_and_clamp=WARP_SIZE - 1,
                    )
                    local_c2 = local_c2 + cute.arch.shuffle_sync_bfly(
                        local_c2, offset=delta, mask=-1,
                        mask_and_clamp=WARP_SIZE - 1,
                    )

                c1 = local_c1 * Float32(inv_k)
                c2 = local_c2 * Float32(inv_k)

                # Pass 2: apply correction; write grad_x in-place over
                # grad_ln_x's cells (per-thread read-then-write at same col,
                # race-free).
                for ki in cutlass.range_constexpr(elems_per_thread):
                    col = lane_id + Int32(ki * WARP_SIZE)
                    gx = (grad_z_local[ki] - c1 - c2 * z_local[ki]) * rstd_val
                    g_grad_ln_x[m_row, col] = g_grad_ln_x.element_type(gx)

            band_iter = band_iter + grid_x

        # ---- Cross-warp reduce grad_γ → gmem partial[cta_idx, :] ----
        for ki in cutlass.range_constexpr(elems_per_thread):
            col = lane_id + Int32(ki * WARP_SIZE)
            smem_part[warp_id, col] = grad_gamma_acc[ki]

        cute.arch.sync_threads()

        # 256 threads cooperate, each handles K/256 = 3 cols.
        for ki in cutlass.range_constexpr(cols_per_thread_final):
            col = tidx + Int32(ki * NUM_THREADS)
            total = Float32(0.0)
            for w in cutlass.range_constexpr(NUM_WARPS):
                total = total + smem_part[w, col]
            g_grad_gamma_partials[Int32(cta_idx), col] = total

    @cute.jit
    def launch(
        GradLnX: cute.Tensor,
        X: cute.Tensor,
        Mean: cute.Tensor,
        Rstd: cute.Tensor,
        Gamma: cute.Tensor,
        GradGammaPartials: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        grid_x = min(n_sm, n_bands)
        kernel(
            GradLnX, X, Mean, Rstd, Gamma, GradGammaPartials
        ).launch(
            grid=(grid_x, 1, 1),
            block=(NUM_THREADS, 1, 1),
            smem=(NUM_WARPS * k * 4 + 1024) & ~127,
            stream=stream,
        )

    return launch


_compiled_for: dict[tuple[int, int, int], object] = {}


def _get_jit(m: int, k: int, n_sm: int):
    key = (m, k, n_sm)
    if key not in _compiled_for:
        _compiled_for[key] = _build_for_mk(m, k, n_sm)
    return _compiled_for[key]


def ln_bwd_inplace(
    grad_ln_x: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-place LN-bwd. ``grad_ln_x`` is OVERWRITTEN with grad_x.

    Returns (grad_x, grad_gamma) where grad_x is the same tensor object as the
    input grad_ln_x (now containing grad_x values) and grad_gamma is freshly
    allocated bf16.
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

    n_sm = torch.cuda.get_device_properties(grad_ln_x.device).multi_processor_count
    n_bands = (M + BM_BAND - 1) // BM_BAND
    grid_x = min(n_sm, n_bands)

    grad_gamma_partials = torch.empty(
        (grid_x, K), dtype=torch.float32, device=grad_ln_x.device
    )

    launcher = _get_jit(M, K, n_sm)
    args = (
        from_dlpack(flat_grad, assumed_align=16),
        from_dlpack(flat_x, assumed_align=16),
        from_dlpack(mean, assumed_align=16),
        from_dlpack(rstd, assumed_align=16),
        from_dlpack(gamma, assumed_align=16),
        from_dlpack(grad_gamma_partials, assumed_align=16),
        current_cute_stream(),
    )
    compiled = get_compiled(
        launcher, args, key=("ln_bwd_inplace", M, K, n_sm)
    )
    compiled(*args)

    grad_gamma = grad_gamma_partials.sum(dim=0).to(torch.bfloat16)
    return grad_ln_x, grad_gamma


# ---------------------------------------------------------------------------
# M-dynamic variant — re-JIT-free across the token count
# ---------------------------------------------------------------------------
#
# `ln_bwd_inplace` above bakes M into the compiled kernel (cache key includes M;
# n_bands is a build-time constexpr; tensors are NOT mark_layout_dynamic), so a
# new sequence length pays a fresh ~350 ms compile (measured re-JIT spike). That
# is fine for the fixed-shape training path it was written for, but it would
# reintroduce torch.compile's per-seqlen recompile churn in the *fused-tail*
# encoder (plan Gate 2). This variant makes M a runtime grid dim:
#
#   - grid is the (constant) SM count; the persistent loop strides over all
#     row-bands, with n_bands computed at RUNTIME from M (passed as a scalar).
#   - the [M, K] / [M] tensors are `mark_layout_dynamic`, so M is not baked.
#   - the compile-cache key is (K, n_sm) only — one artifact serves every S.
#   - grad_gamma partials are [n_sm, K] (fixed); idle CTAs at small M write
#     zeros (correct, cheap).
#
# Also supports a separate output buffer (grad_in != grad_x) so a standalone
# LayerNorm can leave the autograd engine's incoming grad untouched; pass the
# same tensor for both to get the in-place behaviour of `ln_bwd_inplace`.


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
        g_grad_in: cute.Tensor,               # [M, K] bf16  grad wrt LN output
        g_grad_x: cute.Tensor,                # [M, K] bf16  output (may alias g_grad_in)
        g_x: cute.Tensor,                     # [M, K] bf16
        g_mean: cute.Tensor,                  # [M]    fp32
        g_rstd: cute.Tensor,                  # [M]    fp32
        g_gamma: cute.Tensor,                 # [K]    bf16
        g_grad_gamma_partials: cute.Tensor,   # [n_sm, K] fp32
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

        smem_part_ptr = cute.arch.alloc_smem(Float32, NUM_WARPS * k, alignment=16)
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
                        local_c1, offset=delta, mask=-1, mask_and_clamp=WARP_SIZE - 1,
                    )
                    local_c2 = local_c2 + cute.arch.shuffle_sync_bfly(
                        local_c2, offset=delta, mask=-1, mask_and_clamp=WARP_SIZE - 1,
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
        kernel(
            GradIn, GradX, X, Mean, Rstd, Gamma, GradGammaPartials, m
        ).launch(
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
    """M-dynamic LayerNorm backward (no β), re-JIT-free across the token count.

    Same math as :func:`ln_bwd_inplace` but M (= total tokens) is a runtime grid
    dim, so one compiled artifact serves every sequence length (plan Gate 2). This
    is the variant wired into the fused-tail encoder's standalone LayerNorm.

    Args:
        grad_ln_x: [..., K] bf16 — gradient wrt the LN output.
        x:         [..., K] bf16 — saved forward input.
        mean, rstd: flat [M] fp32 — saved per-row stats.
        gamma:     [K] bf16.
        inplace:   if True, write grad_x over ``grad_ln_x``'s storage (saves the
                   [M, K] allocation; caller must own ``grad_ln_x``). Default
                   False allocates a fresh grad_x and leaves the input intact.

    Returns ``(grad_x, grad_gamma)``; grad_gamma is freshly allocated bf16.
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
