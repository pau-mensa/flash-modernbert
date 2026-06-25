"""GeGLU activation forward kernel in CuteDSL.

Drop-in for the elementwise step inside `op_geglu_mlp` in `reference.py`:

    proj  = x @ Wi^T            # done in cuBLAS, shape [..., 2*I]
    a, gate = proj.chunk(2, -1) # halves of last dim
    activated = gelu_exact(a) * gate          # ← THIS KERNEL
    out   = activated @ Wo^T    # done in cuBLAS

Exact GeLU (`approximate="none"`), matching HF mmBERT's `hidden_activation="gelu"`:
`gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))`. Compute is fp32; I/O dtype is the
caller's (bf16 in mmBERT). One block per row of `[N, 2*I]`, threads cooperate along
`I`; specialized by `I` via a closure factory (same pattern as the other kernels).
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack

from flash_modernbert._kernels._compile_cache import current_cute_stream, get_compiled

THREADS_PER_BLOCK = 128
_INV_SQRT2 = float(1.0 / math.sqrt(2.0))


VEC_FWD = 8  # 128-bit vectorized loads/stores for bf16


def _pick_threads_fwd(intermediate: int, vec: int) -> int:
    """Smallest warp-multiple covering one row in a few vec chunks, capped at 256."""
    need_vecs = (intermediate + vec - 1) // vec
    t = ((need_vecs + 31) // 32) * 32
    return max(32, min(t, 256))


def _build_for_i(intermediate: int):
    """Specialize the GeGLU forward for a fixed `I`: vectorized (128-bit
    `autovec_copy`) when `I` is a multiple of `VEC_FWD`, else scalar. It's
    memory-latency-bound — the exact-gelu `erf` is fully hidden, so there's no point
    approximating the activation."""
    assert intermediate > 0
    if intermediate % VEC_FWD == 0:
        return _build_vec_for_i(intermediate, _pick_threads_fwd(intermediate, VEC_FWD), VEC_FWD)
    return _build_scalar_for_i(intermediate, THREADS_PER_BLOCK)


def _build_vec_for_i(intermediate: int, threads: int, vec: int):
    cols_per_iter = threads * vec
    n_chunks = (intermediate + cols_per_iter - 1) // cols_per_iter

    @cute.kernel
    def kernel(gproj: cute.Tensor, gout: cute.Tensor):
        # gproj: [N, 2*I] (a-half | gate-half). gout: [N, I] = gelu(a) * gate.
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        dt = gproj.element_type
        row2 = bidx * (2 * intermediate)
        rowo = bidx * intermediate
        for c in cutlass.range_constexpr(n_chunks):
            base = c * cols_per_iter + tidx * vec
            if base < intermediate:
                a_v = cute.make_rmem_tensor((vec,), dt)
                g_v = cute.make_rmem_tensor((vec,), dt)
                o_v = cute.make_rmem_tensor((vec,), dt)
                cute.autovec_copy(
                    cute.make_tensor(gproj.iterator + row2 + base, cute.make_layout(vec)), a_v)
                cute.autovec_copy(
                    cute.make_tensor(gproj.iterator + row2 + intermediate + base, cute.make_layout(vec)), g_v)
                for j in cutlass.range_constexpr(vec):
                    a = a_v[j].to(Float32)
                    gate = g_v[j].to(Float32)
                    gelu_a = a * Float32(0.5) * (
                        Float32(1.0) + cute.math.erf(a * Float32(_INV_SQRT2)))
                    o_v[j] = o_v.element_type(gelu_a * gate)
                cute.autovec_copy(
                    o_v, cute.make_tensor(gout.iterator + rowo + base, cute.make_layout(vec)))

    @cute.jit
    def launch(proj: cute.Tensor, out: cute.Tensor, stream: cuda_driver.CUstream):
        n_rows = cute.size(proj, mode=[0])
        kernel(proj, out).launch(grid=(n_rows, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


def _build_scalar_for_i(intermediate: int, threads: int):
    """Scalar per-element fallback (for I not divisible by VEC_FWD)."""
    elems_per_thread = (intermediate + threads - 1) // threads

    @cute.kernel
    def kernel(gproj: cute.Tensor, gout: cute.Tensor):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        for k in cutlass.range_constexpr(elems_per_thread):
            col = tidx + k * threads
            if col < intermediate:
                a = gproj[bidx, col].to(Float32)
                gate = gproj[bidx, col + intermediate].to(Float32)
                gelu_a = a * Float32(0.5) * (
                    Float32(1.0) + cute.math.erf(a * Float32(_INV_SQRT2)))
                gout[bidx, col] = gout.element_type(gelu_a * gate)

    @cute.jit
    def launch(proj: cute.Tensor, out: cute.Tensor, stream: cuda_driver.CUstream):
        n_rows = cute.size(proj, mode=[0])
        kernel(proj, out).launch(grid=(n_rows, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


_compiled_for_i: dict[int, object] = {}


def _get_jit(intermediate: int):
    if intermediate not in _compiled_for_i:
        _compiled_for_i[intermediate] = _build_for_i(intermediate)
    return _compiled_for_i[intermediate]


# ---------------------------------------------------------------------------
# Backward — one fused kernel (replaces a 3-launch gelu_backward + multiply + cat),
# recomputing gelu(a)/gelu'(a) once per element:
#   da[i,j]    = dy[i,j] * gate[i,j] * gelu'(a[i,j])
#   dgate[i,j] = dy[i,j] * gelu(a[i,j])
# concatenated into grad_proj [N, 2*I]. No reductions.
# ---------------------------------------------------------------------------

_INV_SQRT_2PI = float(1.0 / math.sqrt(2.0 * math.pi))


def _build_bwd_for_i(intermediate: int):
    """Specialize the backward for a fixed `I`. Scalar, *not* vectorized: unlike the
    forward (where `autovec_copy` won 1.41×), vectorizing the backward measured slower
    on B200 (~1.22× regression) — it streams 5 vec buffers vs the forward's 3, and the
    extra register pressure costs more occupancy than the wider transactions buy."""
    assert intermediate > 0
    threads = THREADS_PER_BLOCK
    elems_per_thread = (intermediate + threads - 1) // threads

    @cute.kernel
    def kernel(
        gproj: cute.Tensor,      # [N, 2*I]
        ggrad_out: cute.Tensor,  # [N, I]
        ggrad_proj: cute.Tensor, # [N, 2*I]
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        for k in range(elems_per_thread):
            col = tidx + k * threads
            if col < intermediate:
                a = gproj[bidx, col].to(Float32)
                gate = gproj[bidx, col + intermediate].to(Float32)
                dy = ggrad_out[bidx, col].to(Float32)

                erf_term = cute.math.erf(a * Float32(_INV_SQRT2))
                half_one_plus_erf = Float32(0.5) * (Float32(1.0) + erf_term)
                gelu_a = a * half_one_plus_erf
                # gelu'(a) = 0.5 * (1 + erf(a/√2)) + a * exp(-a²/2) / √(2π)
                gauss = cute.math.exp(Float32(-0.5) * a * a) * Float32(_INV_SQRT_2PI)
                gelu_grad_a = half_one_plus_erf + a * gauss

                da = dy * gate * gelu_grad_a
                dgate = dy * gelu_a

                ggrad_proj[bidx, col] = ggrad_proj.element_type(da)
                ggrad_proj[bidx, col + intermediate] = ggrad_proj.element_type(dgate)

    @cute.jit
    def launch(
        proj: cute.Tensor,
        grad_out: cute.Tensor,
        grad_proj: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        n_rows = cute.size(proj, mode=[0])
        kernel(proj, grad_out, grad_proj).launch(
            grid=(n_rows, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch


_compiled_bwd_for_i: dict[int, object] = {}


def _get_jit_bwd(intermediate: int):
    if intermediate not in _compiled_bwd_for_i:
        _compiled_bwd_for_i[intermediate] = _build_bwd_for_i(intermediate)
    return _compiled_bwd_for_i[intermediate]


def geglu_bwd(
    proj: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    grad_proj_dtype: torch.dtype | None = None,
    inplace: bool = False,
) -> torch.Tensor:
    """GeGLU backward.

    `proj`: forward input, [..., 2*I]. Saved from forward (any floating dtype).
    `grad_out`: gradient w.r.t. the GeGLU output, [..., I], same dtype as proj
                (caller is responsible for any prior cast).
    `grad_proj_dtype`: dtype of the returned grad_proj. Defaults to proj.dtype.

    Returns grad_proj of `proj.shape`, dtype `grad_proj_dtype`.
    """
    assert proj.is_cuda and grad_out.is_cuda
    assert proj.shape[-1] % 2 == 0
    assert proj.shape[:-1] == grad_out.shape[:-1]
    intermediate = proj.shape[-1] // 2
    assert grad_out.shape[-1] == intermediate
    assert proj.dtype == grad_out.dtype, "proj and grad_out must share dtype"
    if grad_proj_dtype is None:
        grad_proj_dtype = proj.dtype

    proj_d = proj.detach()
    grad_d = grad_out.detach()
    flat_proj = proj_d.reshape(-1, 2 * intermediate).contiguous()
    flat_grad = grad_d.reshape(-1, intermediate).contiguous()
    if inplace and grad_proj_dtype == proj.dtype:
        # Per-element kernel reads (col, col+I) and writes the same positions, so
        # aliasing proj and grad_proj is safe — saves the (B*S, 2I) allocation.
        grad_proj = flat_proj
    else:
        grad_proj = torch.empty(
            flat_proj.shape, dtype=grad_proj_dtype, device=flat_proj.device
        )
    launcher = _get_jit_bwd(intermediate)
    args = (
        from_dlpack(flat_proj, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(flat_grad, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(grad_proj, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        current_cute_stream(),
    )
    compiled = get_compiled(
        launcher, args, key=("bwd", proj.dtype, grad_proj_dtype)
    )
    compiled(*args)
    return grad_proj.view(proj.shape)


def geglu(proj: torch.Tensor, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """GeGLU activation step: takes the concatenated `[input ; gate]` projection
    and returns `gelu(input) * gate`.

    `proj`: [..., 2*I] floating dtype (typically the output of `x @ Wi^T`).
    `out_dtype`: dtype of the result. Defaults to `proj.dtype`.

    Returns: [..., I] of `out_dtype`.
    """
    assert proj.is_cuda and proj.is_floating_point()
    assert proj.shape[-1] % 2 == 0, "last dim must be even (2*I)"
    intermediate = proj.shape[-1] // 2
    if out_dtype is None:
        out_dtype = proj.dtype

    flat = proj.reshape(-1, 2 * intermediate).contiguous()
    out = torch.empty(flat.shape[0], intermediate, dtype=out_dtype, device=flat.device)
    launcher = _get_jit(intermediate)
    args = (
        from_dlpack(flat, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        current_cute_stream(),
    )
    compiled = get_compiled(launcher, args, key=(proj.dtype, out_dtype))
    compiled(*args)
    return out.view(*proj.shape[:-1], intermediate)
