"""LayerNorm forward kernel in CuteDSL (no bias term — mmBERT uses bias=False).

    y = (x - mean(x, -1)) / sqrt(var(x, -1) + eps) * weight

One block per row of the flattened `[N, H]`, one warp (32 threads); each thread
owns H/32 elements and warp shuffles handle the reduction. Accumulates in fp32
even for bf16 I/O (matching `F.layer_norm`). H is baked in via a Python closure so
the per-thread loops unroll; one compiled kernel is cached per H.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack

from flash_modernbert._kernels._compile_cache import current_cute_stream, get_compiled

WARP_SIZE = 32


def _build_for_h(h: int):
    """A `@cute.jit` launcher specialized to a given H.

    Specializing by closure keeps `H`/`elems_per_thread` as Python ints inside the
    kernel (CuteDSL's `Constexpr` lowers to ArithValue, not a Python int, so
    `range(h // WARP_SIZE)` would fail).
    """
    assert h % WARP_SIZE == 0, f"H={h} must be a multiple of warp size {WARP_SIZE}"
    elems_per_thread = h // WARP_SIZE
    h_f32 = float(h)

    @cute.kernel
    def kernel(gx: cute.Tensor, gw: cute.Tensor, gy: cute.Tensor, eps: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # ---- Pass 1: sums (fp32 accumulation) ----
        local_sum = Float32(0.0)
        local_sumsq = Float32(0.0)
        for k in range(elems_per_thread):
            col = tidx + k * WARP_SIZE
            v = gx[bidx, col].to(Float32)
            local_sum = local_sum + v
            local_sumsq = local_sumsq + v * v

        # Warp-wide butterfly reduction across all 32 lanes.
        for delta in (16, 8, 4, 2, 1):
            local_sum = local_sum + cute.arch.shuffle_sync_bfly(
                local_sum, offset=delta, mask=-1, mask_and_clamp=WARP_SIZE - 1
            )
            local_sumsq = local_sumsq + cute.arch.shuffle_sync_bfly(
                local_sumsq, offset=delta, mask=-1, mask_and_clamp=WARP_SIZE - 1
            )

        h_recip = Float32(1.0 / h_f32)
        mean = local_sum * h_recip
        var = local_sumsq * h_recip - mean * mean
        inv_std = cute.math.rsqrt(var + eps)

        # ---- Pass 2: normalize, scale, write back ----
        for k in range(elems_per_thread):
            col = tidx + k * WARP_SIZE
            v = gx[bidx, col].to(Float32)
            w = gw[col].to(Float32)
            y = (v - mean) * inv_std * w
            gy[bidx, col] = gy.element_type(y)

    @cute.jit
    def launch(
        x: cute.Tensor,
        weight: cute.Tensor,
        out: cute.Tensor,
        eps: Float32,
        stream: cuda_driver.CUstream,
    ):
        n_rows = cute.size(x, mode=[0])
        kernel(x, weight, out, eps).launch(
            grid=(n_rows, 1, 1),
            block=(WARP_SIZE, 1, 1),
            stream=stream,
        )

    return launch


def _build_with_stats_for_h(h: int):
    """Like `_build_for_h` but also writes flat per-row mean/rstd (for backward)."""
    assert h % WARP_SIZE == 0, f"H={h} must be a multiple of warp size {WARP_SIZE}"
    elems_per_thread = h // WARP_SIZE
    h_f32 = float(h)

    @cute.kernel
    def kernel(
        gx: cute.Tensor,
        gw: cute.Tensor,
        gy: cute.Tensor,
        gmean: cute.Tensor,
        grstd: cute.Tensor,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        local_sum = Float32(0.0)
        local_sumsq = Float32(0.0)
        for k in range(elems_per_thread):
            col = tidx + k * WARP_SIZE
            v = gx[bidx, col].to(Float32)
            local_sum = local_sum + v
            local_sumsq = local_sumsq + v * v

        for delta in (16, 8, 4, 2, 1):
            local_sum = local_sum + cute.arch.shuffle_sync_bfly(
                local_sum, offset=delta, mask=-1, mask_and_clamp=WARP_SIZE - 1
            )
            local_sumsq = local_sumsq + cute.arch.shuffle_sync_bfly(
                local_sumsq, offset=delta, mask=-1, mask_and_clamp=WARP_SIZE - 1
            )

        h_recip = Float32(1.0 / h_f32)
        mean = local_sum * h_recip
        var = local_sumsq * h_recip - mean * mean
        inv_std = cute.math.rsqrt(var + eps)

        if tidx == 0:
            gmean[bidx] = mean
            grstd[bidx] = inv_std

        for k in range(elems_per_thread):
            col = tidx + k * WARP_SIZE
            v = gx[bidx, col].to(Float32)
            w = gw[col].to(Float32)
            y = (v - mean) * inv_std * w
            gy[bidx, col] = gy.element_type(y)

    @cute.jit
    def launch(
        x: cute.Tensor,
        weight: cute.Tensor,
        out: cute.Tensor,
        mean: cute.Tensor,
        rstd: cute.Tensor,
        eps: Float32,
        stream: cuda_driver.CUstream,
    ):
        n_rows = cute.size(x, mode=[0])
        kernel(x, weight, out, mean, rstd, eps).launch(
            grid=(n_rows, 1, 1),
            block=(WARP_SIZE, 1, 1),
            stream=stream,
        )

    return launch


_compiled_for_h: dict[int, object] = {}
_compiled_with_stats_for_h: dict[int, object] = {}


def _get_jit(h: int):
    if h not in _compiled_for_h:
        _compiled_for_h[h] = _build_for_h(h)
    return _compiled_for_h[h]


def _get_jit_with_stats(h: int):
    if h not in _compiled_with_stats_for_h:
        _compiled_with_stats_for_h[h] = _build_with_stats_for_h(h)
    return _compiled_with_stats_for_h[h]


def layer_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm forward. `x`: [..., H], `weight`: [H]; returns x's shape/dtype.
    H must be a multiple of 32 (warp size)."""
    assert x.is_cuda and weight.is_cuda, "layer_norm kernel requires CUDA tensors"
    assert weight.ndim == 1, "weight must be 1D"
    h = weight.shape[0]
    assert x.shape[-1] == h, f"last dim of x ({x.shape[-1]}) must equal H ({h})"
    assert x.dtype == weight.dtype, "x and weight must have same dtype"

    flat = x.reshape(-1, h).contiguous()
    out = torch.empty_like(flat)
    launcher = _get_jit(h)
    args = (
        from_dlpack(flat, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(weight, assumed_align=16),
        from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        Float32(eps),
        current_cute_stream(),
    )
    # n_rows is a runtime grid dim, so it stays out of the cache key — one artifact serves all M.
    compiled = get_compiled(launcher, args, key=(x.dtype,))
    compiled(*args)
    return out.view(x.shape)


def layer_norm_with_stats(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm forward plus flat fp32 mean/rstd vectors for backward."""
    assert x.is_cuda and weight.is_cuda, "layer_norm kernel requires CUDA tensors"
    assert weight.ndim == 1, "weight must be 1D"
    h = weight.shape[0]
    assert x.shape[-1] == h, f"last dim of x ({x.shape[-1]}) must equal H ({h})"
    assert x.dtype == weight.dtype, "x and weight must have same dtype"

    flat = x.reshape(-1, h).contiguous()
    out = torch.empty_like(flat)
    mean = torch.empty(flat.shape[0], dtype=torch.float32, device=x.device)
    rstd = torch.empty(flat.shape[0], dtype=torch.float32, device=x.device)

    launcher = _get_jit_with_stats(h)
    args = (
        from_dlpack(flat, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(weight, assumed_align=16),
        from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(mean, assumed_align=16).mark_layout_dynamic(),
        from_dlpack(rstd, assumed_align=16).mark_layout_dynamic(),
        Float32(eps),
        current_cute_stream(),
    )
    compiled = get_compiled(launcher, args, key=("stats", x.dtype))
    compiled(*args)
    return out.view(x.shape), mean, rstd
