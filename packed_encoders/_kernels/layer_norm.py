"""LayerNorm forward kernel in CuteDSL (no bias term — mmBERT uses bias=False).

    y = (x - mean(x, -1)) / sqrt(var(x, -1) + eps) * weight

One block per row of the flattened `[N, H]`. Multiple warps cooperate on the
same row: each warp accumulates a partial sum/sumsq over its slice via shuffle,
then lane 0 writes to shared memory. After a barrier, every thread reads and
reduces the per-warp partials (N_WARPS is small enough for scalar adds) to get
the global mean/variance, then normalizes its slice.

Accumulates in fp32 even for bf16 I/O (matching `F.layer_norm`). H is baked in
via a Python closure so the per-thread loops unroll; one compiled kernel is
cached per H.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from packed_encoders._kernels._compile_cache import current_cute_stream, get_compiled

WARP_SIZE = 32


def _pick_warps(h: int) -> int:
    """Choose the number of warps so that H divides evenly across all threads.
    Prefer 8 warps (256 threads) when possible, then 4, then 2, then 1."""
    for nw in (8, 4, 2, 1):
        if h % (nw * WARP_SIZE) == 0:
            return nw
    return 1


def _build_for_h(h: int):
    assert h % WARP_SIZE == 0, f"H={h} must be a multiple of warp size {WARP_SIZE}"
    n_warps = _pick_warps(h)
    n_threads = n_warps * WARP_SIZE
    elems_per_thread = h // n_threads
    h_f32 = float(h)

    @cute.kernel
    def kernel(gx: cute.Tensor, gw: cute.Tensor, gy: cute.Tensor, eps: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_id = tidx // Int32(WARP_SIZE)
        lane_id = tidx % Int32(WARP_SIZE)

        # ---- Pass 1: per-thread partial sums ----
        local_sum = Float32(0.0)
        local_sumsq = Float32(0.0)
        for k in range(elems_per_thread):
            col = tidx + k * n_threads
            v = gx[bidx, col].to(Float32)
            local_sum = local_sum + v
            local_sumsq = local_sumsq + v * v

        # Intra-warp butterfly reduction.
        for delta in (16, 8, 4, 2, 1):
            local_sum = local_sum + cute.arch.shuffle_sync_bfly(
                local_sum, offset=delta, mask=-1, mask_and_clamp=WARP_SIZE - 1
            )
            local_sumsq = local_sumsq + cute.arch.shuffle_sync_bfly(
                local_sumsq, offset=delta, mask=-1, mask_and_clamp=WARP_SIZE - 1
            )

        # Cross-warp reduction via shared memory.
        smem_ptr = cute.arch.get_dyn_smem(Float32, alignment=16)
        smem = cute.make_tensor(
            smem_ptr, cute.make_layout((n_warps * 2,), stride=(1,))
        )
        if lane_id == Int32(0):
            smem[warp_id] = local_sum
            smem[warp_id + Int32(n_warps)] = local_sumsq
        cute.arch.sync_threads()

        total_sum = Float32(0.0)
        total_sumsq = Float32(0.0)
        for w in range(n_warps):
            total_sum = total_sum + smem[w]
            total_sumsq = total_sumsq + smem[w + n_warps]

        h_recip = Float32(1.0 / h_f32)
        mean = total_sum * h_recip
        var = total_sumsq * h_recip - mean * mean
        inv_std = cute.math.rsqrt(var + eps)

        # ---- Pass 2: normalize, scale, write back ----
        for k in range(elems_per_thread):
            col = tidx + k * n_threads
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
            block=(n_threads, 1, 1),
            smem=(n_warps * 2 * 4 + 128) & ~127,
            stream=stream,
        )

    return launch


def _build_with_stats_for_h(h: int):
    """Like `_build_for_h` but also writes flat per-row mean/rstd (for backward)."""
    assert h % WARP_SIZE == 0, f"H={h} must be a multiple of warp size {WARP_SIZE}"
    n_warps = _pick_warps(h)
    n_threads = n_warps * WARP_SIZE
    elems_per_thread = h // n_threads
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
        warp_id = tidx // Int32(WARP_SIZE)
        lane_id = tidx % Int32(WARP_SIZE)

        local_sum = Float32(0.0)
        local_sumsq = Float32(0.0)
        for k in range(elems_per_thread):
            col = tidx + k * n_threads
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

        smem_ptr = cute.arch.get_dyn_smem(Float32, alignment=16)
        smem = cute.make_tensor(
            smem_ptr, cute.make_layout((n_warps * 2,), stride=(1,))
        )
        if lane_id == Int32(0):
            smem[warp_id] = local_sum
            smem[warp_id + Int32(n_warps)] = local_sumsq
        cute.arch.sync_threads()

        total_sum = Float32(0.0)
        total_sumsq = Float32(0.0)
        for w in range(n_warps):
            total_sum = total_sum + smem[w]
            total_sumsq = total_sumsq + smem[w + n_warps]

        h_recip = Float32(1.0 / h_f32)
        mean = total_sum * h_recip
        var = total_sumsq * h_recip - mean * mean
        inv_std = cute.math.rsqrt(var + eps)

        if tidx == 0:
            gmean[bidx] = mean
            grstd[bidx] = inv_std

        for k in range(elems_per_thread):
            col = tidx + k * n_threads
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
            block=(n_threads, 1, 1),
            smem=(n_warps * 2 * 4 + 128) & ~127,
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


class _CachedLN:
    """Caches the per-weight CuTe tensor, eps scalar, and compiled fn so only
    the per-call input/output tensors go through from_dlpack on each invocation."""

    __slots__ = ("_compiled", "_cute_w", "_eps", "_h")

    def __init__(self, h, weight, eps, launcher):
        self._h = h
        self._eps = Float32(eps)
        self._cute_w = from_dlpack(weight, assumed_align=16)
        d = weight.device
        dummy = torch.empty(1, h, dtype=weight.dtype, device=d)
        args = (
            from_dlpack(dummy, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            self._cute_w,
            from_dlpack(dummy, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            self._eps,
            current_cute_stream(),
        )
        self._compiled = get_compiled(launcher, args, key=(weight.dtype,))

    def __call__(self, x):
        flat = x.reshape(-1, self._h)
        out = torch.empty_like(flat)
        self._compiled(
            from_dlpack(flat, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            self._cute_w,
            from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            self._eps,
            current_cute_stream(),
        )
        return out.view(x.shape)


_ln_dispatch: dict[int, _CachedLN] = {}


def layer_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm forward. `x`: [..., H], `weight`: [H]; returns x's shape/dtype.
    H must be a multiple of 32 (warp size)."""
    wid = id(weight)
    cached = _ln_dispatch.get(wid)
    if cached is None:
        assert x.is_cuda and weight.is_cuda, "layer_norm kernel requires CUDA tensors"
        assert weight.ndim == 1, "weight must be 1D"
        h = weight.shape[0]
        assert x.shape[-1] == h, f"last dim of x ({x.shape[-1]}) must equal H ({h})"
        assert x.dtype == weight.dtype, "x and weight must have same dtype"
        cached = _CachedLN(h, weight, eps, _get_jit(h))
        _ln_dispatch[wid] = cached
    return cached(x)


class _CachedLNStats:
    __slots__ = ("_compiled", "_cute_w", "_eps", "_h")

    def __init__(self, h, weight, eps, launcher):
        self._h = h
        self._eps = Float32(eps)
        self._cute_w = from_dlpack(weight, assumed_align=16)
        d = weight.device
        dummy = torch.empty(1, h, dtype=weight.dtype, device=d)
        m1 = torch.empty(1, dtype=torch.float32, device=d)
        args = (
            from_dlpack(dummy, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            self._cute_w,
            from_dlpack(dummy, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(m1, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(m1, assumed_align=16).mark_layout_dynamic(),
            self._eps,
            current_cute_stream(),
        )
        self._compiled = get_compiled(launcher, args, key=("stats", weight.dtype))

    def __call__(self, x):
        flat = x.reshape(-1, self._h)
        out = torch.empty_like(flat)
        n = flat.shape[0]
        mean = torch.empty(n, dtype=torch.float32, device=x.device)
        rstd = torch.empty(n, dtype=torch.float32, device=x.device)
        self._compiled(
            from_dlpack(flat, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            self._cute_w,
            from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(mean, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(rstd, assumed_align=16).mark_layout_dynamic(),
            self._eps,
            current_cute_stream(),
        )
        return out.view(x.shape), mean, rstd


_ln_stats_dispatch: dict[int, _CachedLNStats] = {}


def layer_norm_with_stats(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm forward plus flat fp32 mean/rstd vectors for backward."""
    wid = id(weight)
    cached = _ln_stats_dispatch.get(wid)
    if cached is None:
        assert x.is_cuda and weight.is_cuda, "layer_norm kernel requires CUDA tensors"
        assert weight.ndim == 1, "weight must be 1D"
        h = weight.shape[0]
        assert x.shape[-1] == h, f"last dim of x ({x.shape[-1]}) must equal H ({h})"
        assert x.dtype == weight.dtype, "x and weight must have same dtype"
        cached = _CachedLNStats(h, weight, eps, _get_jit_with_stats(h))
        _ln_stats_dispatch[wid] = cached
    return cached(x)
