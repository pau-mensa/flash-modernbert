"""Unified LayerNorm / residual-add+LayerNorm forward kernel in CuteDSL.

    x_new = bf16(x + residual)  # residual specialization only
    y = (x_new - mean(x_new, -1)) / sqrt(var(x_new, -1) + eps) * weight

One block per row of the flattened `[N, H]`. Each thread caches its x slice in
registers during the stats pass and reads from registers in the normalize pass,
eliminating a second global memory read. This is a significant win on high-BW
cards (B200: 14% faster than Triton at 262K tokens; removes the 1.76x penalty
of the uncached double-read).

Prefers 2 warps (64 threads) to minimize cross-warp reduction overhead. The
butterfly reduction yields per-warp partials, lane 0 writes to shared memory,
then every thread reduces the 2 warp partials with scalar adds.

The residual presence is baked into the closure.  The no-residual cubin therefore
contains neither a residual load nor an updated-residual store; it does not fake the
operation with an allocated zero tensor.  The add specialization deliberately rounds
the fp32 sum to the input dtype before accumulating LayerNorm statistics, matching the
two eager bf16 operations used by Hugging Face.

Accumulates statistics in fp32. H is baked into the closure so the per-thread loops
unroll.  The same kernel body also specializes on whether backward statistics are
written.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from packed_encoders._kernels._compile_cache import current_cute_stream, get_compiled

WARP_SIZE = 32


def _pick_warps(h: int) -> int:
    """Choose the number of warps so that H divides evenly across all threads.
    Prefer 2 warps — fewer warps means more elements per thread (better for
    register caching) and a cheaper cross-warp reduction."""
    for nw in (2, 4, 8, 1):
        if h % (nw * WARP_SIZE) == 0:
            return nw
    return 1


def _build_for_h(h: int, *, has_residual: bool, write_stats: bool):
    assert h % WARP_SIZE == 0, f"H={h} must be a multiple of warp size {WARP_SIZE}"
    n_warps = _pick_warps(h)
    n_threads = n_warps * WARP_SIZE
    elems_per_thread = h // n_threads
    h_f32 = float(h)

    @cute.kernel
    def kernel(
        gx: cute.Tensor,
        gresidual: cute.Tensor,
        gw: cute.Tensor,
        gx_out: cute.Tensor,
        gy: cute.Tensor,
        gmean: cute.Tensor,
        grstd: cute.Tensor,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_id = tidx // Int32(WARP_SIZE)
        lane_id = tidx % Int32(WARP_SIZE)

        # ---- Pass 1: load/update x into registers, accumulate partial sums ----
        x_vals = cute.make_rmem_tensor((elems_per_thread,), Float32)
        local_sum = Float32(0.0)
        local_sumsq = Float32(0.0)
        for k in range(elems_per_thread):
            col = tidx + k * n_threads
            v = gx[bidx, col].to(Float32)
            if has_residual:
                # HF performs a bf16 residual add and then LayerNorm.  Preserve that
                # rounding boundary rather than normalizing the unrounded fp32 sum.
                v = gx.element_type(
                    v + gresidual[bidx, col].to(Float32)
                ).to(Float32)
                gx_out[bidx, col] = gx_out.element_type(v)
            x_vals[k] = v
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

        if write_stats and tidx == 0:
            gmean[bidx] = mean
            grstd[bidx] = inv_std

        # ---- Pass 2: normalize from cached x, scale, write back ----
        for k in range(elems_per_thread):
            col = tidx + k * n_threads
            w = gw[col].to(Float32)
            y = (x_vals[k] - mean) * inv_std * w
            gy[bidx, col] = gy.element_type(y)

    @cute.jit
    def launch(
        x: cute.Tensor,
        residual: cute.Tensor,
        weight: cute.Tensor,
        x_out: cute.Tensor,
        out: cute.Tensor,
        mean: cute.Tensor,
        rstd: cute.Tensor,
        eps: Float32,
        stream: cuda_driver.CUstream,
    ):
        n_rows = cute.size(x, mode=[0])
        kernel(x, residual, weight, x_out, out, mean, rstd, eps).launch(
            grid=(n_rows, 1, 1),
            block=(n_threads, 1, 1),
            smem=(n_warps * 2 * 4 + 128) & ~127,
            stream=stream,
        )

    return launch


_compiled_for_h: dict[tuple[int, bool, bool], object] = {}


def _get_jit(h: int, *, has_residual: bool, write_stats: bool):
    key = (h, has_residual, write_stats)
    if key not in _compiled_for_h:
        _compiled_for_h[key] = _build_for_h(
            h, has_residual=has_residual, write_stats=write_stats
        )
    return _compiled_for_h[key]


class _CachedLN:
    """Caches the per-weight CuTe tensor, eps scalar, and compiled fn so only
    the per-call input/output tensors go through from_dlpack on each invocation."""

    __slots__ = (
        "_compiled",
        "_cute_w",
        "_cute_stats_placeholder",
        "_eps",
        "_h",
        "_has_residual",
    )

    def __init__(self, h, weight, eps, launcher, *, has_residual):
        self._h = h
        self._has_residual = has_residual
        self._eps = Float32(eps)
        self._cute_w = from_dlpack(weight, assumed_align=16)
        d = weight.device
        dummy = torch.empty(1, h, dtype=weight.dtype, device=d)
        stats_placeholder = torch.empty(1, dtype=torch.float32, device=d)
        self._cute_stats_placeholder = from_dlpack(
            stats_placeholder, assumed_align=16
        ).mark_layout_dynamic()
        args = (
            from_dlpack(dummy, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dummy, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            self._cute_w,
            from_dlpack(dummy, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dummy, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            self._cute_stats_placeholder,
            self._cute_stats_placeholder,
            self._eps,
            current_cute_stream(),
        )
        self._compiled = get_compiled(
            launcher, args, key=(weight.dtype, has_residual, False)
        )

    def __call__(self, x, residual=None):
        flat = x.reshape(-1, self._h)
        if self._has_residual:
            assert residual is not None and residual.shape == x.shape
            flat_residual = residual.reshape(-1, self._h)
            x_out = torch.empty_like(flat)
        else:
            assert residual is None
            flat_residual = flat  # compile-time-dead placeholder; never loaded
            x_out = flat  # compile-time-dead placeholder; never written
        out = torch.empty_like(flat)
        self._compiled(
            from_dlpack(flat, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(flat_residual, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            self._cute_w,
            from_dlpack(x_out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            self._cute_stats_placeholder,
            self._cute_stats_placeholder,
            self._eps,
            current_cute_stream(),
        )
        y = out.view(x.shape)
        if self._has_residual:
            return x_out.view(x.shape), y
        return y


_ln_dispatch: dict[int, _CachedLN] = {}
_add_ln_dispatch: dict[int, _CachedLN] = {}


def _validate_inputs(
    x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor | None = None
) -> None:
    assert x.is_cuda and weight.is_cuda, "layer_norm kernel requires CUDA tensors"
    assert weight.ndim == 1, "weight must be 1D"
    h = weight.shape[0]
    assert x.shape[-1] == h, f"last dim of x ({x.shape[-1]}) must equal H ({h})"
    assert x.dtype == weight.dtype, "x and weight must have same dtype"
    if residual is not None:
        assert residual.is_cuda and residual.shape == x.shape
        assert residual.dtype == x.dtype, "x and residual must have same dtype"


def layer_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm forward. `x`: [..., H], `weight`: [H]; returns x's shape/dtype.
    H must be a multiple of 32 (warp size)."""
    wid = id(weight)
    cached = _ln_dispatch.get(wid)
    if cached is None:
        _validate_inputs(x, weight)
        h = weight.shape[0]
        cached = _CachedLN(
            h,
            weight,
            eps,
            _get_jit(h, has_residual=False, write_stats=False),
            has_residual=False,
        )
        _ln_dispatch[wid] = cached
    return cached(x)


def add_layer_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused bf16 residual add plus LayerNorm, returning ``(x_new, y)``."""
    wid = id(weight)
    cached = _add_ln_dispatch.get(wid)
    if cached is None:
        _validate_inputs(x, weight, residual)
        h = weight.shape[0]
        cached = _CachedLN(
            h,
            weight,
            eps,
            _get_jit(h, has_residual=True, write_stats=False),
            has_residual=True,
        )
        _add_ln_dispatch[wid] = cached
    return cached(x, residual)


class _CachedLNStats:
    __slots__ = ("_compiled", "_cute_w", "_eps", "_h", "_has_residual")

    def __init__(self, h, weight, eps, launcher, *, has_residual):
        self._h = h
        self._has_residual = has_residual
        self._eps = Float32(eps)
        self._cute_w = from_dlpack(weight, assumed_align=16)
        d = weight.device
        dummy = torch.empty(1, h, dtype=weight.dtype, device=d)
        m1 = torch.empty(1, dtype=torch.float32, device=d)
        args = (
            from_dlpack(dummy, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dummy, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            self._cute_w,
            from_dlpack(dummy, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dummy, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(m1, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(m1, assumed_align=16).mark_layout_dynamic(),
            self._eps,
            current_cute_stream(),
        )
        self._compiled = get_compiled(
            launcher, args, key=("stats", weight.dtype, has_residual)
        )

    def __call__(self, x, residual=None):
        flat = x.reshape(-1, self._h)
        if self._has_residual:
            assert residual is not None and residual.shape == x.shape
            flat_residual = residual.reshape(-1, self._h)
            x_out = torch.empty_like(flat)
        else:
            assert residual is None
            flat_residual = flat
            x_out = flat
        out = torch.empty_like(flat)
        n = flat.shape[0]
        mean = torch.empty(n, dtype=torch.float32, device=x.device)
        rstd = torch.empty(n, dtype=torch.float32, device=x.device)
        self._compiled(
            from_dlpack(flat, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(flat_residual, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            self._cute_w,
            from_dlpack(x_out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(mean, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(rstd, assumed_align=16).mark_layout_dynamic(),
            self._eps,
            current_cute_stream(),
        )
        y = out.view(x.shape)
        if self._has_residual:
            return x_out.view(x.shape), y, mean, rstd
        return y, mean, rstd


_ln_stats_dispatch: dict[int, _CachedLNStats] = {}
_add_ln_stats_dispatch: dict[int, _CachedLNStats] = {}


def layer_norm_with_stats(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm forward plus flat fp32 mean/rstd vectors for backward."""
    wid = id(weight)
    cached = _ln_stats_dispatch.get(wid)
    if cached is None:
        _validate_inputs(x, weight)
        h = weight.shape[0]
        cached = _CachedLNStats(
            h,
            weight,
            eps,
            _get_jit(h, has_residual=False, write_stats=True),
            has_residual=False,
        )
        _ln_stats_dispatch[wid] = cached
    return cached(x)


def add_layer_norm_with_stats(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused residual add + LayerNorm plus flat fp32 stats for backward."""
    wid = id(weight)
    cached = _add_ln_stats_dispatch.get(wid)
    if cached is None:
        _validate_inputs(x, weight, residual)
        h = weight.shape[0]
        cached = _CachedLNStats(
            h,
            weight,
            eps,
            _get_jit(h, has_residual=True, write_stats=True),
            has_residual=True,
        )
        _add_ln_stats_dispatch[wid] = cached
    return cached(x, residual)
