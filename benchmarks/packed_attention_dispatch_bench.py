"""Captured packed Triton/Flash attention crossover calibration.

This compares two already-packed kernels. Every case measures global and local-64
attention and combines their captured times
with ModernBERT-base's 8-global/14-local layer mix.  The output retains distribution
and tile-work features for fitting one monotonic score per exact GPU/backend pair.
"""

from __future__ import annotations

import gc
import importlib.metadata
import math
import platform
import statistics
import time
from typing import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

from packed_encoders import ops
from packed_encoders._kernels.triton_packed_attention import (
    packed_short_attention,
    select_packed_short_attention_config,
)


DEFAULT_BATCHES = (
    1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256,
    320, 384, 512, 640, 768, 1024, 1536, 2048,
)
DEFAULT_SEQS = (32, 48, 64, 96, 128)


def _profile_lengths(profile: str, n: int, s: int) -> tuple[int, ...]:
    if profile == "equal":
        return (s,) * n
    if profile == "half":
        return tuple(s if index % 2 == 0 else max(1, s // 2) for index in range(n))
    if profile == "skew":
        return (s,) + (max(1, s // 8),) * (n - 1)
    raise ValueError(f"unknown profile {profile!r}")


def build_cases(
    batches: Sequence[int] = DEFAULT_BATCHES,
    seqs: Sequence[int] = DEFAULT_SEQS,
    *,
    profiles: Sequence[str] = ("equal", "half", "skew"),
    max_tokens: int = 131_072,
) -> list[tuple[str, tuple[int, ...]]]:
    cases = []
    for s in seqs:
        for n in batches:
            for profile in profiles:
                lengths = _profile_lengths(profile, int(n), int(s))
                if sum(lengths) <= max_tokens:
                    cases.append((profile, lengths))
    return cases


def _local_pairs(length: int, half_window: int = 64) -> int:
    if length <= half_window + 1:
        return length * length
    return length * (2 * half_window + 1) - half_window * (half_window + 1)


def case_features(profile: str, lengths: Sequence[int]) -> dict[str, object]:
    n = len(lengths)
    m = sum(lengths)
    smax = max(lengths)
    config = select_packed_short_attention_config(smax)
    bm, bn = config["block_m"], config["block_n"]
    global_pairs = sum(length * length for length in lengths)
    local_pairs = sum(_local_pairs(length) for length in lengths)
    global_tile_pairs = sum(
        math.ceil(length / bm) * bm * math.ceil(length / bn) * bn
        for length in lengths
    )
    # The local kernel currently visits the same key tiles and masks within them.
    local_tile_pairs = global_tile_pairs
    return {
        "profile": profile,
        "n_sequences": n,
        "live_tokens": m,
        "max_seqlen": smax,
        "rectangle_tokens": n * smax,
        "fragmentation_tokens": n * smax - m,
        "global_pairs": global_pairs,
        "local_pairs": local_pairs,
        "global_tile_pairs": global_tile_pairs,
        "local_tile_pairs": local_tile_pairs,
        "global_tile_waste": global_tile_pairs - global_pairs,
        "local_tile_waste": local_tile_pairs - local_pairs,
        "config": config,
    }


def _packed_tensors(lengths: Sequence[int], seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (sum(lengths), 12, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    cu = F.pad(lens.cumsum(0, dtype=torch.int32), (1, 0))
    return q, k, v, cu


def _flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu: torch.Tensor,
    max_seqlen: int,
    half_window: int | None,
):
    _dense, varlen, kind = ops._load_flash_attn()
    if kind == "cute":
        window = (None, None) if half_window is None else (half_window, half_window)
    else:
        window = (-1, -1) if half_window is None else (half_window, half_window)
    kwargs = {
        "cu_seqlens_q": cu,
        "cu_seqlens_k": cu,
        "max_seqlen_q": max_seqlen,
        "max_seqlen_k": max_seqlen,
        "softmax_scale": 64 ** -0.5,
        "causal": False,
        "window_size": window,
    }
    if kind == "compiled":
        kwargs["dropout_p"] = 0.0
    out = varlen(q, k, v, **kwargs)
    return out[0] if isinstance(out, (tuple, list)) else out


def _triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu: torch.Tensor,
    max_seqlen: int,
    half_window: int | None,
):
    return packed_short_attention(
        q, k, v, cu, max_seqlen,
        half_window=half_window, softmax_scale=64 ** -0.5,
    )


def _wall_us(fn: Callable[[], object], iterations: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def _capture(fn: Callable[[], torch.Tensor]):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = fn()
    return graph, captured_out


def _timing_summary(values: list[float], iterations: int) -> dict[str, object]:
    return {
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
        "samples_us": values,
        "iterations": iterations,
    }


def _warm_gpu() -> None:
    """One explicit clock warm-up before paired measurements begin."""
    x = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    start = time.perf_counter()
    while time.perf_counter() - start < 1.0:
        torch.mm(x, x)
    torch.cuda.synchronize()
    del x


def _mode_timing(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu: torch.Tensor,
    max_seqlen: int,
    half_window: int | None,
    *,
    samples: int,
    target_ms: float,
) -> dict[str, object]:
    flash_fn = lambda: _flash(q, k, v, cu, max_seqlen, half_window)
    triton_fn = lambda: _triton(q, k, v, cu, max_seqlen, half_window)
    flash_graph, flash_out = _capture(flash_fn)
    triton_graph, triton_out = _capture(triton_fn)
    probe_us = max(
        _wall_us(flash_graph.replay, 1),
        _wall_us(triton_graph.replay, 1),
    )
    iterations = max(
        1, min(10_000, math.ceil(target_ms * 1e3 / max(probe_us, 0.1)))
    )
    flash_values, triton_values = [], []
    # Pair backends inside every sample and alternate order. This is the critical
    # protection against clock/thermal drift near a narrow crossover.
    for sample in range(samples):
        if (sample + (half_window is not None)) % 2 == 0:
            flash_values.append(_wall_us(flash_graph.replay, iterations))
            triton_values.append(_wall_us(triton_graph.replay, iterations))
        else:
            triton_values.append(_wall_us(triton_graph.replay, iterations))
            flash_values.append(_wall_us(flash_graph.replay, iterations))
    result = {
        "flash": _timing_summary(flash_values, iterations),
        "triton": _timing_summary(triton_values, iterations),
    }
    del flash_out, triton_out, flash_graph, triton_graph
    return result


def metadata() -> dict[str, object]:
    import triton

    _dense, _varlen, kind = ops._load_flash_attn()
    try:
        import flash_attn
        flash_version = getattr(flash_attn, "__version__", "unknown")
    except ImportError:
        flash_version = None
    if kind == "cute" and flash_version in (None, "unknown"):
        try:
            flash_version = importlib.metadata.version("flash-attn-4")
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "flash_kind": kind,
        "flash_version": flash_version,
        "dtype": "bfloat16",
        "heads": 12,
        "head_dim": 64,
        "layer_weighting": {"global": 8, "local_64": 14},
        "timing": "captured synchronized wall time; paired alternating backend order",
    }


def correctness_probe() -> list[dict[str, object]]:
    """Small cross-architecture Triton/FA parity probe for the calibrated envelope."""
    rows = []
    for index, lengths in enumerate(((32, 17, 5), (64, 41, 9), (128, 73, 16))):
        q, k, v, cu = _packed_tensors(lengths, seed=911 + index)
        for half_window in (None, 64):
            with torch.inference_mode():
                flash = _flash(q, k, v, cu, max(lengths), half_window)
                triton = _triton(q, k, v, cu, max(lengths), half_window)
            delta = triton.float() - flash.float()
            rows.append({
                "lengths": list(lengths),
                "mode": "global" if half_window is None else "local_64",
                "cosine": F.cosine_similarity(
                    triton.float().flatten(), flash.float().flatten(), dim=0
                ).item(),
                "max_abs": delta.abs().max().item(),
                "mean_abs": delta.abs().mean().item(),
            })
        del q, k, v, cu
    return rows


def run_benchmark(
    cases: Iterable[tuple[str, Sequence[int]]],
    *,
    samples: int = 3,
    target_ms: float = 5.0,
    verbose: bool = True,
) -> dict[str, object]:
    _warm_gpu()
    result = {
        "metadata": metadata(),
        "correctness": correctness_probe(),
        "rows": [],
    }
    for index, (profile, lengths_value) in enumerate(cases):
        lengths = tuple(int(length) for length in lengths_value)
        q, k, v, cu = _packed_tensors(lengths, seed=1701 + index)
        smax = max(lengths)
        with torch.inference_mode():
            global_timing = _mode_timing(
                q, k, v, cu, smax, None, samples=samples, target_ms=target_ms
            )
            local_timing = _mode_timing(
                q, k, v, cu, smax, 64, samples=samples, target_ms=target_ms
            )
        weighted_flash = (
            8 * global_timing["flash"]["median_us"]
            + 14 * local_timing["flash"]["median_us"]
        )
        weighted_triton = (
            8 * global_timing["triton"]["median_us"]
            + 14 * local_timing["triton"]["median_us"]
        )
        row = {
            **case_features(profile, lengths),
            "timing": {"global": global_timing, "local_64": local_timing},
            "triton_time_over_flash_time": weighted_triton / weighted_flash,
            "production_flash_win": weighted_triton / weighted_flash >= 1.03,
        }
        result["rows"].append(row)
        if verbose:
            print(
                f"{profile:<5} N={len(lengths):>4} M={sum(lengths):>7} "
                f"S={smax:>3} T/FA={row['triton_time_over_flash_time']:.3f}",
                flush=True,
            )
        del q, k, v, cu
        if index % 16 == 15:
            gc.collect()
            torch.cuda.empty_cache()
    return result
