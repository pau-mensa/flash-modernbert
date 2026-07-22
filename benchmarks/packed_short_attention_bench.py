"""Benchmark packed short attention against varlen FlashAttention and FP32 PyTorch.

The inputs are already RoPE'd contiguous ``[M, H, 64]`` tensors.  This deliberately
excludes projection, RoPE, and padding work so the result describes the packed-to-
packed attention crossover rather than the existing padded SDPA/FA crossover.

Example:

    .venv/bin/python benchmarks/packed_short_attention_bench.py \
        --batches 1,8,32 --seqs 32,64,128 --output /tmp/packed_short.json

The JSON also contains ``weighted_dispatch_rows`` when both modes are measured.
Those rows compare ``8 * global + 14 * local`` and apply the same 3% production
Flash guard used by the runtime policy.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Callable, Sequence

import torch

from packed_encoders import ops


AGENTIR_BATCHES = (
    (128,) * 8,
    (125, 122, 122, 122, 122, 121, 121, 121),
    (101, 101, 101, 98, 98, 98, 97, 97),
    (67, 66, 65, 58, 56, 56, 56, 55),
    (46, 46, 45, 45, 45, 44, 43, 42),
)


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part) for part in value.split(","))
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return result


def _profiles(batch: int, length: int) -> list[tuple[str, tuple[int, ...]]]:
    rows = [("equal", (length,) * batch)]
    if batch > 1:
        rows.extend(
            (
                ("half", tuple(length if i % 2 == 0 else max(1, length // 2)
                               for i in range(batch))),
                ("skew", (length,) + (max(1, length // 8),) * (batch - 1)),
            )
        )
    return rows


def build_cases(
    batches: Sequence[int],
    lengths: Sequence[int],
    *,
    include_agentir: bool,
) -> list[tuple[str, tuple[int, ...]]]:
    cases = [row for batch in batches for length in lengths
             for row in _profiles(batch, length)]
    if include_agentir:
        cases.extend(("agentir", row) for row in AGENTIR_BATCHES)
    seen: set[tuple[int, ...]] = set()
    return [(name, row) for name, row in cases
            if not (row in seen or seen.add(row))]


def packed_tensors(
    lengths: Sequence[int],
    *,
    heads: int = 12,
    head_dim: int = 64,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (sum(lengths), heads, head_dim)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    seqlens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.nn.functional.pad(
        seqlens.cumsum(0, dtype=torch.int32), (1, 0)
    )
    return q, k, v, cu_seqlens


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    half_window: int | None,
    softmax_scale: float,
) -> torch.Tensor:
    """Per-sequence FP32 reference; intentionally slow and straightforward."""
    boundaries = cu_seqlens.cpu().tolist()
    outputs = []
    for start, end in zip(boundaries, boundaries[1:]):
        qf, kf, vf = q[start:end].float(), k[start:end].float(), v[start:end].float()
        scores = torch.einsum("qhd,khd->hqk", qf, kf) * softmax_scale
        if half_window is not None:
            positions = torch.arange(end - start, device=q.device)
            outside = (positions[:, None] - positions[None, :]).abs() > half_window
            scores.masked_fill_(outside.unsqueeze(0), float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probabilities, vf))
    return torch.cat(outputs).to(q.dtype)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    max_seqlen: int,
    half_window: int | None,
    softmax_scale: float,
) -> torch.Tensor:
    _dense, varlen, kind = ops._load_flash_attn()
    window = ((None, None) if half_window is None else (half_window, half_window))
    if kind == "compiled":
        window = (-1, -1) if half_window is None else window
    kwargs = dict(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=window,
    )
    if kind == "compiled":
        kwargs["dropout_p"] = 0.0
    out = varlen(q, k, v, **kwargs)
    return out[0] if isinstance(out, (tuple, list)) else out


def _load_triton_backend():
    try:
        from packed_encoders._kernels.triton_packed_attention import (
            packed_short_attention,
            select_packed_short_attention_config,
        )
    except ImportError:
        return None, None
    return packed_short_attention, select_packed_short_attention_config


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _wall_ms(fn: Callable[[], object], iterations: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iterations


def time_backend(
    fn: Callable[[], torch.Tensor],
    *,
    samples: int,
    target_ms: float,
) -> dict[str, object]:
    compile_start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - compile_start) * 1e3
    for _ in range(4):
        fn()
    probe_ms = _wall_ms(fn, 1)
    iterations = max(1, min(2000, math.ceil(target_ms / max(probe_ms, 1e-4))))
    eager = _summary([_wall_ms(fn, iterations) for _ in range(samples)])

    # Capture after compilation. Allocations made by the callable become graph-pool
    # allocations and the returned tensor remains live through the graph object.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = fn()
    captured = _summary([_wall_ms(graph.replay, iterations) for _ in range(samples)])
    return {
        "cold_ms": cold_ms,
        "iterations": iterations,
        "eager": eager,
        "captured": captured,
        # Keep it alive until all replay timing is complete.
        "_captured_out": captured_out,
    }


def error_summary(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = actual.float() - reference.float()
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            actual.float().flatten(), reference.float().flatten(), dim=0
        ).item(),
    }


def metadata() -> dict[str, object]:
    import triton

    try:
        import flash_attn
        flash_version = getattr(flash_attn, "__version__", "unknown")
    except ImportError:
        flash_version = None
    return {
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "flash_attn": flash_version,
        "dtype": "bfloat16",
        "heads": 12,
        "head_dim": 64,
        "timing": "synchronized wall time; cold first call separate from warm timing",
    }


def run_case(
    profile: str,
    lengths: tuple[int, ...],
    *,
    half_window: int | None,
    samples: int,
    target_ms: float,
    seed: int,
) -> dict[str, object]:
    q, k, v, cu_seqlens = packed_tensors(lengths, seed=seed)
    max_seqlen = max(lengths)
    scale = q.shape[-1] ** -0.5
    common = dict(
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        half_window=half_window,
        softmax_scale=scale,
    )
    with torch.inference_mode():
        reference = reference_attention(
            q, k, v, cu_seqlens,
            half_window=half_window, softmax_scale=scale,
        )
        flash_fn = lambda: flash_attention(q, k, v, **common)
        flash_out = flash_fn()
        flash_timing = time_backend(flash_fn, samples=samples, target_ms=target_ms)
        flash_timing.pop("_captured_out")
        backends: dict[str, object] = {
            "flash": {"timing": flash_timing,
                      "error": error_summary(flash_out, reference)}
        }

        triton_fn, config_fn = _load_triton_backend()
        if triton_fn is not None and max_seqlen <= 128:
            short_fn = lambda: triton_fn(q, k, v, **common)
            compile_start = time.perf_counter()
            short_out = short_fn()
            torch.cuda.synchronize()
            triton_cold_ms = (time.perf_counter() - compile_start) * 1e3
            short_timing = time_backend(
                short_fn, samples=samples, target_ms=target_ms
            )
            # time_backend's cold call is warm here because correctness compiled it.
            short_timing["cold_ms"] = triton_cold_ms
            short_timing.pop("_captured_out")
            backends["triton_short"] = {
                "config": config_fn(max_seqlen),
                "timing": short_timing,
                "error": error_summary(short_out, reference),
                "speedup_eager": (
                    flash_timing["eager"]["median_ms"]
                    / short_timing["eager"]["median_ms"]
                ),
                "speedup_captured": (
                    flash_timing["captured"]["median_ms"]
                    / short_timing["captured"]["median_ms"]
                ),
            }
    return {
        "profile": profile,
        "lengths": list(lengths),
        "n_sequences": len(lengths),
        "live_tokens": sum(lengths),
        "max_seqlen": max_seqlen,
        "mode": "global" if half_window is None else f"local-{half_window}",
        "backends": backends,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=_parse_ints, default=(1, 2, 4, 8, 16, 32, 64, 128))
    parser.add_argument("--seqs", type=_parse_ints, default=(16, 32, 48, 64, 96, 128, 192, 256))
    parser.add_argument("--modes", choices=("global", "local", "both"), default="both")
    parser.add_argument("--no-agentir", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--target-ms", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    cases = build_cases(args.batches, args.seqs, include_agentir=not args.no_agentir)
    windows = ((None, 64) if args.modes == "both" else
               (None,) if args.modes == "global" else (64,))
    result = {"metadata": metadata(), "rows": []}
    print(json.dumps(result["metadata"], indent=2), flush=True)
    for index, (profile, lengths) in enumerate(cases):
        for half_window in windows:
            row = run_case(
                profile, lengths, half_window=half_window,
                samples=args.samples, target_ms=args.target_ms, seed=17 + index,
            )
            result["rows"].append(row)
            flash = row["backends"]["flash"]["timing"]["eager"]["median_ms"]
            short = row["backends"].get("triton_short")
            suffix = ""
            if short is not None:
                suffix = (f" short={short['timing']['eager']['median_ms']:.4f} ms "
                          f"speedup={short['speedup_eager']:.2f}x")
            print(
                f"{row['mode']:<8} {profile:<7} N={len(lengths):>3} "
                f"M={sum(lengths):>6} S={max(lengths):>3} "
                f"FA={flash:.4f} ms{suffix}", flush=True,
            )
    by_case: dict[tuple[str, tuple[int, ...]], dict[str, object]] = {}
    for row in result["rows"]:
        if "triton_short" not in row["backends"]:
            continue
        key = (row["profile"], tuple(row["lengths"]))
        by_case.setdefault(key, {})[row["mode"]] = row
    weighted_rows = []
    for (profile, lengths), modes in by_case.items():
        if "global" not in modes or "local-64" not in modes:
            continue
        global_row, local_row = modes["global"], modes["local-64"]
        fg = global_row["backends"]["flash"]["timing"]["captured"]["median_ms"]
        fl = local_row["backends"]["flash"]["timing"]["captured"]["median_ms"]
        tg = global_row["backends"]["triton_short"]["timing"]["captured"]["median_ms"]
        tl = local_row["backends"]["triton_short"]["timing"]["captured"]["median_ms"]
        fa_speedup = (8 * tg + 14 * tl) / (8 * fg + 14 * fl)
        weighted_rows.append({
            "profile": profile,
            "lengths": list(lengths),
            "n_sequences": len(lengths),
            "live_tokens": sum(lengths),
            "max_seqlen": max(lengths),
            "triton_time_over_flash_time": fa_speedup,
            "production_flash_win": fa_speedup >= 1.03,
        })
    result["weighted_dispatch_rows"] = weighted_rows
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
