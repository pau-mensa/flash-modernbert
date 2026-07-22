"""Validate packed-forward CUDA-graph routing around attention crossovers.

Unlike the attention-only calibration, this measures complete 22-layer packed
ModernBERT graph replays.  Forced Triton and Flash graphs use identical static
``(M_bucket, N_capacity, S_capacity)`` buffers so only the attention backend differs.
"""

from __future__ import annotations

import gc
import math
import statistics
import time
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from packed_encoders import attention_dispatch, forward
from packed_encoders.config import ModernBertParams
from packed_encoders.graph import GraphConfig, _PackedGraphRunner, _round_up


def _warm_gpu(seconds: float = 1.0) -> None:
    x = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        torch.mm(x, x)
    torch.cuda.synchronize()
    del x


def _model():
    from transformers import ModernBertConfig, ModernBertModel

    config = ModernBertConfig()
    # Transformers 5.x moved this scheduling detail out of the public config even
    # though the fused ModernBERT-base path still needs the checkpoint's 3-layer mix.
    config.global_attn_every_n_layers = 3
    model = ModernBertModel(config).to(
        device="cuda", dtype=torch.bfloat16
    ).eval()
    return model, ModernBertParams.from_hf_config(model.config)


def _inputs(lengths: Sequence[int], seed: int = 17):
    lengths = tuple(int(length) for length in lengths)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(
        5, 50_000, (sum(lengths),), device="cuda", generator=generator
    )
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    cu = F.pad(lens.cumsum(0, dtype=torch.int32), (1, 0))
    positions = torch.cat(
        [torch.arange(length, device="cuda") for length in lengths]
    )
    return ids, cu, positions


def _wall_us(fn, iterations: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def _paired_timing(flash_fn, triton_fn, *, samples: int, target_ms: float):
    probe = max(_wall_us(flash_fn, 1), _wall_us(triton_fn, 1))
    iterations = max(1, min(1000, math.ceil(target_ms * 1e3 / probe)))
    flash, triton = [], []
    for sample in range(samples):
        if sample % 2:
            triton.append(_wall_us(triton_fn, iterations))
            flash.append(_wall_us(flash_fn, iterations))
        else:
            flash.append(_wall_us(flash_fn, iterations))
            triton.append(_wall_us(triton_fn, iterations))
    return {
        "iterations": iterations,
        "flash_us": statistics.median(flash),
        "triton_us": statistics.median(triton),
        "flash_samples_us": flash,
        "triton_samples_us": triton,
    }


def _route(
    workload: attention_dispatch.AttentionWorkload, *, for_cuda_graph: bool = False
) -> str | None:
    policy = attention_dispatch.get_packed_inference_policy(
        for_cuda_graph=for_cuda_graph
    )
    if policy is None:
        return None
    return "flash" if policy.use_flash(workload) else "triton"


def run_case(
    model,
    params,
    *,
    name: str,
    lengths: Sequence[int],
    config: GraphConfig,
    samples: int = 5,
    target_ms: float = 30.0,
) -> dict[str, object]:
    ids, cu, positions = _inputs(lengths)
    m, n, s = ids.numel(), len(lengths), max(lengths)
    mb = _round_up(m, config.pad_to)
    if config.max_batch is None or config.max_seq is None:
        raise ValueError("packed graph check needs max_batch and max_seq")
    actual = attention_dispatch.AttentionWorkload.from_summary(
        n_sequences=n, live_tokens=m, max_seqlen=s
    )
    capacity = attention_dispatch.AttentionWorkload.from_summary(
        n_sequences=config.max_batch,
        live_tokens=mb,
        max_seqlen=config.max_seq,
    )

    runners = {
        backend: _PackedGraphRunner(model, params, config, backend=backend)
        for backend in ("flash", "triton")
    }
    with torch.inference_mode():
        outputs = {
            backend: runner(ids, cu, s, positions)
            for backend, runner in runners.items()
        }
    with torch.inference_mode():
        timing = _paired_timing(
            lambda: runners["flash"](ids, cu, s, positions),
            lambda: runners["triton"](ids, cu, s, positions),
            samples=samples,
            target_ms=target_ms,
        )
    ratio = timing["triton_us"] / timing["flash_us"]
    result = {
        "name": name,
        "n_sequences": n,
        "live_tokens": m,
        "max_seqlen": s,
        "bucket_tokens": mb,
        "capacity_sequences": config.max_batch,
        "capacity_seqlen": config.max_seq,
        "actual_policy_route": _route(actual),
        "current_capture_policy_route": _route(actual, for_cuda_graph=True),
        "legacy_capacity_policy_route": _route(capacity),
        "measured_graph_winner": "flash" if ratio > 1.0 else "triton",
        "triton_over_flash": ratio,
        "cosine": F.cosine_similarity(
            outputs["triton"].float().flatten(),
            outputs["flash"].float().flatten(),
            dim=0,
        ).item(),
        "timing": timing,
    }
    del runners, outputs, ids, cu, positions
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(cases: Sequence[dict], *, samples: int = 5, target_ms: float = 30.0):
    _warm_gpu()
    model, params = _model()
    rows = []
    for case in cases:
        config = GraphConfig(**case["graph"])
        rows.append(run_case(
            model,
            params,
            name=case["name"],
            lengths=case["lengths"],
            config=config,
            samples=samples,
            target_ms=target_ms,
        ))
    return {
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "rows": rows,
    }
