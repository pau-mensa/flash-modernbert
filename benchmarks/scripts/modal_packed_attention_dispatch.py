"""Calibrate packed Triton/FA crossover policies on datacenter GPUs via Modal.

A100/L40S use compiled FA2; H200/B200 use CuteDSL FA4. All requested GPUs are
spawned concurrently. The returned metadata records the loader kind and package
version so an image/backend mismatch is visible in the result rather than assumed.

    .venv/bin/modal run benchmarks/scripts/modal_packed_attention_dispatch.py
    .venv/bin/modal run benchmarks/scripts/modal_packed_attention_dispatch.py --refine
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import modal


REPO = Path(__file__).resolve().parents[2]
FLASH_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/"
    "flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
)

base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.8.0", extra_index_url="https://download.pytorch.org/whl/cu128"
    )
    .pip_install("numpy", "transformers>=4.45")
)


def _with_sources(image: modal.Image) -> modal.Image:
    return (
        image
        .env({
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root",
            "PYTHONWARNINGS": "ignore::DeprecationWarning",
        })
        .add_local_dir(str(REPO / "packed_encoders"), "/root/packed_encoders")
        .add_local_file(
            str(REPO / "benchmarks" / "packed_attention_dispatch_bench.py"),
            "/root/packed_attention_dispatch_bench.py",
        )
        .add_local_file(
            str(REPO / "benchmarks" / "packed_graph_boundary_bench.py"),
            "/root/packed_graph_boundary_bench.py",
        )
    )


fa2_image = _with_sources(
    base_image
    .pip_install("nvidia-cutlass-dsl==4.5.2")
    .pip_install(FLASH_WHEEL)
)

fa4_image = _with_sources(
    base_image
    .pip_install("flash-attn-4==4.0.0b16")
    .pip_install("nvidia-cutlass-dsl==4.5.2", "quack-kernels==0.5.0")
)

app = modal.App("flash-mbert-packed-attention-dispatch-bench")


def _coarse() -> dict:
    import packed_attention_dispatch_bench as bench

    return bench.run_benchmark(
        bench.build_cases(), samples=5, target_ms=20.0, verbose=False
    )


def _refine(cases) -> dict:
    import packed_attention_dispatch_bench as bench

    normalized = [(profile, tuple(lengths)) for profile, lengths in cases]
    return bench.run_benchmark(
        normalized, samples=9, target_ms=100.0, verbose=False
    )


def _validate(cases) -> dict:
    import packed_attention_dispatch_bench as bench
    from packed_encoders.attention_dispatch import (
        AttentionWorkload,
        get_packed_inference_policy,
    )

    result = bench.run_benchmark(
        [(profile, tuple(lengths)) for profile, lengths in cases],
        samples=5,
        target_ms=20.0,
        verbose=False,
    )
    policy = get_packed_inference_policy()
    if policy is None:
        raise RuntimeError("exact-card packed policy did not resolve")
    for row in result["rows"]:
        workload = AttentionWorkload.from_summary(
            n_sequences=row["n_sequences"],
            live_tokens=row["live_tokens"],
            max_seqlen=row["max_seqlen"],
        )
        row["policy_score"] = policy.score(workload)
        row["policy_use_flash"] = policy.use_flash(workload)
    return result


def _graph_validate(cases) -> dict:
    import packed_graph_boundary_bench as bench

    return bench.run(cases, samples=7, target_ms=30.0)


@app.function(image=fa2_image, gpu="A100", timeout=3600)
def coarse_a100() -> dict:
    return _coarse()


@app.function(image=fa2_image, gpu="L40S", timeout=3600)
def coarse_l40s() -> dict:
    return _coarse()


@app.function(image=fa4_image, gpu="H200", timeout=3600)
def coarse_h200() -> dict:
    return _coarse()


@app.function(image=fa4_image, gpu="B200", timeout=3600)
def coarse_b200() -> dict:
    return _coarse()


@app.function(image=fa2_image, gpu="A100", timeout=3600)
def refine_a100(cases) -> dict:
    return _refine(cases)


@app.function(image=fa2_image, gpu="L40S", timeout=3600)
def refine_l40s(cases) -> dict:
    return _refine(cases)


@app.function(image=fa4_image, gpu="H200", timeout=3600)
def refine_h200(cases) -> dict:
    return _refine(cases)


@app.function(image=fa4_image, gpu="B200", timeout=3600)
def refine_b200(cases) -> dict:
    return _refine(cases)


@app.function(image=fa2_image, gpu="A100", timeout=1800)
def validate_a100(cases) -> dict:
    return _validate(cases)


@app.function(image=fa2_image, gpu="L40S", timeout=1800)
def validate_l40s(cases) -> dict:
    return _validate(cases)


@app.function(image=fa4_image, gpu="H200", timeout=1800)
def validate_h200(cases) -> dict:
    return _validate(cases)


@app.function(image=fa4_image, gpu="B200", timeout=1800)
def validate_b200(cases) -> dict:
    return _validate(cases)


@app.function(image=fa2_image, gpu="A100", timeout=3600)
def graph_validate_a100(cases) -> dict:
    return _graph_validate(cases)


@app.function(image=fa2_image, gpu="L40S", timeout=3600)
def graph_validate_l40s(cases) -> dict:
    return _graph_validate(cases)


@app.function(image=fa4_image, gpu="H200", timeout=3600)
def graph_validate_h200(cases) -> dict:
    return _graph_validate(cases)


@app.function(image=fa4_image, gpu="B200", timeout=3600)
def graph_validate_b200(cases) -> dict:
    return _graph_validate(cases)


_COARSE = {
    "A100": coarse_a100,
    "L40S": coarse_l40s,
    "H200": coarse_h200,
    "B200": coarse_b200,
}
_REFINE = {
    "A100": refine_a100,
    "L40S": refine_l40s,
    "H200": refine_h200,
    "B200": refine_b200,
}
_VALIDATE = {
    "A100": validate_a100,
    "L40S": validate_l40s,
    "H200": validate_h200,
    "B200": validate_b200,
}
_GRAPH_VALIDATE = {
    "A100": graph_validate_a100,
    "L40S": graph_validate_l40s,
    "H200": graph_validate_h200,
    "B200": graph_validate_b200,
}

_VALIDATION_CASES = {
    "A100": [
        ("half", (64, 32) * 192),
        ("half", (64, 32) * 256),
    ],
    "L40S": [
        ("equal", (64,) * 384),
        ("equal", (128,) * 192),
    ],
    "H200": [
        ("equal", (128,) * 8),
        ("equal", (128,) * 1024),
    ],
    "B200": [
        ("equal", (128,) * 8),
        ("equal", (128,) * 1024),
    ],
}

_GRAPH_VALIDATION_CASES = {
    "A100": [
        {
            "name": "triton-side-half-s64",
            "lengths": (64, 32) * 192,
            "graph": {
                "pad_to": 512, "max_batch": 512, "max_seq": 64,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
        {
            "name": "flash-side-half-s64",
            "lengths": (64, 32) * 256,
            "graph": {
                "pad_to": 512, "max_batch": 512, "max_seq": 64,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
    ],
    "L40S": [
        {
            "name": "equal-s64-m8192",
            "lengths": (64,) * 128,
            "graph": {
                "pad_to": 512, "max_batch": 512, "max_seq": 64,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
        {
            "name": "equal-s64-m16384",
            "lengths": (64,) * 256,
            "graph": {
                "pad_to": 512, "max_batch": 512, "max_seq": 64,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
        {
            "name": "equal-s64-m20480",
            "lengths": (64,) * 320,
            "graph": {
                "pad_to": 512, "max_batch": 512, "max_seq": 64,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
        {
            "name": "equal-s64-m24576",
            "lengths": (64,) * 384,
            "graph": {
                "pad_to": 512, "max_batch": 512, "max_seq": 64,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
        {
            "name": "equal-s64-m27648",
            "lengths": (64,) * 432,
            "graph": {
                "pad_to": 512, "max_batch": 512, "max_seq": 64,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
        {
            "name": "half-s32-m24576",
            "lengths": (32, 16) * 512,
            "graph": {
                "pad_to": 512, "max_batch": 1024, "max_seq": 32,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
        {
            "name": "equal-s96-m21504",
            "lengths": (96,) * 224,
            "graph": {
                "pad_to": 512, "max_batch": 256, "max_seq": 96,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
        {
            "name": "half-s128-m24576",
            "lengths": (128, 64) * 128,
            "graph": {
                "pad_to": 512, "max_batch": 256, "max_seq": 128,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
        {
            "name": "half-s64-m24576",
            "lengths": (64, 32) * 256,
            "graph": {
                "pad_to": 512, "max_batch": 512, "max_seq": 64,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
    ],
    "H200": [
        {
            "name": "small-equal-s128",
            "lengths": (128,) * 8,
            "graph": {
                "pad_to": 512, "max_batch": 128, "max_seq": 128,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
        {
            "name": "medium-equal-s128",
            "lengths": (128,) * 128,
            "graph": {
                "pad_to": 512, "max_batch": 128, "max_seq": 128,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
    ],
    "B200": [
        {
            "name": "small-equal-s128",
            "lengths": (128,) * 8,
            "graph": {
                "pad_to": 512, "max_batch": 128, "max_seq": 128,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
        {
            "name": "medium-equal-s128",
            "lengths": (128,) * 128,
            "graph": {
                "pad_to": 512, "max_batch": 128, "max_seq": 128,
                "max_tokens": 32_768, "warmup": 2,
            },
        },
    ],
}


def _refine_cases(rows: list[dict], count: int = 72):
    ordered = sorted(
        rows,
        key=lambda row: abs(row["triton_time_over_flash_time"] - 1.03),
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["profile"], row["max_seqlen"])].append(row)
    selected = []
    seen = set()

    def add(row):
        key = row["profile"], row["n_sequences"], row["max_seqlen"]
        if key not in seen:
            seen.add(key)
            selected.append(row)

    # Rerun both sides and immediate neighbors of every winner-label transition.
    for values in grouped.values():
        values.sort(key=lambda row: row["n_sequences"])
        for index, (left, right) in enumerate(zip(values, values[1:])):
            if left["production_flash_win"] != right["production_flash_win"]:
                for neighbor in values[max(0, index - 2) : index + 4]:
                    add(neighbor)
    for row in ordered:
        if len(selected) >= count:
            break
        add(row)

    cases = []
    for row in selected:
        profile, n, s = (
            row["profile"], row["n_sequences"], row["max_seqlen"]
        )
        if profile == "equal":
            lengths = (s,) * n
        elif profile == "half":
            lengths = tuple(s if i % 2 == 0 else s // 2 for i in range(n))
        else:
            lengths = (s,) + (max(1, s // 8),) * (n - 1)
        cases.append((profile, lengths))
    return cases


def _summary(result: dict) -> str:
    rows = result["rows"]
    wins = [row for row in rows if row["production_flash_win"]]
    losses = [row for row in rows if not row["production_flash_win"]]
    meta = result["metadata"]
    return (
        f"{meta['device']} sm_{''.join(map(str, meta['compute_capability']))} "
        f"{meta['flash_kind']} {meta['flash_version']} rows={len(rows)} "
        f"FA-wins={len(wins)} min-win-M="
        f"{min((row['live_tokens'] for row in wins), default=None)} "
        f"max-loss-M={max((row['live_tokens'] for row in losses), default=None)}"
    )


@app.local_entrypoint()
def main(
    refine: bool = False,
    refine_round: int = 1,
    validate: bool = False,
    graph_validate: bool = False,
    gpus: str = "A100,L40S,H200,B200",
):
    requested = [name.strip() for name in gpus.split(",") if name.strip()]
    unknown = [name for name in requested if name not in _COARSE]
    if unknown:
        raise SystemExit(f"unknown GPUs {unknown}; choose from {list(_COARSE)}")
    out_dir = REPO / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    calls = []
    for name in requested:
        if graph_validate:
            call = _GRAPH_VALIDATE[name].spawn(_GRAPH_VALIDATION_CASES[name])
        elif validate:
            call = _VALIDATE[name].spawn(_VALIDATION_CASES[name])
        elif refine:
            source = out_dir / f"packed_attention_dispatch_{name.lower()}.json"
            if not source.exists():
                raise SystemExit(f"missing coarse result: {source}")
            rows_by_key = {
                (row["profile"], row["n_sequences"], row["max_seqlen"]): row
                for row in json.loads(source.read_text())["rows"]
            }
            for round_index in range(1, refine_round):
                prior_suffix = "_refine" if round_index == 1 else f"_refine{round_index}"
                prior = out_dir / (
                    f"packed_attention_dispatch_{name.lower()}{prior_suffix}.json"
                )
                if prior.exists():
                    rows_by_key.update({
                        (row["profile"], row["n_sequences"], row["max_seqlen"]): row
                        for row in json.loads(prior.read_text())["rows"]
                    })
            rows = list(rows_by_key.values())
            call = _REFINE[name].spawn(_refine_cases(rows))
        else:
            call = _COARSE[name].spawn()
        calls.append((name, call))
    print(f"launched concurrently: {', '.join(requested)}", flush=True)

    suffix = (
        "_graph_validation" if graph_validate else
        "_validation" if validate else
        ("_refine" if refine_round == 1 else f"_refine{refine_round}")
        if refine else ""
    )
    for name, call in calls:
        try:
            result = call.get()
        except Exception as exc:
            print(f"{name} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        path = out_dir / f"packed_attention_dispatch_{name.lower()}{suffix}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        if graph_validate:
            decisions = ", ".join(
                f"{row['name']}={row['measured_graph_winner']} "
                f"({row['triton_over_flash']:.4f} T/FA)"
                for row in result["rows"]
            )
            print(f"{name}: {result['device']}: {decisions}", flush=True)
        else:
            print(f"{name}: {_summary(result)}", flush=True)
        print(f"  wrote {path}", flush=True)
