# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["flash-modernbert", "transformers", "matplotlib", "pyyaml"]
#
# [tool.uv.sources]
# flash-modernbert = { path = "../", editable = true }
# torch = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""B0 — training fwd+bwd micro-benchmark: stock HF vs the eager fused tail.

The training counterpart to `inference_bench.py`. Where that measures a no-grad
encoder forward, this measures one **encoder fwd+bwd step** — the region a
training-graph runner (Workstream B) would capture — in plain bf16 weights with
**no autocast** (the clean kernel-only A/B, matching `iso_loss.py`'s regime, so
no fp32-master/autocast confound clouds the kernel delta).

It exists to answer the roadmap's B0 question, before any training-graph code is
written:

    Is the eager fused tail a *speed regression* vs stock at short-S training
    (the classic ColBERT regime, D≈300 / Q≈32)?

If it is, training graphs are competitiveness-critical (they would recover the
launch floor exactly as they do for short-S inference); if not, they are pure
upside. The short-S regime is host-launch-bound — hundreds of tiny CuteDSL
launches per forward, doubled by the backward — so this is where the eager tail
is most at risk and where graphs would help most.

    stock      — the unpatched Hugging Face forward (SDPA attention)
    fused[b]   — flash_modernbert.prepare(model): the eager fused tail, backend b
    graph[b]   — the eager fused tail with **training-step CUDA graphs**: the
                 encoder fwd+bwd captured once per shape and replayed (the
                 `_TrainGraphRunner`). Emitted only when `train_cuda_graph: true`
                 is set in the config. Each shape uses a *fresh* runner so its one
                 captured graph pair is weighed in isolation (no long-S pools
                 stacking up to OOM, and the peak reading is that shape's graph
                 alone).

The step is `out = encoder(ids, mask); out.float().square().mean().backward()`
with grads zeroed each step — a synthetic upstream loss that touches every
encoder parameter, so the backward exercises the full fused-tail gradient path
without dragging in a framework-specific head. Reports ms/step and peak memory
per (variant, shape), the speedup vs stock, and — when graphs are on — the
graph-vs-eager-fused speed/memory delta (the isolated training-graph impact);
writes a JSON + chart.

    uv run benchmarks/train_bench.py benchmarks/configs/train_short.yml          # stock vs eager-fused
    uv run benchmarks/train_bench.py benchmarks/configs/train_graph_short.yml    # + graph[sdpa]

The `graph[b]` variant must zero `.grad` in place (never to None) — the captured
backward replays an add into the persistent `param.grad` address, so the buffer
has to stay live (the `set_to_none=False` training-graph invariant). The stock /
eager steps are free to drop grads to None.

This is the encoder fwd+bwd in isolation (pre-tokenized batches at exact shapes);
the full GradCache training step (with the projection head, scoring, and the
two-pass recompute) is measured by `iso_loss.py` / `pylate_trainer.py`.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

import flash_modernbert as fm
from flash_modernbert.locate import find_encoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    raw: dict

    def section(self, name: str) -> dict:
        return self.raw.get(name, {}) or {}


def load_config(path: str) -> Config:
    with open(path) as handle:
        return Config(yaml.safe_load(handle) or {})


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# ---------------------------------------------------------------------------
# Model — raw HF encoder, the module prepare() patches and a training-graph
# runner would capture. Built in bf16 weights (the clean A/B; no autocast).
# ---------------------------------------------------------------------------


def build_model(cfg: Config, device: torch.device, dtype: torch.dtype):
    section = cfg.section("model")
    name = section["name_or_path"]
    framework = section.get("framework", "huggingface")
    if framework != "huggingface":
        raise ValueError(
            "train_bench measures the bare encoder fwd+bwd; use framework: huggingface "
            "(the full-recipe training step is iso_loss.py / pylate_trainer.py)"
        )
    from transformers import AutoModel

    model = AutoModel.from_pretrained(name, torch_dtype=dtype).to(device).train()
    return model, find_encoder(model)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def make_batch(b: int, s: int, vocab: int, device: torch.device, seed: int):
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.randint(5, vocab, (b, s), generator=g, device=device)
    mask = torch.ones((b, s), dtype=torch.long, device=device)
    return ids, mask


def _last_hidden(out):
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def _zero_grads(params):
    for p in params:
        if p.grad is not None:
            p.grad = None


def measure(step, ids, mask, *, n_warmup: int, n_iters: int, n_trials: int) -> dict:
    """Median ms/step over `n_trials` trials of `n_iters` fwd+bwd steps each, plus
    peak allocated AND reserved memory across the measured window.

    `empty_cache()` before warmup releases the allocator's cached segments from the
    previous shape/variant, so the *reserved* high-water reflects only this
    measurement — otherwise reserved is sticky (a small shape inherits a big prior
    shape's reservation) and graph-vs-eager reserved comparisons are meaningless."""
    torch.cuda.empty_cache()
    for _ in range(n_warmup):
        step(ids, mask)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times_ms = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            step(ids, mask)
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) / n_iters * 1e3)

    return {
        "ms": statistics.median(times_ms),
        "ms_min": min(times_ms),
        "ms_max": max(times_ms),
        "peak_mb": torch.cuda.max_memory_allocated() / 1e6,
        # Reserved, not just allocated: a captured graph keeps its activations in a
        # private pool that `max_memory_allocated` does NOT count but
        # `max_memory_reserved` does. Comparing graph vs eager on allocated alone
        # makes graphs look ~10× cheaper than they are — reserved is the true,
        # graph-fair footprint (what actually decides OOM / max batch).
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }


def variant_steps(model_obj, encoder, validate: bool, backends):
    """Yield (name, step): `stock` once (captured before prepare), then `fused[b]`
    for each attention backend. Each step zeros grads, runs the encoder forward,
    and backprops a square-mean synthetic loss. Built lazily so the consumer
    measures between yields and state mutations land in order."""
    from flash_modernbert.state import get_state

    params = list(encoder.parameters())
    stock_forward = encoder.forward  # bound HF forward, before any patch

    def step_with(forward):
        def step(ids, mask):
            _zero_grads(params)
            out = _last_hidden(forward(input_ids=ids, attention_mask=mask))
            loss = out.float().square().mean()
            loss.backward()
        return step

    yield "stock", step_with(stock_forward)

    fm.prepare(model_obj, validate=validate)  # patches encoder.forward (eager, sdpa)
    state = get_state(model_obj)
    patched = step_with(encoder.forward)

    for b in backends:
        state.attention_backend = b
        yield f"fused[{b}]", patched

    _zero_grads(params)


def _zero_grads_inplace(params):
    """Zero every live `.grad` in a single fused multi-tensor launch (what a real
    `optimizer.zero_grad(set_to_none=False)` does). The training-graph backward
    replays an add into the *address* of each persistent `param.grad`, so the
    buffers must stay live across steps — unlike `_zero_grads`, which drops them to
    None. `_foreach_zero_` keeps this to ~one launch so it doesn't dominate the
    launch-bound short-S step we are trying to measure."""
    grads = [p.grad for p in params if p.grad is not None]
    if grads:
        torch._foreach_zero_(grads)


def graph_step(encoder, params):
    """A fwd+bwd step routed through the training-graph runner: zero grads in place,
    run the (now graph-replayed) encoder forward, backprop the same square-mean
    synthetic loss. `loss.backward()` drives `_TrainGraphFn.backward`, which
    replays the captured backward graph (accumulating into the live `.grad`)."""
    def step(ids, mask):
        _zero_grads_inplace(params)
        out = _last_hidden(encoder.forward(input_ids=ids, attention_mask=mask))
        loss = out.float().square().mean()
        loss.backward()
    return step


def measure_graphed(encoder, state, params, backend, ids, mask, **measure_kwargs):
    """Measure one shape through a *fresh* training-graph runner, then tear it down.

    Fresh-per-shape keeps exactly one captured fwd+bwd graph pair alive at a time,
    so the long-S buckets (whose private pools are GBs) don't stack up and OOM, and
    the peak reading is this shape's graph in isolation. Capture runs inside
    `measure`'s warmup — before `reset_peak_memory_stats` — so the reported peak is
    the steady-state *replay* footprint (the graph's packed activation pool),
    directly comparable to the eager variant's transient fwd+bwd peak."""
    from flash_modernbert.train_graph import TrainGraphConfig, build_train_runner

    state.attention_backend = backend
    # max_seq unbounded here: the bench *characterizes* the graph at every shape in
    # the sweep (incl. S>cutoff), to map where it wins vs loses. The production
    # default (TrainGraphConfig.max_seq=64, the queries-only gate) is a separate
    # policy decision and is exercised by the test suite, not forced here.
    state.train_graph_runner = build_train_runner(
        encoder, state.params, TrainGraphConfig(max_seq=2 ** 31), backend=backend
    )
    state.train_graph_enabled = True
    try:
        return measure(graph_step(encoder, params), ids, mask, **measure_kwargs)
    finally:
        state.train_graph_enabled = False
        state.train_graph_runner = None
        _zero_grads(params)
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(cfg: Config, device: torch.device) -> dict:
    run_cfg = cfg.section("run")
    dtype = _DTYPES[cfg.section("model").get("dtype", "bfloat16")]

    model_obj, encoder = build_model(cfg, device, dtype)
    vocab = int(encoder.config.vocab_size)
    shapes = _shapes(cfg)

    measure_kwargs = dict(
        n_warmup=int(run_cfg.get("n_warmup", 10)),
        n_iters=int(run_cfg.get("n_iters", 30)),
        n_trials=int(run_cfg.get("n_trials", 5)),
    )
    validate = bool(run_cfg.get("validate", True))
    backends = cfg.raw.get("attention_backends", ["sdpa"])

    batches = {
        i: make_batch(sh["b"], sh["s"], vocab, device, seed=2000 + i)
        for i, sh in enumerate(shapes)
    }

    results: dict[str, dict[int, dict]] = {}
    for name, step in variant_steps(model_obj, encoder, validate, backends):
        results[name] = {}
        for i, sh in enumerate(shapes):
            ids, mask = batches[i]
            r = measure(step, ids, mask, **measure_kwargs)
            r["samples_per_s"] = sh["b"] * 1e3 / r["ms"]
            results[name][i] = r
            print(
                f"  {name:<14s} {sh['kind']:<6s} B={sh['b']:<4d} S={sh['s']:<5d} "
                f"{r['ms']:8.3f} ms  {r['samples_per_s']:9.0f} samp/s  "
                f"alloc {r['peak_mb']:6.0f} / resv {r['peak_reserved_mb']:6.0f} MB"
            )
        print()

    if bool(cfg.raw.get("train_cuda_graph", False)):
        from flash_modernbert.state import get_state

        state = get_state(model_obj)
        params = list(encoder.parameters())
        for b in backends:
            name = f"graph[{b}]"
            results[name] = {}
            for i, sh in enumerate(shapes):
                ids, mask = batches[i]
                try:
                    r = measure_graphed(
                        encoder, state, params, b, ids, mask, **measure_kwargs
                    )
                    r["samples_per_s"] = sh["b"] * 1e3 / r["ms"]
                    print(
                        f"  {name:<14s} {sh['kind']:<6s} B={sh['b']:<4d} S={sh['s']:<5d} "
                        f"{r['ms']:8.3f} ms  {r['samples_per_s']:9.0f} samp/s  "
                        f"alloc {r['peak_mb']:6.0f} / resv {r['peak_reserved_mb']:6.0f} MB"
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    nan = float("nan")
                    r = {"ms": nan, "ms_min": nan, "ms_max": nan,
                         "peak_mb": nan, "peak_reserved_mb": nan,
                         "samples_per_s": nan, "oom": True}
                    print(f"  {name:<14s} {sh['kind']:<6s} B={sh['b']:<4d} "
                          f"S={sh['s']:<5d}   OOM during capture (skipped)")
                results[name][i] = r
            print()

    del model_obj, encoder, batches
    gc.collect()
    torch.cuda.empty_cache()

    return _assemble(cfg, shapes, results)


def _assemble(cfg: Config, shapes: list[dict], results: dict) -> dict:
    variants = list(results.keys())
    rows = []
    for i, sh in enumerate(shapes):
        stock = results["stock"][i]
        per_variant = {}
        for v in variants:
            r = dict(results[v][i])
            r["speedup"] = stock["ms"] / r["ms"]  # vs stock HF
            per_variant[v] = r
        rows.append({"shape": sh, "variants": per_variant})
    return {
        "config": cfg.raw,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "variants": variants,
        "rows": rows,
    }


def _shapes(cfg: Config) -> list[dict]:
    shapes = cfg.raw.get("shapes")
    if not shapes:
        raise ValueError("config needs a non-empty `shapes` list")
    return [
        {"b": int(sh["b"]), "s": int(sh["s"]), "kind": str(sh.get("kind", ""))}
        for sh in shapes
    ]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(cfg: Config, payload: dict, config_path: str):
    out = cfg.section("output")
    out_dir = Path(out.get("dir", "benchmarks/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = out.get("name", Path(config_path).stem)

    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    png_path = out_dir / f"{name}.png"
    _plot(payload, png_path, name)
    return json_path, png_path


_PALETTE = ["#3b6ea5", "#d1495b", "#5a9e6f", "#e8a33d", "#9b59b6", "#444444"]


def _plot(payload: dict, png_path: Path, name: str):
    rows = payload["rows"]
    variants = payload["variants"]
    labels = [f"{r['shape']['kind']}\nB{r['shape']['b']} S{r['shape']['s']}" for r in rows]
    x = range(len(rows))
    colors = {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(variants)}
    n = len(variants)
    width = 0.8 / n

    fig, (ax_ms, ax_sp) = plt.subplots(1, 2, figsize=(max(12, 2.6 * len(rows)), 5))

    for vi, v in enumerate(variants):
        off = (vi - (n - 1) / 2) * width
        ms = [r["variants"][v]["ms"] for r in rows]
        ax_ms.bar([i + off for i in x], ms, width, label=v, color=colors[v])
    ax_ms.set(ylabel="ms / step (lower is better)", title="Encoder fwd+bwd latency")
    ax_ms.set_xticks(list(x)); ax_ms.set_xticklabels(labels, fontsize=8)
    ax_ms.legend(fontsize=8); ax_ms.grid(True, axis="y", alpha=0.3)

    speedup_variants = [v for v in variants if v != "stock"]
    m = len(speedup_variants)
    sw = 0.8 / max(m, 1)
    for vi, v in enumerate(speedup_variants):
        off = (vi - (m - 1) / 2) * sw
        sp = [r["variants"][v]["speedup"] for r in rows]
        ax_sp.bar([i + off for i in x], sp, sw, label=v, color=colors[v])
        for i, s in enumerate(sp):
            ax_sp.text(i + off, s, f"{s:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax_sp.axhline(1.0, color="#444", lw=1, ls="--")
    ax_sp.set(ylabel="speedup vs stock HF", title="Fwd+bwd speedup vs stock")
    ax_sp.set_xticks(list(x)); ax_sp.set_xticklabels(labels, fontsize=8)
    ax_sp.legend(fontsize=8); ax_sp.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"{name} — encoder fwd+bwd training ({payload['device']}, "
        f"sm_{''.join(map(str, payload['capability']))}, bf16 weights, no autocast)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a benchmark YAML config")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("error: this benchmark requires a CUDA GPU", file=sys.stderr)
        raise SystemExit(1)

    cfg = load_config(args.config)
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}\n")

    payload = run(cfg, device)
    json_path, png_path = write_outputs(cfg, payload, args.config)

    speedup_variants = [v for v in payload["variants"] if v != "stock"]
    print("=== fwd+bwd speedup vs stock HF (ms/step; peak MB) ===")
    for r in payload["rows"]:
        sh = r["shape"]
        parts = " ".join(
            f"{v} {r['variants'][v]['speedup']:.2f}x" for v in speedup_variants
        )
        print(f"  {sh['kind']:<6s} B={sh['b']:<4d} S={sh['s']:<5d}  {parts}")

    # The isolated training-graph impact: graph[b] vs its eager-fused[b] baseline
    # (same kernels, same dense mask — the only difference is capture/replay). This
    # is the clean A/B the cross-process "graph vs no-graph" runs could never be.
    graph_variants = [v for v in payload["variants"] if v.startswith("graph[")]
    if graph_variants:
        print("\n=== training-graph impact (graph vs eager-fused; speed + peak MB) ===")
        for r in payload["rows"]:
            sh = r["shape"]
            for gv in graph_variants:
                fv = "fused[" + gv[len("graph["):]
                g = r["variants"].get(gv)
                f = r["variants"].get(fv)
                if g is None or f is None:
                    continue
                speed = f["ms"] / g["ms"]
                # Compare on RESERVED (true footprint). Allocated would credit the
                # graph with a fake win — its activation pool isn't in that stat.
                mem = g["peak_reserved_mb"] / f["peak_reserved_mb"]
                print(
                    f"  {sh['kind']:<6s} B={sh['b']:<4d} S={sh['s']:<5d}  {gv}: "
                    f"{speed:.2f}x vs eager-fused, "
                    f"resv {g['peak_reserved_mb']:.0f} vs {f['peak_reserved_mb']:.0f} MB ({mem:.2f}x); "
                    f"alloc {g['peak_mb']:.0f} vs {f['peak_mb']:.0f} MB (excludes graph pool)"
                )
    print(f"\nwrote {json_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
