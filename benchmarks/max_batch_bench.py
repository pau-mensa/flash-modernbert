# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["flash-modernbert", "transformers", "matplotlib", "pyyaml", "flash-attn"]
#
# [tool.uv.sources]
# flash-modernbert = { path = "../", editable = true }
# torch = { index = "pytorch-cu128" }
# # Prebuilt compiled flash-attn FA2 for sm_120 / torch 2.8 / cp311 — so the
# # fused[flash] frontier runs in the isolated uv-script env (mirrors pyproject).
# # On sm_90/sm_100 swap for flash-attn-4 (cute FA4); on a flash-free box, drop
# # both this line and `flash` from `attention_backends` to run sdpa only.
# flash-attn = { url = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""C1 — max-batch / OOM-frontier benchmark.

The iso-batch comparisons (`iso_loss.py`, `train_bench.py`) measure the fused
tail's memory saving *at a fixed batch*; they can only hint at the real headline.
This benchmark states it directly: **the largest fwd+bwd batch each variant fits
before it OOMs**, at a fixed sequence length. The gap is the production memory
story — "the eager fused tail trains a batch stock OOMs on" — because the
fused-tail backward keeps its intermediates in-place (in-place LayerNorm-bwd,
fused GeGLU+Wo-bwd) instead of materializing the saved-activation set HF's
autograd retains. The saving scales with the token count B·S, so it is widest
exactly here, at the frontier.

    uv run benchmarks/max_batch_bench.py benchmarks/configs/max_batch.yml

**Measured on RESERVED, not allocated.** `max_memory_allocated` undercounts a
CUDA graph's private activation pool and overstates how cheaply a variant lives
near the OOM cliff; `max_memory_reserved` is what actually decides OOM. The whole
point of an OOM frontier is the true footprint, so reserved is the only honest
stat here (the iso-batch benchmarks made the same correction).

**The unit is the bare encoder fwd+bwd** (`out.float().square().mean().backward()`,
a synthetic upstream loss touching every parameter), with no optimizer state — the
same clean kernel A/B as `train_bench.py`. Optimizer states (AdamW ≈ 2× params in
fp32) are *variant-invariant*: they lower every frontier by the same fixed amount
without changing the fused-vs-stock gap, so leaving them out hands the whole budget
to activations and shows the gap at its clearest. A real training step adds that
fixed overhead on top.

Variants:

    stock      — unpatched Hugging Face forward (SDPA attention)
    fused[b]   — flash_modernbert.prepare(model): the eager fused tail, backend b
    graph[b]   — (optional, `include_graph: true`) the training-step CUDA-graph
                 runner. Expected to OOM at a *lower* batch than eager-fused: each
                 captured shape pins its own persistent activation pool, so graphs
                 cost more reserved, not less (see docs/roadmap.md Workstream B).
                 Included only to show that honestly — graphs are a short-S latency
                 feature, never a memory/max-batch one.

Method: per (variant, S), an exponential ramp finds a batch that OOMs, then a
binary search narrows to the largest batch that fits a fwd+bwd. OOM is caught
in-process; after every probe the grads/activations are dropped and the allocator
cache is emptied so the next probe starts from a clean reserved high-water. The
frontier batch is then re-measured (ms/step + peak reserved) for the report.
"""

from __future__ import annotations

import argparse
import gc
import json
import os

# expandable_segments cuts allocator fragmentation, so the frontier reflects a
# true capacity ceiling rather than a fragmentation cliff (the false-OOM the
# research repo's max-batch probe hit). Must be set before the CUDA context inits.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import flash_modernbert as fm  # noqa: E402
from flash_modernbert.locate import find_encoder  # noqa: E402


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def section(cfg: dict, name: str) -> dict:
    return cfg.get(name, {}) or {}


# ---------------------------------------------------------------------------
# Model + batch (mirrors train_bench.py: bare HF encoder, bf16, train mode)
# ---------------------------------------------------------------------------


def build_model(cfg: dict, device: torch.device, dtype: torch.dtype):
    name = section(cfg, "model")["name_or_path"]
    from transformers import AutoModel

    model = AutoModel.from_pretrained(name, torch_dtype=dtype).to(device).train()
    return model, find_encoder(model)


def make_batch(b: int, s: int, vocab: int, device: torch.device):
    # Content is irrelevant to the memory frontier; a fixed seed keeps it reproducible.
    g = torch.Generator(device=device).manual_seed(1234)
    ids = torch.randint(5, vocab, (b, s), generator=g, device=device)
    mask = torch.ones((b, s), dtype=torch.long, device=device)
    return ids, mask


def _last_hidden(out):
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def _zero_grads(params):
    for p in params:
        p.grad = None


# ---------------------------------------------------------------------------
# One fwd+bwd probe at a given batch — returns fit/OOM + (on a measured probe)
# ms/step and peak reserved/allocated MB.
# ---------------------------------------------------------------------------


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _cleanup(params):
    _zero_grads(params)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def probe(step, params, b, s, vocab, device, *, measure: bool, n_iters: int):
    """Run a fwd+bwd at batch `b`. Returns a dict with `fit` (bool) and, when it
    fits and `measure` is set, `ms` + peak reserved/allocated MB. Any OOM (during
    the forward, the backward, or the batch alloc) is caught and reported as a
    non-fit; the allocator cache is always emptied afterward so the next probe's
    reserved high-water starts clean."""
    ids = mask = None
    try:
        ids, mask = make_batch(b, s, vocab, device)
        # Warm one step so grad buffers exist; the first backward is the peak.
        step(ids, mask)
        torch.cuda.synchronize()
        if not measure:
            return {"fit": True, "b": b}

        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step(ids, mask)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        return {
            "fit": True,
            "b": b,
            "ms": statistics.median(times),
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
            "peak_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
    except Exception as exc:  # noqa: BLE001 — OOM is the expected outcome here
        if _is_oom(exc):
            return {"fit": False, "b": b}
        raise
    finally:
        del ids, mask
        _cleanup(params)


def find_max_batch(step, params, s, vocab, device, *, start_b, max_b, n_iters):
    """Largest batch that fits a fwd+bwd at sequence length `s`.

    Exponential ramp from `start_b` (×2) until a batch OOMs (or `max_b` is reached
    and still fits — the search is capped there), then binary-search between the
    last fit and the first OOM. Returns (max_fit_b, measured_dict_at_max_fit)."""
    fit_probe = dict(measure=False, n_iters=n_iters)

    # Ramp up to bracket the frontier.
    lo = 0           # largest known-fit (0 = none yet)
    hi = None        # smallest known-OOM
    b = start_b
    while True:
        r = probe(step, params, b, s, vocab, device, **fit_probe)
        if r["fit"]:
            lo = b
            if b >= max_b:
                break  # capped: treat max_b as the frontier (didn't OOM by the cap)
            b = min(b * 2, max_b)
            if b == lo:  # already at cap and it fit
                break
        else:
            hi = b
            break
    if hi is None:  # never OOMed (hit the cap) — frontier is `lo`, capped
        capped = True
    else:
        capped = False
        # Binary search the open interval (lo, hi).
        while hi - lo > 1:
            mid = (lo + hi) // 2
            r = probe(step, params, mid, s, vocab, device, **fit_probe)
            if r["fit"]:
                lo = mid
            else:
                hi = mid

    if lo == 0:
        return 0, {"fit": False, "b": 0, "capped": False}
    measured = probe(step, params, lo, s, vocab, device, measure=True, n_iters=n_iters)
    measured["capped"] = capped
    return lo, measured


# ---------------------------------------------------------------------------
# Variants — stock (pre-patch forward), then fused[b] for each backend, then
# optionally graph[b]. One model: stock uses the bound HF forward captured before
# prepare(); fused toggles state.attention_backend (matches train_bench.py).
# ---------------------------------------------------------------------------


def step_for(forward, params):
    def step(ids, mask):
        _zero_grads(params)
        out = _last_hidden(forward(input_ids=ids, attention_mask=mask))
        loss = out.float().square().mean()
        loss.backward()
    return step


def graph_step_for(encoder, params):
    """fwd+bwd routed through the training-graph runner. Zeroes grads in place (the
    captured backward replays an add into the persistent .grad address)."""
    def step(ids, mask):
        grads = [p.grad for p in params if p.grad is not None]
        if grads:
            torch._foreach_zero_(grads)
        out = _last_hidden(encoder.forward(input_ids=ids, attention_mask=mask))
        loss = out.float().square().mean()
        loss.backward()
    return step


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(cfg: dict, device: torch.device) -> dict:
    run_cfg = section(cfg, "run")
    dtype = _DTYPES[section(cfg, "model").get("dtype", "bfloat16")]
    n_iters = int(run_cfg.get("n_iters", 5))
    start_b = int(run_cfg.get("start_b", 8))
    max_b = int(run_cfg.get("max_b", 8192))
    validate = bool(run_cfg.get("validate", True))
    backends = cfg.get("attention_backends", ["sdpa"])
    seq_lens = [int(x) for x in cfg.get("seq_lens", [512])]
    include_graph = bool(cfg.get("include_graph", False))

    model_obj, encoder = build_model(cfg, device, dtype)
    vocab = int(encoder.config.vocab_size)
    params = list(encoder.parameters())

    # Build the variant list. stock first (captured before prepare), then fused[b].
    variants: list[tuple[str, object]] = [("stock", step_for(encoder.forward, params))]

    fm.prepare(model_obj, validate=validate)
    from flash_modernbert.state import get_state

    state = get_state(model_obj)

    def set_backend(b):
        state.attention_backend = b

    results: dict[str, dict] = {}

    def search_variant(name, step, *, before=None):
        results[name] = {}
        for s in seq_lens:
            if before is not None:
                before(s)
            _cleanup(params)
            torch.cuda.reset_peak_memory_stats()
            try:
                max_fit, measured = find_max_batch(
                    step, params, s, vocab, device,
                    start_b=start_b, max_b=max_b, n_iters=n_iters,
                )
            except Exception as exc:  # noqa: BLE001 — a non-OOM backend failure (e.g.
                # a kernel that can't handle a shape) records an error for that cell
                # and lets the rest of the sweep finish, rather than aborting it.
                if _is_oom(exc):
                    raise
                _cleanup(params)
                results[name][s] = {"max_batch": 0, "fit": False, "error": str(exc)}
                print(f"  {name:<14s} S={s:<5d}  ERROR: {str(exc).splitlines()[0][:80]}")
                continue
            results[name][s] = {"max_batch": max_fit, **measured}
            tag = " (capped)" if measured.get("capped") else ""
            ms = measured.get("ms")
            resv = measured.get("peak_reserved_mb")
            extra = (f"  {ms:7.1f} ms  resv {resv:7.0f} MB"
                     if ms is not None else "")
            print(f"  {name:<14s} S={s:<5d}  max_batch={max_fit:<5d}{tag}{extra}")
        print()

    # stock + fused[b]
    search_variant("stock", variants[0][1])
    for b in backends:
        search_variant(f"fused[{b}]", step_for(encoder.forward, params),
                       before=lambda s, b=b: set_backend(b))

    # optional graph[b] frontier (expected lower — graphs pin more reserved)
    if include_graph:
        from flash_modernbert.train_graph import TrainGraphConfig, build_train_runner

        gstep = graph_step_for(encoder, params)

        def enable_graph(s, b):
            set_backend(b)
            state.train_graph_runner = build_train_runner(
                encoder, state.params, TrainGraphConfig(max_seq=2 ** 31), backend=b
            )
            state.train_graph_enabled = True

        for b in backends:
            name = f"graph[{b}]"
            results[name] = {}
            for s in seq_lens:
                _cleanup(params)
                state.train_graph_runner = None
                state.train_graph_enabled = False
                try:
                    enable_graph(s, b)
                    torch.cuda.reset_peak_memory_stats()
                    max_fit, measured = find_max_batch(
                        gstep, params, s, vocab, device,
                        start_b=start_b, max_b=max_b, n_iters=n_iters,
                    )
                except Exception as exc:  # noqa: BLE001
                    if not _is_oom(exc):
                        raise
                    max_fit, measured = 0, {"fit": False, "b": 0, "capped": False}
                finally:
                    state.train_graph_enabled = False
                    state.train_graph_runner = None
                    _cleanup(params)
                results[name][s] = {"max_batch": max_fit, **measured}
                print(f"  {name:<14s} S={s:<5d}  max_batch={max_fit:<5d}")
            print()

    del model_obj, encoder
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "config": cfg,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_mem_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "seq_lens": seq_lens,
        "variants": list(results.keys()),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(cfg: dict, payload: dict, config_path: str):
    out = section(cfg, "output")
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
    seq_lens = payload["seq_lens"]
    variants = payload["variants"]
    colors = {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(variants)}
    x = range(len(seq_lens))
    n = len(variants)
    width = 0.8 / max(n, 1)

    fig, (ax_b, ax_m) = plt.subplots(1, 2, figsize=(max(11, 3 * len(seq_lens)), 5))

    for vi, v in enumerate(variants):
        off = (vi - (n - 1) / 2) * width
        batches = [payload["results"][v][str(s) if str(s) in payload["results"][v]
                                          else s]["max_batch"] for s in seq_lens]
        bars = ax_b.bar([i + off for i in x], batches, width, label=v, color=colors[v])
        for rect, val in zip(bars, batches):
            ax_b.text(rect.get_x() + rect.get_width() / 2, val, str(val),
                      ha="center", va="bottom", fontsize=7)
    ax_b.set(ylabel="max fwd+bwd batch before OOM", title="Max-batch frontier")
    ax_b.set_xticks(list(x)); ax_b.set_xticklabels([f"S={s}" for s in seq_lens])
    ax_b.legend(fontsize=8); ax_b.grid(True, axis="y", alpha=0.3)

    # Peak reserved at each variant's own frontier batch.
    for vi, v in enumerate(variants):
        off = (vi - (n - 1) / 2) * width
        resv = []
        for s in seq_lens:
            r = payload["results"][v].get(str(s), payload["results"][v].get(s, {}))
            resv.append(r.get("peak_reserved_mb", 0) or 0)
        ax_m.bar([i + off for i in x], resv, width, label=v, color=colors[v])
    ax_m.set(ylabel="peak reserved MB at frontier batch", title="Footprint at the frontier")
    ax_m.set_xticks(list(x)); ax_m.set_xticklabels([f"S={s}" for s in seq_lens])
    ax_m.legend(fontsize=8); ax_m.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"{name} — max-batch frontier ({payload['device']}, "
        f"sm_{''.join(map(str, payload['capability']))}, "
        f"{payload['total_mem_gb']:.0f} GB, bf16, fwd+bwd, reserved)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def _print_headline(payload: dict):
    res = payload["results"]
    print("=== max-batch frontier (fwd+bwd, reserved) ===")
    for s in payload["seq_lens"]:
        stock = res["stock"].get(str(s), res["stock"].get(s, {})).get("max_batch", 0)
        line = [f"S={s:<5d}  stock {stock}"]
        for v in payload["variants"]:
            if v == "stock":
                continue
            vb = res[v].get(str(s), res[v].get(s, {})).get("max_batch", 0)
            gain = f"{vb / stock:.2f}x" if stock else "n/a"
            line.append(f"{v} {vb} ({gain})")
        print("  " + "  ".join(line))


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
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
    print(f"alloc_conf: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}\n")

    payload = run(cfg, device)
    json_path, png_path = write_outputs(cfg, payload, args.config)
    _print_headline(payload)
    print(f"\nwrote {json_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
