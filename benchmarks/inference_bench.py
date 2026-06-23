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
"""Inference benchmark: stock HF vs eager-fused vs graphed encoder forward.

This is the inference counterpart to `iso_loss.py` / `pylate_trainer.py`. Where
those measure a training step, this measures the **encoder forward** — the
region `prepare()` patches and the region the CUDA-graph runner captures — in
plain bf16 inference (`eval()` + `no_grad()`, no autocast), which is exactly the
regime real encode paths run in (`ColBERT.encode`, `SentenceTransformer.encode`,
`AutoModel(...)`). It loads one model, then measures three variants on the same
weights:

    stock    — the unpatched Hugging Face forward (SDPA attention)
    fused    — flash_modernbert.prepare(model): the eager fused tail
    graphed  — fused + the bucketed CUDA-graph runner (set_cuda_graph)

across two regimes the roadmap calls out:

    short S  — queries (query expansion pads every query to a fixed length, so a
               query batch is a single short bucket; this is where the host-launch
               floor collapses hardest under graphs — the headline short-S win)
    long S   — documents / indexing (compute-bound; the eager fused tail already
               wins, graphs add a smaller increment)

It reports ms/call, samples/sec, and peak memory per (variant, shape), plus the
fused- and graphed-over-stock speedups, and writes a JSON + a bar chart.

    uv run benchmarks/inference_bench.py benchmarks/configs/inference_short.yml

The numbers are the encoder forward in isolation (pre-tokenized batches at exact
shapes — no tokenization variance). Tokenization, the ColBERT projection head,
MaxSim, and L2-normalize are framework-side and unaffected by the fused tail; the
encode-path *engagement* and numerical correctness through the real text API are
proven separately by `encode_transparency.py`.
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
from flash_modernbert.graph import GraphConfig
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
# Model — built through the real framework object, encoder located via prepare's
# own locator so we exercise the exact module prepare() patches.
# ---------------------------------------------------------------------------


def build_model(cfg: Config, device: torch.device, dtype: torch.dtype):
    """Return (model_obj, encoder_module). `model_obj` is what `prepare()` takes;
    `encoder` is the located ModernBertModel we drive directly for clean timing."""
    section = cfg.section("model")
    name = section["name_or_path"]
    framework = section.get("framework", "huggingface")

    if framework == "huggingface":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        return model, model

    if framework == "pylate":
        from pylate import models

        model = models.ColBERT(
            model_name_or_path=name,
            device=str(device),
            embedding_size=int(section.get("embedding_size", 128)),
            query_length=int(section.get("query_length", 32)),
            document_length=int(section.get("document_length", 512)),
        )
        model = model.to(device=device, dtype=dtype).eval()
        return model, find_encoder(model)

    if framework == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(name, device=str(device)).to(dtype).eval()
        return model, find_encoder(model)

    raise ValueError(f"unknown framework {framework!r}")


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def make_batch(b: int, s: int, vocab: int, device: torch.device, seed: int):
    """A fixed pre-tokenized batch at an exact (B, S). attention_mask all-ones —
    the clean case (query expansion fills the query mask; un-padded doc batches
    are dense). Padding leak-freeness is covered by the package's own tests."""
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.randint(5, vocab, (b, s), generator=g, device=device)
    mask = torch.ones((b, s), dtype=torch.long, device=device)
    return ids, mask


def _last_hidden(out):
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def measure(call, ids, mask, *, n_warmup: int, n_iters: int, n_trials: int) -> dict:
    """Median ms/call over `n_trials` trials of `n_iters` calls each, plus the
    peak memory allocated across the measured window."""
    for _ in range(n_warmup):
        call(ids, mask)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times_ms = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            call(ids, mask)
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) / n_iters * 1e3)

    return {
        "ms": statistics.median(times_ms),
        "ms_min": min(times_ms),
        "ms_max": max(times_ms),
        "peak_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


def variant_calls(model_obj, encoder, graph_config: GraphConfig, validate: bool):
    """Yield (name, call) for stock, fused, graphed — built lazily so each is
    measured in its own state. The stock forward is captured before prepare()."""
    stock_forward = encoder.forward  # bound HF forward, before any patch

    @torch.no_grad()
    def stock(ids, mask):
        return _last_hidden(stock_forward(input_ids=ids, attention_mask=mask))

    yield "stock", stock

    fm.prepare(model_obj, validate=validate)  # patches encoder.forward (eager)

    @torch.no_grad()
    def fused(ids, mask):
        return _last_hidden(encoder.forward(input_ids=ids, attention_mask=mask))

    yield "fused", fused

    fm.set_cuda_graph(model_obj, True, config=graph_config)

    @torch.no_grad()
    def graphed(ids, mask):
        return _last_hidden(encoder.forward(input_ids=ids, attention_mask=mask))

    yield "graphed", graphed


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(cfg: Config, device: torch.device) -> dict:
    run_cfg = cfg.section("run")
    dtype = _DTYPES[cfg.section("model").get("dtype", "bfloat16")]
    graph_cfg = _graph_config(cfg)

    model_obj, encoder = build_model(cfg, device, dtype)
    vocab = int(encoder.config.vocab_size)
    shapes = _shapes(cfg)

    measure_kwargs = dict(
        n_warmup=int(run_cfg.get("n_warmup", 10)),
        n_iters=int(run_cfg.get("n_iters", 50)),
        n_trials=int(run_cfg.get("n_trials", 5)),
    )
    validate = bool(run_cfg.get("validate", True))

    # Pre-tokenized batches, one per shape, reused across variants.
    batches = {
        i: make_batch(sh["b"], sh["s"], vocab, device, seed=1000 + i)
        for i, sh in enumerate(shapes)
    }

    # variant -> shape_index -> result
    results: dict[str, dict[int, dict]] = {}
    for name, call in variant_calls(model_obj, encoder, graph_cfg, validate):
        results[name] = {}
        for i, sh in enumerate(shapes):
            ids, mask = batches[i]
            r = measure(call, ids, mask, **measure_kwargs)
            r["samples_per_s"] = sh["b"] * 1e3 / r["ms"]
            results[name][i] = r
            print(
                f"  {name:<8s} {sh['kind']:<8s} B={sh['b']:<4d} S={sh['s']:<5d} "
                f"{r['ms']:7.3f} ms  {r['samples_per_s']:9.0f} samp/s  "
                f"{r['peak_mb']:7.0f} MB"
            )
        print()

    del model_obj, encoder, batches
    gc.collect()
    torch.cuda.empty_cache()

    return _assemble(cfg, shapes, results)


def _assemble(cfg: Config, shapes: list[dict], results: dict) -> dict:
    rows = []
    for i, sh in enumerate(shapes):
        stock = results["stock"][i]
        fused = results["fused"][i]
        graphed = results["graphed"][i]
        rows.append({
            "shape": sh,
            "stock": stock,
            "fused": fused,
            "graphed": graphed,
            "fused_speedup": stock["ms"] / fused["ms"],
            "graphed_speedup": stock["ms"] / graphed["ms"],
            "graphed_vs_fused": fused["ms"] / graphed["ms"],
        })
    return {
        "config": cfg.raw,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "rows": rows,
    }


def _shapes(cfg: Config) -> list[dict]:
    shapes = cfg.raw.get("shapes")
    if not shapes:
        raise ValueError("config needs a non-empty `shapes` list")
    out = []
    for sh in shapes:
        out.append({
            "b": int(sh["b"]),
            "s": int(sh["s"]),
            "kind": str(sh.get("kind", "")),
        })
    return out


def _graph_config(cfg: Config) -> GraphConfig:
    section = cfg.section("graph")
    defaults = GraphConfig()
    seq_buckets = section.get("seq_buckets")
    return GraphConfig(
        pad_to=int(section.get("pad_to", defaults.pad_to)),
        max_batch=section.get("max_batch", defaults.max_batch),
        seq_buckets=tuple(seq_buckets) if seq_buckets else None,
        max_graphs=int(section.get("max_graphs", defaults.max_graphs)),
        max_tokens=int(section.get("max_tokens", defaults.max_tokens)),
        warmup=int(section.get("warmup", defaults.warmup)),
    )


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


def _plot(payload: dict, png_path: Path, name: str):
    rows = payload["rows"]
    labels = [f"{r['shape']['kind']}\nB{r['shape']['b']} S{r['shape']['s']}" for r in rows]
    x = range(len(rows))
    width = 0.27
    colors = {"stock": "#3b6ea5", "fused": "#e8a33d", "graphed": "#d1495b"}

    fig, (ax_ms, ax_sp) = plt.subplots(1, 2, figsize=(max(11, 2.2 * len(rows)), 5))

    for off, variant in zip((-width, 0, width), ("stock", "fused", "graphed")):
        ms = [r[variant]["ms"] for r in rows]
        ax_ms.bar([i + off for i in x], ms, width, label=variant, color=colors[variant])
    ax_ms.set(ylabel="ms / call (lower is better)", title="Encoder forward latency")
    ax_ms.set_xticks(list(x))
    ax_ms.set_xticklabels(labels, fontsize=8)
    ax_ms.legend()
    ax_ms.grid(True, axis="y", alpha=0.3)

    fused_sp = [r["fused_speedup"] for r in rows]
    graph_sp = [r["graphed_speedup"] for r in rows]
    ax_sp.bar([i - width / 2 for i in x], fused_sp, width, label="fused", color=colors["fused"])
    ax_sp.bar([i + width / 2 for i in x], graph_sp, width, label="graphed", color=colors["graphed"])
    ax_sp.axhline(1.0, color="#444", lw=1, ls="--")
    for i, (f, g) in enumerate(zip(fused_sp, graph_sp)):
        ax_sp.text(i - width / 2, f, f"{f:.2f}x", ha="center", va="bottom", fontsize=7)
        ax_sp.text(i + width / 2, g, f"{g:.2f}x", ha="center", va="bottom", fontsize=7)
    ax_sp.set(ylabel="speedup vs stock HF", title="Speedup vs stock")
    ax_sp.set_xticks(list(x))
    ax_sp.set_xticklabels(labels, fontsize=8)
    ax_sp.legend()
    ax_sp.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"{name} — encoder forward inference ({payload['device']}, "
        f"sm_{''.join(map(str, payload['capability']))})",
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

    print("=== fused / graphed vs stock HF ===")
    for r in payload["rows"]:
        sh = r["shape"]
        print(f"  {sh['kind']:<8s} B={sh['b']:<4d} S={sh['s']:<5d}  "
              f"fused {r['fused_speedup']:.2f}x  graphed {r['graphed_speedup']:.2f}x "
              f"(graphed/fused {r['graphed_vs_fused']:.2f}x)")
    print(f"\nwrote {json_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
