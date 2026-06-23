# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["flash-modernbert", "pylate", "matplotlib", "pyyaml"]
#
# [tool.uv.sources]
# flash-modernbert = { path = "../", editable = true }
# pylate = { git = "https://github.com/pau-mensa/pylate.git" }
# torch = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""Iso-loss PyLate training comparison: stock vs fused-tail (flash-modernbert).

Trains a ModernBERT ColBERT model with PyLate twice — once stock, once with the
same model `flash_modernbert.prepare()`-ed — on identical data from an identical
initialization, then reports the loss curves side by side with the speed and
peak-memory deltas.

    uv run benchmarks/iso_loss.py benchmarks/configs/synthetic_quick.yml

Everything is driven by the YAML config; see benchmarks/configs/ for examples.
The fused and stock runs share weights init, data order, optimizer, and budget,
so any divergence in the loss curves is the bf16 kernel band and any divergence
in ms/step or peak VRAM is the fused tail.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

import flash_modernbert as fm
from pylate import losses, models, utils


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
        raw = yaml.safe_load(handle) or {}
    return Config(raw)


# ---------------------------------------------------------------------------
# Data — both modes yield a `sentence_features` list [query, positive, *negs]
# ---------------------------------------------------------------------------


class SyntheticSource:
    """A fixed pool of random triplets at controlled lengths.

    Controlled shapes (no tokenization variance, exact seq lengths) make this the
    cleanest source for measuring kernel speed/memory; the pool is fixed so the
    loss still has a signal to descend.
    """

    def __init__(self, section: dict, seed: int, device: torch.device):
        self.batch_size = int(section.get("batch_size", 16))
        self.num_samples = int(section.get("num_samples", 256))
        self.num_negatives = int(section.get("num_negatives", 1))
        query_len = int(section.get("query_len", 32))
        doc_len = int(section.get("doc_len", 512))
        vocab = int(section.get("vocab", 1024))
        self.device = device
        g = torch.Generator().manual_seed(seed)
        n = self.num_samples
        self.query = torch.randint(5, vocab, (n, query_len), generator=g)
        self.positive = torch.randint(5, vocab, (n, doc_len), generator=g)
        self.negatives = [
            torch.randint(5, vocab, (n, doc_len), generator=g)
            for _ in range(self.num_negatives)
        ]

    def describe(self) -> dict:
        return {
            "type": "synthetic",
            "batch_size": self.batch_size,
            "num_samples": self.num_samples,
            "num_negatives": self.num_negatives,
            "query_len": int(self.query.shape[1]),
            "doc_len": int(self.positive.shape[1]),
        }

    def batches(self):
        """Infinite generator of sentence_features, deterministic per construction."""
        n = self.num_samples
        start = 0
        while True:
            idx = [(start + i) % n for i in range(self.batch_size)]
            start = (start + self.batch_size) % n
            yield self._features(torch.tensor(idx))

    def _features(self, idx):
        def feat(ids_pool):
            ids = ids_pool[idx].to(self.device)
            mask = torch.ones_like(ids)
            return {"input_ids": ids, "attention_mask": mask}

        return [feat(self.query), feat(self.positive), *[feat(n) for n in self.negatives]]


class HuggingFaceSource:
    """Real text triplets tokenized through the model's own ColBERT collator."""

    def __init__(self, section: dict, seed: int, device: torch.device, model):
        from datasets import load_dataset

        name = section["name"]
        split = section.get("split", "train")
        dataset = load_dataset(name, split=split)
        self.columns = section.get("columns", ["query", "positive", "negative"])
        self.batch_size = int(section.get("batch_size", 16))
        self.device = device
        self.collator = utils.ColBERTCollator(tokenize_fn=model.tokenize)
        self._name = name
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
        self.rows = [dataset[int(i)] for i in order]

    def describe(self) -> dict:
        return {
            "type": "huggingface",
            "name": self._name,
            "batch_size": self.batch_size,
            "columns": self.columns,
            "rows": len(self.rows),
        }

    def batches(self):
        start = 0
        n = len(self.rows)
        while True:
            rows = [self.rows[(start + i) % n] for i in range(self.batch_size)]
            start = (start + self.batch_size) % n
            collated = self.collator(rows)
            yield self._split(collated)

    def _split(self, collated):
        features = []
        for column in self.columns:
            feat = {}
            for suffix in ("input_ids", "attention_mask", "token_type_ids"):
                key = f"{column}_{suffix}"
                if key in collated:
                    feat[suffix] = collated[key].to(self.device)
            if feat:
                features.append(feat)
        return features


def build_source(cfg: Config, seed: int, device: torch.device, model):
    section = cfg.section("dataset")
    if section.get("type", "synthetic") == "synthetic":
        return SyntheticSource(section, seed, device)
    return HuggingFaceSource(section, seed, device, model)


# ---------------------------------------------------------------------------
# Model + loss
# ---------------------------------------------------------------------------


def build_model(cfg: Config, device: torch.device, dtype: torch.dtype):
    section = cfg.section("model")
    model = models.ColBERT(
        model_name_or_path=section["name_or_path"],
        device=str(device),
        embedding_size=int(section.get("embedding_size", 128)),
        query_length=int(section.get("query_length", 32)),
        document_length=int(section.get("document_length", 512)),
    )
    return model.to(device=device, dtype=dtype)


def build_loss(cfg: Config, model):
    section = cfg.section("loss")
    kind = section.get("type", "cached_contrastive")
    if kind == "contrastive":
        return losses.Contrastive(model=model)
    if kind == "cached_contrastive":
        batch_size = int(cfg.section("dataset").get("batch_size", 16))
        mini = int(section.get("mini_batch_size", batch_size))
        return losses.CachedContrastive(model=model, mini_batch_size=mini)
    raise ValueError(f"unknown loss type {kind!r}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_variant(cfg: Config, variant: str, fused: bool, device: torch.device) -> dict:
    run = cfg.section("run")
    seed = int(run.get("seed", 42))
    torch.manual_seed(seed)

    dtype = _dtype(cfg.section("model").get("dtype", "bfloat16"))
    use_autocast = cfg.section("model").get("dtype") == "float32"

    model = build_model(cfg, device, dtype)
    if _init_state[0] is None:
        _init_state[0] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        model.load_state_dict(_init_state[0])  # identical init to the first variant
    model = model.to(device=device, dtype=dtype)

    train_graph = bool(cfg.raw.get("train_cuda_graph", False))
    if fused:
        fm.prepare(model, cuda_graph=bool(cfg.raw.get("cuda_graph", False)),
                   train_cuda_graph=train_graph,
                   validate=bool(run.get("validate", True)))

    model.train()
    loss_fn = build_loss(cfg, model)
    source = build_source(cfg, seed, device, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.section("optimizer").get("lr", 3e-5)))

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast
        else _nullcontext()
    )

    # Training graphs replay a backward that writes into persistent `.grad` buffers,
    # so those buffers must stay live across steps — zero them in place, never to
    # None (which would free the address the captured graph writes to).
    zero_to_none = not (fused and train_graph)

    def forward_backward(features) -> float:
        optimizer.zero_grad(set_to_none=zero_to_none)
        with autocast:
            loss = loss_fn(features)
        loss.backward()
        return float(loss.detach())

    batches = source.batches()
    warmup_steps = int(run.get("warmup_steps", 8))
    max_steps = run.get("max_steps")
    max_seconds = run.get("max_seconds")
    if max_steps is None and max_seconds is None:
        raise ValueError("run.budget needs at least one of max_steps / max_seconds")

    # Warmup: pay the CuteDSL JIT + cuDNN-plan cost without advancing the weights,
    # so timing and peak memory reflect steady state and the loss curve still
    # starts from the shared init.
    warmup_batch = next(batches)
    for _ in range(warmup_steps):
        forward_backward(warmup_batch)
    optimizer.zero_grad(set_to_none=zero_to_none)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    per_step = []
    source = build_source(cfg, seed, device, model)  # replay from batch 0
    batches = source.batches()
    t_start = time.perf_counter()
    step = 0
    while True:
        if max_steps is not None and step >= int(max_steps):
            break
        if max_seconds is not None and (time.perf_counter() - t_start) >= float(max_seconds):
            break
        features = next(batches)
        torch.cuda.synchronize(device)
        s0 = time.perf_counter()
        loss_value = forward_backward(features)
        optimizer.step()
        torch.cuda.synchronize(device)
        now = time.perf_counter()
        per_step.append({
            "step": step,
            "loss": loss_value,
            "step_ms": (now - s0) * 1e3,
            "wall_s": now - t_start,
        })
        step += 1

    peak_alloc = torch.cuda.max_memory_allocated(device) / 1e9
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1e9
    summary = _summarize(per_step, source.batch_size, peak_alloc, peak_reserved)
    summary["variant"] = variant
    summary["dataset"] = source.describe()

    del model, loss_fn, optimizer, source
    gc.collect()
    torch.cuda.empty_cache()
    return {"summary": summary, "per_step": per_step}


_init_state: list = [None]  # shared init weights so both variants start identical


def _summarize(per_step, batch_size, peak_alloc, peak_reserved) -> dict:
    step_ms = [s["step_ms"] for s in per_step]
    step_ms_sorted = sorted(step_ms)
    median_ms = step_ms_sorted[len(step_ms_sorted) // 2] if step_ms_sorted else float("nan")
    total_s = per_step[-1]["wall_s"] if per_step else 0.0
    return {
        "steps": len(per_step),
        "median_step_ms": median_ms,
        "queries_per_s": (batch_size / median_ms * 1e3) if median_ms else float("nan"),
        "total_wall_s": total_s,
        "peak_mem_allocated_gb": peak_alloc,
        "peak_mem_reserved_gb": peak_reserved,
        "final_loss": _ema([s["loss"] for s in per_step])[-1] if per_step else float("nan"),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _ema(values, span: int = 10):
    if not values:
        return values
    alpha = 2.0 / (span + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def write_outputs(cfg: Config, results: dict, config_path: str):
    out = cfg.section("output")
    out_dir = Path(out.get("dir", "benchmarks/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = out.get("name", Path(config_path).stem)

    stock = results["stock"]["summary"]
    fused = results["fused"]["summary"]
    comparison = {
        "speedup_step": stock["median_step_ms"] / fused["median_step_ms"],
        "throughput_gain": fused["queries_per_s"] / stock["queries_per_s"],
        "peak_mem_allocated_ratio": fused["peak_mem_allocated_gb"] / stock["peak_mem_allocated_gb"],
        "peak_mem_allocated_saved_gb": stock["peak_mem_allocated_gb"] - fused["peak_mem_allocated_gb"],
    }
    payload = {
        "config": cfg.raw,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "results": results,
        "comparison": comparison,
    }
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    png_path = out_dir / f"{name}.png"
    _plot(results, comparison, png_path, name)
    return json_path, png_path, comparison


def _plot(results, comparison, png_path, name):
    fig, (ax_step, ax_time) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"stock": "#3b6ea5", "fused": "#d1495b"}
    for variant in ("stock", "fused"):
        per_step = results[variant]["per_step"]
        steps = [s["step"] for s in per_step]
        wall = [s["wall_s"] for s in per_step]
        loss = [s["loss"] for s in per_step]
        ema = _ema(loss)
        label = f"{variant} ({results[variant]['summary']['median_step_ms']:.1f} ms/step)"
        ax_step.plot(steps, loss, color=colors[variant], alpha=0.25)
        ax_step.plot(steps, ema, color=colors[variant], label=label)
        ax_time.plot(wall, loss, color=colors[variant], alpha=0.25)
        ax_time.plot(wall, ema, color=colors[variant], label=variant)

    ax_step.set(xlabel="training step", ylabel="loss", title="Loss vs step (correctness)")
    ax_time.set(xlabel="wall-clock seconds", ylabel="loss", title="Loss vs wall-clock (speed)")
    for ax in (ax_step, ax_time):
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{name} — fused {comparison['speedup_step']:.2f}x/step, "
        f"peak VRAM {comparison['peak_mem_allocated_ratio']:.2f}x "
        f"(saved {comparison['peak_mem_allocated_saved_gb']:.2f} GB)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers / entry point
# ---------------------------------------------------------------------------


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


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
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")

    results = {}
    for variant, fused in (("stock", False), ("fused", True)):
        print(f"\n=== training [{variant}] ===")
        results[variant] = train_variant(cfg, variant, fused, device)
        s = results[variant]["summary"]
        print(f"  steps={s['steps']} median={s['median_step_ms']:.1f}ms "
              f"queries/s={s['queries_per_s']:.1f} peak={s['peak_mem_allocated_gb']:.2f}GB "
              f"final_loss={s['final_loss']:.4f}")

    json_path, png_path, comparison = write_outputs(cfg, results, args.config)
    print(f"\nfused vs stock: {comparison['speedup_step']:.2f}x faster/step, "
          f"{comparison['peak_mem_allocated_ratio']:.2f}x peak VRAM "
          f"(saved {comparison['peak_mem_allocated_saved_gb']:.2f} GB)")
    print(f"wrote {json_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
