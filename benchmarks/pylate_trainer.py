# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["flash-modernbert", "pylate", "matplotlib", "pyyaml"]
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
"""Native SentenceTransformerTrainer recipe, run stock vs flash-modernbert.

Ordinary training boilerplate around the real `SentenceTransformerTrainer` (which
runs on accelerate). It runs the exact recipe twice from an identical weight init
on identical data, and **the only line that differs between the two runs is
`flash_modernbert.prepare(model)`**:

    model = build_model(cfg)               # PyLate ColBERT or a SentenceTransformer
    if fused:
        fm.prepare(model, cuda_graph=...)  # <-- the entire difference
    trainer = SentenceTransformerTrainer(model=model, args=args, ...)
    trainer.train()

`model.framework` (C2 — framework transparency) selects which framework object the
recipe is built on, proving Decision-1 (the in-place monkeypatch) is transparent
across frameworks — the same one-line `prepare()` accelerates each with no adapter:

    pylate                — `models.ColBERT` + `losses.CachedContrastive` (GradCache),
                            tokenized by `utils.ColBERTCollator` (the default)
    sentence_transformers — `SentenceTransformer` (mean-pooling bi-encoder) +
                            `MultipleNegativesRankingLoss`, tokenized by the
                            trainer's default `SentenceTransformerDataCollator`

Both subclass `SentenceTransformer`, so both run on the same trainer; `prepare()`
locates and patches the wrapped `ModernBertModel` either way. (Raw HF `AutoModel`
training transparency is `train_bench.py` — an AutoModel can't drive the
SentenceTransformerTrainer; inference transparency across all three is
`inference_bench.py`'s `framework` knob + `encode_transparency.py`.)

A `TrainerCallback` records per-step loss / step time / peak VRAM from inside the
real training loop, and the two runs are written out as loss curves plus the
speed and memory deltas.

    uv run benchmarks/pylate_trainer.py benchmarks/configs/msmarco_trainer.yml  # pylate
    uv run benchmarks/pylate_trainer.py benchmarks/configs/st_trainer.yml       # sentence_transformers
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from transformers import TrainerCallback

import flash_modernbert as fm


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


# ---------------------------------------------------------------------------
# The recipe — identical for both runs
# ---------------------------------------------------------------------------


def framework_of(cfg: Config) -> str:
    return cfg.section("model").get("framework", "pylate")


def build_model(cfg: Config):
    """Build the framework object `prepare()` will patch. Both branches return a
    `SentenceTransformer` subclass (ColBERT is one), so both drive the same
    `SentenceTransformerTrainer`; `prepare()` locates the wrapped ModernBertModel
    in either."""
    section = cfg.section("model")
    framework = framework_of(cfg)
    if framework == "pylate":
        from pylate import models

        return models.ColBERT(
            model_name_or_path=section["name_or_path"],
            device="cuda",
            embedding_size=int(section.get("embedding_size", 128)),
            query_length=int(section.get("query_length", 32)),
            document_length=int(section.get("document_length", 512)),
        )
    if framework == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        # Plain bi-encoder: a mean-pooling head over the same ModernBERT encoder
        # (ModernBERT-base ships no ST pooling config, so ST adds mean pooling —
        # fine for a transparency/speed A/B; the encoder is what prepare() patches).
        model = SentenceTransformer(section["name_or_path"], device="cuda")
        max_len = section.get("max_seq_length")
        if max_len is not None:
            model.max_seq_length = int(max_len)
        return model
    raise ValueError(
        f"unknown framework {framework!r} (pylate | sentence_transformers); raw HF "
        "AutoModel training transparency is benchmarks/train_bench.py"
    )


def build_loss(cfg: Config, model):
    """The framework's idiomatic contrastive loss. PyLate's reads the ColBERT
    token embeddings (MaxSim); SentenceTransformers' reads the pooled bi-encoder
    embedding. Both take in-batch negatives and the explicit negatives in the
    triplet, so the recipe shape (anchor, positive, negatives) is shared."""
    section = cfg.section("loss")
    kind = section.get("type", "cached_contrastive")
    framework = framework_of(cfg)
    batch_size = int(cfg.section("trainer").get("batch_size", 16))
    mini = int(section.get("mini_batch_size", batch_size))

    if framework == "pylate":
        from pylate import losses

        if kind == "contrastive":
            return losses.Contrastive(model=model)
        if kind == "cached_contrastive":
            return losses.CachedContrastive(model=model, mini_batch_size=mini)
        raise ValueError(f"unknown pylate loss type {kind!r}")

    # sentence_transformers: MultipleNegativesRankingLoss is the bi-encoder
    # analogue of Contrastive; its Cached* variant is the GradCache analogue of
    # CachedContrastive (same memory regime — chunked, no-grad-then-recompute).
    from sentence_transformers import losses as st_losses

    if kind == "contrastive":
        return st_losses.MultipleNegativesRankingLoss(model=model)
    if kind == "cached_contrastive":
        return st_losses.CachedMultipleNegativesRankingLoss(
            model=model, mini_batch_size=mini
        )
    raise ValueError(f"unknown sentence_transformers loss type {kind!r}")


def build_dataset(cfg: Config):
    section = cfg.section("dataset")
    if section.get("type", "huggingface") == "synthetic":
        return _synthetic_triplets(section)
    dataset = load_dataset(
        section["name"], section.get("config"), split=section.get("split", "train")
    )
    take = section.get("select")
    if take is not None:
        dataset = dataset.select(range(min(int(take), len(dataset))))
    if section.get("format", "triplet") == "agent_ir":
        dataset = _agent_ir_to_triplets(
            dataset,
            num_negatives=int(section.get("num_negatives", 7)),
            instruction=section.get("query_instruction", ""),
        )
    return dataset


def _synthetic_triplets(section: dict):
    """An in-memory `(query, positive, negative_i)` text dataset of random words —
    no network or dataset-schema dependency, so the framework-transparency A/B is
    guaranteed to run anywhere. The loss reads the columns positionally
    [anchor, positive, *negatives]; semantics are irrelevant to a speed/memory A/B
    (both variants see identical data from the shared seed)."""
    import random

    from datasets import Dataset

    n = int(section.get("num_samples", 512))
    num_neg = int(section.get("num_negatives", 1))
    q_words = int(section.get("query_words", 8))
    d_words = int(section.get("doc_words", 90))   # ≈110 ModernBERT tokens
    vocab = [f"tok{i}" for i in range(int(section.get("vocab", 2000)))]
    rng = random.Random(int(section.get("seed", 42)))

    def sentence(k):
        return " ".join(rng.choice(vocab) for _ in range(k))

    cols: dict[str, list] = {"query": [], "positive": []}
    for i in range(num_neg):
        cols[f"negative_{i}"] = []
    for _ in range(n):
        cols["query"].append(sentence(q_words))
        cols["positive"].append(sentence(d_words))
        for i in range(num_neg):
            cols[f"negative_{i}"].append(sentence(d_words))
    return Dataset.from_dict(cols)


def _agent_ir_to_triplets(dataset, num_negatives: int, instruction: str):
    """Flatten Tevatron AgentIR (query / positive_passages / negative_passages
    lists) into the flat `query` / `positive` / `negative_i` text columns the
    ColBERTCollator and the contrastive loss consume — the loss reads them as
    [anchor, positive, *negatives] in column order."""
    dataset = dataset.filter(lambda r: r["positive_passages"] and r["negative_passages"])
    neg_cols = [f"negative_{i}" for i in range(num_negatives)]

    def to_columns(row):
        negs = row["negative_passages"]
        out = {
            "query": instruction + row["query"],
            "positive": row["positive_passages"][0]["text"],
        }
        for i, col in enumerate(neg_cols):
            out[col] = negs[i % len(negs)]["text"]  # cycle if a row has fewer
        return out

    return dataset.map(to_columns, remove_columns=dataset.column_names)


def build_training_args(cfg: Config, output_dir: str, max_steps: int) -> SentenceTransformerTrainingArguments:
    trainer_cfg = cfg.section("trainer")
    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        num_train_epochs=10 ** 6,  # let max_steps / the time callback bound the run
        per_device_train_batch_size=int(trainer_cfg.get("batch_size", 16)),
        gradient_accumulation_steps=int(trainer_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(trainer_cfg.get("learning_rate", 3e-5)),
        warmup_ratio=0.0,
        bf16=bool(trainer_cfg.get("bf16", True)),
        seed=int(trainer_cfg.get("seed", 42)),
        data_seed=int(trainer_cfg.get("seed", 42)),
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        disable_tqdm=True,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )


# ---------------------------------------------------------------------------
# Metrics from inside the real training loop
# ---------------------------------------------------------------------------


class MetricsCallback(TrainerCallback):
    """Records per-step loss / wall time / peak VRAM via the Trainer's hooks."""

    def __init__(self, device: torch.device, max_seconds: float | None):
        self.device = device
        self.max_seconds = max_seconds
        self.per_step: list[dict] = []
        self._t0 = 0.0
        self._step_start = 0.0
        self._last_step_ms = float("nan")
        self._last_wall = float("nan")
        self.peak_alloc = float("nan")
        self.peak_reserved = float("nan")

    def on_train_begin(self, args, state, control, **kw):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._t0 = time.perf_counter()

    def on_step_begin(self, args, state, control, **kw):
        torch.cuda.synchronize(self.device)
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self._last_step_ms = (now - self._step_start) * 1e3
        self._last_wall = now - self._t0
        if self.max_seconds is not None and self._last_wall >= self.max_seconds:
            control.should_training_stop = True

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.per_step.append({
                "step": state.global_step,
                "loss": float(logs["loss"]),
                "step_ms": self._last_step_ms,
                "wall_s": self._last_wall,
            })

    def on_train_end(self, args, state, control, **kw):
        self.peak_alloc = torch.cuda.max_memory_allocated(self.device) / 1e9
        self.peak_reserved = torch.cuda.max_memory_reserved(self.device) / 1e9


# ---------------------------------------------------------------------------
# One run of the recipe
# ---------------------------------------------------------------------------


def run_recipe(cfg: Config, variant: str, fused: bool, init_state: list, device: torch.device) -> dict:
    run = cfg.section("run")
    seed = int(cfg.section("trainer").get("seed", 42))
    torch.manual_seed(seed)

    model = build_model(cfg)
    # Pin both runs to the same weights so the only behavioral difference is
    # prepare() (the ColBERT projection / any head is otherwise randomly initialized).
    if init_state[0] is None:
        init_state[0] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        model.load_state_dict(init_state[0])

    if fused:
        fm.prepare(model, cuda_graph=bool(cfg.raw.get("cuda_graph", False)),
                   validate=bool(run.get("validate", True)))

    loss = build_loss(cfg, model)
    dataset = build_dataset(cfg)
    max_steps = int(run["max_steps"]) if run.get("max_steps") is not None else 10 ** 6
    callback = MetricsCallback(device, run.get("max_seconds"))

    # PyLate tokenizes through its ColBERTCollator; SentenceTransformers lets the
    # trainer pick its default SentenceTransformerDataCollator (data_collator=None).
    collator = None
    if framework_of(cfg) == "pylate":
        from pylate import utils

        collator = utils.ColBERTCollator(model.tokenize)

    with tempfile.TemporaryDirectory() as tmp:
        args = build_training_args(cfg, tmp, max_steps)
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            loss=loss,
            data_collator=collator,
            callbacks=[callback],
        )
        trainer.train()

    summary = _summarize(
        callback.per_step,
        batch_size=int(cfg.section("trainer").get("batch_size", 16)),
        peak_alloc=callback.peak_alloc,
        peak_reserved=callback.peak_reserved,
        skip=int(run.get("warmup_steps", 8)),
    )
    summary["variant"] = variant
    result = {"summary": summary, "per_step": callback.per_step}

    del trainer, model, loss, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _summarize(per_step, batch_size, peak_alloc, peak_reserved, skip) -> dict:
    measured = per_step[skip:] if len(per_step) > skip else per_step
    step_ms = sorted(s["step_ms"] for s in measured if s["step_ms"] == s["step_ms"])
    median_ms = step_ms[len(step_ms) // 2] if step_ms else float("nan")
    return {
        "steps": len(per_step),
        "measured_steps": len(measured),
        "median_step_ms": median_ms,
        "queries_per_s": (batch_size / median_ms * 1e3) if median_ms == median_ms else float("nan"),
        "total_wall_s": per_step[-1]["wall_s"] if per_step else 0.0,
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

    stock, fused = results["stock"]["summary"], results["fused"]["summary"]
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
        f"{name} (SentenceTransformerTrainer) — fused {comparison['speedup_step']:.2f}x/step, "
        f"peak VRAM {comparison['peak_mem_allocated_ratio']:.2f}x "
        f"(saved {comparison['peak_mem_allocated_saved_gb']:.2f} GB)",
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
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")

    init_state: list = [None]  # shared weights so the only difference is prepare()
    results = {}
    for variant, fused in (("stock", False), ("fused", True)):
        print(f"\n=== SentenceTransformerTrainer [{variant}] ===")
        results[variant] = run_recipe(cfg, variant, fused, init_state, device)
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
