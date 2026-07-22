"""Two-family training showcase: stock, patched, and fully packed.

The small configs lock the visual/result contract with real AgentIR groups and
real optimizer steps. The faithful B200 configs use the official effective
batch and full two-epoch schedule.

Outputs:
  - one JSON containing every per-step measurement;
  - an iso-loss chart, faceted by model family;
  - a peak CUDA-reserved-memory bar chart, faceted by model family.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import packed_encoders as pe
from benchmarks.helpers.common import (
    RESULTS_DIR,
    device_banner,
    ema,
    load_config,
    new_figure,
)
from benchmarks.helpers.packed_collator import pack_sequences
from benchmarks.helpers.packed_index import tokenize_colbert_no_padding
from packed_encoders import forward
from packed_encoders.config import ModernBertParams
from packed_encoders.locate import find_encoder


FAMILIES = ("late_interaction", "single_vector")
VARIANTS = ("stock", "pack", "packed")
COMPILED_VARIANTS = frozenset(("pack", "packed"))
LABELS = {"stock": "Stock", "pack": "pe.pack", "packed": "Packed"}
COLORS = {"stock": "#3b6ea5", "pack": "#6a4c93", "packed": "#d1495b"}


def _compile_warmup_enabled(trainer: dict, variant: str) -> bool:
    return bool(trainer.get("compile_warmup", False)) and variant in COMPILED_VARIANTS


def _load_groups(cfg: dict) -> list[dict]:
    from datasets import Dataset

    section = cfg.get("dataset", {})
    path = Path(section.get("cache_dir", "benchmarks/data_cache"))
    # ``num_groups`` is the training pool.  Keep the fixed loss probe disjoint
    # so its curve measures progress rather than memorization of batch zero.
    count = int(section.get("num_groups", 48)) + int(section.get("num_probe_groups", 0))
    negatives = int(section.get("num_negatives", 3))
    if not path.exists():
        from benchmarks.helpers.common import load_agentir_groups

        return load_agentir_groups(
            count,
            num_negatives=negatives,
            instruction=section.get("query_instruction", ""),
        )
    dataset = Dataset.load_from_disk(str(path))
    groups = []
    for row in dataset.select(range(min(count, len(dataset)))):
        groups.append(
            {
                "query": row["query"],
                "positive": row["positive"],
                "negatives": [row[f"negative_{i}"] for i in range(negatives)],
            }
        )
    if len(groups) < count:
        raise RuntimeError(f"requested {count} groups, found {len(groups)}")
    return groups


def _load_model(cfg: dict, family: str, variant: str):
    section = cfg.get("models", {})
    if family == "late_interaction":
        from pylate import models

        model = models.ColBERT(
            model_name_or_path=section["late_interaction"],
            device="cuda",
            query_length=int(section.get("query_length", 128)),
            document_length=int(section.get("document_length", 512)),
            model_kwargs={"torch_dtype": torch.bfloat16},
        ).to(torch.bfloat16)
        tokenizer = model.tokenizer
    else:
        from transformers import AutoModel, AutoTokenizer

        name = section["single_vector"]
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name, dtype=torch.bfloat16).cuda()
    if variant == "pack":
        pe.pack(model, attention_backend="flash", validate=False)
    return model.train(), tokenizer


def _texts(groups: list[dict]) -> tuple[list[str], list[str]]:
    queries = [group["query"] for group in groups]
    documents = []
    for group in groups:
        documents.append(group["positive"])
        documents.extend(group["negatives"])
    return queries, documents


def _tokenize(cfg: dict, family: str, model, tokenizer, groups: list[dict]):
    queries, documents = _texts(groups)
    model_cfg = cfg.get("models", {})
    if family == "late_interaction":
        q = tokenize_colbert_no_padding(model, queries, is_query=True).sequences
        d = tokenize_colbert_no_padding(model, documents, is_query=False).sequences
        q_masks = tuple(torch.ones_like(ids, dtype=torch.bool) for ids in q)
        skiplist = set(int(token) for token in model.skiplist)
        d_masks = tuple(
            torch.tensor([int(token) not in skiplist for token in ids], dtype=torch.bool)
            for ids in d
        )
        return q, d, q_masks, d_masks

    def encode(texts_: list[str], cap: int):
        rows = tokenizer(
            texts_, padding=False, truncation=True, max_length=cap,
            add_special_tokens=True,
        )["input_ids"]
        return tuple(torch.tensor(row, dtype=torch.long) for row in rows)

    q = encode(queries, int(model_cfg.get("query_length", 128)))
    d = encode(documents, int(model_cfg.get("document_length", 512)))
    return q, d, None, None


def _padded_inputs(sequences, pad_id: int):
    ids = pad_sequence(sequences, batch_first=True, padding_value=pad_id).cuda()
    lengths = torch.tensor([row.numel() for row in sequences], device="cuda")
    positions = torch.arange(ids.shape[1], device="cuda")
    mask = (positions[None, :] < lengths[:, None]).to(torch.long)
    return ids, mask


def _colbert_project(model, hidden: torch.Tensor) -> torch.Tensor:
    features = {"token_embeddings": hidden}
    for module in list(model)[1:]:
        features = module(features)
    return F.normalize(features["token_embeddings"], p=2, dim=-1)


def _colbert_padded(model, sequences, scoring_masks):
    ids, attention_mask = _padded_inputs(sequences, int(model.tokenizer.pad_token_id or 0))
    features = model[0]({"input_ids": ids, "attention_mask": attention_mask})
    projected = _colbert_project(model, features["token_embeddings"])
    rows = []
    lengths = []
    for index, (sequence, score_mask) in enumerate(zip(sequences, scoring_masks)):
        selected = projected[index, : sequence.numel()][score_mask.cuda()]
        rows.append(selected)
        lengths.append(selected.shape[0])
    return (
        pad_sequence(rows, batch_first=True),
        torch.tensor(lengths, device="cuda", dtype=torch.int32),
    )


def _colbert_packed(model, sequences, scoring_masks):
    encoder = find_encoder(model)
    params = ModernBertParams.from_hf_config(encoder.config)
    batch = pack_sequences(list(sequences), device="cuda")
    hidden = forward.packed_forward(encoder, params, *batch.forward_args())
    projected = _colbert_project(model, hidden)
    score_mask = torch.cat(list(scoring_masks)).cuda()
    selected = projected[score_mask]
    lengths = torch.tensor(
        [int(mask.sum()) for mask in scoring_masks], device="cuda", dtype=torch.int32
    )
    cu = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    return selected, cu, int(lengths.max())


def _single_padded(model, tokenizer, sequences):
    ids, attention_mask = _padded_inputs(sequences, int(tokenizer.pad_token_id or 0))
    hidden = model(input_ids=ids, attention_mask=attention_mask).last_hidden_state
    return F.normalize(hidden[:, 0], p=2, dim=-1)


def _single_packed(model, sequences):
    encoder = find_encoder(model)
    params = ModernBertParams.from_hf_config(encoder.config)
    batch = pack_sequences(list(sequences), device="cuda")
    hidden = forward.packed_forward(encoder, params, *batch.forward_args())
    # GTE-ModernBERT's published embedding recipe is normalized CLS pooling.
    return F.normalize(hidden[batch.cu_seqlens[:-1].long()], p=2, dim=-1)


def _loss(cfg, family, variant, model, tokenizer, batch):
    q, d, q_masks, d_masks = batch
    if family == "late_interaction":
        temperature = float(cfg.get("loss", {}).get("late_interaction_temperature", 0.01))
        if variant == "packed":
            from benchmarks.helpers.flash_maxsim_packed import (
                flash_maxsim_packed_batched_train,
            )

            q_values, cu_q, max_q = _colbert_packed(model, q, q_masks)
            d_values, cu_d, max_d = _colbert_packed(model, d, d_masks)
            scores = flash_maxsim_packed_batched_train(
                q_values, d_values, cu_q, cu_d, max_q, max_d
            )
        else:
            from flash_maxsim import flash_maxsim_batched_train

            q_values, q_lengths = _colbert_padded(model, q, q_masks)
            d_values, d_lengths = _colbert_padded(model, d, d_masks)
            scores = flash_maxsim_batched_train(
                q_values, d_values, shared_docs=True,
                query_lengths=q_lengths, doc_lengths=d_lengths,
            )
    else:
        temperature = float(cfg.get("loss", {}).get("single_vector_temperature", 0.05))
        if variant == "packed":
            q_values = _single_packed(model, q)
            d_values = _single_packed(model, d)
        else:
            q_values = _single_padded(model, tokenizer, q)
            d_values = _single_padded(model, tokenizer, d)
        scores = q_values.float() @ d_values.float().T

    documents_per_group = len(d) // len(q)
    labels = torch.arange(len(q), device="cuda") * documents_per_group
    return F.cross_entropy(scores / temperature, labels)


def _step(cfg, family, variant, model, tokenizer, batch, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = _loss(cfg, family, variant, model, tokenizer, batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(cfg.get("trainer", {}).get("max_grad_norm", 1.0))
    )
    optimizer.step()
    return loss.detach()


def _probe_loss(cfg, family, model, tokenizer, batch):
    """Evaluate every training path through one identical reference backend."""
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            # packed_forward directly reads the encoder weights, bypassing both
            # the stock padded forward and pe.pack's patched forward.  This
            # makes step-zero and subsequent probe losses directly comparable.
            return _loss(cfg, family, "packed", model, tokenizer, batch).detach()
    finally:
        model.train(was_training)


def _make_batches(tokenized, batch_size: int):
    q_all, d_all, qm_all, dm_all = tokenized
    docs_per_group = len(d_all) // len(q_all)
    batches = []
    for begin in range(0, len(q_all) - batch_size + 1, batch_size):
        end = begin + batch_size
        db, de = begin * docs_per_group, end * docs_per_group
        batches.append(
            (
                q_all[begin:end], d_all[db:de],
                None if qm_all is None else qm_all[begin:end],
                None if dm_all is None else dm_all[db:de],
            )
        )
    return batches


def _build_gradcache_batches(model, groups, cfg, batch_size):
    from benchmarks.helpers.packed_training import (
        ColBERTTrainBatch,
        RoleBatch,
        build_colbert_train_batch,
    )

    num_negatives = int(cfg.get("dataset", {}).get("num_negatives", 7))
    full = build_colbert_train_batch(model, groups, num_negatives=num_negatives)
    dpg = full.documents_per_group
    batches = []
    for begin in range(0, len(groups) - batch_size + 1, batch_size):
        end = begin + batch_size
        db, de = begin * dpg, end * dpg
        batches.append(ColBERTTrainBatch(
            queries=RoleBatch(
                full.queries.sequences[begin:end],
                full.queries.scoring_masks[begin:end],
                True,
            ),
            documents=RoleBatch(
                full.documents.sequences[db:de],
                full.documents.scoring_masks[db:de],
                False,
            ),
            labels=torch.arange(end - begin, dtype=torch.long) * dpg,
            documents_per_group=dpg,
        ))
    return batches


def _gc_step(cfg, variant, model, batch, optimizer):
    from benchmarks.helpers.packed_training import gradcache_step

    loss_cfg = cfg.get("loss", {})
    temperature = float(loss_cfg.get("late_interaction_temperature", 0.01))
    mini_batch_size = int(loss_cfg.get("mini_batch_size", 8))
    score_mbs = loss_cfg.get("score_mini_batch_size")

    optimizer.zero_grad(set_to_none=True)
    stats = gradcache_step(
        model, batch,
        mini_batch_size=mini_batch_size,
        score_mini_batch_size=int(score_mbs) if score_mbs is not None else None,
        temperature=temperature,
        packed=(variant == "packed"),
    )
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(cfg.get("trainer", {}).get("max_grad_norm", 1.0))
    )
    optimizer.step()
    return stats.loss.detach()


def _cpu_state(model):
    """Clone a lightweight BF16 checkpoint without retaining GPU storage."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _worker(cfg: dict, family: str, variant: str, groups_path: str, output: str) -> int:
    trainer = cfg.get("trainer", {})
    dataset_cfg = cfg.get("dataset", {})
    seed = int(trainer.get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    groups = json.loads(Path(groups_path).read_text())
    batch_size = int(trainer.get("batch_size", 4))
    probe_count = int(dataset_cfg.get("num_probe_groups", 0))
    if probe_count:
        if probe_count > len(groups) - batch_size:
            raise ValueError("num_probe_groups leaves fewer than one full training batch")
        train_groups, probe_groups = groups[:-probe_count], groups[-probe_count:]
    else:
        # Backward-compatible fallback for old configs. New showcase configs
        # should always reserve a disjoint fixed probe.
        train_groups = groups
        probe_groups = groups[:batch_size]
    model, tokenizer = _load_model(cfg, family, variant)
    use_gradcache = (
        family == "late_interaction"
        and cfg.get("loss", {}).get("mini_batch_size") is not None
    )
    if use_gradcache:
        batches = _build_gradcache_batches(model, train_groups, cfg, batch_size)
    else:
        tokenized = _tokenize(cfg, family, model, tokenizer, train_groups)
        batches = _make_batches(tokenized, batch_size)
    probe_tokenized = _tokenize(cfg, family, model, tokenizer, probe_groups)
    steps = int(trainer.get("steps", 24))
    probe_batches = _make_batches(probe_tokenized, min(batch_size, len(probe_groups)))
    if not batches or len(probe_batches) != 1:
        raise RuntimeError("training and fixed-probe pools must each produce a batch")
    probe_batch = probe_batches[0]

    def do_step(batch, opt):
        if use_gradcache:
            return _gc_step(cfg, variant, model, batch, opt)
        return _step(cfg, family, variant, model, tokenizer, batch, opt)

    def new_optimizer():
        return torch.optim.AdamW(
            model.parameters(), lr=float(trainer.get("learning_rate", 3e-6)),
            weight_decay=float(trainer.get("weight_decay", 0.0)),
        )

    optimizer = new_optimizer()
    initial_state = _cpu_state(model)
    compile_warmup_seconds = 0.0
    compile_warmup_enabled = _compile_warmup_enabled(trainer, variant)
    if compile_warmup_enabled:
        # Compile/autotune every distinct real-data shape without changing the
        # measured run's initial condition. Saving on CPU also frees us to reset
        # CUDA allocator peaks after discarding the warmup optimizer states.
        torch.cuda.synchronize()
        warmup_started = time.perf_counter()
        for index, batch in enumerate(batches, 1):
            loss = do_step(batch, optimizer)
            torch.cuda.synchronize()
            print(
                f"[{family}/{variant}] compile warmup {index}/{len(batches)}: "
                f"loss={float(loss):.5f}",
                flush=True,
            )
        compile_warmup_seconds = time.perf_counter() - warmup_started
        del optimizer
        model.load_state_dict(initial_state)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        optimizer = new_optimizer()
    elif bool(trainer.get("compile_warmup", False)):
        print(f"[{family}/{variant}] path compile warmup: skipped (eager stock)", flush=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Save checkpoints on CPU and evaluate them only after measurement. Running
    # the packed reference evaluator inline perturbs CUDA allocator/kernel state
    # and therefore changes the performance it is supposed to diagnose.
    probe_states = [(0, 0.0, initial_state)]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    train_wall_s = 0.0
    per_step = []
    for index in range(steps):
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        loss = do_step(batches[index % len(batches)], optimizer)
        torch.cuda.synchronize()
        now = time.perf_counter()
        step_seconds = now - step_started
        train_wall_s += step_seconds
        row = {
            "step": index + 1,
            "loss": float(loss),
            "step_ms": step_seconds * 1e3,
            "wall_s": now - started,
            "train_wall_s": train_wall_s,
        }
        per_step.append(row)
        print(
            f"[{family}/{variant}] {index + 1:02d}/{steps}: "
            f"loss={row['loss']:.5f} step={row['step_ms']:.1f} ms",
            flush=True,
        )
        probe_every = int(trainer.get("probe_every_steps", 1))
        if (index + 1) % probe_every == 0 or index + 1 == steps:
            probe_states.append((index + 1, train_wall_s, _cpu_state(model)))

    skip = int(trainer.get("warmup_steps", 3))
    measured = [row["step_ms"] for row in per_step[skip:]]
    # The smoke/preflight traverses every distinct real-data batch once, then
    # repeats from batch zero. The repeated tail is the only honest steady-state
    # sample when FA4/CuteDSL specializes a previously unseen input shape during
    # the first traversal. Keep the configured warmup median too: the difference
    # between them is useful evidence that shape warmup was incomplete.
    repeated = [row["step_ms"] for row in per_step[len(batches) :]]
    steady_ms = statistics.median(repeated) if repeated else statistics.median(measured)
    # Freeze cumulative memory statistics from the training loop.
    memory_stats = torch.cuda.memory_stats()
    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
    peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
    # Per-step memory with a clean allocator: the CUDA caching allocator
    # accumulates freed blocks across differently-shaped packed batches,
    # inflating cumulative reserved to ~2x allocated.  Measuring each distinct
    # batch individually (empty_cache between them) reports the actual per-step
    # memory requirement without cross-batch cache accumulation.
    skip_memory = bool(trainer.get("skip_memory", False))
    memory_sample = int(trainer.get("memory_sample_count", 0))
    step_peak_reserved_gb = 0.0
    step_peak_allocated_gb = 0.0
    if skip_memory:
        step_peak_reserved_gb = peak_reserved_gb
        step_peak_allocated_gb = peak_allocated_gb
    else:
        if memory_sample and 0 < memory_sample < len(batches):
            def _batch_tokens(b):
                if use_gradcache:
                    return (
                        sum(b.queries.encoder_lengths)
                        + sum(b.documents.encoder_lengths)
                    )
                return sum(s.numel() for s in b[0]) + sum(s.numel() for s in b[1])

            indices = sorted(
                range(len(batches)),
                key=lambda i: _batch_tokens(batches[i]),
                reverse=True,
            )[:memory_sample]
            measure_batches = [batches[i] for i in indices]
            print(
                f"[{family}/{variant}] memory: sampling {len(measure_batches)}"
                f"/{len(batches)} heaviest batches",
                flush=True,
            )
        else:
            measure_batches = batches
        for batch in measure_batches:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            do_step(batch, optimizer)
            torch.cuda.synchronize()
            step_peak_reserved_gb = max(
                step_peak_reserved_gb, torch.cuda.max_memory_reserved() / 1e9
            )
            step_peak_allocated_gb = max(
                step_peak_allocated_gb, torch.cuda.max_memory_allocated() / 1e9
            )

    # The packed fixed-probe evaluator is showcase instrumentation, not part of
    # any training path. Prime it only after training and memory measurement so
    # the eager stock worker performs no compilation before its measured run.
    torch.cuda.synchronize()
    probe_warmup_started = time.perf_counter()
    probe_warmup_loss = _probe_loss(cfg, family, model, tokenizer, probe_batch)
    torch.cuda.synchronize()
    probe_warmup_seconds = time.perf_counter() - probe_warmup_started
    print(
        f"[{family}/{variant}] fixed-probe setup: "
        f"loss={float(probe_warmup_loss):.5f}, {probe_warmup_seconds:.2f}s",
        flush=True,
    )
    probe_per_step = []
    while probe_states:
        probe_step, probe_wall_s, probe_state = probe_states.pop(0)
        model.load_state_dict(probe_state)
        del probe_state
        fixed_loss = _probe_loss(cfg, family, model, tokenizer, probe_batch)
        torch.cuda.synchronize()
        probe_per_step.append(
            {
                "step": probe_step,
                "loss": float(fixed_loss),
                "train_wall_s": probe_wall_s,
            }
        )
    result = {
        "family": family,
        "variant": variant,
        "model": cfg["models"][family],
        "per_step": per_step,
        "probe_per_step": probe_per_step,
        "summary": {
            "median_step_ms": statistics.median(measured),
            "steady_repeat_median_step_ms": steady_ms,
            "cold_first_step_ms": per_step[0]["step_ms"],
            "distinct_batches_before_repeat": len(batches),
            "compile_warmup_enabled": compile_warmup_enabled,
            "compile_warmup_seconds": compile_warmup_seconds,
            "probe_warmup_seconds": probe_warmup_seconds,
            "measured_training_seconds": train_wall_s,
            "compile_accounted_training_seconds": (
                compile_warmup_seconds + train_wall_s
            ),
            "peak_reserved_gb": peak_reserved_gb,
            "peak_allocated_gb": peak_allocated_gb,
            "step_peak_reserved_gb": step_peak_reserved_gb,
            "step_peak_allocated_gb": step_peak_allocated_gb,
            "peak_inactive_split_gb": (
                memory_stats.get("inactive_split_bytes.all.peak", 0) / 1e9
            ),
            "allocator_backend": torch.cuda.get_allocator_backend(),
            "allocator_config": (
                os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
                or os.environ.get("PYTORCH_ALLOC_CONF", "")
            ),
            "final_loss": per_step[-1]["loss"],
            "initial_probe_loss": probe_per_step[0]["loss"],
            "final_probe_loss": probe_per_step[-1]["loss"],
        },
    }
    Path(output).write_text(json.dumps(result, indent=2))
    return 0


def _first_time_at_or_below(rows: list[dict], target: float) -> float | None:
    smooth = ema([row["loss"] for row in rows], span=5)
    for row, value in zip(rows, smooth):
        if value <= target:
            return row["wall_s"]
    return None


def _iso_targets(results: dict) -> dict:
    targets = {}
    for family in results:
        minima = []
        for variant in VARIANTS:
            result = results[family][variant]
            rows = result.get("probe_per_step", result["per_step"])
            minima.append(min(row["loss"] for row in rows))
        # Highest achieved minimum is the lowest target every run demonstrably reaches.
        targets[family] = max(minima)
    return targets


def _timing_comparison(results: dict) -> dict:
    """Compare real training-path time while keeping diagnostics out of it."""
    comparison = {}
    for family, family_results in results.items():
        rows = {}
        for variant in VARIANTS:
            result = family_results[variant]
            summary = result["summary"]
            measured = summary.get("measured_training_seconds")
            if measured is None:
                measured = sum(row["step_ms"] for row in result["per_step"]) / 1e3
            # Stock is eager. Only real training-path warmup for patched and
            # packed variants enters this comparison; fixed-probe setup does not.
            compile_seconds = (
                float(summary.get("compile_warmup_seconds", 0.0))
                if variant in COMPILED_VARIANTS
                else 0.0
            )
            rows[variant] = {
                "measured_training_seconds": measured,
                "path_compile_warmup_seconds": compile_seconds,
                "compile_accounted_training_seconds": measured + compile_seconds,
            }
        stock_total = rows["stock"]["compile_accounted_training_seconds"]
        for row in rows.values():
            row["speedup_vs_stock"] = (
                stock_total / row["compile_accounted_training_seconds"]
            )
        comparison[family] = rows
    return comparison


def _format_duration(seconds: float) -> str:
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _print_timing_comparison(comparison: dict) -> None:
    print("\nCompile-accounted training time (path warmup + measured training):")
    for family, rows in comparison.items():
        values = []
        for variant in VARIANTS:
            row = rows[variant]
            values.append(
                f"{LABELS[variant]} {_format_duration(row['compile_accounted_training_seconds'])} "
                f"({row['speedup_vs_stock']:.2f}x vs stock)"
            )
        print(f"  {family}: " + " | ".join(values))


def _plot_iso_loss(results: dict, targets: dict, path: Path) -> None:
    families = list(results.keys())
    plt, (fig, axes) = new_figure(len(families), 2, figsize=(13, 4.5 * len(families)), sharey="row")
    if len(families) == 1:
        axes = [axes]
    titles = {
        "late_interaction": "Late interaction — GTE-ModernColBERT-v1",
        "single_vector": "Single vector — GTE-ModernBERT-base",
    }
    for row_index, family in enumerate(families):
        step_ax, wall_ax = axes[row_index]
        for variant in VARIANTS:
            result = results[family][variant]
            rows = result.get("probe_per_step")
            fixed_probe = rows is not None
            if rows is None:  # render legacy JSON without changing its meaning
                rows = result["per_step"]
            steps = [row["step"] for row in rows]
            if fixed_probe:
                wall = [row["train_wall_s"] for row in rows]
            else:
                # Legacy training loss was observed before each update.
                wall = [
                    max(0.0, row["wall_s"] - row["step_ms"] / 1e3)
                    for row in rows
                ]
            loss = [row["loss"] for row in rows]
            summary = result["summary"]
            median_ms = summary.get("steady_repeat_median_step_ms", summary["median_step_ms"])
            label = f"{LABELS[variant]} — {median_ms:.0f} ms/step"
            for ax, x in ((step_ax, steps), (wall_ax, wall)):
                ax.plot(
                    x, loss, color=COLORS[variant], linewidth=2,
                    marker="o", markersize=3.5, label=label,
                )
        step_ax.set(
            title=f"{titles[family]} — fixed-probe loss vs step",
            xlabel="Optimizer step",
            ylabel="Fixed-probe loss",
        )
        wall_ax.set(
            title=f"{titles[family]} — fixed-probe loss vs training time",
            xlabel="Cumulative measured training time (s)",
        )
        for ax in (step_ax, wall_ax):
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle(
        "Common fixed-probe loss: equal-step parity (left) and time-to-progress (right)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_memory(results: dict, path: Path) -> None:
    families = list(results.keys())
    plt, (fig, axes) = new_figure(1, len(families), figsize=(5.75 * len(families), 4.8))
    if len(families) == 1:
        axes = [axes]
    titles = {"late_interaction": "Late interaction", "single_vector": "Single vector"}
    for ax, family in zip(axes, families):
        s = [results[family][v]["summary"] for v in VARIANTS]
        allocated = [v.get("step_peak_allocated_gb", v["peak_allocated_gb"]) for v in s]
        reserved = [v.get("step_peak_reserved_gb", v["peak_reserved_gb"]) for v in s]
        x = list(range(len(VARIANTS)))
        width = 0.36
        allocated_bars = ax.bar(
            [value - width / 2 for value in x], allocated, width,
            color=[COLORS[v] for v in VARIANTS], label="Allocated (live tensors)",
        )
        reserved_bars = ax.bar(
            [value + width / 2 for value in x], reserved, width,
            facecolor="none", edgecolor=[COLORS[v] for v in VARIANTS],
            hatch="///", linewidth=1.5, label="Reserved (allocator)",
        )
        ax.bar_label(
            allocated_bars, labels=[f"{value:.1f}" for value in allocated],
            padding=2, fontsize=8,
        )
        ax.bar_label(
            reserved_bars, labels=[f"{value:.1f}" for value in reserved],
            padding=2, fontsize=8,
        )
        ax.set_xticks(x, [LABELS[v] for v in VARIANTS])
        ax.set(title=titles[family], ylabel="Peak CUDA memory (GB)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(reserved) * 1.18)
        ax.legend(fontsize=8)
    fig.suptitle("Training memory — live tensor allocation and allocator reservation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _orchestrate(cfg: dict, config_path: str, families=None) -> dict:
    families = families or FAMILIES
    groups = _load_groups(cfg)
    all_results = {family: {} for family in families}
    with tempfile.TemporaryDirectory(prefix="training-showcase-") as tmp:
        groups_path = Path(tmp) / "groups.json"
        groups_path.write_text(json.dumps(groups))
        for family in families:
            for variant in VARIANTS:
                output = Path(tmp) / f"{family}_{variant}.json"
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.showcases.training",
                    config_path,
                    "--worker",
                    "--family",
                    family,
                    "--variant",
                    variant,
                    "--groups",
                    str(groups_path),
                    "--output",
                    str(output),
                ]
                print(f"\n=== {family} / {variant} ===", flush=True)
                worker_env = os.environ.copy()
                allocator_conf = cfg.get("trainer", {}).get(
                    "cuda_allocator_conf", ""
                )
                if allocator_conf:
                    worker_env["PYTORCH_CUDA_ALLOC_CONF"] = str(allocator_conf)
                subprocess.run(command, check=True, env=worker_env)
                all_results[family][variant] = json.loads(output.read_text())
    for family in families:
        initial = [
            all_results[family][variant]["probe_per_step"][0]["loss"]
            for variant in VARIANTS
        ]
        tolerance = max(1e-5, abs(statistics.mean(initial)) * 1e-6)
        if max(initial) - min(initial) > tolerance:
            raise RuntimeError(
                f"{family} fixed-probe initial losses differ: {initial} "
                f"(tolerance {tolerance})"
            )
    targets = _iso_targets(all_results)
    timing = _timing_comparison(all_results)
    return {
        **device_banner(),
        "config": cfg,
        "targets": targets,
        "timing_comparison": timing,
        "results": all_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--groups")
    parser.add_argument("--output")
    parser.add_argument("--families", nargs="+", choices=FAMILIES)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.worker:
        if not all((args.family, args.variant, args.groups, args.output)):
            parser.error("worker mode requires family, variant, groups, and output")
        return _worker(cfg, args.family, args.variant, args.groups, args.output)

    payload = _orchestrate(cfg, str(Path(args.config).resolve()), families=args.families)
    name = cfg.get("output", {}).get("name", Path(args.config).stem)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{name}.json"
    loss_path = RESULTS_DIR / f"{name}_iso_loss.png"
    memory_path = RESULTS_DIR / f"{name}_memory.png"
    json_path.write_text(json.dumps(payload, indent=2))
    _plot_iso_loss(payload["results"], payload["targets"], loss_path)
    _plot_memory(payload["results"], memory_path)
    _print_timing_comparison(payload["timing_comparison"])
    print(f"wrote {json_path}\nwrote {loss_path}\nwrote {memory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
