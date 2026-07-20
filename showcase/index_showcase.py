# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["packed-encoders", "pylate", "sentence-transformers", "datasets", "matplotlib", "pyyaml", "flash-attn"]
#
# [tool.uv.sources]
# packed-encoders = { path = "../", editable = true }
# pylate = { git = "https://github.com/pau-mensa/pylate.git" }
# torch = { index = "pytorch-cu128" }
# flash-attn = { url = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""Long-document indexing comparison on Tevatron/AgentIR-data.

The ST lane is a pooled-vector deployment control.  The four bars in each lane
compare eager stock, the strongest compiled padded encoder, drop-in pack with a
padded index, and a fully packed encoder/index.  MaxSim is intentionally out of
scope: indexing only creates document representations.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import packed_encoders as pe
from _common import RESULTS_DIR, device_banner, load_agentir_passages, load_config
from packed_encoders.locate import find_encoder
from packed_index import (
    PackedIndex,
    PaddedIndex,
    TokenizedTexts,
    assert_tokenization_matches_pylate,
    encode_packed,
    encode_padded,
    tokenize_colbert_no_padding,
)


# Grouped by the two figures the showcase produces.  Late interaction (PyLate)
# emits one embedding per retained document token; single vector (ST) emits one
# pooled embedding per document.  Within each family the four bars are the same
# comparison: eager stock, strongest compiled stock encoder, drop-in `pe.pack`
# padded, and a fully packed encoder/index.
LATE_INTERACTION = (
    "pylate_eager",
    "pylate_compile_dynamic",
    "pylate_pack",
    "pylate_packed",
)
SINGLE_VECTOR = ("st_eager", "st_compile_dynamic", "st_pack", "st_packed")
MAIN_VARIANTS = (*LATE_INTERACTION, *SINGLE_VECTOR)
VARIANTS = (*MAIN_VARIANTS, "st_compile", "pylate_compile")

# For each model family, the reference variant that parity candidates must match.
PARITY_FAMILIES = {
    "pylate_eager": ("pylate_compile_dynamic", "pylate_pack", "pylate_packed"),
    "st_eager": ("st_compile_dynamic", "st_pack", "st_packed"),
}


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _tree_bytes(path: str | None) -> int:
    if not path or not Path(path).exists():
        return 0
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def _tokenized_st(model, docs: list[str], document_length: int) -> TokenizedTexts:
    encoded = model.tokenizer(
        docs,
        padding=False,
        truncation=True,
        max_length=document_length,
        add_special_tokens=True,
    )
    return TokenizedTexts(
        tuple(torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]),
        is_query=False,
    )


def _next_bucket(length: int, buckets: list[int] | None) -> int:
    if buckets is None:
        return length
    for bucket in sorted(buckets):
        if bucket >= length:
            return bucket
    raise ValueError(f"no configured bucket can hold {length} tokens")


@torch.inference_mode()
def _encode_st(
    model,
    tokenized: TokenizedTexts,
    batch_size: int,
    buckets: list[int] | None,
):
    device = next(model.parameters()).device
    order = sorted(
        range(len(tokenized.sequences)),
        key=lambda i: tokenized.sequences[i].numel(),
        reverse=True,
    )
    output = None
    encoder_tokens = 0
    for offset in range(0, len(order), batch_size):
        indices = order[offset : offset + batch_size]
        length = _next_bucket(
            max(tokenized.sequences[i].numel() for i in indices), buckets
        )
        ids = torch.full(
            (len(indices), length),
            int(model.tokenizer.pad_token_id or 0),
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros_like(ids)
        for row, index in enumerate(indices):
            sequence = tokenized.sequences[index].to(device)
            ids[row, : sequence.numel()] = sequence
            mask[row, : sequence.numel()] = 1
        encoder_tokens += ids.numel()
        encoded = model({"input_ids": ids, "attention_mask": mask})[
            "sentence_embedding"
        ]
        encoded = F.normalize(encoded, p=2, dim=-1)
        if output is None:
            output = torch.empty(
                (len(tokenized.sequences), encoded.shape[-1]),
                device=device,
                dtype=encoded.dtype,
            )
        output[torch.tensor(indices, device=device)] = encoded
    if output is None:
        raise AssertionError("empty ST index")
    return output, encoder_tokens


@torch.inference_mode()
def _encode_st_packed(model, tokenized: TokenizedTexts, batch_size: int):
    """Pooled ST index with no padding: `packed_forward` then segment-wise mean.

    The padded ST path builds a `[B, S, H]` tensor and takes an attention-mask-weighted
    mean over `S`.  Here the encoder returns a flat `[total_real_tokens, H]` tensor plus
    `cu_seqlens`; the same reduction runs directly over each document's token interval.
    The pooled index is identical in size to ordinary ST — both store one `[H]` vector
    per document — so the only wins are encoder throughput and temporary VRAM.
    """
    from packed_encoders import forward
    from packed_encoders.config import ModernBertParams
    from packed_collator import pack_sequences

    encoder = find_encoder(model)
    params = ModernBertParams.from_hf_config(encoder.config)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    order = sorted(
        range(len(tokenized.sequences)),
        key=lambda i: tokenized.sequences[i].numel(),
        reverse=True,
    )
    output = None
    encoder_tokens = 0
    for offset in range(0, len(order), batch_size):
        indices = order[offset : offset + batch_size]
        batch = pack_sequences([tokenized.sequences[i] for i in indices], device=device)
        hidden = forward.packed_forward(encoder, params, *batch.forward_args())
        encoder_tokens += batch.n_tokens
        seqlens = batch.seqlens.to(torch.long)
        segments = torch.repeat_interleave(
            torch.arange(len(indices), device=device), seqlens
        )
        if output is None:
            output = torch.empty(
                (len(tokenized.sequences), hidden.shape[-1]),
                device=device,
                dtype=dtype,
            )
        # Accumulate the mean in fp32: a sequential bf16 index_add over hundreds of
        # tokens compounds rounding error with sequence length, which would otherwise
        # drift the pooled vector of the longest documents away from ST's own pooling.
        sums = torch.zeros(
            (len(indices), hidden.shape[-1]), device=device, dtype=torch.float32
        )
        sums.index_add_(0, segments, hidden.float())
        pooled = sums / seqlens.unsqueeze(1).float()
        pooled = F.normalize(pooled, p=2, dim=-1).to(dtype)
        output[torch.tensor(indices, device=device)] = pooled
    if output is None:
        raise AssertionError("empty packed ST index")
    return output, encoder_tokens


def _subset(tokenized: TokenizedTexts, count: int) -> TokenizedTexts:
    order = sorted(
        range(len(tokenized.sequences)),
        key=lambda i: tokenized.sequences[i].numel(),
        reverse=True,
    )[:count]
    return TokenizedTexts(tuple(tokenized.sequences[i] for i in order), False)


def _index_metrics(index) -> dict:
    if isinstance(index, tuple):
        embeddings, encoder_tokens = index
        logical = _tensor_bytes(embeddings)
        return {
            "representation": "pooled",
            "sequences": embeddings.shape[0],
            "encoder_tokens": encoder_tokens,
            "scoring_tokens": embeddings.shape[0],
            "embedding_bytes": logical,
            "metadata_bytes": 0,
            "storage_bytes": logical,
        }
    if isinstance(index, PaddedIndex):
        embedding_bytes = _tensor_bytes(index.embeddings)
        metadata_bytes = _tensor_bytes(index.mask)
        representation = "padded-token"
    elif isinstance(index, PackedIndex):
        embedding_bytes = _tensor_bytes(index.embeddings)
        metadata_bytes = _tensor_bytes(index.cu_seqlens)
        representation = "packed-token"
    else:
        raise TypeError(type(index))
    return {
        "representation": representation,
        "sequences": index.n_sequences,
        "encoder_tokens": index.encoder_tokens,
        "scoring_tokens": index.scoring_tokens,
        "embedding_bytes": embedding_bytes,
        "metadata_bytes": metadata_bytes,
        "storage_bytes": embedding_bytes + metadata_bytes,
    }


def _serializable(index) -> dict:
    if isinstance(index, tuple):
        return {"embeddings": index[0].detach().cpu()}
    if isinstance(index, PaddedIndex):
        return {
            "embeddings": index.embeddings.detach().cpu(),
            "mask": index.mask.detach().cpu(),
        }
    return {
        "embeddings": index.embeddings.detach().cpu(),
        "cu_seqlens": index.cu_seqlens.detach().cpu(),
    }


def _snapshot(index, count: int = 4) -> list[torch.Tensor]:
    if isinstance(index, tuple):
        return [row.detach().cpu() for row in index[0][:count]]
    return [index.sequence(i).detach().cpu() for i in range(min(count, index.n_sequences))]


def _measure(
    build_index,
    first_batch,
    all_tokens,
    *,
    warmup: int,
    trials: int,
    trial_repeats: int,
) -> tuple[object, dict]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    first = build_index(first_batch)
    torch.cuda.synchronize()
    first_batch_s = time.perf_counter() - start
    first_peak_reserved = torch.cuda.max_memory_reserved() / 1e9
    del first
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    start = time.perf_counter()
    cold_index = build_index(all_tokens)
    torch.cuda.synchronize()
    first_full_pass_s = time.perf_counter() - start
    del cold_index

    for _ in range(warmup):
        warm = build_index(all_tokens)
        del warm
    torch.cuda.synchronize()

    # Steady (already-compiled) first-batch time: isolates compiler/JIT overhead as
    # cold-first-batch minus this warm-first-batch, so the stacked startup chart can
    # separate one-time compilation from the indexing work the first batch also does.
    torch.cuda.synchronize()
    start = time.perf_counter()
    warm_first = build_index(first_batch)
    torch.cuda.synchronize()
    steady_first_batch_s = time.perf_counter() - start
    del warm_first
    jit_overhead_s = max(0.0, first_batch_s - steady_first_batch_s)

    times, peaks_reserved, peaks_allocated = [], [], []
    result = None
    for _ in range(trials):
        if result is not None:
            del result
            result = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(trial_repeats):
            if result is not None:
                del result
            result = build_index(all_tokens)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) / trial_repeats)
        peaks_reserved.append(torch.cuda.max_memory_reserved() / 1e9)
        peaks_allocated.append(torch.cuda.max_memory_allocated() / 1e9)
    assert result is not None
    seconds = statistics.median(times)
    return result, {
        "first_batch_s": first_batch_s,
        "first_batch_peak_reserved_gb": first_peak_reserved,
        "steady_first_batch_s": steady_first_batch_s,
        "jit_overhead_s": jit_overhead_s,
        "first_full_pass_s": first_full_pass_s,
        "cold_ready_s": first_batch_s + first_full_pass_s,
        "steady_seconds": seconds,
        "trial_seconds": times,
        "trial_repeats": trial_repeats,
        "docs_per_second": len(all_tokens.sequences) / seconds,
        "peak_reserved_gb": max(peaks_reserved),
        "peak_allocated_gb": max(peaks_allocated),
    }


def _build_variant(variant: str, cfg: dict, docs: list[str]):
    models_cfg = cfg.get("models", {})
    bench = cfg.get("index", {})
    document_length = int(bench.get("document_length", 2048))
    batch_size = int(bench.get("batch_size", 8))
    buckets = [int(x) for x in bench.get("length_buckets", [256, 512, 1024, 2048])]
    load_start = time.perf_counter()

    if variant in (
        "st_eager",
        "st_compile",
        "st_compile_dynamic",
        "st_pack",
        "st_packed",
    ):
        from sentence_transformers import SentenceTransformer

        model_name = models_cfg.get("sentence_transformers", "answerdotai/ModernBERT-base")
        model = SentenceTransformer(model_name, device="cuda").to(torch.bfloat16).eval()
        if variant in ("st_compile", "st_compile_dynamic"):
            encoder = find_encoder(model)
            encoder.forward = torch.compile(
                encoder.forward,
                mode="max-autotune",
                dynamic=variant == "st_compile_dynamic",
            )
        elif variant in ("st_pack", "st_packed"):
            pe.pack(model, attention_backend="flash", validate=False)
        model_load_s = time.perf_counter() - load_start
        token_start = time.perf_counter()
        tokens = _tokenized_st(model, docs, document_length)
        tokenization_s = time.perf_counter() - token_start
        if variant == "st_packed":
            build = lambda value: _encode_st_packed(model, value, batch_size)
        else:
            build = lambda value: _encode_st(
                model,
                value,
                batch_size,
                buckets if variant == "st_compile" else None,
            )
    else:
        from pylate import models

        model_name = models_cfg.get("pylate", "lightonai/Agent-ModernColBERT")
        model = models.ColBERT(
            model_name_or_path=model_name,
            device="cuda",
            document_length=document_length,
            model_kwargs={"torch_dtype": torch.bfloat16},
        ).to(torch.bfloat16).eval()
        if variant in ("pylate_compile", "pylate_compile_dynamic"):
            encoder = find_encoder(model)
            encoder.forward = torch.compile(
                encoder.forward,
                mode="max-autotune",
                dynamic=variant == "pylate_compile_dynamic",
            )
        elif variant == "pylate_pack":
            pe.pack(model, attention_backend="flash", validate=False)
        model_load_s = time.perf_counter() - load_start
        token_start = time.perf_counter()
        tokens = tokenize_colbert_no_padding(model, docs, is_query=False)
        assert_tokenization_matches_pylate(model, docs, tokens)
        tokenization_s = time.perf_counter() - token_start
        if variant == "pylate_packed":
            build = lambda value: encode_packed(
                model, value, batch_size=batch_size, sort_by_length=True
            )
        else:
            build = lambda value: encode_padded(
                model,
                value,
                batch_size=batch_size,
                sort_by_length=True,
                length_buckets=(
                    None
                    if variant in ("pylate_eager", "pylate_compile_dynamic")
                    else buckets
                ),
            )
    return model_name, model, tokens, build, model_load_s, tokenization_s


def _worker(variant: str, cfg: dict, docs: list[str], result_path: Path) -> None:
    torch.manual_seed(0)
    info = device_banner()
    bench = cfg.get("index", {})
    model_name, model, tokens, build, model_load_s, tokenization_s = _build_variant(
        variant, cfg, docs
    )
    first = _subset(tokens, min(int(bench.get("batch_size", 8)), len(tokens.sequences)))
    baseline_reserved = torch.cuda.memory_reserved() / 1e9
    index, timing = _measure(
        build,
        first,
        tokens,
        warmup=int(bench.get("warmup", 1)),
        trials=int(bench.get("trials", 3)),
        trial_repeats=int(bench.get("trial_repeats", 1)),
    )
    storage = _index_metrics(index)
    storage["storage_gb"] = storage["storage_bytes"] / 1e9
    storage["bytes_per_document"] = storage["storage_bytes"] / storage["sequences"]

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        serialized_path = Path(handle.name)
    serialize_start = time.perf_counter()
    torch.save(_serializable(index), serialized_path)
    serialization_s = time.perf_counter() - serialize_start
    serialized_bytes = serialized_path.stat().st_size
    serialized_path.unlink()

    snapshot_path = result_path.with_suffix(".snapshot.pt")
    torch.save(_snapshot(index), snapshot_path)
    payload = {
        **info,
        "variant": variant,
        "model": model_name,
        "batch_size": int(bench.get("batch_size", 8)),
        "model_load_s": model_load_s,
        "tokenization_s": tokenization_s,
        "documents": len(docs),
        "lengths": tokens.lengths,
        "baseline_reserved_gb": baseline_reserved,
        "timing": timing,
        "storage": storage,
        "serialization_s": serialization_s,
        "serialized_bytes": serialized_bytes,
        "compiler_cache_bytes": _tree_bytes(os.environ.get("TORCHINDUCTOR_CACHE_DIR")),
        "host_max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "snapshot": str(snapshot_path),
    }
    result_path.write_text(json.dumps(payload, indent=2))
    print(
        f"{variant}: {timing['docs_per_second']:.1f} docs/s, "
        f"cold-ready {timing['cold_ready_s']:.1f}s, "
        f"{timing['peak_reserved_gb']:.2f} GB, index {storage['storage_gb']:.3f} GB",
        flush=True,
    )


def _parity(results: dict[str, dict]) -> dict:
    """Gate each family's compiled/patched/packed variants against eager stock.

    Both the late-interaction (per-token) and single-vector (pooled) families are
    checked when present, each against its own unmodified eager encoder.
    """
    output = {}
    for reference_label, candidates in PARITY_FAMILIES.items():
        if reference_label not in results:
            continue
        reference = torch.load(results[reference_label]["snapshot"], map_location="cpu")
        for label in candidates:
            if label not in results:
                continue
            candidate = torch.load(results[label]["snapshot"], map_location="cpu")
            values = []
            for expected, actual in zip(reference, candidate):
                if expected.shape != actual.shape:
                    raise AssertionError(
                        f"{label} shape mismatch: {expected.shape} != {actual.shape}"
                    )
                values.append(
                    F.cosine_similarity(
                        expected.float().flatten(), actual.float().flatten(), dim=0
                    ).item()
                )
            output[label] = {"minimum_flat_cosine": min(values), "per_document": values}
            if output[label]["minimum_flat_cosine"] < 0.997:
                raise AssertionError(f"{label} parity failed: {output[label]}")
    return output


def run(cfg: dict) -> dict:
    section = cfg.get("index", {})
    docs = load_agentir_passages(
        int(section.get("num_docs", 256)), seed=int(section.get("seed", 0))
    )
    with tempfile.TemporaryDirectory(prefix="index-showcase-") as temp:
        temp_path = Path(temp)
        docs_path = temp_path / "docs.json"
        docs_path.write_text(json.dumps(docs))
        results = {}
        selected = tuple(cfg.get("variants", MAIN_VARIANTS))
        unknown = set(selected) - set(VARIANTS)
        if unknown:
            raise ValueError(f"unknown variants: {sorted(unknown)}")
        for variant in selected:
            result_path = temp_path / f"{variant}.json"
            cache_path = temp_path / f"inductor-{variant}"
            env = os.environ.copy()
            env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_path)
            env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
            env["PACKED_ENCODERS_DSL_CACHE"] = "0"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                variant,
                "--config",
                str(cfg["_config_path"]),
                "--docs",
                str(docs_path),
                "--result",
                str(result_path),
            ]
            subprocess.run(command, check=True, env=env)
            results[variant] = json.loads(result_path.read_text())
        parity_ready = any(
            reference in results and any(c in results for c in candidates)
            for reference, candidates in PARITY_FAMILIES.items()
        )
        parity = _parity(results) if parity_ready else {}
        for value in results.values():
            Path(value["snapshot"]).unlink(missing_ok=True)
    return {"dataset": "Tevatron/AgentIR-data", "variants": results, "parity": parity}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?")
    parser.add_argument("--config", dest="worker_config")
    parser.add_argument("--worker", choices=VARIANTS)
    parser.add_argument("--docs")
    parser.add_argument("--result")
    args = parser.parse_args()
    config_path = args.worker_config or args.config
    if not config_path:
        parser.error("a config path is required")
    cfg = load_config(config_path)
    if args.worker:
        _worker(
            args.worker,
            cfg,
            json.loads(Path(args.docs).read_text()),
            Path(args.result),
        )
        return
    cfg["_config_path"] = str(Path(config_path).resolve())
    output = run(cfg)
    name = cfg.get("output", {}).get("name", Path(config_path).stem)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
