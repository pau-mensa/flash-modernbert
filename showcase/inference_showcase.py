# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["packed-encoders", "pylate", "sentence-transformers", "datasets", "pyyaml", "flash-attn"]
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
"""Serving-throughput comparison: compiled stock vs fm.pack vs packed.

Two endpoints, measured independently in process-isolated workers:

* **Query endpoint** -- short sequences (S~32-128).  The ``pack`` variant
  enables CUDA graphs: dispatch-bound at short S, replay eliminates launch
  overhead.
* **Document endpoint** -- long sequences (S~512-2048).  No graphs: compute-
  bound, varlen attention is the lever.

Three variants per model family (late interaction / single vector):

* Compiled stock (``max-autotune``, ``dynamic=True``): strongest deployer
  baseline.
* ``fm.pack`` (flash attention; +CUDA graph on query endpoint): drop-in.
* Packed (``packed_forward`` / segment mean-pool): paradigm.

Each variant runs in a fresh subprocess with a clean ``TORCHINDUCTOR_CACHE_DIR``.
Tokenization is done once per worker and excluded from GPU timings.
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

import packed_encoders as fm
from _common import (
    RESULTS_DIR,
    device_banner,
    load_agentir_passages,
    load_agentir_queries,
    load_config,
)
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


LATE_INTERACTION = ("pylate_compile_dynamic", "pylate_pack", "pylate_packed")
SINGLE_VECTOR = ("st_compile_dynamic", "st_pack", "st_packed")
MAIN_VARIANTS = (*LATE_INTERACTION, *SINGLE_VECTOR)

PARITY_FAMILIES = {
    "pylate_compile_dynamic": ("pylate_pack", "pylate_packed"),
    "st_compile_dynamic": ("st_pack", "st_packed"),
}

ENDPOINTS = ("query", "document")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _tree_bytes(path: str | None) -> int:
    if not path or not Path(path).exists():
        return 0
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# ST encoding (padded and packed)
# ---------------------------------------------------------------------------


def _tokenized_st(
    model, texts: list[str], max_length: int, *, is_query: bool = False
) -> TokenizedTexts:
    encoded = model.tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return TokenizedTexts(
        tuple(torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]),
        is_query=is_query,
    )


@torch.inference_mode()
def _encode_st(model, tokenized: TokenizedTexts, batch_size: int):
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
        length = max(tokenized.sequences[i].numel() for i in indices)
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
        raise AssertionError("empty ST encoding")
    return output, encoder_tokens


@torch.inference_mode()
def _encode_st_packed(model, tokenized: TokenizedTexts, batch_size: int):
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
        batch = pack_sequences(
            [tokenized.sequences[i] for i in indices], device=device
        )
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
        sums = torch.zeros(
            (len(indices), hidden.shape[-1]), device=device, dtype=torch.float32
        )
        sums.index_add_(0, segments, hidden.float())
        pooled = sums / seqlens.unsqueeze(1).float()
        pooled = F.normalize(pooled, p=2, dim=-1).to(dtype)
        output[torch.tensor(indices, device=device)] = pooled
    if output is None:
        raise AssertionError("empty packed ST encoding")
    return output, encoder_tokens


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _subset(tokenized: TokenizedTexts, count: int) -> TokenizedTexts:
    order = sorted(
        range(len(tokenized.sequences)),
        key=lambda i: tokenized.sequences[i].numel(),
        reverse=True,
    )[:count]
    return TokenizedTexts(
        tuple(tokenized.sequences[i] for i in order), tokenized.is_query
    )


def _output_metrics(index) -> dict:
    if isinstance(index, tuple):
        embeddings, encoder_tokens = index
        return {
            "representation": "pooled",
            "sequences": embeddings.shape[0],
            "encoder_tokens": encoder_tokens,
            "embedding_bytes": _tensor_bytes(embeddings),
        }
    if isinstance(index, PaddedIndex):
        return {
            "representation": "padded-token",
            "sequences": index.n_sequences,
            "encoder_tokens": index.encoder_tokens,
            "scoring_tokens": index.scoring_tokens,
            "embedding_bytes": _tensor_bytes(index.embeddings),
        }
    if isinstance(index, PackedIndex):
        return {
            "representation": "packed-token",
            "sequences": index.n_sequences,
            "encoder_tokens": index.encoder_tokens,
            "scoring_tokens": index.scoring_tokens,
            "embedding_bytes": _tensor_bytes(index.embeddings),
        }
    raise TypeError(type(index))


def _snapshot(index, count: int = 4) -> list[torch.Tensor]:
    if isinstance(index, tuple):
        return [row.detach().cpu() for row in index[0][:count]]
    return [
        index.sequence(i).detach().cpu()
        for i in range(min(count, index.n_sequences))
    ]


def _measure(
    build,
    first_batch,
    all_tokens,
    *,
    warmup: int,
    trials: int,
    trial_repeats: int,
) -> tuple[object, dict]:
    # Cold first batch (includes compile / graph capture)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    first = build(first_batch)
    torch.cuda.synchronize()
    first_batch_s = time.perf_counter() - start
    first_peak_reserved = torch.cuda.max_memory_reserved() / 1e9
    del first
    gc.collect()
    torch.cuda.empty_cache()

    # Cold first full pass
    torch.cuda.synchronize()
    start = time.perf_counter()
    cold_result = build(all_tokens)
    torch.cuda.synchronize()
    first_full_pass_s = time.perf_counter() - start
    del cold_result

    # Warmup passes
    for _ in range(warmup):
        warm = build(all_tokens)
        del warm
    torch.cuda.synchronize()

    # Steady first-batch (already-compiled/graphed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    warm_first = build(first_batch)
    torch.cuda.synchronize()
    steady_first_batch_s = time.perf_counter() - start
    del warm_first
    jit_overhead_s = max(0.0, first_batch_s - steady_first_batch_s)

    # Timed trials
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
            result = build(all_tokens)
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
        "seqs_per_second": len(all_tokens.sequences) / seconds,
        "peak_reserved_gb": max(peaks_reserved),
        "peak_allocated_gb": max(peaks_allocated),
    }


# ---------------------------------------------------------------------------
# Variant construction
# ---------------------------------------------------------------------------


def _graph_config(ep_cfg: dict, max_seq: int) -> fm.GraphConfig | bool:
    if not ep_cfg.get("cuda_graph", False):
        return False
    return fm.GraphConfig(
        pad_to=int(ep_cfg.get("cuda_graph_pad_to", 32)),
        max_seq=max_seq,
        max_batch=int(ep_cfg.get("batch_size", 8)),
    )


def _build_variant(variant: str, cfg: dict, texts: list[str], endpoint: str):
    models_cfg = cfg.get("models", {})
    ep_cfg = cfg.get(f"{endpoint}_endpoint", {})
    batch_size = int(ep_cfg.get("batch_size", 8))
    query_length = int(models_cfg.get("query_length", 128))
    document_length = int(models_cfg.get("document_length", 2048))
    is_query = endpoint == "query"
    max_length = query_length if is_query else document_length
    load_start = time.perf_counter()

    if variant.startswith("st_"):
        from sentence_transformers import SentenceTransformer

        model_name = models_cfg.get(
            "sentence_transformers", "answerdotai/ModernBERT-base"
        )
        model = (
            SentenceTransformer(model_name, device="cuda").to(torch.bfloat16).eval()
        )
        if variant == "st_compile_dynamic":
            encoder = find_encoder(model)
            encoder.forward = torch.compile(
                encoder.forward, mode="max-autotune", dynamic=True
            )
        elif variant == "st_pack":
            graph_cfg = (
                _graph_config(ep_cfg, query_length) if is_query else False
            )
            fm.pack(
                model,
                attention_backend="flash",
                validate=False,
                cuda_graph=graph_cfg,
            )
        elif variant == "st_packed":
            graph_cfg = (
                _graph_config(ep_cfg, query_length) if is_query else False
            )
            fm.pack(
                model,
                attention_backend="flash",
                validate=False,
                cuda_graph=graph_cfg,
            )
        model_load_s = time.perf_counter() - load_start

        token_start = time.perf_counter()
        tokens = _tokenized_st(model, texts, max_length, is_query=is_query)
        tokenization_s = time.perf_counter() - token_start

        if variant == "st_packed":
            build = lambda value: _encode_st_packed(model, value, batch_size)
        else:
            build = lambda value: _encode_st(model, value, batch_size)

    else:
        from pylate import models

        model_name = models_cfg.get("pylate", "lightonai/Agent-ModernColBERT")
        model = (
            models.ColBERT(
                model_name_or_path=model_name,
                device="cuda",
                query_length=query_length,
                document_length=document_length,
                model_kwargs={"torch_dtype": torch.bfloat16},
            )
            .to(torch.bfloat16)
            .eval()
        )
        if variant == "pylate_compile_dynamic":
            encoder = find_encoder(model)
            encoder.forward = torch.compile(
                encoder.forward, mode="max-autotune", dynamic=True
            )
        elif variant == "pylate_pack":
            graph_cfg = (
                _graph_config(ep_cfg, query_length) if is_query else False
            )
            fm.pack(
                model,
                attention_backend="flash",
                validate=False,
                cuda_graph=graph_cfg,
            )
        elif variant == "pylate_packed":
            graph_cfg = (
                _graph_config(ep_cfg, query_length) if is_query else False
            )
            fm.pack(
                model,
                attention_backend="flash",
                validate=False,
                cuda_graph=graph_cfg,
            )
        model_load_s = time.perf_counter() - load_start

        token_start = time.perf_counter()
        tokens = tokenize_colbert_no_padding(model, texts, is_query=is_query)
        assert_tokenization_matches_pylate(model, texts, tokens)
        tokenization_s = time.perf_counter() - token_start

        if variant == "pylate_packed":
            build = lambda value: encode_packed(
                model, value, batch_size=batch_size, sort_by_length=True
            )
        else:
            build = lambda value: encode_padded(
                model, value, batch_size=batch_size, sort_by_length=True
            )

    return model_name, model, tokens, build, model_load_s, tokenization_s


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------


def _worker(
    variant: str,
    endpoint: str,
    cfg: dict,
    texts: list[str],
    result_path: Path,
) -> None:
    torch.manual_seed(0)
    info = device_banner()
    ep_cfg = cfg.get(f"{endpoint}_endpoint", {})
    batch_size = int(ep_cfg.get("batch_size", 8))

    model_name, model, tokens, build, model_load_s, tokenization_s = (
        _build_variant(variant, cfg, texts, endpoint)
    )
    first = _subset(
        tokens, min(batch_size, len(tokens.sequences))
    )
    baseline_reserved = torch.cuda.memory_reserved() / 1e9
    index, timing = _measure(
        build,
        first,
        tokens,
        warmup=int(ep_cfg.get("warmup", 1)),
        trials=int(ep_cfg.get("trials", 5)),
        trial_repeats=int(ep_cfg.get("trial_repeats", 10)),
    )
    storage = _output_metrics(index)

    snapshot_path = result_path.with_suffix(".snapshot.pt")
    torch.save(_snapshot(index), snapshot_path)
    payload = {
        **info,
        "variant": variant,
        "endpoint": endpoint,
        "model": model_name,
        "batch_size": batch_size,
        "model_load_s": model_load_s,
        "tokenization_s": tokenization_s,
        "sequences": len(texts),
        "lengths": tokens.lengths,
        "baseline_reserved_gb": baseline_reserved,
        "timing": timing,
        "storage": storage,
        "compiler_cache_bytes": _tree_bytes(
            os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        ),
        "host_max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1e6,
        "snapshot": str(snapshot_path),
    }
    result_path.write_text(json.dumps(payload, indent=2))
    print(
        f"{variant}/{endpoint}: {timing['seqs_per_second']:.1f} seqs/s, "
        f"cold-ready {timing['cold_ready_s']:.1f}s, "
        f"{timing['peak_reserved_gb']:.2f} GB",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def _parity(results: dict[str, dict], *, floor: float = 0.997) -> dict:
    output = {}
    for reference_label, candidates in PARITY_FAMILIES.items():
        if reference_label not in results:
            continue
        reference = torch.load(
            results[reference_label]["snapshot"], map_location="cpu"
        )
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
                        expected.float().flatten(),
                        actual.float().flatten(),
                        dim=0,
                    ).item()
                )
            output[label] = {
                "minimum_flat_cosine": min(values),
                "per_sequence": values,
            }
            if output[label]["minimum_flat_cosine"] < floor:
                raise AssertionError(
                    f"{label} parity below {floor}: {output[label]}"
                )
    return output


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run(cfg: dict) -> dict:
    selected = tuple(cfg.get("variants", MAIN_VARIANTS))
    unknown = set(selected) - set(MAIN_VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")

    instruction = cfg.get("dataset", {}).get("query_instruction", "")
    seed = int(cfg.get("seed", 0))
    query_cfg = cfg.get("query_endpoint", {})
    doc_cfg = cfg.get("document_endpoint", {})

    queries = load_agentir_queries(
        int(query_cfg.get("num_texts", 256)), instruction=instruction
    )
    docs = load_agentir_passages(
        int(doc_cfg.get("num_texts", 256)), seed=seed
    )

    with tempfile.TemporaryDirectory(prefix="inference-showcase-") as temp:
        temp_path = Path(temp)
        data = {
            "query": (queries, temp_path / "queries.json"),
            "document": (docs, temp_path / "docs.json"),
        }
        for texts, path in data.values():
            path.write_text(json.dumps(texts))

        all_results = {}
        for endpoint in ENDPOINTS:
            ep_cfg = cfg.get(f"{endpoint}_endpoint", {})
            if not ep_cfg:
                continue
            _, data_path = data[endpoint]
            results = {}
            for variant in selected:
                result_path = temp_path / f"{variant}_{endpoint}.json"
                cache_path = temp_path / f"inductor-{variant}-{endpoint}"
                env = os.environ.copy()
                env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_path)
                env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
                env["PACKED_ENCODERS_DSL_CACHE"] = "0"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    variant,
                    "--endpoint",
                    endpoint,
                    "--config",
                    str(cfg["_config_path"]),
                    "--data",
                    str(data_path),
                    "--result",
                    str(result_path),
                ]
                subprocess.run(command, check=True, env=env)
                results[variant] = json.loads(result_path.read_text())

            parity_ready = any(
                ref in results and any(c in results for c in cands)
                for ref, cands in PARITY_FAMILIES.items()
            )
            parity_floor = float(ep_cfg.get("parity_floor", 0.997))
            parity = (
                _parity(results, floor=parity_floor)
                if parity_ready
                else {}
            )
            for value in results.values():
                Path(value["snapshot"]).unlink(missing_ok=True)
            all_results[endpoint] = {
                "variants": results,
                "parity": parity,
            }

    return {
        "dataset": "Tevatron/AgentIR-data",
        "endpoints": all_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?")
    parser.add_argument("--config", dest="worker_config")
    parser.add_argument("--worker")
    parser.add_argument("--endpoint", choices=ENDPOINTS)
    parser.add_argument("--data")
    parser.add_argument("--result")
    args = parser.parse_args()

    config_path = args.worker_config or args.config
    if not config_path:
        parser.error("a config path is required")
    cfg = load_config(config_path)

    if args.worker:
        _worker(
            args.worker,
            args.endpoint,
            cfg,
            json.loads(Path(args.data).read_text()),
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
