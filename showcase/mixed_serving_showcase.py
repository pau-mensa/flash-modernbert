#!/usr/bin/env python3
"""Saturated-window mixed query/document serving showcase.

This intentionally small benchmark answers one question: given the same sequence of
mixed 80/20 arrival windows on one GPU, what is gained by replacing conventional
query/document execution with one packed forward?

Three paths make the attribution honest:

* ``split_padded``  -- conventional semantics, two padded framework forwards;
* ``split_packed``  -- two packed forwards, isolating the padding/data-layout effect;
* ``unified_packed`` -- one mixed packed forward, isolating forward unification.

It is a saturated serving-window throughput benchmark, not an HTTP/load-generator
simulation. Tokenization and window construction are outside measured GPU time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

import packed_encoders as pe
from _common import (
    RESULTS_DIR,
    device_banner,
    load_agentir_passages,
    load_agentir_queries,
    load_config,
)
from packed_collator import PackedBatch, pack_sequences
from packed_encoders import forward
from packed_encoders.config import ModernBertParams
from packed_encoders.locate import find_encoder
from packed_index import assert_tokenization_matches_pylate, tokenize_colbert_no_padding


Kind = Literal["query", "document"]
VARIANTS = ("split_padded", "split_packed", "unified_packed")


@dataclass(frozen=True)
class Payload:
    payload_id: int
    kind: Kind
    input_ids: torch.Tensor
    content_hash: str


@dataclass(frozen=True)
class Request:
    request_id: int
    kind: Kind
    payload_id: int


@dataclass
class PreparedWindow:
    requests: list[Request]
    sequences: list[torch.Tensor]
    query_indices: list[int]
    document_indices: list[int]
    query_ids: torch.Tensor
    query_mask: torch.Tensor
    document_ids: torch.Tensor
    document_mask: torch.Tensor
    query_packed: PackedBatch
    document_packed: PackedBatch
    unified_packed: PackedBatch

    @property
    def real_tokens(self) -> int:
        return sum(sequence.numel() for sequence in self.sequences)

    @property
    def padded_tokens(self) -> int:
        return self.query_ids.numel() + self.document_ids.numel()


def _token_hash(kind: Kind, ids: torch.Tensor) -> str:
    value = kind.encode() + b"\0" + ids.numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1 - position + lower) + ordered[upper] * (position - lower)


def _length_summary(payloads: list[Payload], kind: Kind) -> dict:
    values = [p.input_ids.numel() for p in payloads if p.kind == kind]
    return {
        "count": len(values),
        "min": min(values),
        "median": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
        "total_tokens": sum(values),
    }


def _pad(sequences: list[torch.Tensor], pad_id: int, device: torch.device):
    length = max(sequence.numel() for sequence in sequences)
    ids = torch.full((len(sequences), length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        ids[row, : sequence.numel()] = sequence.to(device)
        mask[row, : sequence.numel()] = 1
    return ids, mask


def _build_payloads(model, cfg: dict) -> list[Payload]:
    data = cfg["dataset"]
    queries = load_agentir_queries(
        int(data["queries"]), instruction=data.get("query_instruction", ""),
        streaming=bool(data.get("streaming", True)),
    )
    documents = load_agentir_passages(
        int(data["documents"]), seed=int(cfg.get("seed", 0)),
        streaming=bool(data.get("streaming", True)),
    )
    payloads = []
    for kind, texts in (("query", queries), ("document", documents)):
        tokenized = tokenize_colbert_no_padding(model, texts, is_query=kind == "query")
        assert_tokenization_matches_pylate(model, texts, tokenized)
        for ids in tokenized.sequences:
            ids = ids.contiguous()
            payloads.append(Payload(len(payloads), kind, ids, _token_hash(kind, ids)))
    return payloads


def _prepare_windows(model, payloads: list[Payload], cfg: dict) -> list[PreparedWindow]:
    benchmark = cfg["benchmark"]
    batch_size = int(benchmark["requests_per_window"])
    query_fraction = float(benchmark["query_fraction"])
    n_query = round(batch_size * query_fraction)
    n_document = batch_size - n_query
    if not n_query or not n_document:
        raise ValueError("each window must contain both query and document requests")
    by_kind = {
        kind: [p for p in payloads if p.kind == kind]
        for kind in ("query", "document")
    }
    rng = random.Random(int(cfg.get("seed", 0)))
    device = next(model.parameters()).device
    pad_id = int(model.tokenizer.pad_token_id or 0)
    windows = []
    next_request_id = 0
    for _ in range(int(benchmark["windows"])):
        chosen = rng.sample(by_kind["query"], n_query) + rng.sample(by_kind["document"], n_document)
        rng.shuffle(chosen)
        requests = [Request(next_request_id + i, value.kind, value.payload_id) for i, value in enumerate(chosen)]
        next_request_id += len(requests)
        sequences = [value.input_ids for value in chosen]
        query_indices = [i for i, value in enumerate(chosen) if value.kind == "query"]
        document_indices = [i for i, value in enumerate(chosen) if value.kind == "document"]
        query_sequences = [sequences[i] for i in query_indices]
        document_sequences = [sequences[i] for i in document_indices]
        query_ids, query_mask = _pad(query_sequences, pad_id, device)
        document_ids, document_mask = _pad(document_sequences, pad_id, device)
        windows.append(PreparedWindow(
            requests=requests,
            sequences=sequences,
            query_indices=query_indices,
            document_indices=document_indices,
            query_ids=query_ids,
            query_mask=query_mask,
            document_ids=document_ids,
            document_mask=document_mask,
            query_packed=pack_sequences(query_sequences, device=device),
            document_packed=pack_sequences(document_sequences, device=device),
            unified_packed=pack_sequences(sequences, device=device),
        ))
    return windows


class Benchmark:
    def __init__(self, model, windows: list[PreparedWindow]):
        self.model = model
        self.windows = windows
        self.encoder = find_encoder(model)
        self.params = ModernBertParams.from_hf_config(self.encoder.config)
        self.skiplist = set(int(x) for x in model.skiplist)

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        features = {"token_embeddings": hidden}
        for module in list(self.model)[1:]:
            features = module(features)
        return F.normalize(features["token_embeddings"], p=2, dim=-1)

    def _padded(self, ids: torch.Tensor, mask: torch.Tensor, kind: Kind):
        features = {"input_ids": ids, "attention_mask": mask}
        for module in self.model:
            features = module(features)
        embeddings = F.normalize(features["token_embeddings"], p=2, dim=-1)
        scoring = mask.bool()
        if kind == "document":
            for token in self.skiplist:
                scoring.logical_and_(ids != token)
        checksum = (torch.round(embeddings.float() * 1024).sum(-1) * scoring).sum()
        return embeddings, scoring, checksum

    def _packed(self, batch: PackedBatch, kinds: list[Kind]):
        hidden = forward.packed_forward(self.encoder, self.params, *batch.forward_args())
        embeddings = self._project(hidden)
        masks = []
        for kind, start, end in zip(kinds, batch.cu_seqlens[:-1], batch.cu_seqlens[1:]):
            length = int(end - start)
            if kind == "query":
                masks.append(torch.ones(length, dtype=torch.bool, device=embeddings.device))
            else:
                ids = batch.input_ids[int(start):int(end)]
                keep = torch.ones(length, dtype=torch.bool, device=embeddings.device)
                for token in self.skiplist:
                    keep.logical_and_(ids != token)
                masks.append(keep)
        scoring = torch.cat(masks)
        checksum = (torch.round(embeddings.float() * 1024).sum(-1) * scoring).sum()
        return embeddings, scoring, checksum

    @torch.inference_mode()
    def run_window(self, variant: str, window: PreparedWindow):
        if variant == "split_padded":
            q = self._padded(window.query_ids, window.query_mask, "query")
            d = self._padded(window.document_ids, window.document_mask, "document")
            return q[2] + d[2]
        if variant == "split_packed":
            q = self._packed(window.query_packed, ["query"] * len(window.query_indices))
            d = self._packed(window.document_packed, ["document"] * len(window.document_indices))
            return q[2] + d[2]
        if variant == "unified_packed":
            value = self._packed(window.unified_packed, [request.kind for request in window.requests])
            return value[2]
        raise ValueError(variant)

    @torch.inference_mode()
    def correctness(self, floor: float) -> dict:
        window = self.windows[0]
        reference: dict[int, torch.Tensor] = {}
        for kind, indices, ids, mask in (
            ("query", window.query_indices, window.query_ids, window.query_mask),
            ("document", window.document_indices, window.document_ids, window.document_mask),
        ):
            embeddings, scoring, _ = self._padded(ids, mask, kind)
            for row, index in enumerate(indices):
                reference[index] = embeddings[row][scoring[row]].detach()
        embeddings, scoring, _ = self._packed(
            window.unified_packed, [request.kind for request in window.requests]
        )
        cosines = []
        output_lengths = []
        for index, (start, end) in enumerate(zip(
            window.unified_packed.cu_seqlens[:-1], window.unified_packed.cu_seqlens[1:]
        )):
            start, end = int(start), int(end)
            actual = embeddings[start:end][scoring[start:end]]
            expected = reference[index]
            if actual.shape != expected.shape:
                raise AssertionError(f"output shape mismatch at request {index}: {actual.shape} != {expected.shape}")
            cosines.append(float(F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)))
            output_lengths.append(actual.shape[0])
        if min(cosines) < floor:
            raise AssertionError(f"packed parity {min(cosines):.6f} is below {floor}")
        report = {
            "passed": True,
            "requests": len(window.requests),
            "minimum_flat_cosine": min(cosines),
            "mean_flat_cosine": statistics.mean(cosines),
            "cosine_floor": floor,
            "output_lengths": output_lengths,
        }
        report["sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
        return report

    def measure(self, cfg: dict) -> dict:
        warmup = int(cfg["benchmark"]["warmup_rounds"])
        trials = int(cfg["benchmark"]["trials"])
        rounds = int(cfg["benchmark"]["rounds_per_trial"])
        for _ in range(warmup):
            for window in self.windows:
                for variant in VARIANTS:
                    self.run_window(variant, window)
        torch.cuda.synchronize()

        samples = {variant: [] for variant in VARIANTS}
        checksums = {variant: None for variant in VARIANTS}
        for trial in range(trials):
            order = VARIANTS[trial % len(VARIANTS):] + VARIANTS[:trial % len(VARIANTS)]
            for variant in order:
                torch.cuda.reset_peak_memory_stats()
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                checksum = None
                for repeat in range(rounds):
                    for window in self.windows:
                        checksum = self.run_window(variant, window)
                end.record()
                torch.cuda.synchronize()
                seconds = begin.elapsed_time(end) / 1000.0
                requests = rounds * sum(len(window.requests) for window in self.windows)
                real_tokens = rounds * sum(window.real_tokens for window in self.windows)
                samples[variant].append({
                    "trial": trial,
                    "seconds": seconds,
                    "requests_per_second": requests / seconds,
                    "real_tokens_per_second": real_tokens / seconds,
                    "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                })
                checksums[variant] = int(checksum)

        variants = {}
        for variant, values in samples.items():
            request_rates = [x["requests_per_second"] for x in values]
            token_rates = [x["real_tokens_per_second"] for x in values]
            variants[variant] = {
                "gpu_forwards_per_window": 1 if variant == "unified_packed" else 2,
                "processed_tokens_per_window": (
                    statistics.mean(window.padded_tokens for window in self.windows)
                    if variant == "split_padded"
                    else statistics.mean(window.real_tokens for window in self.windows)
                ),
                "requests_per_second": statistics.median(request_rates),
                "real_tokens_per_second": statistics.median(token_rates),
                "request_rate_min": min(request_rates),
                "request_rate_max": max(request_rates),
                "peak_reserved_gb": max(x["peak_reserved_gb"] for x in values),
                "checksum": checksums[variant],
                "trials": values,
            }
        split_padded = variants["split_padded"]["requests_per_second"]
        split_packed = variants["split_packed"]["requests_per_second"]
        unified = variants["unified_packed"]["requests_per_second"]
        return {
            "variants": variants,
            "comparison": {
                "padding_layout_speedup_split_packed_vs_split_padded": split_packed / split_padded,
                "one_forward_speedup_unified_vs_split_packed": unified / split_packed,
                "end_to_end_speedup_unified_vs_split_padded": unified / split_padded,
                "conventional_gpus_for_one_unified_packed_gpu": unified / split_padded,
            },
        }


def _provenance() -> dict:
    packages = {}
    for name in ("torch", "transformers", "pylate", "sentence-transformers", "flash-attn", "nvidia-cutlass-dsl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        commit = None
    return {"packages": packages, "git_commit": commit}


def run(cfg: dict) -> dict:
    from pylate import models

    torch.manual_seed(int(cfg.get("seed", 0)))
    info = device_banner()
    models_cfg = cfg["models"]
    model = (
        models.ColBERT(
            model_name_or_path=models_cfg["checkpoint"],
            device="cuda",
            query_length=int(models_cfg["query_length"]),
            document_length=int(models_cfg["document_length"]),
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        .to(torch.bfloat16)
        .eval()
    )
    pe.pack(model, attention_backend=cfg["benchmark"].get("attention_backend", "auto"), cuda_graph=False, validate=False)
    payloads = _build_payloads(model, cfg)
    windows = _prepare_windows(model, payloads, cfg)
    benchmark = Benchmark(model, windows)
    correctness = benchmark.correctness(float(cfg["benchmark"].get("cosine_floor", 0.98)))
    measured = benchmark.measure(cfg)
    return {
        "schema_version": 1,
        "purpose": "saturated mixed query/document serving windows",
        **info,
        "model": models_cfg,
        "dataset": {
            "name": cfg["dataset"].get("name", "Tevatron/AgentIR-data"),
            "kind": "real-held-out-text",
            "query_lengths": _length_summary(payloads, "query"),
            "document_lengths": _length_summary(payloads, "document"),
        },
        "workload": {
            **cfg["benchmark"],
            "real_tokens_per_window": statistics.mean(window.real_tokens for window in windows),
            "padded_tokens_per_window": statistics.mean(window.padded_tokens for window in windows),
            "padding_ratio": statistics.mean(window.padded_tokens for window in windows) / statistics.mean(window.real_tokens for window in windows),
            "windows": [
                {
                    "request_ids": [request.request_id for request in window.requests],
                    "kinds": [request.kind for request in window.requests],
                    "payload_ids": [request.payload_id for request in window.requests],
                    "lengths": [sequence.numel() for sequence in window.sequences],
                }
                for window in windows
            ],
        },
        "correctness": correctness,
        **measured,
        "provenance": _provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--output")
    parser.add_argument("--requests-per-window", type=int)
    parser.add_argument("--attention-backend", choices=("auto", "flash", "triton", "sdpa"))
    parser.add_argument("--trials", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this showcase requires CUDA")
    cfg = load_config(args.config)
    if args.requests_per_window is not None:
        cfg["benchmark"]["requests_per_window"] = args.requests_per_window
    if args.attention_backend is not None:
        cfg["benchmark"]["attention_backend"] = args.attention_backend
    if args.trials is not None:
        cfg["benchmark"]["trials"] = args.trials
    result = run(cfg)
    name = cfg.get("output", {}).get("name", Path(args.config).stem)
    output = Path(args.output) if args.output else RESULTS_DIR / f"{name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    comparison = result["comparison"]
    print(f"wrote {output}")
    for variant, value in result["variants"].items():
        print(f"{variant:16s} {value['requests_per_second']:9.1f} req/s  {value['real_tokens_per_second']:11.0f} tok/s")
    print(f"unified vs conventional: {comparison['end_to_end_speedup_unified_vs_split_padded']:.3f}x")
    print(f"forward unification only: {comparison['one_forward_speedup_unified_vs_split_packed']:.3f}x")


if __name__ == "__main__":
    main()
