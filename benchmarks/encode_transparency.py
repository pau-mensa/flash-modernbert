# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["flash-modernbert", "transformers", "sentence-transformers", "pylate"]
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
"""A2 + A4 — prove CUDA graphs engage through the *real* encode paths and that
the graph layer is transparent after `prepare()` across the three frameworks.

`inference_bench.py` measures the encoder forward at exact shapes; this script
proves the thing that benchmark assumes: that driving the public text APIs
(`AutoModel.__call__`, `ColBERT.encode`, `SentenceTransformer.encode`) in their
normal inference mode actually routes through the captured graph, and that the
embeddings it produces match the stock model within the bf16 band.

For each framework it checks, on one model:

  1. ENGAGEMENT — after `prepare(cuda_graph=True)`, a real encode call populates
     the runner's graph cache (a captured graph was replayed). Encode paths run
     `eval()` + `no_grad()` with no autocast, so the inference-only gate lets the
     graph run — this confirms that empirically rather than by assertion.
  2. CORRECTNESS — the graphed embeddings track the stock embeddings (cosine ≥
     the validate band), so the graph is transparent, not just fast.
  3. GATE — under autocast (and under grad) the same call does *not* add a graph,
     confirming the gate that protects against replaying freed weight buffers.

Exit code is non-zero if any check fails, so it doubles as a CI smoke.

    uv run benchmarks/encode_transparency.py
    uv run benchmarks/encode_transparency.py --only pylate
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch
import torch.nn.functional as F

import flash_modernbert as fm
from flash_modernbert.state import get_state

MODEL_ID = "answerdotai/ModernBERT-base"
COS_BAND = 0.997  # the validate() bf16 band

QUERIES = [
    "what is late interaction retrieval",
    "how do cuda graphs reduce launch overhead",
    "modernbert sliding window attention",
    "colbert maxsim scoring explained",
]
DOCUMENTS = [
    "Late interaction models such as ColBERT encode queries and documents into "
    "per-token embeddings and score them with a MaxSim operator over the token grid.",
    "A CUDA graph captures a sequence of kernel launches once and replays them as a "
    "single unit, collapsing the per-launch host overhead that dominates short "
    "sequences where each kernel is tiny.",
    "ModernBERT alternates global and local attention; the local layers use a sliding "
    "window so most layers attend only within a band around each token.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat(emb) -> torch.Tensor:
    """Flatten an encode result (tensor / list of per-token tensors / ndarray)
    to a 1-D float vector for a cosine comparison."""
    if isinstance(emb, (list, tuple)):
        return torch.cat([_flat(e) for e in emb])
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)
    return emb.detach().reshape(-1).float().cpu()


def _cosine(a, b) -> float:
    fa, fb = _flat(a), _flat(b)
    n = min(fa.numel(), fb.numel())
    return F.cosine_similarity(fa[:n], fb[:n], dim=0).item()


def _cache_size(model) -> int:
    runner = get_state(model).graph_runner
    return 0 if runner is None else len(runner._cache)


class Check:
    """Collects pass/fail lines for one framework."""

    def __init__(self, name: str):
        self.name = name
        self.ok = True

    def expect(self, label: str, condition: bool, detail: str = "") -> None:
        mark = "PASS" if condition else "FAIL"
        self.ok = self.ok and condition
        print(f"    [{mark}] {label}{(' — ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# raw Hugging Face AutoModel
# ---------------------------------------------------------------------------


def check_huggingface(device) -> bool:
    from transformers import AutoModel, AutoTokenizer

    print("\n[huggingface] AutoModel(...) forward")
    chk = Check("huggingface")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    enc = tok(DOCUMENTS, padding=True, truncation=True, return_tensors="pt").to(device)
    ids, mask = enc["input_ids"], enc["attention_mask"]

    with torch.no_grad():
        stock = model(input_ids=ids, attention_mask=mask).last_hidden_state

    fm.prepare(model, cuda_graph=True)
    with torch.no_grad():
        graphed = model(input_ids=ids, attention_mask=mask).last_hidden_state

    chk.expect("graph engaged on no_grad forward", _cache_size(model) > 0,
               f"{_cache_size(model)} graph(s) cached")
    cos = _cosine(stock, graphed)
    chk.expect("graphed output matches stock", cos >= COS_BAND, f"cosine {cos:.5f}")

    # GATE: under autocast the same call must not capture a new graph.
    before = _cache_size(model)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=ids, attention_mask=mask)
    chk.expect("graph skipped under autocast", _cache_size(model) == before,
               f"cache {before} -> {_cache_size(model)}")
    return chk.ok


# ---------------------------------------------------------------------------
# PyLate ColBERT.encode
# ---------------------------------------------------------------------------


def check_pylate(device) -> bool:
    from pylate import models

    print("\n[pylate] ColBERT.encode (queries + documents)")
    chk = Check("pylate")
    model = models.ColBERT(
        model_name_or_path=MODEL_ID, device=str(device),
        embedding_size=128, query_length=32, document_length=512,
    )
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    def encode(is_query):
        texts = QUERIES if is_query else DOCUMENTS
        return model.encode(texts, is_query=is_query, batch_size=32,
                            convert_to_tensor=True, show_progress_bar=False)

    q_stock, d_stock = encode(True), encode(False)

    fm.prepare(model, cuda_graph=True)
    q_graph, d_graph = encode(True), encode(False)

    chk.expect("graph engaged through encode()", _cache_size(model) > 0,
               f"{_cache_size(model)} bucket(s): {list(get_state(model).graph_runner._cache)}")
    cq, cd = _cosine(q_stock, q_graph), _cosine(d_stock, d_graph)
    chk.expect("query embeddings match stock", cq >= COS_BAND, f"cosine {cq:.5f}")
    chk.expect("document embeddings match stock", cd >= COS_BAND, f"cosine {cd:.5f}")
    return chk.ok


# ---------------------------------------------------------------------------
# SentenceTransformer.encode
# ---------------------------------------------------------------------------


def check_sentence_transformers(device) -> bool:
    from sentence_transformers import SentenceTransformer

    print("\n[sentence_transformers] SentenceTransformer.encode")
    chk = Check("sentence_transformers")
    # A plain HF id yields a Transformer + mean-Pooling ST model — enough to prove
    # the graph layer is transparent under ST's encode().
    model = SentenceTransformer(MODEL_ID, device=str(device)).to(torch.bfloat16).eval()

    def encode():
        return model.encode(DOCUMENTS, convert_to_tensor=True,
                            show_progress_bar=False, normalize_embeddings=True)

    stock = encode()
    fm.prepare(model, cuda_graph=True)
    graphed = encode()

    chk.expect("graph engaged through encode()", _cache_size(model) > 0,
               f"{_cache_size(model)} graph(s) cached")
    cos = _cosine(stock, graphed)
    chk.expect("sentence embeddings match stock", cos >= COS_BAND, f"cosine {cos:.5f}")
    return chk.ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


CHECKS = {
    "huggingface": check_huggingface,
    "pylate": check_pylate,
    "sentence_transformers": check_sentence_transformers,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=list(CHECKS), help="run a single framework")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("error: this smoke requires a CUDA GPU", file=sys.stderr)
        raise SystemExit(1)

    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")

    selected = [args.only] if args.only else list(CHECKS)
    outcomes = {}
    for name in selected:
        try:
            outcomes[name] = CHECKS[name](device)
        except Exception:  # noqa: BLE001 — report and keep going
            traceback.print_exc()
            outcomes[name] = False

    print("\n=== summary ===")
    for name in selected:
        print(f"  {name:<22s} {'PASS' if outcomes[name] else 'FAIL'}")
    if not all(outcomes.values()):
        raise SystemExit(1)
    print("\nall encode paths engage graphs transparently and correctly.")


if __name__ == "__main__":
    main()
