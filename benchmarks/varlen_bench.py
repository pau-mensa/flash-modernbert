# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["flash-modernbert", "transformers"]
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
"""Varlen benchmark: the flash backend on PADDED, variable-length doc batches.

`inference_bench.py` measures all-ones (unpadded) batches. This one measures the
regime the varlen path exists for: a batch of documents of *different* lengths,
padded to the batch max. There:

    stock / fused[sdpa]  pay the full B×S² dense attention + dense mask on the
                         padded rectangle, including the pad tokens.
    fused[flash]         unpads to the real tokens, runs the whole encoder packed
                         (GEMMs/LN/GeGLU skip pad tokens too) and confines
                         attention within each sequence via cu_seqlens — no dense
                         mask, no pad-token compute.

So this is the head-to-head that decides whether flash can be the default. We
sweep sequence length (at a fixed padding fraction) and padding fraction (at a
fixed S), reporting ms/call, samples/s, peak reserved MB, and the flash-over-sdpa
speedup. Lengths are drawn uniformly in [frac·S, S] (frac=1.0 ⇒ no padding).

    uv run benchmarks/varlen_bench.py
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from pathlib import Path

import torch

import flash_modernbert as fm

MODEL_ID = "answerdotai/ModernBERT-base"
DTYPE = torch.bfloat16


def make_padded_batch(b, s, frac, vocab, device, seed):
    """B sequences with real lengths uniform in [frac·S, S], right-padded to S.
    Returns (input_ids, attention_mask, real_token_fraction)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    lo = max(1, int(frac * s))
    lengths = torch.randint(lo, s + 1, (b,), generator=g)
    lengths[0] = s  # guarantee at least one full-length row (sets the padded S)
    ids = torch.randint(5, vocab, (b, s), generator=g).to(device)
    mask = (torch.arange(s).unsqueeze(0) < lengths.unsqueeze(1)).to(device).long()
    return ids, mask, float(lengths.sum()) / (b * s)


@torch.no_grad()
def measure(call, ids, mask, *, n_warmup=8, n_iters=30, n_trials=5):
    torch.cuda.empty_cache()
    for _ in range(n_warmup):
        call(ids, mask)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            call(ids, mask)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / n_iters * 1e3)
    return {
        "ms": statistics.median(times),
        "reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }


def _hidden(out):
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def build_variants():
    """stock HF (sdpa), fused[sdpa], fused[flash] — three models on equal weights."""
    from transformers import AutoModel

    variants = {}
    stock = AutoModel.from_pretrained(MODEL_ID, dtype=DTYPE).cuda().eval()
    variants["stock"] = lambda ids, m, _f=stock.forward: _hidden(_f(input_ids=ids, attention_mask=m))
    for backend in ("sdpa", "flash"):
        mdl = AutoModel.from_pretrained(MODEL_ID, dtype=DTYPE).cuda().eval()
        fm.prepare(mdl, attention_backend=backend, validate=False)
        variants[f"fused[{backend}]"] = (
            lambda ids, m, _f=mdl.forward: _hidden(_f(input_ids=ids, attention_mask=m))
        )
    vocab = int(stock.config.vocab_size)
    return variants, vocab


def run_point(variants, b, s, frac, vocab, seed):
    ids, mask, real_frac = make_padded_batch(b, s, frac, vocab, "cuda", seed)
    out = {"b": b, "s": s, "pad_frac_target": frac, "real_token_frac": round(real_frac, 3)}
    res = {name: measure(call, ids, mask) for name, call in variants.items()}
    base = res["stock"]["ms"]
    for name, r in res.items():
        out[name] = {
            "ms": round(r["ms"], 3),
            "reserved_mb": round(r["reserved_mb"], 0),
            "speedup_vs_stock": round(base / r["ms"], 3),
        }
    out["flash_vs_sdpa"] = round(res["fused[sdpa]"]["ms"] / res["fused[flash]"]["ms"], 3)
    return out


def main():
    assert torch.cuda.is_available()
    dev = torch.cuda.get_device_name(0)
    cap = "".join(map(str, torch.cuda.get_device_capability(0)))
    print(f"device {dev} sm_{cap}  torch {torch.__version__}\n")

    variants, vocab = build_variants()
    B = 16
    rows = []

    print("== S-sweep, moderate padding (lengths in [0.5S, S]) ==")
    print(f"{'S':>5} {'real%':>6} {'stock':>8} {'sdpa':>8} {'flash':>8} "
          f"{'flash/sdpa':>10} {'flash/stock':>11} {'resv sdpa/flash MB':>20}")
    for s in (256, 512, 1024, 2048, 4096):
        r = run_point(variants, B, s, 0.5, vocab, seed=100 + s)
        rows.append({"sweep": "seq", **r})
        print(f"{s:>5} {r['real_token_frac']*100:>5.0f}% "
              f"{r['stock']['ms']:>8.3f} {r['fused[sdpa]']['ms']:>8.3f} "
              f"{r['fused[flash]']['ms']:>8.3f} {r['flash_vs_sdpa']:>9.2f}x "
              f"{r['fused[flash]']['speedup_vs_stock']:>10.2f}x "
              f"{r['fused[sdpa]']['reserved_mb']:>9.0f}/{r['fused[flash]']['reserved_mb']:<10.0f}")

    print("\n== padding-sweep at S=2048 (lengths in [frac·S, S]) ==")
    print(f"{'frac':>5} {'real%':>6} {'sdpa':>8} {'flash':>8} {'flash/sdpa':>10}")
    for frac in (1.0, 0.75, 0.5, 0.25, 0.1):
        r = run_point(variants, B, 2048, frac, vocab, seed=900 + int(frac * 100))
        rows.append({"sweep": "pad", **r})
        print(f"{frac:>5.2f} {r['real_token_frac']*100:>5.0f}% "
              f"{r['fused[sdpa]']['ms']:>8.3f} {r['fused[flash]']['ms']:>8.3f} "
              f"{r['flash_vs_sdpa']:>9.2f}x")

    del variants
    gc.collect()
    torch.cuda.empty_cache()

    out = Path("benchmarks/results/varlen_bench.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"device": dev, "capability": cap, "torch": torch.__version__,
         "batch": B, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
