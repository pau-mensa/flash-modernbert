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
"""BSHD RoPE A/B: the fused rope+flash inference path vs the transpose+rope path.

The flash path used to transpose q/k/v to [B,H,S,D], `.contiguous()`-copy q/k for the
[B*H,S,D] RoPE kernel, then transpose back — ~4× the q/k tensor's HBM traffic in pure
layout churn. `apply_rope_bshd` reads RoPE straight out of the packed qkv (the Wqkv GEMM
output) and hands flash its native [.., S, H, D] layout (≈2× traffic, no transpose/copy;
`flash_attention_qkv_bshd`). **Bit-identical** (same RoPE math), so it only trades speed;
the win is bandwidth-bound, growing with S and on lower-HBM cards.

Inference only: a BSHD RoPE backward was built and measured a wash-to-regression in
training (the bwd scatters a full [N,3HD] grad and autograd must sum it with the attention
bwd's grad on the v slice — more than the transpose path's single cat), so it was removed
and the path is gated to no-grad. Training keeps the transpose+rope path.

A/B in one process by monkeypatching `ops.use_bshd_rope_flash` off (→ the transpose path).

    uv run benchmarks/bshd_rope_bench.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import flash_modernbert as fm
from flash_modernbert import ops

MODEL_ID = "answerdotai/ModernBERT-base"
DTYPE = torch.bfloat16
_BSHD_OFF = lambda *a, **k: False  # noqa: E731 — A/B override for use_bshd_rope_flash


def make_batch(b, s, pad, vocab, seed):
    g = torch.Generator().manual_seed(seed)
    if pad:
        lengths = torch.randint(max(1, s // 2), s + 1, (b,), generator=g)
        lengths[0] = s
        am = (torch.arange(s).unsqueeze(0) < lengths.unsqueeze(1)).cuda().long()
    else:
        am = torch.ones(b, s, dtype=torch.long).cuda()
    ids = torch.randint(5, vocab, (b, s), generator=g).cuda()
    return ids, am


@torch.no_grad()
def measure(model, ids, am, *, n_warmup=8, n_iters=25, n_trials=5):
    torch.cuda.empty_cache()
    for _ in range(n_warmup):
        model(input_ids=ids, attention_mask=am)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(input_ids=ids, attention_mask=am)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / n_iters * 1e3)
    return statistics.median(times)


def main():
    assert torch.cuda.is_available()
    dev = torch.cuda.get_device_name(0)
    cap = "".join(map(str, torch.cuda.get_device_capability(0)))
    print(f"device {dev} sm_{cap}  torch {torch.__version__}  (inference fwd, flash)\n")

    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL_ID, dtype=DTYPE).cuda().eval()
    fm.prepare(model, attention_backend="flash", validate=False)
    vocab = int(model.config.vocab_size)

    rows = []
    print(f"{'case':16}{'S':>6}{'off ms':>10}{'on ms':>10}{'speedup':>9}{'cos':>9}")
    for pad in (False, True):
        for s in (512, 1024, 2048, 4096):
            b = 8
            ids, am = make_batch(b, s, pad, vocab, seed=s + b + int(pad))
            with torch.no_grad():
                r_on = model(input_ids=ids, attention_mask=am).last_hidden_state.clone()
            orig = ops.use_bshd_rope_flash
            ops.use_bshd_rope_flash = _BSHD_OFF
            try:
                with torch.no_grad():
                    r_off = model(input_ids=ids, attention_mask=am).last_hidden_state.clone()
                off_ms = measure(model, ids, am)
            finally:
                ops.use_bshd_rope_flash = orig
            on_ms = measure(model, ids, am)
            cos = F.cosine_similarity(r_on.flatten().float(), r_off.flatten().float(), dim=0).item()
            tag = "padded(varlen)" if pad else "unpadded(dense)"
            rows.append({"case": tag, "b": b, "s": s, "off_ms": round(off_ms, 3),
                         "on_ms": round(on_ms, 3), "speedup": round(off_ms / on_ms, 3),
                         "out_cos": round(cos, 6)})
            print(f"{tag:16}{s:>6}{off_ms:>10.3f}{on_ms:>10.3f}"
                  f"{off_ms / on_ms:>8.3f}x{cos:>9.5f}")

    out = Path("benchmarks/results/bshd_rope_bench.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"device": dev, "capability": cap, "torch": torch.__version__, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
