# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["flash-modernbert", "transformers", "pytest"]
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
"""Varlen (packed, variable-length) flash path.

Two layers:

- **Plumbing** (`_unpad` / `_repad` / `_has_padding` / `_resolve_eager_backend`):
  pure tensor logic, no GPU or flash kernel — round-trip, cu_seqlens, and the
  HF-matching per-token positions.
- **Numerics** (skipped without a flash kernel): the padded-batch forward through
  the flash backend must match stock HF ModernBERT (which unpads + varlen-flashes
  itself) on the real-token positions, where the dense-flash path scores ~0.32.

    uv run pytest tests/test_varlen.py -q
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flash_modernbert import forward, ops


# --------------------------------------------------------------------------- #
# Plumbing — no GPU, no flash kernel
# --------------------------------------------------------------------------- #


def test_resolve_eager_backend():
    assert forward._resolve_eager_backend("sdpa", 99999) == "sdpa"
    assert forward._resolve_eager_backend("flash", 1) == "flash"
    assert forward._resolve_eager_backend("auto", ops.FLASH_MIN_SEQ - 1) == "sdpa"
    assert forward._resolve_eager_backend("auto", ops.FLASH_MIN_SEQ) == "flash"


def test_has_padding():
    assert forward._has_padding(None) is False
    assert forward._has_padding(torch.ones(2, 5, dtype=torch.long)) is False
    m = torch.ones(2, 5, dtype=torch.long)
    m[0, 3:] = 0
    assert forward._has_padding(m) is True


def test_unpad_shapes_and_positions():
    # B=3, S=5, right-padded lengths 5/2/4
    am = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.long
    )
    x = torch.arange(3 * 5 * 4, dtype=torch.float32).reshape(3, 5, 4)
    x_packed, indices, cu_seqlens, max_seqlen, position_ids = forward._unpad(x, am)

    total = int(am.sum())
    assert x_packed.shape == (total, 4)
    assert cu_seqlens.tolist() == [0, 5, 7, 11]      # cumulative 5, +2, +4
    assert cu_seqlens.dtype == torch.int32
    assert max_seqlen == 5
    # each packed row is a real token, in (b, s) row-major order
    expected_rows = [x[b, s] for b in range(3) for s in range(5) if am[b, s]]
    assert torch.equal(x_packed, torch.stack(expected_rows))
    # position == column index within the padded row (HF arange(S) semantics)
    expected_pos = [s for b in range(3) for s in range(5) if am[b, s]]
    assert position_ids.tolist() == expected_pos


def test_unpad_repad_roundtrip():
    am = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.long)
    x = torch.randn(3, 4, 8)
    x_packed, indices, *_ = forward._unpad(x, am)
    rep = forward._repad(x_packed, indices, 3, 4)
    m = am.bool().unsqueeze(-1).expand_as(x)
    assert torch.equal(rep[m], x[m])          # real tokens recovered exactly
    assert torch.equal(rep[~m], torch.zeros_like(rep[~m]))  # pads zeroed


def test_unpad_no_padding_is_identity_order():
    am = torch.ones(2, 3, dtype=torch.long)
    x = torch.randn(2, 3, 5)
    x_packed, indices, cu_seqlens, max_seqlen, position_ids = forward._unpad(x, am)
    assert x_packed.shape == (6, 5)
    assert cu_seqlens.tolist() == [0, 3, 6]
    assert position_ids.tolist() == [0, 1, 2, 0, 1, 2]


# --------------------------------------------------------------------------- #
# Numerics — needs a CUDA GPU and a flash kernel
# --------------------------------------------------------------------------- #

MODEL_ID = "answerdotai/ModernBERT-base"


def _flash_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        ops._load_flash_attn()
        return True
    except ImportError:
        return False


needs_flash = pytest.mark.skipif(
    not _flash_available(), reason="varlen numerics need a CUDA GPU + flash kernel"
)


@pytest.fixture(scope="module")
def models():
    import flash_modernbert as fm
    from transformers import AutoModel

    tok = __import__("transformers").AutoTokenizer.from_pretrained(MODEL_ID)
    stock = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda().eval()
    flash = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda().eval()
    fm.prepare(flash, attention_backend="flash", validate=False)
    sdpa = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda().eval()
    fm.prepare(sdpa, attention_backend="sdpa", validate=False)
    return tok, stock, flash, sdpa


def _masked_cos(out, ref, am):
    m = am.bool().unsqueeze(-1).expand_as(ref)
    return F.cosine_similarity(out[m].float(), ref[m].float(), dim=0).item()


@needs_flash
@pytest.mark.parametrize(
    "texts",
    [
        ["short doc.", "a considerably longer document with many more tokens than the first one here ok then"],
        ["x", "y", "this middle-length one sits between the two tiny neighbors in the batch"],
        ["only one sequence in this batch, no other rows to pad against at all"],
    ],
)
def test_varlen_flash_matches_hf_on_padded_batch(models, texts):
    tok, stock, flash, sdpa = models
    enc = tok(texts, return_tensors="pt", padding="longest")
    ids = enc["input_ids"].cuda()
    am = enc["attention_mask"].cuda()
    with torch.no_grad():
        ref = stock(input_ids=ids, attention_mask=am).last_hidden_state
        out_flash = flash(input_ids=ids, attention_mask=am).last_hidden_state
        out_sdpa = sdpa(input_ids=ids, attention_mask=am).last_hidden_state
    # flash (varlen) tracks stock HF on the real-token positions...
    assert _masked_cos(out_flash, ref, am) > 0.997
    # ...and agrees with the sdpa fused path (the dep-free padded reference).
    assert _masked_cos(out_flash, out_sdpa, am) > 0.997


@needs_flash
def test_varlen_only_engages_when_padded(monkeypatch, models):
    """A genuinely unpadded (all-ones) batch must take the dense flash path, not
    varlen — _varlen_forward should fire only when there is padding to strip."""
    tok, stock, flash, sdpa = models
    calls = {"varlen": 0}
    orig = forward._varlen_forward

    def spy(*a, **k):
        calls["varlen"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(forward, "_varlen_forward", spy)

    enc = tok(["equal length here please"] * 3, return_tensors="pt", padding="longest")
    ids, am = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
    assert bool((am == 1).all())  # no padding
    with torch.no_grad():
        flash(input_ids=ids, attention_mask=am)
    assert calls["varlen"] == 0

    enc2 = tok(["tiny", "a much longer sequence that forces real padding in the batch"],
               return_tensors="pt", padding="longest")
    ids2, am2 = enc2["input_ids"].cuda(), enc2["attention_mask"].cuda()
    assert not bool((am2 == 1).all())  # padded
    with torch.no_grad():
        flash(input_ids=ids2, attention_mask=am2)
    assert calls["varlen"] == 1
