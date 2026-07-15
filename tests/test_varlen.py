# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["packed-encoders", "transformers", "pytest"]
#
# [tool.uv.sources]
# packed-encoders = { path = "../", editable = true }
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

from packed_encoders import forward, ops


# --------------------------------------------------------------------------- #
# Plumbing — no GPU, no flash kernel
# --------------------------------------------------------------------------- #


def test_resolve_eager_backend():
    assert forward._resolve_eager_backend("sdpa") == "sdpa"
    assert forward._resolve_eager_backend("flash") == "flash"
    # An unresolved auto is capture/CPU-safe SDPA, never an S-length decision.
    assert forward._resolve_eager_backend("auto") == "sdpa"


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
    import packed_encoders as fm
    from transformers import AutoModel

    tok = __import__("transformers").AutoTokenizer.from_pretrained(MODEL_ID)
    stock = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda().eval()
    flash = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda().eval()
    fm.pack(flash, attention_backend="flash", validate=False)
    sdpa = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda().eval()
    fm.pack(sdpa, attention_backend="sdpa", validate=False)
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
def test_5090_auto_routes_by_distribution_score(models, monkeypatch):
    """Exercise the real eager boundary on the card used for score calibration."""
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("the first distribution score is calibrated only for sm_120")

    _tok, _stock, model, _sdpa = models
    state = getattr(model, forward.ATTR)
    original_core = forward.core
    routes = []

    def recording_core(*args, **kwargs):
        routes.append(kwargs.get("backend", "sdpa"))
        return original_core(*args, **kwargs)

    monkeypatch.setattr(forward, "core", recording_core)
    cases = [
        ((1024,), "sdpa"),                 # long but too little aggregate work
        ((1024, 1024), "flash"),           # same S, enough aggregate work
        ((64,) * 64, "sdpa"),              # deliberately omitted ~1-2% boundary win
        ((64, 32) * 32, "flash"),          # same padded shape, padding credits varlen
    ]
    for lengths, expected in cases:
        b, s = len(lengths), max(lengths)
        ids = torch.randint(5, model.config.vocab_size, (b, s), device="cuda")
        lens = torch.tensor(lengths, device="cuda").unsqueeze(1)
        mask = (torch.arange(s, device="cuda").unsqueeze(0) < lens).long()
        routes.clear()
        with torch.inference_mode():
            forward.fused_forward(
                model, state.params, ids, mask, backend="auto"
            )
        assert routes == [expected]


def _grad_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def _grads_of_interest(model) -> dict[str, torch.Tensor]:
    """Output-proximate param grads: the last three encoder layers' attention / MLP /
    norm weights plus the final norm. Deep early-layer grads are *not* a usable fidelity
    gauge in bf16 — they are cancellation-dominated, so even two correct paths (stock HF
    vs our sdpa) agree only ~0.5 there. The output-proximate grads are well-conditioned
    (>0.999 between any two correct paths) and still exercise the whole chain a packing
    bug would corrupt: loss -> repad -> last layers' attention + MLP + LN backward ->
    varlen-attention backward."""
    n = len(model.layers)
    picks = {"final_norm": model.final_norm.weight}
    for li in (n - 1, n - 2, n - 3):
        L = model.layers[li]
        picks[f"L{li}.attn.Wqkv"] = L.attn.Wqkv.weight
        picks[f"L{li}.attn.Wo"] = L.attn.Wo.weight
        picks[f"L{li}.mlp.Wi"] = L.mlp.Wi.weight
        picks[f"L{li}.mlp.Wo"] = L.mlp.Wo.weight
        picks[f"L{li}.mlp_norm"] = L.mlp_norm.weight
    return {k: p.grad.detach().clone() for k, p in picks.items()}


def _fwd_bwd_grads(model, ids, am) -> dict[str, torch.Tensor]:
    """One fwd+bwd under a deterministic, signal-rich loss (sum of squared real-token
    activations). A random output seed leaves early-layer grads cancellation-dominated
    and bf16-noisy; the squared-norm loss conditions grads at every depth so the
    comparison reflects the kernels, not rounding in a near-zero quantity. Pads are
    masked out, so the packed and dense paths see identical signal."""
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, attention_mask=am).last_hidden_state
    loss = (out.float().pow(2) * am.unsqueeze(-1)).sum()
    loss.backward()
    return _grads_of_interest(model)


# Distinct, graded-length sentences (no repetition — repeated tokens make the
# embedding-table grad a near-total-cancellation noise blob for *every* path).
_SENTENCES = [
    "Dense retrieval maps a passage to a single vector.",
    "A bi-encoder pools all token states into one passage embedding for search.",
    "Late interaction keeps one contextual vector per token and defers the comparison.",
    "ColBERT scores a query against a passage by summing, over query tokens, the "
    "largest similarity to any passage token, which rescues exact-match signal.",
    "Sliding-window attention reads long documents at linear cost by restricting most "
    "layers to a local band while a few global layers mix the whole sequence.",
    "Agentic retrieval issues many reformulated queries over long technical documents, "
    "so the encoder must stay fast and memory-frugal as passages stretch to thousands "
    "of tokens, which is exactly where padding waste would otherwise dominate the bill.",
]


def _join(*idx) -> str:
    return " ".join(_SENTENCES[i] for i in idx)


@needs_flash
@pytest.mark.parametrize(
    "texts",
    [
        # graded-length docs (joined sentences) → heavy, varied padding
        [_SENTENCES[0], _join(1, 2), _join(3, 4), _join(5, 3, 0)],
        # six single sentences of differing length
        list(_SENTENCES),
    ],
)
def test_varlen_flash_backward_matches_dense(models, texts):
    """The packed (varlen) training backward must be as correct as the dense paths.

    Both checks are on output-proximate grads (see `_grads_of_interest`): packed agrees
    with the sdpa fused tail (same kernels but attention — isolates pack/unpack + varlen
    backward) and with stock HF (the ground truth), to >0.998. A structural packing bug
    — cross-sequence attention leak, wrong RoPE positions, mis-scattered repad grad —
    would corrupt these by a wide margin (the dense control sits at 0.9999)."""
    tok, stock, flash, sdpa = models
    enc = tok(texts, return_tensors="pt", padding="longest")
    ids = enc["input_ids"].cuda()
    am = enc["attention_mask"].cuda()
    assert not bool((am == 1).all()), "batch must have padding to exercise the packed path"

    # Confirm the packed path actually fired for the flash model.
    calls = {"varlen": 0}
    orig = forward._varlen_forward

    def spy(*a, **k):
        calls["varlen"] += 1
        return orig(*a, **k)

    import unittest.mock as _m
    with _m.patch.object(forward, "_varlen_forward", spy):
        g_flash = _fwd_bwd_grads(flash, ids, am)
    assert calls["varlen"] == 1, "flash backend did not route through the packed path"

    g_sdpa = _fwd_bwd_grads(sdpa, ids, am)
    g_stock = _fwd_bwd_grads(stock, ids, am)

    for name in g_flash:
        flash_sdpa = _grad_cos(g_flash[name], g_sdpa[name])
        flash_hf = _grad_cos(g_flash[name], g_stock[name])
        assert flash_sdpa > 0.998, f"{name}: packed-vs-sdpa grad cos {flash_sdpa:.5f}"
        assert flash_hf > 0.998, f"{name}: packed-vs-HF grad cos {flash_hf:.5f}"


@needs_flash
@pytest.mark.parametrize(
    "texts",
    [
        ["short doc.", "a considerably longer document with many more tokens than the first one here ok then"],
        [_SENTENCES[0], _join(1, 2), _join(3, 4), _join(5, 3, 0)],
        ["only one sequence in this batch, no other rows to pad against at all"],
    ],
)
def test_packed_forward_entry_matches_hf(models, texts):
    """The additive packed entry (`forward.packed_forward`) — the showcase's paradigm
    path, which bypasses the patched `[B, S]` forward and consumes already-packed input
    — must match stock HF on the real tokens. Builds the packed input the way a collator
    will (real ids, `cu_seqlens`, per-sequence within-seq positions) and feeds it straight
    to `packed_forward`, returning `[total, H]` (no repad). This is the P0 forward-parity
    gate for the packed-paradigm showcase: loss-curve parity follows from it."""
    from packed_encoders.config import ModernBertParams
    from packed_encoders.locate import find_encoder

    tok, stock, flash, sdpa = models
    enc = tok(texts, return_tensors="pt", padding="longest")
    ids = enc["input_ids"].cuda()
    am = enc["attention_mask"].cuda()
    b, s = ids.shape

    # packed_forward reads weights straight off the live module — no pack() needed,
    # exactly how the showcase opts into the paradigm (the drop-in [B,S] forward is for
    # padded users; this is the explicit packed call).
    encoder = find_encoder(stock)
    params = ModernBertParams.from_hf_config(encoder.config)

    packed_ids, indices, cu_seqlens, max_seqlen, position_ids = forward._unpad(
        ids.unsqueeze(-1), am
    )
    packed_ids = packed_ids.squeeze(-1)
    with torch.no_grad():
        packed_out = forward.packed_forward(
            encoder, params, packed_ids, cu_seqlens, max_seqlen, position_ids
        )
        ref = stock(input_ids=ids, attention_mask=am).last_hidden_state
    assert packed_out.shape == (int(am.sum()), ref.shape[-1])  # [total, H], no repad
    ref_packed = ref.reshape(b * s, -1).index_select(0, indices)
    cos = F.cosine_similarity(
        packed_out.flatten().float(), ref_packed.flatten().float(), dim=0
    ).item()
    assert cos > 0.997, f"packed entry vs HF real-token cos {cos:.5f}"


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
