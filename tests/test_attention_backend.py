"""Attention-backend dispatch + resolution.

Pure-logic tests (no GPU / model / flash kernel needed): the `attention()` switch
routes correctly, `"auto"` resolves by sequence length around `FLASH_MIN_SEQ`, and
`prepare()`'s backend resolver raises for explicit `"flash"` without a kernel but
downgrades `"auto"` to `"sdpa"`. The kernel call ABIs themselves are exercised on
GPU elsewhere (cute on B200/H200, compiled on sm_120)."""

from __future__ import annotations

import pytest
import torch

from flash_modernbert import ops
from flash_modernbert.errors import FlashModernBertError
from flash_modernbert.prepare import _resolve_attention_backend, prepare


def _qkv(s: int):
    # [B, H, S, D] — only the S dim (shape[2]) matters for the dispatch decision.
    t = torch.zeros(1, 12, s, 64)
    return t, t, t


def test_auto_routes_by_seq_len(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "flash_attention", lambda q, k, v, window, scaling: calls.append("flash"))
    monkeypatch.setattr(ops, "sdpa_attention", lambda q, k, v, mask, scaling: calls.append("sdpa"))

    q, k, v = _qkv(ops.FLASH_MIN_SEQ - 1)
    ops.attention(q, k, v, mask=None, window=(64, 64), scaling=0.125, backend="auto")
    q, k, v = _qkv(ops.FLASH_MIN_SEQ)
    ops.attention(q, k, v, mask=None, window=(64, 64), scaling=0.125, backend="auto")

    assert calls == ["sdpa", "flash"]  # below threshold -> sdpa, at/above -> flash


def test_explicit_backends_route_directly(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "flash_attention", lambda *a, **k: calls.append("flash"))
    monkeypatch.setattr(ops, "sdpa_attention", lambda *a, **k: calls.append("sdpa"))

    q, k, v = _qkv(8)  # short S: explicit "flash" must still use flash, not auto-downgrade
    ops.attention(q, k, v, mask=None, window=(64, 64), scaling=0.125, backend="flash")
    ops.attention(q, k, v, mask=None, window=(64, 64), scaling=0.125, backend="sdpa")

    assert calls == ["flash", "sdpa"]


def test_resolve_flash_available(monkeypatch):
    monkeypatch.setattr(ops, "_load_flash_attn", lambda: (lambda *a, **k: None, "cute"))
    assert _resolve_attention_backend("flash") == "flash"
    assert _resolve_attention_backend("auto") == "auto"
    assert _resolve_attention_backend("sdpa") == "sdpa"


def test_resolve_flash_absent(monkeypatch):
    def boom():
        raise ImportError("no flash kernel here")

    monkeypatch.setattr(ops, "_load_flash_attn", boom)
    assert _resolve_attention_backend("sdpa") == "sdpa"  # never touches flash
    with pytest.warns(UserWarning):
        assert _resolve_attention_backend("auto") == "sdpa"  # graceful downgrade
    with pytest.raises(FlashModernBertError):
        _resolve_attention_backend("flash")  # explicit -> hard error


def test_prepare_rejects_unknown_backend():
    with pytest.raises(FlashModernBertError):
        prepare(object(), attention_backend="bogus")
