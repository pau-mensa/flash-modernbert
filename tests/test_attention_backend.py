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
from flash_modernbert.prepare import (
    _default_backend,
    _resolve_attention_backend,
    prepare,
)


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


def test_flash_min_seq_is_arch_keyed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cases = {(12, 0): 128, (10, 0): 512, (9, 0): 1024, (8, 0): 1024}  # last = unknown -> default
    for cc, expected in cases.items():
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda cc=cc: cc)
        assert ops._resolve_flash_min_seq() == expected


def test_flash_min_seq_falls_back_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert ops._resolve_flash_min_seq() == ops._FLASH_MIN_SEQ_DEFAULT == 1024


def test_default_backend_prefers_flash_when_available(monkeypatch):
    # unset default: flash kernel importable -> "auto" (silently)
    monkeypatch.setattr(ops, "_load_flash_attn", lambda: (None, None, "compiled"))
    assert _default_backend() == "auto"


def test_default_backend_falls_back_to_sdpa(monkeypatch):
    # unset default: no kernel -> "sdpa", and NO warning (unlike explicit "auto")
    def boom():
        raise ImportError("no flash kernel")

    monkeypatch.setattr(ops, "_load_flash_attn", boom)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        assert _default_backend() == "sdpa"
