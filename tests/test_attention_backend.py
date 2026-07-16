"""Attention-backend dispatch + resolution.

Pure-logic tests (no GPU / model / flash kernel needed): the leaf `attention()` switch
routes correctly, its unresolved capture-safe `auto` stays on SDPA, and
`pack()`'s backend resolver validates explicit kernels and downgrades `"auto"` to
`"sdpa"` only when both optimized implementations are absent. Kernel ABIs are exercised on
GPU elsewhere (cute on B200/H200, compiled on sm_120)."""

from __future__ import annotations

import pytest
import torch

from packed_encoders import ops
from packed_encoders.errors import PackedEncodersError
from packed_encoders.pack import (
    _default_backend,
    _resolve_attention_backend,
    pack,
)


def _qkv(s: int):
    # [B, H, S, D] — only the S dim (shape[2]) matters for the dispatch decision.
    t = torch.zeros(1, 12, s, 64)
    return t, t, t


def test_unresolved_leaf_auto_is_sdpa_at_every_seq_len(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "flash_attention", lambda q, k, v, window, scaling: calls.append("flash"))
    monkeypatch.setattr(ops, "sdpa_attention", lambda q, k, v, mask, scaling: calls.append("sdpa"))

    q, k, v = _qkv(16)
    ops.attention(q, k, v, mask=None, window=(64, 64), scaling=0.125, backend="auto")
    q, k, v = _qkv(4096)
    ops.attention(q, k, v, mask=None, window=(64, 64), scaling=0.125, backend="auto")

    assert calls == ["sdpa", "sdpa"]


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
    monkeypatch.setattr(ops, "_load_packed_short_attention", lambda: (None, None))
    assert _resolve_attention_backend("flash") == "flash"
    assert _resolve_attention_backend("triton") == "triton"
    assert _resolve_attention_backend("auto") == "auto"
    assert _resolve_attention_backend("sdpa") == "sdpa"


def test_resolve_flash_absent(monkeypatch):
    def boom():
        raise ImportError("no flash kernel here")

    monkeypatch.setattr(ops, "_load_flash_attn", boom)
    monkeypatch.setattr(ops, "_load_packed_short_attention", lambda: (None, None))
    assert _resolve_attention_backend("sdpa") == "sdpa"  # never touches flash
    assert _resolve_attention_backend("auto") == "auto"  # Triton remains available
    with pytest.raises(PackedEncodersError):
        _resolve_attention_backend("flash")  # explicit -> hard error


def test_resolve_both_optimized_backends_absent(monkeypatch):
    def no_flash():
        raise ImportError("no flash")

    def no_triton():
        raise ImportError("no triton")

    monkeypatch.setattr(ops, "_load_flash_attn", no_flash)
    monkeypatch.setattr(ops, "_load_packed_short_attention", no_triton)
    with pytest.warns(UserWarning):
        assert _resolve_attention_backend("auto") == "sdpa"
    with pytest.raises(PackedEncodersError):
        _resolve_attention_backend("triton")


def test_pack_rejects_unknown_backend():
    with pytest.raises(PackedEncodersError):
        pack(object(), attention_backend="bogus")


def test_default_backend_prefers_flash_when_available(monkeypatch):
    # unset default: flash kernel importable -> "auto" (silently)
    monkeypatch.setattr(ops, "_load_flash_attn", lambda: (None, None, "compiled"))
    assert _default_backend() == "auto"


def test_default_backend_uses_auto_when_only_triton_is_available(monkeypatch):
    monkeypatch.setattr(
        ops, "_load_flash_attn", lambda: (_ for _ in ()).throw(ImportError("no flash"))
    )
    monkeypatch.setattr(ops, "_load_packed_short_attention", lambda: (None, None))
    assert _default_backend() == "auto"


def test_default_backend_falls_back_to_sdpa(monkeypatch):
    # unset default: no kernel -> "sdpa", and NO warning (unlike explicit "auto")
    def boom():
        raise ImportError("no flash kernel")

    monkeypatch.setattr(ops, "_load_flash_attn", boom)
    monkeypatch.setattr(ops, "_load_packed_short_attention", boom)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        assert _default_backend() == "sdpa"
