"""Correctness gates for the experimental packed-short Triton forward."""

from __future__ import annotations

import pytest
import torch

from packed_encoders import ops
from packed_encoders._kernels.triton_packed_attention import (
    packed_short_attention,
    packed_short_attention_supported,
    select_packed_short_attention_config,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _inputs(lengths, *, seed=0, logit_scale=1.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (sum(lengths), 12, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16,
                    generator=generator) * logit_scale
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16,
                    generator=generator) * logit_scale
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16,
                    generator=generator)
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    cu = torch.nn.functional.pad(lens.cumsum(0, dtype=torch.int32), (1, 0))
    return q, k, v, cu


def _reference(q, k, v, cu, half_window):
    boundaries = cu.cpu().tolist()
    rows = []
    for start, end in zip(boundaries, boundaries[1:]):
        qf, kf, vf = q[start:end].float(), k[start:end].float(), v[start:end].float()
        scores = torch.einsum("qhd,khd->hqk", qf, kf) * 0.125
        if half_window is not None:
            pos = torch.arange(end - start, device="cuda")
            scores.masked_fill_(
                ((pos[:, None] - pos[None, :]).abs() > half_window).unsqueeze(0),
                float("-inf"),
            )
        rows.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), vf))
    return torch.cat(rows).to(torch.bfloat16)


def _flash(q, k, v, cu, max_seqlen, half_window):
    window = (-1, -1) if half_window is None else (half_window, half_window)
    return ops.flash_attention_varlen(
        q.permute(1, 0, 2).unsqueeze(0),
        k.permute(1, 0, 2).unsqueeze(0),
        v.permute(1, 0, 2).unsqueeze(0),
        cu,
        max_seqlen,
        window,
        0.125,
    ).reshape_as(q)


@pytest.mark.parametrize("half_window", [None, 64])
@pytest.mark.parametrize(
    "lengths",
    [
        [1, 2, 31, 32, 33, 63, 64, 65, 127, 128],
        [128] * 8,
        [128, 64, 16, 16, 16, 16, 16, 16],
        [46, 46, 45, 45, 45, 44, 43, 42],
    ],
)
def test_matches_fp32_reference_and_flash(lengths, half_window):
    q, k, v, cu = _inputs(lengths, seed=17)
    with torch.inference_mode():
        reference = _reference(q, k, v, cu, half_window)
        flash = _flash(q, k, v, cu, max(lengths), half_window)
        actual = packed_short_attention(
            q, k, v, cu, max(lengths), half_window=half_window,
            softmax_scale=0.125,
        )

    ref_f = reference.float()
    flash_mean = (flash.float() - ref_f).abs().mean()
    actual_mean = (actual.float() - ref_f).abs().mean()
    flash_cos = torch.nn.functional.cosine_similarity(
        flash.float().flatten(), ref_f.flatten(), dim=0
    )
    actual_cos = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), ref_f.flatten(), dim=0
    )
    assert actual_mean <= flash_mean * 1.15 + 1e-5
    assert actual_cos >= flash_cos - 2e-6


@pytest.mark.parametrize("logit_scale", [0.01, 1.0, 8.0])
def test_softmax_stability_and_determinism(logit_scale):
    lengths = [33, 65, 127]
    q, k, v, cu = _inputs(lengths, seed=11, logit_scale=logit_scale)
    with torch.inference_mode():
        first = packed_short_attention(
            q, k, v, cu, 127, half_window=64, softmax_scale=0.125
        )
        second = packed_short_attention(
            q, k, v, cu, 127, half_window=64, softmax_scale=0.125
        )
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_no_cross_sequence_leakage():
    lengths = [31, 65]
    q, k, v, cu = _inputs(lengths, seed=7)
    with torch.inference_mode():
        baseline = packed_short_attention(
            q, k, v, cu, 65, half_window=None, softmax_scale=0.125
        )
        k_changed, v_changed = k.clone(), v.clone()
        k_changed[31:] *= 100
        v_changed[31:] *= -100
        changed = packed_short_attention(
            q, k_changed, v_changed, cu, 65,
            half_window=None, softmax_scale=0.125,
        )
    assert torch.equal(baseline[:31], changed[:31])


def test_capability_guard_and_static_configs():
    q, k, v, cu = _inputs([32, 17])
    with torch.inference_mode():
        assert packed_short_attention_supported(
            q, k, v, cu, 32, half_window=None
        )
        assert not packed_short_attention_supported(
            q, k, v, cu, 129, half_window=None
        )
        assert not packed_short_attention_supported(
            q, k, v, cu, 32, half_window=32
        )
    assert select_packed_short_attention_config(32)["block_m"] == 16
    assert select_packed_short_attention_config(64)["block_n"] == 64
    assert select_packed_short_attention_config(128)["block_m"] == 64
