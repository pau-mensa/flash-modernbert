"""Pure-logic coverage for the distribution-aware attention score."""

from __future__ import annotations

import torch

from packed_encoders import attention_dispatch, forward
from packed_encoders.config import ModernBertParams


def _workload(lengths, padded_seqlen=None):
    return attention_dispatch.AttentionWorkload.from_lengths(
        lengths,
        padded_seqlen=padded_seqlen,
        half_window=64,
        global_layers=8,
        local_layers=14,
    )


def test_workload_preserves_fixed_token_decomposition():
    long = _workload((5000, 5000))
    short = _workload((100,) * 100)
    assert long.live_tokens == short.live_tokens == 10_000
    assert long.global_pairs == 50_000_000
    assert short.global_pairs == 1_000_000
    assert long.effective_seqlen == 5000
    assert short.effective_seqlen == 100
    assert long.local_pairs == 1_281_680
    assert short.local_pairs == 874_000


def test_workload_uses_rectangular_sdpa_width():
    workload = _workload((10, 20), padded_seqlen=32)
    assert workload.max_seqlen == 20
    assert workload.padded_seqlen == 32
    assert workload.live_tokens == 30
    assert workload.padded_tokens == 64
    assert workload.padding_tokens == 34
    assert workload.dense_pairs == 2 * 32 * 32


def test_5090_policy_matches_calibrated_boundary():
    policy = attention_dispatch.get_inference_policy((12, 0))
    assert policy is not None
    # High-confidence boundary reruns: the first member of each pair loses, the second wins.
    assert not policy.use_flash(_workload((1024,)))
    assert policy.use_flash(_workload((1024, 1024)))
    assert not policy.use_flash(_workload((32,) * 128))
    assert policy.use_flash(_workload((128,) * 32))
    # The conservative line deliberately leaves the noisy marginal 64x64 point on SDPA.
    assert not policy.use_flash(_workload((64,) * 64))


def test_5090_policy_credits_real_padding():
    policy = attention_dispatch.get_inference_policy((12, 0))
    equal = _workload((64,) * 64, padded_seqlen=64)
    half = _workload((64, 32) * 32, padded_seqlen=64)
    assert policy.score(half) > policy.score(equal)
    assert not policy.use_flash(equal)
    assert policy.use_flash(half)


def test_a100_policy_matches_refined_fixed_token_boundary():
    policy = attention_dispatch.get_inference_policy(
        (8, 0), "NVIDIA A100-SXM4-40GB"
    )
    assert policy is not None
    # Both contain 5k live tokens. The five longer sequences cross over; the ten
    # shorter attention problems do not.
    assert not policy.use_flash(_workload((500,) * 10))
    assert policy.use_flash(_workload((1000,) * 5))


def test_l40s_policy_matches_refined_boundary():
    policy = attention_dispatch.get_inference_policy((8, 9), "NVIDIA L40S")
    assert policy is not None
    assert not policy.use_flash(_workload((256,) * 16))
    assert policy.use_flash(_workload((512,) * 8))


def test_b200_policy_uses_distribution_at_fixed_rectangle():
    policy = attention_dispatch.get_inference_policy((10, 0), "NVIDIA B200")
    assert policy is not None
    equal = _workload((1024,) * 8)
    half = _workload((1024, 512) * 4, padded_seqlen=1024)
    assert equal.padded_tokens == half.padded_tokens == 8192
    assert not policy.use_flash(equal)
    assert policy.use_flash(half)


def test_h200_policy_is_shared_with_h100_and_matches_refined_boundary():
    h200 = attention_dispatch.get_inference_policy((9, 0), "NVIDIA H200")
    h100 = attention_dispatch.get_inference_policy(
        (9, 0), "NVIDIA H100 80GB HBM3"
    )
    assert h200 is not None
    assert h100 is h200
    assert not h200.use_flash(_workload((1024,) * 8))
    assert h200.use_flash(_workload((1024, 512) * 4, padded_seqlen=1024))


def test_resolver_uses_score_when_workload_is_supplied(monkeypatch):
    policy = attention_dispatch.get_inference_policy((12, 0))
    monkeypatch.setattr(attention_dispatch, "get_inference_policy", lambda: policy)
    assert forward._resolve_eager_backend("auto", _workload((1024,))) == "sdpa"
    assert (
        forward._resolve_eager_backend("auto", _workload((1024, 1024)))
        == "flash"
    )
    assert forward._resolve_eager_backend("flash", _workload((1,))) == "flash"


def test_attention_workload_from_mask():
    params = ModernBertParams(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=22,
        norm_eps=1e-5,
        global_rope_theta=160_000.0,
        local_rope_theta=10_000.0,
        local_attention=128,
        global_attn_every_n_layers=3,
    )
    ids = torch.zeros((3, 8), dtype=torch.long)
    mask = torch.tensor(
        [[1] * 8, [1] * 5 + [0] * 3, [1] * 2 + [0] * 6], dtype=torch.long
    )
    workload = forward._attention_workload(params, ids, mask)
    assert workload.n_sequences == 3
    assert workload.live_tokens == 15
    assert workload.padded_seqlen == 8
    assert workload.padding_tokens == 9


def test_untested_cards_inherit_compute_capability_policy():
    assert attention_dispatch.get_inference_policy(
        (8, 0), "NVIDIA A30"
    ) is attention_dispatch.get_inference_policy(
        (8, 0), "NVIDIA A100-SXM4-40GB"
    )
    assert attention_dispatch.get_inference_policy(
        (8, 9), "NVIDIA GeForce RTX 4090"
    ) is attention_dispatch.get_inference_policy((8, 9), "NVIDIA L40S")
    assert attention_dispatch.get_inference_policy(
        (10, 0), "some other sm100"
    ) is attention_dispatch.get_inference_policy((10, 0), "NVIDIA B200")
    assert attention_dispatch.get_inference_policy((9, 0)) is not None


def test_unknown_compute_capability_uses_generic_score(monkeypatch):
    generic = attention_dispatch.get_inference_policy((12, 0))
    assert attention_dispatch.get_inference_policy(
        (8, 6), "NVIDIA GeForce RTX 3090"
    ) is generic
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert attention_dispatch.get_inference_policy() is None
