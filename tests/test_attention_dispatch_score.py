"""Pure-logic coverage for packed Triton/Flash monotonic scores."""

from __future__ import annotations

import torch

from packed_encoders import attention_dispatch, forward
from packed_encoders.config import ModernBertParams


def _workload(lengths):
    return attention_dispatch.AttentionWorkload.from_lengths(lengths)


def test_workload_retains_capture_safe_shape_statistics():
    workload = _workload((10, 20, 5))
    assert workload.n_sequences == 3
    assert workload.live_tokens == 35
    assert workload.max_seqlen == 20
    assert workload.rectangle_tokens == 60
    assert workload.fragmentation_tokens == 25


def test_workload_rejects_impossible_summary():
    try:
        attention_dispatch.AttentionWorkload.from_summary(
            n_sequences=2, live_tokens=33, max_seqlen=16
        )
    except ValueError:
        pass
    else:  # pragma: no cover - assertion spelling keeps pytest optional here
        raise AssertionError("impossible packed workload was accepted")


def test_5090_policy_matches_guarded_live_token_boundary():
    policy = attention_dispatch.get_packed_inference_policy(
        (12, 0), "NVIDIA GeForce RTX 5090"
    )
    assert policy is not None
    assert not policy.use_flash(_workload((128,) * 160))  # M=20,480
    assert policy.use_flash(_workload((96,) * 216))       # M=20,736


def test_a100_policy_matches_refined_score_gap():
    policy = attention_dispatch.get_packed_inference_policy(
        (8, 0), "NVIDIA A100-SXM4-40GB"
    )
    assert policy is not None
    # Nearest rejected and selected rows under the rounded integer score.
    assert not policy.use_flash(_workload((64, 32) * 192))
    assert policy.use_flash(_workload((64, 32) * 256))


def test_l40s_policy_matches_refined_score_gap():
    policy = attention_dispatch.get_packed_inference_policy(
        (8, 9), "NVIDIA L40S"
    )
    assert policy is not None
    assert not policy.use_flash(_workload((64,) * 384))
    assert policy.use_flash(_workload((128,) * 192))


def test_l40s_graph_policy_moves_only_the_validated_boundary_row():
    eager = attention_dispatch.get_packed_inference_policy(
        (8, 9), "NVIDIA L40S"
    )
    graph = attention_dispatch.get_packed_inference_policy(
        (8, 9), "NVIDIA L40S", for_cuda_graph=True
    )
    assert eager is not None and graph is not None
    assert not eager.use_flash(_workload((64,) * 384))
    assert graph.use_flash(_workload((64,) * 384))
    assert not graph.use_flash(_workload((64,) * 320))
    assert not graph.use_flash(_workload((32, 16) * 512))


def test_h200_and_b200_use_bounded_no_crossover_policy():
    for cc, name in (((9, 0), "NVIDIA H200"), ((10, 0), "NVIDIA B200")):
        policy = attention_dispatch.get_packed_inference_policy(cc, name)
        assert policy is not None
        assert not policy.use_flash(_workload((128,) * 1024))  # measured M=131,072
        assert policy.use_flash(_workload((128,) * 1024 + (1,)))


def test_policies_do_not_transfer_to_uncalibrated_cards():
    assert attention_dispatch.get_packed_inference_policy(
        (12, 0), "some other sm120"
    ) is None
    assert attention_dispatch.get_packed_inference_policy(
        (8, 0), "NVIDIA A30"
    ) is None
    assert attention_dispatch.get_packed_inference_policy(
        (9, 0), "NVIDIA H100"
    ) is None


def test_resolver_uses_packed_score(monkeypatch):
    policy = attention_dispatch.get_packed_inference_policy(
        (12, 0), "NVIDIA RTX 5090"
    )
    monkeypatch.setattr(
        attention_dispatch, "get_packed_inference_policy", lambda **_kwargs: policy
    )
    with torch.no_grad():
        assert forward._resolve_eager_backend(
            "auto", _workload((128,) * 160), triton_supported=True,
            flash_available=True,
        ) == "triton"
        assert forward._resolve_eager_backend(
            "auto", _workload((96,) * 216), triton_supported=True,
            flash_available=True,
        ) == "flash"
        assert forward._resolve_eager_backend(
            "auto", _workload((32,)), triton_supported=False,
            flash_available=True,
        ) == "flash"
        assert forward._resolve_eager_backend(
            "auto", _workload((32,)), triton_supported=True,
            flash_available=False,
        ) == "triton"


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
    assert workload.max_seqlen == 8
    assert workload.rectangle_tokens == 24
    assert workload.fragmentation_tokens == 9


def test_no_cuda_device_has_no_policy(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert attention_dispatch.get_packed_inference_policy() is None
