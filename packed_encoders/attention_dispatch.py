"""Exact-card packed Triton/Flash attention dispatch policies.

All score inputs are host-visible shape statistics. No policy reads CUDA
``cu_seqlens`` values, so the same monotonic threshold comparison is safe before
eager execution and while building a fixed packed CUDA-graph bucket.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch


@dataclass(frozen=True)
class AttentionWorkload:
    """Capture-safe summary of an already-packed short-attention workload."""

    n_sequences: int
    live_tokens: int
    max_seqlen: int
    rectangle_tokens: int
    fragmentation_tokens: int

    @classmethod
    def from_lengths(cls, lengths: Sequence[int]) -> "AttentionWorkload":
        values = tuple(int(length) for length in lengths)
        if not values or any(length < 0 for length in values):
            raise ValueError(
                "sequence lengths must be a non-empty sequence of non-negative integers"
            )
        return cls.from_summary(
            n_sequences=len(values),
            live_tokens=sum(values),
            max_seqlen=max(values),
        )

    @classmethod
    def from_summary(
        cls, *, n_sequences: int, live_tokens: int, max_seqlen: int
    ) -> "AttentionWorkload":
        n, m, s = int(n_sequences), int(live_tokens), int(max_seqlen)
        if n <= 0 or m < 0 or s <= 0 or m > n * s:
            raise ValueError(
                f"invalid packed workload N={n}, M={m}, Smax={s}"
            )
        rectangle = n * s
        return cls(
            n_sequences=n,
            live_tokens=m,
            max_seqlen=s,
            rectangle_tokens=rectangle,
            fragmentation_tokens=rectangle - m,
        )


@dataclass(frozen=True)
class FlashScorePolicy:
    """Integer scalar score; Flash is selected at ``score >= threshold``."""

    threshold: int
    live_token_weight: int = 0
    sequence_weight: int = 0
    max_seqlen_weight: int = 0
    rectangle_pair_weight: int = 0
    mean_square_work_weight: int = 0
    live_max_weight: int = 0
    query_cta_weight: int = 0

    def score(self, workload: AttentionWorkload) -> int:
        n = workload.n_sequences
        m = workload.live_tokens
        s = workload.max_seqlen
        block_m = 16 if s <= 64 else 64
        query_ctas = n * ((s + block_m - 1) // block_m)
        return (
            self.live_token_weight * m
            + self.sequence_weight * n
            + self.max_seqlen_weight * s
            + self.rectangle_pair_weight * n * s * s
            + self.mean_square_work_weight * (m * m // n)
            + self.live_max_weight * m * s
            + self.query_cta_weight * query_ctas
        )

    def use_flash(self, workload: AttentionWorkload) -> bool:
        return self.score(workload) >= self.threshold


# Captured bf16 ModernBERT-base attention, weighted 8 global + 14 local-64 layers.
# A production Flash win requires t_triton/t_flash >= 1.03. Policies are exact-card
# only: GPU-kernel crossovers are not inherited by compute capability.
_RTX_5090_PACKED_POLICY = FlashScorePolicy(
    live_token_weight=1,
    threshold=20_736,
)

_A100_PACKED_POLICY = FlashScorePolicy(
    sequence_weight=-203_715,
    mean_square_work_weight=24,
    query_cta_weight=41_801,
    threshold=9_422_745,
)

_L40S_PACKED_POLICY = FlashScorePolicy(
    live_token_weight=4_260,
    rectangle_pair_weight=-7,
    query_cta_weight=-11_752,
    threshold=77_166_921,
)
# Full packed-forward graph capture moves exactly one guarded L40S calibration row:
# equal S=64, M=24,576. Keep the same monotonic score and lower only the graph
# threshold; the closest lower graph point remains below the 3% production guard.
_L40S_PACKED_GRAPH_POLICY = replace(
    _L40S_PACKED_POLICY,
    threshold=74_000_000,
)

# No guarded FA4 win was observed through M=131,072 on H200. B200 had no stable
# monotonic crossover (an isolated N=8 occupancy island moved between S=96 and S=128
# across repeated runs), so both exact-card
# policies retain Triton through the measured range and conservatively return to FA
# beyond it rather than extrapolating "always Triton".
_H200_PACKED_POLICY = FlashScorePolicy(
    live_token_weight=1,
    threshold=131_073,
)
_B200_PACKED_POLICY = FlashScorePolicy(
    live_token_weight=1,
    threshold=131_073,
)

_PACKED_INFERENCE_POLICY_BY_DEVICE = {
    ((12, 0), "RTX 5090"): _RTX_5090_PACKED_POLICY,
    ((8, 0), "A100-SXM4-40GB"): _A100_PACKED_POLICY,
    ((8, 9), "L40S"): _L40S_PACKED_POLICY,
    ((9, 0), "H200"): _H200_PACKED_POLICY,
    ((10, 0), "B200"): _B200_PACKED_POLICY,
}
_PACKED_GRAPH_POLICY_BY_DEVICE = {
    ((8, 9), "L40S"): _L40S_PACKED_GRAPH_POLICY,
}


def current_compute_capability() -> tuple[int, int] | None:
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability()
    except Exception:  # pragma: no cover - defensive no/duff CUDA
        return None


def current_device_name() -> str | None:
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name()
    except Exception:  # pragma: no cover - defensive no/duff CUDA
        return None


def get_packed_inference_policy(
    compute_capability: tuple[int, int] | None = None,
    device_name: str | None = None,
    *,
    for_cuda_graph: bool = False,
) -> FlashScorePolicy | None:
    """Return an exact-card packed Triton/Flash policy, if calibrated.

    ``None`` means prefer Flash when installed instead of borrowing another GPU's
    crossover. If Flash is absent, the caller still uses Triton in its supported
    envelope and SDPA outside it.
    """
    infer_device = compute_capability is None
    cc = current_compute_capability() if infer_device else compute_capability
    if cc is None:
        return None
    if device_name is None and infer_device:
        device_name = current_device_name()
    if device_name is None:
        return None
    normalized = device_name.casefold()
    if for_cuda_graph:
        graph_policy = next(
            (
                policy
                for (candidate_cc, name_fragment), policy
                in _PACKED_GRAPH_POLICY_BY_DEVICE.items()
                if candidate_cc == cc and name_fragment.casefold() in normalized
            ),
            None,
        )
        if graph_policy is not None:
            return graph_policy
    return next(
        (
            policy
            for (candidate_cc, name_fragment), policy
            in _PACKED_INFERENCE_POLICY_BY_DEVICE.items()
            if candidate_cc == cc and name_fragment.casefold() in normalized
        ),
        None,
    )
