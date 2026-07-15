"""Distribution-aware eager SDPA / FlashAttention dispatch.

The raw workload retains the packed sequence-length distribution. A calibrated policy
compiles those statistics into one monotonically ordered score. Untested cards inherit
a compute-capability anchor, and unknown CUDA capabilities use a generic score; runtime
dispatch never falls back to sequence length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def _local_pairs(length: int, half_window: int) -> int:
    """Ordered pairs satisfying ``abs(query - key) <= half_window``."""
    if length <= half_window + 1:
        return length * length
    return length * (2 * half_window + 1) - half_window * (half_window + 1)


@dataclass(frozen=True)
class AttentionWorkload:
    """Cheap integer statistics describing padded SDPA and packed varlen FA work."""

    n_sequences: int
    live_tokens: int
    max_seqlen: int
    padded_seqlen: int
    padded_tokens: int
    padding_tokens: int
    dense_pairs: int
    global_pairs: int
    local_pairs: int
    saved_pairs: int

    @classmethod
    def from_lengths(
        cls,
        lengths: Sequence[int],
        *,
        padded_seqlen: int | None = None,
        half_window: int = 0,
        global_layers: int = 0,
        local_layers: int = 0,
    ) -> "AttentionWorkload":
        values = tuple(int(length) for length in lengths)
        if not values or any(length < 0 for length in values):
            raise ValueError(
                "sequence lengths must be a non-empty sequence of non-negative integers"
            )
        n_sequences = len(values)
        live_tokens = sum(values)
        max_seqlen = max(values)
        padded_seqlen = max_seqlen if padded_seqlen is None else int(padded_seqlen)
        if padded_seqlen < max_seqlen:
            raise ValueError(
                f"padded_seqlen={padded_seqlen} is smaller than max length {max_seqlen}"
            )
        padded_tokens = n_sequences * padded_seqlen
        dense_pairs = n_sequences * padded_seqlen * padded_seqlen
        global_pairs = sum(length * length for length in values)
        local_pair_count = sum(_local_pairs(length, half_window) for length in values)
        saved_pairs = (
            global_layers * (dense_pairs - global_pairs)
            + local_layers * (dense_pairs - local_pair_count)
        )
        return cls(
            n_sequences=n_sequences,
            live_tokens=live_tokens,
            max_seqlen=max_seqlen,
            padded_seqlen=padded_seqlen,
            padded_tokens=padded_tokens,
            padding_tokens=padded_tokens - live_tokens,
            dense_pairs=dense_pairs,
            global_pairs=global_pairs,
            local_pairs=local_pair_count,
            saved_pairs=saved_pairs,
        )

    @property
    def effective_seqlen(self) -> float:
        return self.global_pairs / self.live_tokens if self.live_tokens else 0.0


@dataclass(frozen=True)
class FlashScorePolicy:
    """Integer linear score and conservative FlashAttention crossover.

    Zero-defaulted coefficients let each calibrated card retain only the workload
    terms that improve its measured winner boundary.  Python integers are deliberate:
    pair-count policies can exceed int32 without adding any tensor work.
    """

    threshold: int
    live_token_weight: int = 0
    padded_token_weight: int = 0
    padded_seqlen_weight: int = 0
    padding_token_weight: int = 0
    sequence_weight: int = 0
    global_pair_weight: int = 0
    local_pair_weight: int = 0
    saved_pair_weight: int = 0

    def score(self, workload: AttentionWorkload) -> int:
        return (
            self.live_token_weight * workload.live_tokens
            + self.padded_token_weight * workload.padded_tokens
            + self.padded_seqlen_weight * workload.padded_seqlen
            + self.padding_token_weight * workload.padding_tokens
            + self.sequence_weight * workload.n_sequences
            + self.global_pair_weight * workload.global_pairs
            + self.local_pair_weight * workload.local_pairs
            + self.saved_pair_weight * workload.saved_pairs
        )

    def use_flash(self, workload: AttentionWorkload) -> bool:
        return self.score(workload) >= self.threshold


_GENERIC_INFERENCE_POLICY = FlashScorePolicy(
    padded_token_weight=2,
    padded_seqlen_weight=6,
    padding_token_weight=1,
    threshold=8_800,
)

_A100_POLICY = FlashScorePolicy(
    live_token_weight=13_312,
    padded_seqlen_weight=-39_936,
    saved_pair_weight=1,
    threshold=85_332_928,
)

_L40S_POLICY = FlashScorePolicy(
    padded_token_weight=7,
    padded_seqlen_weight=8,
    threshold=31_744,
)

_HOPPER_H200_H100_POLICY = FlashScorePolicy(
    live_token_weight=12_183,
    saved_pair_weight=1,
    threshold=204_871_680,
)

_B200_POLICY = FlashScorePolicy(
    local_pair_weight=88,
    saved_pair_weight=1,
    threshold=197_107_200,
)


# bf16 ModernBERT-base eager inference. Each entry was calibrated on 115
# equal/fixed-M/ragged points and refined with synchronized 9-sample boundary runs.
# Exact names retain measurement provenance. Untested cards inherit their compute
# capability's measured anchor below; wholly unknown capabilities use the generic score.
_INFERENCE_POLICY_BY_DEVICE = {
    ((12, 0), "RTX 5090"): _GENERIC_INFERENCE_POLICY,
    # A100 SXM4 40GB, sm_80, compiled FA2.
    ((8, 0), "A100-SXM4-40GB"): _A100_POLICY,
    # L40S, sm_89, compiled FA2.
    ((8, 9), "L40S"): _L40S_POLICY,
    # H200 measurement, shared with H100 by explicit policy choice (both sm_90).
    ((9, 0), "H200"): _HOPPER_H200_H100_POLICY,
    ((9, 0), "H100"): _HOPPER_H200_H100_POLICY,
    # B200, sm_100, CuteDSL FA4 4.0.0b16.
    ((10, 0), "B200"): _B200_POLICY,
}

_INFERENCE_POLICY_BY_CC = {
    (8, 0): _A100_POLICY,
    (8, 9): _L40S_POLICY,
    (9, 0): _HOPPER_H200_H100_POLICY,
    (10, 0): _B200_POLICY,
    (12, 0): _GENERIC_INFERENCE_POLICY,
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


def get_inference_policy(
    compute_capability: tuple[int, int] | None = None,
    device_name: str | None = None,
) -> FlashScorePolicy | None:
    """Return the best available score policy for a CUDA device.

    Resolution order is exact measured card, measured compute-capability anchor, then
    the generic distribution-aware policy. Only a missing CUDA device returns ``None``.
    """
    infer_device = compute_capability is None
    cc = current_compute_capability() if infer_device else compute_capability
    if cc is None:
        return None
    if device_name is None and infer_device:
        device_name = current_device_name()

    candidates = [
        (name_fragment, policy)
        for (candidate_cc, name_fragment), policy in _INFERENCE_POLICY_BY_DEVICE.items()
        if candidate_cc == cc
    ]
    if device_name is not None:
        normalized = device_name.casefold()
        exact = next(
            (
                policy
                for name_fragment, policy in candidates
                if name_fragment.casefold() in normalized
            ),
            None,
        )
        if exact is not None:
            return exact

    return _INFERENCE_POLICY_BY_CC.get(cc, _GENERIC_INFERENCE_POLICY)
