"""Forward-only Triton attention for packed short ModernBERT sequences.

This is intentionally narrow: bf16 BSHD tensors, D=64, self attention, no causal
mask/dropout/backward, and Smax<=128.  The grid exposes sequence/head/query-block
parallelism and reads sequence boundaries directly from ``cu_seqlens``. Unsupported
inputs are rejected here; the integration layer decides whether to fall back to FA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import triton
import triton.language as tl


_LOG2E = tl.constexpr(1.4426950408889634)


@dataclass(frozen=True)
class PackedShortAttentionConfig:
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


def _select_config(max_seqlen: int) -> PackedShortAttentionConfig:
    # A small static table, not runtime autotuning. These are deliberately modest
    # tiles: short packed batches need many independently schedulable CTAs more than
    # they need the long-S reuse strategy of a general FlashAttention kernel.
    if max_seqlen <= 32:
        return PackedShortAttentionConfig(16, 32, 4, 2)
    if max_seqlen <= 64:
        return PackedShortAttentionConfig(16, 64, 4, 2)
    return PackedShortAttentionConfig(64, 64, 4, 2)


def select_packed_short_attention_config(max_seqlen: int) -> dict[str, int]:
    """Return the static launch choice in JSON-friendly form."""
    return asdict(_select_config(int(max_seqlen)))


@triton.jit
def _packed_short_attention_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    cu_seqlens_ptr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_km: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vm: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    MAX_SEQLEN: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    HALF_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    sequence = tl.program_id(0)
    head = tl.program_id(1)
    query_block = tl.program_id(2)

    sequence_start = tl.load(cu_seqlens_ptr + sequence)
    sequence_end = tl.load(cu_seqlens_ptr + sequence + 1)
    sequence_length = sequence_end - sequence_start
    query_start = query_block * BLOCK_M

    # This is uniform across the CTA and avoids doing all key tiles for ragged
    # sequences whose real length does not reach this overlaunched query block.
    if query_start < sequence_length:
        query_pos = query_start + tl.arange(0, BLOCK_M)
        dim = tl.arange(0, HEAD_DIM)
        query_offsets = (
            (sequence_start + query_pos)[:, None] * stride_qm
            + head * stride_qh
            + dim[None, :] * stride_qd
        )
        q = tl.load(q_ptr + query_offsets, mask=query_pos[:, None] < sequence_length,
                    other=0.0)

        running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

        for key_start in range(0, MAX_SEQLEN, BLOCK_N):
            key_pos = key_start + tl.arange(0, BLOCK_N)
            key_valid = key_pos < sequence_length
            key_offsets = (
                (sequence_start + key_pos)[:, None] * stride_km
                + head * stride_kh
                + dim[None, :] * stride_kd
            )
            value_offsets = (
                (sequence_start + key_pos)[:, None] * stride_vm
                + head * stride_vh
                + dim[None, :] * stride_vd
            )
            k = tl.load(k_ptr + key_offsets, mask=key_valid[:, None], other=0.0)
            v = tl.load(v_ptr + value_offsets, mask=key_valid[:, None], other=0.0)

            scores = tl.dot(q, tl.trans(k)) * SOFTMAX_SCALE
            score_valid = key_valid[None, :]
            if IS_LOCAL:
                score_valid = score_valid & (
                    tl.abs(query_pos[:, None] - key_pos[None, :]) <= HALF_WINDOW
                )
            scores = tl.where(score_valid, scores, -float("inf"))

            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            alpha = tl.exp2((running_max - new_max) * _LOG2E)
            probabilities = tl.exp2((scores - new_max[:, None]) * _LOG2E)
            tile_sum = tl.sum(probabilities, axis=1)

            accumulator *= alpha[:, None]
            # bf16 probabilities select tensor-core MMA while tl.dot accumulates in
            # fp32. This is the same precision strategy as Triton's FA tutorial and
            # is checked against both FP32 reference and compiled FA in the harness.
            accumulator += tl.dot(probabilities.to(tl.bfloat16), v)
            running_sum = running_sum * alpha + tile_sum
            running_max = new_max

        accumulator /= running_sum[:, None]
        output_offsets = (
            (sequence_start + query_pos)[:, None] * stride_om
            + head * stride_oh
            + dim[None, :] * stride_od
        )
        tl.store(out_ptr + output_offsets, accumulator,
                 mask=query_pos[:, None] < sequence_length)


def packed_short_attention_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    half_window: int | None,
) -> bool:
    """Pure capability predicate used by the opt-in integration fallback."""
    return (
        q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and cu_seqlens.is_cuda
        and q.ndim == k.ndim == v.ndim == 3
        and q.shape == k.shape == v.shape
        and q.shape[1:] == (12, 64)
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and cu_seqlens.dtype == torch.int32
        and cu_seqlens.ndim == 1
        and cu_seqlens.numel() >= 2
        and 0 < int(max_seqlen) <= 128
        and (half_window is None or int(half_window) == 64)
        and not torch.is_grad_enabled()
    )


def packed_short_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    half_window: int | None,
    softmax_scale: float,
) -> torch.Tensor:
    """Compute packed noncausal self attention and return contiguous ``[M,H,64]``."""
    if not packed_short_attention_supported(
        q, k, v, cu_seqlens, max_seqlen, half_window=half_window
    ):
        raise ValueError(
            "packed short attention requires no-grad CUDA bf16 [M,12,64] Q/K/V, "
            "int32 cu_seqlens, Smax<=128, and global or local-64 attention"
        )
    return _launch_packed_short_attention(
        q, k, v, cu_seqlens, int(max_seqlen), half_window=half_window,
        softmax_scale=softmax_scale, config=_select_config(int(max_seqlen)),
    )


def _launch_packed_short_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    half_window: int | None,
    softmax_scale: float,
    config: PackedShortAttentionConfig,
) -> torch.Tensor:
    """Internal explicit-config entry used only by the offline tile sweep."""
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    n_sequences = cu_seqlens.numel() - 1
    query_blocks = triton.cdiv(int(max_seqlen), config.block_m)
    grid = (n_sequences, q.shape[1], query_blocks)
    _packed_short_attention_fwd[grid](
        q,
        k,
        v,
        output,
        cu_seqlens,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        MAX_SEQLEN=max_seqlen,
        SOFTMAX_SCALE=float(softmax_scale),
        IS_LOCAL=half_window is not None,
        HALF_WINDOW=0 if half_window is None else int(half_window),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        HEAD_DIM=64,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return output


__all__ = [
    "packed_short_attention",
    "packed_short_attention_supported",
    "select_packed_short_attention_config",
]
