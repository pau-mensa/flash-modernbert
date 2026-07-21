"""Autograd MaxSim for all-pairs ragged queries and documents.

Unlike ``flash_maxsim_batched_train``, neither token dimension is rectangular.
Query/document embeddings are flat buffers and both sequence boundaries are
described by ``cu_seqlens``.  Forward stores one document-local winner per real
query token and query/document pair; backward uses those winners for a fused
dQ accumulation + FP32 atomic dD scatter.

Builds on flash-maxsim (https://github.com/roipony/flash-maxsim, Apache 2.0).
"""

import torch
import triton
import triton.language as tl


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def _round_to_bucket(x):
    """Round up to nearest bucket: 32, 64, 128, 256, 512, 1024, 2048, 4096."""
    for b in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        if x <= b:
            return b
    return _next_pow2(x)


@triton.jit
def _packed_train_fwd_kernel(
    Q_ptr, D_ptr, cu_q_ptr, cu_d_ptr, scores_ptr, argmax_ptr,
    Nq, Nd, total_q,
    max_Lq_bucket: tl.constexpr, max_Ld_bucket: tl.constexpr,
    d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_t, stride_q_d,
    stride_d_t, stride_d_d,
    stride_s_q, stride_s_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pair_id = tl.program_id(0)
    query_id = pair_id // Nd
    doc_id = pair_id % Nd
    if query_id >= Nq:
        return

    q_start = tl.load(cu_q_ptr + query_id).to(tl.int64)
    q_end = tl.load(cu_q_ptr + query_id + 1).to(tl.int64)
    d_start = tl.load(cu_d_ptr + doc_id).to(tl.int64)
    d_end = tl.load(cu_d_ptr + doc_id + 1).to(tl.int64)
    q_len = (q_end - q_start).to(tl.int32)
    d_len = (d_end - d_start).to(tl.int32)

    k = tl.arange(0, d_pad)
    k_valid = k < d
    d_tile = tl.arange(0, BLOCK_D)
    score = tl.zeros([], dtype=tl.float32)

    for q_block in range(0, q_len, BLOCK_Q):
        q_offset = q_block + tl.arange(0, BLOCK_Q)
        q_valid = q_offset < q_len
        q_values = tl.load(
            Q_ptr
            + (q_start + q_offset[:, None]) * stride_q_t
            + k[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_valid[None, :],
            other=0.0,
        ).to(tl.float16)

        maxima = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        winners = tl.zeros([BLOCK_Q], dtype=tl.int32)
        for d_block in range(0, d_len, BLOCK_D):
            d_offset = d_block + d_tile
            d_valid = d_offset < d_len
            d_values = tl.load(
                D_ptr
                + (d_start + d_offset[:, None]) * stride_d_t
                + k[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_valid[None, :],
                other=0.0,
            ).to(tl.float16)
            similarities = tl.dot(q_values, tl.trans(d_values))
            similarities = tl.where(
                d_valid[None, :], similarities, float("-inf")
            )
            tile_max = tl.max(similarities, axis=1)
            tile_winner = tl.argmax(similarities, axis=1).to(tl.int32) + d_block
            update = tile_max > maxima
            winners = tl.where(update, tile_winner, winners)
            maxima = tl.maximum(maxima, tile_max)

        score += tl.sum(tl.where(q_valid, maxima, 0.0))
        # Doc-major layout makes the BLOCK_Q stores contiguous. There is no
        # padded Lq dimension: every stored entry belongs to a real query token.
        tl.store(
            argmax_ptr + doc_id * total_q + q_start + q_offset,
            winners,
            mask=q_valid,
        )

    tl.store(scores_ptr + query_id * stride_s_q + doc_id * stride_s_d, score)


@triton.jit
def _packed_train_bwd_kernel(
    Q_ptr, D_ptr, cu_d_ptr, query_ids_ptr, argmax_ptr, grad_scores_ptr,
    grad_Q_ptr, grad_D_ptr,
    Nd, total_q,
    d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_t, stride_q_d,
    stride_d_t, stride_d_d,
    stride_gs_q, stride_gs_d,
    stride_gq_t, stride_gq_d,
    stride_gd_t, stride_gd_d,
):
    query_token = tl.program_id(0)
    if query_token >= total_q:
        return

    query_id = tl.load(query_ids_ptr + query_token)
    k = tl.arange(0, d_pad)
    k_valid = k < d
    q_values = tl.load(
        Q_ptr + query_token * stride_q_t + k * stride_q_d,
        mask=k_valid,
        other=0.0,
    ).to(tl.float32)
    grad_q = tl.zeros([d_pad], dtype=tl.float32)

    for doc_id in range(0, Nd):
        grad_score = tl.load(
            grad_scores_ptr + query_id * stride_gs_q + doc_id * stride_gs_d
        ).to(tl.float32)
        winner = tl.load(argmax_ptr + doc_id * total_q + query_token).to(tl.int64)
        doc_start = tl.load(cu_d_ptr + doc_id).to(tl.int64)
        doc_token = doc_start + winner
        d_values = tl.load(
            D_ptr + doc_token * stride_d_t + k * stride_d_d,
            mask=k_valid,
            other=0.0,
        ).to(tl.float32)
        grad_q += grad_score * d_values
        tl.atomic_add(
            grad_D_ptr + doc_token * stride_gd_t + k * stride_gd_d,
            grad_score * q_values,
            mask=k_valid,
        )

    tl.store(
        grad_Q_ptr + query_token * stride_gq_t + k * stride_gq_d,
        grad_q,
        mask=k_valid,
    )


def _validate_packed(name, values, cu_seqlens):
    if values.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [total_tokens, d]")
    if not values.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"{name} cu_seqlens must be rank-1 with at least two entries")


class _FlashMaxSimPackedBatchedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, D, cu_seqlens_q, cu_seqlens_d, max_seqlen_q, max_seqlen_d):
        _validate_packed("Q", Q, cu_seqlens_q)
        _validate_packed("D", D, cu_seqlens_d)
        if Q.shape[1] != D.shape[1]:
            raise ValueError("query and document embedding dimensions must match")
        if Q.device != D.device:
            raise ValueError("query and document embeddings must share a device")

        device = Q.device
        Q = Q.contiguous()
        D = D.contiguous()
        cu_q = cu_seqlens_q.to(device=device, dtype=torch.int32).contiguous()
        cu_d = cu_seqlens_d.to(device=device, dtype=torch.int32).contiguous()
        Nq = cu_q.numel() - 1
        Nd = cu_d.numel() - 1
        total_q = Q.shape[0]
        d = Q.shape[1]
        d_pad = _next_pow2(max(d, 16))
        if max_seqlen_q is None:
            max_seqlen_q = int((cu_q[1:] - cu_q[:-1]).max())
        if max_seqlen_d is None:
            max_seqlen_d = int((cu_d[1:] - cu_d[:-1]).max())
        if max_seqlen_q <= 0 or max_seqlen_d <= 0:
            raise ValueError("packed sequences must contain at least one token")

        if d < d_pad:
            Q_kernel = torch.zeros((Q.shape[0], d_pad), dtype=Q.dtype, device=device)
            D_kernel = torch.zeros((D.shape[0], d_pad), dtype=D.dtype, device=device)
            Q_kernel[:, :d] = Q
            D_kernel[:, :d] = D
        else:
            Q_kernel, D_kernel = Q, D

        scores = torch.empty((Nq, Nd), dtype=torch.float32, device=device)
        argmax = torch.empty((Nd, total_q), dtype=torch.int32, device=device)
        _packed_train_fwd_kernel[(Nq * Nd,)](
            Q_kernel, D_kernel, cu_q, cu_d, scores, argmax,
            Nq, Nd, total_q,
            _round_to_bucket(max_seqlen_q), _round_to_bucket(max_seqlen_d),
            d, d_pad,
            Q_kernel.stride(0), Q_kernel.stride(1),
            D_kernel.stride(0), D_kernel.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_Q=32,
            BLOCK_D=64,
            num_warps=4,
            num_stages=2,
        )
        query_lengths = cu_q[1:] - cu_q[:-1]
        query_ids = torch.repeat_interleave(
            torch.arange(Nq, device=device, dtype=torch.int32), query_lengths
        )
        ctx.save_for_backward(Q_kernel, D_kernel, cu_d, query_ids, argmax)
        ctx.shape = (Nd, total_q, d, d_pad)
        ctx.orig_shapes = (Q.shape, D.shape)
        ctx.orig_dtypes = (Q.dtype, D.dtype)
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D, cu_d, query_ids, argmax = ctx.saved_tensors
        Nd, total_q, d, d_pad = ctx.shape
        grad_scores = grad_scores.contiguous().float()
        grad_Q_kernel = torch.empty_like(Q)
        grad_D_fp32 = torch.zeros(D.shape, dtype=torch.float32, device=D.device)
        _packed_train_bwd_kernel[(total_q,)](
            Q, D, cu_d, query_ids, argmax, grad_scores,
            grad_Q_kernel, grad_D_fp32,
            Nd, total_q,
            d, d_pad,
            Q.stride(0), Q.stride(1),
            D.stride(0), D.stride(1),
            grad_scores.stride(0), grad_scores.stride(1),
            grad_Q_kernel.stride(0), grad_Q_kernel.stride(1),
            grad_D_fp32.stride(0), grad_D_fp32.stride(1),
            num_warps=4,
            num_stages=2,
        )
        q_shape, d_shape = ctx.orig_shapes
        q_dtype, d_dtype = ctx.orig_dtypes
        grad_Q = grad_Q_kernel[: q_shape[0], : q_shape[1]].to(q_dtype)
        grad_D = grad_D_fp32[: d_shape[0], : d_shape[1]].to(d_dtype)
        return grad_Q, grad_D, None, None, None, None


def flash_maxsim_packed_batched_train(
    Q_packed,
    D_packed,
    cu_seqlens_q,
    cu_seqlens_d,
    max_seqlen_q=None,
    max_seqlen_d=None,
):
    """Differentiable all-pairs MaxSim over two ragged token buffers.

    Returns a float32 ``[Nq, Nd]`` score matrix. Backward returns gradients in
    the input dtypes and never creates padded token dimensions.
    """
    return _FlashMaxSimPackedBatchedFn.apply(
        Q_packed,
        D_packed,
        cu_seqlens_q,
        cu_seqlens_d,
        max_seqlen_q,
        max_seqlen_d,
    )
