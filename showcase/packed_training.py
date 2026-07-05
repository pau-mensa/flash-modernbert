"""Packed and padded GradCache reference implementations for P2.

Both paths use the same three phases and the same groups:

1. chunked no-grad encoder pass, caching projected token representations;
2. matrix-free all-pairs MaxSim + cross entropy, producing representation grads;
3. RNG-replayed chunked encoder recompute/backward with those cached grads.

Only representation changes. ``padded`` keeps rectangular encoder inputs and
rectangular d=128 caches; ``packed`` keeps flat token buffers and cu_seqlens from
the collator through MaxSim backward.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from packed_encoders import forward
from packed_encoders.config import ModernBertParams
from packed_encoders.locate import find_encoder

from packed_collator import pack_sequences
from packed_index import tokenize_colbert_no_padding


@dataclass(frozen=True)
class RoleBatch:
    sequences: tuple[Tensor, ...]
    scoring_masks: tuple[Tensor, ...]
    is_query: bool

    def __post_init__(self) -> None:
        if not self.sequences or len(self.sequences) != len(self.scoring_masks):
            raise ValueError("role sequences/masks must be non-empty and aligned")
        for ids, mask in zip(self.sequences, self.scoring_masks):
            if ids.ndim != 1 or mask.shape != ids.shape or not bool(mask.any()):
                raise ValueError("every role sequence needs a non-empty aligned scoring mask")

    @property
    def n_sequences(self) -> int:
        return len(self.sequences)

    @property
    def encoder_lengths(self) -> list[int]:
        return [x.numel() for x in self.sequences]

    @property
    def scoring_lengths(self) -> list[int]:
        return [int(x.sum()) for x in self.scoring_masks]

    @property
    def max_encoder_length(self) -> int:
        return max(self.encoder_lengths)

    @property
    def max_scoring_length(self) -> int:
        return max(self.scoring_lengths)

    def scoring_cu_seqlens(self, device) -> Tensor:
        lengths = torch.tensor(self.scoring_lengths, dtype=torch.int32, device=device)
        return F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))


@dataclass(frozen=True)
class ColBERTTrainBatch:
    queries: RoleBatch
    documents: RoleBatch
    labels: Tensor
    documents_per_group: int

    @property
    def n_groups(self) -> int:
        return self.queries.n_sequences


@dataclass(frozen=True)
class GradCacheStats:
    loss: Tensor
    cache_bytes: int
    input_bytes: int
    real_encoder_tokens: int
    collated_encoder_tokens: int


@dataclass(frozen=True)
class _RNGState:
    cpu: Tensor
    cuda: Tensor
    device_index: int


def _capture_rng(device: torch.device) -> _RNGState:
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _RNGState(torch.get_rng_state(), torch.cuda.get_rng_state(index), index)


@contextmanager
def _replay_rng(state: _RNGState):
    with torch.random.fork_rng(devices=[state.device_index], enabled=True):
        torch.set_rng_state(state.cpu)
        torch.cuda.set_rng_state(state.cuda, state.device_index)
        yield


def build_colbert_train_batch(
    model,
    groups: Sequence[dict],
    *,
    num_negatives: int,
) -> ColBERTTrainBatch:
    """Tokenize identical query/(positive, negatives) groups without padding.

    Documents stay query-major: ``q0:[pos,neg...]``, then ``q1:[...]``. Thus
    the positive label for query ``i`` is ``i * documents_per_group``, matching
    PyLate CachedContrastive's semantics.
    """
    if not groups or num_negatives < 0:
        raise ValueError("groups must be non-empty and num_negatives non-negative")
    query_texts = [group["query"] for group in groups]
    doc_texts: list[str] = []
    for group in groups:
        negatives = list(group.get("negatives") or [])
        if num_negatives and not negatives:
            raise ValueError("a training group has no negatives")
        doc_texts.append(group["positive"])
        for i in range(num_negatives):
            doc_texts.append(negatives[i % len(negatives)])

    query_tokens = tokenize_colbert_no_padding(model, query_texts, is_query=True)
    doc_tokens = tokenize_colbert_no_padding(model, doc_texts, is_query=False)
    query_masks = tuple(torch.ones_like(ids, dtype=torch.bool) for ids in query_tokens.sequences)
    skiplist = set(int(x) for x in model.skiplist)
    doc_masks = tuple(
        torch.tensor([int(token) not in skiplist for token in ids], dtype=torch.bool)
        for ids in doc_tokens.sequences
    )
    docs_per_group = num_negatives + 1
    labels = torch.arange(len(groups), dtype=torch.long) * docs_per_group
    return ColBERTTrainBatch(
        queries=RoleBatch(query_tokens.sequences, query_masks, True),
        documents=RoleBatch(doc_tokens.sequences, doc_masks, False),
        labels=labels,
        documents_per_group=docs_per_group,
    )


def combine_colbert_train_batches(
    batches: Sequence[ColBERTTrainBatch],
) -> ColBERTTrainBatch:
    """Concatenate pre-tokenized groups without introducing a padded boundary."""
    if not batches:
        raise ValueError("cannot combine an empty batch list")
    documents_per_group = batches[0].documents_per_group
    if any(x.documents_per_group != documents_per_group for x in batches):
        raise ValueError("all groups must have the same negative count")
    q_sequences = tuple(sequence for batch in batches for sequence in batch.queries.sequences)
    q_masks = tuple(mask for batch in batches for mask in batch.queries.scoring_masks)
    d_sequences = tuple(sequence for batch in batches for sequence in batch.documents.sequences)
    d_masks = tuple(mask for batch in batches for mask in batch.documents.scoring_masks)
    labels = torch.arange(len(q_sequences), dtype=torch.long) * documents_per_group
    return ColBERTTrainBatch(
        queries=RoleBatch(q_sequences, q_masks, True),
        documents=RoleBatch(d_sequences, d_masks, False),
        labels=labels,
        documents_per_group=documents_per_group,
    )


def build_token_budget_batches(
    model,
    groups: Sequence[dict],
    *,
    num_negatives: int,
    token_budget: int,
) -> list[ColBERTTrainBatch]:
    """Greedily fill steps by real query+document encoder tokens."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    singletons = [
        build_colbert_train_batch(model, [group], num_negatives=num_negatives)
        for group in groups
    ]
    output = []
    current = []
    current_tokens = 0
    for batch in singletons:
        tokens = sum(batch.queries.encoder_lengths) + sum(batch.documents.encoder_lengths)
        if current and current_tokens + tokens > token_budget:
            output.append(combine_colbert_train_batches(current))
            current = []
            current_tokens = 0
        current.append(batch)
        current_tokens += tokens
    if current:
        output.append(combine_colbert_train_batches(current))
    return output


def _project_and_normalize(model, hidden: Tensor) -> Tensor:
    features = {"token_embeddings": hidden}
    for module in list(model)[1:]:
        features = module(features)
    return F.normalize(features["token_embeddings"], p=2, dim=-1)


def _encode_packed_chunk(model, role: RoleBatch, begin: int, end: int) -> Tensor:
    device = next(model.parameters()).device
    encoder = find_encoder(model)
    params = ModernBertParams.from_hf_config(encoder.config)
    packed = pack_sequences(role.sequences[begin:end], device=device)
    hidden = forward.packed_forward(encoder, params, *packed.forward_args())
    projected = _project_and_normalize(model, hidden)
    offsets = [0]
    for sequence in role.sequences[begin:end]:
        offsets.append(offsets[-1] + sequence.numel())
    selected = [
        projected[offsets[i] : offsets[i + 1]][
            role.scoring_masks[begin + i].to(device)
        ]
        for i in range(end - begin)
    ]
    return torch.cat(selected, dim=0)


def _encode_padded_chunk(model, role: RoleBatch, begin: int, end: int) -> Tensor:
    """Fixed input/cache shapes, with real tokens compacted before the Dense head.

    Compacting at the head boundary gives T1' and T2 the same projection GEMM.
    Otherwise bf16 GEMM algorithm changes from rectangular-M to real-token-M
    perturb scores enough that the faithful 0.01 temperature amplifies them into
    different losses. The persistent T1' cache returned here remains rectangular.
    """
    device = next(model.parameters()).device
    pad_id = int(model.tokenizer.pad_token_id or 0)
    n = end - begin
    input_ids = torch.full(
        (n, role.max_encoder_length), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(role.sequences[begin:end]):
        length = sequence.numel()
        input_ids[row, :length].copy_(sequence.to(device))
        attention_mask[row, :length] = 1

    features = model[0]({"input_ids": input_ids, "attention_mask": attention_mask})
    real_lengths = role.encoder_lengths[begin:end]
    hidden = torch.cat(
        [features["token_embeddings"][row, :length] for row, length in enumerate(real_lengths)]
    )
    projected = _project_and_normalize(model, hidden)
    offsets = [0]
    for length in real_lengths:
        offsets.append(offsets[-1] + length)
    rows = []
    for row in range(n):
        original = begin + row
        selected = projected[offsets[row] : offsets[row + 1]][
            role.scoring_masks[original].to(device)
        ]
        rows.append(F.pad(selected, (0, 0, 0, role.max_scoring_length - selected.shape[0])))
    return torch.stack(rows)


def _cache_role(model, role: RoleBatch, mini_batch_size: int, packed: bool):
    encode = _encode_packed_chunk if packed else _encode_padded_chunk
    device = next(model.parameters()).device
    chunks = []
    states = []
    with torch.no_grad():
        for begin in range(0, role.n_sequences, mini_batch_size):
            end = min(begin + mini_batch_size, role.n_sequences)
            states.append(_capture_rng(device))
            chunks.append(encode(model, role, begin, end).detach())
    return torch.cat(chunks, dim=0), states


def _recompute_role(
    model,
    role: RoleBatch,
    upstream_grad: Tensor,
    states: list[_RNGState],
    mini_batch_size: int,
    packed: bool,
) -> None:
    encode = _encode_packed_chunk if packed else _encode_padded_chunk
    packed_offset = 0
    for chunk_index, begin in enumerate(range(0, role.n_sequences, mini_batch_size)):
        end = min(begin + mini_batch_size, role.n_sequences)
        with _replay_rng(states[chunk_index]):
            output = encode(model, role, begin, end)
        if packed:
            count = sum(role.scoring_lengths[begin:end])
            grad = upstream_grad[packed_offset : packed_offset + count]
            packed_offset += count
        else:
            grad = upstream_grad[begin:end]
        torch.autograd.backward(output, grad)


def _input_and_token_stats(batch: ColBERTTrainBatch, packed: bool) -> tuple[int, int, int]:
    roles = (batch.queries, batch.documents)
    real = sum(sum(role.encoder_lengths) for role in roles)
    if packed:
        collated = real
        # ids int64 + position ids int64 + one int32 length per sequence (+ boundaries)
        input_bytes = real * 16 + sum((role.n_sequences * 2 + 1) * 4 for role in roles)
    else:
        collated = sum(role.n_sequences * role.max_encoder_length for role in roles)
        # input_ids + attention_mask are both int64 in this reference collator.
        input_bytes = collated * 16
    return input_bytes, real, collated


def gradcache_step(
    model,
    batch: ColBERTTrainBatch,
    *,
    mini_batch_size: int,
    score_mini_batch_size: int | None = None,
    temperature: float,
    packed: bool,
) -> GradCacheStats:
    """Run one complete GradCache step and accumulate parameter gradients."""
    if mini_batch_size <= 0 or temperature <= 0:
        raise ValueError("mini_batch_size and temperature must be positive")
    score_mini_batch_size = score_mini_batch_size or mini_batch_size
    if score_mini_batch_size <= 0:
        raise ValueError("score_mini_batch_size must be positive")
    device = next(model.parameters()).device
    q_cache, q_states = _cache_role(model, batch.queries, mini_batch_size, packed)
    d_cache, d_states = _cache_role(model, batch.documents, mini_batch_size, packed)
    labels = batch.labels.to(device)

    from flash_maxsim import flash_maxsim_packed_batched_train

    cu_d = batch.documents.scoring_cu_seqlens(device)
    if packed:
        q_loss = q_cache.detach()
        d_loss = d_cache.detach()
    else:
        # T1' keeps the long-lived cache rectangular, then removes cache padding
        # at the loss boundary. We scatter the resulting flat gradients back
        # into rectangular tensors for pass 3 below.
        q_loss = torch.cat(
            [q_cache[i, :length] for i, length in enumerate(batch.queries.scoring_lengths)]
        )
        d_loss = torch.cat(
            [d_cache[i, :length] for i, length in enumerate(batch.documents.scoring_lengths)]
        )

    q_offsets = [0]
    for length in batch.queries.scoring_lengths:
        q_offsets.append(q_offsets[-1] + length)
    grad_q_loss = torch.empty_like(q_loss)
    grad_d_loss_fp32 = torch.zeros_like(d_loss, dtype=torch.float32)
    loss_value = torch.zeros((), device=device, dtype=torch.float32)
    for begin in range(0, batch.n_groups, score_mini_batch_size):
        end = min(begin + score_mini_batch_size, batch.n_groups)
        q_chunk = q_loss[q_offsets[begin] : q_offsets[end]].detach().requires_grad_(True)
        d_chunk = d_loss.detach().requires_grad_(True)
        chunk_lengths = batch.queries.scoring_lengths[begin:end]
        cu_q_chunk = F.pad(
            torch.tensor(chunk_lengths, device=device, dtype=torch.int32).cumsum(0),
            (1, 0),
        )
        scores = flash_maxsim_packed_batched_train(
            q_chunk,
            d_chunk,
            cu_q_chunk,
            cu_d,
            max(chunk_lengths),
            batch.documents.max_scoring_length,
        )
        loss_chunk = F.cross_entropy(
            scores / temperature,
            labels[begin:end],
            reduction="sum",
        ) / batch.n_groups
        grad_q_chunk, grad_d_chunk = torch.autograd.grad(
            loss_chunk, (q_chunk, d_chunk)
        )
        grad_q_loss[q_offsets[begin] : q_offsets[end]].copy_(grad_q_chunk)
        grad_d_loss_fp32.add_(grad_d_chunk.float())
        loss_value += loss_chunk.detach()
        del scores, loss_chunk, q_chunk, d_chunk, grad_q_chunk, grad_d_chunk

    grad_d_loss = grad_d_loss_fp32.to(d_loss.dtype)
    if packed:
        grad_q = grad_q_loss
        grad_d = grad_d_loss
    else:
        grad_q = torch.zeros_like(q_cache)
        grad_d = torch.zeros_like(d_cache)
        for index, length in enumerate(batch.queries.scoring_lengths):
            grad_q[index, :length].copy_(
                grad_q_loss[q_offsets[index] : q_offsets[index + 1]]
            )
        d_offset = 0
        for index, length in enumerate(batch.documents.scoring_lengths):
            grad_d[index, :length].copy_(grad_d_loss[d_offset : d_offset + length])
            d_offset += length
    cache_bytes = (
        q_cache.numel() * q_cache.element_size()
        + d_cache.numel() * d_cache.element_size()
        + grad_q.numel() * grad_q.element_size()
        + grad_d.numel() * grad_d.element_size()
    )
    del q_cache, d_cache, q_loss, d_loss, grad_q_loss, grad_d_loss, grad_d_loss_fp32

    _recompute_role(
        model, batch.queries, grad_q, q_states, mini_batch_size, packed
    )
    _recompute_role(
        model, batch.documents, grad_d, d_states, mini_batch_size, packed
    )
    input_bytes, real_tokens, collated_tokens = _input_and_token_stats(batch, packed)
    return GradCacheStats(
        loss=loss_value,
        cache_bytes=cache_bytes,
        input_bytes=input_bytes,
        real_encoder_tokens=real_tokens,
        collated_encoder_tokens=collated_tokens,
    )
