"""End-to-end ColBERT encoding structures for the packed-inference showcase.

This is showcase scaffolding, not part of the ``flash_modernbert`` package.  It
keeps the comparison explicit:

* ``encode_padded``: rectangular encoder batches and a rectangular token index.
* ``encode_packed``: no-padding tokenization, ``packed_forward``, a flat Dense
  projection, and a flat scoring index with ``cu_seqlens``.

Both paths consume the same pre-tokenized token-id sequences and apply PyLate's
document skiplist before scoring.  Tokenization is deliberately outside timed GPU
measurements so Python tokenizer throughput cannot obscure the data-structure A/B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from flash_modernbert import forward
from flash_modernbert.config import ModernBertParams
from flash_modernbert.locate import find_encoder

from packed_collator import pack_sequences


@dataclass(frozen=True)
class TokenizedTexts:
    """ColBERT token ids before any rectangular collation."""

    sequences: tuple[Tensor, ...]
    is_query: bool

    def __post_init__(self) -> None:
        if not self.sequences:
            raise ValueError("TokenizedTexts cannot be empty")
        if any(x.ndim != 1 or x.numel() == 0 for x in self.sequences):
            raise ValueError("every tokenized sequence must be a non-empty vector")

    @property
    def lengths(self) -> list[int]:
        return [x.numel() for x in self.sequences]


@dataclass(frozen=True)
class PackedIndex:
    """Ragged token embeddings; document/query ``i`` is ``embeddings[cu[i]:cu[i+1]]``."""

    embeddings: Tensor
    cu_seqlens: Tensor
    max_seqlen: int
    encoder_tokens: int

    @property
    def n_sequences(self) -> int:
        return self.cu_seqlens.numel() - 1

    @property
    def scoring_tokens(self) -> int:
        return self.embeddings.shape[0]

    @property
    def storage_bytes(self) -> int:
        return self.embeddings.numel() * self.embeddings.element_size()

    def sequence(self, index: int) -> Tensor:
        start = int(self.cu_seqlens[index])
        end = int(self.cu_seqlens[index + 1])
        return self.embeddings[start:end]


@dataclass(frozen=True)
class PaddedIndex:
    """Rectangular token embeddings and their scoring mask."""

    embeddings: Tensor
    mask: Tensor
    encoder_tokens: int

    @property
    def n_sequences(self) -> int:
        return self.embeddings.shape[0]

    @property
    def scoring_tokens(self) -> int:
        return int(self.mask.sum())

    @property
    def storage_bytes(self) -> int:
        return self.embeddings.numel() * self.embeddings.element_size()

    def sequence(self, index: int) -> Tensor:
        return self.embeddings[index, self.mask[index]]


def tokenize_colbert_no_padding(model, texts: Sequence[str], *, is_query: bool) -> TokenizedTexts:
    """Mirror ``ColBERT.tokenize`` without ever requesting tokenizer padding.

    PyLate inserts the query/document prefix immediately after the first special
    token.  We reproduce that operation on Python lists so the collator receives
    only real token ids. Query expansion with non-attended mask tokens has two
    contradictory token roles (excluded from attention, retained for scoring),
    which a single ``cu_seqlens`` cannot express; the target GTE checkpoint has
    expansion disabled, and unsupported checkpoints fail loudly here.
    """
    if not texts:
        raise ValueError("cannot tokenize an empty text collection")
    if is_query and model.do_query_expansion and not model.attend_to_expansion_tokens:
        raise ValueError(
            "packed_forward cannot preserve PyLate query expansion tokens that are "
            "masked from attention but retained for scoring; disable query expansion"
        )

    transformer = model._first_module()
    cleaned = [str(text).strip() for text in texts]
    if transformer.do_lower_case:
        cleaned = [text.lower() for text in cleaned]

    max_length = model.query_length if is_query else model.document_length
    prefix_id = model.query_prefix_id if is_query else model.document_prefix_id
    tokenizer_max = max_length - 1 if prefix_id is not None else max_length
    padding: bool | str = False
    if is_query and model.do_query_expansion:
        padding = "max_length"

    encoded = transformer.tokenizer(
        cleaned,
        padding=padding,
        truncation="longest_first",
        max_length=tokenizer_max,
        add_special_tokens=True,
    )
    sequences = []
    for ids in encoded["input_ids"]:
        if prefix_id is not None:
            ids = [ids[0], prefix_id, *ids[1:]]
        sequences.append(torch.tensor(ids, dtype=torch.long))
    return TokenizedTexts(tuple(sequences), is_query=is_query)


def assert_tokenization_matches_pylate(
    model, texts: Sequence[str], tokenized: TokenizedTexts
) -> None:
    """Runtime provenance/parity guard against the installed PyLate tokenizer path."""
    reference = model.tokenize(list(texts), is_query=tokenized.is_query)
    max_length = model.query_length if tokenized.is_query else model.document_length
    prefix_id = model.query_prefix_id if tokenized.is_query else model.document_prefix_id
    if len(texts) != len(tokenized.sequences):
        raise AssertionError("text/token sequence count mismatch")
    for row, ids in enumerate(tokenized.sequences):
        if ids.numel() > max_length:
            raise AssertionError(f"token sequence exceeds configured cap {max_length}")
        if prefix_id is not None and (ids.numel() < 2 or int(ids[1]) != prefix_id):
            raise AssertionError("ColBERT prefix token is not at position 1")
        ref_ids = reference["input_ids"][row][reference["attention_mask"][row].bool()]
        if not torch.equal(ids, ref_ids.cpu()):
            raise AssertionError(f"no-padding tokenization differs from PyLate at row {row}")


def _scoring_masks(model, tokenized: TokenizedTexts) -> list[Tensor]:
    if tokenized.is_query:
        return [torch.ones_like(ids, dtype=torch.bool) for ids in tokenized.sequences]
    skip = set(int(x) for x in model.skiplist)
    masks = [torch.tensor([int(x) not in skip for x in ids], dtype=torch.bool) for ids in tokenized.sequences]
    if any(not bool(mask.any()) for mask in masks):
        raise ValueError("document skiplist removed every token from a document")
    return masks


def _embedding_dim(model) -> int:
    module = model[-1]
    dim = getattr(module, "out_features", None)
    if dim is None:
        raise TypeError("packed showcase expects a ColBERT projection with out_features")
    return int(dim)


def _project_and_normalize(model, hidden: Tensor) -> Tensor:
    features = {"token_embeddings": hidden}
    for module in list(model)[1:]:
        features = module(features)
    return F.normalize(features["token_embeddings"], p=2, dim=-1)


def _batch_order(tokenized: TokenizedTexts, sort_by_length: bool) -> list[int]:
    order = list(range(len(tokenized.sequences)))
    if sort_by_length:
        order.sort(key=lambda i: tokenized.sequences[i].numel(), reverse=True)
    return order


@torch.inference_mode()
def encode_packed(
    model,
    tokenized: TokenizedTexts,
    *,
    batch_size: int,
    sort_by_length: bool = True,
) -> PackedIndex:
    """Encode and project a collection without a token-padding allocation."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    encoder = find_encoder(model)
    params = ModernBertParams.from_hf_config(encoder.config)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    scoring_masks = _scoring_masks(model, tokenized)
    scoring_lengths = [int(mask.sum()) for mask in scoring_masks]
    cu_cpu = torch.nn.functional.pad(
        torch.tensor(scoring_lengths, dtype=torch.int32).cumsum(0), (1, 0)
    )
    output = torch.empty(
        (int(cu_cpu[-1]), _embedding_dim(model)), device=device, dtype=dtype
    )
    order = _batch_order(tokenized, sort_by_length)

    for offset in range(0, len(order), batch_size):
        indices = order[offset : offset + batch_size]
        batch = pack_sequences([tokenized.sequences[i] for i in indices], device=device)
        hidden = forward.packed_forward(encoder, params, *batch.forward_args())
        projected = _project_and_normalize(model, hidden)
        # Vectorized scatter: the batch's scoring mask (aligned to the packed output) and
        # its destination rows in `output`, then one masked select + one index_copy. This
        # replaces a per-document slice / H2D / copy loop whose host dispatch dominated at
        # short docs / small batch (a per-document loop is never faster and adds launches).
        batch_mask = torch.cat([scoring_masks[i] for i in indices]).to(device)
        dest = torch.cat(
            [
                torch.arange(int(cu_cpu[i]), int(cu_cpu[i + 1]), device=device)
                for i in indices
            ]
        )
        output.index_copy_(0, dest, projected[batch_mask])

    return PackedIndex(
        embeddings=output,
        cu_seqlens=cu_cpu.to(device),
        max_seqlen=max(scoring_lengths),
        encoder_tokens=sum(tokenized.lengths),
    )


@torch.inference_mode()
def encode_padded(
    model,
    tokenized: TokenizedTexts,
    *,
    batch_size: int,
    sort_by_length: bool = True,
    length_buckets: Sequence[int] | None = None,
) -> PaddedIndex:
    """Encode rectangular batches and retain a rectangular scoring index."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    pad_id = int(model.tokenizer.pad_token_id or 0)
    scoring_masks = _scoring_masks(model, tokenized)
    scoring_lengths = [int(mask.sum()) for mask in scoring_masks]
    max_scoring = max(scoring_lengths)
    n = len(tokenized.sequences)
    output = torch.zeros((n, max_scoring, _embedding_dim(model)), device=device, dtype=dtype)
    output_mask = torch.zeros((n, max_scoring), device=device, dtype=torch.bool)
    order = _batch_order(tokenized, sort_by_length)
    encoder_tokens = 0

    for offset in range(0, len(order), batch_size):
        indices = order[offset : offset + batch_size]
        sequences = [tokenized.sequences[i] for i in indices]
        real_batch_max = max(ids.numel() for ids in sequences)
        if length_buckets:
            batch_max = next(
                (int(bucket) for bucket in sorted(length_buckets) if bucket >= real_batch_max),
                None,
            )
            if batch_max is None:
                raise ValueError(
                    f"no length bucket can hold a {real_batch_max}-token sequence"
                )
        else:
            batch_max = real_batch_max
        ids = torch.full((len(indices), batch_max), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(indices), batch_max), dtype=torch.long, device=device)
        for row, sequence in enumerate(sequences):
            length = sequence.numel()
            ids[row, :length].copy_(sequence.to(device))
            attention_mask[row, :length] = 1
        encoder_tokens += ids.numel()

        features = {"input_ids": ids, "attention_mask": attention_mask}
        for module in model:
            features = module(features)
        projected = F.normalize(features["token_embeddings"], p=2, dim=-1)
        for row, original_i in enumerate(indices):
            real_len = tokenized.sequences[original_i].numel()
            selected = projected[row, :real_len][scoring_masks[original_i].to(device)]
            score_len = selected.shape[0]
            output[original_i, :score_len].copy_(selected)
            output_mask[original_i, :score_len] = True

    return PaddedIndex(output, output_mask, encoder_tokens=encoder_tokens)


def index_summary(index: PackedIndex | PaddedIndex) -> dict:
    return {
        "sequences": index.n_sequences,
        "encoder_tokens": index.encoder_tokens,
        "scoring_tokens": index.scoring_tokens,
        "storage_bytes": index.storage_bytes,
        "storage_gb": index.storage_bytes / 1e9,
        "max_scoring_length": (
            index.max_seqlen if isinstance(index, PackedIndex) else index.embeddings.shape[1]
        ),
        "representation": "packed" if isinstance(index, PackedIndex) else "padded",
    }
