"""Packed token-budget collator — **showcase scaffolding** (P0 of the packed paradigm).

NOT part of the shipped flash-modernbert surface (see `docs/packed_paradigm_showcase.md`
§8). It produces the already-packed input that `flash_modernbert.forward.packed_forward`
consumes — real tokens only, no padding, end-to-end. The shipped package keeps the `[B, S]`
drop-in forward; this collator is how the *showcase* opts into the fully-packed paradigm.

The one detail that has to be exact (or padded-vs-packed parity breaks): per-token
`position_ids` are the **within-sequence index** `arange(L_i)` for each sequence — HF
ModernBERT assigns RoPE positions by `arange(S)` regardless of padding, so for right-padded
data this is identical to the patched path's `indices % S` (`forward._unpad`). The unit
check `assert_positions_match_unpad` pins that equivalence.

Two modes (roadmap §5.1):
  - **same_groups** (P0): pack the exact same (query, positive, negatives) groups as the
    padded baseline, in the same order → apples-to-apples loss semantics.
  - **token_budget** (P2+): fill ~T real tokens/step regardless of the length distribution
    → the "you can now go bigger" upside. Stubbed here; built when P2 needs it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PackedBatch:
    """Already-packed encoder input — the argument bundle for `packed_forward`.

    `input_ids`/`position_ids` are `[total]`; `cu_seqlens` is `[n_seq + 1]` int32 prefix
    sums; `seqlens` is `[n_seq]` int32 (convenience). `groups` (optional) records, per
    packed sequence, a `(group_id, role)` tag so a downstream packed loss can gather each
    query/doc back out of the flat `[total, H]` encoder output without any repad."""

    input_ids: Tensor
    cu_seqlens: Tensor
    max_seqlen: int
    position_ids: Tensor
    seqlens: Tensor
    groups: list[tuple[int, str]] | None = None

    @property
    def n_seq(self) -> int:
        return self.seqlens.numel()

    @property
    def n_tokens(self) -> int:
        return self.input_ids.numel()

    def to(self, device) -> "PackedBatch":
        return PackedBatch(
            self.input_ids.to(device),
            self.cu_seqlens.to(device),
            self.max_seqlen,
            self.position_ids.to(device),
            self.seqlens.to(device),
            self.groups,
        )

    def forward_args(self) -> tuple[Tensor, Tensor, int, Tensor]:
        """The positional tuple `packed_forward(model, params, *args)` wants."""
        return self.input_ids, self.cu_seqlens, self.max_seqlen, self.position_ids


def pack_sequences(
    sequences: list[Tensor] | list[list[int]],
    *,
    device=None,
    groups: list[tuple[int, str]] | None = None,
) -> PackedBatch:
    """Pack variable-length token-id sequences into a `PackedBatch`.

    The core of the paradigm: concatenate real tokens (no padding), build `cu_seqlens`,
    and assign each token its within-sequence position `arange(L_i)`. This is the inverse
    construction of `forward._unpad` for right-padded data — verified by
    `assert_positions_match_unpad`."""
    seqs = [
        s if isinstance(s, Tensor) else torch.tensor(s, dtype=torch.long)
        for s in sequences
    ]
    if not seqs:
        raise ValueError("pack_sequences got no sequences")
    seqlens = torch.tensor([s.numel() for s in seqs], dtype=torch.int32)
    if int(seqlens.min()) == 0:
        raise ValueError("pack_sequences got an empty sequence (length 0)")
    input_ids = torch.cat([s.to(torch.long).reshape(-1) for s in seqs])
    cu_seqlens = torch.nn.functional.pad(
        seqlens.cumsum(0, dtype=torch.int32), (1, 0)
    )
    max_seqlen = int(seqlens.max())
    position_ids = torch.cat([torch.arange(int(n), dtype=torch.long) for n in seqlens])
    batch = PackedBatch(input_ids, cu_seqlens, max_seqlen, position_ids, seqlens, groups)
    return batch.to(device) if device is not None else batch


def pack_texts(
    texts: list[str],
    tokenizer,
    *,
    max_length: int = 8192,
    device=None,
    groups: list[tuple[int, str]] | None = None,
) -> PackedBatch:
    """Tokenize (no padding) and pack a flat list of texts — the convenience the parity
    script and inference indexer use. Adds special tokens to match the padded baseline's
    tokenization exactly."""
    seqs = [
        torch.tensor(
            tokenizer(t, truncation=True, max_length=max_length, add_special_tokens=True)[
                "input_ids"
            ],
            dtype=torch.long,
        )
        for t in texts
    ]
    return pack_sequences(seqs, device=device, groups=groups)


# --------------------------------------------------------------------------- #
# same_groups ColBERT collator (P0): pack the identical (q, pos, negs) groups
# --------------------------------------------------------------------------- #


def pack_colbert_groups(
    groups: list[dict],
    tokenizer,
    *,
    query_max_length: int = 256,
    doc_max_length: int = 8192,
    device=None,
) -> tuple[PackedBatch, PackedBatch]:
    """same_groups mode: pack a batch of ColBERT examples into one packed **query** batch
    and one packed **document** batch, preserving group order so the loss can pair them.

    Each example is `{"query": str, "positive": str, "negatives": [str, ...]}` — the exact
    same groups as the padded baseline (no token-budget repacking). Returns
    `(packed_queries, packed_docs)`:

    - `packed_queries`: one sequence per group's query; `groups[i] = (i, "query")`.
    - `packed_docs`: that group's positive then its negatives, contiguous;
      `groups[j] = (group_id, "pos"|"neg")` tags each doc's owning group and role.

    Encode each batch with `packed_forward` → `[total_q, H]` / `[total_d, H]`; slice per
    sequence with the respective `cu_seqlens` to feed a packed MaxSim loss. (The loss glue
    is P2; this fixes the data contract now and is what the P0 parity script drives.)"""
    q_texts: list[str] = []
    d_texts: list[str] = []
    q_groups: list[tuple[int, str]] = []
    d_groups: list[tuple[int, str]] = []
    for gid, ex in enumerate(groups):
        q_texts.append(ex["query"])
        q_groups.append((gid, "query"))
        d_texts.append(ex["positive"])
        d_groups.append((gid, "pos"))
        for neg in ex.get("negatives", []) or []:
            d_texts.append(neg)
            d_groups.append((gid, "neg"))
    packed_q = pack_texts(
        q_texts, tokenizer, max_length=query_max_length, device=device, groups=q_groups
    )
    packed_d = pack_texts(
        d_texts, tokenizer, max_length=doc_max_length, device=device, groups=d_groups
    )
    return packed_q, packed_d


# --------------------------------------------------------------------------- #
# Parity guard — the position_ids equivalence the whole claim rests on
# --------------------------------------------------------------------------- #


def assert_positions_match_unpad(packed: PackedBatch, attention_mask: Tensor) -> None:
    """Assert the collator's `position_ids`/`cu_seqlens` equal what `forward._unpad`
    derives from the right-padded `attention_mask` of the *same* sequences. This is the
    P0 correctness lock: if these diverge, RoPE positions differ and packed≠padded.

    `attention_mask` is the padded baseline's `[B, S]` mask for the same sequences in the
    same order (right-padded)."""
    from flash_modernbert import forward as _fwd  # local: showcase-only dep on internals

    am = attention_mask.cpu()
    b, s = am.shape
    dummy = torch.zeros(b, s, 1)
    _, _, cu_ref, max_ref, pos_ref = _fwd._unpad(dummy, am)
    if not torch.equal(packed.cu_seqlens.cpu(), cu_ref.cpu()):
        raise AssertionError(
            f"cu_seqlens mismatch:\n  collator={packed.cu_seqlens.tolist()}\n"
            f"  _unpad  ={cu_ref.tolist()}"
        )
    if packed.max_seqlen != max_ref:
        raise AssertionError(f"max_seqlen mismatch: {packed.max_seqlen} != {max_ref}")
    if not torch.equal(packed.position_ids.cpu(), pos_ref.cpu()):
        raise AssertionError("position_ids mismatch vs _unpad (HF arange convention)")
