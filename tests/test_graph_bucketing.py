# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["flash-modernbert", "transformers", "pytest"]
#
# [tool.uv.sources]
# flash-modernbert = { path = "../", editable = true }
# torch = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""A3 — the graph runner's bucketing policy: batch-dim bucketing, a bounded LRU
cache under a stream of variable shapes, and seq_bucket pre-capture.

Drives the `_GraphRunner` directly (no `prepare()`) so each policy is exercised
in isolation and checked against the eager fused forward for correctness.

    uv run pytest tests/test_graph_bucketing.py -q     # on a validated GPU (sm_120 here)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused/graph path requires a CUDA GPU"
)

MODEL_ID = "answerdotai/ModernBERT-base"


@pytest.fixture(scope="module")
def encoder():
    from transformers import AutoModel

    return AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda().eval()


@pytest.fixture(scope="module")
def params(encoder):
    from flash_modernbert.config import ModernBertParams

    return ModernBertParams.from_hf_config(encoder.config)


def _batch(b, s, vocab, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(5, vocab, (b, s), generator=g, device="cuda")
    mask = torch.ones((b, s), dtype=torch.long, device="cuda")
    return ids, mask


FIDELITY = 0.9999  # graph replay vs eager run of the same captured region


def _cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def _fused(encoder, params, ids, mask):
    """The eager fused forward — what an out-of-bucket batch falls back to."""
    from flash_modernbert import forward

    with torch.no_grad():
        return forward.fused_forward(encoder, params, ids, mask)


def _replay_ref(encoder, params, ids, mask, bb, sb):
    """An eager run of the *exact region the runner captures* — prologue with a
    dense mask + core, at the runner's padded (bb, sb) shape, sliced back. The
    graph replay must reproduce this near bit-for-bit (capture/replay fidelity);
    comparing against the eager fused_forward instead would fold in the flash-vs-
    dense SDPA backend band, which is a fused-tail property, not a graph one."""
    from flash_modernbert import forward

    b, s = ids.shape
    pad_ids = torch.zeros((bb, sb), dtype=ids.dtype, device="cuda")
    pad_ids[:b, :s] = ids
    pad_mask = torch.zeros((bb, sb), dtype=torch.long, device="cuda")
    pad_mask[:b, :s] = mask
    if bb > b:
        pad_mask[b:bb, :s] = 1
    with torch.no_grad():
        p = forward.prologue(encoder, params, pad_ids, pad_mask, dense_mask=True)
        out = forward.core(
            encoder, params, p.x, p.cos_global, p.sin_global,
            p.cos_local, p.sin_local, p.full_mask, p.sliding_mask,
        )
    return out[:b, :s]


# ---------------------------------------------------------------------------
# Batch-dim bucketing: one graph at max_batch, partial batches padded + sliced,
# oversized batches eager — all numerically equal to the eager forward.
# ---------------------------------------------------------------------------


def test_batch_bucketing_one_graph_for_all_b_le_max(encoder, params):
    from flash_modernbert.graph import GraphConfig, build_runner

    vocab = int(encoder.config.vocab_size)
    runner = build_runner(encoder, params, GraphConfig(pad_to=64, max_batch=8))

    with torch.no_grad():
        for b in (8, 5, 1):  # full, partial, single — all <= max_batch, same S bucket
            ids, mask = _batch(b, 64, vocab, seed=b)
            out = runner(ids, mask)
            assert out.shape[0] == b  # sliced back to the real batch
            ref = _replay_ref(encoder, params, ids, mask, bb=8, sb=64)
            assert _cos(out, ref) >= FIDELITY

    # Every b <= max_batch shared the single (max_batch, seq_bucket) graph.
    assert list(runner._cache.keys()) == [(8, 64)]


def test_batch_over_max_falls_back_to_eager(encoder, params):
    from flash_modernbert.graph import GraphConfig, build_runner

    vocab = int(encoder.config.vocab_size)
    runner = build_runner(encoder, params, GraphConfig(pad_to=64, max_batch=4))

    with torch.no_grad():
        ids, mask = _batch(16, 64, vocab)  # b > max_batch
        out = runner(ids, mask)
        assert out.shape[0] == 16
        assert _cos(out, _fused(encoder, params, ids, mask)) >= FIDELITY
    assert len(runner._cache) == 0  # nothing captured — went straight to eager


def test_per_b_graphs_when_max_batch_none(encoder, params):
    from flash_modernbert.graph import GraphConfig, build_runner

    vocab = int(encoder.config.vocab_size)
    runner = build_runner(encoder, params, GraphConfig(pad_to=64, max_graphs=8))

    with torch.no_grad():
        for b in (2, 4):
            ids, mask = _batch(b, 64, vocab, seed=b)
            runner(ids, mask)
    assert set(runner._cache.keys()) == {(2, 64), (4, 64)}  # a graph per distinct B


# ---------------------------------------------------------------------------
# Bounded cache: a stream of distinct sequence buckets must not grow the cache
# past max_graphs; LRU keeps the most-recently-replayed buckets.
# ---------------------------------------------------------------------------


def test_lru_cache_stays_bounded(encoder, params):
    from flash_modernbert.graph import GraphConfig, build_runner

    vocab = int(encoder.config.vocab_size)
    runner = build_runner(encoder, params, GraphConfig(pad_to=64, max_batch=2, max_graphs=3))

    seq_buckets = [64, 128, 192, 256, 320]  # five distinct buckets, cap is three
    with torch.no_grad():
        for s in seq_buckets:
            ids, mask = _batch(2, s, vocab, seed=s)
            out = runner(ids, mask)
            assert _cos(out, _replay_ref(encoder, params, ids, mask, bb=2, sb=s)) >= FIDELITY
            assert len(runner._cache) <= 3  # never exceeds the cap mid-stream

    # The three most-recently-seen buckets survive; the first two were evicted.
    assert list(runner._cache.keys()) == [(2, 192), (2, 256), (2, 320)]


def test_lru_promotes_on_replay(encoder, params):
    from flash_modernbert.graph import GraphConfig, build_runner

    vocab = int(encoder.config.vocab_size)
    runner = build_runner(encoder, params, GraphConfig(pad_to=64, max_batch=2, max_graphs=2))

    with torch.no_grad():
        a = _batch(2, 64, vocab, seed=1)
        b = _batch(2, 128, vocab, seed=2)
        runner(*a)            # cache: [64]
        runner(*b)            # cache: [64, 128]
        runner(*a)            # replay 64 -> promoted: [128, 64]
        c = _batch(2, 192, vocab, seed=3)
        runner(*c)            # insert 192 evicts LRU (128): [64, 192]
    assert list(runner._cache.keys()) == [(2, 64), (2, 192)]


# ---------------------------------------------------------------------------
# seq_bucket pre-capture at build time (the prepare()-time path).
# ---------------------------------------------------------------------------


def test_seq_buckets_precaptured_at_build(encoder, params):
    from flash_modernbert.graph import GraphConfig, build_runner

    runner = build_runner(
        encoder, params,
        GraphConfig(pad_to=64, max_batch=4, seq_buckets=(64, 128)),
    )
    assert set(runner._cache.keys()) == {(4, 64), (4, 128)}  # captured eagerly at build


def test_seq_buckets_without_max_batch_warns(encoder, params):
    from flash_modernbert.graph import GraphConfig, build_runner

    with pytest.warns(UserWarning, match="seq_buckets needs max_batch"):
        runner = build_runner(encoder, params, GraphConfig(seq_buckets=(64,)))
    assert len(runner._cache) == 0  # nothing pre-captured without a batch dim
