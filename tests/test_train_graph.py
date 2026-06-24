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
"""B1 — the training-graph runner: capture the encoder fwd+bwd and replay it
inside autograd, reproducing the eager fused tail's gradients.

Drives the `_TrainGraphRunner` directly (no `prepare()`). The reference is the
*same* dense-mask, capture-safe forward the runner captures — the only forward a
fixed graph can replay — so any gap is the capture/replay itself, not the
dense-vs-flash mask choice. Checks the three properties GradCache relies on:
one-chunk fidelity, cross-chunk accumulation into a persistent `.grad`, and
deterministic replay.

    uv run pytest tests/test_train_graph.py -q     # on a validated GPU (sm_120 here)
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

    return AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda().train()


@pytest.fixture(scope="module")
def params(encoder):
    from flash_modernbert.config import ModernBertParams

    return ModernBertParams.from_hf_config(encoder.config)


@pytest.fixture(scope="module")
def weights(encoder):
    ws = [p for p in encoder.parameters() if p.requires_grad]
    for p in ws:  # prime .grad buffers (training-graph mode never set_to_none's them)
        p.grad = torch.zeros_like(p)
    return ws


@pytest.fixture(scope="module")
def runner(encoder, params):
    """One runner holding all buckets — the real-usage pattern. Every bucket
    captures on the shared default capture stream, so cross-bucket AccumulateGrad
    streams stay consistent (a fresh runner per test would leave a stale-stream
    graph alive that breaks the next capture)."""
    from flash_modernbert.train_graph import TrainGraphConfig, build_train_runner

    # max_seq high so every parametrized shape (incl. S=288) actually captures —
    # this fixture tests capture/replay *fidelity*; the runtime queries-only gate
    # (max_seq, default 64) is covered separately by test_long_seq_falls_back.
    return build_train_runner(encoder, params, TrainGraphConfig(warmup=3, max_seq=4096))


def _batch(b, s, vocab, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(5, vocab, (b, s), generator=g, device="cuda")
    mask = torch.ones((b, s), dtype=torch.long, device="cuda")
    grad_seed = torch.randn((b, s, 768), generator=g, device="cuda", dtype=torch.bfloat16)
    return ids, mask, grad_seed


def _cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def _rel_l2(a, b):
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-9)).item()


def _zero(weights):
    for p in weights:
        p.grad.zero_()


def _flat_grads(weights):
    return torch.cat([p.grad.float().flatten() for p in weights])


def _eager_dense_grads(encoder, params, weights, ids, mask, grad_seed):
    """The eager equivalent of what the runner captures: the dense-mask,
    capture-safe forward, differentiated by `grad_seed`. Uses `autograd.grad`
    (not `.backward()`) so it leaves no AccumulateGrad node on the encoder params
    — which would otherwise linger on the default stream and break the next
    capture (the real GradCache path never accumulates eagerly into them)."""
    from flash_modernbert import forward

    p = forward.prologue(encoder, params, ids, mask, dense_mask=True, capture_safe=True)
    out = forward.core(
        encoder, params, p.x, p.cos_global, p.sin_global, p.cos_local, p.sin_local,
        p.full_mask, p.sliding_mask, backend="sdpa",
    )
    grads = torch.autograd.grad(out, weights, grad_seed, allow_unused=True)
    flat = torch.cat([g.float().flatten() for g in grads if g is not None])
    return out.detach(), flat


# Short S is bit-exact; a longer S sits in the bf16 band (a different SDPA
# backward kernel is selected under capture than under eager .backward()).
# validate.py's gate is cos 0.997, so cos > 0.999 here is comfortably inside it.
GRAD_COS = 0.999
GRAD_RELL2 = 2e-2


@pytest.mark.parametrize("b,s", [(16, 32), (32, 32), (64, 32), (32, 64), (16, 288)])
def test_graph_grads_match_eager(encoder, params, weights, runner, b, s):
    ids, mask, grad_seed = _batch(b, s, encoder.config.vocab_size, seed=b * 1000 + s)

    # Capture happens here, before any eager .backward() — the real GradCache
    # situation (encoder grads only ever flow through this graph).
    _zero(weights)
    out_graph = runner(ids, mask)
    out_graph.backward(grad_seed)
    g_graph = _flat_grads(weights)
    fwd_graph = out_graph.detach()

    fwd_eager, g_eager = _eager_dense_grads(encoder, params, weights, ids, mask, grad_seed)

    assert _cos(fwd_graph, fwd_eager) > 0.99
    assert _cos(g_graph, g_eager) > GRAD_COS
    assert _rel_l2(g_graph, g_eager) < GRAD_RELL2


def test_replay_is_deterministic(encoder, params, weights, runner):
    ids, mask, grad_seed = _batch(16, 32, encoder.config.vocab_size, seed=7)

    _zero(weights)
    runner(ids, mask).backward(grad_seed)
    g1 = _flat_grads(weights)
    _zero(weights)
    runner(ids, mask).backward(grad_seed)
    g2 = _flat_grads(weights)

    assert torch.equal(g1, g2)  # graph replay is bit-stable


def test_grads_accumulate_across_chunks(encoder, params, weights, runner):
    """GradCache replays the chunk fwd+bwd many times per step without zeroing
    between chunks; the per-chunk grads must sum into the persistent .grad."""
    ids, mask, grad_seed = _batch(16, 32, encoder.config.vocab_size, seed=11)

    _zero(weights)
    runner(ids, mask).backward(grad_seed)
    g_one = _flat_grads(weights)

    _zero(weights)
    for _ in range(3):  # three chunks, no zero between
        runner(ids, mask).backward(grad_seed)
    g_three = _flat_grads(weights)

    # 3× accumulation carries bf16 rounding (2× is exact via exponent shift, the
    # third add rounds), so check the band, not bit-equality.
    assert _rel_l2(g_three, 3 * g_one) < 5e-3


def test_autocast_capture_matches_eager_and_tracks_weight_updates():
    """B2 — the PyLate recipe: fp32 master weights under bf16 autocast. The graph
    must capture with the bf16 weight casts recorded *inline* (cache_enabled=False)
    so each replay recasts from the persistent fp32 master, tracking the
    optimizer's in-place updates rather than replaying a stale captured copy."""
    from transformers import AutoModel
    from flash_modernbert.config import ModernBertParams
    from flash_modernbert import forward
    from flash_modernbert.train_graph import TrainGraphConfig, build_train_runner

    enc = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).cuda().train()
    params = ModernBertParams.from_hf_config(enc.config)
    w = [p for p in enc.parameters() if p.requires_grad]
    for p in w:
        p.grad = torch.zeros_like(p)
    opt = torch.optim.AdamW(w, lr=1e-3)
    runner = build_train_runner(enc, params, TrainGraphConfig(warmup=3))

    ids, mask, _ = _batch(16, 32, enc.config.vocab_size, seed=5)
    grad_seed = torch.randn((16, 32, 768), device="cuda", dtype=torch.bfloat16)

    def eager_out():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = forward.prologue(enc, params, ids, mask, dense_mask=True, capture_safe=True)
            return forward.core(
                enc, params, p.x, p.cos_global, p.sin_global, p.cos_local, p.sin_local,
                p.full_mask, p.sliding_mask, backend="sdpa",
            )

    prev = None
    for _ in range(3):
        opt.zero_grad(set_to_none=False)  # keep .grad buffers live (graph invariant)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = runner(ids, mask)
        assert out.dtype == torch.bfloat16          # autocast output
        out.backward(grad_seed)
        assert w[0].grad.dtype == torch.float32      # fp32 master grads
        go = out.detach().float()

        assert _cos(go, eager_out().float()) > 0.99  # graph tracks current weights
        if prev is not None:
            assert (go - prev).abs().max().item() > 0  # output moved with the update
        prev = go
        opt.step()  # updates fp32 master in place → next replay must recast


def test_oversized_shape_falls_back_to_eager(encoder, params, weights):
    """A shape beyond max_tokens is never captured — it runs the eager fused
    forward (still autograd-correct, just un-graphed)."""
    from flash_modernbert.train_graph import TrainGraphConfig, build_train_runner

    ids, mask, grad_seed = _batch(8, 64, encoder.config.vocab_size, seed=3)
    runner = build_train_runner(encoder, params, TrainGraphConfig(max_tokens=1, warmup=3))

    _zero(weights)
    out = runner(ids, mask)
    out.backward(grad_seed)

    assert len(runner._cache) == 0  # nothing captured
    assert all(p.grad is not None for p in weights)


def test_long_seq_falls_back_to_eager(encoder, params, weights):
    """The queries-only gate: with max_seq=64, a short (query) shape is captured but
    a long (doc) shape falls straight through to eager — the per-call queries-vs-docs
    split. Both still produce correct grads."""
    from flash_modernbert.train_graph import TrainGraphConfig, build_train_runner

    runner = build_train_runner(encoder, params, TrainGraphConfig(max_seq=64, warmup=3))
    vocab = encoder.config.vocab_size

    q_ids, q_mask, q_grad = _batch(16, 32, vocab, seed=101)   # query: s=32 <= 64
    d_ids, d_mask, d_grad = _batch(16, 300, vocab, seed=202)  # doc:   s=300 > 64

    _zero(weights)
    runner(q_ids, q_mask).backward(q_grad)
    assert (16, 32) in runner._cache          # query captured
    assert all(p.grad is not None for p in weights)

    _zero(weights)
    runner(d_ids, d_mask).backward(d_grad)
    assert (16, 300) not in runner._cache     # doc never captured (ran eager)
    assert len(runner._cache) == 1            # only the query bucket lives
    assert all(p.grad is not None for p in weights)


def test_end_to_end_prepare_train_graph_engages_and_matches_eager():
    """B4 wiring — `prepare(train_cuda_graph=True)` routes the grad-enabled forward
    through the train runner (it engages, the bucket cache populates), and a real
    training step through the patched forward reproduces the eager fused (dense)
    backward bit-exactly. (The graph captures the dense-mask forward; comparing to
    the *default* flash-path forward would instead surface the bf16 dense-vs-flash
    backward band, which is characterized in the benchmarks, not gated here.)"""
    import flash_modernbert as fm
    from flash_modernbert import forward
    from flash_modernbert.config import ModernBertParams
    from transformers import AutoModel
    from flash_modernbert.state import get_state

    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda().train()
    params = ModernBertParams.from_hf_config(model.config)
    w = [p for p in model.parameters() if p.requires_grad]
    for p in w:
        p.grad = torch.zeros_like(p)
    ids, mask, grad_seed = _batch(16, 32, model.config.vocab_size, seed=21)

    fm.prepare(model, train_cuda_graph=True, validate=False)

    # A real fwd+bwd through the patched forward (grad on → routes to train runner).
    for p in w:
        p.grad.zero_()
    out_graph = model(input_ids=ids, attention_mask=mask).last_hidden_state
    out_graph.backward(grad_seed)
    g_graph = torch.cat([p.grad.float().flatten() for p in w])

    state = get_state(model)
    assert state.train_graph_runner is not None
    assert len(state.train_graph_runner._cache) == 1  # the (16, 32) bucket captured

    # Reference: the same dense-mask forward the runner captures, eager.
    for p in w:
        p.grad.zero_()
    _, g_eager = _eager_dense_grads(model, params, w, ids, mask, grad_seed)
    assert _cos(g_graph, g_eager) > GRAD_COS
    assert _rel_l2(g_graph, g_eager) < GRAD_RELL2
