"""CUDA-graph layer for **training** — capturing the encoder fwd+bwd.

The inference runner captures a no-grad forward and can't drive a backward. This
runner captures forward *and* backward and replays both inside autograd, so a
training step differentiates through it transparently. It targets short-S training,
where the launch floor (doubled by the backward) regresses vs stock (B16/S32: 0.83×).

Per exact `(batch, seq_len)` shape: a forward graph (from static id/mask buffers,
producing a static `out`) and a backward graph that copies in `grad_output` and
replays `autograd.grad`, accumulating into each persistent `param.grad` *inside* the
graph. That accumulation-inside-the-graph is the point: unlike `make_graphed_callables`
(one shared `grad_inputs` buffer, clobbered across GradCache chunks → wrong grads),
every chunk replay just `+=`'s into the static `.grad` — exactly the cross-chunk sum
GradCache wants.

Two caller invariants (enforced at the integration layer):

- **`.grad` buffers must stay live across replays** — the backward graph records an add
  into each `param.grad`'s address at capture, so train with `set_to_none=False`.
- **fwd→bwd must be paired per chunk** before the next forward, else the static
  forward activations are overwritten. GradCache's pass-3 already does this.
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from flash_modernbert import forward
from flash_modernbert.config import ModernBertParams


@dataclass(frozen=True)
class TrainGraphConfig:
    """Bucketing policy: one fwd+bwd graph pair per **exact** `(batch, seq_len)`.

    Unlike inference, the sequence is **not** padded into buckets. Padding leaves the
    forward bit-identical at real positions but shifts the bf16 backward drastically
    (padded-vs-unpadded grads come out near-uncorrelated, cos≈0.09, though fp32 grads
    are identical), so exact keying is required to match eager grads.

    `max_seq` makes this a queries-only feature: graphs win only at short S (by S≈300
    eager wins on latency and memory), so `s > max_seq` runs eager. In ColBERT this
    separates queries from documents at zero cost (PyLate encodes each group in its own
    call). Default 64 (ColBERT `query_length` ≤64); 5090 break-even ~128.

    Memory scales with `max_graphs`: each bucket pins its own activation pool for the
    runner's lifetime (the backward retains the forward's saved tensors). With `max_seq`
    keeping docs out, only the small query pool is pinned (~parity with eager); capturing
    docs too pushes reserved to ~1.2× eager (measure with `max_memory_reserved`, not
    `allocated`, which hides the private pool). A shape over `max_tokens` is never captured.
    """

    max_seq: int = 64       # graph only when s <= this (queries); longer runs eager
    max_graphs: int = 4
    max_tokens: int = 2 ** 20
    warmup: int = 3


@dataclass
class _StaticInputs:
    input_ids: Tensor
    attention_mask: Tensor

    def stage(self, input_ids: Tensor, attention_mask: Tensor | None) -> None:
        """Copy a real `(b, s)` batch into the static buffers (same exact shape —
        no padding). The mask defaults to all-ones when not supplied."""
        self.input_ids.copy_(input_ids)
        if attention_mask is None:
            self.attention_mask.fill_(1)
        else:
            self.attention_mask.copy_(attention_mask.to(self.attention_mask.dtype))


@dataclass
class _Captured:
    fwd_graph: "torch.cuda.CUDAGraph"
    bwd_graph: "torch.cuda.CUDAGraph"
    inputs: _StaticInputs
    out: Tensor          # static forward output (B, S, H), carries the autograd graph
    grad_out: Tensor     # static incoming-gradient buffer, same shape as out


class _TrainGraphFn(torch.autograd.Function):
    """Autograd glue: forward replays the fwd graph, backward replays the bwd graph
    (which accumulates into `param.grad`). Params are threaded through as inputs only
    so the output requires grad and our backward runs; we return `None` for their
    grads (the captured graph already accumulated — returning grads would double-count)."""

    @staticmethod
    def forward(ctx, captured, b, s, input_ids, attention_mask, *params):
        captured.inputs.stage(input_ids, attention_mask)
        captured.fwd_graph.replay()
        ctx.captured = captured
        ctx.num_params = len(params)
        # Clone the static output: `captured.out` is overwritten on every replay, so
        # handing it back raw would alias across calls (a second forward before this
        # one's backward corrupts it). The clone gives each call its own output and a
        # history-free tensor to hang our grad_fn on. It's identity, so backward is
        # unaffected.
        return captured.out.clone()

    @staticmethod
    def backward(ctx, grad_output):
        captured = ctx.captured
        if grad_output is None:  # output unused downstream
            return (None, None, None, None, None) + (None,) * ctx.num_params
        captured.grad_out.copy_(grad_output)
        captured.bwd_graph.replay()
        return (None, None, None, None, None) + (None,) * ctx.num_params


class _TrainGraphRunner:
    """Holds one captured fwd+bwd graph pair per exact `(batch, seq_len)`."""

    def __init__(
        self,
        model: nn.Module,
        params: ModernBertParams,
        config: TrainGraphConfig,
        backend: str = "sdpa",
    ):
        self._model = model
        self._params = params
        self._config = config
        self._backend = backend
        self._weights = [p for p in model.parameters() if p.requires_grad]
        self._cache: "OrderedDict[tuple[int, int], _Captured]" = OrderedDict()
        self._device = next(model.parameters()).device

    def __call__(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        b, s = input_ids.shape

        # Graph only short sequences (queries); longer calls (documents) run eager.
        if s > self._config.max_seq:
            return self._eager(input_ids, attention_mask)

        key = (b, s)  # exact shape — the bf16 backward is padding-sensitive

        captured = self._cache.get(key)
        if captured is None:
            if b * s > self._config.max_tokens:
                return self._eager(input_ids, attention_mask)
            captured = self._capture(b, s)
            self._insert(key, captured)
        else:
            self._cache.move_to_end(key)

        return _TrainGraphFn.apply(
            captured, b, s, input_ids, attention_mask, *self._weights
        )

    def _eager(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        return forward.fused_forward(
            self._model, self._params, input_ids, attention_mask, backend=self._backend
        )

    def _insert(self, key: tuple[int, int], captured: _Captured) -> None:
        self._cache[key] = captured
        while len(self._cache) > self._config.max_graphs:
            old_key, old = self._cache.popitem(last=False)
            del old

    def _fwd(self, static: _StaticInputs) -> Tensor:
        """The captured forward: the whole fused forward with grad enabled, capture-safe."""
        p = forward.prologue(
            self._model, self._params, static.input_ids, static.attention_mask,
            dense_mask=True, capture_safe=True,
        )
        return forward.core(
            self._model, self._params,
            p.x, p.cos_global, p.sin_global, p.cos_local, p.sin_local,
            p.full_mask, p.sliding_mask,
            backend=self._backend,
        )

    def _capture(self, b: int, s: int) -> _Captured:
        # Autograd runs backward on per-device worker threads by default, which issue
        # their kernels on the *default* stream, not the capture stream — illegal under
        # capture. Single-threaded backward runs the whole pass on the capturing thread
        # so every kernel is recorded.
        # The mismatch warning is a false alarm here (we differentiate with
        # `autograd.grad` and accumulate explicitly, never running AccumulateGrad), so
        # silence it around capture. The toggle is torch 2.11+; on the pinned 2.8 neither
        # the warning nor the toggle exist, so skip it.
        warn_mismatch = getattr(
            torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch", None
        )
        if warn_mismatch is not None:
            warn_mismatch(False)
        try:
            with torch.autograd.set_multithreading_enabled(False):
                return self._capture_inner(b, s)
        finally:
            if warn_mismatch is not None:
                warn_mismatch(True)

    def _capture_inner(self, b: int, s: int) -> _Captured:
        static = _StaticInputs(
            input_ids=torch.zeros((b, s), dtype=torch.long, device=self._device),
            attention_mask=torch.ones((b, s), dtype=torch.long, device=self._device),
        )

        # Every param.grad must be a live, stable-address buffer before capture: the
        # backward graph records an in-place add into it (hence the set_to_none invariant).
        for p in self._weights:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        with self._autocast_ctx():
            return self._capture_under_autocast(static, b, s)

    def _autocast_ctx(self):
        """Capture under the ambient autocast but with `cache_enabled=False`.

        With the cache on (default), autocast stashes ephemeral bf16 weight copies and
        frees them at context exit — a graph would replay against freed memory. With it
        off, the bf16 casts are recomputed inline and recorded into the graph, reading
        the persistent fp32 master each replay. Off (plain bf16 weights) this is a no-op.

        The autocast state is baked in at capture; replaying under a different regime
        would be stale (fine in practice — autocast config is fixed across a run)."""
        if not torch.is_autocast_enabled("cuda"):
            return contextlib.nullcontext()
        return torch.autocast(
            "cuda", dtype=torch.get_autocast_dtype("cuda"), cache_enabled=False
        )

    def _capture_under_autocast(self, static: _StaticInputs, b: int, s: int) -> _Captured:
        # Warmup on a side stream: build the autograd graph, allocate activations and
        # grad buffers, let cuBLAS/cuDNN pick plans — all before capture, so capture
        # records steady-state ops only. Backward goes through `autograd.grad` (not
        # `.backward()`, which synchronizes against the default stream — illegal under
        # capture). The `synchronize()` bookends drain default-stream work first.
        torch.cuda.synchronize()
        with torch.cuda.stream(torch.cuda.Stream()):
            for _ in range(self._config.warmup):
                out = self._fwd(static)
                grad_out = torch.ones_like(out)
                torch.autograd.grad(
                    out, self._weights, grad_outputs=grad_out, allow_unused=True
                )
        torch.cuda.synchronize()

        # Capture forward. `out` keeps the autograd graph (and saved activations) alive
        # for backward capture and every replay; the backward shares this bucket's pool.
        pool = torch.cuda.graph_pool_handle()
        fwd_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(fwd_graph, pool=pool):
            out = self._fwd(static)
        grad_out = torch.empty_like(out)

        # Capture backward: `autograd.grad` computes each gradient into static buffers
        # (private to this bucket, no cross-chunk clobber), then `add_` accumulates into
        # the shared persistent `param.grad`. On replay this re-accumulates — the
        # cross-chunk sum GradCache wants. `retain_graph` keeps the saved activations.
        # `.grad` is not zeroed here: warmup used `autograd.grad` (never writes `.grad`)
        # so the primed zeros are intact, and a pending default-stream `zero_()` would
        # make the capture stream depend on it ("legacy stream" error).
        bwd_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(bwd_graph, pool=pool):
            grads = torch.autograd.grad(
                out, self._weights, grad_outputs=grad_out,
                retain_graph=True, allow_unused=True,
            )
            for p, g in zip(self._weights, grads):
                if g is not None:
                    p.grad.add_(g)

        # Defensive re-zero so the first real step starts from clean grads.
        self._zero_grads()
        return _Captured(fwd_graph, bwd_graph, static, out, grad_out)

    def _zero_grads(self) -> None:
        for p in self._weights:
            if p.grad is not None:
                p.grad.zero_()


def build_train_runner(
    model: nn.Module,
    params: ModernBertParams,
    config: TrainGraphConfig,
    backend: str = "sdpa",
) -> _TrainGraphRunner:
    return _TrainGraphRunner(model, params, config, backend=backend)


def set_train_cuda_graph(
    model: object, enabled: bool, *, config: TrainGraphConfig | None = None
) -> None:
    """Turn the training-graph layer on or off after `prepare()`. See the module
    docstring for the two caller invariants (`set_to_none=False`, exact `(B, S)`)."""
    from flash_modernbert.locate import find_encoder
    from flash_modernbert.state import get_state

    state = get_state(model)
    if enabled and state.train_graph_runner is None:
        state.train_graph_runner = build_train_runner(
            find_encoder(model), state.params, config or TrainGraphConfig(),
            backend=state.attention_backend,
        )
    state.train_graph_enabled = enabled
