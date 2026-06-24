"""CUDA-graph layer for **training** — capturing the encoder fwd+bwd.

The inference runner (`graph.py`) captures a *no-grad* forward; it cannot drive a
backward. Training needs a different runner: one that captures the encoder
forward **and** its backward, and replays both inside the autograd machinery so a
training step differentiates through it transparently.

Why short-S training needs this (roadmap B0, measured): the eager fused tail is
host-launch-bound at short S — hundreds of tiny CuteDSL launches per forward,
*doubled* by the backward — so at the smallest query shape (B16/S32) it regresses
to 0.83× vs stock. Collapsing fwd+bwd into two graph replays recovers the launch
floor, exactly as the inference runner does for short-S encode.

Design (the PyTorch manual graph-training recipe, **not**
`make_graphed_callables`). For each exact `(batch, seq_len)` shape:

- a **forward graph** replayed by `_TrainGraphFn.forward`, from static
  `input_ids`/`attention_mask` buffers, producing a static `out` activation;
- a **backward graph** replayed by `_TrainGraphFn.backward`, which copies the
  incoming `grad_output` into a static buffer and replays
  `torch.autograd.backward(out, grad, inputs=params)` — so grads accumulate
  straight into each persistent `param.grad` via `AccumulateGrad`, *inside* the
  captured graph.

That accumulation-inside-the-graph is the whole point. `make_graphed_callables`
keeps one static `grad_inputs` buffer per callable; in the GradCache loop the same
model is called once per chunk, many chunks per step, and that single buffer is
clobbered before the matching backward runs → NaN/wrong grads. Here there is no
shared intermediate: every chunk replay just `+=`'s its contribution into the
static `.grad`, which is precisely the cross-chunk gradient sum GradCache wants.

Two invariants this imposes on the caller (enforced at the integration layer):

- **`.grad` buffers must stay live across replays.** The backward graph records an
  add into the *address* of each `param.grad` captured at capture time; freeing it
  (`zero_grad(set_to_none=True)`) and replaying would write to freed memory. Train
  with `set_to_none=False` so the optimizer zeroes `.grad` in place between steps.
- **fwd→bwd must be paired per chunk** before the next chunk's forward. The
  forward graph's static activations live until the backward graph reads them; a
  second forward in between would overwrite them. GradCache's pass-3 already runs
  per-chunk forward-then-backward, so this holds.

Like the inference runner this is bf16-weights-clean first; capturing under the
fp32-master/autocast regime is handled separately (roadmap B2).
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
    """Bucketing policy for the training-graph runner.

    One captured fwd+bwd graph pair per **exact** `(batch, seq_len)`. Unlike the
    inference runner, the sequence length is **not** rounded into padded buckets:
    measured on the 5090 (sm_120), padding the sequence and masking the tail
    leaves the *forward* bit-identical at the real positions (cos 1.0, what the
    inference runner relies on) but shifts the **bf16 backward** drastically —
    padded-vs-unpadded grads come out near-uncorrelated (cos≈0.09, relL2≈1.1),
    even though fp32 grads are identical. So a padded training graph would *not*
    train like the eager fused tail. Keying on exact `(B, S)` keeps the graph
    grads bit-exact to eager.

    **`max_seq` is the policy that makes this a queries-only feature.** Training
    graphs only win at short S; by S≈300 the eager fused tail already wins and the
    graph loses on *both* latency (dense mask precludes the flash fast path) and
    memory. So the runner graphs a call only when `s <= max_seq` and runs everything
    longer eager. In ColBERT this cleanly separates queries from documents at *zero*
    runtime cost: PyLate encodes each text group in its own forward call, so the
    runner sees a homogeneous `(B, S)` per call — a `(B, 32)` query batch graphs, a
    `(B, 300)` doc batch falls straight through to eager. Default 64 is a safe,
    device-independent floor (ColBERT `query_length` is ≤64); the measured 5090
    break-even is ~128, so raise it if you measure a higher crossover on another arch.

    For the regime where training graphs matter — short-S queries — PyLate pads
    every query to a fixed length, so a query stream is a single `(B, S)` bucket and
    exact keying costs nothing. A shape larger than `max_tokens` is never captured.

    **Memory cost scales with `max_graphs`.** Every captured bucket pins its *own*
    full activation pool for the runner's lifetime (the backward graph retains the
    forward's saved tensors), so N live graphs hold N activation sets at once —
    unlike the eager path, which reuses one chunk's peak. With `max_seq` keeping
    docs out of the cache, only the (small) query pool is pinned, so the graphed
    footprint stays ~parity with eager. (Capturing docs too — large `max_seq` —
    pins the doc pool as well and pushes reserved to ~1.2× eager; that "graphs halve
    VRAM" reading was an artifact of `max_memory_allocated`, which does not count a
    graph's private pool — always measure with `max_memory_reserved`.)
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
    """Autograd glue: forward replays the fwd graph, backward replays the bwd
    graph (which accumulates into `param.grad`). The params are threaded through
    as inputs purely so the output `requires_grad` and our backward is invoked;
    we return `None` for their grads because the captured graph already did the
    accumulation (returning a grad too would double-count)."""

    @staticmethod
    def forward(ctx, captured, b, s, input_ids, attention_mask, *params):
        captured.inputs.stage(input_ids, attention_mask)
        captured.fwd_graph.replay()
        ctx.captured = captured
        ctx.num_params = len(params)
        # Clone the static output: `captured.out` is a fixed buffer the fwd graph
        # overwrites on every replay, so handing it back raw aliases across calls —
        # a second forward of this bucket (before this one's backward) would corrupt
        # it. The clone (a plain copy here, since Function.forward runs grad-off)
        # gives each call its own output and hands autograd a history-free tensor to
        # hang our grad_fn on, instead of one still carrying the capture-time graph.
        # Backward is unaffected: the clone is identity, so grad_output still arrives
        # as the gradient w.r.t. `captured.out`.
        return captured.out.clone()

    @staticmethod
    def backward(ctx, grad_output):
        captured = ctx.captured
        if grad_output is None:  # output unused downstream — nothing to backprop
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
        param = next(model.parameters())
        self._device = param.device
        self._dtype = param.dtype

    def __call__(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        b, s = input_ids.shape

        # Graph only short sequences (queries). Past max_seq the graph loses to the
        # eager fused tail on both latency (dense mask precludes flash) and memory
        # (a captured doc pool stays pinned), so longer calls — documents — run
        # eager and are never captured. Each PyLate forward call is one homogeneous
        # (B, S), so this is the whole queries-vs-docs split, decided per call.
        if s > self._config.max_seq:
            return self._eager(input_ids, attention_mask)

        key = (b, s)  # exact shape — no padding (the bf16 backward is padding-sensitive)

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
        """The captured forward: the whole fused forward (prologue + core) with
        grad enabled, run capture-safe from the static id/mask buffers."""
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
        # The autograd engine runs backward on per-device worker threads by
        # default. Those threads issue the CuteDSL launches (which resolve the
        # stream via `torch.cuda.current_stream()`) and AccumulateGrad on the
        # *default* stream, not the capture stream — illegal under capture
        # ("legacy stream depend on a capturing blocking stream"). Forcing
        # single-threaded backward runs the whole pass on the capturing thread,
        # so every kernel lands on the capture stream and is recorded.
        # During capture the param AccumulateGrad nodes (made on a warmup-side
        # stream) don't match the capture stream; autograd warns it *could* break
        # capture. It doesn't here — our path differentiates with `autograd.grad`
        # and accumulates explicitly, never running those AccumulateGrad nodes — so
        # silence the false alarm just around capture, restoring the user's setting.
        warn_mismatch = torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch
        warn_mismatch(False)
        try:
            with torch.autograd.set_multithreading_enabled(False):
                return self._capture_inner(b, s)
        finally:
            warn_mismatch(True)

    def _capture_inner(self, b: int, s: int) -> _Captured:
        static = _StaticInputs(
            input_ids=torch.zeros((b, s), dtype=torch.long, device=self._device),
            attention_mask=torch.ones((b, s), dtype=torch.long, device=self._device),
        )

        # Every param.grad must be a live, stable-address buffer before capture:
        # the backward graph records an in-place add into it. (`set_to_none` would
        # break this, hence the training-mode invariant.)
        for p in self._weights:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        with self._autocast_ctx():
            return self._capture_under_autocast(static, b, s)

    def _autocast_ctx(self):
        """Capture under the ambient autocast but with `cache_enabled=False`.

        The PyLate recipe trains fp32 master weights under bf16 autocast. With the
        cache *on* (its default), autocast stashes ephemeral bf16 weight copies
        and frees them at context exit — a captured graph would replay against that
        freed memory (the exact reason the inference path gates graphs off under
        autocast). With the cache *off*, the bf16 weight casts are recomputed
        inline and so are *recorded into the graph*, reading the persistent fp32
        master each replay. When autocast is off (plain bf16 weights), this is a
        no-op and the kernels run directly.

        The autocast state is baked in at capture: a bucket captured under autocast
        records the casts, one captured without does not. Replaying under a
        *different* autocast regime than capture would be stale — fine in practice
        (autocast config is fixed across a run), but an assumption, not enforced."""
        if not torch.is_autocast_enabled("cuda"):
            return contextlib.nullcontext()
        return torch.autocast(
            "cuda", dtype=torch.get_autocast_dtype("cuda"), cache_enabled=False
        )

    def _capture_under_autocast(self, static: _StaticInputs, b: int, s: int) -> _Captured:
        # Warmup on a side stream: build the autograd graph, allocate every
        # activation and every grad buffer, let cuBLAS/cuDNN pick plans — all
        # before capture so capture records steady-state ops only. Backward goes
        # through `torch.autograd.grad` (not `.backward()`): the full engine
        # synchronizes against the legacy default stream, which is illegal under
        # capture; `autograd.grad` is the capture-safe path (as in
        # `make_graphed_callables`). The `synchronize()` bookends match its recipe
        # — they drain pending default-stream work so capture sees a clean slate.
        torch.cuda.synchronize()
        with torch.cuda.stream(torch.cuda.Stream()):
            for _ in range(self._config.warmup):
                out = self._fwd(static)
                grad_out = torch.ones_like(out)
                torch.autograd.grad(
                    out, self._weights, grad_outputs=grad_out, allow_unused=True
                )
        torch.cuda.synchronize()

        # Capture forward. `out` keeps the autograd graph (and its saved static
        # activations) alive; backward capture and every replay reference it. The
        # backward shares this bucket's pool so it sees the forward's activations.
        pool = torch.cuda.graph_pool_handle()
        fwd_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(fwd_graph, pool=pool):
            out = self._fwd(static)
        grad_out = torch.empty_like(out)

        # Capture backward: `autograd.grad` recomputes each param's gradient into
        # static buffers (private to this bucket's graph, so no cross-chunk
        # clobber), then an in-place `add_` accumulates them into the persistent,
        # shared `param.grad`. On replay this re-accumulates — exactly the
        # cross-chunk gradient sum GradCache wants. `retain_graph` keeps the saved
        # static activations alive for every future replay.
        #
        # `.grad` is *not* zeroed here: warmup goes through `autograd.grad`, which
        # never writes `.grad`, so the primed zeros are still intact — and a
        # pending in-place `zero_()` on the default stream at this point would make
        # the capture stream depend on it ("legacy stream" error).
        bwd_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(bwd_graph, pool=pool):
            grads = torch.autograd.grad(
                out, self._weights, grad_outputs=grad_out,
                retain_graph=True, allow_unused=True,
            )
            for p, g in zip(self._weights, grads):
                if g is not None:
                    p.grad.add_(g)

        # Defensive re-zero. Graph *capture* records the `add_` without executing
        # it, and warmup used `autograd.grad` (which never touches `.grad`), so the
        # primed zeros should already be intact — this just guarantees the first
        # real step starts from clean grads regardless.
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
    """Turn the training-graph layer on or off after `prepare()`.

    Unlike the inference runner, this captures the encoder fwd+bwd and replays it
    inside autograd, so it engages on the grad-enabled training forward (incl.
    under bf16 autocast). Two caller invariants (see TrainGraphConfig and the
    module docstring): train with `optimizer.zero_grad(set_to_none=False)` so the
    `.grad` buffers the backward graph writes into stay live, and feed exact
    `(B, S)` shapes (no sequence padding — the bf16 backward is padding-sensitive).
    """
    from flash_modernbert.state import encoder_of, get_state

    state = get_state(model)
    if enabled and state.train_graph_runner is None:
        state.train_graph_runner = build_train_runner(
            encoder_of(model), state.params, config or TrainGraphConfig(),
            backend=state.attention_backend,
        )
    state.train_graph_enabled = enabled
