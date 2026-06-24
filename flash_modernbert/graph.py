"""CUDA-graph layer — a separable speedup on top of the eager fused forward.

Off by default. Its job is the short-sequence regime where the eager fused tail
is host-launch-bound (hundreds of tiny CuteDSL launches per forward); capturing
the encoder core into a graph collapses that launch floor to a single replay.

One graph captures one shape, so the runner buckets: sequence length is rounded
up to `pad_to` and the padded region is masked out (the fused tail is leak-free
under padding). The capture boundary is the **whole** fused forward — prologue
(embeddings, RoPE tables, SDPA masks) *and* `core` (the layer loop + final norm)
— captured from the static `input_ids`/`attention_mask` buffers. So the only
per-replay host work is copying those two `(B, S)` int tensors in; everything
downstream is replayed. This matters most at long S, where the prologue's `S×S`
sliding mask would otherwise be rebuilt and copied every call. The prologue's one
host sync (the mask `isinf().any()` probe) is removed under `capture_safe=True`.
Out-of-bucket or oversized shapes fall back to the eager forward, which is
numerically identical, just un-graphed.

Correctness rests on the CuteDSL launches inside the captured region taking the
capture stream: `_kernels._compile_cache.current_cute_stream()` threads the
active stream into every launch, so capture records them. Without it,
default-stream launches are silently missed and replay produces stale output.

A captured graph replays against fixed memory addresses, so it is an
**inference** feature: it must not be used while autograd is building a graph or
while autocast is active. Autocast caches ephemeral bf16 weight copies and frees
them when its context exits, so a graph captured under autocast would replay
against freed memory. The patched forward enforces this — it only routes through
the runner when both autograd and autocast are off (plain bf16 inference).
"""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from flash_modernbert import forward
from flash_modernbert.config import ModernBertParams
from flash_modernbert.state import encoder_of, get_state

GRAPH_ENV_VAR = "FLASH_MODERNBERT_GRAPH"


def graphs_globally_disabled() -> bool:
    """The `FLASH_MODERNBERT_GRAPH=0` kill switch (mirrors PYLATE_DISABLE_LIK)."""
    return os.environ.get(GRAPH_ENV_VAR, "1") == "0"


@dataclass(frozen=True)
class GraphConfig:
    """Bucketing policy for the graph runner.

    The runner holds one captured graph per `(batch, seq_bucket)` key, where
    `seq_bucket = ceil(seq_len / pad_to) * pad_to`. Two knobs bound how many keys
    that produces under variable-length, variable-batch inference:

    - `pad_to` collapses the **sequence** dimension into buckets (the padded tail
      is masked out — the fused tail is leak-free under padding).
    - `max_batch` collapses the **batch** dimension. Left `None`, each distinct
      batch size captures its own graph (fine when batch size is fixed). Set to
      the encode batch size, every batch with `b <= max_batch` replays a single
      `max_batch`-row graph (batch dim padded, the first `b` rows sliced out), and
      `b > max_batch` falls back to eager. This is the "round B up to max_batch"
      policy: it collapses the batch dimension to one graph per sequence bucket —
      the strongest cache bound for variable-batch workloads (e.g. an indexing
      run whose last batch is partial) — at the cost of computing the few padded
      rows on a partial batch.

    Capture is lazy by default. Set `seq_buckets` (together with `max_batch`, which
    supplies the batch dim) to pre-capture those buckets at `prepare()` time, when
    deterministic warm-up latency matters more than flexibility.

    The cache is bounded at `max_graphs` by LRU eviction: once full, capturing a
    new bucket evicts the least-recently-replayed one (freeing its graph), so the
    hot working set stays graphed even under an unbounded stream of shapes. A
    single shape larger than `max_tokens` is never captured (always eager).

    `max_seq` (default None = no cutoff, graph any length) gates by sequence length:
    a call with `s > max_seq` runs eager and is never captured. Graphs win at short
    S (the launch-floor collapse) but at long S the eager fused tail is faster
    (a captured graph needs a static dense mask, which precludes SDPA's flash fast
    path) and — measured with `max_memory_reserved`, not `allocated` — graphs do not
    actually save memory there either (the private pool is real, just invisible to
    `allocated`). So for short-S query *serving* a cutoff routes long docs to the
    faster eager path at no cost. It is left None by default so explicit long-S
    *indexing* configs (`seq_buckets` + `max_batch`) keep graphing as before;
    `prepare(cuda_graph=True, cuda_graph_seq_cutoff=...)` sets it for the plain
    bool-enable path.
    """

    pad_to: int = 64
    max_batch: int | None = None
    seq_buckets: tuple[int, ...] | None = None
    max_graphs: int = 32
    max_tokens: int = 2 ** 20
    max_seq: int | None = None
    warmup: int = 3


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


@dataclass
class _Captured:
    graph: "torch.cuda.CUDAGraph"
    inputs: "_StaticInputs"
    out: Tensor


@dataclass
class _StaticInputs:
    """The graph's fixed input buffers. The capture boundary is the *whole*
    forward (prologue + core), so the only things that vary per replay are the
    token ids and the attention mask — everything downstream (embeddings, RoPE
    tables, SDPA masks) is recomputed inside the captured region. This keeps the
    per-replay copy to two `(B, S)` int tensors instead of the post-prologue
    activations and the `S×S` masks."""

    input_ids: Tensor
    attention_mask: Tensor

    def stage(self, input_ids: Tensor, attention_mask: Tensor | None) -> None:
        """Write a real `(b, s)` batch into the `(bb, sb)` static buffers: zero
        them (so a smaller shape leaves the padded tail masked/empty), then fill
        the live region; dummy rows beyond `b` are marked valid (sliced off)."""
        b, s = input_ids.shape
        bb = self.input_ids.shape[0]
        self.input_ids.zero_()
        self.input_ids[:b, :s] = input_ids
        self.attention_mask.zero_()
        if attention_mask is None:
            self.attention_mask[:b, :s] = 1
        else:
            self.attention_mask[:b, :s] = attention_mask.to(self.attention_mask.dtype)
        if bb > b:
            self.attention_mask[b:bb, :s] = 1


class _GraphRunner:
    """Holds one captured graph per `(batch, seq_bucket)` and replays it."""

    def __init__(
        self,
        model: nn.Module,
        params: ModernBertParams,
        config: GraphConfig,
        backend: str = "sdpa",
    ):
        self._model = model
        self._params = params
        self._config = config
        self._backend = backend
        self._cache: "OrderedDict[tuple[int, int], _Captured]" = OrderedDict()
        param = next(model.parameters())
        self._device = param.device
        self._dtype = param.dtype

    def __call__(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        b, s = input_ids.shape
        max_seq = self._config.max_seq
        if max_seq is not None and s > max_seq:
            return self._eager(input_ids, attention_mask)  # long S: eager fused is faster, no mem win
        max_batch = self._config.max_batch
        if max_batch is not None and b > max_batch:
            return self._eager(input_ids, attention_mask)  # batch too large for the bucket
        bb = b if max_batch is None else max_batch
        sb = _round_up(s, self._config.pad_to)
        key = (bb, sb)

        captured = self._cache.get(key)
        if captured is None:
            if bb * sb > self._config.max_tokens:
                return self._eager(input_ids, attention_mask)
            captured = self._capture(bb, sb)
            self._insert(key, captured)
        else:
            self._cache.move_to_end(key)  # LRU: mark as recently replayed

        captured.inputs.stage(input_ids, attention_mask)
        captured.graph.replay()
        return captured.out[:b, :s].clone()

    def _eager(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        return forward.fused_forward(
            self._model, self._params, input_ids, attention_mask, backend=self._backend
        )

    def _insert(self, key: tuple[int, int], captured: "_Captured") -> None:
        """Insert a freshly captured graph, evicting the least-recently-replayed
        one if the cache is at capacity (LRU), so it stays bounded at max_graphs."""
        self._cache[key] = captured
        while len(self._cache) > self._config.max_graphs:
            old_key, old = self._cache.popitem(last=False)
            del old  # drop the CUDAGraph + static buffers so their memory frees

    def _capture(self, b: int, sb: int) -> _Captured:
        static = _StaticInputs(
            input_ids=torch.zeros((b, sb), dtype=torch.long, device=self._device),
            attention_mask=torch.ones((b, sb), dtype=torch.long, device=self._device),
        )
        with torch.no_grad():
            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for _ in range(self._config.warmup):
                    self._run_forward(static)
            torch.cuda.current_stream().wait_stream(warm)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._run_forward(static)
        return _Captured(graph=graph, inputs=static, out=out)

    def _run_forward(self, static: _StaticInputs) -> Tensor:
        """The captured region: the whole fused forward (prologue + core), run
        capture-safe (no host sync in the prologue) from the static id/mask
        buffers."""
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

    def precapture(self, batch: int, seq_buckets) -> None:
        for s in seq_buckets:
            sb = _round_up(int(s), self._config.pad_to)
            key = (batch, sb)
            if key not in self._cache:
                self._insert(key, self._capture(batch, sb))


def build_runner(
    model: nn.Module,
    params: ModernBertParams,
    config: GraphConfig,
    backend: str = "sdpa",
) -> _GraphRunner:
    runner = _GraphRunner(model, params, config, backend=backend)
    if config.seq_buckets:
        if config.max_batch is None:
            warnings.warn(
                "flash-modernbert: GraphConfig.seq_buckets needs max_batch to know "
                "the batch dimension to pre-capture at; skipping pre-capture (graphs "
                "will still be captured lazily on first sight).",
                stacklevel=2,
            )
        else:
            runner.precapture(config.max_batch, config.seq_buckets)
    return runner


# ---------------------------------------------------------------------------
# Runtime toggles
# ---------------------------------------------------------------------------


def set_cuda_graph(model: object, enabled: bool, *, config: GraphConfig | None = None) -> None:
    """Turn the graph layer on or off after `prepare()`. Building a runner the
    first time it is enabled requires CUDA and is lazy per shape."""
    state = get_state(model)
    if enabled and state.graph_runner is None:
        state.graph_runner = build_runner(
            encoder_of(model), state.params, config or GraphConfig(),
            backend=state.attention_backend,
        )
    state.graph_enabled = enabled


class _NoCudaGraph:
    def __init__(self, model: object):
        self._state = get_state(model)
        self._previous = self._state.graph_enabled

    def __enter__(self):
        self._previous = self._state.graph_enabled
        self._state.graph_enabled = False
        return self

    def __exit__(self, *exc):
        self._state.graph_enabled = self._previous
        return False


def no_cuda_graph(model: object) -> _NoCudaGraph:
    """Context manager that bypasses captured graphs for a one-off odd shape."""
    return _NoCudaGraph(model)
