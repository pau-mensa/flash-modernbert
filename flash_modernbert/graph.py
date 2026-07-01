"""CUDA-graph layer for inference — a separable speedup, off by default.

Targets the short-sequence regime where the eager fused tail is host-launch-bound;
capturing the forward into a graph collapses the launch floor to a single replay.
One graph per shape, so the runner buckets by `(batch, seq_bucket)`; out-of-bucket
or oversized shapes fall back to the (numerically identical) eager forward.

The capture boundary is the *whole* fused forward (prologue + core), captured from
the static `input_ids`/`attention_mask` buffers, so the only per-replay host work is
copying those two `(B, S)` int tensors. Two correctness requirements:

- The CuteDSL launches inside the captured region must take the capture stream —
  `_compile_cache.current_cute_stream()` threads it into every launch. Without it,
  default-stream launches are silently missed and replay produces stale output.
- A graph replays against fixed addresses, so it is inference-only: never under
  autograd (building a graph) or autocast (which frees its ephemeral bf16 weight
  copies at context exit — the graph would replay against freed memory). The patched
  forward enforces this, routing through the runner only when both are off.
"""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from flash_modernbert import forward, ops
from flash_modernbert.config import ModernBertParams
from flash_modernbert.locate import find_encoder
from flash_modernbert.state import get_state

GRAPH_ENV_VAR = "FLASH_MODERNBERT_GRAPH"


def graphs_globally_disabled() -> bool:
    """The `FLASH_MODERNBERT_GRAPH=0` kill switch."""
    return os.environ.get(GRAPH_ENV_VAR, "1") == "0"


@dataclass(frozen=True)
class GraphConfig:
    """Bucketing policy for the graph runner: one graph per `(batch, seq_bucket)`,
    `seq_bucket = ceil(seq_len / pad_to) * pad_to` (the padded tail is masked out).

    - `pad_to` buckets the sequence dimension.
    - `max_batch` buckets the batch dimension: left `None`, each batch size captures
      its own graph; set, every `b <= max_batch` replays one padded `max_batch`-row
      graph (the first `b` rows sliced out) and `b > max_batch` falls back to eager —
      the strongest cache bound for variable-batch workloads (e.g. partial last batch).
    - `seq_buckets` (with `max_batch`) pre-captures those buckets at `prepare()` time
      instead of lazily.
    - `max_graphs` bounds the cache by LRU eviction; a shape over `max_tokens` is
      never captured.
    - `max_seq` (default None = no cutoff) runs `s > max_seq` eager. Graphs win at
      short S but lose at long S on both latency (the static dense mask precludes
      SDPA's flash path) and memory — measured with `max_memory_reserved`, not
      `allocated`, which hides a graph's private pool. So the cutoff keeps graphs on
      short-S query *serving*; left None so long-S indexing configs keep graphing.
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
    """The graph's fixed input buffers — only the token ids and mask vary per replay."""

    input_ids: Tensor
    attention_mask: Tensor

    def stage(self, input_ids: Tensor, attention_mask: Tensor | None) -> None:
        """Write a real `(b, s)` batch into the `(bb, sb)` static buffers: zero them
        (so the padded tail stays empty), fill the live region; rows beyond `b` are
        marked valid (they get sliced off)."""
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
        self._device = next(model.parameters()).device

    def __call__(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        b, s = input_ids.shape
        max_seq = self._config.max_seq
        if max_seq is not None and s > max_seq:
            return self._eager(input_ids, attention_mask)  # long S: eager is faster, no mem win
        max_batch = self._config.max_batch
        if max_batch is not None and b > max_batch:
            return self._eager(input_ids, attention_mask)
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
            self._cache.move_to_end(key)  # LRU

        captured.inputs.stage(input_ids, attention_mask)
        captured.graph.replay()
        return captured.out[:b, :s].clone()

    def _eager(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        return forward.fused_forward(
            self._model, self._params, input_ids, attention_mask, backend=self._backend
        )

    def _insert(self, key: tuple[int, int], captured: "_Captured") -> None:
        self._cache[key] = captured
        while len(self._cache) > self._config.max_graphs:
            old_key, old = self._cache.popitem(last=False)
            del old  # frees the CUDAGraph + static buffers

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
        """The captured region: the whole fused forward, run capture-safe."""
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


# ---------------------------------------------------------------------------
# Packed graph runner — captures packed_forward at a fixed token budget
# ---------------------------------------------------------------------------


@dataclass
class _PackedStaticInputs:
    """Fixed-address buffers for packed graph replay."""
    input_ids: Tensor
    cu_seqlens: Tensor
    position_ids: Tensor
    max_seqlen: int

    def stage(
        self, packed_ids: Tensor, cu_seqlens: Tensor, position_ids: Tensor
    ) -> int:
        m_real = packed_ids.shape[0]
        b_real = cu_seqlens.shape[0] - 1
        self.input_ids[:m_real].copy_(packed_ids)
        self.input_ids[m_real:].zero_()
        self.position_ids[:m_real].copy_(position_ids)
        self.position_ids[m_real:].zero_()
        self.cu_seqlens[:b_real + 1].copy_(cu_seqlens)
        if b_real + 1 < self.cu_seqlens.shape[0]:
            self.cu_seqlens[b_real + 1 :].fill_(m_real)
        return m_real


@dataclass
class _PackedCaptured:
    graph: "torch.cuda.CUDAGraph"
    inputs: _PackedStaticInputs
    out: Tensor


class _PackedGraphRunner:
    """Graph runner for ``packed_forward``: one graph per M-bucket.

    The packed tensor shape is ``[1, M, H]`` (batch dim is always 1; sequences
    are encoded in ``cu_seqlens``).  The graph captures at a fixed ``M_bucket``
    with ``max_batch`` sequences and ``max_seq`` RoPE range.  Padding tokens
    beyond the real M are processed (wasted work) but their output is discarded.
    """

    def __init__(
        self,
        model: nn.Module,
        params: ModernBertParams,
        config: GraphConfig,
    ):
        self._model = model
        self._params = params
        self._config = config
        self._device = next(model.parameters()).device
        self._cache: OrderedDict[int, _PackedCaptured] = OrderedDict()

    def __call__(
        self,
        packed_ids: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        position_ids: Tensor,
    ) -> Tensor:
        m = packed_ids.shape[0]
        max_seq = self._config.max_seq
        if max_seq is not None and max_seqlen > max_seq:
            return self._eager(packed_ids, cu_seqlens, max_seqlen, position_ids)
        if m > self._config.max_tokens:
            return self._eager(packed_ids, cu_seqlens, max_seqlen, position_ids)

        mb = _round_up(m, self._config.pad_to)
        captured = self._cache.get(mb)
        if captured is not None:
            self._cache.move_to_end(mb)
        else:
            captured = self._capture(mb)
            while len(self._cache) >= self._config.max_graphs:
                self._cache.popitem(last=False)
            self._cache[mb] = captured

        m_real = captured.inputs.stage(packed_ids, cu_seqlens, position_ids)
        captured.graph.replay()
        return captured.out[:m_real].clone()

    def _eager(self, packed_ids, cu_seqlens, max_seqlen, position_ids):
        return forward.packed_forward(
            self._model,
            self._params,
            packed_ids,
            cu_seqlens,
            max_seqlen,
            position_ids,
        )

    def _capture(self, mb: int) -> _PackedCaptured:
        max_seq = self._config.max_seq
        bb = self._config.max_batch
        if max_seq is None or bb is None:
            raise ValueError(
                "packed graph runner requires both max_seq and max_batch"
            )
        static = _PackedStaticInputs(
            input_ids=torch.zeros(mb, dtype=torch.long, device=self._device),
            cu_seqlens=torch.zeros(
                bb + 1, dtype=torch.int32, device=self._device
            ),
            position_ids=torch.zeros(
                mb, dtype=torch.long, device=self._device
            ),
            max_seqlen=max_seq,
        )
        per_seq = max(1, mb // bb)
        for i in range(bb):
            end = min((i + 1) * per_seq, mb)
            static.cu_seqlens[i + 1] = end
            start = int(static.cu_seqlens[i])
            static.position_ids[start:end] = torch.arange(
                end - start, device=self._device
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
        return _PackedCaptured(graph=graph, inputs=static, out=out)

    def _run_forward(self, static: _PackedStaticInputs) -> Tensor:
        model, params = self._model, self._params
        emb = model.embeddings.tok_embeddings(static.input_ids)
        x = ops.fused_layer_norm(emb, model.embeddings.norm.weight,
                                 model.embeddings.norm.eps)
        dtype, device = x.dtype, x.device
        hd = params.head_dim
        cos_g, sin_g = forward._rope_tables(
            static.max_seqlen, hd, params.global_rope_theta, device, dtype
        )
        if params.local_rope_theta == params.global_rope_theta:
            cos_l, sin_l = cos_g, sin_g
        else:
            cos_l, sin_l = forward._rope_tables(
                static.max_seqlen, hd, params.local_rope_theta, device, dtype
            )

        def gather(t: Tensor) -> Tensor:
            return t[0].index_select(0, static.position_ids)

        out = forward.core(
            model, params,
            x.unsqueeze(0),
            gather(cos_g), gather(sin_g),
            gather(cos_l), gather(sin_l),
            None, None,
            backend="flash",
            cu_seqlens=static.cu_seqlens,
            max_seqlen=static.max_seqlen,
        )
        return out.squeeze(0)


def build_packed_runner(
    model: nn.Module, params: ModernBertParams, config: GraphConfig
) -> _PackedGraphRunner | None:
    if config.max_batch is None or config.max_seq is None:
        return None
    return _PackedGraphRunner(model, params, config)


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
            find_encoder(model), state.params, config or GraphConfig(),
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
