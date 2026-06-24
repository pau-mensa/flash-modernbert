"""`prepare()` — install the fused forward onto a live model, in place.

Decision from the M3 design: patch, don't re-implement. The kernels already
consume HF's exact weight layout, so instead of building a parallel `nn.Module`
and a per-framework shim, `prepare()` swaps the encoder's bound `forward` for the
fused one and leaves everything else — `state_dict`, `save_pretrained`, `resize`,
`from_pretrained`, gradient checkpointing — as HF's own. One patch, and HF /
SentenceTransformers / PyLate all inherit the speedup with no adapter class.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput

from flash_modernbert import ops
from flash_modernbert.config import ModernBertParams
from flash_modernbert.errors import FlashModernBertError
from flash_modernbert.forward import fused_forward
from flash_modernbert.graph import GraphConfig, build_runner, graphs_globally_disabled
from flash_modernbert.train_graph import TrainGraphConfig, build_train_runner
from flash_modernbert.locate import find_encoder
from flash_modernbert.state import ATTR, PatchState
from flash_modernbert.validate import validate as _validate


def prepare(
    target: object,
    *,
    cuda_graph: bool | GraphConfig = False,
    train_cuda_graph: bool | TrainGraphConfig = False,
    cuda_graph_seq_cutoff: int = 64,
    attention_backend: str | None = None,
    validate: bool = True,
) -> object:
    """Install the fused-tail forward onto `target` and return the same object.

    `target` may be a Hugging Face `ModernBertModel`, a SentenceTransformer or
    PyLate ColBERT wrapping one, or a task model exposing it; the encoder is
    located and patched in place.

    `cuda_graph` enables the bucketed **inference** CUDA-graph runner (off by
    default — the plain eager fused forward), which engages on the no-grad,
    no-autocast forward. `train_cuda_graph` enables the **training** runner, which
    captures the encoder fwd+bwd and replays it inside autograd, engaging on the
    grad-enabled forward (incl. under bf16 autocast — the PyLate `bf16=True`
    recipe). It targets the short-S regime where the eager fused tail is
    host-launch-bound and regresses vs stock (e.g. B16/S32). Two caller invariants:
    train with `optimizer.zero_grad(set_to_none=False)` (the backward graph writes
    into persistent `.grad` buffers) and feed exact `(B, S)` shapes (the bf16
    backward is padding-sensitive, so the runner does not bucket the sequence).

    `cuda_graph_seq_cutoff` (default 64) bounds *which* calls get graphed when the
    runner is enabled via the bool shortcut (`cuda_graph=True` / `train_cuda_graph=
    True`): only sequences with `s <= cutoff` are captured; longer ones run eager.
    Graphs win only at short S (by S≈300 the eager fused tail wins on latency, and —
    measured on reserved memory, not allocated — graphs don't save memory either),
    so this keeps graphs on the **queries** and off the **documents**. Since PyLate
    encodes each text group in its own forward call, that split happens
    automatically per call at no runtime cost. 64 is a safe device-independent floor
    (ColBERT `query_length` ≤64); the 5090 break-even is ~128, so raise it if a
    longer crossover is measured on another arch. Passing an explicit `GraphConfig`
    / `TrainGraphConfig` overrides this via that config's own `max_seq` (which
    defaults to None for inference — so long-S **indexing** configs with
    `seq_buckets` keep graphing — and 64 for training).

    `attention_backend` selects the attention kernel:

    - **unset (`None`, the default)** — best available: the flash path when a kernel
      is importable (so flash is the out-of-the-box backend wherever it runs — varlen
      makes it correct on padded batches too), else the dependency-free `"sdpa"`.
      Resolves *silently*: an unset backend makes no demand, so a missing kernel is
      not a warning. The resolved value is recorded in the patch state.
    - `"sdpa"` — dependency-free, dense-mask; the safe universal choice. Wins or ties
      at the short-S query regime (flash's fixed cost loses on tiny tensors).
    - `"flash"` — FlashAttention with sliding-window pruning on the local layers
      (O(S·W) vs O(S²)). On a **padded** batch it unpads and uses the varlen kernel
      (confines attention within each sequence, skips pad-token compute); on an
      unpadded batch it runs dense flash. Needs a kernel (FA4-cute on sm_90/sm_100,
      compiled flash-attn on sm_120).
    - `"auto"` — sdpa below `ops.FLASH_MIN_SEQ` (short-S queries), flash at or above
      it (long-S docs), decided per call. Falls back to `"sdpa"` (with a warning,
      since it was asked for) if no flash kernel is present.

    Measured on padded document batches (5090, ModernBERT-base, bf16), flash beats
    sdpa from S≈256 up (1.16× → 2.7× at S4096), the win growing with both S and
    padding fraction (up to 3.25× at ~56% padding) — the varlen path prunes the
    local layers *and* skips pad-token compute. See `benchmarks/varlen_bench.py`.

    `validate` runs the hard gate during prepare.
    """
    if attention_backend is None:
        attention_backend = _default_backend()
    if attention_backend not in ("sdpa", "flash", "auto"):
        raise FlashModernBertError(
            "attention_backend must be 'sdpa', 'flash', 'auto', or None, got "
            f"{attention_backend!r}"
        )
    encoder = find_encoder(target)

    existing = getattr(encoder, ATTR, None)
    if existing is not None:  # idempotent — only (re)configure graphs if asked
        if cuda_graph and existing.graph_runner is None:
            _enable_graphs(encoder, existing, cuda_graph, cuda_graph_seq_cutoff)
        if train_cuda_graph and existing.train_graph_runner is None:
            _enable_train_graphs(encoder, existing, train_cuda_graph, cuda_graph_seq_cutoff)
        return target

    if validate:
        _validate(encoder)
    else:
        _require_cuda(encoder)

    attention_backend = _resolve_attention_backend(attention_backend)

    params = ModernBertParams.from_hf_config(encoder.config)
    state = PatchState(
        params=params, original_forward=encoder.forward,
        attention_backend=attention_backend,
    )
    if cuda_graph:
        _enable_graphs(encoder, state, cuda_graph, cuda_graph_seq_cutoff)
    if train_cuda_graph:
        _enable_train_graphs(encoder, state, train_cuda_graph, cuda_graph_seq_cutoff)

    setattr(encoder, ATTR, state)
    encoder.forward = _make_forward(encoder, state)
    return target


def _default_backend() -> str:
    """Resolve an unset `attention_backend`: `"auto"` when a flash kernel is
    importable (flash wherever it can run), else dep-free `"sdpa"`. Silent — an
    unset backend is not a request, so a missing kernel needs no warning."""
    try:
        ops._load_flash_attn()
        return "auto"
    except ImportError:
        return "sdpa"


def _resolve_attention_backend(backend: str) -> str:
    """Confirm a flash kernel is importable up front for `"flash"`/`"auto"`, so a
    captured graph never hits a mid-capture `ImportError`. `"flash"` raises if none
    is available (the user asked for it explicitly); `"auto"` downgrades to
    `"sdpa"` with a warning, keeping it dependency-optional. `"sdpa"` is a no-op."""
    if backend == "sdpa":
        return backend
    try:
        ops._load_flash_attn()
    except ImportError as exc:
        if backend == "flash":
            raise FlashModernBertError(str(exc)) from exc
        warnings.warn(
            f"attention_backend='auto' but no FlashAttention kernel is available "
            f"({exc}); falling back to 'sdpa'.",
            stacklevel=2,
        )
        return "sdpa"
    return backend


def _enable_graphs(
    encoder: nn.Module,
    state: PatchState,
    cuda_graph: bool | GraphConfig,
    seq_cutoff: int | None = None,
) -> None:
    # An explicit GraphConfig wins as-is (incl. its own max_seq, default None so
    # long-S indexing configs keep graphing). A bare `True` gets the default config
    # with the caller's seq cutoff applied — the safe short-S-serving default.
    config = (
        cuda_graph if isinstance(cuda_graph, GraphConfig)
        else GraphConfig(max_seq=seq_cutoff)
    )
    state.graph_runner = build_runner(
        encoder, state.params, config, backend=state.attention_backend
    )
    state.graph_enabled = True


def _enable_train_graphs(
    encoder: nn.Module,
    state: PatchState,
    train_cuda_graph: bool | TrainGraphConfig,
    seq_cutoff: int = 64,
) -> None:
    # An explicit config wins as-is; a bare `True` gets the default config with the
    # caller's seq cutoff threaded into max_seq (the queries-only gate).
    config = (
        train_cuda_graph if isinstance(train_cuda_graph, TrainGraphConfig)
        else TrainGraphConfig(max_seq=seq_cutoff)
    )
    state.train_graph_runner = build_train_runner(
        encoder, state.params, config, backend=state.attention_backend
    )
    state.train_graph_enabled = True


def unprepare(target: object) -> object:
    """Restore the original forward, reverting `prepare()`."""
    encoder = find_encoder(target)
    state = getattr(encoder, ATTR, None)
    if state is None:
        return target
    encoder.forward = state.original_forward
    delattr(encoder, ATTR)
    return target


def _make_forward(encoder: nn.Module, state: PatchState):
    def forward(
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        *,
        return_dict: bool = True,
        **kwargs,
    ):
        if input_ids is None:
            raise NotImplementedError(
                "flash-modernbert requires input_ids (inputs_embeds is unsupported)"
            )
        _reject_unsupported(kwargs)

        graphs_off = graphs_globally_disabled()
        # The **training** runner captures the encoder fwd+bwd and replays it
        # inside autograd, so — unlike the inference runner — it is exactly the
        # grad-enabled path it serves, and it captures autocast-safe (the bf16
        # weight casts are recorded inline, reading the persistent fp32 master).
        # So it engages whenever autograd is building a graph; it has no quarrel
        # with autocast. This is the relaxation of the old inference-only gate.
        use_train_graph = (
            state.train_graph_runner is not None
            and state.train_graph_enabled
            and not graphs_off
            and torch.is_grad_enabled()
        )
        # The **inference** runner replays against fixed memory addresses, so it is
        # only safe when those addresses are stable across replays: no autograd
        # graph being built, and no autocast (which caches ephemeral bf16 weight
        # copies and frees them at context exit — a captured graph would replay
        # against freed memory). Both hold only for plain bf16 inference.
        graph_requested = (
            state.graph_runner is not None
            and state.graph_enabled
            and not graphs_off
        )
        use_graph = (
            graph_requested
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled("cuda")
        )
        if (
            graph_requested and not use_graph and not use_train_graph
            and not state.graph_skip_warned
        ):
            state.graph_skip_warned = True
            warnings.warn(
                "flash-modernbert: inference CUDA graphs are enabled but skipped here "
                "because autocast or autograd is active (e.g. inside a training step). "
                "Inference graphs apply to plain bf16 inference; for training-step "
                "graphs pass train_cuda_graph=True. Running the eager fused forward.",
                stacklevel=2,
            )
        if use_train_graph:
            hidden = state.train_graph_runner(input_ids, attention_mask)
        elif use_graph:
            hidden = state.graph_runner(input_ids, attention_mask)
        else:
            hidden = fused_forward(
                encoder, state.params, input_ids, attention_mask,
                backend=state.attention_backend,
            )

        if not return_dict:
            return (hidden,)
        return BaseModelOutput(last_hidden_state=hidden)

    return forward


def _reject_unsupported(kwargs: dict) -> None:
    if kwargs.get("inputs_embeds") is not None:
        raise NotImplementedError("flash-modernbert does not support inputs_embeds")
    if kwargs.get("output_attentions"):
        raise NotImplementedError("flash-modernbert does not support output_attentions")
    if kwargs.get("output_hidden_states"):
        raise NotImplementedError("flash-modernbert does not support output_hidden_states")
    token_type_ids = kwargs.get("token_type_ids")
    if token_type_ids is not None and bool(token_type_ids.any()):
        raise NotImplementedError("flash-modernbert does not support non-zero token_type_ids")


def _require_cuda(encoder: nn.Module) -> None:
    try:
        device = next(encoder.parameters()).device
    except StopIteration as exc:
        raise FlashModernBertError("the model has no parameters") from exc
    if device.type != "cuda":
        raise FlashModernBertError(
            f"the fused path requires CUDA weights; the model is on {device!r}"
        )
