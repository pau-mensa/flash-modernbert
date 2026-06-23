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
    attention_backend: str = "sdpa",
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

    `attention_backend` selects the attention kernel: `"sdpa"` (default,
    dependency-free, dense-mask) or `"flash"` (FlashAttention with sliding-window
    pruning on local layers — needs the flash-attn package and targets unpadded
    inference). `validate` runs the hard gate during prepare.
    """
    if attention_backend not in ("sdpa", "flash"):
        raise FlashModernBertError(
            f"attention_backend must be 'sdpa' or 'flash', got {attention_backend!r}"
        )
    encoder = find_encoder(target)

    existing = getattr(encoder, ATTR, None)
    if existing is not None:  # idempotent — only (re)configure graphs if asked
        if cuda_graph and existing.graph_runner is None:
            _enable_graphs(encoder, existing, cuda_graph)
        if train_cuda_graph and existing.train_graph_runner is None:
            _enable_train_graphs(encoder, existing, train_cuda_graph)
        return target

    if validate:
        _validate(encoder)
    else:
        _require_cuda(encoder)

    params = ModernBertParams.from_hf_config(encoder.config)
    state = PatchState(
        params=params, original_forward=encoder.forward,
        attention_backend=attention_backend,
    )
    if cuda_graph:
        _enable_graphs(encoder, state, cuda_graph)
    if train_cuda_graph:
        _enable_train_graphs(encoder, state, train_cuda_graph)

    setattr(encoder, ATTR, state)
    encoder.forward = _make_forward(encoder, state)
    return target


def _enable_graphs(encoder: nn.Module, state: PatchState, cuda_graph: bool | GraphConfig) -> None:
    config = cuda_graph if isinstance(cuda_graph, GraphConfig) else GraphConfig()
    state.graph_runner = build_runner(
        encoder, state.params, config, backend=state.attention_backend
    )
    state.graph_enabled = True


def _enable_train_graphs(
    encoder: nn.Module, state: PatchState, train_cuda_graph: bool | TrainGraphConfig
) -> None:
    config = (
        train_cuda_graph if isinstance(train_cuda_graph, TrainGraphConfig)
        else TrainGraphConfig()
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
