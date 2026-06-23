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
from flash_modernbert.locate import find_encoder
from flash_modernbert.state import ATTR, PatchState
from flash_modernbert.validate import validate as _validate


def prepare(
    target: object,
    *,
    cuda_graph: bool | GraphConfig = False,
    validate: bool = True,
) -> object:
    """Install the fused-tail forward onto `target` and return the same object.

    `target` may be a Hugging Face `ModernBertModel`, a SentenceTransformer or
    PyLate ColBERT wrapping one, or a task model exposing it; the encoder is
    located and patched in place.

    `cuda_graph` enables the bucketed CUDA-graph runner (off by default — the
    plain eager fused forward). `validate` runs the hard gate during prepare.
    """
    encoder = find_encoder(target)

    existing = getattr(encoder, ATTR, None)
    if existing is not None:  # idempotent — only (re)configure graphs if asked
        if cuda_graph and existing.graph_runner is None:
            _enable_graphs(encoder, existing, cuda_graph)
        return target

    if validate:
        _validate(encoder)
    else:
        _require_cuda(encoder)

    params = ModernBertParams.from_hf_config(encoder.config)
    state = PatchState(params=params, original_forward=encoder.forward)
    if cuda_graph:
        _enable_graphs(encoder, state, cuda_graph)

    setattr(encoder, ATTR, state)
    encoder.forward = _make_forward(encoder, state)
    return target


def _enable_graphs(encoder: nn.Module, state: PatchState, cuda_graph: bool | GraphConfig) -> None:
    config = cuda_graph if isinstance(cuda_graph, GraphConfig) else GraphConfig()
    state.graph_runner = build_runner(encoder, state.params, config)
    state.graph_enabled = True


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

        graph_requested = (
            state.graph_runner is not None
            and state.graph_enabled
            and not graphs_globally_disabled()
        )
        # A captured graph replays against fixed memory addresses, so it is only
        # safe when those addresses are stable across replays: no autograd graph
        # being built, and no autocast. Autocast caches ephemeral bf16 weight
        # copies and frees them when its context exits, so a graph captured under
        # autocast would replay against freed memory (illegal access). Both hold
        # for plain bf16 inference; neither holds inside a training step (grad +
        # autocast), where the eager fused tail runs instead.
        use_graph = (
            graph_requested
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled("cuda")
        )
        if graph_requested and not use_graph and not state.graph_skip_warned:
            state.graph_skip_warned = True
            warnings.warn(
                "flash-modernbert: CUDA graphs are enabled but skipped here because "
                "autocast or autograd is active (e.g. inside a training step). Graphs "
                "apply to plain bf16 inference; running the eager fused forward instead.",
                stacklevel=2,
            )
        if use_graph:
            hidden = state.graph_runner(input_ids, attention_mask)
        else:
            hidden = fused_forward(encoder, state.params, input_ids, attention_mask)

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
