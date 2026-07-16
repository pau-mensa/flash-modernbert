"""`pack()` — install the fused forward onto a live model, in place.

Patch, don't re-implement: the kernels already consume HF's exact weight layout, so
`pack()` swaps the encoder's bound `forward` and leaves everything else
(`state_dict`, `save_pretrained`, ...) as HF's own. One patch, and HF /
SentenceTransformers / PyLate all inherit the speedup with no adapter class.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput

from packed_encoders import ops
from packed_encoders.config import ModernBertParams
from packed_encoders.errors import PackedEncodersError
from packed_encoders.forward import fused_forward
from packed_encoders.graph import GraphConfig, build_packed_runner, build_runner, graphs_globally_disabled
from packed_encoders.train_graph import TrainGraphConfig, build_train_runner
from packed_encoders.locate import find_encoder
from packed_encoders.state import ATTR, PatchState
from packed_encoders.validate import validate as _validate


def pack(
    target: object,
    *,
    cuda_graph: bool | GraphConfig = False,
    train_cuda_graph: bool | TrainGraphConfig = False,
    cuda_graph_seq_cutoff: int = 64,
    attention_backend: str | None = None,
    validate: bool = True,
) -> object:
    """Install the fused-tail forward onto `target` and return the same object.

    `target` may be a HF `ModernBertModel`, a SentenceTransformer / PyLate ColBERT
    wrapping one, or a task model exposing it; the encoder is located and patched.

    `cuda_graph` enables the bucketed **inference** graph runner (engages on the
    no-grad, no-autocast forward). `train_cuda_graph` enables the **training** runner,
    which captures fwd+bwd and replays inside autograd (incl. under bf16 autocast).
    Training has two caller invariants: `optimizer.zero_grad(set_to_none=False)` (the
    backward graph writes into persistent `.grad`) and exact `(B, S)` shapes (the bf16
    backward is padding-sensitive).

    Inference `auto`/`triton` graph capture applies to `packed_forward()` only; a
    rectangular graph cannot derive dynamic packed boundaries capture-safely. It needs
    an explicit `GraphConfig` with both `max_batch` and `max_seq`.

    `cuda_graph_seq_cutoff` (default 64) bounds which calls get graphed via the bool
    shortcut: only `s <= cutoff` is captured. Graphs win only at short S, so this keeps
    them on queries and off documents (PyLate encodes each group in its own call, so
    the split is automatic). An explicit `GraphConfig`/`TrainGraphConfig` overrides it
    via that config's own `max_seq`.

    `attention_backend` selects the attention kernel:

    - **`None` (default)** — `"auto"` when Flash or packed Triton is importable,
      else `"sdpa"`.
      Resolves silently (an unset backend makes no demand).
    - `"sdpa"` — dependency-free, dense-mask; the safe universal choice.
    - `"flash"` — FlashAttention with sliding-window pruning; on a padded batch it
      uses the varlen kernel. Needs a kernel (FA4-cute on sm_90/sm_100, compiled
      flash-attn on sm_120); raises if none is present.
    - `"triton"` — the packed short-attention kernel for no-grad bf16 CUDA
      ModernBERT-base workloads with `Smax<=128`; invariant misses prefer Flash and
      then SDPA.
    - `"auto"` — exact-card calibrated packed Triton/Flash dispatch for RTX 5090,
      A100, L40S, H200, and B200. Uncalibrated cards prefer Flash. SDPA is only the
      final no-kernel fallback.

    `validate` runs the hard gate during pack.
    """
    if attention_backend is None:
        attention_backend = _default_backend()
    if attention_backend not in ("sdpa", "flash", "triton", "auto"):
        raise PackedEncodersError(
            "attention_backend must be 'sdpa', 'flash', 'triton', 'auto', or None, got "
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
    """Use auto if either optimized attention implementation imports."""
    try:
        ops._load_flash_attn()
        return "auto"
    except ImportError:
        try:
            ops._load_packed_short_attention()
            return "auto"
        except ImportError:
            return "sdpa"


def _resolve_attention_backend(backend: str) -> str:
    """Validate explicit dependencies and collapse auto only when neither exists."""
    if backend == "sdpa":
        return backend
    flash_error = None
    triton_error = None
    try:
        ops._load_flash_attn()
    except ImportError as exc:
        flash_error = exc
    try:
        ops._load_packed_short_attention()
    except ImportError as exc:
        triton_error = exc

    if backend == "flash" and flash_error is not None:
        raise PackedEncodersError(str(flash_error)) from flash_error
    if backend == "triton" and triton_error is not None:
        raise PackedEncodersError(
            f"attention_backend='triton' is unavailable: {triton_error}"
        ) from triton_error
    if backend == "auto" and flash_error is not None and triton_error is not None:
        warnings.warn(
            "attention_backend='auto' but neither FlashAttention nor the packed "
            f"Triton kernel is available ({flash_error}; {triton_error}); falling "
            "back to 'sdpa'.",
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
    # Explicit config wins as-is; a bare `True` gets the default config with the
    # caller's seq cutoff applied.
    config = (
        cuda_graph if isinstance(cuda_graph, GraphConfig)
        else GraphConfig(max_seq=seq_cutoff)
    )
    graph_backend = state.attention_backend
    if graph_backend in ("auto", "triton"):
        # A rectangular graph cannot turn a dynamic padding mask into capture-safe
        # cu_seqlens. Auto/Triton graph only through the already-packed runner; using
        # dense Flash here would attend to padding, while SDPA would violate auto.
        graph_backend = None
    state.graph_runner = (
        build_runner(encoder, state.params, config, backend=graph_backend)
        if graph_backend is not None else None
    )
    state.packed_graph_runner = build_packed_runner(
        encoder, state.params, config, backend=state.attention_backend
    )
    state.graph_enabled = True


def _enable_train_graphs(
    encoder: nn.Module,
    state: PatchState,
    train_cuda_graph: bool | TrainGraphConfig,
    seq_cutoff: int = 64,
) -> None:
    # Explicit config wins as-is; a bare `True` gets the default config with the
    # caller's seq cutoff threaded into max_seq.
    config = (
        train_cuda_graph if isinstance(train_cuda_graph, TrainGraphConfig)
        else TrainGraphConfig(max_seq=seq_cutoff)
    )
    train_backend = state.attention_backend
    if train_backend in ("auto", "triton"):
        # The packed Triton kernel is inference-only. Keep the existing dense SDPA
        # training graph until a separate forward+backward crossover is calibrated.
        train_backend = "sdpa"
    state.train_graph_runner = build_train_runner(
        encoder, state.params, config, backend=train_backend
    )
    state.train_graph_enabled = True


def unpack(target: object) -> object:
    """Restore the original forward, reverting `pack()`."""
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
                "packed-encoders requires input_ids (inputs_embeds is unsupported)"
            )
        _reject_unsupported(kwargs)

        graphs_off = graphs_globally_disabled()
        # The training runner captures fwd+bwd and replays inside autograd, so it
        # engages whenever a grad graph is being built (autocast-safe — see train_graph).
        use_train_graph = (
            state.train_graph_runner is not None
            and state.train_graph_enabled
            and not graphs_off
            and torch.is_grad_enabled()
        )
        # The inference runner replays against fixed addresses, so it is safe only with
        # no autograd graph and no autocast (which frees its ephemeral bf16 weight copies
        # at context exit) — i.e. plain bf16 inference.
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
                "packed-encoders: inference CUDA graphs are enabled but skipped here "
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
        raise NotImplementedError("packed-encoders does not support inputs_embeds")
    if kwargs.get("output_attentions"):
        raise NotImplementedError("packed-encoders does not support output_attentions")
    if kwargs.get("output_hidden_states"):
        raise NotImplementedError("packed-encoders does not support output_hidden_states")
    token_type_ids = kwargs.get("token_type_ids")
    if token_type_ids is not None and bool(token_type_ids.any()):
        raise NotImplementedError("packed-encoders does not support non-zero token_type_ids")


def _require_cuda(encoder: nn.Module) -> None:
    try:
        device = next(encoder.parameters()).device
    except StopIteration as exc:
        raise PackedEncodersError("the model has no parameters") from exc
    if device.type != "cuda":
        raise PackedEncodersError(
            f"the fused path requires CUDA weights; the model is on {device!r}"
        )
