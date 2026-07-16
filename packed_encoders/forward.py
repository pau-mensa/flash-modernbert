"""The fused-tail forward, reading weights straight off the live HF submodules.

Split into two regions for the CUDA-graph layer: `prologue` (embeddings, RoPE
tables, SDPA masks) carries a host sync in mask prep so it runs eager; `core`
(the layer loop + final norm) is pure device work and is what the graph captures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from packed_encoders import attention_dispatch, ops
from packed_encoders.config import ModernBertParams
from packed_encoders.state import ATTR

_GRAPH_ENV = "PACKED_ENCODERS_GRAPH"


@dataclass(frozen=True)
class _Prologue:
    x: Tensor
    cos_global: Tensor
    sin_global: Tensor
    cos_local: Tensor
    sin_local: Tensor
    full_mask: Tensor | None
    sliding_mask: Tensor


# ---------------------------------------------------------------------------
# Prologue helpers
# ---------------------------------------------------------------------------


def _rope_tables(
    seq_len: int, head_dim: int, theta: float, device, dtype
) -> tuple[Tensor, Tensor]:
    """cos, sin of shape [1, S, D] (the rope kernel only reads the [S, D] slice)."""
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().unsqueeze(0).to(dtype), emb.sin().unsqueeze(0).to(dtype)


def _build_masks(
    attention_mask: Tensor | None, seq_len: int, sliding_half_window: int, device
) -> tuple[Tensor | None, Tensor]:
    """(full, sliding) additive masks, fp32, broadcastable to [B, 1, S, S].
    `full` is None when there is no padding."""
    pad_mask = None
    if attention_mask is not None:
        am = attention_mask.to(torch.float32)
        pad_mask = torch.zeros_like(am).masked_fill_(am == 0, float("-inf"))
        pad_mask = pad_mask[:, None, None, :]

    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    band_outside = (i - j).abs() > sliding_half_window
    band = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=device)
    band = band.masked_fill_(band_outside, float("-inf"))[None, None, :, :]

    full = pad_mask
    sliding = band if pad_mask is None else (band + pad_mask)
    return full, sliding


def _sdpa_ready_mask(
    mask: Tensor | None, dtype, *, drop_if_zero: bool, capture_safe: bool = False
) -> Tensor | None:
    """Cast an additive mask to `dtype` and replace `-inf` with the finite dtype min
    (vendor SDPA wants a finite mask). `drop_if_zero` returns None for an all-zero
    mask so SDPA takes the flash fast path.

    `capture_safe` skips the `isinf().any()` probe (a device→host sync, illegal under
    graph capture) and unconditionally produces a finite dense mask instead.
    """
    if mask is None:
        return None
    if capture_safe:
        m = mask.to(dtype)
        return m.masked_fill(torch.isinf(m), torch.finfo(dtype).min)
    has_inf = bool(torch.isinf(mask).any())
    if drop_if_zero and not has_inf:
        return None
    m = mask.to(dtype)
    if has_inf:
        m = m.masked_fill(torch.isinf(m), torch.finfo(dtype).min)
    return m


def prologue(
    model: nn.Module,
    params: ModernBertParams,
    input_ids: Tensor,
    attention_mask: Tensor | None,
    *,
    dense_mask: bool = False,
    capture_safe: bool = False,
) -> _Prologue:
    """Embeddings + RoPE tables + SDPA-ready masks.

    `dense_mask=True` forces `full_mask` to a finite tensor (never None) so a graph
    keyed on a fixed shape sees a consistent mask on replay. `capture_safe=True`
    removes this region's only host sync, letting the whole prologue be captured.
    """
    b, s = input_ids.shape
    device = input_ids.device

    emb = model.embeddings.tok_embeddings(input_ids)
    norm = model.embeddings.norm
    x = ops.fused_layer_norm(emb, norm.weight, norm.eps)
    dtype = x.dtype

    head_dim = params.head_dim
    cos_g, sin_g = _rope_tables(s, head_dim, params.global_rope_theta, device, dtype)
    if params.local_rope_theta == params.global_rope_theta:
        cos_l, sin_l = cos_g, sin_g
    else:
        cos_l, sin_l = _rope_tables(s, head_dim, params.local_rope_theta, device, dtype)

    full_mask, sliding_mask = _build_masks(
        attention_mask, s, params.sliding_half_window, device
    )
    full_mask = _sdpa_ready_mask(
        full_mask, dtype, drop_if_zero=not dense_mask, capture_safe=capture_safe
    )
    if dense_mask and full_mask is None:
        full_mask = torch.zeros((b, 1, 1, s), dtype=dtype, device=device)
    sliding_mask = _sdpa_ready_mask(
        sliding_mask, dtype, drop_if_zero=False, capture_safe=capture_safe
    )

    return _Prologue(x, cos_g, sin_g, cos_l, sin_l, full_mask, sliding_mask)


# ---------------------------------------------------------------------------
# Core (capturable)
# ---------------------------------------------------------------------------


def _layer_is_global(
    layer: nn.Module, layer_idx: int, params: ModernBertParams
) -> bool:
    attention_type = getattr(layer, "attention_type", None)
    if attention_type == "full_attention":
        return True
    if attention_type == "sliding_attention":
        return False
    return params.is_global_layer(layer_idx)


def _encoder_layer(
    x: Tensor,
    pending_residual: Tensor | None,
    layer: nn.Module,
    params: ModernBertParams,
    cos: Tensor,
    sin: Tensor,
    mask: Tensor | None,
    window: tuple[int, int],
    backend: str,
    cu_seqlens: Tensor | None = None,
    max_seqlen: int | None = None,
) -> tuple[Tensor, Tensor]:
    b, s = x.shape[:2]
    h, d = params.num_attention_heads, params.head_dim

    attn_norm = layer.attn_norm
    if isinstance(attn_norm, nn.Identity):
        # ModernBERT uses Identity only for layer 0, before any residual is pending.
        # Keep this defensive fallback correct for architecture variants.
        if pending_residual is not None:
            x = x + pending_residual
        qkv = F.linear(x, layer.attn.Wqkv.weight, None)
    else:
        if pending_residual is None:
            normed = ops.fused_layer_norm(x, attn_norm.weight, attn_norm.eps)
        else:
            x, normed = ops.fused_add_layer_norm(
                x, pending_residual, attn_norm.weight, attn_norm.eps
            )
        qkv = ops._linear(normed, layer.attn.Wqkv.weight)

    if ops.use_bshd_rope_flash(backend, h, d, cu_seqlens):
        # Inference flash fast path: RoPE reads the packed qkv and hands flash its native
        # [.., S, H, D] layout — no transpose to [B,H,S,D], no .contiguous() copy.
        ctx = ops.flash_attention_qkv_bshd(
            qkv, h, d, cos=cos, sin=sin, window=window, scaling=params.scaling,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, backend=backend,
        )
    else:
        qkv = qkv.view(b, s, 3, h, d)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = ops.fused_apply_rope(q, k, cos, sin)
        ctx = ops.attention(
            q,
            k,
            v,
            mask=mask,
            window=window,
            scaling=params.scaling,
            backend=backend,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
    wo_out = ops._linear(ctx, layer.attn.Wo.weight)
    mlp_norm = layer.mlp_norm
    x, normed = ops.fused_add_layer_norm(
        x, wo_out, mlp_norm.weight, mlp_norm.eps
    )
    # Do not materialize this residual add.  The next layer's attn_norm (or the
    # model's final_norm) consumes it in its fused add+LN kernel.
    pending_residual = ops.geglu_mlp(
        normed, layer.mlp.Wi.weight, layer.mlp.Wo.weight
    )
    return x, pending_residual


def core(
    model: nn.Module,
    params: ModernBertParams,
    x: Tensor,
    cos_global: Tensor,
    sin_global: Tensor,
    cos_local: Tensor,
    sin_local: Tensor,
    full_mask: Tensor | None,
    sliding_mask: Tensor,
    *,
    backend: str = "sdpa",
    cu_seqlens: Tensor | None = None,
    max_seqlen: int | None = None,
) -> Tensor:
    """Encoder-layer loop + final norm — the capturable region.

    Each layer gets a dense mask (sdpa backend) and a window (flash backend): global
    layers `(-1, -1)`, local layers the sliding band. `cu_seqlens`/`max_seqlen` (the
    packed varlen path) pass through to flash; masks are unused there."""
    half = params.sliding_half_window
    global_window = (-1, -1)
    local_window = (half, half)
    pending_residual = None
    for layer_idx, layer in enumerate(model.layers):
        if _layer_is_global(layer, layer_idx, params):
            cos, sin, mask, window = cos_global, sin_global, full_mask, global_window
        else:
            cos, sin, mask, window = cos_local, sin_local, sliding_mask, local_window
        x, pending_residual = _encoder_layer(
            x,
            pending_residual,
            layer,
            params,
            cos,
            sin,
            mask,
            window,
            backend,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
    final_norm = model.final_norm
    if pending_residual is None:
        return ops.fused_layer_norm(x, final_norm.weight, final_norm.eps)
    _, out = ops.fused_add_layer_norm(
        x, pending_residual, final_norm.weight, final_norm.eps
    )
    return out


# ---------------------------------------------------------------------------
# Varlen (packed, variable-length) path
# ---------------------------------------------------------------------------


def _has_padding(attention_mask: Tensor | None) -> bool:
    # `.all()` is a host sync — fine here, the varlen path is never captured.
    return attention_mask is not None and not bool((attention_mask == 1).all())


def _resolve_eager_backend(
    backend: str,
    workload: attention_dispatch.AttentionWorkload | None = None,
    *,
    triton_supported: bool = False,
    flash_available: bool | None = None,
    for_cuda_graph: bool = False,
) -> str:
    """Resolve the public backend before choosing packed versus rectangular compute.

    On a calibrated card auto compares packed Triton with Flash; elsewhere it
    conservatively prefers Flash.
    SDPA is reached only when neither optimized backend can serve the call.
    """
    if flash_available is None:
        flash_available = _flash_available()
    inference = not torch.is_grad_enabled()
    if backend == "triton":
        if inference and triton_supported:
            return "triton"
        return "flash" if flash_available else "sdpa"
    if backend == "auto":
        if inference and triton_supported:
            policy = attention_dispatch.get_packed_inference_policy(
                for_cuda_graph=for_cuda_graph
            )
            if policy is not None and workload is not None:
                return "flash" if flash_available and policy.use_flash(workload) else "triton"
            if not flash_available:
                return "triton"
        return "flash" if flash_available else "sdpa"
    return backend


def _flash_available() -> bool:
    try:
        ops._load_flash_attn()
        return True
    except ImportError:
        return False


def _triton_available() -> bool:
    try:
        ops._load_packed_short_attention()
        return True
    except ImportError:
        return False


def _packed_short_invariants(
    params: ModernBertParams,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_seqlen: int,
    cu_seqlens: Tensor | None = None,
) -> bool:
    return (
        not torch.is_grad_enabled()
        and device.type == "cuda"
        and dtype == torch.bfloat16
        and params.num_attention_heads == 12
        and params.head_dim == 64
        and 0 < int(max_seqlen) <= 128
        and params.sliding_half_window == 64
        and (cu_seqlens is None or (
            cu_seqlens.is_cuda and cu_seqlens.dtype == torch.int32
        ))
        and _triton_available()
    )


def _attention_workload(
    params: ModernBertParams,
    input_ids: Tensor,
    attention_mask: Tensor | None,
) -> attention_dispatch.AttentionWorkload:
    """Build score inputs once at the eager boundary.

    A mask requires one device-to-host transfer of `B` lengths. The result retains
    only shape statistics available to the packed eager/graph dispatcher.
    """
    b, padded_s = input_ids.shape
    if attention_mask is None:
        lengths = (padded_s,) * b
    else:
        lengths = tuple(
            int(length)
            for length in attention_mask.sum(dim=1, dtype=torch.int64).tolist()
        )
    return attention_dispatch.AttentionWorkload.from_lengths(lengths)


def _unpad(
    x: Tensor, attention_mask: Tensor
) -> tuple[Tensor, Tensor, Tensor, int, Tensor]:
    """Strip padding from `x` [B, S, H] using `attention_mask` [B, S] (1 = real),
    returning what `flash_attn_varlen_func` needs: packed tokens, the flat `indices`
    for re-padding, `cu_seqlens`, `max_seqlen`, and `position_ids`.

    `position_ids = indices % S` is each token's column in the padded row. Stock
    ModernBERT assigns RoPE positions by `arange(S)` regardless of padding, so the
    per-token gather `cos_table[position_ids]` reproduces HF exactly.
    """
    b, s = attention_mask.shape
    hidden = x.shape[-1]
    seqlens = attention_mask.sum(dim=1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max())
    x_packed = x.reshape(b * s, hidden).index_select(0, indices)
    position_ids = indices % s
    return x_packed, indices, cu_seqlens, max_seqlen, position_ids


def _repad(x_packed: Tensor, indices: Tensor, b: int, s: int) -> Tensor:
    """Scatter packed tokens back into a padded [B, S, H] tensor (the inverse of
    `_unpad`); pad rows stay zero."""
    hidden = x_packed.shape[-1]
    out = x_packed.new_zeros(b * s, hidden)
    out.index_copy_(0, indices, x_packed)
    return out.reshape(b, s, hidden)


def packed_forward(
    model: nn.Module,
    params: ModernBertParams,
    packed_ids: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    position_ids: Tensor,
    *,
    backend: str | None = None,
    _allow_graph: bool = True,
) -> Tensor:
    """Run the encoder on an *already-packed* batch; return packed `[total, H]`.

    The packed-paradigm entry: `_varlen_forward` **minus its two padding-boundary ops**
    — no `_unpad` of an incoming `[B, S]` batch (the caller's collator already packed it)
    and no `_repad` of the result (it stays `[total, H]` for a packed loss). `core` runs
    unchanged: attention takes `cu_seqlens` (confining it within each sequence) and RoPE
    takes per-token-gathered cos/sin.

    When ``fm.pack(cuda_graph=...)`` is active, the packed graph runner captures this
    function at a fixed token budget and replays on subsequent calls, eliminating all
    per-kernel dispatch overhead.  Falls back to eager when inputs exceed the graph
    budget or when grad/autocast is enabled.

    Args:
        packed_ids: real token ids `[total]`, all sequences concatenated.
        cu_seqlens: `[n_seq + 1]` int32 prefix sums of sequence lengths (flash varlen).
        max_seqlen: longest sequence length (bounds the RoPE table).
        position_ids: `[total]` within-sequence index per token (0,1,2,… per sequence).
            Must match HF's `arange`-per-position convention (identical to `_unpad`'s
            `indices % S` for right-padded data) or RoPE positions — and any
            padded-vs-packed parity — break.

    Additive and non-breaking: the shipped `[B, S]` HF forward is untouched. The
    backend defaults to the model's packed-encoders setting; without patch state it
    retains the historical Flash default.
    """
    state = getattr(model, ATTR, None)
    requested_backend = (
        backend if backend is not None
        else (state.attention_backend if state is not None else "flash")
    )
    if (
        _allow_graph
        and state is not None
        and requested_backend == state.attention_backend
        and state.packed_graph_runner is not None
        and os.environ.get(_GRAPH_ENV, "1") != "0"
        and not torch.is_grad_enabled()
        and not torch.is_autocast_enabled("cuda")
    ):
        return state.packed_graph_runner(
            packed_ids, cu_seqlens, max_seqlen, position_ids
        )

    device = packed_ids.device

    # Embed only the real tokens.
    emb = model.embeddings.tok_embeddings(packed_ids)
    norm = model.embeddings.norm
    x_packed = ops.fused_layer_norm(emb, norm.weight, norm.eps)
    dtype = x_packed.dtype

    # RoPE tables up to the longest real sequence only (position_ids < max_seqlen),
    # then gather per token — same values as the full [0, S) table the dense path uses.
    head_dim = params.head_dim
    cos_g, sin_g = _rope_tables(
        max_seqlen, head_dim, params.global_rope_theta, device, dtype
    )
    if params.local_rope_theta == params.global_rope_theta:
        cos_l, sin_l = cos_g, sin_g
    else:
        cos_l, sin_l = _rope_tables(
            max_seqlen, head_dim, params.local_rope_theta, device, dtype
        )

    def gather(t: Tensor) -> Tensor:
        return t[0].index_select(0, position_ids)

    packed_backend = _resolve_packed_backend(
        requested_backend, params, x_packed, cu_seqlens, max_seqlen
    )
    if packed_backend == "sdpa":
        return _packed_sdpa_fallback(
            model, params, packed_ids, cu_seqlens, max_seqlen, position_ids
        )

    out = core(
        model,
        params,
        x_packed.unsqueeze(0),
        gather(cos_g),
        gather(sin_g),
        gather(cos_l),
        gather(sin_l),
        None,
        None,
        backend=packed_backend,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    return out.squeeze(0)


def _resolve_packed_backend(
    backend: str,
    params: ModernBertParams,
    x: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
) -> str:
    """Resolve packed Triton versus Flash once, before the layer loop.

    The predicate is host-only and graph-capture-safe: the 5090 policy reads only
    static tensor shapes, never ``cu_seqlens`` values. The selected backend is baked
    into each packed graph bucket.
    """
    return _resolve_packed_backend_from_shape(
        backend,
        params,
        device=x.device,
        dtype=x.dtype,
        total_tokens=x.shape[-2],
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )


def _resolve_packed_backend_from_shape(
    backend: str,
    params: ModernBertParams,
    *,
    device: torch.device,
    dtype: torch.dtype,
    total_tokens: int,
    cu_seqlens: Tensor,
    max_seqlen: int,
    for_cuda_graph: bool = False,
) -> str:
    """Resolve from host-visible packed shapes before eager execution or capture."""
    supported = _packed_short_invariants(
        params,
        device=device,
        dtype=dtype,
        max_seqlen=max_seqlen,
        cu_seqlens=cu_seqlens,
    )
    flash_available = _flash_available()
    workload = _packed_workload(
        total_tokens=total_tokens,
        n_sequences=cu_seqlens.numel() - 1,
        max_seqlen=max_seqlen,
    )
    resolved = _resolve_eager_backend(
        backend,
        workload,
        triton_supported=supported,
        flash_available=flash_available,
        for_cuda_graph=for_cuda_graph,
    )
    return resolved


def _packed_sdpa_fallback(
    model: nn.Module,
    params: ModernBertParams,
    packed_ids: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    position_ids: Tensor,
) -> Tensor:
    """Re-pad an already-packed batch for the dependency-free SDPA fallback."""
    n_sequences = cu_seqlens.numel() - 1
    total = packed_ids.numel()
    token_index = torch.arange(total, device=packed_ids.device)
    sequence_ids = torch.bucketize(token_index, cu_seqlens[1:], right=True)
    flat_indices = sequence_ids * int(max_seqlen) + position_ids
    padded_ids = packed_ids.new_zeros(n_sequences * int(max_seqlen))
    padded_ids.index_copy_(0, flat_indices, packed_ids)
    attention_mask = packed_ids.new_zeros(n_sequences * int(max_seqlen))
    attention_mask.index_fill_(0, flat_indices, 1)
    padded_out = fused_forward(
        model,
        params,
        padded_ids.view(n_sequences, int(max_seqlen)),
        attention_mask.view(n_sequences, int(max_seqlen)),
        backend="sdpa",
    )
    return padded_out.reshape(-1, padded_out.shape[-1]).index_select(0, flat_indices)


def _packed_workload(
    *, total_tokens: int, n_sequences: int, max_seqlen: int
) -> attention_dispatch.AttentionWorkload:
    """Shape-only workload used by eager and captured packed policies."""
    return attention_dispatch.AttentionWorkload.from_summary(
        n_sequences=n_sequences,
        live_tokens=total_tokens,
        max_seqlen=max_seqlen,
    )


def _varlen_forward(
    model: nn.Module,
    params: ModernBertParams,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    backend: str = "flash",
) -> Tensor:
    """Run the encoder on the packed (unpadded) batch as a single `b=1` sequence,
    then re-pad. Internally: `_unpad` → `packed_forward` → `_repad`, i.e. the packed
    entry wrapped in the two padding-boundary ops that the patched `[B, S]` forward
    needs (the showcase's packed path calls `packed_forward` directly and skips both)."""
    b, s = input_ids.shape
    # Unpad ids first (treat [B, S] as [B, S, 1] so _unpad's index math is reused).
    packed_ids, indices, cu_seqlens, max_seqlen, position_ids = _unpad(
        input_ids.unsqueeze(-1), attention_mask
    )
    packed_ids = packed_ids.squeeze(-1)
    out = packed_forward(
        model, params, packed_ids, cu_seqlens, max_seqlen, position_ids,
        backend=backend,
    )
    return _repad(out, indices, b, s)


def fused_forward(
    model: nn.Module,
    params: ModernBertParams,
    input_ids: Tensor,
    attention_mask: Tensor | None,
    *,
    backend: str = "sdpa",
) -> Tensor:
    """Eager fused-tail forward — returns the final-normed hidden state.

    A padded batch with the flash backend routes through the packed varlen path;
    otherwise the rectangular `[B, S]` path (dense flash on unpadded, or sdpa)."""
    workload = None
    if backend in ("auto", "triton"):
        workload = _attention_workload(params, input_ids, attention_mask)
    weight = model.embeddings.tok_embeddings.weight
    triton_supported = backend in ("auto", "triton") and _packed_short_invariants(
        params,
        device=input_ids.device,
        dtype=weight.dtype,
        max_seqlen=(workload.max_seqlen if workload is not None else input_ids.shape[1]),
    )
    resolved = _resolve_eager_backend(
        backend, workload, triton_supported=triton_supported
    )
    if resolved in ("flash", "triton"):
        has_padding = (
            workload.fragmentation_tokens > 0
            if workload is not None
            else _has_padding(attention_mask)
        )
        if resolved == "triton" or has_padding:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            return _varlen_forward(
                model, params, input_ids, attention_mask, backend=backend
            )
    p = prologue(model, params, input_ids, attention_mask)
    return core(
        model,
        params,
        p.x,
        p.cos_global,
        p.sin_global,
        p.cos_local,
        p.sin_local,
        p.full_mask,
        p.sliding_mask,
        backend=resolved,
    )
