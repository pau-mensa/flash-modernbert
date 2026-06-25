"""The fused-tail forward, reading weights straight off the live HF submodules.

Split into two regions for the CUDA-graph layer: `prologue` (embeddings, RoPE
tables, SDPA masks) carries a host sync in mask prep so it runs eager; `core`
(the layer loop + final norm) is pure device work and is what the graph captures.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flash_modernbert import ops
from flash_modernbert.config import ModernBertParams


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
    layer: nn.Module,
    params: ModernBertParams,
    cos: Tensor,
    sin: Tensor,
    mask: Tensor | None,
    window: tuple[int, int],
    backend: str,
    cu_seqlens: Tensor | None = None,
    max_seqlen: int | None = None,
) -> Tensor:
    b, s = x.shape[:2]
    h, d = params.num_attention_heads, params.head_dim

    attn_norm = layer.attn_norm
    if isinstance(attn_norm, nn.Identity):
        qkv = F.linear(x, layer.attn.Wqkv.weight, None)
    else:
        qkv = ops.ln_linear(x, layer.attn.Wqkv.weight, attn_norm.weight, attn_norm.eps)

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
    x = x + F.linear(ctx, layer.attn.Wo.weight, None)

    mlp_norm = layer.mlp_norm
    x = x + ops.geglu_mlp_pre_ln(
        x, layer.mlp.Wi.weight, layer.mlp.Wo.weight, mlp_norm.weight, mlp_norm.eps
    )
    return x


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
    for layer_idx, layer in enumerate(model.layers):
        if _layer_is_global(layer, layer_idx, params):
            cos, sin, mask, window = cos_global, sin_global, full_mask, global_window
        else:
            cos, sin, mask, window = cos_local, sin_local, sliding_mask, local_window
        x = _encoder_layer(
            x,
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
    return ops.fused_layer_norm(x, final_norm.weight, final_norm.eps)


# ---------------------------------------------------------------------------
# Varlen (packed, variable-length) path
# ---------------------------------------------------------------------------


def _has_padding(attention_mask: Tensor | None) -> bool:
    # `.all()` is a host sync — fine here, the varlen path is never captured.
    return attention_mask is not None and not bool((attention_mask == 1).all())


def _resolve_eager_backend(backend: str, seq_len: int) -> str:
    """Resolve `"auto"` once at the forward level (so the varlen-vs-dense decision is
    made before the loop), using the same `FLASH_MIN_SEQ` rule as `ops.attention`."""
    if backend == "auto":
        return "flash" if seq_len >= ops.FLASH_MIN_SEQ else "sdpa"
    return backend


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


def _varlen_forward(
    model: nn.Module,
    params: ModernBertParams,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """Run the encoder on the packed (unpadded) batch as a single `b=1` sequence,
    then re-pad. `core` runs unchanged — only attention takes `cu_seqlens` (confining
    it within each original sequence) and RoPE takes per-token-gathered cos/sin."""
    b, s = input_ids.shape
    device = input_ids.device
    # Unpad ids first (treat [B, S] as [B, S, 1] so _unpad's index math is reused).
    packed_ids, indices, cu_seqlens, max_seqlen, position_ids = _unpad(
        input_ids.unsqueeze(-1), attention_mask
    )
    packed_ids = packed_ids.squeeze(-1)

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
        backend="flash",
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    return _repad(out.squeeze(0), indices, b, s)


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
    resolved = _resolve_eager_backend(backend, input_ids.shape[1])
    if resolved == "flash" and _has_padding(attention_mask):
        return _varlen_forward(model, params, input_ids, attention_mask)
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
