"""The fused-tail forward, run against a live HF `ModernBertModel`.

The forward reads weights straight off the HF submodules (`layers[i].attn.Wqkv`,
`mlp_norm`, `final_norm`, ...) — HF's layout already is the layout the kernels
consume, so there is nothing to re-pack. It splits into two regions on the clean
boundary the CUDA-graph layer needs:

- `prologue` — token embeddings, the RoPE cos/sin tables, and the SDPA-ready
  attention masks. The mask preparation has a host sync (`isinf().any()`), so this
  region runs eager and is never captured.
- `core` — the encoder-layer loop plus the final norm. Pure device work; this is
  the callable the graph runner captures.
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
    """Cast an additive mask to the activation dtype once per forward and replace
    `-inf` with the dtype's finite min (vendor SDPA wants a finite mask). With
    `drop_if_zero`, a mask carrying no `-inf` is dropped to None so SDPA can take
    the fast flash path.

    `capture_safe` drops the `isinf().any()` probe — a device→host sync that is
    illegal inside CUDA-graph capture. It powers the two probe-driven branches:
    dropping an all-zero mask to None (an eager-only flash fast-path), and
    skipping the `masked_fill` when there is no `-inf`. Under capture we always
    want a finite dense mask, so we unconditionally `masked_fill` (a no-op when
    there is no `-inf`) and never drop — no probe, no sync.
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

    `dense_mask=True` forces `full_mask` to a finite tensor (never None) so a
    captured graph keyed on a fixed shape sees a consistent mask on replay.

    `capture_safe=True` removes the only host sync in this region (the mask
    `isinf().any()` probe), so the whole prologue — embeddings, RoPE tables, mask
    build — can run *inside* a CUDA graph instead of eager. The graph runner uses
    it to capture from `input_ids`/`attention_mask` rather than from the
    post-prologue tensors, which collapses the prologue's host launches and drops
    the per-replay copy from the full `S×S` masks down to just the token ids.
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


def _layer_is_global(layer: nn.Module, layer_idx: int, params: ModernBertParams) -> bool:
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
    ctx = ops.attention(q, k, v, mask=mask, window=window, scaling=params.scaling, backend=backend)
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
) -> Tensor:
    """Encoder-layer loop + final norm. The capturable region.

    Each layer gets both the dense mask (for the sdpa backend) and its attention
    *window* (for the flash backend): global layers attend fully `(-1, -1)`, local
    layers over the sliding band `(±sliding_half_window)`."""
    half = params.sliding_half_window
    global_window = (-1, -1)
    local_window = (half, half)
    for layer_idx, layer in enumerate(model.layers):
        if _layer_is_global(layer, layer_idx, params):
            cos, sin, mask, window = cos_global, sin_global, full_mask, global_window
        else:
            cos, sin, mask, window = cos_local, sin_local, sliding_mask, local_window
        x = _encoder_layer(x, layer, params, cos, sin, mask, window, backend)
    final_norm = model.final_norm
    return ops.fused_layer_norm(x, final_norm.weight, final_norm.eps)


def fused_forward(
    model: nn.Module,
    params: ModernBertParams,
    input_ids: Tensor,
    attention_mask: Tensor | None,
    *,
    backend: str = "sdpa",
) -> Tensor:
    """Eager fused-tail forward — returns the final-normed hidden state."""
    p = prologue(model, params, input_ids, attention_mask)
    return core(
        model, params,
        p.x, p.cos_global, p.sin_global, p.cos_local, p.sin_local,
        p.full_mask, p.sliding_mask,
        backend=backend,
    )
