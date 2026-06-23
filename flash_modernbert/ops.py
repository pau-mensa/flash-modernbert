"""The fused-tail numerical ops: autograd Functions + their dispatch wrappers.

Only the three tails that the fused path swaps onto ModernBERT live here —
LayerNorm, RoPE, and GeGLU. Each is a `torch.autograd.Function` whose forward is
a CuteDSL kernel and whose backward is the equivalent math run directly (the same
ATen op the engine would call, or a hand-rolled exact gradient), with no second
forward and no autograd subgraph built per call. GEMMs stay on cuBLAS (`F.linear`)
and attention stays on vendor SDPA; those are not autograd Functions, just the
plain ops below.

The `fused_*` wrappers pick the autograd Function when a gradient is actually
needed (training, or fp32-master-weights under autocast) and the raw detached
kernel otherwise (inference, GradCache pass-1), so the no-grad path never pays for
ctx bookkeeping it won't use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flash_modernbert._kernels.geglu import (
    geglu as _geglu_kernel,
    geglu_bwd as _geglu_bwd_kernel,
)
from flash_modernbert._kernels.layer_norm import (
    layer_norm as _ln_kernel,
    layer_norm_with_stats as _ln_kernel_with_stats,
)
from flash_modernbert._kernels.ln_bwd_inplace import ln_bwd as _ln_bwd_dyn_kernel
from flash_modernbert._kernels.rope import apply_rope as _rope_kernel


# ---------------------------------------------------------------------------
# LayerNorm (bias-free)
# ---------------------------------------------------------------------------


class _LayerNormFn(torch.autograd.Function):
    """Forward: `layer_norm_with_stats` yields y plus the flat fp32 mean/rstd in
    one kernel. Backward: the M-dynamic `ln_bwd` computes grad_x and grad_gamma in
    one launch from those stats. Both are re-JIT-free across the token count, so a
    new sequence length never triggers a recompile. Falls back to ATen for the
    dtype/H the cute reduction does not cover (H must be a multiple of 256)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        x_d = x.detach()
        w_d = weight.detach()
        h = x_d.shape[-1]
        use_cute = (
            x_d.is_cuda
            and x_d.dtype == torch.bfloat16
            and w_d.dtype == torch.bfloat16
            and h % 256 == 0
        )
        if use_cute:
            y, mean, rstd = _ln_kernel_with_stats(x_d, w_d, eps)
        else:
            y_aten, mean, rstd = torch.ops.aten.native_layer_norm.default(
                x_d, (h,), w_d, None, eps
            )
            cute_fwd_ok = (
                x_d.is_cuda
                and h % 32 == 0
                and x_d.dtype == torch.bfloat16
                and w_d.dtype == torch.bfloat16
            )
            y = _ln_kernel(x_d, w_d, eps) if cute_fwd_ok else y_aten
        ctx.save_for_backward(x_d, w_d, mean, rstd)
        ctx.normalized_shape = (h,)
        ctx.use_cute = use_cute
        return y

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, mean, rstd = ctx.saved_tensors
        if ctx.use_cute and grad_out.dtype == torch.bfloat16:
            grad_x, grad_weight = _ln_bwd_dyn_kernel(
                grad_out.contiguous(), x, mean, rstd, weight, inplace=False
            )
            return grad_x, grad_weight, None
        mean_a = mean.reshape(*x.shape[:-1], 1)
        rstd_a = rstd.reshape(*x.shape[:-1], 1)
        grad_input, grad_weight, _ = torch.ops.aten.native_layer_norm_backward.default(
            grad_out.contiguous(),
            x,
            ctx.normalized_shape,
            mean_a,
            rstd_a,
            weight,
            None,
            [True, True, False],
        )
        return grad_input, grad_weight, None


def layer_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Autograd-aware bias-free LayerNorm."""
    return _LayerNormFn.apply(x, weight, eps)


def fused_layer_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm that routes through autograd only when a gradient is live.

    fp32 master weights under bf16 autocast (HF Trainer / PyLate) must go through
    the `custom_fwd` cast wrapper even in a no-grad pass so x and gamma are cast
    together — the raw kernel requires equal dtypes.
    """
    if torch.is_autocast_enabled("cuda") or (
        torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
    ):
        return _LayerNormFn.apply(x, weight, eps)
    return _ln_kernel(x.detach(), weight.detach(), eps)


# ---------------------------------------------------------------------------
# RoPE (split-half / concat convention, matching HF ModernBERT)
# ---------------------------------------------------------------------------


class _ApplyRopeFn(torch.autograd.Function):
    """RoPE is linear, so its transpose Jacobian is the conjugate rotation:
    backward is the same kernel with sin negated. No autograd graph, no recompute.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, q, k, cos, sin):
        cos_d = cos.detach()
        sin_d = sin.detach()
        ctx.save_for_backward(cos_d, sin_d)
        return _rope_kernel(q.detach(), k.detach(), cos_d, sin_d)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_q_out, grad_k_out):
        cos, sin = ctx.saved_tensors
        neg_sin = (-sin).contiguous()
        grad_q, grad_k = _rope_kernel(
            grad_q_out.contiguous(), grad_k_out.contiguous(), cos, neg_sin
        )
        return grad_q, grad_k, None, None


def apply_rope(q, k, cos, sin):
    """Autograd-aware RoPE."""
    return _ApplyRopeFn.apply(q, k, cos, sin)


def fused_apply_rope(q, k, cos, sin):
    if torch.is_grad_enabled() and (
        q.requires_grad or k.requires_grad or cos.requires_grad or sin.requires_grad
    ):
        return _ApplyRopeFn.apply(q, k, cos, sin)
    return _rope_kernel(q.detach(), k.detach(), cos.detach(), sin.detach())


# ---------------------------------------------------------------------------
# GeGLU activation
# ---------------------------------------------------------------------------


class _GegluFn(torch.autograd.Function):
    """Forward: `gelu(a) * gate` in one kernel. Backward: a single fused kernel
    recomputes gelu(a)/gelu'(a) and writes (da, dgate) into proj's storage."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, proj, out_dtype):
        proj_d = proj.detach()
        ctx.save_for_backward(proj_d)
        return _geglu_kernel(proj_d, out_dtype=out_dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        (proj,) = ctx.saved_tensors
        grad_act = grad_out.contiguous()
        if grad_act.dtype != proj.dtype:
            grad_act = grad_act.to(proj.dtype)
        grad_proj = _geglu_bwd_kernel(
            proj, grad_act, grad_proj_dtype=proj.dtype, inplace=True
        )
        return grad_proj, None


def geglu(proj, *, out_dtype=None):
    """Autograd-aware GeGLU."""
    return _GegluFn.apply(proj, out_dtype)


def fused_geglu(proj, *, out_dtype=None):
    if torch.is_grad_enabled() and proj.requires_grad:
        return _GegluFn.apply(proj, out_dtype)
    return _geglu_kernel(proj.detach(), out_dtype=out_dtype)


# ---------------------------------------------------------------------------
# Composite layer ops (cuBLAS GEMMs + the tails above)
# ---------------------------------------------------------------------------


def ln_linear(x, weight, gamma, eps):
    return F.linear(fused_layer_norm(x, gamma, eps), weight, None)


def geglu_mlp(x, w_in, w_out):
    proj = F.linear(x, w_in, None)
    return F.linear(fused_geglu(proj), w_out, None)


def geglu_mlp_pre_ln(x, w_in, w_out, gamma, eps):
    return geglu_mlp(fused_layer_norm(x, gamma, eps), w_in, w_out)


# ---------------------------------------------------------------------------
# Attention (vendor SDPA — no FlashAttention kernel in this package)
# ---------------------------------------------------------------------------


def sdpa_attention(q, k, v, additive_mask, scaling):
    """q/k/v are strided [B, H, S, D] transpose views; SDPA reads them as-is
    (no contiguous copy, bit-identical to contiguous on the validated arches).
    `additive_mask` must already be SDPA-ready (finite, dtype-matched, or None) —
    the prologue prepares it once per forward."""
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=additive_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scaling,
    )
    out = out.transpose(1, 2)
    b, s = out.shape[:2]
    return out.reshape(b, s, -1)


_flash_attn_func = None


def _load_flash_attn():
    """Import flash_attn lazily so the dependency is only required when the flash
    backend is actually selected (the default sdpa path stays dependency-free)."""
    global _flash_attn_func
    if _flash_attn_func is None:
        try:
            from flash_attn import flash_attn_func
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise ImportError(
                "attention_backend='flash' needs the flash-attn package "
                "(pip install flash-attn). The default 'sdpa' backend needs nothing."
            ) from exc
        _flash_attn_func = flash_attn_func
    return _flash_attn_func


def flash_attention(q, k, v, window, scaling):
    """FlashAttention with the layer's *structure* (window) instead of a dense
    mask: `window=(-1, -1)` is full attention (global layers), `window=(w, w)` is
    a symmetric sliding band `|i - j| <= w` (local layers), which FA prunes to the
    band rather than computing the full `S×S` — the long-S win.

    q/k/v arrive as the [B, H, S, D] transpose views the rest of the tail uses;
    FA wants [B, S, H, D]. The `.transpose(1, 2)` back is a free view — and FA
    only requires the head dim (last) to be contiguous, which it is (RoPE outputs
    contiguous [B, H, S, D], and v's unbind view keeps D packed), so we pass the
    strided views directly with no copy. Assumes an unpadded batch — padding would
    need `flash_attn_varlen_func` with cu_seqlens; sdpa remains the path for that.
    """
    flash_attn_func = _load_flash_attn()
    out = flash_attn_func(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=0.0, softmax_scale=scaling, causal=False, window_size=window,
    )
    b, s = out.shape[:2]
    return out.reshape(b, s, -1)


def attention(q, k, v, *, mask, window, scaling, backend):
    """Dispatch attention to the selected backend. `mask` feeds sdpa; `window`
    feeds flash. Both are precomputed per layer so this stays a thin switch."""
    if backend == "flash":
        return flash_attention(q, k, v, window, scaling)
    return sdpa_attention(q, k, v, mask, scaling)
