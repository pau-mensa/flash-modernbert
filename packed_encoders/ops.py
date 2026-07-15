"""The fused-tail numerical ops: LayerNorm, RoPE, GeGLU.

Each is a `torch.autograd.Function` wrapping a CuteDSL forward kernel and an exact
backward (no second forward, no per-call autograd subgraph). GEMMs and attention
stay on cuBLAS / vendor SDPA. The `fused_*` wrappers route through autograd only
when a gradient is live (training, or fp32-master under autocast) and the raw
detached kernel otherwise, so the no-grad path skips ctx bookkeeping.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from packed_encoders._kernels.geglu import (
    geglu as _geglu_kernel,
    geglu_bwd as _geglu_bwd_kernel,
)
from packed_encoders._kernels.layer_norm import (
    add_layer_norm as _add_ln_kernel,
    add_layer_norm_with_stats as _add_ln_kernel_with_stats,
    layer_norm as _ln_kernel,
    layer_norm_with_stats as _ln_kernel_with_stats,
)
from packed_encoders._kernels.ln_bwd_inplace import ln_bwd as _ln_bwd_dyn_kernel
from packed_encoders._kernels.rope import (
    apply_rope as _rope_kernel,
    apply_rope_bshd as _rope_bshd_kernel,
    _bshd_applicable,
)


# ---------------------------------------------------------------------------
# LayerNorm (bias-free)
# ---------------------------------------------------------------------------


class _LayerNormFn(torch.autograd.Function):
    """Cute forward yields y + fp32 mean/rstd; cute backward computes grad_x/grad_gamma
    from those stats in one launch. Both are re-JIT-free across token count. Falls back
    to ATen when the cute reduction can't cover the dtype/H (H must be a multiple of 256)."""

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


class _AddLayerNormFn(torch.autograd.Function):
    """Two-input/two-output residual-add+LN autograd boundary.

    Returning ``x_new`` as well as the normalized value lets the encoder carry the
    residual stream without redoing the add.  Backward folds ``grad_x_new`` into the
    LN input gradient and returns that same combined gradient to both add inputs.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(
        ctx,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ):
        x_d = x.detach()
        residual_d = residual.detach()
        w_d = weight.detach()
        h = x_d.shape[-1]
        use_cute = (
            x_d.is_cuda
            and x_d.dtype == torch.bfloat16
            and residual_d.dtype == torch.bfloat16
            and w_d.dtype == torch.bfloat16
            and h % 256 == 0
        )
        if use_cute:
            x_new, y, mean, rstd = _add_ln_kernel_with_stats(
                x_d, residual_d, w_d, eps
            )
        else:
            # This add intentionally happens in the (possibly autocast-forced) input
            # dtype so LN sees the same rounded residual stream as stock ModernBERT.
            x_new = x_d + residual_d
            y_aten, mean, rstd = torch.ops.aten.native_layer_norm.default(
                x_new, (h,), w_d, None, eps
            )
            cute_fwd_ok = (
                x_d.is_cuda
                and h % 32 == 0
                and x_d.dtype == residual_d.dtype == w_d.dtype == torch.bfloat16
            )
            y = (
                _add_ln_kernel(x_d, residual_d, w_d, eps)[1]
                if cute_fwd_ok
                else y_aten
            )
        # Save a detached alias: returning ``x_new`` makes autograd attach this
        # Function's grad_fn to that output tensor, which DLPack correctly refuses
        # to export from backward.
        ctx.save_for_backward(x_new.detach(), w_d, mean, rstd)
        ctx.normalized_shape = (h,)
        ctx.use_cute = use_cute
        return x_new, y

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx, grad_x_new: torch.Tensor, grad_out: torch.Tensor
    ):
        x_new, weight, mean, rstd = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x_new = grad_x_new.contiguous()
        if ctx.use_cute and grad_out.dtype == grad_x_new.dtype == torch.bfloat16:
            grad_input, grad_weight = _ln_bwd_dyn_kernel(
                grad_out,
                x_new,
                mean,
                rstd,
                weight,
                inplace=False,
                grad_residual=grad_x_new,
            )
        else:
            mean_a = mean.reshape(*x_new.shape[:-1], 1)
            rstd_a = rstd.reshape(*x_new.shape[:-1], 1)
            grad_input, grad_weight, _ = (
                torch.ops.aten.native_layer_norm_backward.default(
                    grad_out,
                    x_new,
                    ctx.normalized_shape,
                    mean_a,
                    rstd_a,
                    weight,
                    None,
                    [True, True, False],
                )
            )
            grad_input = grad_input + grad_x_new
        return grad_input, grad_input, grad_weight, None


def fused_layer_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm that routes through autograd only when a gradient is live.

    fp32 master weights under bf16 autocast must still go through the `custom_fwd`
    cast wrapper even in a no-grad pass, so x and gamma are cast to the same dtype
    (the raw kernel requires equal dtypes).

    The inference path (no grad, no autocast) uses a Triton kernel whose dispatch
    overhead is ~7 µs — lower than the CuTe DSL variant's ~15 µs from_dlpack
    wrapping. Training stays on CuTe DSL for the backward stats path.
    """
    if torch.is_autocast_enabled("cuda") or (
        torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
    ):
        return _LayerNormFn.apply(x, weight, eps)
    from packed_encoders._kernels.triton_layer_norm import layer_norm as _triton_ln

    return _triton_ln(x.detach(), weight.detach(), eps)


# ---------------------------------------------------------------------------
# RoPE (split-half / concat convention, matching HF ModernBERT)
# ---------------------------------------------------------------------------


class _ApplyRopeFn(torch.autograd.Function):
    """RoPE is linear, so backward is the same kernel with sin negated."""

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
    """Forward: `gelu(a) * gate` in one kernel. Backward: one fused kernel recomputes
    gelu and writes (da, dgate) back into proj's storage."""

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


def fused_geglu(proj, *, out_dtype=None):
    if torch.is_grad_enabled() and proj.requires_grad:
        return _GegluFn.apply(proj, out_dtype)
    from packed_encoders._kernels.triton_layer_norm import geglu as _triton_geglu

    return _triton_geglu(proj.detach())


# ---------------------------------------------------------------------------
# Composite layer ops (cuBLAS GEMMs + the tails above)
# ---------------------------------------------------------------------------

_TRITON_MM_MAX_ROWS = 4096


def _linear(x, weight):
    n = x.shape[0] if x.ndim == 2 else x.shape[0] * x.shape[1]
    if (
        not torch.is_grad_enabled()
        and n <= _TRITON_MM_MAX_ROWS
        and x.dtype == weight.dtype
    ):
        from packed_encoders._kernels.matmul import triton_linear

        return triton_linear(x, weight)
    return F.linear(x, weight, None)


def fused_add_layer_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``x_new = x + residual; y = LN(x_new)`` → ``(x_new, y)``."""
    if torch.is_autocast_enabled("cuda") or (
        torch.is_grad_enabled()
        and (x.requires_grad or residual.requires_grad or weight.requires_grad)
    ):
        return _AddLayerNormFn.apply(x, residual, weight, eps)
    from packed_encoders._kernels.triton_layer_norm import add_layer_norm

    return add_layer_norm(x.detach(), residual.detach(), weight.detach(), eps)


def geglu_mlp(x, w_in, w_out):
    proj = _linear(x, w_in)
    return _linear(fused_geglu(proj), w_out)


# ---------------------------------------------------------------------------
# Attention (vendor SDPA — no FlashAttention kernel in this package)
# ---------------------------------------------------------------------------


def sdpa_attention(q, k, v, additive_mask, scaling):
    """q/k/v are strided [B, H, S, D] transpose views; SDPA reads them with no copy.
    `additive_mask` must already be SDPA-ready (finite, dtype-matched, or None)."""
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


_flash_attn = None  # cached (func, varlen_func, kind), kind in {"cute", "compiled"}


def _import_cute_flash():
    from flash_attn.cute import (  # CuteDSL FA4 (sm_90/100/110)
        flash_attn_func,
        flash_attn_varlen_func,
    )

    return flash_attn_func, flash_attn_varlen_func, "cute"


def _import_compiled_flash():
    from flash_attn import (  # compiled FA2/FA3 (incl. sm_120 / PTX)
        flash_attn_func,
        flash_attn_varlen_func,
    )

    return flash_attn_func, flash_attn_varlen_func, "compiled"


def _load_flash_attn():
    """Resolve and cache the flash kernel for this GPU, lazily (so the default sdpa
    path stays dependency-free). Arch-dependent: cute FA4 on sm_90/sm_100 (no sm_120
    kernel — no tcgen05), compiled flash-attn FA2 on sm_120. Tries the arch-appropriate
    one first, then the other; raises only if neither imports. `kind` selects the call
    ABI (cute and compiled funcs differ)."""
    global _flash_attn
    if _flash_attn is not None:
        return _flash_attn
    major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    if major == 12:
        order = [_import_compiled_flash]  # cute FA4 has no sm_120 kernel
    elif major in (9, 10, 11):
        order = [_import_cute_flash, _import_compiled_flash]
    else:
        order = [_import_compiled_flash, _import_cute_flash]
    errors = []
    for importer in order:
        try:
            _flash_attn = importer()
            return _flash_attn
        except ImportError as exc:  # pragma: no cover - import-time guard
            errors.append(f"{importer.__name__}: {exc}")
    raise ImportError(
        "attention_backend='flash'/'auto' needs a FlashAttention kernel: "
        "flash-attn-4 on sm_90/sm_100 (pip install flash-attn-4) or flash-attn on "
        "sm_120 (pip install flash-attn). The default 'sdpa' backend needs neither. "
        "Tried: " + "; ".join(errors)
    )


def flash_attention(q, k, v, window, scaling):
    """FlashAttention with the layer's window instead of a dense mask: `(-1, -1)` full
    (global layers), `(w, w)` the symmetric sliding band `|i-j| <= w` (local layers),
    which FA prunes rather than computing the full `S×S` — the long-S win. Assumes an
    unpadded batch (padding goes through `flash_attention_varlen`).

    q/k/v arrive as [B, H, S, D] transpose views; the `.transpose(1, 2)` to FA's
    [B, S, H, D] is a free view (only the head dim must be contiguous, which it is).
    The cute (FA4) and compiled (FA2) ABIs differ: compiled takes `dropout_p` and
    spells full attention `(-1, -1)`; cute takes neither, spells it `(None, None)`,
    and returns `(out, lse)`.
    """
    func, _varlen, kind = _load_flash_attn()
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if kind == "cute":
        w = (None, None) if window == (-1, -1) else window
        out = func(qt, kt, vt, softmax_scale=scaling, causal=False, window_size=w)
        if isinstance(out, (tuple, list)):
            out = out[0]
    else:
        out = func(
            qt, kt, vt, dropout_p=0.0, softmax_scale=scaling,
            causal=False, window_size=window,
        )
    b, s = out.shape[:2]
    return out.reshape(b, s, -1)


def flash_attention_varlen(q, k, v, cu_seqlens, max_seqlen, window, scaling):
    """FlashAttention over a packed, variable-length batch (padding stripped, see
    `forward._unpad`). q/k/v arrive as `[1, H, total, D]` views; `flash_attn_varlen_func`
    wants `[total, H, D]` plus `cu_seqlens` so it confines attention within each
    sequence (no cross-document leak, no `S×S` mask). `window` is the same per-layer
    structure as the dense path. Returns `[1, total, hidden]` so the layer code is
    identical to the dense path."""
    _dense, func, kind = _load_flash_attn()
    qt = q.squeeze(0).transpose(0, 1)  # [1,H,total,D] -> [total,H,D] (view)
    kt = k.squeeze(0).transpose(0, 1)
    vt = v.squeeze(0).transpose(0, 1)
    if kind == "cute":
        w = (None, None) if window == (-1, -1) else window
        out = func(
            qt, kt, vt,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            softmax_scale=scaling, causal=False, window_size=w,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
    else:
        out = func(
            qt, kt, vt,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            dropout_p=0.0, softmax_scale=scaling, causal=False, window_size=window,
        )
    total = out.shape[0]
    return out.reshape(1, total, -1)


def use_bshd_rope_flash(backend, h, d, cu_seqlens) -> bool:
    """Whether to take the fused BSHD rope+flash path: the flash backend, a supported
    `(H, D)` (warp-packable), and **no autograd**. The no-grad gate is a *measured*
    dispatch decision, not laziness: BSHD is a clean inference win (bit-identical,
    1.05–1.08× fwd, growing with S), but in training (fwd+bwd) it is a wash-to-regression
    — a BSHD RoPE backward must scatter a full `[N,3HD]` grad and have autograd sum it with
    the attention backward's grad on the v slice, which costs more than the transpose
    path's single cat (5090: 0.81–1.01× fwd+bwd). So
    training keeps the transpose+rope path (where sequence packing already dominates), and
    BSHD is wired where it actually pays. Eager `"auto"` is resolved to an explicit
    backend before this point; an unresolved capture-safe `"auto"` stays off this path.
    `_bshd_applicable` is a pure capability check (unsupported shapes fall back to
    transpose+rope), not a toggle."""
    if torch.is_grad_enabled() or not _bshd_applicable(h, d):
        return False
    if backend == "flash":
        return True
    return False


def flash_attention_qkv_bshd(
    qkv, h, d, *, cos, sin, window, scaling, cu_seqlens=None, max_seqlen=None
):
    """Fused BSHD RoPE + FlashAttention reading the packed `qkv` `[.., S, 3*H*D]`
    directly. RoPE'd q/k come out contiguous `[.., S, H, D]` and v is the qkv slice (D
    contiguous) — all in flash's native layout, so there is **no transpose to [B,H,S,D]
    and no `.contiguous()` copy** (the transpose path pays both: q/k strided→contiguous,
    then flash transposes back). `cu_seqlens` selects the packed varlen kernel. Returns
    `[.., S, hidden]`. No-grad only (see `use_bshd_rope_flash` — training is a wash)."""
    q, k = _rope_bshd_kernel(qkv, h, d, cos, sin)            # contiguous [.., S, H, D]
    v = qkv.view(*qkv.shape[:-1], 3, h, d)[..., 2, :, :]     # [.., S, H, D] (D contiguous)
    func, varlen, kind = _load_flash_attn()
    cute = kind == "cute"
    w = ((None, None) if window == (-1, -1) else window) if cute else window
    if cu_seqlens is not None:
        qf, kf, vf = q.squeeze(0), k.squeeze(0), v.squeeze(0)  # [total, H, D]
        kw = dict(cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                  max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                  softmax_scale=scaling, causal=False, window_size=w)
        out = varlen(qf, kf, vf, **(kw if cute else {**kw, "dropout_p": 0.0}))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.reshape(1, out.shape[0], -1)
    kw = dict(softmax_scale=scaling, causal=False, window_size=w)
    out = func(q, k, v, **(kw if cute else {**kw, "dropout_p": 0.0}))
    if isinstance(out, (tuple, list)):
        out = out[0]
    b, s = out.shape[:2]
    return out.reshape(b, s, -1)


def attention(q, k, v, *, mask, window, scaling, backend, cu_seqlens=None, max_seqlen=None):
    """Dispatch attention to the selected backend (`mask` feeds sdpa, `window` feeds
    flash). `cu_seqlens` routes flash to the varlen path. Eager `"auto"` resolves
    earlier from the distribution-aware score. An unresolved leaf-level `"auto"`
    means a CUDA-graph/capture path without host-visible sequence statistics and stays
    on SDPA; it never dispatches from sequence length.
    `pack()` downgrades `"auto"` to `"sdpa"` up front if no kernel is loadable, so
    this never imports mid-capture."""
    if backend == "auto":
        backend = "sdpa"
    if backend == "flash":
        if cu_seqlens is not None:
            return flash_attention_varlen(q, k, v, cu_seqlens, max_seqlen, window, scaling)
        return flash_attention(q, k, v, window, scaling)
    return sdpa_attention(q, k, v, mask, scaling)
