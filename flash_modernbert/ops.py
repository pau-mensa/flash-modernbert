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


# Sequence-length threshold for the "auto" backend: at/above it flash beats sdpa,
# below it flash's fixed-cost floor loses. The crossover is **arch-dependent**, so
# this is keyed on compute capability rather than one global line. The values are the
# measured **end-to-end** crossovers (full encoder forward, not attention in
# isolation) — `benchmarks/{varlen_bench.py (5090), results/varlen_b200_h200.json}`:
#
#   sm_120 (5090, compiled FA2): 256  — ties ~S256, wins by S512 (B32). Low FA2 floor.
#   sm_100 (B200, cute FA4):     1024 — loses at S512 (0.80×), wins at S1024 (1.10×).
#   sm_90  (H200, cute FA4):     2048 — loses through S1024 (0.83×); crossover >1024.
#
# Two caveats baked into these (both push the threshold UP vs the attention-only
# `attn_backend_micro.py` crossover, which gave an over-optimistic 128/512/1024):
#   (1) End-to-end ≠ attention-only. The non-attention ops (GEMMs/LN/GeGLU) and the
#       varlen unpad/repad + per-layer kernel launches dilute and offset flash's
#       attention win, so flash needs a larger S to come out ahead than the isolated
#       attention op does — especially on the high-fixed-cost cute-FA4 datacenter arches.
#   (2) Batch-dependent. These were measured at small batch (B=8 on B200/H200), where
#       those very fast GPUs are *launch-bound* (sdpa time is ~flat in S) and flash's
#       per-layer launch overhead dominates. At larger B (compute-bound) the crossover
#       drops. So this S-only table is a conservative floor; a token-budget (B·S)
#       threshold would capture the large-B mid-S wins it leaves on the table.
# H200's 2048 is conservative — measured a loss through S1024, did not measure S2048.
# Unknown arch / no CUDA → 1024.
_FLASH_MIN_SEQ_BY_CC = {
    (12, 0): 256,    # sm_120 — consumer Blackwell (5090), compiled FA2
    (10, 0): 1024,   # sm_100 — datacenter Blackwell (B200), cute FA4
    (9, 0): 2048,    # sm_90  — Hopper (H200), cute FA4: strong cuDNN sdpa + high FA4 floor
}
_FLASH_MIN_SEQ_DEFAULT = 1024


def _resolve_flash_min_seq() -> int:
    """The per-GPU `"auto"` flash threshold (see `_FLASH_MIN_SEQ_BY_CC`). Guarded so
    a CPU-only import (e.g. the pure-logic tests) gets the safe 1024 fallback rather
    than touching CUDA."""
    try:
        if not torch.cuda.is_available():
            return _FLASH_MIN_SEQ_DEFAULT
        cc = torch.cuda.get_device_capability()
    except Exception:  # pragma: no cover - defensive: no/duff CUDA
        return _FLASH_MIN_SEQ_DEFAULT
    return _FLASH_MIN_SEQ_BY_CC.get(cc, _FLASH_MIN_SEQ_DEFAULT)


# Resolved once for this process's GPU (the device is fixed per process). Read by
# both dispatch sites (`attention` here, `forward._resolve_eager_backend`).
FLASH_MIN_SEQ = _resolve_flash_min_seq()


_flash_attn = None  # cached (func, varlen_func, kind) where kind in {"cute", "compiled"}


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
    """Resolve the flash kernel for *this* GPU, lazily (so the default sdpa path
    stays dependency-free) and cached. The right kernel is arch-dependent:

    - sm_90 / sm_100 (Hopper / datacenter Blackwell): CuteDSL **FA4**
      (`pip install flash-attn-4`), the per-arch SOTA, built on wgmma / tcgen05.
    - sm_120 (consumer Blackwell): FA4-cute has no kernel (no tcgen05), so the
      **compiled** flash-attn FA2 path (`pip install flash-attn`, runs from PTX).

    Tries the arch-appropriate kernel first, then the other, and raises only if
    neither imports. Returns `(func, varlen_func, kind)`; `kind` selects the call
    ABI (the cute and compiled funcs differ); `varlen_func` is the packed
    padded-batch entry point (`flash_attention_varlen`)."""
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
    """FlashAttention with the layer's *structure* (window) instead of a dense
    mask: full attention (global layers) is `window=(-1, -1)`, and `window=(w, w)`
    is a symmetric sliding band `|i - j| <= w` (local layers), which FA prunes to
    the band rather than computing the full `S×S` — the long-S win.

    q/k/v arrive as the [B, H, S, D] transpose views the rest of the tail uses;
    FA wants [B, S, H, D]. The `.transpose(1, 2)` back is a free view — and FA
    only requires the head dim (last) to be contiguous, which it is (RoPE outputs
    contiguous [B, H, S, D], and v's unbind view keeps D packed), so we pass the
    strided views directly with no copy. Assumes an unpadded batch — padding would
    need `flash_attn_varlen_func` with cu_seqlens; sdpa remains the path for that.

    The cute (FA4) and compiled (FA2) `flash_attn_func` differ in ABI: compiled
    takes `dropout_p` and spells full attention `(-1, -1)`; cute takes neither,
    spells it `(None, None)`, and returns `(out, lse)`. `_load_flash_attn` reports
    which kind loaded, and we adapt the call accordingly.
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
    """FlashAttention over a **packed, variable-length** batch — the path for
    padded document batches (different lengths padded to a common `S`).

    The encoder runs the whole forward as a single `b=1` sequence of `total`
    concatenated real tokens (padding already stripped, see `forward._unpad`), so
    q/k/v arrive as the `[1, H, total, D]` transpose views the rest of the tail
    uses. `flash_attn_varlen_func` wants `[total, H, D]` plus `cu_seqlens` (the
    per-sequence boundaries) so it confines attention to within each sequence — no
    cross-document leakage, and no `S×S` mask. `.squeeze(0).transpose(0, 1)` is a
    free view; FA only needs the head dim (last) contiguous, which holds.

    `window` is the same per-layer structure as the dense path: `(-1, -1)` full
    (global layers), `(half, half)` the symmetric sliding band (local layers).
    Under varlen the band is measured *within* each sequence (FA aligns it to
    each sub-sequence's own positions), exactly the local-attention semantics.

    Returns `[1, total, hidden]` to match the dense attention's `[b, s, hidden]`
    so the surrounding layer code (residual add, Wo) is identical."""
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


def attention(q, k, v, *, mask, window, scaling, backend, cu_seqlens=None, max_seqlen=None):
    """Dispatch attention to the selected backend. `mask` feeds sdpa; `window`
    feeds flash; both are precomputed per layer so this stays a thin switch.

    When `cu_seqlens` is supplied (the packed varlen path, decided once at the
    forward level — see `forward._varlen_forward`), the flash backend routes to
    `flash_attention_varlen`; the dense flash/sdpa kernels are for the
    rectangular `[B, S]` path.

    `"auto"` resolves per call from `q`'s sequence length: flash for `S >=
    FLASH_MIN_SEQ` (long-S docs, where windowed pruning wins on every arch) and
    sdpa below (short-S queries, where flash is pure fixed-cost overhead and sdpa
    wins or ties). Resolving here — at the leaf, where S is known — means a
    captured graph bakes in the right path per `(B, S)` bucket. `"auto"` assumes a
    flash kernel is loadable; `prepare()` downgrades it to `"sdpa"` up front when
    none is, so this never imports mid-capture."""
    if backend == "auto":
        backend = "flash" if q.shape[2] >= FLASH_MIN_SEQ else "sdpa"
    if backend == "flash":
        if cu_seqlens is not None:
            return flash_attention_varlen(q, k, v, cu_seqlens, max_seqlen, window, scaling)
        return flash_attention(q, k, v, window, scaling)
    return sdpa_attention(q, k, v, mask, scaling)
