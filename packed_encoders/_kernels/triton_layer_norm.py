"""Triton LayerNorm forward — inference-only replacement for the CuTe DSL
variant when CuTe dispatch overhead dominates (small M, large H).

The CuTe DSL GPU kernel beats ATen (4.2 µs vs 6.8 µs at M=2048 H=768),
but each CuTe call pays ~10 µs of from_dlpack + stream wrapping that
PyTorch's C++ dispatch avoids. Triton sits in between: ~1 µs dispatch,
same GPU kernel quality. This kernel exists so the inference path gets
the fast kernel without the slow wrapper.

Training (backward, autocast stats) stays on CuTe DSL — the backward's
persistent-CTA layout and grad_gamma partials are more complex than a
simple Triton kernel, and the training path is not launch-bound.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ln_fwd_kernel(
    X, W, Y,
    stride_x,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    eps,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / H
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / H
    inv_std = tl.math.rsqrt(var + eps)

    y = (xc * inv_std * w).to(Y.dtype.element_ty)
    tl.store(Y + row * stride_x + cols, y, mask=mask)


@triton.jit
def _add_ln_fwd_kernel(
    X, RESIDUAL, W, Y, X_OUT,
    stride_x,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    eps,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(RESIDUAL + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    x_new = x + res
    tl.store(X_OUT + row * stride_x + cols, x_new.to(X.dtype.element_ty), mask=mask)

    mean = tl.sum(x_new, axis=0) / H
    xc = x_new - mean
    var = tl.sum(xc * xc, axis=0) / H
    inv_std = tl.math.rsqrt(var + eps)

    y = (xc * inv_std * w).to(Y.dtype.element_ty)
    tl.store(Y + row * stride_x + cols, y, mask=mask)


@triton.jit
def _geglu_fwd_kernel(
    PROJ, OUT,
    stride_proj, stride_out,
    I: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_I)
    mask = cols < I
    INV_SQRT2: tl.constexpr = 0.7071067811865476

    a = tl.load(PROJ + row * stride_proj + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(PROJ + row * stride_proj + I + cols, mask=mask, other=0.0).to(tl.float32)

    gelu_a = a * 0.5 * (1.0 + tl.math.erf(a * INV_SQRT2))
    out = (gelu_a * gate).to(OUT.dtype.element_ty)
    tl.store(OUT + row * stride_out + cols, out, mask=mask)


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    assert x.shape[-1] == weight.shape[0]
    h = weight.shape[0]
    flat = x.reshape(-1, h)
    out = torch.empty_like(flat)
    n_rows = flat.shape[0]
    block_h = triton.next_power_of_2(h)
    _ln_fwd_kernel[(n_rows,)](
        flat, weight, out,
        flat.stride(0),
        H=h,
        BLOCK_H=block_h,
        eps=eps,
    )
    return out.view(x.shape)


def add_layer_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add + LayerNorm: ``x_new = x + residual; y = LN(x_new)``.

    Returns ``(x_new, y)`` — the updated residual stream and the normalized
    output, in one kernel instead of two.
    """
    h = weight.shape[0]
    flat_x = x.reshape(-1, h)
    flat_res = residual.reshape(-1, h)
    x_out = torch.empty_like(flat_x)
    y = torch.empty_like(flat_x)
    n_rows = flat_x.shape[0]
    block_h = triton.next_power_of_2(h)
    _add_ln_fwd_kernel[(n_rows,)](
        flat_x, flat_res, weight, y, x_out,
        flat_x.stride(0),
        H=h,
        BLOCK_H=block_h,
        eps=eps,
    )
    return x_out.view(x.shape), y.view(x.shape)


def geglu(proj: torch.Tensor) -> torch.Tensor:
    """Triton GeGLU: ``gelu(a) * gate`` from concatenated ``[..., 2*I]`` input."""
    assert proj.shape[-1] % 2 == 0
    intermediate = proj.shape[-1] // 2
    flat = proj.reshape(-1, proj.shape[-1])
    out = torch.empty(flat.shape[0], intermediate, dtype=proj.dtype, device=proj.device)
    n_rows = flat.shape[0]
    block_i = triton.next_power_of_2(intermediate)
    _geglu_fwd_kernel[(n_rows,)](
        flat, out,
        flat.stride(0), out.stride(0),
        I=intermediate,
        BLOCK_I=block_i,
    )
    return out.view(*proj.shape[:-1], intermediate)
