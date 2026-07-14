"""Unified residual-add + LayerNorm kernel and autograd contract."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="residual LayerNorm kernels require CUDA"
)


@needs_cuda
@pytest.mark.parametrize("with_stats", [False, True])
def test_cute_add_layer_norm_preserves_bf16_rounding(with_stats):
    from packed_encoders._kernels.layer_norm import (
        add_layer_norm,
        add_layer_norm_with_stats,
    )

    torch.manual_seed(0)
    x = torch.randn(17, 768, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(768, device="cuda", dtype=torch.bfloat16)

    expected_x = x + residual
    expected_y = F.layer_norm(expected_x, (768,), weight, None, 1e-5)
    if with_stats:
        x_new, y, mean, rstd = add_layer_norm_with_stats(
            x, residual, weight, 1e-5
        )
        assert mean.shape == rstd.shape == (17,)
        assert mean.dtype == rstd.dtype == torch.float32
    else:
        x_new, y = add_layer_norm(x, residual, weight, 1e-5)

    assert torch.equal(x_new, expected_x)
    assert torch.equal(y, expected_y)


@needs_cuda
def test_fused_add_layer_norm_two_output_backward_matches_eager():
    from packed_encoders.ops import fused_add_layer_norm

    torch.manual_seed(1)
    shape = (19, 768)
    x0 = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    residual0 = torch.randn_like(x0)
    weight0 = torch.randn(768, device="cuda", dtype=torch.bfloat16)
    grad_x_new = torch.randn_like(x0)
    grad_y = torch.randn_like(x0)

    x = x0.detach().requires_grad_()
    residual = residual0.detach().requires_grad_()
    weight = weight0.detach().requires_grad_()
    x_new, y = fused_add_layer_norm(x, residual, weight, 1e-5)
    actual = torch.autograd.grad(
        (x_new, y), (x, residual, weight), (grad_x_new, grad_y)
    )

    x_ref = x0.detach().clone().requires_grad_()
    residual_ref = residual0.detach().clone().requires_grad_()
    weight_ref = weight0.detach().clone().requires_grad_()
    x_new_ref = x_ref + residual_ref
    y_ref = F.layer_norm(x_new_ref, (768,), weight_ref, None, 1e-5)
    expected = torch.autograd.grad(
        (x_new_ref, y_ref),
        (x_ref, residual_ref, weight_ref),
        (grad_x_new, grad_y),
    )

    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        assert torch.equal(actual_grad, expected_grad)
