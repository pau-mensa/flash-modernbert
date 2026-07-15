from __future__ import annotations

import pytest
import torch

from packed_encoders._kernels.matmul import (
    _FALLBACK_CONFIG,
    _M16_N128_K64_W4,
    _M16_N128_K64_W8,
    _M32_N128_K64,
    _M64_N64_K64,
    _M64_N128_K64,
    _pick_config,
    triton_linear,
)


@pytest.mark.parametrize(
    ("m", "n", "k", "expected"),
    [
        (32, 2304, 768, _M16_N128_K64_W4),
        (128, 2304, 768, _M16_N128_K64_W4),
        (256, 2304, 768, _M64_N64_K64),
        (1024, 2304, 768, _M64_N128_K64),
        (2048, 2304, 768, _M64_N64_K64),
        (32, 768, 768, _M16_N128_K64_W8),
        (128, 768, 768, _M16_N128_K64_W4),
        (256, 768, 768, _M16_N128_K64_W8),
        (1024, 768, 768, _M64_N128_K64),
        (256, 768, 1152, _M16_N128_K64_W8),
        (512, 768, 1152, _M32_N128_K64),
        (1024, 768, 1152, _M64_N128_K64),
        (256, 1024, 1024, _FALLBACK_CONFIG),
    ],
)
def test_pick_config(m, n, k, expected):
    assert _pick_config(m, n, k) == expected


def test_fp32_uses_shared_memory_safe_fallback():
    assert _pick_config(32, 768, 1152, element_size=4) == _FALLBACK_CONFIG


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(32, 2304, 768), (128, 768, 768), (512, 768, 1152), (1024, 2304, 768)],
)
def test_triton_linear_matches_torch(m, n, k):
    torch.manual_seed(0)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x, weight)
    actual = triton_linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_linear_preserves_leading_dimensions_and_empty_rows():
    x = torch.randn((2, 16, 768), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((768, 768), device="cuda", dtype=torch.bfloat16)
    assert triton_linear(x, weight).shape == (2, 16, 768)

    empty = torch.empty((0, 768), device="cuda", dtype=torch.bfloat16)
    assert triton_linear(empty, weight).shape == (0, 768)
