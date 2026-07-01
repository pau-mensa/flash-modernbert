"""Autotuned Triton matmul for shapes where cuBLAS picks a suboptimal kernel.

cuBLAS's heuristic selects a generic CUTLASS wmma kernel at small M (total
tokens ≤ ~4096) for the [M, 768, 2304] shapes that dominate the encoder's
QKV and gated-MLP projections, running 1.5–1.9x slower than an autotuned
Triton kernel on sm_120. At large M (≥ 8192) cuBLAS wins and F.linear should
be used directly.

The autotune grid is kept small (five configs, all ≤ 99 KB shared memory on
sm_120) so the first-call warm-up is <0.5 s. The grid key includes M so each
distinct token count autotuned at most once.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=2, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=3, num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=3, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=3, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=3, num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def triton_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b with autotuned Triton. a: [M, K], b: [K, N] → [M, N]."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner dim mismatch: {K} vs {K2}"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def triton_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Drop-in for F.linear(x, weight): x @ weight.T."""
    if x.ndim > 2:
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        out_2d = triton_mm(x_2d, weight.t())
        return out_2d.reshape(*shape[:-1], out_2d.shape[-1])
    return triton_mm(x, weight.t())
