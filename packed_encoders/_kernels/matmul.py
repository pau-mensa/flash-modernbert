"""Fixed-dispatch Triton matmul for small ModernBERT inference shapes.

At the ``M=256..4096`` token counts that dominate the short-batch path, cuBLAS
selects a suboptimal tile on sm_120 (RTX 5090) for ModernBERT's projections and
a hand-picked Triton config is faster.  On all other measured architectures —
sm_80 (A100), sm_89 (L40S), sm_100 (B200) — cuBLAS wins at every shape and M
by 1.5-2.3x, so the Triton path is sm_120-only.

Cross-arch data: ``benchmarks/results/matmul_{a100,l40s,b200}.json``, generated
by ``modal run scripts/modal_matmul_bench.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class _Config:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int
    group_m: int


_M16_N128_K64_W4 = _Config(16, 128, 64, 3, 4, 8)
_M16_N128_K64_W8 = _Config(16, 128, 64, 3, 8, 8)
_M32_N128_K64 = _Config(32, 128, 64, 3, 8, 8)
_M64_N64_K64 = _Config(64, 64, 64, 3, 8, 1)
_M64_N128_K64 = _Config(64, 128, 64, 3, 8, 8)

_FALLBACK_CONFIG = _M64_N64_K64


def _pick_config(m: int, n: int, k: int, element_size: int = 2) -> _Config:
    """Return the fixed sm_120 winner for a row-count bucket and projection."""
    if element_size != 2:
        return _FALLBACK_CONFIG

    if (n, k) == (2304, 768):
        if m <= 128:
            return _M16_N128_K64_W4
        if m <= 512:
            return _M64_N64_K64
        if m <= 1024:
            return _M64_N128_K64
        if m <= 2048:
            return _M64_N64_K64
        return _M64_N128_K64

    if (n, k) == (768, 768):
        if m <= 32:
            return _M16_N128_K64_W8
        if m <= 128:
            return _M16_N128_K64_W4
        if m <= 256:
            return _M16_N128_K64_W8
        if m <= 512:
            return _M64_N64_K64
        if m <= 1024:
            return _M64_N128_K64
        return _M64_N64_K64

    if (n, k) == (768, 1152):
        if m <= 256:
            return _M16_N128_K64_W8
        if m <= 512:
            return _M32_N128_K64
        if m <= 1024:
            return _M64_N128_K64
        return _M64_N64_K64

    return _FALLBACK_CONFIG


def _is_sm120() -> bool:
    return torch.cuda.get_device_capability() == (12, 0)


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    num_n_blocks = tl.cdiv(N, BLOCK_N)

    programs_per_group = GROUP_M * num_n_blocks
    group_id = pid // programs_per_group
    first_m = group_id * GROUP_M
    group_m = tl.minimum(num_m_blocks - first_m, GROUP_M)
    pid_m = first_m + (pid % programs_per_group) % group_m
    pid_n = (pid % programs_per_group) // group_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    mask_m = offs_m[:, None] < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K
        a = tl.load(a_ptrs, mask=mask_m & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def triton_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute ``a @ b`` via Triton on sm_120, cuBLAS everywhere else."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner dim mismatch: {K} vs {K2}"
    if M == 0:
        return torch.empty((M, N), device=a.device, dtype=a.dtype)

    if not _is_sm120():
        return torch.mm(a, b)

    config = _pick_config(M, N, K, a.element_size())
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (
        triton.cdiv(M, config.block_m) * triton.cdiv(N, config.block_n),
    )
    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        GROUP_M=config.group_m,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
    )
    return c


def triton_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Drop-in for ``F.linear(x, weight)`` without bias."""
    if x.ndim > 2:
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        out_2d = triton_mm(x_2d, weight.t())
        return out_2d.reshape(*shape[:-1], out_2d.shape[-1])
    return triton_mm(x, weight.t())
