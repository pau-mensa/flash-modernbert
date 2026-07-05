"""RoPE (rotary position embedding) forward kernel in CuteDSL.

Drop-in for `op_apply_rope` in `reference.py`. For each (b, h, s, d):

    y[..., d_low]  = x[..., d_low]  * cos[s, d_low]  - x[..., d_high] * sin[s, d_low]
    y[..., d_high] = x[..., d_high] * cos[s, d_high] + x[..., d_low]  * sin[s, d_high]

where d_low ∈ [0, D/2) and d_high = d_low + D/2. This is the "half-rotation"
RoPE (concat-half), matching HuggingFace's ModernBert. Compute is fp32, I/O bf16.

**Design — coalesced shuffle.** A naive kernel (thread `t` owns the pair
`(t, t+D/2)`, reading the two halves as separate strided streams) stalls at ~12% HBM
peak: packing short D=64 rows into a warp makes each load's footprint non-contiguous.
The fix keeps global I/O contiguous and resolves the low↔high pairing with an
intra-warp butterfly shuffle instead of a second memory stream:

- Flatten q/k to `[B*H, S, D]`; a block owns R consecutive `s` rows of one (b,h),
  contiguous in memory.
- Each thread owns one `vec=8` chunk and loads only its 8 elements, so the 32 lanes
  read 256 contiguous elements per load — coalesced.
- The rotation pairs chunk `c` with chunk `c ^ (D/2/vec)`; the partner's data arrives
  via `shuffle_sync_bfly`, no second load. Low chunks subtract `partner*sin`, high add.
- q and k share cos/sin and the shuffle setup in one kernel.

Lands at ~29% peak / 2.34× the naive kernel; it's latency-bound (D=64 rows are short),
so splitting q/k or widening to vec=16 both regress. cos/sin are `[S, D]` (the broadcast
batch dim is sliced upstream — identical across batch).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from packed_encoders._kernels._compile_cache import current_cute_stream, get_compiled


VEC = 8                  # 128-bit (8×bf16) coalesced loads/stores
ROWS_PER_BLOCK = 8       # best on B200 sweep; smaller blocks win (latency-bound)
WARP_SIZE = 32


def _shfl_applicable(d: int) -> bool:
    """The shuffle path needs vec-aligned chunks whose count is a power of two
    ≤ 32 so each row's threads sit inside one warp and the partner lane
    (tidx ^ low_chunks) never crosses a warp boundary. True for mmBERT D=64."""
    if d % VEC != 0:
        return False
    chunks = d // VEC
    return chunks <= 32 and (chunks & (chunks - 1)) == 0


def _build_shfl_for_d(d: int):
    """Coalesced-shuffle kernel for a fixed head_dim D (see module docstring)."""
    half_d = d // 2
    chunks_per_row = d // VEC           # 8 for D=64
    low_chunks = half_d // VEC          # 4  -> partner lane = tidx ^ low_chunks
    threads = chunks_per_row * ROWS_PER_BLOCK

    @cute.kernel
    def kernel(
        gq: cute.Tensor,    # [B*H, S, D]
        gk: cute.Tensor,
        gcos: cute.Tensor,  # [S, D]
        gsin: cute.Tensor,
        gq_out: cute.Tensor,
        gk_out: cute.Tensor,
    ):
        bh, sb, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        dt = gq.element_type
        s_size = cute.size(gq, mode=[1])
        chunk = tidx % chunks_per_row
        row_local = tidx // chunks_per_row
        s = sb * ROWS_PER_BLOCK + row_local
        if s < s_size:
            d0 = chunk * VEC
            i_base = cute.crd2idx((bh, s, d0), gq.layout)
            i_c = cute.crd2idx((s, d0), gcos.layout)
            cv = cute.make_rmem_tensor((VEC,), dt)
            sv = cute.make_rmem_tensor((VEC,), dt)
            cute.autovec_copy(cute.make_tensor(gcos.iterator + i_c, cute.make_layout(VEC)), cv)
            cute.autovec_copy(cute.make_tensor(gsin.iterator + i_c, cute.make_layout(VEC)), sv)
            # low chunk: out = mine*cos - partner*sin ; high chunk: + partner*sin.
            sign = Float32(1.0)
            if chunk < low_chunks:
                sign = Float32(-1.0)
            for (gin, gout) in ((gq, gq_out), (gk, gk_out)):
                mine = cute.make_rmem_tensor((VEC,), dt)
                out = cute.make_rmem_tensor((VEC,), dt)
                cute.autovec_copy(
                    cute.make_tensor(gin.iterator + i_base, cute.make_layout(VEC)), mine)
                for j in cutlass.range_constexpr(VEC):
                    myf = mine[j].to(Float32)
                    pf = cute.arch.shuffle_sync_bfly(
                        myf, offset=low_chunks, mask=-1, mask_and_clamp=31)
                    out[j] = out.element_type(
                        myf * cv[j].to(Float32) + sign * pf * sv[j].to(Float32))
                cute.autovec_copy(
                    out, cute.make_tensor(gout.iterator + i_base, cute.make_layout(VEC)))

    @cute.jit
    def launch(
        q: cute.Tensor,
        k: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        q_out: cute.Tensor,
        k_out: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        bh_size = cute.size(q, mode=[0])
        s_size = cute.size(q, mode=[1])
        s_blocks = (s_size + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
        kernel(q, k, cos, sin, q_out, k_out).launch(
            grid=(bh_size, s_blocks, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch


def _bshd_applicable(h: int, d: int) -> bool:
    """The BSHD coalesced-shuffle path needs D vec-aligned with a power-of-two chunk
    count (the partner lane stays inside D via `^ low_chunks`) AND H a multiple of the
    heads packed per warp, so every head's chunks sit inside one warp. True for the
    ModernBERT D=64 / H∈{12,16} family."""
    if not _shfl_applicable(d):
        return False
    chunks_per_head = d // VEC
    if WARP_SIZE % chunks_per_head != 0:
        return False
    heads_per_warp = WARP_SIZE // chunks_per_head
    return h % heads_per_warp == 0


def _build_bshd_for_hd(h: int, d: int):
    """Coalesced-shuffle RoPE that reads q/k straight out of the **packed qkv**
    `[N, 3*H*D]` (the fused-Wqkv GEMM output, contiguous) and writes contiguous
    `[N, H, D]` q_out/k_out — the flash path's layout, with no transpose to [B,H,S,D]
    and no `.contiguous()` copy (the [B*H,S,D] kernel needs q/k packed-contiguous, which
    a transpose-view is not). Fusing the q/k slice-extract in also drops the unbind.

    One block owns one token row `bs` (its 3HD run is contiguous → coalesced reads);
    `pos = bs % s_mod` indexes cos/sin (`s_mod = S` for a dense [B,S,..] batch, or the
    packed `total` for varlen, where rows map 1:1 to the gathered cos/sin so the modulo
    is identity). q lives at row offset 0, k at offset H*D. The low/high D pairing is
    resolved by an intra-warp butterfly shuffle as in the [B*H,S,D] kernel; heads are
    packed `WARP_SIZE//chunks` per warp so a head's partner lane never crosses a warp."""
    hd = h * d
    half_d = d // 2
    chunks_per_head = d // VEC          # 8 for D=64
    low_chunks = half_d // VEC          # 4 -> partner lane = tidx ^ low_chunks
    threads = h * chunks_per_head       # all heads of one row, e.g. 96 for H=12,D=64

    @cute.kernel
    def kernel(
        gqkv: cute.Tensor,  # [N, 3*H*D]  (N = B*S dense, or total varlen)
        gcos: cute.Tensor,  # [s_mod, D]
        gsin: cute.Tensor,
        gq_out: cute.Tensor,  # [N, H, D]
        gk_out: cute.Tensor,
        s_mod: cutlass.Int32,
    ):
        bs, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        dt = gqkv.element_type
        head = tidx // chunks_per_head
        chunk = tidx % chunks_per_head
        pos = bs % s_mod
        d0 = chunk * VEC
        row_in = cute.crd2idx((bs, head * d + d0), gqkv.layout)   # q at offset 0
        i_out = cute.crd2idx((bs, head, d0), gq_out.layout)
        i_c = cute.crd2idx((pos, d0), gcos.layout)
        cv = cute.make_rmem_tensor((VEC,), dt)
        sv = cute.make_rmem_tensor((VEC,), dt)
        cute.autovec_copy(cute.make_tensor(gcos.iterator + i_c, cute.make_layout(VEC)), cv)
        cute.autovec_copy(cute.make_tensor(gsin.iterator + i_c, cute.make_layout(VEC)), sv)
        sign = Float32(1.0)
        if chunk < low_chunks:
            sign = Float32(-1.0)
        for (in_off, gout) in ((0, gq_out), (hd, gk_out)):
            mine = cute.make_rmem_tensor((VEC,), dt)
            out = cute.make_rmem_tensor((VEC,), dt)
            cute.autovec_copy(
                cute.make_tensor(gqkv.iterator + row_in + in_off, cute.make_layout(VEC)), mine)
            for j in cutlass.range_constexpr(VEC):
                myf = mine[j].to(Float32)
                pf = cute.arch.shuffle_sync_bfly(
                    myf, offset=low_chunks, mask=-1, mask_and_clamp=WARP_SIZE - 1)
                out[j] = out.element_type(
                    myf * cv[j].to(Float32) + sign * pf * sv[j].to(Float32))
            cute.autovec_copy(
                out, cute.make_tensor(gout.iterator + i_out, cute.make_layout(VEC)))

    @cute.jit
    def launch(
        qkv: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        q_out: cute.Tensor,
        k_out: cute.Tensor,
        s_mod: cutlass.Int32,
        stream: cuda_driver.CUstream,
    ):
        n = cute.size(qkv, mode=[0])
        kernel(qkv, cos, sin, q_out, k_out, s_mod).launch(
            grid=(n, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch


def _build_scalar_for_d(d: int):
    """Scalar fallback for head_dims the shuffle path can't take (D not a
    power-of-two multiple of VEC). One block per (b,h,s); thread `t` owns the
    pair (t, t+D/2). ~12% peak — only used off the mmBERT/Nomic/NeoBERT D=64 path.
    """
    assert d % 2 == 0, f"head_dim D={d} must be even"
    half_d = d // 2

    @cute.kernel
    def kernel(
        gq: cute.Tensor,
        gk: cute.Tensor,
        gcos: cute.Tensor,
        gsin: cute.Tensor,
        gq_out: cute.Tensor,
        gk_out: cute.Tensor,
    ):
        b, h, s = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        d_low = tidx              # 0..half_d-1
        d_high = tidx + half_d    # half_d..d-1

        # Fetch cos/sin once, share between q and k (and pair halves).
        cos_low = gcos[s, d_low].to(Float32)
        cos_high = gcos[s, d_high].to(Float32)
        sin_low = gsin[s, d_low].to(Float32)
        sin_high = gsin[s, d_high].to(Float32)

        # ----- q -----
        q_low = gq[b, h, s, d_low].to(Float32)
        q_high = gq[b, h, s, d_high].to(Float32)
        gq_out[b, h, s, d_low] = gq_out.element_type(q_low * cos_low - q_high * sin_low)
        gq_out[b, h, s, d_high] = gq_out.element_type(q_high * cos_high + q_low * sin_high)

        # ----- k -----
        k_low = gk[b, h, s, d_low].to(Float32)
        k_high = gk[b, h, s, d_high].to(Float32)
        gk_out[b, h, s, d_low] = gk_out.element_type(k_low * cos_low - k_high * sin_low)
        gk_out[b, h, s, d_high] = gk_out.element_type(k_high * cos_high + k_low * sin_high)

    @cute.jit
    def launch(
        q: cute.Tensor,
        k: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        q_out: cute.Tensor,
        k_out: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        b_size = cute.size(q, mode=[0])
        h_size = cute.size(q, mode=[1])
        s_size = cute.size(q, mode=[2])
        kernel(q, k, cos, sin, q_out, k_out).launch(
            grid=(b_size, h_size, s_size),
            block=(half_d, 1, 1),
            stream=stream,
        )

    return launch


_compiled_for_d: dict[tuple[int, bool], object] = {}
_compiled_bshd: dict[tuple[int, int], object] = {}


def _get_jit(d: int, shfl: bool):
    key = (d, shfl)
    if key not in _compiled_for_d:
        _compiled_for_d[key] = _build_shfl_for_d(d) if shfl else _build_scalar_for_d(d)
    return _compiled_for_d[key]


def _get_jit_bshd(h: int, d: int):
    key = (h, d)
    if key not in _compiled_bshd:
        _compiled_bshd[key] = _build_bshd_for_hd(h, d)
    return _compiled_bshd[key]


def apply_rope_bshd(
    qkv: torch.Tensor,
    h: int,
    d: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE'd q, k extracted straight from the packed **qkv** `[..., 3*H*D]` (the fused
    Wqkv GEMM output) in the flash path's [..., H, D] layout — no transpose to [B,H,S,D]
    and no `.contiguous()` copy (the [B*H,S,D] kernel needs packed-contiguous q/k, which
    a transpose-view is not), and the q/k slice-extract is fused in.

    `qkv`: [..., 3*H*D], contiguous in the last dim (leading dims flattened to N rows).
    `cos`, `sin`: the position tables — dense `[S, D]` (or `[1, S, D]`), or the varlen
    per-token gathered `[total, D]`; `s_mod` (rows of cos) is derived, so dense uses
    `pos = bs % S` and varlen `pos = bs` (identity). Returns contiguous q, k of shape
    `qkv.shape[:-1] + (H, D)`. Requires `_bshd_applicable(H, D)`."""
    assert qkv.is_cuda and cos.is_cuda and sin.is_cuda
    assert qkv.dtype == cos.dtype == sin.dtype
    assert qkv.shape[-1] == 3 * h * d, f"qkv last dim {qkv.shape[-1]} != 3*{h}*{d}"
    assert _bshd_applicable(h, d), f"BSHD rope unsupported for H={h}, D={d}"
    if cos.ndim == 3:  # [1, s_mod, D] -> [s_mod, D]
        cos, sin = cos[0], sin[0]
    s_mod = cos.shape[0]
    assert cos.shape == (s_mod, d), f"cos {tuple(cos.shape)} != ({s_mod}, {d})"
    cos = cos.contiguous()
    sin = sin.contiguous()

    qkv_f = qkv.reshape(-1, 3 * h * d)   # contiguous view of the GEMM output
    n = qkv_f.shape[0]
    out_shape = tuple(qkv.shape[:-1]) + (h, d)
    q_out = torch.empty(n, h, d, dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty(n, h, d, dtype=qkv.dtype, device=qkv.device)

    launcher = _get_jit_bshd(h, d)
    args = (
        from_dlpack(qkv_f, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(cos, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(sin, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(q_out, assumed_align=16).mark_layout_dynamic(leading_dim=2),
        from_dlpack(k_out, assumed_align=16).mark_layout_dynamic(leading_dim=2),
        Int32(s_mod),
        current_cute_stream(),
    )
    # N is a runtime grid dim; H/D are baked in the closure. One artifact per (H, D).
    compiled = get_compiled(launcher, args, key=(qkv.dtype, h, d))
    compiled(*args)
    return q_out.view(out_shape), k_out.view(out_shape)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for `op_apply_rope` from reference.py.

    `q`, `k`: [B, H, S, D]
    `cos`, `sin`: [B, S, D] (or [1, S, D] or [S, D]) — identical across batch,
                  the kernel only uses the [S, D] slice.
    Returns (q_out, k_out) of same shape and dtype as q/k.
    """
    assert q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda
    assert q.shape == k.shape, "q and k must have the same shape"
    assert q.ndim == 4, "q must be [B, H, S, D]"
    assert q.dtype == k.dtype == cos.dtype == sin.dtype, "all tensors must share dtype"

    *_, d = q.shape
    s = q.shape[2]

    # Reduce cos/sin to a contiguous [S, D] view. HF/our reference usually
    # ships them as [B, S, D] via .expand (zero-stride on batch); take batch 0.
    if cos.ndim == 3:
        cos = cos[0]
        sin = sin[0]
    assert cos.shape == (s, d), f"cos shape {tuple(cos.shape)} != ({s}, {d})"
    assert sin.shape == (s, d), f"sin shape {tuple(sin.shape)} != ({s}, {d})"
    cos = cos.contiguous()
    sin = sin.contiguous()

    q = q.contiguous()
    k = k.contiguous()
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    shfl = _shfl_applicable(d)
    launcher = _get_jit(d, shfl)
    if shfl:
        # Flatten heads: the shuffle kernel works on [B*H, S, D] so a block owns
        # contiguous S rows of one (b,h). q is contiguous so .view is free.
        b, h = q.shape[0], q.shape[1]
        qk_in = (q.view(b * h, s, d), k.view(b * h, s, d))
        qk_out = (q_out.view(b * h, s, d), k_out.view(b * h, s, d))
        lead = 2
    else:
        qk_in = (q, k)
        qk_out = (q_out, k_out)
        lead = 3
    args = (
        from_dlpack(qk_in[0], assumed_align=16).mark_layout_dynamic(leading_dim=lead),
        from_dlpack(qk_in[1], assumed_align=16).mark_layout_dynamic(leading_dim=lead),
        from_dlpack(cos, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(sin, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(qk_out[0], assumed_align=16).mark_layout_dynamic(leading_dim=lead),
        from_dlpack(qk_out[1], assumed_align=16).mark_layout_dynamic(leading_dim=lead),
        current_cute_stream(),
    )
    # Grid is derived at runtime from q via cute.size, so B/H/S are not part of
    # the compilation signature. D is already captured by the launcher closure.
    compiled = get_compiled(launcher, args, key=(q.dtype, shfl))
    compiled(*args)
    return q_out, k_out
