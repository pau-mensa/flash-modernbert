"""Attention-only micro-benchmark: FA4 (CuteDSL) vs vendor SDPA, per layer type.

Isolates the single attention op the ModernBERT encoder calls per layer, so we can
characterize — *per device / arch* — exactly what the flash backend buys over
vendor SDPA, and where the crossover is, without the noise of the full forward.
This is the core measurement behind the Workstream-D dispatch decision (sdpa vs
flash): the crossover-S and the fixed-cost floor are both arch-dependent and must
be re-measured per device before committing a default.

ModernBERT-base attention: H=12, D=64, bf16, sliding window |i-j| <= 64. There are
two layer types, run differently by each end-to-end variant:

  global (full attention)
    eager-fused[sdpa]  F.sdpa(attn_mask=None)            -> SDPA flash fast path
    graph[sdpa]        F.sdpa(attn_mask=zeros[B,1,1,S])  -> finite mask, SDPA falls
                                                            OFF flash to mem-efficient
                                                            (a graph can't drop the
                                                            mask to None host-side)
    flash              flash_attn_func(window=(None,None))

  local (sliding window)
    sdpa (eager+graph) F.sdpa(attn_mask=band[1,1,S,S])   -> dense S*S band, O(S^2)
    flash              flash_attn_func(window=(64,64))    -> banded, O(S*W)

So the five primitives we time are:
    sdpa_nomask      global, eager-fused[sdpa]
    sdpa_dense_full  global, graph[sdpa]
    sdpa_band        local,  sdpa (eager AND graph)
    fa_full          global, flash
    fa_band          local,  flash

From those we synthesize the per-encoder attention cost for each end-to-end variant
(ModernBERT-base = 8 global + 14 local layers):
    eager[sdpa] = 8*sdpa_nomask     + 14*sdpa_band
    graph[sdpa] = 8*sdpa_dense_full + 14*sdpa_band
    flash       = 8*fa_full         + 14*fa_band   (same eager or graphed; FA is
                                                    bit-stable under capture)

The flash backend is the CuteDSL FA4 (`flash_attn.cute.interface.flash_attn_func`,
`pip install fa4`), the per-arch SOTA (sm_90/sm_100/sm_120 kernels), JIT-compiled.

    python benchmarks/attn_backend_micro.py [--out results/attn_micro.json]

Runs with whatever Python has torch + fa4; not tied to the package's uv env (the
package ships the *compiled* flash backend — this script measures the cute path).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time

import torch

# ModernBERT-base attention geometry.
H = 12
D = 64
WINDOW = 64           # sliding_half_window = local_attention // 2 = 128 // 2
N_GLOBAL = 8          # layers where layer_idx % 3 == 0, for 22 layers (0..21)
N_LOCAL = 14
SCALE = 1.0 / math.sqrt(D)
DTYPE = torch.bfloat16

# S-scaling sweep at a fixed batch (clean crossover; B=8 lets S=4096 fit).
S_SWEEP_B = 8
S_SWEEP = [32, 64, 128, 256, 512, 1024, 2048, 4096]
# Query fixed-cost sweep at a fixed short S (the tiny-S launch floor at real query
# batch sizes — where FA's higher fixed cost is pure overhead, window >= S).
Q_SWEEP_S = 32
Q_SWEEP_B = [32, 128, 256]


def _load_fa():
    """CuteDSL FA4 callable (window-as-structure). None if fa4 is absent.

    Install on the target arch with `pip install flash-attn-4==4.0.0b16` (sm_90/
    sm_100/sm_110; not sm_120). Imported from `flash_attn.cute` (the cute namespace
    package), independent of the package's own compiled flash backend."""
    try:
        from flash_attn.cute import flash_attn_func
        return flash_attn_func
    except Exception as exc:  # noqa: BLE001
        print(f"  (FA4 cute path unavailable: {exc})", file=sys.stderr)
        return None


def make_qkv(b: int, s: int, device, seed: int):
    """q/k/v as [B, H, S, D] contiguous — the layout SDPA receives in the real
    forward (RoPE outputs contiguous [B,H,S,D]). The flash path takes the strided
    [B,S,H,D] transpose view (D stays contiguous), exactly as ops.flash_attention
    does — so each backend sees the layout it sees in production."""
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (b, H, s, D)
    q = torch.randn(shape, generator=g, device=device, dtype=DTYPE)
    k = torch.randn(shape, generator=g, device=device, dtype=DTYPE)
    v = torch.randn(shape, generator=g, device=device, dtype=DTYPE)
    return q, k, v


def band_mask(s: int, device):
    """Additive [1,1,S,S] sliding band: 0 inside |i-j|<=W, finfo.min outside —
    the exact mask forward.py builds for local layers (finite, SDPA-ready)."""
    i = torch.arange(s, device=device)[:, None]
    j = torch.arange(s, device=device)[None, :]
    outside = (i - j).abs() > WINDOW
    m = torch.zeros((s, s), dtype=DTYPE, device=device)
    m = m.masked_fill(outside, torch.finfo(DTYPE).min)
    return m[None, None, :, :]


def full_dense_mask(b: int, s: int, device):
    """The all-zeros [B,1,1,S] finite mask a captured graph feeds global layers
    (forward.py: `dense_mask and full_mask is None -> zeros((b,1,1,s))`). Finite,
    so SDPA cannot take the flash fast path — the graph[sdpa] global penalty."""
    return torch.zeros((b, 1, 1, s), dtype=DTYPE, device=device)


def _backends(fa_func):
    """name -> callable(q,k,v, *, band, full). q/k/v are [B,H,S,D]."""
    def sdpa_nomask(q, k, v, **_):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, scale=SCALE
        ).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    def sdpa_dense_full(q, k, v, *, full, **_):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=full, scale=SCALE
        ).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    def sdpa_band(q, k, v, *, band, **_):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=band, scale=SCALE
        ).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    def _fa(q, k, v, window):
        # FA4 `flash_attn_func` returns (out, lse) — always a tuple; take [0].
        # ([B,S,H,D] native layout: pass the strided transpose views, D contiguous.)
        out = fa_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            softmax_scale=SCALE, causal=False, window_size=window,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.reshape(out.shape[0], out.shape[1], -1)

    def fa_full(q, k, v, **_):
        return _fa(q, k, v, (None, None))

    def fa_band(q, k, v, **_):
        return _fa(q, k, v, (WINDOW, WINDOW))

    names = {
        "sdpa_nomask": sdpa_nomask,
        "sdpa_dense_full": sdpa_dense_full,
        "sdpa_band": sdpa_band,
    }
    if fa_func is not None:
        names["fa_full"] = fa_full
        names["fa_band"] = fa_band
    return names


def time_call(fn, q, k, v, *, band, full, n_warmup=15, n_iters=50, n_trials=7):
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        fn(q, k, v, band=band, full=full)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn(q, k, v, band=band, full=full)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / n_iters * 1e3)
    return statistics.median(times)


def cosine(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def correctness(backends, device):
    """One shape: confirm FA matches its SDPA twin (global vs nomask, band vs band)."""
    if "fa_full" not in backends:
        return {}
    q, k, v = make_qkv(8, 512, device, seed=7)
    band = band_mask(512, device)
    full = full_dense_mask(8, 512, device)
    try:
        ref_full = backends["sdpa_nomask"](q, k, v, band=band, full=full)
        ref_band = backends["sdpa_band"](q, k, v, band=band, full=full)
        out_full = backends["fa_full"](q, k, v, band=band, full=full)
        out_band = backends["fa_band"](q, k, v, band=band, full=full)
    except Exception as exc:  # noqa: BLE001 — FA can be arch-unsupported (e.g. sm_120)
        print(f"  (FA correctness skipped: {exc})", file=sys.stderr)
        return {}
    return {
        "cos_global_fa_vs_sdpa": cosine(out_full, ref_full),
        "cos_local_fa_vs_sdpa": cosine(out_band, ref_band),
    }


def run_shape(backends, b: int, s: int, device):
    q, k, v = make_qkv(b, s, device, seed=1234 + s + b)
    band = band_mask(s, device)
    full = full_dense_mask(b, s, device)
    row = {}
    for name, fn in backends.items():
        try:
            row[name] = time_call(fn, q, k, v, band=band, full=full)
        except Exception as exc:  # noqa: BLE001
            print(f"    {name} failed at B={b} S={s}: {exc}", file=sys.stderr)
            row[name] = None
    del q, k, v, band, full
    torch.cuda.empty_cache()
    return row


def encoder_attn_cost(row):
    """Synthetic per-encoder attention time (ms) for each e2e variant from the
    per-call primitives. Predicts the attention contribution to the full forward."""
    def g(name):
        return row.get(name)

    out = {}
    if g("sdpa_nomask") is not None and g("sdpa_band") is not None:
        out["eager[sdpa]"] = N_GLOBAL * g("sdpa_nomask") + N_LOCAL * g("sdpa_band")
    if g("sdpa_dense_full") is not None and g("sdpa_band") is not None:
        out["graph[sdpa]"] = N_GLOBAL * g("sdpa_dense_full") + N_LOCAL * g("sdpa_band")
    if g("fa_full") is not None and g("fa_band") is not None:
        out["flash"] = N_GLOBAL * g("fa_full") + N_LOCAL * g("fa_band")
    return out


def fmt_row(b, s, row):
    cols = ["sdpa_nomask", "sdpa_dense_full", "sdpa_band", "fa_full", "fa_band"]
    parts = []
    for c in cols:
        v = row.get(c)
        parts.append(f"{c}={v:7.4f}" if v is not None else f"{c}=  --   ")
    line = f"  B={b:<4d} S={s:<5d}  " + "  ".join(parts)
    # decision-relevant ratios
    extra = []
    if row.get("fa_band") and row.get("sdpa_band"):
        extra.append(f"local FA/SDPA={row['sdpa_band'] / row['fa_band']:.2f}x")
    if row.get("fa_full") and row.get("sdpa_nomask"):
        extra.append(f"glob FA/SDPAflash={row['sdpa_nomask'] / row['fa_full']:.2f}x")
    cost = encoder_attn_cost(row)
    if "flash" in cost and "eager[sdpa]" in cost:
        extra.append(f"enc flash/eager[sdpa]={cost['eager[sdpa]'] / cost['flash']:.2f}x")
    if "flash" in cost and "graph[sdpa]" in cost:
        extra.append(f"flash/graph[sdpa]={cost['graph[sdpa]'] / cost['flash']:.2f}x")
    return line + ("   | " + "  ".join(extra) if extra else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="write JSON here")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("error: needs a CUDA GPU", file=sys.stderr)
        raise SystemExit(1)

    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"device: {name}  sm_{cap[0]}{cap[1]}  torch {torch.__version__}")
    print(f"geom: H={H} D={D} window={WINDOW} layers={N_GLOBAL}G+{N_LOCAL}L  dtype={DTYPE}\n")

    fa_func = _load_fa()
    backends = _backends(fa_func)

    corr = correctness(backends, device)
    if corr:
        print("correctness (cosine vs SDPA twin, B8 S512):")
        for k, v in corr.items():
            print(f"  {k}: {v:.5f}")
        print()

    print("=== S-scaling sweep (B=8) — ms/call per primitive + ratios ===")
    print("  (local FA/SDPA>1 -> FA local wins;  glob FA/SDPAflash>1 -> FA global wins)")
    sweep_a = {}
    for s in S_SWEEP:
        row = run_shape(backends, S_SWEEP_B, s, device)
        sweep_a[s] = row
        print(fmt_row(S_SWEEP_B, s, row))

    print("\n=== query fixed-cost sweep (S=32) — tiny-S launch floor ===")
    sweep_b = {}
    for b in Q_SWEEP_B:
        row = run_shape(backends, b, Q_SWEEP_S, device)
        sweep_b[b] = row
        print(fmt_row(b, Q_SWEEP_S, row))

    # crossover-S: smallest S where the synthetic flash encoder cost beats eager[sdpa]
    crossover = None
    for s in S_SWEEP:
        cost = encoder_attn_cost(sweep_a[s])
        if "flash" in cost and "eager[sdpa]" in cost and cost["flash"] < cost["eager[sdpa]"]:
            crossover = s
            break
    print(f"\nflash-beats-eager[sdpa] encoder-attn crossover: "
          f"{'S=' + str(crossover) if crossover else 'never in sweep'}")

    payload = {
        "device": name,
        "capability": list(cap),
        "torch": torch.__version__,
        "geom": {"H": H, "D": D, "window": WINDOW, "n_global": N_GLOBAL, "n_local": N_LOCAL},
        "correctness": corr,
        "s_sweep": {str(s): sweep_a[s] for s in S_SWEEP},
        "q_sweep": {str(b): sweep_b[b] for b in Q_SWEEP_B},
        "crossover_s": crossover,
        "encoder_cost_s_sweep": {str(s): encoder_attn_cost(sweep_a[s]) for s in S_SWEEP},
        "encoder_cost_q_sweep": {str(b): encoder_attn_cost(sweep_b[b]) for b in Q_SWEEP_B},
    }
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
