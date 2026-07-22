# Attention dispatch benchmarks

These benchmarks calibrate `attention_backend="auto"`: the runtime choice between
the specialized packed Triton attention kernel and FlashAttention. The benchmark
sources and fitted policy are committed; raw sweep JSON is generated under the
ignored `benchmarks/results/` directory.

## Files

| File | Purpose |
|---|---|
| `packed_attention_dispatch_bench.py` | Shared attention-only Triton/Flash sweep used on every calibrated GPU. |
| `packed_graph_boundary_bench.py` | Full ModernBERT forward validation around the selected graph boundary. |
| `fit_packed_attention_dispatch.py` | Offline linear-separator search over capture-safe shape features. |
| `packed_short_attention_bench.py` | Local kernel microbenchmark and parity harness, including AgentIR length profiles. |
| `../scripts/modal_packed_attention_dispatch_bench.py` | Concurrent A100, L40S, H200, and B200 orchestration. |

The obsolete `token_dispatch_bench.py` files are historical padded-dispatch
experiments and are not callers of the current runtime policy.

## Runtime contract

Inference `attention_backend="auto"` compares the specialized packed Triton kernel
with FA2/FA4. SDPA is not a crossover candidate: it remains an explicit backend, the
training path, and the final dependency/invariant fallback.

The Triton kernel's supported envelope is bf16 CUDA ModernBERT-base attention with 12
heads, head dimension 64, global or local-64 attention, no autograd, and `Smax<=128`.
Outside that envelope, auto prefers Flash and then SDPA.

Every calibrated policy produces one integer score and makes one monotonic comparison:

```text
score <  threshold -> Triton
score >= threshold -> FlashAttention
```

The policies use only host-visible shape arguments. Let:

```text
N = number of packed sequences
M = number of live packed tokens
S = maximum sequence length
F = N*S - M                         fragmentation tokens
A = floor(M*M/N)                    mean-square work proxy
Q = N*ceil(S/BM)                    Triton query CTAs per head
BM = 16 for S<=64, otherwise 64
```

No policy reads CUDA `cu_seqlens` values. Eager packed calls know `N`, `M`, and `S`
from tensor shapes and the Python `max_seqlen` argument. A packed CUDA graph scores its
fixed `(max_batch, M_bucket, max_seq)` work during capture, and replay has no dispatch
overhead.

## Calibrated policies

Measurements are captured bf16 attention-only timings, weighted for ModernBERT-base's
8 global and 14 local-64 layers. A point is labeled a production Flash win only when
`t_triton / t_flash >= 1.03`.

| Exact card | Flash implementation | Score | Eager threshold | Graph threshold |
|---|---|---:|---:|---:|
| RTX 5090 | compiled FA2 2.8.3.post1 | `M` | `20,736` | same |
| A100-SXM4-40GB | compiled FA2 2.8.3.post1 | `-203,715 N + 24 A + 41,801 Q` | `9,422,745` | same |
| L40S | compiled FA2 2.8.3.post1 | `4,260 M - 7 N S² - 11,752 Q` | `77,166,921` | `74,000,000` |
| H200 | Cute FA4 4.0.0b16 | `M` | `131,073` | same |
| B200 | Cute FA4 4.0.0b16 | `M` | `131,073` | same |

The H200/B200 thresholds are bounded calibration sentinels, not observed crossovers.
H200 had no guarded FA4 win through the largest measured `M=131,072`. B200 had no
stable monotonic crossover: its isolated `N=8` guarded win moved from skewed `S=96`
to skewed `S=128` between two long refinement runs, while neighboring points favored
Triton. Production therefore keeps Triton through the measured range and
conservatively returns to FA above it instead of extrapolating “always Triton.”

Policies are exact-card only. A100 does not transfer to A30, L40S does not transfer to
other sm_89 cards, and H200 does not transfer to H100/GH200. An uncalibrated card
prefers Flash when installed; if Flash is absent it uses Triton inside the supported
envelope and SDPA otherwise.

## Datacenter calibration

The coarse sweep contains 308 points per GPU:

- `S in {32,48,64,96,128}`;
- `N in {1,2,4,8,16,32,64,96,128,160,192,224,256,320,384,512,640,768,1024,1536,2048}`;
- equal, alternating full/half, and one-long/rest-eighth profiles;
- at most 131,072 live tokens;
- five paired samples with a 20 ms replay target per backend and attention mode.

Every timing captures both backend graphs, alternates their order within each sample,
and starts after an explicit GPU clock warm-up. The first refinement reruns 72 ratios
nearest the 3% guard with nine samples and a 100 ms target per backend and mode. The
second reruns 72 points covering both sides and adjacent batch sizes of every observed
label transition. Later rows replace coarse rows at the same `(profile,N,S)`.

After both refinements, the rounded integer A100 policy selected 55/308 points with no
measured false selections or missed guarded wins. Its minimum selected speedup was
1.0343 and maximum rejected speedup 1.0260. The L40S policy also selected 55/308
cleanly; its minimum selected speedup was 1.0408 and maximum rejected speedup 1.0159.
Independent validation reruns preserved both cards' selected/rejected boundary choices.

H200's largest Triton/FA4 ratio was only 0.7556. B200's single final guarded island was
1.0999, but it was not repeatable at a fixed shape; its largest rejected ratio was
1.0268. The parity probes across all four cards had cosine at least 0.9999994 and
maximum absolute bf16 difference 0.0078125.

The fitted feature set is deliberately restricted to capture-safe values. Pair counts
computed from the actual length distribution can improve retrospective fit, but using
them would require a device-to-host read or an extra caller-supplied statistic and would
make eager and graph policies differ.

## Packed CUDA-graph boundary check

Packed graphs cache actual `(M_bucket, N, Smax, backend)`. `max_batch` and `max_seq`
are admission bounds, not padded workload statistics. Keeping the actual `N` and
`Smax` in the key is necessary both for the calibrated score and for Triton's static
tile selection; keeping the backend prevents equal-M distributions on opposite sides
of the policy from aliasing one graph.

The graph check captures complete random-weight 22-layer ModernBERT-base packed
forwards with forced Triton and Flash backends, then times paired replays. A routing
difference is actionable only at the same 3% production guard used by attention
calibration; smaller full-forward differences are treated as ties.

| Card / representative point | Auto route | `t_triton/t_flash` | Result |
|---|---:|---:|---|
| RTX 5090, `M=20,480` | Triton | `0.9850` | Triton side retained |
| RTX 5090, `M=20,736` | FA2 | `1.0138` | FA2 direction retained; full-forward tie band |
| A100, half `S=64`, `M=18,432` | Triton | `1.0087` | tie band |
| A100, half `S=64`, `M=24,576` | FA2 | `1.0057` | tie band |
| L40S, equal `S=64`, `M=16,384` | Triton | `0.9732` | Triton side retained |
| L40S, equal `S=64`, `M=24,576` | FA2 | `1.0561` | guarded graph crossover |
| L40S, equal `S=64`, `M=27,648` | FA2 | `1.0965` | FA2 side retained |
| H200, equal `S=128`, `M=1,024/16,384` | Triton | `0.9647/0.8823` | Triton retained |
| B200, equal `S=128`, `M=1,024/16,384` | Triton | `0.9990/0.8652` | tie/Triton retained |

The L40S sweep additionally covered equal and half profiles at `S=32,64,96,128`.
Exactly one previously rejected row became a guarded Flash win under full graph
capture: equal `S=64`, `M=24,576`. Lowering only the L40S graph threshold to
`74,000,000` selects it while leaving the measured ragged Triton wins and every
sub-3% point on Triton. No separate feature set or lookup table was introduced.

## Reproduction

The shared benchmark is
`benchmarks/packed_attention_dispatch_bench.py`; the compact linear-separator fitter is
`benchmarks/fit_packed_attention_dispatch.py`. Run all datacenter cards concurrently:

```bash
uvx modal run scripts/modal_packed_attention_dispatch_bench.py
uvx modal run scripts/modal_packed_attention_dispatch_bench.py --refine
uvx modal run scripts/modal_packed_attention_dispatch_bench.py \
  --refine --refine-round 2 --gpus A100,L40S,B200
uvx modal run scripts/modal_packed_attention_dispatch_bench.py --validate
uvx modal run scripts/modal_packed_attention_dispatch_bench.py --graph-validate
```

Run the local kernel sweep on an installed RTX 5090 environment with:

```bash
uv run benchmarks/packed_short_attention_bench.py \
  --batches 1,8,32 --seqs 32,64,128 \
  --output benchmarks/results/packed_short_attention_5090.json
```

The launcher provisions the remote CUDA/PyTorch/FlashAttention environments, so Modal
is not a runtime dependency of `packed-encoders`. `uvx` installs the CLI in an isolated
environment; authenticate it before the first run.

Fit a capture-safe separator after the coarse and refinement sweeps:

```bash
uv run benchmarks/fit_packed_attention_dispatch.py \
  benchmarks/results/packed_attention_dispatch_a100.json \
  --refine benchmarks/results/packed_attention_dispatch_a100_refine.json \
  --refine benchmarks/results/packed_attention_dispatch_a100_refine2.json
```

Raw coarse, refinement, and validation JSON files are written under
`benchmarks/results/` as `packed_attention_dispatch_<card>*.json`. They are local
measurement artifacts and are intentionally not committed. The remote metadata
verifies the actual loader kind: `compiled` on A100/L40S and `cute` on H200/B200. The
FA4 image is pinned to `flash-attn-4==4.0.0b16`; that package does not expose a useful
top-level version string, so result metadata reports the loader kind while the image
definition records the pin.

The obsolete padded SDPA/Flash runtime registry was removed when auto stopped
selecting SDPA. Historical scripts and local results may remain for provenance, but
they have no runtime caller.
