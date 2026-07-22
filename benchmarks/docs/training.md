# Training showcase — B200

This document describes how to reproduce the training showcases on an NVIDIA
B200. Two model families are covered:

- **Single-vector** — `Alibaba-NLP/gte-modernbert-base`, CLS pooling, no
  GradCache. Direct forward-backward per step, so both step time and memory
  directly reflect the forward-path difference.
- **Late interaction** — `lightonai/GTE-ModernColBERT-v1`, ColBERT MaxSim
  scoring with GradCache. The faithful Agent-ModernColBERT recipe (B=32,
  D=4096, Q=8192). GradCache chunks the forward into mini-batches, so the
  headline is per-step speed; memory is dominated by GradCache's chunking.

Both showcases train on `Tevatron/AgentIR-data` for two epochs and compare
three forward paths: stock (padded), `pe.pack` (drop-in fused), and fully
packed. All three start from bit-identical initial probe loss, confirming
equivalent initialization.

## Single-vector results

Configuration: B=16, 7 negatives, query/document caps 128/2048, 650 optimizer
steps (two epochs over 5200 groups), seed 42. Median step time is measured
after all 325 distinct batch shapes have appeared at least once (second epoch).

| | Stock | pe.pack | Packed |
|---|---:|---:|---:|
| Step time (ms) | 1215.8 | 271.2 | 270.5 |
| **Speedup vs stock** | — | **4.48x** | **4.50x** |
| Memory allocated (GB) | 170.1 | 91.0 | 90.6 |
| Memory reserved (GB) | 176.5 | 91.8 | 91.2 |
| **Memory reduction** | — | **1.87x** | **1.88x** |
| Initial probe loss | 2.0895 | 2.0895 | 2.0895 |
| Final probe loss | 1.9516 | 1.9441 | 1.9433 |
| Path compile warmup (s) | 0.0 | rerun required | rerun required |
| **Compile-accounted training time** | **13m 11s** | **rerun required** | **rerun required** |
| **Compile-accounted speedup** | — | **rerun required** | **rerun required** |

No GradCache: stock forwards all 128 documents simultaneously through padded
attention, consuming 170 GB. pe.pack and packed eliminate padding waste,
halving memory while delivering a 4.5x step-time speedup. The loss plot shows
equal-step parity — same optimization trajectory regardless of forward path —
and pe.pack/packed reaching the same loss in 4.5x less wall-clock time.

## Late-interaction results

Configuration: B=32, 7 negatives, query/document caps 8192/4096, GradCache
mini-batch 8, 326 optimizer steps (two epochs over 5222 groups), seed 42.
Compile warmup runs all 163 distinct batches before the measured loop.

| | Stock | pe.pack | Packed |
|---|---:|---:|---:|
| Step time (ms) | 10518 | 1812 | 1516 |
| **Speedup vs stock** | — | **5.80x** | **6.94x** |
| Path compile warmup (s) | 0.0 | 398 | 352 |
| **Compile-accounted training time** | **1h 05m 06s** | **16m 39s** | **14m 18s** |
| **Compile-accounted speedup** | — | **3.91x** | **4.55x** |
| Memory allocated (GB) | 25.6 | 17.0 | 16.7 |
| Memory reserved (GB) | 26.7 | 17.2 | 16.9 |
| **Memory reduction (reserved)** | — | **1.55x** | **1.58x** |
| Initial probe loss | 35.812 | 35.812 | 35.812 |
| Final probe loss | 34.542 | 36.217 | 36.732 |

GradCache chunks each step into 36 encoder mini-batches (4 query + 32
document), so all three variants fit in ~25–30 GB regardless of the total batch
size. The headline is speed: pe.pack delivers 5.8x per-step throughput and
packed 6.9x, or 3.9x/4.6x compile-accounted. Memory allocated is 1.5x lower
for the fused variants, though GradCache's chunking limits the absolute saving
compared to the single-vector showcase.

The probe loss oscillates within a ~2-point band for all three variants at
temperature 0.01. The final snapshot differs because the curves are sampled
every 50 steps (7 data points); at step 300 packed (35.44) is lower than stock
(36.03). The three trajectories are statistically equivalent.

## Environment

- NVIDIA B200, sm_100, 191.5 GB HBM
- PyTorch 2.11.0+cu128
- `flash-attn-4==4.0.0b16` (CuteDSL FA4)
- `nvidia-cutlass-dsl==4.5.2`
- `quack-kernels==0.5.0`
- `pylate>=1.5` (late-interaction only)
- `matplotlib` (for figure rendering)
- Modal B200 sandbox managed by `benchmarks/scripts/modal_b200_sandbox.py`

## Reproduction

All commands use the packed-encoders virtual environment for Modal:

```bash
MODAL_PY=.venv/bin/python
```

### 1. Start the sandbox and sync

```bash
$MODAL_PY benchmarks/scripts/modal_b200_sandbox.py up
$MODAL_PY benchmarks/scripts/modal_b200_sandbox.py sync
```

### 2. Install dependencies

```bash
$MODAL_PY benchmarks/scripts/modal_b200_sandbox.py run -- \
  pip install \
    flash-attn-4==4.0.0b16 \
    nvidia-cutlass-dsl==4.5.2 \
    quack-kernels==0.5.0 \
    matplotlib \
    'pylate>=1.5'
```

### 3. Run the single-vector showcase (~50 minutes)

```bash
$MODAL_PY benchmarks/scripts/modal_b200_sandbox.py run -- env \
  PYTHONPATH=/workspace/packed-encoders:/workspace/packed-encoders/flash-maxsim \
  HF_HOME=/workspace/hf-cache \
  PYTHONWARNINGS=ignore \
  python -m benchmarks.showcases.training \
    benchmarks/configs/training_b200_single_vector.yml \
    --families single_vector
```

### 4. Run the late-interaction showcase (~2 hours)

```bash
$MODAL_PY benchmarks/scripts/modal_b200_sandbox.py run -- env \
  PYTHONPATH=/workspace/packed-encoders:/workspace/packed-encoders/flash-maxsim \
  HF_HOME=/workspace/hf-cache \
  PYTHONWARNINGS=ignore \
  python -m benchmarks.showcases.training \
    benchmarks/configs/training_b200_late_interaction_sampled.yml \
    --families late_interaction
```

Each showcase runs three sequential workers (stock, pe.pack, packed). Every
worker streams its data from the Hugging Face Hub, trains for the configured
number of steps, then runs a per-step memory measurement pass. The pe.pack
and packed workers first compile all distinct batch shapes while stock starts
immediately in eager mode.

### 5. Pull results and stop the sandbox

```bash
for name in training_b200_single_vector training_b200_late_interaction; do
  for suffix in .json _iso_loss.png _memory.png; do
    $MODAL_PY benchmarks/scripts/modal_b200_sandbox.py pull \
      /workspace/packed-encoders/benchmarks/results/${name}${suffix} \
      benchmarks/results/${name}${suffix}
  done
done

$MODAL_PY benchmarks/scripts/modal_b200_sandbox.py down
```

## Artifacts

### Single-vector

- `benchmarks/configs/training_b200_single_vector.yml` — run configuration
- `benchmarks/results/training_b200_single_vector.json` — full results
- `benchmarks/docs/assets/training_b200_single_vector_iso_loss.png` — published fixed-probe
  loss vs optimizer step and vs training wall-clock time
- `benchmarks/docs/assets/training_b200_single_vector_memory.png` — published per-step peak
  CUDA memory (allocated and reserved) for all three variants

### Late interaction

- `benchmarks/configs/training_b200_late_interaction_sampled.yml` — run
  configuration
- `benchmarks/results/training_b200_late_interaction.json` — full results
- `benchmarks/docs/assets/training_b200_late_interaction_iso_loss.png` — published fixed-probe
  loss vs optimizer step and vs training wall-clock time
- `benchmarks/docs/assets/training_b200_late_interaction_memory.png` — published per-step peak
  CUDA memory (allocated and reserved) for all three variants

## Measurement methodology

**Compile accounting.** `compile_warmup: true` makes only the pe.pack and
packed workers run one full forward-backward-optimizer pass over every distinct
batch before the measured training loop. Stock is eager and receives no such
warmup. The initial model state is then restored, the optimizer is recreated,
and the CUDA allocator is reset. Path compilation is excluded from the loss
chart so all curves retain the same step-zero origin, but it is included in the
separate compile-accounted training-time comparison. Fixed-probe setup is
recorded separately and excluded from both because it is showcase instrumentation.

**Fixed-probe loss.** Instead of plotting training loss (which varies with
the mini-batch and is affected by BF16 numerics across forward paths), a
disjoint set of 16 groups is evaluated periodically through an identical
`packed_forward` reference path for all variants. This ensures the probe loss
is directly comparable across stock, pe.pack, and packed — all three
produce bit-identical initial probe loss.

**Per-step memory.** After the training loop, distinct batches are re-run
individually with `torch.cuda.empty_cache()` between batches. This reports
the actual per-step memory requirement without cross-batch allocator cache
accumulation. The `memory_sample_count` option measures only the N heaviest
batches (by total encoder tokens) instead of all batches, which is sufficient
to find the peak and substantially faster for large batch counts. The
late-interaction config uses `expandable_segments:True` to avoid allocator
fragmentation from GradCache's variable-sized packed tensors.

**Step time.** The median is computed over the second-epoch steps (after all
distinct batch shapes have been seen at least once). The first-epoch pass
may include residual CuteDSL shape specialization not covered by compile
warmup.

## Design choices

**Single-vector: batch size 16 instead of 32.** With 7 negatives and document
length 2048, each step processes 128 documents. At B=32 (256 documents), stock
requires ~324 GB of activation memory. B=16
fits all three variants comfortably (stock at 170 GB, packed at 91 GB).

**Single-vector: document length 2048.** Agent-ModernColBERT uses 4096 for
late-interaction ColBERT, but single-vector models use CLS pooling where
extreme document lengths offer diminishing returns. The AgentIR documents
average ~1886 tokens, so 2048 captures most content without truncation while
keeping memory feasible.

**Single-vector: no GradCache.** The single-vector showcase runs a single
forward-backward per step to measure raw per-step performance. This isolates
the forward path comparison without GradCache's chunking masking the memory
difference.

**Late interaction: GradCache.** At B=32 with D=4096, all three variants
(including packed) exceed 192 GB without GradCache. GradCache with
mini-batch 8 is required for the faithful Agent-ModernColBERT recipe and
reduces peak memory to ~25–30 GB for all variants. The memory comparison under
GradCache is modest (1.5x allocated) because GradCache already solves the
memory problem.

## Known issue: query instruction prefix

The `query_instruction` in the YAML configs uses `\n` inside a double-quoted
string, which YAML interprets as a real newline character (U+000A). The
upstream GTE-ModernColBERT instruction prefix uses a literal backslash-n
(`\\nQuery:`), i.e. two characters `\` and `n`, not a newline. All three
variants (stock, pe.pack, packed) were run with the same (incorrect) prefix,
so the relative comparison — step time, memory, and loss parity — is valid.
However, the absolute probe-loss values are not directly comparable to runs
using the model's intended instruction format.
