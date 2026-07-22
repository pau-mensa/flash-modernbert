# Indexing showcase

This guide reproduces the RTX 5090 long-document indexing comparison on
`Tevatron/AgentIR-data`. It compares four encoder/index paths for two retrieval
model families at batch size 128.

## What is measured

The runner loads 256 real AgentIR passages, deduplicated by document ID, and
gives the same ordered corpus to every variant.

Two model families are measured separately:

- **Late interaction:** `lightonai/Agent-ModernColBERT`, which stores one
  embedding per retained document token.
- **Single vector:** `answerdotai/ModernBERT-base` through
  SentenceTransformers, which mean-pools one embedding per document.

These families implement different retrieval systems. Do not compare their
absolute index sizes with each other. Compare variants only within a family.
MaxSim is not part of this showcase; indexing only creates document
representations.

## Files

- Runner: `showcase/index_showcase.py`
- RTX 5090 configuration: `showcase/configs/index_5090_b128.yml`
- Packed index helpers: `showcase/packed_index.py`
- Generated result: `showcase/results/index_5090_b128.json` (ignored)
- Published figures:
  `docs/assets/index_5090_b128_{late_interaction,single_vector}.png`

## Variants

Each family has four equivalent-output paths:

| role | late interaction | single vector | behavior |
|---|---|---|---|
| Eager stock | `pylate_eager` | `st_eager` | Unmodified eager encoder with dynamic per-batch padding. |
| Compiled stock | `pylate_compile_dynamic` | `st_compile_dynamic` | Stock encoder wrapped in `torch.compile(mode="max-autotune", dynamic=True)`. |
| Prepared padded | `pylate_pack` | `st_pack` | `pe.pack(..., attention_backend="flash")`; external batches and outputs remain padded while the encoder processes real tokens with FlashAttention varlen. |
| Fully packed | `pylate_packed` | `st_packed` | No-padding tokenization and `packed_forward`; outputs are reduced or stored directly from packed token ranges. |

For late interaction, the first three paths produce a rectangular token index
plus a mask. The fully packed path produces a flat token index plus
`cu_seqlens`.

For single-vector retrieval, all four paths ultimately store one pooled vector
per document. Fully packed mean pooling accumulates token sums in FP32 before
division, then converts back to the model dtype.

## Reproduction environment

The checked-in configuration was designed for:

- NVIDIA GeForce RTX 5090, sm_120, 32 GB
- NVIDIA driver 580.95.05
- PyTorch 2.8.0 with CUDA 12.8
- BF16 inference
- FlashAttention 2.8.3.post1
- Python 3.11

The runner has PEP 723 dependency metadata, including the sm_120-compatible
FlashAttention wheel and the required PyLate Git source. Install
[`uv`](https://docs.astral.sh/uv/) and ensure the machine can access the
Hugging Face Hub and GitHub. A Hugging Face token is optional for public models
and datasets but avoids anonymous rate limits.

Other CUDA GPUs can run an adapted configuration, but their timing, compiler
choices, and memory use are not directly comparable with the RTX 5090 result.

## Measurement protocol

`showcase/configs/index_5090_b128.yml` fixes:

- 256 documents;
- maximum document length 2048;
- batch size 128;
- descending length sorting;
- BF16 and inference mode;
- one complete warm-up pass;
- five trials;
- ten complete corpus passes per trial.

The reported steady time is the median of the five per-pass trial times.
Throughput is `256 / median_seconds`.

Each variant runs in a fresh subprocess with a fresh
`TORCHINDUCTOR_CACHE_DIR`. This prevents model state, allocator state, graph
pools, or compiler caches from leaking between variants.

The runner also records:

- model-load and tokenization time;
- cold first-batch and first full-pass time;
- steady first-batch time and inferred first-call/JIT overhead;
- peak allocated and peak reserved VRAM;
- logical and serialized index size;
- compiler-cache size and host RSS;
- raw trial times and encoder-token counts.

Use peak **reserved** VRAM for deployment comparisons. It includes allocator
and graph-pool memory that allocated-only measurements can miss.

## Run

From the repository root:

```bash
nvidia-smi
uv run showcase/index_showcase.py showcase/configs/index_5090_b128.yml
```

The benchmark takes exclusive use of one GPU. Avoid running other GPU workloads
during measurement.

The command writes the ignored `showcase/results/index_5090_b128.json`. The two
committed figures are snapshots from the published reference run and keep the
incompatible model families in separate comparisons.

## Correctness checks

Every worker saves a small embedding snapshot. After all workers finish, the
orchestrator compares compiled, prepared, and packed outputs with eager stock
within the same model family.

The indexing parity floor is cosine similarity `>= 0.997`. A failed parity
check aborts the run instead of publishing the result JSON.

The result is complete only when:

1. all eight variants are present;
2. every variant contains five raw trial times with ten repeats;
3. both family parity groups pass.

## Reading the figures

Each family figure has three panels:

1. startup plus one steady indexing pass;
2. peak reserved VRAM;
3. final logical index size.

Interpret the comparisons as follows:

- **Eager versus compiled stock** isolates the effect of dynamic compilation,
  including its startup cost.
- **Compiled stock versus prepared/packed** compares deployable stock and fused
  implementations, not packing alone.
- **Prepared padded versus fully packed** is the cleanest measure of external
  padding and index-layout overhead because both use the fused varlen encoder.
- The single-vector index sizes should be identical. Its relevant differences
  are throughput and temporary VRAM.
- The late-interaction packed index can be smaller because it does not retain a
  rectangular token tensor and mask.

## Changing the experiment

Copy the YAML file and change its `output.name` before altering batch size,
document count, sequence length, warm-up, or trial counts. Keep the original
configuration and result together; otherwise differently sampled protocols can
silently overwrite or be mistaken for the canonical run.

Triton may print `OutOfMemoryError: out of resource` while rejecting autotuner
candidates that exceed the GPU shared-memory limit. These are not benchmark
OOMs if the worker continues and emits its result. A subprocess exit or missing
final JSON is a real failure.
