# Inference serving showcase

This guide reproduces the RTX 5090 inference-serving comparison on real
`Tevatron/AgentIR-data` queries and passages. It measures query and document
encoding independently for four execution paths and two model families.

## What is measured

Two model families are measured:

- **Late interaction:** `lightonai/Agent-ModernColBERT`, producing one embedding
  per retained token.
- **Single vector:** `answerdotai/ModernBERT-base` through
  SentenceTransformers, producing one mean-pooled embedding per sequence.

Two serving endpoints are intentionally separate:

- **Query endpoint:** 256 instructed AgentIR queries, maximum length 128,
  batch size 8. This is a short-sequence, dispatch-sensitive workload.
- **Document endpoint:** 256 deduplicated AgentIR passages, maximum length 2048,
  batch size 8. This is a long-sequence, compute-sensitive workload.

This showcase measures encoding only. It does not run MaxSim or another
retrieval scoring stage.

## Files

- Runner: `showcase/inference_showcase.py`
- RTX 5090 configuration: `showcase/configs/inference_5090.yml`
- Packed encoding helpers: `showcase/packed_index.py`
- Generated result: `showcase/results/inference_5090.json` (ignored)
- Published figures: `docs/assets/inference_5090_{query,document}.png`

## Variants

Each family has four equivalent-output paths:

| role | late interaction | single vector | behavior |
|---|---|---|---|
| Eager stock | `pylate_eager` | `st_eager` | Unmodified eager encoder with dynamic per-batch padding. |
| Compiled stock | `pylate_compile_dynamic` | `st_compile_dynamic` | Stock encoder wrapped in `torch.compile(mode="max-autotune", dynamic=True)`. |
| Prepared padded | `pylate_pack` | `st_pack` | `pe.pack(..., attention_backend="flash")`; padded public inputs/outputs with varlen encoder compute. |
| Fully packed | `pylate_packed` | `st_packed` | No-padding collation and `packed_forward`; token outputs remain packed or are pooled directly by sequence range. |

Eager stock is the numerical reference for every other path.

## CUDA graphs

The query endpoint enables CUDA graphs for the prepared and fully packed paths.
The eager and compiled stock paths do not use these graph runners.

- Prepared padded uses fixed sequence buckets controlled by
  `cuda_graph_pad_to: 32`.
- Fully packed uses fixed total-token buckets while preserving sequence
  boundaries in `cu_seqlens`.
- Both graph paths cap sequence length at 128 and batch size at 8 through the
  endpoint configuration.

The document endpoint deliberately disables CUDA graphs. At long sequence
lengths, encoder compute dominates launch overhead, while graph pools would
complicate the memory comparison without a material throughput benefit.

CUDA graphs reserve private memory pools. This is why the query graph variants
can trade higher peak reserved VRAM for higher steady throughput. Report peak
**reserved** VRAM, not allocated VRAM.

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

For every endpoint and variant, `showcase/configs/inference_5090.yml` fixes:

- 256 sequences;
- batch size 8;
- descending length sorting;
- BF16 and inference mode;
- one complete warm-up pass;
- five trials;
- ten complete endpoint passes per trial.

The reported steady time is the median of the five per-pass trial times.
Throughput is `256 / median_seconds`.

Each endpoint/variant pair runs in a fresh subprocess with a fresh
`TORCHINDUCTOR_CACHE_DIR`. The orchestrator loads each AgentIR corpus once and
passes identical ordered texts to every worker.

The runner records:

- model-load and tokenization time;
- cold first-batch and first full-pass time;
- steady first-batch time and inferred first-call/JIT overhead;
- steady sequences per second;
- peak allocated and peak reserved VRAM;
- compiler-cache size, host RSS, token counts, and raw trial times.

## Run

From the repository root:

```bash
nvidia-smi
uv run showcase/inference_showcase.py showcase/configs/inference_5090.yml
```

The benchmark takes exclusive use of one GPU. Avoid running other GPU workloads
during measurement.

The runner writes one ignored JSON containing both endpoints. The committed query
and document figures are snapshots from the published reference run; each groups
late-interaction and single-vector results without treating their output
representations as interchangeable.

## Correctness checks

Every worker saves a small embedding snapshot. After an endpoint finishes, the
orchestrator compares compiled, prepared, and packed outputs with eager stock
within the same model family.

Parity thresholds are endpoint-specific:

- Query endpoint: cosine similarity `>= 0.98`.
- Document endpoint: cosine similarity `>= 0.997`.

The lower query floor accounts for the prepared CUDA graph path using fixed
padding buckets rather than each batch's exact maximum length. The fully packed
graph path should normally remain much closer to eager stock.

A failed parity check aborts the run instead of publishing the result JSON. The
result is complete only when:

1. both endpoints contain all eight variants;
2. every variant contains five raw trial times with ten repeats;
3. both parity groups pass for both endpoints.

## Reading the figures

Each endpoint figure has two panels:

1. startup plus one steady encoding pass;
2. peak reserved VRAM.

Interpret the comparisons as follows:

- **Eager versus compiled stock** shows the steady benefit and startup cost of
  dynamic compilation.
- **Eager/compiled versus prepared/packed on queries** includes the deliberate
  CUDA graph throughput-for-memory tradeoff.
- **Prepared versus packed on queries** compares padded and packed graph paths;
  both eliminate most Python/kernel-launch dispatch overhead.
- **Prepared versus packed on documents** is the clean external-padding
  comparison because neither path uses CUDA graphs.
- Do not rank late-interaction and single-vector throughput as if they produced
  the same retrieval representation.

Cold readiness and steady throughput answer different deployment questions.
Eager stock can have the shortest cold start while a compiled or graphed path
has higher sustained throughput.

## Changing the experiment

Copy the YAML file and change its `output.name` before altering endpoint batch
size, sequence length, corpus size, graph buckets, warm-up, or trial counts.
Keep each configuration with its generated JSON when archiving a run so different
protocols are not presented as one comparison.

Triton may print `OutOfMemoryError: out of resource` while rejecting autotuner
candidates that exceed the GPU shared-memory limit. These are not benchmark
OOMs if the worker continues and emits its result. A subprocess exit or missing
final JSON is a real failure.
