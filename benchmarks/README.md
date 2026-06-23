# Benchmarks

Config-driven comparisons of stock training/inference against the same model
`flash_modernbert.prepare()`-ed with the fused tail.

Two training benchmarks that answer different questions:

| script | path | answers |
| --- | --- | --- |
| `pylate_trainer.py` | real `SentenceTransformerTrainer` (on accelerate) | "what does a real PyLate training run get?" |
| `iso_loss.py` | manual loop, bf16 weights | "what is the clean kernel-only A/B?" |

Both run the same recipe twice — stock, then with one extra `prepare()` call —
from an identical weight init, data order, optimizer, and budget, and write a
JSON of per-step metrics plus a PNG of the two loss curves (vs step for
correctness, vs wall-clock for speed). Both require a CUDA GPU (sm_90 / sm_100 /
sm_120) and error otherwise.

## pylate_trainer.py — the real recipe, only `prepare()` differs

Ordinary PyLate boilerplate — `models.ColBERT`, `losses.CachedContrastive`, the
`ColBERTCollator`, and the actual `SentenceTransformerTrainer` (which runs on
accelerate). The **only** difference between the two runs is one line:

```python
model = build_colbert(cfg)
if fused:
    fm.prepare(model, cuda_graph=cfg["cuda_graph"])   # <-- the entire difference
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                                     loss=loss, data_collator=collator, callbacks=[metrics])
trainer.train()
```

A `TrainerCallback` records per-step loss / step time / peak VRAM from inside the
real loop. Run it:

```bash
uv run benchmarks/pylate_trainer.py benchmarks/configs/msmarco_trainer.yml
```

Measured (RTX 5090, ModernBERT-base, MS MARCO, batch 16, doc 512, 60 steps,
`bf16=True`): fused **1.22×/step** and **0.85× peak VRAM (saved 1.20 GB)**.

Configs:

- `msmarco_trainer.yml` — ModernBERT-base on MS MARCO triplets.
- `agent_modern_colbert.yml` — replicates the
  [Agent-ModernColBERT](https://huggingface.co/lightonai/Agent-ModernColBERT)
  recipe (`GTE-ModernColBERT-v1` + `Tevatron/agent_ir-data`, CachedContrastive,
  7 negatives); set `run.max_seconds: 300` for the 5-min-per-variant run. The
  faithful document length (4096) needs a large GPU (B200-class).

A `dataset.format: agent_ir` flattens the AgentIR `positive_passages` /
`negative_passages` lists into the flat `query` / `positive` / `negative_i`
columns the collator wants; plain triplet datasets (`[query, positive,
negative]`) need no `format`.

**CUDA graphs are an inference feature** and are skipped during training (they
can't be captured under autocast/autograd — autocast frees the ephemeral bf16
weight copies a captured graph would replay against). Setting `cuda_graph: true`
in a training config is a no-op with a one-time warning; it belongs on the
inference benchmarks.

Two things to read honestly:

- The memory delta here (1.20 GB) is larger than the clean kernel A/B below
  (0.43 GB at the same shape). That is because the canonical recipe uses
  `bf16=True` (fp32 master weights + autocast), and GradCache's pass-3 recompute
  runs *outside* autocast — so stock runs pass-3 in fp32 while the fused model
  casts to bf16. Part of the win is the fused model handling that regime more
  robustly, which is a real thing a user gets, but it is not *only* the kernels.
- Over a short, noisy run the loss curves track in trend but the endpoints can
  separate (compounding bf16 differences); this script is the *speed/memory*
  story. For the tight loss-overlap correctness story, use `iso_loss.py`, which
  trains in bf16 weights with no autocast confound.

## iso_loss.py — clean kernel A/B (bf16 weights, manual loop)

Trains a ModernBERT ColBERT model twice — stock PyLate, then the identical model
with `prepare()` — from the **same weight init, same data order, same optimizer,
same budget** — and reports the two loss curves with the speed and peak-memory
deltas. Any gap in the loss curves is the bf16 kernel band; any gap in ms/step or
peak VRAM is the fused tail.

```bash
uv run benchmarks/iso_loss.py benchmarks/configs/synthetic_quick.yml
```

It requires a CUDA GPU (sm_90 / sm_100 / sm_120) and errors otherwise. Outputs go
to `output.dir`:

- `<name>.json` — config echo, per-step loss/time arrays, per-variant summaries,
  and the stock-vs-fused comparison.
- `<name>.png` — loss vs step (correctness: the curves overlap) and loss vs
  wall-clock (speed: the fused curve covers the budget sooner / reaches further).

### Configs

| config | source | what it shows |
| --- | --- | --- |
| `synthetic_quick.yml` | synthetic, 40 steps | fast smoke; correctness + a modest delta |
| `synthetic_showcase.yml` | synthetic, batch 32 × 3 neg | the speed/memory win at a representative shape |
| `msmarco.yml` | MS MARCO triplets, 120 s | real loss curves over an iso-*time* budget |

Measured on an RTX 5090 (sm_120), bf16, doc length 512:

| shape | fused speed | peak VRAM saved |
| --- | --- | --- |
| batch 16, 1 neg | 1.20×/step | 0.43 GB |
| batch 32, 3 neg | 1.39×/step | 0.86 GB |

### Config schema

```yaml
run:
  max_steps: 40           # stop after N optimizer steps  (iso-step)
  max_seconds: 120        # …or after T seconds           (iso-time); set one or both
  warmup_steps: 8         # forward+backward without stepping, to pay CuteDSL JIT
  seed: 42                # shared init + data order
  validate: true          # run the flash-modernbert hard gate on the fused model
model:
  name_or_path: answerdotai/ModernBERT-base
  dtype: bfloat16         # bfloat16 | float32 (see precision note below)
  embedding_size: 128
  query_length: 32
  document_length: 512
dataset:
  type: synthetic         # synthetic (controlled shapes) | huggingface (real text)
  batch_size: 16
  # synthetic: num_samples, num_negatives, query_len, doc_len, vocab
  # huggingface: name, split, columns: [query, positive, negative]
loss:
  type: cached_contrastive  # cached_contrastive (GradCache) | contrastive
  mini_batch_size: 16
cuda_graph: false
optimizer: { lr: 3.0e-5 }
output: { dir: benchmarks/results, name: my_run }
```

## inference_bench.py — encoder forward: stock vs fused vs graphed

The inference counterpart. It measures the **encoder forward** (the region
`prepare()` patches and the CUDA-graph runner captures) in plain bf16 inference
(`eval()` + `no_grad()`, no autocast — the regime every real encode path runs
in), on one model, three ways: stock HF, `prepare()`-ed (eager fused), and fused
+ graphs. Pre-tokenized batches at exact `(B, S)` so there is no tokenization
variance; tokenization / projection head / MaxSim / normalize are framework-side
and unaffected by the fused tail.

```bash
uv run benchmarks/inference_bench.py benchmarks/configs/inference_short.yml
uv run benchmarks/inference_bench.py benchmarks/configs/inference_indexing.yml
```

Measured (RTX 5090 sm_120, ModernBERT-base, bf16), speedup vs stock HF and
graphed peak memory (the runner captures the *whole* forward — prologue + core —
from the static id/mask buffers, so the per-replay copy is just the token ids):

| regime | shape | fused | graphed | graphed peak | fused peak |
| --- | --- | --- | --- | --- | --- |
| short-S query | B32 S32 | **0.95×** | **1.58×** | 338 MB | 343 MB |
| short-S query | B128 S32 | 1.28× | 1.39× | 357 MB | 415 MB |
| short-S query | B256 S32 | 1.32× | 1.40× | 385 MB | 513 MB |
| short-S doc | B32 S320 | 1.33× | 1.35× | 414 MB | 570 MB |
| long-S index | B32 S512 | 1.48× | 1.46× | 386 MB | 725 MB |
| long-S index | B8 S2048 | 1.33× | 1.23× | 453 MB | 776 MB |
| long-S index | B4 S4096 | 1.23× | 1.09× | **487 MB** | 974 MB |

Three honest takeaways:

- **At small short-S shapes the eager fused tail *loses* to stock (0.95× at
  B32/S32) and CUDA graphs are what recover competitiveness** (1.58×). This is
  the classic ColBERT query regime, so graphs are not upside there — they are
  necessary. The win is widest at the smallest shape and narrows as the batch
  grows (more compute per launch to amortize).
- **Graphs are the memory-cheapest path at every length** — roughly half the peak
  of the eager fused tail at long S (487 MB vs 974 MB at S4096, below even stock).
  Capturing the whole forward drops the per-replay copy to the token ids and lets
  the graph's private pool pack the 22-layer activations tightly.
- **At long S the eager fused tail still wins on latency (~1.2–1.5×) and graphs
  match-or-trail it.** The cause is *not* host launches (capturing the whole
  prologue, masks included, left long-S latency unchanged): to capture, the graph
  must use a **static dense attention mask**, which forecloses the data-dependent
  flash-attention fast path. Eager-fused drops the all-ones `full_mask` to `None`
  so SDPA runs flash on the ~7 global layers; the graph can't make that
  host-side decision, so those layers fall to the slower mem-efficient backend,
  and at long S that backend gap is the whole difference. Recovering it would need
  an "assume-unpadded" capture mode that passes `None` for the global mask (valid
  for query expansion / fixed-length unpadded indexing, wrong for padded batches).

**Guidance:** enable `cuda_graph` for short-S query serving (it is what makes the
fused tail competitive there) and whenever peak memory is the constraint. For
long-S document indexing, the `flash` attention backend below is the bigger lever.

### attention_backend="flash" — windowed FlashAttention for long S

The sdpa speedup *shrinks* as S grows because the sliding-window local layers
(≈2/3 of the model) are expressed as a dense `S×S` mask and run full O(S²).
`prepare(model, attention_backend="flash")` swaps attention for FlashAttention,
which takes the window as *structure* (`window_size`) and prunes the local layers
to the band — O(S·W) — instead of masking. It needs the `flash-attn` package and
targets unpadded inference (queries with expansion, fixed-length docs); the
default `"sdpa"` backend is dependency-free and handles padded batches.

Backend comparison (5090, ModernBERT-base, bf16), speedup vs stock HF, run via
`inference_indexing_fa.yml` / `inference_short_fa.yml`:

| shape | graphed[sdpa] | graphed[flash] | graphed[flash] peak |
| --- | --- | --- | --- |
| query B32 S32 | **1.63×** | 1.52× | 372 MB |
| query B256 S32 | **1.45×** | 1.35× | 404 MB |
| doc B32 S320 | 1.41× | **1.45×** | 433 MB |
| doc B32 S512 | 1.45× | **1.63×** | 420 MB |
| doc B8 S2048 | 1.22× | **1.94×** | 488 MB |
| doc B4 S4096 | 1.09× | **2.24×** | 521 MB |

The two backends are **complementary**, crossing over at ≈S320:

- **Short S (queries, S≤~256): use sdpa.** FlashAttention can't prune (window ≥ S
  → full attention) and its fixed launch cost loses to dense SDPA on tiny tensors
  — `fused[flash]` even regresses to 0.77× at B32/S32. `graphed[sdpa]` (1.63×) is
  the best short-S path.
- **Long S (indexing, S≥320): use flash.** Where the sdpa speedup decays with S
  (graphed 1.45→1.09), flash *grows* (graphed 1.63→**2.24×** at S4096), because
  the local-layer O(S²)→O(S·W) win compounds — and `graphed[flash]` stays the
  memory-cheapest path (521 MB at S4096 vs eager-fused 1017, stock 765). So at
  long S, graphed[flash] is simultaneously fastest *and* lightest.

The flash path passes q/k/v as **strided `[B,S,H,D]` views** (a free `transpose`,
no copy): FA only needs the head dim contiguous, which RoPE's contiguous
`[B,H,S,D]` output and v's unbind view both satisfy. Materializing them with
`.contiguous()` instead cost 3 copies/layer and dragged the short/mid-S flash
numbers down (e.g. doc S512 graphed 1.46→1.63× once removed); at long S attention
compute dominates so it was already negligible.

Correctness: the flash forward matches stock HF at cosine 0.9995 (tighter than
sdpa's 0.99918). Measured on a torch-2.8 env with the prebuilt FA 2.8.3 wheel
(runs on sm_120 from PTX — no toolkit build).

Configs: `inference_short.yml` (classic ColBERT Q≈32 / D≈300 — also the roadmap's
C3 short-S config) and `inference_indexing.yml` (long documents). Both take a
`graph:` block (`pad_to`, `max_batch`, `seq_buckets`, `max_graphs`) and run on raw
HF `AutoModel` by default; set `model.framework` to `pylate` /
`sentence_transformers` to drive the real wrapper object.

## encode_transparency.py — graphs engage through the real encode paths

A smoke (non-zero exit on failure) proving what `inference_bench.py` assumes: that
the public text APIs — `AutoModel(...)`, `ColBERT.encode`, `SentenceTransformer.
encode` — route through the captured graph and stay correct. For each framework
it checks (1) a real encode call populates the graph cache (graphs engage because
encode runs no-grad + eval with no autocast), (2) the graphed embeddings match
stock within the validate band (cosine ≥ 0.997), and (3) under autocast the call
falls back to eager (the gate that protects against replaying freed bf16 weight
buffers). Measured: all three frameworks PASS, cosine 0.999+.

```bash
uv run benchmarks/encode_transparency.py            # HF + ST + PyLate
uv run benchmarks/encode_transparency.py --only pylate
```

It uses the PyLate fork (`github.com/pau-mensa/pylate`, autocast-in-CachedContrastive
fix) until that lands upstream.

## Notes on a fair comparison

These are baked into the harness; they matter because a careless setup can hide
the win or inflate it:

- **Sequence length is the dominant variable.** The fused tail wins at S ≥ 512
  and can *lose* at short S (host-launch-bound). Keep document length ≥ 512.
- **Warmup excludes the one-time CuteDSL JIT compile.** Warmup runs
  forward+backward without optimizer steps, so weights stay at the shared init
  and timing/peak-memory reflect steady state.
- **Precision.** `dtype: bfloat16` (the default) runs both variants entirely in
  bf16 — the cleanest, fairest comparison, isolating the kernels. `dtype:
  float32` uses fp32 master weights under bf16 autocast; note that on stock
  PyLate the GradCache pass-3 recompute runs *outside* autocast (a known PyLate
  issue), which the fused model is robust to but the stock model is not — so that
  mode is realistic but not a clean kernel A/B.
- **`cuda_graph` during training** only accelerates the no-grad GradCache pass-1;
  training-step graphs are deferred (the eager fused tail already wins training
  at S ≥ 512). Graphs are primarily an inference lever.
