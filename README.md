# packed-encoders

A fast, monkeypatching encoder for ModernBERT / mmBERT.

`pack(model)` installs a validated **fused-tail** forward — CuteDSL LayerNorm,
RoPE, and GeGLU kernels, cuBLAS GEMMs, and vendor SDPA attention — onto a live
Hugging Face `ModernBertModel` **in place**. Because the kernels already consume
HF's exact weight layout, nothing is re-packed: `state_dict`, `save_pretrained`,
`from_pretrained`, and gradient checkpointing all stay HF's own, and Hugging
Face, SentenceTransformers, and PyLate inherit the speedup with no per-framework
adapter.

```python
import packed_encoders as fm
from transformers import AutoModel

model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", dtype="bfloat16").cuda()
fm.pack(model)                      # eager fused forward (default)
out = model(input_ids=ids, attention_mask=mask).last_hidden_state
```

It also works through the framework wrappers — `pack()` locates the encoder
inside a `SentenceTransformer` or a PyLate `ColBERT` and patches it:

```python
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("some-modernbert-model").cuda().bfloat16()
fm.pack(st)
emb = st.encode(texts)
```

## CUDA graphs (optional, off by default)

For the short-sequence regime where the eager fused tail is host-launch-bound,
the bucketed CUDA-graph runner collapses the launch floor to a single replay:

```python
fm.pack(model, cuda_graph=True)
# or with an explicit bucketing policy:
fm.pack(model, cuda_graph=fm.GraphConfig(pad_to=64, max_graphs=32))

with fm.no_cuda_graph(model):          # bypass graphs for a one-off odd shape
    out = model(input_ids=huge_ids, attention_mask=huge_mask)
fm.set_cuda_graph(model, False)        # or flip after the fact
```

The kill switch `PACKED_ENCODERS_GRAPH=0` disables graphs globally. Out-of-bucket
or oversized shapes fall back to the eager forward automatically — numerically
identical, just un-graphed.

## `validate()` — a hard gate, not a silent fallback

`pack()` runs `validate()` by default (`validate=False` to skip). It **raises**
rather than degrading silently, because enabling on a mismatched architecture or
device would mean *wrong embeddings*, not just slower ones. It checks:

1. the model architecture (`config.model_type` — and with it the split-half RoPE
   convention the kernels assume),
2. the compute capability against the validated matrix — sm_90, sm_100, sm_120,
3. that the CuteDSL toolchain can JIT a kernel on this machine,
4. that the fused forward tracks stock HF within the bf16 band.

## Requirements

- An NVIDIA GPU of compute capability sm_90, sm_100, or sm_120, with a working
  `ptxas` (CuteDSL JIT-compiles at runtime).
- bf16 weights for inference, or fp32 master weights under bf16 autocast for
  training.

## Scope

Supports the ModernBERT architecture (ModernBERT, mmBERT) at `hidden_size`
divisible by 256. Does **not** support `inputs_embeds`, `output_attentions`,
`output_hidden_states`, or non-zero `token_type_ids`.
