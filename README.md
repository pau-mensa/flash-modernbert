# packed-encoders

A fast, monkeypatching encoder for ModernBERT / mmBERT.

`pack(model)` installs a validated **fused-tail** forward — fused LayerNorm,
RoPE, and GeGLU kernels, cuBLAS GEMMs, and packed Triton/Flash attention — onto a live
Hugging Face `ModernBertModel` **in place**. Because the kernels already consume
HF's exact weight layout, nothing is re-packed: `state_dict`, `save_pretrained`,
`from_pretrained`, and gradient checkpointing all stay HF's own, and Hugging
Face, SentenceTransformers, and PyLate inherit the speedup with no per-framework
adapter.

```python
import packed_encoders as pe
from transformers import AutoModel

model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", dtype="bfloat16").cuda()
pe.pack(model)                      # eager fused forward (default)
out = model(input_ids=ids, attention_mask=mask).last_hidden_state
```

Attention can be selected with `attention_backend="auto"`, `"triton"`, `"flash"`,
or `"sdpa"`. Auto is the default when an optimized kernel is available: on the
calibrated RTX 5090 it uses the specialized packed Triton kernel below 20,736 live
tokens and FA2 above that boundary. Exact-card policies are also calibrated for A100,
L40S, H200, and B200; unmeasured cards conservatively prefer Flash. SDPA remains the
explicit dependency-free option and final fallback.

It also works through the framework wrappers — `pack()` locates the encoder
inside a `SentenceTransformer` or a PyLate `ColBERT` and patches it:

```python
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("some-modernbert-model").cuda().bfloat16()
pe.pack(st)
emb = st.encode(texts)
```

## CUDA graphs (optional, off by default)

For the short-sequence regime where the eager fused tail is host-launch-bound,
the bucketed CUDA-graph runner collapses the launch floor to a single replay:

```python
# Rectangular graph (dynamic padding remains a dense SDPA mask):
pe.pack(model, cuda_graph=True, attention_backend="sdpa")

# Already-packed auto/Triton graph needs fixed sequence-count and S bounds:
pe.pack(model, cuda_graph=pe.GraphConfig(
    pad_to=64, max_batch=256, max_seq=128, max_graphs=32,
))

with pe.no_cuda_graph(model):          # bypass graphs for a one-off odd shape
    out = model(input_ids=huge_ids, attention_mask=huge_mask)
pe.set_cuda_graph(model, False)        # or flip after the fact
```

The kill switch `PACKED_ENCODERS_GRAPH=0` disables graphs globally. Out-of-bucket
or oversized shapes fall back to the eager forward automatically — numerically
identical, just un-graphed. `auto`/`triton` do not use the rectangular graph runner:
their capture-safe dispatch applies to the already-packed entry, whose fixed graph
bucket provides the token budget without reading dynamic lengths back to the host.

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

- An NVIDIA GPU of compute capability sm_80, sm_89, sm_90, sm_100, or sm_120, with a working
  `ptxas` (CuteDSL JIT-compiles at runtime).
- bf16 weights for inference, or fp32 master weights under bf16 autocast for
  training.

## Scope

Supports the ModernBERT architecture (ModernBERT, mmBERT) at `hidden_size`
divisible by 256. Does **not** support `inputs_embeds`, `output_attentions`,
`output_hidden_states`, or non-zero `token_type_ids`.
