"""packed-encoders — a fast, monkeypatching ModernBERT/mmBERT encoder.

`pack(model)` installs a validated fused-tail forward (CuteDSL LayerNorm,
RoPE, and GeGLU; cuBLAS GEMMs; vendor SDPA attention) onto a live Hugging Face
`ModernBertModel` in place, so HF, SentenceTransformers, and PyLate all inherit
the speedup with no per-framework adapter. CUDA graphs are an optional layer on
top, off by default.

    import packed_encoders as fm
    fm.pack(model)                 # eager fused forward
    fm.pack(model, cuda_graph=True)  # bucketed CUDA graphs
"""

from __future__ import annotations

from packed_encoders.errors import (
    PackedEncodersError,
    UnsupportedTargetError,
    ValidationError,
)
from packed_encoders.graph import GraphConfig, no_cuda_graph, set_cuda_graph
from packed_encoders.pack import pack, unpack
from packed_encoders.train_graph import TrainGraphConfig, set_train_cuda_graph
from packed_encoders.validate import ValidationReport, validate

__all__ = [
    "pack",
    "unpack",
    "validate",
    "ValidationReport",
    "GraphConfig",
    "set_cuda_graph",
    "no_cuda_graph",
    "TrainGraphConfig",
    "set_train_cuda_graph",
    "PackedEncodersError",
    "UnsupportedTargetError",
    "ValidationError",
]
