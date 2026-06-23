"""flash-modernbert — a fast, monkeypatching ModernBERT/mmBERT encoder.

`prepare(model)` installs a validated fused-tail forward (CuteDSL LayerNorm,
RoPE, and GeGLU; cuBLAS GEMMs; vendor SDPA attention) onto a live Hugging Face
`ModernBertModel` in place, so HF, SentenceTransformers, and PyLate all inherit
the speedup with no per-framework adapter. CUDA graphs are an optional layer on
top, off by default.

    import flash_modernbert as fm
    fm.prepare(model)                 # eager fused forward
    fm.prepare(model, cuda_graph=True)  # bucketed CUDA graphs
"""

from __future__ import annotations

from flash_modernbert.errors import (
    FlashModernBertError,
    UnsupportedTargetError,
    ValidationError,
)
from flash_modernbert.graph import GraphConfig, no_cuda_graph, set_cuda_graph
from flash_modernbert.prepare import prepare, unprepare
from flash_modernbert.validate import ValidationReport, validate

__all__ = [
    "prepare",
    "unprepare",
    "validate",
    "ValidationReport",
    "GraphConfig",
    "set_cuda_graph",
    "no_cuda_graph",
    "FlashModernBertError",
    "UnsupportedTargetError",
    "ValidationError",
]
