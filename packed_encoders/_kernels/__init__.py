"""Vendored CuteDSL kernels for the ModernBERT fused tail.

The fused-tail closure — LayerNorm (fwd + stats), the M-dynamic LayerNorm
backward, RoPE, and GeGLU (fwd + bwd) — plus the compile cache they share.
The numerical bodies are carried over verbatim from the research monorepo; only
their package path changed. Nothing here imports framework code, so the kernels
stay a self-contained substrate the rest of the package builds glue on top of.
"""
