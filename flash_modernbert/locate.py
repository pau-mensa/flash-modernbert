"""Find the ModernBERT encoder inside whatever the caller hands `prepare()`.

HF `AutoModel`, SentenceTransformers, and PyLate ColBERT all ultimately hold one
`ModernBertModel`. This walks the known wrapper shapes to it and raises on an unknown
container rather than guessing — a wrong guess would patch the wrong weights.
"""

from __future__ import annotations

from torch import nn

from flash_modernbert.config import SUPPORTED_MODEL_TYPES
from flash_modernbert.errors import UnsupportedTargetError

# Attribute chains a wrapper uses to reach the backbone, most-specific first.
# `auto_model`: SentenceTransformers Transformer / PyLate. `model`: HF task heads
# (ModernBertForMaskedLM, ...). `modernbert`/`bert`: occasional custom heads.
_WRAPPER_ATTRS = ("auto_model", "model", "modernbert", "bert", "encoder")


def is_modernbert_encoder(module: object) -> bool:
    """True for a `ModernBertModel`-shaped backbone: a supported `model_type` and
    the embeddings / layers / final_norm trio the forward reads."""
    if not isinstance(module, nn.Module):
        return False
    config = getattr(module, "config", None)
    if getattr(config, "model_type", None) not in SUPPORTED_MODEL_TYPES:
        return False
    return (
        hasattr(module, "embeddings")
        and hasattr(module, "layers")
        and hasattr(module, "final_norm")
    )


def find_encoder(target: object, *, _depth: int = 0) -> nn.Module:
    """Return the `ModernBertModel` reachable from `target`, or raise."""
    if is_modernbert_encoder(target):
        return target  # type: ignore[return-value]

    if _depth >= 4:
        raise UnsupportedTargetError(_describe(target))

    # SentenceTransformers / PyLate: the first pipeline module carries auto_model.
    first = _first_submodule(target)
    if first is not None and first is not target:
        try:
            return find_encoder(first, _depth=_depth + 1)
        except UnsupportedTargetError:
            pass

    for attr in _WRAPPER_ATTRS:
        sub = getattr(target, attr, None)
        if isinstance(sub, nn.Module) and sub is not target:
            try:
                return find_encoder(sub, _depth=_depth + 1)
            except UnsupportedTargetError:
                continue

    raise UnsupportedTargetError(_describe(target))


def _first_submodule(target: object):
    """`target[0]` for indexable containers (ST `SentenceTransformer` is one),
    without assuming the framework is installed."""
    if not hasattr(target, "__getitem__"):
        return None
    try:
        return target[0]
    except (KeyError, IndexError, TypeError):
        return None


def _describe(target: object) -> str:
    return (
        f"could not locate a ModernBERT encoder in {type(target).__name__!r}. "
        "flash-modernbert patches a Hugging Face ModernBertModel, a "
        "SentenceTransformer / PyLate ColBERT wrapping one, or a task model "
        "exposing it via .auto_model / .model. Pass the encoder directly if it "
        "lives somewhere else."
    )
