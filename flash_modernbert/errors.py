"""Exceptions raised by the package.

`validate()` is a hard gate, not a silent fallback: enabling the fused path on a
mismatched architecture or device would mean *wrong embeddings*, not just slower
ones. Every refusal is one of these, never a quiet downgrade.
"""

from __future__ import annotations


class FlashModernBertError(Exception):
    """Base class for every error this package raises."""


class UnsupportedTargetError(FlashModernBertError):
    """`prepare()` could not locate a ModernBERT encoder inside the target."""


class ValidationError(FlashModernBertError):
    """A `validate()` gate failed — the fused path was refused."""
