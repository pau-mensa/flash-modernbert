"""Per-model state attached by `prepare()`.

`prepare()` patches in place, so the state hangs off the encoder module under one
attribute. Keeping the original forward here makes the patch reversible and gives
`validate()` an oracle (stock HF) to compare the fused path against even after the
swap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from flash_modernbert.config import ModernBertParams
from flash_modernbert.errors import FlashModernBertError
from flash_modernbert.locate import find_encoder

ATTR = "_flash_modernbert"


@dataclass
class PatchState:
    params: ModernBertParams
    original_forward: Callable[..., Any]
    attention_backend: str = "sdpa"  # "sdpa" (default, dep-free) | "flash" | "auto" (sdpa<FLASH_MIN_SEQ, flash above)
    graph_runner: Any = None        # graph._GraphRunner | None (kept loose to avoid a cycle)
    graph_enabled: bool = False
    graph_skip_warned: bool = False  # one-time warning when graphs are skipped (autocast/grad)
    packed_graph_runner: Any = None  # graph._PackedGraphRunner | None
    train_graph_runner: Any = None   # train_graph._TrainGraphRunner | None
    train_graph_enabled: bool = False


def get_state(target: object) -> PatchState:
    """Return the `PatchState` for an already-prepared target, or raise."""
    encoder = find_encoder(target)
    state = getattr(encoder, ATTR, None)
    if not isinstance(state, PatchState):
        raise FlashModernBertError(
            "this model has not been prepared with flash_modernbert.prepare()"
        )
    return state
