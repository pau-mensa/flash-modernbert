"""Per-model state attached by `pack()`.

`pack()` patches in place, so the state hangs off the encoder module under one
attribute. Keeping the original forward here makes the patch reversible and gives
`validate()` an oracle (stock HF) to compare the fused path against even after the
swap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from packed_encoders.config import ModernBertParams
from packed_encoders.errors import PackedEncodersError
from packed_encoders.locate import find_encoder

ATTR = "_packed_encoders"


@dataclass
class PatchState:
    params: ModernBertParams
    original_forward: Callable[..., Any]
    # "sdpa" (general fallback) | "flash" | "triton" | "auto" (packed score)
    attention_backend: str = "sdpa"
    graph_runner: Any = None        # graph._GraphRunner | None (kept loose to avoid a cycle)
    graph_enabled: bool = False
    graph_skip_warned: bool = False  # one-time warning when graphs are skipped (autocast/grad)
    packed_graph_runner: Any = None  # graph._PackedGraphRunner | None
    train_graph_runner: Any = None   # train_graph._TrainGraphRunner | None
    train_graph_enabled: bool = False


def get_state(target: object) -> PatchState:
    """Return the `PatchState` for an already-patched target, or raise."""
    encoder = find_encoder(target)
    state = getattr(encoder, ATTR, None)
    if not isinstance(state, PatchState):
        raise PackedEncodersError(
            "this model has not been patched with packed_encoders.pack()"
        )
    return state
