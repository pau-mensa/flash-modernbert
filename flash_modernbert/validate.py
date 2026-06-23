"""`validate()` — the hard gate that decides whether the fused path is safe.

This is the package's headline rule: enabling on a mismatched architecture or an
unvalidated device would produce *wrong embeddings*, not merely slower ones, so
every check below raises rather than degrading silently. (A perf fallback — an
out-of-bucket shape running eager instead of graphed — is fine and lives in the
graph runner; that path is numerically identical.)

The gate covers: the model architecture (and, implicitly, the split-half RoPE
convention the kernels assume), the compute capability against the validated
matrix, that the CuteDSL toolchain can actually JIT a kernel on this machine, and
that the fused forward tracks stock HF within the bf16 band.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flash_modernbert import forward
from flash_modernbert.config import ModernBertParams
from flash_modernbert.errors import ValidationError
from flash_modernbert.locate import find_encoder
from flash_modernbert.state import ATTR

# Compute capabilities the fused tail is confirmed bit-identical on (2026-06-22):
# sm_90 (H100/H200), sm_100 (B200), sm_120 (RTX 5090).
VALIDATED_CAPABILITIES = frozenset({(9, 0), (10, 0), (12, 0)})

MIN_CUTLASS_DSL = (4, 5, 2)
DEFAULT_SEQ_LENS = (128, 512, 2048)
# The bf16 band, not an exactness check. At S>=512 stock HF's own SDPA and eager
# backends only agree to ~0.998 in bf16, and the fused path tracks SDPA at least
# that tightly; 0.997 sits just under that floor so a genuine divergence (which
# lands far lower) is still caught.
DEFAULT_COS_THRESHOLD = 0.997


@dataclass
class ValidationReport:
    model_type: str
    capability: tuple[int, int]
    cutlass_version: str
    cosine: dict[int, float] = field(default_factory=dict)


def validate(
    target: object,
    *,
    seq_lens: tuple[int, ...] = DEFAULT_SEQ_LENS,
    cos_threshold: float = DEFAULT_COS_THRESHOLD,
) -> ValidationReport:
    """Run every gate and return a report, or raise `ValidationError` on a miss."""
    encoder = find_encoder(target)
    params = ModernBertParams.from_hf_config(encoder.config)  # gate 1: architecture
    model_type = encoder.config.model_type

    device = _encoder_device(encoder)  # gate 2: capability
    capability = torch.cuda.get_device_capability(device)
    if capability not in VALIDATED_CAPABILITIES:
        raise ValidationError(
            f"compute capability sm_{capability[0]}{capability[1]} is not in the "
            f"validated set {sorted(VALIDATED_CAPABILITIES)}; refusing to enable "
            "the fused path (it would risk wrong embeddings on an untested arch)"
        )

    cutlass_version = _check_cutlass_toolchain(device)  # gate 3: JIT smoke

    report = ValidationReport(model_type, capability, cutlass_version)
    _check_numerics(encoder, params, device, seq_lens, cos_threshold, report)  # gate 4
    return report


def _encoder_device(encoder: nn.Module) -> torch.device:
    try:
        device = next(encoder.parameters()).device
    except StopIteration as exc:
        raise ValidationError("the model has no parameters to validate") from exc
    if device.type != "cuda":
        raise ValidationError(
            f"the fused path requires CUDA weights; the model is on {device!r}. "
            "Move it to a GPU before calling prepare()."
        )
    return device


def _check_cutlass_toolchain(device: torch.device) -> str:
    import cutlass  # imported lazily so import errors surface here, not at module load

    version = getattr(cutlass, "__version__", "0.0.0")
    if _version_tuple(version) < MIN_CUTLASS_DSL:
        raise ValidationError(
            f"nvidia-cutlass-dsl {version} is below the validated floor "
            f"{'.'.join(map(str, MIN_CUTLASS_DSL))}"
        )
    from flash_modernbert._kernels.layer_norm import layer_norm as _ln_kernel

    try:
        x = torch.randn(32, 256, dtype=torch.bfloat16, device=device)
        w = torch.randn(256, dtype=torch.bfloat16, device=device)
        _ln_kernel(x, w, 1e-5)
        torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001 — surface any JIT/ptxas failure loudly
        raise ValidationError(
            "the CuteDSL toolchain could not compile a smoke kernel on this "
            f"machine (is a matching ptxas on PATH?): {exc}"
        ) from exc
    return version


def _check_numerics(
    encoder: nn.Module,
    params: ModernBertParams,
    device: torch.device,
    seq_lens: tuple[int, ...],
    cos_threshold: float,
    report: ValidationReport,
) -> None:
    dtype = next(encoder.parameters()).dtype
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValidationError(
            f"the fused path requires bf16 or fp32 weights; got {dtype}"
        )
    # fp32 master weights are the bf16-autocast training regime; run the
    # comparison under autocast so both sides see the bf16 the kernels target.
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if dtype == torch.float32
        else contextlib.nullcontext()
    )
    vocab = int(encoder.config.vocab_size)
    generator = torch.Generator(device=device).manual_seed(0)

    oracle = _oracle(encoder)
    with torch.no_grad(), autocast:
        for seq_len in seq_lens:
            ids = torch.randint(5, vocab, (2, seq_len), generator=generator, device=device)
            mask = torch.ones((2, seq_len), dtype=torch.long, device=device)
            expected = oracle(ids, mask)
            actual = forward.fused_forward(encoder, params, ids, mask)
            cos = F.cosine_similarity(
                expected.flatten().float(), actual.flatten().float(), dim=0
            ).item()
            report.cosine[seq_len] = cos
            if cos < cos_threshold:
                raise ValidationError(
                    f"fused forward diverged from stock HF at S={seq_len}: "
                    f"cosine {cos:.6f} < {cos_threshold} — refusing to enable"
                )


def _oracle(encoder: nn.Module):
    """A callable (ids, mask) -> last_hidden_state running the stock HF forward.
    Uses the saved original forward when the encoder is already patched."""
    state = getattr(encoder, ATTR, None)
    hf_forward = state.original_forward if state is not None else encoder.forward

    def run(ids: Tensor, mask: Tensor) -> Tensor:
        out = hf_forward(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    return run


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in version.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)
