"""Opt-in exact rotary patch for the pinned MiniMax-H3 Diffusers source."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
from typing import Any, Callable

import torch


PINNED_ROTARY_SOURCE_SHA256 = "897848a7dac3aed2d8906bdb9c71b0f3f4bcd321fe39ecbd47b67a7203ae7604"
EXPECTED_FUSED_NOGRAD_CALLS_PER_CYCLE = 25 * 50 * 2
EXPECTED_REFERENCE_GRAD_CALLS_PER_CYCLE = 6 * 50 * 2 * 2
EXPECTED_FUSED_GRAD_CALLS_PER_CYCLE = EXPECTED_REFERENCE_GRAD_CALLS_PER_CYCLE
EXPECTED_FUSED_GRAD_BACKWARD_CALLS_PER_CYCLE = 6 * 50 * 2


@dataclasses.dataclass
class FusedRotaryStats:
    fused_nograd_calls: int = 0
    fused_grad_calls: int = 0
    fused_grad_backward_calls: int = 0
    reference_grad_calls: int = 0


@dataclasses.dataclass
class FusedRotaryRegistration:
    enabled: bool
    grad_enabled: bool
    source_sha256: str | None
    stats: FusedRotaryStats
    original: Callable[..., torch.Tensor] | None = None

    def snapshot(self) -> dict[str, int]:
        return dataclasses.asdict(self.stats)

    def receipt(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "grad_enabled": self.grad_enabled,
            "source_sha256": self.source_sha256,
            "stats": self.snapshot(),
        }


def _source_sha256(function: Callable[..., Any]) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


class _FusedRotaryAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        stats: FusedRotaryStats,
    ) -> torch.Tensor:
        if cos.requires_grad or sin.requires_grad:
            raise RuntimeError("H3 fused rotary grad path requires non-trainable cos/sin")
        from .triton_fused_rotary import fused_apply_rotary_emb

        ctx.save_for_backward(cos, sin)
        ctx.stats = stats
        return fused_apply_rotary_emb(hidden_states, cos, sin)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        from .triton_fused_rotary import fused_apply_rotary_emb_backward

        cos, sin = ctx.saved_tensors
        ctx.stats.fused_grad_backward_calls += 1
        grad_hidden = fused_apply_rotary_emb_backward(grad_output.contiguous(), cos, sin)
        return grad_hidden, None, None, None


def install_fused_rotary(
    *, enabled: bool, grad_enabled: bool = False
) -> FusedRotaryRegistration:
    """Patch the module-global rotary helper used by all 50 attention blocks."""

    stats = FusedRotaryStats()
    if grad_enabled and not enabled:
        raise RuntimeError("H3 fused grad rotary requires base rotary fusion enabled")
    if not enabled:
        return FusedRotaryRegistration(False, False, None, stats)

    module = importlib.import_module("diffusers.models.transformers.transformer_minimax_h3")
    original = getattr(module, "_apply_rotary_emb", None)
    if not callable(original):
        raise RuntimeError("pinned MiniMax-H3 module has no callable _apply_rotary_emb")
    if getattr(original, "_h3_fused_rotary_wrapper", False):
        raise RuntimeError("H3 fused rotary was installed more than once")
    source_sha256 = _source_sha256(original)
    if source_sha256 != PINNED_ROTARY_SOURCE_SHA256:
        raise RuntimeError(
            "H3 rotary source identity mismatch: "
            f"observed={source_sha256} expected={PINNED_ROTARY_SOURCE_SHA256}"
        )

    def wrapped(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            if grad_enabled:
                stats.fused_grad_calls += 1
                return _FusedRotaryAutograd.apply(hidden_states, cos, sin, stats)
            stats.reference_grad_calls += 1
            return original(hidden_states, cos, sin)
        from .triton_fused_rotary import fused_apply_rotary_emb

        stats.fused_nograd_calls += 1
        return fused_apply_rotary_emb(hidden_states, cos, sin)

    setattr(wrapped, "_h3_fused_rotary_wrapper", True)
    module._apply_rotary_emb = wrapped
    return FusedRotaryRegistration(True, grad_enabled, source_sha256, stats, original)


def validate_cycle(registration: FusedRotaryRegistration, start: dict[str, int]) -> dict[str, int]:
    """Fail closed if no-grad or grad/replay rotary calls change."""

    if not registration.enabled:
        return {}
    current = registration.snapshot()
    delta = {key: int(current[key]) - int(start.get(key, 0)) for key in current}
    expected = {
        "fused_nograd_calls": EXPECTED_FUSED_NOGRAD_CALLS_PER_CYCLE,
        "fused_grad_calls": (
            EXPECTED_FUSED_GRAD_CALLS_PER_CYCLE if registration.grad_enabled else 0
        ),
        "fused_grad_backward_calls": (
            EXPECTED_FUSED_GRAD_BACKWARD_CALLS_PER_CYCLE if registration.grad_enabled else 0
        ),
        "reference_grad_calls": (
            0 if registration.grad_enabled else EXPECTED_REFERENCE_GRAD_CALLS_PER_CYCLE
        ),
    }
    errors = [
        f"{key}={delta[key]} expected={value}"
        for key, value in expected.items()
        if delta[key] != value
    ]
    if errors:
        raise RuntimeError(f"H3 fused rotary cycle contract failed: {'; '.join(errors)}")
    return delta
