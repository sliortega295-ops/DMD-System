"""Opt-in SwiGLU fusion for the 50 pinned MiniMax-H3 main blocks."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import types
from typing import Any, Callable

import torch


PINNED_SWIGLU_FORWARD_SHA256 = "c229f3514b2da3ce3c0e81a9fdb79c510a0665dcb73cb6cc61d1e6c58cf7aa19"
EXPECTED_BLOCK_COUNT = 50
EXPECTED_NOGRAD_CALLS_PER_CYCLE = 25 * EXPECTED_BLOCK_COUNT
EXPECTED_GRAD_REPLAY_CALLS_PER_CYCLE = 6 * EXPECTED_BLOCK_COUNT * 2
EXPECTED_BACKWARD_CALLS_PER_CYCLE = 6 * EXPECTED_BLOCK_COUNT


@dataclasses.dataclass
class FusedSwiGLUStats:
    fused_nograd_calls: int = 0
    fused_grad_calls: int = 0
    fused_backward_calls: int = 0
    reference_grad_calls: int = 0


@dataclasses.dataclass
class FusedSwiGLURegistration:
    enabled: bool
    grad_enabled: bool
    source_sha256: str | None
    module_count: int
    stats: FusedSwiGLUStats

    def snapshot(self) -> dict[str, int]:
        return dataclasses.asdict(self.stats)

    def receipt(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "grad_enabled": self.grad_enabled,
            "source_sha256": self.source_sha256,
            "module_count": self.module_count,
            "stats": self.snapshot(),
        }


def _source_sha256(function: Callable[..., Any]) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


class _FusedSwiGLUAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, stats):
        from .triton_fused_swiglu import fused_swiglu

        ctx.save_for_backward(projected)
        ctx.stats = stats
        return fused_swiglu(projected)

    @staticmethod
    def backward(ctx, grad_output):
        from .triton_fused_swiglu import fused_swiglu_backward

        (projected,) = ctx.saved_tensors
        ctx.stats.fused_backward_calls += 1
        grad_projected = fused_swiglu_backward(grad_output.contiguous(), projected)
        return grad_projected, None


def _patched_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    stats: FusedSwiGLUStats = self._h3_fused_swiglu_stats
    if torch.is_grad_enabled():
        if not self._h3_fused_swiglu_grad_enabled:
            stats.reference_grad_calls += 1
            return self._h3_original_swiglu_forward(hidden_states)
        # Count entry because non-reentrant checkpoint replay may exit early.
        stats.fused_grad_calls += 1
        projected = self.proj(hidden_states)
        return _FusedSwiGLUAutograd.apply(projected, stats)

    from .triton_fused_swiglu import fused_swiglu

    stats.fused_nograd_calls += 1
    return fused_swiglu(self.proj(hidden_states))


def install_fused_swiglu(
    transformer: torch.nn.Module, *, enabled: bool, grad_enabled: bool = False
) -> FusedSwiGLURegistration:
    """Patch only the main transformer's 50 SwiGLU instances."""

    blocks = list(getattr(transformer, "transformer_blocks", ()))
    if not enabled:
        if grad_enabled:
            raise RuntimeError("H3 fused grad SwiGLU requires base SwiGLU fusion enabled")
        return FusedSwiGLURegistration(False, False, None, 0, FusedSwiGLUStats())
    if len(blocks) != EXPECTED_BLOCK_COUNT:
        raise RuntimeError(
            f"H3 fused SwiGLU requires {EXPECTED_BLOCK_COUNT} blocks, got {len(blocks)}"
        )

    modules = []
    for index, block in enumerate(blocks):
        net = getattr(getattr(block, "ff", None), "net", None)
        if net is None or len(net) < 1:
            raise RuntimeError(f"H3 block {index} has no pinned ff.net[0]")
        module = net[0]
        if type(module).__name__ != "SwiGLU":
            raise RuntimeError(
                f"H3 block {index} ff.net[0] is {type(module).__name__}, expected SwiGLU"
            )
        modules.append(module)
    module_type = type(modules[0])
    if any(type(module) is not module_type for module in modules):
        raise RuntimeError("H3 fused SwiGLU requires one homogeneous pinned module type")
    source_sha256 = _source_sha256(module_type.forward)
    if source_sha256 != PINNED_SWIGLU_FORWARD_SHA256:
        raise RuntimeError(
            "H3 SwiGLU source identity mismatch: "
            f"observed={source_sha256} expected={PINNED_SWIGLU_FORWARD_SHA256}"
        )

    stats = FusedSwiGLUStats()
    for module in modules:
        if hasattr(module, "_h3_original_swiglu_forward"):
            raise RuntimeError("H3 fused SwiGLU was installed more than once")
        object.__setattr__(module, "_h3_original_swiglu_forward", module.forward)
        object.__setattr__(module, "_h3_fused_swiglu_stats", stats)
        object.__setattr__(module, "_h3_fused_swiglu_grad_enabled", bool(grad_enabled))
        object.__setattr__(module, "forward", types.MethodType(_patched_forward, module))
    return FusedSwiGLURegistration(
        True, bool(grad_enabled), source_sha256, len(modules), stats
    )


def validate_cycle(
    registration: FusedSwiGLURegistration, start: dict[str, int]
) -> dict[str, int]:
    """Fail closed if any main-block SwiGLU call escapes the selected path."""

    if not registration.enabled:
        return {}
    current = registration.snapshot()
    delta = {key: int(current[key]) - int(start.get(key, 0)) for key in current}
    expected = {
        "fused_nograd_calls": EXPECTED_NOGRAD_CALLS_PER_CYCLE,
        "fused_grad_calls": (
            EXPECTED_GRAD_REPLAY_CALLS_PER_CYCLE if registration.grad_enabled else 0
        ),
        "fused_backward_calls": (
            EXPECTED_BACKWARD_CALLS_PER_CYCLE if registration.grad_enabled else 0
        ),
        "reference_grad_calls": (
            0 if registration.grad_enabled else EXPECTED_GRAD_REPLAY_CALLS_PER_CYCLE
        ),
    }
    errors = [
        f"{key}={delta[key]} expected={value}"
        for key, value in expected.items()
        if delta[key] != value
    ]
    if errors:
        raise RuntimeError(f"H3 fused SwiGLU cycle contract failed: {'; '.join(errors)}")
    return delta
