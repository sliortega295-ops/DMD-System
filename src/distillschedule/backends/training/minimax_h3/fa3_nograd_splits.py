"""Opt-in FA3 split scheduling for the pinned Hub kernel.

The original candidate rewrote only no-grad calls.  ``grad_num_splits`` is an
independent, default-off extension so checkpoint forward/replay calls can be
measured without changing the already validated no-grad policy.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable

import torch


# The controlled cycle contains 25 no-grad DiT calls and 12 grad/replay DiT
# executions. MiniMax-H3 has 50 main plus 2 refiner attention layers.
EXPECTED_NOGRAD_CALLS_PER_CYCLE = 25 * 52
EXPECTED_GRAD_CALLS_PER_CYCLE = 12 * 52


@dataclasses.dataclass
class FA3NoGradSplitStats:
    total_calls: int = 0
    no_grad_calls: int = 0
    grad_calls: int = 0
    rewritten_calls: int = 0
    no_grad_rewritten_calls: int = 0
    grad_rewritten_calls: int = 0
    unexpected_input_num_splits: int = 0


@dataclasses.dataclass
class FA3NoGradSplitRegistration:
    enabled: bool
    num_splits: int
    grad_num_splits: int
    original_module: str | None
    original_qualname: str | None
    stats: FA3NoGradSplitStats

    def snapshot(self) -> dict[str, int]:
        return dataclasses.asdict(self.stats)

    def receipt(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "num_splits": self.num_splits,
            "grad_num_splits": self.grad_num_splits,
            "original_module": self.original_module,
            "original_qualname": self.original_qualname,
            "stats": self.snapshot(),
        }


def install_fa3_nograd_splits(
    *,
    num_splits: int,
    grad_num_splits: int = 1,
    kernel_config: Any | None = None,
) -> FA3NoGradSplitRegistration:
    """Rewrite selected FA3 calls from ``num_splits=1`` to split two.

    The Diffusers backend, model processor, Q/K/V, mask, and scale remain
    unchanged. The grad path scheduling is touched only when
    ``grad_num_splits=2``.
    Passing kernel_config is supported only to make the wrapper independently
    testable without importing Diffusers.
    """

    num_splits = int(num_splits)
    grad_num_splits = int(grad_num_splits)
    if num_splits not in {1, 2}:
        raise ValueError(
            "The bounded H3 FA3 no-grad candidate only authorizes "
            f"num_splits=1 or 2, got {num_splits}"
        )
    if grad_num_splits not in {1, 2}:
        raise ValueError(
            "The bounded H3 FA3 grad candidate only authorizes num_splits=1 "
            f"or 2, got {grad_num_splits}"
        )
    if num_splits == 1 and grad_num_splits == 1:
        return FA3NoGradSplitRegistration(
            False, 1, 1, None, None, FA3NoGradSplitStats()
        )
    if kernel_config is None:
        from diffusers.models.attention_dispatch import (
            AttentionBackendName,
            _HUB_KERNELS_REGISTRY,
        )

        kernel_config = _HUB_KERNELS_REGISTRY[AttentionBackendName._FLASH_3_HUB]
    original: Callable[..., Any] | None = getattr(kernel_config, "kernel_fn", None)
    if original is None:
        raise RuntimeError(
            "H3 FA3 no-grad split candidate requires the pinned local Hub "
            "kernel to be bound before installation"
        )
    if getattr(original, "_h3_fa3_nograd_split_wrapper", False):
        raise RuntimeError("H3 FA3 no-grad split wrapper was installed more than once")

    stats = FA3NoGradSplitStats()

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        stats.total_calls += 1
        observed = int(kwargs.get("num_splits", 1))
        if observed != 1:
            stats.unexpected_input_num_splits += 1
            raise RuntimeError(
                "H3 FA3 split candidate expected Diffusers num_splits=1, "
                f"observed {observed}"
            )
        if torch.is_grad_enabled():
            stats.grad_calls += 1
            if grad_num_splits == 2:
                kwargs["num_splits"] = 2
                stats.rewritten_calls += 1
                stats.grad_rewritten_calls += 1
            return original(*args, **kwargs)
        stats.no_grad_calls += 1
        if num_splits == 2:
            kwargs["num_splits"] = 2
            stats.rewritten_calls += 1
            stats.no_grad_rewritten_calls += 1
        return original(*args, **kwargs)

    wrapped._h3_fa3_nograd_split_wrapper = True
    kernel_config.kernel_fn = wrapped
    return FA3NoGradSplitRegistration(
        True,
        num_splits,
        grad_num_splits,
        getattr(original, "__module__", None),
        getattr(original, "__qualname__", None),
        stats,
    )


def validate_cycle(
    registration: FA3NoGradSplitRegistration, start: dict[str, int]
) -> dict[str, int]:
    if not registration.enabled:
        return {}
    current = registration.snapshot()
    delta = {key: int(current[key]) - int(start.get(key, 0)) for key in current}
    expected = {
        "total_calls": EXPECTED_NOGRAD_CALLS_PER_CYCLE + EXPECTED_GRAD_CALLS_PER_CYCLE,
        "no_grad_calls": EXPECTED_NOGRAD_CALLS_PER_CYCLE,
        "grad_calls": EXPECTED_GRAD_CALLS_PER_CYCLE,
        "rewritten_calls": (
            (EXPECTED_NOGRAD_CALLS_PER_CYCLE if registration.num_splits == 2 else 0)
            + (
                EXPECTED_GRAD_CALLS_PER_CYCLE
                if registration.grad_num_splits == 2
                else 0
            )
        ),
        "no_grad_rewritten_calls": (
            EXPECTED_NOGRAD_CALLS_PER_CYCLE if registration.num_splits == 2 else 0
        ),
        "grad_rewritten_calls": (
            EXPECTED_GRAD_CALLS_PER_CYCLE
            if registration.grad_num_splits == 2
            else 0
        ),
        "unexpected_input_num_splits": 0,
    }
    errors = [
        f"{key}={delta[key]} expected={value}"
        for key, value in expected.items()
        if delta[key] != value
    ]
    if errors:
        raise RuntimeError(f"H3 FA3 no-grad split cycle contract failed: {'; '.join(errors)}")
    return delta
