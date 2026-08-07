"""Non-invasive call tracing for nested generator rollouts.

The tracer intentionally wraps a callable rather than changing model/source code. It
records the execution boundary that a scheduler can observe: pyramid shape, grad mode,
selected-call autograd state, latency, and caller-provided memory snapshots.
"""
from __future__ import annotations

import math
import time
import traceback
from collections.abc import Callable, Mapping
from typing import Any

import torch


def infer_pyramid_stage(
    hidden_states: torch.Tensor,
    *,
    base_height: int,
    base_width: int,
    num_stages: int,
) -> int:
    """Infer a power-of-two pyramid stage from a BCTHW latent tensor."""
    if hidden_states.ndim != 5:
        raise ValueError(f"expected BCTHW hidden_states, got {tuple(hidden_states.shape)}")
    height, width = map(int, hidden_states.shape[-2:])
    first_height = base_height // (2 ** (num_stages - 1))
    first_width = base_width // (2 ** (num_stages - 1))
    if first_height <= 0 or first_width <= 0:
        raise ValueError("base resolution is too small for the requested pyramid")
    height_ratio = height / first_height
    width_ratio = width / first_width
    if height_ratio != width_ratio or height_ratio <= 0:
        raise ValueError(
            f"non-uniform pyramid ratio for {(height, width)} from "
            f"{(first_height, first_width)}"
        )
    stage = round(math.log2(height_ratio))
    if not math.isclose(height_ratio, 2**stage) or not 0 <= stage < num_stages:
        raise ValueError(
            f"shape {(height, width)} is not a valid stage of "
            f"{num_stages}-level base {(base_height, base_width)}"
        )
    return stage


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError("wrapped model output must be a Tensor or start with a Tensor")


def _json_scalar(value: torch.Tensor | None) -> float | int | None:
    if value is None or value.numel() == 0:
        return None
    item = value.detach().flatten()[0].cpu().item()
    return int(item) if isinstance(item, int) else float(item)


class ForceNoGradCalls:
    """Force only the wrapped physical call under no-grad.

    This is intentionally narrower than wrapping an entire logical stage: source-side
    state updates and control flow keep their original grad context, while the physical
    model call cannot construct an autograd graph.
    """

    def __init__(self, target: Callable[..., Any]) -> None:
        self.target = target

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with torch.no_grad():
            return self.target(*args, **kwargs)


class LightweightNestedCallAudit:
    """Audit call count/stage/autograd without timing or CUDA synchronization.

    Formal timing only needs to fail closed on the physical no-grad and
    operation-trace contract.  Unlike :class:`NestedCallTracer`, this wrapper
    performs no per-call device synchronization, allocator reset, memory
    snapshot, timestep CPU copy, or tensor retention.
    """

    def __init__(
        self,
        target: Callable[..., Any],
        *,
        base_height: int,
        base_width: int,
        num_stages: int,
    ) -> None:
        self.target = target
        self.base_height = base_height
        self.base_width = base_width
        self.num_stages = num_stages
        self.calls: list[dict[str, Any]] = []
        self._stage_call_counts = [0 for _ in range(num_stages)]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("LightweightNestedCallAudit requires tensor hidden_states")
        stage = infer_pyramid_stage(
            hidden_states,
            base_height=self.base_height,
            base_width=self.base_width,
            num_stages=self.num_stages,
        )
        denoise_call = self._stage_call_counts[stage]
        self._stage_call_counts[stage] += 1
        grad_enabled = torch.is_grad_enabled()
        output = self.target(*args, **kwargs)
        primary = _first_tensor(output)
        self.calls.append(
            {
                "call_index": len(self.calls),
                "stage": stage,
                "denoise_call_in_stage": denoise_call,
                "hidden_states_shape": list(hidden_states.shape),
                "grad_enabled": grad_enabled,
                "output_shape": list(primary.shape),
                "output_requires_grad": bool(primary.requires_grad),
                "status": "success",
            }
        )
        return output


class NestedCallTracer:
    """Trace every call without detaching or otherwise changing its output graph."""

    def __init__(
        self,
        target: Callable[..., Any],
        *,
        base_height: int,
        base_width: int,
        num_stages: int,
        synchronize: Callable[[], None] | None = None,
        before_call: Callable[[], None] | None = None,
        memory_snapshot: Callable[[], Mapping[str, Any]] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        retain_output_gradients: bool = True,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.target = target
        self.base_height = base_height
        self.base_width = base_width
        self.num_stages = num_stages
        self.synchronize = synchronize or (lambda: None)
        self.before_call = before_call or (lambda: None)
        self.memory_snapshot = memory_snapshot or (lambda: {})
        self.on_event = on_event
        self.retain_output_gradients = retain_output_gradients
        self.clock = clock
        self.calls: list[dict[str, Any]] = []
        self._retained_outputs: list[tuple[int, torch.Tensor]] = []
        self._stage_call_counts = [0 for _ in range(num_stages)]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("NestedCallTracer requires tensor hidden_states")
        stage = infer_pyramid_stage(
            hidden_states,
            base_height=self.base_height,
            base_width=self.base_width,
            num_stages=self.num_stages,
        )
        denoise_call = self._stage_call_counts[stage]
        self._stage_call_counts[stage] += 1
        index = len(self.calls)
        timestep = kwargs.get("timestep")
        grad_enabled = torch.is_grad_enabled()
        self.synchronize()
        self.before_call()
        before = dict(self.memory_snapshot())
        started = self.clock()
        event: dict[str, Any] = {
            "call_index": index,
            "stage": stage,
            "denoise_call_in_stage": denoise_call,
            "hidden_states_shape": list(hidden_states.shape),
            "timestep": _json_scalar(timestep if isinstance(timestep, torch.Tensor) else None),
            "grad_enabled": grad_enabled,
            "is_first_denoising_step": bool(kwargs.get("is_first_denoising_step", False)),
            "memory_before": before,
            "status": "running",
        }
        try:
            output = self.target(*args, **kwargs)
            self.synchronize()
            primary = _first_tensor(output)
            event.update(
                {
                    "status": "success",
                    "duration_s": self.clock() - started,
                    "output_shape": list(primary.shape),
                    "output_requires_grad": bool(primary.requires_grad),
                    "memory_after": dict(self.memory_snapshot()),
                }
            )
            if primary.requires_grad and self.retain_output_gradients:
                primary.retain_grad()
                self._retained_outputs.append((index, primary))
            self.calls.append(event)
            if self.on_event is not None:
                self.on_event(event)
            return output
        except BaseException as exc:
            try:
                self.synchronize()
            except BaseException:
                pass
            event.update(
                {
                    "status": "failed",
                    "duration_s": self.clock() - started,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "memory_after": dict(self.memory_snapshot()),
                }
            )
            self.calls.append(event)
            if self.on_event is not None:
                self.on_event(event)
            raise

    def selected_call_indices(self) -> list[int]:
        return [
            int(event["call_index"])
            for event in self.calls
            if event.get("grad_enabled") and event.get("output_requires_grad")
        ]

    def exit_call_by_stage(self) -> dict[str, int]:
        exits: dict[str, int] = {}
        for event in self.calls:
            if event.get("grad_enabled") and event.get("output_requires_grad"):
                exits[str(event["stage"])] = int(event["denoise_call_in_stage"])
        return exits

    def retained_gradient_summary(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for call_index, tensor in self._retained_outputs:
            gradient = tensor.grad
            summaries.append(
                {
                    "call_index": call_index,
                    "gradient_present": gradient is not None,
                    "gradient_elements": 0 if gradient is None else gradient.numel(),
                    "gradient_abs_sum": (
                        None if gradient is None else float(gradient.detach().float().abs().sum().cpu())
                    ),
                    "gradient_abs_max": (
                        None if gradient is None else float(gradient.detach().float().abs().max().cpu())
                    ),
                }
            )
        return summaries
