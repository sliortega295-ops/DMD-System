"""Bounded source attribution for one known Critic D2H copy shape.

The tracer is diagnostic-only and deliberately narrow.  It does not install
``saved_tensors_hooks`` (which would change the hook stack being diagnosed),
retain tensors, synchronize CUDA, or intercept unrelated copies.  On one
representative rank it records Python call stacks only for CUDA-to-CPU ATen
copies whose logical size matches the previously observed Critic tensor.
"""
from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils._python_dispatch import TorchDispatchMode


DEFAULT_TARGET_BYTES = 113_049_600
_CHECKPOINT_PHASE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "distillschedule_checkpoint_phase", default="outside_checkpoint"
)


def _now_ns() -> int:
    return time.monotonic_ns()


def _device(value: Any) -> str | None:
    try:
        device = getattr(value, "device", None)
        return None if device is None else str(device)
    except BaseException:
        return None


def _shape(value: Any) -> list[int] | None:
    try:
        shape = getattr(value, "shape", None)
        return None if shape is None else [int(x) for x in shape]
    except BaseException:
        return None


def _dtype(value: Any) -> str | None:
    try:
        dtype = getattr(value, "dtype", None)
        return None if dtype is None else str(dtype)
    except BaseException:
        return None


def _bytes(value: Any) -> int | None:
    try:
        return int(value.numel() * value.element_size())
    except BaseException:
        return None


def _stack(limit: int = 32) -> list[str]:
    # Store strings only: frames/locals can retain tensors and alter lifetimes.
    return traceback.format_stack(limit=limit)[:-1]


@contextlib.contextmanager
def operation_source_marker(label: str) -> Iterator[None]:
    enabled = torch.cuda.is_available()
    if enabled:
        torch.cuda.nvtx.range_push(str(label))
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


@contextlib.contextmanager
def _checkpoint_context(phase: str) -> Iterator[None]:
    token = _CHECKPOINT_PHASE.set(phase)
    try:
        with operation_source_marker(f"CHECKPOINT/{phase}"):
            yield
    finally:
        _CHECKPOINT_PHASE.reset(token)


def checkpoint_context_fn() -> tuple[Any, Any]:
    """Return non-reentrant checkpoint contexts without changing saved hooks."""

    return (
        _checkpoint_context("original_forward"),
        _checkpoint_context("recompute"),
    )


@dataclasses.dataclass
class OperationSourceTrace:
    rank: int
    target_bytes: int = DEFAULT_TARGET_BYTES
    max_events: int = 256
    events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    matched_events: int = 0
    dropped_events: int = 0

    def append(self, event: dict[str, Any]) -> None:
        self.matched_events += 1
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return
        self.events.append(
            {"ts_monotonic_ns": _now_ns(), "rank": self.rank, **event}
        )

    def dump(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        value = {
            "schema_version": "distillschedule.fsdp2_operation_source.v2",
            "rank": self.rank,
            "target_bytes": self.target_bytes,
            "max_events": self.max_events,
            "matched_events": self.matched_events,
            "recorded_events": len(self.events),
            "dropped_events": self.dropped_events,
            "metadata": metadata or {},
            "events": self.events,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)


def _requested_device(kwargs: dict[str, Any]) -> str | None:
    value = kwargs.get("device")
    if value is None:
        return None
    try:
        return str(torch.device(value))
    except (TypeError, ValueError, RuntimeError):
        return str(value)


class TargetedD2HTraceMode(TorchDispatchMode):
    """Record only the exact known CUDA-to-pageable-CPU copy candidates."""

    def __init__(self, trace: OperationSourceTrace) -> None:
        super().__init__()
        self.trace = trace

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: Any = (),
        kwargs: Any = None,
    ) -> Any:
        del types
        kwargs = {} if kwargs is None else kwargs
        name = str(func)
        source: Any | None = None
        target: Any | None = None
        requested_target: str | None = None
        if "aten._to_copy" in name and args:
            source = args[0]
            requested_target = _requested_device(kwargs)
        elif "aten.copy_" in name and len(args) >= 2:
            target, source = args[0], args[1]
            requested_target = _device(target)
        else:
            return func(*args, **kwargs)

        source_device = _device(source)
        logical_bytes = _bytes(source)
        candidate = (
            source_device is not None
            and source_device.startswith("cuda")
            and requested_target == "cpu"
            and logical_bytes == self.trace.target_bytes
        )
        if not candidate:
            return func(*args, **kwargs)

        stack = _stack()
        started = _now_ns()
        with operation_source_marker(
            f"TARGET_D2H/{logical_bytes}/{_CHECKPOINT_PHASE.get()}"
        ):
            result = func(*args, **kwargs)
        if target is None:
            target = result
        self.trace.append(
            {
                "event": "TARGET_D2H_ATEN_COPY",
                "op": name,
                "op_start_monotonic_ns": started,
                "op_end_monotonic_ns": _now_ns(),
                "checkpoint_phase": _CHECKPOINT_PHASE.get(),
                "source_device": source_device,
                "target_device": _device(target),
                "source_shape": _shape(source),
                "target_shape": _shape(target),
                "source_dtype": _dtype(source),
                "target_dtype": _dtype(target),
                "logical_bytes": logical_bytes,
                "stack": stack,
            }
        )
        return result


class OperationSourceCapture:
    """Scope the targeted copy mode to one Critic update on one rank."""

    def __init__(
        self,
        *,
        enabled: bool,
        rank: int,
        target_bytes: int = DEFAULT_TARGET_BYTES,
        max_events: int = 256,
    ) -> None:
        if target_bytes <= 0 or max_events <= 0:
            raise ValueError("target_bytes and max_events must be positive")
        self.enabled = bool(enabled)
        self.trace = OperationSourceTrace(
            rank=int(rank), target_bytes=int(target_bytes), max_events=int(max_events)
        )
        self._mode: TargetedD2HTraceMode | None = None

    def __enter__(self) -> "OperationSourceCapture":
        if self.enabled:
            self._mode = TargetedD2HTraceMode(self.trace)
            self._mode.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._mode is not None:
            self._mode.__exit__(exc_type, exc, tb)
            self._mode = None


__all__ = [
    "DEFAULT_TARGET_BYTES",
    "OperationSourceCapture",
    "OperationSourceTrace",
    "TargetedD2HTraceMode",
    "checkpoint_context_fn",
    "operation_source_marker",
]
