"""Explicit physical placement for the frozen sixteen-sample workload.

The default is the original A100 world16 path. Hardware is never inferred
from visible devices. A smaller physical mesh must explicitly accumulate
the original rank-qualified B1 rows before committing an optimizer update.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

PROFILE_ENV = "DISTILLSCHEDULE_HARDWARE_PROFILE"


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    physical_world_size: int
    global_batch: int = 16

    @property
    def microbatch_count(self) -> int:
        return self.global_batch // self.physical_world_size

    def logical_slots(self, physical_rank: int) -> tuple[int, ...]:
        if not 0 <= physical_rank < self.physical_world_size:
            raise ValueError(f"invalid physical rank {physical_rank} for {self.name}")
        return tuple(physical_rank + i * self.physical_world_size
                     for i in range(self.microbatch_count))

    def mesh_shape(self, policy: str) -> tuple[int, ...]:
        if policy == "world16-fsdp":
            return (self.physical_world_size,)
        if policy == "node-hsdp":
            return (2, self.physical_world_size // 2)
        raise ValueError(f"unsupported mesh policy {policy!r}")


def hardware_profile(name: str | None = None) -> HardwareProfile:
    name = name or os.environ.get(PROFILE_ENV, "a100-16-b16")
    profiles = {"a100-16-b16": 16, "h100-8-b16": 8}
    if name not in profiles:
        raise ValueError(f"unknown {PROFILE_ENV}={name!r}")
    return HardwareProfile(name=name, physical_world_size=profiles[name])


def runtime_physical_world_size() -> int:
    return hardware_profile().physical_world_size


class MicrobatchCommitEngine:
    """Scale each logical B1 loss once and commit only at the outer boundary.

    An inner GAN helper may toggle its denoise/GAN boundary, but it cannot
    turn the first physical rank's B1 row into an optimizer commit. The
    wrapped engine remains responsible for its native gradient reduction.
    """

    def __init__(self, engine: Any, *, microbatch_count: int):
        if not isinstance(microbatch_count, int) or microbatch_count < 1:
            raise ValueError("microbatch_count must be a positive integer")
        self.engine = engine
        self.microbatch_count = microbatch_count
        self._next_index = 0
        self._active_index: int | None = None
        self._inner_boundary = True
        self._step_seen = False
        self._failed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.engine(*args, **kwargs)

    @contextmanager
    def microbatch(self, index: int) -> Iterator["MicrobatchCommitEngine"]:
        if self._failed or self._active_index is not None or index != self._next_index:
            raise RuntimeError(f"microbatch sequence violation: expected {self._next_index}, got {index}")
        self._active_index = index
        self._inner_boundary = True
        self._step_seen = False
        self._set_raw_boundary()
        try:
            yield self
            if not self._step_seen:
                raise RuntimeError("logical microbatch did not reach its step boundary")
        except BaseException:
            self._failed = True
            raise
        else:
            self._next_index = (index + 1) % self.microbatch_count
        finally:
            self._active_index = None

    def _require_active(self) -> None:
        if self._active_index is None or self._failed:
            raise RuntimeError("backward/step requires an active microbatch")

    def is_gradient_accumulation_boundary(self) -> bool:
        self._require_active()
        return self._active_index == self.microbatch_count - 1 and self._inner_boundary

    def _set_raw_boundary(self) -> None:
        setter = getattr(self.engine, "set_gradient_accumulation_boundary", None)
        if setter is not None:
            setter(is_boundary=self.is_gradient_accumulation_boundary())

    def set_gradient_accumulation_boundary(self, *, is_boundary: bool) -> None:
        self._require_active()
        self._inner_boundary = bool(is_boundary)
        self._set_raw_boundary()

    def backward(self, loss: Any, *args: Any, **kwargs: Any) -> Any:
        self._require_active()
        if self._step_seen:
            raise RuntimeError("backward after a microbatch step")
        return self.engine.backward(loss / self.microbatch_count, *args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        self._require_active()
        if self._step_seen:
            raise RuntimeError("duplicate microbatch step")
        self._step_seen = True
        if self._active_index == self.microbatch_count - 1:
            if not self._inner_boundary:
                raise RuntimeError("final microbatch has not reached its inner commit boundary")
            return self.engine.step(*args, **kwargs)
        return None
