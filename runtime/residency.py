"""Backend-owned physical-state residency with alias and version safety.

The backend is intentionally synchronous. It establishes correctness and exposes
copy costs before an asynchronous copy-stream implementation is attempted.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn


class ResidencyPolicy(str, Enum):
    RESIDENT = "resident"
    TIME_MULTIPLEX = "time_multiplex"


class ResidencyError(RuntimeError):
    pass


class ParameterVersionError(ResidencyError):
    pass


class AdapterModeConflictError(ResidencyError):
    pass


@dataclass
class ResidencyEvent:
    action: str
    state: str
    alias: Optional[str] = None
    source_device: Optional[str] = None
    target_device: Optional[str] = None
    tensor_bytes: int = 0
    duration_ms: float = 0.0
    version: int = 0
    mode: Optional[str] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _StateRecord:
    name: str
    module: nn.Module
    aliases: Tuple[str, ...]
    modes: Tuple[str, ...]
    current_device: str
    version: int
    before_offload: Optional[Callable[[], None]] = None
    active_alias_counts: Dict[str, int] = field(default_factory=dict)
    active_mode: Optional[str] = None

    @property
    def active_references(self) -> int:
        return sum(self.active_alias_counts.values())


MoveFunction = Callable[[nn.Module, str], None]
SynchronizeFunction = Callable[[str], None]


class ModuleResidencyBackend:
    """Own module materialization for logical aliases of physical state.

    `TIME_MULTIPLEX` offloads inactive non-target states before materializing the
    requested one. Active aliases are reference counted, preventing a live graph
    or nested operation from being invalidated. `RESIDENT` records acquisitions
    but does not move modules.
    """

    def __init__(
        self,
        *,
        accelerator_device: str,
        host_device: str = "cpu",
        policy: ResidencyPolicy | str = ResidencyPolicy.RESIDENT,
        move_function: Optional[MoveFunction] = None,
        synchronize_function: Optional[SynchronizeFunction] = None,
    ) -> None:
        self.accelerator_device = str(accelerator_device)
        self.host_device = str(host_device)
        self.policy = ResidencyPolicy(policy)
        self._move_function = move_function or self._default_move
        self._synchronize_function = synchronize_function or self._default_synchronize
        self._states: Dict[str, _StateRecord] = {}
        self._aliases: Dict[str, str] = {}
        self.events: list[ResidencyEvent] = []

    @staticmethod
    def _default_move(module: nn.Module, target_device: str) -> None:
        module.to(device=torch.device(target_device), non_blocking=False)

    @staticmethod
    def _default_synchronize(device: str) -> None:
        parsed = torch.device(device)
        if parsed.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(parsed)

    @staticmethod
    def _tensor_bytes(module: nn.Module) -> int:
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in module.parameters()
        )
        buffer_bytes = sum(
            buffer.numel() * buffer.element_size() for buffer in module.buffers()
        )
        gradient_bytes = sum(
            parameter.grad.numel() * parameter.grad.element_size()
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        return parameter_bytes + buffer_bytes + gradient_bytes

    def register_state(
        self,
        name: str,
        module: nn.Module,
        *,
        aliases: Sequence[str],
        modes: Sequence[str] = (),
        initial_device: Optional[str] = None,
        version: int = 0,
        before_offload: Optional[Callable[[], None]] = None,
    ) -> None:
        if name in self._states:
            raise ValueError(f"duplicate residency state: {name}")
        aliases_tuple = tuple(dict.fromkeys((name, *aliases)))
        for alias in aliases_tuple:
            if alias in self._aliases:
                raise ValueError(
                    f"alias {alias!r} already maps to {self._aliases[alias]!r}"
                )
        record = _StateRecord(
            name=name,
            module=module,
            aliases=aliases_tuple,
            modes=tuple(modes),
            current_device=str(initial_device or self.accelerator_device),
            version=int(version),
            before_offload=before_offload,
        )
        self._states[name] = record
        for alias in aliases_tuple:
            self._aliases[alias] = name

    def resolve(self, alias_or_state: str) -> str:
        try:
            return self._aliases[alias_or_state]
        except KeyError as error:
            raise KeyError(f"unknown residency alias/state: {alias_or_state}") from error

    def initialize(self, preferred_state: str) -> None:
        preferred = self.resolve(preferred_state)
        if self.policy != ResidencyPolicy.TIME_MULTIPLEX:
            return
        for state in self._states.values():
            if state.name != preferred and state.current_device == self.accelerator_device:
                self._move(state, self.host_device, reason="initial_placement")
        target = self._states[preferred]
        if target.current_device != self.accelerator_device:
            self._move(target, self.accelerator_device, reason="initial_placement")

    def _move(self, state: _StateRecord, target: str, *, reason: str) -> None:
        target = str(target)
        if state.current_device == target:
            return
        if state.active_references:
            raise ResidencyError(
                f"cannot move active state {state.name}: {state.active_alias_counts}"
            )
        source = state.current_device
        if target == self.host_device and state.before_offload is not None:
            state.before_offload()
        tensor_bytes = self._tensor_bytes(state.module)
        self._synchronize_function(source)
        start = time.perf_counter()
        self._move_function(state.module, target)
        self._synchronize_function(target)
        duration_ms = (time.perf_counter() - start) * 1000.0
        state.current_device = target
        action = (
            "materialize"
            if target == self.accelerator_device
            else "offload"
            if target == self.host_device
            else "move"
        )
        self.events.append(
            ResidencyEvent(
                action=action,
                state=state.name,
                source_device=source,
                target_device=target,
                tensor_bytes=tensor_bytes,
                duration_ms=duration_ms,
                version=state.version,
                reason=reason,
            )
        )

    def _prepare(self, target: _StateRecord) -> None:
        if self.policy == ResidencyPolicy.TIME_MULTIPLEX:
            for other in self._states.values():
                if (
                    other.name == target.name
                    or other.current_device != self.accelerator_device
                ):
                    continue
                if other.active_references:
                    self.events.append(
                        ResidencyEvent(
                            action="retain_active",
                            state=other.name,
                            target_device=self.accelerator_device,
                            version=other.version,
                            reason=f"needed while acquiring {target.name}",
                            metadata={"active_alias_counts": dict(other.active_alias_counts)},
                        )
                    )
                    continue
                self._move(other, self.host_device, reason=f"prepare:{target.name}")
        if target.current_device != self.accelerator_device:
            self._move(target, self.accelerator_device, reason="acquire")

    @contextmanager
    def acquire(
        self,
        alias_or_state: str,
        *,
        mode: Optional[str] = None,
        expected_version: Optional[int] = None,
    ) -> Iterator[nn.Module]:
        state_name = self.resolve(alias_or_state)
        state = self._states[state_name]
        if expected_version is not None and state.version != expected_version:
            raise ParameterVersionError(
                f"state {state.name} has version {state.version}, expected {expected_version}"
            )
        if mode is not None and state.modes and mode not in state.modes:
            raise ValueError(f"mode {mode!r} is invalid for state {state.name}")
        if (
            mode is not None
            and state.active_references
            and state.active_mode is not None
            and state.active_mode != mode
        ):
            raise AdapterModeConflictError(
                f"state {state.name} already has active mode {state.active_mode!r}; "
                f"cannot acquire {mode!r}"
            )
        self._prepare(state)
        state.active_alias_counts[alias_or_state] = (
            state.active_alias_counts.get(alias_or_state, 0) + 1
        )
        if mode is not None:
            state.active_mode = mode
        self.events.append(
            ResidencyEvent(
                action="acquire",
                state=state.name,
                alias=alias_or_state,
                target_device=self.accelerator_device,
                version=state.version,
                mode=mode,
            )
        )
        try:
            yield state.module
        finally:
            count = state.active_alias_counts[alias_or_state] - 1
            if count:
                state.active_alias_counts[alias_or_state] = count
            else:
                state.active_alias_counts.pop(alias_or_state)
            if not state.active_references:
                state.active_mode = None
            self.events.append(
                ResidencyEvent(
                    action="release_reference",
                    state=state.name,
                    alias=alias_or_state,
                    target_device=state.current_device,
                    version=state.version,
                    mode=mode,
                )
            )

    def mark_updated(self, alias_or_state: str, version: int) -> None:
        state = self._states[self.resolve(alias_or_state)]
        if int(version) <= state.version:
            raise ParameterVersionError(
                f"state {state.name} cannot move version {state.version} -> {version}"
            )
        state.version = int(version)
        self.events.append(
            ResidencyEvent(
                action="mark_updated",
                state=state.name,
                target_device=state.current_device,
                version=state.version,
            )
        )

    def reset_metrics(self) -> None:
        self.events.clear()

    def snapshot(self) -> Dict[str, Any]:
        transfer_events = [
            event for event in self.events if event.action in {"materialize", "offload", "move"}
        ]
        return {
            "policy": self.policy.value,
            "accelerator_device": self.accelerator_device,
            "host_device": self.host_device,
            "states": {
                name: {
                    "current_device": state.current_device,
                    "version": state.version,
                    "aliases": list(state.aliases),
                    "modes": list(state.modes),
                    "active_alias_counts": dict(state.active_alias_counts),
                    "active_mode": state.active_mode,
                    "tensor_bytes": self._tensor_bytes(state.module),
                }
                for name, state in self._states.items()
            },
            "transfer": {
                "count": len(transfer_events),
                "bytes": sum(event.tensor_bytes for event in transfer_events),
                "time_ms": sum(event.duration_ms for event in transfer_events),
                "h2d_bytes": sum(
                    event.tensor_bytes
                    for event in transfer_events
                    if event.target_device == self.accelerator_device
                ),
                "h2d_time_ms": sum(
                    event.duration_ms
                    for event in transfer_events
                    if event.target_device == self.accelerator_device
                ),
                "d2h_bytes": sum(
                    event.tensor_bytes
                    for event in transfer_events
                    if event.target_device == self.host_device
                ),
                "d2h_time_ms": sum(
                    event.duration_ms
                    for event in transfer_events
                    if event.target_device == self.host_device
                ),
            },
            "events": [asdict(event) for event in self.events],
        }
