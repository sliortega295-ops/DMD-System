"""Monotonic component-version ledger shared by role runners and rollout engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from distillschedule.core.contracts import ContractViolation


@dataclass(frozen=True)
class VersionSnapshot:
    values: Mapping[str, int]


class VersionLedger:
    def __init__(self, initial: Mapping[str, int] | None = None) -> None:
        self._values = {name: int(value) for name, value in (initial or {}).items()}
        if any(value < 0 for value in self._values.values()):
            raise ValueError("versions must be non-negative")

    def get(self, component: str) -> int:
        return self._values.get(component, 0)

    def require(self, component: str, expected: int) -> None:
        actual = self.get(component)
        if actual != expected:
            raise ContractViolation(
                f"expected {component}@{expected}, observed {component}@{actual}",
                path="runtime.version",
            )

    def advance(self, component: str, *, expected: int | None = None) -> int:
        if expected is not None:
            self.require(component, expected)
        value = self.get(component) + 1
        self._values[component] = value
        return value

    def set(self, component: str, value: int) -> None:
        if value < self.get(component):
            raise ContractViolation(
                f"cannot move {component} version backwards",
                path="runtime.version",
            )
        self._values[component] = value

    def snapshot(self) -> VersionSnapshot:
        return VersionSnapshot(values=dict(self._values))
