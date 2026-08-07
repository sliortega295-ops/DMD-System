"""Fail-closed validation helpers used by the public DistillSchedule schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn


class ContractViolation(ValueError):
    """Raised when a workload or execution plan violates a declared contract."""

    def __init__(self, message: str, *, path: str = "plan") -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def fail(message: str, *, path: str) -> NoReturn:
    raise ContractViolation(message, path=path)


def require(condition: bool, message: str, *, path: str) -> None:
    if not condition:
        fail(message, path=path)


def require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"expected a mapping, got {type(value).__name__}", path=path)
    return value


def require_sequence(value: Any, *, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        fail(f"expected a sequence, got {type(value).__name__}", path=path)
    return value


def reject_unknown_keys(
    mapping: Mapping[str, Any],
    allowed: set[str] | frozenset[str],
    *,
    path: str,
) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        fail(f"unknown fields: {', '.join(unknown)}", path=path)


def required(mapping: Mapping[str, Any], key: str, *, path: str) -> Any:
    if key not in mapping:
        fail(f"missing required field {key!r}", path=path)
    return mapping[key]


def non_empty_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail("expected a non-empty string", path=path)
    return value.strip()


def positive_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail("expected a positive integer", path=path)
    return value


def nonnegative_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail("expected a non-negative integer", path=path)
    return value


def positive_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        fail("expected a positive number", path=path)
    return float(value)


def ensure_unique(values: Sequence[Any], *, path: str) -> None:
    if len(values) != len(set(values)):
        fail("entries must be unique", path=path)
