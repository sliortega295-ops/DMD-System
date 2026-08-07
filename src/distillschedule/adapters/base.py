"""Model-adapter boundary between generic DMD semantics and repository code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Mapping

from distillschedule.core.capabilities import AdapterCapabilities
from distillschedule.core.plan import ExecutionPlan
from distillschedule.workloads.dmd import CompiledDMDProtocol


@dataclass(frozen=True)
class LegacyInvocation:
    command: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str] = field(default_factory=dict)
    generated_files: Mapping[Path, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "cwd": str(self.cwd),
            "environment": dict(self.environment),
            "generated_files": {
                str(path): {
                    "bytes": len(content.encode("utf-8")),
                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
                for path, content in self.generated_files.items()
            },
        }


@dataclass(frozen=True)
class CompiledAdapterPlan:
    protocol: CompiledDMDProtocol
    invocation: LegacyInvocation | None
    warnings: tuple[str, ...] = ()


class DMDModelAdapter(ABC):
    """The only layer allowed to know model-family-specific execution details."""

    name: str

    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities:
        raise NotImplementedError

    @abstractmethod
    def validate(self, plan: ExecutionPlan) -> None:
        raise NotImplementedError

    @abstractmethod
    def compile(self, plan: ExecutionPlan, *, source_dir: Path, repo_root: Path) -> CompiledAdapterPlan:
        raise NotImplementedError


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, DMDModelAdapter] = {}

    def register(self, adapter: DMDModelAdapter) -> None:
        key = adapter.name.strip().lower()
        if not key:
            raise ValueError("adapter name must not be empty")
        if key in self._adapters:
            raise ValueError(f"adapter {key!r} is already registered")
        self._adapters[key] = adapter

    def get(self, name: str) -> DMDModelAdapter:
        key = name.strip().lower()
        try:
            return self._adapters[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._adapters)) or "<none>"
            raise KeyError(f"unknown adapter {name!r}; available: {available}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))
