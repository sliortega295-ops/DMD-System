"""Minimal training-backend protocol used by future native role runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from distillschedule.core.enums import TrainingBackendKind


@dataclass(frozen=True)
class BackendState:
    component: str
    model_version: int
    optimizer_step: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TrainingBackend(ABC):
    kind: TrainingBackendKind

    @abstractmethod
    def initialize(self) -> BackendState:
        raise NotImplementedError

    @abstractmethod
    def backward(self, loss: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self) -> BackendState:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        raise NotImplementedError
