"""Capability declarations keep optional backends out of core scheduling logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .enums import RolloutEngineKind, TrainingBackendKind


@dataclass(frozen=True)
class AdapterCapabilities:
    training_backends: tuple[TrainingBackendKind, ...]
    rollout_engines: tuple[RolloutEngineKind, ...]
    supports_dynamic_shape: bool = False
    supports_exact_version_sync: bool = True
    supports_external_rollout: bool = False
    features: Mapping[str, bool] = field(default_factory=dict)

    def supports_training(self, backend: TrainingBackendKind) -> bool:
        return backend in self.training_backends

    def supports_rollout(self, engine: RolloutEngineKind) -> bool:
        return engine in self.rollout_engines
