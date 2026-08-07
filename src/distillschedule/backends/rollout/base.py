"""Version-safe rollout-provider interface for native and external engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from distillschedule.core.contracts import ContractViolation, require
from distillschedule.core.enums import RolloutEngineKind


@dataclass(frozen=True, order=True)
class ModelVersion:
    component: str
    value: int
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        require(bool(self.component.strip()), "component must not be empty", path="model_version.component")
        require(self.value >= 0, "version value must be non-negative", path="model_version.value")


@dataclass(frozen=True)
class WeightSyncRequest:
    version: ModelVersion
    checkpoint: str | None = None
    delta: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WeightSyncAck:
    engine: RolloutEngineKind
    version: ModelVersion
    accepted: bool
    message: str = ""


@dataclass(frozen=True)
class RolloutRequest:
    request_id: str
    required_version: ModelVersion
    payload: Any
    sample_ids: tuple[str, ...] = ()
    operation_trace_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutArtifact:
    request_id: str
    source_version: ModelVersion
    samples: Any
    histories: Any = None
    conditioning: Any = None
    sample_ids: tuple[str, ...] = ()
    operation_trace_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RemoteRolloutClient(Protocol):
    def synchronize(self, request: WeightSyncRequest) -> WeightSyncAck:
        ...

    def rollout(self, request: RolloutRequest) -> RolloutArtifact:
        ...

    def close(self) -> None:
        ...


class RolloutProvider(ABC):
    engine: RolloutEngineKind

    @property
    @abstractmethod
    def current_version(self) -> ModelVersion | None:
        raise NotImplementedError

    @abstractmethod
    def synchronize(self, request: WeightSyncRequest) -> WeightSyncAck:
        raise NotImplementedError

    @abstractmethod
    def rollout(self, request: RolloutRequest) -> RolloutArtifact:
        raise NotImplementedError

    def close(self) -> None:
        return None


class VersionedRolloutProvider(RolloutProvider):
    def __init__(self, *, max_version_staleness: int = 0) -> None:
        if max_version_staleness < 0:
            raise ValueError("max_version_staleness must be non-negative")
        self.max_version_staleness = max_version_staleness
        self._current_version: ModelVersion | None = None

    @property
    def current_version(self) -> ModelVersion | None:
        return self._current_version

    def _accept_ack(self, ack: WeightSyncAck, requested: ModelVersion) -> WeightSyncAck:
        if ack.engine != self.engine:
            raise ContractViolation(
                f"engine acknowledged as {ack.engine.value!r}, expected {self.engine.value!r}",
                path="rollout.version_sync.engine",
            )
        if not ack.accepted:
            raise ContractViolation(
                ack.message or "rollout engine rejected the weight synchronization",
                path="rollout.version_sync",
            )
        if ack.version.component != requested.component or ack.version.value != requested.value:
            raise ContractViolation(
                f"engine acknowledged {ack.version} instead of requested {requested}",
                path="rollout.version_sync",
            )
        self._current_version = ack.version
        return ack

    def _require_version(self, required_version: ModelVersion) -> None:
        current = self.current_version
        if current is None:
            raise ContractViolation("rollout engine has no synchronized model version", path="rollout.version")
        if current.component != required_version.component:
            raise ContractViolation(
                f"rollout engine owns {current.component!r}, request requires {required_version.component!r}",
                path="rollout.version.component",
            )
        staleness = required_version.value - current.value
        if staleness < 0:
            raise ContractViolation(
                f"rollout engine version {current.value} is newer than requested {required_version.value}",
                path="rollout.version",
            )
        if staleness > self.max_version_staleness:
            raise ContractViolation(
                f"rollout engine version is stale by {staleness}; allowed {self.max_version_staleness}",
                path="rollout.version",
            )


class NativeRolloutProvider(VersionedRolloutProvider):
    engine = RolloutEngineKind.NATIVE

    def __init__(
        self,
        rollout_fn: Callable[[RolloutRequest], RolloutArtifact],
        *,
        sync_fn: Callable[[WeightSyncRequest], None] | None = None,
        max_version_staleness: int = 0,
    ) -> None:
        super().__init__(max_version_staleness=max_version_staleness)
        self._rollout_fn = rollout_fn
        self._sync_fn = sync_fn

    def synchronize(self, request: WeightSyncRequest) -> WeightSyncAck:
        if self._sync_fn is not None:
            self._sync_fn(request)
        ack = WeightSyncAck(engine=self.engine, version=request.version, accepted=True)
        return self._accept_ack(ack, request.version)

    def rollout(self, request: RolloutRequest) -> RolloutArtifact:
        self._require_version(request.required_version)
        artifact = self._rollout_fn(request)
        if artifact.request_id != request.request_id:
            raise ContractViolation("rollout request_id mismatch", path="rollout.artifact.request_id")
        if artifact.source_version != self.current_version:
            raise ContractViolation("rollout artifact reports the wrong source version", path="rollout.artifact.source_version")
        return artifact


class ExternalRolloutProvider(VersionedRolloutProvider):
    engine = RolloutEngineKind.REMOTE

    def __init__(self, client: RemoteRolloutClient, *, max_version_staleness: int = 0) -> None:
        super().__init__(max_version_staleness=max_version_staleness)
        self.client = client

    def synchronize(self, request: WeightSyncRequest) -> WeightSyncAck:
        return self._accept_ack(self.client.synchronize(request), request.version)

    def rollout(self, request: RolloutRequest) -> RolloutArtifact:
        self._require_version(request.required_version)
        artifact = self.client.rollout(request)
        if artifact.request_id != request.request_id:
            raise ContractViolation("rollout request_id mismatch", path="rollout.artifact.request_id")
        self._require_artifact_version(request, artifact)
        return artifact

    def _require_artifact_version(self, request: RolloutRequest, artifact: RolloutArtifact) -> None:
        if artifact.source_version.component != request.required_version.component:
            raise ContractViolation("artifact component version mismatch", path="rollout.artifact.source_version")
        staleness = request.required_version.value - artifact.source_version.value
        if staleness < 0 or staleness > self.max_version_staleness:
            raise ContractViolation(
                f"artifact staleness {staleness} violates allowed range [0, {self.max_version_staleness}]",
                path="rollout.artifact.source_version",
            )

    def close(self) -> None:
        self.client.close()
