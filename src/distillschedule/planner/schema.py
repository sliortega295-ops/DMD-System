"""Typed schema for profile-guided DistillSchedule planning.

The schema deliberately separates measured evidence from analytical inputs.
A profile may contain full-plan anchors before it contains enough primitive
measurements to predict unseen plans. The simulator reports that distinction
instead of silently treating calibration anchors as generalization evidence.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "distillschedule.profile_guided_planner.v0"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _positive(name: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclasses.dataclass(frozen=True)
class HardwareSpec:
    name: str
    nodes: int
    gpus_per_node: int
    gpu_hbm_bytes: int
    node_cpu_limit_bytes: int
    intra_node_bandwidth_gbps: float | None = None
    inter_node_bandwidth_gbps: float | None = None
    pinned_h2d_bandwidth_gbps: float | None = None
    pageable_h2d_bandwidth_gbps: float | None = None
    d2h_bandwidth_gbps: float | None = None
    collective_latency_us: float | None = None
    transfer_launch_latency_us: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("hardware name cannot be empty")
        for name in ("nodes", "gpus_per_node", "gpu_hbm_bytes", "node_cpu_limit_bytes"):
            _positive(name, getattr(self, name))

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node


@dataclasses.dataclass(frozen=True)
class ModelInventory:
    role: str
    parameter_bytes: int
    frozen_parameter_bytes: int
    trainable_parameter_bytes: int
    buffer_bytes: int
    block_parameter_bytes: tuple[int, ...] = ()
    trainable_numel: int = 0
    optimizer_state_bytes: int | None = None
    ema_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("model role cannot be empty")
        for name in (
            "parameter_bytes",
            "frozen_parameter_bytes",
            "trainable_parameter_bytes",
            "buffer_bytes",
            "trainable_numel",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.parameter_bytes != self.frozen_parameter_bytes + self.trainable_parameter_bytes:
            raise ValueError("parameter bytes must equal frozen + trainable bytes")
        if any(value <= 0 for value in self.block_parameter_bytes):
            raise ValueError("block parameter sizes must be positive")


@dataclasses.dataclass(frozen=True)
class PrimitiveKey:
    workload: str
    operation: str
    role: str
    backend: str
    mesh_size: int
    local_batch: int
    stage: int | None = None
    grad_mode: str = "no_grad"
    residency: str = "resident"
    activation_policy: str = "none"
    offload_blocks: tuple[int, ...] = ()
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        for name in ("workload", "operation", "role", "backend", "grad_mode", "residency"):
            if not getattr(self, name):
                raise ValueError(f"{name} cannot be empty")
        _positive("mesh_size", self.mesh_size)
        _positive("local_batch", self.local_batch)
        if self.stage is not None and self.stage < 0:
            raise ValueError("stage cannot be negative")
        if len(set(self.offload_blocks)) != len(self.offload_blocks):
            raise ValueError("offload block indices must be unique")

    @property
    def id(self) -> str:
        payload = dataclasses.asdict(self)
        payload["offload_blocks"] = list(self.offload_blocks)
        return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class PrimitiveMeasurement:
    key: PrimitiveKey
    wall_seconds: float
    samples: int
    provenance: str
    cuda_device_seconds: float | None = None
    compute_seconds: float | None = None
    nccl_seconds: float | None = None
    h2d_seconds: float | None = None
    d2h_seconds: float | None = None
    transferred_h2d_bytes: int = 0
    transferred_d2h_bytes: int = 0
    collective_bytes: int = 0
    gpu_peak_bytes: int | None = None
    cpu_peak_bytes: int | None = None
    activation_peak_bytes: int | None = None
    parameter_materialization_count: int = 0
    confidence: str = "measured"
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        _positive("wall_seconds", self.wall_seconds)
        _positive("samples", self.samples)
        if not self.provenance:
            raise ValueError("measurement provenance cannot be empty")
        for name in (
            "transferred_h2d_bytes",
            "transferred_d2h_bytes",
            "collective_bytes",
            "parameter_materialization_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.confidence not in {"measured", "derived", "diagnostic", "synthetic"}:
            raise ValueError(f"unsupported confidence: {self.confidence}")


@dataclasses.dataclass(frozen=True)
class TransitionMeasurement:
    name: str
    source_layout: str
    target_layout: str
    wall_seconds: float
    samples: int
    provenance: str
    source_release_seconds: float | None = None
    target_prepare_seconds: float | None = None
    target_materialize_seconds: float | None = None
    optimizer_restore_seconds: float | None = None
    collective_seconds: float | None = None
    barrier_seconds: float | None = None
    cleanup_seconds: float | None = None
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    node_cpu_peak_bytes: int | None = None
    gpu_peak_bytes: int | None = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.source_layout or not self.target_layout:
            raise ValueError("transition identity cannot be empty")
        _positive("wall_seconds", self.wall_seconds)
        _positive("samples", self.samples)


@dataclasses.dataclass(frozen=True)
class PlanObservation:
    name: str
    workload: str
    full5_seconds: tuple[float, ...]
    status: str
    provenance: str
    phase_seconds: Mapping[str, float] = dataclasses.field(default_factory=dict)
    compute_fraction: float | None = None
    nccl_fraction: float | None = None
    copy_fraction: float | None = None
    gpu_peak_bytes: int | None = None
    node_cpu_peak_bytes: int | None = None
    claimable_timing: bool = True
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.workload or not self.provenance:
            raise ValueError("plan observation identity cannot be empty")
        if not self.full5_seconds or any(value <= 0 for value in self.full5_seconds):
            raise ValueError("plan observation needs positive Full5 samples")
        if self.status not in {"PASS", "DIAGNOSTIC", "CAPACITY_FAIL", "PARTIAL"}:
            raise ValueError(f"unsupported plan status {self.status}")
        for value in (self.compute_fraction, self.nccl_fraction, self.copy_fraction):
            if value is not None and not 0 <= value <= 1:
                raise ValueError("activity fractions must be within [0, 1]")


@dataclasses.dataclass(frozen=True)
class PhaseSpec:
    name: str
    kind: str
    primitive_keys: tuple[PrimitiveKey, ...] = ()
    repeats: int = 1
    fixed_seconds: float | None = None
    producer_key: PrimitiveKey | None = None
    consumer_key: PrimitiveKey | None = None
    queue_depth: int = 1
    batches: int = 1
    overlap_credit: str = "consumer_commit"
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("phase name cannot be empty")
        if self.kind not in {"sequential", "pipeline", "fixed"}:
            raise ValueError(f"unsupported phase kind: {self.kind}")
        _positive("repeats", self.repeats)
        if self.kind == "fixed":
            if self.fixed_seconds is None or self.fixed_seconds <= 0:
                raise ValueError("fixed phase requires positive fixed_seconds")
        elif self.kind == "sequential":
            if not self.primitive_keys:
                raise ValueError("sequential phase requires primitive keys")
        else:
            if self.producer_key is None or self.consumer_key is None:
                raise ValueError("pipeline phase requires producer and consumer keys")
            _positive("queue_depth", self.queue_depth)
            _positive("batches", self.batches)


@dataclasses.dataclass(frozen=True)
class PlanSpec:
    name: str
    workload: str
    phases: tuple[PhaseSpec, ...]
    transition_names: tuple[str, ...] = ()
    backend_family: str = "mixed"
    gpu_meshes: tuple[int, ...] = ()
    global_batch_by_phase: Mapping[str, int] = dataclasses.field(default_factory=dict)
    memory_policy: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.workload or not self.phases:
            raise ValueError("plan name, workload, and phases are required")
        if any(value <= 0 for value in self.gpu_meshes):
            raise ValueError("GPU meshes must be positive")


@dataclasses.dataclass(frozen=True)
class PlannerProfile:
    hardware: HardwareSpec
    models: tuple[ModelInventory, ...]
    primitives: tuple[PrimitiveMeasurement, ...]
    transitions: tuple[TransitionMeasurement, ...]
    plans: tuple[PlanObservation, ...]
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported planner profile schema {self.schema_version}")
        roles = [value.role for value in self.models]
        if len(roles) != len(set(roles)):
            raise ValueError("model inventory roles must be unique")

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if dataclasses.is_dataclass(value):
                return {field.name: convert(getattr(value, field.name)) for field in dataclasses.fields(value)}
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            return value
        return convert(self)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(target)


def _primitive_key(value: Mapping[str, Any]) -> PrimitiveKey:
    payload = dict(value)
    payload["offload_blocks"] = tuple(int(item) for item in payload.get("offload_blocks", ()))
    return PrimitiveKey(**payload)


def profile_from_dict(value: Mapping[str, Any]) -> PlannerProfile:
    hardware = HardwareSpec(**dict(value["hardware"]))
    models = tuple(
        ModelInventory(**{**dict(item), "block_parameter_bytes": tuple(int(x) for x in item.get("block_parameter_bytes", ()))})
        for item in value.get("models", ())
    )
    primitives = []
    for item in value.get("primitives", ()):
        payload = dict(item)
        payload["key"] = _primitive_key(payload["key"])
        primitives.append(PrimitiveMeasurement(**payload))
    transitions = tuple(TransitionMeasurement(**dict(item)) for item in value.get("transitions", ()))
    plans = tuple(
        PlanObservation(**{**dict(item), "full5_seconds": tuple(float(x) for x in item["full5_seconds"])})
        for item in value.get("plans", ())
    )
    return PlannerProfile(
        hardware=hardware,
        models=models,
        primitives=tuple(primitives),
        transitions=transitions,
        plans=plans,
        metadata=dict(value.get("metadata", {})),
        schema_version=str(value.get("schema_version", SCHEMA_VERSION)),
    )


def load_profile(path: str | Path) -> PlannerProfile:
    return profile_from_dict(json.loads(Path(path).read_text()))


def phase_from_dict(value: Mapping[str, Any]) -> PhaseSpec:
    payload = dict(value)
    payload["primitive_keys"] = tuple(_primitive_key(item) for item in payload.get("primitive_keys", ()))
    if payload.get("producer_key") is not None:
        payload["producer_key"] = _primitive_key(payload["producer_key"])
    if payload.get("consumer_key") is not None:
        payload["consumer_key"] = _primitive_key(payload["consumer_key"])
    return PhaseSpec(**payload)


def plan_from_dict(value: Mapping[str, Any]) -> PlanSpec:
    payload = dict(value)
    payload["phases"] = tuple(phase_from_dict(item) for item in payload["phases"])
    payload["transition_names"] = tuple(str(item) for item in payload.get("transition_names", ()))
    payload["gpu_meshes"] = tuple(int(item) for item in payload.get("gpu_meshes", ()))
    return PlanSpec(**payload)


def load_plan(path: str | Path) -> PlanSpec:
    return plan_from_dict(json.loads(Path(path).read_text()))


__all__ = [
    "HardwareSpec", "ModelInventory", "PhaseSpec", "PlanObservation", "PlanSpec",
    "PlannerProfile", "PrimitiveKey", "PrimitiveMeasurement", "SCHEMA_VERSION",
    "TransitionMeasurement", "load_plan", "load_profile", "phase_from_dict",
    "plan_from_dict", "profile_from_dict",
]
