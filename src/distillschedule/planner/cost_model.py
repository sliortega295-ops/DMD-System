"""Hybrid analytical/profile-guided cost model for DistillSchedule."""
from __future__ import annotations

import dataclasses
import math
from typing import Mapping, Sequence

from .profile_db import LookupResult, ProfileDB
from .schema import HardwareSpec, ModelInventory, PrimitiveKey

GIB = 2**30
GB = 10**9


@dataclasses.dataclass(frozen=True)
class MemoryPolicy:
    backend: str
    mesh_size: int
    optimizer_offload: bool = True
    parameter_offload: bool = False
    node_shared_frozen: bool = False
    offload_blocks: tuple[int, ...] = ()
    fsdp_prefetch_units: int = 1
    activation_peak_bytes: int = 0
    activation_cpu_bytes: int = 0
    temporary_gpu_bytes: int = 0
    temporary_cpu_bytes: int = 0
    persistent_pinned_target_copies: int = 0
    construction_duplicates_target: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"replicated", "zero1", "zero3", "fsdp2"}:
            raise ValueError(f"unsupported memory backend {self.backend}")
        if self.mesh_size <= 0:
            raise ValueError("mesh_size must be positive")
        if self.fsdp_prefetch_units < 0 or self.persistent_pinned_target_copies < 0:
            raise ValueError("memory-policy counts cannot be negative")


@dataclasses.dataclass(frozen=True)
class MemoryEstimate:
    gpu_steady_bytes: int
    gpu_peak_bytes: int
    cpu_steady_bytes: int
    cpu_peak_bytes: int
    gpu_breakdown: Mapping[str, int]
    cpu_breakdown: Mapping[str, int]
    feasible_gpu: bool
    feasible_cpu: bool
    confidence: str

    @property
    def feasible(self) -> bool:
        return self.feasible_gpu and self.feasible_cpu


@dataclasses.dataclass(frozen=True)
class OperationEstimate:
    key: PrimitiveKey
    wall_seconds: float | None
    match: str
    confidence: str
    explanation: str
    breakdown: Mapping[str, float]


@dataclasses.dataclass(frozen=True)
class OffloadCandidate:
    index: int
    freed_gpu_bytes: int
    added_seconds: float

    def __post_init__(self) -> None:
        if self.index < 0 or self.freed_gpu_bytes <= 0 or self.added_seconds < 0:
            raise ValueError("invalid offload candidate")


@dataclasses.dataclass(frozen=True)
class OffloadSelection:
    indices: tuple[int, ...]
    freed_gpu_bytes: int
    added_seconds: float
    feasible: bool


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _optimizer_bytes(model: ModelInventory) -> int:
    return model.optimizer_state_bytes if model.optimizer_state_bytes is not None else model.trainable_numel * 12


def estimate_model_memory(
    model: ModelInventory,
    policy: MemoryPolicy,
    hardware: HardwareSpec,
    *,
    node_rank_copies: int = 1,
) -> MemoryEstimate:
    """Estimate per-rank GPU and per-node CPU memory."""
    mesh = policy.mesh_size
    full_parameter = model.parameter_bytes + model.buffer_bytes
    trainable = model.trainable_parameter_bytes
    optimizer = _optimizer_bytes(model)
    offload_set = set(policy.offload_blocks)
    if any(index >= len(model.block_parameter_bytes) for index in offload_set):
        raise ValueError("offload block index exceeds model inventory")
    offloaded_block_full_bytes = sum(
        model.block_parameter_bytes[index] for index in offload_set
    )

    if policy.backend in {"fsdp2", "zero3"}:
        parameter_shard = _ceil_div(full_parameter, mesh)
        gradient_shard = _ceil_div(trainable, mesh)
        optimizer_gpu = 0 if policy.optimizer_offload else _ceil_div(optimizer, mesh)
        gather_unit = max(model.block_parameter_bytes or (full_parameter,))
        gather_peak = gather_unit * max(1, policy.fsdp_prefetch_units)
        offloaded_block_rank_bytes = _ceil_div(offloaded_block_full_bytes, mesh)
    else:
        parameter_shard = full_parameter
        gradient_shard = trainable
        optimizer_gpu = 0 if policy.optimizer_offload else (
            _ceil_div(optimizer, mesh) if policy.backend == "zero1" else optimizer
        )
        gather_peak = 0
        offloaded_block_rank_bytes = offloaded_block_full_bytes

    parameter_gpu = (
        max(0, parameter_shard - offloaded_block_rank_bytes)
        if policy.parameter_offload
        else parameter_shard
    )
    gpu_steady = parameter_gpu + gradient_shard + optimizer_gpu
    gpu_peak = gpu_steady + gather_peak + policy.activation_peak_bytes + policy.temporary_gpu_bytes

    if policy.optimizer_offload:
        optimizer_cpu_per_rank = _ceil_div(optimizer, mesh) if policy.backend in {"zero1", "zero3", "fsdp2"} else optimizer
    else:
        optimizer_cpu_per_rank = 0
    frozen_host_per_rank = (
        offloaded_block_rank_bytes if policy.parameter_offload else 0
    )
    frozen_host_per_node = model.frozen_parameter_bytes if policy.node_shared_frozen else frozen_host_per_rank * node_rank_copies
    pinned_target_bytes = policy.persistent_pinned_target_copies * full_parameter
    cpu_steady = (
        optimizer_cpu_per_rank * node_rank_copies
        + frozen_host_per_node
        + pinned_target_bytes
        + policy.activation_cpu_bytes
        + policy.temporary_cpu_bytes
    )
    construction_duplicate = pinned_target_bytes if policy.construction_duplicates_target else 0
    cpu_peak = cpu_steady + construction_duplicate

    gpu_breakdown = {
        "parameter_or_shard": parameter_gpu,
        "trainable_gradient_or_shard": gradient_shard,
        "optimizer_gpu": optimizer_gpu,
        "largest_gather_prefetch": gather_peak,
        "activation": policy.activation_peak_bytes,
        "temporary": policy.temporary_gpu_bytes,
    }
    cpu_breakdown = {
        "optimizer_offload": optimizer_cpu_per_rank * node_rank_copies,
        "frozen_or_offloaded_parameter": frozen_host_per_node,
        "persistent_pinned_targets": pinned_target_bytes,
        "activation_offload": policy.activation_cpu_bytes,
        "temporary": policy.temporary_cpu_bytes,
        "construction_duplicate": construction_duplicate,
    }
    return MemoryEstimate(
        gpu_steady, gpu_peak, cpu_steady, cpu_peak,
        gpu_breakdown, cpu_breakdown,
        gpu_peak <= hardware.gpu_hbm_bytes,
        cpu_peak <= hardware.node_cpu_limit_bytes,
        "analytical_plus_profile_residual_required",
    )


def transfer_seconds(byte_count: int, *, bandwidth_gbps: float | None, event_count: int = 1, launch_latency_us: float | None = None) -> float | None:
    if byte_count < 0 or event_count < 0:
        raise ValueError("transfer bytes/events cannot be negative")
    if byte_count == 0 and event_count == 0:
        return 0.0
    if bandwidth_gbps is None or bandwidth_gbps <= 0:
        return None
    latency = 0.0 if launch_latency_us is None else event_count * launch_latency_us / 1e6
    return byte_count / (bandwidth_gbps * GB) + latency


def ring_collective_seconds(byte_count: int, *, world_size: int, bandwidth_gbps: float | None, latency_us: float | None, kind: str) -> float | None:
    if world_size <= 0 or byte_count < 0:
        raise ValueError("invalid collective inputs")
    if bandwidth_gbps is None or bandwidth_gbps <= 0:
        return None
    if kind == "all_reduce":
        factor = 2 * (world_size - 1) / world_size
    elif kind in {"all_gather", "reduce_scatter"}:
        factor = (world_size - 1) / world_size
    else:
        raise ValueError(f"unsupported collective kind {kind}")
    latency = 0.0 if latency_us is None else (world_size - 1) * latency_us / 1e6
    return factor * byte_count / (bandwidth_gbps * GB) + latency


class HybridCostModel:
    def __init__(self, profile_db: ProfileDB) -> None:
        self.db = profile_db

    def operation(self, key: PrimitiveKey, *, allow_interpolation: bool = True) -> OperationEstimate:
        result: LookupResult = self.db.lookup(key, allow_interpolation=allow_interpolation)
        if result.measurement is None:
            return OperationEstimate(key, None, "missing", "missing", result.explanation, {})
        measurement = result.measurement
        breakdown = {
            name: value for name, value in {
                "compute": measurement.compute_seconds,
                "nccl": measurement.nccl_seconds,
                "h2d": measurement.h2d_seconds,
                "d2h": measurement.d2h_seconds,
            }.items() if value is not None
        }
        return OperationEstimate(key, measurement.wall_seconds, result.match, measurement.confidence, result.explanation, breakdown)

    def estimate_unprofiled_transfer(self, *, h2d_bytes: int = 0, d2h_bytes: int = 0, events: int = 1, pinned: bool = True):
        hardware = self.db.profile.hardware
        return {
            "h2d_seconds": transfer_seconds(
                h2d_bytes,
                bandwidth_gbps=hardware.pinned_h2d_bandwidth_gbps if pinned else hardware.pageable_h2d_bandwidth_gbps,
                event_count=events,
                launch_latency_us=hardware.transfer_launch_latency_us,
            ),
            "d2h_seconds": transfer_seconds(
                d2h_bytes,
                bandwidth_gbps=hardware.d2h_bandwidth_gbps,
                event_count=events,
                launch_latency_us=hardware.transfer_launch_latency_us,
            ),
        }


def select_offload_blocks(candidates: Sequence[OffloadCandidate], *, required_free_bytes: int) -> OffloadSelection:
    """Exact Pareto-frontier knapsack for tens of model blocks."""
    if required_free_bytes <= 0:
        return OffloadSelection((), 0, 0.0, True)
    states: list[tuple[int, float, tuple[int, ...]]] = [(0, 0.0, ())]
    for candidate in candidates:
        expanded = states + [
            (freed + candidate.freed_gpu_bytes, cost + candidate.added_seconds, indices + (candidate.index,))
            for freed, cost, indices in states
        ]
        best_by_freed: dict[int, tuple[float, tuple[int, ...]]] = {}
        for freed, cost, indices in expanded:
            prior = best_by_freed.get(freed)
            if prior is None or cost < prior[0]:
                best_by_freed[freed] = (cost, indices)
        ordered = sorted(((freed, cost, indices) for freed, (cost, indices) in best_by_freed.items()), key=lambda item: (-item[0], item[1]))
        frontier = []
        cheapest = math.inf
        for state in ordered:
            if state[1] < cheapest:
                frontier.append(state)
                cheapest = state[1]
        states = frontier
    feasible = [state for state in states if state[0] >= required_free_bytes]
    if not feasible:
        best = max(states, key=lambda item: item[0])
        return OffloadSelection(tuple(sorted(best[2])), best[0], best[1], False)
    best = min(feasible, key=lambda item: (item[1], item[0]))
    return OffloadSelection(tuple(sorted(best[2])), best[0], best[1], True)


__all__ = [
    "GIB", "HybridCostModel", "MemoryEstimate", "MemoryPolicy", "OffloadCandidate",
    "OffloadSelection", "OperationEstimate", "estimate_model_memory",
    "ring_collective_seconds", "select_offload_blocks", "transfer_seconds",
]
