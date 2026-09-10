"""Performance-control primitives for the GAN Role-Phase runtime.

The module does not change Helios model math. It provides transition subphase
receipts, a persistent-target policy, host-memory admission, service-rate
accounting, bounded layout estimates, and a DeepSpeed Stage-3-safe validation
guard.
"""
from __future__ import annotations

import contextlib
import dataclasses
import enum
import json
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


CGROUP_MEMORY_ROOT = Path("/sys/fs/cgroup")


class TransitionSubphase(str, enum.Enum):
    SOURCE_EXPORT = "source_export"
    SOURCE_RELEASE = "source_release"
    PROCESS_GROUP_LIFECYCLE = "process_group_lifecycle"
    HOST_ADMISSION = "host_memory_admission"
    TARGET_PREPARE = "target_prepare"
    TARGET_MATERIALIZE = "target_materialize"
    OPTIMIZER_RESTORE = "optimizer_restore"
    BROADCAST_RESHARD = "broadcast_or_reshard"
    ENGINE_INITIALIZE = "engine_initialize"
    BARRIER_PEER_WAIT = "barrier_or_peer_wait"
    ALLOCATOR_CLEANUP = "allocator_cleanup"
    PROFILER_TAIL = "profiler_only_tail"
    PHASE_B_GENERATOR = "phase_b_generator_service"
    PHASE_B_CRITIC = "phase_b_critic_service"
    REVERSE_TOTAL = "reverse_handoff_total"


@dataclasses.dataclass(frozen=True)
class PersistentTargetPolicy:
    forward_targets_preconstructed_before_timing: bool = True
    forward_targets_pinned_before_timing: bool = True
    reverse_target_policy: str = "lazy_after_critic_release"
    process_groups_created_outside_stable_compute: bool = True
    repeated_frozen_streaming_in_stable_phase: bool = False
    forward_handoff_timed: bool = True
    reverse_handoff_timed: bool = True

    def __post_init__(self) -> None:
        if not self.forward_targets_preconstructed_before_timing:
            raise ValueError("forward target construction may not remain in A->B")
        if not self.forward_targets_pinned_before_timing:
            raise ValueError("forward target host preparation may not remain in A->B")
        if self.reverse_target_policy not in {
            "lazy_after_critic_release",
            "preconstruct_before_timing_capacity_canary",
        }:
            raise ValueError("unsupported reverse-target policy")
        if self.repeated_frozen_streaming_in_stable_phase:
            raise ValueError("stable phases may not repeatedly stream frozen backbones")
        if not (self.forward_handoff_timed and self.reverse_handoff_timed):
            raise ValueError("both handoff directions belong to steady-state timing")


@dataclasses.dataclass(frozen=True)
class CgroupMemorySnapshot:
    current_bytes: int
    max_bytes: int | None
    oom: int
    oom_kill: int

    def __post_init__(self) -> None:
        if self.current_bytes < 0:
            raise ValueError("cgroup memory.current must be non-negative")
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValueError("finite cgroup memory.max must be positive")
        if self.oom < 0 or self.oom_kill < 0:
            raise ValueError("cgroup OOM counters must be non-negative")

    @property
    def headroom_bytes(self) -> int | None:
        if self.max_bytes is None:
            return None
        return max(0, self.max_bytes - self.current_bytes)

    @property
    def used_fraction(self) -> float | None:
        if self.max_bytes is None:
            return None
        return float(self.current_bytes) / float(self.max_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_bytes": self.current_bytes,
            "max_bytes": self.max_bytes,
            "headroom_bytes": self.headroom_bytes,
            "used_fraction": self.used_fraction,
            "oom": self.oom,
            "oom_kill": self.oom_kill,
        }


def _read_cgroup_counter_file(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        key, raw = fields
        try:
            result[key] = int(raw)
        except ValueError:
            continue
    return result


def read_cgroup_memory_snapshot(
    root: str | Path = CGROUP_MEMORY_ROOT,
) -> CgroupMemorySnapshot:
    """Read the cgroup-v2 memory state used by the cluster OOM killer."""

    root = Path(root)
    current = int((root / "memory.current").read_text().strip())
    raw_max = (root / "memory.max").read_text().strip()
    maximum = None if raw_max == "max" else int(raw_max)
    events = _read_cgroup_counter_file(root / "memory.events")
    return CgroupMemorySnapshot(
        current_bytes=current,
        max_bytes=maximum,
        oom=int(events.get("oom", 0)),
        oom_kill=int(events.get("oom_kill", 0)),
    )


def build_preconstruction_waves(
    admitted_local_ranks: Iterable[int],
    *,
    local_world_size: int = 8,
    max_concurrent: int = 1,
) -> tuple[tuple[int, ...], ...]:
    """Return a deterministic rank-ordered cold-preparation admission schedule."""

    if local_world_size <= 0:
        raise ValueError("local_world_size must be positive")
    if not 1 <= max_concurrent <= local_world_size:
        raise ValueError("max_concurrent must be in [1, local_world_size]")
    ranks = tuple(sorted({int(rank) for rank in admitted_local_ranks}))
    if any(rank < 0 or rank >= local_world_size for rank in ranks):
        raise ValueError("preconstruction rank is outside the local world")
    return tuple(
        ranks[index : index + max_concurrent]
        for index in range(0, len(ranks), max_concurrent)
    )


def assert_preconstruction_admitted(
    snapshot: CgroupMemorySnapshot,
    *,
    max_used_fraction: float = 0.90,
    min_headroom_bytes: int = 64 * 2**30,
) -> dict[str, Any]:
    """Fail before the next 14B target can trigger a real cgroup OOM kill.

    A finite cgroup must keep both a fractional reserve and an absolute reserve.
    The default 10% reserve is intentionally conservative for the 760-GiB BAAI
    cgroup because one target build plus pageable->pinned replacement can have a
    tens-of-GiB transient even when the final pinned representation fits.
    """

    if not 0.0 < float(max_used_fraction) < 1.0:
        raise ValueError("max_used_fraction must be in (0, 1)")
    if int(min_headroom_bytes) < 0:
        raise ValueError("min_headroom_bytes must be non-negative")
    receipt = {
        "snapshot": snapshot.as_dict(),
        "max_used_fraction": float(max_used_fraction),
        "min_headroom_bytes": int(min_headroom_bytes),
        "finite_limit": snapshot.max_bytes is not None,
        "admitted": True,
    }
    if snapshot.max_bytes is None:
        return receipt
    fractional_reserve = int(
        snapshot.max_bytes * (1.0 - float(max_used_fraction))
    )
    required_headroom = max(int(min_headroom_bytes), fractional_reserve)
    receipt["required_headroom_bytes"] = required_headroom
    headroom = snapshot.headroom_bytes
    if headroom is None or headroom < required_headroom:
        receipt["admitted"] = False
        raise RuntimeError(
            "Role-Phase target preconstruction rejected by host admission: "
            f"current={snapshot.current_bytes} max={snapshot.max_bytes} "
            f"headroom={headroom} required={required_headroom}"
        )
    return receipt


def assert_no_new_cgroup_oom(
    before: CgroupMemorySnapshot,
    after: CgroupMemorySnapshot,
) -> None:
    """Reject a cold-preparation turn if the kernel observed a new OOM event."""

    if after.oom > before.oom or after.oom_kill > before.oom_kill:
        raise RuntimeError(
            "cgroup OOM counters increased during Role-Phase target preparation: "
            f"oom {before.oom}->{after.oom}, "
            f"oom_kill {before.oom_kill}->{after.oom_kill}"
        )


@dataclasses.dataclass(frozen=True)
class StageRecord:
    stage: str
    start_unix_ns: int
    stop_unix_ns: int
    duration_seconds: float
    rank: int
    metadata: Mapping[str, Any]


class JsonlStageRecorder:
    """Low-overhead append-only transition/service receipt writer."""

    def __init__(self, path: str | Path, *, rank: int) -> None:
        self.path = Path(path)
        self.rank = int(rank)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[StageRecord] = []
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def stage(self, stage: str | TransitionSubphase, **metadata: Any) -> Iterator[None]:
        name = stage.value if isinstance(stage, TransitionSubphase) else str(stage)
        started_perf = time.perf_counter()
        started_ns = time.time_ns()
        error: BaseException | None = None
        try:
            yield
        except BaseException as caught:
            error = caught
            raise
        finally:
            stopped_ns = time.time_ns()
            payload = dict(metadata)
            payload["status"] = "FAIL" if error is not None else "PASS"
            if error is not None:
                payload["error_type"] = type(error).__name__
                payload["error"] = str(error)
            record = StageRecord(
                stage=name,
                start_unix_ns=started_ns,
                stop_unix_ns=stopped_ns,
                duration_seconds=time.perf_counter() - started_perf,
                rank=self.rank,
                metadata=payload,
            )
            with self._lock:
                self._records.append(record)
                with self.path.open("a") as handle:
                    handle.write(json.dumps(dataclasses.asdict(record), sort_keys=True) + "\n")

    @property
    def records(self) -> tuple[StageRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def summary(self) -> dict[str, Any]:
        grouped: dict[str, list[float]] = {}
        for record in self.records:
            grouped.setdefault(record.stage, []).append(record.duration_seconds)
        return {
            "schema_version": "distillschedule.gan_rolephase_transition_receipt.v2",
            "rank": self.rank,
            "event_count": len(self.records),
            "stages": {
                name: {
                    "count": len(values),
                    "total_seconds": sum(values),
                    "median_seconds": statistics.median(values),
                    "max_seconds": max(values),
                }
                for name, values in sorted(grouped.items())
            },
        }

    def write_summary(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path.with_suffix(".summary.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(self.summary(), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)
        return target


@dataclasses.dataclass(frozen=True)
class LayoutContract:
    name: str
    generator_ranks: int
    critic_ranks: int
    global_batch: int
    generator_samples_per_rank: int
    critic_samples_per_rank: int
    contract_id: str

    def __post_init__(self) -> None:
        if self.generator_ranks + self.critic_ranks != 16:
            raise ValueError("layout must use exactly world16")
        if self.generator_ranks * self.generator_samples_per_rank != self.global_batch:
            raise ValueError("Generator lanes do not cover the declared batch")
        if self.critic_ranks * self.critic_samples_per_rank != self.global_batch:
            raise ValueError("Critic lanes do not cover the declared batch")


DEFAULT_LAYOUT_CONTRACTS = (
    LayoutContract("4G12C-B12", 4, 12, 12, 3, 1, "gan-phase-b-b12-v1"),
    LayoutContract("6G10C-B30", 6, 10, 30, 5, 3, "gan-phase-b-b30-v1"),
    LayoutContract("8G8C-B16", 8, 8, 16, 2, 2, "gan-phase-b-b16-v1"),
)


@dataclasses.dataclass(frozen=True)
class LayoutEstimate:
    layout: LayoutContract
    producer_seconds: float
    critic_seconds: float
    steady_batch_seconds: float
    bottleneck: str
    predicted_samples_per_second: float
    matched_to_current_b12: bool


def estimate_layouts(
    *,
    producer_seconds_per_sample: float,
    critic_seconds_per_sample: float,
    layouts: Sequence[LayoutContract] = DEFAULT_LAYOUT_CONTRACTS,
) -> tuple[LayoutEstimate, ...]:
    if producer_seconds_per_sample <= 0 or critic_seconds_per_sample <= 0:
        raise ValueError("service times must be positive")
    estimates: list[LayoutEstimate] = []
    for layout in layouts:
        producer = producer_seconds_per_sample * layout.generator_samples_per_rank
        critic = critic_seconds_per_sample * layout.critic_samples_per_rank
        steady = max(producer, critic)
        estimates.append(
            LayoutEstimate(
                layout=layout,
                producer_seconds=producer,
                critic_seconds=critic,
                steady_batch_seconds=steady,
                bottleneck="generator" if producer > critic else "critic",
                predicted_samples_per_second=layout.global_batch / steady,
                matched_to_current_b12=layout.contract_id == "gan-phase-b-b12-v1",
            )
        )
    return tuple(sorted(estimates, key=lambda value: value.predicted_samples_per_second, reverse=True))


def summarize_service_records(rank_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    producer_sample_seconds: list[float] = []
    producer_send_wait: list[float] = []
    producer_commit_wait: list[float] = []
    critic_compute_seconds: list[float] = []
    critic_receive_wait: list[float] = []
    phase_b_walls: list[float] = []
    for result in rank_results:
        phase_b = result.get("phase_b") or {}
        role = phase_b.get("role")
        for sample in phase_b.get("samples", []):
            if role == "generator":
                if sample.get("produce_seconds") is not None:
                    producer_sample_seconds.append(float(sample["produce_seconds"]))
                if sample.get("send_completion_wait_seconds") is not None:
                    producer_send_wait.append(float(sample["send_completion_wait_seconds"]))
                for key in ("critic_commit_wait_seconds", "commit_wait_seconds"):
                    if sample.get(key) is not None:
                        producer_commit_wait.append(float(sample[key]))
                        break
            elif role == "critic":
                if sample.get("critic_compute_commit_seconds") is not None:
                    critic_compute_seconds.append(float(sample["critic_compute_commit_seconds"]))
                if sample.get("receive_wait_seconds") is not None:
                    critic_receive_wait.append(float(sample["receive_wait_seconds"]))
        if result.get("phase_b_wall_seconds") is not None:
            phase_b_walls.append(float(result["phase_b_wall_seconds"]))

    def stats(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "median": None, "max": None}
        return {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "max": max(values),
        }

    producer_median = statistics.median(producer_sample_seconds) if producer_sample_seconds else None
    critic_median = statistics.median(critic_compute_seconds) if critic_compute_seconds else None
    estimates = (
        [
            {
                "layout": dataclasses.asdict(value.layout),
                "producer_seconds": value.producer_seconds,
                "critic_seconds": value.critic_seconds,
                "steady_batch_seconds": value.steady_batch_seconds,
                "bottleneck": value.bottleneck,
                "predicted_samples_per_second": value.predicted_samples_per_second,
                "matched_to_current_b12": value.matched_to_current_b12,
                "claimable_native_speedup": False,
            }
            for value in estimate_layouts(
                producer_seconds_per_sample=producer_median,
                critic_seconds_per_sample=critic_median,
            )
        ]
        if producer_median is not None and critic_median is not None
        else []
    )
    return {
        "schema_version": "distillschedule.gan_rolephase_service_rates.v2",
        "producer_sample_seconds": stats(producer_sample_seconds),
        "producer_send_wait_seconds": stats(producer_send_wait),
        "producer_commit_wait_seconds": stats(producer_commit_wait),
        "critic_compute_commit_seconds": stats(critic_compute_seconds),
        "critic_receive_wait_seconds": stats(critic_receive_wait),
        "phase_b_wall_seconds": stats(phase_b_walls),
        "layout_estimates": estimates,
        "claim_boundary": {
            "diagnostic_only": True,
            "different_batch_contracts_not_direct_speedups": True,
            "matched_unprofiled_trials_required": True,
        },
    }


def detect_zero_stage(engine: Any) -> int:
    method = getattr(engine, "zero_optimization_stage", None)
    if callable(method):
        return int(method())
    optimizer = getattr(engine, "optimizer", None)
    for owner in (engine, optimizer):
        if owner is None:
            continue
        for name in ("zero_stage", "stage"):
            value = getattr(owner, name, None)
            if value is not None and not callable(value):
                return int(value)
    raise RuntimeError("unable to determine DeepSpeed ZeRO stage")


def validate_zero_engine(engine: Any, *, expected_stage: int) -> dict[str, Any]:
    """Validate ZeRO without using Stage-1/2-only fields on Stage-3."""

    stage = detect_zero_stage(engine)
    if stage != int(expected_stage):
        raise AssertionError(f"ZeRO stage changed: {stage} != {expected_stage}")
    optimizer = getattr(engine, "optimizer", None)
    receipt: dict[str, Any] = {
        "zero_stage": stage,
        "optimizer_type": type(optimizer).__name__ if optimizer is not None else None,
        "partition_gradients_checked": False,
    }
    if stage in (1, 2):
        value = getattr(optimizer, "partition_gradients", None)
        if value is not None:
            receipt["partition_gradients_checked"] = True
            receipt["partition_gradients"] = bool(value)
            expected = stage == 2
            if bool(value) != expected:
                raise AssertionError(
                    f"ZeRO-{stage} partition_gradients drift: {bool(value)} != {expected}"
                )
    elif stage == 3:
        receipt.update(
            stage3_public_contract=True,
            stage3_guard_does_not_access_partition_gradients=True,
            parameter_offload_present=bool(
                getattr(optimizer, "parameter_offload", None)
                or getattr(optimizer, "offload_param", None)
            ),
        )
    else:
        raise AssertionError(f"unsupported ZeRO stage {stage}")
    return receipt


__all__ = [
    "CGROUP_MEMORY_ROOT",
    "CgroupMemorySnapshot",
    "DEFAULT_LAYOUT_CONTRACTS",
    "JsonlStageRecorder",
    "LayoutContract",
    "LayoutEstimate",
    "PersistentTargetPolicy",
    "StageRecord",
    "TransitionSubphase",
    "assert_no_new_cgroup_oom",
    "assert_preconstruction_admitted",
    "build_preconstruction_waves",
    "detect_zero_stage",
    "estimate_layouts",
    "read_cgroup_memory_snapshot",
    "summarize_service_records",
    "validate_zero_engine",
]
