"""Append-only evidence and coarse 50-cycle trajectory comparison.

This module is deliberately independent of the strict P0 tensor-parity gate.
P0 answers whether a physical rewrite is numerically interchangeable at one
boundary.  ``trajectory_coarse_v1`` instead answers whether two complete,
matched, single-seed runs follow broadly similar training trajectories over
50 cycles.  Its numerical thresholds are evaluated *after* the run; they are
never an early-stop policy.

Workload entry points own tensor extraction and streaming state comparison.
The generic layer stores compact per-cycle receipts, enforces the matched
execution contract, aggregates rank losses, and renders auditable reports.
"""
from __future__ import annotations

import csv
import dataclasses
import fcntl
import hashlib
import io
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "distillschedule.trajectory_coarse_v1"
COMPARISON_SCHEMA_VERSION = "distillschedule.trajectory_coarse_comparison.v1"
STATE_COMPARISON_SCHEMA_VERSION = "distillschedule.trajectory_state_comparison.v1"


@dataclasses.dataclass(frozen=True)
class TrajectoryThresholds:
    """Loose, post-run thresholds for a single-seed 50-cycle comparison."""

    min_cosine: float = 0.999
    max_normalized_l2: float = 0.05
    min_norm_ratio: float = 0.95
    max_norm_ratio: float = 1.05
    min_named_tensor_cosine: float = 0.995
    max_curve_normalized_rmse: float = 0.10
    min_moving_average_correlation: float = 0.95
    max_tail_symmetric_relative_difference: float = 0.10
    moving_average_window: int = 5
    tail_cycles: int = 10
    trend_cycles: int = 10
    near_zero: float = 1e-12
    trend_relative_significance: float = 0.01
    noise_floor_multiplier: float = 3.0

    def __post_init__(self) -> None:
        for name in ("min_cosine", "min_named_tensor_cosine", "min_moving_average_correlation"):
            value = float(getattr(self, name))
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")
        for name in (
            "max_normalized_l2",
            "max_curve_normalized_rmse",
            "max_tail_symmetric_relative_difference",
            "near_zero",
            "trend_relative_significance",
            "noise_floor_multiplier",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.min_norm_ratio < 0.0 or self.max_norm_ratio < self.min_norm_ratio:
            raise ValueError("invalid norm-ratio interval")
        for name in ("moving_average_window", "tail_cycles", "trend_cycles"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


COARSE_50C_THRESHOLDS = TrajectoryThresholds()


def _require_nonempty(value: str, *, name: str) -> str:
    value = str(value)
    if not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value


def _string_mapping(value: Mapping[str, Any], *, name: str, allow_empty: bool = False) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise TypeError(f"{name} keys must be strings")
        key = _require_nonempty(raw_key, name=f"{name} key")
        if not isinstance(raw_value, str):
            raise TypeError(f"{name}.{key} must be a string")
        result[key] = _require_nonempty(raw_value, name=f"{name}.{key}")
    if not allow_empty and not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _integer_mapping(value: Mapping[str, Any], *, name: str, allow_empty: bool = False) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        key = _require_nonempty(str(raw_key), name=f"{name} key")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise TypeError(f"{name}.{key} must be an integer")
        if raw_value < 0:
            raise ValueError(f"{name}.{key} must be non-negative")
        result[key] = int(raw_value)
    if not allow_empty and not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _metric_mapping(
    value: Mapping[str, Any],
    *,
    name: str,
    allow_empty: bool = False,
    non_negative: bool = False,
) -> dict[str, float | None]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, float | None] = {}
    for raw_key, raw_value in value.items():
        key = _require_nonempty(str(raw_key), name=f"{name} key")
        if raw_value is None:
            result[key] = None
            continue
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(f"{name}.{key} must be a number or null")
        number = float(raw_value)
        if not math.isfinite(number):
            raise ValueError(f"{name}.{key} must be finite or null")
        if non_negative and number < 0.0:
            raise ValueError(f"{name}.{key} must be non-negative")
        result[key] = number
    if not allow_empty and not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _tensor_contracts(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError("tensor_contracts must be a mapping")
    result: dict[str, dict[str, Any]] = {}
    for raw_key, raw_contract in value.items():
        key = _require_nonempty(str(raw_key), name="tensor_contracts key")
        if not isinstance(raw_contract, Mapping):
            raise TypeError(f"tensor_contracts.{key} must be a mapping")
        unknown = set(raw_contract) - {"shape", "dtype"}
        if unknown:
            raise ValueError(f"tensor_contracts.{key} has unknown fields: {sorted(unknown)}")
        shape = raw_contract.get("shape")
        dtype = raw_contract.get("dtype")
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
            raise TypeError(f"tensor_contracts.{key}.shape must be a sequence")
        normalized_shape: list[int] = []
        for dimension in shape:
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
                raise ValueError(f"tensor_contracts.{key}.shape contains an invalid dimension")
            normalized_shape.append(int(dimension))
        if not isinstance(dtype, str):
            raise TypeError(f"tensor_contracts.{key}.dtype must be a string")
        result[key] = {
            "shape": normalized_shape,
            "dtype": _require_nonempty(dtype, name=f"tensor_contracts.{key}.dtype"),
        }
    return result


@dataclasses.dataclass(frozen=True)
class TrajectoryCycleReceipt:
    """One rank's observed evidence for one completed logical DMD cycle."""

    run_id: str
    workload: str
    variant: str
    rank: int
    world_size: int
    seed: int
    cycle: int
    expected_total_cycles: int
    logical_step: int
    matched_contract: Mapping[str, str]
    implementation: Mapping[str, str]
    sample_identities: Sequence[str]
    rng_identities: Mapping[str, str]
    versions: Mapping[str, int]
    cycle_updates: Mapping[str, int]
    cumulative_updates: Mapping[str, int]
    call_census: Mapping[str, int]
    losses: Mapping[str, float | None]
    gradient_norms: Mapping[str, float | None]
    tensor_contracts: Mapping[str, Mapping[str, Any]]
    state_fingerprints: Mapping[str, str]
    finite: bool
    runtime_status: str = "PASS"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema {self.schema_version!r}")
        object.__setattr__(self, "run_id", _require_nonempty(self.run_id, name="run_id"))
        object.__setattr__(self, "workload", _require_nonempty(self.workload, name="workload"))
        object.__setattr__(self, "variant", _require_nonempty(self.variant, name="variant"))
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("rank must be an integer")
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int) or self.world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            isinstance(self.expected_total_cycles, bool)
            or not isinstance(self.expected_total_cycles, int)
            or self.expected_total_cycles <= 0
        ):
            raise ValueError("expected_total_cycles must be a positive integer")
        if isinstance(self.cycle, bool) or not isinstance(self.cycle, int):
            raise TypeError("cycle must be an integer")
        if not 1 <= self.cycle <= self.expected_total_cycles:
            raise ValueError("cycle must be in [1, expected_total_cycles]")
        if isinstance(self.logical_step, bool) or not isinstance(self.logical_step, int) or self.logical_step < 0:
            raise ValueError("logical_step must be a non-negative integer")
        if not isinstance(self.finite, bool):
            raise TypeError("finite must be a boolean")
        object.__setattr__(
            self, "runtime_status", _require_nonempty(self.runtime_status, name="runtime_status").upper()
        )
        object.__setattr__(
            self, "matched_contract", _string_mapping(self.matched_contract, name="matched_contract")
        )
        object.__setattr__(
            self, "implementation", _string_mapping(self.implementation, name="implementation")
        )
        if isinstance(self.sample_identities, (str, bytes)) or not isinstance(
            self.sample_identities, Sequence
        ):
            raise TypeError("sample_identities must be a sequence")
        samples = tuple(
            _require_nonempty(str(value), name="sample_identities entry")
            for value in self.sample_identities
        )
        if not samples:
            raise ValueError("sample_identities cannot be empty")
        if len(samples) != len(set(samples)):
            raise ValueError("sample_identities must be unique within a rank-cycle receipt")
        object.__setattr__(self, "sample_identities", samples)
        object.__setattr__(
            self, "rng_identities", _string_mapping(self.rng_identities, name="rng_identities")
        )
        object.__setattr__(self, "versions", _integer_mapping(self.versions, name="versions"))
        object.__setattr__(
            self, "cycle_updates", _integer_mapping(self.cycle_updates, name="cycle_updates")
        )
        object.__setattr__(
            self,
            "cumulative_updates",
            _integer_mapping(self.cumulative_updates, name="cumulative_updates"),
        )
        object.__setattr__(
            self, "call_census", _integer_mapping(self.call_census, name="call_census")
        )
        object.__setattr__(self, "losses", _metric_mapping(self.losses, name="losses"))
        object.__setattr__(
            self,
            "gradient_norms",
            _metric_mapping(
                self.gradient_norms,
                name="gradient_norms",
                allow_empty=True,
                non_negative=True,
            ),
        )
        object.__setattr__(self, "tensor_contracts", _tensor_contracts(self.tensor_contracts))
        object.__setattr__(
            self,
            "state_fingerprints",
            _string_mapping(
                self.state_fingerprints,
                name="state_fingerprints",
                allow_empty=True,
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "workload": self.workload,
            "variant": self.variant,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "cycle": self.cycle,
            "expected_total_cycles": self.expected_total_cycles,
            "logical_step": self.logical_step,
            "matched_contract": dict(self.matched_contract),
            "implementation": dict(self.implementation),
            "sample_identities": list(self.sample_identities),
            "rng_identities": dict(self.rng_identities),
            "versions": dict(self.versions),
            "cycle_updates": dict(self.cycle_updates),
            "cumulative_updates": dict(self.cumulative_updates),
            "call_census": dict(self.call_census),
            "losses": dict(self.losses),
            "gradient_norms": dict(self.gradient_norms),
            "tensor_contracts": {
                key: {"shape": list(value["shape"]), "dtype": str(value["dtype"])}
                for key, value in self.tensor_contracts.items()
            },
            "state_fingerprints": dict(self.state_fingerprints),
            "finite": self.finite,
            "runtime_status": self.runtime_status,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrajectoryCycleReceipt:
        if not isinstance(value, Mapping):
            raise TypeError("trajectory receipt must be a mapping")
        fields = {field.name for field in dataclasses.fields(cls)}
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"trajectory receipt has unknown fields: {sorted(unknown)}")
        missing = fields - set(value) - {"runtime_status", "schema_version"}
        if missing:
            raise ValueError(f"trajectory receipt is missing fields: {sorted(missing)}")
        return cls(**dict(value))


@dataclasses.dataclass(frozen=True)
class TrajectoryStateComparison:
    """Compact streaming comparison for a gradient or state delta."""

    cycle: int
    component: str
    state_kind: str
    cosine: float
    normalized_l2: float
    norm_ratio: float
    min_named_tensor_cosine: float
    finite: bool = True
    reference_digest: str | None = None
    candidate_digest: str | None = None
    schema_version: str = STATE_COMPARISON_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATE_COMPARISON_SCHEMA_VERSION:
            raise ValueError(f"unsupported state-comparison schema {self.schema_version!r}")
        if isinstance(self.cycle, bool) or not isinstance(self.cycle, int) or self.cycle <= 0:
            raise ValueError("state comparison cycle must be positive")
        object.__setattr__(self, "component", _require_nonempty(self.component, name="component"))
        object.__setattr__(self, "state_kind", _require_nonempty(self.state_kind, name="state_kind"))
        if not isinstance(self.finite, bool):
            raise TypeError("finite must be a boolean")
        for name in ("cosine", "normalized_l2", "norm_ratio", "min_named_tensor_cosine"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.normalized_l2 < 0.0 or self.norm_ratio < 0.0:
            raise ValueError("normalized_l2 and norm_ratio must be non-negative")

    def evaluate(self, thresholds: TrajectoryThresholds) -> dict[str, Any]:
        passed = (
            self.finite
            and self.cosine >= thresholds.min_cosine
            and self.normalized_l2 <= thresholds.max_normalized_l2
            and thresholds.min_norm_ratio <= self.norm_ratio <= thresholds.max_norm_ratio
            and self.min_named_tensor_cosine >= thresholds.min_named_tensor_cosine
        )
        return {
            **dataclasses.asdict(self),
            "status": "PASS" if passed else "REVIEW",
            "thresholds": {
                "min_cosine": thresholds.min_cosine,
                "max_normalized_l2": thresholds.max_normalized_l2,
                "norm_ratio": [thresholds.min_norm_ratio, thresholds.max_norm_ratio],
                "min_named_tensor_cosine": thresholds.min_named_tensor_cosine,
            },
        }


def load_trajectory_jsonl(path: Path | str) -> list[TrajectoryCycleReceipt]:
    """Load a complete append-only rank file; malformed lines are evidence failure."""

    source = Path(path)
    rows: list[TrajectoryCycleReceipt] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row {source}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL {source}:{line_number}") from exc
            if not isinstance(value, Mapping):
                raise TypeError(f"non-object JSONL row {source}:{line_number}")
            try:
                rows.append(TrajectoryCycleReceipt.from_mapping(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid trajectory row {source}:{line_number}: {exc}") from exc
    return rows


class AppendOnlyTrajectoryWriter:
    """Process-safe append-only writer for one rank's trajectory receipts."""

    def __init__(self, path: Path | str, *, run_id: str, rank: int, fsync: bool = True) -> None:
        self.path = Path(path)
        self.run_id = _require_nonempty(run_id, name="run_id")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a non-negative integer")
        self.rank = rank
        self.fsync = bool(fsync)

    def append(self, receipt: TrajectoryCycleReceipt | Mapping[str, Any]) -> None:
        value = (
            receipt
            if isinstance(receipt, TrajectoryCycleReceipt)
            else TrajectoryCycleReceipt.from_mapping(receipt)
        )
        if value.run_id != self.run_id or value.rank != self.rank:
            raise ValueError(
                "receipt does not match writer identity: "
                f"writer=({self.run_id},rank={self.rank}) "
                f"receipt=({value.run_id},rank={value.rank})"
            )
        serialized = (
            json.dumps(value.to_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o640)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            existing_bytes = self.path.read_bytes()
            if existing_bytes and not existing_bytes.endswith(b"\n"):
                raise ValueError("existing append-only trajectory file does not end with a newline")
            existing = load_trajectory_jsonl(self.path) if existing_bytes else []
            if existing:
                first = existing[0]
                for row in existing:
                    if (
                        row.run_id != self.run_id
                        or row.rank != self.rank
                        or row.workload != first.workload
                        or row.variant != first.variant
                        or row.world_size != first.world_size
                        or row.expected_total_cycles != first.expected_total_cycles
                    ):
                        raise ValueError("existing trajectory file contains mixed writer identities")
                expected_cycle = existing[-1].cycle + 1
            else:
                expected_cycle = 1
            if value.cycle != expected_cycle:
                raise ValueError(
                    "trajectory cycles must strictly increase by one: "
                    f"expected={expected_cycle} observed={value.cycle}"
                )
            written = os.write(descriptor, serialized)
            if written != len(serialized):
                raise OSError(f"short trajectory append: wrote {written}/{len(serialized)} bytes")
            if self.fsync:
                os.fsync(descriptor)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _coerce_rows(rows: Iterable[TrajectoryCycleReceipt | Mapping[str, Any]]) -> list[TrajectoryCycleReceipt]:
    return [
        value
        if isinstance(value, TrajectoryCycleReceipt)
        else TrajectoryCycleReceipt.from_mapping(value)
        for value in rows
    ]


def _moving_average(values: Sequence[float], window: int) -> list[float]:
    if len(values) < window:
        return []
    running = sum(values[:window])
    result = [running / window]
    for index in range(window, len(values)):
        running += values[index] - values[index - window]
        result.append(running / window)
    return result


def _pearson(reference: Sequence[float], candidate: Sequence[float], near_zero: float) -> float | None:
    if len(reference) != len(candidate) or not reference:
        return None
    ref_mean = sum(reference) / len(reference)
    cand_mean = sum(candidate) / len(candidate)
    ref_centered = [value - ref_mean for value in reference]
    cand_centered = [value - cand_mean for value in candidate]
    ref_norm = math.sqrt(sum(value * value for value in ref_centered))
    cand_norm = math.sqrt(sum(value * value for value in cand_centered))
    if ref_norm <= near_zero or cand_norm <= near_zero:
        return None
    return sum(left * right for left, right in zip(ref_centered, cand_centered)) / (
        ref_norm * cand_norm
    )


def _symmetric_relative_difference(left: float, right: float, near_zero: float) -> float | None:
    denominator = (abs(left) + abs(right)) / 2.0
    if denominator <= near_zero:
        return None
    return abs(left - right) / denominator


def _run_inventory(
    rows: Sequence[TrajectoryCycleReceipt],
    *,
    label: str,
    expected_cycles: int,
    expected_world_size: int,
) -> tuple[dict[tuple[int, int], TrajectoryCycleReceipt], list[str], bool, dict[str, Any]]:
    errors: list[str] = []
    indexed: dict[tuple[int, int], TrajectoryCycleReceipt] = {}
    for row in rows:
        address = (row.cycle, row.rank)
        if address in indexed:
            errors.append(f"{label}: duplicate receipt cycle={row.cycle} rank={row.rank}")
        else:
            indexed[address] = row
    expected = {
        (cycle, rank)
        for cycle in range(1, expected_cycles + 1)
        for rank in range(expected_world_size)
    }
    observed = set(indexed)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        errors.append(f"{label}: missing rank-cycle receipts {missing[:20]} (count={len(missing)})")
    if extra:
        errors.append(f"{label}: unexpected rank-cycle receipts {extra[:20]} (count={len(extra)})")

    if rows:
        run_ids = sorted({row.run_id for row in rows})
        workloads = sorted({row.workload for row in rows})
        variants = sorted({row.variant for row in rows})
        seeds = sorted({row.seed for row in rows})
        world_sizes = sorted({row.world_size for row in rows})
        expected_counts = sorted({row.expected_total_cycles for row in rows})
        for field, values, expected_value in (
            ("run_id", run_ids, None),
            ("workload", workloads, None),
            ("variant", variants, None),
            ("seed", seeds, None),
            ("world_size", world_sizes, expected_world_size),
            ("expected_total_cycles", expected_counts, expected_cycles),
        ):
            if len(values) != 1:
                errors.append(f"{label}: mixed {field} values {values}")
            elif expected_value is not None and values[0] != expected_value:
                errors.append(
                    f"{label}: {field} expected={expected_value} observed={values[0]}"
                )
    else:
        run_ids, workloads, variants, seeds = [], [], [], []
        world_sizes, expected_counts = [], []
        errors.append(f"{label}: no trajectory receipts")

    runtime_failures = [
        (row.cycle, row.rank, row.runtime_status, row.finite)
        for row in rows
        if row.runtime_status != "PASS" or not row.finite
    ]
    if runtime_failures:
        errors.append(f"{label}: runtime/finite failures {runtime_failures[:20]}")
    complete = not missing and not extra and len(indexed) == len(expected)
    summary = {
        "run_ids": run_ids,
        "workloads": workloads,
        "variants": variants,
        "seeds": seeds,
        "world_sizes": world_sizes,
        "expected_total_cycles": expected_counts,
        "row_count": len(rows),
        "expected_row_count": len(expected),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "runtime_failure_count": len(runtime_failures),
    }
    return indexed, errors, complete and not runtime_failures, summary


def _curve_metrics(
    name: str,
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    thresholds: TrajectoryThresholds,
    noise_floor: float,
) -> dict[str, Any]:
    points = [
        {
            "cycle": index + 1,
            "reference": left,
            "candidate": right,
            "difference": right - left,
        }
        for index, (left, right) in enumerate(zip(reference, candidate))
    ]
    ref_rms = math.sqrt(sum(value * value for value in reference) / len(reference))
    cand_rms = math.sqrt(sum(value * value for value in candidate) / len(candidate))
    difference_rms = math.sqrt(
        sum((left - right) ** 2 for left, right in zip(reference, candidate)) / len(reference)
    )
    if ref_rms <= thresholds.near_zero and cand_rms <= thresholds.near_zero:
        normalized_rmse = 0.0
        normalized_rmse_policy = "BOTH_NEAR_ZERO_DEFINED_AS_ZERO"
    elif ref_rms <= thresholds.near_zero:
        normalized_rmse = None
        normalized_rmse_policy = "REFERENCE_NEAR_ZERO_CANDIDATE_NONZERO"
    else:
        normalized_rmse = difference_rms / ref_rms
        normalized_rmse_policy = "REFERENCE_RMS"

    ref_smoothed = _moving_average(reference, thresholds.moving_average_window)
    cand_smoothed = _moving_average(candidate, thresholds.moving_average_window)
    correlation = _pearson(ref_smoothed, cand_smoothed, thresholds.near_zero)
    correlation_policy = "STANDARD" if correlation is not None else "N/A_NEAR_CONSTANT"

    tail_count = min(thresholds.tail_cycles, len(reference))
    ref_tail = sum(reference[-tail_count:]) / tail_count
    cand_tail = sum(candidate[-tail_count:]) / tail_count
    tail_difference = _symmetric_relative_difference(ref_tail, cand_tail, thresholds.near_zero)
    tail_policy = "STANDARD" if tail_difference is not None else "N/A_BOTH_NEAR_ZERO"

    trend_count = min(thresholds.trend_cycles, len(reference))
    ref_first = sum(reference[:trend_count]) / trend_count
    ref_last = sum(reference[-trend_count:]) / trend_count
    cand_first = sum(candidate[:trend_count]) / trend_count
    cand_last = sum(candidate[-trend_count:]) / trend_count
    ref_trend = ref_last - ref_first
    cand_trend = cand_last - cand_first
    trend_scale = max(ref_rms, abs(ref_first), abs(ref_last), thresholds.near_zero)
    trend_threshold = max(
        thresholds.near_zero,
        thresholds.trend_relative_significance * trend_scale,
        thresholds.noise_floor_multiplier * max(0.0, float(noise_floor)),
    )
    trend_significant = abs(ref_trend) > trend_threshold
    trend_direction_match: bool | None = (
        (ref_trend > 0.0) == (cand_trend > 0.0) if trend_significant else None
    )

    checks = {
        "normalized_rmse": (
            normalized_rmse is not None
            and normalized_rmse <= thresholds.max_curve_normalized_rmse
        ),
        "moving_average_correlation": (
            True if correlation is None else correlation >= thresholds.min_moving_average_correlation
        ),
        "tail_symmetric_relative_difference": (
            True
            if tail_difference is None
            else tail_difference <= thresholds.max_tail_symmetric_relative_difference
        ),
        "trend_direction": True if trend_direction_match is None else trend_direction_match,
    }
    return {
        "metric": name,
        "status": "PASS" if all(checks.values()) else "REVIEW",
        "points": points,
        "normalized_rmse": normalized_rmse,
        "normalized_rmse_policy": normalized_rmse_policy,
        "moving_average_correlation": correlation,
        "moving_average_correlation_policy": correlation_policy,
        "tail_symmetric_relative_difference": tail_difference,
        "tail_difference_policy": tail_policy,
        "reference_tail_mean": ref_tail,
        "candidate_tail_mean": cand_tail,
        "reference_trend": ref_trend,
        "candidate_trend": cand_trend,
        "trend_significant": trend_significant,
        "trend_direction_match": trend_direction_match,
        "trend_significance_threshold": trend_threshold,
        "noise_floor": noise_floor,
        "checks": checks,
    }


def compare_trajectories(
    reference_rows: Iterable[TrajectoryCycleReceipt | Mapping[str, Any]],
    candidate_rows: Iterable[TrajectoryCycleReceipt | Mapping[str, Any]],
    *,
    state_comparisons: Iterable[TrajectoryStateComparison] = (),
    thresholds: TrajectoryThresholds = COARSE_50C_THRESHOLDS,
    expected_cycles: int = 50,
    expected_world_size: int = 16,
    comparison_kind: str = "semantic",
    curve_noise_floor: Mapping[str, float] | None = None,
    required_state_cycles: Sequence[int] = (),
) -> dict[str, Any]:
    """Compare two completed matched runs without imposing an early-stop gate."""

    if expected_cycles <= 0 or expected_world_size <= 0:
        raise ValueError("expected_cycles and expected_world_size must be positive")
    if comparison_kind not in {"semantic", "approximation"}:
        raise ValueError("comparison_kind must be 'semantic' or 'approximation'")
    reference = _coerce_rows(reference_rows)
    candidate = _coerce_rows(candidate_rows)
    ref_by_address, ref_errors, ref_complete, ref_summary = _run_inventory(
        reference,
        label="reference",
        expected_cycles=expected_cycles,
        expected_world_size=expected_world_size,
    )
    cand_by_address, cand_errors, cand_complete, cand_summary = _run_inventory(
        candidate,
        label="candidate",
        expected_cycles=expected_cycles,
        expected_world_size=expected_world_size,
    )
    inventory_errors = [*ref_errors, *cand_errors]
    complete = ref_complete and cand_complete
    base_result: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "comparison_kind": comparison_kind,
        "expected_cycles": expected_cycles,
        "expected_world_size": expected_world_size,
        "complete": complete,
        "evaluation_policy": "POST_RUN_ONLY_NO_NUMERICAL_EARLY_STOP",
        "thresholds": dataclasses.asdict(thresholds),
        "reference": ref_summary,
        "candidate": cand_summary,
        "contract": {"status": "FAIL" if inventory_errors else "PASS", "mismatches": inventory_errors},
        "curves": {},
        "gradient_norm_curves": {},
        "state_comparisons": [],
    }
    if not complete:
        base_result.update(
            {
                "status": "RUNTIME_FAIL",
                "claim_boundary": "INCOMPLETE_OR_NONFINITE_RUN_NO_TRAJECTORY_CLAIM",
            }
        )
        return base_result

    contract_mismatches = list(inventory_errors)
    ref_run = reference[0]
    cand_run = candidate[0]
    if ref_run.workload != cand_run.workload:
        contract_mismatches.append(
            f"workload mismatch reference={ref_run.workload!r} candidate={cand_run.workload!r}"
        )
    if ref_run.seed != cand_run.seed:
        contract_mismatches.append(
            f"seed mismatch reference={ref_run.seed} candidate={cand_run.seed}"
        )

    exact_fields = (
        "logical_step",
        "matched_contract",
        "sample_identities",
        "rng_identities",
        "versions",
        "cycle_updates",
        "cumulative_updates",
        "call_census",
        "tensor_contracts",
    )
    for cycle in range(1, expected_cycles + 1):
        for rank in range(expected_world_size):
            reference_row = ref_by_address[(cycle, rank)]
            candidate_row = cand_by_address[(cycle, rank)]
            for field in exact_fields:
                left = getattr(reference_row, field)
                right = getattr(candidate_row, field)
                if left != right:
                    contract_mismatches.append(
                        f"cycle={cycle} rank={rank} {field} mismatch: "
                        f"reference={left!r} candidate={right!r}"
                    )
            if set(reference_row.state_fingerprints) != set(candidate_row.state_fingerprints):
                contract_mismatches.append(
                    f"cycle={cycle} rank={rank} state_fingerprint key mismatch: "
                    f"reference={sorted(reference_row.state_fingerprints)} "
                    f"candidate={sorted(candidate_row.state_fingerprints)}"
                )

    for label, indexed in (("reference", ref_by_address), ("candidate", cand_by_address)):
        for cycle in range(1, expected_cycles + 1):
            identities = [
                identity
                for rank in range(expected_world_size)
                for identity in indexed[(cycle, rank)].sample_identities
            ]
            if len(identities) != len(set(identities)):
                contract_mismatches.append(
                    f"{label} cycle={cycle} has duplicate rank-qualified sample identities"
                )

    reference_loss_keys = {key for row in reference for key in row.losses}
    candidate_loss_keys = {key for row in candidate for key in row.losses}
    if reference_loss_keys != candidate_loss_keys:
        contract_mismatches.append(
            "loss key mismatch: "
            f"reference={sorted(reference_loss_keys)} candidate={sorted(candidate_loss_keys)}"
        )
    loss_keys = sorted(reference_loss_keys & candidate_loss_keys)
    for key in loss_keys:
        for label, rows in (("reference", reference), ("candidate", candidate)):
            missing = [(row.cycle, row.rank) for row in rows if key not in row.losses or row.losses[key] is None]
            if missing:
                contract_mismatches.append(
                    f"{label} loss {key!r} missing/nonfinite at {missing[:20]} (count={len(missing)})"
                )

    curves: dict[str, dict[str, Any]] = {}
    gradient_norm_curves: dict[str, dict[str, Any]] = {}
    noise = dict(curve_noise_floor or {})
    if not any("loss " in value and "missing/nonfinite" in value for value in contract_mismatches):
        for key in loss_keys:
            ref_values = [
                sum(float(ref_by_address[(cycle, rank)].losses[key]) for rank in range(expected_world_size))
                / expected_world_size
                for cycle in range(1, expected_cycles + 1)
            ]
            cand_values = [
                sum(float(cand_by_address[(cycle, rank)].losses[key]) for rank in range(expected_world_size))
                / expected_world_size
                for cycle in range(1, expected_cycles + 1)
            ]
            curves[key] = _curve_metrics(
                key,
                ref_values,
                cand_values,
                thresholds=thresholds,
                noise_floor=float(noise.get(key, 0.0)),
            )

    reference_gradient_keys = {key for row in reference for key in row.gradient_norms}
    candidate_gradient_keys = {key for row in candidate for key in row.gradient_norms}
    if reference_gradient_keys != candidate_gradient_keys:
        contract_mismatches.append(
            "gradient-norm key mismatch: "
            f"reference={sorted(reference_gradient_keys)} candidate={sorted(candidate_gradient_keys)}"
        )
    gradient_keys = sorted(reference_gradient_keys & candidate_gradient_keys)
    gradient_values_missing = False
    for key in gradient_keys:
        for label, rows in (("reference", reference), ("candidate", candidate)):
            missing = [
                (row.cycle, row.rank)
                for row in rows
                if key not in row.gradient_norms or row.gradient_norms[key] is None
            ]
            if missing:
                gradient_values_missing = True
                contract_mismatches.append(
                    f"{label} gradient norm {key!r} missing/nonfinite at "
                    f"{missing[:20]} (count={len(missing)})"
                )
    if not gradient_values_missing:
        for key in gradient_keys:
            ref_values = [
                sum(
                    float(ref_by_address[(cycle, rank)].gradient_norms[key])
                    for rank in range(expected_world_size)
                )
                / expected_world_size
                for cycle in range(1, expected_cycles + 1)
            ]
            cand_values = [
                sum(
                    float(cand_by_address[(cycle, rank)].gradient_norms[key])
                    for rank in range(expected_world_size)
                )
                / expected_world_size
                for cycle in range(1, expected_cycles + 1)
            ]
            gradient_norm_curves[key] = _curve_metrics(
                key,
                ref_values,
                cand_values,
                thresholds=thresholds,
                noise_floor=float(noise.get(f"gradient_norm/{key}", 0.0)),
            )

    state_rows = [value.evaluate(thresholds) for value in state_comparisons]
    required = {int(value) for value in required_state_cycles}
    observed = {int(value["cycle"]) for value in state_rows}
    missing_state_cycles = sorted(required - observed)
    if missing_state_cycles:
        contract_mismatches.append(f"missing required state-comparison cycles {missing_state_cycles}")

    contract_status = "PASS" if not contract_mismatches else "FAIL"
    numerical_review = (
        any(value["status"] != "PASS" for value in curves.values())
        or any(value["status"] != "PASS" for value in gradient_norm_curves.values())
        or any(value["status"] != "PASS" for value in state_rows)
    )
    if contract_mismatches:
        status = "CONTRACT_MISMATCH_50C_COMPLETED"
        claim_boundary = "COMPLETE_RUN_BUT_CONTRACT_NOT_MATCHED"
    elif numerical_review:
        status = (
            "APPROXIMATION_REVIEW_50C_COMPLETED"
            if comparison_kind == "approximation"
            else "TRAJECTORY_REVIEW_50C_COMPLETED"
        )
        claim_boundary = (
            "APPROXIMATION_NOT_EXACT_EQUIVALENCE"
            if comparison_kind == "approximation"
            else "SINGLE_MATCHED_SEED_COARSE_TRAJECTORY_ONLY"
        )
    elif comparison_kind == "approximation":
        status = "APPROXIMATION_CURVE_CLOSE_50C_SINGLE_SEED"
        claim_boundary = "APPROXIMATION_NOT_EXACT_EQUIVALENCE"
    else:
        status = "TRAJECTORY_MATCHED_50C_SINGLE_SEED"
        claim_boundary = "SINGLE_MATCHED_SEED_COARSE_TRAJECTORY_ONLY"

    base_result.update(
        {
            "status": status,
            "claim_boundary": claim_boundary,
            "contract": {"status": contract_status, "mismatches": contract_mismatches},
            "curves": curves,
            "gradient_norm_curves": gradient_norm_curves,
            "state_comparisons": state_rows,
            "summary": {
                "curve_count": len(curves),
                "curve_review_count": sum(value["status"] != "PASS" for value in curves.values()),
                "gradient_norm_curve_count": len(gradient_norm_curves),
                "gradient_norm_curve_review_count": sum(
                    value["status"] != "PASS" for value in gradient_norm_curves.values()
                ),
                "state_comparison_count": len(state_rows),
                "state_review_count": sum(value["status"] != "PASS" for value in state_rows),
                "contract_mismatch_count": len(contract_mismatches),
            },
        }
    )
    return base_result


def _render_csv(result: Mapping[str, Any]) -> str:
    handle = io.StringIO()
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(
        (
            "metric",
            "cycle",
            "reference",
            "candidate",
            "difference",
            "metric_status",
            "normalized_rmse",
            "moving_average_correlation",
            "tail_symmetric_relative_difference",
        )
    )
    series = [
        (metric, curve)
        for metric, curve in sorted(result.get("curves", {}).items())
    ] + [
        (f"gradient_norm/{metric}", curve)
        for metric, curve in sorted(result.get("gradient_norm_curves", {}).items())
    ]
    for metric, curve in series:
        for point in curve.get("points", []):
            writer.writerow(
                (
                    metric,
                    point["cycle"],
                    point["reference"],
                    point["candidate"],
                    point["difference"],
                    curve["status"],
                    curve["normalized_rmse"],
                    curve["moving_average_correlation"],
                    curve["tail_symmetric_relative_difference"],
                )
            )
    return handle.getvalue()


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Trajectory coarse comparison",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Complete: `{result.get('complete')}`",
        f"- Comparison kind: `{result.get('comparison_kind')}`",
        f"- Claim boundary: `{result.get('claim_boundary')}`",
        f"- Evaluation policy: `{result.get('evaluation_policy')}`",
        "",
        "## Loss curves",
        "",
        "| Metric | Status | normalized RMSE | moving-average correlation | tail symmetric difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, curve in sorted(result.get("curves", {}).items()):
        def display(value: Any) -> str:
            return "N/A" if value is None else f"{float(value):.6g}"

        lines.append(
            f"| {metric} | {curve['status']} | {display(curve['normalized_rmse'])} | "
            f"{display(curve['moving_average_correlation'])} | "
            f"{display(curve['tail_symmetric_relative_difference'])} |"
        )
    lines.extend(
        [
            "",
            "## Gradient-norm curves",
            "",
            "| Metric | Status | normalized RMSE | moving-average correlation | tail symmetric difference |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric, curve in sorted(result.get("gradient_norm_curves", {}).items()):
        def display_gradient(value: Any) -> str:
            return "N/A" if value is None else f"{float(value):.6g}"

        lines.append(
            f"| {metric} | {curve['status']} | {display_gradient(curve['normalized_rmse'])} | "
            f"{display_gradient(curve['moving_average_correlation'])} | "
            f"{display_gradient(curve['tail_symmetric_relative_difference'])} |"
        )
    mismatches = result.get("contract", {}).get("mismatches", [])
    lines.extend(["", "## Contract", "", f"- Status: `{result.get('contract', {}).get('status')}`"])
    if mismatches:
        lines.extend(f"- {value}" for value in mismatches)
    else:
        lines.append("- No matched-contract mismatch observed.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This is a complete-run, single-seed coarse trajectory comparison. Numerical "
                "thresholds are evaluated after the run and are not an early-stop rule. It does "
                "not establish multi-seed statistical equivalence or final generation quality."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _render_png(result: Mapping[str, Any], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised in minimal deployments
        raise RuntimeError("matplotlib is required to render trajectory PNG reports") from exc

    curves = {
        **result.get("curves", {}),
        **{
            f"gradient_norm/{key}": value
            for key, value in result.get("gradient_norm_curves", {}).items()
        },
    }
    if not curves:
        raise ValueError("cannot render a trajectory plot without complete loss curves")
    count = len(curves)
    columns = min(2, count)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(7.0 * columns, 3.8 * rows), squeeze=False)
    flat_axes = [axis for row in axes for axis in row]
    for axis, (metric, curve) in zip(flat_axes, sorted(curves.items())):
        points = curve["points"]
        cycles = [point["cycle"] for point in points]
        axis.plot(cycles, [point["reference"] for point in points], label="baseline", linewidth=1.7)
        axis.plot(cycles, [point["candidate"] for point in points], label="optimized", linewidth=1.7)
        axis.set_title(f"{metric} [{curve['status']}]")
        axis.set_xlabel("cycle")
        axis.set_ylabel("rank-mean loss")
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in flat_axes[count:]:
        axis.set_visible(False)
    figure.suptitle(str(result.get("status", "trajectory comparison")))
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=160)
    plt.close(figure)


def write_trajectory_report(
    result: Mapping[str, Any],
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write JSON, CSV, Markdown, PNG, and hashes without silent overwrite."""

    root = Path(output_dir)
    outputs = {
        "json": root / "trajectory_comparison.json",
        "csv": root / "loss_curves.csv",
        "markdown": root / "trajectory_comparison.md",
        "png": root / "trajectory_comparison.png",
        "sha256": root / "artifact_sha256.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"trajectory report would overwrite existing artifacts: {existing}")
    root.mkdir(parents=True, exist_ok=True)
    nonce = f"{os.getpid()}.{os.urandom(4).hex()}"
    temporary = {key: path.with_name(f".{path.name}.{nonce}.tmp") for key, path in outputs.items()}
    try:
        temporary["json"].write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary["csv"].write_text(_render_csv(result), encoding="utf-8")
        temporary["markdown"].write_text(_render_markdown(result), encoding="utf-8")
        _render_png(result, temporary["png"])
        hashes = {
            outputs[key].name: hashlib.sha256(temporary[key].read_bytes()).hexdigest()
            for key in ("json", "csv", "markdown", "png")
        }
        temporary["sha256"].write_text(
            json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for key in ("json", "csv", "markdown", "png", "sha256"):
            os.replace(temporary[key], outputs[key])
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    return outputs


__all__ = [
    "COARSE_50C_THRESHOLDS",
    "COMPARISON_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "STATE_COMPARISON_SCHEMA_VERSION",
    "AppendOnlyTrajectoryWriter",
    "TrajectoryCycleReceipt",
    "TrajectoryStateComparison",
    "TrajectoryThresholds",
    "compare_trajectories",
    "load_trajectory_jsonl",
    "write_trajectory_report",
]
