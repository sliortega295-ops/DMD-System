"""Post-run orchestration for the five preregistered 50-cycle DMD pairs.

The suite layer intentionally does not launch training.  It consumes immutable,
rank-qualified :mod:`trajectory_coarse` JSONL receipts after both arms finish,
checks the fixed 16-rank/50-cycle execution contract, and publishes one compact
matrix.  A separate append-only JSONL manifest records every run-matrix status
transition, including ``NOT_RUN`` and failed attempts, without rewriting prior
evidence.

MiniMax-H3 Exact versus Grid1000 is always labelled as an approximation study;
the suite cannot promote it to exact semantic equivalence.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .trajectory_coarse import (
    COARSE_50C_THRESHOLDS,
    TrajectoryCycleReceipt,
    TrajectoryStateComparison,
    TrajectoryThresholds,
    compare_trajectories,
    load_trajectory_jsonl,
)

MATRIX_RECORD_SCHEMA_VERSION = "distillschedule.trajectory_run_matrix_record.v1"
SUITE_REPORT_SCHEMA_VERSION = "distillschedule.trajectory_suite_report.v1"
EXPECTED_WORLD_SIZE = 16
EXPECTED_CYCLES = 50
ALLOWED_EXECUTION_STATUSES = frozenset({"PASS", "FAIL", "NOT_RUN"})
ANCHOR_CYCLES = (1, 10, 25, 50)


@dataclasses.dataclass(frozen=True)
class WorkloadDefinition:
    display_name: str
    comparison_kind: str
    claim_boundary: str


WORKLOAD_DEFINITIONS: dict[str, WorkloadDefinition] = {
    "wan1_3b_nogan": WorkloadDefinition(
        "Wan1.3B No-GAN", "semantic", "SINGLE_MATCHED_SEED_COARSE_TRAJECTORY_ONLY"
    ),
    "helios14b_nogan": WorkloadDefinition(
        "Helios14B No-GAN", "semantic", "SINGLE_MATCHED_SEED_COARSE_TRAJECTORY_ONLY"
    ),
    "helios14b_gan_on": WorkloadDefinition(
        "Helios14B GAN-on", "semantic", "SINGLE_MATCHED_SEED_COARSE_TRAJECTORY_ONLY"
    ),
    "livewan_crossscale": WorkloadDefinition(
        "LiveWan 14B-to-1.3B", "semantic", "SINGLE_MATCHED_SEED_COARSE_TRAJECTORY_ONLY"
    ),
    "minimax_h3_exact_vs_grid1000": WorkloadDefinition(
        "MiniMax-H3 Exact vs Grid1000",
        "approximation",
        "APPROXIMATION_NOT_EXACT_EQUIVALENCE",
    ),
}

_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _nonempty(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_map(value: Any, *, name: str, allow_empty: bool = False) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        normalized_key = _nonempty(key, name=f"{name} key")
        result[normalized_key] = _nonempty(item, name=f"{name}.{normalized_key}")
    if not allow_empty and not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _absolute_path_or_none(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    path = Path(_nonempty(value, name=name))
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return str(path)


def _parse_utc(value: str) -> None:
    if not value.endswith("Z"):
        raise ValueError("recorded_at_utc must end in Z")
    try:
        dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("recorded_at_utc must be an ISO-8601 UTC timestamp") from exc


@dataclasses.dataclass(frozen=True)
class TrajectorySuiteArm:
    """Immutable identity and receipt location for one comparison arm."""

    run_id: str
    variant: str
    source_head: str
    assets: Mapping[str, str]
    rank_jsonl_root: str | None = None
    implementation_expectations: Mapping[str, str] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _nonempty(self.run_id, name="arm.run_id"))
        object.__setattr__(self, "variant", _nonempty(self.variant, name="arm.variant"))
        head = _nonempty(self.source_head, name="arm.source_head").lower()
        if not _FULL_GIT_SHA.fullmatch(head):
            raise ValueError(
                "arm.source_head must be an exact 40-character lowercase git SHA"
            )
        object.__setattr__(self, "source_head", head)
        object.__setattr__(self, "assets", _string_map(self.assets, name="arm.assets"))
        object.__setattr__(
            self,
            "rank_jsonl_root",
            _absolute_path_or_none(self.rank_jsonl_root, name="arm.rank_jsonl_root"),
        )
        object.__setattr__(
            self,
            "implementation_expectations",
            _string_map(
                self.implementation_expectations,
                name="arm.implementation_expectations",
                allow_empty=True,
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "variant": self.variant,
            "source_head": self.source_head,
            "assets": dict(self.assets),
            "rank_jsonl_root": self.rank_jsonl_root,
            "implementation_expectations": dict(self.implementation_expectations),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrajectorySuiteArm:
        if not isinstance(value, Mapping):
            raise TypeError("trajectory suite arm must be a mapping")
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"trajectory suite arm has unknown fields: {sorted(unknown)}"
            )
        required = {
            "run_id",
            "variant",
            "source_head",
            "assets",
            "rank_jsonl_root",
            "implementation_expectations",
        }
        missing = required - set(value)
        if missing:
            raise ValueError(
                f"trajectory suite arm is missing fields: {sorted(missing)}"
            )
        return cls(**dict(value))


@dataclasses.dataclass(frozen=True)
class TrajectoryRunMatrixRecord:
    """One append-only status row for one canonical workload pair."""

    suite_id: str
    workload_id: str
    attempt: int
    recorded_at_utc: str
    execution_status: str
    reason: str
    baseline: TrajectorySuiteArm
    candidate: TrajectorySuiteArm
    shared_assets: Mapping[str, str]
    state_comparison_paths: Sequence[str] = ()
    required_state_cycles: Sequence[int] = ()
    curve_noise_floor: Mapping[str, float] = dataclasses.field(default_factory=dict)
    expected_world_size: int = EXPECTED_WORLD_SIZE
    expected_cycles: int = EXPECTED_CYCLES
    schema_version: str = MATRIX_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MATRIX_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported matrix-record schema {self.schema_version!r}"
            )
        object.__setattr__(self, "suite_id", _nonempty(self.suite_id, name="suite_id"))
        workload_id = _nonempty(self.workload_id, name="workload_id")
        if workload_id not in WORKLOAD_DEFINITIONS:
            raise ValueError(
                f"workload_id must be one of {sorted(WORKLOAD_DEFINITIONS)}, got {workload_id!r}"
            )
        object.__setattr__(self, "workload_id", workload_id)
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt <= 0
        ):
            raise ValueError("attempt must be a positive integer")
        timestamp = _nonempty(self.recorded_at_utc, name="recorded_at_utc")
        _parse_utc(timestamp)
        object.__setattr__(self, "recorded_at_utc", timestamp)
        status = _nonempty(self.execution_status, name="execution_status").upper()
        if status not in ALLOWED_EXECUTION_STATUSES:
            raise ValueError(
                f"execution_status must be one of {sorted(ALLOWED_EXECUTION_STATUSES)}"
            )
        object.__setattr__(self, "execution_status", status)
        object.__setattr__(self, "reason", _nonempty(self.reason, name="reason"))
        if not isinstance(self.baseline, TrajectorySuiteArm):
            object.__setattr__(
                self, "baseline", TrajectorySuiteArm.from_mapping(self.baseline)
            )
        if not isinstance(self.candidate, TrajectorySuiteArm):
            object.__setattr__(
                self, "candidate", TrajectorySuiteArm.from_mapping(self.candidate)
            )
        object.__setattr__(
            self, "shared_assets", _string_map(self.shared_assets, name="shared_assets")
        )
        if self.expected_world_size != EXPECTED_WORLD_SIZE:
            raise ValueError(
                f"expected_world_size must be exactly {EXPECTED_WORLD_SIZE}"
            )
        if self.expected_cycles != EXPECTED_CYCLES:
            raise ValueError(f"expected_cycles must be exactly {EXPECTED_CYCLES}")
        if status == "PASS" and (
            self.baseline.rank_jsonl_root is None
            or self.candidate.rank_jsonl_root is None
        ):
            raise ValueError(
                "PASS records require explicit baseline and candidate rank-JSONL roots"
            )

        state_paths: list[str] = []
        for index, raw_path in enumerate(self.state_comparison_paths):
            normalized = _absolute_path_or_none(
                raw_path, name=f"state_comparison_paths[{index}]"
            )
            assert normalized is not None
            state_paths.append(normalized)
        object.__setattr__(self, "state_comparison_paths", tuple(state_paths))

        state_cycles: list[int] = []
        for cycle in self.required_state_cycles:
            if isinstance(cycle, bool) or not isinstance(cycle, int):
                raise TypeError("required_state_cycles entries must be integers")
            if cycle not in ANCHOR_CYCLES:
                raise ValueError(
                    f"required_state_cycles must be a subset of {ANCHOR_CYCLES}"
                )
            state_cycles.append(cycle)
        if len(state_cycles) != len(set(state_cycles)):
            raise ValueError("required_state_cycles cannot contain duplicates")
        if state_cycles and not state_paths:
            raise ValueError("required_state_cycles require state_comparison_paths")
        object.__setattr__(self, "required_state_cycles", tuple(sorted(state_cycles)))

        noise: dict[str, float] = {}
        if not isinstance(self.curve_noise_floor, Mapping):
            raise TypeError("curve_noise_floor must be a mapping")
        for key, value in self.curve_noise_floor.items():
            normalized_key = _nonempty(key, name="curve_noise_floor key")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"curve_noise_floor.{normalized_key} must be numeric")
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(
                    f"curve_noise_floor.{normalized_key} must be finite and non-negative"
                )
            noise[normalized_key] = number
        object.__setattr__(self, "curve_noise_floor", noise)

    @property
    def comparison_kind(self) -> str:
        return WORKLOAD_DEFINITIONS[self.workload_id].comparison_kind

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "workload_id": self.workload_id,
            "comparison_kind": self.comparison_kind,
            "attempt": self.attempt,
            "recorded_at_utc": self.recorded_at_utc,
            "execution_status": self.execution_status,
            "reason": self.reason,
            "expected_world_size": self.expected_world_size,
            "expected_cycles": self.expected_cycles,
            "baseline": self.baseline.to_mapping(),
            "candidate": self.candidate.to_mapping(),
            "shared_assets": dict(self.shared_assets),
            "state_comparison_paths": list(self.state_comparison_paths),
            "required_state_cycles": list(self.required_state_cycles),
            "curve_noise_floor": dict(self.curve_noise_floor),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrajectoryRunMatrixRecord:
        if not isinstance(value, Mapping):
            raise TypeError("trajectory run-matrix record must be a mapping")
        data = dict(value)
        has_comparison_kind = "comparison_kind" in data
        comparison_kind = data.pop("comparison_kind", None)
        fields = {field.name for field in dataclasses.fields(cls)}
        unknown = set(data) - fields
        if unknown:
            raise ValueError(
                f"trajectory run-matrix record has unknown fields: {sorted(unknown)}"
            )
        required = {
            "schema_version",
            "suite_id",
            "workload_id",
            "attempt",
            "recorded_at_utc",
            "execution_status",
            "reason",
            "baseline",
            "candidate",
            "shared_assets",
            "state_comparison_paths",
            "required_state_cycles",
            "curve_noise_floor",
            "expected_world_size",
            "expected_cycles",
        }
        missing = required - set(data)
        if not has_comparison_kind:
            missing.add("comparison_kind")
        if missing:
            raise ValueError(
                f"trajectory run-matrix record is missing fields: {sorted(missing)}"
            )
        result = cls(**data)
        if comparison_kind is not None and comparison_kind != result.comparison_kind:
            raise ValueError(
                f"comparison_kind for {result.workload_id} must be {result.comparison_kind!r}"
            )
        return result


def load_run_matrix_manifest(path: Path | str) -> list[TrajectoryRunMatrixRecord]:
    """Load an append-only run matrix; malformed/blank rows invalidate evidence."""

    source = Path(path)
    records: list[TrajectoryRunMatrixRecord] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank run-matrix JSONL row {source}:{line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed run-matrix JSONL {source}:{line_number}"
                ) from exc
            try:
                records.append(TrajectoryRunMatrixRecord.from_mapping(payload))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid run-matrix row {source}:{line_number}: {exc}"
                ) from exc
    return records


def _immutable_record_identity(record: TrajectoryRunMatrixRecord) -> dict[str, Any]:
    """Identity that must not drift between attempts under one suite id."""

    return {
        "workload_id": record.workload_id,
        "comparison_kind": record.comparison_kind,
        "expected_world_size": record.expected_world_size,
        "expected_cycles": record.expected_cycles,
        "shared_assets": dict(record.shared_assets),
        "baseline_source_head": record.baseline.source_head,
        "baseline_variant": record.baseline.variant,
        "candidate_source_head": record.candidate.source_head,
        "candidate_variant": record.candidate.variant,
        "baseline_assets": dict(record.baseline.assets),
        "candidate_assets": dict(record.candidate.assets),
        "baseline_implementation_expectations": dict(
            record.baseline.implementation_expectations
        ),
        "candidate_implementation_expectations": dict(
            record.candidate.implementation_expectations
        ),
    }


class AppendOnlyTrajectoryRunMatrixWriter:
    """Process-safe append-only writer for suite status/history records."""

    def __init__(self, path: Path | str, *, suite_id: str, fsync: bool = True) -> None:
        self.path = Path(path)
        self.suite_id = _nonempty(suite_id, name="suite_id")
        self.fsync = bool(fsync)

    def append(self, record: TrajectoryRunMatrixRecord | Mapping[str, Any]) -> None:
        value = (
            record
            if isinstance(record, TrajectoryRunMatrixRecord)
            else TrajectoryRunMatrixRecord.from_mapping(record)
        )
        if value.suite_id != self.suite_id:
            raise ValueError(
                f"record suite_id {value.suite_id!r} does not match writer {self.suite_id!r}"
            )
        serialized = (
            json.dumps(
                value.to_mapping(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o640)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            existing_bytes = self.path.read_bytes()
            if existing_bytes and not existing_bytes.endswith(b"\n"):
                raise ValueError(
                    "existing append-only run-matrix manifest lacks final newline"
                )
            existing = load_run_matrix_manifest(self.path) if existing_bytes else []
            if any(item.suite_id != self.suite_id for item in existing):
                raise ValueError(
                    "existing run-matrix manifest contains another suite_id"
                )
            same_workload = [
                item for item in existing if item.workload_id == value.workload_id
            ]
            expected_attempt = same_workload[-1].attempt + 1 if same_workload else 1
            if value.attempt != expected_attempt:
                raise ValueError(
                    f"attempt must advance by one for {value.workload_id}: "
                    f"expected={expected_attempt} observed={value.attempt}"
                )
            if same_workload and _immutable_record_identity(
                same_workload[0]
            ) != _immutable_record_identity(value):
                raise ValueError(
                    "source/assets/workload identity drifted within one suite_id; start a new suite"
                )
            written = os.write(descriptor, serialized)
            if written != len(serialized):
                raise OSError(
                    f"short run-matrix append: wrote {written}/{len(serialized)} bytes"
                )
            if self.fsync:
                os.fsync(descriptor)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _latest_records(
    records: Iterable[TrajectoryRunMatrixRecord | Mapping[str, Any]],
) -> tuple[str, dict[str, TrajectoryRunMatrixRecord], list[TrajectoryRunMatrixRecord]]:
    normalized = [
        item
        if isinstance(item, TrajectoryRunMatrixRecord)
        else TrajectoryRunMatrixRecord.from_mapping(item)
        for item in records
    ]
    if not normalized:
        raise ValueError("run-matrix manifest has no records")
    suite_ids = {item.suite_id for item in normalized}
    if len(suite_ids) != 1:
        raise ValueError(f"run-matrix manifest mixes suite ids: {sorted(suite_ids)}")
    latest: dict[str, TrajectoryRunMatrixRecord] = {}
    seen_attempts: set[tuple[str, int]] = set()
    identities: dict[str, dict[str, Any]] = {}
    for item in normalized:
        address = (item.workload_id, item.attempt)
        if address in seen_attempts:
            raise ValueError(
                f"duplicate workload/attempt in run-matrix manifest: {address}"
            )
        seen_attempts.add(address)
        identity = _immutable_record_identity(item)
        previous_identity = identities.setdefault(item.workload_id, identity)
        if previous_identity != identity:
            raise ValueError(
                f"source/assets identity drifted for {item.workload_id} within one suite_id"
            )
        previous = latest.get(item.workload_id)
        if previous is None or item.attempt > previous.attempt:
            latest[item.workload_id] = item
    for workload_id in {item.workload_id for item in normalized}:
        attempts = [
            item.attempt for item in normalized if item.workload_id == workload_id
        ]
        expected = list(range(1, max(attempts) + 1))
        if attempts != expected:
            raise ValueError(
                f"append-only attempt history for {workload_id} is not contiguous: "
                f"observed={attempts} expected={expected}"
            )
    missing = sorted(set(WORKLOAD_DEFINITIONS) - set(latest))
    extra = sorted(set(latest) - set(WORKLOAD_DEFINITIONS))
    if missing or extra:
        raise ValueError(
            f"run-matrix must record all five workloads: missing={missing} extra={extra}"
        )
    return next(iter(suite_ids)), latest, normalized


def _rank_jsonl_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"rank-JSONL root is not a directory: {root}")
    files = sorted(path for path in root.rglob("rank_*.jsonl") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no rank_*.jsonl files below {root}")
    return files


def load_rank_jsonl_root(
    root: Path | str,
    *,
    arm: TrajectorySuiteArm,
    expected_world_size: int = EXPECTED_WORLD_SIZE,
) -> list[TrajectoryCycleReceipt]:
    """Load one dedicated rank root and bind it to its manifest identity."""

    source = Path(root)
    by_rank: dict[int, tuple[Path, list[TrajectoryCycleReceipt]]] = {}
    for path in _rank_jsonl_files(source):
        rows = load_trajectory_jsonl(path)
        if not rows:
            raise ValueError(f"empty trajectory rank file: {path}")
        ranks = {row.rank for row in rows}
        if len(ranks) != 1:
            raise ValueError(f"trajectory file mixes ranks {sorted(ranks)}: {path}")
        rank = next(iter(ranks))
        if rank in by_rank:
            raise ValueError(
                f"multiple JSONL files claim rank {rank}: {by_rank[rank][0]} and {path}"
            )
        by_rank[rank] = (path, rows)
    if set(by_rank) != set(range(expected_world_size)):
        raise ValueError(
            "rank-JSONL root must contain exactly one file per rank: "
            f"observed={sorted(by_rank)} expected={list(range(expected_world_size))}"
        )
    result = [row for rank in range(expected_world_size) for row in by_rank[rank][1]]
    run_ids = {row.run_id for row in result}
    variants = {row.variant for row in result}
    if run_ids != {arm.run_id}:
        raise ValueError(
            f"arm run_id mismatch: manifest={arm.run_id!r} receipts={sorted(run_ids)}"
        )
    if variants != {arm.variant}:
        raise ValueError(
            f"arm variant mismatch: manifest={arm.variant!r} receipts={sorted(variants)}"
        )
    for row in result:
        for key, expected in arm.implementation_expectations.items():
            observed = row.implementation.get(key)
            if observed != expected:
                raise ValueError(
                    f"rank={row.rank} cycle={row.cycle} implementation.{key} "
                    f"expected={expected!r} observed={observed!r}"
                )
    return result


def load_state_comparisons(
    paths: Sequence[str | Path],
) -> list[TrajectoryStateComparison]:
    """Load state comparisons from an object, list, or ``state_comparisons`` wrapper."""

    result: list[TrajectoryStateComparison] = []
    for path_like in paths:
        path = Path(path_like)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and "state_comparisons" in payload:
            payload = payload["state_comparisons"]
        if isinstance(payload, Mapping):
            values = [payload]
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            values = list(payload)
        else:
            raise TypeError(
                f"state comparison JSON must contain an object or list: {path}"
            )
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                raise TypeError(f"state comparison {path}[{index}] is not an object")
            normalized = dict(value)
            normalized.pop("status", None)
            normalized.pop("thresholds", None)
            try:
                result.append(TrajectoryStateComparison(**normalized))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid state comparison {path}[{index}]: {exc}"
                ) from exc
    addresses: set[tuple[int, str, str]] = set()
    for row in result:
        address = (row.cycle, row.component, row.state_kind)
        if address in addresses:
            raise ValueError(f"duplicate state comparison address: {address}")
        addresses.add(address)
    return result


def _comparison_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    curves = [
        *result.get("curves", {}).values(),
        *result.get("gradient_norm_curves", {}).values(),
    ]
    state_rows = list(result.get("state_comparisons", []))

    def values(key: str, rows: Sequence[Mapping[str, Any]]) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    nrmse = values("normalized_rmse", curves)
    correlations = values("moving_average_correlation", curves)
    tails = values("tail_symmetric_relative_difference", curves)
    return {
        "comparison_status": result.get("status"),
        "complete": bool(result.get("complete")),
        "contract_status": result.get("contract", {}).get("status"),
        "contract_mismatch_count": len(
            result.get("contract", {}).get("mismatches", [])
        ),
        "curve_count": len(result.get("curves", {})),
        "gradient_norm_curve_count": len(result.get("gradient_norm_curves", {})),
        "curve_review_count": sum(row.get("status") != "PASS" for row in curves),
        "state_comparison_count": len(state_rows),
        "state_review_count": sum(row.get("status") != "PASS" for row in state_rows),
        "worst_normalized_rmse": max(nrmse, default=None),
        "minimum_moving_average_correlation": min(correlations, default=None),
        "worst_tail_symmetric_relative_difference": max(tails, default=None),
        "minimum_state_cosine": min(values("cosine", state_rows), default=None),
        "worst_state_normalized_l2": max(
            values("normalized_l2", state_rows), default=None
        ),
        "minimum_named_tensor_cosine": min(
            values("min_named_tensor_cosine", state_rows), default=None
        ),
    }


def build_trajectory_suite_report(
    records: Iterable[TrajectoryRunMatrixRecord | Mapping[str, Any]],
    *,
    thresholds: TrajectoryThresholds = COARSE_50C_THRESHOLDS,
) -> dict[str, Any]:
    """Build one five-workload post-run report from the latest manifest rows."""

    suite_id, latest, history = _latest_records(records)
    pairs: list[dict[str, Any]] = []
    for workload_id, definition in WORKLOAD_DEFINITIONS.items():
        record = latest[workload_id]
        pair: dict[str, Any] = {
            "workload_id": workload_id,
            "display_name": definition.display_name,
            "comparison_kind": definition.comparison_kind,
            "fixed_claim_boundary": definition.claim_boundary,
            "manifest_execution_status": record.execution_status,
            "attempt": record.attempt,
            "recorded_at_utc": record.recorded_at_utc,
            "reason": record.reason,
            "expected_world_size": record.expected_world_size,
            "expected_cycles": record.expected_cycles,
            "baseline": record.baseline.to_mapping(),
            "candidate": record.candidate.to_mapping(),
            "shared_assets": dict(record.shared_assets),
            "state_comparison_paths": list(record.state_comparison_paths),
            "required_state_cycles": list(record.required_state_cycles),
            "comparison": None,
            "evidence_error": None,
        }
        if record.execution_status != "PASS":
            pair["matrix_status"] = record.execution_status
            pair["summary"] = {
                "comparison_status": record.execution_status,
                "complete": False,
                "contract_status": "NOT_EVALUATED",
            }
            pairs.append(pair)
            continue
        try:
            assert record.baseline.rank_jsonl_root is not None
            assert record.candidate.rank_jsonl_root is not None
            baseline_rows = load_rank_jsonl_root(
                record.baseline.rank_jsonl_root,
                arm=record.baseline,
                expected_world_size=record.expected_world_size,
            )
            candidate_rows = load_rank_jsonl_root(
                record.candidate.rank_jsonl_root,
                arm=record.candidate,
                expected_world_size=record.expected_world_size,
            )
            states = load_state_comparisons(record.state_comparison_paths)
            comparison = compare_trajectories(
                baseline_rows,
                candidate_rows,
                state_comparisons=states,
                thresholds=thresholds,
                expected_cycles=record.expected_cycles,
                expected_world_size=record.expected_world_size,
                comparison_kind=definition.comparison_kind,
                curve_noise_floor=record.curve_noise_floor,
                required_state_cycles=record.required_state_cycles,
            )
            pair["comparison"] = comparison
            pair["summary"] = _comparison_summary(comparison)
            comparison_status = str(comparison.get("status"))
            if comparison_status in {"RUNTIME_FAIL", "CONTRACT_MISMATCH_50C_COMPLETED"}:
                pair["matrix_status"] = "FAIL"
            elif comparison_status in {
                "TRAJECTORY_REVIEW_50C_COMPLETED",
                "APPROXIMATION_REVIEW_50C_COMPLETED",
            }:
                pair["matrix_status"] = "REVIEW"
            else:
                pair["matrix_status"] = "PASS"
        except (AssertionError, KeyError, OSError, TypeError, ValueError) as exc:
            # One malformed pair must not erase the other four reports.
            pair["matrix_status"] = "FAIL"
            pair["evidence_error"] = f"{type(exc).__name__}: {exc}"
            pair["summary"] = {
                "comparison_status": "FAIL_EVIDENCE",
                "complete": False,
                "contract_status": "FAIL",
            }
        pairs.append(pair)

    statuses = {pair["matrix_status"] for pair in pairs}
    if "FAIL" in statuses:
        suite_status = "FAIL"
    elif "NOT_RUN" in statuses:
        suite_status = "NOT_RUN"
    elif "REVIEW" in statuses:
        suite_status = "REVIEW"
    else:
        suite_status = "PASS"
    history_payload = [item.to_mapping() for item in history]
    history_sha256 = hashlib.sha256(
        json.dumps(
            history_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return {
        "schema_version": SUITE_REPORT_SCHEMA_VERSION,
        "suite_id": suite_id,
        "status": suite_status,
        "evaluation_policy": "POST_RUN_ONLY_NO_NUMERICAL_EARLY_STOP",
        "expected_world_size": EXPECTED_WORLD_SIZE,
        "expected_cycles": EXPECTED_CYCLES,
        "workload_count": len(pairs),
        "manifest_history_record_count": len(history),
        "manifest_history_sha256": history_sha256,
        "thresholds": dataclasses.asdict(thresholds),
        "pairs": pairs,
        "claim_boundary": (
            "FOUR_SINGLE_SEED_COARSE_SEMANTIC_TRAJECTORIES_PLUS_ONE_H3_APPROXIMATION_STUDY"
        ),
    }


def _fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}g}"
    return str(value)


def _matrix_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in report["pairs"]:
        summary = pair["summary"]
        rows.append(
            {
                "workload_id": pair["workload_id"],
                "workload": pair["display_name"],
                "comparison_kind": pair["comparison_kind"],
                "manifest_execution_status": pair["manifest_execution_status"],
                "matrix_status": pair["matrix_status"],
                "comparison_status": summary.get("comparison_status"),
                "complete_16x50": bool(summary.get("complete")),
                "contract_status": summary.get("contract_status"),
                "curve_review_count": summary.get("curve_review_count"),
                "state_review_count": summary.get("state_review_count"),
                "worst_normalized_rmse": summary.get("worst_normalized_rmse"),
                "minimum_moving_average_correlation": summary.get(
                    "minimum_moving_average_correlation"
                ),
                "worst_tail_symmetric_relative_difference": summary.get(
                    "worst_tail_symmetric_relative_difference"
                ),
                "minimum_state_cosine": summary.get("minimum_state_cosine"),
                "worst_state_normalized_l2": summary.get("worst_state_normalized_l2"),
                "minimum_named_tensor_cosine": summary.get(
                    "minimum_named_tensor_cosine"
                ),
                "baseline_run_id": pair["baseline"]["run_id"],
                "baseline_head": pair["baseline"]["source_head"],
                "baseline_rank_jsonl_root": pair["baseline"]["rank_jsonl_root"],
                "candidate_run_id": pair["candidate"]["run_id"],
                "candidate_head": pair["candidate"]["source_head"],
                "candidate_rank_jsonl_root": pair["candidate"]["rank_jsonl_root"],
                "shared_assets_json": json.dumps(
                    pair["shared_assets"], sort_keys=True, separators=(",", ":")
                ),
                "baseline_assets_json": json.dumps(
                    pair["baseline"]["assets"], sort_keys=True, separators=(",", ":")
                ),
                "candidate_assets_json": json.dumps(
                    pair["candidate"]["assets"], sort_keys=True, separators=(",", ":")
                ),
                "claim_boundary": pair["fixed_claim_boundary"],
                "evidence_error": pair["evidence_error"],
            }
        )
    return rows


def _render_suite_csv(report: Mapping[str, Any]) -> str:
    rows = _matrix_rows(report)
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _render_suite_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Five-workload 50-cycle trajectory matrix",
        "",
        f"- Suite ID: `{report['suite_id']}`",
        f"- Status: `{report['status']}`",
        "- Required completeness: `16 ranks x 50 continuous cycles per arm`",
        f"- Evaluation policy: `{report['evaluation_policy']}`",
        f"- Claim boundary: `{report['claim_boundary']}`",
        "",
        "| Workload | Kind | Execution | Matrix | Complete | Contract | worst nRMSE | min corr | min state cosine |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in report["pairs"]:
        summary = pair["summary"]
        lines.append(
            f"| {pair['display_name']} | {pair['comparison_kind']} | "
            f"{pair['manifest_execution_status']} | {pair['matrix_status']} | "
            f"{summary.get('complete', False)} | {summary.get('contract_status')} | "
            f"{_fmt(summary.get('worst_normalized_rmse'))} | "
            f"{_fmt(summary.get('minimum_moving_average_correlation'))} | "
            f"{_fmt(summary.get('minimum_state_cosine'))} |"
        )
    lines.extend(["", "## Pair evidence", ""])
    for pair in report["pairs"]:
        lines.extend(
            [
                f"### {pair['display_name']}",
                "",
                f"- Workload ID: `{pair['workload_id']}`",
                f"- Comparison kind: `{pair['comparison_kind']}`",
                f"- Matrix status: `{pair['matrix_status']}`",
                f"- Baseline: `{pair['baseline']['run_id']}` at `{pair['baseline']['source_head']}`",
                f"- Candidate: `{pair['candidate']['run_id']}` at `{pair['candidate']['source_head']}`",
                f"- Claim boundary: `{pair['fixed_claim_boundary']}`",
            ]
        )
        if pair["comparison_kind"] == "approximation":
            lines.append(
                "- **Grid1000 is an approximation study and is not labelled exact-equivalent.**"
            )
        if pair["evidence_error"]:
            lines.append(f"- Evidence error: `{pair['evidence_error']}`")
        comparison = pair.get("comparison") or {}
        mismatches = comparison.get("contract", {}).get("mismatches", [])
        if mismatches:
            lines.append("- Contract mismatches:")
            lines.extend(f"  - {item}" for item in mismatches)
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "These are complete-run, one-seed coarse trajectory checks. They do not establish "
                "multi-seed convergence or final-generation quality, and numerical thresholds were "
                "not used to stop an otherwise healthy run. The H3 row changes the sigma distribution "
                "by snapping to Grid1000 and therefore remains an approximation comparison."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _primary_curve(pair: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    comparison = pair.get("comparison") or {}
    curves = comparison.get("curves", {})
    if not curves:
        return None
    preferred = [key for key in sorted(curves) if "total" in key.lower()]
    key = preferred[0] if preferred else min(curves)
    return key, curves[key]


def _render_suite_png(report: Mapping[str, Any], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to render the suite matrix PNG"
        ) from exc

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), squeeze=False)
    flat = [axis for row in axes for axis in row]
    status_colors = {
        "PASS": "#2e7d32",
        "REVIEW": "#ed6c02",
        "FAIL": "#c62828",
        "NOT_RUN": "#616161",
    }
    for axis, pair in zip(flat, report["pairs"]):
        axis.set_title(
            f"{pair['display_name']} [{pair['matrix_status']}]",
            color=status_colors.get(pair["matrix_status"], "black"),
        )
        primary = _primary_curve(pair)
        if primary is None:
            axis.axis("off")
            message = pair["evidence_error"] or pair["reason"]
            axis.text(
                0.5, 0.55, pair["matrix_status"], ha="center", va="center", fontsize=18
            )
            axis.text(
                0.5, 0.38, message, ha="center", va="center", fontsize=9, wrap=True
            )
            continue
        metric, curve = primary
        points = curve["points"]
        cycles = [item["cycle"] for item in points]
        axis.plot(
            cycles,
            [item["reference"] for item in points],
            label="baseline",
            linewidth=1.6,
        )
        axis.plot(
            cycles,
            [item["candidate"] for item in points],
            label="candidate",
            linewidth=1.6,
        )
        axis.set_xlabel("cycle")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        if pair["comparison_kind"] == "approximation":
            axis.text(
                0.02,
                0.03,
                "APPROXIMATION - not exact equivalence",
                transform=axis.transAxes,
                fontsize=8,
                color="#8a3b12",
                bbox={"facecolor": "#fff3e0", "edgecolor": "#ffb74d", "alpha": 0.9},
            )
    flat[-1].axis("off")
    flat[-1].text(
        0.5,
        0.62,
        f"Suite: {report['suite_id']}\nStatus: {report['status']}",
        ha="center",
        va="center",
        fontsize=16,
    )
    flat[-1].text(
        0.5,
        0.38,
        "16 ranks x 50 cycles per arm\npost-run only; one matched seed",
        ha="center",
        va="center",
        fontsize=11,
    )
    figure.suptitle("DMD 50-cycle trajectory suite", fontsize=17)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, format="png", dpi=160)
    plt.close(figure)


def write_trajectory_suite_report(
    report: Mapping[str, Any], output_dir: Path | str, *, overwrite: bool = False
) -> dict[str, Path]:
    """Atomically write one JSON/CSV/Markdown/PNG matrix and its checksums."""

    root = Path(output_dir)
    outputs = {
        "json": root / "trajectory_suite_matrix.json",
        "csv": root / "trajectory_suite_matrix.csv",
        "markdown": root / "trajectory_suite_matrix.md",
        "png": root / "trajectory_suite_matrix.png",
        "sha256": root / "artifact_sha256.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"suite report would overwrite existing artifacts: {existing}"
        )
    root.mkdir(parents=True, exist_ok=True)
    nonce = f"{os.getpid()}.{os.urandom(4).hex()}"
    temporary = {
        key: path.with_name(f".{path.name}.{nonce}.tmp")
        for key, path in outputs.items()
    }
    try:
        temporary["json"].write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary["csv"].write_text(_render_suite_csv(report), encoding="utf-8")
        temporary["markdown"].write_text(
            _render_suite_markdown(report), encoding="utf-8"
        )
        _render_suite_png(report, temporary["png"])
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
    "ALLOWED_EXECUTION_STATUSES",
    "ANCHOR_CYCLES",
    "EXPECTED_CYCLES",
    "EXPECTED_WORLD_SIZE",
    "MATRIX_RECORD_SCHEMA_VERSION",
    "SUITE_REPORT_SCHEMA_VERSION",
    "WORKLOAD_DEFINITIONS",
    "AppendOnlyTrajectoryRunMatrixWriter",
    "TrajectoryRunMatrixRecord",
    "TrajectorySuiteArm",
    "build_trajectory_suite_report",
    "load_rank_jsonl_root",
    "load_run_matrix_manifest",
    "load_state_comparisons",
    "write_trajectory_suite_report",
]
