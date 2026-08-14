"""Exact interval-union analysis for Nsight Systems SQLite exports.

Semantic NVTX ranges are host-side envelopes, not GPU utilization.  This
module keeps the two layers separate and reports, for every rank/scope/phase:

* non-NCCL CUDA compute-kernel interval union;
* NCCL kernel interval union;
* H2D, D2H, D2D and all-copy interval unions plus bytes/events;
* active union = union(compute, NCCL, all CUDA copies); and
* no-traced-GPU-activity = phase wall minus active union.

Percentages are never added across categories because their intervals may
overlap.  Aggregates use GPU-seconds: sum(rank activity union) divided by
sum(rank phase wall).
"""
from __future__ import annotations

import dataclasses
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


COMPUTE = "compute"
NCCL = "nccl"
H2D = "h2d"
D2H = "d2h"
D2D = "d2d"
OTHER_COPY = "other_copy"
ALL_COPY = "all_copy"
ACTIVE = "active"
NO_TRACED_ACTIVITY = "no_traced_gpu_activity"

_COPY_KIND_NUMERIC = {
    1: H2D,
    2: D2H,
    3: D2D,
    8: D2D,
    10: D2D,
}


@dataclasses.dataclass(frozen=True, order=True)
class Interval:
    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        if self.start_ns < 0 or self.end_ns <= self.start_ns:
            raise ValueError(f"invalid interval [{self.start_ns}, {self.end_ns})")

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    def clip(self, window: "Interval") -> "Interval | None":
        start = max(self.start_ns, window.start_ns)
        end = min(self.end_ns, window.end_ns)
        return Interval(start, end) if start < end else None


@dataclasses.dataclass(frozen=True)
class ActivityInterval:
    interval: Interval
    category: str
    name: str = ""
    bytes: int = 0

    def __post_init__(self) -> None:
        if self.category not in {COMPUTE, NCCL, H2D, D2H, D2D, OTHER_COPY}:
            raise ValueError(f"unknown activity category {self.category!r}")
        if self.bytes < 0:
            raise ValueError("activity bytes cannot be negative")


@dataclasses.dataclass(frozen=True)
class PhaseWindow:
    name: str
    interval: Interval
    coverage: str = "exact"
    source: str = "nvtx"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("phase name cannot be empty")
        if self.coverage not in {"exact", "inferred", "receipt_reconstructed"}:
            raise ValueError(f"unsupported phase coverage {self.coverage!r}")


def merge_intervals(intervals: Iterable[Interval]) -> tuple[Interval, ...]:
    ordered = sorted(intervals)
    if not ordered:
        return ()
    merged: list[Interval] = [ordered[0]]
    for value in ordered[1:]:
        previous = merged[-1]
        if value.start_ns <= previous.end_ns:
            merged[-1] = Interval(previous.start_ns, max(previous.end_ns, value.end_ns))
        else:
            merged.append(value)
    return tuple(merged)


def interval_union_ns(intervals: Iterable[Interval]) -> int:
    return sum(value.duration_ns for value in merge_intervals(intervals))


def clip_intervals_to_windows(
    intervals: Iterable[Interval], windows: Iterable[Interval]
) -> tuple[Interval, ...]:
    merged_windows = merge_intervals(windows)
    clipped: list[Interval] = []
    for value in intervals:
        for window in merged_windows:
            if window.end_ns <= value.start_ns:
                continue
            if window.start_ns >= value.end_ns:
                break
            part = value.clip(window)
            if part is not None:
                clipped.append(part)
    return merge_intervals(clipped)


def complement_intervals(intervals: Iterable[Interval], window: Interval) -> tuple[Interval, ...]:
    covered = clip_intervals_to_windows(intervals, (window,))
    cursor = window.start_ns
    result: list[Interval] = []
    for value in covered:
        if cursor < value.start_ns:
            result.append(Interval(cursor, value.start_ns))
        cursor = max(cursor, value.end_ns)
    if cursor < window.end_ns:
        result.append(Interval(cursor, window.end_ns))
    return tuple(result)


def _quote(identifier: str) -> str:
    if not identifier or any(character not in "_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for character in identifier):
        raise ValueError(f"unsafe SQLite identifier {identifier!r}")
    return f'"{identifier}"'


def sqlite_schema(connection: sqlite3.Connection) -> dict[str, tuple[str, ...]]:
    tables = [
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    ]
    return {
        table: tuple(str(row[1]) for row in connection.execute(f"PRAGMA table_info({_quote(table)})"))
        for table in tables
    }


def _first(mapping: Mapping[str, Sequence[str]], table_names: Sequence[str]) -> str | None:
    for name in table_names:
        if name in mapping:
            return name
    return None


def _first_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _load_string_ids(connection: sqlite3.Connection, schema: Mapping[str, Sequence[str]]) -> dict[int, str]:
    table = _first(schema, ("StringIds", "STRING_IDS", "StringTable"))
    if table is None:
        return {}
    columns = schema[table]
    id_column = _first_column(columns, ("id", "stringId", "key"))
    value_column = _first_column(columns, ("value", "string", "text"))
    if id_column is None or value_column is None:
        return {}
    query = f"SELECT {_quote(id_column)}, {_quote(value_column)} FROM {_quote(table)}"
    result: dict[int, str] = {}
    for raw_id, raw_value in connection.execute(query):
        if raw_id is not None and raw_value is not None:
            result[int(raw_id)] = str(raw_value)
    return result


def _resolve_string(value: Any, strings: Mapping[int, str]) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)) and int(value) in strings:
        return strings[int(value)]
    return str(value)


def _classify_kernel(name: str) -> str:
    lowered = name.lower()
    return NCCL if any(token in lowered for token in ("nccl", "nvls", "collnet")) else COMPUTE


def _classify_copy(value: Any, strings: Mapping[int, str]) -> str:
    if isinstance(value, (int, float)):
        return _COPY_KIND_NUMERIC.get(int(value), OTHER_COPY)
    text = _resolve_string(value, strings).lower().replace(" ", "").replace("_", "")
    if any(token in text for token in ("htod", "h2d", "hosttodevice")):
        return H2D
    if any(token in text for token in ("dtoh", "d2h", "devicetohost")):
        return D2H
    if any(token in text for token in ("dtod", "d2d", "devicetodevice", "peer")):
        return D2D
    return OTHER_COPY


def read_cuda_activities(connection: sqlite3.Connection) -> tuple[ActivityInterval, ...]:
    """Read kernels and memcpy activities from common Nsight SQLite schemas."""

    schema = sqlite_schema(connection)
    strings = _load_string_ids(connection, schema)
    result: list[ActivityInterval] = []

    kernel_table = _first(
        schema,
        (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
            "CUDA_KERNEL",
        ),
    )
    if kernel_table is not None:
        columns = schema[kernel_table]
        start = _first_column(columns, ("start", "startNs", "start_ns"))
        end = _first_column(columns, ("end", "endNs", "end_ns"))
        name = _first_column(columns, ("demangledName", "shortName", "name", "nameId", "textId"))
        if start is not None and end is not None:
            selected = [_quote(start), _quote(end)]
            if name is not None:
                selected.append(_quote(name))
            query = f"SELECT {', '.join(selected)} FROM {_quote(kernel_table)}"
            for row in connection.execute(query):
                start_ns, end_ns = int(row[0]), int(row[1])
                if end_ns <= start_ns:
                    continue
                kernel_name = _resolve_string(row[2], strings) if len(row) > 2 else ""
                result.append(
                    ActivityInterval(
                        Interval(start_ns, end_ns),
                        _classify_kernel(kernel_name),
                        name=kernel_name,
                    )
                )

    for copy_table in (
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMCPY2",
        "CUDA_MEMCPY",
    ):
        if copy_table not in schema:
            continue
        columns = schema[copy_table]
        start = _first_column(columns, ("start", "startNs", "start_ns"))
        end = _first_column(columns, ("end", "endNs", "end_ns"))
        kind = _first_column(columns, ("copyKind", "kind", "memcpyKind", "name"))
        byte_column = _first_column(columns, ("bytes", "size", "numBytes"))
        if start is None or end is None:
            continue
        selected = [_quote(start), _quote(end)]
        if kind is not None:
            selected.append(_quote(kind))
        if byte_column is not None:
            selected.append(_quote(byte_column))
        query = f"SELECT {', '.join(selected)} FROM {_quote(copy_table)}"
        for row in connection.execute(query):
            start_ns, end_ns = int(row[0]), int(row[1])
            if end_ns <= start_ns:
                continue
            cursor = 2
            raw_kind: Any = None
            if kind is not None:
                raw_kind = row[cursor]
                cursor += 1
            num_bytes = int(row[cursor] or 0) if byte_column is not None else 0
            result.append(
                ActivityInterval(
                    Interval(start_ns, end_ns),
                    _classify_copy(raw_kind, strings),
                    name=_resolve_string(raw_kind, strings),
                    bytes=num_bytes,
                )
            )
    return tuple(result)


def read_nvtx_phase_windows(
    connection: sqlite3.Connection,
    patterns: Mapping[str, str],
    *,
    coverage: str = "exact",
) -> tuple[PhaseWindow, ...]:
    schema = sqlite_schema(connection)
    strings = _load_string_ids(connection, schema)
    table = _first(schema, ("NVTX_EVENTS", "NVTX_EVENTS_EXPORT", "NVTX"))
    if table is None:
        return ()
    columns = schema[table]
    start = _first_column(columns, ("start", "startNs", "start_ns"))
    end = _first_column(columns, ("end", "endNs", "end_ns"))
    text = _first_column(columns, ("text", "textId", "name", "message"))
    if start is None or end is None or text is None:
        return ()
    compiled = {name: re.compile(pattern) for name, pattern in patterns.items()}
    query = f"SELECT {_quote(start)}, {_quote(end)}, {_quote(text)} FROM {_quote(table)}"
    result: list[PhaseWindow] = []
    for raw_start, raw_end, raw_text in connection.execute(query):
        # Nsight can emit a metadata-only NVTX row with NULL boundaries when
        # an outer process is traced together with forked torchrun children.
        # It is not an interval and must not invalidate otherwise complete
        # node-level activity exports.
        if raw_start is None or raw_end is None or raw_text is None:
            continue
        start_ns, end_ns = int(raw_start), int(raw_end)
        if end_ns <= start_ns:
            continue
        label = _resolve_string(raw_text, strings)
        for name, pattern in compiled.items():
            if pattern.search(label):
                result.append(PhaseWindow(name, Interval(start_ns, end_ns), coverage, f"nvtx:{label}"))
                break
    return tuple(result)


def infer_capture_window(
    activities: Sequence[ActivityInterval], phases: Sequence[PhaseWindow]
) -> Interval:
    starts = [value.interval.start_ns for value in activities]
    ends = [value.interval.end_ns for value in activities]
    starts.extend(value.interval.start_ns for value in phases)
    ends.extend(value.interval.end_ns for value in phases)
    if not starts or not ends:
        raise ValueError("cannot infer capture window from an empty SQLite export")
    return Interval(min(starts), max(ends))


def _measure_category(
    activities: Sequence[ActivityInterval],
    category: str,
    windows: Sequence[Interval],
) -> dict[str, int]:
    selected = [value for value in activities if value.category == category]
    clipped = clip_intervals_to_windows((value.interval for value in selected), windows)
    num_bytes = sum(
        value.bytes
        for value in selected
        if any(value.interval.clip(window) is not None for window in windows)
    )
    events = sum(
        1
        for value in selected
        if any(value.interval.clip(window) is not None for window in windows)
    )
    return {"union_ns": interval_union_ns(clipped), "events": events, "bytes": num_bytes}


def analyze_windows(
    activities: Sequence[ActivityInterval],
    windows: Sequence[Interval],
) -> dict[str, Any]:
    windows = merge_intervals(windows)
    wall_ns = interval_union_ns(windows)
    if wall_ns <= 0:
        raise ValueError("analysis window has zero duration")
    categories = {
        category: _measure_category(activities, category, windows)
        for category in (COMPUTE, NCCL, H2D, D2H, D2D, OTHER_COPY)
    }
    all_copy_intervals = clip_intervals_to_windows(
        (
            value.interval
            for value in activities
            if value.category in {H2D, D2H, D2D, OTHER_COPY}
        ),
        windows,
    )
    active_intervals = clip_intervals_to_windows(
        (value.interval for value in activities), windows
    )
    categories[ALL_COPY] = {
        "union_ns": interval_union_ns(all_copy_intervals),
        "events": sum(categories[name]["events"] for name in (H2D, D2H, D2D, OTHER_COPY)),
        "bytes": sum(categories[name]["bytes"] for name in (H2D, D2H, D2D, OTHER_COPY)),
    }
    active_ns = interval_union_ns(active_intervals)
    categories[ACTIVE] = {"union_ns": active_ns, "events": 0, "bytes": 0}
    categories[NO_TRACED_ACTIVITY] = {
        "union_ns": max(0, wall_ns - active_ns),
        "events": 0,
        "bytes": 0,
    }
    for value in categories.values():
        value["union_seconds"] = value["union_ns"] / 1e9
        value["percent_of_wall"] = 100.0 * value["union_ns"] / wall_ns
    return {
        "wall_ns": wall_ns,
        "wall_seconds": wall_ns / 1e9,
        "categories": categories,
        "non_additivity_warning": "compute, NCCL and copy interval unions may overlap",
    }


def _binned_fraction(
    intervals: Iterable[Interval],
    window: Interval,
    *,
    bin_ns: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return exact union occupancy for fixed display bins.

    The interval arithmetic remains exact; only the visualization is binned.
    A fraction of 0.4 means the category occupied exactly 40% of that bin.
    """

    if bin_ns <= 0:
        raise ValueError("timeline bin_ns must be positive")
    merged = clip_intervals_to_windows(intervals, (window,))
    starts: list[int] = []
    fractions: list[float] = []
    cursor = window.start_ns
    index = 0
    while cursor < window.end_ns:
        stop = min(window.end_ns, cursor + bin_ns)
        display_bin = Interval(cursor, stop)
        while index < len(merged) and merged[index].end_ns <= cursor:
            index += 1
        overlaps: list[Interval] = []
        scan = index
        while scan < len(merged) and merged[scan].start_ns < stop:
            part = merged[scan].clip(display_bin)
            if part is not None:
                overlaps.append(part)
            scan += 1
        starts.append(cursor)
        fractions.append(interval_union_ns(overlaps) / display_bin.duration_ns)
        cursor = stop
    return tuple(starts), tuple(fractions)


def build_activity_timeline(
    activities: Sequence[ActivityInterval],
    phases: Sequence[PhaseWindow],
    *,
    capture_window: Interval,
    bin_ns: int = 500_000_000,
) -> dict[str, Any]:
    """Build aligned semantic and exact-within-bin CUDA activity strips."""

    category_intervals: dict[str, tuple[Interval, ...]] = {}
    for category in (COMPUTE, NCCL, H2D, D2H, D2D, OTHER_COPY):
        category_intervals[category] = merge_intervals(
            value.interval for value in activities if value.category == category
        )
    category_intervals[ALL_COPY] = merge_intervals(
        value.interval
        for value in activities
        if value.category in {H2D, D2H, D2D, OTHER_COPY}
    )
    category_intervals[ACTIVE] = merge_intervals(value.interval for value in activities)

    starts, active_fraction = _binned_fraction(
        category_intervals[ACTIVE], capture_window, bin_ns=bin_ns
    )
    activity_fraction: dict[str, list[float]] = {}
    for category in (COMPUTE, NCCL, H2D, D2H, D2D, ALL_COPY, ACTIVE):
        category_starts, fractions = _binned_fraction(
            category_intervals[category], capture_window, bin_ns=bin_ns
        )
        if category_starts != starts:
            raise AssertionError("activity display bins drifted across categories")
        activity_fraction[category] = list(fractions)
    activity_fraction[NO_TRACED_ACTIVITY] = [
        max(0.0, min(1.0, 1.0 - value)) for value in active_fraction
    ]

    clipped_phases: list[dict[str, Any]] = []
    for phase in phases:
        clipped = phase.interval.clip(capture_window)
        if clipped is None:
            continue
        clipped_phases.append(
            {
                "name": phase.name,
                "start_seconds": (clipped.start_ns - capture_window.start_ns) / 1e9,
                "duration_seconds": clipped.duration_ns / 1e9,
                "coverage": phase.coverage,
                "source": phase.source,
            }
        )
    return {
        "bin_ns": int(bin_ns),
        "bin_start_seconds": [
            (start - capture_window.start_ns) / 1e9 for start in starts
        ],
        "bin_duration_seconds": [
            min(capture_window.end_ns, start + bin_ns) / 1e9 - start / 1e9
            for start in starts
        ],
        "activity_fraction": activity_fraction,
        "semantic_windows": clipped_phases,
        "display_semantics": {
            "activity_fraction_is_exact_interval_union_inside_each_bin": True,
            "semantic_window_is_host_side_envelope": True,
            "no_traced_gpu_activity_is_not_physical_idle": True,
        },
    }


def analyze_sqlite(
    path: str | Path,
    *,
    rank: int,
    phase_patterns: Mapping[str, str] | None = None,
    capture_window: Interval | None = None,
    capture_coverage: str = "inferred",
    timeline_bin_ns: int = 500_000_000,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with sqlite3.connect(str(path)) as connection:
        activities = read_cuda_activities(connection)
        phases = read_nvtx_phase_windows(
            connection,
            phase_patterns or {},
            coverage="exact",
        )
    if capture_window is None:
        capture_window = infer_capture_window(activities, phases)
        denominator_source = "activity_or_nvtx_extents_fallback"
    else:
        denominator_source = "provided_closed_profiler_range"
    report: dict[str, Any] = {
        "schema_version": "distillschedule.nsight_activity.v2",
        "rank": int(rank),
        "sqlite": str(path),
        "capture": {
            "start_ns": capture_window.start_ns,
            "end_ns": capture_window.end_ns,
            "wall_seconds": capture_window.duration_ns / 1e9,
            "coverage": capture_coverage,
            "denominator_source": denominator_source,
        },
        "activity_count": len(activities),
        "capture_metrics": analyze_windows(activities, (capture_window,)),
        "phases": {},
        "claim_boundary": {
            "diagnostic_only": True,
            "semantic_envelope_is_not_compute_utilization": True,
            "no_traced_gpu_activity_is_not_proof_of_cpu_or_scheduler_idle": True,
        },
    }
    grouped: dict[str, list[PhaseWindow]] = defaultdict(list)
    for phase in phases:
        clipped = phase.interval.clip(capture_window)
        if clipped is not None:
            grouped[phase.name].append(dataclasses.replace(phase, interval=clipped))
    for name, values in sorted(grouped.items()):
        metric = analyze_windows(activities, [value.interval for value in values])
        metric["coverage"] = sorted({value.coverage for value in values})
        metric["sources"] = [value.source for value in values]
        metric["windows"] = [dataclasses.asdict(value.interval) for value in values]
        report["phases"][name] = metric
    report["timeline"] = build_activity_timeline(
        activities,
        phases,
        capture_window=capture_window,
        bin_ns=timeline_bin_ns,
    )
    return report


def aggregate_rank_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one rank report is required")
    phase_names = sorted(
        set.intersection(
            *(set(report.get("phases", {})) for report in reports)
        )
    ) if all(report.get("phases") for report in reports) else []

    def aggregate_metric(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        wall_ns = sum(int(metric["wall_ns"]) for metric in metrics)
        categories: dict[str, Any] = {}
        for category in (COMPUTE, NCCL, H2D, D2H, D2D, OTHER_COPY, ALL_COPY, ACTIVE, NO_TRACED_ACTIVITY):
            union_ns = sum(int(metric["categories"][category]["union_ns"]) for metric in metrics)
            rank_percentages = [float(metric["categories"][category]["percent_of_wall"]) for metric in metrics]
            categories[category] = {
                "gpu_seconds": union_ns / 1e9,
                "pooled_percent": 100.0 * union_ns / wall_ns if wall_ns else 0.0,
                "rank_mean_percent": sum(rank_percentages) / len(rank_percentages),
                "events": sum(int(metric["categories"][category]["events"]) for metric in metrics),
                "bytes": sum(int(metric["categories"][category]["bytes"]) for metric in metrics),
            }
        return {
            "rank_count": len(metrics),
            "gpu_wall_seconds": wall_ns / 1e9,
            "categories": categories,
            "denominator": "sum(rank activity union) / sum(rank phase wall)",
        }

    result = {
        "schema_version": "distillschedule.nsight_activity_aggregate.v2",
        "rank_count": len(reports),
        "ranks": sorted(int(report["rank"]) for report in reports),
        "capture": aggregate_metric([report["capture_metrics"] for report in reports]),
        "phases": {},
    }
    for phase in phase_names:
        result["phases"][phase] = aggregate_metric(
            [report["phases"][phase] for report in reports]
        )
    return result


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


__all__ = [
    "ACTIVE",
    "ALL_COPY",
    "ActivityInterval",
    "COMPUTE",
    "D2D",
    "D2H",
    "H2D",
    "Interval",
    "NCCL",
    "NO_TRACED_ACTIVITY",
    "OTHER_COPY",
    "PhaseWindow",
    "aggregate_rank_reports",
    "build_activity_timeline",
    "analyze_sqlite",
    "analyze_windows",
    "clip_intervals_to_windows",
    "complement_intervals",
    "infer_capture_window",
    "interval_union_ns",
    "merge_intervals",
    "read_cuda_activities",
    "read_nvtx_phase_windows",
    "sqlite_schema",
    "write_json",
]
