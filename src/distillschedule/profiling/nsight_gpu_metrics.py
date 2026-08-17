"""Read Nsight Systems periodic GPU Metrics from an exported SQLite trace.

Nsight Systems GPU Metrics are device-level periodic samples, not process-level
samples. In DistillSchedule world16 profiles each CUDA device is exclusively
owned by one training rank, so a rank-qualified NVTX window may be aligned with
that device's samples. The helpers below keep that assumption explicit and do
not reinterpret CUDA-kernel temporal coverage as hardware utilization.
"""
from __future__ import annotations

import math
import re
import sqlite3
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


# Canonical names used by DistillSchedule reports. Nsight metric display names
# have changed slightly across releases/metric sets, so matching is deliberately
# tolerant while the chosen source name is always recorded in the output.
_METRIC_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "sm_active_percent",
        (
            re.compile(r"^SMs? Active$", re.I),
        ),
    ),
    (
        "sm_issue_percent",
        (
            re.compile(r"^SM Issue$", re.I),
        ),
    ),
    (
        "tensor_active_percent",
        (
            re.compile(r"^Tensor Active(?:\s*/\s*FP16 Active)?$", re.I),
        ),
    ),
    (
        "vram_bandwidth_percent",
        (
            re.compile(r"^(?:VRAM|DRAM) Bandwidth$", re.I),
        ),
    ),
    (
        "dram_read_percent",
        (
            re.compile(r"^(?:DRAM|VRAM) Read (?:Throughput|Bandwidth)$", re.I),
        ),
    ),
    (
        "dram_write_percent",
        (
            re.compile(r"^(?:DRAM|VRAM) Write (?:Throughput|Bandwidth)$", re.I),
        ),
    ),
)

CORE_METRICS = ("sm_active_percent", "sm_issue_percent", "tensor_active_percent")
BANDWIDTH_METRICS = (
    "vram_bandwidth_percent",
    "dram_read_percent",
    "dram_write_percent",
)


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "select 1 from sqlite_master where type='table' and name=? limit 1", (name,)
    ).fetchone()
    return row is not None


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError(q)
    xs = sorted(float(value) for value in values)
    if len(xs) == 1:
        return xs[0]
    position = q * (len(xs) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return xs[lo]
    weight = position - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def _canonical_source_names(metric_names: Iterable[str]) -> dict[str, str]:
    """Choose one concrete Nsight display name for each canonical metric."""

    names = sorted({str(name) for name in metric_names})
    selected: dict[str, str] = {}
    for canonical, patterns in _METRIC_PATTERNS:
        for pattern in patterns:
            matches = [name for name in names if pattern.search(name)]
            if matches:
                # Prefer the shortest display name when a metric set exposes
                # multiple aliases for the same hardware counter.
                selected[canonical] = sorted(matches, key=lambda value: (len(value), value))[0]
                break
    return selected


def load_gpu_metric_series(connection: sqlite3.Connection) -> dict[str, Any]:
    """Load selected GPU metric series from an Nsight Systems SQLite export.

    `typeId & 0xff` is the GPU id encoding documented by Nsight Systems. Values
    are integer percentages in the standard GPU metric sets.
    """

    required_tables = ("GPU_METRICS", "TARGET_INFO_GPU_METRICS")
    missing_tables = [name for name in required_tables if not _table_exists(connection, name)]
    if missing_tables:
        return {
            "status": "NOT_CAPTURED",
            "reason": "missing_gpu_metrics_tables",
            "missing_tables": missing_tables,
            "available_metric_names": [],
            "selected_source_names": {},
            "series": {},
        }

    catalog = [
        str(row[0])
        for row in connection.execute(
            "select distinct metricName from TARGET_INFO_GPU_METRICS order by metricName"
        )
    ]
    selected = _canonical_source_names(catalog)
    if not selected:
        return {
            "status": "NOT_CAPTURED",
            "reason": "gpu_metrics_tables_have_no_supported_utilization_metrics",
            "missing_tables": [],
            "available_metric_names": catalog,
            "selected_source_names": {},
            "series": {},
        }

    wanted = set(selected.values())
    placeholders = ",".join("?" for _ in wanted)
    query = f"""
        select g.timestamp, g.typeId, i.metricName, g.value
        from GPU_METRICS g
        join TARGET_INFO_GPU_METRICS i
          on i.typeId=g.typeId and i.metricId=g.metricId
        where i.metricName in ({placeholders})
        order by g.typeId, i.metricName, g.timestamp
    """
    reverse = {source: canonical for canonical, source in selected.items()}
    grouped: dict[tuple[int, str], list[tuple[int, float]]] = defaultdict(list)
    for timestamp, type_id, metric_name, value in connection.execute(query, tuple(sorted(wanted))):
        gpu_id = int(type_id) & 0xFF
        canonical = reverse[str(metric_name)]
        grouped[(gpu_id, canonical)].append((int(timestamp), float(value)))

    series = {
        f"gpu{gpu_id}/{canonical}": {
            "gpu_id": gpu_id,
            "canonical_name": canonical,
            "source_metric_name": selected[canonical],
            "samples": samples,
        }
        for (gpu_id, canonical), samples in sorted(grouped.items())
    }
    missing_core = [name for name in CORE_METRICS if name not in selected]
    status = "PASS" if not missing_core else "PARTIAL"
    return {
        "status": status,
        "reason": None if status == "PASS" else "one_or_more_core_metrics_missing",
        "missing_tables": [],
        "missing_core_metrics": missing_core,
        "available_metric_names": catalog,
        "selected_source_names": selected,
        "series": series,
    }


def _median_period_ns(samples: Sequence[tuple[int, float]]) -> int | None:
    deltas = [
        int(right[0]) - int(left[0])
        for left, right in zip(samples, samples[1:])
        if int(right[0]) > int(left[0])
    ]
    return int(statistics.median(deltas)) if deltas else None


def summarize_series_window(
    samples: Sequence[tuple[int, float]], *, start_ns: int, end_ns: int
) -> dict[str, Any]:
    """Time-weight one periodic metric series inside an NVTX window.

    A sample is treated as piecewise constant until the next sample. The final
    sample is extended by one median sampling period only; this prevents a
    truncated metrics capture from being mistaken for complete phase coverage.
    """

    start_ns = int(start_ns)
    end_ns = int(end_ns)
    if end_ns <= start_ns:
        raise ValueError("GPU metric window must have positive duration")
    ordered = sorted((int(ts), float(value)) for ts, value in samples)
    if not ordered:
        return {
            "status": "NO_SAMPLES",
            "sample_count": 0,
            "covered_ns": 0,
            "coverage_fraction": 0.0,
            "time_weighted_mean_pct": None,
        }

    median_period = _median_period_ns(ordered)
    weighted_sum = 0.0
    covered = 0
    values_in_or_covering_window: list[float] = []
    contributing_samples = 0

    for index, (timestamp, value) in enumerate(ordered):
        if index + 1 < len(ordered):
            next_timestamp = ordered[index + 1][0]
        elif median_period is not None:
            next_timestamp = timestamp + median_period
        else:
            next_timestamp = timestamp + 1
        lo = max(start_ns, timestamp)
        hi = min(end_ns, next_timestamp)
        if hi <= lo:
            continue
        duration = hi - lo
        weighted_sum += value * duration
        covered += duration
        contributing_samples += 1
        values_in_or_covering_window.append(value)

    wall = end_ns - start_ns
    if covered == 0:
        return {
            "status": "NO_SAMPLES_IN_WINDOW",
            "sample_count": 0,
            "covered_ns": 0,
            "coverage_fraction": 0.0,
            "time_weighted_mean_pct": None,
        }

    coverage = covered / wall
    status = "PASS" if coverage >= 0.95 else "PARTIAL_COVERAGE"
    return {
        "status": status,
        "sample_count": contributing_samples,
        "covered_ns": covered,
        "coverage_fraction": coverage,
        "sampling_period_median_ns": median_period,
        "time_weighted_mean_pct": weighted_sum / covered,
        "sample_median_pct": statistics.median(values_in_or_covering_window),
        "sample_p10_pct": _percentile(values_in_or_covering_window, 0.10),
        "sample_p90_pct": _percentile(values_in_or_covering_window, 0.90),
        "sample_min_pct": min(values_in_or_covering_window),
        "sample_max_pct": max(values_in_or_covering_window),
    }


def summarize_device_window(
    loaded: Mapping[str, Any], *, gpu_id: int, start_ns: int, end_ns: int
) -> dict[str, Any]:
    """Summarize all selected metrics for one device-qualified NVTX window."""

    if loaded.get("status") == "NOT_CAPTURED":
        return {
            "status": "NOT_CAPTURED",
            "reason": loaded.get("reason"),
            "metrics": {},
        }

    metrics: dict[str, Any] = {}
    for canonical, source_name in loaded.get("selected_source_names", {}).items():
        row = loaded.get("series", {}).get(f"gpu{int(gpu_id)}/{canonical}")
        if row is None:
            continue
        summary = summarize_series_window(
            row["samples"], start_ns=int(start_ns), end_ns=int(end_ns)
        )
        metrics[canonical] = {"source_metric_name": source_name, **summary}

    missing_core = [name for name in CORE_METRICS if name not in metrics]
    if not metrics:
        status = "NO_DEVICE_SAMPLES"
    elif missing_core:
        status = "PARTIAL"
    elif any(value["status"] != "PASS" for value in metrics.values()):
        status = "PARTIAL_COVERAGE"
    else:
        status = "PASS"
    return {
        "status": status,
        "gpu_id": int(gpu_id),
        "device_level_not_process_attributed": True,
        "missing_core_metrics": missing_core,
        "metrics": metrics,
    }


def aggregate_window_metrics(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-window hardware metrics, weighting by sampled time."""

    totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"weighted": 0.0, "covered_ns": 0, "window_means": [], "windows": 0}
    )
    source_names: dict[str, str] = {}
    status_counts: dict[str, int] = defaultdict(int)
    for window in windows:
        gpu_metrics = window.get("gpu_metrics") or {}
        status_counts[str(gpu_metrics.get("status", "MISSING"))] += 1
        for canonical, metric in (gpu_metrics.get("metrics") or {}).items():
            mean = metric.get("time_weighted_mean_pct")
            covered_ns = int(metric.get("covered_ns") or 0)
            if mean is None or covered_ns <= 0:
                continue
            totals[canonical]["weighted"] += float(mean) * covered_ns
            totals[canonical]["covered_ns"] += covered_ns
            totals[canonical]["window_means"].append(float(mean))
            totals[canonical]["windows"] += 1
            source_names[canonical] = str(metric.get("source_metric_name", canonical))

    metrics: dict[str, Any] = {}
    for canonical, value in sorted(totals.items()):
        covered = int(value["covered_ns"])
        means = list(value["window_means"])
        metrics[canonical] = {
            "source_metric_name": source_names.get(canonical),
            "window_count": int(value["windows"]),
            "covered_gpu_seconds": covered / 1e9,
            "time_weighted_mean_pct": value["weighted"] / covered,
            "window_mean_stats": {
                "min": min(means),
                "median": statistics.median(means),
                "mean": statistics.mean(means),
                "max": max(means),
            },
        }

    missing_core = [name for name in CORE_METRICS if name not in metrics]
    if not metrics:
        status = "NOT_CAPTURED"
    elif missing_core:
        status = "PARTIAL"
    else:
        status = "PASS"
    return {
        "status": status,
        "device_level_not_process_attributed": True,
        "missing_core_metrics": missing_core,
        "window_status_counts": dict(sorted(status_counts.items())),
        "metrics": metrics,
    }


__all__ = [
    "BANDWIDTH_METRICS",
    "CORE_METRICS",
    "aggregate_window_metrics",
    "load_gpu_metric_series",
    "summarize_device_window",
    "summarize_series_window",
]
