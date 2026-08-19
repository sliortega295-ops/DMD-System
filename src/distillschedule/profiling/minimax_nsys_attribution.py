"""Nsight Systems attribution for the matched MiniMax-H3 DMD workload.

The analyzer answers a deliberately narrow systems question: inside the exact
MiniMax cycle, how much NCCL and host/device copy activity is visible, how much
is overlapped by non-NCCL GPU compute, and how much remains *exposed* on the GPU
timeline.

"Exposed" is an interval property, not a causal speedup claim.  For example,
``NCCL - COMPUTE`` is time where an NCCL kernel is active and no non-NCCL CUDA
kernel is active on that rank.  Removing NCCL would not necessarily recover all
of that wall time because CPU dependencies, synchronization, or another rank
may still be critical.

Activation offload is identified conservatively as large H2D/D2H copies.  The
threshold is configurable and the report always preserves the raw H2D/D2H
metrics and a copy-size histogram so the heuristic can be audited.
"""
from __future__ import annotations

import dataclasses
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from distillschedule.profiling.nsight_activity import Interval, interval_union_ns, merge_intervals


COMPUTE = "compute"
NCCL = "nccl"
NCCL_ALL_GATHER = "nccl_all_gather"
NCCL_REDUCE_SCATTER = "nccl_reduce_scatter"
NCCL_ALL_REDUCE = "nccl_all_reduce"
NCCL_OTHER = "nccl_other"
H2D = "h2d"
D2H = "d2h"
D2D = "d2d"
OTHER_COPY = "other_copy"
ALL_COPY = "all_copy"
OFFLOAD_H2D = "offload_h2d_candidate"
OFFLOAD_D2H = "offload_d2h_candidate"
OFFLOAD = "offload_candidate"
ACTIVE = "active"
NO_TRACED_ACTIVITY = "no_traced_gpu_activity"

_COPY_KIND = {
    1: H2D,
    2: D2H,
    8: D2D,
    10: D2D,
    11: H2D,
    12: D2H,
    13: D2D,
}

_STAGE_RE = re.compile(
    r"^MINIMAX_NSYS_STAGE/(?P<mode>[^/]+)/rank=(?P<rank>\d+)/"
    r"(?P<stage>student|fake)/(?P<index>\d+)$"
)
_DIT_RE = re.compile(
    r"^MINIMAX_NSYS_DIT/(?P<mode>[^/]+)/rank=(?P<rank>\d+)/"
    r"(?P<role>student|fake|teacher|unknown)/(?P<grad>grad|nograd)/(?P<index>\d+)$"
)


@dataclasses.dataclass(frozen=True)
class StageWindow:
    rank: int
    pid: int
    stage: str
    index: int
    interval: Interval

    @property
    def key(self) -> str:
        return f"{self.stage}/{self.index}"


@dataclasses.dataclass
class _Series:
    intervals: list[Interval] = dataclasses.field(default_factory=list)
    events: int = 0
    bytes: int = 0
    sum_duration_ns: int = 0
    out_of_order_rows: int = 0

    def add(self, interval: Interval, *, num_bytes: int = 0) -> None:
        self.events += 1
        self.bytes += int(num_bytes)
        self.sum_duration_ns += interval.duration_ns
        if not self.intervals:
            self.intervals.append(interval)
            return
        previous = self.intervals[-1]
        if interval.start_ns < previous.start_ns:
            self.out_of_order_rows += 1
            self.intervals.append(interval)
            return
        if interval.start_ns <= previous.end_ns:
            self.intervals[-1] = Interval(
                previous.start_ns,
                max(previous.end_ns, interval.end_ns),
            )
        else:
            self.intervals.append(interval)

    def merged(self) -> tuple[Interval, ...]:
        if self.out_of_order_rows:
            return merge_intervals(self.intervals)
        return tuple(self.intervals)


@dataclasses.dataclass
class _ScopeAccumulator:
    window: Interval
    series: dict[str, _Series] = dataclasses.field(
        default_factory=lambda: defaultdict(_Series)
    )
    copy_sizes: Counter[tuple[str, int]] = dataclasses.field(default_factory=Counter)

    def add(
        self,
        category: str,
        interval: Interval,
        *,
        num_bytes: int = 0,
        copy_size_key: tuple[str, int] | None = None,
    ) -> None:
        clipped = interval.clip(self.window)
        if clipped is None:
            return
        self.series[category].add(clipped, num_bytes=num_bytes)
        if copy_size_key is not None:
            self.copy_sizes[copy_size_key] += 1


@dataclasses.dataclass
class _RankAccumulator:
    rank: int
    pid: int
    cycle: _ScopeAccumulator
    stages: dict[str, _ScopeAccumulator]
    stage_windows: dict[str, StageWindow]
    offload_correlation_ids: set[int] = dataclasses.field(default_factory=set)
    offload_runtime_api: dict[str, _Series] = dataclasses.field(
        default_factory=lambda: defaultdict(_Series)
    )


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _strings(connection: sqlite3.Connection) -> dict[int, str]:
    if not _table_exists(connection, "StringIds"):
        return {}
    columns = _columns(connection, "StringIds")
    value_column = "value" if "value" in columns else ("string" if "string" in columns else None)
    if value_column is None:
        return {}
    return {
        int(row[0]): str(row[1])
        for row in connection.execute(f'SELECT id,"{value_column}" FROM StringIds')
        if row[0] is not None and row[1] is not None
    }


def _known_pids(connection: sqlite3.Connection) -> set[int]:
    if not _table_exists(connection, "PROCESSES"):
        return set()
    columns = _columns(connection, "PROCESSES")
    if "pid" not in columns:
        return set()
    return {int(row[0]) for row in connection.execute("SELECT pid FROM PROCESSES") if row[0] is not None}


def decode_pid(value: Any, known_pids: set[int] | None = None) -> int:
    """Decode Nsight globalPid/globalTid while tolerating plain-PID exports."""
    raw = int(value)
    known = known_pids or set()
    if raw in known:
        return raw
    encoded = (raw // 0x1000000) % 0x1000000
    if encoded in known:
        return encoded
    return encoded if encoded else raw


def _resolve_string(value: Any, strings: Mapping[int, str]) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return strings.get(int(value), str(value))
    return str(value)


def _nvtx_rows(
    connection: sqlite3.Connection,
    strings: Mapping[int, str],
    known_pids: set[int],
) -> list[tuple[int, int, int, str]]:
    if not _table_exists(connection, "NVTX_EVENTS"):
        raise ValueError("Nsight SQLite has no NVTX_EVENTS table")
    columns = _columns(connection, "NVTX_EVENTS")
    required = {"start", "end", "globalTid"}
    if not required.issubset(columns):
        raise ValueError(f"NVTX_EVENTS is missing required columns: {required - columns}")
    text_column = "text" if "text" in columns else None
    text_id_column = "textId" if "textId" in columns else None
    if text_column is None and text_id_column is None:
        raise ValueError("NVTX_EVENTS has neither text nor textId")
    selected = ["start", "end", "globalTid"]
    if text_column:
        selected.append("text")
    if text_id_column:
        selected.append("textId")
    query = f"SELECT {','.join(selected)} FROM NVTX_EVENTS WHERE end IS NOT NULL AND end>start"
    result: list[tuple[int, int, int, str]] = []
    for row in connection.execute(query):
        start, end, global_tid = row[:3]
        if start is None or end is None or global_tid is None:
            continue
        cursor = 3
        text = ""
        if text_column:
            text = _resolve_string(row[cursor], strings)
            cursor += 1
        if not text and text_id_column:
            text = _resolve_string(row[cursor], strings)
        if not text.startswith("MINIMAX_"):
            continue
        result.append((int(start), int(end), decode_pid(global_tid, known_pids), text))
    return result


def _discover_windows(
    rows: Sequence[tuple[int, int, int, str]],
    *,
    mode: str,
) -> tuple[dict[int, tuple[int, Interval]], dict[int, list[StageWindow]], dict[int, dict[str, Any]]]:
    pid_to_rank: dict[int, int] = {}
    stages_by_rank: dict[int, list[StageWindow]] = defaultdict(list)
    dit_counts: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "roles": Counter(),
            "grad_roles": Counter(),
            "nograd_roles": Counter(),
        }
    )
    cycle_rows_by_pid: dict[int, list[Interval]] = defaultdict(list)
    cycle_label = f"MINIMAX_MATCHED_CYCLE/{mode}"

    pending_stages: list[tuple[int, int, int, str, int]] = []
    for start, end, pid, text in rows:
        if text == cycle_label:
            cycle_rows_by_pid[pid].append(Interval(start, end))
            continue
        stage = _STAGE_RE.match(text)
        if stage and stage.group("mode") == mode:
            rank = int(stage.group("rank"))
            previous = pid_to_rank.setdefault(pid, rank)
            if previous != rank:
                raise ValueError(f"PID {pid} maps to both rank {previous} and rank {rank}")
            pending_stages.append(
                (rank, pid, int(stage.group("index")), stage.group("stage"), start)
            )
            # Store the end separately by replacing the tuple below.
            stages_by_rank[rank].append(
                StageWindow(
                    rank=rank,
                    pid=pid,
                    stage=stage.group("stage"),
                    index=int(stage.group("index")),
                    interval=Interval(start, end),
                )
            )
            continue
        dit = _DIT_RE.match(text)
        if dit and dit.group("mode") == mode:
            rank = int(dit.group("rank"))
            previous = pid_to_rank.setdefault(pid, rank)
            if previous != rank:
                raise ValueError(f"PID {pid} maps to both rank {previous} and rank {rank}")
            role = dit.group("role")
            grad = dit.group("grad")
            value = dit_counts[rank]
            value["total"] += 1
            value["roles"][role] += 1
            value[f"{grad}_roles"][role] += 1

    cycles: dict[int, tuple[int, Interval]] = {}
    for pid, rank in pid_to_rank.items():
        values = cycle_rows_by_pid.get(pid, [])
        if len(values) != 1:
            raise ValueError(
                f"rank {rank} / pid {pid} expected one {cycle_label!r} range, found {len(values)}"
            )
        cycles[rank] = (pid, values[0])

    for rank, values in stages_by_rank.items():
        if rank not in cycles:
            raise ValueError(f"rank {rank} has stage markers but no exact cycle marker")
        cycle = cycles[rank][1]
        clipped: list[StageWindow] = []
        for value in sorted(values, key=lambda item: (item.interval.start_ns, item.index)):
            interval = value.interval.clip(cycle)
            if interval is None:
                continue
            clipped.append(dataclasses.replace(value, interval=interval))
        stages_by_rank[rank] = clipped

    normalized_counts: dict[int, dict[str, Any]] = {}
    for rank, value in dit_counts.items():
        normalized_counts[rank] = {
            "total": int(value["total"]),
            "roles": dict(value["roles"]),
            "grad_roles": dict(value["grad_roles"]),
            "nograd_roles": dict(value["nograd_roles"]),
        }
    return cycles, stages_by_rank, normalized_counts


def _validate_marker_contract(
    cycles: Mapping[int, tuple[int, Interval]],
    stages: Mapping[int, Sequence[StageWindow]],
    dit_counts: Mapping[int, Mapping[str, Any]],
    *,
    mode: str,
) -> None:
    if mode not in {"native", "distillschedule"}:
        raise ValueError(f"unsupported mode: {mode}")
    expected_stage_keys = ["student/0"] + [f"fake/{index}" for index in range(5)]
    expected_roles = {"student": 24, "fake": 6, "teacher": 1}
    expected_grad = {"student": 1, "fake": 5}
    expected_nograd = {"student": 23, "fake": 1, "teacher": 1}
    if not cycles:
        raise ValueError("no rank-qualified MiniMax cycle markers were found")
    for rank in sorted(cycles):
        stage_keys = [value.key for value in stages.get(rank, [])]
        if stage_keys != expected_stage_keys:
            raise ValueError(
                f"rank {rank} stage marker contract mismatch: {stage_keys} != {expected_stage_keys}"
            )
        counts = dit_counts.get(rank, {})
        if int(counts.get("total", 0)) != 31:
            raise ValueError(f"rank {rank} DIT marker count={counts.get('total')} != 31")
        roles = {key: int(counts.get("roles", {}).get(key, 0)) for key in expected_roles}
        grad = {key: int(counts.get("grad_roles", {}).get(key, 0)) for key in expected_grad}
        nograd = {key: int(counts.get("nograd_roles", {}).get(key, 0)) for key in expected_nograd}
        if roles != expected_roles or grad != expected_grad or nograd != expected_nograd:
            raise ValueError(
                f"rank {rank} DIT role/grad contract mismatch: roles={roles}, grad={grad}, nograd={nograd}"
            )


def _intersect(a: Iterable[Interval], b: Iterable[Interval]) -> tuple[Interval, ...]:
    left = merge_intervals(a)
    right = merge_intervals(b)
    result: list[Interval] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i].start_ns, right[j].start_ns)
        end = min(left[i].end_ns, right[j].end_ns)
        if start < end:
            result.append(Interval(start, end))
        if left[i].end_ns <= right[j].end_ns:
            i += 1
        else:
            j += 1
    return tuple(result)


def _subtract(a: Iterable[Interval], b: Iterable[Interval]) -> tuple[Interval, ...]:
    left = merge_intervals(a)
    right = merge_intervals(b)
    result: list[Interval] = []
    j = 0
    for value in left:
        cursor = value.start_ns
        while j < len(right) and right[j].end_ns <= cursor:
            j += 1
        scan = j
        while scan < len(right) and right[scan].start_ns < value.end_ns:
            other = right[scan]
            if cursor < other.start_ns:
                result.append(Interval(cursor, min(other.start_ns, value.end_ns)))
            cursor = max(cursor, other.end_ns)
            if cursor >= value.end_ns:
                break
            scan += 1
        if cursor < value.end_ns:
            result.append(Interval(cursor, value.end_ns))
    return tuple(result)


def _metric_from_intervals(intervals: Iterable[Interval], wall_ns: int) -> dict[str, Any]:
    union_ns = interval_union_ns(intervals)
    return {
        "union_ns": union_ns,
        "union_seconds": union_ns / 1e9,
        "percent_of_wall": 100.0 * union_ns / wall_ns if wall_ns else 0.0,
    }


def _series_metric(series: _Series, wall_ns: int) -> dict[str, Any]:
    union_ns = interval_union_ns(series.merged())
    return {
        "union_ns": union_ns,
        "union_seconds": union_ns / 1e9,
        "percent_of_wall": 100.0 * union_ns / wall_ns if wall_ns else 0.0,
        "events": int(series.events),
        "bytes": int(series.bytes),
        "gib": series.bytes / (1024**3),
        "sum_duration_ns": int(series.sum_duration_ns),
        "sum_duration_seconds": series.sum_duration_ns / 1e9,
        "out_of_order_rows": int(series.out_of_order_rows),
    }


def _finish_scope(scope: _ScopeAccumulator) -> dict[str, Any]:
    wall_ns = scope.window.duration_ns
    category_names = (
        COMPUTE,
        NCCL,
        NCCL_ALL_GATHER,
        NCCL_REDUCE_SCATTER,
        NCCL_ALL_REDUCE,
        NCCL_OTHER,
        H2D,
        D2H,
        D2D,
        OTHER_COPY,
        ALL_COPY,
        OFFLOAD_H2D,
        OFFLOAD_D2H,
        OFFLOAD,
    )
    categories = {
        name: _series_metric(scope.series.get(name, _Series()), wall_ns)
        for name in category_names
    }
    compute = scope.series.get(COMPUTE, _Series()).merged()
    nccl = scope.series.get(NCCL, _Series()).merged()
    offload = scope.series.get(OFFLOAD, _Series()).merged()
    all_copy = scope.series.get(ALL_COPY, _Series()).merged()
    movement = merge_intervals((*nccl, *offload))
    active = merge_intervals((*compute, *nccl, *all_copy))
    categories[ACTIVE] = _metric_from_intervals(active, wall_ns)
    no_activity_ns = max(0, wall_ns - interval_union_ns(active))
    categories[NO_TRACED_ACTIVITY] = {
        "union_ns": no_activity_ns,
        "union_seconds": no_activity_ns / 1e9,
        "percent_of_wall": 100.0 * no_activity_ns / wall_ns if wall_ns else 0.0,
    }

    compute_or_offload = merge_intervals((*compute, *offload))
    compute_or_nccl = merge_intervals((*compute, *nccl))
    nccl_offload_overlap = _intersect(nccl, offload)
    exposure = {
        "nccl_exposed_vs_compute": _metric_from_intervals(_subtract(nccl, compute), wall_ns),
        "offload_exposed_vs_compute": _metric_from_intervals(_subtract(offload, compute), wall_ns),
        "movement_exposed_vs_compute": _metric_from_intervals(_subtract(movement, compute), wall_ns),
        "nccl_unique_exposed": _metric_from_intervals(_subtract(nccl, compute_or_offload), wall_ns),
        "offload_unique_exposed": _metric_from_intervals(_subtract(offload, compute_or_nccl), wall_ns),
        "nccl_offload_shared_exposed": _metric_from_intervals(
            _subtract(nccl_offload_overlap, compute), wall_ns
        ),
        "all_copy_exposed_vs_compute": _metric_from_intervals(_subtract(all_copy, compute), wall_ns),
        "nccl_hidden_by_compute": _metric_from_intervals(_intersect(nccl, compute), wall_ns),
        "offload_hidden_by_compute": _metric_from_intervals(_intersect(offload, compute), wall_ns),
        "nccl_offload_overlap": _metric_from_intervals(nccl_offload_overlap, wall_ns),
    }
    histogram = [
        {
            "direction": direction,
            "bytes": num_bytes,
            "mib": num_bytes / (1024**2),
            "events": count,
        }
        for (direction, num_bytes), count in sorted(
            scope.copy_sizes.items(), key=lambda item: (-item[1], -item[0][1], item[0][0])
        )
    ]
    return {
        "wall_ns": wall_ns,
        "wall_seconds": wall_ns / 1e9,
        "categories": categories,
        "exposure": exposure,
        "candidate_offload_copy_size_histogram": histogram[:32],
    }


def _kernel_category(name: str) -> tuple[str, str | None]:
    lowered = name.lower()
    is_nccl = any(token in lowered for token in ("nccl", "nvls", "collnet"))
    if not is_nccl:
        return COMPUTE, None
    normalized = lowered.replace("_", "").replace("-", "")
    if "allgather" in normalized:
        return NCCL, NCCL_ALL_GATHER
    if "reducescatter" in normalized:
        return NCCL, NCCL_REDUCE_SCATTER
    if "allreduce" in normalized:
        return NCCL, NCCL_ALL_REDUCE
    return NCCL, NCCL_OTHER


def _build_rank_accumulators(
    cycles: Mapping[int, tuple[int, Interval]],
    stages: Mapping[int, Sequence[StageWindow]],
) -> dict[int, _RankAccumulator]:
    result: dict[int, _RankAccumulator] = {}
    for rank, (pid, cycle) in cycles.items():
        stage_windows = {value.key: value for value in stages.get(rank, [])}
        result[rank] = _RankAccumulator(
            rank=rank,
            pid=pid,
            cycle=_ScopeAccumulator(cycle),
            stages={key: _ScopeAccumulator(value.interval) for key, value in stage_windows.items()},
            stage_windows=stage_windows,
        )
    return result


def _add_to_rank_scopes(
    rank_acc: _RankAccumulator,
    category: str,
    interval: Interval,
    *,
    num_bytes: int = 0,
    copy_size_key: tuple[str, int] | None = None,
) -> None:
    rank_acc.cycle.add(
        category,
        interval,
        num_bytes=num_bytes,
        copy_size_key=copy_size_key,
    )
    for key, scope in rank_acc.stages.items():
        if interval.clip(scope.window) is not None:
            scope.add(category, interval, num_bytes=num_bytes)


def _scan_kernels(
    connection: sqlite3.Connection,
    rank_accumulators: Mapping[int, _RankAccumulator],
    pid_to_rank: Mapping[int, int],
    strings: Mapping[int, str],
    known_pids: set[int],
) -> None:
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if not _table_exists(connection, table):
        raise ValueError("Nsight SQLite has no CUPTI_ACTIVITY_KIND_KERNEL table")
    columns = _columns(connection, table)
    if not {"start", "end", "globalPid"}.issubset(columns):
        raise ValueError("kernel table lacks start/end/globalPid")
    name_column = None
    for candidate in ("demangledName", "shortName", "name"):
        if candidate in columns:
            name_column = candidate
            break
    selected = ["start", "end", "globalPid"] + ([name_column] if name_column else [])
    global_start = min(value.cycle.window.start_ns for value in rank_accumulators.values())
    global_end = max(value.cycle.window.end_ns for value in rank_accumulators.values())
    query = (
        f"SELECT {','.join(selected)} FROM {table} "
        "WHERE end>? AND start<?"
    )
    for row in connection.execute(query, (global_start, global_end)):
        start, end, raw_pid = row[:3]
        if start is None or end is None or raw_pid is None or int(end) <= int(start):
            continue
        pid = decode_pid(raw_pid, known_pids)
        rank = pid_to_rank.get(pid)
        if rank is None:
            continue
        interval = Interval(int(start), int(end))
        if interval.clip(rank_accumulators[rank].cycle.window) is None:
            continue
        name = _resolve_string(row[3], strings) if name_column else ""
        category, nccl_kind = _kernel_category(name)
        _add_to_rank_scopes(rank_accumulators[rank], category, interval)
        if nccl_kind is not None:
            _add_to_rank_scopes(rank_accumulators[rank], nccl_kind, interval)


def _scan_memcpy(
    connection: sqlite3.Connection,
    rank_accumulators: Mapping[int, _RankAccumulator],
    pid_to_rank: Mapping[int, int],
    known_pids: set[int],
    *,
    offload_min_bytes: int,
) -> None:
    table = "CUPTI_ACTIVITY_KIND_MEMCPY"
    if not _table_exists(connection, table):
        return
    columns = _columns(connection, table)
    required = {"start", "end", "globalPid", "bytes", "copyKind"}
    if not required.issubset(columns):
        raise ValueError(f"memcpy table lacks required columns: {required - columns}")
    has_correlation = "correlationId" in columns
    selected = ["start", "end", "globalPid", "bytes", "copyKind"]
    if has_correlation:
        selected.append("correlationId")
    global_start = min(value.cycle.window.start_ns for value in rank_accumulators.values())
    global_end = max(value.cycle.window.end_ns for value in rank_accumulators.values())
    query = (
        f"SELECT {','.join(selected)} FROM {table} "
        "WHERE end>? AND start<?"
    )
    for row in connection.execute(query, (global_start, global_end)):
        start, end, raw_pid, raw_bytes, raw_kind = row[:5]
        if start is None or end is None or raw_pid is None or int(end) <= int(start):
            continue
        pid = decode_pid(raw_pid, known_pids)
        rank = pid_to_rank.get(pid)
        if rank is None:
            continue
        interval = Interval(int(start), int(end))
        rank_acc = rank_accumulators[rank]
        if interval.clip(rank_acc.cycle.window) is None:
            continue
        num_bytes = int(raw_bytes or 0)
        category = _COPY_KIND.get(int(raw_kind), OTHER_COPY)
        _add_to_rank_scopes(
            rank_acc,
            category,
            interval,
            num_bytes=num_bytes,
            copy_size_key=(category, num_bytes),
        )
        _add_to_rank_scopes(rank_acc, ALL_COPY, interval, num_bytes=num_bytes)
        if category in {H2D, D2H} and num_bytes >= offload_min_bytes:
            specific = OFFLOAD_H2D if category == H2D else OFFLOAD_D2H
            _add_to_rank_scopes(rank_acc, specific, interval, num_bytes=num_bytes)
            _add_to_rank_scopes(rank_acc, OFFLOAD, interval, num_bytes=num_bytes)
            if has_correlation and row[5] is not None:
                rank_acc.offload_correlation_ids.add(int(row[5]))


def _scan_offload_runtime_api(
    connection: sqlite3.Connection,
    rank_accumulators: Mapping[int, _RankAccumulator],
    pid_to_rank: Mapping[int, int],
    strings: Mapping[int, str],
    known_pids: set[int],
) -> None:
    table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if not _table_exists(connection, table):
        return
    columns = _columns(connection, table)
    required = {"start", "end", "globalTid", "correlationId"}
    if not required.issubset(columns):
        return
    name_column = "nameId" if "nameId" in columns else ("name" if "name" in columns else None)
    selected = ["start", "end", "globalTid", "correlationId"]
    if name_column:
        selected.append(name_column)
    global_start = min(value.cycle.window.start_ns for value in rank_accumulators.values())
    global_end = max(value.cycle.window.end_ns for value in rank_accumulators.values())
    query = (
        f"SELECT {','.join(selected)} FROM {table} "
        "WHERE end>? AND start<?"
    )
    for row in connection.execute(query, (global_start, global_end)):
        start, end, raw_tid, correlation_id = row[:4]
        if None in (start, end, raw_tid, correlation_id) or int(end) <= int(start):
            continue
        pid = decode_pid(raw_tid, known_pids)
        rank = pid_to_rank.get(pid)
        if rank is None:
            continue
        rank_acc = rank_accumulators[rank]
        if int(correlation_id) not in rank_acc.offload_correlation_ids:
            continue
        interval = Interval(int(start), int(end))
        if interval.clip(rank_acc.cycle.window) is None:
            continue
        name = _resolve_string(row[4], strings) if name_column else "cuda_runtime_api"
        rank_acc.offload_runtime_api[name].add(interval)


def _finish_rank(
    value: _RankAccumulator,
    dit_counts: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_apis = {
        name: _series_metric(series, value.cycle.window.duration_ns)
        for name, series in sorted(value.offload_runtime_api.items())
    }
    return {
        "rank": value.rank,
        "pid": value.pid,
        "cycle_window": dataclasses.asdict(value.cycle.window),
        "cycle": _finish_scope(value.cycle),
        "stages": {
            key: {
                "stage": stage.stage,
                "index": stage.index,
                "window": dataclasses.asdict(stage.interval),
                "metrics": _finish_scope(value.stages[key]),
            }
            for key, stage in value.stage_windows.items()
        },
        "dit_marker_census": dict(dit_counts),
        "offload_runtime_api": runtime_apis,
    }


def analyze_minimax_node_sqlite(
    path: str | Path,
    *,
    mode: str = "native",
    offload_min_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if offload_min_bytes <= 0:
        raise ValueError("offload_min_bytes must be positive")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA mmap_size=268435456")
    connection.execute("PRAGMA cache_size=-65536")
    try:
        strings = _strings(connection)
        known_pids = _known_pids(connection)
        nvtx = _nvtx_rows(connection, strings, known_pids)
        cycles, stages, dit_counts = _discover_windows(nvtx, mode=mode)
        _validate_marker_contract(cycles, stages, dit_counts, mode=mode)
        rank_accumulators = _build_rank_accumulators(cycles, stages)
        pid_to_rank = {value.pid: rank for rank, value in rank_accumulators.items()}
        _scan_kernels(connection, rank_accumulators, pid_to_rank, strings, known_pids)
        _scan_memcpy(
            connection,
            rank_accumulators,
            pid_to_rank,
            known_pids,
            offload_min_bytes=offload_min_bytes,
        )
        _scan_offload_runtime_api(
            connection,
            rank_accumulators,
            pid_to_rank,
            strings,
            known_pids,
        )
        ranks = [
            _finish_rank(rank_accumulators[rank], dit_counts[rank])
            for rank in sorted(rank_accumulators)
        ]
        return {
            "schema_version": "distillschedule.lightx2v_minimax_nsys_node.v1",
            "sqlite": str(path),
            "mode": mode,
            "offload_min_bytes": int(offload_min_bytes),
            "offload_min_mib": offload_min_bytes / (1024**2),
            "rank_count": len(ranks),
            "ranks": ranks,
            "claim_boundary": {
                "diagnostic_only": True,
                "formal_timing_unchanged": True,
                "exposed_is_interval_attribution_not_causal_speedup": True,
                "offload_is_large_h2d_d2h_candidate_heuristic": True,
                "nccl_operation_breakdown_is_kernel_name_based": True,
                "advanced_nccl_copy_engine_collectives_may_require_separate_attribution": True,
            },
        }
    finally:
        connection.close()


def _aggregate_scope_reports(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot aggregate an empty scope")
    wall_ns = sum(int(value["wall_ns"]) for value in values)
    wall_seconds = [float(value["wall_seconds"]) for value in values]
    category_names = sorted(set.intersection(*(set(value["categories"]) for value in values)))
    exposure_names = sorted(set.intersection(*(set(value["exposure"]) for value in values)))
    categories: dict[str, Any] = {}
    for name in category_names:
        rows = [value["categories"][name] for value in values]
        union_ns = sum(int(row["union_ns"]) for row in rows)
        categories[name] = {
            "gpu_seconds": union_ns / 1e9,
            "pooled_percent": 100.0 * union_ns / wall_ns if wall_ns else 0.0,
            "events": sum(int(row.get("events", 0)) for row in rows),
            "bytes": sum(int(row.get("bytes", 0)) for row in rows),
            "gib": sum(int(row.get("bytes", 0)) for row in rows) / (1024**3),
            "rank_mean_seconds": statistics.fmean(float(row["union_seconds"]) for row in rows),
            "rank_median_seconds": statistics.median(float(row["union_seconds"]) for row in rows),
            "rank_max_seconds": max(float(row["union_seconds"]) for row in rows),
        }
    exposure: dict[str, Any] = {}
    for name in exposure_names:
        rows = [value["exposure"][name] for value in values]
        union_ns = sum(int(row["union_ns"]) for row in rows)
        exposure[name] = {
            "gpu_seconds": union_ns / 1e9,
            "pooled_percent": 100.0 * union_ns / wall_ns if wall_ns else 0.0,
            "rank_mean_seconds": statistics.fmean(float(row["union_seconds"]) for row in rows),
            "rank_median_seconds": statistics.median(float(row["union_seconds"]) for row in rows),
            "rank_max_seconds": max(float(row["union_seconds"]) for row in rows),
        }
    return {
        "rank_count": len(values),
        "gpu_wall_seconds": wall_ns / 1e9,
        "rank_mean_wall_seconds": statistics.fmean(wall_seconds),
        "rank_median_wall_seconds": statistics.median(wall_seconds),
        "rank_max_wall_seconds": max(wall_seconds),
        "categories": categories,
        "exposure": exposure,
        "denominator": "sum(rank interval union) / sum(rank exact MiniMax NVTX wall)",
    }


def aggregate_minimax_world(
    node_reports: Sequence[Mapping[str, Any]],
    *,
    expected_world_size: int = 16,
) -> dict[str, Any]:
    per_rank = [rank for node in node_reports for rank in node["ranks"]]
    per_rank.sort(key=lambda value: int(value["rank"]))
    ranks = [int(value["rank"]) for value in per_rank]
    if len(per_rank) != expected_world_size or ranks != list(range(expected_world_size)):
        raise ValueError(
            f"world profile ranks={ranks}; expected exact ranks 0..{expected_world_size - 1}"
        )
    cycle = _aggregate_scope_reports([value["cycle"] for value in per_rank])
    stage_keys = ["student/0"] + [f"fake/{index}" for index in range(5)]
    stages: dict[str, Any] = {}
    for key in stage_keys:
        stages[key] = _aggregate_scope_reports(
            [value["stages"][key]["metrics"] for value in per_rank]
        )
    return {
        "schema_version": "distillschedule.lightx2v_minimax_nsys_world16.v1",
        "mode": node_reports[0]["mode"],
        "offload_min_bytes": node_reports[0]["offload_min_bytes"],
        "rank_count": len(per_rank),
        "cycle": cycle,
        "stages": stages,
        "per_rank": per_rank,
        "claim_boundary": {
            "diagnostic_only": True,
            "profiled_wall_is_not_formal_timing": True,
            "rank_max_exposed_seconds_is_not_a_speedup_prediction": True,
            "exposed_nccl_or_copy_means_no_non_nccl_compute_kernel_overlap": True,
            "offload_candidate_uses_size_and_direction_not_source_level_identity": True,
        },
    }


def analyze_minimax_world16(
    node_sqlites: Sequence[str | Path],
    *,
    mode: str = "native",
    offload_min_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    if len(node_sqlites) != 2:
        raise ValueError(f"world16 MiniMax analysis requires two node SQLite files, got {len(node_sqlites)}")
    nodes = [
        analyze_minimax_node_sqlite(
            path,
            mode=mode,
            offload_min_bytes=offload_min_bytes,
        )
        for path in node_sqlites
    ]
    if any(node["mode"] != mode for node in nodes):
        raise ValueError("node profile mode mismatch")
    result = aggregate_minimax_world(nodes)
    result["nodes"] = nodes
    return result


__all__ = [
    "ACTIVE",
    "ALL_COPY",
    "COMPUTE",
    "D2D",
    "D2H",
    "H2D",
    "NCCL",
    "NCCL_ALL_GATHER",
    "NCCL_ALL_REDUCE",
    "NCCL_OTHER",
    "NCCL_REDUCE_SCATTER",
    "NO_TRACED_ACTIVITY",
    "OFFLOAD",
    "OFFLOAD_D2H",
    "OFFLOAD_H2D",
    "aggregate_minimax_world",
    "analyze_minimax_node_sqlite",
    "analyze_minimax_world16",
    "decode_pid",
]
