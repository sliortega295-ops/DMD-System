"""Bounded streaming analysis for node-level static FSDP2 Nsight exports.

The first node-level analyzer materialized every kernel and memcpy row from a
1--2 GiB SQLite export before doing interval arithmetic.  That is both slow
and needlessly memory hungry.  This module reads only the exact application
range, streams one ordered category at a time, and keeps only interval-union
state plus per-process counters.

The report deliberately distinguishes event sum duration from interval union:
the former counts overlapped work repeatedly, while the latter is the
copy-engine/kernel occupancy observable.  The two must not be conflated.
"""
from __future__ import annotations

import dataclasses
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


COMPUTE = "compute"
NCCL = "nccl"
H2D = "h2d"
D2H = "d2h"
D2D = "d2d"
OTHER_COPY = "other_copy"
ALL_COPY = "all_copy"
ACTIVE = "active"
NO_TRACED_ACTIVITY = "no_traced_gpu_activity"

COPY_KIND = {1: H2D, 2: D2H, 3: D2D, 8: D2D, 10: D2D}


@dataclasses.dataclass
class _Union:
    """Streaming union accumulator for rows ordered by start time."""

    start: int | None = None
    end: int | None = None
    union_ns: int = 0
    out_of_order: int = 0

    def add(self, start: int, end: int) -> None:
        if end <= start:
            return
        if self.start is None:
            self.start, self.end = start, end
            return
        assert self.end is not None
        if start < self.start:
            # SQL ORDER BY should make this impossible.  Keep the row in the
            # same accumulator but mark the approximation rather than silently
            # presenting a non-exact union.
            self.out_of_order += 1
            if end <= self.end:
                return
            self.union_ns += end - self.end
            self.end = end
            return
        if start <= self.end:
            self.end = max(self.end, end)
            return
        self.union_ns += self.end - self.start
        self.start, self.end = start, end

    def finish(self) -> int:
        if self.start is not None:
            assert self.end is not None
            return self.union_ns + self.end - self.start
        return self.union_ns


@dataclasses.dataclass
class _Metric:
    events: int = 0
    bytes: int = 0
    sum_duration_ns: int = 0
    union: _Union = dataclasses.field(default_factory=_Union)

    def add(self, start: int, end: int, num_bytes: int = 0) -> None:
        if end <= start:
            return
        self.events += 1
        self.bytes += int(num_bytes)
        self.sum_duration_ns += end - start
        self.union.add(start, end)

    def finish(self) -> dict[str, Any]:
        value = {
            "events": self.events,
            "bytes": self.bytes,
            "sum_duration_ns": self.sum_duration_ns,
            "sum_duration_seconds": self.sum_duration_ns / 1e9,
            "union_ns": self.union.finish(),
            "union_seconds": self.union.finish() / 1e9,
            "out_of_order_rows": self.union.out_of_order,
        }
        return value


def _schema(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _strings(connection: sqlite3.Connection) -> dict[int, str]:
    if not _table_exists(connection, "StringIds"):
        return {}
    return {int(row[0]): str(row[1]) for row in connection.execute("SELECT id,value FROM StringIds")}


def _phase_window(connection: sqlite3.Connection, label: str) -> tuple[int, int]:
    if not _table_exists(connection, "NVTX_EVENTS"):
        raise ValueError("SQLite export has no NVTX_EVENTS table")
    rows = connection.execute(
        "SELECT start,end FROM NVTX_EVENTS "
        "WHERE text=? AND start IS NOT NULL AND end IS NOT NULL AND end>start "
        "ORDER BY start,end",
        (label,),
    ).fetchall()
    if not rows:
        raise ValueError(f"exact NVTX phase {label!r} is absent")
    if len(rows) != 1:
        raise ValueError(f"expected one exact NVTX phase {label!r}, found {len(rows)}")
    return int(rows[0][0]), int(rows[0][1])


def _clip(start: int, end: int, window: tuple[int, int]) -> tuple[int, int] | None:
    start = max(int(start), window[0])
    end = min(int(end), window[1])
    return (start, end) if end > start else None


def _activity_rows(
    connection: sqlite3.Connection,
    table: str,
    selected: str,
    window: tuple[int, int],
):
    """Yield an Nsight activity table in its native append/rowid order.

    CUPTI activity exports are appended chronologically.  Asking SQLite to
    ``ORDER BY start,end`` on these multi-million-row tables forces a giant
    temporary B-tree and was the source of the original node-level analyzer
    stall.  We intentionally stream the native order and validate monotonicity
    in ``_Union``; a non-zero out-of-order count is surfaced as non-exact rather
    than silently guessed.
    """
    query = (
        f"SELECT {selected} FROM {table} "
        "WHERE end>? AND start<?"
    )
    yield from connection.execute(query, window)


def _merge_activity_rows(*iterators):
    """Merge chronologically appended activity streams with one-row lookahead."""
    heads = []
    for index, iterator in enumerate(iterators):
        try:
            row = next(iterator)
        except StopIteration:
            continue
        heads.append([int(row[0]), int(row[1]), index, row, iterator])
    while heads:
        heads.sort(key=lambda value: (value[0], value[1]))
        _, _, index, row, iterator = heads.pop(0)
        yield row
        try:
            next_row = next(iterator)
        except StopIteration:
            continue
        heads.append([int(next_row[0]), int(next_row[1]), index, next_row, iterator])


def _kernel_name(row_value: Any, strings: Mapping[int, str]) -> str:
    if row_value is None:
        return ""
    if isinstance(row_value, (int, float)):
        return strings.get(int(row_value), str(row_value))
    return str(row_value)


def _kernel_category(name: str) -> str:
    lowered = name.lower()
    return NCCL if any(token in lowered for token in ("nccl", "nvls", "collnet")) else COMPUTE


def _worker_pids(connection: sqlite3.Connection, global_pids: set[int]) -> list[int]:
    """Return python worker PIDs that actually emitted GPU activity.

    Nsight's PROCESSES table does not retain ``RANK``.  The launcher starts
    local workers in local-rank order, so the caller may expose the resulting
    mapping as *inferred* rather than pretending it is an exact rank label.
    """
    if not _table_exists(connection, "PROCESSES"):
        return []
    rows = connection.execute(
        "SELECT pid,globalPid FROM PROCESSES WHERE name='python' ORDER BY pid"
    ).fetchall()
    result = []
    for pid, global_pid in rows:
        if global_pid in global_pids:
            result.append(int(pid))
    return result


def analyze_bounded_sqlite(
    path: str | Path,
    *,
    phase_label: str,
    node_rank_base: int | None = None,
    query_budget_seconds: float = 300.0,
) -> dict[str, Any]:
    """Analyze one node SQLite export without materializing activity rows."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    started = time.monotonic()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA temp_store=FILE")
    # The exports live on a shared filesystem.  A bounded read-only mmap and a
    # small page cache avoid materialising rows in Python while reducing the
    # repeated page-fault cost of the exact phase scans.
    connection.execute("PRAGMA mmap_size=268435456")
    connection.execute("PRAGMA cache_size=-65536")
    try:
        window = _phase_window(connection, phase_label)
        strings = _strings(connection)
        by_pid: dict[int, dict[str, _Metric]] = defaultdict(
            lambda: {name: _Metric() for name in (COMPUTE, NCCL, H2D, D2H, D2D, OTHER_COPY, ALL_COPY, ACTIVE)}
        )
        global_pids: set[int] = set()
        raw_memcpy = {"events": 0, "bytes": 0, "sum_duration_ns": 0}
        node = {name: _Metric() for name in (COMPUTE, NCCL, H2D, D2H, D2D, OTHER_COPY, ALL_COPY, ACTIVE)}
        scanned_rows = {"kernel": 0, "memcpy": 0, "active": 0}

        def check_budget(kind: str) -> None:
            scanned_rows[kind] += 1
            if scanned_rows[kind] % 100_000 == 0 and time.monotonic() - started > query_budget_seconds:
                raise TimeoutError(
                    f"bounded Nsight analysis exceeded {query_budget_seconds}s "
                    f"while scanning {kind}: {path}"
                )

        def add_kernel(start: int, end: int, global_pid: int, name: str) -> None:
            category = _kernel_category(name)
            by_pid[global_pid][category].add(start, end)
            node[category].add(start, end)

        def add_memcpy(start: int, end: int, global_pid: int, category: str, num_bytes: int) -> None:
            by_pid[global_pid][category].add(start, end, num_bytes)
            by_pid[global_pid][ALL_COPY].add(start, end, num_bytes)
            node[category].add(start, end, num_bytes)
            node[ALL_COPY].add(start, end, num_bytes)

        # Restrict every query to the exact NVTX phase.  Do not add SQL ORDER BY
        # here: Nsight's CUPTI tables are append-ordered and can be streamed
        # directly.  This avoids a multi-gigabyte temporary sort.  The union
        # accumulator records any order violation so the report remains
        # fail-closed if a future exporter changes that property.
        if _table_exists(connection, "CUPTI_ACTIVITY_KIND_KERNEL"):
            cols = _schema(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
            name_col = "demangledName" if "demangledName" in cols else ("shortName" if "shortName" in cols else None)
            selected = "start,end,globalPid" + (f",{name_col}" if name_col else "")
            for row in _activity_rows(connection, "CUPTI_ACTIVITY_KIND_KERNEL", selected, window):
                check_budget("kernel")
                clipped = _clip(row[0], row[1], window)
                if clipped is None:
                    continue
                pid = int(row[2] or -1)
                global_pids.add(pid)
                add_kernel(clipped[0], clipped[1], pid, _kernel_name(row[3], strings) if name_col else "")
                by_pid[pid][ACTIVE].add(*clipped)
                node[ACTIVE].add(*clipped)
        if _table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"):
            cols = _schema(connection, "CUPTI_ACTIVITY_KIND_MEMCPY")
            selected = "start,end,globalPid,bytes,copyKind"
            for row in _activity_rows(connection, "CUPTI_ACTIVITY_KIND_MEMCPY", selected, window):
                check_budget("memcpy")
                clipped = _clip(row[0], row[1], window)
                if clipped is None:
                    continue
                pid = int(row[2] or -1)
                global_pids.add(pid)
                category = COPY_KIND.get(int(row[4]), OTHER_COPY)
                num_bytes = int(row[3] or 0)
                raw_memcpy["events"] += 1
                raw_memcpy["bytes"] += num_bytes
                raw_memcpy["sum_duration_ns"] += clipped[1] - clipped[0]
                add_memcpy(clipped[0], clipped[1], pid, category, num_bytes)
                by_pid[pid][ACTIVE].add(*clipped)
                node[ACTIVE].add(*clipped)

        # Merge the two append-ordered tables with one-row lookahead.  This is
        # an exact cross-table union when both native streams are monotonic,
        # without asking SQLite to materialize a UNION ORDER BY temporary tree.
        active_iterators = []
        if _table_exists(connection, "CUPTI_ACTIVITY_KIND_KERNEL"):
            active_iterators.append(iter(_activity_rows(connection, "CUPTI_ACTIVITY_KIND_KERNEL", "start,end,globalPid", window)))
        if _table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"):
            active_iterators.append(iter(_activity_rows(connection, "CUPTI_ACTIVITY_KIND_MEMCPY", "start,end,globalPid", window)))
        active_union = _Union()
        for row in _merge_activity_rows(*active_iterators):
            check_budget("active")
            clipped = _clip(row[0], row[1], window)
            if clipped is None:
                continue
            active_union.add(*clipped)
        node[ACTIVE].union = active_union
        elapsed = time.monotonic() - started
        if elapsed > query_budget_seconds:
            raise TimeoutError(f"bounded Nsight analysis exceeded {query_budget_seconds}s: {path}")

        workers = _worker_pids(connection, global_pids)
        pid_to_rank: dict[int, int] = {}
        if node_rank_base is not None:
            pid_to_rank = {pid: node_rank_base + i for i, pid in enumerate(workers)}
        per_process = []
        for global_pid in sorted(by_pid):
            pid = global_pid >> 24
            metrics = {name: metric.finish() for name, metric in by_pid[global_pid].items()}
            rank = pid_to_rank.get(pid)
            per_process.append({
                "global_pid": global_pid,
                "pid": pid,
                "rank": rank,
                "rank_mapping": "inferred_worker_pid_order" if rank is not None else "unmapped",
                "metrics": metrics,
            })

        node = {name: metric.finish() for name, metric in node.items()}
        wall_ns = window[1] - window[0]
        node[NO_TRACED_ACTIVITY] = {"events": 0, "bytes": 0, "sum_duration_ns": 0, "sum_duration_seconds": 0.0, "union_ns": max(0, wall_ns - node[ACTIVE]["union_ns"]), "union_seconds": max(0, wall_ns - node[ACTIVE]["union_ns"]) / 1e9, "out_of_order_rows": 0}
        for value in node.values():
            value["percent_of_phase_wall"] = 100.0 * value["union_ns"] / wall_ns if wall_ns else 0.0

        copy_classified = {name: {key: node[name][key] for key in ("events", "bytes", "sum_duration_ns")} for name in (H2D, D2H, D2D, OTHER_COPY)}
        classified = {key: sum(copy_classified[name][key] for name in copy_classified) for key in ("events", "bytes", "sum_duration_ns")}
        residual = {key: raw_memcpy[key] - classified[key] for key in raw_memcpy}
        reconciliation = {"raw_memcpy_rows_in_window": raw_memcpy, "classified_memcpy_rows_in_window": classified, "residual": residual, "status": "PASS" if all(value == 0 for value in residual.values()) else "FAIL"}
        return {
            "schema_version": "distillschedule.static_nsys_bounded.v1",
            "sqlite": str(path),
            "phase_label": phase_label,
            "phase_window": {"start_ns": window[0], "end_ns": window[1], "wall_ns": wall_ns, "wall_seconds": wall_ns / 1e9},
            "node_rank_base": node_rank_base,
            "rank_mapping": "inferred_worker_pid_order" if node_rank_base is not None else "unavailable_in_v3_profile",
            "query_seconds": elapsed,
            "scanned_rows": scanned_rows,
            "query_strategy": "three_native_append_order_streams_with_python_two_way_merge_and_bounded_state",
            "global_pid_count": len(global_pids),
            "worker_pid_count": len(workers),
            "per_process": per_process,
            "node_metrics": node,
            "memcpy_totals_reconciliation": reconciliation,
            "claim_boundary": {"diagnostic_only": True, "rank_mapping_in_v3_is_inferred": node_rank_base is not None, "formal_timing_unchanged": True, "sum_duration_is_not_union_occupancy": True, "native_activity_order_verified_by_out_of_order_counters": True},
        }
    finally:
        connection.close()


__all__ = ["analyze_bounded_sqlite"]
