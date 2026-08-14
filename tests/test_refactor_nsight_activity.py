from __future__ import annotations

import sqlite3

import pytest

from distillschedule.profiling.nsight_activity import (
    ACTIVE,
    ALL_COPY,
    COMPUTE,
    D2H,
    H2D,
    Interval,
    NCCL,
    NO_TRACED_ACTIVITY,
    aggregate_rank_reports,
    analyze_sqlite,
    complement_intervals,
    interval_union_ns,
    merge_intervals,
)
from distillschedule.profiling.static_nsys_bounded import analyze_bounded_sqlite
from distillschedule.profiling.static_profile_diagnostics import StaticProfileDiagnostics


def test_interval_union_and_complement_are_exact() -> None:
    merged = merge_intervals((Interval(0, 10), Interval(5, 12), Interval(20, 25)))
    assert merged == (Interval(0, 12), Interval(20, 25))
    assert interval_union_ns(merged) == 17
    assert complement_intervals(merged, Interval(0, 30)) == (
        Interval(12, 20),
        Interval(25, 30),
    )


def _sqlite(path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, shortName INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER, copyKind INTEGER, bytes INTEGER);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, textId INTEGER);
            """
        )
        connection.executemany(
            "INSERT INTO StringIds(id, value) VALUES (?, ?)",
            ((1, "my_compute_kernel"), (2, "ncclAllReduceRingLLKernel"), (3, "phase_a"), (4, "phase_b")),
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)",
            ((0, 20, 1), (10, 30, 2), (60, 80, 1)),
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?)",
            ((25, 50, 1, 1000), (70, 90, 2, 2000)),
        )
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)",
            ((0, 50, 3), (50, 100, 4)),
        )


def test_sqlite_report_separates_semantic_and_cuda_activity(tmp_path) -> None:
    path = tmp_path / "rank.sqlite"
    _sqlite(path)
    report = analyze_sqlite(
        path,
        rank=2,
        phase_patterns={"phase_a": "phase_a", "phase_b": "phase_b"},
        capture_window=Interval(0, 100),
        capture_coverage="exact",
    )
    capture = report["capture_metrics"]
    assert capture["categories"][COMPUTE]["union_ns"] == 40
    assert capture["categories"][NCCL]["union_ns"] == 20
    assert capture["categories"][H2D]["union_ns"] == 25
    assert capture["categories"][D2H]["union_ns"] == 20
    assert capture["categories"][ALL_COPY]["union_ns"] == 45
    assert capture["categories"][ACTIVE]["union_ns"] == 80
    assert capture["categories"][NO_TRACED_ACTIVITY]["union_ns"] == 20
    assert report["phases"]["phase_a"]["wall_ns"] == 50
    assert report["phases"]["phase_b"]["wall_ns"] == 50
    assert report["claim_boundary"]["semantic_envelope_is_not_compute_utilization"]
    timeline = report["timeline"]
    assert timeline["display_semantics"][
        "activity_fraction_is_exact_interval_union_inside_each_bin"
    ]
    assert len(timeline["activity_fraction"][COMPUTE]) == 1
    assert timeline["activity_fraction"][COMPUTE][0] == pytest.approx(0.4)
    assert timeline["activity_fraction"][NO_TRACED_ACTIVITY][0] == pytest.approx(0.2)


def test_gpu_seconds_aggregate_uses_summed_rank_wall(tmp_path) -> None:
    path = tmp_path / "rank.sqlite"
    _sqlite(path)
    reports = [
        analyze_sqlite(path, rank=rank, capture_window=Interval(0, 100))
        for rank in (0, 1)
    ]
    aggregate = aggregate_rank_reports(reports)
    assert aggregate["capture"]["gpu_wall_seconds"] == pytest.approx(200 / 1e9)
    assert aggregate["capture"]["categories"][COMPUTE]["pooled_percent"] == pytest.approx(40.0)
    assert aggregate["capture"]["denominator"] == "sum(rank activity union) / sum(rank phase wall)"


def test_nvtx_null_metadata_row_is_ignored(tmp_path) -> None:
    path = tmp_path / "rank.sqlite"
    _sqlite(path)
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)", (None, None, None))
    report = analyze_sqlite(
        path,
        rank=0,
        phase_patterns={"phase_a": "phase_a"},
        capture_window=Interval(0, 100),
    )
    assert report["phases"]["phase_a"]["wall_ns"] == 50


def test_bounded_node_analyzer_reconciles_copy_totals_and_keeps_process_detail(tmp_path) -> None:
    path = tmp_path / "node.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE PROCESSES(globalPid INTEGER, pid INTEGER, name TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, globalPid INTEGER, demangledName INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER, globalPid INTEGER, bytes INTEGER, copyKind INTEGER);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT);
            """
        )
        connection.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            ((1, "compute_kernel"), (2, "ncclAllReduceRingLLKernel")),
        )
        # Nsight globalPid is pid << 24 in the real export.
        connection.executemany(
            "INSERT INTO PROCESSES VALUES (?, ?, ?)",
            ((11 << 24, 11, "python"), (12 << 24, 12, "python")),
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
            ((0, 20, 11 << 24, 1), (10, 30, 12 << 24, 2)),
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?, ?)",
            ((25, 50, 11 << 24, 1000, 1), (40, 60, 12 << 24, 2000, 2)),
        )
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)",
            (0, 100, "STATIC_FSDP2_FULL5/no-gan"),
        )
    report = analyze_bounded_sqlite(
        path, phase_label="STATIC_FSDP2_FULL5/no-gan", node_rank_base=0
    )
    assert report["node_metrics"][COMPUTE]["union_ns"] == 20
    assert report["node_metrics"][NCCL]["union_ns"] == 20
    assert report["node_metrics"][ALL_COPY]["union_ns"] == 35
    assert report["node_metrics"][H2D]["bytes"] == 1000
    assert report["node_metrics"][D2H]["bytes"] == 2000
    assert report["memcpy_totals_reconciliation"]["status"] == "PASS"
    assert report["memcpy_totals_reconciliation"]["residual"] == {
        "events": 0,
        "bytes": 0,
        "sum_duration_ns": 0,
    }
    assert [item["rank"] for item in report["per_process"]] == [0, 1]


def test_profile_diagnostics_clean_mode_does_not_materialize_scalars(tmp_path) -> None:
    recorder = StaticProfileDiagnostics(
        enabled=False,
        rank=0,
        output_path=tmp_path / "rank.json",
        collect_cpu_diagnostics=False,
    )
    assert recorder.diagnostic_scalar("loss", 1.0) is None
    assert recorder.diagnostic_sha256("x0", object()) is None
    assert recorder.counters == {}
