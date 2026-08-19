from __future__ import annotations

import sqlite3

import pytest

from distillschedule.profiling.minimax_nsys_attribution import (
    COMPUTE,
    NCCL,
    OFFLOAD,
    analyze_minimax_node_sqlite,
    analyze_minimax_world16,
    decode_pid,
)


def _global_pid(pid: int) -> int:
    return pid << 24


def _global_tid(pid: int, tid: int = 7) -> int:
    return (pid << 24) | tid


def _insert_rank_profile(connection: sqlite3.Connection, *, rank: int, pid: int) -> None:
    gpid = _global_pid(pid)
    gtid = _global_tid(pid)
    connection.execute("INSERT INTO PROCESSES(pid, name) VALUES (?, 'python')", (pid,))

    connection.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
        (0, 1000, "MINIMAX_MATCHED_CYCLE/native", gtid),
    )
    stage_windows = [
        (0, 200, "student", 0),
        (200, 360, "fake", 0),
        (360, 520, "fake", 1),
        (520, 680, "fake", 2),
        (680, 840, "fake", 3),
        (840, 1000, "fake", 4),
    ]
    for start, end, stage, index in stage_windows:
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
            (start, end, f"MINIMAX_NSYS_STAGE/native/rank={rank}/{stage}/{index}", gtid),
        )

    calls: list[tuple[str, str]] = []
    calls.extend([("student", "nograd")] * 23)
    calls.append(("student", "grad"))
    calls.append(("fake", "nograd"))
    calls.extend([("fake", "grad")] * 5)
    calls.append(("teacher", "nograd"))
    assert len(calls) == 31
    for index, (role, grad) in enumerate(calls, start=1):
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
            (
                100 + index,
                101 + index,
                f"MINIMAX_NSYS_DIT/native/rank={rank}/{role}/{grad}/{index}",
                gtid,
            ),
        )

    # Compute [0,50], NCCL [40,70].  NCCL exposed vs compute is [50,70] = 20.
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
        (0, 50, gpid, 1),
    )
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
        (40, 70, gpid, 2),
    )

    # Large D2H is hidden by compute; large H2D is exposed in [60,80].
    # Their correlation IDs are also represented in the CUDA runtime table.
    large = 100 * 1024 * 1024
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?, ?, ?)",
        (10, 30, gpid, large, 2, 1000 + rank * 2),
    )
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?, ?, ?)",
        (60, 80, gpid, large, 1, 1001 + rank * 2),
    )
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        (8, 9, gtid, 1000 + rank * 2, 3),
    )
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        (58, 59, gtid, 1001 + rank * 2, 3),
    )


def _make_node_sqlite(path, ranks: range) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE PROCESSES(pid INTEGER, name TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                start INTEGER, end INTEGER, globalPid INTEGER, demangledName INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
                start INTEGER, end INTEGER, globalPid INTEGER, bytes INTEGER,
                copyKind INTEGER, correlationId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
                start INTEGER, end INTEGER, globalTid INTEGER,
                correlationId INTEGER, nameId INTEGER
            );
            """
        )
        connection.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            (
                (1, "my_compute_kernel"),
                (2, "ncclAllGatherRingLLKernel"),
                (3, "cudaMemcpyAsync"),
            ),
        )
        for rank in ranks:
            _insert_rank_profile(connection, rank=rank, pid=4000 + rank)


def test_decode_pid_handles_serialized_global_id_and_plain_pid() -> None:
    pid = 4321
    known = {pid}
    assert decode_pid(_global_pid(pid), known) == pid
    assert decode_pid(_global_tid(pid), known) == pid
    assert decode_pid(pid, known) == pid


def test_node_analyzer_reports_exposed_nccl_and_offload(tmp_path) -> None:
    path = tmp_path / "node0.sqlite"
    _make_node_sqlite(path, range(8))
    report = analyze_minimax_node_sqlite(
        path,
        mode="native",
        offload_min_bytes=64 * 1024 * 1024,
    )
    assert report["rank_count"] == 8
    rank0 = report["ranks"][0]
    cycle = rank0["cycle"]
    assert cycle["categories"][COMPUTE]["union_ns"] == 50
    assert cycle["categories"][NCCL]["union_ns"] == 30
    assert cycle["categories"][OFFLOAD]["union_ns"] == 40
    assert cycle["exposure"]["nccl_exposed_vs_compute"]["union_ns"] == 20
    assert cycle["exposure"]["offload_exposed_vs_compute"]["union_ns"] == 20
    assert cycle["exposure"]["movement_exposed_vs_compute"]["union_ns"] == 30
    assert cycle["exposure"]["nccl_unique_exposed"]["union_ns"] == 10
    assert cycle["exposure"]["offload_unique_exposed"]["union_ns"] == 10
    assert cycle["exposure"]["nccl_offload_shared_exposed"]["union_ns"] == 10
    assert cycle["categories"][OFFLOAD]["bytes"] == 200 * 1024 * 1024
    assert rank0["offload_runtime_api"]["cudaMemcpyAsync"]["events"] == 2
    assert rank0["dit_marker_census"]["total"] == 31
    assert rank0["dit_marker_census"]["roles"] == {
        "student": 24,
        "fake": 6,
        "teacher": 1,
    }


def test_world16_aggregate_uses_exact_rank_markers(tmp_path) -> None:
    node0 = tmp_path / "node0.sqlite"
    node1 = tmp_path / "node1.sqlite"
    _make_node_sqlite(node0, range(0, 8))
    _make_node_sqlite(node1, range(8, 16))
    report = analyze_minimax_world16(
        (node0, node1),
        mode="native",
        offload_min_bytes=64 * 1024 * 1024,
    )
    assert report["rank_count"] == 16
    cycle = report["cycle"]
    assert cycle["rank_max_wall_seconds"] == pytest.approx(1000 / 1e9)
    assert cycle["categories"][COMPUTE]["pooled_percent"] == pytest.approx(5.0)
    assert cycle["categories"][NCCL]["pooled_percent"] == pytest.approx(3.0)
    assert cycle["categories"][OFFLOAD]["pooled_percent"] == pytest.approx(4.0)
    assert cycle["exposure"]["nccl_exposed_vs_compute"]["pooled_percent"] == pytest.approx(2.0)
    assert cycle["exposure"]["offload_exposed_vs_compute"]["pooled_percent"] == pytest.approx(2.0)
    assert cycle["exposure"]["movement_exposed_vs_compute"]["pooled_percent"] == pytest.approx(3.0)
    assert cycle["categories"][OFFLOAD]["bytes"] == 16 * 200 * 1024 * 1024
    assert list(report["stages"]) == ["student/0", "fake/0", "fake/1", "fake/2", "fake/3", "fake/4"]
