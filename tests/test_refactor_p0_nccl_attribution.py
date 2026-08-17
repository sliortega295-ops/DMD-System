from __future__ import annotations

import sqlite3

from distillschedule.profiling.p0_nccl_attribution import (
    analyze_p0_nccl_sqlite,
    classify_nccl_kernel,
)


def test_nccl_kernel_name_classification() -> None:
    assert classify_nccl_kernel("ncclDevKernel_AllGather_RING_LL") == "all_gather"
    assert classify_nccl_kernel("ncclDevKernel_ReduceScatter_RING_LL") == "reduce_scatter"
    assert classify_nccl_kernel("ncclKernel_AllReduce_TREE") == "all_reduce"
    assert classify_nccl_kernel("ncclSomethingUnknown") == "other_nccl"


def test_p0_nccl_sqlite_attributes_rollout_operations(tmp_path) -> None:
    path = tmp_path / "capture.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, demangledName INTEGER)"
        )
        connection.execute("CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text INTEGER)")
        strings = {
            1: "STATIC_FSDP2_P0_FULL5/no-gan/world16-fsdp-rollout-b5-realpair-0",
            2: "FSDP_BATCHED_GENERATOR_ROLLOUT/batch0/B5",
            3: "ncclDevKernel_AllGather_RING_LL",
            4: "ncclDevKernel_ReduceScatter_RING_LL",
            5: "ncclKernel_AllReduce_TREE",
        }
        connection.executemany("INSERT INTO StringIds(id, value) VALUES (?, ?)", strings.items())
        connection.execute("INSERT INTO NVTX_EVENTS VALUES (0, 1000, 1)")
        connection.execute("INSERT INTO NVTX_EVENTS VALUES (100, 900, 2)")
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (120, 220, 3)")
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (300, 380, 4)")
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (500, 560, 5)")
        connection.commit()

    result = analyze_p0_nccl_sqlite(
        path,
        variant="no-gan",
        policy_id="world16-fsdp-rollout-b5-realpair-0",
    )
    assert len(result["rollout_windows"]) == 1
    by_operation = result["rollout_windows"][0]["nccl"]["by_operation"]
    assert by_operation["all_gather"]["kernel_events"] == 1
    assert by_operation["reduce_scatter"]["kernel_events"] == 1
    assert by_operation["all_reduce"]["kernel_events"] == 1
    assert by_operation["all_gather"]["payload_bytes"] is None
