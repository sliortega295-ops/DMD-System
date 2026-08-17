from __future__ import annotations

import sqlite3

from distillschedule.profiling.nsight_gpu_metrics import (
    aggregate_window_metrics,
    load_gpu_metric_series,
    summarize_device_window,
    summarize_series_window,
)


def _db() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "create table TARGET_INFO_GPU_METRICS("
        "typeId integer not null, sourceId integer not null, typeName text not null, "
        "metricId integer not null, metricName text not null)"
    )
    connection.execute(
        "create table GPU_METRICS("
        "timestamp integer not null, typeId integer not null, metricId integer not null, "
        "value integer not null)"
    )
    return connection


def test_time_weighted_metric_window() -> None:
    summary = summarize_series_window(
        [(0, 10.0), (10, 20.0), (20, 30.0), (30, 40.0)],
        start_ns=5,
        end_ns=25,
    )
    assert summary["status"] == "PASS"
    assert summary["coverage_fraction"] == 1.0
    assert summary["time_weighted_mean_pct"] == 20.0


def test_gpu_metric_loader_maps_typeid_low_byte_to_device() -> None:
    connection = _db()
    metric_names = [
        (1, "SM Active"),
        (2, "SM Issue"),
        (3, "Tensor Active"),
        (4, "DRAM Read Throughput"),
        (5, "DRAM Write Throughput"),
    ]
    for gpu_id in (0, 1):
        type_id = 0x100 | gpu_id
        for metric_id, name in metric_names:
            connection.execute(
                "insert into TARGET_INFO_GPU_METRICS values(?,?,?,?,?)",
                (type_id, 7, "gpu-metrics", metric_id, name),
            )
            for timestamp, value in ((0, 10 + gpu_id), (10, 20 + gpu_id), (20, 30 + gpu_id)):
                connection.execute(
                    "insert into GPU_METRICS values(?,?,?,?)",
                    (timestamp, type_id, metric_id, value),
                )
    loaded = load_gpu_metric_series(connection)
    assert loaded["status"] == "PASS"
    assert loaded["selected_source_names"]["sm_active_percent"] == "SM Active"
    assert loaded["selected_source_names"]["tensor_active_percent"] == "Tensor Active"
    assert "gpu1/sm_active_percent" in loaded["series"]

    window = summarize_device_window(loaded, gpu_id=1, start_ns=0, end_ns=20)
    assert window["status"] == "PASS"
    assert window["metrics"]["sm_active_percent"]["time_weighted_mean_pct"] == 16.0
    assert window["metrics"]["dram_read_percent"]["source_metric_name"] == "DRAM Read Throughput"


def test_gpu_metric_missing_tables_fail_closed_without_fabricating_utilization() -> None:
    connection = sqlite3.connect(":memory:")
    loaded = load_gpu_metric_series(connection)
    assert loaded["status"] == "NOT_CAPTURED"
    window = summarize_device_window(loaded, gpu_id=0, start_ns=0, end_ns=100)
    assert window["status"] == "NOT_CAPTURED"
    assert window["metrics"] == {}


def test_aggregate_window_metrics_uses_sampled_time_weighting() -> None:
    windows = [
        {
            "gpu_metrics": {
                "status": "PASS",
                "metrics": {
                    "sm_active_percent": {
                        "source_metric_name": "SM Active",
                        "covered_ns": 10,
                        "time_weighted_mean_pct": 20.0,
                    }
                },
            }
        },
        {
            "gpu_metrics": {
                "status": "PASS",
                "metrics": {
                    "sm_active_percent": {
                        "source_metric_name": "SM Active",
                        "covered_ns": 30,
                        "time_weighted_mean_pct": 60.0,
                    }
                },
            }
        },
    ]
    aggregate = aggregate_window_metrics(windows)
    # Only SM Active is present, so the aggregate is intentionally PARTIAL.
    assert aggregate["status"] == "PARTIAL"
    assert aggregate["metrics"]["sm_active_percent"]["time_weighted_mean_pct"] == 50.0
