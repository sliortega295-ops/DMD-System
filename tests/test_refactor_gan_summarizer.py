from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/distillschedule/summarize_routea_gan_world16.py"
SPEC = importlib.util.spec_from_file_location("gan_world16_summarizer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _version_events() -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {
            "phase": "parameter_versions_initialized",
            "generator_version": 200,
            "critic_version": 1000,
        },
        {
            "phase": "generator_optimizer_step",
            "global_step": 1000,
            "generator_version": 201,
        },
    ]
    for step, version in zip(range(1000, 1005), range(1001, 1006)):
        events.append(
            {
                "phase": "critic_optimizer_step",
                "global_step": step,
                "critic_version": version,
            }
        )
    events.extend(
        [
            {
                "phase": "ema_step",
                "global_step": 1000,
                "optimization_step_before": 800,
                "ema": {"optimization_step": 801},
                "predicate": {
                    "before_ema_start": False,
                    "not_train_dmd": False,
                    "generator_update": True,
                },
            },
            {"phase": "training_complete_skip_final_save", "global_step": 1005},
        ]
    )
    return events


def test_real_runtime_version_receipts_form_exact_trajectory() -> None:
    result = MODULE.validate_versions(_version_events(), rank=0)
    assert result["status"] == "OBSERVED_EXACT"
    assert result["initial"] == {"generator": 200, "critic": 1000, "ema": 800}
    assert result["final"] == {"generator": 201, "critic": 1005, "ema": 801}
    assert result["final_evidence"] == "last_component_update_receipts"


def test_operation_trace_authority_is_bound_by_exact_hash_and_rank() -> None:
    expected = {"trace_hash": "trace-0", "source_authority_aliases": ["rank0"]}
    event = {
        "phase": "operation_trace_input_validated",
        "trace_hash": "trace-0",
        "rank": 0,
        "source_authority": "operation_trace",
    }
    assert MODULE.validate_trace([event], expected, rank=0)["source_authority"] == "operation_trace"


def test_resource_summary_accepts_procfs_scientific_notation(tmp_path: Path) -> None:
    path = tmp_path / "resource.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["unix_time", "cgroup_bytes", "host_used_bytes", "gpu_used_mib", "gpu_util_percent"]
        )
        writer.writerow(["1.0", "600195186688", "5.89132e+11", "1:2", "0:0"])
    result = MODULE.resource_summary(path)
    assert result["status"] == "MEASURED"
    assert result["cgroup_peak_bytes"] == 600195186688
    assert result["host_used_peak_bytes"] == 589132000000
