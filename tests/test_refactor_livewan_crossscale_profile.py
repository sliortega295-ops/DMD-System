from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
HEAD = "1" * 40


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _summary(mode: str) -> dict:
    return {
        "status": "PASS",
        "run_kind": "correctness",
        "execution_mode": mode,
        "head": HEAD,
        "identity": ["trace", "bootstrap", "source"],
        "call_inventory": {
            "student/nograd/plain": 1,
            "student/grad/checkpoint": 1,
            "teacher/nograd/plain": 1,
            "critic/nograd/plain": 1,
            "critic/grad/checkpoint": 1,
        },
    }


def test_profile_gate_is_diagnostic_only_and_preserves_failed_pair(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "baseline.json", _summary("baseline"))
    pipeline = _write(tmp_path / "pipeline.json", _summary("pipeline"))
    pair = _write(
        tmp_path / "pair.json",
        {
            "status": "FAIL",
            "claimable_for_timing": False,
            "heads": {"baseline": HEAD, "pipeline": HEAD},
            "rollout": {"failure_count": 63},
        },
    )
    output = tmp_path / "profile_gate.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/distillschedule/build_livewan_crossscale_profile_gate.py"),
            "--baseline-summary", str(baseline),
            "--pipeline-summary", str(pipeline),
            "--failed-pair-receipt", str(pair),
            "--output", str(output),
        ],
        check=True,
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["status"] == "PASS_DIAGNOSTIC_ONLY"
    assert value["profile_authorized"] is True
    assert value["formal_timing_authorized"] is False
    assert value["heads"] == {
        "baseline": HEAD,
        "pipeline": HEAD,
        "profile": HEAD,
    }
    assert value["profile_source"] == {
        "diff_paths": [],
        "instrumentation_only": True,
        "profile_head": HEAD,
        "verification": "exact_correctness_head",
        "workload_head": HEAD,
    }
    assert value["failed_pair_receipt"]["rollout_failure_count"] == 63
    assert value["failed_pair_receipt"]["sha256"] == hashlib.sha256(
        pair.read_bytes()
    ).hexdigest()


def test_profile_launcher_is_node_outer_and_fail_closed() -> None:
    launcher = (
        ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh"
    ).read_text(encoding="utf-8")
    controller = (
        ROOT / "scripts/distillschedule/run_livewan_crossscale_world16.py"
    ).read_text(encoding="utf-8")
    assert 'choices=("validate", "correctness", "timing", "profile")' in controller
    assert "profile requires --nsys-bin" in controller
    assert "profile_authorized" in launcher
    assert "formal_timing_authorized" in launcher
    assert 'source.get("profile_head") != expected_head' in launcher
    assert 'source.get("instrumentation_only") is not True' in launcher
    assert "claimable_for_timing" in launcher
    assert '"${NSYS_BIN}" profile' in launcher
    assert "--capture-range=cudaProfilerApi" in launcher
    assert "--trace-fork-before-exec=true" in launcher
    assert '"${workload[@]}"' in launcher
    assert "missing_nsys_report" in launcher
    assert "missing_nsys_sqlite" in launcher
    assert '--format=csv "${PROFILE_SQLITE}"' in launcher
    assert "nsys_profile_validation_failed" in launcher
    assert "incomplete_nvtx_rank_ranges" in launcher
    assert "incomplete_kernel_device_coverage" in launcher
    gate = (
        ROOT / "scripts/distillschedule/build_livewan_crossscale_profile_gate.py"
    ).read_text(encoding="utf-8")
    assert '"scripts/distillschedule/analyze_livewan_crossscale_nsys.py"' in gate


def test_profile_markers_and_capture_are_absent_from_formal_timing_path() -> None:
    worker = (
        ROOT / "src/distillschedule/runtime/livewan_crossscale_worker.py"
    ).read_text(encoding="utf-8")
    helper = worker.split("def _profile_range", 1)[1].split(
        "def _canonical_without_sha", 1
    )[0]
    assert 'if run_kind == "profile"' in helper
    assert "torch.cuda.nvtx.range(label)" in helper
    assert "cudaProfilerStart" in worker
    assert "cudaProfilerStop" in worker
    assert "if is_profile and local_rank == 0" not in worker
    assert worker.count("torch.cuda.cudart().cudaProfilerStart()") == 1
    assert worker.count("torch.cuda.cudart().cudaProfilerStop()") == 1
    assert "time.sleep(10.0)" in worker
    assert worker.index("time.sleep(10.0)") < worker.index("start = time.perf_counter()")
    assert worker.index("dist.destroy_process_group()") > worker.index(
        "_write_rank_receipt(output_dir=output_dir, rank=rank, value=receipt)"
    )
    assert worker.index("os._exit(0)") > worker.index("cudaProfilerStop")
    assert worker.index("os._exit(0)") < worker.index("dist.destroy_process_group()")
    for label in (
        "rollout_wave_0",
        "rollout_wave_1",
        "recv_gradient_0",
        "dmd_wave_0",
        "critic_backward_wave_0",
        "critic_commit",
    ):
        assert label in worker


def test_profile_analyzer_streams_each_large_activity_table_once() -> None:
    analyzer = (
        ROOT / "scripts/distillschedule/analyze_livewan_crossscale_nsys.py"
    ).read_text(encoding="utf-8")
    # Nsight activity tables do not carry useful process/time indices in this
    # export.  A query per rank performs eight full scans of an 819 MB SQLite.
    # The production analyzer must instead scan each table once and dispatch
    # rows through the process-qualified NVTX rank map.
    assert analyzer.count("FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE") == 1
    assert analyzer.count("FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE") == 1
    assert "state_by_pid" in analyzer
    assert "ORDER BY globalPid,start" not in analyzer
    assert '"livewan/pipeline_runahead/g/"' in analyzer
    assert '"livewan/pipeline_runahead/score/"' in analyzer
    assert "def _phase_sort_key" in analyzer


def test_profile_analyzer_attributes_collectives_inside_each_phase() -> None:
    path = ROOT / "scripts/distillschedule/analyze_livewan_crossscale_nsys.py"
    spec = importlib.util.spec_from_file_location("livewan_nsys_analyzer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = module._collective_metric(
        (10, 30),
        {
            kind: ([(5, 20), (25, 40)] if kind == "all_gather" else [])
            for kind in module.COLLECTIVES
        },
        [(15, 28)],
    )
    assert value["all_gather"] == {
        "events": 2,
        "union_ns": 15,
        "hidden_by_compute_ns": 8,
        "exposed_vs_compute_ns": 7,
    }
