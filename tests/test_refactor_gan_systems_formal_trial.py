from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "distillschedule"
    / "run_gan_systems_formal_trial.py"
)
SPEC = importlib.util.spec_from_file_location("run_gan_systems_formal_trial", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_workload_digest_requires_explicit_sha256(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"workload_contract_digest": "a" * 64}))
    assert MODULE._workload_digest(contract) == "a" * 64

    contract.write_text(json.dumps({"workload_contract_digest": "short"}))
    try:
        MODULE._workload_digest(contract)
    except RuntimeError as exc:
        assert "lacks a SHA256" in str(exc)
    else:
        raise AssertionError("malformed workload digest was accepted")


def test_pipeline_worker_freezes_exact_formal_and_gas2_contract(tmp_path: Path) -> None:
    args = SimpleNamespace(
        master_addr="10.0.0.1",
        master_port=30001,
        timeout_seconds=20000,
        run_nonce=123,
        pipeline_remote_root=Path("/remote/dmd"),
        pipeline_runtime_root=Path("/remote/runtime"),
        config=Path("/assets/config.yaml"),
        target_package=Path("/assets/target"),
        trace_bundle=Path("/assets/trace"),
        generator_root=Path("/assets/generator"),
        score_root=Path("/assets/score"),
        negative_prompt=Path("/assets/negative.pt"),
        pipeline_mmap=Path("/assets/mmap"),
    )
    script = MODULE._pipeline_worker_script(args, 1, tmp_path / "run")
    assert "--nnodes=2" in script
    assert "--nproc-per-node=8" in script
    assert "--node-rank=1" in script
    assert "DISTILLSCHEDULE_GAN_EXACT_TIMING=1" in script
    assert "DISTILLSCHEDULE_GAN_NSYS_PROFILE=0" in script
    assert "--detailed-events" in script
    assert "distillschedule_gpu_node.lock" in script


def test_pipeline_worker_profiles_same_command_but_disables_formal_timing(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        master_addr="10.0.0.1",
        master_port=30001,
        timeout_seconds=20000,
        run_nonce=123,
        profile_nsys=True,
        pipeline_remote_root=Path("/remote/dmd"),
        pipeline_runtime_root=Path("/remote/runtime"),
        config=Path("/assets/config.yaml"),
        target_package=Path("/assets/target"),
        trace_bundle=Path("/assets/trace"),
        generator_root=Path("/assets/generator"),
        score_root=Path("/assets/score"),
        negative_prompt=Path("/assets/negative.pt"),
        pipeline_mmap=Path("/assets/mmap"),
    )
    script = MODULE._pipeline_worker_script(args, 0, tmp_path / "run")
    assert "/remote/dmd/scripts/distillschedule/gan_launch_shim.sh" in script
    assert "DISTILLSCHEDULE_GAN_NSYS_PROFILE=1" in script
    assert "DISTILLSCHEDULE_GAN_EXACT_TIMING=0" in script
    assert "DISTILLSCHEDULE_NODE_RANK=0" in script
    assert "DISTILLSCHEDULE_OUTPUT_DIR=" in script
    assert "DISTILLSCHEDULE_PYTHON=" in script


def test_pipeline_capacity_candidate_changes_only_streamed_tail_contract(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        master_addr="10.0.0.1",
        master_port=30001,
        timeout_seconds=20000,
        run_nonce=123,
        profile_nsys=False,
        capacity_search=True,
        streamed_block_count=4,
        pipeline_remote_root=Path("/remote/dmd"),
        pipeline_runtime_root=Path("/remote/runtime"),
        config=Path("/assets/config.yaml"),
        target_package=Path("/assets/target"),
        trace_bundle=Path("/assets/trace"),
        generator_root=Path("/assets/generator"),
        score_root=Path("/assets/score"),
        negative_prompt=Path("/assets/negative.pt"),
        pipeline_mmap=Path("/assets/mmap"),
    )
    script = MODULE._pipeline_worker_script(args, 0, tmp_path / "run")
    assert "DISTILLSCHEDULE_GAN_CAPACITY_SEARCH=1" in script
    assert "DISTILLSCHEDULE_GAN_GENERATOR_STREAMED_BLOCK_COUNT=4" in script
    assert "DISTILLSCHEDULE_GAN_GENERATOR_STREAMED_BLOCK_INDICES=36,37,38,39" in script
    assert "DISTILLSCHEDULE_GAN_SCORE_STREAMED_BLOCK_COUNT=4" in script
    assert "DISTILLSCHEDULE_GAN_SCORE_STREAMED_BLOCK_INDICES=36,37,38,39" in script
    assert "DISTILLSCHEDULE_GAN_NSYS_PROFILE=0" in script


def test_pipeline_role_aware_phase_resident_candidate_is_fail_closed(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        master_addr="10.0.0.1",
        master_port=30001,
        timeout_seconds=20000,
        run_nonce=123,
        profile_nsys=False,
        capacity_search=True,
        streamed_block_count=40,
        generator_streamed_block_count=40,
        score_streamed_block_count=4,
        generator_critic_phase_resident=True,
        pipeline_remote_root=Path("/remote/dmd"),
        pipeline_runtime_root=Path("/remote/runtime"),
        config=Path("/assets/config.yaml"),
        target_package=Path("/assets/target"),
        trace_bundle=Path("/assets/trace"),
        generator_root=Path("/assets/generator"),
        score_root=Path("/assets/score"),
        negative_prompt=Path("/assets/negative.pt"),
        pipeline_mmap=Path("/assets/mmap"),
    )
    script = MODULE._pipeline_worker_script(args, 0, tmp_path / "run")
    assert "DISTILLSCHEDULE_GAN_GENERATOR_STREAMED_BLOCK_COUNT=40" in script
    assert "DISTILLSCHEDULE_GAN_SCORE_STREAMED_BLOCK_COUNT=4" in script
    assert "DISTILLSCHEDULE_GAN_SCORE_STREAMED_BLOCK_INDICES=36,37,38,39" in script
    assert "DISTILLSCHEDULE_GAN_GENERATOR_CRITIC_PHASE_RESIDENT=1" in script


def test_formal_run_id_contract_accepts_artifact_convention() -> None:
    pattern = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
    assert pattern.fullmatch("gan_pipeline_formal_gate_20260811_001")
    assert pattern.fullmatch("pair0-baseline.001")
    assert pattern.fullmatch("../escape") is None


def test_prepare_result_path_creates_parent_but_preserves_immutability(
    tmp_path: Path,
) -> None:
    result = tmp_path / "nested" / "result.json"
    MODULE._prepare_result_path(result)
    assert result.parent.is_dir()
    result.write_text("{}")
    try:
        MODULE._prepare_result_path(result)
    except RuntimeError as exc:
        assert "immutable result already exists" in str(exc)
    else:
        raise AssertionError("existing formal result was accepted")


def test_pipeline_failure_classifier_requires_explicit_capacity_evidence(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "launcher"
    results = tmp_path / "results"
    launcher.mkdir()
    results.mkdir()
    (launcher / "node_0.log").write_text("NCCL watchdog timeout\n")
    (launcher / "node_0.memory.events.before").write_text(
        "oom 7\noom_kill 3\n"
    )
    (launcher / "node_0.memory.events.after").write_text(
        "oom 7\noom_kill 3\n"
    )
    first = MODULE._classify_pipeline_failure_evidence(tmp_path)
    assert first["actual_capacity_failure"] is False

    (results / "rank_000.json").write_text(
        json.dumps({"status": "FAIL", "error": "CUDA out of memory"})
    )
    second = MODULE._classify_pipeline_failure_evidence(tmp_path)
    assert second["actual_capacity_failure"] is True
    assert second["cuda_signatures"]["cuda_out_of_memory"] is True


def test_pipeline_failure_classifier_accepts_positive_cgroup_delta(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "launcher"
    launcher.mkdir()
    (launcher / "node_1.memory.events.before").write_text(
        "oom 4\noom_kill 1\n"
    )
    (launcher / "node_1.memory.events.after").write_text(
        "oom 5\noom_kill 2\n"
    )
    value = MODULE._classify_pipeline_failure_evidence(tmp_path)
    assert value["actual_capacity_failure"] is True
    assert value["cgroup"][1]["oom_delta"] == 1
    assert value["cgroup"][1]["oom_kill_delta"] == 1


def test_formal_wrapper_pins_baseline_execution_fix_head(tmp_path: Path) -> None:
    args = MODULE.parser().parse_args(
        [
            "--family",
            "baseline",
            "--run-id",
            "formal_001",
            "--result-path",
            str(tmp_path / "result.json"),
            "--workload-contract",
            str(tmp_path / "contract.json"),
        ]
    )
    assert args.baseline_expected_head == "99aa0a31cd1fd5aebcb1d72bd721ece5fd09b683"
