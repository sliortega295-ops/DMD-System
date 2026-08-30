from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.adapters.livewan import LiveWanSFDMDAdapter
from distillschedule.core.config import load_plan
from distillschedule.core.contracts import ContractViolation
from distillschedule.workloads.livewan_crossscale import (
    PIPELINE_8G8S_LAYOUT,
    balanced_slots,
    compile_trace_identities,
    full5_routes,
    speedup_receipt,
    trace_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_fsdp2.json"
PIPELINE = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8.json"


@pytest.fixture(autouse=True)
def plan_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    values = {
        "LIVEWAN_STUDENT_CHECKPOINT": tmp_path / "student",
        "LIVEWAN_CRITIC_CHECKPOINT": tmp_path / "critic",
        "LIVEWAN_TEACHER_CHECKPOINT": tmp_path / "teacher",
        "LIVEWAN_OPERATION_TRACE": tmp_path / "trace.json",
        "LIVEWAN_SYSTEMS_BOOTSTRAP": tmp_path / "bootstrap.json",
        "DISTILLSCHEDULE_OUTPUT_DIR": tmp_path / "output",
        "DISTILLSCHEDULE_PYTHON": "python3",
        "DMD_SYSTEM_ROOT": ROOT,
        "DISTILLSCHEDULE_NODE_RANK": "0",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, str(value))


def test_livewan_plans_compile_and_freeze_only_placement() -> None:
    baseline = load_plan(BASELINE).plan
    pipeline = load_plan(PIPELINE).plan
    adapter = LiveWanSFDMDAdapter()
    adapter.validate(baseline)
    adapter.validate(pipeline)
    assert baseline.workload.to_dict() == pipeline.workload.to_dict()
    assert baseline.hardware.to_dict() == pipeline.hardware.to_dict()
    assert baseline.workload.teacher is not None
    assert "t-wan2.1-14b-bf16" in baseline.profile_key
    assert baseline.metadata["execution_mode"] == "baseline"
    assert pipeline.metadata["execution_mode"] == "pipeline"
    assert baseline.rank_groups[0].ranks == tuple(range(16))
    assert pipeline.rank_groups[0].ranks == tuple(range(8))


def test_livewan_adapter_fails_closed_on_semantic_drift() -> None:
    plan = load_plan(BASELINE).plan
    changed_workload = replace(
        plan.workload,
        schedule=replace(plan.workload.schedule, generator_update_every=4),
    )
    with pytest.raises(ContractViolation, match="G1/C5"):
        LiveWanSFDMDAdapter().validate(replace(plan, workload=changed_workload))


def test_pipeline_8x8_routes_form_two_b1_waves() -> None:
    PIPELINE_8G8S_LAYOUT.validate()
    routes = full5_routes()
    assert len(routes) == 80
    for step in range(5):
        rows = [route for route in routes if route.logical_step == step]
        assert sorted(route.logical_slot for route in rows) == list(range(16))
        assert {route.wave for route in rows} == {0, 1}
        assert all(route.score_rank == route.generator_rank + 8 for route in rows)
    assert balanced_slots(tuple(range(8))) == {
        rank: (rank, rank + 8) for rank in range(8)
    }


def test_operation_trace_identity_is_stable_and_rank_qualified() -> None:
    rows = compile_trace_identities(sample_count=96, seed=20260830)
    assert len(rows) == 80
    assert len({(row.logical_step, row.logical_slot) for row in rows}) == 80
    assert len({row.operation_key for row in rows}) == 80
    first = trace_sha256(rows)
    assert first == trace_sha256(compile_trace_identities(sample_count=96, seed=20260830))
    assert first != trace_sha256(compile_trace_identities(sample_count=96, seed=20260831))


def test_speedup_target_is_exactly_thirty_percent() -> None:
    passing = speedup_receipt(baseline_seconds=130.0, pipeline_seconds=100.0)
    failing = speedup_receipt(baseline_seconds=130.0, pipeline_seconds=100.01)
    assert passing["status"] == "PASS"
    assert passing["target_pipeline_seconds"] == 100.0
    assert failing["status"] == "BELOW_TARGET"


def test_submodules_are_pinned_and_source_prepare_is_non_installing() -> None:
    assert (ROOT / "third_party/livewan/.git").is_file()
    assert (ROOT / "third_party/wan21/.git").is_file()
    modules = (ROOT / ".gitmodules").read_text()
    assert "third_party/livewan" in modules
    assert "third_party/wan21" in modules
    assert subprocess.check_output(
        ["git", "-C", str(ROOT / "third_party/livewan"), "rev-parse", "HEAD"], text=True
    ).strip() == "1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5"
    assert subprocess.check_output(
        ["git", "-C", str(ROOT / "third_party/wan21"), "rev-parse", "HEAD"], text=True
    ).strip() == "9737cba9c1c3c4d04b33fcad41c111989865d315"
    script = (ROOT / "scripts/distillschedule/prepare_livewan_crossscale_source.sh").read_text()
    assert "pip install" not in script
    assert "archive --format=tar" in script
    assert "1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5" in script
    assert "9737cba9c1c3c4d04b33fcad41c111989865d315" in script
    overlay = (
        ROOT / "scripts/distillschedule/prepare_livewan_crossscale_python_overlay.sh"
    ).read_text()
    assert "easydict==1.13" in overlay
    assert "--target" in overlay
    assert "pip install" not in overlay.replace('"$(dirname "${PYTHON}")/pip" install', "")


def test_controller_keeps_remote_evidence_outside_source_checkout() -> None:
    controller = (
        ROOT / "scripts/distillschedule/run_livewan_crossscale_world16.py"
    ).read_text()
    assert '"--remote-output-root", required=True' in controller
    assert 'args.remote_output_root / args.run_id' in controller
    assert 'args.remote_root / "artifacts/runs"' not in controller


def test_cold_loader_avoids_redundant_full_fp32_copies() -> None:
    worker = (
        ROOT / "src/distillschedule/runtime/livewan_crossscale_worker.py"
    ).read_text()
    assert 'mmap=True' in worker
    assert '"complete_stage1_direct_mmap"' in worker
    assert '_wan_model(WanModel, size="14B", config=config).to(torch.bfloat16)' in worker
    assert '_wan_model(WanModel, size="1.3B", config=config).to(torch.bfloat16)' in worker


def test_step0_p2p_posts_gradient_receive_before_second_wave_send() -> None:
    """Default-PG P2P peers must enqueue matching operations in one order."""
    worker = (
        ROOT / "src/distillschedule/runtime/livewan_crossscale_worker.py"
    ).read_text()
    pipeline = worker.split("def _run_pipeline_8x8(", 1)[1]
    generator_path = pipeline.split("for step in range(FULL5_STEPS):", 1)[1].split(
        "else:\n            bundle.critic_optimizer.zero_grad", 1
    )[0]
    step0 = generator_path.split("if step == 0:", 1)[1]
    assert step0.index("dist.recv(grad0, src=peer)") < step0.index(
        "sends = _send_pair(destination=peer, prepared=second, asynchronous=True)"
    )


def test_trace_builder_requires_80_distinct_source_samples(tmp_path: Path) -> None:
    data = tmp_path / "latents"
    data.mkdir()
    prompts = tmp_path / "prompts.pt"
    prompts.write_bytes(b"prompt-bundle")
    for index in range(79):
        (data / f"sample_{index:03d}.pt").write_bytes(f"sample-{index}".encode())
    script = ROOT / "scripts/distillschedule/build_livewan_crossscale_trace.py"
    failed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data",
            str(data),
            "--prompts",
            str(prompts),
            "--output",
            str(tmp_path / "trace.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "needs 80 distinct latent samples" in failed.stderr

    (data / "sample_079.pt").write_bytes(b"sample-79")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--data",
            str(data),
            "--prompts",
            str(prompts),
            "--output",
            str(tmp_path / "trace.json"),
        ],
        check=True,
    )
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert trace["unique_samples_consumed"] == 80
    assert len({row["dataset_index"] for row in trace["rows"]}) == 80


def test_launcher_fails_before_model_load_on_world16_and_p2p_drift() -> None:
    launcher = (
        ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh"
    ).read_text()
    assert "probe_world16_topology.py" in launcher
    assert "probe_livewan_crossscale_transport.py" in launcher
    assert 'master-port="$((MASTER_PORT + 1))"' in launcher
    assert 'master-port="$((MASTER_PORT + 2))"' in launcher
    probe = (
        ROOT / "scripts/distillschedule/probe_livewan_crossscale_transport.py"
    ).read_text()
    assert "default_world_initialized_collectively" in probe
    assert "range(2)" in probe
