from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from distillschedule.evidence.trajectory_coarse import (
    AppendOnlyTrajectoryWriter,
    TrajectoryCycleReceipt,
)
from distillschedule.adapters.livewan.adapter import LiveWanSFDMDAdapter
from distillschedule.core.config import load_plan
from distillschedule.runtime.livewan_crossscale_worker import (
    RunEvidence,
    TRAJECTORY_ALLOWED_CYCLES,
    _apply_trajectory_contract,
    _continuous_pipeline_contract,
    _critic_b1_attribution_reducer_accumulation,
    _physical_critic_backward_calls_for_step,
    _run_pipeline_8x8,
    _run_pipeline_8x8_continuous,
    _trajectory_critic_batch_ablation_enabled,
)


ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / "configs/distillschedule/plans"
C24 = PLANS / "livewan_wan13b_14b_sf_dmd_c24_continuous_b16_score_modulation.json"
C24_B1 = PLANS / (
    "livewan_wan13b_14b_sf_dmd_c24_continuous_b16_score_modulation_"
    "b1_attribution.json"
)
B2_AST_SHA256 = "c4447e4a4a4e2009b8a49635c13cd8a29f4bea25e074bf50995f05496bba9880"


@pytest.fixture(autouse=True)
def plan_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in (
        "LIVEWAN_STUDENT_CHECKPOINT",
        "LIVEWAN_CRITIC_CHECKPOINT",
        "LIVEWAN_TEACHER_CHECKPOINT",
        "LIVEWAN_OPERATION_TRACE",
        "LIVEWAN_SYSTEMS_BOOTSTRAP",
        "DISTILLSCHEDULE_OUTPUT_DIR",
    ):
        monkeypatch.setenv(key, str(tmp_path / key.lower()))


def test_c24_b1_plan_changes_only_registered_critic_physical_execution() -> None:
    reference = load_plan(C24).plan
    candidate = load_plan(C24_B1).plan
    adapter = LiveWanSFDMDAdapter()
    adapter.validate(reference)
    adapter.validate(candidate)

    expected = reference.to_dict()
    observed = candidate.to_dict()
    expected["id"] = observed["id"]
    expected["metadata"]["critic_physical_batch_size"] = 1
    expected["metadata"]["trajectory_critic_batch_ablation"] = copy.deepcopy(
        observed["metadata"]["trajectory_critic_batch_ablation"]
    )
    expected["training_backends"]["score"]["options"][
        "critic_physical_batch_policy"
    ] = "two_b1_accumulated_single_commit_attribution"
    assert observed == expected
    assert _trajectory_critic_batch_ablation_enabled(
        observed, mode="pipeline"
    ) is True
    assert _trajectory_critic_batch_ablation_enabled(
        reference.to_dict(), mode="pipeline"
    ) is False


def test_two_cycle_contract_and_b1_accumulation_are_exact() -> None:
    plan = load_plan(C24_B1).plan.to_dict()
    assert 2 in TRAJECTORY_ALLOWED_CYCLES
    _apply_trajectory_contract(plan, cycles=2, mode="pipeline")
    contract = _continuous_pipeline_contract(plan, mode="pipeline")
    assert contract is not None
    assert contract["cycles"] == 2
    assert contract["critic_shadow_sync_versions"] == [5]
    assert _critic_b1_attribution_reducer_accumulation(0) == (0, 2)
    assert _critic_b1_attribution_reducer_accumulation(1) == (1, 2)
    with pytest.raises(RuntimeError, match="wave must be 0 or 1"):
        _critic_b1_attribution_reducer_accumulation(2)


def test_physical_backward_counter_is_not_logical_sample_inventory() -> None:
    assert _physical_critic_backward_calls_for_step(
        critic_b2_only=True, split_step1=False
    ) == 1
    assert _physical_critic_backward_calls_for_step(
        critic_b2_only=False, split_step1=False
    ) == 2
    assert _physical_critic_backward_calls_for_step(
        critic_b2_only=True, split_step1=True
    ) == 2

    evidence = RunEvidence()
    evidence.add_physical_critic_backward_calls(1)
    evidence.add_physical_critic_backward_calls(2)
    assert evidence.physical_critic_backward_calls == 3
    with pytest.raises(RuntimeError, match="must be positive"):
        evidence.add_physical_critic_backward_calls(0)

    continuous_source = inspect.getsource(_run_pipeline_8x8_continuous)
    assert "physical_critic_backward_before" in continuous_source
    assert (
        'inventory_after.get("critic/grad/checkpoint"' not in continuous_source
    )


def test_c24_production_b2_branch_ast_is_unchanged() -> None:
    tree = ast.parse(inspect.getsource(_run_pipeline_8x8))
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "critic_b2_only"
    )
    body = ast.Module(body=branch.body, type_ignores=[])
    digest = hashlib.sha256(
        # Python 3.13 omits empty lists by default; preserve the original
        # serialized AST and the existing frozen hash, not a new acceptance hash.
        ast.dump(body, include_attributes=False,
                 **({"show_empty": True} if sys.version_info >= (3, 13) else {})).encode()
    ).hexdigest()
    assert digest == B2_AST_SHA256
    calls = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "critic_backward_for_batch"
    ]
    assert len(calls) == 1
    keywords = {item.arg: item.value for item in calls[0].keywords}
    assert ast.literal_eval(keywords["loss_scale"]) == 1.0
    assert "reducer_accumulation" not in keywords


def test_trajectory_summarizer_accepts_two_cycle_b1_and_b2_receipts(
    tmp_path: Path,
) -> None:
    head = "a" * 40
    components = {
        name: {"coordinate_count": 1, "coordinates": [{"value_fp32": 1.0}]}
        for name in (
            "generator_trainable",
            "generator_adam",
            "critic_trainable",
            "critic_adam",
        )
    }
    anchors: dict[int, tuple[dict[str, object], str]] = {}
    for cycle in (1, 2):
        body = {
            "schema_version": "distillschedule.livewan_trajectory_state_anchor.v1",
            "cycle": cycle,
            "writer_rank": 0,
            "topology": {},
            "components": components,
            "comparison_scope": "test",
        }
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        anchors[cycle] = ({**body, "anchor_sha256": digest}, digest)

    cases = (
        ("baseline", "baseline", "b1", "1"),
        (
            "pipeline_b2",
            "pipeline",
            "c24-continuous-b16-score-modulation",
            "2",
        ),
        (
            "pipeline_b1",
            "pipeline",
            "c24-continuous-b16-score-modulation-b1-attribution",
            "1",
        ),
    )
    for label, mode, variant, critic_batch in cases:
        run_root = tmp_path / label
        for node in range(2):
            manifest = run_root / "launcher" / f"node_{node}" / "launch_manifest.txt"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                f"run_id={label}\nhead={head}\nexecution_mode={mode}\n"
                f"pipeline_variant={variant}\n"
                "trajectory_cycles=2\n"
                "trajectory_seed=20260903\n"
                f"trajectory_arm_id={label}_arm\nclaim_boundary=DIAGNOSTIC_ONLY\n"
            )
        for cycle, (anchor, _digest) in anchors.items():
            anchor_path = (
                run_root / "trajectory/anchors" / f"cycle_{cycle:02d}_global.json"
            )
            anchor_path.parent.mkdir(parents=True, exist_ok=True)
            anchor_path.write_text(json.dumps(anchor))

        for rank in range(16):
            writer = AppendOnlyTrajectoryWriter(
                run_root / "trajectory" / f"rank_{rank:03d}.jsonl",
                run_id=label,
                rank=rank,
            )
            for cycle in (1, 2):
                samples = tuple(
                    f"cycle={cycle - 1}/step={step}/slot={rank}"
                    for step in range(5)
                )
                state = {name: f"{name}_sha_{cycle}" for name in components}
                state["anchor"] = anchors[cycle][1]
                physical_calls = (
                    "5"
                    if mode == "baseline"
                    else ("0" if rank < 8 else ("10" if critic_batch == "1" else "5"))
                )
                physical_commits = (
                    "5" if mode == "baseline" else ("0" if rank < 8 else "5")
                )
                writer.append(
                    TrajectoryCycleReceipt(
                        run_id=label,
                        workload="livewan_wan13b_teacher14b_sfdmd",
                        variant=f"{label}_arm",
                        rank=rank,
                        world_size=16,
                        seed=20260903,
                        cycle=cycle,
                        expected_total_cycles=2,
                        logical_step=(cycle - 1) * 5,
                        matched_contract={"contract": "same"},
                        implementation={
                            "execution_mode": mode,
                            "critic_physical_batch_size": critic_batch,
                            "critic_physical_backward_calls": physical_calls,
                            "critic_optimizer_commits": physical_commits,
                        },
                        sample_identities=samples,
                        rng_identities={sample: "b" * 64 for sample in samples},
                        versions={
                            "generator": cycle,
                            "critic": 5 * cycle,
                            "ema": 0,
                        },
                        cycle_updates={"generator": 1, "critic": 5, "ema": 0},
                        cumulative_updates={
                            "generator": cycle,
                            "critic": 5 * cycle,
                            "ema": 0,
                        },
                        call_census={"application/student": 20},
                        losses={"generator_mean": 1.0, "critic_mean": 2.0},
                        gradient_norms={
                            "generator_mean": 3.0,
                            "critic_mean": 4.0,
                        },
                        tensor_contracts={
                            "rollout_latent": {
                                "shape": [1, 16, 3, 8, 8],
                                "dtype": "torch.float32",
                            }
                        },
                        state_fingerprints=state,
                        finite=True,
                    )
                )
            commits = (
                (2, 10)
                if mode == "baseline"
                else ((2, 0) if rank < 8 else (0, 10))
            )
            receipt = run_root / "receipts" / f"rank_{rank:03d}.json"
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "run_kind": "trajectory",
                        "trace_sha256": "1" * 64,
                        "bootstrap_sha256": "2" * 64,
                        "source_receipt_sha256": "3" * 64,
                        "final_versions": {"generator": 2, "critic": 10, "ema": 0},
                        "physical_generator_commits": commits[0],
                        "physical_critic_commits": commits[1],
                        "trajectory": {
                            "formal_timing_eligible": False,
                            "arm_id": f"{label}_arm",
                        },
                    }
                )
            )
        output = run_root / "summary.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/distillschedule/summarize_livewan_trajectory_world16.py"),
                "--run-root",
                str(run_root),
                "--cycles",
                "2",
                "--expected-head",
                head,
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        summary = json.loads(output.read_text())
        assert summary["status"] == "PASS"
        assert summary["cycles"] == 2
        assert summary["pipeline_variant"] == variant
        assert summary["run_id"] == label
        assert summary["seed"] == 20260903
        assert summary["identity"] == {
            "trace_sha256": "1" * 64,
            "bootstrap_sha256": "2" * 64,
            "source_receipt_sha256": "3" * 64,
        }
        assert [row["cycle"] for row in summary["numeric_state_anchors"]] == [1, 2]
