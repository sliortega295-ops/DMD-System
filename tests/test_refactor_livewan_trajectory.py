from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from distillschedule.evidence.trajectory_state_sketch import (
    canonical_state_name,
    deterministic_flat_indices,
)
from distillschedule.evidence.trajectory_coarse import (
    AppendOnlyTrajectoryWriter,
    TrajectoryCycleReceipt,
)
from distillschedule.core.config import load_plan
from distillschedule.runtime.livewan_crossscale_worker import (
    OperationTraceRow,
    TRAJECTORY_CONTINUOUS_NAME,
    _apply_trajectory_contract,
    _trajectory_normalized_cycle_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
LIVEWAN = "1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5"
WAN21 = "9737cba9c1c3c4d04b33fcad41c111989865d315"


def _source_trace(path: Path) -> None:
    rows = []
    seed_names = (
        "rollout",
        "generator_dmd_t",
        "generator_dmd_noise",
        "generator_grad_blocks",
        "critic_t",
        "critic_noise",
    )
    for step in range(5):
        for slot in range(16):
            rows.append(
                {
                    "logical_step": step,
                    "logical_slot": slot,
                    "dataset_index": step * 16 + slot,
                    "sample_path": f"/immutable/sample_{step * 16 + slot}.pt",
                    "sample_sha256": None,
                    "operation_key": f"step={step}/slot={slot}",
                    "world_frames": 3,
                    "rope_gap": 1,
                    "seeds": {
                        name: 100_000 * step + 100 * slot + index
                        for index, name in enumerate(seed_names)
                    },
                    "generator_update": step == 0,
                    "critic_update": True,
                }
            )
    path.write_text(
        json.dumps(
            {
                "livewan_commit": LIVEWAN,
                "wan21_commit": WAN21,
                "seed": 20260903,
                "world_size": 16,
                "global_batch": 16,
                "steps": 5,
                "prompts": {"path": "/immutable/prompts", "sha256": "a" * 64},
                "source_sample_count": 80,
                "rows": rows,
            }
        )
    )


def test_trace_builder_defaults_to_legacy_three_cycles(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "derived.json"
    _source_trace(source)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/distillschedule/build_livewan_continuous_trace.py"),
            "--source-trace",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
    )
    value = json.loads(output.read_text())
    assert value["cycles"] == 3
    assert value["steps"] == 15
    assert len(value["rows"]) == 240
    assert value["claim_boundary"].startswith(
        "Three continuous Full5 cycles for pipeline steady-state diagnosis"
    )


def test_trace_builder_emits_50_fresh_operation_keyed_cycles(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "derived.json"
    _source_trace(source)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/distillschedule/build_livewan_continuous_trace.py"),
            "--source-trace",
            str(source),
            "--output",
            str(output),
            "--cycles",
            "50",
        ],
        check=True,
    )
    value = json.loads(output.read_text())
    assert value["cycles"] == 50
    assert value["steps"] == 250
    assert value["generator_updates"] == 50
    assert value["critic_updates"] == 250
    assert len(value["rows"]) == 4_000
    keys = [row["operation_key"] for row in value["rows"]]
    assert len(keys) == len(set(keys))
    assert value["rows"][0]["seeds"] != value["rows"][80]["seeds"]
    for cycle in range(50):
        addresses = {
            (row["cycle_step"], row["logical_slot"])
            for row in value["rows"][cycle * 80 : (cycle + 1) * 80]
        }
        assert addresses == {(step, slot) for step in range(5) for slot in range(16)}


def test_trace_builder_supports_registered_canary_lengths(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    _source_trace(source)
    for cycles in (1, 5):
        output = tmp_path / f"derived_{cycles}.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/distillschedule/build_livewan_continuous_trace.py"),
                "--source-trace",
                str(source),
                "--output",
                str(output),
                "--cycles",
                str(cycles),
            ],
            check=True,
        )
        value = json.loads(output.read_text())
        assert value["cycles"] == cycles
        assert value["steps"] == cycles * 5
        assert value["source_reuse_across_cycles"] is (cycles > 1)
        parsed = [
            OperationTraceRow.from_mapping(
                row,
                total_steps=cycles * 5,
                cycle_length=5,
                global_batch=16,
            )
            for row in value["rows"]
        ]
        assert len(parsed) == cycles * 5 * 16
        if cycles == 1:
            assert parsed[0].operation_key == "step=0/slot=0"


def test_continuous_bootstrap_tracks_each_registered_trajectory_length(
    tmp_path: Path,
) -> None:
    source_trace = tmp_path / "source_trace.json"
    source_bootstrap = tmp_path / "source_bootstrap.json"
    _source_trace(source_trace)
    source_bootstrap.write_text(
        json.dumps(
            {
                "kind": "systems_only_livewan_crossscale_bootstrap",
                "initial_versions": {"generator": 0, "critic": 0, "ema": 0},
                "bootstrap_sha256": "source-placeholder",
            }
        )
    )
    for cycles in (1, 5, 50):
        trace = tmp_path / f"trace_{cycles}.json"
        bootstrap = tmp_path / f"bootstrap_{cycles}.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/distillschedule/build_livewan_continuous_trace.py"),
                "--source-trace",
                str(source_trace),
                "--output",
                str(trace),
                "--cycles",
                str(cycles),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/distillschedule/derive_livewan_continuous_bootstrap.py"),
                "--source-bootstrap",
                str(source_bootstrap),
                "--trace",
                str(trace),
                "--output",
                str(bootstrap),
            ],
            check=True,
        )
        value = json.loads(bootstrap.read_text())
        assert value["continuous_diagnostic"] == {
            "cycles": cycles,
            "source_reuse_across_cycles": cycles > 1,
            "initial_versions": {"generator": 0, "critic": 0, "ema": 0},
            "expected_final_versions": {
                "generator": cycles,
                "critic": 5 * cycles,
                "ema": 0,
            },
        }
        assert value["trace"]["sha256"] == json.loads(trace.read_text())[
            "trace_sha256"
        ]


def test_trajectory_worker_keeps_legacy_c24_and_adds_explicit_mode() -> None:
    source = (
        ROOT / "src/distillschedule/runtime/livewan_crossscale_worker.py"
    ).read_text()
    assert 'run_kind == "trajectory"' in source
    assert 'trace.cycles == 3' in source
    assert (
        "_apply_trajectory_contract(plan, cycles=trajectory_cycles, mode=mode)"
        in source
    )
    assert "def _run_baseline_trajectory(" in source
    assert "def _trajectory_normalized_cycle_evidence(" in source
    assert 'workload="livewan_wan13b_teacher14b_sfdmd"' in source
    assert '"generator": 1, "critic": 5, "ema": 0' in source
    assert "continuous trajectory changed its per-cycle call census" in source
    assert "collect_numeric_state_sketch" in source
    assert "ordinary_numerical_drift_early_stop" in source


def test_trajectory_contract_supports_both_matched_arms_without_mutating_files(
    monkeypatch,
) -> None:
    for key in (
        "LIVEWAN_STUDENT_CHECKPOINT",
        "LIVEWAN_CRITIC_CHECKPOINT",
        "LIVEWAN_TEACHER_CHECKPOINT",
        "LIVEWAN_OPERATION_TRACE",
        "LIVEWAN_SYSTEMS_BOOTSTRAP",
        "DISTILLSCHEDULE_OUTPUT_DIR",
    ):
        monkeypatch.setenv(key, f"/tmp/{key.lower()}")
    baseline_path = (
        ROOT
        / "configs/distillschedule/plans/"
        "livewan_wan13b_14b_sf_dmd_g1c5_world16_fsdp2.json"
    )
    pipeline_path = (
        ROOT
        / "configs/distillschedule/plans/"
        "livewan_wan13b_14b_sf_dmd_c24_continuous_b16_score_modulation.json"
    )
    baseline = load_plan(baseline_path).plan.to_dict()
    pipeline = load_plan(pipeline_path).plan.to_dict()
    _apply_trajectory_contract(baseline, cycles=5, mode="baseline")
    _apply_trajectory_contract(pipeline, cycles=50, mode="pipeline")
    assert baseline["metadata"]["trajectory_diagnostic"] == {
        "schema": "trajectory_coarse_v1",
        "cycles": 5,
        "schedule": "world16_fsdp2_sequential",
        "source_reuse_across_cycles": True,
        "formal_timing_eligible": False,
    }
    continuous = pipeline["metadata"]["continuous_pipeline"]
    assert continuous["name"] == TRAJECTORY_CONTINUOUS_NAME
    assert continuous["cycles"] == 50
    assert continuous["critic_shadow_sync_versions"][-1] == 245
    assert "trajectory_diagnostic" not in json.loads(baseline_path.read_text())[
        "metadata"
    ]


def test_numeric_state_coordinates_are_stable_and_wrapper_independent() -> None:
    assert canonical_state_name("module._orig_mod.blocks.0.weight") == (
        "blocks.0.weight"
    )
    first = deterministic_flat_indices("blocks.0.weight", 123_457, count=16)
    second = deterministic_flat_indices("blocks.0.weight", 123_457, count=16)
    assert first == second
    assert len(first) == len(set(first)) == 16
    assert first[0] == 0 and first[-1] == 123_456


def test_controller_and_launcher_make_trajectory_opt_in() -> None:
    controller = (
        ROOT / "scripts/distillschedule/run_livewan_crossscale_world16.py"
    ).read_text()
    launcher = (
        ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh"
    ).read_text()
    entrypoint = (
        ROOT / "scripts/distillschedule/livewan_crossscale_entrypoint.py"
    ).read_text()
    assert '"trajectory"' in controller
    assert '"--trajectory-cycles", type=int, default=0' in controller
    assert "trajectory baseline requires sequential B1" in controller
    assert "trajectory pipeline requires exact C24" in controller
    assert "LIVEWAN_TRAJECTORY_CYCLES" in launcher
    assert "trajectory_baseline_requires_b1" in launcher
    assert "trajectory_pipeline_requires_exact_c24" in launcher
    assert 'if [[ "${RUN_KIND}" == trajectory ]]; then' in launcher
    assert '"trajectory"' in entrypoint


def test_pipeline_physical_b2_is_normalized_to_logical_rank_census() -> None:
    contributions = []
    for rank in range(16):
        lane = rank if rank < 8 else rank - 8
        slots = (lane, lane + 8)
        if rank < 8:
            losses = [
                {
                    "kind": "generator",
                    "operation_key": f"cycle=0/step=0/slot={slot}",
                    "mean": float(slot + 1),
                }
                for slot in slots
            ]
            optimizer = [{"role": "generator", "clip_input_norm": 2.0}]
            census = {"application/student": 2}
        else:
            losses = [
                {
                    "kind": "critic",
                    "operation_key": f"cycle=0/step={step}/slot={slot}",
                    "mean": float(step + slot + 1),
                }
                for step in range(5)
                for slot in slots
            ]
            optimizer = [
                {"role": "critic", "clip_input_norm": 3.0} for _ in range(5)
            ]
            census = {
                "logical_score/critic/grad/checkpoint": 10,
                "logical_score/critic/nograd/plain": 2,
                "logical_score/teacher/nograd/plain": 2,
            }
        contributions.append(
            {"losses": losses, "optimizer": optimizer, "call_census": census}
        )

    class FakeDistributed:
        @staticmethod
        def all_gather_object(output, _local):
            output[:] = contributions

    class FakeTrace:
        global_batch = 16

        @staticmethod
        def row(global_step, rank):
            return SimpleNamespace(
                operation_key=f"cycle=0/step={global_step}/slot={rank}",
                seeds={"rollout": global_step * 100 + rank},
            )

    result = _trajectory_normalized_cycle_evidence(
        torch=SimpleNamespace(distributed=FakeDistributed()),
        rank=0,
        cycle=0,
        trace=FakeTrace(),
        local_losses=(),
        local_optimizer=(),
        local_call_census={},
    )
    samples, _rng, census, losses, gradients, finite = result
    assert samples == tuple(f"cycle=0/step={step}/slot=0" for step in range(5))
    assert census == {
        "application/student": 1,
        "logical_score/critic/grad/checkpoint": 5,
        "logical_score/critic/nograd/plain": 1,
        "logical_score/teacher/nograd/plain": 1,
    }
    assert losses == {"generator_mean": 1.0, "critic_mean": 3.0}
    assert gradients == {"generator_mean": 2.0, "critic_mean": 3.0}
    assert finite is True


def test_trajectory_summarizer_accepts_baseline_and_pipeline_receipts(
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
    anchor_body = {
        "schema_version": "distillschedule.livewan_trajectory_state_anchor.v1",
        "cycle": 1,
        "writer_rank": 0,
        "topology": {},
        "components": components,
        "comparison_scope": "test",
    }
    anchor_digest = hashlib.sha256(
        json.dumps(anchor_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    anchor = {**anchor_body, "anchor_sha256": anchor_digest}
    state = {name: name + "_sha" for name in components}
    state["anchor"] = anchor_digest

    for mode in ("baseline", "pipeline"):
        run_root = tmp_path / mode
        for node in range(2):
            manifest = run_root / "launcher" / f"node_{node}" / "launch_manifest.txt"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                f"head={head}\nexecution_mode={mode}\ntrajectory_cycles=1\n"
                f"trajectory_arm_id={mode}_arm\nclaim_boundary=DIAGNOSTIC_ONLY\n"
            )
        anchor_path = run_root / "trajectory/anchors/cycle_01_global.json"
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        anchor_path.write_text(json.dumps(anchor))
        for rank in range(16):
            samples = tuple(
                f"cycle=0/step={step}/slot={rank}" for step in range(5)
            )
            writer = AppendOnlyTrajectoryWriter(
                run_root / "trajectory" / f"rank_{rank:03d}.jsonl",
                run_id=mode,
                rank=rank,
            )
            writer.append(
                TrajectoryCycleReceipt(
                    run_id=mode,
                    workload="livewan_wan13b_teacher14b_sfdmd",
                    variant=f"{mode}_arm",
                    rank=rank,
                    world_size=16,
                    seed=20260903,
                    cycle=1,
                    expected_total_cycles=1,
                    logical_step=0,
                    matched_contract={"contract": "same"},
                    implementation={"execution_mode": mode},
                    sample_identities=samples,
                    rng_identities={sample: "b" * 64 for sample in samples},
                    versions={"generator": 1, "critic": 5, "ema": 0},
                    cycle_updates={"generator": 1, "critic": 5, "ema": 0},
                    cumulative_updates={"generator": 1, "critic": 5, "ema": 0},
                    call_census={"application/student": 20},
                    losses={"generator_mean": 1.0, "critic_mean": 2.0},
                    gradient_norms={"generator_mean": 3.0, "critic_mean": 4.0},
                    tensor_contracts={
                        "rollout_latent": {"shape": [1, 16, 3, 8, 8], "dtype": "torch.float32"}
                    },
                    state_fingerprints=state,
                    finite=True,
                )
            )
            commits = (
                (1, 5) if mode == "baseline" else ((1, 0) if rank < 8 else (0, 5))
            )
            receipt = run_root / "receipts" / f"rank_{rank:03d}.json"
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "run_kind": "trajectory",
                        "final_versions": {"generator": 1, "critic": 5, "ema": 0},
                        "physical_generator_commits": commits[0],
                        "physical_critic_commits": commits[1],
                        "trajectory": {
                            "formal_timing_eligible": False,
                            "arm_id": f"{mode}_arm",
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
                "1",
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
        assert json.loads(output.read_text())["status"] == "PASS"
