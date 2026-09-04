from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from typing import Any
from pathlib import Path

from distillschedule.evidence.trajectory_coarse import TrajectoryCycleReceipt


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py"
SUMMARIZER = ROOT / "scripts/distillschedule/summarize_wan13b_two_cycle_timing_world16.py"
COMPARATOR = ROOT / "scripts/distillschedule/compare_wan13b_two_cycle_timing.py"
LAUNCHER = ROOT / "scripts/distillschedule/launch_wan13b_fair_world16_node.sh"
CONTROLLER = ROOT / "scripts/distillschedule/run_wan13b_fair_world16.py"
HEAD = "2" * 40
SHA = "a" * 64
ROLLOUT_OPERATIONS = (
    "step/10/generator/rollout",
    *(f"step/{step}/critic/rollout" for step in range(10, 15)),
)
ROLLOUT_STAGE_CALLS = (
    (2, 1, 1),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 1),
    (2, 1, 1),
    (2, 2, 2),
)


def _rollout_contract() -> list[dict[str, object]]:
    result = []
    for operation, stage_values in zip(
        ROLLOUT_OPERATIONS, ROLLOUT_STAGE_CALLS, strict=True
    ):
        stages = dict(enumerate(stage_values))
        total = sum(stage_values)
        result.append(
            {
                "operation": operation,
                "inventory": {
                    "call_count": total,
                    "logical_call_count": total,
                    "stage_call_count": stages,
                    "logical_stage_call_count": dict(stages),
                    "dummy_call_count": 0,
                },
            }
        )
    return result


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_two_cycle_worker_is_a_dedicated_timing_path() -> None:
    source = WORKER.read_text()
    tree = ast.parse(source)
    assert "TIMED_CYCLES = 2" in source
    assert '"generator": 3, "critic": 15, "ema": 15' in source
    assert '"generator": 4, "critic": 20, "ema": 20' in source
    assert '"FIXED_FIVE_STEP_COHORT_REPLAY"' in source
    assert "expected_trace_rollout_call_inventory(" in source
    assert '"trace_rollout_contract_sha256"' in source
    assert '"model_call_count", -1)) != 56' in source
    assert '"fused_invocation_count", -1)) != 17_080' in source
    assert '"reference_fallback_count", -1)) != 224' in source

    workload = _function(WORKER, "_run_workload_cycle")
    calls = [node for node in ast.walk(workload) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "cpu"
        for call in calls
    )
    assert not any(
        isinstance(call.func, ast.Name) and call.func.id == "_atomic_json"
        for call in calls
    )
    workload_source = ast.get_source_segment(source, workload)
    assert "collect_scalars=False" in workload_source
    assert "collect_diagnostics=False" in workload_source

    main_source = ast.get_source_segment(source, _function(WORKER, "main"))
    assert main_source.count("torch.cuda.synchronize(device)") >= 2
    assert main_source.count("dist.barrier()") >= 2
    assert main_source.index("_run_workload_cycle(") < main_source.index(
        "torch.stack(deferred_finite_checks"
    )
    assert main_source.index("torch.stack(deferred_finite_checks") < main_source.index(
        "_atomic_json(output, result)"
    )
    assert any(isinstance(node, ast.FunctionDef) for node in tree.body)


def test_two_cycle_worker_populates_every_shared_builder_mode_flag() -> None:
    helper = _function(WORKER, "_configure_shared_mode_args")
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {"argparse": argparse}
    exec(compile(module, str(WORKER), "exec"), namespace)
    args = argparse.Namespace()
    namespace["_configure_shared_mode_args"](args)  # type: ignore[index,operator]
    assert vars(args) == {
        "timing_mode": True,
        "profile_mode": False,
        "trajectory_mode": False,
        "validate_only": False,
    }


def test_controller_and_launcher_register_fail_closed_two_cycle_mode() -> None:
    controller = CONTROLLER.read_text()
    launcher = LAUNCHER.read_text()
    assert '"two-cycle-timing"' in controller
    assert "DISTILLSCHEDULE_TWO_CYCLE_SEED" in controller
    assert "DISTILLSCHEDULE_TWO_CYCLE_ARM_ID" in controller
    assert "DISTILLSCHEDULE_WAN13B_TRAJECTORY_COMPARISON_RECEIPT" in controller
    assert "DISTILLSCHEDULE_WAN13B_TRAJECTORY_COMPARISON_SHA256" in controller
    assert "summarize_wan13b_two_cycle_timing_world16.py" in controller
    assert '"${MODE}" == two-cycle-timing' in launcher
    assert "probe_wan13b_two_cycle_timing_world16.py" in launcher
    assert "STEADY_STATE_CYCLE2_UNPROFILED" in launcher
    assert "two_cycle_correctness_sha256=" in launcher
    assert "two_cycle_trajectory_comparison_sha256=" in launcher
    assert "teardown_status.txt" in launcher
    assert "gpu_process_count=0" in launcher
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_two_cycle_correctness_receipt_schema_accepts_only_scalar_census() -> None:
    receipt = TrajectoryCycleReceipt(
        run_id="r", workload="w", variant="arm", rank=0, world_size=16,
        seed=1, cycle=1, expected_total_cycles=2, logical_step=10,
        matched_contract={
            "bootstrap_hash": "a", "model_identity_sha256": "b",
            "trace_bundle_hash": "c", "trace_binding_hash": "d",
            "trace_replay_policy": "FIXED_FIVE_STEP_COHORT_REPLAY",
            "negative_prompt_manifest_hash": "e", "seed": "1",
            "global_batch": "16", "updates": "G1_C5_EMA5",
            "score_packing": "false", "rollout_batch_size": "1",
            "trace_rollout_contract_sha256": "f" * 64,
        },
        implementation={"placement": "world16-fsdp", "arm_contract": "baseline"},
        sample_identities=("g", "c0", "c1", "c2", "c3", "c4"),
        rng_identities={"runtime": "1"},
        versions={"generator": 3, "critic": 15, "ema": 15},
        cycle_updates={"generator": 1, "critic": 5, "ema": 5},
        cumulative_updates={"generator": 1, "critic": 5, "ema": 5},
        call_census={
            "generator_update_rollout_dit": 4,
            "generator_critic_rollout_dit": 27,
            "rollout_inventory_count": 6,
            "rollout_dummy_calls_total": 0,
        },
        losses={"generator": 1.0}, gradient_norms={},
        tensor_contracts={"latent": {"shape": [1, 16, 9, 48, 80], "dtype": "torch.bfloat16"}},
        state_fingerprints={}, finite=True,
    )
    assert receipt.expected_total_cycles == 2
    assert receipt.call_census["rollout_inventory_count"] == 6


def test_two_cycle_worker_binds_arm_correctness_and_cross_arm_comparison(
    tmp_path: Path,
) -> None:
    tree = ast.parse(WORKER.read_text())
    names = {
        "_sha256",
        "_canonical_hash",
        "_load_correctness_binding",
        "_load_trajectory_comparison_binding",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {
        "Any": Any,
        "Path": Path,
        "argparse": argparse,
        "hashlib": hashlib,
        "json": json,
        "TIMED_CYCLES": 2,
        "WORLD_SIZE": 16,
    }
    exec(compile(module, str(WORKER), "exec"), namespace)
    arm = {
        "arm_contract": "WORLD16_FSDP_BASELINE",
        "generator_nograd_lora_fused_epilogue": False,
        "generator_nograd_lora_cast_cache": False,
    }
    matched = {
        "bootstrap_hash": "b" * 64,
        "trace_bundle_hash": "a" * 64,
        "trace_replay_policy": "FIXED_FIVE_STEP_COHORT_REPLAY",
        "seed": 20260903,
        "global_batch": 16,
        "updates": "G1_C5_EMA5",
        "rollout_batch_size": 1,
        "score_packing": False,
        "trace_rollout_contract_sha256": _digest(_rollout_contract()),
    }
    implementation = dict(arm, trajectory_arm_id="fsdp-two-cycle")
    correctness = {
        "schema_version": "distillschedule.wan13b_trajectory_summary.v1",
        "status": "PASS",
        "placement": "world16-fsdp",
        "cycles": 2,
        "rank_count": 16,
        "final_versions": {"generator": 4, "critic": 20, "ema": 20},
        "diagnostic_only": True,
        "formal_timing_eligible": False,
        "expected_head": HEAD,
        "run_id": "fsdp-correctness-c2",
        "matched_contract": matched,
        "workload_fingerprint": _digest(matched),
        "implementation_contract": implementation,
        "implementation_fingerprint": _digest(implementation),
    }
    correctness_path = tmp_path / "correctness.json"
    correctness_path.write_text(json.dumps(correctness))
    comparison = {
        "schema_version": "distillschedule.trajectory_coarse_comparison.v1",
        "comparison_kind": "semantic",
        "expected_cycles": 2,
        "expected_world_size": 16,
        "complete": True,
        "status": "TRAJECTORY_REVIEW_50C_COMPLETED",
        "contract": {"status": "PASS", "mismatches": []},
        "reference": {
            "variants": ["fsdp-two-cycle"],
            "run_ids": ["fsdp-correctness-c2"],
        },
        "candidate": {
            "variants": ["resident-two-cycle"],
            "run_ids": ["resident-correctness-c2"],
        },
    }
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison))
    args = argparse.Namespace(
        expected_correctness_hash=hashlib.sha256(correctness_path.read_bytes()).hexdigest(),
        correctness_receipt=correctness_path,
        placement="world16-fsdp",
        expected_head=HEAD,
        timing_seed=20260903,
        expected_bootstrap_hash="b" * 64,
        expected_trace_hash="a" * 64,
        timing_arm_id="fsdp-two-cycle",
        expected_trajectory_comparison_hash=hashlib.sha256(
            comparison_path.read_bytes()
        ).hexdigest(),
        trajectory_comparison_receipt=comparison_path,
    )
    binding = namespace["_load_correctness_binding"](args, expected_arm=arm)  # type: ignore[index,operator]
    assert binding["status"] == "PASS"
    comparison_binding = namespace["_load_trajectory_comparison_binding"](  # type: ignore[index,operator]
        args, correctness_run_id="fsdp-correctness-c2"
    )
    assert comparison_binding["contract_status"] == "PASS"
    assert comparison_binding["numerical_review_recorded"] is True
    assert comparison_binding["matched_side"] == "reference"

    comparison["reference"]["run_ids"] = ["unrelated-correctness-run"]
    comparison_path.write_text(json.dumps(comparison))
    args.expected_trajectory_comparison_hash = hashlib.sha256(
        comparison_path.read_bytes()
    ).hexdigest()
    try:
        namespace["_load_trajectory_comparison_binding"](  # type: ignore[index,operator]
            args, correctness_run_id="fsdp-correctness-c2"
        )
    except ValueError as error:
        assert "not bound" in str(error)
    else:
        raise AssertionError("unrelated comparison/correctness runs were accepted")


def _manifest(run_root: Path, node: int, placement: str, run_id: str) -> None:
    target = run_root / "launcher" / f"node_{node}"
    target.mkdir(parents=True)
    fields = {
        "mode": "two-cycle-timing",
        "run_id": run_id,
        "dmd_system_head": HEAD,
        "placement": placement,
        "world_size": "16",
        "global_batch": "16",
        "rollout_batch_size": "1",
        "score_packing": "false",
        "trace_hash": SHA,
        "bootstrap_hash": "b" * 64,
        "model_root": "/model",
        "helios_transformer_sha256": "c" * 64,
        "two_cycle_count": "2",
        "two_cycle_claimable_cycle": "2",
        "two_cycle_trace_replay_policy": "FIXED_FIVE_STEP_COHORT_REPLAY",
        "two_cycle_seed": "20260903",
        "two_cycle_arm_id": "fsdp-two-cycle",
        "two_cycle_correctness_sha256": "d" * 64,
        "two_cycle_trajectory_comparison_sha256": "8" * 64,
        "generator_nograd_lora_fused_epilogue": "0",
        "generator_nograd_lora_cast_cache": "0",
        "claim_boundary": "STEADY_STATE_CYCLE2_UNPROFILED",
        "parameter_offload": "false",
        "activation_offload": "false",
        "optimizer_offload": "false",
        "parameter_streaming": "false",
        "model_swap": "false",
        "ema_implementation": "reference",
        "optimizer_implementation": "PYTORCH_DEFAULT",
        "nccl_algo": "AUTO",
        "modulation_chunk_tokens": "RUNTIME_DEFAULT",
    }
    (target / "launch_manifest.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in fields.items())
    )
    (target / "cgroup_memory_events_before.txt").write_text("oom 10\noom_kill 2\n")
    (target / "cgroup_memory_events_after.txt").write_text("oom 10\noom_kill 2\n")
    (target / "teardown_status.txt").write_text(
        f"status=PASS\ngpu_process_count=0\nowner_run_id={run_id}\n"
    )


def _rank_receipt(rank: int, run_id: str) -> dict[str, object]:
    trace_rollout_contract = _rollout_contract()
    matched = {
        "bootstrap_hash": "b" * 64,
        "model_identity_sha256": "e" * 64,
        "trace_bundle_hash": SHA,
        "trace_binding_hash": "f" * 64,
        "trace_replay_policy": "FIXED_FIVE_STEP_COHORT_REPLAY",
        "negative_prompt_manifest_hash": "1" * 64,
        "seed": 20260903,
        "global_batch": 16,
        "updates": "G1_C5_EMA5",
        "rollout_batch_size": 1,
        "score_packing": False,
        "trace_rollout_contract_sha256": _digest(trace_rollout_contract),
    }
    arm = {
        "arm_contract": "WORLD16_FSDP_BASELINE",
        "generator_nograd_lora_fused_epilogue": False,
        "generator_nograd_lora_cast_cache": False,
    }
    census = []
    for cycle in (1, 2):
        payloads = [
            hashlib.sha256(f"rank={rank}/slot={slot}".encode()).hexdigest()
            for slot in range(6)
        ]
        census.append({
            "generator_updates": 1,
            "critic_updates": 5,
            "ema_updates": 5,
            "generator_score_forwards": 3,
            "critic_score_forwards": 5,
            "physical_rollout_batches": 5,
            "rollout_batch_size": 1,
            "rollout_operation_keys": list(ROLLOUT_OPERATIONS),
            "rollout_call_inventory": [
                dict(row["inventory"]) for row in trace_rollout_contract
            ],
            "generator_trace_payload_identity": payloads[0],
            "critic_trace_payload_identities": payloads[1:],
            "sample_payload_identities": payloads,
            "sample_occurrence_identities": [
                f"cycle={cycle}/slot={slot}/payload={payload}"
                for slot, payload in enumerate(payloads)
            ],
        })
    return {
        "schema_version": "distillschedule.wan13b_two_cycle_timing_rank.v1",
        "status": "PASS",
        "run_id": run_id,
        "rank": rank,
        "local_rank": rank % 8,
        "hostname": f"node{rank // 8}",
        "world_size": 16,
        "placement": "world16-fsdp",
        "expected_head": HEAD,
        "timing_arm_id": "fsdp-two-cycle",
        "arm_contract": arm,
        "correctness_binding": {
            "status": "PASS",
            "sha256": "d" * 64,
            "run_id": "fsdp-correctness-c2",
        },
        "trajectory_comparison_binding": {
            "sha256": "8" * 64,
            "schema_version": "distillschedule.trajectory_coarse_comparison.v1",
            "status": "TRAJECTORY_MATCHED_50C_SINGLE_SEED",
            "contract_status": "PASS",
            "complete": True,
            "reference_arms": ["fsdp-two-cycle"],
            "candidate_arms": ["resident-two-cycle"],
            "matched_side": "reference",
            "matched_correctness_run_id": "fsdp-correctness-c2",
            "numerical_review_recorded": False,
        },
        "profiled": False,
        "profiler_status": "NOT_RUN",
        "profiler_environment": {"exact_range": None, "lora_operator_attribution": "0"},
        "matched_contract": matched,
        "trace_rollout_contract": trace_rollout_contract,
        "trace_rollout_contract_sha256": _digest(trace_rollout_contract),
        "workload_fingerprint": _digest(matched),
        "implementation_fingerprint": _digest(arm),
        "configuration_fingerprint": _digest({
            "workload_fingerprint": _digest(matched),
            "implementation_fingerprint": _digest(arm),
        }),
        "initial_versions": {"generator": 2, "critic": 10, "ema": 10},
        "cycles": [
            {"cycle": 1, "role": "WARMUP_DIAGNOSTIC", "start_unix_ns": 1, "stop_unix_ns": 2, "local_elapsed_seconds": 10.0 + rank / 100},
            {"cycle": 2, "role": "STEADY_STATE_CLAIMABLE", "start_unix_ns": 3, "stop_unix_ns": 4, "local_elapsed_seconds": 9.0 + rank / 100},
        ],
        "cycle_versions": [
            {"generator": 3, "critic": 15, "ema": 15},
            {"generator": 4, "critic": 20, "ema": 20},
        ],
        "cycle_census": census,
        "finite_status": [True, True],
        "final_versions": {"generator": 4, "critic": 20, "ema": 20},
        "updates": {"generator": 2, "critic": 10, "ema": 10},
        "construction_seconds": 100.0 + rank,
        "memory": {
            "allocated_bytes": 1,
            "reserved_bytes": 2,
            "max_allocated_bytes": 3 + rank,
            "max_reserved_bytes": 4 + rank,
            "driver_free_bytes": 100 - rank,
            "driver_total_bytes": 1000,
        },
        "runtime_contract": {
            "helios_transformer_sha256": "c" * 64,
            "ema_implementation": "reference",
            "nccl_algo": "AUTO",
            "residual_storage": "cuda_out_of_place",
            "parameter_offload": False,
            "activation_offload": False,
            "optimizer_offload": False,
            "parameter_streaming": False,
            "model_swap": False,
        },
        "fused_epilogue_receipt": {
            "generator": {"requested": False, "enabled": False},
            "critic": {"requested": False, "enabled": False},
        },
        "evidence_extraction": "AFTER_BOTH_TIMED_WINDOWS",
        "state_anchors": "NOT_COLLECTED_IN_TIMING_MODE",
    }


def _synthetic_run(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "run"
    run_id = "two-cycle-test"
    for node in (0, 1):
        _manifest(run_root, node, "world16-fsdp", run_id)
    ranks = run_root / "execution" / "ranks"
    ranks.mkdir(parents=True)
    for rank in range(16):
        (ranks / f"rank_{rank:02d}.json").write_text(
            json.dumps(_rank_receipt(rank, run_id)) + "\n"
        )
    return run_root, tmp_path / "summary.json"


def test_two_cycle_summarizer_uses_cycle2_world_max_and_reconciles_16_ranks(tmp_path: Path) -> None:
    run_root, output = _synthetic_run(tmp_path)
    completed = subprocess.run(
        [
            sys.executable, str(SUMMARIZER), "--run-root", str(run_root),
            "--placement", "world16-fsdp", "--expected-head", HEAD,
            "--output", str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    value = json.loads(output.read_text())
    assert value["status"] == "PASS"
    assert value["host_topology"] == {"node0": list(range(8)), "node1": list(range(8))}
    assert value["cycle_1"]["claimable"] is False
    assert value["cycle_2"]["claimable"] is True
    assert value["cycle_2"]["world_max_seconds"] == 9.15
    assert value["speed_claim_denominator"] == "CYCLE2_WORLD_MAX_ONLY"
    assert value["trajectory_comparison_sha256"] == "8" * 64
    assert value["sample_identity_contract"] == {
        "rank_qualified_payloads_per_cycle": 96,
        "critic_payloads_per_cycle": 80,
        "cycle_qualified_occurrences": 192,
        "payload_cohort_replayed": True,
        "ordered_rank_slot_mapping_replayed": True,
    }


def test_two_cycle_summarizer_rejects_duplicate_cycle_or_nan(tmp_path: Path) -> None:
    run_root, output = _synthetic_run(tmp_path)
    path = run_root / "execution" / "ranks" / "rank_00.json"
    value = json.loads(path.read_text())
    value["cycles"][1]["cycle"] = 1
    value["cycles"][1]["local_elapsed_seconds"] = float("nan")
    path.write_text(json.dumps(value) + "\n")
    completed = subprocess.run(
        [sys.executable, str(SUMMARIZER), "--run-root", str(run_root),
         "--placement", "world16-fsdp", "--expected-head", HEAD,
         "--output", str(output)],
        check=False,
    )
    assert completed.returncode == 1
    summary = json.loads(output.read_text())
    assert summary["status"] == "FAIL"
    assert any("cycle timer identity" in error for error in summary["errors"])
    assert any("invalid cycle duration" in error for error in summary["errors"])


def test_two_cycle_summarizer_rejects_rank_slot_payload_permutation(
    tmp_path: Path,
) -> None:
    run_root, output = _synthetic_run(tmp_path)
    path = run_root / "execution" / "ranks" / "rank_00.json"
    value = json.loads(path.read_text())
    cycle2 = value["cycle_census"][1]
    cycle2["critic_trace_payload_identities"][0:2] = reversed(
        cycle2["critic_trace_payload_identities"][0:2]
    )
    cycle2["sample_payload_identities"] = [
        cycle2["generator_trace_payload_identity"],
        *cycle2["critic_trace_payload_identities"],
    ]
    cycle2["sample_occurrence_identities"] = [
        f"cycle=2/slot={slot}/payload={payload}"
        for slot, payload in enumerate(cycle2["sample_payload_identities"])
    ]
    path.write_text(json.dumps(value) + "\n")
    completed = subprocess.run(
        [sys.executable, str(SUMMARIZER), "--run-root", str(run_root),
         "--placement", "world16-fsdp", "--expected-head", HEAD,
         "--output", str(output)],
        check=False,
        stdout=subprocess.DEVNULL,
    )
    assert completed.returncode == 1
    summary = json.loads(output.read_text())
    assert any("rank-slot mapping" in error for error in summary["errors"])


def test_two_cycle_summarizer_rejects_trace_specific_c12_inventory_drift(
    tmp_path: Path,
) -> None:
    run_root, output = _synthetic_run(tmp_path)
    path = run_root / "execution" / "ranks" / "rank_00.json"
    value = json.loads(path.read_text())
    # This remains an internally consistent exact receipt, but it is not the
    # C12 schedule frozen in the trace.
    drifted = {
        "call_count": 6,
        "logical_call_count": 6,
        "stage_call_count": {"0": 2, "1": 2, "2": 2},
        "logical_stage_call_count": {"0": 2, "1": 2, "2": 2},
        "dummy_call_count": 0,
    }
    value["cycle_census"][0]["rollout_call_inventory"][3] = drifted
    path.write_text(json.dumps(value) + "\n")
    completed = subprocess.run(
        [sys.executable, str(SUMMARIZER), "--run-root", str(run_root),
         "--placement", "world16-fsdp", "--expected-head", HEAD,
         "--output", str(output)],
        check=False,
        stdout=subprocess.DEVNULL,
    )
    assert completed.returncode == 1
    summary = json.loads(output.read_text())
    assert any(
        "step/12/critic/rollout" in error for error in summary["errors"]
    )


def test_two_cycle_summarizer_rejects_same_total_operation_swap(
    tmp_path: Path,
) -> None:
    run_root, output = _synthetic_run(tmp_path)
    path = run_root / "execution" / "ranks" / "rank_00.json"
    value = json.loads(path.read_text())
    calls = value["cycle_census"][0]["rollout_call_inventory"]
    calls[3], calls[4] = calls[4], calls[3]
    path.write_text(json.dumps(value) + "\n")
    completed = subprocess.run(
        [sys.executable, str(SUMMARIZER), "--run-root", str(run_root),
         "--placement", "world16-fsdp", "--expected-head", HEAD,
         "--output", str(output)],
        check=False,
        stdout=subprocess.DEVNULL,
    )
    assert completed.returncode == 1
    summary = json.loads(output.read_text())
    assert any(
        "trace-derived DiT census changed" in error for error in summary["errors"]
    )


def test_two_cycle_summarizer_rejects_cycle2_inventory_mutation(
    tmp_path: Path,
) -> None:
    run_root, output = _synthetic_run(tmp_path)
    path = run_root / "execution" / "ranks" / "rank_00.json"
    value = json.loads(path.read_text())
    value["cycle_census"][1]["rollout_call_inventory"][5] = dict(
        value["cycle_census"][1]["rollout_call_inventory"][3]
    )
    path.write_text(json.dumps(value) + "\n")
    completed = subprocess.run(
        [sys.executable, str(SUMMARIZER), "--run-root", str(run_root),
         "--placement", "world16-fsdp", "--expected-head", HEAD,
         "--output", str(output)],
        check=False,
        stdout=subprocess.DEVNULL,
    )
    assert completed.returncode == 1
    summary = json.loads(output.read_text())
    assert any(
        "rank0/cycle2" in error and "step/14/critic/rollout" in error
        for error in summary["errors"]
    )


def test_two_cycle_comparator_claims_only_matched_cycle2(tmp_path: Path) -> None:
    run_root, baseline_path = _synthetic_run(tmp_path)
    subprocess.run(
        [sys.executable, str(SUMMARIZER), "--run-root", str(run_root),
         "--placement", "world16-fsdp", "--expected-head", HEAD,
         "--output", str(baseline_path)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    baseline = json.loads(baseline_path.read_text())
    optimized = dict(baseline)
    optimized["placement"] = "resident-async128-tree-chunk2048"
    optimized["correctness_receipt_sha256"] = "9" * 64
    optimized["implementation_contract"] = {
        "arm_contract": "RESIDENT_ASYNC128_TREE_CHUNK2048_FUSED_EPILOGUE",
        "generator_nograd_lora_fused_epilogue": True,
        "generator_nograd_lora_cast_cache": False,
    }
    optimized["implementation_fingerprint"] = _digest(
        optimized["implementation_contract"]
    )
    optimized["cycle_1"] = dict(optimized["cycle_1"], world_max_seconds=8.0)
    optimized["cycle_2"] = dict(optimized["cycle_2"], world_max_seconds=7.0)
    optimized_path = tmp_path / "optimized.json"
    optimized_path.write_text(json.dumps(optimized) + "\n")
    output = tmp_path / "pair.json"
    completed = subprocess.run(
        [sys.executable, str(COMPARATOR), "--baseline-summary", str(baseline_path),
         "--optimized-summary", str(optimized_path), "--output", str(output)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    value = json.loads(output.read_text())
    assert value["status"] == "PASS"
    assert value["cycle_2_speedup"] == 9.15 / 7.0
    assert value["claim_boundary"] == "MATCHED_UNPROFILED_CYCLE2_WORLD_MAX_ONLY"
    assert value["cycle_1_use"] == "WARMUP_DIAGNOSTIC_EXCLUDED"
