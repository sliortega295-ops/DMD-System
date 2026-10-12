#!/usr/bin/env python3
"""Fail-closed world16 summary for same-process Wan1.3B two-cycle timing."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


WORLD_SIZE = 16
RESIDENT_WINNER = "resident-async128-tree-chunk2048"
FSDP_BASELINE = "world16-fsdp"
RESIDENT_REFERENCE = "resident"
REGISTERED_PLACEMENTS = frozenset({RESIDENT_WINNER, FSDP_BASELINE, RESIDENT_REFERENCE})
EXPECTED_VERSIONS = (
    {"generator": 3, "critic": 15, "ema": 15},
    {"generator": 4, "critic": 20, "ema": 20},
)
EXPECTED_ROLLOUT_OPERATIONS = (
    "step/10/generator/rollout",
    *(f"step/{step}/critic/rollout" for step in range(10, 15)),
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _normalize_rollout_contract(
    value: Any, *, label: str, errors: list[str]
) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or len(value) != len(EXPECTED_ROLLOUT_OPERATIONS):
        errors.append(f"{label}: trace rollout contract has the wrong operation count")
        return None
    result: list[dict[str, Any]] = []
    for index, (row, expected_operation) in enumerate(
        zip(value, EXPECTED_ROLLOUT_OPERATIONS, strict=True)
    ):
        if not isinstance(row, dict) or row.get("operation") != expected_operation:
            errors.append(f"{label}: trace rollout operation {index} changed")
            return None
        raw = row.get("inventory")
        if not isinstance(raw, dict):
            errors.append(f"{label}: {expected_operation} inventory is missing")
            return None
        try:
            required = {
                "call_count",
                "logical_call_count",
                "stage_call_count",
                "logical_stage_call_count",
                "dummy_call_count",
            }
            if set(raw) != required:
                raise ValueError("inventory fields changed")
            normalized = dict(raw)
            for key in ("stage_call_count", "logical_stage_call_count"):
                stages = normalized[key]
                if not isinstance(stages, dict) or not stages:
                    raise TypeError(key)
                converted: dict[int, int] = {}
                for stage, count in stages.items():
                    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                        raise TypeError(key)
                    stage_index = int(stage)
                    if stage_index in converted:
                        raise ValueError(f"duplicate stage {stage_index}")
                    converted[stage_index] = count
                if sorted(converted) != list(range(len(converted))):
                    raise ValueError("stage indices are not contiguous from zero")
                normalized[key] = converted
            for key in ("call_count", "logical_call_count", "dummy_call_count"):
                if isinstance(normalized[key], bool) or not isinstance(normalized[key], int):
                    raise TypeError(key)
            if (
                normalized["call_count"] <= 0
                or normalized["call_count"] != normalized["logical_call_count"]
                or normalized["stage_call_count"]
                != normalized["logical_stage_call_count"]
                or sum(normalized["stage_call_count"].values())
                != normalized["call_count"]
                or normalized["dummy_call_count"] != 0
            ):
                raise ValueError("inconsistent call inventory")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{label}: invalid {expected_operation} inventory: {error}")
            return None
        result.append({"operation": expected_operation, "inventory": normalized})
    return result


def _parse_manifest(path: Path, *, node: int, errors: list[str]) -> dict[str, str]:
    try:
        rows = [line.split("=", 1) for line in path.read_text().splitlines() if "=" in line]
    except OSError as error:
        errors.append(f"node{node}: invalid launch manifest: {error}")
        return {}
    value = {key: item for key, item in rows}
    required = {
        "mode": "two-cycle-timing",
        "world_size": "16",
        "global_batch": "16",
        "rollout_batch_size": "1",
        "score_packing": "false",
        "two_cycle_count": "2",
        "two_cycle_claimable_cycle": "2",
        "two_cycle_trace_replay_policy": "FIXED_FIVE_STEP_COHORT_REPLAY",
        "claim_boundary": "STEADY_STATE_CYCLE2_UNPROFILED",
        "parameter_offload": "false",
        "activation_offload": "false",
        "optimizer_offload": "false",
        "parameter_streaming": "false",
        "model_swap": "false",
        "ema_implementation": "reference",
        "optimizer_implementation": "PYTORCH_DEFAULT",
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            errors.append(f"node{node}: launch manifest {key} changed")
    expected_algo = "Tree" if value.get("placement") == RESIDENT_WINNER else "AUTO"
    if value.get("nccl_algo") != expected_algo:
        errors.append(f"node{node}: launch NCCL algorithm changed")
    expected_chunk = "2048" if value.get("placement") == RESIDENT_WINNER else "RUNTIME_DEFAULT"
    if value.get("modulation_chunk_tokens") != expected_chunk:
        errors.append(f"node{node}: launch modulation chunk changed")
    return value


def _cgroup_counts(path: Path) -> dict[str, int]:
    try:
        return {
            key: int(value)
            for line in path.read_text().splitlines()
            if len(parts := line.split()) == 2
            for key, value in (parts,)
        }
    except (OSError, ValueError):
        return {}


def _cgroup_delta(run_root: Path, *, node: int, errors: list[str]) -> dict[str, Any]:
    root = run_root / "launcher" / f"node_{node}"
    before = _cgroup_counts(root / "cgroup_memory_events_before.txt")
    after = _cgroup_counts(root / "cgroup_memory_events_after.txt")
    if not before or not after:
        errors.append(f"node{node}: cgroup before/after receipt missing")
        return {"status": "MISSING"}
    missing = [key for key in ("oom", "oom_kill") if key not in before or key not in after]
    if missing:
        errors.append(f"node{node}: cgroup counters missing: {missing}")
        return {"status": "MISSING", "missing": missing}
    deltas = {
        key: int(after.get(key, 0) - before.get(key, 0))
        for key in sorted(set(before) | set(after))
    }
    if any(value < 0 for value in deltas.values()):
        errors.append(f"node{node}: cgroup counters moved backwards")
    for key in ("oom", "oom_kill"):
        if deltas.get(key, 0) != 0:
            errors.append(f"node{node}: cgroup {key} delta is nonzero")
    return {"status": "PASS", **{f"{key}_delta": value for key, value in deltas.items()}}


def _validate_arm(
    row: dict[str, Any], *, placement: str, rank: int, errors: list[str]
) -> None:
    arm = row.get("arm_contract", {})
    fused = arm.get("generator_nograd_lora_fused_epilogue")
    cast_cache = arm.get("generator_nograd_lora_cast_cache")
    grad_fused = arm.get("grad_lora_fused_epilogue", False)
    norm_fused = arm.get("modulated_norm_fusion", False)
    expected_name = {
        RESIDENT_WINNER: "RESIDENT_ASYNC128_TREE_CHUNK2048_FUSED_EPILOGUE",
        FSDP_BASELINE: "WORLD16_FSDP_BASELINE",
        RESIDENT_REFERENCE: "RESIDENT_REPLICATED_REFERENCE",
    }[placement]
    if arm.get("arm_contract") != expected_name:
        errors.append(f"rank{rank}: arm contract changed")
    if placement == RESIDENT_WINNER:
        if fused is not True or cast_cache is not False:
            errors.append(f"rank{rank}: Resident fused/cast-cache flags changed")
    elif fused is not False or cast_cache is not False:
        errors.append(f"rank{rank}: FSDP enabled Resident-only LoRA arm")

    runtime = row.get("runtime_contract", {})
    expected_runtime = {
        "ema_implementation": "reference",
        "nccl_algo": "Tree" if placement == RESIDENT_WINNER else "AUTO",
        "residual_storage": "cuda_out_of_place",
        "parameter_offload": False,
        "activation_offload": False,
        "optimizer_offload": False,
        "parameter_streaming": False,
        "model_swap": False,
    }
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            errors.append(f"rank{rank}: runtime contract changed: {key}")
    if len(str(runtime.get("helios_transformer_sha256", ""))) != 64:
        errors.append(f"rank{rank}: Helios runtime identity missing")

    receipt = row.get("fused_epilogue_receipt", {})
    generator = receipt.get("generator", {})
    critic = receipt.get("critic", {})
    if placement == RESIDENT_WINNER:
        expected_counts = {
            "model_call_count": 56,
            "fused_invocation_count": 17_080,
            "reference_fallback_count": 224,
        }
        if generator.get("requested") is not True or generator.get("enabled") is not True:
            errors.append(f"rank{rank}: fused epilogue is not enabled")
        for key, expected in expected_counts.items():
            if int(generator.get(key, -1)) != expected:
                errors.append(f"rank{rank}: fused epilogue {key} changed")
        if generator.get("reference_fallback_reasons") != {"BASE_DTYPE_torch.float32": 224}:
            errors.append(f"rank{rank}: unexpected fused fallback census/reason")
        if critic.get("enabled") is not False or critic.get("disabled_reason") != "GENERATOR_ONLY":
            errors.append(f"rank{rank}: Critic entered Generator fused epilogue")
    elif generator.get("requested") or generator.get("enabled") or critic.get("enabled"):
        errors.append(f"rank{rank}: unrequested fused epilogue executed")


def _teardown(
    run_root: Path, *, node: int, expected_run_id: str | None, errors: list[str]
) -> dict[str, str]:
    path = run_root / "launcher" / f"node_{node}" / "teardown_status.txt"
    try:
        value = {
            key: item for line in path.read_text().splitlines() if "=" in line
            for key, item in (line.split("=", 1),)
        }
    except OSError as error:
        errors.append(f"node{node}: teardown receipt missing: {error}")
        return {}
    if value.get("status") != "PASS" or value.get("gpu_process_count") != "0":
        errors.append(f"node{node}: teardown cleanup is not PASS")
    if value.get("owner_run_id") != expected_run_id:
        errors.append(f"node{node}: teardown run identity changed")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    if args.placement not in REGISTERED_PLACEMENTS:
        errors.append("unregistered two-cycle placement")
    if len(args.expected_head) != 40:
        errors.append("expected HEAD must be a full Git SHA")

    manifests = [
        _parse_manifest(
            args.run_root / "launcher" / f"node_{node}" / "launch_manifest.txt",
            node=node,
            errors=errors,
        )
        for node in range(2)
    ]
    for node, manifest in enumerate(manifests):
        if manifest.get("dmd_system_head") != args.expected_head:
            errors.append(f"node{node}: source HEAD changed")
        if manifest.get("placement") != args.placement:
            errors.append(f"node{node}: placement changed")
    launch_identity_keys = (
        "run_id", "dmd_system_head", "placement", "trace_hash", "bootstrap_hash",
        "two_cycle_seed", "two_cycle_arm_id", "two_cycle_correctness_sha256",
        "two_cycle_trajectory_comparison_sha256",
        "generator_nograd_lora_fused_epilogue", "generator_nograd_lora_cast_cache",
        "helios_transformer_sha256", "model_root",
    )
    launch_identities = {
        tuple(manifest.get(key) for key in launch_identity_keys)
        for manifest in manifests if manifest
    }
    if len(launch_identities) != 1:
        errors.append("node launch identities differ")
    manifest_run_id = manifests[0].get("run_id") if manifests else None
    manifest_arm_id = manifests[0].get("two_cycle_arm_id") if manifests else None

    rows: list[dict[str, Any]] = []
    hosts: dict[str, list[int]] = defaultdict(list)
    payloads_by_cycle: dict[int, list[str]] = defaultdict(list)
    generator_payloads_by_cycle: dict[int, list[str]] = defaultdict(list)
    critic_payloads_by_cycle: dict[int, list[str]] = defaultdict(list)
    payload_mapping_by_cycle: dict[int, dict[tuple[int, int], str]] = defaultdict(dict)
    occurrences: list[str] = []
    cycle_values: dict[int, list[float]] = {1: [], 2: []}
    trajectory_comparison_hashes: set[str | None] = set()
    trace_rollout_contracts: list[list[dict[str, Any]]] = []
    for rank in range(WORLD_SIZE):
        path = args.run_root / "execution" / "ranks" / f"rank_{rank:02d}.json"
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"rank{rank}: invalid timing receipt: {error}")
            continue
        rows.append(row)
        if row.get("schema_version") != "distillschedule.wan13b_two_cycle_timing_rank.v1":
            errors.append(f"rank{rank}: timing receipt schema changed")
        if row.get("status") != "PASS" or row.get("rank") != rank:
            errors.append(f"rank{rank}: rank receipt is not PASS")
        if row.get("run_id") != manifest_run_id or row.get("timing_arm_id") != manifest_arm_id:
            errors.append(f"rank{rank}: run/arm identity changed")
        if row.get("expected_head") != args.expected_head:
            errors.append(f"rank{rank}: source HEAD changed")
        if row.get("world_size") != WORLD_SIZE or row.get("placement") != args.placement:
            errors.append(f"rank{rank}: world/placement identity changed")
        comparison = row.get("trajectory_comparison_binding", {})
        trajectory_comparison_hashes.add(comparison.get("sha256"))
        if (
            comparison.get("schema_version")
            != "distillschedule.trajectory_coarse_comparison.v1"
            or comparison.get("contract_status") != "PASS"
            or comparison.get("complete") is not True
        ):
            errors.append(f"rank{rank}: cross-arm trajectory comparison is not PASS")
        compared_arms = {
            *comparison.get("reference_arms", []),
            *comparison.get("candidate_arms", []),
        }
        if row.get("timing_arm_id") not in compared_arms:
            errors.append(f"rank{rank}: arm absent from cross-arm trajectory comparison")
        if comparison.get("matched_correctness_run_id") != row.get(
            "correctness_binding", {}
        ).get("run_id"):
            errors.append(f"rank{rank}: comparison/correctness run binding changed")
        hostname, local_rank = row.get("hostname"), row.get("local_rank")
        if not isinstance(hostname, str) or not isinstance(local_rank, int):
            errors.append(f"rank{rank}: host/local-rank identity missing")
        else:
            hosts[hostname].append(local_rank)
        if row.get("profiled") is not False or row.get("profiler_status") != "NOT_RUN":
            errors.append(f"rank{rank}: profiled run cannot enter native timing")
        profile_env = row.get("profiler_environment", {})
        if profile_env.get("exact_range") not in (None, "", "0") or profile_env.get("lora_operator_attribution") != "0":
            errors.append(f"rank{rank}: profiler environment was enabled")
        if row.get("evidence_extraction") != "AFTER_BOTH_TIMED_WINDOWS":
            errors.append(f"rank{rank}: evidence extraction entered timer contract")
        if row.get("state_anchors") != "NOT_COLLECTED_IN_TIMING_MODE":
            errors.append(f"rank{rank}: timing path collected state anchors")
        if row.get("initial_versions") != {"generator": 2, "critic": 10, "ema": 10}:
            errors.append(f"rank{rank}: initial versions changed")
        if tuple(row.get("cycle_versions", ())) != EXPECTED_VERSIONS:
            errors.append(f"rank{rank}: per-cycle versions changed")
        if row.get("final_versions") != EXPECTED_VERSIONS[-1]:
            errors.append(f"rank{rank}: final versions changed")
        if row.get("updates") != {"generator": 2, "critic": 10, "ema": 10}:
            errors.append(f"rank{rank}: two-cycle update inventory changed")
        if row.get("finite_status") != [True, True]:
            errors.append(f"rank{rank}: finite gate failed")
        trace_rollout_contract = _normalize_rollout_contract(
            row.get("trace_rollout_contract"),
            label=f"rank{rank}",
            errors=errors,
        )
        if trace_rollout_contract is not None:
            trace_rollout_contracts.append(trace_rollout_contract)
            contract_hash = _canonical_hash(trace_rollout_contract)
            if row.get("trace_rollout_contract_sha256") != contract_hash:
                errors.append(f"rank{rank}: trace rollout contract SHA changed")
            if (
                row.get("matched_contract", {}).get(
                    "trace_rollout_contract_sha256"
                )
                != contract_hash
            ):
                errors.append(f"rank{rank}: matched rollout contract SHA changed")

        cycles = row.get("cycles", [])
        if len(cycles) != 2:
            errors.append(f"rank{rank}: expected exactly two cycle timers")
        else:
            expected_roles = ("WARMUP_DIAGNOSTIC", "STEADY_STATE_CLAIMABLE")
            for expected_cycle, (cycle, expected_role) in enumerate(zip(cycles, expected_roles, strict=True), start=1):
                if cycle.get("cycle") != expected_cycle or cycle.get("role") != expected_role:
                    errors.append(f"rank{rank}: cycle timer identity changed")
                duration = float(cycle.get("local_elapsed_seconds", float("nan")))
                if not math.isfinite(duration) or duration <= 0:
                    errors.append(f"rank{rank}: invalid cycle duration")
                elif expected_cycle in cycle_values:
                    cycle_values[expected_cycle].append(duration)
                if int(cycle.get("stop_unix_ns", 0)) <= int(cycle.get("start_unix_ns", 0)):
                    errors.append(f"rank{rank}: invalid cycle timer endpoints")
            if int(cycles[0].get("stop_unix_ns", 0)) > int(cycles[1].get("start_unix_ns", 0)):
                errors.append(f"rank{rank}: cycle timers overlap or are out of order")

        census = row.get("cycle_census", [])
        if len(census) != 2:
            errors.append(f"rank{rank}: missing per-cycle census")
        for index, cycle in enumerate(census, start=1):
            required = {
                "generator_updates": 1, "critic_updates": 5, "ema_updates": 5,
                "generator_score_forwards": 3, "critic_score_forwards": 5,
                "physical_rollout_batches": 5, "rollout_batch_size": 1,
            }
            for key, expected in required.items():
                if cycle.get(key) != expected:
                    errors.append(f"rank{rank}/cycle{index}: {key} changed")
            inventory = cycle.get("rollout_call_inventory", [])
            operation_keys = cycle.get("rollout_operation_keys", [])
            if len(inventory) != 6:
                errors.append(f"rank{rank}/cycle{index}: rollout inventory changed")
            elif operation_keys != list(EXPECTED_ROLLOUT_OPERATIONS):
                errors.append(f"rank{rank}/cycle{index}: rollout operation order changed")
            elif trace_rollout_contract is None:
                errors.append(
                    f"rank{rank}/cycle{index}: trace rollout contract unavailable"
                )
            else:
                for call, expected in zip(
                    inventory, trace_rollout_contract, strict=True
                ):
                    normalized = dict(call)
                    for key in ("stage_call_count", "logical_stage_call_count"):
                        normalized[key] = {int(k): int(v) for k, v in normalized.get(key, {}).items()}
                    if normalized != expected["inventory"]:
                        errors.append(
                            f"rank{rank}/cycle{index}: {expected['operation']} "
                            "trace-derived DiT census changed"
                        )
            payloads = cycle.get("sample_payload_identities", [])
            generator_payload = cycle.get("generator_trace_payload_identity")
            critic_payloads = cycle.get("critic_trace_payload_identities", [])
            occurrence_ids = cycle.get("sample_occurrence_identities", [])
            if len(payloads) != 6 or any(len(str(value)) != 64 for value in payloads):
                errors.append(f"rank{rank}/cycle{index}: trace payload inventory changed")
            payloads_by_cycle[index].extend(str(value) for value in payloads)
            for slot, value in enumerate(payloads):
                payload_mapping_by_cycle[index][(rank, slot)] = str(value)
            if (
                not isinstance(generator_payload, str)
                or len(generator_payload) != 64
                or len(critic_payloads) != 5
                or any(len(str(value)) != 64 for value in critic_payloads)
                or list(payloads) != [generator_payload, *critic_payloads]
            ):
                errors.append(f"rank{rank}/cycle{index}: Generator/Critic payload roles changed")
            generator_payloads_by_cycle[index].append(str(generator_payload))
            critic_payloads_by_cycle[index].extend(str(value) for value in critic_payloads)
            if len(occurrence_ids) != 6:
                errors.append(f"rank{rank}/cycle{index}: occurrence inventory changed")
            occurrences.extend(f"rank={rank}/{value}" for value in occurrence_ids)

        binding = row.get("correctness_binding", {})
        if binding.get("status") != "PASS" or len(str(binding.get("sha256", ""))) != 64:
            errors.append(f"rank{rank}: invalid correctness binding")
        _validate_arm(row, placement=args.placement, rank=rank, errors=errors)

        memory = row.get("memory", {})
        required_memory = (
            "allocated_bytes", "reserved_bytes", "max_allocated_bytes",
            "max_reserved_bytes", "driver_free_bytes", "driver_total_bytes",
        )
        for key in required_memory:
            if not isinstance(memory.get(key), int) or int(memory[key]) < 0:
                errors.append(f"rank{rank}: memory field missing/invalid: {key}")
        construction = row.get("construction_seconds")
        if not isinstance(construction, (int, float)) or not math.isfinite(float(construction)) or float(construction) <= 0:
            errors.append(f"rank{rank}: construction time missing/invalid")

    if len(rows) != WORLD_SIZE:
        errors.append(f"rank receipt count={len(rows)}, expected={WORLD_SIZE}")
    if len(hosts) != 2 or any(sorted(local_ranks) != list(range(8)) for local_ranks in hosts.values()):
        errors.append(f"world topology is not two hosts x local-rank0..7: {dict(hosts)}")
    for cycle in (1, 2):
        payloads = payloads_by_cycle[cycle]
        if len(payloads) != 96 or len(set(payloads)) != 96:
            errors.append(f"cycle{cycle}: expected 96 unique rank-qualified sample payloads")
        if len(set(generator_payloads_by_cycle[cycle])) != 16:
            errors.append(f"cycle{cycle}: expected 16 unique Generator payloads")
        if len(critic_payloads_by_cycle[cycle]) != 80 or len(set(critic_payloads_by_cycle[cycle])) != 80:
            errors.append(f"cycle{cycle}: expected 80 unique Critic payloads")
        if len(cycle_values[cycle]) != WORLD_SIZE:
            errors.append(f"cycle{cycle}: duration rank count changed")
    if payload_mapping_by_cycle[1] != payload_mapping_by_cycle[2]:
        errors.append(
            "immutable five-step payload rank-slot mapping was not replayed in cycle2"
        )
    if len(occurrences) != 192 or len(set(occurrences)) != 192:
        errors.append("expected 192 distinct cycle-qualified sample occurrences")

    rollout_contract_hashes = {
        _canonical_hash(contract) for contract in trace_rollout_contracts
    }
    trace_rollout_contract = (
        trace_rollout_contracts[0]
        if len(trace_rollout_contracts) == WORLD_SIZE
        and len(rollout_contract_hashes) == 1
        else None
    )
    if trace_rollout_contract is None:
        errors.append("rank trace-derived rollout contracts differ or are incomplete")

    matched_hashes = {row.get("workload_fingerprint") for row in rows}
    matched_contract_hashes = {
        _canonical_hash(row.get("matched_contract", {})) for row in rows
    }
    if len(matched_hashes) != 1 or None in matched_hashes or len(matched_contract_hashes) != 1:
        errors.append("rank matched workload fingerprints differ")
    elif next(iter(matched_hashes)) != next(iter(matched_contract_hashes)):
        errors.append("workload fingerprint does not match matched contract")
    matched_contract = rows[0].get("matched_contract") if rows else None
    if (
        trace_rollout_contract is not None
        and isinstance(matched_contract, dict)
        and matched_contract.get("trace_rollout_contract_sha256")
        != _canonical_hash(trace_rollout_contract)
    ):
        errors.append("world matched rollout contract SHA changed")

    implementation_hashes = {row.get("implementation_fingerprint") for row in rows}
    arm_hashes = {_canonical_hash(row.get("arm_contract", {})) for row in rows}
    if len(implementation_hashes) != 1 or None in implementation_hashes or len(arm_hashes) != 1:
        errors.append("rank implementation fingerprints differ")
    elif next(iter(implementation_hashes)) != next(iter(arm_hashes)):
        errors.append("implementation fingerprint does not match arm contract")
    implementation_contract = rows[0].get("arm_contract") if rows else None

    config_fingerprints = {row.get("configuration_fingerprint") for row in rows}
    if len(config_fingerprints) != 1 or None in config_fingerprints:
        errors.append("rank configuration fingerprints differ")
    correctness_hashes = {row.get("correctness_binding", {}).get("sha256") for row in rows}
    if len(correctness_hashes) != 1 or None in correctness_hashes:
        errors.append("rank correctness bindings differ")
    elif manifests and next(iter(correctness_hashes)) != manifests[0].get("two_cycle_correctness_sha256"):
        errors.append("rank correctness binding differs from launch manifest")
    if len(trajectory_comparison_hashes) != 1 or None in trajectory_comparison_hashes:
        errors.append("rank trajectory comparison bindings differ")
    elif manifests and next(iter(trajectory_comparison_hashes)) != manifests[0].get(
        "two_cycle_trajectory_comparison_sha256"
    ):
        errors.append("rank trajectory comparison binding differs from launch manifest")

    cgroup = [_cgroup_delta(args.run_root, node=node, errors=errors) for node in range(2)]
    teardown = [
        _teardown(
            args.run_root,
            node=node,
            expected_run_id=manifest_run_id,
            errors=errors,
        )
        for node in range(2)
    ]
    profiler_artifacts = [
        str(path) for suffix in ("*.nsys-rep", "*.sqlite")
        for path in args.run_root.rglob(suffix)
    ]
    if profiler_artifacts:
        errors.append("native two-cycle run unexpectedly contains profiler artifacts")

    memory_rows = [row.get("memory", {}) for row in rows]
    construction = [float(row.get("construction_seconds", 0.0)) for row in rows]
    formal = not errors and len(rows) == WORLD_SIZE
    summary = {
        "schema_version": "distillschedule.wan13b_two_cycle_timing_summary.v1",
        "status": "PASS" if formal else "FAIL",
        "errors": errors,
        "run_id": manifest_run_id or args.run_root.name,
        "run_root": str(args.run_root.resolve()),
        "expected_head": args.expected_head,
        "placement": args.placement,
        "rank_count": len(rows),
        "host_topology": {host: sorted(values) for host, values in sorted(hosts.items())},
        "cycles_in_same_process": 2,
        "matched_contract": matched_contract,
        "trace_rollout_contract": trace_rollout_contract,
        "workload_fingerprint": next(iter(matched_hashes)) if len(matched_hashes) == 1 else None,
        "implementation_contract": implementation_contract,
        "implementation_fingerprint": next(iter(implementation_hashes)) if len(implementation_hashes) == 1 else None,
        "configuration_fingerprint": next(iter(config_fingerprints)) if len(config_fingerprints) == 1 else None,
        "correctness_receipt_sha256": next(iter(correctness_hashes)) if len(correctness_hashes) == 1 else None,
        "trajectory_comparison_sha256": (
            next(iter(trajectory_comparison_hashes))
            if len(trajectory_comparison_hashes) == 1
            else None
        ),
        "sample_identity_contract": {
            "rank_qualified_payloads_per_cycle": 96,
            "critic_payloads_per_cycle": 80,
            "cycle_qualified_occurrences": len(occurrences),
            "payload_cohort_replayed": payload_mapping_by_cycle[1]
            == payload_mapping_by_cycle[2],
            "ordered_rank_slot_mapping_replayed": payload_mapping_by_cycle[1]
            == payload_mapping_by_cycle[2],
        },
        "cycle_1": {
            "role": "WARMUP_DIAGNOSTIC",
            "rank_local_seconds": cycle_values[1],
            "world_max_seconds": max(cycle_values[1]) if cycle_values[1] else None,
            "claimable": False,
        },
        "cycle_2": {
            "role": "STEADY_STATE_CLAIMABLE",
            "rank_local_seconds": cycle_values[2],
            "world_max_seconds": max(cycle_values[2]) if cycle_values[2] else None,
            "claimable": formal,
        },
        "speed_claim_denominator": "CYCLE2_WORLD_MAX_ONLY",
        "cycle_1_excluded_from_speed_claim": True,
        "initial_versions": {"generator": 2, "critic": 10, "ema": 10},
        "cycle_versions": list(EXPECTED_VERSIONS),
        "final_versions": EXPECTED_VERSIONS[-1],
        "updates": {"generator": 2, "critic": 10, "ema": 10},
        "construction": {
            "rank_local_seconds": construction,
            "world_max_seconds": max(construction) if construction else None,
            "included_in_cycle_timers": False,
        },
        "memory": {
            "max_allocated_bytes": max((int(row.get("max_allocated_bytes", 0)) for row in memory_rows), default=None),
            "max_reserved_bytes": max((int(row.get("max_reserved_bytes", 0)) for row in memory_rows), default=None),
            "min_driver_free_bytes": min((int(row.get("driver_free_bytes", 0)) for row in memory_rows), default=None),
        },
        "resource_status": {
            "cgroup_memory_events": cgroup,
            "teardown": teardown,
            "profiler_artifacts": profiler_artifacts,
        },
        "profiled": False,
        "profiler_status": "NOT_RUN",
        "formal_eligibility": formal,
        "timing_claim": (
            "same process and one construction; cycle1 is warm-up only; "
            "cycle2 world-max rank-local wall is the sole claimable measurement"
        ),
        "rank_receipts": [
            str(args.run_root / "execution" / "ranks" / f"rank_{rank:02d}.json")
            for rank in range(WORLD_SIZE)
        ],
    }
    summary["summary_contract_sha256"] = _canonical_hash({
        "expected_head": summary["expected_head"],
        "placement": summary["placement"],
        "workload_fingerprint": summary["workload_fingerprint"],
        "implementation_fingerprint": summary["implementation_fingerprint"],
        "speed_claim_denominator": summary["speed_claim_denominator"],
    })
    _atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
