#!/usr/bin/env python3
"""Fail-closed summary for one completed Wan1.3B trajectory run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from distillschedule.evidence.trajectory_coarse import load_trajectory_jsonl
from distillschedule.evidence.trajectory_state_sketch import (
    SCHEMA_VERSION as STATE_SKETCH_SCHEMA_VERSION,
)


WORLD_SIZE = 16
REGISTERED_CYCLE_COUNTS = frozenset({1, 2, 5, 50})
RESIDENT_WINNER = "resident-async128-tree-chunk2048"
FSDP_BASELINE = "world16-fsdp"
REGISTERED_PLACEMENTS = frozenset({RESIDENT_WINNER, FSDP_BASELINE})
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


def _expected_call_census(contract: list[dict[str, Any]]) -> dict[str, int]:
    inventories = [row["inventory"] for row in contract]
    generator, *critics = inventories
    result = {
        "generator_update_rollout_dit": int(generator["call_count"]),
        "generator_critic_rollout_dit": sum(
            int(inventory["call_count"]) for inventory in critics
        ),
        "generator_score_forwards": 3,
        "critic_score_forwards": 5,
        "physical_rollout_batches": 5,
        "rollout_batch_size": 1,
        "rollout_inventory_count": 6,
        "rollout_dummy_calls_total": sum(
            int(inventory["dummy_call_count"]) for inventory in inventories
        ),
    }
    role_names = (
        "generator_update",
        *(f"critic_step_{step}" for step in range(10, 15)),
    )
    for role, inventory in zip(role_names, inventories, strict=True):
        result[f"{role}_rollout_dit"] = int(inventory["call_count"])
        for stage, count in sorted(inventory["stage_call_count"].items()):
            result[f"{role}_rollout_stage{stage}_dit"] = int(count)
    return result


def _read_manifest(path: Path, *, node: int, errors: list[str]) -> dict[str, str]:
    try:
        value = {
            key: item
            for line in path.read_text().splitlines()
            if "=" in line
            for key, item in (line.split("=", 1),)
        }
    except OSError as error:
        errors.append(f"node{node}: invalid launch manifest: {error}")
        return {}
    required = {
        "mode": "trajectory",
        "world_size": "16",
        "global_batch": "16",
        "rollout_batch_size": "1",
        "score_packing": "false",
        "parameter_offload": "false",
        "activation_offload": "false",
        "optimizer_offload": "false",
        "parameter_streaming": "false",
        "model_swap": "false",
        "claim_boundary": "DIAGNOSTIC_ONLY",
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            errors.append(f"node{node}: launch manifest {key} changed")
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


def _validate_runtime_build(
    final: dict[str, Any], *, placement: str, cycles: int, rank: int, errors: list[str]
) -> dict[str, Any]:
    build = final.get("build", {})
    static = build.get("static_residency", {})
    required_static = {
        "parameter_offload": False,
        "activation_offload": False,
        "optimizer_offload": False,
        "frozen_parameter_streaming": False,
        "role_transition": False,
        "model_swap": False,
        "residual_storage": "cuda_out_of_place",
        "ema_device": "cuda",
    }
    for key, expected in required_static.items():
        if static.get(key) != expected:
            errors.append(f"rank{rank}: static residency {key} changed")
    runtime_hashes: set[str] = set()
    for role in ("generator", "critic"):
        component = build.get(role, {})
        metadata = component.get("metadata", {})
        placement_receipt = component.get("placement", {})
        residual = metadata.get("chunked_residual_runtime", {})
        if metadata.get("chunked_residual_storage") != "cuda_out_of_place":
            errors.append(f"rank{rank}/{role}: residual storage is not CUDA")
        if residual.get("requested_storage") != "cuda_out_of_place" or residual.get("storage_capable") is not True:
            errors.append(f"rank{rank}/{role}: CUDA residual runtime contract changed")
        source = residual.get("transformer_source_sha256")
        if isinstance(source, str) and len(source) == 64:
            runtime_hashes.add(source)
        else:
            errors.append(f"rank{rank}/{role}: residual runtime identity missing")
        if placement_receipt.get("trainable_parameter_dtypes") != ["torch.float32"]:
            errors.append(f"rank{rank}/{role}: LoRA dtype changed")
        if "torch.bfloat16" not in placement_receipt.get("frozen_parameter_dtypes", []):
            errors.append(f"rank{rank}/{role}: frozen dtype changed")
        if placement_receipt.get("gradient_reduction") != "SUM_THEN_DIVIDE_WORLD_SIZE":
            errors.append(f"rank{rank}/{role}: gradient mean convention changed")
        if component.get("engine", {}).get("optimizer_implementation") != "PYTORCH_DEFAULT":
            errors.append(f"rank{rank}/{role}: optimizer implementation changed")
        dtype = component.get("lora_compute_dtype_contract", {})
        if dtype.get("frozen_base") != "torch.bfloat16" or dtype.get("trainable_lora") != "torch.float32":
            errors.append(f"rank{rank}/{role}: runtime compute dtype contract changed")
        if placement == RESIDENT_WINNER:
            modulation = metadata.get("modulation_chunk_runtime", {})
            if modulation.get("effective_tokens") != [2048] or modulation.get("block_count") != 30 or modulation.get("all_blocks_match") is not True:
                errors.append(f"rank{rank}/{role}: modulation chunk2048 contract changed")
    if len(runtime_hashes) != 1:
        errors.append(f"rank{rank}: Generator/Critic residual runtime identities differ")

    generator = (
        build.get("generator", {})
        .get("metadata", {})
        .get("generator_nograd_lora_fused_epilogue", {})
    )
    critic = (
        build.get("critic", {})
        .get("metadata", {})
        .get("generator_nograd_lora_fused_epilogue", {})
    )
    if placement == RESIDENT_WINNER:
        from distillschedule.runtime.wan13b_rollout_shadow import expected_epilogue_counts

        shadow_enabled = bool(final.get("generator_rollout_merged_shadow_receipt", {}).get("enabled", False))
        expected_counts = expected_epilogue_counts(
            cycles=cycles, shadow_enabled=shadow_enabled, diagnostic_reference=shadow_enabled,
            graph_receipt=build.get("generator", {}).get("metadata", {}).get("generator_rollout_cuda_graph", {}),
        )
        if generator.get("requested") is not True or generator.get("enabled") is not True:
            errors.append(f"rank{rank}: fused epilogue is not enabled")
        for key, expected in expected_counts.items():
            if int(generator.get(key, -1)) != expected:
                errors.append(f"rank{rank}: fused epilogue {key} changed")
        if generator.get("reference_fallback_reasons") != {
            "BASE_DTYPE_torch.float32": expected_counts["reference_fallback_count"]
        }:
            errors.append(f"rank{rank}: fused epilogue fallback reason changed")
        if critic.get("enabled") is not False or critic.get("disabled_reason") != "GENERATOR_ONLY":
            errors.append(f"rank{rank}: Critic entered Generator fused epilogue")
    elif generator.get("requested") or generator.get("enabled") or critic.get("enabled"):
        errors.append(f"rank{rank}: FSDP correctness run enabled Resident-only epilogue")
    from distillschedule.runtime.wan13b_norm_policy import validate as validate_norm
    norm_enabled = bool(final.get("modulated_norm_fusion_receipt", {}).get("enabled", False))
    if "modulated_norm_fusion_receipt" in final:
        try:
            norm_receipt = validate_norm(build, enabled=norm_enabled)
        except AssertionError as error:
            errors.append(f"rank{rank}: {error}")
            norm_receipt = {"enabled": norm_enabled, "status": "FAIL", "error": str(error)}
    else:
        norm_receipt = {"enabled": False, "status": "NOT_REQUESTED_LEGACY_RECEIPT"}
        if any(
            build.get(role, {}).get("metadata", {}).get("modulated_norm_fusion", {}).get("enabled")
            for role in ("generator", "critic")
        ):
            errors.append(f"rank{rank}: enabled normalization fusion has no final receipt")
    from distillschedule.runtime.wan13b_rollout_shadow import validate as validate_shadow

    try:
        shadow_receipt = validate_shadow(
            build,
            enabled=bool(final.get("generator_rollout_merged_shadow_receipt", {}).get("enabled", False)),
            cycles=cycles,
            diagnostic_reference=bool(final.get("generator_rollout_merged_shadow_receipt", {}).get("enabled", False)),
        )
    except AssertionError as error:
        errors.append(f"rank{rank}: {error}")
        shadow_receipt = {"status": "FAIL", "error": str(error)}
    from distillschedule.runtime.wan13b_rollout_cuda_graph import validate as validate_graph

    graph_enabled = bool(final.get("generator_rollout_cuda_graph_receipt", {}).get("enabled", False))
    try:
        graph_receipt = validate_graph(build, enabled=graph_enabled, cycles=cycles,
                                       diagnostic_reference=graph_enabled)
    except AssertionError as error:
        errors.append(f"rank{rank}: {error}")
        graph_receipt = {"status": "FAIL", "error": str(error)}
    return {
        "helios_transformer_sha256": next(iter(runtime_hashes)) if len(runtime_hashes) == 1 else None,
        "fused_epilogue": generator,
        "modulated_norm_fusion": norm_receipt,
        "generator_rollout_merged_shadow": shadow_receipt,
        "generator_rollout_cuda_graph": graph_receipt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    if args.cycles not in REGISTERED_CYCLE_COUNTS:
        errors.append("trajectory cycles must be one of 1, 2, 5, or 50")
    if args.placement not in REGISTERED_PLACEMENTS:
        errors.append("trajectory placement is not a registered comparison arm")

    manifests = [
        _read_manifest(
            args.run_root / "launcher" / f"node_{node}" / "launch_manifest.txt",
            node=node,
            errors=errors,
        )
        for node in range(2)
    ]
    for node, manifest in enumerate(manifests):
        if manifest.get("dmd_system_head") != args.expected_head:
            errors.append(f"node{node}: launch HEAD differs from expected HEAD")
        if manifest.get("trajectory_cycles") != str(args.cycles):
            errors.append(f"node{node}: launch trajectory cycle count changed")
        if manifest.get("placement") != args.placement:
            errors.append(f"node{node}: launch placement changed")
        if manifest.get("modulated_norm_fusion", "0") not in {"0", "1"}:
            errors.append(f"node{node}: invalid normalization fusion launch flag")
        if manifest.get("generator_rollout_merged_shadow", "0") not in {"0", "1"}:
            errors.append(f"node{node}: invalid merged shadow launch flag")
        if manifest.get("generator_rollout_cuda_graph", "0") not in {"0", "1"}:
            errors.append(f"node{node}: invalid CUDA Graph launch flag")
    norm_requested = manifests[0].get("modulated_norm_fusion", "0") == "1"
    shadow_requested = manifests[0].get("generator_rollout_merged_shadow", "0") == "1"
    graph_requested = manifests[0].get("generator_rollout_cuda_graph", "0") == "1"
    if graph_requested and (args.placement != RESIDENT_WINNER or shadow_requested):
        errors.append("CUDA Graph requires Resident without merged weights")
    if norm_requested and args.placement != RESIDENT_WINNER:
        errors.append("normalization fusion is not registered for this placement")
    if shadow_requested and args.placement != RESIDENT_WINNER:
        errors.append("merged shadow is registered only for the Resident arm")
    identity_keys = (
        "run_id", "dmd_system_head", "trajectory_seed", "trajectory_arm_id",
        "trace_hash", "bootstrap_hash", "generator_nograd_lora_fused_epilogue",
        "generator_nograd_lora_cast_cache", "nccl_algo",
        "grad_lora_fused_epilogue", "tf32_cublas_override",
        "modulated_norm_fusion",
        "generator_rollout_merged_shadow",
        "generator_rollout_cuda_graph",
    )
    manifest_identities = {
        tuple(manifest.get(key) for key in identity_keys) for manifest in manifests if manifest
    }
    if len(manifest_identities) != 1:
        errors.append("node launch identities differ")

    rank_rows: dict[int, list[Any]] = {}
    finals: dict[int, dict[str, Any]] = {}
    payloads_by_cycle: dict[int, list[str]] = defaultdict(list)
    generator_payloads_by_cycle: dict[int, list[str]] = defaultdict(list)
    critic_payloads_by_cycle: dict[int, list[str]] = defaultdict(list)
    payload_mapping_by_cycle: dict[int, dict[tuple[int, int], str]] = defaultdict(dict)
    occurrences: list[str] = []
    matched_contracts: list[dict[str, Any]] = []
    implementations: list[dict[str, Any]] = []
    trace_rollout_contracts: list[list[dict[str, Any]]] = []
    hosts: dict[str, list[int]] = defaultdict(list)
    runtime_receipts: list[dict[str, Any]] = []
    for rank in range(WORLD_SIZE):
        trajectory_path = args.run_root / "execution" / "trajectory" / f"rank_{rank:02d}.jsonl"
        try:
            rows = load_trajectory_jsonl(trajectory_path)
        except (TypeError, ValueError, OSError) as error:
            errors.append(f"rank{rank}: invalid trajectory JSONL: {error}")
            continue
        rank_rows[rank] = rows
        if len(rows) != args.cycles:
            errors.append(f"rank{rank}: cycles {len(rows)} != {args.cycles}")
        for cycle, row in enumerate(rows, 1):
            if row.rank != rank or row.cycle != cycle:
                errors.append(f"rank{rank}: non-contiguous cycle identity")
            if row.implementation.get("placement") != args.placement:
                errors.append(f"rank{rank}: placement changed")
            if dict(row.cycle_updates) != {"generator": 1, "critic": 5, "ema": 5}:
                errors.append(f"rank{rank}: cycle{cycle} update inventory changed")
            if dict(row.cumulative_updates) != {"generator": cycle, "critic": 5 * cycle, "ema": 5 * cycle}:
                errors.append(f"rank{rank}: cycle{cycle} cumulative inventory changed")
            if dict(row.versions) != {"generator": 2 + cycle, "critic": 10 + 5 * cycle, "ema": 10 + 5 * cycle}:
                errors.append(f"rank{rank}: cycle{cycle} versions changed")
            if not row.finite or row.runtime_status != "PASS":
                errors.append(f"rank{rank}: cycle{cycle} is non-finite/failed")
            if len(row.sample_identities) != 6 or any(len(str(item)) != 64 for item in row.sample_identities):
                errors.append(f"rank{rank}: cycle{cycle} trace payload inventory changed")
            payloads_by_cycle[cycle].extend(str(item) for item in row.sample_identities)
            for slot, item in enumerate(row.sample_identities):
                payload_mapping_by_cycle[cycle][(rank, slot)] = str(item)
            if row.sample_identities:
                generator_payloads_by_cycle[cycle].append(str(row.sample_identities[0]))
                critic_payloads_by_cycle[cycle].extend(
                    str(item) for item in row.sample_identities[1:]
                )
            occurrences.extend(
                f"cycle={cycle}/rank={rank}/slot={slot}/payload={payload}"
                for slot, payload in enumerate(row.sample_identities)
            )
            matched_contracts.append(dict(row.matched_contract))
            implementations.append(dict(row.implementation))

        final_path = args.run_root / "execution" / "ranks" / f"rank_{rank:02d}.json"
        try:
            final = json.loads(final_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"rank{rank}: invalid final rank receipt: {error}")
            continue
        finals[rank] = final
        for trajectory_row in rows:
            observed = trajectory_row.implementation.get("modulated_norm_fusion", "false")
            if observed not in {"false", "true"} or (observed == "true") != norm_requested:
                errors.append(f"rank{rank}: normalization fusion launch/trajectory binding changed")
        norm_receipt = final.get("modulated_norm_fusion_receipt", {})
        if norm_receipt.get("enabled", False) is not norm_requested:
            errors.append(f"rank{rank}: normalization fusion launch/runtime binding changed")
        if norm_requested and not norm_receipt:
            errors.append(f"rank{rank}: requested normalization fusion has no runtime proof")
        for trajectory_row in rows:
            observed = trajectory_row.implementation.get("generator_rollout_merged_shadow", "false")
            if observed not in {"false", "true"} or (observed == "true") != shadow_requested:
                errors.append(f"rank{rank}: merged shadow launch/trajectory binding changed")
        shadow_receipt = final.get("generator_rollout_merged_shadow_receipt", {})
        if shadow_receipt.get("enabled", False) is not shadow_requested:
            errors.append(f"rank{rank}: merged shadow launch/runtime binding changed")
        if shadow_requested and not shadow_receipt:
            errors.append(f"rank{rank}: requested merged shadow has no runtime proof")
        for trajectory_row in rows:
            observed = trajectory_row.implementation.get("generator_rollout_cuda_graph", "false")
            if observed not in {"false", "true"} or (observed == "true") != graph_requested:
                errors.append(f"rank{rank}: CUDA Graph launch/trajectory binding changed")
        graph_receipt = final.get("generator_rollout_cuda_graph_receipt", {})
        if graph_receipt.get("enabled", False) is not graph_requested:
            errors.append(f"rank{rank}: CUDA Graph launch/runtime binding changed")
        if graph_requested and not graph_receipt:
            errors.append(f"rank{rank}: requested CUDA Graph has no runtime proof")
        if rows and rows[0].implementation.get("grad_lora_fused_epilogue") == "true":
            for role in ("generator", "critic"):
                observed = final.get("grad_epilogue_receipt", {}).get(role, {})
                if observed.get("enabled") is not True or int(observed.get("fused_invocation_count", 0)) <= 0:
                    errors.append(f"rank{rank}: {role} grad epilogue did not execute")
        if final.get("status") != "PASS" or final.get("rank") != rank:
            errors.append(f"rank{rank}: final rank receipt is not PASS")
        if final.get("placement") != args.placement or final.get("world_size") != WORLD_SIZE:
            errors.append(f"rank{rank}: final placement/world changed")
        if final.get("completed_cycles") != args.cycles:
            errors.append(f"rank{rank}: final completed-cycle count changed")
        expected_final = {"generator": 2 + args.cycles, "critic": 10 + 5 * args.cycles, "ema": 10 + 5 * args.cycles}
        if final.get("final_versions") != expected_final:
            errors.append(f"rank{rank}: final versions changed")
        if final.get("updates") != {"generator": args.cycles, "critic": 5 * args.cycles, "ema": 5 * args.cycles}:
            errors.append(f"rank{rank}: final update inventory changed")
        trace_rollout_contract = _normalize_rollout_contract(
            final.get("trace_rollout_contract"),
            label=f"rank{rank}",
            errors=errors,
        )
        if trace_rollout_contract is not None:
            trace_rollout_contracts.append(trace_rollout_contract)
            observed_hash = final.get("trace_rollout_contract_sha256")
            expected_hash = _canonical_hash(trace_rollout_contract)
            if observed_hash != expected_hash:
                errors.append(f"rank{rank}: trace rollout contract SHA changed")
        local_rank = final.get("local_rank")
        hostname = final.get("hostname")
        if not isinstance(hostname, str) or not isinstance(local_rank, int):
            errors.append(f"rank{rank}: host/local-rank identity missing")
        else:
            hosts[hostname].append(local_rank)
        runtime_receipts.append(
            _validate_runtime_build(final, placement=args.placement, cycles=args.cycles, rank=rank, errors=errors)
        )

    if len(rank_rows) != WORLD_SIZE or len(finals) != WORLD_SIZE:
        errors.append("trajectory requires 16/16 rank receipts")
    if len(hosts) != 2 or any(sorted(local_ranks) != list(range(8)) for local_ranks in hosts.values()):
        errors.append(f"trajectory topology is not two hosts x local-rank0..7: {dict(hosts)}")
    for cycle in range(1, args.cycles + 1):
        values = payloads_by_cycle.get(cycle, [])
        if len(values) != 96 or len(set(values)) != 96:
            errors.append(f"cycle{cycle}: expected 96 unique rank-qualified sample payloads")
        if len(set(generator_payloads_by_cycle[cycle])) != 16:
            errors.append(f"cycle{cycle}: expected 16 unique Generator payloads")
        critic_values = critic_payloads_by_cycle[cycle]
        if len(critic_values) != 80 or len(set(critic_values)) != 80:
            errors.append(f"cycle{cycle}: expected 80 unique Critic payloads")
    if args.cycles >= 2 and any(
        payload_mapping_by_cycle[cycle] != payload_mapping_by_cycle[1]
        for cycle in range(2, args.cycles + 1)
    ):
        errors.append(
            "fixed five-step payload rank-slot mapping was not replayed across cycles"
        )
    if len(occurrences) != 96 * args.cycles or len(set(occurrences)) != 96 * args.cycles:
        errors.append("cycle-qualified sample occurrence inventory changed")

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
    else:
        required_census = _expected_call_census(trace_rollout_contract)
        for rank, rows in rank_rows.items():
            for row in rows:
                if dict(row.call_census) != required_census:
                    errors.append(
                        f"rank{rank}: cycle{row.cycle} trace-derived call census changed"
                    )

    contract_hashes = {_canonical_hash(value) for value in matched_contracts}
    raw_matched = matched_contracts[0] if len(contract_hashes) == 1 and matched_contracts else None
    matched_contract = (
        {
            "bootstrap_hash": raw_matched.get("bootstrap_hash"),
            "model_identity_sha256": raw_matched.get("model_identity_sha256"),
            "trace_bundle_hash": raw_matched.get("trace_bundle_hash"),
            "trace_binding_hash": raw_matched.get("trace_binding_hash"),
            "trace_rollout_contract_sha256": raw_matched.get(
                "trace_rollout_contract_sha256"
            ),
            "trace_replay_policy": raw_matched.get("trace_replay_policy"),
            "negative_prompt_manifest_hash": raw_matched.get("negative_prompt_manifest_hash"),
            "seed": int(raw_matched.get("seed", -1)),
            "global_batch": int(raw_matched.get("global_batch", -1)),
            "updates": raw_matched.get("updates"),
            "score_packing": raw_matched.get("score_packing", "").lower() == "true",
            "rollout_batch_size": int(raw_matched.get("rollout_batch_size", -1)),
            **{key: raw_matched[key] for key in (
                "tf32_cublas_override", "float32_matmul_precision",
                "cuda_matmul_allow_tf32", "cudnn_allow_tf32") if key in raw_matched},
        }
        if raw_matched is not None
        else None
    )
    if len(contract_hashes) != 1:
        errors.append("rank/cycle matched workload identities differ")
    expected_matched = {
        "trace_replay_policy": "FIXED_FIVE_STEP_COHORT_REPLAY",
        "global_batch": 16,
        "updates": "G1_C5_EMA5",
        "score_packing": False,
        "rollout_batch_size": 1,
    }
    if matched_contract is not None:
        for key, expected in expected_matched.items():
            if matched_contract.get(key) != expected:
                errors.append(f"matched workload contract changed: {key}")
        for key in (
            "bootstrap_hash", "model_identity_sha256", "trace_bundle_hash",
            "trace_binding_hash", "negative_prompt_manifest_hash",
            "trace_rollout_contract_sha256",
        ):
            if len(str(matched_contract.get(key, ""))) != 64:
                errors.append(f"matched workload identity missing: {key}")
        if (
            trace_rollout_contract is not None
            and matched_contract.get("trace_rollout_contract_sha256")
            != _canonical_hash(trace_rollout_contract)
        ):
            errors.append("matched workload trace rollout contract SHA changed")

    implementation_hashes = {_canonical_hash(value) for value in implementations}
    raw_implementation = implementations[0] if len(implementation_hashes) == 1 and implementations else None
    implementation_contract = (
        {
            "placement": raw_implementation.get("placement"),
            "arm_id": raw_implementation.get("arm_id"),
            "trajectory_arm_id": raw_implementation.get("trajectory_arm_id"),
            "arm_contract": raw_implementation.get("arm_contract"),
            "generator_nograd_lora_fused_epilogue": (
                raw_implementation.get("generator_nograd_lora_fused_epilogue", "").lower() == "true"
            ),
            "generator_nograd_lora_cast_cache": (
                raw_implementation.get("generator_nograd_lora_cast_cache", "").lower() == "true"
            ),
            "grad_lora_fused_epilogue": raw_implementation.get("grad_lora_fused_epilogue", "false") == "true",
            "modulated_norm_fusion": raw_implementation.get("modulated_norm_fusion", "false") == "true",
            "generator_rollout_merged_shadow": raw_implementation.get("generator_rollout_merged_shadow", "false") == "true",
            "generator_rollout_cuda_graph": raw_implementation.get("generator_rollout_cuda_graph", "false") == "true",
            "fused_input_lora_a": raw_implementation.get("fused_input_lora_a", "false") == "true",
            "layout_only_ablation": raw_implementation.get("layout_only_ablation", "false") == "true",
            "critic_grad_cuda_graph": raw_implementation.get("critic_grad_cuda_graph", "false") == "true",
            "score_cuda_graph": raw_implementation.get("score_cuda_graph", "false") == "true",
            "worker": raw_implementation.get("worker"),
        }
        if raw_implementation is not None
        else None
    )
    if len(implementation_hashes) != 1:
        errors.append("rank/cycle implementation identities differ")
    expected_arm = (
        ("RESIDENT_ASYNC128_TREE_CHUNK2048_LAYOUT_ONLY" if implementation_contract and implementation_contract.get("layout_only_ablation",False) else "RESIDENT_ASYNC128_TREE_CHUNK2048_FUSED_EPILOGUE")
        if args.placement == RESIDENT_WINNER
        else "WORLD16_FSDP_BASELINE"
    )
    if implementation_contract is not None and implementation_contract.get("arm_contract") != expected_arm:
        errors.append("trajectory implementation arm contract changed")
    if implementation_contract is not None:
        if implementation_contract.get("placement") != args.placement:
            errors.append("trajectory implementation placement changed")
        if not implementation_contract.get("trajectory_arm_id"):
            errors.append("trajectory implementation arm id is missing")
        expected_fused = args.placement == RESIDENT_WINNER and not implementation_contract.get("layout_only_ablation",False)
        if implementation_contract.get("generator_nograd_lora_fused_epilogue") is not expected_fused:
            errors.append("trajectory fused-epilogue arm flag changed")
        from distillschedule.runtime.wan13b_cast_graph_composition import check as check_cast_composition
        try:
            check_cast_composition(args.placement, cast_cache=implementation_contract.get("generator_nograd_lora_cast_cache",False), fused=implementation_contract.get("generator_nograd_lora_fused_epilogue",False), graph=implementation_contract.get("generator_rollout_cuda_graph",False), input_gemm=implementation_contract.get("fused_input_lora_a",False), shadow=implementation_contract.get("generator_rollout_merged_shadow",False))
        except ValueError as exc: errors.append(str(exc))
        if args.placement == FSDP_BASELINE and implementation_contract["grad_lora_fused_epilogue"]:
            errors.append("FSDP baseline enabled Resident grad epilogue")

    if implementation_contract is not None:
        from distillschedule.runtime.wan13b_layout_ablation import check as check_layout
        try:
            check_layout(args.placement,enabled=implementation_contract.get("layout_only_ablation",False),flags={k:implementation_contract.get(k,False) for k in ("generator_nograd_lora_fused_epilogue","generator_nograd_lora_cast_cache","grad_lora_fused_epilogue","modulated_norm_fusion","generator_rollout_merged_shadow","generator_rollout_cuda_graph","fused_input_lora_a","score_cuda_graph","critic_grad_cuda_graph")})
        except ValueError as exc:errors.append(str(exc))
    expected_anchor_cycles = sorted(
        {cycle for cycle in (1, 10, 25, 50, args.cycles) if cycle <= args.cycles}
    )
    anchors: list[dict[str, Any]] = []
    for cycle in expected_anchor_cycles:
        path = args.run_root / "execution" / "trajectory" / "anchors" / f"cycle_{cycle:02d}_global.json"
        try:
            anchor = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cycle{cycle}: invalid numeric state anchor: {error}")
            continue
        recorded = anchor.get("anchor_sha256")
        canonical = dict(anchor)
        canonical.pop("anchor_sha256", None)
        actual = _canonical_hash(canonical)
        if recorded != actual:
            errors.append(f"cycle{cycle}: state anchor SHA mismatch")
        if anchor.get("schema_version") != "distillschedule.wan13b_trajectory_state_anchor.v2":
            errors.append(f"cycle{cycle}: state anchor schema changed")
        if anchor.get("cycle") != cycle or anchor.get("placement") != args.placement:
            errors.append(f"cycle{cycle}: state anchor identity changed")
        components = dict(anchor.get("components", {}))
        expected_components = {
            "generator_trainable", "generator_adam", "generator_ema",
            "critic_trainable", "critic_adam",
        }
        if set(components) != expected_components:
            errors.append(f"cycle{cycle}: numeric state component inventory changed")
        for component, evidence in components.items():
            sketch = dict(evidence.get("numeric_sketch", {}))
            if sketch.get("schema_version") != STATE_SKETCH_SCHEMA_VERSION:
                errors.append(f"cycle{cycle}: {component} numeric sketch schema changed")
            if sketch.get("world_size") != WORLD_SIZE or int(sketch.get("coordinate_count", 0)) <= 0:
                errors.append(f"cycle{cycle}: {component} numeric sketch is incomplete")
            if sketch.get("replica_validation") != "ALL_RANKS_CONTRIBUTE_SAMPLED_COORDINATES_MUST_MATCH":
                errors.append(f"cycle{cycle}: {component} replica agreement is unverified")
            if not sketch.get("coordinates") or evidence.get("global_moments", {}).get("status") != "PASS":
                errors.append(f"cycle{cycle}: {component} numeric evidence is incomplete")
        anchors.append({"cycle": cycle, "path": str(path), "sha256": recorded})

    cgroup = [_cgroup_delta(args.run_root, node=node, errors=errors) for node in range(2)]
    run_id = manifests[0].get("run_id") if manifests else None
    teardown: list[dict[str, Any]] = []
    for node in range(2):
        path = args.run_root / "launcher" / f"node_{node}" / "teardown_status.txt"
        try:
            receipt = {
                key: value for line in path.read_text().splitlines() if "=" in line
                for key, value in (line.split("=", 1),)
            }
        except OSError as error:
            errors.append(f"node{node}: teardown receipt missing: {error}")
            receipt = {}
        if receipt.get("status") != "PASS" or receipt.get("gpu_process_count") != "0":
            errors.append(f"node{node}: teardown cleanup is not PASS")
        if receipt.get("owner_run_id") != run_id:
            errors.append(f"node{node}: teardown run identity changed")
        teardown.append(receipt)

    value = {
        "schema_version": "distillschedule.wan13b_trajectory_summary.v1",
        "status": "PASS" if not errors and len(rank_rows) == WORLD_SIZE else "FAIL",
        "errors": errors,
        "run_id": run_id,
        "run_root": str(args.run_root.resolve()),
        "placement": args.placement,
        "expected_head": args.expected_head,
        "cycles": args.cycles,
        "rank_count": len(rank_rows),
        "rank_cycle_receipts": sum(len(rows) for rows in rank_rows.values()),
        "expected_rank_cycle_receipts": WORLD_SIZE * args.cycles,
        "matched_contract": matched_contract,
        "workload_fingerprint": _canonical_hash(matched_contract) if matched_contract else None,
        "trace_rollout_contract": trace_rollout_contract,
        "implementation_contract": implementation_contract,
        "implementation_fingerprint": _canonical_hash(implementation_contract) if implementation_contract else None,
        "sample_identity_contract": {
            "rank_qualified_payloads_per_cycle": 96,
            "critic_payloads_per_cycle": 80,
            "cycle_qualified_occurrences": len(occurrences),
            "payload_cohort_replayed": args.cycles < 2 or not any(
                payload_mapping_by_cycle[cycle] != payload_mapping_by_cycle[1]
                for cycle in range(2, args.cycles + 1)
            ),
            "ordered_rank_slot_mapping_replayed": args.cycles < 2 or not any(
                payload_mapping_by_cycle[cycle] != payload_mapping_by_cycle[1]
                for cycle in range(2, args.cycles + 1)
            ),
        },
        "final_versions": {"generator": 2 + args.cycles, "critic": 10 + 5 * args.cycles, "ema": 10 + 5 * args.cycles},
        "numeric_state_anchors": anchors,
        "runtime_receipts": runtime_receipts,
        "resource_status": {"cgroup_memory_events": cgroup, "teardown": teardown},
        "diagnostic_only": True,
        "formal_timing_eligible": False,
    }
    _atomic_json(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
