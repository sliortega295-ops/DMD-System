#!/usr/bin/env python3
"""Low-overhead, same-process Wan1.3B two-cycle timing worker.

The first cycle is an in-process warm-up.  Only the second cycle is eligible
for a steady-state timing claim.  Generator, Critic, optimizers, EMA, trace,
and runtime caches are constructed once and remain live across both cycles.

This worker intentionally does not emit scalar/state evidence while either
timer is open.  It is bound to a completed two-cycle trajectory correctness
receipt, and performs only lightweight census and finite checks after both
timed cycles have ended.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _load_native_runtime() -> None:
    """Defer model dependencies so CPU control-flow tests can inject math seams."""
    global fair, base, p0
    if "fair" in globals():
        return
    path = ROOT / "scripts/distillschedule/probe_wan13b_fair_world16.py"
    spec = importlib.util.spec_from_file_location("wan13b_two_cycle_fair_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    fair = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fair
    spec.loader.exec_module(fair)
    base = fair.base
    p0 = base.p0


from distillschedule.profiling.static_profile_diagnostics import (  # noqa: E402
    StaticProfileDiagnostics,
)
from distillschedule.runtime.fsdp2_p0 import (  # noqa: E402
    StaticFSDP2P0Policy,
    expected_trace_rollout_call_inventory,
    validate_exact_rollout_call_inventory,
)
from runtime.gst_pool_contract import NegativePromptAsset  # noqa: E402
from runtime.static_world16_trace import StaticWorld16Trace  # noqa: E402
from distillschedule.runtime.wan13b_lora_grad_epilogue import (  # noqa: E402
    grad_epilogue_requested,
    runtime_precision_contract,
    validate_grad_epilogue_receipts,
)


WORLD_SIZE = 16
SOURCE_STEPS = tuple(range(10, 15))
TIMED_CYCLES = 2
RESIDENT_WINNER = "resident-async128-tree-chunk2048"
FSDP_BASELINE = "world16-fsdp"
REGISTERED_PLACEMENTS = frozenset({RESIDENT_WINNER, FSDP_BASELINE})
RESIDENT_ARM_CONTRACT = "RESIDENT_ASYNC128_TREE_CHUNK2048_FUSED_EPILOGUE"
FSDP_ARM_CONTRACT = "WORLD16_FSDP_BASELINE"


def _rollout_operations() -> tuple[str, ...]:
    return (
        "step/10/generator/rollout",
        *(f"step/{step}/critic/rollout" for step in SOURCE_STEPS),
    )


def _trace_rollout_contract(trace: StaticWorld16Trace) -> list[dict[str, Any]]:
    return [
        {
            "operation": operation,
            "inventory": expected_trace_rollout_call_inventory(
                trace.rank_control_validation,
                operation=operation,
                expected_rank_count=WORLD_SIZE,
            ),
        }
        for operation in _rollout_operations()
    ]


def parser() -> argparse.ArgumentParser:
    _load_native_runtime()
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-id", required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--placement", choices=fair.FAIR_PLACEMENTS, required=True)
    value.add_argument("--bootstrap", type=Path, required=True)
    value.add_argument("--expected-bootstrap-hash", required=True)
    value.add_argument("--model-root", type=Path, required=True)
    value.add_argument("--trace-bundle", type=Path, required=True)
    value.add_argument("--expected-trace-hash", required=True)
    value.add_argument("--negative-prompt", type=Path, required=True)
    value.add_argument("--timing-seed", type=int, required=True)
    value.add_argument("--timing-arm-id", required=True)
    value.add_argument("--expected-head", required=True)
    value.add_argument("--correctness-receipt", type=Path, required=True)
    value.add_argument("--expected-correctness-hash", required=True)
    value.add_argument("--trajectory-comparison-receipt", type=Path, required=True)
    value.add_argument("--expected-trajectory-comparison-hash", required=True)
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _clean_runtime(placement: str) -> bool:
    return placement == RESIDENT_WINNER


def _expected_arm_contract(placement: str) -> str:
    if placement == RESIDENT_WINNER:
        from distillschedule.runtime.wan13b_layout_ablation import requested, ARM
        return ARM if requested() else RESIDENT_ARM_CONTRACT
    if placement == FSDP_BASELINE:
        return FSDP_ARM_CONTRACT
    raise ValueError(f"unregistered two-cycle placement: {placement}")


def _configure_shared_mode_args(args: argparse.Namespace) -> None:
    """Populate every mode attribute consumed by the shared engine builders."""

    args.timing_mode = True
    args.profile_mode = False
    args.trajectory_mode = False
    args.validate_only = False


def _validate_arm_environment(placement: str, *, factorial_configuration: dict[str, Any] | None = None) -> dict[str, Any]:
    from distillschedule.runtime.wan13b_factorial_r2 import fsdp_compatible_b
    compatible = fsdp_compatible_b(factorial_configuration)
    if compatible:
        from distillschedule.runtime.wan13b_factorial_r2 import assert_runtime_environment
        assert_runtime_environment(factorial_configuration)
    if compatible and placement != FSDP_BASELINE:
        raise ValueError('compatible B requires the original world16 FSDP placement')
    grad_fused = grad_epilogue_requested()
    from distillschedule.runtime.wan13b_norm_policy import requested as norm_requested

    norm_fused = norm_requested()
    from distillschedule.runtime.wan13b_rollout_shadow import requested as shadow_requested

    shadow_enabled = shadow_requested()
    from distillschedule.runtime.wan13b_rollout_cuda_graph import requested as graph_requested

    graph_enabled = graph_requested()
    if graph_enabled and (placement != RESIDENT_WINNER or shadow_enabled):
        raise ValueError("CUDA Graph requires Resident without merged weights")
    if shadow_enabled and placement != RESIDENT_WINNER:
        raise ValueError("merged shadow requires the registered Resident control")
    if norm_fused and placement != RESIDENT_WINNER and not compatible:
        raise ValueError("FSDP control cannot enable Resident normalization fusion")
    fused = (
        os.environ.get(
            "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE", "0"
        )
        == "1"
    )
    cast_cache = (
        os.environ.get(
            "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE", "0"
        )
        == "1"
    )
    from distillschedule.runtime.wan13b_layout_ablation import requested as layout_requested, check as check_layout
    layout_only=layout_requested()
    check_layout(placement,enabled=layout_only,flags={"fused":fused,"grad":grad_fused,"norm":norm_fused,"graph":graph_enabled,"shadow":shadow_enabled,"cast":cast_cache,"input":os.environ.get("DISTILLSCHEDULE_WAN13B_FUSED_INPUT_LORA_A","0")=="1","score":os.environ.get("DISTILLSCHEDULE_WAN13B_SCORE_CUDA_GRAPH","0")=="1"})
    from distillschedule.runtime.wan13b_cast_graph_composition import check as check_cast_composition
    check_cast_composition(placement, cast_cache=cast_cache, fused=fused, graph=graph_enabled, input_gemm=os.environ.get("DISTILLSCHEDULE_WAN13B_FUSED_INPUT_LORA_A", "0") == "1", shadow=shadow_enabled)
    if placement == RESIDENT_WINNER and (not fused and not layout_only):
        raise ValueError(
            "two-cycle Resident winner requires fused epilogue ON and cast cache OFF"
        )
    if placement == FSDP_BASELINE and (fused or cast_cache or grad_fused) and not compatible:
        raise ValueError(
            "two-cycle world16-FSDP baseline requires fused epilogue and cast cache OFF"
        )
    return {
        "layout_only_ablation": layout_only,
        "arm_contract": _expected_arm_contract(placement),
        "generator_nograd_lora_fused_epilogue": fused,
        "generator_nograd_lora_cast_cache": cast_cache,
        "grad_lora_fused_epilogue": grad_fused,
        "modulated_norm_fusion": norm_fused,
        "generator_rollout_merged_shadow": shadow_enabled,
        "generator_rollout_cuda_graph": graph_enabled,
        "fused_input_lora_a": os.environ.get("DISTILLSCHEDULE_WAN13B_FUSED_INPUT_LORA_A", "0") == "1",
        "critic_grad_cuda_graph": os.environ.get("DISTILLSCHEDULE_WAN13B_CRITIC_GRAD_CUDA_GRAPH", "0") == "1",
        "score_cuda_graph": os.environ.get("DISTILLSCHEDULE_WAN13B_SCORE_CUDA_GRAPH", "0") == "1",
    }


def _load_correctness_binding(
    args: argparse.Namespace, *, expected_arm: dict[str, Any]
) -> dict[str, Any]:
    if len(args.expected_correctness_hash) != 64:
        raise ValueError("expected correctness hash must be SHA-256")
    observed = _sha256(args.correctness_receipt)
    if observed != args.expected_correctness_hash:
        raise ValueError("two-cycle correctness receipt SHA-256 changed")
    value = json.loads(args.correctness_receipt.read_text())
    expected_final = {"generator": 4, "critic": 20, "ema": 20}
    required = {
        "schema_version": "distillschedule.wan13b_trajectory_summary.v1",
        "status": "PASS",
        "placement": args.placement,
        "cycles": TIMED_CYCLES,
        "rank_count": WORLD_SIZE,
        "final_versions": expected_final,
        "diagnostic_only": True,
        "formal_timing_eligible": False,
        "expected_head": args.expected_head,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(
                f"two-cycle correctness binding changed {key}: "
                f"{value.get(key)!r} != {expected!r}"
            )
    matched = value.get("matched_contract", {})
    implementation = value.get("implementation_contract", {})
    if value.get("workload_fingerprint") != _canonical_hash(matched):
        raise ValueError("correctness workload fingerprint is invalid")
    if value.get("implementation_fingerprint") != _canonical_hash(implementation):
        raise ValueError("correctness implementation fingerprint is invalid")
    for key, expected in expected_arm.items():
        observed_flag = implementation.get(key, False) if key in {
            "modulated_norm_fusion", "generator_rollout_merged_shadow", "generator_rollout_cuda_graph", "fused_input_lora_a", "score_cuda_graph", "critic_grad_cuda_graph", "layout_only_ablation"
        } else implementation.get(key)
        if observed_flag != expected:
            raise ValueError(f"correctness receipt arm contract changed: {key}")
    if matched.get("trace_replay_policy") != "FIXED_FIVE_STEP_COHORT_REPLAY":
        raise ValueError("correctness receipt trace replay policy changed")
    if int(matched.get("seed", -1)) != args.timing_seed:
        raise ValueError("timing seed differs from correctness trajectory seed")
    expected_matched = {
        "bootstrap_hash": args.expected_bootstrap_hash,
        "trace_bundle_hash": args.expected_trace_hash,
        "global_batch": 16,
        "updates": "G1_C5_EMA5",
        "rollout_batch_size": 1,
        "score_packing": False,
    }
    for key, expected in expected_matched.items():
        if matched.get(key) != expected:
            raise ValueError(f"correctness receipt matched workload changed: {key}")
    if implementation.get("trajectory_arm_id") != args.timing_arm_id:
        raise ValueError("timing arm id differs from correctness trajectory arm")
    return {
        "path": str(args.correctness_receipt),
        "sha256": observed,
        "schema_version": value["schema_version"],
        "run_id": value.get("run_id"),
        "expected_head": value.get("expected_head"),
        "status": value["status"],
        "matched_contract": matched,
        "implementation_contract": implementation,
    }


def _load_trajectory_comparison_binding(
    args: argparse.Namespace, *, correctness_run_id: str
) -> dict[str, Any]:
    if len(args.expected_trajectory_comparison_hash) != 64:
        raise ValueError("expected trajectory comparison hash must be SHA-256")
    observed = _sha256(args.trajectory_comparison_receipt)
    if observed != args.expected_trajectory_comparison_hash:
        raise ValueError("trajectory comparison receipt SHA-256 changed")
    value = json.loads(args.trajectory_comparison_receipt.read_text())
    required = {
        "schema_version": "distillschedule.trajectory_coarse_comparison.v1",
        "comparison_kind": "semantic",
        "expected_cycles": 2,
        "expected_world_size": 16,
        "complete": True,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"trajectory comparison changed: {key}")
    contract = value.get("contract", {})
    if contract.get("status") != "PASS" or contract.get("mismatches"):
        raise ValueError("trajectory comparison semantic contract is not PASS")
    arm_matches: list[tuple[str, dict[str, Any]]] = []
    for side_name in ("reference", "candidate"):
        side = value.get(side_name, {})
        variants = side.get("variants", [])
        if args.timing_arm_id in variants:
            arm_matches.append((side_name, side))
    if len(arm_matches) != 1:
        raise ValueError("timing arm must identify exactly one trajectory comparison side")
    matched_side, matched_inventory = arm_matches[0]
    run_ids = matched_inventory.get("run_ids", [])
    if run_ids != [correctness_run_id]:
        raise ValueError(
            "trajectory comparison side is not bound to this arm's correctness run"
        )
    reference_arms = value.get("reference", {}).get("variants", [])
    candidate_arms = value.get("candidate", {}).get("variants", [])
    return {
        "path": str(args.trajectory_comparison_receipt),
        "sha256": observed,
        "schema_version": value["schema_version"],
        "status": value.get("status"),
        "contract_status": contract["status"],
        "complete": value["complete"],
        "reference_arms": reference_arms,
        "candidate_arms": candidate_arms,
        "matched_side": matched_side,
        "matched_correctness_run_id": correctness_run_id,
        "numerical_review_recorded": value.get("status")
        in {
            "TRAJECTORY_REVIEW_50C_COMPLETED",
            "TRAJECTORY_MATCHED_50C_SINGLE_SEED",
        },
    }


def _run_workload_cycle(
    *,
    rank: int,
    device: torch.device,
    generator_engine: Any,
    critic_engine: Any,
    ema: Any,
    trace: Any,
    negative: torch.Tensor,
    diagnostics: Any,
    policy: Any,
    clean_runtime: bool,
    runtime_cache: dict[str, Any] | None,
    deferred_finite_checks: list[torch.Tensor],
) -> dict[str, Any]:
    """Execute G1/C5/EMA5 without CPU evidence extraction or serialization."""

    critic_entries: dict[int, Any] = {}
    generator_record = p0._run_nogan_generator_update_p0(
        rank=rank,
        device=device,
        generator_engine=generator_engine,
        critic_engine=critic_engine,
        trace=trace,
        negative=negative,
        collect_diagnostics=False,
        collect_scalars=False,
        diagnostics=diagnostics,
        policy=policy,
        score_pack_parity_check=False,
        clean_runtime=clean_runtime,
        runtime_cache=runtime_cache,
        deferred_finite_checks=deferred_finite_checks,
    )
    rollout_outputs, rollout_records = p0._precompute_critic_rollouts(
        rank=rank,
        steps=SOURCE_STEPS,
        device=device,
        generator_engine=generator_engine,
        trace=trace,
        policy=policy,
        collect_diagnostics=False,
        diagnostics=diagnostics,
        rollout_parity_check=False,
        clean_runtime=clean_runtime,
        runtime_cache=runtime_cache,
        entry_outputs=critic_entries,
    )
    critic_records: list[dict[str, Any]] = []
    for source_step in SOURCE_STEPS:
        critic_records.append(
            p0._run_nogan_critic_precomputed(
                rank=rank,
                step=source_step,
                x0=rollout_outputs[source_step],
                device=device,
                critic_engine=critic_engine,
                trace=trace,
                collect_diagnostics=False,
                collect_scalars=False,
                diagnostics=diagnostics,
                entry=critic_entries[source_step],
                runtime_cache=runtime_cache,
                deferred_finite_checks=deferred_finite_checks,
            )
        )
        ema.step()
        del rollout_outputs[source_step]
    return {
        "generator": generator_record,
        "rollouts": rollout_records,
        "critic": critic_records,
    }


def _cycle_census(
    raw: dict[str, Any],
    *,
    cycle: int,
    trace_rollout_contract: list[dict[str, Any]],
) -> dict[str, Any]:
    generator = raw["generator"]
    rollouts = list(raw["rollouts"])
    if len(rollouts) != 5 or any(
        int(row.get("batch_size", 1)) != 1 for row in rollouts
    ):
        raise AssertionError(f"cycle{cycle}: physical rollout B1 inventory changed")
    if len(raw["critic"]) != 5:
        raise AssertionError(f"cycle{cycle}: Critic update inventory changed")
    critic_steps = [int(row.get("logical_step", -1)) for row in raw["critic"]]
    if critic_steps != list(SOURCE_STEPS) or any(
        int(row.get("score_forwards", -1)) != 1 for row in raw["critic"]
    ):
        raise AssertionError(f"cycle{cycle}: Critic Score/step census changed")
    if int(generator["physical_score_forwards"]) != 3:
        raise AssertionError(f"cycle{cycle}: no-GAN Score forward inventory changed")
    inventories = [generator["rollout"], *rollouts]
    normalized = [
        validate_exact_rollout_call_inventory(
            row, label=f"two-cycle/cycle{cycle}/rollout{index}"
        )
        for index, row in enumerate(inventories)
    ]
    if len(trace_rollout_contract) != len(normalized):
        raise AssertionError(f"cycle{cycle}: trace rollout operation count changed")
    for index, (inventory, expected) in enumerate(
        zip(normalized, trace_rollout_contract, strict=True)
    ):
        operation = expected.get("operation")
        if operation != _rollout_operations()[index]:
            raise AssertionError(f"cycle{cycle}: rollout operation order changed")
        if inventory != expected.get("inventory"):
            raise AssertionError(
                f"cycle{cycle}/{operation}: exact trace-derived DiT census changed"
            )
    generator_trace = str(generator["rollout"].get("trace_identity_sha256", ""))
    critic_traces = [
        str(row.get("member_trace_identities", [""])[0]) for row in rollouts
    ]
    if len(generator_trace) != 64 or any(len(value) != 64 for value in critic_traces):
        raise AssertionError(f"cycle{cycle}: missing consumed trace payload identity")
    payloads = [generator_trace, *critic_traces]
    return {
        "generator_updates": 1,
        "critic_updates": 5,
        "ema_updates": 5,
        "generator_score_forwards": int(generator["physical_score_forwards"]),
        "critic_score_forwards": 5,
        "physical_rollout_batches": 5,
        "rollout_batch_size": 1,
        "rollout_operation_keys": [
            str(row["operation"]) for row in trace_rollout_contract
        ],
        "rollout_call_inventory": normalized,
        "generator_trace_payload_identity": generator_trace,
        "critic_trace_payload_identities": critic_traces,
        "sample_payload_identities": payloads,
        "sample_occurrence_identities": [
            f"cycle={cycle}/slot={index}/payload={identity}"
            for index, identity in enumerate(payloads)
        ],
    }


def _validate_fused_receipt(
    build: dict[str, Any], *, placement: str, cycles: int = 2,
    layout_only: bool | None = None,
    fsdp_compatible: bool = False,
) -> dict[str, Any]:
    if fsdp_compatible and placement != FSDP_BASELINE:
        raise AssertionError('compatible B epilogue requires FSDP placement')
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
    if not isinstance(generator, dict) or not isinstance(critic, dict):
        raise AssertionError("missing fused-epilogue runtime receipt")
    from distillschedule.runtime.wan13b_layout_ablation import requested as layout_requested, check as check_layout
    if layout_requested() if layout_only is None else layout_only:
        check_layout(placement, enabled=True, flags={})
        for role, receipt in (("generator", generator), ("critic", critic)):
            if receipt.get("requested") is not False or receipt.get("enabled") is not False:
                raise AssertionError(f"layout-only {role} must report fused epilogue disabled")
            if any(int(receipt.get(key, 0)) for key in (
                "fused_invocation_count", "model_call_count", "reference_fallback_count"
            )):
                raise AssertionError(f"layout-only {role} unexpectedly executed fused epilogue")
        return {"generator": generator, "critic": critic}
    if placement == RESIDENT_WINNER or fsdp_compatible:
        if generator.get("requested") is not True or generator.get("enabled") is not True:
            raise AssertionError("Resident winner fused epilogue is not enabled")
        shadow_enabled = bool(build.get("generator", {}).get("metadata", {}).get(
            "generator_rollout_merged_shadow", {}).get("enabled", False))
        from distillschedule.runtime.wan13b_rollout_shadow import expected_epilogue_counts

        graph_receipt = build.get("generator", {}).get("metadata", {}).get(
            "generator_rollout_cuda_graph", {})
        expected_counts = expected_epilogue_counts(
            cycles=cycles, shadow_enabled=shadow_enabled, graph_receipt=graph_receipt)
        for key, expected in expected_counts.items():
            if generator.get(key) != expected:
                raise AssertionError(f"training-model epilogue census changed {key}")
        if generator.get("reference_fallback_reasons") != {
            "BASE_DTYPE_torch.float32": expected_counts["reference_fallback_count"]
        }:
            raise AssertionError("Resident winner has an unexpected fallback reason")
        if critic.get("enabled") is not False or critic.get("disabled_reason") != "GENERATOR_ONLY":
            raise AssertionError("Critic unexpectedly entered Generator fused epilogue")
    else:
        if generator.get("requested") or generator.get("enabled") or critic.get("enabled"):
            raise AssertionError("world16-FSDP baseline enabled a Resident-only epilogue")
    return {"generator": generator, "critic": critic}


def _validate_build_contract(
    build: dict[str, Any], *, placement: str, ema: Any,
    nccl_algo: str | None = None,
) -> dict[str, Any]:
    runtime_hashes: set[str] = set()
    for role in ("generator", "critic"):
        component = build.get(role, {})
        metadata = component.get("metadata", {})
        residual = metadata.get("chunked_residual_runtime", {})
        if metadata.get("chunked_residual_storage") != "cuda_out_of_place":
            raise AssertionError(f"{role}: residual storage is not GPU resident")
        if residual.get("requested_storage") != "cuda_out_of_place":
            raise AssertionError(f"{role}: residual runtime request changed")
        if residual.get("storage_capable") is not True:
            raise AssertionError(f"{role}: residual runtime cannot honor CUDA storage")
        runtime_sha = residual.get("transformer_source_sha256")
        if not isinstance(runtime_sha, str) or len(runtime_sha) != 64:
            raise AssertionError(f"{role}: residual runtime source identity missing")
        runtime_hashes.add(runtime_sha)
        placement_receipt = component.get("placement", {})
        if placement_receipt.get("trainable_parameter_dtypes") != ["torch.float32"]:
            raise AssertionError(f"{role}: LoRA parameter dtype changed")
        if "torch.bfloat16" not in placement_receipt.get("frozen_parameter_dtypes", []):
            raise AssertionError(f"{role}: frozen backbone dtype changed")
        if placement_receipt.get("gradient_reduction") != "SUM_THEN_DIVIDE_WORLD_SIZE":
            raise AssertionError(f"{role}: gradient mean convention changed")
        if component.get("engine", {}).get("optimizer_implementation") != "PYTORCH_DEFAULT":
            raise AssertionError(f"{role}: optimizer implementation changed")
        dtype_contract = component.get("lora_compute_dtype_contract", {})
        if dtype_contract.get("frozen_base") != "torch.bfloat16":
            raise AssertionError(f"{role}: frozen compute dtype changed")
        if dtype_contract.get("trainable_lora") != "torch.float32":
            raise AssertionError(f"{role}: LoRA compute dtype changed")
        if placement == RESIDENT_WINNER:
            modulation = metadata.get("modulation_chunk_runtime", {})
            if modulation.get("effective_tokens") != [2048]:
                raise AssertionError(f"{role}: modulation chunk is not 2048")
            if modulation.get("block_count") != 30 or modulation.get("all_blocks_match") is not True:
                raise AssertionError(f"{role}: modulation block census changed")
    if len(runtime_hashes) != 1:
        raise AssertionError("Generator/Critic residual runtime sources differ")
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
            raise AssertionError(f"static residency contract changed: {key}")
    if getattr(ema, "implementation", None) != "reference":
        raise AssertionError("two-cycle arms require reference GPU EMA")
    expected_algo = "Tree" if placement == RESIDENT_WINNER else "AUTO"
    observed_algo = os.environ.get("NCCL_ALGO", "AUTO") if nccl_algo is None else nccl_algo
    if observed_algo != expected_algo:
        raise AssertionError(
            f"NCCL algorithm changed: {observed_algo!r} != {expected_algo!r}"
        )
    return {
        "helios_transformer_sha256": next(iter(runtime_hashes)),
        "ema_implementation": "reference",
        "nccl_algo": observed_algo,
        "residual_storage": "cuda_out_of_place",
        "parameter_offload": False,
        "activation_offload": False,
        "optimizer_offload": False,
        "parameter_streaming": False,
        "model_swap": False,
    }


def main() -> int:
    args = parser().parse_args()
    # Construction helpers use these shared mode attributes to disable audit
    # hooks.  This dedicated worker never enters the legacy one-cycle timer or
    # the trajectory evidence path.
    _configure_shared_mode_args(args)
    if args.placement not in REGISTERED_PLACEMENTS:
        raise ValueError("two-cycle timing placement is not registered")
    if args.timing_seed < 0:
        raise ValueError("timing seed must be non-negative")
    if not args.run_id.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("unsafe run id")
    if not args.timing_arm_id:
        raise ValueError("timing arm id must be non-empty")
    if len(args.expected_head) != 40:
        raise ValueError("expected HEAD must be a full Git SHA")
    profiler_environment = {
        "exact_range": os.environ.get("DISTILLSCHEDULE_STATIC_FSDP2_NSYS_EXACT_RANGE"),
        "lora_operator_attribution": os.environ.get(
            "DISTILLSCHEDULE_WAN13B_LORA_OPERATOR_ATTRIBUTION", "0"
        ),
    }
    if profiler_environment["exact_range"] not in (None, "", "0"):
        raise ValueError("two-cycle native timing cannot enable an Nsight range")
    if profiler_environment["lora_operator_attribution"] != "0":
        raise ValueError("two-cycle native timing cannot enable LoRA profiling")

    arm = _validate_arm_environment(args.placement)
    correctness = _load_correctness_binding(args, expected_arm=arm)
    correctness_run_id = correctness.get("run_id")
    if not isinstance(correctness_run_id, str) or not correctness_run_id:
        raise ValueError("correctness receipt run identity is missing")
    trajectory_comparison = _load_trajectory_comparison_binding(
        args, correctness_run_id=correctness_run_id
    )
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != WORLD_SIZE:
        raise AssertionError(f"Wan1.3B two-cycle timing requires world16, got {world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    runtime_seed = args.timing_seed + rank
    random.seed(runtime_seed)
    torch.manual_seed(runtime_seed)
    torch.cuda.manual_seed_all(runtime_seed)
    output = args.output_dir / "ranks" / f"rank_{rank:02d}.json"
    result: dict[str, Any] = {
        "schema_version": "distillschedule.wan13b_two_cycle_timing_rank.v1",
        "status": "RUNNING",
        "run_id": args.run_id,
        "expected_head": args.expected_head,
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "world_size": world,
        "placement": args.placement,
        "timing_seed": args.timing_seed,
        "rank_runtime_seed": runtime_seed,
        "timing_arm_id": args.timing_arm_id,
        "arm_contract": arm,
        "correctness_binding": correctness,
        "trajectory_comparison_binding": trajectory_comparison,
        "profiled": False,
        "profiler_status": "NOT_RUN",
        "profiler_environment": profiler_environment,
        "cycle_contract": {
            "cycles_in_same_process": TIMED_CYCLES,
            "cycle_1_role": "WARMUP_DIAGNOSTIC",
            "cycle_2_role": "STEADY_STATE_CLAIMABLE",
            "claimable_cycle": 2,
            "trace_replay_policy": "FIXED_FIVE_STEP_COHORT_REPLAY",
            "updates_per_cycle": {"generator": 1, "critic": 5, "ema": 5},
        },
    }
    try:
        contract = base.contract_for_variant("no-gan")
        bootstrap = base.Wan13BBootstrap.load(
            args.bootstrap,
            expected_hash=args.expected_bootstrap_hash,
            verify_payloads=rank == 0,
        )
        trace = StaticWorld16Trace(args.trace_bundle, variant="no-gan")
        if trace.bundle_hash != args.expected_trace_hash:
            raise ValueError("trace bundle identity changed")
        if Path(bootstrap.manifest["model_root"]).resolve() != args.model_root.resolve():
            raise ValueError("bootstrap/model-root path identity changed")
        trace_rollout_contract = _trace_rollout_contract(trace)
        trace_rollout_contract_sha256 = _canonical_hash(trace_rollout_contract)
        negative_asset = NegativePromptAsset.load(args.negative_prompt)
        matched_contract = {
            **runtime_precision_contract(),
            "bootstrap_hash": bootstrap.manifest_hash,
            "model_identity_sha256": bootstrap.manifest["model_identity_sha256"],
            "trace_bundle_hash": trace.bundle_hash,
            "trace_binding_hash": trace.binding_hash,
            "trace_replay_policy": "FIXED_FIVE_STEP_COHORT_REPLAY",
            "negative_prompt_manifest_hash": negative_asset.manifest["manifest_hash"],
            "seed": args.timing_seed,
            "global_batch": 16,
            "updates": "G1_C5_EMA5",
            "rollout_batch_size": 1,
            "score_packing": False,
            "trace_rollout_contract_sha256": trace_rollout_contract_sha256,
        }
        result["trace_rollout_contract"] = trace_rollout_contract
        result["trace_rollout_contract_sha256"] = trace_rollout_contract_sha256
        result["matched_contract"] = matched_contract
        result["workload_fingerprint"] = _canonical_hash(matched_contract)
        result["implementation_fingerprint"] = _canonical_hash(arm)
        result["configuration_fingerprint"] = _canonical_hash(
            {
                "workload_fingerprint": result["workload_fingerprint"],
                "implementation_fingerprint": result["implementation_fingerprint"],
            }
        )
        correctness_matched = correctness["matched_contract"]
        for key, observed in matched_contract.items():
            if correctness_matched.get(key) != observed:
                raise AssertionError(
                    f"timing workload differs from correctness trajectory: {key}"
                )

        negative = negative_asset.embedding.to(
            device=device, dtype=torch.bfloat16
        )
        construction_started = time.perf_counter()
        generator_engine, critic_engine, ema, build = base._build_engines(
            args=args, bootstrap=bootstrap, device=device
        )
        torch.cuda.synchronize(device)
        construction_seconds = time.perf_counter() - construction_started
        runtime_contract = _validate_build_contract(
            build, placement=args.placement, ema=ema
        )
        if correctness.get("expected_head") != args.expected_head:
            raise AssertionError("correctness source HEAD changed")
        clean_runtime = _clean_runtime(args.placement)
        runtime_cache: dict[str, Any] | None = {} if clean_runtime else None
        policy = StaticFSDP2P0Policy(
            mesh_policy="world16-fsdp",
            rollout_batch_size=1,
            pack_real_score_pair=False,
            rollout_backend="native-fsdp2-batch",
        )
        diagnostics = StaticProfileDiagnostics(
            enabled=False,
            rank=rank,
            output_path=None,
            collect_cpu_diagnostics=False,
        )
        initial_versions = {
            "generator": int(generator_engine.global_steps),
            "critic": int(critic_engine.global_steps),
            "ema": int(ema.optimization_step),
        }
        if initial_versions != {"generator": 2, "critic": 10, "ema": 10}:
            raise AssertionError("two-cycle bootstrap versions changed")

        torch.cuda.reset_peak_memory_stats(device)
        deferred_finite_checks: list[torch.Tensor] = []
        finite_slices: list[tuple[int, int]] = []
        raw_cycles: list[dict[str, Any]] = []
        timing_rows: list[dict[str, Any]] = []
        version_rows: list[dict[str, int]] = []
        for cycle in range(1, TIMED_CYCLES + 1):
            dist.barrier()
            torch.cuda.synchronize(device)
            check_start = len(deferred_finite_checks)
            start_ns = time.time_ns()
            started = time.perf_counter()
            raw = _run_workload_cycle(
                rank=rank,
                device=device,
                generator_engine=generator_engine,
                critic_engine=critic_engine,
                ema=ema,
                trace=trace,
                negative=negative,
                diagnostics=diagnostics,
                policy=policy,
                clean_runtime=clean_runtime,
                runtime_cache=runtime_cache,
                deferred_finite_checks=deferred_finite_checks,
            )
            torch.cuda.synchronize(device)
            dist.barrier()
            elapsed = time.perf_counter() - started
            stop_ns = time.time_ns()
            finite_slices.append((check_start, len(deferred_finite_checks)))
            raw_cycles.append(raw)
            timing_rows.append(
                {
                    "cycle": cycle,
                    "role": "WARMUP_DIAGNOSTIC" if cycle == 1 else "STEADY_STATE_CLAIMABLE",
                    "start_unix_ns": start_ns,
                    "stop_unix_ns": stop_ns,
                    "local_elapsed_seconds": elapsed,
                }
            )
            version_rows.append(
                {
                    "generator": int(generator_engine.global_steps),
                    "critic": int(critic_engine.global_steps),
                    "ema": int(ema.optimization_step),
                }
            )

        # All device-to-host evidence extraction occurs after both timer ranges.
        finite_status = [
            bool(torch.stack(deferred_finite_checks[start:stop]).all().detach().cpu())
            for start, stop in finite_slices
        ]
        if finite_status != [True, True]:
            raise FloatingPointError(f"two-cycle finite gate failed: {finite_status}")
        expected_versions = (
            {"generator": 3, "critic": 15, "ema": 15},
            {"generator": 4, "critic": 20, "ema": 20},
        )
        if tuple(version_rows) != expected_versions:
            raise AssertionError(f"two-cycle version inventory changed: {version_rows}")
        cycle_census = [
            _cycle_census(
                raw,
                cycle=index,
                trace_rollout_contract=trace_rollout_contract,
            )
            for index, raw in enumerate(raw_cycles, start=1)
        ]
        fused_receipt = _validate_fused_receipt(build, placement=args.placement)
        grad_receipt = validate_grad_epilogue_receipts(
            build, requested=bool(arm["grad_lora_fused_epilogue"]))
        from distillschedule.runtime.wan13b_norm_policy import validate as validate_norm

        norm_receipt = validate_norm(build, enabled=bool(arm["modulated_norm_fusion"]))
        from distillschedule.runtime.wan13b_rollout_shadow import validate as validate_shadow

        shadow_receipt = validate_shadow(
            build, enabled=bool(arm["generator_rollout_merged_shadow"]), cycles=2
        )
        from distillschedule.runtime.wan13b_rollout_cuda_graph import validate as validate_graph

        graph_receipt = validate_graph(
            build, enabled=bool(arm["generator_rollout_cuda_graph"]), cycles=2)

        from distillschedule.runtime.wan13b_score_cuda_graph import validate_build as validate_score_graph
        score_graph_receipt = validate_score_graph(build, cycles=2, reference_checks=False)
        from distillschedule.runtime.wan13b_cast_graph_composition import validate_build as validate_cast_composition
        cast_composition_receipt = validate_cast_composition(build, enabled=arm["generator_nograd_lora_cast_cache"])
        from distillschedule.runtime.wan13b_critic_grad_graph import validate_build as validate_critic_graph
        result.update(
            cast_graph_composition_receipt=cast_composition_receipt,
            critic_grad_cuda_graph_receipt=validate_critic_graph(build, cycles=2, reference_checks=False),
            score_cuda_graph_receipt=score_graph_receipt,
            status="PASS",
            construction_seconds=construction_seconds,
            initial_versions=initial_versions,
            cycles=timing_rows,
            cycle_versions=version_rows,
            cycle_census=cycle_census,
            finite_status=finite_status,
            final_versions=version_rows[-1],
            updates={"generator": 2, "critic": 10, "ema": 10},
            memory=base._memory(device),
            runtime_cache_keys=(sorted(runtime_cache) if runtime_cache is not None else []),
            fused_epilogue_receipt=fused_receipt,
            grad_epilogue_receipt=grad_receipt,
            modulated_norm_fusion_receipt=norm_receipt,
            generator_rollout_merged_shadow_receipt=shadow_receipt,
            generator_rollout_cuda_graph_receipt=graph_receipt,
            runtime_contract=runtime_contract,
            gradient_sync={
                "generator_receipt_count": len(
                    getattr(generator_engine, "gradient_sync_receipts", [])
                ),
                "critic_receipt_count": len(
                    getattr(critic_engine, "gradient_sync_receipts", [])
                ),
            },
            evidence_extraction="AFTER_BOTH_TIMED_WINDOWS",
            state_anchors="NOT_COLLECTED_IN_TIMING_MODE",
            claimable_cycle=2,
        )
        _atomic_json(output, result)
        dist.barrier()
        return 0
    except BaseException as error:
        result.update(
            status=(
                "OOM"
                if isinstance(error, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(error).lower()
                else "FAIL"
            ),
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        try:
            result["memory_failure"] = base._memory(device)
        except BaseException:
            pass
        _atomic_json(output, result)
        raise
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
