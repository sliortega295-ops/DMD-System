#!/usr/bin/env python3
"""Fail-closed summary for one LiveWan world16 baseline or pipeline run."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from distillschedule.workloads.livewan_crossscale import (
    FULL5_STEPS,
    GLOBAL_BATCH,
    TRACE_BALANCED_8G8S_SLOT_PAIRS,
    WORLD_SIZE,
)
from distillschedule.runtime.livewan_crossscale_worker import (
    FormalTrace,
    PHASE_EXPANDED_CRITIC16_POOL_POLICY,
    atomic_json,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-root", required=True, type=Path)
    value.add_argument("--execution-mode", required=True, choices=("baseline", "pipeline"))
    value.add_argument(
        "--run-kind",
        required=True,
        choices=("validate", "capacity", "correctness", "timing", "profile"),
    )
    value.add_argument("--trace", required=True, type=Path)
    value.add_argument("--plan", required=True, type=Path)
    value.add_argument("--expected-head", required=True)
    value.add_argument("--output", required=True, type=Path)
    return value


def _close(values: list[float], *, rtol: float = 1e-7, atol: float = 1e-7) -> bool:
    return not values or max(values) - min(values) <= atol + rtol * max(abs(value) for value in values)


def main() -> int:
    args = parser().parse_args()
    errors: list[str] = []
    if args.run_kind == "validate":
        rows = []
        for node in (0, 1):
            path = args.run_root / f"validate_only.node{node}.json"
            if not path.is_file():
                errors.append(f"node{node}: validate receipt missing")
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            rows.append(row)
            if row.get("status") != "PASS" or row.get("node_rank") != node:
                errors.append(f"node{node}: validate receipt is not PASS")
        value = {
            "status": "PASS" if not errors else "FAIL",
            "run_kind": "validate",
            "execution_mode": args.execution_mode,
            "node_count": len(rows),
            "gpu_workload": "NOT_RUN",
            "errors": errors,
        }
        atomic_json(args.output, value)
        return 0 if not errors else 1

    trace = FormalTrace.load(args.trace)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    pool_policy = plan.get("metadata", {}).get(
        "pool_policy", "world16_fsdp2_sequential"
    )
    full_context_lens_elision = bool(
        plan.get("metadata", {}).get("full_context_lens_elision", False)
    )
    score_context_prefetch = bool(
        plan.get("metadata", {}).get("score_context_prefetch", False)
    )
    critic_lora_kernel_contract = dict(
        plan.get("metadata", {}).get("critic_lora_kernel", {})
    )
    critic_lora_kernel_enabled = bool(
        critic_lora_kernel_contract.get("enabled", False)
    )
    score_modulation_kernel_contract = dict(
        plan.get("metadata", {}).get("score_modulation_kernel", {})
    )
    score_modulation_kernel_enabled = bool(
        score_modulation_kernel_contract.get("enabled", False)
    )
    score_rope_kernel_contract = dict(
        plan.get("metadata", {}).get("score_rope_kernel", {})
    )
    score_rope_kernel_enabled = bool(
        score_rope_kernel_contract.get("enabled", False)
    )
    score_attention_backend_contract = dict(
        plan.get("metadata", {}).get("score_attention_backend", {})
    )
    score_attention_backend_enabled = bool(
        score_attention_backend_contract.get("enabled", False)
    )
    generator_cross_kv_cache_contract = dict(
        plan.get("metadata", {}).get("generator_cross_attention_kv_cache", {})
    )
    generator_cross_kv_cache_enabled = bool(
        generator_cross_kv_cache_contract.get("enabled", False)
    )
    generator_attention_backend_contract = dict(
        plan.get("metadata", {}).get("generator_attention_backend", {})
    )
    generator_attention_backend_enabled = bool(
        generator_attention_backend_contract.get("enabled", False)
    )
    generator_runahead_transport_contract = dict(
        plan.get("metadata", {}).get("generator_runahead_transport", {})
    )
    generator_runahead_transport_enabled = bool(
        generator_runahead_transport_contract.get("enabled", False)
    )
    split_dmd_shadow_contract = dict(
        plan.get("metadata", {}).get("split_dmd_shadow", {})
    )
    split_dmd_shadow_enabled = bool(
        split_dmd_shadow_contract.get("enabled", False)
    )
    rank_groups = {
        str(row["name"]): tuple(int(rank) for rank in row["ranks"])
        for row in plan.get("rank_groups", [])
    }
    generator_ranks = (
        tuple(range(WORLD_SIZE))
        if args.execution_mode == "baseline"
        else rank_groups.get("generator", ())
    )
    score_ranks = (
        tuple(range(WORLD_SIZE))
        if args.execution_mode == "baseline"
        else rank_groups.get("score", ())
    )
    if not generator_ranks or not score_ranks:
        errors.append("plan role ranks are missing")
    critic_ranks = (
        tuple(range(WORLD_SIZE))
        if args.execution_mode == "baseline"
        else rank_groups.get("critic", score_ranks)
    )

    if args.run_kind == "capacity":
        capacity_rows: list[dict[str, Any]] = []
        for rank in range(WORLD_SIZE):
            path = args.run_root / "receipts" / f"rank_{rank:03d}.json"
            if not path.is_file():
                errors.append(f"rank{rank}: capacity receipt missing")
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            capacity_rows.append(row)
            if (
                row.get("status") != "PASS"
                or row.get("stage") != "construction_capacity"
                or int(row.get("rank", -1)) != rank
                or row.get("mode") != args.execution_mode
                or row.get("run_kind") != "capacity"
            ):
                errors.append(f"rank{rank}: capacity receipt identity mismatch")
            if row.get("trace_sha256") != trace.sha256:
                errors.append(f"rank{rank}: capacity trace SHA mismatch")
            if row.get("pool_policy") != pool_policy:
                errors.append(f"rank{rank}: capacity pool policy mismatch")
            before = row.get("cgroup_oom_before", {})
            after = row.get("cgroup_oom_after", {})
            for key in ("oom", "oom_kill"):
                if int(after.get(key, 0)) != int(before.get(key, 0)):
                    errors.append(f"rank{rank}: cgroup {key} changed")
            build = row.get("build", {})
            if build.get("generator_ranks") != list(generator_ranks):
                errors.append(f"rank{rank}: capacity Generator rank map mismatch")
            if build.get("score_ranks") != list(score_ranks):
                errors.append(f"rank{rank}: capacity Score rank map mismatch")
            if build.get("critic_ranks") != list(critic_ranks):
                errors.append(f"rank{rank}: capacity Critic rank map mismatch")
            if pool_policy == PHASE_EXPANDED_CRITIC16_POOL_POLICY:
                if rank in generator_ranks:
                    if (
                        (build.get("student") or {}).get("placement")
                        != "gpu_resident_replica"
                        or build.get("critic") is None
                        or build.get("teacher") is not None
                    ):
                        errors.append(f"rank{rank}: invalid phase-expanded G residency")
                elif (
                    build.get("student") is not None
                    or build.get("critic") is None
                    or (build.get("teacher") or {}).get("placement")
                    != "fsdp2_group8_reshard"
                ):
                    errors.append(f"rank{rank}: invalid phase-expanded Score residency")
                sync = build.get("critic_gradient_sync", {})
                if (
                    sync.get("group_ranks") != list(range(WORLD_SIZE))
                    or sync.get("group_size") != WORLD_SIZE
                    or sync.get("reduction") != "SUM_THEN_DIVIDE_GROUP_SIZE"
                ):
                    errors.append(f"rank{rank}: invalid world16 Critic reducer residency")
            if split_dmd_shadow_enabled:
                shadow = build.get("critic_shadow")
                critic = build.get("critic")
                if rank in generator_ranks:
                    if (
                        build.get("student") is None
                        or critic is not None
                        or build.get("teacher") is not None
                        or (shadow or {}).get("placement")
                        != "generator_rank_frozen_replica"
                        or (shadow or {}).get("scope") != "step0_fake_score_only"
                        or (shadow or {}).get("trainable_parameter_count") != 0
                        or (shadow or {}).get("optimizer") is not None
                    ):
                        errors.append(f"rank{rank}: invalid split-DMD Generator residency")
                elif (
                    build.get("student") is not None
                    or shadow is not None
                    or (critic or {}).get("placement") != "gpu_resident_replica"
                    or (build.get("teacher") or {}).get("placement")
                    != "fsdp2_group8_reshard"
                ):
                    errors.append(f"rank{rank}: invalid split-DMD Score residency")
        identities = {
            (
                row.get("trace_sha256"),
                row.get("bootstrap_sha256"),
                row.get("source_receipt_sha256"),
            )
            for row in capacity_rows
        }
        if len(capacity_rows) != WORLD_SIZE:
            errors.append(
                f"capacity rank receipt count={len(capacity_rows)}, expected={WORLD_SIZE}"
            )
        if len(identities) != 1:
            errors.append("capacity source/bootstrap/trace identity differs by rank")
        if split_dmd_shadow_enabled:
            initial_lora_shas = {
                (row.get("build", {}).get("critic_shadow") or row.get("build", {}).get("critic") or {}).get(
                    "initial_lora_sha256"
                )
                for row in capacity_rows
            }
            if len(initial_lora_shas) != 1 or None in initial_lora_shas:
                errors.append("split-DMD C0 Critic/Shadow initial LoRA SHA mismatch")
        allocated = [
            int(row.get("peak_memory", {}).get("max_allocated_bytes", 0))
            for row in capacity_rows
        ]
        reserved = [
            int(row.get("peak_memory", {}).get("max_reserved_bytes", 0))
            for row in capacity_rows
        ]
        free = [
            int(row.get("peak_memory", {}).get("driver_free_bytes", 0))
            for row in capacity_rows
        ]
        value = {
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "run_kind": "capacity",
            "execution_mode": args.execution_mode,
            "rank_count": len(capacity_rows),
            "trace_sha256": trace.sha256,
            "identity": next(iter(identities)) if len(identities) == 1 else None,
            "pool_policy": pool_policy,
            "generator_ranks": list(generator_ranks),
            "score_ranks": list(score_ranks),
            "critic_ranks": list(critic_ranks),
            "split_dmd_shadow": split_dmd_shadow_contract,
            "peak_allocated_gib": max(allocated) / 2**30 if allocated else None,
            "peak_reserved_gib": max(reserved) / 2**30 if reserved else None,
            "minimum_driver_free_gib": min(free) / 2**30 if free else None,
            "gpu_training_workload": "NOT_RUN",
            "formal_timing_eligible": False,
            "head": args.expected_head,
        }
        atomic_json(args.output, value)
        return 0 if not errors else 1
    score_physical_batch_size = int(plan.get("metadata", {}).get("score_physical_batch_size", 1))
    if score_physical_batch_size not in {1, 2}:
        errors.append(f"unsupported Score physical batch size={score_physical_batch_size}")
    if args.execution_mode == "baseline" and score_physical_batch_size != 1:
        errors.append("baseline physical Score batch must remain B1")
    critic_physical_batch_size = int(
        plan.get("metadata", {}).get("critic_physical_batch_size", 1)
    )
    if critic_physical_batch_size not in {1, 2}:
        errors.append(
            f"unsupported Critic physical batch size={critic_physical_batch_size}"
        )
    if args.execution_mode == "baseline" and critic_physical_batch_size != 1:
        errors.append("baseline physical Critic batch must remain B1")
    rows: list[dict[str, Any]] = []
    for rank in range(WORLD_SIZE):
        path = args.run_root / "receipts" / f"rank_{rank:03d}.json"
        if not path.is_file():
            errors.append(f"rank{rank}: receipt missing")
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        rows.append(row)
        if row.get("status") != "PASS" or int(row.get("rank", -1)) != rank:
            errors.append(f"rank{rank}: status/rank mismatch")
        if row.get("mode") != args.execution_mode or row.get("run_kind") != args.run_kind:
            errors.append(f"rank{rank}: run identity mismatch")
        if row.get("trace_sha256") != trace.sha256:
            errors.append(f"rank{rank}: trace SHA mismatch")
        if row.get("logical_generator_updates") != 1 or row.get("logical_critic_updates") != 5:
            errors.append(f"rank{rank}: logical update inventory changed")
        if row.get("final_versions") != {"generator": 1, "critic": 5, "ema": 0}:
            errors.append(f"rank{rank}: final versions changed")
        before = row.get("cgroup_oom_before", {})
        after = row.get("cgroup_oom_after", {})
        for key in ("oom", "oom_kill"):
            if int(after.get(key, 0)) != int(before.get(key, 0)):
                errors.append(f"rank{rank}: cgroup {key} changed")
    if len(rows) != WORLD_SIZE:
        errors.append(f"rank receipt count={len(rows)}, expected=16")

    identities = {
        (row.get("trace_sha256"), row.get("bootstrap_sha256"), row.get("source_receipt_sha256"))
        for row in rows
    }
    if len(identities) != 1:
        errors.append("source/bootstrap/trace identity differs by rank")

    expected_physical = {
        rank: (
            1 if rank in generator_ranks else 0,
            5 if rank in critic_ranks else 0,
        )
        for rank in range(WORLD_SIZE)
    }
    for row in rows:
        rank = int(row["rank"])
        actual = (row.get("physical_generator_commits"), row.get("physical_critic_commits"))
        if actual != expected_physical[rank]:
            errors.append(f"rank{rank}: physical commit inventory={actual}, expected={expected_physical[rank]}")

    expected_logical_inventory = {
        # Each sample first runs one clean world-prefix DiT to prime the
        # deployment K/V cache, then five calls per generated block (four
        # denoising evaluations plus the clean commit pass).
        "student/nograd/plain": sum(1 + row.new_blocks * 5 for row in trace.rows),
        "student/grad/checkpoint": 16 * 3,
        "teacher/nograd/plain": 16,
        "critic/nograd/plain": 16,
        "critic/grad/checkpoint": 80,
    }
    expected_physical_inventory = dict(expected_logical_inventory)
    if pool_policy == "persistent_8g_8score_teacher_fsdp8_score_b2":
        expected_physical_inventory.update(
            {
                "teacher/nograd/plain": 8,
                "critic/nograd/plain": 8,
                "critic/grad/checkpoint": 40,
            }
        )
    elif pool_policy in {
        "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2",
        "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2_local_context",
        "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2_local_context_critic_async32",
        "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2_local_context_critic_async32_trace_balanced",
    }:
        expected_physical_inventory.update(
            {
                # Step0 DMD/Teacher remains two physical B1 calls per Score
                # rank; only the two Critic training samples are exact B2.
                "teacher/nograd/plain": 16,
                "critic/nograd/plain": 16,
                "critic/grad/checkpoint": 40,
            }
        )
    if (
        pool_policy
        in {
            "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2_local_context_critic_async32",
            "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2_local_context_critic_async32_trace_balanced",
        }
    ):
        for row in rows:
            if int(row["rank"]) not in score_ranks:
                continue
            receipts = [
                value
                for value in row.get("gradient_sync", [])
                if value.get("role") == "critic"
            ]
            if len(receipts) != 5:
                errors.append(f"rank{row['rank']}: async Critic sync receipt count={len(receipts)}")
                continue
            for value in receipts:
                if (
                    value.get("backend") != "GRAD_READY_ASYNC_FLAT_ALL_REDUCE"
                    or value.get("reduction") != "SUM_THEN_DIVIDE_GROUP_SIZE"
                    or value.get("group_size") != 8
                    or value.get("gradient_tensors") != 360
                    or value.get("gradient_bytes") != 127795200
                    or not isinstance(value.get("buckets"), int)
                    or value.get("buckets") < 4
                    or value.get("bucket_issue_order")
                    != list(range(value.get("buckets", 0)))
                    or sum(value.get("bucket_bytes_observed", [])) != 127795200
                    or max(value.get("bucket_bytes_observed", [2**63]))
                    > 32 * 1024 * 1024
                    or value.get("hook_count_min") != 1
                    or value.get("hook_count_max") != 1
                    or value.get("overlap_with_backward") is not True
                ):
                    errors.append(
                        f"rank{row['rank']}: invalid async Critic sync receipt at "
                        f"step {value.get('step')}"
                    )
    elif pool_policy == PHASE_EXPANDED_CRITIC16_POOL_POLICY:
        for row in rows:
            receipts = [
                value
                for value in row.get("gradient_sync", [])
                if value.get("role") == "critic"
            ]
            if len(receipts) != FULL5_STEPS:
                errors.append(
                    f"rank{row['rank']}: FSDP2 Critic sync receipt count={len(receipts)}"
                )
                continue
            for value in receipts:
                if value != {
                    "step": value.get("step"),
                    "role": "critic",
                    "backend": "FSDP2_REDUCE_SCATTER",
                    "group_size": WORLD_SIZE,
                    "reduction": "SUM_THEN_DIVIDE_GROUP_SIZE",
                    "overlap_with_backward": True,
                } or value.get("step") not in range(FULL5_STEPS):
                    errors.append(
                        f"rank{row['rank']}: invalid FSDP2 Critic sync receipt at "
                        f"step {value.get('step')}"
                    )
    elif pool_policy in {
        "persistent_6g_10score_teacher_fsdp10",
        "persistent_6g_10score_teacher_node_local_hierarchical",
    }:
        expected_physical_inventory.update(
            {
                "teacher/nograd/plain": 10,
                "critic/nograd/plain": 10,
                "critic/grad/checkpoint": 50,
            }
        )
    actual_logical_inventory: dict[str, int] = {}
    actual_physical_inventory: dict[str, int] = {}
    for row in rows:
        logical = row.get("logical_call_inventory", row.get("call_inventory", {}))
        physical = row.get("physical_call_inventory", row.get("call_inventory", {}))
        for key, count in logical.items():
            actual_logical_inventory[key] = actual_logical_inventory.get(key, 0) + int(count)
        for key, count in physical.items():
            actual_physical_inventory[key] = actual_physical_inventory.get(key, 0) + int(count)
        if int(row.get("score_physical_batch_size", score_physical_batch_size)) != score_physical_batch_size:
            errors.append(f"rank{row['rank']}: Score physical batch receipt differs from plan")
        if int(
            row.get("critic_physical_batch_size", critic_physical_batch_size)
        ) != critic_physical_batch_size:
            errors.append(
                f"rank{row['rank']}: Critic physical batch receipt differs from plan"
            )
        if row.get("pool_policy") != pool_policy:
            errors.append(f"rank{row['rank']}: pool policy receipt differs from plan")
        expected_lens_receipt = {
            "enabled": full_context_lens_elision,
            "context_length": 512,
            "padding_policy": (
                "all_512_tokens_valid_no_padding"
                if full_context_lens_elision
                else "explicit_context_lens"
            ),
            "attention_mask_equivalence": (
                "ctx_lens_512_equivalent_to_none_at_exact_tensor_length"
                if full_context_lens_elision
                else "not_applied"
            ),
        }
        if row.get("full_context_lens_elision") != expected_lens_receipt:
            errors.append(
                f"rank{row['rank']}: full-context lens-elision receipt mismatch"
            )
        prefetch_receipt = row.get("score_context_prefetch", {})
        if score_context_prefetch:
            rank = int(row["rank"])
            expected_prefetch = {
                "enabled": True,
                "role": "generator" if rank in generator_ranks else "score",
                "execution_status": (
                    "NOT_APPLICABLE" if rank in generator_ranks else "PASS"
                ),
                "logical_context_count": 0 if rank in generator_ranks else 10,
                "critic_context_count": 0 if rank in generator_ranks else 10,
                "teacher_context_count": 0 if rank in generator_ranks else 2,
                "timing_position": "inside_full5_before_first_latent_receive",
                "calls_preserved": True,
            }
            for key, expected_value in expected_prefetch.items():
                if prefetch_receipt.get(key) != expected_value:
                    errors.append(
                        f"rank{rank}: Score context prefetch receipt mismatch for {key}"
                    )
            resident_bytes = int(prefetch_receipt.get("gpu_resident_bytes", -1))
            if rank in generator_ranks and resident_bytes != 0:
                errors.append(f"rank{rank}: Generator unexpectedly owns prefetched contexts")
            if rank in score_ranks and resident_bytes <= 0:
                errors.append(f"rank{rank}: Score prefetched-context bytes missing")
        elif prefetch_receipt.get("enabled") not in {False, None}:
            errors.append(f"rank{row['rank']}: unregistered Score context prefetch")
        lora_kernel_receipt = row.get("critic_lora_kernel", {})
        if critic_lora_kernel_enabled:
            rank = int(row["rank"])
            phase_expanded = pool_policy == PHASE_EXPANDED_CRITIC16_POOL_POLICY
            expected_lora_kernel = {
                "enabled": True,
                "name": critic_lora_kernel_contract.get("name"),
                "source_forward_sha256": (
                    "70d00d496aa2bd8ea06d64e6b04af873644eff52539396f74ec586a4cbbe97a3"
                ),
                "class_forward_patched": True,
                "role": (
                    "critic"
                    if phase_expanded
                    else (
                        "generator_shadow"
                        if split_dmd_shadow_enabled and rank in generator_ranks
                        else ("generator" if rank in generator_ranks else "score")
                    )
                ),
                "execution_status": (
                    "PASS"
                    if phase_expanded or (split_dmd_shadow_enabled and rank in generator_ranks)
                    else ("NOT_APPLICABLE" if rank in generator_ranks else "PASS")
                ),
                "module_count": (
                    180
                    if phase_expanded or split_dmd_shadow_enabled or rank not in generator_ranks
                    else 0
                ),
                "all_scales_one": True,
            }
            for key, expected_value in expected_lora_kernel.items():
                if lora_kernel_receipt.get(key) != expected_value:
                    errors.append(
                        f"rank{rank}: Critic LoRA kernel receipt mismatch for {key}"
                    )
            if lora_kernel_receipt.get("install_status") not in {
                "PASS", "ALREADY_INSTALLED"
            }:
                errors.append(f"rank{rank}: Critic LoRA kernel installation failed")
            fused_expected = (
                critic_lora_kernel_contract.get("name")
                == "scale_one_fused_cast_add_v1"
            )
            if lora_kernel_receipt.get("fused_cast_add") is not fused_expected:
                errors.append(f"rank{rank}: Critic LoRA fused-epilogue mode mismatch")
            autocast_elision_expected = (
                critic_lora_kernel_contract.get("name")
                in {"scale_one_autocast_bf16_input_elision_v1",
                    "scale_one_autocast_shared_bf16_input_v1"}
            )
            if critic_lora_kernel_contract.get("name") == "scale_one_autocast_shared_bf16_input_v1":
                from distillschedule.runtime.livewan_lora_shared_input import runtime_receipt_errors
                errors.extend(f"rank{rank}: {error}" for error in
                    runtime_receipt_errors(lora_kernel_receipt, rank=rank))
            if (
                lora_kernel_receipt.get("autocast_input_elision")
                is not autocast_elision_expected
            ):
                errors.append(f"rank{rank}: Critic LoRA autocast-elision mode mismatch")
            if autocast_elision_expected:
                expected_invocations = (
                    phase_expanded
                    or split_dmd_shadow_enabled
                    or rank not in generator_ranks
                )
                invocation_count = int(
                    lora_kernel_receipt.get(
                        "autocast_input_elision_invocation_count", -1
                    )
                )
                if (invocation_count > 0) is not expected_invocations:
                    errors.append(
                        f"rank{rank}: Critic LoRA autocast-elision invocation mismatch"
                    )
                if int(lora_kernel_receipt.get("autocast_contract_failure_count", -1)) != 0:
                    errors.append(f"rank{rank}: Critic LoRA autocast contract failed")
            if fused_expected and (
                lora_kernel_receipt.get("fused_min_numel") != 8_000_000
                or lora_kernel_receipt.get("triton_block_size") != 256
                or not lora_kernel_receipt.get("triton_version")
            ):
                errors.append(f"rank{rank}: Critic LoRA fused-kernel receipt invalid")
        elif lora_kernel_receipt.get("enabled") not in {False, None}:
            errors.append(f"rank{row['rank']}: unregistered Critic LoRA kernel")
        modulation_receipt = row.get("score_modulation_kernel", {})
        if score_modulation_kernel_enabled:
            rank = int(row["rank"])
            expected_modulation = {
                "enabled": True,
                "name": "score_bf16_modulation_fusion_v1",
                "source_modulate_sha256": (
                    "f244152360d40d6745e28bed4add081604312d04fe2f342842a32432f4bd20ad"
                ),
                "function_patched": rank in score_ranks,
                "install_status": "PASS" if rank in score_ranks else "NOT_APPLICABLE",
                "role": "score" if rank in score_ranks else "generator",
                "execution_status": (
                    "PASS" if rank in score_ranks else "NOT_APPLICABLE"
                ),
                "invocation_count": 889 if rank in score_ranks else 0,
                # Teacher real-score is one physical B2 call per Score rank;
                # Critic no-grad remains two B1 calls.  Only the 2*30 block
                # modulation functions per Critic update replay in backward;
                # the frozen output head does not request an input gradient.
                "backward_invocation_count": 300 if rank in score_ranks else 0,
                "batch_histogram": {"1": 122, "2": 767} if rank in score_ranks else {},
                "dimension_histogram": (
                    {"1536": 727, "5120": 162} if rank in score_ranks else {}
                ),
                "observed_contract": (
                    {
                        "normalized_dtype": "torch.bfloat16",
                        "shift_dtype": "torch.bfloat16",
                        "scale_dtype": "torch.bfloat16",
                        "output_dtype": "torch.float32",
                        "per_frame": False,
                    }
                    if rank in score_ranks else None
                ),
            }
            for key, expected_value in expected_modulation.items():
                if modulation_receipt.get(key) != expected_value:
                    errors.append(
                        f"rank{rank}: Score modulation receipt mismatch for {key}: "
                        f"observed={modulation_receipt.get(key)!r}, "
                        f"expected={expected_value!r}"
                    )
            if rank in score_ranks:
                if modulation_receipt.get("triton_block_size") != 256:
                    errors.append(f"rank{rank}: Score modulation block size changed")
                if not modulation_receipt.get("triton_version"):
                    errors.append(f"rank{rank}: Score modulation Triton identity missing")
                if int(modulation_receipt.get("element_count", 0)) <= 0:
                    errors.append(f"rank{rank}: Score modulation executed no elements")
        elif modulation_receipt.get("enabled") not in {False, None}:
            errors.append(f"rank{row['rank']}: unregistered Score modulation kernel")
        rope_receipt = row.get("score_rope_kernel", {})
        if score_rope_kernel_enabled:
            rank = int(row["rank"])
            expected_rope = {
                "enabled": True,
                "name": "score_bf16_rope_fusion_v1",
                "source_apply_rope_sha256": (
                    "8ab587e9f15cc969659609ec2f9cc79f4d4cf6fba468ae8c54267dc949224482"
                ),
                "function_patched": rank in score_ranks,
                "install_status": "PASS" if rank in score_ranks else "NOT_APPLICABLE",
                "role": "score" if rank in score_ranks else "generator",
                "execution_status": (
                    "PASS" if rank in score_ranks else "NOT_APPLICABLE"
                ),
                "triton_block_size": 512,
                "fp32_rounding_order": "mul_then_first_product_fma_rn",
                # Five B2 Critic forward/replay pairs contribute 600 Q/K
                # rotations.  The two logical Teacher rows are executed as the
                # pinned B2 CFG path and contribute another 160 rotations.
                "invocation_count": 760 if rank in score_ranks else 0,
                "backward_invocation_count": 300 if rank in score_ranks else 0,
                "element_count": 67_264_512_000 if rank in score_ranks else 0,
                "backward_element_count": 17_805_312_000 if rank in score_ranks else 0,
                "shape_histogram": (
                    {
                        "2x19320x12x128": 600,
                        "2x19320x40x128": 160,
                    }
                    if rank in score_ranks else {}
                ),
                "table_rank_histogram": {"3": 760} if rank in score_ranks else {},
                "contract_failure_count": 0,
            }
            for key, expected_value in expected_rope.items():
                if rope_receipt.get(key) != expected_value:
                    errors.append(
                        f"rank{rank}: Score RoPE receipt mismatch for {key}: "
                        f"observed={rope_receipt.get(key)!r}, "
                        f"expected={expected_value!r}"
                    )
            if rank in score_ranks:
                if not rope_receipt.get("triton_version"):
                    errors.append(f"rank{rank}: Score RoPE Triton identity missing")
                if int(rope_receipt.get("gradient_contiguous_copy_count", -1)) < 0:
                    errors.append(f"rank{rank}: Score RoPE gradient-copy receipt invalid")
        elif rope_receipt.get("enabled") not in {False, None}:
            errors.append(f"rank{row['rank']}: unregistered Score RoPE kernel")
        attention_receipt = row.get("score_attention_backend", {})
        if score_attention_backend_enabled:
            rank = int(row["rank"])
            expected_attention = {
                "enabled": True,
                "name": "score_flash_attn2_unmasked_bf16_v1",
                "source_flash_attention_sha256": (
                    "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59"
                ),
                "artifact_metadata_sha256": (
                    "7749dd68b81a544c1973d68c30be5aa316ee1a718ae0228e0368c2cd04365516"
                ),
                "artifact_package_sha256": (
                    "0c560f96b857c188c4a8297ff27d1299619957c3459a5e433e132329fbfdf066"
                ),
                "artifact_extension_sha256": (
                    "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
                ),
                "function_patched": rank in score_ranks,
                "install_status": "PASS" if rank in score_ranks else "NOT_APPLICABLE",
                "role": "score" if rank in score_ranks else "generator",
                "execution_status": (
                    "PASS" if rank in score_ranks else "NOT_APPLICABLE"
                ),
                "invocation_count": 760 if rank in score_ranks else 0,
                "backward_invocation_count": 300 if rank in score_ranks else 0,
                "element_count": 67_264_512_000 if rank in score_ranks else 0,
                "backward_element_count": 17_805_312_000 if rank in score_ranks else 0,
                "shape_histogram": (
                    {
                        "q=2x19320x12x128|k=2x19320x12x128": 300,
                        "q=2x19320x12x128|k=2x512x12x128": 300,
                        "q=2x19320x40x128|k=2x19320x40x128": 80,
                        "q=2x19320x40x128|k=2x512x40x128": 80,
                    }
                    if rank in score_ranks else {}
                ),
                "contract_failure_count": 0,
            }
            for key, expected_value in expected_attention.items():
                if attention_receipt.get(key) != expected_value:
                    errors.append(
                        f"rank{rank}: Score attention receipt mismatch for {key}: "
                        f"observed={attention_receipt.get(key)!r}, "
                        f"expected={expected_value!r}"
                    )
            if rank in score_ranks and not attention_receipt.get("artifact_root"):
                errors.append(f"rank{rank}: Score attention artifact root missing")
        elif attention_receipt.get("enabled") not in {False, None}:
            errors.append(f"rank{row['rank']}: unregistered Score attention backend")
        cross_kv_receipt = row.get("generator_cross_attention_kv_cache", {})
        if generator_cross_kv_cache_enabled:
            rank = int(row["rank"])
            is_generator = rank in generator_ranks
            expected_cross_kv = {
                "enabled": True,
                "name": "generator_nograd_text_kv_cache_v1",
                "source_set_text_sha256": (
                    "def2452ea9528abb9af2aa166d78afdf9b8b7c5daa52c0fc87bfd63bc9832386"
                ),
                "source_cross_forward_sha256": (
                    "0df9365afb37bb8a876d8c8659ca95d1caaffb89bf3e64918b782eb86b0a8253"
                ),
                "source_flash_attention_sha256": (
                    "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59"
                ),
                "function_patched": is_generator,
                "install_status": "PASS" if is_generator else "NOT_APPLICABLE",
                "role": "generator" if is_generator else "score",
                "execution_status": "PASS" if is_generator else "NOT_APPLICABLE",
                "cache_build_context_count": 10 if is_generator else 0,
                "cache_build_block_count": 300 if is_generator else 0,
                "cache_build_bytes": 1_415_577_600 if is_generator else 0,
                "cache_invalidation_count": 8 if is_generator else 0,
                "grad_bypass_count": 360 if is_generator else 0,
                "uncached_fallback_count": 60 if is_generator else 0,
                "contract_failure_count": 0,
            }
            if is_generator:
                lane = generator_ranks.index(rank)
                slots = TRACE_BALANCED_8G8S_SLOT_PAIRS[lane]
                lane_rows = [
                    trace.row(step, slot)
                    for step in range(FULL5_STEPS)
                    for slot in slots
                ]
                expected_cross_kv["cache_hit_count"] = sum(
                    (1 + trace_row.new_blocks * 5) * 30
                    for trace_row in lane_rows
                )
                shape_histogram: dict[str, int] = {}
                for trace_row in lane_rows:
                    world_key = f"1x{trace_row.world_frames * 920}x1536"
                    shape_histogram[world_key] = shape_histogram.get(world_key, 0) + 30
                    event_key = "1x2760x1536"
                    shape_histogram[event_key] = shape_histogram.get(event_key, 0) + (
                        trace_row.new_blocks * 5 * 30
                    )
                expected_cross_kv["query_shape_histogram"] = shape_histogram
            else:
                expected_cross_kv["cache_hit_count"] = 0
                expected_cross_kv["query_shape_histogram"] = {}
            for key, expected_value in expected_cross_kv.items():
                if cross_kv_receipt.get(key) != expected_value:
                    errors.append(
                        f"rank{rank}: Generator cross K/V receipt mismatch for {key}: "
                        f"observed={cross_kv_receipt.get(key)!r}, "
                        f"expected={expected_value!r}"
                    )
        elif cross_kv_receipt.get("enabled") not in {False, None}:
            errors.append(
                f"rank{row['rank']}: unregistered Generator cross K/V cache"
            )
        generator_attention_receipt = row.get("generator_attention_backend", {})
        if generator_attention_backend_enabled:
            rank = int(row["rank"])
            is_generator = rank in generator_ranks
            expected_generator_attention = {
                "enabled": True,
                "name": "generator_flash_attn2_unmasked_exact_cast_v1",
                "source_flash_attention_sha256": (
                    "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59"
                ),
                "artifact_metadata_sha256": (
                    "7749dd68b81a544c1973d68c30be5aa316ee1a718ae0228e0368c2cd04365516"
                ),
                "artifact_package_sha256": (
                    "0c560f96b857c188c4a8297ff27d1299619957c3459a5e433e132329fbfdf066"
                ),
                "artifact_extension_sha256": (
                    "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
                ),
                "function_patched": is_generator,
                "install_status": "PASS" if is_generator else "NOT_APPLICABLE",
                "role": "generator" if is_generator else "score",
                "execution_status": "PASS" if is_generator else "NOT_APPLICABLE",
                "contract_failure_count": 0,
            }
            for key, expected_value in expected_generator_attention.items():
                if generator_attention_receipt.get(key) != expected_value:
                    errors.append(
                        f"rank{rank}: Generator attention receipt mismatch for {key}: "
                        f"observed={generator_attention_receipt.get(key)!r}, "
                        f"expected={expected_value!r}"
                    )
            invocation_count = int(
                generator_attention_receipt.get("invocation_count", -1)
            )
            backward_count = int(
                generator_attention_receipt.get("backward_invocation_count", -1)
            )
            shape_histogram = dict(
                generator_attention_receipt.get("shape_histogram", {})
            )
            dtype_histogram = dict(
                generator_attention_receipt.get("raw_dtype_histogram", {})
            )
            if is_generator:
                if invocation_count <= 0 or sum(shape_histogram.values()) != invocation_count:
                    errors.append(
                        f"rank{rank}: Generator attention invocation/shape inventory invalid"
                    )
                if sum(dtype_histogram.values()) != invocation_count:
                    errors.append(
                        f"rank{rank}: Generator attention dtype inventory invalid"
                    )
                allowed_dtypes = {
                    "bfloat16/bfloat16/bfloat16",
                    "float32/float32/bfloat16",
                }
                if not dtype_histogram or not set(dtype_histogram).issubset(allowed_dtypes):
                    errors.append(
                        f"rank{rank}: Generator attention raw dtype contract changed: "
                        f"{dtype_histogram}"
                    )
                fp32_cross_count = int(
                    dtype_histogram.get("float32/float32/bfloat16", 0)
                )
                if int(generator_attention_receipt.get("input_cast_count", -1)) != (
                    2 * fp32_cross_count
                ):
                    errors.append(
                        f"rank{rank}: Generator attention input cast inventory invalid"
                    )
                if int(generator_attention_receipt.get("output_cast_count", -1)) != (
                    fp32_cross_count
                ):
                    errors.append(
                        f"rank{rank}: Generator attention output cast inventory invalid"
                    )
                if backward_count <= 0 or backward_count > invocation_count:
                    errors.append(
                        f"rank{rank}: Generator attention backward inventory invalid"
                    )
                if int(generator_attention_receipt.get("element_count", 0)) <= 0:
                    errors.append(f"rank{rank}: Generator attention element inventory empty")
                if not generator_attention_receipt.get("artifact_root"):
                    errors.append(f"rank{rank}: Generator attention artifact root missing")
            else:
                for key, expected_value in {
                    "invocation_count": 0,
                    "backward_invocation_count": 0,
                    "element_count": 0,
                    "backward_element_count": 0,
                    "shape_histogram": {},
                    "raw_dtype_histogram": {},
                    "input_cast_count": 0,
                    "output_cast_count": 0,
                }.items():
                    if generator_attention_receipt.get(key) != expected_value:
                        errors.append(
                            f"rank{rank}: non-Generator attention receipt mismatch for {key}"
                        )
        elif generator_attention_receipt.get("enabled") not in {False, None}:
            errors.append(
                f"rank{row['rank']}: unregistered Generator attention backend"
            )
        runahead_transport_receipt = row.get("generator_runahead_transport", {})
        if generator_runahead_transport_enabled:
            rank = int(row["rank"])
            is_generator = rank in generator_ranks
            expected_runahead_transport = {
                "enabled": True,
                "name": "pinned_cpu_gloo_latent_v1",
                "role": "generator" if is_generator else "score",
                "execution_status": "PASS",
                "backend": "gloo",
                "scope": "critic_only_steps_1_4_latent_only",
                "step0_transport": "cuda_nccl_p2p_unchanged",
                "context_transport": "trace_local_replay_unchanged",
                "payload_shape": [1, 16, 21, 46, 80],
                "payload_dtype": "torch.float32",
                "payload_bytes": 4_945_920,
                "buffer_count": 8,
                "pinned_buffer_count": 8,
                "send_event_count": 8 if is_generator else 0,
                "receive_event_count": 0 if is_generator else 8,
                "d2h_bytes": 39_567_360 if is_generator else 0,
                "h2d_bytes": 0 if is_generator else 39_567_360,
                "tags": [27102, 27103, 27104, 27105, 27106, 27107, 27108, 27109],
                "sender_threads": 1 if is_generator else 0,
                "preposted_receives": not is_generator,
            }
            if runahead_transport_receipt != expected_runahead_transport:
                errors.append(
                    f"rank{rank}: Generator run-ahead host transport receipt mismatch"
                )
        elif runahead_transport_receipt.get("enabled") not in {False, None}:
            errors.append(
                f"rank{row['rank']}: unregistered Generator run-ahead transport"
            )
        split_receipt = row.get("split_dmd_shadow", {})
        if split_dmd_shadow_enabled:
            rank = int(row["rank"])
            expected_split_receipt = {
                "enabled": True,
                "role": "generator" if rank in generator_ranks else "score",
                "execution_status": "PASS",
                "fake_velocity_calls": 2 if rank in generator_ranks else 0,
                "teacher_velocity_calls": 0 if rank in generator_ranks else 2,
                "received_teacher_velocities": 2 if rank in generator_ranks else 0,
                "received_final_gradients": 0,
            }
            if split_receipt != expected_split_receipt:
                errors.append(f"rank{rank}: split-DMD execution receipt mismatch")
            build = row.get("build", {})
            identity = build.get("critic_shadow") if rank in generator_ranks else build.get("critic")
            if (
                (identity or {}).get("initial_version") != 0
                or not (identity or {}).get("initial_lora_sha256")
            ):
                errors.append(f"rank{rank}: split-DMD C0 identity receipt missing")
        elif split_receipt.get("enabled") not in {False, None}:
            errors.append(f"rank{row['rank']}: unregistered split-DMD execution")
        if pool_policy in {
            "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2_local_context_critic_async32_trace_balanced",
            PHASE_EXPANDED_CRITIC16_POOL_POLICY,
        }:
            rank = int(row["rank"])
            lane = rank if rank < 8 else rank - 8
            expected_routing = {
                "policy": (
                    "trace_balanced_phase_expanded_critic16_v1"
                    if pool_policy == PHASE_EXPANDED_CRITIC16_POOL_POLICY
                    else "trace_balanced_fixed_rank_pairs_v1"
                ),
                "lane": lane,
                "logical_slots": list(TRACE_BALANCED_8G8S_SLOT_PAIRS[lane]),
                "peer_rank": rank + 8 if rank < 8 else rank - 8,
                "all_pairs": [list(pair) for pair in TRACE_BALANCED_8G8S_SLOT_PAIRS],
            }
            if pool_policy == PHASE_EXPANDED_CRITIC16_POOL_POLICY:
                expected_routing["critic_slot"] = int(
                    TRACE_BALANCED_8G8S_SLOT_PAIRS[lane][1 if rank < 8 else 0]
                )
            if row.get("slot_routing") != expected_routing:
                errors.append(f"rank{rank}: trace-balanced slot-routing receipt mismatch")
    if split_dmd_shadow_enabled:
        split_initial_lora_shas = {
            (
                row.get("build", {}).get("critic_shadow")
                or row.get("build", {}).get("critic")
                or {}
            ).get("initial_lora_sha256")
            for row in rows
        }
        if len(split_initial_lora_shas) != 1 or None in split_initial_lora_shas:
            errors.append("split-DMD C0 Critic/Shadow initial LoRA SHA mismatch")
    for key, expected in expected_logical_inventory.items():
        if actual_logical_inventory.get(key, 0) != expected:
            errors.append(
                f"logical call inventory {key}={actual_logical_inventory.get(key, 0)}, expected={expected}"
            )
    for key, expected in expected_physical_inventory.items():
        if actual_physical_inventory.get(key, 0) != expected:
            errors.append(
                f"physical call inventory {key}={actual_physical_inventory.get(key, 0)}, expected={expected}"
            )
    if any(
        count
        for key, count in actual_physical_inventory.items()
        if key == "unknown" or key.startswith("unknown/")
    ):
        errors.append("unknown model calls were observed")

    output_keys: list[str] = []
    generator_loss_count = critic_loss_count = 0
    for row in rows:
        output_keys.extend(item["operation_key"] for item in row.get("output_fingerprints", []))
        for item in row.get("loss_fingerprints", []):
            if item.get("kind") == "generator":
                generator_loss_count += 1
            elif item.get("kind") == "critic":
                critic_loss_count += 1
    if args.run_kind == "correctness":
        expected_keys = {f"step={step}/slot={slot}" for step in range(FULL5_STEPS) for slot in range(GLOBAL_BATCH)}
        if len(output_keys) != 80 or set(output_keys) != expected_keys:
            errors.append("rank-qualified rollout evidence is not a 5x16 bijection")
        if generator_loss_count != 48 or critic_loss_count != 80:
            errors.append(
                f"backward loss census G={generator_loss_count}/48 C={critic_loss_count}/80"
            )
        for row in rows:
            artifact = row.get("output_tensor_artifact")
            owns_generator = int(row["rank"]) in generator_ranks
            if owns_generator and (not artifact or not Path(artifact["path"]).is_file()):
                errors.append(f"rank{row['rank']}: rollout tensor artifact missing")

    # Owner replicas must finish with the same trainable-state statistics.
    for name, owners in (
        ("generator_state_metrics", [row for row in rows if row.get("generator_state_metrics")]),
        ("critic_state_metrics", [row for row in rows if row.get("critic_state_metrics")]),
    ):
        if not owners:
            errors.append(f"{name} missing")
            continue
        for key in ("numel", "sum", "sum_sq", "max_abs", "weighted_sum", "weighted_sum_sq"):
            values = [float(row[name][key]) for row in owners]
            if not _close(values):
                errors.append(f"{name}/{key} differs across owner ranks")

    walls = [float(row["full5_wall_seconds"]) for row in rows]
    peak_allocated = [int(row["peak_memory"]["max_allocated_bytes"]) for row in rows]
    peak_reserved = [int(row["peak_memory"]["max_reserved_bytes"]) for row in rows]
    min_free = [int(row["peak_memory"]["driver_free_bytes"]) for row in rows]
    value = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "run_kind": args.run_kind,
        "execution_mode": args.execution_mode,
        "rank_count": len(rows),
        "trace_sha256": trace.sha256,
        "identity": next(iter(identities)) if len(identities) == 1 else None,
        "workload": {
            "world_size": 16,
            "global_batch": 16,
            "steps": 5,
            "generator_updates": 1,
            "critic_updates": 5,
            "ema_updates_in_full5": 0,
            "source_shape": "B1x16x21x46x80@368x640",
            "attention": "LiveWan pinned source",
        },
        # ``call_inventory`` remains the logical workload inventory for paired
        # compatibility. Exact physical batching is reported separately.
        "call_inventory": actual_logical_inventory,
        "logical_call_inventory": actual_logical_inventory,
        "expected_logical_call_inventory": expected_logical_inventory,
        "physical_call_inventory": actual_physical_inventory,
        "expected_physical_call_inventory": expected_physical_inventory,
        "score_physical_batch_size": score_physical_batch_size,
        "critic_physical_batch_size": critic_physical_batch_size,
        "pool_policy": pool_policy,
        "full_context_lens_elision": full_context_lens_elision,
        "score_context_prefetch": score_context_prefetch,
        "critic_lora_kernel": critic_lora_kernel_enabled,
        "score_modulation_kernel": score_modulation_kernel_enabled,
        "score_rope_kernel": score_rope_kernel_enabled,
        "score_attention_backend": score_attention_backend_enabled,
        "generator_cross_attention_kv_cache": generator_cross_kv_cache_enabled,
        "generator_attention_backend": generator_attention_backend_enabled,
        "generator_runahead_transport": generator_runahead_transport_contract,
        "split_dmd_shadow": split_dmd_shadow_contract,
        "generator_ranks": list(generator_ranks),
        "score_ranks": list(score_ranks),
        "critic_ranks": list(critic_ranks),
        "full5_seconds": max(walls) if walls else None,
        "rank_wall_min_seconds": min(walls) if walls else None,
        "rank_wall_max_seconds": max(walls) if walls else None,
        "peak_allocated_gib": max(peak_allocated) / 2**30 if peak_allocated else None,
        "peak_reserved_gib": max(peak_reserved) / 2**30 if peak_reserved else None,
        "minimum_driver_free_gib": min(min_free) / 2**30 if min_free else None,
        "transport_endpoint_bytes": sum(int(row.get("transport_bytes", 0)) for row in rows),
        "formal_timing_eligible": args.run_kind == "timing" and not errors,
        "profiled_time_is_formal": False,
        "head": args.expected_head,
    }
    atomic_json(args.output, value)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
