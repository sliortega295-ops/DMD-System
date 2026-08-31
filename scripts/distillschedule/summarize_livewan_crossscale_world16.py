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

from distillschedule.workloads.livewan_crossscale import FULL5_STEPS, GLOBAL_BATCH, WORLD_SIZE
from distillschedule.runtime.livewan_crossscale_worker import FormalTrace, atomic_json


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-root", required=True, type=Path)
    value.add_argument("--execution-mode", required=True, choices=("baseline", "pipeline"))
    value.add_argument("--run-kind", required=True, choices=("validate", "correctness", "timing", "profile"))
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
            5 if rank in score_ranks else 0,
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
        == "persistent_8g_8score_teacher_fsdp8_runahead_critic_b2_local_context_critic_async32"
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
        "generator_ranks": list(generator_ranks),
        "score_ranks": list(score_ranks),
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
