#!/usr/bin/env python3
"""Summarize one exact-head, native, same-process LiveWan two-cycle timing arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from distillschedule.runtime.livewan_crossscale_worker import (  # noqa: E402
    FormalTrace,
    TWO_CYCLE_TIMING_CONTINUOUS_NAME,
    atomic_json,
)


WORLD_SIZE = 16
GENERATOR_RANKS = tuple(range(8))
SCORE_RANKS = tuple(range(8, 16))
BASELINE_VARIANT = "b1"
PIPELINE_VARIANT = "c24-continuous-b16-score-modulation"
FINAL_VERSIONS = {"generator": 2, "critic": 10, "ema": 0}
CORE_CALL_KEYS = (
    "student/nograd/plain",
    "student/grad/checkpoint",
    "teacher/nograd/plain",
    "critic/nograd/plain",
    "critic/grad/checkpoint",
)


def _manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _status(path: Path, *, owner_run_id: str, wait_seconds: float = 30.0) -> dict[str, str]:
    """Wait only for missing/partial shared-FS publication, outside all timers.

    Explicit FAIL, nonzero GPU count, or a different owner is never retried or
    repaired. Timeout returns the observed failure for the ordinary gate.
    """
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            observed = _manifest(path)
        except FileNotFoundError:
            observed = {}
        if (observed.get("status") not in {None, "PASS"}
            or observed.get("gpu_process_count") not in {None, "0"}
            or observed.get("owner_run_id") not in {None, owner_run_id}
            or set(observed) >= {"status", "gpu_process_count", "owner_run_id"}
            or time.monotonic() >= deadline):
            return observed
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))


def _memory_events(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2:
            result[fields[0]] = int(fields[1])
    return result


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        for name, count in dict(row.get(key, {})).items():
            result[str(name)] = result.get(str(name), 0) + int(count)
    return result


def _core(value: dict[str, int]) -> dict[str, int]:
    return {key: int(value.get(key, 0)) for key in CORE_CALL_KEYS}


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _effective_plan(path: Path) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(value.pop("effective_plan_sha256", ""))
    actual = _sha256_json(value)
    if recorded != actual:
        raise ValueError("effective plan SHA mismatch")
    return value, actual


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "max": max(values) if values else None,
    }


def _expected_inventory(trace: FormalTrace, *, mode: str, pipeline_variant: str | None = None) -> tuple[dict[str, int], dict[str, int]]:
    logical = {
        "student/nograd/plain": sum(1 + row.new_blocks * 5 for row in trace.rows),
        "student/grad/checkpoint": trace.cycles * 16 * 3,
        "teacher/nograd/plain": trace.cycles * 16,
        "critic/nograd/plain": trace.cycles * 16,
        "critic/grad/checkpoint": trace.cycles * 80,
    }
    physical = dict(logical)
    if mode == "pipeline":
        from distillschedule.runtime.livewan_step1_overlap import VARIANT
        physical["critic/grad/checkpoint"] = trace.cycles * (48 if pipeline_variant == VARIANT else 40)
    return logical, physical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=("baseline", "pipeline"), required=True)
    parser.add_argument(
        "--pipeline-variant", choices=(BASELINE_VARIANT, PIPELINE_VARIANT, "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3'), required=True
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    allowed_variants = {BASELINE_VARIANT} if args.execution_mode == "baseline" else {PIPELINE_VARIANT, "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3'}
    if args.pipeline_variant not in allowed_variants:
        errors.append("execution mode/physical implementation pair changed")
    if len(args.expected_head) != 40:
        errors.append("expected HEAD is not a full Git SHA")

    trace = FormalTrace.load(
        args.trace, expected_cycles=2, expected_global_batch=16
    )
    expected_logical, expected_physical = _expected_inventory(
        trace, mode=args.execution_mode, pipeline_variant=args.pipeline_variant
    )

    manifests: list[dict[str, str]] = []
    effective_plan_shas: set[str] = set()
    for node in range(2):
        launch_root = args.run_root / "launcher" / f"node_{node}"
        try:
            manifest = _manifest(launch_root / "launch_manifest.txt")
            teardown = _status(launch_root / "teardown_status.txt", owner_run_id=manifest.get("run_id", ""))
            before = _memory_events(launch_root / "cgroup_memory_events_before.txt")
            after = _memory_events(launch_root / "cgroup_memory_events_after.txt")
            effective_plan, effective_plan_sha = _effective_plan(
                args.run_root / f"effective_plan.node{node}.json"
            )
        except (OSError, ValueError) as error:
            errors.append(f"node{node}: launcher evidence missing/invalid: {error}")
            continue
        manifests.append(manifest)
        effective_plan_shas.add(effective_plan_sha)
        if int(dict(effective_plan.get("evidence", {})).get("steps", -1)) != 10:
            errors.append(f"node{node}: effective plan is not a 10-step/two-cycle plan")
        if dict(effective_plan.get("metadata", {})).get("execution_mode") != args.execution_mode:
            errors.append(f"node{node}: effective plan execution mode changed")
        required = {
            "node_rank": str(node),
            "head": args.expected_head,
            "execution_mode": args.execution_mode,
            "pipeline_variant": args.pipeline_variant,
            "run_kind": "two-cycle-timing",
            "two_cycle_count": "2",
            "profiled": "false",
            "profiler_status": "NOT_RUN",
            "claim_boundary": "STEADY_STATE_CYCLE2_SCORE_COMPLETION_ENDPOINT_DELTA",
        }
        for key, expected in required.items():
            if manifest.get(key) != expected:
                errors.append(f"node{node}: launch manifest changed {key}")
        if (
            teardown.get("status") != "PASS"
            or teardown.get("gpu_process_count") != "0"
            or teardown.get("owner_run_id") != manifest.get("run_id")
        ):
            errors.append(f"node{node}: teardown proof is incomplete")
        for key in ("oom", "oom_kill"):
            if int(after.get(key, 0)) != int(before.get(key, 0)):
                errors.append(f"node{node}: cgroup {key} changed")
    run_ids = {row.get("run_id") for row in manifests}
    seeds = {row.get("two_cycle_seed") for row in manifests}
    arm_ids = {row.get("two_cycle_arm_id") for row in manifests}
    gate_shas = {row.get("two_cycle_correctness_sha256") for row in manifests}
    if len(run_ids) != 1 or None in run_ids:
        errors.append("two-node run identity differs")
    if len(seeds) != 1 or next(iter(seeds), "").isdigit() is False:
        errors.append("two-node seed differs or is invalid")
    if len(arm_ids) != 1 or not next(iter(arm_ids), ""):
        errors.append("two-node arm identity differs")
    if len(gate_shas) != 1 or len(str(next(iter(gate_shas), ""))) != 64:
        errors.append("two-node correctness gate SHA differs or is invalid")
    if len(effective_plan_shas) != 1:
        errors.append("two-node effective plans differ")

    rows: list[dict[str, Any]] = []
    endpoint_rows: dict[int, dict[str, float]] = {}
    for rank in range(WORLD_SIZE):
        path = args.run_root / "receipts" / f"rank_{rank:03d}.json"
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"rank{rank}: receipt missing/invalid: {error}")
            continue
        rows.append(row)
        if (
            row.get("status") != "PASS"
            or int(row.get("rank", -1)) != rank
            or row.get("run_kind") != "two-cycle-timing"
            or row.get("mode") != args.execution_mode
            or row.get("profiled") is not False
            or row.get("trace_sha256") != trace.sha256
        ):
            errors.append(f"rank{rank}: rank/run/source identity changed")
        if (
            int(row.get("logical_generator_updates", -1)) != 2
            or int(row.get("logical_critic_updates", -1)) != 10
            or row.get("final_versions") != FINAL_VERSIONS
        ):
            errors.append(f"rank{rank}: logical updates or final versions changed")
        expected_commits = (
            (2, 10)
            if args.execution_mode == "baseline"
            else ((2, 0) if rank in GENERATOR_RANKS else (0, 10))
        )
        if (
            int(row.get("physical_generator_commits", -1)),
            int(row.get("physical_critic_commits", -1)),
        ) != expected_commits:
            errors.append(f"rank{rank}: physical commit inventory changed")
        for key in ("oom", "oom_kill"):
            before = int(dict(row.get("cgroup_oom_before", {})).get(key, 0))
            after = int(dict(row.get("cgroup_oom_after", {})).get(key, 0))
            if after != before:
                errors.append(f"rank{rank}: cgroup {key} changed")

        timing = dict(row.get("two_cycle_timing", {}))
        if args.pipeline_variant in {"c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3'}:
            from distillschedule.runtime.livewan_score_gate import runtime_receipt_errors as score_gate_errors
            errors.extend(f"rank{rank}: {error}" for error in score_gate_errors(
                dict(row.get("score_residual_gate", {})), rank=rank,
                inventory=dict(row.get("physical_call_inventory", {}))))
        if args.pipeline_variant in {"c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3'}:
            from distillschedule.runtime.livewan_score_layernorm import runtime_receipt_errors as score_ln_errors
            errors.extend(f"rank{rank}: {error}" for error in score_ln_errors(
                dict(row.get("score_layernorm", {})), rank=rank,
                inventory=dict(row.get("physical_call_inventory", {}))))
        if args.pipeline_variant == "c37-continuous-b16-step1-overlap":
            from distillschedule.runtime.livewan_step1_overlap import runtime_receipt_errors as step1_errors
            errors.extend(f"rank{rank}: {error}" for error in step1_errors(
                dict(row.get("continuous_pipeline", {})), rank=rank, cycles=2,
                inventory=dict(row.get("physical_call_inventory", {}))))
        if args.pipeline_variant in {"c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3'}:
            from distillschedule.runtime.livewan_score_rmsnorm import runtime_receipt_errors

            errors.extend(f"rank{rank}: {error}" for error in runtime_receipt_errors(
                dict(row.get("score_rmsnorm_kernel", {})), rank=rank))
        if args.pipeline_variant in {"c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3'}:
            from distillschedule.runtime.livewan_generator_rms_epilogue import runtime_receipt_errors as rms_epilogue_errors

            errors.extend(f"rank{rank}: {error}" for error in rms_epilogue_errors(
                dict(row.get("generator_rms_epilogue", {})), rank=rank, cycles=2,
                fuse_cast_square=args.pipeline_variant in {"c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3'}))
        if args.pipeline_variant in {"c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3'}:
            from distillschedule.runtime.livewan_rollout_cast_cache import runtime_receipt_errors as cast_errors

            errors.extend(f"rank{rank}: {error}" for error in cast_errors(
                dict(row.get("generator_rollout_cast_cache", {})), rank=rank, cycles=2))
        if args.pipeline_variant in {"c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3'}:
            from distillschedule.runtime.livewan_generator_modulation import runtime_receipt_errors as modulation_errors

            errors.extend(f"rank{rank}: {error}" for error in modulation_errors(
                dict(row.get("generator_modulation", {})), rank=rank, cycles=2))
        if args.pipeline_variant in {"c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3'}:
            from distillschedule.runtime.livewan_generator_gate import runtime_receipt_errors as gate_errors

            errors.extend(f"rank{rank}: {error}" for error in gate_errors(
                dict(row.get("generator_gate", {})), rank=rank, cycles=2,
                output_layout="row_major" if args.pipeline_variant in {"c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3'} else "native"))
        if args.pipeline_variant == "c27-continuous-b16-pinned-prompt-staging":
            from distillschedule.runtime.livewan_context_staging import runtime_receipt_errors as staging_errors

            errors.extend(f"rank{rank}: {error}" for error in staging_errors(
                dict(row.get("score_prompt_staging", {})), rank=rank, cycles=2))
        if args.pipeline_variant in {"c26-continuous-b16-critic-attention-checkpoint", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3'}:
            if args.pipeline_variant == "c38-continuous-b16-score-fa3":
                from distillschedule.runtime.livewan_score_fa3 import runtime_receipt_errors as sac_errors, backend_receipt_errors
                errors.extend(f"rank{rank}: {error}" for error in backend_receipt_errors(
                    dict(row.get("score_attention_backend", {})), rank=rank))
            else:
                from distillschedule.runtime.livewan_attention_checkpoint import runtime_receipt_errors as sac_errors

            errors.extend(f"rank{rank}: {error}" for error in sac_errors(
                dict(row.get("critic_attention_checkpoint", {})), rank=rank,
                # Use observed physical forwards; the independent inventory and
                # C37 per-step gates below/above still enforce planned totals.
                expected_calls=int(dict(row.get("physical_call_inventory", {}))
                                   .get("critic/grad/checkpoint", -1)) * 30))
        cycle1 = dict(timing.get("cycle_1", {}))
        cycle2 = dict(timing.get("cycle_2", {}))
        if (
            timing.get("schema_version")
            != "distillschedule.livewan_two_cycle_timing_rank.v1"
            or timing.get("cycles_in_same_process") != 2
            or timing.get("profiled") is not False
            or timing.get("profiler_status") != "NOT_RUN"
            or timing.get("state_anchors") != "NOT_COLLECTED_IN_TIMING_MODE"
            or timing.get("evidence_extraction")
            != "AFTER_BOTH_TIMED_CYCLE_ENDPOINTS"
            or cycle1.get("role") != "WARMUP_DIAGNOSTIC"
            or cycle1.get("claimable") is not False
            or cycle2.get("role") != "STEADY_STATE_CLAIMABLE"
            or cycle2.get("claimable") is not True
        ):
            errors.append(f"rank{rank}: two-cycle timing receipt contract changed")
        try:
            c1 = float(cycle1["completion_offset_seconds"])
            c2 = float(cycle2["completion_offset_seconds"])
            local = float(cycle2["local_endpoint_delta_seconds"])
            if not (math.isfinite(c1) and math.isfinite(c2) and math.isfinite(local)):
                raise ValueError("non-finite endpoint")
            if not (0.0 < c1 < c2 and local > 0.0):
                raise ValueError("endpoint ordering")
            if abs((c2 - c1) - local) > 1e-6:
                raise ValueError("local endpoint delta mismatch")
            endpoint_rows[rank] = {"cycle1": c1, "cycle2": c2, "local": local}
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"rank{rank}: invalid timing endpoint: {error}")
        binding = dict(timing.get("correctness_binding", {}))
        if (
            binding.get("sha256") != next(iter(gate_shas), None)
            or binding.get("trajectory_comparison_sha256") is None
            or binding.get("arm_id") != next(iter(arm_ids), None)
            or binding.get("pipeline_variant") != args.pipeline_variant
            or binding.get("status") != "PASS"
        ):
            errors.append(f"rank{rank}: correctness evidence binding changed")
        if row.get("effective_plan_sha256") not in effective_plan_shas:
            errors.append(f"rank{rank}: effective plan SHA changed")

        cycle_contracts = list(timing.get("per_cycle_contract", []))
        if len(cycle_contracts) != 2:
            errors.append(f"rank{rank}: per-cycle contract receipts are incomplete")
        for index, cycle_contract in enumerate(cycle_contracts, start=1):
            expected_local_versions = (
                {"generator": index, "critic": index * 5, "ema": 0}
                if args.execution_mode == "baseline"
                else (
                    {"generator": index, "critic": 0, "ema": 0}
                    if rank in GENERATOR_RANKS
                    else {"generator": 0, "critic": index * 5, "ema": 0}
                )
            )
            if (
                cycle_contract.get("cycle") != index
                or cycle_contract.get("local_versions") != expected_local_versions
            ):
                errors.append(f"rank{rank}/cycle{index}: role-local version contract changed")

        if args.execution_mode == "baseline":
            if row.get("continuous_pipeline"):
                errors.append(f"rank{rank}: sequential arm unexpectedly used a pipeline")
        else:
            continuous = dict(row.get("continuous_pipeline", {}))
            transport = dict(row.get("generator_runahead_transport", {}))
            hot_sync = dict(row.get("critic_shadow_hot_sync", {}))
            if (
                continuous.get("name") != TWO_CYCLE_TIMING_CONTINUOUS_NAME
                or continuous.get("cycles") != 2
                or continuous.get("execution_status") != "PASS"
                or continuous.get("critic_physical_batch_size") != (
                    "mixed_step1_b1_other_b2" if args.pipeline_variant == "c37-continuous-b16-step1-overlap" else 2)
                or continuous.get("critic_physical_backward_calls_per_step") != (
                    [1, 2, 1, 1, 1] if args.pipeline_variant == "c37-continuous-b16-step1-overlap" else 1)
                or continuous.get("critic_step1_single_commit") is not True
                or continuous.get("source_reuse_across_cycles") is not True
            ):
                errors.append(f"rank{rank}: C24 continuous contract changed")
            expected_role = "generator" if rank in GENERATOR_RANKS else "score"
            if (
                transport.get("execution_status") != "PASS"
                or transport.get("role") != expected_role
                or transport.get("cycles") != 2
                or transport.get("cross_cycle_lookahead") is not True
            ):
                errors.append(f"rank{rank}: depth-1 transport receipt changed")
            expected_sync_role = (
                "generator_receiver" if rank in GENERATOR_RANKS else "score_source"
            )
            if (
                hot_sync.get("execution_status") != "PASS"
                or hot_sync.get("role") != expected_sync_role
                or [int(item.get("version", -1)) for item in hot_sync.get("events", [])]
                != [5]
            ):
                errors.append(f"rank{rank}: C5 hot-sync receipt changed")

    if len(rows) != WORLD_SIZE:
        errors.append(f"rank receipt count={len(rows)}, expected=16")
    identities = {
        (
            row.get("trace_sha256"),
            row.get("bootstrap_sha256"),
            row.get("source_receipt_sha256"),
        )
        for row in rows
    }
    if len(identities) != 1 or None in next(iter(identities), (None,)):
        errors.append("trace/bootstrap/source-receipt identity differs")
        identity = None
    else:
        trace_sha256, bootstrap_sha256, source_receipt_sha256 = next(iter(identities))
        identity = {
            "trace_sha256": trace_sha256,
            "bootstrap_sha256": bootstrap_sha256,
            "source_receipt_sha256": source_receipt_sha256,
        }

    logical = _aggregate(rows, "logical_call_inventory")
    physical = _aggregate(rows, "physical_call_inventory")
    if _core(logical) != expected_logical:
        errors.append(f"logical call inventory changed: {_core(logical)}")
    if _core(physical) != expected_physical:
        errors.append(f"physical call inventory changed: {_core(physical)}")
    expected_cycle_logical = {
        key: count // 2 for key, count in expected_logical.items()
    }
    expected_cycle_physical = {
        key: count // 2 for key, count in expected_physical.items()
    }
    per_cycle_inventory: list[dict[str, Any]] = []
    for cycle_index in range(2):
        logical_delta: dict[str, int] = {}
        physical_delta: dict[str, int] = {}
        for row in rows:
            contracts = list(dict(row.get("two_cycle_timing", {})).get("per_cycle_contract", []))
            if len(contracts) != 2:
                continue
            for key, count in dict(
                contracts[cycle_index].get("logical_call_inventory_delta", {})
            ).items():
                logical_delta[key] = logical_delta.get(key, 0) + int(count)
            for key, count in dict(
                contracts[cycle_index].get("physical_call_inventory_delta", {})
            ).items():
                physical_delta[key] = physical_delta.get(key, 0) + int(count)
        if _core(logical_delta) != expected_cycle_logical:
            errors.append(
                f"cycle{cycle_index + 1}: logical call inventory changed: {_core(logical_delta)}"
            )
        if _core(physical_delta) != expected_cycle_physical:
            errors.append(
                f"cycle{cycle_index + 1}: physical call inventory changed: {_core(physical_delta)}"
            )
        per_cycle_inventory.append(
            {
                "cycle": cycle_index + 1,
                "logical": logical_delta,
                "physical": physical_delta,
            }
        )

    for row in rows:
        for name in ("generator_state_metrics", "critic_state_metrics"):
            metrics = row.get(name)
            if metrics is None:
                continue
            for key in ("sum", "sum_sq", "max_abs", "weighted_sum", "weighted_sum_sq"):
                try:
                    finite = math.isfinite(float(metrics[key]))
                except (KeyError, TypeError, ValueError):
                    finite = False
                if not finite:
                    errors.append(f"rank{row.get('rank')}: non-finite {name}/{key}")

    critical_ranks = (
        tuple(range(WORLD_SIZE))
        if args.execution_mode == "baseline"
        else SCORE_RANKS
    )
    critical = [endpoint_rows[rank] for rank in critical_ranks if rank in endpoint_rows]
    if len(critical) != len(critical_ranks):
        errors.append("critical-rank timing endpoints are incomplete")
    c1_endpoint = max((row["cycle1"] for row in critical), default=0.0)
    c2_endpoint = max((row["cycle2"] for row in critical), default=0.0)
    c2_seconds = c2_endpoint - c1_endpoint
    if not (c1_endpoint > 0.0 and c2_seconds > 0.0):
        errors.append("world cycle endpoint delta is not positive")

    contract = {
        "schema": "livewan_two_cycle_timing_contract_v1",
        "head": args.expected_head,
        "identity": identity,
        "seed": int(next(iter(seeds), -1)) if len(seeds) == 1 else None,
        "cycles_in_same_process": 2,
        "world_size": 16,
        "global_batch": 16,
        "source_reuse_across_cycles": True,
        "logical_updates": FINAL_VERSIONS,
        "logical_call_inventory": expected_logical,
        "profiled": False,
        "timer_boundary": (
            "one pre-run barrier+synchronize; no midpoint barrier/synchronize; "
            "one post-cycle2 synchronize+barrier"
        ),
    }
    result = {
        "schema_version": "distillschedule.livewan_two_cycle_timing_summary.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "run_id": next(iter(run_ids), None) if len(run_ids) == 1 else None,
        "run_root": str(args.run_root),
        "head": args.expected_head,
        "heads": {args.execution_mode: args.expected_head},
        "execution_mode": args.execution_mode,
        "pipeline_variant": args.pipeline_variant,
        "arm_id": next(iter(arm_ids), None) if len(arm_ids) == 1 else None,
        "run_kind": "two-cycle-timing",
        "native_unprofiled": True,
        "profiled": False,
        "profiler_status": "NOT_RUN",
        "formal_timing_eligible": not errors,
        "claim_boundary": "CYCLE2_ONLY_NATIVE_UNPROFILED",
        "matched_workload_contract": contract,
        "matched_workload_fingerprint": _sha256_json(contract),
        "implementation": {
            "mode": args.execution_mode,
            "variant": args.pipeline_variant,
            "effective_plan_sha256": (
                next(iter(effective_plan_shas))
                if len(effective_plan_shas) == 1
                else None
            ),
            "physical_call_inventory": expected_physical,
            "fingerprint": _sha256_json(
                {
                    "mode": args.execution_mode,
                    "variant": args.pipeline_variant,
                    "effective_plan_sha256": (
                        next(iter(effective_plan_shas))
                        if len(effective_plan_shas) == 1
                        else None
                    ),
                    "physical_call_inventory": expected_physical,
                }
            ),
        },
        "identity": identity,
        "trace_sha256": trace.sha256,
        "cycles_in_same_process": 2,
        "final_versions": FINAL_VERSIONS,
        "logical_call_inventory": logical,
        "physical_call_inventory": physical,
        "per_cycle_call_inventory": per_cycle_inventory,
        "cycle_1": {
            "role": "WARMUP_DIAGNOSTIC",
            "claimable": False,
            "world_completion_endpoint_seconds": c1_endpoint,
            "critical_rank_local_endpoint_deltas": _distribution(
                [row["cycle1"] for row in critical]
            ),
        },
        "cycle_2": {
            "role": "STEADY_STATE_CLAIMABLE",
            "claimable": not errors,
            "seconds": c2_seconds,
            "endpoint_2_world_seconds": c2_endpoint,
            "endpoint_1_world_seconds": c1_endpoint,
            "denominator": (
                "MAX_SCORE_POOL_COMPLETION_2_MINUS_MAX_SCORE_POOL_COMPLETION_1"
                if args.execution_mode == "pipeline"
                else "MAX_WORLD_COMPLETION_2_MINUS_MAX_WORLD_COMPLETION_1"
            ),
            "critical_rank_local_endpoint_deltas": _distribution(
                [row["local"] for row in critical]
            ),
            "includes_cross_cycle_lookahead_and_hot_sync": (
                args.execution_mode == "pipeline"
            ),
        },
        "correctness_binding": {
            "gate_sha256": next(iter(gate_shas), None),
            "trajectory_comparison_sha256": (
                dict(rows[0].get("two_cycle_timing", {}))
                .get("correctness_binding", {})
                .get("trajectory_comparison_sha256")
                if rows
                else None
            ),
        },
        "outer_two_cycle_wall_diagnostic": _distribution(
            [float(row.get("full5_wall_seconds", 0.0)) for row in rows]
        ),
        "rank_count": len(rows),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
