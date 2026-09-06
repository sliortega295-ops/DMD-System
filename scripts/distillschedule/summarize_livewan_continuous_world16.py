#!/usr/bin/env python3
"""Fail-closed evidence summary for the C21 continuous B16 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from distillschedule.runtime.livewan_crossscale_worker import FormalTrace, atomic_json


WORLD_SIZE = 16
GENERATOR_RANKS = tuple(range(8))
SCORE_RANKS = tuple(range(8, 16))
EXPECTED_LOGICAL = {
    "student/nograd/plain": 6180,
    "student/grad/checkpoint": 144,
    "critic/nograd/plain": 48,
    "critic/grad/checkpoint": 240,
    "teacher/nograd/plain": 48,
}
C21_C0_NAME = "continuous_full5_b16_depth1_v1"
C21_C1_NAME = "continuous_full5_b16_depth1_step1_b1_overlap_v1"
C23_B32_NAME = "continuous_full5_b32_depth1_systems_v1"
EXPECTED_LOGICAL_B32 = {
    key: value * 2 for key, value in EXPECTED_LOGICAL.items()
}
EXPECTED_PHYSICAL_BY_CONTRACT = {
    C21_C0_NAME: dict(EXPECTED_LOGICAL, **{"critic/grad/checkpoint": 120}),
    C21_C1_NAME: dict(EXPECTED_LOGICAL, **{"critic/grad/checkpoint": 144}),
    C23_B32_NAME: dict(
        EXPECTED_LOGICAL_B32, **{"critic/grad/checkpoint": 240}
    ),
}


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        for name, count in dict(row.get(key, {})).items():
            result[name] = result.get(name, 0) + int(count)
    return result


def _core_inventory(value: dict[str, int]) -> dict[str, int]:
    return {key: int(value.get(key, 0)) for key in EXPECTED_LOGICAL}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=("pipeline",), required=True)
    parser.add_argument(
        "--run-kind", choices=("validate", "capacity", "correctness", "timing", "profile"),
        required=True,
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-two-cycle", action="store_true")
    args = parser.parse_args()
    if args.diagnostic_two_cycle:
        if args.run_kind != "profile":
            raise SystemExit("two-cycle diagnostic cannot produce a native timing summary")
        from distillschedule.runtime.livewan_two_cycle_profile import summarize
        result = summarize(run_root=args.run_root, trace_path=args.trace, expected_head=args.expected_head)
        atomic_json(args.output, result)
        return 0 if result['status'] == 'PASS' else 1
    errors: list[str] = []
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    contract = dict(plan.get("metadata", {}).get("continuous_pipeline", {}))
    checkpoint_rng_contract = dict(
        plan.get("metadata", {}).get("critic_checkpoint_rng", {})
    )
    checkpoint_rng_enabled = checkpoint_rng_contract.get("enabled") is True
    contract_name = str(contract.get("name", ""))
    global_batch = int(plan["workload"]["shape"]["batch_size"])
    trace = FormalTrace.load(
        args.trace,
        expected_cycles=3,
        expected_global_batch=global_batch,
    )
    if contract_name not in EXPECTED_PHYSICAL_BY_CONTRACT:
        errors.append("resolved plan lacks the registered C21 contract")
    expected_physical = EXPECTED_PHYSICAL_BY_CONTRACT.get(
        contract_name, EXPECTED_PHYSICAL_BY_CONTRACT[C21_C0_NAME]
    )
    expected_logical = (
        EXPECTED_LOGICAL_B32 if contract_name == C23_B32_NAME else EXPECTED_LOGICAL
    )
    expected_transport_buffers = 50 if contract_name == C23_B32_NAME else 26
    expected_transport_waves = 4 if contract_name == C23_B32_NAME else 2
    expected_rollout_tensors = 60 if contract_name == C23_B32_NAME else 30

    if args.run_kind == "validate":
        values = []
        for node in (0, 1):
            path = args.run_root / f"validate_only.node{node}.json"
            if not path.is_file():
                errors.append(f"node{node}: validate receipt missing")
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            values.append(row)
            if row.get("status") != "PASS" or int(row.get("node_rank", -1)) != node:
                errors.append(f"node{node}: validate receipt failed")
            if row.get("trace_sha256") != trace.sha256:
                errors.append(f"node{node}: trace identity mismatch")
        result = {
            "status": "PASS" if not errors else "FAIL",
            "run_kind": args.run_kind,
            "rank_count": 0,
            "node_count": len(values),
            "gpu_workload": "NOT_RUN",
            "trace_sha256": trace.sha256,
            "head": args.expected_head,
            "heads": {"pipeline": args.expected_head},
            "continuous_contract_name": contract_name,
            "errors": errors,
        }
        atomic_json(args.output, result)
        return 0 if not errors else 1

    rows: list[dict[str, Any]] = []
    for rank in range(WORLD_SIZE):
        path = args.run_root / "receipts" / f"rank_{rank:03d}.json"
        if not path.is_file():
            errors.append(f"rank{rank}: receipt missing")
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        rows.append(row)
        if row.get("status") != "PASS" or int(row.get("rank", -1)) != rank:
            errors.append(f"rank{rank}: receipt identity failed")
        if row.get("run_kind") != args.run_kind or row.get("mode") != "pipeline":
            errors.append(f"rank{rank}: run kind/mode mismatch")
        if row.get("trace_sha256") != trace.sha256:
            errors.append(f"rank{rank}: trace SHA mismatch")
        if row.get("logical_generator_updates") != 3 or row.get("logical_critic_updates") != 15:
            errors.append(f"rank{rank}: logical commit inventory changed")
        expected_commits = (3, 0) if rank in GENERATOR_RANKS else (0, 15)
        observed_commits = (
            int(row.get("physical_generator_commits", -1)),
            int(row.get("physical_critic_commits", -1)),
        )
        if observed_commits != expected_commits:
            errors.append(f"rank{rank}: physical commits={observed_commits}")
        if row.get("final_versions") != {"generator": 3, "critic": 15, "ema": 0}:
            errors.append(f"rank{rank}: final versions changed")
        for oom_key in ("oom", "oom_kill"):
            before = int(row.get("cgroup_oom_before", {}).get(oom_key, 0))
            after = int(row.get("cgroup_oom_after", {}).get(oom_key, 0))
            if before != after:
                errors.append(f"rank{rank}: cgroup {oom_key} changed")
        continuous = dict(row.get("continuous_pipeline", {}))
        if (
            continuous.get("execution_status") != "PASS"
            or continuous.get("name") != contract_name
            or continuous.get("cycles") != 3
            or len(continuous.get("events", ())) != 3
        ):
            errors.append(f"rank{rank}: continuous execution receipt incomplete")
        expected_step1_schedule = (
            "two_b2_backward_accumulations_single_commit"
            if contract_name == C23_B32_NAME
            else (
                "wave0_b1_backward_then_wave1_b1_backward_single_commit"
                if contract_name == C21_C1_NAME
                else "single_b2_backward"
            )
        )
        if (
            continuous.get("critic_step1_physical_schedule")
            != expected_step1_schedule
            or continuous.get("critic_step1_single_commit") is not True
        ):
            errors.append(f"rank{rank}: step1 physical schedule changed")
        transport = dict(row.get("generator_runahead_transport", {}))
        expected_transport_role = "generator" if rank in GENERATOR_RANKS else "score"
        if (
            transport.get("execution_status") != "PASS"
            or transport.get("role") != expected_transport_role
            or transport.get("cycles") != 3
            or transport.get("buffer_count") != expected_transport_buffers
            or transport.get("waves_per_step") != expected_transport_waves
            or transport.get("cross_cycle_lookahead") is not True
        ):
            errors.append(f"rank{rank}: continuous host transport incomplete")
        sync = dict(row.get("critic_shadow_hot_sync", {}))
        events = list(sync.get("events", ()))
        expected_sync_role = "generator_receiver" if rank in GENERATOR_RANKS else "score_source"
        if (
            sync.get("execution_status") != "PASS"
            or sync.get("role") != expected_sync_role
            or [int(event.get("version", -1)) for event in events] != [5, 10]
        ):
            errors.append(f"rank{rank}: Critic-shadow hot sync incomplete")
        if args.run_kind == "correctness":
            if rank in GENERATOR_RANKS:
                artifact = row.get("output_tensor_artifact") or {}
                if int(artifact.get("tensor_count", -1)) != expected_rollout_tensors:
                    errors.append(f"rank{rank}: rollout artifact count changed")
            elif row.get("output_tensor_artifact") is not None:
                errors.append(f"rank{rank}: Score unexpectedly saved rollout tensors")
        checkpoint_rng = dict(row.get("critic_checkpoint_rng", {}))
        if checkpoint_rng_enabled:
            if rank in GENERATOR_RANKS:
                if (
                    checkpoint_rng.get("execution_status") != "NOT_APPLICABLE"
                    or int(checkpoint_rng.get("checkpoint_calls", -1)) != 0
                ):
                    errors.append(f"rank{rank}: Generator checkpoint RNG scope changed")
            else:
                expected_checkpoint_calls = int(
                    row.get("physical_call_inventory", {}).get(
                        "critic/grad/checkpoint", 0
                    )
                ) * 30
                if (
                    checkpoint_rng.get("execution_status") != "PASS"
                    or checkpoint_rng.get("preserve_rng_state") is not False
                    or int(checkpoint_rng.get("checkpoint_calls", -1))
                    != expected_checkpoint_calls
                    or int(checkpoint_rng.get("expected_checkpoint_calls", -1))
                    != expected_checkpoint_calls
                ):
                    errors.append(f"rank{rank}: Critic checkpoint RNG receipt changed")
        elif checkpoint_rng.get("execution_status") != "DISABLED":
            errors.append(f"rank{rank}: unregistered checkpoint RNG policy active")

    if len(rows) != WORLD_SIZE:
        errors.append(f"rank receipt count={len(rows)}, expected=16")
    identities = {
        (row.get("trace_sha256"), row.get("bootstrap_sha256"), row.get("source_receipt_sha256"))
        for row in rows
    }
    if len(identities) != 1:
        errors.append("rank source/bootstrap/trace identity differs")

    logical = _aggregate(rows, "logical_call_inventory")
    physical = _aggregate(rows, "physical_call_inventory")
    if _core_inventory(logical) != expected_logical:
        errors.append(f"logical call inventory changed: {_core_inventory(logical)}")
    if _core_inventory(physical) != expected_physical:
        errors.append(f"physical call inventory changed: {_core_inventory(physical)}")

    for lane in range(8):
        generator = rows[lane] if len(rows) == 16 else None
        score = rows[lane + 8] if len(rows) == 16 else None
        if generator is None or score is None:
            continue
        generator_sync = dict(generator.get("critic_shadow_hot_sync", {}))
        score_sync = dict(score.get("critic_shadow_hot_sync", {}))
        if generator_sync.get("layout_sha256") != score_sync.get("layout_sha256"):
            errors.append(f"lane{lane}: Critic-shadow parameter layout differs")
        if args.run_kind == "correctness":
            receiver_events = list(generator_sync.get("events", ()))
            source_events = list(score_sync.get("events", ()))
            for source, receiver in zip(source_events, receiver_events):
                if not source.get("source_sha256") or (
                    source.get("source_sha256") != receiver.get("loaded_sha256")
                ):
                    errors.append(
                        f"lane{lane}: Critic-shadow SHA differs at C{source.get('version')}"
                    )

    rank_walls = [float(row.get("full5_wall_seconds", 0.0)) for row in rows]
    role_offsets: dict[str, list[list[float]]] = {
        "generator": [[], [], []], "score": [[], [], []]
    }
    for row in rows:
        for event in row.get("continuous_pipeline", {}).get("events", []):
            role = str(event.get("role"))
            cycle = int(event.get("cycle", -1))
            if role in role_offsets and 0 <= cycle < 3:
                role_offsets[role][cycle].append(
                    float(event.get("role_return_offset_seconds", 0.0))
                )
    role_cycle_ends = {
        role: [max(values) if values else None for values in cycles]
        for role, cycles in role_offsets.items()
    }
    score_ends = role_cycle_ends["score"]
    score_intervals = [
        score_ends[index] - score_ends[index - 1]
        for index in (1, 2)
        if score_ends[index] is not None and score_ends[index - 1] is not None
    ]
    receive_waits = [
        float(value.get("wall_seconds", 0.0))
        for row in rows if int(row.get("rank", -1)) in SCORE_RANKS
        for value in row.get("score_receive_waits", [])
    ]
    generator_backpressure = [
        float(value.get("wall_seconds", 0.0))
        for row in rows if int(row.get("rank", -1)) in GENERATOR_RANKS
        for value in row.get("generator_backpressure", [])
    ]
    peak_allocated = [int(row.get("peak_memory", {}).get("max_allocated_bytes", 0)) for row in rows]
    min_free = [int(row.get("peak_memory", {}).get("driver_free_bytes", 0)) for row in rows]
    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "run_kind": args.run_kind,
        "execution_mode": "pipeline",
        "diagnostic_only": True,
        "formal_speedup_claimable": False,
        "claim_boundary": (
            "Continuous B32 source-reuse-within-step throughput diagnostic; not a "
            "training-quality or convergence claim."
            if contract_name == C23_B32_NAME
            else "Continuous source-reuse steady-state systems diagnostic; not a "
            "training-quality or sequential-equivalence claim."
        ),
        "head": args.expected_head,
        "heads": {"pipeline": args.expected_head},
        "continuous_contract_name": contract_name,
        "critic_checkpoint_rng": {
            "enabled": checkpoint_rng_enabled,
            "name": checkpoint_rng_contract.get("name"),
            "score_rank_checkpoint_calls": {
                str(row["rank"]): int(
                    row.get("critic_checkpoint_rng", {}).get(
                        "checkpoint_calls", 0
                    )
                )
                for row in rows
                if int(row["rank"]) in SCORE_RANKS
            },
        },
        "rank_count": len(rows),
        "trace_sha256": trace.sha256,
        "trace_file_sha256": _sha256(args.trace),
        "identity": next(iter(identities)) if len(identities) == 1 else None,
        "cycles": 3,
        "global_batch": global_batch,
        "source_reuse_within_step": contract_name == C23_B32_NAME,
        "training_quality_claimable": False,
        "logical_call_inventory": logical,
        "physical_call_inventory": physical,
        "expected_logical_call_inventory": expected_logical,
        "expected_physical_call_inventory": expected_physical,
        "final_versions": {"generator": 3, "critic": 15, "ema": 0},
        "world_wall_seconds": max(rank_walls) if rank_walls else None,
        "world_wall_per_cycle_seconds": max(rank_walls) / 3 if rank_walls else None,
        "role_cycle_end_offsets_seconds": role_cycle_ends,
        "score_steady_intervals_seconds": score_intervals,
        "score_steady_interval_median_seconds": (
            statistics.median(score_intervals) if score_intervals else None
        ),
        "logical_samples_per_cycle": global_batch * 5,
        "steady_samples_per_second": (
            (global_batch * 5) / statistics.median(score_intervals)
            if score_intervals
            else None
        ),
        "b16_equivalent_steady_interval_seconds": (
            statistics.median(score_intervals) * 16 / global_batch
            if score_intervals
            else None
        ),
        "score_receive_wait_sum_seconds_per_rank": {
            str(row["rank"]): sum(
                float(value.get("wall_seconds", 0.0))
                for value in row.get("score_receive_waits", [])
            )
            for row in rows if int(row["rank"]) in SCORE_RANKS
        },
        "score_receive_wait_median_seconds": (
            statistics.median(receive_waits) if receive_waits else None
        ),
        "generator_backpressure_sum_seconds_per_rank": {
            str(row["rank"]): sum(
                float(value.get("wall_seconds", 0.0))
                for value in row.get("generator_backpressure", [])
            )
            for row in rows if int(row["rank"]) in GENERATOR_RANKS
        },
        "generator_backpressure_median_seconds": (
            statistics.median(generator_backpressure) if generator_backpressure else None
        ),
        "peak_allocated_gib": max(peak_allocated) / 2**30 if peak_allocated else None,
        "minimum_driver_free_gib": min(min_free) / 2**30 if min_free else None,
        "claimable_for_timing": args.run_kind == "correctness" and not errors,
    }
    atomic_json(args.output, result)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
