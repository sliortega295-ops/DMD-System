#!/usr/bin/env python3
"""Fail-closed per-run summarizer for RB16-F5-PIPE-2G6C."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from runtime.routeb_full5 import RouteBCandidateBTopology, RouteBFull5Contract


CONTRACT = RouteBFull5Contract.candidate_b()
TOPOLOGY = RouteBCandidateBTopology(CONTRACT)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _require(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def _resource_peak(path: Path) -> int:
    if not path.is_file():
        return -1
    with path.open(newline="") as handle:
        return max(
            (int(row["cgroup_bytes"]) for row in csv.DictReader(handle)),
            default=-1,
        )


def _launch_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _memory_counter(raw: str, name: str) -> int:
    for item in raw.replace(";", "\n").splitlines():
        parts = item.split()
        if len(parts) == 2 and parts[0] == name:
            return int(parts[1])
    return -1


def _stage_peak(value: dict[str, Any], key: str) -> float:
    return max(
        (float(stage.get("cuda", {}).get(key, 0.0)) for stage in value.get("stages", [])),
        default=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--schedule", choices=("sequential", "pipeline"), required=True)
    parser.add_argument("--mode", choices=("correctness", "timing"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted((args.run_root / "execution" / "ranks").glob("rank_*.json"))
    values = [json.loads(path.read_text()) for path in paths]
    by_rank = {int(value["rank"]): value for value in values}
    errors: list[str] = []
    _require(len(values) == 16, errors, f"rank result count {len(values)} != 16")
    _require(set(by_rank) == set(range(16)), errors, "rank IDs are not exactly 0..15")

    expected_contract = CONTRACT.__dict__
    identity_fields = (
        "execution_trace_bundle_hash",
        "routeb_trace_binding_hash",
        "logical_manifest_hash",
        "transition_package_hash",
        "source_transition_manifest_sha256",
    )
    identities: dict[str, Any] = {}
    for field in identity_fields:
        observed = {value.get("identities", {}).get(field) for value in values}
        _require(len(observed) == 1 and None not in observed, errors, f"{field} differs")
        identities[field] = next(iter(observed)) if observed else None

    timing_bindings = {
        json.dumps(value.get("timing_correctness_binding"), sort_keys=True)
        for value in values
    }
    if args.mode == "timing":
        _require(len(timing_bindings) == 1, errors, "timing correctness binding differs")
        binding = values[0].get("timing_correctness_binding") if values else None
        _require(isinstance(binding, dict) and binding.get("status") == "PASS", errors, "timing correctness binding is not PASS")
    else:
        _require(timing_bindings == {"null"}, errors, "correctness run unexpectedly has timing binding")

    generated: dict[tuple[int, int, int], dict[str, Any]] = {}
    consumed: dict[tuple[int, int, int], dict[str, Any]] = {}
    ema_records: list[dict[str, Any]] = []
    for rank in range(16):
        value = by_rank.get(rank, {})
        role = TOPOLOGY.role(rank)
        _require(value.get("status") == "PASS", errors, f"rank {rank} status={value.get('status')}")
        _require(value.get("schedule") == args.schedule, errors, f"rank {rank} schedule changed")
        _require(value.get("role") == role, errors, f"rank {rank} role changed")
        _require(value.get("contract") == expected_contract, errors, f"rank {rank} contract changed")
        _require(value.get("execution_scope") == "critic_phase_from_post_step10a_target_layout", errors, f"rank {rank} scope changed")
        cycle = value.get("cycle", {})
        if role == "generator":
            init = value.get("generator_initialization", {})
            _require(init.get("parameter_version") == 3, errors, f"rank {rank} is not G3")
            _require(init.get("optimizer_materialized") is False, errors, f"rank {rank} materialized G optimizer")
            _require(cycle.get("generator_version_before") == cycle.get("generator_version_after") == 3, errors, f"rank {rank} G version changed")
            if args.mode == "correctness":
                _require(cycle.get("generator_trainable_sha256_before") == cycle.get("generator_trainable_sha256_after"), errors, f"rank {rank} G trainable changed")
                gradients = cycle.get("generator_gradients", {})
                _require(gradients.get("trainable_grad_present", 0) == gradients.get("frozen_grad_present", 0) == 0, errors, f"rank {rank} retained G gradients")
            queue = cycle.get("queue", {})
            _require(queue.get("produced") == queue.get("sent") == queue.get("acknowledged") == 15, errors, f"rank {rank} producer counts changed")
            _require(queue.get("producer_slot") == "EMPTY" and queue.get("outstanding_transport") == 0, errors, f"rank {rank} producer did not flush")
            _require(queue.get("producer_ready_not_sent_max") == 3, errors, f"rank {rank} producer batch width changed")
            _require(queue.get("producer_global_batches_ahead_max") == (1 if args.schedule == "pipeline" else 0), errors, f"rank {rank} queue-depth contract changed")
            waves = cycle.get("waves", [])
            _require(len(waves) == 15, errors, f"rank {rank} generated {len(waves)} payloads")
            for record in waves:
                update = int(record.get("update", -1))
                production_round = int(record.get("production_round", -1))
                destination = int(record.get("destination_global_rank", -1))
                key = (update, production_round, rank)
                _require(key not in generated, errors, f"duplicate generated slot {key}")
                _require(destination == TOPOLOGY.critic_for(rank, production_round), errors, f"slot {key} destination changed")
                generated[key] = record
            ema_records.extend(cycle.get("ema", []))
        else:
            init = value.get("student_initialization", {})
            resolved = init.get("resolved", {})
            _require(resolved == {"dp_world_size": 12, "train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 1, "train_batch_size": 12}, errors, f"rank {rank} DP12 batch changed: {resolved}")
            runtime = init.get("runtime", {})
            _require(runtime.get("parameter_policy") == "RESIDENT" and runtime.get("frozen_block_streaming") is False, errors, f"rank {rank} Student is not resident")
            _require(cycle.get("critic_version_before") == 10 and cycle.get("critic_version_after") == 15, errors, f"rank {rank} C versions changed")
            _require(cycle.get("optimizer_step_before") == 10 and cycle.get("optimizer_step_after") == 15, errors, f"rank {rank} optimizer versions changed")
            _require(cycle.get("scheduler_step_before") == 10 and cycle.get("scheduler_step_after") == 15, errors, f"rank {rank} scheduler versions changed")
            queue = cycle.get("queue", {})
            _require(queue == {"received": 5, "consumed": 5, "committed": 5, "consumer_slot": "EMPTY", "outstanding_transport": 0}, errors, f"rank {rank} consumer did not flush")
            waves = cycle.get("waves", [])
            _require(len(waves) == 5, errors, f"rank {rank} consumed {len(waves)} payloads")
            producer, production_round = TOPOLOGY.producer_for_critic(rank)
            for record in waves:
                update = int(record.get("update", -1))
                key = (update, production_round, producer)
                _require(key not in consumed, errors, f"duplicate consumed slot {key}")
                _require(record.get("producer_global_rank") == producer, errors, f"slot {key} producer changed")
                _require(record.get("gradient_accumulation_boundary") is True, errors, f"slot {key} was not GAS1 commit")
                _require(record.get("critic_version_before") == 10 + update and record.get("critic_version_after") == 11 + update, errors, f"slot {key} C version changed")
                consumed[key] = record

    _require(len(generated) == len(consumed) == 60, errors, f"payload coverage G={len(generated)} C={len(consumed)}")
    _require(set(generated) == set(consumed), errors, "generated and consumed slot sets differ")
    trace_identities: set[str] = set()
    for key in sorted(set(generated) & set(consumed)):
        left, right = generated[key], consumed[key]
        _require(left.get("trace_key") == right.get("trace_key"), errors, f"slot {key} trace key changed")
        _require(left.get("trace_identity_sha256") == right.get("trace_identity_sha256"), errors, f"slot {key} trace identity changed")
        _require(left.get("x0_identity") == right.get("x0_identity"), errors, f"slot {key} x0 identity changed")
        trace_identities.add(str(left.get("trace_identity_sha256")))
    _require(len(trace_identities) == 60, errors, f"rank-qualified trace identity count {len(trace_identities)} != 60")

    _require(len(ema_records) == 5, errors, f"EMA record count {len(ema_records)} != 5")
    for update, record in enumerate(ema_records):
        _require((record.get("before"), record.get("after"), record.get("decay")) == (10 + update, 11 + update, 0.0), errors, f"EMA update {update} changed")
        if args.mode == "correctness":
            _require(record.get("matches_generator_exact") is True, errors, f"EMA update {update} no longer equals G3")

    node_records = []
    for node in (0, 1):
        launcher = args.run_root / "launcher" / f"node_{node}"
        launch = _launch_values(launcher / "launch_manifest.txt")
        before = launch.get("memory_events_before", "")
        after = launch.get("memory_events_after", "")
        oom_delta = _memory_counter(after, "oom") - _memory_counter(before, "oom")
        kill_delta = _memory_counter(after, "oom_kill") - _memory_counter(before, "oom_kill")
        _require(launch.get("final_return_code") == "0", errors, f"node {node} launcher return changed")
        _require(oom_delta == 0 and kill_delta == 0, errors, f"node {node} new cgroup OOM: {oom_delta}/{kill_delta}")
        node_records.append({
            "node": node,
            "cgroup_peak_bytes": _resource_peak(launcher / "resource.csv"),
            "oom_delta": oom_delta,
            "oom_kill_delta": kill_delta,
        })

    makespan = max((float(value.get("cycle_wall_seconds", 0.0)) for value in values), default=0.0)
    result = {
        "schema_version": "distillschedule.routeb_candidate_b_run_summary.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "run_id": args.run_root.name,
        "schedule": args.schedule,
        "mode": args.mode,
        "experiment_id": CONTRACT.name,
        "claim_scope": "batch12_resource_throughput_pareto_not_candidate_a_speedup",
        "identities": identities,
        "contract": {
            "world_size": 16,
            "generator_ranks": list(CONTRACT.generator_ranks),
            "critic_ranks": list(CONTRACT.critic_ranks),
            "critic_dp": 12,
            "micro_batch_per_rank": 1,
            "gradient_accumulation_steps": 1,
            "global_batch_per_update": 12,
            "logical_updates": 5,
            "logical_samples": 60,
        },
        "coverage": {
            "generated_payloads": len(generated),
            "consumed_payloads": len(consumed),
            "rank_qualified_trace_identities": len(trace_identities),
            "ema_updates": len(ema_records),
        },
        "performance": {
            "makespan_seconds": makespan,
            "samples": 60,
            "samples_per_second": 60.0 / makespan if makespan else 0.0,
            "samples_per_second_per_gpu": 60.0 / (16.0 * makespan) if makespan else 0.0,
            "generator_gpu_max_allocated_gib": max((_stage_peak(value, "max_allocated_gib") for rank, value in by_rank.items() if TOPOLOGY.role(rank) == "generator"), default=0.0),
            "generator_gpu_max_reserved_gib": max((_stage_peak(value, "max_reserved_gib") for rank, value in by_rank.items() if TOPOLOGY.role(rank) == "generator"), default=0.0),
            "critic_gpu_max_allocated_gib": max((_stage_peak(value, "max_allocated_gib") for rank, value in by_rank.items() if TOPOLOGY.role(rank) == "critic"), default=0.0),
            "critic_gpu_max_reserved_gib": max((_stage_peak(value, "max_reserved_gib") for rank, value in by_rank.items() if TOPOLOGY.role(rank) == "critic"), default=0.0),
        },
        "node_resources": node_records,
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
