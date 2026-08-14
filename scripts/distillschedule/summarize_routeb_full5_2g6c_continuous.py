#!/usr/bin/env python3
"""Fail-closed per-run summary for continuous Step10-A -> 2G+6C Full5."""
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
STEP10A_G = tuple(range(8))
STEP10A_S = tuple(range(8, 12))
STEP10A_T = tuple(range(12, 16))
TARGET_G = tuple(CONTRACT.generator_ranks)
TARGET_C = tuple(CONTRACT.critic_ranks)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _require(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


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


def _resource_peak(path: Path) -> int:
    if not path.is_file():
        return -1
    with path.open(newline="") as handle:
        return max(
            (int(row["cgroup_bytes"]) for row in csv.DictReader(handle)),
            default=-1,
        )


def _stage_peak(value: dict[str, Any], key: str) -> float:
    return max(
        (float(stage.get("cuda", {}).get(key, 0.0)) for stage in value.get("stages", [])),
        default=0.0,
    )


def _sample_ids(rank: int) -> list[int]:
    lane = rank - (8 if rank in STEP10A_S else 12)
    return list(range(lane, 16, 4))


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

    identity_fields = (
        "execution_trace_bundle_hash",
        "routeb_trace_binding_hash",
        "logical_manifest_hash",
        "state_provenance_trace_hash",
    )
    identities: dict[str, Any] = {}
    for field in identity_fields:
        observed = {value.get("identities", {}).get(field) for value in values}
        _require(len(observed) == 1 and None not in observed, errors, f"{field} differs")
        identities[field] = next(iter(observed)) if observed else None

    expected_step10_contract = {
        "generator_ranks": list(STEP10A_G),
        "student_ranks": list(STEP10A_S),
        "teacher_ranks": list(STEP10A_T),
        "generator_backend": "ZeRO-3",
        "generator_parameter_offload": False,
        "generator_optimizer_offload": "cpu",
        "generator_checkpoint_policy": (
            "zero3_metadata_tolerant_block_local_non_reentrant"
        ),
        "dp": 8,
        "micro_batch": 1,
        "gradient_accumulation_steps": 2,
        "global_batch": 16,
    }
    expected_target_resolved = {
        "dp_world_size": 12,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "train_batch_size": 12,
    }

    step10_g: dict[int, dict[str, Any]] = {}
    step10_s: dict[int, dict[str, Any]] = {}
    step10_t: dict[int, dict[str, Any]] = {}
    generated: dict[tuple[int, int, int], dict[str, Any]] = {}
    consumed: dict[tuple[int, int, int], dict[str, Any]] = {}
    ema_records: list[dict[str, Any]] = []
    target_generator_shas: set[str] = set()

    for rank in range(16):
        value = by_rank.get(rank, {})
        _require(value.get("status") == "PASS", errors, f"rank {rank} status={value.get('status')}")
        _require(value.get("schema_version") == "distillschedule.rb16_full5_2g6c_continuous_rank.v1", errors, f"rank {rank} schema changed")
        _require(value.get("experiment_id") == "RB16-F5-CONT-8G4S4T-TO-2G6C", errors, f"rank {rank} experiment changed")
        _require(value.get("execution_scope") == "continuous_full5_step10_to_step14", errors, f"rank {rank} scope changed")
        _require(value.get("claim_scope") == "batch12_resource_throughput_pareto_not_matched_batch16_speedup", errors, f"rank {rank} claim scope changed")
        _require(value.get("schedule") == args.schedule, errors, f"rank {rank} schedule changed")
        _require(value.get("step10a_contract") == expected_step10_contract, errors, f"rank {rank} Step10-A contract changed")
        expected_critic_contract = json.loads(json.dumps(CONTRACT.__dict__))
        _require(value.get("critic_contract") == expected_critic_contract, errors, f"rank {rank} target contract changed")
        _require(value.get("transport_policy") == "ACK_FREE_NCCL_SEND_COMPLETION_BACKPRESSURE", errors, f"rank {rank} transport changed")
        _require(value.get("final_versions") == {
            "generator": 3, "critic": 15, "ema": 15,
            "next_boundary": "Step15-A_not_executed",
        }, errors, f"rank {rank} final versions changed")

        binding = value.get("timing_correctness_binding")
        if args.mode == "timing":
            _require(isinstance(binding, dict) and binding.get("status") == "PASS", errors, f"rank {rank} timing binding is not PASS")
        else:
            _require(binding is None, errors, f"rank {rank} correctness run has timing binding")

        step = value.get("step10a", {})
        samples = step.get("samples", [])
        if rank in STEP10A_G:
            init = value.get("step10a_generator_initialization", {})
            _require(init.get("resolved") == {
                "dp_world_size": 8,
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 2,
                "train_batch_size": 16,
            }, errors, f"rank {rank} Step10-A DP8 batch changed")
            _require(init.get("config", {}).get("zero_optimization", {}).get("stage") == 3, errors, f"rank {rank} Step10-A is not ZeRO-3")
            _require(init.get("checkpoint_policy") == "zero3_metadata_tolerant_block_local_non_reentrant", errors, f"rank {rank} Step10-A checkpoint policy changed")
            _require(step.get("role") == "generator", errors, f"rank {rank} Step10-A role changed")
            _require(step.get("schedule") == "two_selected_graphs_first_then_two_backwards", errors, f"rank {rank} Step10-A ordering regressed")
            _require((step.get("generator_before"), step.get("generator_after")) == (2, 3), errors, f"rank {rank} G2->G3 changed")
            _require((step.get("global_steps_before"), step.get("global_steps_after"), step.get("accumulation_boundaries")) == (2, 3, 1), errors, f"rank {rank} Step10-A update count changed")
            _require(len(samples) == 2, errors, f"rank {rank} Step10-A generated {len(samples)} samples")
            expected_ids = [rank, rank + 8]
            _require([record.get("source_id") for record in samples] == expected_ids, errors, f"rank {rank} Step10-A source IDs changed")
            for record in samples:
                source_id = int(record.get("source_id", -1))
                _require(source_id not in step10_g, errors, f"duplicate Step10-A G source {source_id}")
                _require(record.get("gradient_accumulation_boundary") == (source_id >= 8), errors, f"source {source_id} GAS2 boundary changed")
                step10_g[source_id] = record
        elif rank in STEP10A_S:
            _require(step.get("role") == "student" and step.get("parameter_version") == 10, errors, f"rank {rank} Step10-A Student version changed")
            _require([record.get("source_id") for record in samples] == _sample_ids(rank), errors, f"rank {rank} Step10-A Student lane changed")
            for record in samples:
                source_id = int(record.get("source_id", -1))
                _require(source_id not in step10_s, errors, f"duplicate Step10-A S source {source_id}")
                step10_s[source_id] = record
        else:
            _require(step.get("role") == "teacher" and step.get("parameter_version") == "official_frozen_base", errors, f"rank {rank} Step10-A Teacher version changed")
            _require([record.get("source_id") for record in samples] == _sample_ids(rank), errors, f"rank {rank} Step10-A Teacher lane changed")
            for record in samples:
                source_id = int(record.get("source_id", -1))
                _require(source_id not in step10_t, errors, f"duplicate Step10-A T source {source_id}")
                step10_t[source_id] = record

        target_role = TOPOLOGY.role(rank)
        _require(value.get("target_role") == target_role, errors, f"rank {rank} target role changed")
        transition = value.get("transition", {})
        phase = value.get("critic_phase", {})
        if target_role == "generator":
            _require(transition.get("model_reused") is False, errors, f"rank {rank} illegally reused a ZeRO-3 module")
            _require(transition.get("step10a_backend") == "ZeRO-3_DP8", errors, f"rank {rank} Step10-A backend changed")
            redistribution = transition.get("generator_g3_redistribution", {})
            _require(redistribution.get("parameter_version") == 3, errors, f"rank {rank} G3 redistribution version changed")
            _require(redistribution.get("target_generator_ranks") == list(TARGET_G), errors, f"rank {rank} G target ranks changed")
            if args.mode == "correctness":
                sha = redistribution.get("trainable_sha256_after")
                _require(isinstance(sha, str) and len(sha) == 64, errors, f"rank {rank} G3 SHA missing")
                if isinstance(sha, str):
                    target_generator_shas.add(sha)
                _require(phase.get("generator_trainable_sha256_before") == phase.get("generator_trainable_sha256_after") == sha, errors, f"rank {rank} Generator changed in Critic phase")
                gradients = phase.get("generator_gradients", {})
                _require(gradients.get("trainable_grad_present", 0) == gradients.get("frozen_grad_present", 0) == 0, errors, f"rank {rank} retained Generator gradients")
            queue = phase.get("queue", {})
            _require(queue.get("produced_global_batches") == queue.get("committed_global_batches") == 5, errors, f"rank {rank} global batch queue count changed")
            _require(queue.get("produced_payloads") == queue.get("sent_payloads") == 15, errors, f"rank {rank} payload count changed")
            _require(queue.get("per_route_commit_ack") is False, errors, f"rank {rank} reintroduced ACK")
            _require(queue.get("producer_ready_not_sent_max") <= 1, errors, f"rank {rank} ready queue exceeded one")
            expected_ahead = 1 if args.schedule == "sequential" else 2
            _require(queue.get("produced_minus_committed_max") == expected_ahead, errors, f"rank {rank} queue depth changed")
            _require(queue.get("producer_slot") == "EMPTY" and queue.get("outstanding_transport") == 0, errors, f"rank {rank} producer did not flush")
            waves = phase.get("waves", [])
            _require(len(waves) == 15, errors, f"rank {rank} generated {len(waves)} target payloads")
            for record in waves:
                update = int(record.get("update", -1))
                production_round = int(record.get("production_round", -1))
                destination = int(record.get("destination_global_rank", -1))
                key = (update, production_round, rank)
                _require(key not in generated, errors, f"duplicate generated slot {key}")
                _require(destination == TOPOLOGY.critic_for(rank, production_round), errors, f"slot {key} destination changed")
                generated[key] = record
            ema_records.extend(phase.get("ema", []))
        else:
            init = value.get("student_initialization", {})
            _require(init.get("resolved") == expected_target_resolved, errors, f"rank {rank} DP12 batch changed")
            runtime = init.get("runtime", {})
            _require(runtime.get("parameter_policy") == "RESIDENT" and runtime.get("frozen_block_streaming") is False, errors, f"rank {rank} Student is not resident")
            _require((phase.get("critic_version_before"), phase.get("critic_version_after")) == (10, 15), errors, f"rank {rank} C versions changed")
            _require((phase.get("optimizer_step_before"), phase.get("optimizer_step_after")) == (10, 15), errors, f"rank {rank} optimizer versions changed")
            _require((phase.get("scheduler_step_before"), phase.get("scheduler_step_after")) == (10, 15), errors, f"rank {rank} scheduler versions changed")
            queue = phase.get("queue", {})
            _require(queue == {
                "received": 5, "consumed": 5, "committed": 5,
                "per_route_commit_ack": False,
                "consumer_slot": "EMPTY", "outstanding_transport": 0,
            }, errors, f"rank {rank} consumer did not flush")
            waves = phase.get("waves", [])
            _require(len(waves) == 5, errors, f"rank {rank} consumed {len(waves)} target payloads")
            producer, production_round = TOPOLOGY.producer_for_critic(rank)
            for record in waves:
                update = int(record.get("update", -1))
                key = (update, production_round, producer)
                _require(key not in consumed, errors, f"duplicate consumed slot {key}")
                _require(record.get("producer_global_rank") == producer, errors, f"slot {key} producer changed")
                _require(record.get("gradient_accumulation_boundary") is True, errors, f"slot {key} was not GAS1")
                _require(record.get("gradient_clipping", {}).get("configured_max_norm") == 1.0, errors, f"slot {key} clipping changed")
                _require((record.get("critic_version_before"), record.get("critic_version_after")) == (10 + update, 11 + update), errors, f"slot {key} C version changed")
                consumed[key] = record

    _require(set(step10_g) == set(step10_s) == set(step10_t) == set(range(16)), errors, "Step10-A source coverage changed")
    for source_id in sorted(set(step10_g) & set(step10_s) & set(step10_t)):
        g, s, t = step10_g[source_id], step10_s[source_id], step10_t[source_id]
        _require(g.get("x0_sha256") == s.get("x0_sha256") == t.get("x0_sha256"), errors, f"Step10-A source {source_id} x0 changed")
        _require(s.get("fake_sha256") == t.get("fake_sha256"), errors, f"Step10-A source {source_id} fake changed")
        _require(g.get("gradient_sha256") == t.get("gradient_sha256"), errors, f"Step10-A source {source_id} gradient changed")

    if args.mode == "correctness":
        _require(len(target_generator_shas) == 1, errors, f"G3 redistribution SHAs differ: {len(target_generator_shas)}")
    _require(len(generated) == len(consumed) == 60, errors, f"target payload coverage G={len(generated)} C={len(consumed)}")
    _require(set(generated) == set(consumed), errors, "target generated/consumed slot sets differ")
    trace_identities: set[str] = set()
    for key in sorted(set(generated) & set(consumed)):
        left, right = generated[key], consumed[key]
        _require(left.get("trace_key") == right.get("trace_key"), errors, f"slot {key} trace key changed")
        _require(left.get("trace_identity_sha256") == right.get("trace_identity_sha256"), errors, f"slot {key} trace identity changed")
        _require(left.get("x0_identity") == right.get("x0_identity"), errors, f"slot {key} x0 identity changed")
        trace_identities.add(str(left.get("trace_identity_sha256")))
    _require(len(trace_identities) == 60, errors, f"target trace identity count {len(trace_identities)} != 60")

    _require(len(ema_records) == 5, errors, f"EMA record count {len(ema_records)} != 5")
    for update, record in enumerate(ema_records):
        _require((record.get("before"), record.get("after"), record.get("decay")) == (10 + update, 11 + update, 0.0), errors, f"EMA update {update} changed")
        if args.mode == "correctness":
            _require(record.get("matches_generator_exact") is True, errors, f"EMA update {update} no longer equals G3")

    node_records = []
    for node in (0, 1):
        launcher = args.run_root / "launcher" / f"node_{node}"
        launch = _launch_values(launcher / "launch_manifest.txt")
        before, after = launch.get("memory_events_before", ""), launch.get("memory_events_after", "")
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

    full5 = max((float(value.get("full5_execution_seconds", 0.0)) for value in values), default=0.0)
    step10a = max((float(value.get("step10a_wall_seconds", 0.0)) for value in values), default=0.0)
    transition = max((float(value.get("transition_wall_seconds", 0.0)) for value in values), default=0.0)
    critic = max((float(value.get("critic_phase_wall_seconds", 0.0)) for value in values), default=0.0)
    cold = max((float(value.get("duration_seconds", 0.0)) - float(value.get("full5_execution_seconds", 0.0)) for value in values), default=0.0)
    result = {
        "schema_version": "distillschedule.rb16_full5_2g6c_continuous_summary.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "run_id": args.run_root.name,
        "schedule": args.schedule,
        "mode": args.mode,
        "experiment_id": "RB16-F5-CONT-8G4S4T-TO-2G6C",
        "claim_scope": "batch12_resource_throughput_pareto_not_matched_batch16_speedup",
        "identities": identities,
        "contract": {
            "step10a": expected_step10_contract,
            "target_generator_ranks": list(TARGET_G),
            "target_critic_ranks": list(TARGET_C),
            "target_critic_dp": 12,
            "target_micro_batch": 1,
            "target_gas": 1,
            "target_global_batch": 12,
            "critic_updates": 5,
            "ema_updates": 5,
            "critic_samples": 60,
        },
        "coverage": {
            "step10a_samples": len(step10_g),
            "target_generated_payloads": len(generated),
            "target_consumed_payloads": len(consumed),
            "target_trace_identities": len(trace_identities),
            "ema_updates": len(ema_records),
        },
        "performance": {
            "full5_execution_seconds": full5,
            "step10a_wall_seconds": step10a,
            "transition_wall_seconds": transition,
            "critic_phase_wall_seconds": critic,
            "excluded_cold_preparation_seconds": cold,
            "critic_samples_per_second": 60.0 / full5 if full5 else 0.0,
            "critic_samples_per_second_per_gpu": 60.0 / (16.0 * full5) if full5 else 0.0,
            "target_generator_gpu_max_allocated_gib": max((_stage_peak(by_rank[rank], "max_allocated_gib") for rank in TARGET_G if rank in by_rank), default=0.0),
            "target_generator_gpu_max_reserved_gib": max((_stage_peak(by_rank[rank], "max_reserved_gib") for rank in TARGET_G if rank in by_rank), default=0.0),
            "target_critic_gpu_max_allocated_gib": max((_stage_peak(by_rank[rank], "max_allocated_gib") for rank in TARGET_C if rank in by_rank), default=0.0),
            "target_critic_gpu_max_reserved_gib": max((_stage_peak(by_rank[rank], "max_reserved_gib") for rank in TARGET_C if rank in by_rank), default=0.0),
        },
        "node_resources": node_records,
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
