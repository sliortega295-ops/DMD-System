#!/usr/bin/env python3
"""Fail-closed evidence summarizer for RB16-F5-PIPE-MATCH16."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


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


def _checkpoint_inventory(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()]
    generator = root / "pytorch_model"
    critic = root / "critic" / "pytorch_model"
    expected = {
        f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        for rank in range(8)
    }
    for component, path in (("generator", generator), ("critic", critic)):
        observed = {item.name for item in path.glob("*optim_states.pt")}
        if observed != expected:
            raise ValueError(f"checkpoint-15 {component} optimizer shards changed")
        if not (path / "mp_rank_00_model_states.pt").is_file():
            raise ValueError(f"checkpoint-15 {component} model state missing")
    rng = sorted(root.glob("random_states_*.pkl"))
    if len(rng) != 16:
        raise ValueError("checkpoint-15 RNG rank inventory changed")
    if not (root / "model_ema" / "sparse_ema_manifest.json").is_file():
        raise ValueError("checkpoint-15 sparse EMA missing")
    return {
        "path": str(root.resolve()),
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "generator_optimizer_shards": 8,
        "critic_optimizer_shards": 8,
        "rng_files": 16,
        "sparse_ema": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--schedule", choices=("sequential", "pipeline"), required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "transition",
            "transition-save",
            "correctness",
            "critic-correctness",
            "critic-timing",
            "timing",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rank_paths = sorted((args.run_root / "execution" / "ranks").glob("rank_*.json"))
    ranks = [json.loads(path.read_text()) for path in rank_paths]
    critic_phase_only = args.mode in ("critic-correctness", "critic-timing")
    critic_timing = args.mode == "critic-timing"
    errors: list[str] = []
    _require(len(ranks) == 16, errors, f"rank result count {len(ranks)} != 16")
    by_rank = {int(value["rank"]): value for value in ranks}
    _require(set(by_rank) == set(range(16)), errors, "rank IDs are not exactly 0..15")
    for rank, value in sorted(by_rank.items()):
        _require(value.get("status") == "PASS", errors, f"rank {rank} status={value.get('status')}")
        _require(value.get("schedule") == args.schedule, errors, f"rank {rank} schedule changed")
        expected_scope = (
            "critic_phase_from_common_transition_checkpoint"
            if critic_phase_only
            else "continuous_full5_step10_to_step14"
        )
        _require(
            value.get("execution_scope") == expected_scope,
            errors,
            f"rank {rank} execution scope changed",
        )
        contract = value.get("contract", {})
        _require(contract.get("name") == "RB16-F5-PIPE-MATCH16", errors, f"rank {rank} contract changed")
        _require(contract.get("gradient_accumulation_steps") == 2, errors, f"rank {rank} GAS changed")
        _require(contract.get("gradient_clipping") == 1.0, errors, f"rank {rank} clipping changed")

    identity_fields = (
        "logical_manifest_hash",
        "state_provenance_trace_hash",
        "execution_trace_bundle_hash",
        "routeb_trace_binding_hash",
    )
    if critic_phase_only:
        identity_fields += ("transition_manifest_sha256",)
    identities: dict[str, Any] = {}
    for field in identity_fields:
        values = {value.get("identities", {}).get(field) for value in ranks}
        _require(len(values) == 1 and None not in values, errors, f"{field} differs across ranks")
        identities[field] = next(iter(values)) if values else None

    timing_correctness_binding = None
    if critic_timing:
        bindings = [value.get("timing_correctness_binding") for value in ranks]
        serialized = {json.dumps(value, sort_keys=True) for value in bindings}
        _require(
            len(serialized) == 1 and bindings[0] is not None,
            errors,
            "timing correctness binding differs across ranks",
        )
        timing_correctness_binding = bindings[0]
        _require(
            timing_correctness_binding.get("status") == "PASS",
            errors,
            "timing correctness binding is not PASS",
        )
        _require(
            timing_correctness_binding.get("transition_manifest_sha256")
            == identities.get("transition_manifest_sha256"),
            errors,
            "timing correctness transition identity changed",
        )

    generator_samples: dict[int, dict[str, Any]] = {}
    student_samples: dict[int, dict[str, Any]] = {}
    teacher_samples: dict[int, dict[str, Any]] = {}
    for rank in range(16):
        record = by_rank.get(rank, {})
        if critic_phase_only:
            transition = record.get("transition", {})
            restored = record.get("transition_resume", {})
            expected_component = "generator" if rank < 8 else "critic"
            expected_versions = [3, 3, 3] if rank < 8 else [10, 10, 10]
            _require(
                restored.get("component") == expected_component,
                errors,
                f"rank {rank} restored component changed",
            )
            _require(
                restored.get("versions") == expected_versions,
                errors,
                f"rank {rank} restored versions changed",
            )
            _require(
                restored.get("transition_manifest_sha256")
                == identities.get("transition_manifest_sha256"),
                errors,
                f"rank {rank} transition manifest changed",
            )
            _require(
                isinstance(restored.get("trainable_sha256"), str),
                errors,
                f"rank {rank} restored trainable SHA missing",
            )
            if rank < 8:
                _require(
                    transition.get("target_role")
                    == "generator_critic_phase_resident",
                    errors,
                    f"rank {rank} G restore target changed",
                )
                _require(
                    transition.get("parameter_policy") == "RESIDENT",
                    errors,
                    f"rank {rank} G restore is not resident",
                )
            else:
                _require(
                    transition.get("target_role") == "critic",
                    errors,
                    f"rank {rank} C transition missing",
                )
                _require(
                    transition.get("parameter_policy") == "RESIDENT",
                    errors,
                    f"rank {rank} Critic is not resident",
                )
            continue
        step10a = record.get("step10a", {})
        if rank < 8:
            _require(step10a.get("generator_before") == 2, errors, f"rank {rank} lacks G2")
            _require(step10a.get("generator_after") == 3, errors, f"rank {rank} lacks G3")
            _require(step10a.get("accumulation_boundaries") == 1, errors, f"rank {rank} G commit count changed")
            for sample in step10a.get("samples", []):
                source = int(sample["source_id"])
                _require(source not in generator_samples, errors, f"duplicate Generator source {source}")
                generator_samples[source] = sample
            transition = record.get("transition", {})
            _require(transition.get("target_role") == "generator_critic_phase_resident", errors, f"rank {rank} G transition changed")
            promotion = transition.get("promotion", {})
            _require(promotion.get("streamed_block_count") == 4, errors, f"rank {rank} did not promote g4")
            _require(int(promotion.get("promoted_bytes", 0)) > 0, errors, f"rank {rank} promoted zero bytes")
        else:
            role = step10a.get("role")
            target = student_samples if role == "student" else teacher_samples
            _require(role in ("student", "teacher"), errors, f"rank {rank} Step10-A role changed")
            for sample in step10a.get("samples", []):
                source = int(sample["source_id"])
                _require(source not in target, errors, f"duplicate {role} source {source}")
                target[source] = sample
            transition = record.get("transition", {})
            _require(transition.get("target_role") == "critic", errors, f"rank {rank} C transition missing")
            _require(transition.get("critic_version") == 10, errors, f"rank {rank} did not restore C10")
            _require(transition.get("parameter_policy") == "RESIDENT", errors, f"rank {rank} Critic is not resident")
            resolved = transition.get("resolved", {})
            _require(
                resolved == {
                    "dp_world_size": 8,
                    "train_micro_batch_size_per_gpu": 1,
                    "gradient_accumulation_steps": 2,
                    "train_batch_size": 16,
                },
                errors,
                f"rank {rank} resolved batch contract changed: {resolved}",
            )

    if not critic_phase_only:
        _require(set(generator_samples) == set(range(16)), errors, "Generator Step10-A source coverage changed")
        _require(set(student_samples) == set(range(16)), errors, "Student Step10-A source coverage changed")
        _require(set(teacher_samples) == set(range(16)), errors, "Teacher Step10-A source coverage changed")
        for source in range(16):
            if source not in generator_samples or source not in student_samples or source not in teacher_samples:
                continue
            g, s, t = generator_samples[source], student_samples[source], teacher_samples[source]
            _require(g["x0_sha256"] == s["x0_sha256"] == t["x0_sha256"], errors, f"source {source} x0 changed")
            _require(s["fake_sha256"] == t["fake_sha256"], errors, f"source {source} fake score changed")
            _require(g["gradient_sha256"] == t["gradient_sha256"], errors, f"source {source} DMD gradient changed")

    if args.mode in ("transition", "transition-save"):
        for rank, value in sorted(by_rank.items()):
            gate = value.get("transition_only", {})
            _require(gate.get("status") == "PASS", errors, f"rank {rank} transition gate missing")
            _require(
                (gate.get("generator_version"), gate.get("critic_version"), gate.get("ema_optimization_step")) == (3, 10, 10),
                errors,
                f"rank {rank} transition version tuple changed",
            )
        phase_summary: dict[str, Any] = {
            "scope": "transition_only",
            "critic_phase": "Not Run",
        }
        if args.mode == "transition-save":
            try:
                phase_summary["transition_checkpoint"] = _checkpoint_inventory(
                    args.run_root
                    / "execution"
                    / "checkpoint-transition-g3-c10-ema10"
                )
            except Exception as error:
                errors.append(str(error))
    else:
        g_waves: dict[tuple[int, int], dict[str, Any]] = {}
        c_waves: dict[tuple[int, int], dict[str, Any]] = {}
        ema_records: list[dict[str, Any]] = []
        generator_final_sha = set()
        for rank in range(8):
            phase = by_rank[rank].get("critic_phase", {})
            _require(phase.get("generator_version_before") == 3 and phase.get("generator_version_after") == 3, errors, f"rank {rank} Generator version changed")
            if critic_timing:
                _require(
                    phase.get("instrumentation_profile") == "low_overhead_timing_v1",
                    errors,
                    f"rank {rank} did not use low-overhead timing instrumentation",
                )
                _require(
                    phase.get("generator_trainable_sha256_before")
                    == phase.get("generator_trainable_sha256_after")
                    == "omitted_in_low_overhead_timing_mode",
                    errors,
                    f"rank {rank} executed correctness SHA in timing mode",
                )
            else:
                _require(phase.get("generator_trainable_sha256_before") == phase.get("generator_trainable_sha256_after"), errors, f"rank {rank} Generator changed in Critic phase")
            generator_final_sha.add(phase.get("generator_trainable_sha256_after"))
            gradients = phase.get("generator_gradients", {})
            if critic_timing:
                _require(
                    gradients.get("measurement")
                    == gradients.get("trainable_grad_present")
                    == gradients.get("frozen_grad_present")
                    == "omitted_in_low_overhead_timing_mode",
                    errors,
                    f"rank {rank} timing gradient sentinel changed",
                )
            else:
                _require(gradients.get("trainable_grad_present", 0) == 0 and gradients.get("frozen_grad_present", 0) == 0, errors, f"rank {rank} retained Generator gradients")
            queue = phase.get("queue", {})
            _require(queue == {"produced": 10, "sent": 10, "acknowledged": 10, "producer_slot": "EMPTY", "outstanding_transport": 0}, errors, f"rank {rank} producer queue did not flush")
            for wave in phase.get("micro_waves", []):
                g_waves[(rank, int(wave["micro_wave"]))] = wave
            ema_records.extend(phase.get("ema", []))
        _require(len(generator_final_sha) == 1, errors, "G3 differs across Generator ranks")
        _require(len(ema_records) == 5, errors, f"EMA record count {len(ema_records)} != 5")
        for index, ema in enumerate(ema_records):
            _require(
                (ema.get("before"), ema.get("after"), ema.get("decay"))
                == (10 + index, 11 + index, 0.0),
                errors,
                f"EMA update {index} changed",
            )
            if critic_timing:
                _require(
                    ema.get("matches_generator_exact")
                    == ema.get("checksum")
                    == "omitted_in_low_overhead_timing_mode",
                    errors,
                    f"EMA update {index} executed correctness instrumentation",
                )
            else:
                _require(
                    ema.get("matches_generator_exact") is True,
                    errors,
                    f"EMA update {index} no longer matches Generator",
                )

        critic_final = []
        for rank in range(8, 16):
            phase = by_rank[rank].get("critic_phase", {})
            if critic_timing:
                _require(
                    phase.get("instrumentation_profile") == "low_overhead_timing_v1",
                    errors,
                    f"rank {rank} did not use low-overhead timing instrumentation",
                )
                _require(
                    phase.get("initial_trainable")
                    == phase.get("final_trainable")
                    == phase.get("initial_optimizer")
                    == phase.get("final_optimizer")
                    == "omitted_in_low_overhead_timing_mode",
                    errors,
                    f"rank {rank} executed state checksums in timing mode",
                )
            _require((phase.get("critic_version_before"), phase.get("critic_version_after")) == (10, 15), errors, f"rank {rank} C version changed")
            _require((phase.get("optimizer_step_before"), phase.get("optimizer_step_after")) == (10, 15), errors, f"rank {rank} optimizer step changed")
            _require((phase.get("scheduler_step_before"), phase.get("scheduler_step_after")) == (10, 15), errors, f"rank {rank} scheduler step changed")
            queue = phase.get("queue", {})
            _require(queue == {"received": 10, "consumed": 10, "optimizer_commits": 5, "consumer_slot": "EMPTY", "outstanding_transport": 0}, errors, f"rank {rank} consumer queue did not flush")
            critic_final.append(phase.get("final_trainable"))
            for wave in phase.get("micro_waves", []):
                c_waves[(rank - 8, int(wave["micro_wave"]))] = wave
        _require(len(g_waves) == 80 and len(c_waves) == 80, errors, "Critic-phase wave cardinality changed")
        if critic_timing:
            _require(
                set(critic_final) == {"omitted_in_low_overhead_timing_mode"},
                errors,
                "Critic timing state sentinel changed",
            )
        else:
            _require(len({json.dumps(value, sort_keys=True) for value in critic_final}) == 1, errors, "C15 trainables differ across Critic ranks")
        for key in sorted(set(g_waves) & set(c_waves)):
            g, c = g_waves[key], c_waves[key]
            wire = g.get("wire", {})
            _require(g.get("trace_key") == c.get("trace_key"), errors, f"wave {key} trace key changed")
            _require(g.get("trace_identity_sha256") == c.get("trace_identity_sha256"), errors, f"wave {key} trace identity changed")
            _require(g.get("source_trace_rank") == c.get("source_trace_rank"), errors, f"wave {key} source trace rank changed")
            _require(wire.get("x0_checksum") == c.get("x0_identity", {}).get("value"), errors, f"wave {key} x0 checksum changed")
            expected_boundary = key[1] % 2 == 1
            _require(bool(c.get("gradient_accumulation_boundary")) == expected_boundary, errors, f"wave {key} accumulation boundary changed")
            _require(bool(g.get("ack", {}).get("optimizer_committed")) == expected_boundary, errors, f"wave {key} ACK commit changed")
        checkpoint = None
        if args.mode in ("correctness", "critic-correctness"):
            try:
                checkpoint = _checkpoint_inventory(args.run_root / "execution" / "checkpoint-15")
            except Exception as error:
                errors.append(str(error))
        phase_summary = {
            "scope": (
                "critic_phase_from_common_transition_checkpoint"
                if critic_phase_only
                else "continuous_full5_step10_to_step14"
            ),
            "generator_waves": len(g_waves),
            "critic_waves": len(c_waves),
            "critic_commits": 5,
            "ema_commits": len(ema_records),
            "checkpoint_15": checkpoint if checkpoint is not None else "Not Run",
        }

    max_full5 = max((float(value.get("full5_execution_seconds", 0.0)) for value in ranks), default=0.0)
    max_critic_subgraph = max(
        (
            float(value.get("critic_subgraph_execution_seconds", 0.0))
            for value in ranks
        ),
        default=0.0,
    )
    max_critic = max((float(value.get("critic_phase_wall_seconds", 0.0)) for value in ranks), default=0.0)
    peaks = {
        f"node_{node}": _resource_peak(args.run_root / "launcher" / f"node_{node}" / "resource.csv")
        for node in (0, 1)
    }
    result = {
        "schema_version": "distillschedule.routeb_full5_candidate_a_summary.v1",
        "status": "PASS" if not errors else "FAIL",
        "experiment_id": "RB16-F5-PIPE-MATCH16",
        "run_id": args.run_root.name,
        "schedule": args.schedule,
        "mode": args.mode,
        "rank_count": len(ranks),
        "identities": identities,
        "timing_correctness_binding": (
            timing_correctness_binding if critic_timing else "Not Run"
        ),
        "step10a": (
            "Not Run"
            if critic_phase_only
            else {
                "generator_sources": len(generator_samples),
                "student_sources": len(student_samples),
                "teacher_sources": len(teacher_samples),
                "generator_updates": 1,
            }
        ),
        "phase": phase_summary,
        "timing": {
            "full5_makespan_seconds": max_full5,
            "critic_subgraph_makespan_seconds": max_critic_subgraph,
            "critic_phase_makespan_seconds": max_critic,
            "label": (
                "formal_candidate_run"
                if args.mode in ("timing", "critic-timing")
                else "diagnostic"
            ),
        },
        "node_cgroup_sampled_peaks_bytes": peaks,
        "errors": errors,
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
