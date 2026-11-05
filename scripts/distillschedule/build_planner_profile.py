#!/usr/bin/env python3
"""Aggregate planner calibration receipts into one immutable profile JSON."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from distillschedule.planner.calibration import calibration_coverage
from distillschedule.planner.profile_db import ProfileDB
from distillschedule.planner.schema import (
    ModelInventory,
    PlanObservation,
    PlannerProfile,
    PrimitiveKey,
    PrimitiveMeasurement,
    TransitionMeasurement,
    load_profile,
)
from distillschedule.profiling.planner_profile import (
    EVENT_SCHEMA,
    HARDWARE_SCHEMA,
    MODEL_SCHEMA,
    load_profile_events,
)


def _maximum(values: Iterable[Any]) -> int | None:
    present = [int(value) for value in values if value is not None]
    return max(present) if present else None


def _median(values: Iterable[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def _resource_max(event: Mapping[str, Any], section: str, key: str) -> int | None:
    values = []
    for boundary in ("before", "after"):
        snapshot = event.get(boundary, {}).get(section, {})
        value = snapshot.get(key)
        if value is not None:
            values.append(int(value))
    return max(values) if values else None


def _primitive_key(event: Mapping[str, Any]) -> PrimitiveKey:
    return PrimitiveKey(
        workload=str(event["workload"]),
        operation=str(event["operation"]),
        role=str(event["role"]),
        backend=str(event["backend"]),
        mesh_size=int(event["mesh_size"]),
        local_batch=int(event["local_batch"]),
        stage=(None if event.get("stage") is None else int(event["stage"])),
        grad_mode=str(event.get("grad_mode", "no_grad")),
        residency=str(event.get("residency", "resident")),
        activation_policy=str(event.get("activation_policy", "none")),
        offload_blocks=tuple(int(value) for value in event.get("offload_blocks", ())),
        dtype=str(event.get("dtype", "bfloat16")),
    )


def aggregate_events(seed: PlannerProfile, events: list[dict[str, Any]]) -> PlannerProfile:
    primitive_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    primitive_keys: dict[str, PrimitiveKey] = {}
    model_records: dict[str, dict[str, Any]] = {
        value.role: {
            "role": value.role,
            "parameter_bytes": value.parameter_bytes,
            "frozen_parameter_bytes": value.frozen_parameter_bytes,
            "trainable_parameter_bytes": value.trainable_parameter_bytes,
            "buffer_bytes": value.buffer_bytes,
            "block_parameter_bytes": list(value.block_parameter_bytes),
            "trainable_numel": value.trainable_numel,
            "optimizer_state_bytes": value.optimizer_state_bytes,
            "ema_bytes": value.ema_bytes,
        }
        for value in seed.models
    }
    transition_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hardware_updates: dict[str, Any] = {}

    for event in events:
        schema = event.get("schema_version")
        if schema == EVENT_SCHEMA and event.get("status") == "PASS":
            key = _primitive_key(event)
            primitive_keys[key.id] = key
            primitive_groups[key.id].append(event)
            transition_name = event.get("metadata", {}).get("transition_name")
            if transition_name:
                transition_groups[str(transition_name)].append(event)
        elif schema == MODEL_SCHEMA:
            role = str(event["role"])
            prior = model_records.get(role)
            current = {
                "role": role,
                "parameter_bytes": int(event["parameter_bytes"]),
                "frozen_parameter_bytes": int(event["frozen_parameter_bytes"]),
                "trainable_parameter_bytes": int(event["trainable_parameter_bytes"]),
                "buffer_bytes": int(event["buffer_bytes"]),
                "block_parameter_bytes": [int(value) for value in event.get("block_parameter_bytes", ())],
                "trainable_numel": int(event.get("trainable_numel", 0)),
                "optimizer_state_bytes": event.get("optimizer_state_bytes"),
                "ema_bytes": event.get("ema_bytes"),
            }
            if prior is not None and prior != current:
                raise ValueError(f"model inventory drift for role {role}")
            model_records[role] = current
        elif schema == HARDWARE_SCHEMA:
            for key, value in event.items():
                if key in {"schema_version", "run_id", "rank", "source_sha"} or value is None:
                    continue
                if key in hardware_updates and hardware_updates[key] != value:
                    raise ValueError(f"hardware calibration drift for {key}")
                hardware_updates[key] = value

    measurements: dict[str, PrimitiveMeasurement] = {value.key.id: value for value in seed.primitives}
    for key_id, values in primitive_groups.items():
        key = primitive_keys[key_id]
        measurements[key_id] = PrimitiveMeasurement(
            key=key,
            wall_seconds=float(statistics.median(float(value["wall_seconds"]) for value in values)),
            samples=len(values),
            provenance=";".join(sorted({str(value.get("run_id", "unknown")) for value in values})),
            cuda_device_seconds=_median(value.get("cuda_device_seconds") for value in values),
            compute_seconds=_median(value.get("compute_seconds") for value in values),
            nccl_seconds=_median(value.get("nccl_seconds") for value in values),
            h2d_seconds=_median(value.get("h2d_seconds") for value in values),
            d2h_seconds=_median(value.get("d2h_seconds") for value in values),
            transferred_h2d_bytes=int(statistics.median(int(value.get("h2d_bytes", 0)) for value in values)),
            transferred_d2h_bytes=int(statistics.median(int(value.get("d2h_bytes", 0)) for value in values)),
            collective_bytes=int(statistics.median(int(value.get("collective_bytes", 0)) for value in values)),
            gpu_peak_bytes=_maximum(
                _resource_max(value, "cuda", "max_allocated_bytes") for value in values
            ),
            cpu_peak_bytes=_maximum(
                _resource_max(value, "cgroup", "memory_current_bytes") for value in values
            ),
            activation_peak_bytes=_maximum(value.get("activation_peak_bytes") for value in values),
            parameter_materialization_count=int(
                statistics.median(
                    int(value.get("parameter_materialization_count", 0))
                    for value in values
                )
            ),
            confidence="measured",
            metadata={
                "rank_count": len({int(value["rank"]) for value in values}),
                "source_shas": sorted({str(value.get("source_sha")) for value in values}),
                "trace_shas": sorted({str(value.get("trace_sha")) for value in values if value.get("trace_sha")}),
            },
        )

    transitions: dict[str, TransitionMeasurement] = {value.name: value for value in seed.transitions}
    for name, values in transition_groups.items():
        metadata = dict(values[0].get("metadata", {}))
        transitions[name] = TransitionMeasurement(
            name=name,
            source_layout=str(metadata["source_layout"]),
            target_layout=str(metadata["target_layout"]),
            wall_seconds=float(statistics.median(float(value["wall_seconds"]) for value in values)),
            samples=len(values),
            provenance=";".join(sorted({str(value.get("run_id", "unknown")) for value in values})),
            h2d_bytes=int(statistics.median(int(value.get("h2d_bytes", 0)) for value in values)),
            d2h_bytes=int(statistics.median(int(value.get("d2h_bytes", 0)) for value in values)),
            node_cpu_peak_bytes=_maximum(
                _resource_max(value, "cgroup", "memory_current_bytes") for value in values
            ),
            gpu_peak_bytes=_maximum(
                _resource_max(value, "cuda", "max_allocated_bytes") for value in values
            ),
            metadata={"event_count": len(values)},
        )

    hardware_payload = {**seed.hardware.__dict__, **hardware_updates}
    return PlannerProfile(
        hardware=type(seed.hardware)(**hardware_payload),
        models=tuple(ModelInventory(**value) for _, value in sorted(model_records.items())),
        primitives=tuple(measurements[key] for key in sorted(measurements)),
        transitions=tuple(transitions[key] for key in sorted(transitions)),
        plans=seed.plans,
        metadata={
            **dict(seed.metadata),
            "aggregated_profile_event_count": len(events),
            "profile_builder": "build_planner_profile.py",
        },
    )


def load_plan_observation(spec: str) -> PlanObservation:
    if "=" not in spec:
        raise ValueError("--plan-observation must use NAME=JSON")
    name, raw_path = spec.split("=", 1)
    value = json.loads(Path(raw_path).read_text())
    value.setdefault("name", name)
    value["full5_seconds"] = tuple(float(item) for item in value["full5_seconds"])
    return PlanObservation(**value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-profile", required=True)
    parser.add_argument("--event", action="append", default=[])
    parser.add_argument("--plan-observation", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--coverage-output")
    args = parser.parse_args()

    seed = load_profile(args.seed_profile)
    profile = aggregate_events(seed, load_profile_events(args.event))
    observations = {value.name: value for value in profile.plans}
    for spec in args.plan_observation:
        value = load_plan_observation(spec)
        observations[value.name] = value
    profile = PlannerProfile(
        hardware=profile.hardware,
        models=profile.models,
        primitives=profile.primitives,
        transitions=profile.transitions,
        plans=tuple(observations[key] for key in sorted(observations)),
        metadata=profile.metadata,
    )
    profile.write(args.output)
    coverage = calibration_coverage(ProfileDB(profile))
    if args.coverage_output:
        target = Path(args.coverage_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "primitive_count": len(profile.primitives),
        "transition_count": len(profile.transitions),
        "plan_count": len(profile.plans),
        "calibration": {key: value for key, value in coverage.items() if key != "missing"},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
