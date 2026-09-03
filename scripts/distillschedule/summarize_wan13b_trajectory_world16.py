#!/usr/bin/env python3
"""Fail-closed summary for one completed Wan1.3B trajectory run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from distillschedule.evidence.trajectory_coarse import load_trajectory_jsonl
from distillschedule.evidence.trajectory_state_sketch import (
    SCHEMA_VERSION as STATE_SKETCH_SCHEMA_VERSION,
)


REGISTERED_CYCLE_COUNTS = frozenset({1, 5, 50})


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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
        errors.append("trajectory cycles must be one of 1, 5, or 50")
    rank_rows: dict[int, list[Any]] = {}
    launch_identities: list[tuple[str | None, ...]] = []
    for node in range(2):
        path = args.run_root / "launcher" / f"node_{node}" / "launch_manifest.txt"
        try:
            manifest = {
                key: value
                for line in path.read_text().splitlines()
                if "=" in line
                for key, value in (line.split("=", 1),)
            }
        except OSError as error:
            errors.append(f"node{node}: invalid launch manifest: {error}")
            continue
        if manifest.get("dmd_system_head") != args.expected_head:
            errors.append(f"node{node}: launch HEAD differs from expected HEAD")
        if manifest.get("trajectory_cycles") != str(args.cycles):
            errors.append(f"node{node}: launch trajectory cycle count changed")
        if manifest.get("claim_boundary") != "DIAGNOSTIC_ONLY":
            errors.append(f"node{node}: trajectory claim boundary changed")
        if manifest.get("placement") != args.placement:
            errors.append(f"node{node}: launch placement changed")
        launch_identities.append(
            (
                manifest.get("run_id"),
                manifest.get("trajectory_seed"),
                manifest.get("trajectory_arm_id"),
                manifest.get("trace_hash"),
                manifest.get("bootstrap_hash"),
            )
        )
    if launch_identities and len(set(launch_identities)) != 1:
        errors.append("node launch identities differ")
    for rank in range(16):
        path = args.run_root / "execution" / "trajectory" / f"rank_{rank:02d}.jsonl"
        if not path.is_file():
            errors.append(f"rank{rank}: missing trajectory JSONL")
            continue
        try:
            rows = load_trajectory_jsonl(path)
        except (TypeError, ValueError, OSError) as error:
            errors.append(f"rank{rank}: invalid trajectory JSONL: {error}")
            continue
        rank_rows[rank] = rows
        if len(rows) != args.cycles:
            errors.append(f"rank{rank}: cycles {len(rows)} != {args.cycles}")
            continue
        for cycle, row in enumerate(rows, 1):
            if row.rank != rank or row.cycle != cycle:
                errors.append(f"rank{rank}: non-contiguous cycle identity")
            if row.implementation.get("placement") != args.placement:
                errors.append(f"rank{rank}: placement changed")
            if dict(row.cycle_updates) != {"generator": 1, "critic": 5, "ema": 5}:
                errors.append(f"rank{rank}: cycle{cycle} update inventory changed")
            if dict(row.cumulative_updates) != {
                "generator": cycle,
                "critic": cycle * 5,
                "ema": cycle * 5,
            }:
                errors.append(f"rank{rank}: cycle{cycle} cumulative inventory changed")
            expected_versions = {
                "generator": 2 + cycle,
                "critic": 10 + cycle * 5,
                "ema": 10 + cycle * 5,
            }
            if dict(row.versions) != expected_versions:
                errors.append(f"rank{rank}: cycle{cycle} versions changed")
            if not row.finite or row.runtime_status != "PASS":
                errors.append(f"rank{rank}: cycle{cycle} is non-finite/failed")
            if len(row.sample_identities) != 6:
                errors.append(f"rank{rank}: cycle{cycle} sample inventory changed")
            if row.call_census.get("generator_score_forwards") != 3:
                errors.append(f"rank{rank}: cycle{cycle} Generator Score census changed")
            if row.call_census.get("critic_score_forwards") != 5:
                errors.append(f"rank{rank}: cycle{cycle} Critic Score census changed")
            if row.call_census.get("physical_rollout_batches") != 5:
                errors.append(f"rank{rank}: cycle{cycle} physical rollout census changed")
        final_path = args.run_root / "execution" / "ranks" / f"rank_{rank:02d}.json"
        try:
            final = json.loads(final_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            errors.append(f"rank{rank}: invalid final rank receipt: {error}")
            continue
        expected_final_versions = {
            "generator": 2 + args.cycles,
            "critic": 10 + 5 * args.cycles,
            "ema": 10 + 5 * args.cycles,
        }
        if final.get("status") != "PASS" or final.get("rank") != rank:
            errors.append(f"rank{rank}: final rank receipt is not PASS")
        if final.get("placement") != args.placement:
            errors.append(f"rank{rank}: final rank placement changed")
        if final.get("completed_cycles") != args.cycles:
            errors.append(f"rank{rank}: final completed-cycle count changed")
        if final.get("updates") != {
            "generator": args.cycles,
            "critic": 5 * args.cycles,
            "ema": 5 * args.cycles,
        }:
            errors.append(f"rank{rank}: final update inventory changed")
        if final.get("final_versions") != expected_final_versions:
            errors.append(f"rank{rank}: final versions changed")
    fingerprints = {
        (
            row.matched_contract.get("trace_bundle_hash"),
            row.matched_contract.get("trace_binding_hash"),
            row.seed,
        )
        for rows in rank_rows.values()
        for row in rows
    }
    if len(fingerprints) != 1:
        errors.append(f"mixed matched identities: {sorted(fingerprints)}")
    expected_anchor_cycles = sorted(
        {cycle for cycle in (1, 10, 25, 50, args.cycles) if cycle <= args.cycles}
    )
    anchors: list[dict[str, Any]] = []
    for cycle in expected_anchor_cycles:
        path = (
            args.run_root
            / "execution"
            / "trajectory"
            / "anchors"
            / f"cycle_{cycle:02d}_global.json"
        )
        try:
            anchor = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            errors.append(f"cycle{cycle}: invalid numeric state anchor: {error}")
            continue
        recorded = anchor.get("anchor_sha256")
        canonical = dict(anchor)
        canonical.pop("anchor_sha256", None)
        actual = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if recorded != actual:
            errors.append(f"cycle{cycle}: state anchor SHA mismatch")
        if anchor.get("schema_version") != "distillschedule.wan13b_trajectory_state_anchor.v2":
            errors.append(f"cycle{cycle}: state anchor schema changed")
        if anchor.get("cycle") != cycle or anchor.get("placement") != args.placement:
            errors.append(f"cycle{cycle}: state anchor identity changed")
        components = dict(anchor.get("components", {}))
        expected_components = {
            "generator_trainable",
            "generator_adam",
            "generator_ema",
            "critic_trainable",
            "critic_adam",
        }
        if set(components) != expected_components:
            errors.append(f"cycle{cycle}: numeric state component inventory changed")
        for component, evidence in components.items():
            sketch = dict(evidence.get("numeric_sketch", {}))
            if sketch.get("schema_version") != STATE_SKETCH_SCHEMA_VERSION:
                errors.append(f"cycle{cycle}: {component} numeric sketch schema changed")
            if sketch.get("world_size") != 16:
                errors.append(f"cycle{cycle}: {component} sketch world size changed")
            if sketch.get("replica_validation") != (
                "ALL_RANKS_CONTRIBUTE_SAMPLED_COORDINATES_MUST_MATCH"
            ):
                errors.append(f"cycle{cycle}: {component} replica agreement is unverified")
            if int(sketch.get("coordinate_count", 0)) <= 0:
                errors.append(f"cycle{cycle}: {component} has no numeric coordinates")
            if not sketch.get("coordinates"):
                errors.append(f"cycle{cycle}: {component} coordinate values are missing")
            if evidence.get("global_moments", {}).get("status") != "PASS":
                errors.append(f"cycle{cycle}: {component} global moments are invalid")
        anchors.append({"cycle": cycle, "path": str(path), "sha256": recorded})
    value = {
        "schema_version": "distillschedule.wan13b_trajectory_summary.v1",
        "status": "PASS" if not errors and len(rank_rows) == 16 else "FAIL",
        "errors": errors,
        "run_root": str(args.run_root.resolve()),
        "placement": args.placement,
        "expected_head": args.expected_head,
        "cycles": args.cycles,
        "rank_count": len(rank_rows),
        "rank_cycle_receipts": sum(len(rows) for rows in rank_rows.values()),
        "expected_rank_cycle_receipts": 16 * args.cycles,
        "matched_identity": list(next(iter(fingerprints))) if len(fingerprints) == 1 else None,
        "final_versions": {"generator": 2 + args.cycles, "critic": 10 + 5 * args.cycles, "ema": 10 + 5 * args.cycles},
        "numeric_state_anchors": anchors,
        "diagnostic_only": True,
        "formal_timing_eligible": False,
    }
    _atomic_json(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
