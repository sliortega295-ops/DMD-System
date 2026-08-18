#!/usr/bin/env python3
"""Fail-closed summary for one LightX2V MiniMax-H3 world16 DMD benchmark cycle."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    return p


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _memory_event_delta(before: Path, after: Path, key: str) -> int | None:
    if not before.exists() or not after.exists():
        return None
    def parse(path: Path) -> dict[str, int]:
        result: dict[str, int] = {}
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    result[parts[0]] = int(parts[1])
                except ValueError:
                    pass
        return result
    a, b = parse(before), parse(after)
    if key not in a or key not in b:
        return None
    return b[key] - a[key]


def main() -> int:
    args = parser().parse_args()
    contract = _load(args.contract)
    errors: list[str] = []
    receipt_paths = sorted((args.run_root / "receipts").glob("rank_*.json"))
    if len(receipt_paths) != 16:
        errors.append(f"expected 16 rank receipts, found {len(receipt_paths)}")
    receipts = [_load(path) for path in receipt_paths]

    expected_ranks = list(range(16))
    observed_ranks = sorted(int(row.get("rank", -1)) for row in receipts)
    if observed_ranks != expected_ranks:
        errors.append(f"rank coverage mismatch: {observed_ranks}")
    expected_upstream = contract["upstream"]["commit"]
    expected_cycle = contract["cycle"]
    patch_values = {row.get("benchmark_patch_sha256") for row in receipts}
    if len(patch_values) != 1 or None in patch_values:
        errors.append("benchmark patch identity is missing or inconsistent")

    for row in receipts:
        if row.get("schema_version") != "distillschedule.lightx2v_minimax_cycle.v1":
            errors.append(f"rank {row.get('rank')} has wrong receipt schema")
        if int(row.get("world_size", -1)) != 16:
            errors.append(f"rank {row.get('rank')} did not execute world16")
        if int(row.get("student_updates", -1)) != expected_cycle["student_updates"]:
            errors.append(f"rank {row.get('rank')} student update count mismatch")
        if int(row.get("fake_updates", -1)) != expected_cycle["fake_updates"]:
            errors.append(f"rank {row.get('rank')} fake update count mismatch")
        if str(row.get("fixed_end_step_idx")) != str(expected_cycle["benchmark_end_step_idx"]):
            errors.append(f"rank {row.get('rank')} benchmark end-step mismatch")
        if row.get("upstream_commit") != expected_upstream:
            errors.append(f"rank {row.get('rank')} upstream source mismatch")
        if row.get("dmd_system_head") != args.expected_head:
            errors.append(f"rank {row.get('rank')} DMD-System source mismatch")
        for key in ("local_cycle_seconds", "global_cycle_seconds", "dmd_loss", "fake_loss"):
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                errors.append(f"rank {row.get('rank')} missing numeric {key}")
                continue
            if not math.isfinite(value):
                errors.append(f"rank {row.get('rank')} non-finite {key}")

    global_times = [float(row["global_cycle_seconds"]) for row in receipts if "global_cycle_seconds" in row]
    cycle_seconds = max(global_times) if global_times else None
    if global_times and max(global_times) - min(global_times) > 1e-6:
        errors.append("global cycle seconds differ across ranks after MAX all-reduce")

    node_results = []
    oom_deltas = []
    oom_kill_deltas = []
    for node in (0, 1):
        node_dir = args.run_root / f"node_{node}"
        result_path = node_dir / "result.json"
        if not result_path.exists():
            errors.append(f"missing node result {node}")
            continue
        result = _load(result_path)
        node_results.append(result)
        if int(result.get("return_code", -1)) != 0:
            errors.append(f"node {node} returned {result.get('return_code')}")
        if result.get("dmd_system_head") != args.expected_head:
            errors.append(f"node {node} DMD-System head mismatch")
        oom = _memory_event_delta(node_dir / "memory.events.before", node_dir / "memory.events.after", "oom")
        oom_kill = _memory_event_delta(node_dir / "memory.events.before", node_dir / "memory.events.after", "oom_kill")
        oom_deltas.append(oom)
        oom_kill_deltas.append(oom_kill)
        if oom is not None and oom > 0:
            errors.append(f"node {node} cgroup oom increased by {oom}")
        if oom_kill is not None and oom_kill > 0:
            errors.append(f"node {node} cgroup oom_kill increased by {oom_kill}")

    payload = {
        "schema_version": "distillschedule.lightx2v_minimax_world16_summary.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "contract_name": contract["name"],
        "scientific_scope": (
            "MiniMax-H3 LightX2V upstream DMD cycle, 2x8 A100 world16 FSDP2, "
            "global batch16, 1 Student update + 5 Fake updates, fixed full 4-eval back-simulation"
        ),
        "cycle_seconds": cycle_seconds,
        "rank_count": len(receipts),
        "student_updates": expected_cycle["student_updates"],
        "fake_updates": expected_cycle["fake_updates"],
        "ema_updates": expected_cycle["ema_updates"],
        "benchmark_end_step_idx": expected_cycle["benchmark_end_step_idx"],
        "model_evaluations_per_back_simulation": expected_cycle["model_evaluations_per_back_simulation"],
        "global_batch": expected_cycle["global_batch"],
        "upstream_commit": expected_upstream,
        "benchmark_patch_sha256": next(iter(patch_values)) if len(patch_values) == 1 else None,
        "dmd_system_head": args.expected_head,
        "max_allocated_bytes": max((int(row["max_allocated_bytes"]) for row in receipts if row.get("max_allocated_bytes") is not None), default=None),
        "max_reserved_bytes": max((int(row["max_reserved_bytes"]) for row in receipts if row.get("max_reserved_bytes") is not None), default=None),
        "node_oom_delta": oom_deltas,
        "node_oom_kill_delta": oom_kill_deltas,
        "claim_boundary": {
            "cycle_timing_claimable": not errors,
            "startup_excluded": True,
            "profiled_timing": False,
            "cross_model_helios_speedup_claimable": False,
            "matched_system_speedup_requires_minimax_distillschedule_variant": True
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
