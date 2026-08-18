#!/usr/bin/env python3
"""Fail-closed summarizer for one staged MiniMax-H3 world16 canary."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--mode", choices=("native", "distillschedule"), required=True)
    value.add_argument("--expected-head", required=True)
    value.add_argument("--canary-stage", choices=("student_forward", "student_update"), required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def memory_events(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    result: dict[str, int] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            result[fields[0]] = int(fields[1])
    return result


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def read_debug_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    events: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed worker may leave only a partial final line.  Preserve it
            # as an explicit evidence warning rather than silently discarding it.
            events.append({"event": "INVALID_JSONL", "raw": line})
    return events


def main() -> int:
    args = parser().parse_args()
    receipts_dir = args.run_root / "receipts"
    errors: list[str] = []
    receipts: list[dict] = []
    for rank in range(16):
        path = receipts_dir / f"rank_{rank:03d}.json"
        if not path.is_file():
            errors.append(f"missing rank receipt: {path}")
            continue
        receipt = load_json(path)
        receipts.append(receipt)
        if receipt.get("rank") != rank:
            errors.append(f"rank receipt mismatch at {rank}")
        for key, expected in (("world_size", 16), ("mode", args.mode), ("canary_stage", args.canary_stage)):
            if receipt.get(key) != expected:
                errors.append(f"rank {rank}: {key}={receipt.get(key)!r} != {expected!r}")
        if receipt.get("dmd_system_head") != args.expected_head:
            errors.append(f"rank {rank}: DMD-System head mismatch")
        if receipt.get("status") != "PASS":
            errors.append(f"rank {rank}: receipt status={receipt.get('status')!r}")
        inventory = receipt.get("application_inventory", {})
        if args.canary_stage == "student_update":
            # This stage executes the complete native Student DMD update,
            # including the one Fake Score and one Teacher Score evaluation
            # needed to form the Student loss.  It is not a rollout-only
            # forward canary.
            expected = {
                "forward_counts": {"student": 4, "fake": 1, "teacher": 1},
                "grad_forward_counts": {"student": 1, "fake": 0, "teacher": 0},
                "backward_counts": {"student": 1, "fake": 0},
                "total_forwards": 6,
                "total_backward_graphs": 1,
            }
        else:
            expected = {
                "forward_counts": {"student": 4, "fake": 0, "teacher": 0},
                "grad_forward_counts": {"student": 1, "fake": 0, "teacher": 0},
                "backward_counts": {"student": 0, "fake": 0},
                "total_forwards": 4,
                "total_backward_graphs": 0,
            }
        for key, value in expected.items():
            if inventory.get(key) != value:
                errors.append(f"rank {rank}: inventory {key}={inventory.get(key)!r} != {value!r}")

    if len(receipts) == 16:
        identities = (
            "upstream_commit",
            "config_sha256",
            "prompt_metadata_sha256",
            "model_path",
            "prompt_cache",
            "attention_backend",
            "benchmark_seed",
            "checkpoint_segment",
        )
        for key in identities:
            values = {json.dumps(receipt.get(key), sort_keys=True) for receipt in receipts}
            if len(values) != 1:
                errors.append(f"rank disagreement for {key}: {sorted(values)}")

        policies = {json.dumps(receipt.get("checkpoint_policy"), sort_keys=True) for receipt in receipts}
        if len(policies) != 1:
            errors.append("rank disagreement for checkpoint_policy")
        alloc = [int(receipt.get("max_allocated_bytes", 0)) for receipt in receipts]
        reserved = [int(receipt.get("max_reserved_bytes", 0)) for receipt in receipts]
        max_alloc = max(alloc)
        max_reserved = max(reserved)
        alloc_stats = {"min": min(alloc), "median": statistics.median(alloc), "max": max_alloc}
        reserved_stats = {"min": min(reserved), "median": statistics.median(reserved), "max": max_reserved}
    else:
        alloc_stats = reserved_stats = {}

    debug_events_by_rank: dict[str, list[dict]] = {}
    for rank in range(16):
        debug_events_by_rank[str(rank)] = read_debug_events(
            receipts_dir / f"rank_{rank:03d}.segment_debug.jsonl"
        )
    # Segment debug JSONL is only required for the segmented-checkpoint
    # canaries.  Native boundary-offload and ordinary forward/update canaries
    # intentionally do not enable this optional instrumentation.
    if any(receipt.get("checkpoint_segment") not in (None, "0", 0) for receipt in receipts):
        if not any(debug_events_by_rank.values()):
            errors.append("no segment debug JSONL was produced")

    node_oom: dict[str, dict[str, int]] = {}
    for node in (0, 1):
        node_dir = args.run_root / f"node_{node}"
        result_path = node_dir / "result.json"
        if not result_path.is_file():
            errors.append(f"missing node_{node}/result.json")
            continue
        result = load_json(result_path)
        if result.get("return_code") != 0:
            errors.append(f"node {node}: return_code={result.get('return_code')}")
        before = memory_events(node_dir / "memory.events.before")
        after = memory_events(node_dir / "memory.events.after")
        delta = {key: after.get(key, 0) - before.get(key, 0) for key in set(before) | set(after)}
        node_oom[str(node)] = delta
        if delta.get("oom", 0) or delta.get("oom_kill", 0):
            errors.append(f"node {node}: cgroup OOM delta={delta}")

    summary = {
        "schema_version": "distillschedule.lightx2v_minimax_canary_summary.v1",
        "status": "PASS" if not errors else "FAIL",
        "mode": args.mode,
        "canary_stage": args.canary_stage,
        "world_size": 16,
        "rank_count": len(receipts),
        "max_allocated_bytes": max((int(item.get("max_allocated_bytes", 0)) for item in receipts), default=0),
        "max_reserved_bytes": max((int(item.get("max_reserved_bytes", 0)) for item in receipts), default=0),
        "allocated_bytes_by_rank": alloc_stats,
        "reserved_bytes_by_rank": reserved_stats,
        "application_inventory": receipts[0].get("application_inventory") if receipts else {},
        "checkpoint_policy": receipts[0].get("checkpoint_policy") if receipts else {},
        "segment_debug_events_by_rank": debug_events_by_rank,
        "node_memory_event_deltas": node_oom,
        "errors": errors,
    }
    atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
