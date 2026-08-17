#!/usr/bin/env python3
"""Fail-closed summarizer for one matched MiniMax-H3 world16 mode."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from runtime.lightx2v_minimax_matched import validate_matched_receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--mode", choices=("native", "distillschedule"), required=True)
    value.add_argument("--expected-head", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def load_json(path: Path):
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


def main() -> int:
    args = parser().parse_args()
    receipts_dir = args.run_root / "receipts"
    receipts = [load_json(receipts_dir / f"rank_{rank:03d}.json") for rank in range(16)]
    errors: list[str] = []
    for rank, receipt in enumerate(receipts):
        if receipt.get("rank") != rank:
            errors.append(f"rank receipt mismatch at {rank}")
        if receipt.get("world_size") != 16:
            errors.append(f"rank {rank}: world_size={receipt.get('world_size')}")
        if receipt.get("mode") != args.mode:
            errors.append(f"rank {rank}: mode={receipt.get('mode')}")
        if receipt.get("dmd_system_head") != args.expected_head:
            errors.append(f"rank {rank}: DMD-System head mismatch")
        errors.extend(f"rank {rank}: {item}" for item in validate_matched_receipt(receipt))
        if args.mode == "distillschedule" and receipt.get("rng_neutral_fake_updates") != [True] * 5:
            errors.append(f"rank {rank}: Fake compute RNG-neutrality gate failed")

    identities = (
        "upstream_commit",
        "config_sha256",
        "prompt_metadata_sha256",
        "model_path",
        "prompt_cache",
        "attention_backend",
        "benchmark_seed",
    )
    for key in identities:
        values = {json.dumps(receipt.get(key), sort_keys=True) for receipt in receipts}
        if len(values) != 1:
            errors.append(f"rank disagreement for {key}: {sorted(values)}")

    sample_sequences = [receipt.get("sample_sequence") for receipt in receipts]
    if any(len(value or []) != 6 for value in sample_sequences):
        errors.append("every rank must consume exactly six samples")

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

    walls = [float(receipt["global_cycle_seconds"]) for receipt in receipts]
    if max(walls) - min(walls) > 1e-6:
        errors.append("global cycle wall differs across rank receipts")

    result = {
        "schema_version": "distillschedule.lightx2v_minimax_matched_summary.v1",
        "status": "PASS" if not errors else "FAIL",
        "mode": args.mode,
        "world_size": 16,
        "global_cycle_seconds": statistics.median(walls),
        "max_allocated_bytes": max(int(receipt["max_allocated_bytes"]) for receipt in receipts),
        "max_reserved_bytes": max(int(receipt["max_reserved_bytes"]) for receipt in receipts),
        "application_inventory": receipts[0]["application_inventory"],
        "phase_order": receipts[0]["phase_order"],
        "sample_sequences_by_rank": sample_sequences,
        "state_hashes_by_rank": [receipt["state_hashes"] for receipt in receipts],
        "identities": {key: receipts[0].get(key) for key in identities},
        "node_memory_event_deltas": node_oom,
        "errors": errors,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
