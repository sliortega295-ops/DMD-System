#!/usr/bin/env python3
"""Fail-closed summary for the strict-v4 Step10-A saved-tensor census."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED = (
    ((1, 11040, 5120), "torch.bfloat16", 113_049_600, 80),
    ((1, 4560, 5120), "torch.bfloat16", 46_694_400, 80),
    ((1, 2940, 5120), "torch.bfloat16", 30_105_600, 80),
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    ranks = []
    for rank in range(16):
        path = args.run_root / "execution" / "ranks" / f"rank_{rank:02d}.json"
        if not path.is_file():
            errors.append(f"missing rank {rank}")
            continue
        value = json.loads(path.read_text())
        ranks.append(value)
        if value.get("status") != "PASS":
            errors.append(f"rank {rank} status {value.get('status')}")

    observed: dict[str, list[dict[str, Any]]] = {}
    for rank in range(8):
        if rank >= len(ranks):
            continue
        value = next((item for item in ranks if item.get("rank") == rank), None)
        if value is None:
            continue
        census = value.get("step10a_saved_tensor_census")
        if not isinstance(census, dict):
            errors.append(f"rank {rank} missing census")
            continue
        if census.get("tensor_references_retained_by_census") != 0:
            errors.append(f"rank {rank} retained tensor references")
        if census.get("pack_returns_original_tensor") is not True:
            errors.append(f"rank {rank} pack was not pass-through")
        if census.get("unpack_returns_same_tensor") is not True:
            errors.append(f"rank {rank} unpack was not pass-through")
        signatures = census.get("signatures", [])
        rank_observed = []
        for shape, dtype, logical_bytes, expected_count in EXPECTED:
            count = sum(
                int(item.get("count", 0))
                for item in signatures
                if tuple(item.get("shape", ())) == shape
                and item.get("dtype") == dtype
                and int(item.get("logical_bytes", -1)) == logical_bytes
            )
            record = {
                "shape": list(shape),
                "dtype": dtype,
                "logical_bytes": logical_bytes,
                "observed_count": count,
                "expected_count": expected_count,
                "match": count == expected_count,
            }
            rank_observed.append(record)
            if count != expected_count:
                errors.append(
                    f"rank {rank} signature {shape}/{logical_bytes}: {count} != {expected_count}"
                )
        observed[str(rank)] = rank_observed

    result = {
        "schema_version": "distillschedule.step10a_saved_tensor_census_summary.v1",
        "run_id": args.run_root.name,
        "status": "PASS" if not errors else "REFUTED_SIGNATURE_OR_CONTRACT",
        "claim_scope": "causal ownership diagnostic only; no speedup",
        "errors": errors,
        "generator_rank_count": len(observed),
        "expected_derivation": "two GAS waves x forty blocks per selected stage = eighty",
        "observed": observed,
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
