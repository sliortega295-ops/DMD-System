#!/usr/bin/env python3
"""Fail-closed summary for one static FSDP2 world16 Full5 run."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from distillschedule.runtime.fsdp2_static import contract_for_variant


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("no-gan", "gan"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = contract_for_variant(args.variant)
    rank_root = args.run_root / "execution" / "ranks"
    errors: list[str] = []
    ranks: list[dict[str, Any]] = []
    for rank in range(16):
        path = rank_root / f"rank_{rank:02d}.json"
        if not path.is_file():
            errors.append(f"missing rank {rank}")
            continue
        value = json.loads(path.read_text())
        ranks.append(value)
        if value.get("status") != "PASS":
            errors.append(f"rank {rank} status={value.get('status')}")
        if value.get("variant") != args.variant or int(value.get("rank", -1)) != rank:
            errors.append(f"rank {rank} identity mismatch")
    modes = {(item.get("timing_mode"), item.get("profile_mode")) for item in ranks}
    if len(modes) != 1:
        errors.append(f"rank mode mismatch: {sorted(modes)}")
    validate_only = bool(ranks) and all(item.get("validate_only") for item in ranks)
    starts = [int(item["timing_start_unix_ns"]) for item in ranks if "timing_start_unix_ns" in item]
    stops = [int(item["timing_stop_unix_ns"]) for item in ranks if "timing_stop_unix_ns" in item]
    walls = [float(item["full5_seconds"]) for item in ranks if "full5_seconds" in item]
    if not validate_only and (len(starts) != 16 or len(stops) != 16 or len(walls) != 16):
        errors.append("incomplete Full5 timing receipts")
    if not validate_only:
        for item in ranks:
            if item.get("updates") != {
                "generator": contract.generator_updates,
                "critic": contract.critic_updates,
                "ema": len(contract.ema_update_steps),
            }:
                errors.append(f"rank {item.get('rank')} update inventory mismatch")
            if int(item.get("transition_count", -1)) != 0:
                errors.append(f"rank {item.get('rank')} performed a transition")
            if int(item.get("model_swap_count", -1)) != 0:
                errors.append(f"rank {item.get('rank')} performed a model swap")
            final = item.get("final", {})
            if int(final.get("generator_logical_version", -1)) != int(
                contract.final_versions["generator"]
            ):
                errors.append(f"rank {item.get('rank')} Generator version mismatch")
            if int(final.get("critic_logical_version", -1)) != int(
                contract.final_versions["critic"]
            ):
                errors.append(f"rank {item.get('rank')} Critic version mismatch")
            if int(final.get("ema_step", -1)) != int(contract.final_versions["ema"]):
                errors.append(f"rank {item.get('rank')} EMA version mismatch")
    timing_mode = bool(ranks and ranks[0].get("timing_mode"))
    profile_mode = bool(ranks and ranks[0].get("profile_mode"))
    summary = {
        "schema_version": "distillschedule.static_fsdp2_full5_summary.v1",
        "status": "PASS" if not errors else "FAIL",
        "variant": args.variant,
        "rank_count": len(ranks),
        "validate_only": validate_only,
        "timing_mode": timing_mode,
        "profile_mode": profile_mode,
        "errors": errors,
        "global_full5_seconds": (
            (max(stops) - min(starts)) / 1e9 if len(starts) == len(stops) == 16 else None
        ),
        "rank_full5_seconds": {
            "count": len(walls),
            "median": statistics.median(walls) if walls else None,
            "max": max(walls) if walls else None,
            "min": min(walls) if walls else None,
        },
        "updates": {
            "generator": contract.generator_updates,
            "critic": contract.critic_updates,
            "ema": len(contract.ema_update_steps),
        },
        "final_versions": dict(contract.final_versions),
        "static_baseline": {
            "world_size": 16,
            "global_batch": 16,
            "both_backbones_fsdp2": True,
            "transition_count": 0,
            "model_swap_count": 0,
            "parameter_streaming": False,
        },
        "claimable_correctness": (
            not errors and not validate_only and not timing_mode and not profile_mode
        ),
        "claimable_formal_timing": (
            not errors and timing_mode and not profile_mode and not validate_only
        ),
        "profiled_timing_is_diagnostic_only": profile_mode,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
