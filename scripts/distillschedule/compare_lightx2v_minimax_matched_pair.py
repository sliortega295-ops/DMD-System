#!/usr/bin/env python3
"""Compare one native and one DistillSchedule MiniMax-H3 matched pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--native", type=Path, required=True)
    value.add_argument("--distillschedule", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def load(path: Path):
    return json.loads(path.read_text())


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    args = parser().parse_args()
    native = load(args.native)
    candidate = load(args.distillschedule)
    errors: list[str] = []
    if native.get("status") != "PASS" or native.get("mode") != "native":
        errors.append("native summary is not a PASS native run")
    if candidate.get("status") != "PASS" or candidate.get("mode") != "distillschedule":
        errors.append("candidate summary is not a PASS distillschedule run")

    if native.get("application_inventory") != candidate.get("application_inventory"):
        errors.append("DiT forward/backward inventory differs")
    if native.get("sample_sequences_by_rank") != candidate.get("sample_sequences_by_rank"):
        errors.append("rank-qualified six-sample sequence differs")
    if native.get("state_hashes_by_rank") != candidate.get("state_hashes_by_rank"):
        errors.append("final Student/Fake parameter or optimizer shard hashes differ")

    native_ids = native.get("identities", {})
    candidate_ids = candidate.get("identities", {})
    if native_ids != candidate_ids:
        errors.append("model/config/data/seed identities differ")

    native_seconds = float(native["global_cycle_seconds"])
    candidate_seconds = float(candidate["global_cycle_seconds"])
    result = {
        "schema_version": "distillschedule.lightx2v_minimax_matched_pair.v1",
        "status": "PASS" if not errors else "NOT_COMPARABLE",
        "native_seconds": native_seconds,
        "distillschedule_seconds": candidate_seconds,
        "speedup": native_seconds / candidate_seconds if not errors else None,
        "wall_reduction_fraction": (native_seconds - candidate_seconds) / native_seconds if not errors else None,
        "native_max_allocated_bytes": native.get("max_allocated_bytes"),
        "distillschedule_max_allocated_bytes": candidate.get("max_allocated_bytes"),
        "inventory": native.get("application_inventory"),
        "native_phase_order": native.get("phase_order"),
        "distillschedule_phase_order": candidate.get("phase_order"),
        "equivalence_gates": {
            "same_application_dit_inventory": native.get("application_inventory") == candidate.get("application_inventory"),
            "same_rank_sample_sequence": native.get("sample_sequences_by_rank") == candidate.get("sample_sequences_by_rank"),
            "same_final_training_state_hashes": native.get("state_hashes_by_rank") == candidate.get("state_hashes_by_rank"),
            "same_model_config_data_seed_identity": native_ids == candidate_ids,
        },
        "errors": errors,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
