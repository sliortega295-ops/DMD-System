#!/usr/bin/env python3
"""Print the bounded initial primitive matrix required by planner v0."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from distillschedule.planner.calibration import calibration_coverage, initial_calibration_matrix
from distillschedule.planner.profile_db import ProfileDB
from distillschedule.planner.schema import load_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--output")
    args = parser.parse_args()

    matrix = initial_calibration_matrix()
    result = {
        "schema_version": "distillschedule.planner_calibration_matrix.v0",
        "primitive_count": len(matrix),
        "primitives": [
            {**dataclasses.asdict(key), "offload_blocks": list(key.offload_blocks), "id": key.id}
            for key in matrix
        ],
        "non_primitive_required_records": [
            "exact model and block byte inventory",
            "optimizer/EMA byte inventory",
            "saved-tensor activation census and liveness",
            "pinned/pageable H2D and D2H calibration",
            "intra-node and inter-node collective bandwidth and latency",
            "transition subphase timing and transfer bytes",
            "GPU HBM and node cgroup stable/transient peaks",
            "one Full5 validation anchor per backend family",
        ],
    }
    if args.profile:
        result["coverage"] = calibration_coverage(ProfileDB(load_profile(args.profile)))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
