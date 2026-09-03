#!/usr/bin/env python3
"""Compare per-rank trajectory state anchors across distributed placements."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from distillschedule.evidence.distributed_state_sketch import (  # noqa: E402
    compare_state_sketch_roots,
    write_state_sketch,
)


def _cycles(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("cycles must be comma-separated integers") from error
    if not result or result[0] <= 0:
        raise argparse.ArgumentTypeError("cycles must contain positive integers")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Resident/FSDP2/HSDP rank sketches by global coordinates and "
            "compute post-run trajectory metrics"
        )
    )
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--required-cycles", default=(1, 10, 25, 50), type=_cycles)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = compare_state_sketch_roots(
        args.reference_root,
        args.candidate_root,
        required_cycles=args.required_cycles,
    )
    artifact_sha256 = write_state_sketch(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "comparison_count": result["comparison_count"],
                "review_count": result["review_count"],
                "output": str(args.output),
                "artifact_sha256": artifact_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    # REVIEW is a completed post-run numerical result, not an execution error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
