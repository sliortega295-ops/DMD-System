#!/usr/bin/env python3
"""Validate and summarize all five 16-rank, 50-cycle trajectory pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from distillschedule.evidence.trajectory_suite import (
    build_trajectory_suite_report,
    load_run_matrix_manifest,
    write_trajectory_suite_report,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Consume explicit baseline/candidate rank-JSONL roots from an append-only "
            "manifest and emit one five-workload trajectory matrix."
        )
    )
    value.add_argument("--manifest-jsonl", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing derived report; raw inputs are never changed.",
    )
    return value


def main() -> int:
    args = parser().parse_args()
    records = load_run_matrix_manifest(args.manifest_jsonl)
    report = build_trajectory_suite_report(records)
    outputs = write_trajectory_suite_report(
        report, args.output_dir, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "suite_id": report["suite_id"],
                "outputs": {key: str(path) for key, path in outputs.items()},
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] in {"PASS", "REVIEW", "NOT_RUN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
