#!/usr/bin/env python3
"""Append one immutable status record to a 50-cycle suite run matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from distillschedule.evidence.trajectory_suite import (
    AppendOnlyTrajectoryRunMatrixWriter,
    TrajectoryRunMatrixRecord,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Append a PASS/FAIL/NOT_RUN row without rewriting prior suite evidence."
    )
    value.add_argument("--manifest-jsonl", type=Path, required=True)
    value.add_argument("--record-json", type=Path, required=True)
    value.add_argument("--suite-id", required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    payload = json.loads(args.record_json.read_text(encoding="utf-8"))
    record = TrajectoryRunMatrixRecord.from_mapping(payload)
    AppendOnlyTrajectoryRunMatrixWriter(
        args.manifest_jsonl, suite_id=args.suite_id
    ).append(record)
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest_jsonl": str(args.manifest_jsonl),
                "suite_id": record.suite_id,
                "workload_id": record.workload_id,
                "attempt": record.attempt,
                "execution_status": record.execution_status,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
