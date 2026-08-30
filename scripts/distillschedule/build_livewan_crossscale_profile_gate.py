#!/usr/bin/env python3
"""Authorize diagnostic profiling without weakening the formal timing gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--baseline-summary", required=True, type=Path)
    value.add_argument("--pipeline-summary", required=True, type=Path)
    value.add_argument("--failed-pair-receipt", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    return value


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    args = parser().parse_args()
    baseline = _load(args.baseline_summary)
    pipeline = _load(args.pipeline_summary)
    pair = _load(args.failed_pair_receipt)
    if baseline.get("status") != "PASS" or baseline.get("execution_mode") != "baseline":
        raise SystemExit("baseline correctness summary is not PASS")
    if pipeline.get("status") != "PASS" or pipeline.get("execution_mode") != "pipeline":
        raise SystemExit("pipeline correctness summary is not PASS")
    if baseline.get("run_kind") != "correctness" or pipeline.get("run_kind") != "correctness":
        raise SystemExit("profile gate requires correctness runs")
    if baseline.get("identity") != pipeline.get("identity"):
        raise SystemExit("baseline/pipeline source-bootstrap-trace identity differs")
    if baseline.get("call_inventory") != pipeline.get("call_inventory"):
        raise SystemExit("baseline/pipeline application call inventory differs")
    heads = {baseline.get("head"), pipeline.get("head")}
    if len(heads) != 1 or None in heads:
        raise SystemExit("baseline/pipeline HEAD differs")
    if pair.get("status") != "FAIL" or pair.get("claimable_for_timing") is not False:
        raise SystemExit("expected a preserved failed paired numerical receipt")
    if set(pair.get("heads", {}).values()) != heads:
        raise SystemExit("failed pair receipt belongs to another HEAD")
    value = {
        "schema_version": 1,
        "status": "PASS_DIAGNOSTIC_ONLY",
        "profile_authorized": True,
        "formal_timing_authorized": False,
        "claim_boundary": (
            "Profiled walls are diagnostic only. The paired numerical receipt is FAIL; "
            "no formal speedup may be claimed."
        ),
        "heads": {"baseline": next(iter(heads)), "pipeline": next(iter(heads))},
        "identity": baseline["identity"],
        "baseline_summary": {"path": str(args.baseline_summary), "sha256": _sha256(args.baseline_summary)},
        "pipeline_summary": {"path": str(args.pipeline_summary), "sha256": _sha256(args.pipeline_summary)},
        "failed_pair_receipt": {
            "path": str(args.failed_pair_receipt),
            "sha256": _sha256(args.failed_pair_receipt),
            "rollout_failure_count": pair.get("rollout", {}).get("failure_count"),
        },
    }
    _write(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
