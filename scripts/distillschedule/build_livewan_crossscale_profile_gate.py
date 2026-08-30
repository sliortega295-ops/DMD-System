#!/usr/bin/env python3
"""Authorize diagnostic profiling without weakening the formal timing gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--baseline-summary", required=True, type=Path)
    value.add_argument("--pipeline-summary", required=True, type=Path)
    value.add_argument("--failed-pair-receipt", required=True, type=Path)
    value.add_argument("--profile-head")
    value.add_argument("--repo-root", type=Path)
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


def _profile_source_identity(
    *, workload_head: str, profile_head: str, repo_root: Path | None
) -> dict:
    if profile_head == workload_head:
        return {
            "workload_head": workload_head,
            "profile_head": profile_head,
            "instrumentation_only": True,
            "diff_paths": [],
            "verification": "exact_correctness_head",
        }
    if repo_root is None:
        raise SystemExit("a descendant --profile-head requires --repo-root")
    root = repo_root.resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != profile_head:
        raise SystemExit("profile repository is not at --profile-head")
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise SystemExit("profile repository is dirty")
    if subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", workload_head, profile_head],
        check=False,
    ).returncode != 0:
        raise SystemExit("profile HEAD is not a descendant of the correctness workload HEAD")
    paths = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--name-only", workload_head, profile_head],
        text=True,
    ).splitlines()
    allowed = {
        "scripts/distillschedule/build_livewan_crossscale_profile_gate.py",
        "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh",
        "scripts/distillschedule/run_livewan_crossscale_world16.py",
        "src/distillschedule/runtime/livewan_crossscale_worker.py",
        "tests/test_refactor_livewan_crossscale_profile.py",
    }
    unexpected = sorted(set(paths) - allowed)
    if unexpected:
        raise SystemExit("profile descendant changes workload files: " + ", ".join(unexpected))
    diff = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--binary", workload_head, profile_head]
    )
    return {
        "workload_head": workload_head,
        "profile_head": profile_head,
        "instrumentation_only": True,
        "diff_paths": paths,
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "verification": "clean_descendant_allowlisted_diff",
    }


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
    workload_head = next(iter(heads))
    profile_head = args.profile_head or workload_head
    if len(profile_head) != 40:
        raise SystemExit("--profile-head must be a full Git SHA")
    profile_source = _profile_source_identity(
        workload_head=workload_head,
        profile_head=profile_head,
        repo_root=args.repo_root,
    )
    value = {
        "schema_version": 1,
        "status": "PASS_DIAGNOSTIC_ONLY",
        "profile_authorized": True,
        "formal_timing_authorized": False,
        "claim_boundary": (
            "Profiled walls are diagnostic only. The paired numerical receipt is FAIL; "
            "no formal speedup may be claimed."
        ),
        "heads": {
            "baseline": workload_head,
            "pipeline": workload_head,
            "profile": profile_head,
        },
        "profile_source": profile_source,
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
