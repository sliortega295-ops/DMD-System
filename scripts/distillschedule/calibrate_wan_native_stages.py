#!/usr/bin/env python3
"""Collect ONE source-bound native implementation; default is source-only plan.

Explicit --execute uses current-source native correctness + one two-cycle
diagnostic calibration. It does not run the seven-arm matrix or formal timing.
Large native trace/state artifacts stay remote, using the existing dispatcher.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from distillschedule.search.contracts import require
from distillschedule.search.execution import NativeTrialRunner, resolve_candidate
from distillschedule.search.registry import get_candidate
from distillschedule.search.runtime_identity import observed_binding
from distillschedule.search.wan_implementation import WAN, PRECISION, WanImplementation
from distillschedule.workloads.best_known.registry import git_head, fingerprint, write_new_json


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--implementation", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets-config", type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    head = git_head(ROOT)
    require(head == args.expected_head, "calibration source HEAD changed")
    impl = WanImplementation.parse(args.implementation)
    plan = resolve_candidate(WAN, impl.id, mode="calibration", run_id=args.run_id)
    target = args.output.resolve()
    require(target != ROOT and ROOT not in target.parents,
            "controller evidence must be outside the source checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    write_new_json(args.output / "intent.json", dict(status="NOT_RUN", source_head=head,
        implementation=impl.assignment, source_only_plan=plan,
        native_calibration_runs=1, cycles=2, formal_timing_attempts=0,
        prerequisite="CURRENT_HEAD_MATCHED_CORRECTNESS_REQUIRED; NOT_HISTORICAL_GATES",
        calibration_only=True, formal_timing_eligible=False))
    if not args.execute:
        print(json.dumps({"status": "NOT_RUN", "output": str(args.output)}))
        return 0
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
            "calibration requires a clean committed checkout")
    require(args.assets_config is not None, "--execute requires assets-config")
    assets = json.loads(args.assets_config.read_text())
    identity = dict(source_head=head, asset_fingerprint=fingerprint(assets["assets"]),
        precision_fingerprint=PRECISION.digest, semantic_fingerprint=plan["workload_semantics_hash"])
    reference = WanImplementation("fsdp", 512, False).id
    runner = NativeTrialRunner(workload=WAN, identity=identity, assets=assets, output=args.output,
        run_prefix=args.run_id, reference_id=reference,
        bootstrap_peer=impl.id if impl.id != reference else WanImplementation("resident", 2048, True).id)
    try:
        probe = runner._rpc("resolve", workload=WAN, candidate_id=impl.id, assets=assets,
                            run_id=args.run_id + "-identity", mode="correctness")
        nodes = [runner._rpc("runtime_identity", _node_rank=i, node_rank=i, receipt=probe) for i in (0, 1)]
        identity.update(observed_binding(nodes, source_head=head))
        write_new_json(args.output / "observed-binding.json", {"binding": identity, "nodes": nodes})
        result = runner.calibrate(get_candidate(WAN, impl.id), directory=args.output / "calibration")
        print(json.dumps(result, indent=2))
        return 0
    except BaseException as error:
        write_new_json(args.output / "failure.json", dict(status="FAIL", error_type=type(error).__name__,
            error=str(error), source_head=head, run_id=args.run_id,
            formal_timing_eligible=False, next_action="PRESERVE_FAILED_RUN; NO_AUTOMATIC_RETRY"))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
