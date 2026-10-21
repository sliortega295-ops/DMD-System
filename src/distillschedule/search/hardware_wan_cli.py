"""Local receipt-host CLI; remote transport is owned by the outer controller."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from distillschedule.workloads.best_known.registry import _object
from .contracts import require
from . import hardware_wan_dispatch as dispatch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="distillctl workloads hardware")
    subs = parser.add_subparsers(dest="action", required=True)
    inspect = subs.add_parser("inspect", help="Verify source/manifests without importing Torch or querying GPUs")
    inspect.add_argument("--deployment", type=Path, required=True)
    calibration = subs.add_parser("calibrate", help="Fresh serial correctness, native comparison, then timing")
    calibration.add_argument("--deployment", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--run-prefix", required=True)
    calibration.add_argument("--placements", nargs="+", choices=dispatch.PLACEMENTS,
                             default=list(dispatch.DEFAULT_PLACEMENTS))
    calibration.add_argument("--reference", choices=dispatch.PLACEMENTS, default=dispatch.PLACEMENTS[0])
    search = subs.add_parser("search", help="Use calibrated costs and the public hierarchical planner")
    search.add_argument("--deployment", type=Path, required=True)
    search.add_argument("--observations", type=Path, required=True,
                        help="Calibration session containing its content-hashed observations")
    search.add_argument("--output", type=Path, required=True)
    search.add_argument("--run-id", required=True, help="Fresh run id saved in the selected plan")
    search.add_argument("--max-proposals", type=int, default=10000)
    search.add_argument("--search-timeout", type=float, default=600)
    search.add_argument("--beam-width", type=int, default=0)
    run = subs.add_parser("run", help="Validate a saved selection and execute its actual native placement")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--deployment", type=Path)
    run.add_argument("--output", type=Path, required=True)
    for command in (calibration, run):
        command.add_argument("--timeout-seconds", type=float, default=1800)
        command.add_argument("--inherited-locks", action="store_true",
                             help="Explicitly inherit already-held flock descriptors 7, 8, and 9")
    args = parser.parse_args(argv)
    try:
        deployment = _object(args.deployment) if args.deployment else None
        if args.action == "inspect":
            result = dispatch.inspect_deployment(deployment)
        elif args.action == "calibrate":
            result = dispatch.calibrate(deployment, output_dir=args.output, run_prefix=args.run_prefix,
                placements=tuple(args.placements), reference=args.reference,
                timeout_seconds=args.timeout_seconds, inherited_locks=args.inherited_locks)
        elif args.action == "search":
            session = _object(args.observations)
            require(session.get("deployment_sha256") == dispatch.fingerprint(deployment),
                    "calibration session belongs to another deployment")
            require(session.get("evidence_scope") == deployment.get("evidence_scope"),
                    "calibration session evidence scope changed")
            result = dispatch.search(deployment, session["observations"], output_dir=args.output,
                run_id=args.run_id, max_proposals=args.max_proposals,
                timeout_seconds=args.search_timeout, beam_width=args.beam_width)
        else:
            result = dispatch.run_selected(_object(args.plan), output_dir=args.output,
                deployment=deployment, timeout_seconds=args.timeout_seconds,
                inherited_locks=args.inherited_locks)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
