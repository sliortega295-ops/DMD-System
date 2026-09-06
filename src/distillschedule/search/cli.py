"""CLI extension of distillctl workloads, not a second workload launcher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from distillschedule.workloads.best_known.registry import (
    _object, fingerprint, git_head, load_assets, load_profile, repository_root, write_new_json,
)
from .contracts import PILOTS, require
from .engine import SearchSession
from .execution import NativeTrialRunner, resolve_candidate
from .profile import feature_weights, parse_profile, validate_profile
from .registry import catalog, candidates, search_identity


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="distillctl workloads")
    subs = parser.add_subparsers(dest="action", required=True)
    p = subs.add_parser("profile", help="Import exact per-rank cycle2 SQLite evidence; no capture")
    p.add_argument("--input", type=Path, required=True, help="profile_input.v1 ownership/NVTX manifest")
    p.add_argument("--output", type=Path, required=True)
    p = subs.add_parser("search", help="Bounded registered strategy search, no workload math changes")
    p.add_argument("--workload", required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--assets-config", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--run-id", default="search")
    p.add_argument("--max-trials", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--historical-profile-prior", action="store_true",
                   help="Explicitly allow an old source/assets/cycle profile as ranking hints only")
    args = parser.parse_args(argv)
    try:
        if args.action == "profile":
            result = parse_profile(_object(args.input))
            write_new_json(args.output, result)
        elif args.dry_run:
            require(1 <= args.max_trials <= 8, "v1 budget is 1..8")
            profile = _object(args.profile) if args.profile else None
            if profile is not None:
                validate_profile(profile)
            entry = next((x for x in catalog() if x["workload"] == args.workload), None)
            require(entry is not None, "unknown workload")
            plans = [resolve_candidate(args.workload, c.id) for c in candidates(args.workload)]
            result = {"status": "PASS_CPU_PLAN_ONLY", "gpu": "NOT_RUN", "registration": entry,
                      "source_head": git_head(repository_root()), "max_candidate_trials": args.max_trials,
                      "cycles": 2, "measured_cycle": 2, "baseline_and_confirmation_runs": 2,
                      "profile_priorities": feature_weights(profile) if profile else "PROFILE_REQUIRED_FOR_RANKING",
                      "plans": plans}
            if args.output:
                write_new_json(args.output, result)
        else:
            require(args.workload in PILOTS, "CPU-only registration; GPU search NOT_RUN for this workload")
            require(all((args.profile, args.assets_config, args.output)),
                    "search execution requires profile, assets-config and output")
            profile = _object(args.profile)
            assets = load_assets(args.assets_config, load_profile(args.workload))
            identity = search_identity(args.workload, source_head=git_head(repository_root()),
                                       assets=assets["assets"], hardware=profile["hardware"])
            session = SearchSession(workload=args.workload, identity=identity, profile=profile,
                                    output=args.output, max_trials=args.max_trials, resume=args.resume,
                                    historical_prior=args.historical_profile_prior)
            controls = [c for c in candidates(args.workload) if c.executable and c.id != "fsdp-reference"]
            # Bootstrap the first implementation edge solely to establish the
            # native matched numerical gate before measuring the baseline.
            peer = next((c for c in controls if c.parent == "fsdp-reference"), controls[0])
            runner = NativeTrialRunner(workload=args.workload, identity=identity, assets=assets,
                                       output=args.output, run_prefix=args.run_id, bootstrap_peer=peer.id)
            result = session.run(runner)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, sqlite3.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
