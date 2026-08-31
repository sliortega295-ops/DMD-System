#!/usr/bin/env python3
"""CLI entrypoint for the pinned LiveWan baseline/pipeline worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from distillschedule.runtime.livewan_crossscale_worker import run_worker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--plan", required=True, type=Path)
    value.add_argument("--mode", required=True, choices=("baseline", "pipeline"))
    value.add_argument(
        "--run-kind",
        choices=("validate", "capacity", "correctness", "timing", "profile"),
        default=os.environ.get("DISTILLSCHEDULE_RUN_KIND", "validate"),
    )
    value.add_argument("--repo-root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    result = run_worker(
        plan_path=args.plan,
        mode=args.mode,
        run_kind=args.run_kind,
        repo_root=args.repo_root,
    )
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0 or args.run_kind == "validate":
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
