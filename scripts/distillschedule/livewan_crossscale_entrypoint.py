#!/usr/bin/env python3
"""CLI entrypoint for the pinned LiveWan baseline/pipeline worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from distillschedule.runtime.livewan_crossscale_worker import run_worker


def write_failure_receipt(args, error):
    """Best-effort evidence only after failure; never repair or suppress it."""
    rank = int(os.environ.get("RANK", "0"))
    value = {"status": "FAIL", "rank": rank, "error": repr(error),
             "traceback": traceback.format_exc(), "run_kind": args.run_kind,
             "head": os.environ.get("LIVEWAN_EXPECTED_HEAD"), "frames": []}
    for frame, line in traceback.walk_tb(error.__traceback__):
        fields = {k: frame.f_locals[k] for k in
                  ("rank", "step", "cycle", "cycle_index", "versions")
                  if k in frame.f_locals and isinstance(frame.f_locals[k], (int, str, dict))}
        value["frames"].append({"file": frame.f_code.co_filename,
                                "function": frame.f_code.co_name,
                                "line": line, "context": fields})
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        for key, fn in (("allocated", torch.cuda.memory_allocated),
                        ("reserved", torch.cuda.memory_reserved),
                        ("max_allocated", torch.cuda.max_memory_allocated),
                        ("max_reserved", torch.cuda.max_memory_reserved),
                        ("driver_free_total", torch.cuda.mem_get_info),
                        ("allocator_stats", torch.cuda.memory_stats)):
            try:
                value[key] = fn()
            except Exception as telemetry_error:
                value[key] = {"error": repr(telemetry_error)}
    model_module = sys.modules.get("wan.modules.model")
    if model_module is not None:
        value["rmsnorm"] = getattr(model_module.WanRMSNorm,
                                  "_distillschedule_rmsnorm_receipt", None)
    plan = json.loads(args.plan.read_text())
    path = Path(plan["evidence"]["output_dir"]) / "failures" / f"rank_{rank:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, default=str)
        f.write("\n")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--plan", required=True, type=Path)
    value.add_argument("--mode", required=True, choices=("baseline", "pipeline"))
    value.add_argument(
        "--run-kind",
        choices=(
            "validate", "capacity", "correctness", "timing", "profile",
            "trajectory", "two-cycle-timing", "system-factorial-timing",
        ),
        default=os.environ.get("DISTILLSCHEDULE_RUN_KIND", "validate"),
    )
    value.add_argument("--repo-root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        result = run_worker(
            plan_path=args.plan,
            mode=args.mode,
            run_kind=args.run_kind,
            repo_root=args.repo_root,
        )
    except Exception as error:
        try:
            write_failure_receipt(args, error)
        except Exception as telemetry_error:
            print(f"failure receipt unavailable: {telemetry_error!r}", file=sys.stderr)
        raise
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0 or args.run_kind == "validate":
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
