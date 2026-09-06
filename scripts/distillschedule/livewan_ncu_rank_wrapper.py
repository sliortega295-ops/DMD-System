#!/usr/bin/env python3
"""Wrap exactly one torchrun worker in Nsight Compute.

The wrapper is intentionally rank-selective: every non-target worker execs the
original training entry point directly, while one preregistered worker execs it
under ``ncu``.  It adds no synchronization or workload-side instrumentation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--target-script", required=True, type=Path)
    value.add_argument("--ncu-bin", required=True, type=Path)
    value.add_argument("--output-base", required=True, type=Path)
    value.add_argument("--manifest", required=True, type=Path)
    value.add_argument("--target-node-rank", required=True, type=int, choices=(0, 1))
    value.add_argument("--target-local-rank", required=True, type=int, choices=range(8))
    value.add_argument("--kernel-regex", required=True)
    value.add_argument("--launch-skip-before-match", required=True, type=int)
    value.add_argument("--launch-count", type=int, default=1)
    value.add_argument("--metric-set", default="full")
    value.add_argument("target_args", nargs=argparse.REMAINDER)
    return value


def _rank_identity(environ: Mapping[str, str]) -> tuple[int, int, int]:
    try:
        return (
            int(environ["DISTILLSCHEDULE_NODE_RANK"]),
            int(environ["LOCAL_RANK"]),
            int(environ["RANK"]),
        )
    except (KeyError, ValueError) as error:
        raise RuntimeError("rank-selective NCU wrapper requires torchrun rank identity") from error


def _target_args(args: argparse.Namespace) -> list[str]:
    values = list(args.target_args)
    if values[:1] == ["--"]:
        values = values[1:]
    return values


def build_exec_command(
    args: argparse.Namespace,
    environ: Mapping[str, str],
) -> tuple[str, list[str], bool]:
    node_rank, local_rank, _ = _rank_identity(environ)
    target = node_rank == args.target_node_rank and local_rank == args.target_local_rank
    application = [sys.executable, str(args.target_script), *_target_args(args)]
    if not target:
        return sys.executable, application, False
    command = [
        str(args.ncu_bin),
        "--set", args.metric_set,
        "--target-processes", "application-only",
        "--kernel-name", f"regex:{args.kernel_regex}",
        "--launch-skip-before-match", str(args.launch_skip_before_match),
        "--launch-count", str(args.launch_count),
        "--export", str(args.output_base),
        *application,
    ]
    return str(args.ncu_bin), command, True


def _write_manifest(
    destination: Path,
    *,
    args: argparse.Namespace,
    command: Sequence[str],
    environ: Mapping[str, str],
) -> None:
    node_rank, local_rank, global_rank = _rank_identity(environ)
    value = {
        "schema_version": 1,
        "status": "STARTED",
        "diagnostic_only": True,
        "formal_timing_eligible": False,
        "node_rank": node_rank,
        "local_rank": local_rank,
        "global_rank": global_rank,
        "kernel_regex": args.kernel_regex,
        "launch_skip_before_match": args.launch_skip_before_match,
        "launch_count": args.launch_count,
        "metric_set": args.metric_set,
        "output_base": str(args.output_base),
        "target_script": str(args.target_script),
        "command": list(command),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)


def main() -> int:
    args = parser().parse_args()
    if args.launch_skip_before_match < 0 or args.launch_count != 1:
        raise ValueError("bounded NCU capture requires non-negative skip and launch-count=1")
    executable, command, target = build_exec_command(args, os.environ)
    if target:
        if not args.ncu_bin.is_file() or not os.access(args.ncu_bin, os.X_OK):
            raise FileNotFoundError(args.ncu_bin)
        if not args.target_script.is_file():
            raise FileNotFoundError(args.target_script)
        report = Path(str(args.output_base) + ".ncu-rep")
        for path in (args.output_base, report, args.manifest):
            if path.exists():
                raise FileExistsError(path)
        args.output_base.parent.mkdir(parents=True, exist_ok=True)
        _write_manifest(args.manifest, args=args, command=command, environ=os.environ)
        os.execvpe(executable, command, dict(os.environ))
    os.execv(executable, command)
    raise AssertionError("exec returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
