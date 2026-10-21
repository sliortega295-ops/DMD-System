"""Read-only discovery of six protocols and their existing native entrypoints.

The catalog is not a strategy registry, a cost model, or a launcher. In
particular, discovery cannot turn old timing records into executable costs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
from typing import Any

SCHEMA = "distillschedule.workload_catalog.v1"
IDS = tuple(f"W{i}" for i in range(1, 7))


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    path = path or Path(__file__).with_name("catalog.json")
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate catalog field: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError("unsupported workload catalog schema")
    rows = value.get("workloads", [])
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows) or [r.get("id") for r in rows] != list(IDS):
        raise ValueError("catalog must contain W1 through W6 exactly once, in order")
    for row in rows:
        if not row.get("name") or not row.get("protocol"):
            raise ValueError("workload identity and original protocol are required")
        if not re.fullmatch(r"[0-9a-f]{40}", row["integrated_planner_source"]):
            raise ValueError("planner origin requires an exact commit")
        for key in ("measured_source", "published_source"):
            if not re.fullmatch(r"[0-9a-f]{40}", row["h100"][key]):
                raise ValueError("hardware provenance requires exact commits")
        command = row.get("inspect_args", [])
        if not command or any(arg in command for arg in ("--execute", "--execute-native")):
            raise ValueError("discovery must not authorize execution")
    return value


def describe(workload: str, *, hardware: str | None = None) -> dict[str, Any]:
    key = workload.upper()
    rows = load_catalog()["workloads"]
    found = next((r for r in rows if r["id"] == key or r["slug"] == workload.lower()), None)
    if found is None:
        raise ValueError(f"unknown workload {workload!r}; choose {', '.join(IDS)}")
    result = copy.deepcopy(found)
    result["inspection_hardware_scope"] = "a100_native_adapter"
    result["inspection_command"] = (shlex.join(["distillctl", *result["inspect_args"]])
                                    if hardware != "h100" else None)
    if hardware == "h100":
        result["hardware_guide"] = "docs/status/H100_SOURCE_SNAPSHOT_20261019.md"
    result["requested_hardware"] = hardware
    result["discovery_only"] = True
    result["native_execution"] = "NOT_RUN"
    return result


def verify_checkout(root: Path) -> dict[str, Any]:
    root = root.resolve()
    catalog = load_catalog()
    for relative, expected in catalog["evidence_files"].items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"archived source/evidence changed or missing: {relative}")
    for row in catalog["workloads"]:
        for relative in [row["runbook"], row["recipe"], *row["code"], *([row["task"]] if row["task"] else [])]:
            if not (root / relative).is_file():
                raise ValueError(f"{row['id']} local entrypoint missing: {relative}")
    return {"status": "PASS_SOURCE_CATALOG", "workloads": list(IDS),
            "source_index": catalog["source_index"], "gpu_execution": "NOT_RUN"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="distillctl workloads",
        description="Six-protocol discovery. Existing native and legacy commands remain separate.",
        epilog="Other existing commands: native, candidates, calibrate, assemble-calibration, "
               "search, explain, validate, render, run. Use <command> --help. "
               "workloads list --legacy retains the original five-profile inventory.")
    sub = parser.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("list", help="list all six current protocols without model imports")
    listing.add_argument("--json", action="store_true", dest="as_json")
    info = sub.add_parser("describe", help="show source pins and the existing native entrypoint")
    info.add_argument("--workload", required=True)
    info.add_argument("--hardware", choices=("a100", "h100"))
    check = sub.add_parser("verify-catalog", help="verify local sources and archived evidence, no GPUs")
    check.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        if args.action == "list":
            rows = load_catalog()["workloads"]
            if args.as_json:
                print(json.dumps(rows, indent=2, ensure_ascii=False))
            else:
                for row in rows:
                    print(f"{row['id']}\t{row['name']}\t{row['recipe']}")
        elif args.action == "describe":
            print(json.dumps(describe(args.workload, hardware=args.hardware), indent=2, ensure_ascii=False))
        else:
            print(json.dumps(verify_checkout(args.repo_root), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
