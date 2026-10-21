"""Read-only discovery of six protocols and their existing native entrypoints.

The catalog is not a strategy registry, a cost model, or a launcher. In
particular, discovery cannot turn old timing records into executable costs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
from typing import Any

SCHEMA = "distillschedule.workload_catalog.v1"
IDS = tuple(f"W{i}" for i in range(1, 7))


def _relative_path(value: Any) -> str:
    """Catalog paths are checkout-relative, not locations to follow off-tree."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("catalog paths must be nonempty relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in value or value != path.as_posix() or value == ".":
        raise ValueError(f"unsafe catalog path: {value!r}")
    return value


def _checkout_path(root: Path, relative: str) -> Path:
    path = root / _relative_path(relative)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"catalog path resolves outside checkout: {relative}")
    return path


def _exact_digest(value: Any, length: int, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, value):
        raise ValueError(f"{label} requires an exact lowercase {length}-character digest")


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
    _exact_digest(value.get("source_index"), 40, "source index")
    evidence = value.get("evidence_files")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("catalog requires source-bound evidence files")
    for relative, expected in evidence.items():
        _relative_path(relative)
        _exact_digest(expected, 64, "evidence file")
    slugs: set[str] = set()
    for row in rows:
        if any(not isinstance(row.get(key), str) or not row[key] for key in ("name", "protocol")):
            raise ValueError("workload identity and original protocol are required")
        slug = row.get("slug")
        if not isinstance(slug, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", slug) or slug in slugs:
            raise ValueError("catalog slugs must be valid and unique")
        slugs.add(slug)
        _exact_digest(row.get("integrated_planner_source"), 40, "planner origin")
        hardware = row.get("h100")
        if not isinstance(hardware, dict):
            raise ValueError("hardware provenance is required")
        for key in ("measured_source", "published_source"):
            _exact_digest(hardware.get(key), 40, "hardware provenance")
        code = row.get("code")
        if not isinstance(code, list) or not code:
            raise ValueError("each workload needs existing native code entrypoints")
        for relative in [row.get("runbook"), row.get("recipe"), *code]:
            _relative_path(relative)
        if row.get("task") is not None:
            _relative_path(row["task"])
        command = row.get("inspect_args")
        if (not isinstance(command, list) or not command
                or any(not isinstance(arg, str) or not arg for arg in command)
                or any(arg.split("=", 1)[0] in {"--execute", "--execute-native"} for arg in command)
                or not (command[:2] == ["workloads", "candidates"]
                        or command[:3] == ["workloads", "native", "validate"])):
            raise ValueError("discovery must not authorize execution; only read-only inspection commands are allowed")
    return value


def describe(workload: str, *, hardware: str | None = None) -> dict[str, Any]:
    if hardware not in {None, "a100", "h100"}:
        raise ValueError("hardware must be a100 or h100")
    key = workload.upper()
    rows = load_catalog()["workloads"]
    found = next((r for r in rows if r["id"] == key or r["slug"] == workload.lower()), None)
    if found is None:
        raise ValueError(f"unknown workload {workload!r}; choose {', '.join(IDS)}")
    result = copy.deepcopy(found)
    result["inspection_hardware_scope"] = (
        "h100_separately_pinned_source" if hardware == "h100" else "a100_native_adapter"
    )
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
        path = _checkout_path(root, relative)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"archived source/evidence changed or missing: {relative}")
    for row in catalog["workloads"]:
        for relative in [row["runbook"], row["recipe"], *row["code"], *([row["task"]] if row["task"] else [])]:
            if not _checkout_path(root, relative).is_file():
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
