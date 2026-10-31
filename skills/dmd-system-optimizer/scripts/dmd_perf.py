#!/usr/bin/env python3
"""Small read-only inspection and arithmetic helpers; no campaign or admission gate.

Uses Python 3.10+ standard library. Does not launch workloads, reserve budgets,
certify correctness, or promote candidates. Use the project's own runner/checker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

MAX_JSON = 8 * 1024 * 1024
MAX_EVIDENCE = 32 * 1024 * 1024

class InputError(ValueError):
    pass

def require(ok: bool, message: str) -> None:
    if not ok:
        raise InputError(message)

def number(value: Any, name: str, *, minimum: float = 0, positive: bool = False) -> float:
    require(type(value) in (int, float), f"{name}: finite number required (not bool/null)")
    value = float(value)
    require(math.isfinite(value), f"{name}: non-finite number")
    require(value > minimum if positive else value >= minimum, f"{name}: below allowed range")
    return value

def text(value: Any, name: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{name}: nonblank text required")
    return value

def unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result

def load(path: Path) -> Any:
    require(path.is_file(), f"not a regular file: {path}")
    require(path.stat().st_size <= MAX_JSON, f"JSON exceeds {MAX_JSON} bytes")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs,
                       parse_constant=lambda x: (_ for _ in ()).throw(InputError(f"invalid constant: {x}")))
    # Exponent overflow such as 1e999 is accepted by Python's parser without this check.
    def check(x: Any) -> None:
        if isinstance(x, float):
            require(math.isfinite(x), "non-finite JSON number")
        elif isinstance(x, dict):
            for v in x.values():
                check(v)
        elif isinstance(x, list):
            for v in x:
                check(v)
    check(value)
    require(isinstance(value, dict), "top-level JSON must be an object")
    return value

def sha_file(path: Path, limit: int = MAX_EVIDENCE) -> str:
    require(path.is_file(), f"not a regular evidence file: {path}")
    require(path.stat().st_size <= limit, f"use a small evidence report, not raw large data: {path}")
    with path.open("rb") as f:
        h = hashlib.sha256()
        size = 0
        for block in iter(lambda: f.read(1024 * 1024), b""):
            size += len(block)
            require(size <= limit, f"evidence grew beyond limit: {path}")
            h.update(block)
    return h.hexdigest()

def target(baseline: Any, control: Any, speedup: Any) -> dict[str, float]:
    b = number(baseline, "baseline_seconds", positive=True)
    c = number(control, "control_seconds", positive=True)
    s = number(speedup, "target_speedup", minimum=1)
    t = b / s
    return {"baseline_seconds": b, "control_seconds": c, "target_seconds": t,
            "target_speedup": s, "achieved_speedup": b / c,
            "remaining_saving_seconds": max(0, c - t),
            "required_speedup_from_control": c / t,
            "target_latency_reduction_percent": 100 * (1 - 1 / s)}

def check_evidence(items: Any, base: Path, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(items, list) or not items:
        return [f"{label}: evidence required"]
    for i, item in enumerate(items):
        try:
            require(isinstance(item, dict), "evidence row must be object")
            path = Path(text(item.get("path"), "evidence.path"))
            if not path.is_absolute():
                path = base / path
            expected = text(item.get("sha256"), "evidence.sha256").lower()
            require(len(expected) == 64 and all(x in "0123456789abcdef" for x in expected),
                    "invalid sha256")
            text(item.get("locator"), "evidence.locator")
            require(sha_file(path) == expected, "evidence digest mismatch")
        except (InputError, OSError) as exc:
            errors.append(f"{label}[{i}]: {exc}")
    return errors

def git_read(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], check=False, capture_output=True,
                       text=True, timeout=20)
    require(p.returncode == 0, f"read-only git command failed: {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout.strip()

def inspect(repo: Path) -> dict[str, Any]:
    """Inspect a checkout or source archive without loading historical policies."""
    repo = repo.resolve()
    require(repo.is_dir(), "--repo must be an existing source directory")
    head = status = None
    warnings = []
    try:
        top = Path(git_read(repo, "rev-parse", "--show-toplevel")).resolve()
    except (InputError, OSError, subprocess.SubprocessError):
        warnings.append("No Git metadata; source archive inspection only. Record the source version separately.")
    else:
        require(top == repo, "--repo must be the Git checkout root")
        try:
            head = git_read(repo, "rev-parse", "HEAD")
            status = git_read(repo, "status", "--porcelain", "--untracked-files=normal")
        except (InputError, OSError, subprocess.SubprocessError):
            warnings.append("Git metadata is incomplete; do not assume a clean committed source.")
    native = (repo / "src/distillschedule").is_dir()
    anchors = ["AGENTS.md", "CONTRIBUTING.md", "pyproject.toml", "docs/guides/support_matrix.md",
               "src/distillschedule/workloads/catalog.json", "recipes/README.md",
               "docs/architecture/framework.md", "docs/guides/optimizer_skill.md"]
    return {"repo": str(repo), "head": head, "git_status": status,
            "clean_status": not status if status is not None else None,
            "mode": "dmd-system" if native else "external-source",
            "anchors": {p: sha_file(repo/p) if (repo/p).is_file() else None for p in anchors},
            "warnings": warnings, "gpu_validation_performed": False,
            "warning": "Read-only source inspection, not numerical validation. Use current project instructions "
                       "and recipes, not a historical best-known policy."}


def dag_model(data: dict[str, Any]) -> dict[str, Any]:
    nodes = data.get("nodes")
    require(isinstance(nodes, list) and nodes, "DAG requires nonempty nodes")
    nd: dict[str, Any] = {}
    require(all(isinstance(n, dict) for n in nodes), "every node must be an object")
    for n in nodes:
        key = text(n.get("id"), "node.id")
        require(key not in nd, "duplicate node id")
        number(n.get("seconds"), f"{key}.seconds")
        require(isinstance(n.get("depends_on", []), list) and all(isinstance(d, str) for d in n.get("depends_on", [])), "depends_on must be a string array")
        nd[key] = n
    changes = data.get("candidate_seconds", {})
    require(isinstance(changes, dict) and set(changes) <= set(nd), "unknown candidate node")
    for key, value in changes.items():
        number(value, f"candidate.{key}")
    def solve(overrides: dict[str, Any]) -> tuple[float, list[str]]:
        memo: dict[str, tuple[float, list[str]]] = {}
        visiting: set[str] = set()
        def end(key: str) -> tuple[float, list[str]]:
            require(key in nd, f"unknown dependency: {key}")
            require(key not in visiting, "cyclic DAG")
            if key in memo:
                return memo[key]
            visiting.add(key)
            choices = [end(d) for d in nd[key].get("depends_on", [])]
            prior, path = max(choices, key=lambda x: x[0]) if choices else (0., [])
            result = (prior + overrides.get(key, nd[key]["seconds"]), path + [key])
            visiting.remove(key)
            memo[key] = result
            return result
        return max((end(k) for k in nd), key=lambda x: x[0])
    b, bp = solve({})
    c, cp = solve(changes)
    return {"base_model_seconds": b, "candidate_model_seconds": c, "model_saving_seconds": b-c,
            "base_critical_path": bp, "candidate_critical_path": cp,
            "assumptions": data.get("assumptions", "not supplied"),
            "classification": "user-supplied fixed-order DAG calculation; NOT measurement or an automatic physical bound",
            "warning": "Capacity, contention, dynamic arrivals and unlisted resource serialization are NOT modeled."}

def compare(data: dict[str, Any]) -> dict[str, Any]:
    for key in ("timer_scope", "semantic_contract", "hardware_id"):
        text(data.get(key), key)
    require(data.get("unprofiled") is True, "comparison input must explicitly state unprofiled")
    a, b = data.get("control_seconds"), data.get("candidate_seconds")
    require(isinstance(a, list) and isinstance(b, list) and a and b, "nonempty independent-run arrays required")
    a = [number(x, "control sample", positive=True) for x in a]
    b = [number(x, "candidate sample", positive=True) for x in b]
    ma, mb = statistics.median(a), statistics.median(b)
    result = {"control_n": len(a), "candidate_n": len(b), "control_median_seconds": ma,
              "candidate_median_seconds": mb, "median_ratio_speedup": ma/mb,
              "latency_reduction_percent": 100*(1-mb/ma),
              "control_range_seconds": [min(a), max(a)], "candidate_range_seconds": [min(b), max(b)],
              "formal_acceptance": "NOT_DECIDED",
              "warning": "Descriptive statistics only. No significance test or numerical acceptance. "
                         "Inputs must be independent complete runs, not correlated steps or cherry-picked timings."}
    if data.get("paired") is True:
        require(len(a) == len(b), "paired sample counts differ")
        result["paired_saving_seconds"] = [x-y for x, y in zip(a, b)]
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("inspect", help="read local Git/source metadata; no GPU or network")
    s.add_argument("--repo", type=Path, required=True)
    s = sub.add_parser("target", help="calculate target and remaining gap")
    s.add_argument("--baseline-seconds", type=float, required=True)
    s.add_argument("--control-seconds", type=float)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--speedup", type=float)
    g.add_argument("--latency-reduction", type=float)
    for command in ("dag", "compare"):
        sub.add_parser(command).add_argument("--input", type=Path, required=True)
    a = p.parse_args()
    try:
        if a.command == "inspect":
            out = inspect(a.repo)
        elif a.command == "target":
            speedup = a.speedup
            if a.latency_reduction is not None:
                r = number(a.latency_reduction, "latency_reduction")
                require(r < 1, "latency_reduction must be in [0,1)")
                speedup = 1/(1-r)
            out = target(a.baseline_seconds, a.control_seconds if a.control_seconds is not None else a.baseline_seconds, speedup)
        elif a.command == "dag":
            out = dag_model(load(a.input))
        else:
            out = compare(load(a.input))
        print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (InputError, ValueError, TypeError, KeyError, AttributeError, OSError,
            OverflowError, RecursionError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
