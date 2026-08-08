#!/usr/bin/env python3
"""Run matched, repeated Route-A/Route-B formal timing pairs.

The driver alternates launch order, gives every trial an immutable run ID and
refuses to compute a speedup from partial, profiled, OOM or contract-drifted runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "distillschedule.paired_formal_timing.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--run-id", required=True)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument("--workload-contract", required=True, type=Path)
    result.add_argument("--routea-command-template", required=True)
    result.add_argument("--routeb-command-template", required=True)
    result.add_argument("--trials", type=int, default=3)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def command(template: str, *, run_id: str, result_path: Path) -> list[str]:
    return shlex.split(template.format(run_id=run_id, result_path=str(result_path)))


def workload_digest(contract: Mapping[str, Any]) -> str:
    value = contract.get("workload_contract_digest") or contract.get("shared_digest")
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError("workload contract lacks a 64-character digest")
    return value


def extract_window(result: Mapping[str, Any]) -> float:
    timing = result.get("timing")
    if not isinstance(timing, Mapping):
        raise RuntimeError("formal result lacks timing object")
    value = timing.get("global_training_window_seconds")
    if value is None:
        raise RuntimeError("formal result lacks global_training_window_seconds")
    window = float(value)
    if not math.isfinite(window) or window <= 0:
        raise RuntimeError("formal timing window is invalid")
    return window


def validate_result(path: Path, expected_digest: str) -> dict[str, Any]:
    result = load_json(path)
    if result.get("status") != "PASS" or result.get("claimable") is not True:
        raise RuntimeError(f"{path}: result is not claimable PASS")
    if result.get("profiled") is True or result.get("diagnostic_only") is True:
        raise RuntimeError(f"{path}: profiled/diagnostic run cannot enter formal timing")
    observed = result.get("workload_contract_digest")
    if observed is None and isinstance(result.get("evidence_contract"), Mapping):
        observed = result["evidence_contract"].get("shared_digest")
    if observed != expected_digest:
        raise RuntimeError(f"{path}: workload contract drift")
    result = dict(result)
    result["formal_window_seconds"] = extract_window(result)
    return result


def run_one(
    *,
    family: str,
    pair_index: int,
    root: Path,
    parent_id: str,
    template: str,
    expected_digest: str,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    run_id = f"{parent_id}-pair{pair_index:02d}-{family.lower()}"
    run_dir = root / "trials" / run_id
    result_path = run_dir / "result.json"
    command_value = command(template, run_id=run_id, result_path=result_path)
    if result_path.exists():
        existing = validate_result(result_path, expected_digest)
        if resume:
            return {
                "family": family,
                "run_id": run_id,
                "status": "PASS",
                "reused": True,
                "window_seconds": existing["formal_window_seconds"],
                "result_sha256": sha256_file(result_path),
            }
        raise RuntimeError(f"immutable trial already exists: {run_id}")
    if dry_run:
        return {
            "family": family,
            "run_id": run_id,
            "status": "DRY_RUN",
            "command": command_value,
            "result_path": str(result_path),
        }
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "console.log").open("wb") as handle:
        completed = subprocess.run(
            command_value,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        failure = {
            "schema_version": SCHEMA,
            "status": "FAIL",
            "family": family,
            "run_id": run_id,
            "return_code": completed.returncode,
            "downstream": "NOT_RUN",
        }
        atomic_json(run_dir / "driver_failure.json", failure)
        return failure
    if not result_path.is_file():
        raise RuntimeError(f"{run_id}: command returned zero without result manifest")
    result = validate_result(result_path, expected_digest)
    return {
        "family": family,
        "run_id": run_id,
        "status": "PASS",
        "reused": False,
        "window_seconds": result["formal_window_seconds"],
        "result_sha256": sha256_file(result_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.trials < 3:
        print(json.dumps({"status": "FAIL", "error": "formal timing requires at least 3 paired trials"}))
        return 2
    contract = load_json(args.workload_contract)
    expected_digest = workload_digest(contract)
    root = args.output_root.expanduser().resolve() / args.run_id
    prereg_path = root / "preregistration.json"
    prereg = {
        "schema_version": SCHEMA,
        "run_id": args.run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workload_contract": str(args.workload_contract.resolve()),
        "workload_contract_sha256": sha256_file(args.workload_contract),
        "workload_contract_digest": expected_digest,
        "trials": args.trials,
        "routea_command_digest": digest(args.routea_command_template),
        "routeb_command_digest": digest(args.routeb_command_template),
        "order_policy": "alternate AB/BA",
        "profiled_runs_allowed": False,
    }
    if prereg_path.exists():
        previous = load_json(prereg_path)
        current = dict(prereg); current.pop("created_at", None)
        old = dict(previous); old.pop("created_at", None)
        if current != old:
            print(json.dumps({"status": "FAIL", "error": "immutable formal run ID collision"}))
            return 2
        if not args.resume and not args.dry_run:
            print(json.dumps({"status": "FAIL", "error": "formal run exists; use --resume or a new ID"}))
            return 2
    else:
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(prereg_path, prereg)

    pairs: list[dict[str, Any]] = []
    try:
        for index in range(args.trials):
            order = ("A", "B") if index % 2 == 0 else ("B", "A")
            trial_results: dict[str, Any] = {}
            for family in order:
                outcome = run_one(
                    family=family,
                    pair_index=index,
                    root=root,
                    parent_id=args.run_id,
                    template=(
                        args.routea_command_template
                        if family == "A"
                        else args.routeb_command_template
                    ),
                    expected_digest=expected_digest,
                    resume=args.resume,
                    dry_run=args.dry_run,
                )
                trial_results[family] = outcome
                if outcome.get("status") not in {"PASS", "DRY_RUN"}:
                    raise RuntimeError(
                        f"pair {index} family {family} failed; remaining formal work is NOT_RUN"
                    )
            pairs.append(
                {
                    "pair_index": index,
                    "launch_order": list(order),
                    "routea": trial_results["A"],
                    "routeb": trial_results["B"],
                }
            )
        if args.dry_run:
            result = {
                "schema_version": SCHEMA,
                "status": "DRY_RUN",
                "pairs": pairs,
                "workload_contract_digest": expected_digest,
            }
            atomic_json(root / "result.json", result)
            print(json.dumps({"status": "DRY_RUN", "result": str(root / 'result.json')}))
            return 0
        ratios = [
            pair["routea"]["window_seconds"] / pair["routeb"]["window_seconds"]
            for pair in pairs
        ]
        a_values = [pair["routea"]["window_seconds"] for pair in pairs]
        b_values = [pair["routeb"]["window_seconds"] for pair in pairs]
        result = {
            "schema_version": SCHEMA,
            "status": "PASS",
            "claimable": True,
            "workload_contract_digest": expected_digest,
            "pair_count": len(pairs),
            "pairs": pairs,
            "routea_seconds": a_values,
            "routeb_seconds": b_values,
            "paired_speedup_routea_over_routeb": ratios,
            "median_paired_speedup": statistics.median(ratios),
            "geometric_mean_paired_speedup": math.exp(
                statistics.fmean(math.log(value) for value in ratios)
            ),
            "median_routea_seconds": statistics.median(a_values),
            "median_routeb_seconds": statistics.median(b_values),
            "claim_boundary": "Matched unprofiled pairs only; diagnostic and failed attempts excluded.",
        }
        atomic_json(root / "result.json", result)
        print(json.dumps({"status": "PASS", "result": str(root / 'result.json')}, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        failure = {
            "schema_version": SCHEMA,
            "status": "FAIL",
            "claimable": False,
            "error": str(exc),
            "completed_pairs": pairs,
            "downstream": "NOT_RUN",
        }
        atomic_json(root / "result.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
