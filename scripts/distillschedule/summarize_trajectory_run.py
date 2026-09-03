#!/usr/bin/env python3
"""Validate one complete append-only world trajectory without comparing arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from distillschedule.evidence import load_trajectory_jsonl
from distillschedule.evidence.distributed_state_sketch import (
    validate_rank_state_anchor,
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--run-id", required=True)
    value.add_argument("--workload", required=True)
    value.add_argument("--variant", required=True)
    value.add_argument("--cycles", type=int, required=True)
    value.add_argument("--world-size", type=int, default=16)
    value.add_argument("--expected-final-generator", type=int, required=True)
    value.add_argument("--expected-final-critic", type=int, required=True)
    value.add_argument("--expected-final-ema", type=int, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def _validate_state_anchor(
    *,
    run_root: Path,
    run_id: str,
    workload: str,
    variant: str,
    rank: int,
    cycle: int,
    world_size: int,
    references: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    expected_keys = {
        "state_sketch_path",
        "state_sketch_sha256",
        "state_content_sha256",
    }
    if set(references) != expected_keys:
        return [
            f"rank {rank}: state anchor reference keys changed at cycle {cycle}: "
            f"{sorted(references)}"
        ]
    path = Path(references["state_sketch_path"])
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        return [f"rank {rank}: missing state anchor {path} at cycle {cycle}"]
    expected_root = (run_root / "trajectory" / "state").resolve()
    if not resolved.is_relative_to(expected_root):
        return [
            f"rank {rank}: state anchor escaped run root at cycle {cycle}: {resolved}"
        ]
    try:
        value = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [
            f"rank {rank}: invalid state anchor at cycle {cycle}: "
            f"{type(error).__name__}: {error}"
        ]
    try:
        validated = validate_rank_state_anchor(resolved)
    except (OSError, TypeError, ValueError) as error:
        return [
            f"rank {rank}: state anchor validation failed at cycle {cycle}: "
            f"{type(error).__name__}: {error}"
        ]
    expected = {
        "run_id": run_id,
        "workload": workload,
        "variant": variant,
        "rank": rank,
        "cycle": cycle,
        "world_size": world_size,
    }
    for key, target in expected.items():
        if validated.get(key) != target:
            errors.append(
                f"rank {rank}: state anchor {key}={validated.get(key)!r} != "
                f"{target!r} at cycle {cycle}"
            )
    file_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if file_sha256 != references["state_sketch_sha256"]:
        errors.append(
            f"rank {rank}: state anchor file SHA reference mismatch at cycle {cycle}"
        )
    body = dict(value)
    supplied_sha = body.pop("sha256", None)
    calculated_sha = hashlib.sha256(
        json.dumps(
            body, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()
    if supplied_sha != calculated_sha:
        errors.append(f"rank {rank}: state anchor content SHA failed at cycle {cycle}")
    if supplied_sha != references["state_content_sha256"]:
        errors.append(
            f"rank {rank}: state anchor content SHA reference mismatch at cycle {cycle}"
        )
    expected_states = {
        "generator_parameter",
        "critic_parameter",
        "generator_adam",
        "critic_adam",
        "generator_ema",
        "generator_gradient",
        *{f"critic_{index}_gradient" for index in range(5)},
    }
    states = value.get("states")
    if not isinstance(states, dict) or set(states) != expected_states:
        errors.append(
            f"rank {rank}: state component inventory changed at cycle {cycle}: "
            f"{sorted(states) if isinstance(states, dict) else type(states).__name__}"
        )
    return errors


def main() -> int:
    args = parser().parse_args()
    expected = {
        "generator": args.expected_final_generator,
        "critic": args.expected_final_critic,
        "ema": args.expected_final_ema,
    }
    errors: list[str] = []
    rank_summaries: list[dict[str, Any]] = []
    contract_identity: dict[str, Any] | None = None
    implementation_identity: dict[str, str] | None = None
    call_census_identity: dict[str, int] | None = None
    rows_by_cycle: dict[int, list[Any]] = {
        cycle: [] for cycle in range(1, args.cycles + 1)
    }
    start_step = 1000 if args.variant == "gan" else 10
    initial = {"generator": 200, "critic": 1000, "ema": 800} if args.variant == "gan" else {"generator": 2, "critic": 10, "ema": 10}
    anchors = {1, 10, 25, 50, args.cycles}
    anchors = {cycle for cycle in anchors if cycle <= args.cycles}
    for rank in range(args.world_size):
        path = args.run_root / "trajectory" / f"rank_{rank:02d}.jsonl"
        if not path.is_file():
            errors.append(f"rank {rank}: missing {path}")
            continue
        try:
            rows = load_trajectory_jsonl(path)
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"rank {rank}: {type(error).__name__}: {error}")
            continue
        cycles = [row.cycle for row in rows]
        if cycles != list(range(1, args.cycles + 1)):
            errors.append(f"rank {rank}: cycle inventory {cycles}")
        for row in rows:
            if row.run_id != args.run_id or row.rank != rank:
                errors.append(f"rank {rank}: writer identity drift at cycle {row.cycle}")
            if row.workload != args.workload or row.variant != args.variant:
                errors.append(f"rank {rank}: workload identity drift at cycle {row.cycle}")
            if row.world_size != args.world_size or row.expected_total_cycles != args.cycles:
                errors.append(f"rank {rank}: world/cycle contract drift at cycle {row.cycle}")
            if not row.finite or row.runtime_status != "PASS":
                errors.append(f"rank {rank}: runtime/finite failure at cycle {row.cycle}")
            if len(row.gradient_norms) != 6 or any(
                value is None for value in row.gradient_norms.values()
            ):
                errors.append(
                    f"rank {rank}: expected six finite gradient norms at cycle {row.cycle}"
                )
            expected_step = start_step + 5 * (row.cycle - 1)
            if row.logical_step != expected_step:
                errors.append(
                    f"rank {rank}: logical step {row.logical_step} != {expected_step} "
                    f"at cycle {row.cycle}"
                )
            expected_cycle_updates = {
                "generator": 1,
                "critic": 5,
                "ema": 1 if args.variant == "gan" else 5,
            }
            expected_cumulative = {
                key: count * row.cycle
                for key, count in expected_cycle_updates.items()
            }
            expected_versions = {
                key: initial[key] + expected_cumulative[key]
                for key in initial
            }
            if dict(row.cycle_updates) != expected_cycle_updates:
                errors.append(
                    f"rank {rank}: cycle update drift at cycle {row.cycle}: "
                    f"{dict(row.cycle_updates)}"
                )
            if dict(row.cumulative_updates) != expected_cumulative:
                errors.append(
                    f"rank {rank}: cumulative update drift at cycle {row.cycle}: "
                    f"{dict(row.cumulative_updates)}"
                )
            if dict(row.versions) != expected_versions:
                errors.append(
                    f"rank {rank}: intermediate version drift at cycle {row.cycle}: "
                    f"{dict(row.versions)} != {expected_versions}"
                )
            if len(row.sample_identities) != 6 or len(set(row.sample_identities)) != 6:
                errors.append(
                    f"rank {rank}: expected six unique sample identities at cycle {row.cycle}"
                )
            if len(row.rng_identities) != 6 or len(set(row.rng_identities.values())) != 6:
                errors.append(
                    f"rank {rank}: expected six unique RNG identities at cycle {row.cycle}"
                )
            if row.cycle in anchors and not row.state_fingerprints:
                errors.append(
                    f"rank {rank}: missing state anchor evidence at cycle {row.cycle}"
                )
            elif row.cycle in anchors:
                errors.extend(
                    _validate_state_anchor(
                        run_root=args.run_root,
                        run_id=args.run_id,
                        workload=args.workload,
                        variant=args.variant,
                        rank=rank,
                        cycle=row.cycle,
                        world_size=args.world_size,
                        references=dict(row.state_fingerprints),
                    )
                )
            elif row.state_fingerprints:
                errors.append(
                    f"rank {rank}: unexpected state anchor at cycle {row.cycle}"
                )
            rows_by_cycle.setdefault(row.cycle, []).append(row)
        if rows:
            identity = {
                "matched_contract": dict(rows[0].matched_contract),
                "seed": rows[0].seed,
            }
            if contract_identity is None:
                contract_identity = identity
            elif identity != contract_identity:
                errors.append(f"rank {rank}: matched contract or seed differs")
            implementation = dict(rows[0].implementation)
            if implementation_identity is None:
                implementation_identity = implementation
            elif implementation != implementation_identity:
                errors.append(f"rank {rank}: implementation identity differs")
            call_census = dict(rows[0].call_census)
            if call_census_identity is None:
                call_census_identity = call_census
            elif call_census != call_census_identity:
                errors.append(f"rank {rank}: call census differs")
            for row in rows[1:]:
                if dict(row.matched_contract) != dict(rows[0].matched_contract):
                    errors.append(
                        f"rank {rank}: matched contract changed at cycle {row.cycle}"
                    )
                if dict(row.implementation) != implementation:
                    errors.append(
                        f"rank {rank}: implementation changed at cycle {row.cycle}"
                    )
                if dict(row.call_census) != call_census:
                    errors.append(
                        f"rank {rank}: call census changed at cycle {row.cycle}"
                    )
            if dict(rows[-1].versions) != expected:
                errors.append(
                    f"rank {rank}: final versions {dict(rows[-1].versions)} != {expected}"
                )
            rank_summaries.append(
                {
                    "rank": rank,
                    "row_count": len(rows),
                    "final_versions": dict(rows[-1].versions),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    for cycle, rows in rows_by_cycle.items():
        if len(rows) != args.world_size:
            continue
        samples = [value for row in rows for value in row.sample_identities]
        rng = [value for row in rows for value in row.rng_identities.values()]
        if len(samples) != 6 * args.world_size or len(set(samples)) != len(samples):
            errors.append(f"cycle {cycle}: rank-qualified sample identities collide")
        if len(rng) != 6 * args.world_size or len(set(rng)) != len(rng):
            errors.append(f"cycle {cycle}: rank-qualified RNG identities collide")
    status = "PASS" if not errors and len(rank_summaries) == args.world_size else "FAIL"
    result = {
        "schema_version": "distillschedule.trajectory_run_summary.v1",
        "status": status,
        "run_id": args.run_id,
        "workload": args.workload,
        "variant": args.variant,
        "cycles": args.cycles,
        "world_size": args.world_size,
        "contract_identity": contract_identity,
        "implementation_identity": implementation_identity,
        "call_census_identity": call_census_identity,
        "required_state_anchors": sorted(anchors),
        "expected_final_versions": expected,
        "rank_count": len(rank_summaries),
        "rank_summaries": rank_summaries,
        "errors": errors,
        "claim_boundary": "TRAJECTORY_DIAGNOSTIC_NOT_FORMAL_TIMING",
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
