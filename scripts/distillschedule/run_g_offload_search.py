#!/usr/bin/env python3
"""Search the minimum Generator block-offload set in the target No-GAN graph.

Historical tail-1/3/5/all-40 results are never accepted as search bounds.  This
script first runs a fresh conservative anchor under the target G/S/T pool contract,
freezes a nested block order from that anchor's low-overhead profile, and then runs
a bounded coarse-to-linear search.  A candidate is claimable only after a separate
continuous-transition/Full5 validation under the same evidence-contract digest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "distillschedule.g_offload_search.v1"


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


def parse_sizes(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate sizes must be integers") from exc
    if not result or any(size < 0 for size in result):
        raise argparse.ArgumentTypeError("candidate sizes must be non-negative")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--search-id", required=True)
    result.add_argument("--run-root", required=True, type=Path)
    result.add_argument("--contract", required=True, type=Path)
    result.add_argument("--block-count", type=int, default=40)
    result.add_argument("--max-candidates", type=int, default=12)
    result.add_argument(
        "--coarse-sizes", type=parse_sizes, default=parse_sizes("32,24,16,12,8,4,2,1,0")
    )
    result.add_argument(
        "--candidate-command-template",
        help="argv template with {run_id}, {candidate_manifest}, {result_path}",
    )
    result.add_argument(
        "--continuous-command-template",
        help="continuous gate argv template with the same placeholders",
    )
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def command_from_template(
    template: str, *, run_id: str, candidate_manifest: Path, result_path: Path
) -> list[str]:
    rendered = template.format(
        run_id=run_id,
        candidate_manifest=str(candidate_manifest),
        result_path=str(result_path),
    )
    return shlex.split(rendered)


def validate_result(
    result_path: Path,
    *,
    candidate_path: Path,
    contract_digest: str,
    accepted_statuses: Sequence[str] = ("PASS", "OOM", "FAIL"),
) -> dict[str, Any]:
    result = load_json(result_path)
    status = result.get("status")
    if status not in accepted_statuses:
        raise RuntimeError(f"{result_path}: invalid status {status!r}")
    if result.get("contract_digest") != contract_digest:
        raise RuntimeError(f"{result_path}: target execution-graph contract drift")
    if result.get("candidate_manifest_sha256") != sha256_file(candidate_path):
        raise RuntimeError(f"{result_path}: candidate manifest identity mismatch")
    if result.get("execution_graph") != "target_no_gan_gst_pool":
        raise RuntimeError(f"{result_path}: result is not the target G/S/T pool graph")
    if result.get("student_residency") != "resident" or result.get(
        "teacher_residency"
    ) != "resident":
        raise RuntimeError(f"{result_path}: S/T residency contract changed")
    return result


def run_candidate(
    *,
    search_root: Path,
    search_id: str,
    label: str,
    blocks: Sequence[int],
    order_digest: str | None,
    contract_digest: str,
    command_template: str,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    candidate_id = f"{search_id}-{label}"
    candidate_dir = search_root / "candidates" / candidate_id
    candidate_path = candidate_dir / "candidate.json"
    result_path = candidate_dir / "result.json"
    candidate = {
        "schema_version": "distillschedule.g_offload_candidate.v1",
        "run_id": candidate_id,
        "execution_graph": "target_no_gan_gst_pool",
        "student_residency": "resident",
        "teacher_residency": "resident",
        "generator_offload_blocks": list(blocks),
        "generator_offload_block_count": len(blocks),
        "order_digest": order_digest,
        "contract_digest": contract_digest,
        "activation_policy_mutable": False,
        "loss_optimizer_batch_trace_version_ema_mutable": False,
        "historical_candidate_substitution_allowed": False,
    }
    if candidate_path.exists():
        if load_json(candidate_path) != candidate:
            raise RuntimeError(f"immutable candidate ID collision: {candidate_id}")
    else:
        atomic_json(candidate_path, candidate)
    command = command_from_template(
        command_template,
        run_id=candidate_id,
        candidate_manifest=candidate_path,
        result_path=result_path,
    )
    if result_path.exists():
        existing = validate_result(
            result_path,
            candidate_path=candidate_path,
            contract_digest=contract_digest,
        )
        if resume and existing["status"] == "PASS":
            return {"run_id": candidate_id, "reused": True, **existing}
        raise RuntimeError(
            f"immutable candidate result already exists with status={existing['status']}; use a new search ID"
        )
    if dry_run:
        return {
            "run_id": candidate_id,
            "status": "DRY_RUN",
            "blocks": list(blocks),
            "command": command,
            "candidate_manifest": str(candidate_path),
        }
    completed = subprocess.run(command, check=False)
    if not result_path.is_file():
        synthetic = {
            "schema_version": "distillschedule.g_offload_candidate_result.v1",
            "status": "FAIL",
            "return_code": completed.returncode,
            "contract_digest": contract_digest,
            "candidate_manifest_sha256": sha256_file(candidate_path),
            "execution_graph": "target_no_gan_gst_pool",
            "student_residency": "resident",
            "teacher_residency": "resident",
            "failure": "candidate command did not produce result.json",
        }
        atomic_json(result_path, synthetic)
    result = validate_result(
        result_path, candidate_path=candidate_path, contract_digest=contract_digest
    )
    return {"run_id": candidate_id, "reused": False, **result}


def freeze_order(anchor: Mapping[str, Any], block_count: int) -> list[int]:
    profile = anchor.get("block_profile")
    if not isinstance(profile, list) or len(profile) != block_count:
        raise RuntimeError(
            "fresh target-graph anchor must emit one low-overhead block_profile entry per Generator block"
        )
    normalized: list[dict[str, Any]] = []
    for item in profile:
        if not isinstance(item, Mapping):
            raise RuntimeError("block_profile entries must be objects")
        block = int(item.get("block"))
        normalized.append(
            {
                "block": block,
                "memory_relief_bytes": float(item.get("memory_relief_bytes", 0)),
                "lifetime_seconds": float(item.get("lifetime_seconds", 0)),
                "allocator_pressure_bytes": float(
                    item.get("allocator_pressure_bytes", 0)
                ),
            }
        )
    if sorted(item["block"] for item in normalized) != list(range(block_count)):
        raise RuntimeError("block_profile must cover each Generator block exactly once")
    normalized.sort(
        key=lambda item: (
            -item["memory_relief_bytes"],
            -item["allocator_pressure_bytes"],
            -item["lifetime_seconds"],
            item["block"],
        )
    )
    return [int(item["block"]) for item in normalized]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.block_count <= 0 or args.max_candidates <= 0:
        print(json.dumps({"status": "FAIL", "error": "invalid search budget"}))
        return 2
    if not args.candidate_command_template:
        print(
            json.dumps(
                {
                    "status": "NOT_RUN",
                    "reason": "target G/S/T candidate command is not configured",
                    "historical_bounds_used": False,
                },
                sort_keys=True,
            )
        )
        return 3
    contract = load_json(args.contract)
    contract_digest = str(
        contract.get("contract_digest") or contract.get("shared_digest") or ""
    )
    if len(contract_digest) != 64:
        print(json.dumps({"status": "FAIL", "error": "invalid target contract"}))
        return 2
    search_root = args.run_root.expanduser().resolve() / args.search_id
    manifest_path = search_root / "search_manifest.json"
    manifest = {
        "schema_version": SCHEMA,
        "search_id": args.search_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract_path": str(args.contract.resolve()),
        "contract_digest": contract_digest,
        "block_count": args.block_count,
        "max_candidates": args.max_candidates,
        "strategy": "fresh_target_anchor_then_profile_ordered_coarse_to_linear",
        "historical_tail_or_all40_bounds_used": False,
        "student_resident": True,
        "teacher_resident": True,
        "generator_only_offload_search": True,
    }
    if manifest_path.exists():
        previous = load_json(manifest_path)
        immutable = dict(previous)
        immutable.pop("created_at", None)
        current = dict(manifest)
        current.pop("created_at", None)
        if immutable != current:
            print(json.dumps({"status": "FAIL", "error": "search ID collision"}))
            return 2
        if not args.resume and not args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "FAIL",
                        "error": "search ID exists; resume only reuses exact PASS candidates",
                    }
                )
            )
            return 2
    else:
        atomic_json(manifest_path, manifest)

    anchor_blocks = list(range(args.block_count))
    try:
        anchor = run_candidate(
            search_root=search_root,
            search_id=args.search_id,
            label=f"anchor-g{args.block_count}",
            blocks=anchor_blocks,
            order_digest=None,
            contract_digest=contract_digest,
            command_template=args.candidate_command_template,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "DRY_RUN",
                        "anchor": anchor,
                        "historical_bounds_used": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if anchor.get("status") != "PASS":
            result = {
                "schema_version": SCHEMA,
                "status": "ANCHOR_FAILED",
                "anchor": anchor,
                "downstream": "NOT_RUN",
                "historical_bounds_used": False,
            }
            atomic_json(search_root / "result.json", result)
            print(json.dumps(result, sort_keys=True))
            return 1
        order = freeze_order(anchor, args.block_count)
        order_payload = {
            "schema_version": "distillschedule.g_offload_order.v1",
            "contract_digest": contract_digest,
            "source_run_id": anchor["run_id"],
            "order": order,
            "order_digest": digest(order),
        }
        atomic_json(search_root / "frozen_order.json", order_payload)
        order_digest = order_payload["order_digest"]

        observations: dict[int, dict[str, Any]] = {args.block_count: anchor}
        budget_used = 0
        coarse = sorted(
            {
                size
                for size in args.coarse_sizes
                if 0 <= size < args.block_count
            },
            reverse=True,
        )
        passing_size = args.block_count
        failing_size: int | None = None
        for size in coarse:
            if budget_used >= args.max_candidates:
                break
            outcome = run_candidate(
                search_root=search_root,
                search_id=args.search_id,
                label=f"probe-g{size}",
                blocks=order[:size],
                order_digest=order_digest,
                contract_digest=contract_digest,
                command_template=args.candidate_command_template,
                resume=args.resume,
                dry_run=False,
            )
            budget_used += 1
            observations[size] = outcome
            if outcome.get("status") == "PASS":
                passing_size = size
            else:
                failing_size = size
                break

        if failing_size is not None and passing_size - failing_size > 1:
            for size in range(passing_size - 1, failing_size, -1):
                if budget_used >= args.max_candidates:
                    break
                outcome = run_candidate(
                    search_root=search_root,
                    search_id=args.search_id,
                    label=f"refine-g{size}",
                    blocks=order[:size],
                    order_digest=order_digest,
                    contract_digest=contract_digest,
                    command_template=args.candidate_command_template,
                    resume=args.resume,
                    dry_run=False,
                )
                budget_used += 1
                observations[size] = outcome
                if outcome.get("status") == "PASS":
                    passing_size = size
                else:
                    failing_size = size
                    break

        tested = sorted(observations)
        nonmonotonic = any(
            observations[smaller].get("status") == "PASS"
            and observations[larger].get("status") != "PASS"
            for smaller in tested
            for larger in tested
            if smaller < larger
        )
        exact_minimum = (
            not nonmonotonic
            and failing_size is not None
            and passing_size == failing_size + 1
        )
        validation: dict[str, Any] | None = None
        if exact_minimum and args.continuous_command_template:
            validation = run_candidate(
                search_root=search_root,
                search_id=args.search_id,
                label=f"continuous-g{passing_size}",
                blocks=order[:passing_size],
                order_digest=order_digest,
                contract_digest=contract_digest,
                command_template=args.continuous_command_template,
                resume=args.resume,
                dry_run=False,
            )
        status = (
            "PASS"
            if exact_minimum
            and validation is not None
            and validation.get("status") == "PASS"
            else "BOUNDED_NOT_CLAIMABLE"
        )
        result = {
            "schema_version": SCHEMA,
            "status": status,
            "contract_digest": contract_digest,
            "anchor": anchor,
            "frozen_order": order_payload,
            "observations": {str(key): value for key, value in sorted(observations.items())},
            "budget_used": budget_used,
            "nonmonotonic_observed": nonmonotonic,
            "minimum_passing_block_count": passing_size if exact_minimum else None,
            "minimum_passing_blocks": order[:passing_size] if exact_minimum else None,
            "continuous_validation": validation,
            "final_baseline_label": (
                "S/T resident + measured-minimum G block offload under the target No-GAN execution graph"
                if status == "PASS"
                else None
            ),
            "historical_tail_or_all40_bounds_used": False,
            "limitations": [
                "A target-graph fresh all-block anchor is required; historical all-40 is diagnostic only.",
                "Search is profile-ordered coarse-to-linear, not direct historical binary search.",
                "The minimum is claimable only after target continuous transition/Full5 validation.",
            ],
        }
        atomic_json(search_root / "result.json", result)
        print(json.dumps({"status": status, "result": str(search_root / 'result.json')}, sort_keys=True))
        return 0 if status == "PASS" else 1
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        failure = {
            "schema_version": SCHEMA,
            "status": "FAIL",
            "error": str(exc),
            "historical_bounds_used": False,
            "downstream": "NOT_RUN",
        }
        atomic_json(search_root / "result.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
