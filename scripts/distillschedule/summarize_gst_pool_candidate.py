#!/usr/bin/env python3
"""Validate and summarize one fresh N2 target-graph candidate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from runtime.gst_pool_contract import GeneratorOffloadCandidate
from runtime.gst_pool_exchange import lane_source_ids


def _parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--candidate-manifest", type=Path, required=True)
    result.add_argument("--contract-digest", required=True)
    result.add_argument("--generator-return-code", type=int, required=True)
    result.add_argument("--score-return-code", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"rank result is not an object: {path}")
    return value


def _max_peak(records: list[dict[str, Any]], field: str) -> int:
    return max(
        (
            int(snapshot.get(field, 0))
            for record in records
            for snapshot in record.get("memory", [])
        ),
        default=0,
    )


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = _parser().parse_args()
    candidate = GeneratorOffloadCandidate.load(
        args.candidate_manifest,
        expected_contract_digest=args.contract_digest,
    )
    root = args.run_root / "rank_results"
    expected = {
        *(f"generator/rank_{rank:02d}.json" for rank in range(8)),
        *(f"student/rank_{rank:02d}.json" for rank in range(4)),
        *(f"teacher/rank_{rank:02d}.json" for rank in range(4, 8)),
    }
    observed = {
        str(path.relative_to(root))
        for path in root.glob("*/*.json")
        if path.is_file()
    }
    records = [_load(root / relative) for relative in sorted(observed)]
    rank_statuses = [str(record.get("status")) for record in records]
    if "OOM" in rank_statuses:
        status = "OOM"
    elif observed != expected or any(value != "PASS" for value in rank_statuses):
        status = "FAIL"
    elif args.generator_return_code != 0 or args.score_return_code != 0:
        status = "FAIL"
    else:
        status = "PASS"

    validation: dict[str, Any] = {
        "expected_rank_results": sorted(expected),
        "observed_rank_results": sorted(observed),
        "missing_rank_results": sorted(expected - observed),
        "unexpected_rank_results": sorted(observed - expected),
    }
    block_profile: list[dict[str, Any]] = []
    if status == "PASS":
        generators = [_load(root / f"generator/rank_{rank:02d}.json") for rank in range(8)]
        students = [_load(root / f"student/rank_{rank:02d}.json") for rank in range(4)]
        teachers = [_load(root / f"teacher/rank_{rank:02d}.json") for rank in range(4, 8)]
        for record in generators + students + teachers:
            if record.get("candidate_manifest_sha256") != candidate.manifest_sha256:
                raise AssertionError("rank result candidate identity drift")
            if record.get("contract_digest") != candidate.contract_digest:
                raise AssertionError("rank result evidence-contract drift")
        for rank, record in enumerate(generators):
            if record.get("streamed_blocks") != list(candidate.streamed_blocks):
                raise AssertionError(f"Generator rank {rank} candidate drift")
            if {int(item["source_id"]) for item in record["waves"]} != {rank, rank + 8}:
                raise AssertionError(f"Generator rank {rank} source mapping changed")
            final = record["final"]
            if (
                int(final["generator_version_before"]),
                int(final["generator_version_after"]),
                int(final["accumulation_boundaries"]),
            ) != (3, 4, 1):
                raise AssertionError(f"Generator rank {rank} version/update drift")
        for lane, record in enumerate(students):
            if {int(item["source_id"]) for item in record["samples"]} != set(lane_source_ids(lane)):
                raise AssertionError(f"Student lane {lane} source mapping changed")
        for lane, record in enumerate(teachers):
            if {int(item["source_id"]) for item in record["samples"]} != set(lane_source_ids(lane)):
                raise AssertionError(f"Teacher lane {lane} source mapping changed")
        g_by_source = {
            int(item["source_id"]): item
            for record in generators
            for item in record["waves"]
        }
        s_by_source = {
            int(item["source_id"]): item
            for record in students
            for item in record["samples"]
        }
        t_by_source = {
            int(item["source_id"]): item
            for record in teachers
            for item in record["samples"]
        }
        if set(g_by_source) != set(range(16)) or set(s_by_source) != set(range(16)) or set(t_by_source) != set(range(16)):
            raise AssertionError("target graph did not cover all 16 logical samples")
        for source_id in range(16):
            if not (
                g_by_source[source_id]["trace_identity_sha256"]
                == s_by_source[source_id]["trace_identity_sha256"]
                == t_by_source[source_id]["trace_identity_sha256"]
            ):
                raise AssertionError(f"trace identity drift for source {source_id}")
            if not (
                g_by_source[source_id]["x0_sha256"]
                == s_by_source[source_id]["x0_sha256"]
                == t_by_source[source_id]["x0_sha256"]
            ):
                raise AssertionError(f"x0 identity drift for source {source_id}")
        block_profile = list(generators[0].get("block_profile", []))
        if len(candidate.streamed_blocks) == 40:
            if sorted(int(item["block"]) for item in block_profile) != list(range(40)):
                raise AssertionError("all-40 anchor lacks an exact 40-block profile")
        validation.update(
            {
                "logical_samples": 16,
                "generator_versions": "G3_to_G4",
                "generator_optimizer_updates": 1,
                "student_version": 11,
                "teacher_version": "official_frozen_base",
                "trace_and_x0_identity": "exact",
            }
        )

    result = {
        "schema_version": "distillschedule.g_offload_candidate_result.v1",
        "status": status,
        "run_id": candidate.run_id,
        "contract_digest": candidate.contract_digest,
        "candidate_manifest_sha256": candidate.manifest_sha256,
        "execution_graph": "target_no_gan_gst_pool",
        "student_residency": "resident",
        "teacher_residency": "resident",
        "generator_offload_blocks": list(candidate.streamed_blocks),
        "generator_offload_block_count": len(candidate.streamed_blocks),
        "generator_return_code": args.generator_return_code,
        "score_return_code": args.score_return_code,
        "validation": validation,
        "block_profile": block_profile,
        "gpu_peak_allocated_bytes": _max_peak(records, "max_allocated_bytes"),
        "gpu_peak_reserved_bytes": _max_peak(records, "max_reserved_bytes"),
        "remote_run_root": str(args.run_root),
        "measurement_scope": "real14b_384x640_isolated_step15a_capacity_not_timing",
        "transport": "shared_filesystem_correctness_exchange_not_timing",
    }
    if status != "PASS":
        result["failures"] = [
            {
                "role": record.get("role", "generator"),
                "rank": record.get("rank"),
                "status": record.get("status"),
                "error_type": record.get("error_type"),
                "error": record.get("error"),
            }
            for record in records
            if record.get("status") != "PASS"
        ]
    _atomic(args.output, result)
    print(json.dumps({"status": status, "output": str(args.output)}, sort_keys=True))
    return 0 if status == "PASS" else (42 if status == "OOM" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
