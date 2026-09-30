#!/usr/bin/env python3
"""Compare matched LiveWan two-cycle arms using cycle 2, and cycle 2 only."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def _load(path: Path, *, mode: str, variant: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "distillschedule.livewan_two_cycle_timing_summary.v1",
        "status": "PASS",
        "execution_mode": mode,
        "pipeline_variant": variant,
        "run_kind": "two-cycle-timing",
        "native_unprofiled": True,
        "profiled": False,
        "profiler_status": "NOT_RUN",
        "formal_timing_eligible": True,
        "cycles_in_same_process": 2,
        "final_versions": {"generator": 2, "critic": 10, "ema": 0},
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"{mode} summary changed {key}")
    if dict(value.get("cycle_1", {})).get("claimable") is not False:
        raise ValueError(f"{mode} cycle 1 is not warm-up-only")
    cycle2 = dict(value.get("cycle_2", {}))
    if cycle2.get("claimable") is not True:
        raise ValueError(f"{mode} cycle 2 is not claimable")
    seconds = float(cycle2.get("seconds", 0.0))
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"{mode} cycle 2 wall is invalid")
    return value


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--pipeline-summary", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--pipeline-variant", choices=("c24-continuous-b16-score-modulation", "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm"), default="c24-continuous-b16-score-modulation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = _load(args.baseline_summary, mode="baseline", variant="b1")
    pipeline = _load(
        args.pipeline_summary,
        mode="pipeline",
        variant=args.pipeline_variant,
    )
    errors: list[str] = []
    for label, value in (("baseline", baseline), ("pipeline", pipeline)):
        if value.get("head") != args.expected_head:
            errors.append(f"{label} source HEAD differs")
    for key in (
        "matched_workload_fingerprint",
        "identity",
        "trace_sha256",
        "final_versions",
        "logical_call_inventory",
        "correctness_binding",
    ):
        if baseline.get(key) != pipeline.get(key):
            errors.append(f"matched arms differ in {key}")
    if (
        dict(baseline.get("implementation", {})).get("fingerprint")
        == dict(pipeline.get("implementation", {})).get("fingerprint")
    ):
        errors.append("baseline and pipeline implementation fingerprints are identical")

    baseline_c2 = float(baseline["cycle_2"]["seconds"])
    pipeline_c2 = float(pipeline["cycle_2"]["seconds"])
    result = {
        "schema_version": "distillschedule.livewan_two_cycle_timing_comparison.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "expected_head": args.expected_head,
        "heads": {"baseline": baseline.get("head"), "pipeline": pipeline.get("head")},
        "timing_contract": {
            "same_process_cycles": 2,
            "cycle_1": "WARMUP_DIAGNOSTIC_EXCLUDED",
            "cycle_2": "STEADY_STATE_ONLY_CLAIMABLE",
            "profiled": False,
            "aggregation": "CRITICAL_ROLE_CUDA_COMPLETION_ENDPOINT_DELTA",
            "average_cycles": False,
        },
        "baseline": {
            "run_id": baseline.get("run_id"),
            "cycle_1_seconds": baseline["cycle_1"]["world_completion_endpoint_seconds"],
            "cycle_2_seconds": baseline_c2,
        },
        "pipeline": {
            "run_id": pipeline.get("run_id"),
            "cycle_1_seconds": pipeline["cycle_1"]["world_completion_endpoint_seconds"],
            "cycle_2_seconds": pipeline_c2,
        },
        "cycle_2_speedup": baseline_c2 / pipeline_c2,
        "cycle_2_reduction_seconds": baseline_c2 - pipeline_c2,
        "cycle_2_reduction_fraction": 1.0 - pipeline_c2 / baseline_c2,
        "formal_claimable": not errors,
        "claim_boundary": "MATCHED_NATIVE_UNPROFILED_CYCLE2_ONLY",
    }
    _atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
