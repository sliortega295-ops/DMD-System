"""CPU-only H3 native two-cycle receipt checks and bounded sampled comparison.

Rank-max host-fenced durations are not synchronized cross-node completion
endpoints. Local-shard samples are not full-state parity, even when every
sample passes DEFAULT_P0_PARITY. Neither fact establishes search readiness.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from distillschedule.backends.training.minimax_h3.matched_contract import validate_global_snapshots
from distillschedule.runtime.p0_parity import DEFAULT_P0_PARITY
from distillschedule.workloads.best_known.registry import file_sha256, write_new_json
from .contracts import finite_number, require
from .h3_implementation import H3Implementation

_GRID = dict(checkpoint_wrap_count=300, scoped_execution_count=600,
             cache_hit_count=600, cache_miss_count=0, missing_key_count=0)
_COMPONENTS = {"student_parameters", "student_optimizer", "fake_parameters", "fake_optimizer"}
_SHA = re.compile(r"[0-9a-f]{64}")


def summarize_native_two_cycle(receipts: Sequence[Mapping[str, Any]], *, expected_head: str,
                               candidate_id: str) -> dict[str, Any]:
    require(isinstance(expected_head, str) and re.fullmatch(r"[0-9a-f]{40}", expected_head) is not None,
            "H3 summary requires exact source HEAD")
    impl = H3Implementation.parse(candidate_id)
    runtime = impl.lower_profile()["runtime"]
    binding = impl.receipt_binding(runtime["model_config"], runtime["environment"])
    require(len(receipts) == 16 and all(type(r.get("rank")) is int for r in receipts)
            and {r["rank"] for r in receipts} == set(range(16)), "H3 requires 16 unique rank receipts")
    rows = sorted(receipts, key=lambda row: row["rank"])
    calibration = rows[0].get("calibration_only")
    require(type(calibration) is bool, "H3 calibration mode absent")
    for row in rows:
        rank = row["rank"]
        for key, expected in {"schema_version": "distillschedule.h3_native_two_cycle.v1",
                              "status": "PASS_NATIVE_RUNTIME_CONTRACTS", "integration_head": expected_head,
                              "semantics": "approximate_grid1000", **binding,
                              "final_versions": dict(generator=2, critic=10, ema=0),
                              "updates": dict(student=2, fake=10, ema=0),
                              "native_wall_clock": "HOST_MONOTONIC_CUDA_FENCED",
                              "native_boundary_fences": 4, "formal_timing_eligible": False,
                              "calibration_only": calibration}.items():
            require(row.get(key) == expected, f"H3 rank{rank} {key} mismatch")
        env = row.get("source_environment", {})
        for key, expected in runtime["environment"].items():
            if not (key.startswith("H3_") or key == "DISTILLSCHEDULE_SEARCH_CANDIDATE_ID"):
                continue
            if key == "H3_PLANNER_CALIBRATION":
                expected = str(int(calibration))
            require(env.get(key) == expected, f"H3 rank{rank} effective {key} mismatch")
        require(calibration or row.get("stage_calibration") is None,
                "H3 disabled calibration contains profiling ranges")
        cycles = row.get("cycles", [])
        require(len(cycles) == 2 and [c.get("cycle") for c in cycles] == [1, 2],
                f"H3 rank{rank} two cycles incomplete")
        for cycle, item in enumerate(cycles, 1):
            for key, expected in {"status": "PASS_NATIVE_RUNTIME_CONTRACTS",
                                  "updates": dict(student=1, fake=5, ema=0),
                                  "cumulative_updates": dict(student=cycle, fake=5 * cycle, ema=0),
                                  "grid_replay_delta": _GRID, "finite_losses": True}.items():
                require(item.get(key) == expected, f"H3 rank{rank} cycle{cycle} {key} mismatch")
            require(all(item.get("grid_replay", {}).get(k) == v * cycle for k, v in _GRID.items()),
                    f"H3 rank{rank} cumulative Grid1000 mismatch")
            finite_number(item.get("native_wall_seconds"), "native_wall_seconds", positive=True)
            finite_number(item.get("gpu_completion_elapsed_s"), "gpu_completion_elapsed_s", positive=True)
            require(item.get("census", {}).get("fixed_end_step_idx") == 3, "H3 fixed end step changed")
            samples = item["census"].get("samples", [])
            require(all(isinstance(s.get("identity"), str) and bool(s["identity"]) for s in samples),
                    "H3 sample identity missing")
            losses = item.get("losses", {})
            require(set(losses) == {"student", "fake"} and len(losses["student"]) == 1
                    and len(losses["fake"]) == 5, "H3 loss inventory incomplete")
            require(all(type(v) in (int, float) and math.isfinite(v)
                        for values in losses.values() for v in values), "H3 loss nonfinite")
    for index in range(2):
        errors = validate_global_snapshots([row["cycles"][index]["census"] for row in rows],
                                          expected_world_size=16, require_unique_samples=True)
        require(not errors, f"H3 cycle{index + 1} census failed: {errors}")
    wall = {str(c + 1): max(float(r["cycles"][c]["native_wall_seconds"]) for r in rows) for c in range(2)}
    cuda = {str(c + 1): max(float(r["cycles"][c]["gpu_completion_elapsed_s"]) for r in rows) for c in range(2)}
    return {"schema_version": "distillschedule.h3_native_two_cycle_summary.v1", "status": "PASS",
            "errors": [], "integration_head": expected_head, **binding, "rank_count": 16,
            "cycles": 2, "cycles_in_same_process": 2, "final_versions": dict(generator=2, critic=10, ema=0),
            "cycle_wall_seconds": wall, "cycle_cuda_elapsed_seconds": cuda,
            "timer_scope": "RANK_MAX_NATIVE_HOST_FENCED_CYCLE_WALL",
            "cross_rank_clock_alignment": "NOT_ESTABLISHED", "calibration_only": calibration,
            "formal_timing_eligible": False, "search_ready": False, "strict_pair_gate": "NOT_RUN",
            "claim_boundary": "NATIVE_DIAGNOSTIC_NOT_SEARCH_READY_OR_FORMAL_SPEEDUP",
            "ranks": [{"rank": r["rank"], "cycle_wall_seconds": {
                str(c["cycle"]): c["native_wall_seconds"] for c in r["cycles"]}} for r in rows]}


def _parity(reference: list, candidate: list) -> dict[str, Any]:
    """DEFAULT_P0_PARITY formulas on serialized FP32 samples, no Torch/GPU.

    This intentionally compares the captured values, not unobserved tensors.
    Shape/dtype/local coordinates are checked separately before this call.
    """
    threshold = DEFAULT_P0_PARITY
    result = {"status": "FAIL", "thresholds": asdict(threshold)}
    if not reference or len(reference) != len(candidate):
        return {**result, "reason": "sample_shape_mismatch"}
    if not all(type(v) in (int, float) and math.isfinite(v) for v in reference + candidate):
        return {**result, "finite": False, "reason": "non_finite_samples"}
    ref, cand = list(map(float, reference)), list(map(float, candidate))
    differences = [a - b for a, b in zip(ref, cand)]
    ref_l2, cand_l2 = math.hypot(*ref), math.hypot(*cand)
    mean_abs = math.fsum(map(abs, differences)) / len(ref)
    mean_relative = mean_abs / max(math.fsum(map(abs, ref)) / len(ref), 1e-6)
    normalized_l2 = math.hypot(*differences) / max(ref_l2, 1e-12)
    cosine = (1. if ref_l2 <= 1e-12 and cand_l2 <= 1e-12 else 0.
              if min(ref_l2, cand_l2) <= 1e-12 else math.fsum((a / ref_l2) * (b / cand_l2) for a, b in zip(ref, cand)))
    peak, max_abs = max(map(abs, ref)), max(map(abs, differences))
    limit = threshold.max_abs_atol + threshold.max_abs_rtol * peak
    passed = (mean_relative <= threshold.mean_relative and normalized_l2 <= threshold.normalized_l2
              and cosine >= threshold.min_cosine and max_abs <= limit)
    return {**result, "status": "PASS" if passed else "FAIL", "finite": True,
            "mean_abs": mean_abs, "mean_relative": mean_relative, "normalized_l2": normalized_l2,
            "cosine": cosine, "max_abs": max_abs, "max_abs_limit": limit, "reference_peak_abs": peak}


def _state(value: Mapping[str, Any], rank: int) -> Mapping[str, Any]:
    require(value.get("schema_version") == "distillschedule.h3_local_shard_fingerprint.v1"
            and value.get("rank") == rank and value.get("status") == "CAPTURED_FINITE_SAMPLES"
            and value.get("scope") == "SAME_RANK_LOCAL_LAYOUT_SAMPLED_COORDINATES_ONLY",
            "H3 sampled state source/rank/scope mismatch")
    components = value.get("components", {})
    require(set(components) == _COMPONENTS, "H3 sampled state component inventory incomplete")
    for component in components.values():
        rows = component.get("rows", [])
        count = component.get("inventory_tensor_count")
        require(type(count) is int and 0 < len(rows) <= min(count, 32)
                and component.get("sampled_tensor_count") == len(rows), "H3 sampled tensor inventory changed")
        require(isinstance(component.get("inventory_sha256"), str)
                and _SHA.fullmatch(component["inventory_sha256"]) is not None, "H3 state inventory binding absent")
        require(len({r.get("name") for r in rows}) == len(rows)
                and all(isinstance(r.get("name"), str) and r["name"] for r in rows), "H3 sampled state duplicate names")
        for row in rows:
            vals, indices = row.get("values_fp32", []), row.get("local_flat_indices", [])
            require(0 < len(vals) <= 16 and len(vals) == len(indices) and len(set(indices)) == len(indices)
                    and all(type(i) is int and i >= 0 for i in indices), "H3 sample coordinates missing")
            require(all(type(v) in (int, float) and math.isfinite(v) for v in vals), "H3 state samples nonfinite")
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(component.get("sample_sha256") == digest, "H3 sampled state content hash mismatch")
    return components


def _compare_state(left: Mapping[str, Any], right: Mapping[str, Any], rank: int) -> dict[str, Any]:
    if left.get("state_fingerprints") is None or right.get("state_fingerprints") is None:
        return {"status": "NOT_RUN", "reason": "sampled final state absent"}
    try:
        ref, cand = (_state(r["state_fingerprints"], rank) for r in (left, right))
        comparisons = {}
        for key in sorted(_COMPONENTS):
            a, b = ref[key], cand[key]
            require(all(a[k] == b[k] for k in ("inventory_tensor_count", "inventory_sha256", "sampled_tensor_count")),
                    "H3 sampled state inventory mismatch")
            pairs = []
            for ar, br in zip(a["rows"], b["rows"]):
                require({k: v for k, v in ar.items() if k != "values_fp32"}
                        == {k: v for k, v in br.items() if k != "values_fp32"},
                        "H3 sampled state layout/dtype/coordinate mismatch")
                pairs.append({"name": ar["name"], "metrics": _parity(ar["values_fp32"], br["values_fp32"])})
            comparisons[key] = pairs
        passed = all(p["metrics"]["status"] == "PASS" for pairs in comparisons.values() for p in pairs)
        return {"status": "PASS" if passed else "FAIL", "components": comparisons,
                "scope": "SAME_RANK_LOCAL_LAYOUT_SAMPLED_COORDINATES_ONLY"}
    except (ValueError, KeyError, TypeError) as error:
        return {"status": "FAIL", "reason": str(error)}


def _aggregate(values: Sequence[str]) -> str:
    return "FAIL" if "FAIL" in values else "NOT_RUN" if "NOT_RUN" in values else "PASS"


def _rng(left: Mapping[str, Any], right: Mapping[str, Any], rank: int, head: str) -> str:
    # Seed/config equality is not proof of matching consumed RNG streams.
    a, b = left.get("rng_source_binding"), right.get("rng_source_binding")
    if a is None or b is None:
        return "NOT_RUN"
    if not isinstance(a, dict) or not isinstance(b, dict):
        return "FAIL"
    fields = {"integration_head", "rank", "initial_sha256", "final_sha256", "scope"}
    if set(a) != fields or set(b) != fields or a != b:
        return "FAIL"
    if not (a["integration_head"] == head and a["rank"] == rank
            and a["scope"] == "NATIVE_RNG_STATE_BEFORE_AND_AFTER_TWO_CYCLES"
            and all(isinstance(a[k], str) and _SHA.fullmatch(a[k]) for k in
                    ("initial_sha256", "final_sha256"))):
        return "FAIL"
    boundaries = []
    for row, binding in ((left, a), (right, b)):
        states = [cycle.get(key) for cycle in row["cycles"] for key in ("rng_before", "rng_after")]
        if any(state is None for state in states):
            return "NOT_RUN"
        if not all(isinstance(state, dict) and set(state) == {"torch_cpu", "torch_cuda"}
                   and all(isinstance(v, str) and _SHA.fullmatch(v) for v in state.values()) for state in states):
            return "FAIL"
        for field, state in (("initial_sha256", states[0]), ("final_sha256", states[-1])):
            if binding[field] != hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest():
                return "FAIL"
        if states[1] != states[2]:  # Probes/logging between cycles must not consume these generators.
            return "FAIL"
        boundaries.append(states)
    return "PASS" if boundaries[0] == boundaries[1] else "FAIL"


def compare_native_two_cycle(reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]], *,
                             expected_head: str) -> dict[str, Any]:
    require(bool(reference) and bool(candidate), "H3 pair receipts missing")
    summaries = [summarize_native_two_cycle(rows, expected_head=expected_head, candidate_id=rows[0].get("candidate_id"))
                 for rows in (reference, candidate)]
    require(not any(s["calibration_only"] for s in summaries), "H3 correctness pair cannot use calibration receipts")
    ranks = []
    for left, right in zip(sorted(reference, key=lambda r: r["rank"]), sorted(candidate, key=lambda r: r["rank"])):
        rank = left["rank"]
        losses = [{role: _parity(a["losses"][role], b["losses"][role]) for role in ("student", "fake")}
                  for a, b in zip(left["cycles"], right["cycles"])]
        ranks.append({"rank": rank, "sampled_state": _compare_state(left, right, rank), "losses": losses,
                      "loss_status": _aggregate([m["status"] for cycle in losses for m in cycle.values()]),
                      "sample_status": "PASS" if all(a["census"]["samples"] == b["census"]["samples"]
                          for a, b in zip(left["cycles"], right["cycles"])) else "FAIL",
                      "rng_status": _rng(left, right, rank, expected_head)})
    state = _aggregate([r["sampled_state"]["status"] for r in ranks])
    statuses = {key: _aggregate([r[key] for r in ranks]) for key in ("loss_status", "sample_status", "rng_status")}
    failed = "FAIL" in [state, *statuses.values()]
    return {"schema_version": "distillschedule.h3_native_two_cycle_comparison.v1",
            "status": "FAIL" if failed else "NOT_RUN" if "NOT_RUN" in [state, *statuses.values()]
            else "REVIEW_SAMPLED_PARITY_ONLY",
            "integration_head": expected_head, "reference_candidate_id": summaries[0]["candidate_id"],
            "candidate_id": summaries[1]["candidate_id"], "rank_count": 16, "cycles": 2,
            "sampled_state_status": state, **statuses, "ranks": ranks,
            "thresholds": asdict(DEFAULT_P0_PARITY), "strict_numerical_gate": "NOT_RUN",
            "rng_scope": "TORCH_DEFAULT_CPU_AND_RANK_CUDA_BOUNDARIES_ONLY; OTHER_GENERATORS_NOT_OBSERVED",
            "full_state_comparison": "NOT_RUN", "formal_timing_eligible": False, "search_ready": False,
            "claim_boundary": "BOUNDED_LOCAL_SHARD_SAMPLES_NOT_FULL_STATE_OR_STRICT_PAIR_GATE"}


def read_native_receipts(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    directory = run_root / "execution/ranks"
    paths = [directory / f"rank_{rank:03d}.json" for rank in range(16)]
    require(set(directory.glob("rank_*.json")) == set(paths), "H3 native receipt paths incomplete or duplicated")
    rows, bindings = [], []
    for rank, path in enumerate(paths):
        value = json.loads(path.read_text())
        require(isinstance(value, dict) and value.get("rank") == rank, f"H3 rank path/owner mismatch: {path}")
        rows.append(value)
        bindings.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    return rows, bindings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--run-root", type=Path, required=True)
    summary.add_argument("--candidate-id", required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--reference-run-root", type=Path, required=True)
    comparison.add_argument("--candidate-run-root", type=Path, required=True)
    for command in (summary, comparison):
        command.add_argument("--expected-head", required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "summarize":
        rows, bindings = read_native_receipts(args.run_root)
        value = summarize_native_two_cycle(rows, expected_head=args.expected_head, candidate_id=args.candidate_id)
        value["native_evidence"] = bindings
    else:
        left, lb = read_native_receipts(args.reference_run_root)
        right, rb = read_native_receipts(args.candidate_run_root)
        value = compare_native_two_cycle(left, right, expected_head=args.expected_head)
        value["native_evidence"] = {"reference": lb, "candidate": rb}
    write_new_json(args.output, value)
    return 1 if value["status"] in {"FAIL", "NOT_RUN"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
