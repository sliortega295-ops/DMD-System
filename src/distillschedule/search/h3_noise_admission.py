"""Opt-in, cache-only admission for planner diagnostics with a control repeat.

The original comparator and thresholds remain authoritative for strict parity.
Sampled noise qualification cannot establish full-state or trajectory equality.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from distillschedule.workloads.best_known.registry import file_sha256
from .contracts import require
from .h3_implementation import H3Implementation
from .h3_native import _state, compare_native_two_cycle, summarize_native_two_cycle

POLICY_SCHEMA = "distillschedule.h3_noise_diagnostic_policy.v1"
SCHEMA = "distillschedule.h3_noise_diagnostic_admission.v1"
REVIEW = "REVIEW_NOISE_QUALIFIED_DIAGNOSTIC"


def validate_policy(value, *, expected_head):
    fields = {"schema_version", "source_head", "reference_candidate_id", "candidate_id",
              "max_cross_self_delta_l2_ratio", "purpose", "formal_timing_eligible",
              "default_gate_unchanged"}
    require(isinstance(value, dict) and set(value) == fields,
            "noise diagnostic policy requires exact explicit fields")
    require(value["schema_version"] == POLICY_SCHEMA and value["source_head"] == expected_head,
            "noise diagnostic policy source/schema changed")
    require(value["purpose"] == "PLANNER_DIAGNOSTIC_ONLY"
            and value["formal_timing_eligible"] is False and value["default_gate_unchanged"] is True,
            "noise policy cannot authorize formal timing or replace strict gates")
    ratio = value["max_cross_self_delta_l2_ratio"]
    require(type(ratio) in (int, float) and math.isfinite(ratio) and 1 <= ratio <= 2,
            "noise diagnostic ratio must be explicit and at most two")
    ref, cand = (H3Implementation.parse(value[k]) for k in ("reference_candidate_id", "candidate_id"))
    require(ref.cache_policy == "cpu_staged_trim" and cand.cache_policy == "cuda_no_trim"
            and ref.request == cand.request and ref.prefetch_depth == cand.prefetch_depth
            and ref.lora_epilogue_block_n == cand.lora_epilogue_block_n
            and ref.rollout_window is cand.rollout_window is None,
            "noise diagnostic admits only matched CPU/GPU cache storage; no kernel or schedule change")
    return value


def _metric(name, reference, repeat, candidate, limit):
    require(reference and len(reference) == len(repeat) == len(candidate),
            "noise diagnostic requires matched nonempty samples")
    require(all(type(x) in (int, float) and math.isfinite(x)
                for values in (reference, repeat, candidate) for x in values),
            "noise diagnostic samples must be finite")
    noise = math.hypot(*(b-a for a, b in zip(reference, repeat)))
    cross = math.hypot(*(b-a for a, b in zip(reference, candidate)))
    passed = cross == 0 if noise == 0 else (cross <= limit*noise
        or math.isclose(cross, limit*noise, rel_tol=1e-12, abs_tol=0.))
    return dict(name=name, count=len(reference), control_repeat_delta_l2=noise,
                cross_delta_l2=cross,
                cross_self_delta_l2_ratio=cross/noise if noise else (0. if cross == 0 else None),
                zero_noise_cross_nonzero=noise == 0 and cross > 0, within_noise_bound=passed)


def qualify(reference, repeat, candidate, *, policy, expected_head):
    validate_policy(policy, expected_head=expected_head)
    groups = [sorted(rows, key=lambda r: r["rank"]) for rows in (reference, repeat, candidate)]
    wanted = [policy["reference_candidate_id"], policy["reference_candidate_id"], policy["candidate_id"]]
    for rows, cid in zip(groups, wanted):
        summary = summarize_native_two_cycle(rows, expected_head=expected_head, candidate_id=cid)
        require(summary["run_kind"] == "correctness" and not summary["calibration_only"],
                "noise diagnostic requires independent correctness runs")
    cross = compare_native_two_cycle(groups[0], groups[2], expected_head=expected_head)
    self_comparison = compare_native_two_cycle(groups[0], groups[1], expected_head=expected_head)
    for comparison in (cross, self_comparison):
        require(comparison["sample_status"] == comparison["rng_status"] == "PASS",
                "noise cannot excuse sample/RNG mismatch or missing boundaries")
        require(comparison["sampled_state_status"] in ("PASS", "FAIL")
                and all("components" in r["sampled_state"] for r in comparison["ranks"]),
                "noise diagnostic requires complete state samples and matched layouts")
    components = []
    for rank in range(16):
        states = [_state(rows[rank]["state_fingerprints"], rank) for rows in groups]
        for name in states[0]:
            signatures = [(s[name]["inventory_tensor_count"], s[name]["inventory_sha256"],
                           s[name]["sampled_tensor_count"],
                           [{k: v for k, v in row.items() if k != "values_fp32"}
                            for row in s[name]["rows"]]) for s in states]
            require(signatures[0] == signatures[1] == signatures[2],
                    "noise diagnostic requires identical state inventory/dtype/coordinates")
        components.append(states)
    metrics = []
    limit = policy["max_cross_self_delta_l2_ratio"]
    for cycle in range(2):
        for role in ("student", "fake"):
            vectors = [[v for r in rows for v in r["cycles"][cycle]["losses"][role]] for rows in groups]
            metrics.append(_metric(f"cycle{cycle+1}_{role}_loss", *vectors, limit))
    for name in sorted(components[0][0]):
        vectors = [[v for rank in components for row in rank[index][name]["rows"]
                    for v in row["values_fp32"]] for index in range(3)]
        metrics.append(_metric(name, *vectors, limit))
    allowed = all(row["within_noise_bound"] for row in metrics)
    return dict(schema_version=SCHEMA, status=REVIEW if allowed else "FAIL_DIAGNOSTIC_ADMISSION",
                integration_head=expected_head, reference_candidate_id=wanted[0], candidate_id=wanted[2],
                rank_count=16, cycles=2, policy=policy, metrics=metrics,
                strict_comparison=cross, strict_control_repeat_comparison=self_comparison,
                sample_status="PASS", rng_status="PASS", numerical_status="REVIEW",
                diagnostic_search_allowed=allowed, formal_timing_eligible=False,
                strict_numerical_gate="UNCHANGED_ORIGINAL_COMPARISON_PRESERVED",
                full_state_comparison="NOT_RUN", trajectory_equivalence="NOT_ESTABLISHED",
                claim_boundary="SAMPLED_NOISE_QUALIFIED_PLANNER_DIAGNOSTIC_ONLY")


def validate_admission(value):
    """Recompute admission from immutable raw witnesses, not caller status flags."""
    require(value.get("schema_version") == SCHEMA and value.get("status") == REVIEW
            and value.get("diagnostic_search_allowed") is True,
            "noise diagnostic admission failed or incomplete")
    evidence = value.get("native_evidence", {})
    require(set(evidence) == {"reference", "repeat", "candidate"}, "noise diagnostic witnesses missing")
    from .h3_native_bridge import _checked_native
    plans = value.get("execution_plans", {})
    require(set(plans) == set(evidence), "noise diagnostic execution plans missing")
    groups, executions, inodes = [], [], []
    paths = []
    for name in ("reference", "repeat", "candidate"):
        plan_ref = plans[name]
        require(isinstance(plan_ref, dict) and set(plan_ref) == {"path", "sha256"}, "noise execution binding invalid")
        plan_path = Path(plan_ref["path"])
        require(plan_path.is_absolute() and plan_path.is_file()
                and file_sha256(plan_path) == plan_ref["sha256"], "noise diagnostic execution plan changed")
        plan = json.loads(plan_path.read_text())
        require(plan["integration_head"] == value["integration_head"] and plan["mode"] == "correctness",
                "noise diagnostic execution source/mode changed")
        out, _, checked_rows, checked_sources = _checked_native(plan)
        refs = evidence[name]
        require(isinstance(refs, list) and len(refs) == 16, "noise diagnostic requires all sixteen rank witnesses")
        rows = []
        for ref in refs:
            require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "noise witness binding invalid")
            path = Path(ref["path"])
            require(path.is_absolute() and path.is_file() and file_sha256(path) == ref["sha256"],
                    "noise diagnostic witness changed or missing")
            paths.append(path.resolve())
            info = path.stat()
            inodes.append((info.st_dev, info.st_ino))
            rows.append(json.loads(path.read_text()))
        require(refs == checked_sources and rows == checked_rows,
                "noise witnesses do not belong to verified native execution")
        require(Path(out).resolve() == (Path(plan["asset_config"]["run_root"]) / plan["run_id"]).resolve(),
                "noise diagnostic execution output changed")
        executions.append(plan)
        groups.append(rows)
    require(len(set(paths)) == len(set(inodes)) == 48,
            "noise diagnostic requires three independent run witness sets")
    require(len({p["run_id"] for p in executions}) == 3, "noise diagnostic requires three native run identities")
    for field in ("profile_id", "integration_head", "asset_fingerprint", "workload_semantics_hash"):
        require(len({p[field] for p in executions}) == 1, "noise diagnostic execution identities differ")
    require(executions[0]["search_candidate"] == executions[1]["search_candidate"]
            and executions[0]["profile_sha256"] == executions[1]["profile_sha256"],
            "noise diagnostic repeat is not the same native control")
    expected = qualify(*groups, policy=value["policy"], expected_head=value["integration_head"])
    require({k: v for k, v in value.items() if k not in ("native_evidence", "execution_plans")} == expected,
            "noise diagnostic result does not reconcile with raw witnesses")
