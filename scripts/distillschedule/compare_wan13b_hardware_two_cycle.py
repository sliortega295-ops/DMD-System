#!/usr/bin/env python3
"""Fresh hardware-bound semantic admission and unchanged numerical drift report.

Uses actual 160 rollout tensors and global-coordinate state sketches. Sampled
state is explicitly NOT a full-state parity proof. Like the existing coarse
trajectory path, numerical REVIEW permits diagnostic timing, never strict parity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from distillschedule.runtime.p0_parity import tensor_parity_metrics


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(reference_path: Path, candidate_path: Path):
    import torch
    reference = json.loads(reference_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    mismatches = []
    for field in ("expected_head", "hardware_profile", "logical_world_size", "rank_count", "cycles", "final_versions", "matched_contract"):
        if reference.get(field) != candidate.get(field):
            mismatches.append(field)
    for side in (reference, candidate):
        if side.get("status") != "PASS" or side.get("mode") != "correctness":
            mismatches.append("missing passing correctness/health summary")
    ref_rows = {row["rank"]: row for row in reference["ranks"]}
    cand_rows = {row["rank"]: row for row in candidate["ranks"]}
    if ref_rows.keys() != cand_rows.keys():
        mismatches.append("physical ranks")
    for rank in ref_rows.keys() & cand_rows.keys():
        if ref_rows[rank]["cycle_census"] != cand_rows[rank]["cycle_census"]:
            mismatches.append(f"rank{rank}: actual trace/DiT census")
    if mismatches:
        return {"status": "FAIL", "contract": {"status": "FAIL", "mismatches": mismatches}, "complete": False}

    def rollouts(summary):
        result = {}
        for row in summary["ranks"]:
            for artifact in row["rollout_evidence"]:
                path = Path(artifact["path"])
                if file_sha(path) != artifact["sha256"]:
                    raise ValueError("rollout artifact identity changed")
                values = torch.load(path, map_location="cpu", weights_only=True)
                for step, tensor in values.items():
                    key = (artifact["cycle"], artifact["logical_slot"], int(step))
                    if key in result:
                        raise ValueError(f"duplicate logical rollout {key}")
                    result[key] = tensor
        expected = {(cycle, slot, step) for cycle in (1, 2) for slot in range(16) for step in range(10, 15)}
        if result.keys() != expected:
            raise ValueError("incomplete logical rollout evidence")
        return result

    a, b = rollouts(reference), rollouts(candidate)
    numeric = []
    for key in sorted(a):
        metric = tensor_parity_metrics(a[key], b[key])
        numeric.append({"kind": "rollout", "cycle": key[0], "logical_slot": key[1], "step": key[2], **metric})

    def anchor(summary):
        spec = summary["ranks"][0]["state_anchors"]
        value = json.loads(Path(spec["path"]).read_text())
        recorded = value.pop("anchor_sha256")
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(payload).hexdigest() != recorded or recorded != spec["sha256"]:
            raise ValueError("final state anchor identity changed")
        return value["components"]

    aa, bb = anchor(reference), anchor(candidate)
    if aa.keys() != bb.keys():
        raise ValueError("state component inventory mismatch")
    for component in sorted(aa):
        first, second = aa[component]["numeric_sketch"], bb[component]["numeric_sketch"]
        # Representation differs by design; coordinates must not.
        for key in ("inventory_sha256", "inventory_tensor_count", "tensor_count", "coordinate_count"):
            if first[key] != second[key]:
                raise ValueError(f"state sketch {component}/{key} mismatch")
        def indexed(sketch):
            return {(r["tensor"], r["flat_index"]): r["value_fp32"] for r in sketch["coordinates"]}
        av, bv = indexed(first), indexed(second)
        if av.keys() != bv.keys() or len(av) != first["coordinate_count"]:
            raise ValueError("state sketch coordinate mismatch")
        keys = sorted(av)
        numeric.append({"kind": "sampled_state", "component": component,
                        **tensor_parity_metrics(torch.tensor([av[k] for k in keys], dtype=torch.float32),
                                                torch.tensor([bv[k] for k in keys], dtype=torch.float32))})
    if any(not r["finite"] or not r["same_shape"] or not r["same_dtype"] for r in numeric):
        raise ValueError("finite/shape/dtype contract failed")
    passed = all(r["status"] == "PASS" for r in numeric)
    return {
        "schema_version": "distillschedule.trajectory_coarse_comparison.v1",
        "comparison_kind": "semantic", "expected_cycles": 2,
        "expected_world_size": reference["rank_count"], "hardware_profile": reference["hardware_profile"],
        "expected_head": reference["expected_head"], "complete": True,
        "status": "HARDWARE_TWO_CYCLE_PASS" if passed else "HARDWARE_TWO_CYCLE_REVIEW",
        "contract": {"status": "PASS", "mismatches": []},
        "reference": {"variants": [reference["implementation_contract"]["trajectory_arm_id"]], "run_ids": [reference["run_id"]]},
        "candidate": {"variants": [candidate["implementation_contract"]["trajectory_arm_id"]], "run_ids": [candidate["run_id"]]},
        "summary_sha256": {"reference": file_sha(reference_path), "candidate": file_sha(candidate_path)},
        "numerical_status": "PASS_OBSERVED_ROLLOUTS_AND_SAMPLED_STATE" if passed else "REVIEW",
        "strict_full_state_parity": "NOT_RUN", "strict_claimable_for_paired_timing": False,
        "diagnostic_timing_eligible": True,
        "numeric_metrics": numeric,
        "claim_boundary": "same-head H100 logical-B16 controlled diagnostic comparison; original strict thresholds unchanged; sampled state is not full-state proof",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.reference_summary, args.candidate_summary)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "numeric_metrics"}, indent=2))
    return 0 if result.get("contract", {}).get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
