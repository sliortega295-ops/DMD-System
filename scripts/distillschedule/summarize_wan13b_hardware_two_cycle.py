#!/usr/bin/env python3
"""Admission receipts for an explicit physical mesh and sixteen logical rows."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from distillschedule.runtime.hardware_profile import hardware_profile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summarize(run_root: Path, *, profile_name: str):
    profile = hardware_profile(profile_name)
    files = sorted((run_root / "execution/ranks").glob("rank_*.json"))
    errors = []
    rows = [json.loads(path.read_text()) for path in files]
    ranks = [row.get("rank") for row in rows]
    if sorted(ranks) != list(range(profile.physical_world_size)):
        errors.append("incomplete or duplicate physical rank coverage")
    all_slots = []
    final = {"generator": 4, "critic": 20, "ema": 20}
    health_path = run_root / "run_status.json"
    health = json.loads(health_path.read_text()) if health_path.exists() else {}
    if health.get("status") != "PASS" or health.get("exit_code") != 0:
        errors.append("missing or failed launcher/teardown receipt")
    if health.get("cgroup_oom_delta") != {"oom": 0, "oom_kill": 0}:
        errors.append("cgroup OOM evidence missing or increased")
    if health.get("gpu_processes_after") != []:
        errors.append("GPU cleanup evidence missing or not empty")
    for row in rows:
        rank = row["rank"]
        if row.get("status") != "PASS" or row.get("world_size") != profile.physical_world_size:
            errors.append(f"rank{rank}: workload/physical world failed")
        if row.get("hardware_profile") != profile.name or row.get("logical_world_size") != 16:
            errors.append(f"rank{rank}: hardware/logical contract changed")
        if row.get("logical_slots") != list(profile.logical_slots(rank)):
            errors.append(f"rank{rank}: logical slot mapping changed")
        all_slots.extend(row.get("logical_slots", []))
        if row.get("final_versions") != final or row.get("finite_status") != [True, True]:
            errors.append(f"rank{rank}: version/finite gate failed")
        if row.get("updates") != {"generator": 2, "critic": 10, "ema": 10}:
            errors.append(f"rank{rank}: optimizer/EMA update count changed")
        censuses = row.get("cycle_census", [])
        if len(censuses) != 2:
            errors.append(f"rank{rank}: missing cycle census")
        for census in censuses:
            if any(census.get(k) != v for k, v in {"generator_updates": 1, "critic_updates": 5, "ema_updates": 5}.items()):
                errors.append(f"rank{rank}: cycle commit census changed")
            if profile.microbatch_count > 1:
                if census.get("logical_slots") != list(profile.logical_slots(rank)) or len(census.get("logical_sample_census", [])) != profile.microbatch_count:
                    errors.append(f"rank{rank}: logical contribution census changed")
            leaves = census.get("logical_sample_census", [census])
            for leaf in leaves:
                if leaf.get("generator_score_forwards") != 3 or leaf.get("critic_score_forwards") != 5:
                    errors.append(f"rank{rank}: physical Score call census changed")
                if leaf.get("physical_rollout_batches") != 5 or leaf.get("rollout_batch_size") != 1:
                    errors.append(f"rank{rank}: B1 rollout census changed")
                if len(leaf.get("rollout_call_inventory", [])) != 6 or len(leaf.get("sample_payload_identities", [])) != 6:
                    errors.append(f"rank{rank}: missing DiT/trace consumption inventory")
                expected_inventory = [item["inventory"] for item in row.get("trace_rollout_contract", [])]
                if leaf.get("rollout_call_inventory") != expected_inventory:
                    errors.append(f"rank{rank}: consumed DiT inventory differs from immutable trace")
        cycles = row.get("cycles", [])
        if len(cycles) != 2 or any(not math.isfinite(c.get("local_elapsed_seconds", float("nan"))) or c.get("local_elapsed_seconds", 0) <= 0 for c in cycles):
            errors.append(f"rank{rank}: invalid timing receipt")
        if row.get("profiled") is not False:
            errors.append(f"rank{rank}: native receipt lacks unprofiled evidence")
        norm_enabled = row.get("arm_contract", {}).get("modulated_norm_fusion", False)
        norm_receipt = row.get("modulated_norm_fusion_receipt", {})
        if norm_receipt.get("enabled") is not norm_enabled:
            errors.append(f"rank{rank}: missing actual normalization policy receipt")
        if norm_enabled:
            # The installer is process-shared; role aliases are not independent
            # executions and must not be summed into an inflated call count.
            from distillschedule.runtime.wan13b_norm_policy import validate
            try:
                validate({role: {"metadata": {"modulated_norm_fusion": value}}
                          for role, value in norm_receipt.get("roles", {}).items()}, enabled=True)
            except AssertionError as error:
                errors.append(f"rank{rank}: normalization runtime counter gate: {error}")
        if row.get("mode") == "correctness":
            if len(row.get("rollout_evidence", [])) != 2 * profile.microbatch_count:
                errors.append(f"rank{rank}: missing actual rollout evidence")
            build = row.get("build", {})
            for role in ("generator", "critic"):
                dtype = build.get(role, {}).get("lora_compute_dtype_contract", {})
                observations = dtype.get("records")
                if not isinstance(observations, list) or not observations:
                    errors.append(f"rank{rank}/{role}: missing runtime dtype observations")
                elif any(o.get("base_weight_dtype") != "torch.bfloat16" or o.get("lora_a_weight_dtype") != "torch.float32" or o.get("lora_b_weight_dtype") != "torch.float32" for o in observations):
                    errors.append(f"rank{rank}/{role}: actual compute dtype gate failed")
            for artifact in row.get("rollout_evidence", []):
                path = Path(artifact["path"])
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact.get("sha256"):
                    errors.append(f"rank{rank}: rollout artifact missing or hash mismatch")
            anchor = row.get("state_anchors", {})
            anchor_path = Path(anchor.get("path", "/__missing_anchor__"))
            if not anchor_path.is_file():
                errors.append(f"rank{rank}: final state evidence missing")
            else:
                payload = json.loads(anchor_path.read_text())
                recorded = payload.pop("anchor_sha256", None)
                if digest(payload) != recorded or recorded != anchor.get("sha256"):
                    errors.append(f"rank{rank}: final state artifact hash mismatch")
    if sorted(all_slots) != list(range(16)):
        errors.append("logical slot coverage is not exactly sixteen")
    for field in ("expected_head", "placement", "matched_contract", "arm_contract", "mode"):
        if rows and any(row.get(field) != rows[0].get(field) for row in rows):
            errors.append(f"cross-rank {field} mismatch")
    sample = rows[0] if rows else {}
    matched = sample.get("matched_contract", {})
    implementation = dict(sample.get("arm_contract", {}),
                          hardware_profile=profile.name,
                          microbatch_count=profile.microbatch_count,
                          trajectory_arm_id=sample.get("timing_arm_id"))
    result = {
        "schema_version": "distillschedule.wan13b_trajectory_summary.v1",
        "status": "FAIL" if errors else "PASS", "errors": errors,
        "run_id": sample.get("run_id"), "expected_head": sample.get("expected_head"),
        "placement": sample.get("placement"), "rank_count": len(rows),
        "mode": sample.get("mode"),
        "hardware_profile": profile.name, "logical_world_size": 16,
        "logical_slots": all_slots, "cycles": 2, "final_versions": final,
        "diagnostic_only": sample.get("mode") != "timing",
        "formal_timing_eligible": not errors and sample.get("mode") == "timing",
        "numeric_comparison": "NOT_RUN",
        "launcher_health": health,
        "matched_contract": matched, "workload_fingerprint": digest(matched),
        "implementation_contract": implementation, "implementation_fingerprint": digest(implementation),
        "ranks": rows,
        "cycle_seconds_max_rank": [max((row["cycles"][index]["local_elapsed_seconds"] for row in rows if len(row.get("cycles", [])) == 2), default=None) for index in range(2)],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--hardware-profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(args.run_root, profile_name=args.hardware_profile)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("ranks", "matched_contract", "implementation_contract")}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
