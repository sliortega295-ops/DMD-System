from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wan_hw_summary", ROOT / "scripts/distillschedule/summarize_wan13b_hardware_two_cycle.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root):
    ranks = root / "execution/ranks"
    ranks.mkdir(parents=True)
    health = {"status": "PASS", "exit_code": 0, "cgroup_oom_delta": {"oom": 0, "oom_kill": 0}, "gpu_processes_after": []}
    (root / "run_status.json").write_text(json.dumps(health))
    anchor = {"cycle": 2, "components": {}}
    anchor["anchor_sha256"] = module.digest(anchor)
    anchor_path = root / "anchor.json"
    anchor_path.write_text(json.dumps(anchor))
    artifact_path = root / "rollout.pt"
    artifact_path.write_bytes(b"unit-test fixture, not real model evidence")
    artifact = {"path": str(artifact_path), "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest()}
    inventory = [{"call_count": 4}] * 6
    leaf = {"generator_score_forwards": 3, "critic_score_forwards": 5,
            "physical_rollout_batches": 5, "rollout_batch_size": 1,
            "rollout_call_inventory": inventory, "sample_payload_identities": ["a"] * 6}
    dtype = {"records": [{"base_weight_dtype": "torch.bfloat16", "lora_a_weight_dtype": "torch.float32", "lora_b_weight_dtype": "torch.float32"}]}
    for rank in range(8):
        census = {"generator_updates": 1, "critic_updates": 5, "ema_updates": 5,
                  "logical_slots": [rank, rank+8], "logical_sample_census": [leaf, leaf]}
        row = {"rank": rank, "status": "PASS", "world_size": 8, "hardware_profile": "h100-8-b16",
               "logical_world_size": 16, "logical_slots": [rank, rank+8],
               "final_versions": {"generator": 4, "critic": 20, "ema": 20},
               "finite_status": [True, True], "updates": {"generator": 2, "critic": 10, "ema": 10},
               "cycle_census": [census, census], "mode": "correctness", "profiled": False,
               "cycles": [{"local_elapsed_seconds": 2.0}, {"local_elapsed_seconds": 1.0}],
               "trace_rollout_contract": [{"inventory": item} for item in inventory],
               "rollout_evidence": [artifact] * 4, "build": {"generator": {"lora_compute_dtype_contract": dtype}, "critic": {"lora_compute_dtype_contract": dtype}},
               "state_anchors": {"path": str(anchor_path), "sha256": anchor["anchor_sha256"]},
               "expected_head": "a" * 40, "placement": "world16-fsdp", "matched_contract": {}, "arm_contract": {}}
        (ranks / f"rank_{rank:02d}.json").write_text(json.dumps(row))


def test_actual_hardware_receipts_pass_without_claiming_numerical_parity(tmp_path):
    fixture(tmp_path)
    result = module.summarize(tmp_path, profile_name="h100-8-b16")
    assert result["status"] == "PASS", result["errors"]
    assert result["numeric_comparison"] == "NOT_RUN"
    assert result["formal_timing_eligible"] is False
    assert result["cycle_seconds_max_rank"] == [2.0, 1.0]


def test_missing_rank_or_cleanup_or_dtype_fails_closed(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "execution/ranks/rank_07.json"
    path.unlink()
    (tmp_path / "run_status.json").unlink()
    result = module.summarize(tmp_path, profile_name="h100-8-b16")
    assert result["status"] == "FAIL"
    assert any("rank coverage" in error for error in result["errors"])
    assert any("teardown" in error for error in result["errors"])


def test_changed_artifact_or_compute_dtype_cannot_pass(tmp_path):
    fixture(tmp_path)
    (tmp_path / "rollout.pt").write_bytes(b"corrupt")
    path = tmp_path / "execution/ranks/rank_00.json"
    row = json.loads(path.read_text())
    row["build"]["generator"]["lora_compute_dtype_contract"]["records"][0]["lora_a_weight_dtype"] = "torch.bfloat16"
    path.write_text(json.dumps(row))
    result = module.summarize(tmp_path, profile_name="h100-8-b16")
    assert result["status"] == "FAIL"
    assert any("hash mismatch" in error for error in result["errors"])
    assert any("dtype" in error for error in result["errors"])
