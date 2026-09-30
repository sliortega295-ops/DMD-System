from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wan_hw_compare", ROOT / "scripts/distillschedule/compare_wan13b_hardware_two_cycle.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def summaries(root, delta=0.0):
    torch = pytest.importorskip("torch")
    paths = []
    for side, offset in (("reference", 0.0), ("candidate", delta)):
        folder = root / side
        folder.mkdir()
        sketch = {"inventory_sha256": "a" * 64, "inventory_tensor_count": 1,
                  "tensor_count": 1, "coordinate_count": 2,
                  "coordinates": [{"tensor": "lora", "flat_index": i, "value_fp32": 1.0 + offset} for i in range(2)]}
        anchor = {"cycle": 2, "components": {"generator_trainable": {"numeric_sketch": sketch}}}
        canonical = json.dumps(anchor, sort_keys=True, separators=(",", ":")).encode()
        anchor["anchor_sha256"] = hashlib.sha256(canonical).hexdigest()
        ap = folder / "anchor.json"
        ap.write_text(json.dumps(anchor))
        ranks = []
        for rank in range(8):
            artifacts = []
            for cycle in (1, 2):
                for slot in (rank, rank+8):
                    path = folder / f"cycle{cycle}_slot{slot}.pt"
                    torch.save({step: torch.ones(4) + offset for step in range(10, 15)}, path)
                    artifacts.append({"cycle": cycle, "logical_slot": slot, "path": str(path), "sha256": module.file_sha(path)})
            ranks.append({"rank": rank, "cycle_census": {"trace": "a" * 64},
                          "rollout_evidence": artifacts, "state_anchors": {"path": str(ap), "sha256": anchor["anchor_sha256"]}})
        row = {"status": "PASS", "mode": "correctness", "rank_count": 8,
               "expected_head": "a" * 40, "hardware_profile": "h100-8-b16", "logical_world_size": 16,
               "cycles": 2, "matched_contract": {}, "final_versions": {"generator": 4, "critic": 20, "ema": 20},
               "implementation_contract": {"trajectory_arm_id": side}, "run_id": side, "ranks": ranks}
        sp = folder / "summary.json"
        sp.write_text(json.dumps(row))
        paths.append(sp)
    return paths


def test_same_data_passes_observed_scope_not_full_state(tmp_path):
    a, b = summaries(tmp_path)
    result = module.compare(a, b)
    assert result["status"] == "HARDWARE_TWO_CYCLE_PASS"
    assert len(result["numeric_metrics"]) == 161
    assert result["strict_full_state_parity"] == "NOT_RUN"
    assert result["strict_claimable_for_paired_timing"] is False


def test_numeric_drift_is_recorded_without_relaxing_thresholds(tmp_path):
    a, b = summaries(tmp_path, 0.2)
    result = module.compare(a, b)
    assert result["status"] == "HARDWARE_TWO_CYCLE_REVIEW"
    assert result["contract"]["status"] == "PASS"
    assert result["diagnostic_timing_eligible"] is True
    assert result["numeric_metrics"][0]["thresholds"]["normalized_l2"] == 5e-3


def test_source_or_census_changes_cannot_be_admitted(tmp_path):
    a, b = summaries(tmp_path)
    row = json.loads(b.read_text())
    row["ranks"][0]["cycle_census"] = {"trace": "wrong"}
    b.write_text(json.dumps(row))
    result = module.compare(a, b)
    assert result["status"] == "FAIL"
    assert result["complete"] is False
