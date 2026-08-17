from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.lightx2v_minimax_matched import (
    EXPECTED_BACKWARD_COUNTS,
    EXPECTED_FORWARD_COUNTS,
    EXPECTED_GRAD_FORWARD_COUNTS,
    expected_application_inventory,
    validate_matched_receipt,
)

RUNTIME = ROOT / "runtime/lightx2v_minimax_matched.py"
ENTRY = ROOT / "scripts/distillschedule/lightx2v_minimax_matched_entry.py"
PAIR = ROOT / "scripts/distillschedule/run_lightx2v_minimax_matched_pair.py"
CONFIG = ROOT / "configs/distillschedule/baselines/lightx2v_minimax_h3_world16_dmd.yaml"


def test_matched_minimax_application_inventory_is_exact() -> None:
    assert EXPECTED_FORWARD_COUNTS == {"student": 24, "fake": 6, "teacher": 1}
    assert EXPECTED_GRAD_FORWARD_COUNTS == {"student": 1, "fake": 5, "teacher": 0}
    assert EXPECTED_BACKWARD_COUNTS == {"student": 1, "fake": 5}
    value = expected_application_inventory()
    assert value["total_forwards"] == 31
    assert value["total_backward_graphs"] == 6
    assert value["fixed_end_step_idx"] == 3


def test_matched_receipt_gate_rejects_compute_drift() -> None:
    receipt = {
        "status": "PASS",
        "application_inventory": expected_application_inventory(),
    }
    assert validate_matched_receipt(receipt) == []
    receipt = json.loads(json.dumps(receipt))
    receipt["application_inventory"]["forward_counts"]["student"] = 23
    assert validate_matched_receipt(receipt)


def test_distillschedule_candidate_only_reorders_fake_stage_student_rollouts() -> None:
    source = RUNTIME.read_text()
    assert 'self._benchmark_phase_order.append("student_update")' in source
    assert 'self._benchmark_phase_order.append(f"student_nograd_rollout_{fake_index}")' in source
    assert 'self._benchmark_phase_order.append(f"fake_update_{fake_index}")' in source
    assert "self.run_back_simulation(" in source
    assert "weighted_mse_pair(" in source
    assert "self.fake_optimizer.step()" in source
    assert "Fake compute changed global RNG state" in source
    assert "state_hashes" in source


def test_native_and_candidate_share_the_same_lightx2v_model_and_config() -> None:
    entry = ENTRY.read_text()
    assert "from lightx2v_train.model_zoo import build_model" in entry
    assert "create_minimax_matched_trainer" in entry
    config = CONFIG.read_text()
    assert "method: minimax_h3_t2av_dmd" in config
    assert "fake_update_ratio: 5" in config
    assert "num_inference_steps: 4" in config
    assert "gradient_accumulation_iters: 1" in config


def test_pair_runner_is_sequential_not_concurrent() -> None:
    source = PAIR.read_text()
    assert "for command in commands:" in source
    assert "subprocess.run(command, check=True)" in source
    assert "Popen" not in source
    assert 'choices=("native-first", "distillschedule-first")' in source
