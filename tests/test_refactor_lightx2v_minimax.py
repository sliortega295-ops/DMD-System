from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/distillschedule/baselines/lightx2v_minimax_h3_world16_dmd_contract.json"
CONFIG = ROOT / "configs/distillschedule/baselines/lightx2v_minimax_h3_world16_dmd.yaml"
PATCH = ROOT / "third_party/lightx2v_benchmark/benchmark.patch"
README = ROOT / "third_party/lightx2v_benchmark/README.md"

EXPECTED_UPSTREAM = "d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be"
EXPECTED_PATCH = "89935efd36d8a149f121b428cad700363d987918718b81b1ea56b887f196d9c3"


def test_lightx2v_minimax_contract_is_world16_g1f5() -> None:
    value = json.loads(CONTRACT.read_text())
    assert value["upstream"]["commit"] == EXPECTED_UPSTREAM
    assert value["hardware"]["world_size"] == 16
    assert value["parallel"]["fsdp2_size"] == 16
    assert value["parallel"]["sequence_parallel_size"] == 1
    assert value["cycle"]["student_updates"] == 1
    assert value["cycle"]["fake_updates"] == 5
    assert value["cycle"]["ema_updates"] == 0
    assert value["cycle"]["benchmark_end_step_idx"] == 3
    assert value["cycle"]["model_evaluations_per_back_simulation"] == 4
    assert value["cycle"]["global_batch"] == 16
    assert value["cycle"]["global_batches_consumed"] == 6
    assert value["cycle"]["minimum_unique_cache_samples"] == 96


def test_lightx2v_minimax_config_preserves_upstream_semantics() -> None:
    text = CONFIG.read_text()
    for needle in (
        "method: minimax_h3_t2av_dmd",
        "fake_update_ratio: 5",
        "num_inference_steps: 4",
        "height: 768",
        "width: 1344",
        "num_frames: 124",
        "rank: 128",
        "enabled: false\n        size: 1",
        "enabled: true\n        size: ${oc.env:WORLD_SIZE,16}",
        "max_train_iters: 1",
        "auto_resume: false"
    ):
        assert needle in text


def test_lightx2v_benchmark_patch_identity_and_scope() -> None:
    data = PATCH.read_bytes()
    assert hashlib.sha256(data).hexdigest() == EXPECTED_PATCH
    text = data.decode()
    assert "LIGHTX2V_BENCHMARK_END_STEP_IDX" in text
    assert "LIGHTX2V_BENCHMARK_RECEIPT_DIR" in text
    assert "LIGHTX2V_DMD_CYCLE" in text
    assert "LIGHTX2V_MINIMAX_FSDP_FIRST_INIT" in text
    assert "transformer.to(self.device)" in text
    assert "dmd_loss_pair" not in text
    assert "weighted_mse_pair" not in text


def test_fsdp_first_construction_is_opt_in_and_shared_by_benchmark_launchers() -> None:
    model_patch = PATCH.read_text()
    launcher = (ROOT / "scripts/distillschedule/launch_lightx2v_minimax_matched_world16_node.sh").read_text()
    gate = (ROOT / "scripts/distillschedule/launch_lightx2v_minimax_fsdp_first_node.sh").read_text()
    assert "and bool(fsdp2_config.get(\"enabled\", False))" in model_patch
    assert 'if not self._fsdp2_first_init:' in model_patch
    assert 'export LIGHTX2V_MINIMAX_FSDP_FIRST_INIT="${LIGHTX2V_MINIMAX_FSDP_FIRST_INIT:-1}"' in launcher
    assert 'export LIGHTX2V_MINIMAX_FSDP_FIRST_INIT=1' in gate
    assert "after_cpu_load" in (ROOT / "scripts/distillschedule/lightx2v_minimax_fsdp_first_construction.py").read_text()


def test_cross_workload_speedup_is_explicitly_forbidden() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["comparison"]["helios_pr16_absolute_time_is_cross_workload_only"] is True
    readme = README.read_text().lower()
    assert "does **not** itself claim a speedup" in readme
    assert "same minimax contract" in readme
