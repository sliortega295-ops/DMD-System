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
EXPECTED_PATCH = "99a66efb5984eb46bd00ae2760b57088605b52f255449dad48a7c9fc47d0e4dc"


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
    assert "dmd_loss_pair" not in text
    assert "weighted_mse_pair" not in text


def test_cross_workload_speedup_is_explicitly_forbidden() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["comparison"]["helios_pr16_absolute_time_is_cross_workload_only"] is True
    assert "speedup" in README.read_text().lower()
