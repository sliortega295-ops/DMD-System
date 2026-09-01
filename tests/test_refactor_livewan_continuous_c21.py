from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from distillschedule.core.config import load_plan
from distillschedule.adapters.livewan.adapter import LiveWanSFDMDAdapter
from distillschedule.runtime.livewan_critic_shadow_sync import (
    layout_sha256,
    parameter_layout,
)
from distillschedule.runtime.livewan_host_runahead import PinnedHostRunaheadTransport


ROOT = Path(__file__).resolve().parents[1]


def _load_controller():
    path = ROOT / "scripts/distillschedule/run_livewan_crossscale_world16.py"
    spec = importlib.util.spec_from_file_location("livewan_c21_controller", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
C20 = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c20_generator_flash_attn2.json"
C21 = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c21_continuous_b16.json"


def test_c21_changes_only_registered_continuous_schedule_metadata(monkeypatch) -> None:
    for key in (
        "LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT",
        "LIVEWAN_TEACHER_CHECKPOINT", "LIVEWAN_OPERATION_TRACE",
        "LIVEWAN_SYSTEMS_BOOTSTRAP", "DISTILLSCHEDULE_OUTPUT_DIR",
    ):
        monkeypatch.setenv(key, f"/tmp/{key.lower()}")
    reference = load_plan(C20).plan.to_dict()
    candidate_plan = load_plan(C21).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    candidate = candidate_plan.to_dict()
    continuous = dict(candidate["metadata"])
    assert continuous.pop("continuous_pipeline") == {
        "enabled": True,
        "name": "continuous_full5_b16_depth1_v1",
        "cycles": 3,
        "cycle_length": 5,
        "lookahead_depth": 1,
        "lookahead_scope": "next_cycle_step0_wave0_rollout_only",
        "source_reuse_across_cycles": True,
        "fresh_rank_qualified_rng_per_cycle": True,
        "critic_shadow_hot_sync": "paired_score_to_generator_fp32_lora_after_c5",
        "critic_shadow_sync_versions": [5, 10],
        "single_generator_commit_per_cycle": True,
        "five_critic_commits_per_cycle": True,
        "math_and_call_inventory_preserved": True,
    }
    assert continuous == reference["metadata"]
    for key in ("workload", "hardware", "rank_groups", "lifecycle", "training_backends"):
        assert candidate[key] == reference[key]


def test_continuous_host_transport_has_bounded_exact_key_inventory() -> None:
    instance = SimpleNamespace(cycles=3, cross_cycle_lookahead=True)
    keys = PinnedHostRunaheadTransport.expected_keys.fget(instance)
    assert len(keys) == 26
    assert {(5, 0), (10, 0)}.issubset(keys)
    assert (5, 1) not in keys and (10, 1) not in keys
    assert (0, 0) not in keys


def test_critic_shadow_layout_hash_is_order_and_shape_sensitive() -> None:
    class Parameter:
        def __init__(self, shape, dtype="torch.float32") -> None:
            self.shape = shape
            self.dtype = dtype

        def numel(self) -> int:
            result = 1
            for value in self.shape:
                result *= value
            return result

    first = parameter_layout([Parameter((2, 3)), Parameter((4,))])
    same = parameter_layout([Parameter((2, 3)), Parameter((4,))])
    different = parameter_layout([Parameter((4,)), Parameter((2, 3))])
    assert first == same
    assert layout_sha256(first) == layout_sha256(same)
    assert layout_sha256(first) != layout_sha256(different)


def test_depth_one_wait_precedes_next_cycle_wave_one() -> None:
    source = (
        ROOT / "src/distillschedule/runtime/livewan_crossscale_worker.py"
    ).read_text(encoding="utf-8")
    receive = source.index("critic_shadow_receive_wait_before_wave1")
    wave_one = source.index(
        'label=f"livewan/pipeline/g/step_{step}/rollout_wave_1"', receive
    )
    assert receive < wave_one
    assert "_run_pipeline_8x8_continuous" in source
    assert "finalize_host_transport=False" in source


def test_controller_launcher_and_probe_register_c21() -> None:
    controller = (ROOT / "scripts/distillschedule/run_livewan_crossscale_world16.py").read_text()
    launcher = (ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh").read_text()
    probe = (ROOT / "scripts/distillschedule/probe_livewan_crossscale_transport.py").read_text()
    assert '"c21-continuous-b16"' in controller
    assert "livewan_wan13b_14b_sf_dmd_c21_continuous_b16.json" in launcher
    assert 'args.pipeline_variant == "c21-continuous-b16"' in probe
    assert "paired Critic-shadow broadcast mismatch" in probe


def test_controller_extracts_json_from_mandatory_pty_banner() -> None:
    controller = _load_controller()
    payload = b'{"status":"PASS"}\n'
    raw = (
        b"platform banner\r\n"
        + controller._PAYLOAD_BEGIN
        + b"\r\n"
        + base64.b64encode(payload)
        + b"\r\n"
        + controller._PAYLOAD_END
        + b"\r\nConnection closed\r\n"
    )
    assert controller.decode_remote_payload(raw) == payload
