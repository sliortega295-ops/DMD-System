from __future__ import annotations

from pathlib import Path

import pytest

from distillschedule.adapters.livewan.adapter import LiveWanSFDMDAdapter
from distillschedule.core.config import load_plan
from distillschedule.runtime.livewan_crossscale_worker import (
    _CheckpointRngElisionProxy,
)


ROOT = Path(__file__).resolve().parents[1]
C21 = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c21_continuous_b16.json"
C22 = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c22_critic_checkpoint_rng_elision_continuous_b16.json"


def _fake_evidence_environment(monkeypatch) -> None:
    for key in (
        "LIVEWAN_STUDENT_CHECKPOINT",
        "LIVEWAN_CRITIC_CHECKPOINT",
        "LIVEWAN_TEACHER_CHECKPOINT",
        "LIVEWAN_OPERATION_TRACE",
        "LIVEWAN_SYSTEMS_BOOTSTRAP",
        "DISTILLSCHEDULE_OUTPUT_DIR",
    ):
        monkeypatch.setenv(key, f"/tmp/{key.lower()}")


def test_c22_changes_only_registered_checkpoint_rng_metadata(monkeypatch) -> None:
    _fake_evidence_environment(monkeypatch)
    reference = load_plan(C21).plan.to_dict()
    candidate_plan = load_plan(C22).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    candidate = candidate_plan.to_dict()

    metadata = dict(candidate["metadata"])
    assert metadata.pop("critic_checkpoint_rng") == {
        "enabled": True,
        "name": "critic_checkpoint_rng_state_elision_v1",
        "scope": "score_rank_critic_grad_checkpoint_only",
        "use_reentrant": False,
        "preserve_rng_state": False,
        "checkpoint_block_stochastic_ops": False,
        "generator_checkpoint_unchanged": True,
        "teacher_checkpoint_unchanged": True,
        "math_and_call_inventory_preserved": True,
    }
    assert metadata == reference["metadata"]
    for key in (
        "workload",
        "hardware",
        "rank_groups",
        "lifecycle",
        "training_backends",
    ):
        assert candidate[key] == reference[key]


def test_checkpoint_proxy_forces_only_rng_preservation_flag() -> None:
    class FakeCheckpointModule:
        def __init__(self) -> None:
            self.calls = []

        def checkpoint(self, function, *args, **kwargs):
            self.calls.append((function, args, kwargs))
            return function(*args)

    module = FakeCheckpointModule()
    proxy = _CheckpointRngElisionProxy(module)
    result = proxy.checkpoint(lambda value: value + 1, 4, use_reentrant=False)
    assert result == 5
    assert module.calls[0][2] == {
        "use_reentrant": False,
        "preserve_rng_state": False,
    }
    assert proxy.receipt(expected_calls=1)["execution_status"] == "PASS"

    with pytest.raises(RuntimeError, match="controls RNG preservation"):
        proxy.checkpoint(
            lambda value: value,
            4,
            use_reentrant=False,
            preserve_rng_state=True,
        )


def test_c22_is_wired_to_controller_launcher_and_continuous_summarizer() -> None:
    controller = (
        ROOT / "scripts/distillschedule/run_livewan_crossscale_world16.py"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh"
    ).read_text(encoding="utf-8")
    summarizer = (
        ROOT / "scripts/distillschedule/summarize_livewan_continuous_world16.py"
    ).read_text(encoding="utf-8")
    for source in (controller, launcher):
        assert "c22-critic-checkpoint-rng" in source
    assert "critic_checkpoint_rng" in summarizer
    assert "expected_checkpoint_calls" in summarizer
