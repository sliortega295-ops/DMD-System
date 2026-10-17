"""Versioned opt-in extension; original bounded2 rejects remain unchanged."""
from dataclasses import replace
import pytest
from distillschedule.search.helios_implementation import HeliosImplementation, current_implementation, MODE_ENV
from distillschedule.search.helios_model_adapter import safe_incumbent


def test_tail4_source_lowering_roundtrip_and_same_training_contract():
    control = replace(safe_incumbent(), retained_critic_blocks=(38, 39))
    child = replace(control, retained_critic_blocks=(36, 37, 38, 39), retention_policy="tail4.v1")
    assert child.id.endswith("-rc36_37_38_39-rp-tail4-v1")
    assert HeliosImplementation.parse(child.id) == child
    assert HeliosImplementation.from_assignment(child.assignment) == child
    env = child.environment(); env[MODE_ENV] = "native"
    assert current_implementation(env) == child
    assert child.lower_profile()["contract"] == control.lower_profile()["contract"]
    assert child.lower_profile()["runtime"]["planner_implementation"] == child.assignment
    assert "retention_policy" not in control.assignment
    assert not safe_incumbent().retained_critic_blocks


@pytest.mark.parametrize("blocks", [(), (39,), (38, 39), (0, 1, 2, 3), (35, 36, 37, 38, 39)])
def test_tail4_cannot_widen_or_hide_other_policy(blocks):
    with pytest.raises(Exception):
        replace(safe_incumbent(), retained_critic_blocks=blocks, retention_policy="tail4.v1")


def test_legacy_three_or_four_still_rejected_and_no_calibration_reuse():
    for blocks in ((37, 38, 39), (36, 37, 38, 39)):
        with pytest.raises(Exception):
            replace(safe_incumbent(), retained_critic_blocks=blocks)
    child = replace(safe_incumbent(), retained_critic_blocks=(36, 37, 38, 39), retention_policy="tail4.v1")
    env = child.environment(); env[MODE_ENV] = "calibration"
    with pytest.raises(Exception, match="uncalibrated"):
        current_implementation(env)
    with pytest.raises(Exception):
        replace(child, variant="gan")
