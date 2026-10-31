"""Supported typed adapters; retired Helios launch plans cannot execute."""
from pathlib import Path
import pytest
from distillschedule.adapters import default_registry
from distillschedule.core.config import load_plan
from distillschedule.runtime import DistillExecutor

ROOT = Path(__file__).resolve().parents[1]


def test_generic_adapter_compiles_vllm_plan_without_importing_vllm():
    loaded = load_plan(ROOT / 'configs/distillschedule/plans/generic_autoregressive_dmd_vllm_critic_only.json')
    compiled = DistillExecutor().compile(loaded, repo_root=ROOT)
    assert compiled.adapter_plan.invocation is None
    assert compiled.adapter_plan.protocol.steps[0].intent.kind.value == 'critic_only'
    assert 'plan-only' in compiled.adapter_plan.warnings[0]


def test_retired_helios_launcher_is_not_registered():
    with pytest.raises(KeyError, match='unknown adapter'):
        default_registry().get('helios')


def test_current_helios_reference_helpers_remain_importable():
    from distillschedule.adapters.helios import reference_model, reference_helpers
    assert callable(reference_model.load_reference_model_class)
    assert callable(reference_helpers.load_reference_primitives)
