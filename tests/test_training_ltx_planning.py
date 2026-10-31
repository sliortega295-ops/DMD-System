"""Actual tiny LTX AV models through the existing measured common planner.

Procedural media/random native models only, not pretrained or GPU evidence.
Reuse the same execution/recovery assertions as the other flow-DMD adapters;
run in the separate pinned LTX environment, never the modern-flow TF5 lane.
"""
from pathlib import Path

import pytest

import test_training_modern_flow_planning_cpu as checks
from test_training_ltx_cpu import source_root  # existing explicit optional-source fixture


@pytest.fixture(scope='module')
def native_assets(tmp_path_factory, source_root):
    from distillschedule.training.plugins import load_callable
    repo = Path(__file__).resolve().parents[1]
    build, _ = load_callable(str(repo/'recipes/ltx25/create_cpu_fixture.py')+':build_fixture')
    result = build(tmp_path_factory.mktemp('ltx-plan')/'fixture', source_root, cycles=3)
    return dict(result, root=Path(result['assets']), assets=Path(result['assets'])/'assets.json',
                config=result['lora_config'], family='ltx25', planning_limit_bytes=100_000_000)


@pytest.fixture(scope='module')
def native_plan(native_assets, tmp_path_factory):
    return checks.measured_plan(native_assets, tmp_path_factory.mktemp('ltx-native-plan'))


def test_actual_measured_ltx_policy_preserves_all_updates(native_assets, tmp_path, monkeypatch):
    checks.test_real_measured_policy_preserves_every_update_and_persistent_owner(native_assets, tmp_path, monkeypatch)


def test_public_ltx_auto_saved_plan_and_new_process_resume(native_assets, tmp_path):
    checks.test_public_auto_saved_plan_and_new_process_resume(native_assets, tmp_path)


def test_ltx_trial_failure_preserves_live_state(native_assets, tmp_path, monkeypatch):
    checks.test_trial_budget_refusal_and_trial_exception_preserve_live_native_state(native_assets, tmp_path, monkeypatch)


@pytest.mark.parametrize('layout', ['bf16', 'meta'])
def test_ltx_clone_layout_refusal(native_assets, monkeypatch, layout):
    checks.test_fixture_label_does_not_admit_unsupported_native_clone_layout(native_assets, monkeypatch, layout)


def test_ltx_qualified_copy_keeps_optimizer_aliases(native_assets):
    checks.test_qualified_clone_preserves_parameter_optimizer_aliases_and_native_module_state(native_assets)


@pytest.mark.parametrize('field', ['source', 'shape', 'dtype', 'model-parameter', 'algorithm', 'hardware',
                                  'cache', 'plan-hash', 'unmeasured'])
def test_ltx_saved_plan_rejects_drift(native_assets, native_plan, tmp_path, monkeypatch, field):
    checks.test_saved_native_plan_rejects_drift_before_any_update(native_assets, native_plan, tmp_path, monkeypatch, field)


def test_ltx_unqualified_pretrained_trials_refuse_before_copy(native_assets, tmp_path, monkeypatch):
    checks.assert_unqualified_trials_refused(native_assets, tmp_path, monkeypatch)


def test_ltx_sharing_declarations_exclude_mutable_lora(native_assets):
    from distillschedule.training.low_rank import TrainableLowRankLinear
    _, components, _, _, _ = checks.owned(native_assets)
    expected = {'teacher', 'media'}
    for role in ('student', 'fake'):
        for path, module in components.state[role].named_modules():
            if isinstance(module, TrainableLowRankLinear):
                expected.add(role+'.'+path+'.base')
                assert not any(p.requires_grad for p in module.base.parameters())
                assert module.lora_a.requires_grad and module.lora_b.requires_grad
    assert len(expected) == 6 and set(components.immutable_modules) == expected
