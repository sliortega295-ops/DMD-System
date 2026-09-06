"""C31 changes only G no-grad cast-square; C30 remains exactly registered."""
import copy
import json
from pathlib import Path

import pytest

from distillschedule.core.config import load_plan
from distillschedule.runtime.livewan_c30_contract import C31_PLAN_CONTRACT
from distillschedule.runtime.livewan_generator_rms_epilogue import (
    CAST_SQUARE_PLAN_CONTRACT, PLAN_CONTRACT, runtime_receipt_errors)
from distillschedule.runtime.livewan_crossscale_worker import (
    _generator_rms_cast_square_enabled, _generator_rms_epilogue_enabled,
    _c30_checkpoint_composition_enabled, _apply_trajectory_contract,
    _validate_plan_identity)

ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / 'configs/distillschedule/plans'
PARENT = PLANS / 'livewan_wan13b_14b_sf_dmd_c30_continuous_b16_generator_rms_critic_checkpoint.json'
CHILD = PLANS / 'livewan_wan13b_14b_sf_dmd_c31_continuous_b16_generator_rms_cast_square.json'
VARIANT = 'c31-continuous-b16-generator-rms-cast-square'


@pytest.fixture
def plans(monkeypatch):
    for key in ('LIVEWAN_STUDENT_CHECKPOINT', 'LIVEWAN_CRITIC_CHECKPOINT',
                'LIVEWAN_TEACHER_CHECKPOINT', 'LIVEWAN_OPERATION_TRACE',
                'LIVEWAN_SYSTEMS_BOOTSTRAP', 'DISTILLSCHEDULE_OUTPUT_DIR'):
        monkeypatch.setenv(key, '/unread/' + key)
    return load_plan(PARENT).plan.to_dict(), load_plan(CHILD).plan.to_dict()


def test_one_factor_with_truthful_metadata():
    a, b = json.loads(PARENT.read_text()), json.loads(CHILD.read_text())
    assert b['metadata']['checkpoint_composition'] == C31_PLAN_CONTRACT
    assert b['metadata']['generator_rms_epilogue'] == CAST_SQUARE_PLAN_CONTRACT
    for key in ('checkpoint_composition', 'generator_rms_epilogue'):
        b['metadata'][key] = a['metadata'][key]
    assert a == b


def test_parent_unchanged_and_child_explicit(plans):
    a, b = plans
    assert not _generator_rms_cast_square_enabled(a, mode='pipeline')
    assert _generator_rms_epilogue_enabled(a, mode='pipeline')
    assert _generator_rms_cast_square_enabled(b, mode='pipeline')
    assert _c30_checkpoint_composition_enabled(b, mode='pipeline')
    with pytest.raises(RuntimeError):
        _generator_rms_cast_square_enabled(b, mode='baseline')
    _apply_trajectory_contract(b, cycles=2, mode='pipeline')
    _validate_plan_identity(b, mode='pipeline')


@pytest.mark.parametrize('key,value', [
    ('checkpoint_composition', None), ('critic_attention_checkpoint', None),
    ('generator_rms_epilogue', PLAN_CONTRACT), ('generator_rollout_cast_cache', None),
    ('score_prompt_staging', {}), ('score_context_prefetch', True)])
def test_missing_or_extra_factor_rejected(plans, key, value):
    _, b = plans
    b['metadata'][key] = value
    with pytest.raises(RuntimeError):
        _generator_rms_cast_square_enabled(b, mode='pipeline')


def test_c31_receipt_cannot_silently_use_native_pow_or_change_reduction():
    r = dict(CAST_SQUARE_PLAN_CONTRACT, execution_status='PASS', scope_entries=20,
             scope_exits=20, scope_failures=0, contract_failures=0, active=False,
             scope_forward_calls=[1920]*20, calls=38400, square_calls=38400)
    assert not runtime_receipt_errors(r, rank=0, cycles=2, fuse_cast_square=True)
    assert runtime_receipt_errors(r, rank=0, cycles=2)
    for key, value in [('square_calls', 0), ('native_mean_preserved', False),
                       ('fp32_square_preserved', False), ('fused_cast_square', False),
                       ('native_pow_mean_preserved', True), ('name', PLAN_CONTRACT['name']),
                       ('output_dtype', 'bfloat16'), ('grad_path_changed', True)]:
        assert runtime_receipt_errors(dict(r, **{key: value}), rank=0, cycles=2,
                                      fuse_cast_square=True)


def test_runtime_helper_forwards_opt_in(monkeypatch):
    from distillschedule.runtime import livewan_generator_rms_epilogue as mod
    calls = []
    def build(torch, *, fuse_cast_square=False):
        calls.append(fuse_cast_square)
        return None, {'calls': 0}
    monkeypatch.setattr(mod, 'build_nograd_generator_rms_epilogue', build)
    mod.GeneratorRMSEpilogue(None, None)
    mod.GeneratorRMSEpilogue(None, None, fuse_cast_square=True)
    assert calls == [False, True]


@pytest.mark.parametrize('kind', ['trajectory', 'two-cycle-timing'])
def test_actual_shell_argument_gate_no_gpu(tmp_path, kind):
    import os
    import subprocess
    source = (ROOT/'scripts/distillschedule/launch_livewan_crossscale_world16_node.sh').read_text()
    prefix = source.split('\nif [[ "${EXECUTION_MODE}" == baseline ]]; then\n')[0]
    env = dict(os.environ, DISTILLSCHEDULE_PYENV='/unused', DISTILLSCHEDULE_NODE_LOCK='/tmp/test-livewan.lock', DMD_SYSTEM_ROOT=str(tmp_path), LIVEWAN_PYTHON_OVERLAY='/unused',
        DISTILLSCHEDULE_OUTPUT_DIR='/unused', LIVEWAN_WAN21_SOURCE='/unused',
        LIVEWAN_SOURCE_RECEIPT='/unused', LIVEWAN_PIPELINE_VARIANT=VARIANT,
        LIVEWAN_PROFILE_TWO_CYCLES='0', LIVEWAN_TRAJECTORY_CYCLES='2',
        LIVEWAN_TRAJECTORY_SEED='20260830', LIVEWAN_TRAJECTORY_ARM_ID='test',
        LIVEWAN_TWO_CYCLE_SEED='20260830', LIVEWAN_TWO_CYCLE_ARM_ID='test')
    p = subprocess.run(['bash', '-c', prefix+'\necho ARGUMENT_GATE_PASS\n', 'gate',
        '0', 'canary-validate', kind, 'pipeline', 'a'*40, '127.0.0.1', '31811'],
        env=env, text=True, capture_output=True, timeout=10)
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert 'ARGUMENT_GATE_PASS' in p.stdout


def test_both_summaries_require_actual_square_counter():
    for filename in ('summarize_livewan_trajectory_world16.py',
                     'summarize_livewan_two_cycle_timing_world16.py'):
        s = (ROOT/'scripts/distillschedule'/filename).read_text()
        assert 'fuse_cast_square=' in s and VARIANT in s
