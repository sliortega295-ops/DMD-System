"""C30 adds one existing exact checkpoint factor; never mutates C29 defaults."""
import copy
import json
from pathlib import Path
import pytest

from distillschedule.core.config import load_plan
from distillschedule.runtime.livewan_c30_contract import PLAN_CONTRACT
from distillschedule.runtime.livewan_attention_checkpoint import PLAN_CONTRACT as SAC
from distillschedule.runtime.livewan_crossscale_worker import (
    _c30_checkpoint_composition_enabled, _generator_rollout_cast_cache_enabled,
    _generator_rms_epilogue_enabled, _critic_attention_checkpoint_enabled,
    _validate_plan_identity, _apply_trajectory_contract)

ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT/'configs/distillschedule/plans'
PARENT = PLANS/'livewan_wan13b_14b_sf_dmd_c29_continuous_b16_generator_rms_epilogue.json'
CHILD = PLANS/'livewan_wan13b_14b_sf_dmd_c30_continuous_b16_generator_rms_critic_checkpoint.json'

@pytest.fixture
def plans(monkeypatch):
    for key in ('LIVEWAN_STUDENT_CHECKPOINT','LIVEWAN_CRITIC_CHECKPOINT','LIVEWAN_TEACHER_CHECKPOINT',
                'LIVEWAN_OPERATION_TRACE','LIVEWAN_SYSTEMS_BOOTSTRAP','DISTILLSCHEDULE_OUTPUT_DIR'):
        monkeypatch.setenv(key,'/unread/'+key)
    return load_plan(PARENT).plan.to_dict(),load_plan(CHILD).plan.to_dict()


def test_exact_one_factor_plan_diff():
    a,b=json.loads(PARENT.read_text()),json.loads(CHILD.read_text())
    assert b['metadata'].pop('checkpoint_composition')==PLAN_CONTRACT
    assert b['metadata'].pop('critic_attention_checkpoint')==SAC
    assert a==b


def test_default_c29_unchanged_and_c30_accepts_exact_policies(plans):
    a,b=plans
    assert not _c30_checkpoint_composition_enabled(a,mode='pipeline')
    assert not _critic_attention_checkpoint_enabled(a,mode='pipeline')
    for fn in (_c30_checkpoint_composition_enabled,_critic_attention_checkpoint_enabled,
               _generator_rms_epilogue_enabled,_generator_rollout_cast_cache_enabled):
        assert fn(b,mode='pipeline')
        with pytest.raises(RuntimeError):fn(b,mode='baseline')
    _apply_trajectory_contract(b,cycles=2,mode='pipeline')
    _validate_plan_identity(b,mode='pipeline')


def test_c29_cannot_silently_enable_sac_without_c30_marker(plans):
    a,_=plans;a['metadata']['critic_attention_checkpoint']=copy.deepcopy(SAC)
    with pytest.raises(RuntimeError):_generator_rollout_cast_cache_enabled(a,mode='pipeline')

@pytest.mark.parametrize('key,value',[
    ('checkpoint_composition',{}),('generator_rms_epilogue',None),
    ('generator_rollout_cast_cache',None),('critic_attention_checkpoint',None),
    ('score_prompt_staging',{}),('score_context_prefetch',True)])
def test_c30_fail_closed_missing_or_extra_factor(plans,key,value):
    _,b=plans;b['metadata'][key]=value
    with pytest.raises(RuntimeError):_c30_checkpoint_composition_enabled(b,mode='pipeline')


def test_modified_fa2_math_rng_or_storage_rejected(plans):
    for key,value in [('cache_location','cpu'),('preserve_rng_state',False),
                      ('query_tokens',512),('allow_cache_entry_mutation',True)]:
        _,b=copy.deepcopy(plans);b['metadata']['critic_attention_checkpoint'][key]=value
        with pytest.raises(RuntimeError):_c30_checkpoint_composition_enabled(b,mode='pipeline')


def test_receipt_summarizers_require_both_g_and_score_checks():
    name='c30-continuous-b16-generator-rms-critic-checkpoint'
    for p in ('summarize_livewan_trajectory_world16.py','summarize_livewan_two_cycle_timing_world16.py'):
        s=(ROOT/'scripts/distillschedule'/p).read_text()
        assert '"c29-continuous-b16-generator-rms-epilogue", "'+name+'", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation"}' in s
        assert '"c26-continuous-b16-critic-attention-checkpoint", "'+name+'", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation"}' in s

@pytest.mark.parametrize('kind',['trajectory','two-cycle-timing'])
@pytest.mark.parametrize('variant,expected',[
    ('c29-continuous-b16-generator-rms-epilogue',0),
    ('c30-continuous-b16-generator-rms-critic-checkpoint',0),
    ('unregistered-combination',2)])
def test_actual_shell_argument_gate_no_gpu(tmp_path,kind,variant,expected):
    import os,subprocess
    source=(ROOT/'scripts/distillschedule/launch_livewan_crossscale_world16_node.sh').read_text()
    # Execute real validation prefix; never model/asset/process/lock setup.
    prefix=source.split('\nif [[ "${EXECUTION_MODE}" == baseline ]]; then\n')[0]
    env=dict(os.environ, DMD_SYSTEM_ROOT=str(tmp_path),LIVEWAN_PYTHON_OVERLAY='/unused',
        DISTILLSCHEDULE_OUTPUT_DIR='/unused',LIVEWAN_WAN21_SOURCE='/unused',
        LIVEWAN_SOURCE_RECEIPT='/unused',LIVEWAN_PIPELINE_VARIANT=variant,
        LIVEWAN_PROFILE_TWO_CYCLES='0',LIVEWAN_TRAJECTORY_CYCLES='2',
        LIVEWAN_TRAJECTORY_SEED='20260830',LIVEWAN_TRAJECTORY_ARM_ID='test',
        LIVEWAN_TWO_CYCLE_SEED='20260830',LIVEWAN_TWO_CYCLE_ARM_ID='test')
    cmd=['bash','-c',prefix+'\necho ARGUMENT_GATE_PASS\n','launcher-gate',
         '0','canary-validate',kind,'pipeline','a'*40,'127.0.0.1','31811']
    p=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=10)
    assert p.returncode==expected,(p.returncode,p.stdout,p.stderr)
    assert ('ARGUMENT_GATE_PASS' in p.stdout)==(expected==0)
