"""Real runtime C33 opt-in contract, counters and unchanged C32 defaults."""
import copy
import json
from pathlib import Path
import pytest
from distillschedule.core.config import load_plan
from distillschedule.runtime.livewan_c30_contract import C33_PLAN_CONTRACT
from distillschedule.runtime.livewan_generator_gate import PLAN_CONTRACT, runtime_receipt_errors
from distillschedule.runtime.livewan_crossscale_worker import (
    _generator_gate_enabled, _generator_rms_cast_square_enabled,
    _apply_trajectory_contract, _validate_plan_identity)

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'configs/distillschedule/plans'
PARENT=P/'livewan_wan13b_14b_sf_dmd_c32_continuous_b16_generator_modulation.json'
CHILD=P/'livewan_wan13b_14b_sf_dmd_c33_continuous_b16_generator_gate.json'
VARIANT='c33-continuous-b16-generator-gate'

@pytest.fixture
def plans(monkeypatch):
    for key in ('LIVEWAN_STUDENT_CHECKPOINT','LIVEWAN_CRITIC_CHECKPOINT','LIVEWAN_TEACHER_CHECKPOINT',
                'LIVEWAN_OPERATION_TRACE','LIVEWAN_SYSTEMS_BOOTSTRAP','DISTILLSCHEDULE_OUTPUT_DIR'):
        monkeypatch.setenv(key,'/unread/'+key)
    return load_plan(PARENT).plan.to_dict(),load_plan(CHILD).plan.to_dict()

def test_only_new_factor_and_parent_marker():
    a,b=json.loads(PARENT.read_text()),json.loads(CHILD.read_text())
    assert b['metadata'].pop('generator_gate')==PLAN_CONTRACT
    assert b['metadata']['checkpoint_composition']==C33_PLAN_CONTRACT
    b['metadata']['checkpoint_composition']=a['metadata']['checkpoint_composition']
    assert a==b

def test_explicit_opt_in(plans):
    a,b=plans
    assert not _generator_gate_enabled(a,mode='pipeline')
    assert _generator_gate_enabled(b,mode='pipeline')
    assert _generator_rms_cast_square_enabled(b,mode='pipeline')
    with pytest.raises(RuntimeError):_generator_gate_enabled(b,mode='baseline')
    _apply_trajectory_contract(b,cycles=2,mode='pipeline');_validate_plan_identity(b,mode='pipeline')

@pytest.mark.parametrize('key,value',[('out_of_place',False),('fma',True),
    ('product_dtype','bfloat16'),('output_dtype','bfloat16'),('grad_path_changed',True)])
def test_math_dtype_change_rejected(plans,key,value):
    _,b=plans;b['metadata']['generator_gate'][key]=value
    with pytest.raises(RuntimeError):_generator_gate_enabled(b,mode='pipeline')

def test_unregistered_composition_rejected(plans):
    a,b=plans
    a['metadata']['generator_gate']=copy.deepcopy(PLAN_CONTRACT)
    with pytest.raises(RuntimeError):_generator_gate_enabled(a,mode='pipeline')
    b['metadata']['critic_attention_checkpoint']=None
    with pytest.raises(RuntimeError):_generator_gate_enabled(b,mode='pipeline')

def test_actual_receipt_counts_fail_closed():
    r=dict(PLAN_CONTRACT,execution_status='PASS',scope_entries=20,scope_exits=20,
        scope_forward_calls=[1260]*20,calls=25200,active=False,scope_failures=0,contract_failures=0)
    assert not runtime_receipt_errors(r,rank=0,cycles=2)
    for k,v in [('calls',0),('scope_forward_calls',[1259]*20),('scope_entries',19),
                 ('out_of_place',False),('scope_failures',1),('active',True)]:
        assert runtime_receipt_errors(dict(r,**{k:v}),rank=0,cycles=2)
    assert runtime_receipt_errors(r,rank=8,cycles=2)
    assert not runtime_receipt_errors({'execution_status':'NOT_APPLICABLE'},rank=8,cycles=2)
    from distillschedule.search.livewan_request import LiveWanRequest
    new = dict(r,scope_forward_calls=[960]*20,calls=960*20)
    assert runtime_receipt_errors(new,rank=0,cycles=2)
    assert not runtime_receipt_errors(new,rank=0,cycles=2,request=LiveWanRequest(frames=69))

@pytest.mark.parametrize('kind',['trajectory','two-cycle-timing'])
def test_actual_launcher_prefix(tmp_path,kind):
    import os,subprocess
    s=(ROOT/'scripts/distillschedule/launch_livewan_crossscale_world16_node.sh').read_text()
    prefix=s.split('\nif [[ "${EXECUTION_MODE}" == baseline ]]; then\n')[0]
    env=dict(os.environ,DISTILLSCHEDULE_PYENV='/unused', DISTILLSCHEDULE_NODE_LOCK='/tmp/test-livewan.lock', DMD_SYSTEM_ROOT=str(tmp_path),LIVEWAN_PYTHON_OVERLAY='/unused',
        DISTILLSCHEDULE_OUTPUT_DIR='/unused',LIVEWAN_WAN21_SOURCE='/unused',LIVEWAN_SOURCE_RECEIPT='/unused',
        LIVEWAN_PIPELINE_VARIANT=VARIANT,LIVEWAN_PROFILE_TWO_CYCLES='0',LIVEWAN_TRAJECTORY_CYCLES='2',
        LIVEWAN_TRAJECTORY_SEED='20260830',LIVEWAN_TRAJECTORY_ARM_ID='test',
        LIVEWAN_TWO_CYCLE_SEED='20260830',LIVEWAN_TWO_CYCLE_ARM_ID='test')
    p=subprocess.run(['bash','-c',prefix+'\necho ARGUMENT_GATE_PASS\n','gate','0','validate',kind,
        'pipeline','a'*40,'127.0.0.1','31811'],env=env,text=True,capture_output=True,timeout=10)
    assert p.returncode==0,(p.stdout,p.stderr)
    assert 'ARGUMENT_GATE_PASS' in p.stdout

def test_grad_path_has_no_scope_and_summaries_check_census():
    import inspect
    from distillschedule.runtime.livewan_crossscale_worker import prepare_rollout,generator_backward_for_sample
    assert 'generator_gate_scope' in inspect.getsource(prepare_rollout)
    assert 'generator_gate_scope' not in inspect.getsource(generator_backward_for_sample)
    for name in ('summarize_livewan_trajectory_world16.py','summarize_livewan_two_cycle_timing_world16.py'):
        s=(ROOT/'scripts/distillschedule'/name).read_text()
        assert 'generator_gate' in s and 'gate_errors(' in s and VARIANT in s
