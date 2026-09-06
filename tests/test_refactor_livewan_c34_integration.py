"""C34 changes internal output layout only, after real-row bitwise canary."""
import copy
import inspect
import json
from pathlib import Path
import pytest
from distillschedule.core.config import load_plan
from distillschedule.runtime.livewan_c30_contract import C34_PLAN_CONTRACT
from distillschedule.runtime.livewan_generator_gate import LAYOUT_PLAN_CONTRACT, PLAN_CONTRACT, runtime_receipt_errors
from distillschedule.runtime.livewan_crossscale_worker import (
    _generator_residual_layout_enabled, _generator_gate_enabled,
    _apply_trajectory_contract, _validate_plan_identity, generator_backward_for_sample)
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'configs/distillschedule/plans'
PARENT=P/'livewan_wan13b_14b_sf_dmd_c33_continuous_b16_generator_gate.json'
CHILD=P/'livewan_wan13b_14b_sf_dmd_c34_continuous_b16_residual_layout.json'
VARIANT='c34-continuous-b16-residual-layout'

@pytest.fixture
def plans(monkeypatch):
    for key in ('LIVEWAN_STUDENT_CHECKPOINT','LIVEWAN_CRITIC_CHECKPOINT','LIVEWAN_TEACHER_CHECKPOINT',
                'LIVEWAN_OPERATION_TRACE','LIVEWAN_SYSTEMS_BOOTSTRAP','DISTILLSCHEDULE_OUTPUT_DIR'):
        monkeypatch.setenv(key,'/unread/'+key)
    return load_plan(PARENT).plan.to_dict(),load_plan(CHILD).plan.to_dict()

def test_only_layout_changes():
    a,b=json.loads(PARENT.read_text()),json.loads(CHILD.read_text())
    assert b['metadata'].pop('generator_residual_layout')==LAYOUT_PLAN_CONTRACT
    assert b['metadata']['checkpoint_composition']==C34_PLAN_CONTRACT
    b['metadata']['checkpoint_composition']=a['metadata']['checkpoint_composition']
    assert a==b

def test_explicit_composition(plans):
    a,b=plans
    assert not _generator_residual_layout_enabled(a,mode='pipeline')
    assert _generator_gate_enabled(a,mode='pipeline')
    assert _generator_residual_layout_enabled(b,mode='pipeline')
    with pytest.raises(RuntimeError):_generator_residual_layout_enabled(b,mode='baseline')
    _apply_trajectory_contract(b,cycles=2,mode='pipeline');_validate_plan_identity(b,mode='pipeline')
    a['metadata']['generator_residual_layout']=copy.deepcopy(LAYOUT_PLAN_CONTRACT)
    with pytest.raises(RuntimeError):_generator_residual_layout_enabled(a,mode='pipeline')

@pytest.mark.parametrize('key,value',[('output_layout','native'),('public_output_contract_unchanged',False),
    ('grad_path_changed',True),('scope','all_roles'),('internal_stride_changed',False)])
def test_contract_fail_closed(plans,key,value):
    _,b=plans;b['metadata']['generator_residual_layout'][key]=value
    with pytest.raises(RuntimeError):_generator_residual_layout_enabled(b,mode='pipeline')

def test_actual_layout_receipt():
    r=dict(PLAN_CONTRACT,output_layout='row_major',execution_status='PASS',scope_entries=20,scope_exits=20,
        scope_forward_calls=[1260]*20,calls=25200,active=False,scope_failures=0,contract_failures=0)
    assert not runtime_receipt_errors(r,rank=0,cycles=2,output_layout='row_major')
    assert runtime_receipt_errors(r,rank=0,cycles=2)
    assert runtime_receipt_errors(dict(r,output_layout='native'),rank=0,cycles=2,output_layout='row_major')
    assert runtime_receipt_errors(dict(r,calls=25199),rank=0,cycles=2,output_layout='row_major')
    assert 'generator_gate_scope' not in inspect.getsource(generator_backward_for_sample)

@pytest.mark.parametrize('kind',['trajectory','two-cycle-timing'])
def test_actual_launcher_prefix(tmp_path,kind):
    import os,subprocess
    s=(ROOT/'scripts/distillschedule/launch_livewan_crossscale_world16_node.sh').read_text()
    prefix=s.split('\nif [[ "${EXECUTION_MODE}" == baseline ]]; then\n')[0]
    env=dict(os.environ,DMD_SYSTEM_ROOT=str(tmp_path),LIVEWAN_PYTHON_OVERLAY='/unused',
        DISTILLSCHEDULE_OUTPUT_DIR='/unused',LIVEWAN_WAN21_SOURCE='/unused',LIVEWAN_SOURCE_RECEIPT='/unused',
        LIVEWAN_PIPELINE_VARIANT=VARIANT,LIVEWAN_PROFILE_TWO_CYCLES='0',LIVEWAN_TRAJECTORY_CYCLES='2',
        LIVEWAN_TRAJECTORY_SEED='20260830',LIVEWAN_TRAJECTORY_ARM_ID='test',
        LIVEWAN_TWO_CYCLE_SEED='20260830',LIVEWAN_TWO_CYCLE_ARM_ID='test')
    p=subprocess.run(['bash','-c',prefix+'\necho ARGUMENT_GATE_PASS\n','gate','0','validate',kind,
        'pipeline','a'*40,'127.0.0.1','32111'],env=env,text=True,capture_output=True,timeout=10)
    assert p.returncode==0,(p.stdout,p.stderr)
    assert 'ARGUMENT_GATE_PASS' in p.stdout
    for name in ('summarize_livewan_trajectory_world16.py','summarize_livewan_two_cycle_timing_world16.py'):
        text=(ROOT/'scripts/distillschedule'/name).read_text()
        assert VARIANT in text and 'output_layout="row_major" if' in text and 'gate_errors(' in text
