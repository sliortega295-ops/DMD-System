"""C35 inherits C34 unchanged, with one explicitly opt-in Score gate."""
import copy
import json
from pathlib import Path
import pytest

from distillschedule.core.config import load_plan
from distillschedule.runtime.livewan_crossscale_worker import (
    _score_residual_gate_enabled, _apply_trajectory_contract, _validate_plan_identity)
from distillschedule.runtime.livewan_score_gate import PLAN_CONTRACT, runtime_receipt_errors

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'configs/distillschedule/plans'
CHILD=P/'livewan_wan13b_14b_sf_dmd_c35_continuous_b16_score_gate.json'
PARENT=P/'livewan_wan13b_14b_sf_dmd_c34_continuous_b16_residual_layout.json'


def test_single_factor_plan():
    a,b=json.loads(PARENT.read_text()),json.loads(CHILD.read_text())
    assert b['metadata'].pop('score_residual_gate')==PLAN_CONTRACT
    assert a==b


@pytest.fixture
def plans(monkeypatch):
    for key in ('LIVEWAN_STUDENT_CHECKPOINT','LIVEWAN_CRITIC_CHECKPOINT',
                'LIVEWAN_TEACHER_CHECKPOINT','LIVEWAN_OPERATION_TRACE',
                'LIVEWAN_SYSTEMS_BOOTSTRAP','DISTILLSCHEDULE_OUTPUT_DIR'):
        monkeypatch.setenv(key,'/unread/'+key)
    return load_plan(PARENT).plan.to_dict(),load_plan(CHILD).plan.to_dict()


def test_default_and_explicit_child(plans):
    a,b=plans
    assert not _score_residual_gate_enabled(a,mode='pipeline')
    assert _score_residual_gate_enabled(b,mode='pipeline')
    with pytest.raises(RuntimeError): _score_residual_gate_enabled(b,mode='baseline')
    _apply_trajectory_contract(b,cycles=2,mode='pipeline')
    _validate_plan_identity(b,mode='pipeline')


@pytest.mark.parametrize('key,value',[('product_dtype','float32'),('fma',True),
    ('gate_requires_grad',True),('scope','all_ranks'),('output_layout','native'),
    ('native_layernorm_unchanged',False)])
def test_fail_closed(plans,key,value):
    _,p=plans;p['metadata']['score_residual_gate'][key]=value
    with pytest.raises(RuntimeError): _score_residual_gate_enabled(p,mode='pipeline')


def test_counter_source_and_rank_guards():
    calls={'teacher/nograd/plain':4,'critic/nograd/plain':4,'critic/grad/checkpoint':10}
    good=dict(PLAN_CONTRACT,install_status='PASS',execution_status='PASS',
        contract_failures=0,forward_calls=1760,backward_calls=600)
    assert not runtime_receipt_errors(good,rank=8,inventory=calls)
    assert runtime_receipt_errors(dict(good,forward_calls=1759),rank=8,inventory=calls)
    assert runtime_receipt_errors(dict(good,backward_calls=599),rank=8,inventory=calls)
    assert runtime_receipt_errors(good,rank=0,inventory=calls)
    assert not runtime_receipt_errors({'execution_status':'NOT_APPLICABLE'},rank=0,inventory={})


@pytest.mark.parametrize('kind',['trajectory','two-cycle-timing'])
def test_controller_routes_child_without_model_or_assets(tmp_path,kind):
    import subprocess,sys
    args=[sys.executable,str(ROOT/'scripts/distillschedule/run_livewan_crossscale_world16.py'),
        '--hosts','node0,node1','--run-id','c35-dryrun','--run-kind',kind,
        '--execution-mode','pipeline','--pipeline-variant','c35-continuous-b16-score-gate',
        '--expected-head','a'*40,'--master-addr','127.0.0.1','--dry-run',
        '--local-evidence-root',str(tmp_path)]
    for key in ['remote-root','remote-output-root','student-checkpoint','critic-checkpoint',
        'teacher-checkpoint','trace','bootstrap','source-root','source-receipt','python-overlay']:
        args+=['--'+key,str(tmp_path/key)]
    if kind=='trajectory':
        args+=['--trajectory-cycles','2','--trajectory-arm-id','c35','--trajectory-seed','20260830']
    else:
        args+=['--two-cycle-arm-id','c35','--two-cycle-seed','20260830',
            '--correctness-receipt',str(tmp_path/'gate.json'),'--correctness-sha256','b'*64]
    r=subprocess.run(args,check=True,capture_output=True,text=True,timeout=15)
    result=json.loads(r.stdout)
    assert len(result['commands'])==2
    assert all('c35-continuous-b16-score-gate' in ' '.join(v) for v in result['commands'])
