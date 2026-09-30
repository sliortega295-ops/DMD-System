"""One opt-in Score norm factor; C35/default/affine/Generator are unchanged."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.core.config import load_plan
from distillschedule.runtime.livewan_crossscale_worker import (
    _score_layernorm_enabled, _apply_trajectory_contract, _validate_plan_identity)
from distillschedule.runtime.livewan_score_layernorm import PLAN_CONTRACT, runtime_receipt_errors

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'configs/distillschedule/plans'
CHILD=P/'livewan_wan13b_14b_sf_dmd_c36_continuous_b16_score_layernorm.json'
PARENT=P/'livewan_wan13b_14b_sf_dmd_c35_continuous_b16_score_gate.json'


def test_single_factor_plan_and_no_schedule_change():
    a,b=json.loads(PARENT.read_text()),json.loads(CHILD.read_text())
    assert b['metadata'].pop('score_layernorm')==PLAN_CONTRACT
    assert a==b


@pytest.fixture
def plans(monkeypatch):
    for key in ('LIVEWAN_STUDENT_CHECKPOINT','LIVEWAN_CRITIC_CHECKPOINT',
                'LIVEWAN_TEACHER_CHECKPOINT','LIVEWAN_OPERATION_TRACE',
                'LIVEWAN_SYSTEMS_BOOTSTRAP','DISTILLSCHEDULE_OUTPUT_DIR'):
        monkeypatch.setenv(key,'/unread/'+key)
    return load_plan(PARENT).plan.to_dict(),load_plan(CHILD).plan.to_dict()


def test_default_and_explicit_score_only_parent_binding(plans):
    a,b=plans
    assert not _score_layernorm_enabled(a,mode='pipeline')
    assert _score_layernorm_enabled(b,mode='pipeline')
    with pytest.raises(RuntimeError): _score_layernorm_enabled(b,mode='baseline')
    _apply_trajectory_contract(b,cycles=2,mode='pipeline')
    _validate_plan_identity(b,mode='pipeline')
    p=copy.deepcopy(b);p['metadata'].pop('score_residual_gate')
    with pytest.raises(RuntimeError): _score_layernorm_enabled(p,mode='pipeline')


@pytest.mark.parametrize('key,value',[('input_dtype','float32'),('statistics_dtype','bfloat16'),
    ('eps',1e-5),('elementwise_affine',True),('scope','all_ranks'),
    ('native_affine_and_unsupported_preserved',False)])
def test_fail_closed_plan_metadata(plans,key,value):
    _,p=plans;p['metadata']['score_layernorm'][key]=value
    with pytest.raises(RuntimeError): _score_layernorm_enabled(p,mode='pipeline')


def test_actual_inventory_and_selected_fallback_reconciliation():
    calls={'teacher/nograd/plain':4,'critic/grad/checkpoint':10}
    good=dict(PLAN_CONTRACT,install_status='PASS',execution_status='PASS',
        contract_failures=0,total_norm_calls=2294,selected_entries=1510,
        forward_calls=1510,backward_calls=600,native_fallback_calls=784,
        native_fallback_histogram={'affine_preserved':760,'layout_preserved':24})
    assert not runtime_receipt_errors(good,rank=8,inventory=calls)
    for key,value in [('total_norm_calls',2293),('forward_calls',0),
                      ('selected_entries',0),('backward_calls',0),
                      ('native_fallback_calls',785),('contract_failures',1)]:
        assert runtime_receipt_errors(dict(good,**{key:value}),rank=8,inventory=calls)
    assert runtime_receipt_errors(good,rank=0,inventory=calls)
    assert not runtime_receipt_errors({'execution_status':'NOT_APPLICABLE'},rank=0,inventory={})


@pytest.mark.parametrize('kind',['trajectory','two-cycle-timing'])
def test_controller_routes_explicit_child_without_assets(tmp_path,kind):
    args=[sys.executable,str(ROOT/'scripts/distillschedule/run_livewan_crossscale_world16.py'),
        '--hosts','node0,node1','--run-id','c36-dryrun','--run-kind',kind,
        '--execution-mode','pipeline','--pipeline-variant','c36-continuous-b16-score-layernorm',
        '--expected-head','a'*40,'--master-addr','127.0.0.1','--dry-run',
        '--local-evidence-root',str(tmp_path)]
    for key in ['remote-root','remote-output-root','student-checkpoint','critic-checkpoint',
        'teacher-checkpoint','trace','bootstrap','source-root','source-receipt','python-overlay']:
        args+=['--'+key,str(tmp_path/key)]
    if kind=='trajectory':
        args+=['--trajectory-cycles','2','--trajectory-arm-id','c36','--trajectory-seed','20260830']
    else:
        args+=['--two-cycle-arm-id','c36','--two-cycle-seed','20260830',
            '--correctness-receipt',str(tmp_path/'gate.json'),'--correctness-sha256','b'*64]
    r=subprocess.run(args,check=True,capture_output=True,text=True,timeout=15)
    x=json.loads(r.stdout)
    assert len(x['commands'])==2
    assert all('c36-continuous-b16-score-layernorm' in ' '.join(v) for v in x['commands'])


def test_registration_and_runtime_receipt_are_not_simulated_only():
    source=(ROOT/'src/distillschedule/runtime/livewan_crossscale_worker.py').read_text()
    assert 'score_layernorm = install_frozen_score_layernorm(torch)' in source
    assert '"score_layernorm": score_layernorm' in source
    assert 'score_ln_errors(score_layernorm, rank=rank, inventory=inventory.counts)' in source
    launcher=(ROOT/'scripts/distillschedule/launch_livewan_crossscale_world16_node.sh').read_text()
    assert 'livewan_wan13b_14b_sf_dmd_c36_continuous_b16_score_layernorm.json' in launcher
