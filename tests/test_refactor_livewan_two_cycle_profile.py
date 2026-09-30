import copy
import importlib.util
import json
from pathlib import Path

import pytest

from distillschedule.runtime.livewan_two_cycle_profile import (
    enabled,apply_contract,plan_variant,NAME,VARIANT,C34_VARIANT)

ROOT=Path(__file__).resolve().parents[1]

def gate_module():
    p=ROOT/'scripts/distillschedule/build_livewan_two_cycle_profile_gate.py'
    s=importlib.util.spec_from_file_location('profile_gate_test',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_profile_default_off_and_cannot_enter_native(monkeypatch):
    monkeypatch.delenv('LIVEWAN_PROFILE_TWO_CYCLES',raising=False)
    assert not enabled('profile') and not enabled('two-cycle-timing')
    monkeypatch.setenv('LIVEWAN_PROFILE_TWO_CYCLES','1')
    assert enabled('profile')
    for kind in ('trajectory','two-cycle-timing','timing','correctness'):
        with pytest.raises(RuntimeError):enabled(kind)
    monkeypatch.setenv('LIVEWAN_PROFILE_TWO_CYCLES','false')
    with pytest.raises(RuntimeError):enabled('profile')

@pytest.mark.parametrize('suffix,variant',[
    ('c29_continuous_b16_generator_rms_epilogue',VARIANT),
    ('c34_continuous_b16_residual_layout',C34_VARIANT),
])
def test_profile_plan_diff_is_metadata_only(monkeypatch,suffix,variant):
    from distillschedule.core.config import load_plan
    from distillschedule.runtime.livewan_crossscale_worker import (
        _apply_two_cycle_timing_contract,_validate_plan_identity)
    for key in ('LIVEWAN_STUDENT_CHECKPOINT','LIVEWAN_CRITIC_CHECKPOINT','LIVEWAN_TEACHER_CHECKPOINT',
                'LIVEWAN_OPERATION_TRACE','LIVEWAN_SYSTEMS_BOOTSTRAP','DISTILLSCHEDULE_OUTPUT_DIR'):
        monkeypatch.setenv(key,'/unread/'+key)
    plan=load_plan(ROOT/f'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_{suffix}.json').plan.to_dict()
    assert plan_variant(plan,mode='pipeline')==variant
    a,b=copy.deepcopy(plan),copy.deepcopy(plan)
    _apply_two_cycle_timing_contract(a,mode='pipeline');apply_contract(b,mode='pipeline')
    _validate_plan_identity(b,mode='pipeline')
    c=b['metadata']['continuous_pipeline']
    assert c['name']==NAME and c['profiled'] and not c['formal_timing_eligible']
    for key in ('name','profiled','formal_timing_eligible'):
        c[key]=a['metadata']['continuous_pipeline'][key]
    assert a==b
    with pytest.raises(RuntimeError):apply_contract(copy.deepcopy(plan),mode='baseline')

def test_profile_gate_rejects_changed_math_helpers():
    m=gate_module()
    before='def prepare_rollout():\n return 1\ndef run_worker():\n return 1\n'
    after=before.replace('def run_worker():\n return 1','def run_worker():\n return 2')
    assert m.verify_worker_ast(before,after)['changed_functions']==['run_worker']
    with pytest.raises(ValueError):m.verify_worker_ast(before,before.replace('return 1','return 2'))

@pytest.mark.parametrize('variant',[VARIANT,C34_VARIANT])
def test_native_gate_must_be_unchanged_review_not_profile(variant):
    m=gate_module()
    g=dict(schema_version='distillschedule.livewan_two_cycle_timing_gate.v2',status='PASS',
        claimable_for_two_cycle_timing=True,cycles=2,final_versions=dict(generator=2,critic=10,ema=0),
        numerical_review=dict(extreme_failure_count=0,policy='REJECT_EXTREME_ONLY_RECORD_ORDINARY_REVIEW'),
        arms=dict(pipeline=dict(pipeline_variant=variant)),expected_head='a'*40)
    p=dict(status='PASS',formal_claimable=True,timing_contract=dict(profiled=False),expected_head='a'*40)
    m.validate_native(g,p)
    for key,value in [('status','FAIL'),('cycles',3),('claimable_for_two_cycle_timing',False)]:
        with pytest.raises(ValueError):m.validate_native(dict(g,**{key:value}),p)
    with pytest.raises(ValueError):m.validate_native(g,dict(p,timing_contract=dict(profiled=True)))
    bad=copy.deepcopy(g);bad['arms']['pipeline']['pipeline_variant']='c35-unverified'
    with pytest.raises(ValueError):m.validate_native(bad,p)


def test_profile_rejects_intermediate_unaudited_plan(monkeypatch):
    from distillschedule.core.config import load_plan
    for key in ('LIVEWAN_STUDENT_CHECKPOINT','LIVEWAN_CRITIC_CHECKPOINT','LIVEWAN_TEACHER_CHECKPOINT',
                'LIVEWAN_OPERATION_TRACE','LIVEWAN_SYSTEMS_BOOTSTRAP','DISTILLSCHEDULE_OUTPUT_DIR'):
        monkeypatch.setenv(key,'/unread/'+key)
    plan=load_plan(ROOT/'configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c30_continuous_b16_generator_rms_critic_checkpoint.json').plan.to_dict()
    with pytest.raises(RuntimeError,match='intermediate'):
        apply_contract(plan,mode='pipeline')

def test_profile_load_gate_rejects_tamper_before_runtime(tmp_path):
    from distillschedule.runtime.livewan_two_cycle_profile import load_gate,sha256
    p=tmp_path/'gate.json';p.write_text('{}')
    args=dict(path=p,expected_sha256=sha256(p),expected_head='a'*40,mode='pipeline',
              pipeline_variant=VARIANT,arm_id='x',seed=1,trace=None,bootstrap=None,source_receipt={})
    with pytest.raises(RuntimeError,match='identity'):load_gate(**args)
    args['expected_sha256']='0'*64
    with pytest.raises(RuntimeError,match='SHA'):load_gate(**args)


def test_relocated_profile_retains_exact_native_source_receipt(tmp_path,monkeypatch):
    import subprocess
    from distillschedule.runtime.livewan_two_cycle_profile import source_prepare_script,sha256
    native,current=tmp_path/'native',tmp_path/'profile'
    relative=Path('scripts/distillschedule/prepare_livewan_crossscale_source.sh')
    upstream_relative=Path('third_party/livewan/wanstreamer/blockcausal.py')
    for root in (native,current):
        (root/relative).parent.mkdir(parents=True)
        (root/relative).write_text('immutable preparation source')
        (root/upstream_relative).parent.mkdir(parents=True)
        (root/upstream_relative).write_text('pinned upstream source')
    receipt=tmp_path/'receipt.json'
    receipt.write_text(json.dumps(dict(pythonpath=[str(native/'third_party/livewan')],
        files=dict(block=dict(path=str(native/upstream_relative),sha256=sha256(native/upstream_relative))))))
    gate=tmp_path/'gate.json'
    gate.write_text(json.dumps(dict(status='PASS_DIAGNOSTIC_ONLY',pipeline_variant=C34_VARIANT,
        profile_source=dict(instrumentation_only=True,profile_head='b'*40,workload_head='a'*40))))
    monkeypatch.setattr(subprocess,'check_output',lambda cmd,**kw: ('a'*40 if str(native) in cmd else 'b'*40)+'\n')
    before=receipt.read_bytes()
    assert source_prepare_script(repo_root=current,receipt_path=receipt,gate_path=gate)==native/relative
    assert receipt.read_bytes()==before
    (current/upstream_relative).write_text('wrong source')
    with pytest.raises(RuntimeError,match='relocated upstream'):
        source_prepare_script(repo_root=current,receipt_path=receipt,gate_path=gate)
    (current/upstream_relative).write_text('pinned upstream source')
    (current/relative).write_text('modified preparation')
    with pytest.raises(RuntimeError,match='byte-identical'):
        source_prepare_script(repo_root=current,receipt_path=receipt,gate_path=gate)
