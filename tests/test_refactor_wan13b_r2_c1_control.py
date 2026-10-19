"""C1 metadata transport only: no GPU/model and no numerical gate weakening."""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
REL='scripts/distillschedule/probe_wan13b_factorial_r2_world16.py'

def worker():
    spec=importlib.util.spec_from_file_location('r2_c1_cpu_worker',ROOT/REL)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.mark.parametrize('rank',range(16))
def test_cpu_group_preserves_all_observations_rng_and_tree(monkeypatch,rank):
    m=worker();monkeypatch.setenv('NCCL_ALGO','Tree')
    observation={'rank':rank,'finite':False,'versions':{'generator':3,'critic':15,'ema':15},'data':[1,2.5,'payload']}
    expected=[dict(copy.deepcopy(observation),rank=i)for i in range(16)]
    group=object();events=[]
    def create(**kwargs):
        assert kwargs['ranks']==list(range(16))and kwargs['backend']=='gloo'
        assert kwargs['timeout'].total_seconds()==180
        events.append('create');return group
    def gather(out,own,*,group):
        assert group is not None and own==observation
        events.append('gather');out[:]=copy.deepcopy(expected)
    d=SimpleNamespace(new_group=create,get_backend=lambda g=None:'gloo'if g is group else'nccl',all_gather_object=gather,destroy_process_group=lambda g:events.append('destroy'))
    a={};t=torch.get_rng_state().clone();p=random.getstate()
    result=m.gather_resident_c1_metadata(observation,world=16,audit=a,distributed=d)
    assert result==expected and events==['create','gather','destroy']
    assert torch.equal(t,torch.get_rng_state())and p==random.getstate()
    assert os.environ['NCCL_ALGO']=='Tree'
    assert a['created_count']==a['gather_count']==a['destroyed_count']==1
    assert a['metadata_backend']=='gloo'and a['default_training_backend']=='nccl'

def test_gather_error_is_not_swallowed_and_group_destroyed(monkeypatch):
    m=worker();g=object();events=[]
    def fail(*args,**kwargs):raise RuntimeError('metadata exchange failed')
    d=SimpleNamespace(new_group=lambda **kw:g,get_backend=lambda x=None:'gloo'if x is g else'nccl',all_gather_object=fail,destroy_process_group=lambda x:events.append(x))
    with pytest.raises(RuntimeError,match='metadata exchange failed'):
        m.gather_resident_c1_metadata({},world=16,audit={},distributed=d)
    assert events==[g]

def test_wrong_backend_rejected_and_group_destroyed():
    m=worker();g=object();destroy=Mock();gather=Mock()
    d=SimpleNamespace(new_group=lambda **kw:g,get_backend=lambda x=None:'nccl',all_gather_object=gather,destroy_process_group=destroy)
    with pytest.raises(AssertionError,match='gloo'):
        m.gather_resident_c1_metadata({},world=16,audit={},distributed=d)
    gather.assert_not_called();destroy.assert_called_once_with(g)

def test_non_world16_rejected_before_any_collective():
    m=worker();d=Mock()
    with pytest.raises(ValueError,match='world16'):
        m.gather_resident_c1_metadata({},world=8,audit={},distributed=d)
    d.new_group.assert_not_called()

def test_cycle_math_timer_rng_and_qualification_gate_unchanged():
    old=subprocess.check_output(['git','-C',str(ROOT),'show','45fcff74cd30a57903b30b2fddf124a95fb11682:'+REL],text=True)
    new=(ROOT/REL).read_text();a=ast.parse(old);b=ast.parse(new)
    funcs=lambda t:{n.name:ast.dump(n,include_attributes=False)for n in t.body if isinstance(n,ast.FunctionDef)}
    fa,fb=funcs(a),funcs(b)
    for name in fa.keys()-{'main'}:assert fa[name]==fb[name],name
    main=next(n for n in b.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    gather=next(n for n in ast.walk(main)if isinstance(n,ast.FunctionDef)and n.name=='gather')
    assert 'gather_resident_c1_metadata'in ast.unparse(gather)
    assert "cfg['placement'] == r2.RESIDENT"in ast.unparse(gather)
    # The FSDP branch keeps the original default NCCL gather; only Resident calls the helper.
    assert any(isinstance(n,ast.Call)and ast.unparse(n.func)=='dist.all_gather_object'for n in ast.walk(gather))
    policy=(ROOT/'src/distillschedule/runtime/wan13b_factorial_r2.py').read_bytes()
    assert policy==subprocess.check_output(['git','-C',str(ROOT),'show','45fcff74cd30a57903b30b2fddf124a95fb11682:src/distillschedule/runtime/wan13b_factorial_r2.py'])
