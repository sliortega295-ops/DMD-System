"""SYNTHETIC_TEST_ONLY small CPU model; original native projection class AST."""
from pathlib import Path
import ast,copy,hashlib,sys,os
import pytest
import torch
from torch import nn
import torch.utils.checkpoint as cp
from distillschedule.strategies.h3_ar_manual_frozen_bank import (
    FrozenAdaLNBank, Key, current_key, key_scope, NativeSigmaKeys,
    ManualFrozenAdaLNPolicy, validate_manual_policy, POLICY_ID)
# SYNTHETIC_TEST_ONLY geometry; same projection implementation as pinned native
# bd617a4312c70405e916707e1908100b484b2259dfbeba4af7de5bb145206d80.
class MiniMaxH3AdalnProj(nn.Module):
    def __init__(self, hidden_size, time_embed_dim, out_features, *, expand_ratio, modality_num):
        super().__init__()
        if out_features != expand_ratio * hidden_size * modality_num:
            raise ValueError(
                f"adaln out_features mismatch: {out_features} != {expand_ratio}*{hidden_size}*{modality_num}"
            )
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = hidden_size
        self.linear = nn.Linear(time_embed_dim, out_features, bias=True)

    def forward(self, t_emb):
        x = nn.functional.silu(t_emb)
        x = self.linear(x)
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))
def native_projection_type():return MiniMaxH3AdalnProj
class Block(nn.Module):
    def __init__(self):
        super().__init__();self.adaln_proj=native_projection_type()(8,4,144,expand_ratio=6,modality_num=3).double().requires_grad_(False)
        self.trainable=nn.Linear(8,8,bias=False).double()
    def forward(self,x,t):
        shift,scale,gate,*_=self.adaln_proj(t)
        return x*(scale[0]+1)+shift[0]+self.trainable(torch.nn.functional.dropout(x,p=.2,training=True))*gate[0]
class Model(nn.Module):
    def __init__(self):super().__init__();self.blocks=nn.ModuleList([Block(),Block()])
    def forward(self,x,t,checkpoint):
        for b in self.blocks:x=cp.checkpoint(b,x,t,use_reentrant=False)if checkpoint else b(x,t)
        return x

def installed():
    torch.manual_seed(74);m=Model();bank=FrozenAdaLNBank(m,expected_blocks=2,hidden_size=8,input_size=4);bank.bind_frozen_versions();return m,bank

def test_relocation_keeps_same_objects_storages_and_trainable_order():
    m=Model();before={id(p):p.data_ptr()for p in m.parameters()};train=[id(p)for p in m.parameters()if p.requires_grad]
    b=FrozenAdaLNBank(m,expected_blocks=2,hidden_size=8,input_size=4)
    assert before=={id(p):p.data_ptr()for p in m.parameters()}
    assert train==[id(p)for p in m.parameters()if p.requires_grad]
    assert all(not list(x.adaln_proj.parameters())for x in m.blocks)
    assert len(list(m._manual_frozen_adaln_bank.parameters()))==4
    with pytest.raises(ValueError,match='ALREADY'):FrozenAdaLNBank(m,expected_blocks=2,hidden_size=8,input_size=4)

@pytest.mark.parametrize('args',[('fixed',8,0),('fixed',0,1),('dynamic',2,1),('dynamic',0,0),('bad',0,0)])
def test_invalid_keys_fail(args):
    with pytest.raises(ValueError):Key(*args)

def test_missing_scope_and_unbound_versions_fail_closed():
    m,b=installed();x=torch.zeros(2,4,dtype=torch.float64)
    with pytest.raises(RuntimeError,match='MISSING'):m.blocks[0].adaln_proj(x)
    b._versions=None
    with key_scope(Key('fixed',0)),pytest.raises(RuntimeError,match='BIND'):m.blocks[0].adaln_proj(x)

def test_wrong_value_same_key_rejected_at_original_fence_not_fallback():
    m,b=installed();x=torch.zeros(2,4,dtype=torch.float64)
    with key_scope(Key('fixed',0)):
        first=m.blocks[0].adaln_proj(x);second=m.blocks[0].adaln_proj(x+1)
        assert first is second
    assert b.counts['bank_forward_calls']==1
    with pytest.raises(RuntimeError,match='VALUES_DIFFER'):b.finish_cycle(synchronized=True)

def test_input_grad_and_shape_fail():
    m,b=installed()
    with key_scope(Key('fixed',0)):
        with pytest.raises(ValueError):m.blocks[0].adaln_proj(torch.ones(2,4,dtype=torch.float64,requires_grad=True))
        with pytest.raises(ValueError):m.blocks[0].adaln_proj(torch.ones(3,4,dtype=torch.float64))

def test_frozen_shard_mutation_detected():
    m,b=installed()
    with torch.no_grad():m._manual_frozen_adaln_bank[0].linear.weight.add_(1)
    with key_scope(Key('fixed',0)),pytest.raises(RuntimeError,match='PARTITION_CHANGED'):m.blocks[0].adaln_proj(torch.ones(2,4,dtype=torch.float64))

def test_dynamic_entries_evicted_and_static_survive():
    m,b=installed();t=torch.ones(2,4,dtype=torch.float64)
    for key in(Key('fixed',0),Key('dynamic',0,1)):
        with key_scope(key):m.blocks[0].adaln_proj(t)
    r=b.finish_cycle(synchronized=True);assert r['static_entries']==1 and r['dynamic_entries_released']==1
    assert all(k.kind=='fixed'for _,k in b.cache)
    with pytest.raises(RuntimeError,match='FENCED'):b.finish_cycle(synchronized=False)

def test_actual_nonreentrant_backward_after_scope_exit_preserves_output_grad_RNG_Adam():
    torch.manual_seed(91);base=Model();inputs=[torch.randn(2,8,dtype=torch.float64)for _ in range(4)];ts=[torch.randn(2,4,dtype=torch.float64)for _ in range(2)]
    def run(cached):
        model=copy.deepcopy(base);bank=FrozenAdaLNBank(model,expected_blocks=2,hidden_size=8,input_size=4)if cached else None
        if bank:bank.bind_frozen_versions()
        params=[p for p in model.parameters()if p.requires_grad];opt=torch.optim.AdamW(params,lr=.001)
        torch.manual_seed(988);out=[];grad=[];receipts=[]
        for cycle in(1,2):
            key=Key('fixed',0);x=inputs[(cycle-1)*2].clone().requires_grad_()
            if bank:
                with key_scope(key),bank.capture_checkpoints():y=model(x,ts[0],True)
            else:y=model(x,ts[0],True)
            assert current_key()is None
            # Backward occurs after every key / temporary checkpoint patch scope.
            y.square().mean().backward();out.append(y.detach().clone());grad.append([p.grad.clone()for p in params]);opt.step();opt.zero_grad(set_to_none=True)
            key=Key('dynamic',0,cycle);x=inputs[(cycle-1)*2+1].clone().requires_grad_()
            if bank:
                with key_scope(key),bank.capture_checkpoints():y=model(x,ts[1],True)
            else:y=model(x,ts[1],True)
            y.square().mean().backward();out.append(y.detach().clone());grad.append([p.grad.clone()for p in params]);opt.step();opt.zero_grad(set_to_none=True)
            if bank:receipts.append(bank.finish_cycle(synchronized=True))
        state=[p.detach().clone()for p in params]+[v.clone()for p in params for v in opt.state[p].values()if torch.is_tensor(v)]
        return out,grad,state,torch.random.get_rng_state().clone(),bank,receipts
    A=run(False);B=run(True)
    for left,right in zip(A[:3],B[:3]):
        def flat(v):
            for x in v:
                if isinstance(x,list):yield from flat(x)
                else:yield x
        assert all(torch.equal(a,b)for a,b in zip(flat(left),flat(right)))
    assert torch.equal(A[3],B[3])and current_key()is None
    b=B[4];assert b.counts['checkpoint_wraps']==8
    # Non-reentrant early-stop is legal. Entry/finally must always pair.
    assert b.counts['scoped_execution_entry']==b.counts['scoped_execution_finally']==16
    assert B[5][0]['values_equal']and B[5][1]['values_equal']

def test_temporary_checkpoint_patch_always_restored_and_kwargs_forwarded(monkeypatch):
    m,b=installed();original=cp.checkpoint;seen=[]
    def spy(function,*a,**kw):seen.append(dict(kw));return original(function,*a,**kw)
    monkeypatch.setattr(cp,'checkpoint',spy)
    with key_scope(Key('fixed',0)),b.capture_checkpoints():
        x=torch.ones(2,8,dtype=torch.float64,requires_grad=True);t=torch.ones(2,4,dtype=torch.float64)
        cp.checkpoint(m.blocks[0],x,t,use_reentrant=False,preserve_rng_state=True)
    assert cp.checkpoint is spy and seen==[dict(use_reentrant=False,preserve_rng_state=True)]
    with pytest.raises(ZeroDivisionError):
        with key_scope(Key('fixed',1)),b.capture_checkpoints():1/0
    assert cp.checkpoint is spy and current_key()is None

def test_two_cycle_native_order_8_rollout_C1_T_F_G1_complete_census():
    torch.manual_seed(83);model=Model();b=FrozenAdaLNBank(model,expected_blocks=2,hidden_size=8,input_size=4);b.bind_frozen_versions()
    fixed=[torch.randn(2,4,dtype=torch.float64)for _ in range(8)]
    opt=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.001)
    previous={}
    for cycle in(1,2):
        def call(key,t,grad):
            x=torch.ones(2,8,dtype=torch.float64,requires_grad=grad)
            with torch.set_grad_enabled(grad),key_scope(key),b.capture_checkpoints():out=model(x,t,grad)
            if grad:out.square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
        for i,t in enumerate(fixed):call(Key('fixed',i),t,False)
        t=torch.randn(2,4,dtype=torch.float64);call(Key('dynamic',0,cycle),t,True)
        t=torch.randn(2,4,dtype=torch.float64)
        call(Key('dynamic',1,cycle),t,False);call(Key('dynamic',1,cycle),t,False)
        call(Key('fixed',5),fixed[5],True)
        receipt=b.finish_cycle(synchronized=True);counts=receipt['counts']
        delta={k:v-previous.get(k,0)for k,v in counts.items()};previous=counts
        assert delta['proxy_calls']==28 and delta['checkpoint_wraps']==4
        assert delta['scoped_execution_entry']==delta['scoped_execution_finally']==8
        assert delta['bank_forward_calls']==(20 if cycle==1 else 4)
        assert delta['cache_reads']==(8 if cycle==1 else 24)
        assert receipt['static_entries']==16 and receipt['dynamic_entries_released']==4


@pytest.mark.parametrize('policy,mode,impl,plan',[
    (POLICY_ID, 'run', 'zero3-ap', 'saved.json'),
    (POLICY_ID, 'calibrate', 'zero3-ap', None),
    (POLICY_ID, 'manual', 'zero3-native', None),
    (POLICY_ID, 'manual', 'zero3-ap-gpu-tail1', None),
    ('unreviewed', 'manual', 'zero3-ap', None),
    ('off', 'run', 'zero3-ap', None)])
def test_manual_optin_never_accepts_automatic_or_unknown_paths(policy,mode,impl,plan):
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='MANUAL_ONLY'):
        validate_manual_policy(policy,mode=mode,implementation=SimpleNamespace(id=impl),plan_path=plan)


def test_default_policy_and_explicit_control_are_not_search_candidates():
    from types import SimpleNamespace
    from distillschedule.strategies.h3_ar_task import H3ARImplementation
    validate_manual_policy(None,mode='calibrate',implementation=None,plan_path=None)
    for p in ('off',POLICY_ID):
        validate_manual_policy(p,mode='manual',implementation=SimpleNamespace(id='zero3-ap'),plan_path=None)
    with pytest.raises(ValueError):H3ARImplementation(POLICY_ID).to_dict()


def test_sigma_identity_uses_no_host_value_reads_and_rejects_mutation():
    schedule=(torch.linspace(1,0,9),torch.linspace(1,0,9))
    keys=NativeSigmaKeys(schedule);keys.begin(1)
    for i in range(8):assert keys.resolve(tuple(s[i] for s in schedule))==Key('fixed',i)
    pair=(torch.tensor(.36),torch.tensor(.44))
    assert keys.resolve(pair)==keys.resolve(pair)==Key('dynamic',0,1)
    with torch.no_grad():pair[0].add_(.1)
    with pytest.raises(RuntimeError,match='MUTATED'):keys.resolve(pair)
    with torch.no_grad():schedule[0].add_(.1)
    with pytest.raises(RuntimeError,match='SCHEDULE_MUTATED'):keys.check_schedule()


def test_native_sigma_keys_hold_dynamic_references_and_fail_closed():
    schedule=(torch.linspace(1,0,9),torch.linspace(1,0,9));keys=NativeSigmaKeys(schedule)
    with pytest.raises(RuntimeError):keys.resolve((schedule[0][0],schedule[1][0]))
    keys.begin(1)
    first=(torch.tensor(.1),torch.tensor(.2));second=(torch.tensor(.3),torch.tensor(.4))
    assert keys.resolve(first)==Key('dynamic',0,1)
    assert keys.dynamic[0][0][0] is first[0]
    assert keys.resolve(second)==Key('dynamic',1,1)
    with pytest.raises(RuntimeError,match='MORE_THAN_TWO'):keys.resolve(tuple(x.clone()for x in first))
    with pytest.raises(RuntimeError,match='KEY_CENSUS'):keys.finish()


def test_manual_real_predict_checkpoint_capture_two_cycles_complete_policy_counts():
    # This is CPU SYNTHETIC_TEST_ONLY, not native GPU qualification.
    from types import SimpleNamespace
    torch.manual_seed(772);model=Model();model.blocks=nn.ModuleList([Block()for _ in range(50)])
    policy=ManualFrozenAdaLNPolicy()
    policy.bank=FrozenAdaLNBank(model,expected_blocks=50,hidden_size=8,input_size=4)
    policy.registered_frozen_numel=sum(p.numel()for p in model._manual_frozen_adaln_bank.parameters())
    policy.previous_counts=__import__('collections').Counter()
    policy.bind_engine(SimpleNamespace(module=model))
    schedule=(torch.linspace(1,0,9,dtype=torch.float64),torch.linspace(1,0,9,dtype=torch.float64))
    def predict(latents,sigmas,inputs,checkpoint=False):
        t=torch.stack(sigmas).view(2,1).expand(2,4).contiguous()
        return model(latents,t,checkpoint)
    adapter=SimpleNamespace(predict=predict);policy.attach(adapter,SimpleNamespace(schedule=schedule))
    for cycle in (1,2):
        policy.begin_cycle(cycle)
        def call(pair,grad):
            x=torch.ones(2,8,dtype=torch.float64,requires_grad=grad)*1e-10
            with torch.set_grad_enabled(grad):y=adapter.predict(x,pair,{},checkpoint=grad)
            if grad:y.sum().backward()
        for i in range(8):call(tuple(s[i]for s in schedule),False)
        first=(torch.tensor(.25,dtype=torch.float64),torch.tensor(.27,dtype=torch.float64));call(first,True)
        second=(torch.tensor(.47,dtype=torch.float64),torch.tensor(.49,dtype=torch.float64))
        call(second,False);call(second,False);call(tuple(s[5]for s in schedule),True)
        receipt=policy.finish_cycle(cycle)
        assert receipt['exact_input_gate']['values_equal']
        assert receipt['per_cycle_counts']['cache_reads']==(200 if cycle==1 else 600)
        assert receipt['per_cycle_counts']['bank_forward_calls']==(500 if cycle==1 else 100)
        assert receipt['key_resolution']['application_calls']==12
        assert not receipt['parameters_eliminated']and not receipt['weights_merged']
        assert not policy.keys.active and not policy.keys.dynamic


def test_policy_source_is_fingerprinted_without_new_planner_candidate():
    from distillschedule.strategies import h3_ar_planning as planning
    import inspect
    assert 'h3_ar_*.py' in inspect.getsource(planning.source_fingerprint)
    from distillschedule.strategies import h3_ar_worker as worker
    body=inspect.getsource(worker.run_worker)
    assert body.index('validate_manual_policy') < body.index('host_admission')
    assert 'distillschedule.manual_h3_ar_execution.v1' in body
    assert 'automatic_planning_eligible=False' in body
    assert not torch.cuda.is_initialized()
