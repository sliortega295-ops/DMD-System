"""Selective checkpointing changes actual native recomputation, not math.

Models/graphs are new to the planner. All gradient/Adam comparisons remain on
real Torch; CPU SDPA does not validate CUDA attention or H3 custom FA caches.
"""
from datetime import timedelta
from pathlib import Path
import json
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.workloads.program import Program, Call
from distillschedule.open_workload.torch_backend import Workload, Session, candidates, compare
from distillschedule.open_workload.replica_contract import ReplicaSemantics


class UnregisteredAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.qkv = nn.Linear(width,width*3,bias=False)
    def forward(self,x):
        q,k,v = self.qkv(x).chunk(3,dim=-1)
        return torch.sin(F.scaled_dot_product_attention(q.unsqueeze(1),k.unsqueeze(1),v.unsqueeze(1)))


def predict(ctx): return ctx.state['model'](next(iter(ctx.inputs.values())))
def objective(ctx): return next(iter(ctx.inputs.values())).square().mean()
def backward(ctx): next(iter(ctx.inputs.values())).backward()
def update(ctx):
    ctx.state['adam'].step();ctx.state['adam'].zero_grad(set_to_none=True)


def workload(width=8,length=7,frequency=1,rank=0,world=1,global_batch=False):
    with torch.random.fork_rng():
        torch.manual_seed(77)
        model=UnregisteredAttention(width)
        x=torch.randn(2,length,width)
    x=torch.cat([x+.15*i for i in range(world)]) if global_batch else x+.15*rank
    state={'model':model,'adam':torch.optim.Adam(model.parameters(),lr=.001),'grad':None}
    calls=[]
    for i in range(frequency):
        calls.extend((Call(f'f{i}','forward',('x',),('model',),grad='grad',replay_safe=True),
                      Call(f'l{i}','loss',(f'f{i}',),grad='grad'),
                      Call(f'b{i}','backward',(f'l{i}',),('grad',),('grad',),grad='grad',backward_of=(f'l{i}',)),
                      Call(f'u{i}','update',(f'b{i}',),('model','adam','grad'),('model','adam','grad'))))
    program=Program(tuple(calls),dict.fromkeys(state,0),('x',),(),
                    {'width':width,'length':length,'frequency':frequency})
    w=Workload(program,{'forward':predict,'loss':objective,'backward':backward,'update':update},state,{'x':x})
    w.replica_semantics=ReplicaSemantics({'forward':'model'},tuple(f'b{i}' for i in range(frequency)),2)
    return w


class AttentionCounter(TorchDispatchMode):
    def __init__(self): super().__init__();self.forward=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if func is torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:self.forward+=1
        return func(*args,**(kwargs or {}))


@pytest.mark.parametrize('width,length,frequency',[(8,7,1),(12,11,3)])
def test_native_attention_forward_is_not_recomputed_and_adam_matches(width,length,frequency):
    w=workload(width,length,frequency)
    options=candidates(w,limit=16)['candidates']
    reference=Session(w,options[0]);expected=reference.result_state(reference.run())
    counts={}
    for option in options:
        s=Session(w,option); counter=AttentionCounter()
        with counter: actual=s.result_state(s.run(measure=True))
        assert compare(expected,actual)['status']=='PASS'
        policy=option.get('checkpoint_policy','recompute' if option['checkpoint_bindings'] else 'reference')
        counts[policy]=counter.forward
        if policy=='save_attention':
            assert s.effects['selective_replay_hits']>=frequency
            assert s.effects['selective_native_cache_bytes']>0
            assert s.profile['f0']['retained_bytes']>0
    assert counts['reference']==counts['save_attention']==frequency
    assert counts['recompute']==2*frequency


def test_selective_candidates_are_measured_and_replayable(tmp_path):
    from distillschedule.open_workload.planning import calibrate,plan,run
    from distillschedule.workloads.best_known.registry import fingerprint,write_new_json
    w=workload();manifest=calibrate(w,tmp_path/'cal',samples=1,max_candidates=4,max_program_runs=5)
    assert len(manifest['variants'])==4
    p=plan(tmp_path/'cal/calibration.json',memory_limit_bytes=10**7)
    option=next(x for x in p['search']['ranked'] if x['implementation'].get('checkpoint_policy')=='save_attention')
    p['selected']=option;p.pop('plan_sha256');p['plan_sha256']=fingerprint(p)
    write_new_json(tmp_path/'plan.json',p)
    result,session,_=run(w,tmp_path/'plan.json',tmp_path/'run')
    assert result['status']=='PASS_SELECTED_PROGRAM' and session.effects['selective_replay_hits']>0
    # This replay tests measured lowering, not a claim of an automatic CPU winner.


def test_cache_mutation_detection_remains_enabled():
    from distillschedule.open_workload.checkpoint_policy import contexts
    from torch.utils.checkpoint import checkpoint
    x=torch.randn(3,4,requires_grad=True);w=torch.randn(4,5,requires_grad=True)
    def invalid(a,b):
        value=a@b;value.add_(1)
        return torch.sin(value)
    y=checkpoint(invalid,x,w,use_reentrant=False,
                 context_fn=lambda:contexts('save_matmul',{},set(),{}))
    with pytest.raises(RuntimeError,match='mutated'):y.sum().backward()


def worker(rank,rendezvous,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        from distillschedule.open_workload.execution import ExecutionBackend
        w=workload(frequency=2,rank=rank,world=2);backend=ExecutionBackend(w)
        g=workload(frequency=2,world=2,global_batch=True)
        ref=Session(g,candidates(g,limit=1)['candidates'][0]);expected=ref.result_state(ref.run())['state']
        implementations=backend.inventory(5)['candidates']
        for base in implementations:
            option={**base,'local':{**base['local'],'checkpoint_bindings':['forward'],
                                   'checkpoint_policy':'save_attention'}}
            session=backend.session(option)
            actual=session.result_state(session.run())['state']
            assert compare(expected,actual,rtol=4e-5,atol=4e-7)['status']=='PASS'
            assert session.effects['selective_replay_hits']>0
        Path(output,f'rank{rank}.json').write_text(json.dumps({'variants':len(implementations)}))
    finally:dist.destroy_process_group()


def test_selective_attention_composes_with_real_ddp_and_fsdp(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'rdzv'),str(tmp_path)),nprocs=2,join=True)
    assert json.loads((tmp_path/'rank0.json').read_text())['variants']==5
