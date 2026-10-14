"""Actual arbitrary Torch models/autograd/Adam, not a canonical DMD speed claim."""
from copy import deepcopy
import pytest
import torch
from torch import nn
from distillschedule.workloads.program import Call,Program
from distillschedule.open_workload.torch_backend import Workload,Session,candidates,compare,storage_bytes


class NewModel(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.base=nn.Linear(width,width).requires_grad_(False)
        self.adapter=nn.Linear(width,width,bias=False)
    def forward(self,x):
        return torch.sin(self.base(x)+self.adapter(x))


def forward(ctx):
    return ctx.state['learner'](next(iter(ctx.inputs.values())))

def teacher(ctx):
    return ctx.state['guide'](next(iter(ctx.inputs.values())))

def loss(ctx):
    x,y=ctx.inputs.values()
    return (x-y).square().mean()

def backward(ctx):
    next(iter(ctx.inputs.values())).backward()

def update(ctx):
    ctx.state['adam'].step();ctx.state['adam'].zero_grad(set_to_none=True)


def build(n=2,width=4,length=8):
    with torch.random.fork_rng():
        torch.manual_seed(11)
        learner=NewModel(width);guide=deepcopy(learner).requires_grad_(False)
        guide.adapter.weight.fill_(.05)
        x=torch.randn(length,width)
    state={'learner':learner,'guide':guide,'adam':torch.optim.Adam(learner.adapter.parameters(),lr=.001),'grad':None}
    calls=[]
    for i in range(n):
        calls.extend((Call(f'p{i}','forward',('x',),('learner',),grad='grad',replay_safe=True),
                      Call(f't{i}','teacher',('x',),('guide',)),
                      Call(f'l{i}','loss',(f'p{i}',f't{i}'),grad='grad'),
                      Call(f'b{i}','backward',(f'l{i}',),('grad',),('grad',),grad='grad',backward_of=(f'l{i}',)),
                      Call(f'u{i}','update',(f'b{i}',),('learner','adam','grad'),('learner','adam','grad'))))
    p=Program(tuple(calls),dict.fromkeys(state,0),('x',),(f'l{n-1}',),{'n':n,'width':width,'length':length})
    return Workload(p,{'forward':forward,'teacher':teacher,'loss':loss,'backward':backward,'update':update},
                    state,{'x':x},immutable_modules=('learner.base','guide.base'))


@pytest.mark.parametrize('n,width,length',[(1,4,8),(2,8,13),(5,4,3)])
def test_new_models_all_capabilities_preserve_actual_updates(n,width,length):
    w=build(n,width,length); identity=w.identity
    ref=Session(w,candidates(w)['candidates'][0]); out=ref.run();expected=ref.result_state(out)
    for impl in candidates(w)['candidates']:
        s=Session(w,impl);out=s.run(measure=True)
        assert compare(expected,s.result_state(out))['status']=='PASS'
        assert s.receipt['dispatcher']['final_versions']['learner']==n
        assert s.receipt['gpu_execution']=='NOT_RUN'
        if impl['share_immutable']:
            assert s.state['learner'].base.weight is s.state['guide'].base.weight
            assert s.initial_storage < ref.initial_storage
            assert s.state['learner'].adapter.weight is not s.state['guide'].adapter.weight
        if impl['checkpoint_bindings']:
            assert s.effects['checkpoint_calls']==n
            assert s.effects['recompute_calls']>=n
    assert w.identity==identity


def test_reject_external_model_closure():
    w=build(); external=w.state['learner']
    def invalid(ctx): return external(ctx.inputs['x'])
    w.bindings={**w.bindings,'forward':invalid}
    with pytest.raises(ValueError,match='captured'): w.validate()


def test_candidates_are_not_a_workload_name_map():
    w=build()
    assert len(candidates(w)['candidates'])==8
    w.immutable_modules=()
    assert len(candidates(w)['candidates'])==4
    assert candidates(w,limit=1)['exhaustive'] is False


def test_unavailable_or_malformed_implementation_rejected():
    w=build(); impl=candidates(w)['candidates'][0]
    with pytest.raises(ValueError): Session(w,{**impl,'backend':'unknown-fast-model'})
    with pytest.raises(ValueError): Session(w,{**impl,'checkpoint_bindings':['update']})


def test_mutating_declared_frozen_state_fails():
    w=build()
    def bad(ctx):
        ctx.state['guide'].base.weight.add_(1)
        return teacher(ctx)
    w.bindings={**w.bindings,'teacher':bad}
    s=Session(w,candidates(w)['candidates'][0])
    with pytest.raises(ValueError,match='immutable'): s.run()


def test_original_workload_state_not_mutated_by_calibration():
    w=build(); before=w.identity
    for impl in candidates(w)['candidates']: Session(w,impl).run()
    assert w.identity==before and not w.state['adam'].state
