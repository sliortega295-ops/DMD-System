import random
import pytest
import torch
from distillschedule.open_workload.torch_backend import Session,candidates,compare
from test_open_workload_execution import build,forward


def stochastic(ctx):
    y=ctx.state['learner'](next(iter(ctx.inputs.values())))
    return torch.nn.functional.dropout(y,p=.2,training=True)*(.5+random.random())


def test_python_and_torch_rng_are_replayed_without_global_side_effects():
    w=build();w.bindings={**w.bindings,'forward':stochastic}
    before=torch.get_rng_state().clone(); py=random.getstate()
    reference=Session(w,candidates(w)['candidates'][0]);out=reference.run();expected=reference.result_state(out)
    for impl in candidates(w)['candidates']:
        s=Session(w,impl);out=s.run()
        assert compare(expected,s.result_state(out))['status']=='PASS'
    assert torch.equal(before,torch.get_rng_state()) and py==random.getstate()


def test_scalar_closure_changes_identity():
    w=build()
    def factory(scale):
        def wrapped(ctx): return forward(ctx)*scale
        return wrapped
    w.bindings={**w.bindings,'forward':factory(1.)};a=w.identity
    w.bindings={**w.bindings,'forward':factory(2.)};b=w.identity
    assert a!=b


def test_shared_trainable_roles_are_not_silently_admitted():
    w=build();w.state['guide']=w.state['learner']
    with pytest.raises(ValueError,match='trainable storage'): w.validate()
