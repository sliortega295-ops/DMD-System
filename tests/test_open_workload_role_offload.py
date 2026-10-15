"""Saved producer activations are genuinely stored and fetched before native VJP."""
import torch
from torch import nn
import pytest
from distillschedule.open_workload.role_offload import HostBudget,ActivationOffloader


def test_offload_prefetch_restores_original_autograd_and_adam():
    with torch.random.fork_rng():
        torch.manual_seed(37)
        base=nn.Sequential(nn.Linear(8,16),nn.GELU(),nn.Linear(16,4))
    import copy
    a=copy.deepcopy(base);b=copy.deepcopy(base)
    oa=torch.optim.Adam(a.parameters(),lr=.001);ob=torch.optim.Adam(b.parameters(),lr=.001)
    x=torch.randn(2,8)
    reference=a(x).square().mean();reference.backward();oa.step()
    budget=HostBudget(2**20);off=ActivationOffloader('cpu',b,budget)
    with off.hooks(): y=b(x).square().mean()
    assert off.statistics['packed_tensors']>0
    assert budget.used>0
    off.prefetch();y.backward();ob.step();off.close()
    assert off.statistics['unpacked_tensors']>0
    assert off.statistics['prefetched_tensors']>0
    assert budget.used==0
    for p,q in zip(a.parameters(),b.parameters()):assert torch.allclose(p,q,atol=1e-7,rtol=1e-6)
    for p,q in zip(oa.state.values(),ob.state.values()):
        for k in ('exp_avg','exp_avg_sq'):assert torch.allclose(p[k],q[k],atol=1e-7,rtol=1e-6)


def test_budget_failure_and_idempotent_release():
    m=nn.Linear(4,8);b=HostBudget(1);o=ActivationOffloader('cpu',m,b)
    with pytest.raises(ValueError):
        with o.hooks():m(torch.randn(2,4))
    assert b.used==0;o.close();o.close()


def test_inplace_change_of_original_saved_input_rejected():
    m=nn.Linear(3,2)
    x=torch.randn(2,3,requires_grad=True);b=HostBudget(1024)
    o=ActivationOffloader('cpu',m,b)
    with o.hooks():y=m(x).sum()
    with torch.no_grad():x.add_(1)
    with pytest.raises(ValueError,match='modified in-place'):y.backward()
    o.close();assert b.used==0
