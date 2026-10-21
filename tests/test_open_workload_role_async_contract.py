"""CUDA ordering contracts plus an optional real-device race regression.

Fake streams assert the host-side API ordering only; the CUDA test is explicitly
skipped without hardware. Neither fake events nor CPU parity prove GPU safety.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import weakref
import pytest
import torch
from torch import nn
from distillschedule.open_workload.role_offload import ActivationOffloader, HostBudget, SavedLease


class Event:
    def __init__(self, log): self.log=log
    def record(self, stream): self.log.append(('record',stream.name))
    def synchronize(self): self.log.append(('event_sync',))


class Stream:
    def __init__(self, name, log): self.name=name;self.log=log
    def wait_stream(self, other): self.log.append(('wait_stream',self.name,other.name))
    def wait_event(self, event): self.log.append(('wait_event',self.name))
    def synchronize(self): self.log.append(('stream_sync',self.name))


class TensorStub:
    def __init__(self, device, log): self.device=device;self.log=log;self._version=0
    def untyped_storage(self): return SimpleNamespace(data_ptr=lambda:100)
    def numel(self): return 4
    def element_size(self): return 4
    def detach(self): return self
    def record_stream(self, stream): self.log.append(('record_storage',stream.name))
    def copy_(self, source, non_blocking): self.log.append(('d2h',));return self
    def to(self, device, non_blocking): self.log.append(('h2d',));return TensorStub(device,self.log)


def cuda_stubs(monkeypatch):
    log=[];device=torch.device('cuda:0');producer=Stream('producer',log)
    monkeypatch.setattr(torch.cuda,'Stream',lambda **_:Stream('copy',log))
    monkeypatch.setattr(torch.cuda,'current_stream',lambda *_:producer)
    monkeypatch.setattr(torch.cuda,'Event',lambda:Event(log))
    @contextmanager
    def scope(stream):
        log.append(('enter',stream.name));yield
    monkeypatch.setattr(torch.cuda,'stream',scope)
    return log,device


def test_pack_orders_producer_before_copy_then_records_lifetime(monkeypatch):
    log,device=cuda_stubs(monkeypatch)
    o=ActivationOffloader(device,nn.Identity(),HostBudget(1024))
    t=TensorStub(device,log);host=TensorStub(torch.device('cpu'),log)
    monkeypatch.setattr(torch,'Tensor',TensorStub)
    monkeypatch.setattr(torch,'empty_like',lambda *a,**k:host)
    lease=o.pack(t)
    assert log.index(('wait_stream','copy','producer'))<log.index(('d2h',))
    assert log.index(('d2h',))<log.index(('record','copy'))
    assert ('record_storage','copy') in log and o.budget.used==16
    o.close();assert o.budget.used==0


def test_on_demand_h2d_lease_waits_for_backward_stream(monkeypatch):
    log,device=cuda_stubs(monkeypatch)
    o=ActivationOffloader(device,nn.Identity(),HostBudget(1024))
    original=TensorStub(device,log)
    lease=SavedLease(weakref.ref(original),0,TensorStub(torch.device('cpu'),log),device,Event(log))
    o.leases.append(lease);o.budget.reserve(16);o.packed_bytes=16
    value=o.unpack(lease)
    assert value.device==device and len(lease.host_read_events)==1
    assert log.index(('wait_event','producer'))<log.index(('h2d',))<log.index(('record','producer'))
    o.close();assert ('event_sync',) in log and o.budget.used==0
    with pytest.raises(ValueError,match='unpack after'):o.unpack(lease)


def test_failed_pack_drains_copy_before_returning_host_budget(monkeypatch):
    log,device=cuda_stubs(monkeypatch)
    o=ActivationOffloader(device,nn.Identity(),HostBudget(1024))
    t=TensorStub(device,log)
    monkeypatch.setattr(torch,'Tensor',TensorStub)
    monkeypatch.setattr(torch,'empty_like',lambda *a,**k:TensorStub(torch.device('cpu'),log))
    def fail(): raise RuntimeError('event creation failed')
    monkeypatch.setattr(torch.cuda,'Event',fail)
    with pytest.raises(RuntimeError,match='event creation'):o.pack(t)
    assert log.index(('d2h',))<log.index(('stream_sync','copy'))
    assert o.budget.used==0 and not o.leases


def test_cpu_prefetch_does_not_allocate_unbudgeted_duplicate():
    o=ActivationOffloader('cpu',nn.Identity(),HostBudget(1024))
    t=torch.ones(4,requires_grad=True);lease=o.pack(t);used=o.budget.used
    o.prefetch()
    assert lease.prefetched is lease.host and o.budget.used==used
    o.close();assert o.budget.used==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA device required; not covered by mocks')
@pytest.mark.parametrize('prefetch',[False,True])
def test_real_cuda_producer_copy_backward_dependency(prefetch):
    device=torch.device('cuda',torch.cuda.current_device())
    producer=torch.cuda.Stream(device=device)
    model=nn.Linear(32,16,device=device)
    model.requires_grad_(False)
    x=torch.randn(8,32,device=device,requires_grad=True)
    producer.wait_stream(torch.cuda.current_stream(device))
    o=ActivationOffloader(device,model,HostBudget(2**20))
    with torch.cuda.stream(producer):
        # Delay generation to make a missing D2H dependency observable.
        torch.cuda._sleep(2_000_000)
        with o.hooks(): y=model(x.sin()).square().mean()
        if prefetch:o.prefetch()
        y.backward()
        o.close()
    torch.cuda.current_stream(device).wait_stream(producer)
    expected=x.detach().clone().requires_grad_()
    model(expected.sin()).square().mean().backward()
    torch.testing.assert_close(x.grad,expected.grad)
    assert o.budget.used==0
