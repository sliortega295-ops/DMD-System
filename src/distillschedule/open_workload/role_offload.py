"""Autograd activation CPU-offload/prefetch with bounded pinned host leases.

This offloads only tensors saved for the producer's OWN delayed backward graph.
Parameters/buffers remain resident and are excluded by physical storage keys.
For CUDA, data is staged on an explicit copy stream with events; prefetch can
start before waiting for Score's returned VJP. CPU/Gloo is a correctness fixture
and does not prove pinned-bandwidth or accelerator overlap on A100/H100.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import weakref
import torch
from distillschedule.core.contracts import require


@dataclass
class HostBudget:
    limit: int
    used: int = 0
    peak: int = 0

    def reserve(self, size):
        require(type(self.limit) is int and self.limit>0,
                'positive explicit per-rank host budget required',path='role.offload')
        require(self.used+size<=self.limit,
                'offloaded autograd activations exceed host budget',path='role.offload')
        self.used+=size;self.peak=max(self.peak,self.used)

    def release(self, size):
        self.used-=size
        require(self.used>=0,'host lease accounting underflow',path='role.offload')


@dataclass
class SavedLease:
    original: object
    version: int
    host: torch.Tensor
    device: torch.device
    completed: object = None
    prefetched: torch.Tensor | None = None
    prefetched_event: object = None


class ActivationOffloader:
    def __init__(self, device, model, budget:HostBudget):
        self.device=torch.device(device);self.budget=budget
        self.parameter_storage={(str(x.device),x.untyped_storage().data_ptr())
                                for x in (*model.parameters(),*model.buffers())}
        self.leases=[];self.closed=False
        self.packed=0;self.unpacked=0;self.prefetched=0;self.packed_bytes=0
        self._stream=(torch.cuda.Stream(device=self.device) if self.device.type=='cuda' else None)

    def pack(self, tensor):
        require(not self.closed, 'cannot pack after context is closed',path='role.offload')
        if not isinstance(tensor,torch.Tensor):return tensor
        key=(str(tensor.device),tensor.untyped_storage().data_ptr())
        if key in self.parameter_storage or tensor.device!=self.device:
            return tensor
        byte_count=tensor.numel()*tensor.element_size()
        self.budget.reserve(byte_count)
        try:
            if self.device.type=='cuda':
                host=torch.empty_like(tensor.detach(),device='cpu',pin_memory=True,
                                      memory_format=torch.preserve_format)
                with torch.cuda.stream(self._stream):
                    host.copy_(tensor.detach(),non_blocking=True)
                    event=torch.cuda.Event();event.record(self._stream)
                    tensor.record_stream(self._stream)
            else:
                host=tensor.detach().to('cpu',copy=True)
                event=None
        except BaseException:
            self.budget.release(byte_count)
            raise
        lease=SavedLease(weakref.ref(tensor),tensor._version,host,self.device,event)
        self.leases.append(lease)
        self.packed+=1;self.packed_bytes+=byte_count
        return lease

    def unpack(self, item):
        if not isinstance(item,SavedLease):return item
        source=item.original()
        require(source is None or source._version==item.version,
                'a saved tensor was modified in-place before backward',path='role.offload')
        self.unpacked+=1
        if self.device.type=='cuda':
            stream=torch.cuda.current_stream(self.device)
            if item.prefetched is not None:
                stream.wait_event(item.prefetched_event)
                item.prefetched.record_stream(stream)
                return item.prefetched
            stream.wait_event(item.completed)
            return item.host.to(self.device,non_blocking=True)
        return item.prefetched if item.prefetched is not None else item.host

    def prefetch(self):
        require(not self.closed,'prefetch after release',path='role.offload')
        for item in self.leases:
            if item.prefetched is not None:continue
            if self.device.type=='cuda':
                with torch.cuda.stream(self._stream):
                    self._stream.wait_event(item.completed)
                    staged=item.host.to(self.device,non_blocking=True)
                    event=torch.cuda.Event();event.record(self._stream)
                item.prefetched=staged;item.prefetched_event=event
            else:
                item.prefetched=item.host.clone()
            self.prefetched+=1

    @contextmanager
    def hooks(self):
        require(not self.closed,'cannot reopen closed offload',path='role.offload')
        with torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack):
            yield

    def close(self):
        if self.closed:return
        # A CUDA stream may still read pinned staging buffers when the selected
        # VJP finishes. Synchronize before releasing host leases to avoid UAF.
        if self._stream is not None:self._stream.synchronize()
        self.budget.release(self.packed_bytes)
        self.closed=True
        self.leases.clear()

    @property
    def statistics(self):
        return {'packed_tensors':self.packed,'packed_host_bytes':self.packed_bytes,
                'prefetched_tensors':self.prefetched,'unpacked_tensors':self.unpacked,
                'device':self.device.type,
                'true_cuda_async_copy':self.device.type=='cuda'}
