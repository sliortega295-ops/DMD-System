"""Opt-in component observations from the EXISTING native role executor.

Only replicated, resident, window-one anchors are timed. Synchronization and
saved-tensor accounting deliberately perturb this diagnostic run, not the
selected production cycle. These are isolated operation observations, never a
measured end-to-end speedup or a model of communication contention.
"""
from contextlib import contextmanager
import time

import torch
import torch.distributed as dist
from distillschedule.core.contracts import require


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class ComponentRecorder:
    def __init__(self, device, enabled=False):
        self.device = torch.device(device)
        self.enabled = enabled
        self.records = []
        self.saved_bytes = {}

    @contextmanager
    def span(self, action, index=None):
        if not self.enabled:
            yield
            return
        synchronize(self.device)
        before = (torch.cuda.memory_allocated(self.device)
                  if self.device.type == 'cuda' else 0)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter_ns()
        yield
        synchronize(self.device)
        elapsed = time.perf_counter_ns() - started
        peak = (torch.cuda.max_memory_allocated(self.device)
                if self.device.type == 'cuda' else 0)
        self.records.append({'action': action, 'microbatch': index,
                             'duration_ns': elapsed, 'peak_allocated_bytes': peak,
                             'extra_allocated_bytes': max(0, peak-before)})

    @contextmanager
    def saved(self, module, index):
        if not self.enabled:
            yield
            return
        # Keep exactly autograd's saved values, not an additional strong-reference
        # census. Parameter storages are already included in static model memory.
        excluded = {(str(t.device), t.untyped_storage().data_ptr())
                    for t in (*module.parameters(), *module.buffers())}
        sizes = {}
        def pack(t):
            key = (str(t.device), t.untyped_storage().data_ptr())
            if key not in excluded:
                sizes[key] = t.untyped_storage().nbytes()
            return (t, t._version)
        def unpack(value):
            tensor, version = value
            require(tensor._version == version,
                    'saved component-profile tensor mutated before backward',
                    path='role.components')
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield
        self.saved_bytes[str(index)] = sum(sizes.values())

    def result(self):
        return {'enabled': self.enabled, 'records': self.records,
                'saved_bytes_by_microbatch': self.saved_bytes,
                'timing': 'synchronized_native_calls' if self.enabled else 'NOT_RUN',
                'scope': 'same_task_same_pools_replicated_resident_only'}


def profile_links(task, layout, device, *, samples=3):
    """Measure each used direction independently, never time producer waiting.

    Every rank joins barriers in the same deterministic order. Timing includes
    host dispatch/jitter and CUDA completion, not a fabricated half-RTT. There
    is no concurrent model work; NIC/copy-engine contention remains unmodeled.
    Scratch tensors use no model state, inputs, or random-number generators.
    """
    require(dist.is_initialized(), 'allocated process group required', path='role.components')
    require(type(samples) is int and 1 <= samples <= 8,
            'invalid link sample budget', path='role.components')
    device = torch.device(device)
    peers = sorted({layout.owners(i) for i in range(task.microbatches)})
    require(len(peers) <= 64, 'bounded link profiling supports at most 64 pairs',
            path='role.components')
    rank = dist.get_rank()
    rows = {}
    for producer, score in peers:
        for source, target in ((producer, score), (score, producer)):
            tensor = (torch.zeros(task.output_shape, device=device,
                                  dtype=getattr(torch, task.dtype))
                      if rank in (source, target) else None)
            observations = []
            for repeat in range(samples+1):
                synchronize(device)
                dist.barrier()
                started = time.perf_counter_ns()
                if rank == source:
                    dist.send(tensor, dst=target)
                elif rank == target:
                    dist.recv(tensor, src=source)
                synchronize(device)
                duration = time.perf_counter_ns()-started
                reports = [None]*dist.get_world_size()
                dist.all_gather_object(reports, duration if rank in (source, target) else 0)
                if repeat:
                    observations.append(max(reports[source], reports[target]))
            rows[f'{source}:{target}'] = observations
            del tensor
    return {'samples_ns': rows, 'warmup_per_direction': 1,
            'samples_per_direction': samples, 'bytes': task.bytes_per_latent,
            'scope': 'isolated_directional_send_recv_max_endpoint_wall',
            'gpu_execution': 'NOT_RUN' if device.type=='cpu' else 'CUDA_NOT_VALIDATED'}
