"""Hardware-bound, bounded micro-profiling for the generic workload planner.

The objective is to measure THIS allocation rather than assume A100/H100 winners.
Microbenchmarks are diagnostics: they never substitute their timings for a
call's native measured cost, and CPU/Gloo records never claim GPU performance.
No GPU or process group is allocated by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import statistics
import time
from typing import Any

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint

SCHEMA = 'distillschedule.hardware_probe.v1'


def gpu_family(name: str, capability: tuple[int, int]) -> str:
    """Classify known A100/H100 parts without assuming other SMs are equivalent."""
    normalized = name.lower().replace('nvidia', '').strip()
    if 'a100' in normalized and tuple(capability) == (8, 0):
        return 'NVIDIA_A100_SM80'
    if 'h100' in normalized and tuple(capability) == (9, 0):
        return 'NVIDIA_H100_SM90'
    return 'OTHER_CUDA_UNVALIDATED'


def hardware_facts(device) -> dict[str, Any]:
    """Stable hardware/runtime facts for calibration and playback identity.

    Dynamic power and clock measurements are intentionally excluded: adapting
    to their drift is a new profile, not a mismatch in model identity.
    """
    import torch
    target = torch.device(device)
    facts = {'device_type': target.type,
             'torch_cuda_build': torch.version.cuda,
             'precision': {'allow_tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
                           'allow_tf32_cudnn': bool(torch.backends.cudnn.allow_tf32)}}
    if target.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA hardware fingerprint needs an available GPU', path='open.hardware')
        properties = torch.cuda.get_device_properties(target)
        capability = (properties.major, properties.minor)
        facts.update({'gpu_name': properties.name, 'gpu_family': gpu_family(properties.name, capability),
                      'compute_capability': list(capability), 'memory_total_bytes': int(properties.total_memory),
                      'device_uuid': str(getattr(properties, 'uuid', 'UNKNOWN'))})
        try:
            facts['nccl_version'] = list(torch.cuda.nccl.version())
        except (RuntimeError, AttributeError, TypeError):
            facts['nccl_version'] = None
    else:
        facts['gpu_family'] = 'CPU_NOT_GPU_EVIDENCE'
    return facts


@dataclass(frozen=True)
class ProbeConfig:
    warmup: int = 1
    samples: int = 3
    transfer_mib: int = 4
    allreduce_mib: tuple[int, ...] = (1, 8)

    def __post_init__(self):
        require(type(self.warmup) is int and 0 <= self.warmup <= 8, 'invalid hardware warmup', path='open.hardware')
        require(type(self.samples) is int and 1 <= self.samples <= 16, 'invalid hardware samples', path='open.hardware')
        require(type(self.transfer_mib) is int and 0 <= self.transfer_mib <= 64,
                'hardware H2D transfer limit exceeded', path='open.hardware')
        require(isinstance(self.allreduce_mib, tuple) and len(self.allreduce_mib) <= 4
                and all(type(n) is int and 1 <= n <= 64 for n in self.allreduce_mib),
                'allreduce probe sizes must be <=64 MiB', path='open.hardware')


def _timed_cuda(action, device, cfg: ProbeConfig) -> dict[str, Any]:
    """Synchronized GPU-event and host wall timings, reported separately."""
    import torch
    for _ in range(cfg.warmup):
        action()
    torch.cuda.synchronize(device)
    events_ns, wall_ns = [], []
    for _ in range(cfg.samples):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record(torch.cuda.current_stream(device))
        start = time.perf_counter_ns()
        action()
        end.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
        wall_ns.append(max(1, time.perf_counter_ns() - start))
        events_ns.append(max(1, int(round(begin.elapsed_time(end) * 1_000_000))))
    return {'event_samples_ns': events_ns, 'wall_samples_ns': wall_ns,
            'event_median_ns': int(statistics.median(events_ns)),
            'wall_median_ns': int(statistics.median(wall_ns)),
            'timing': 'cuda_event_default_stream_plus_host_wall'}


def probe(device, *, distributed: bool, config: ProbeConfig | None = None) -> dict[str, Any]:
    """Measure isolated pinned H2D and optional world-group NCCL collectives.

    Every rank in a NCCL group must call this function in the same order.
    The benchmark uses new scratch tensors and never modifies model/optimizer.
    """
    import torch
    import torch.distributed as dist
    config = config or ProbeConfig()
    target = torch.device(device)
    hardware = hardware_facts(target)
    if target.type != 'cuda':
        return {'schema_version': SCHEMA, 'status': 'NOT_GPU_PROFILE',
                'hardware': hardware, 'settings': asdict(config), 'benchmarks': [],
                'gpu_execution': 'NOT_RUN', 'claimable_speedup': False}
    require(distributed == (dist.is_initialized() and dist.get_world_size() > 1),
            'distributed probe declaration does not match process group', path='open.hardware')
    if distributed:
        require(dist.get_backend() == 'nccl', 'CUDA allreduce probe requires NCCL', path='open.hardware')
    results = []
    if config.transfer_mib:
        byte_count = config.transfer_mib * 2**20
        cpu = torch.empty(byte_count, dtype=torch.uint8, device='cpu', pin_memory=True)
        dst = torch.empty_like(cpu, device=target)
        timing = _timed_cuda(lambda: dst.copy_(cpu, non_blocking=True), target, config)
        results.append({'kind': 'pinned_h2d', 'bytes': byte_count, **timing})
        del dst, cpu
    if distributed:
        for mib in config.allreduce_mib:
            byte_count = mib * 2**20
            tensor = torch.ones(byte_count // 4, dtype=torch.float32, device=target)
            # Each rank executes the exact same sequence. Never call a barrier
            # inside only one rank's branch: that can deadlock NCCL.
            dist.barrier()
            timing = _timed_cuda(lambda: dist.all_reduce(tensor), target, config)
            dist.barrier()
            results.append({'kind': 'nccl_allreduce_sum', 'bytes': byte_count,
                            'group_size': dist.get_world_size(), **timing})
            del tensor
    record = {'schema_version': SCHEMA, 'status': 'MEASURED_ISOLATED_GPU_PROBES',
              'hardware': hardware, 'settings': asdict(config), 'benchmarks': results,
              'gpu_execution': 'CUDA_PROBE_ONLY', 'claimable_speedup': False,
              'limitations': ['scratch transfers are not full workload memory traffic',
                              'isolated NCCL does not estimate overlapped communication',
                              'CUDA event scopes cannot replace Nsight critical-path analysis']}
    record['digest'] = fingerprint(record)
    return record


def plan_memory_limit(device, *, distributed: bool, consensus, requested: int | None = None,
                      reserve_bytes: int = 2 * 2**30) -> dict[str, Any]:
    """Safe, current-allocation-specific admission budget for A100/H100.

    Include tensors already allocated by the workload factory as well as
    allocator-reserved reusable cache; free VRAM alone is not comparable
    with the planner's total static+workspace estimate.
    Return the minimum headroom across all ranks; never infer GPU capacity from
    an A100/H100 product label. On CPU the caller must provide a memory limit.
    """
    import torch
    device = torch.device(device)
    require(type(reserve_bytes) is int and 0 <= reserve_bytes <= 16 * 2**30,
            'GPU reserve must be 0..16 GiB', path='open.hardware')
    require(requested is None or (type(requested) is int and requested > 0),
            'positive explicit memory limit required', path='open.hardware')
    if device.type != 'cuda':
        require(requested is not None,
                'CPU memory limit must be explicit (CPU tensor storage is not RSS)', path='open.hardware')
        return {'limit_bytes':requested, 'source':'explicit_cpu_tracked_tensor_storage',
                'rank_budgets':[], 'reserve_bytes':0}
    require(torch.cuda.is_available(), 'CUDA capacity requires actual GPU', path='open.hardware')
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    require(0 <= allocated <= reserved and free + reserved <= total,
            'CUDA allocator capacity snapshot inconsistent', path='open.hardware')
    local = {'free_bytes':int(free),'total_bytes':int(total),
             'torch_allocated_bytes':int(allocated),
             'torch_reserved_bytes':int(reserved),
             'conservative_limit_bytes':int(free + reserved - reserve_bytes)}
    records = consensus(local) if distributed else [local]
    require(records and all(type(row.get('conservative_limit_bytes')) is int
                            and row['conservative_limit_bytes'] > 0 for row in records),
            'not enough real free CUDA memory after reserve', path='open.hardware')
    allowed = min(row['conservative_limit_bytes'] for row in records)
    require(requested is None or requested <= allowed,
            'explicit budget exceeds observed minimum GPU capacity', path='open.hardware')
    return {'limit_bytes':allowed if requested is None else requested,
            'source':'automatic_min_free_plus_allocator_reserved_after_reserve' if requested is None
                      else 'explicit_bounded_by_live_gpu_headroom',
            'rank_budgets':records, 'reserve_bytes':reserve_bytes}
