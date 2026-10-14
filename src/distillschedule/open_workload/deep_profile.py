"""Opt-in deep trace of ONE validated native plan, never cost calibration.

PyTorch Kineto traces are diagnostic. The profiler is forbidden from replacing
unprofiled matched candidate timings in the scheduler, and trace wall times
cannot be presented as a speedup result. This works without Nsight access.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import file_sha256

SCHEMA = 'distillschedule.selected_program_profile.v1'


def selected_plan_trace(call: Callable, output, *, rank: int, device) -> tuple[object, dict]:
    import torch
    require(callable(call) and type(rank) is int and rank>=0,
            'profile requires an executable plan and rank', path='open.deep_profile')
    device = torch.device(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA trace requires real device', path='open.deep_profile')
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    path = Path(output) / f'torch-profiler-rank{rank:04d}.json'
    require(not path.exists(), 'do not overwrite a recorded selected-plan trace', path='open.deep_profile')
    with torch.profiler.profile(activities=activities, profile_memory=True,
                                record_shapes=False, with_stack=False, with_flops=False) as profiler:
        result = call()
    profiler.export_chrome_trace(str(path))
    stats = profiler.key_averages()
    return result, {'schema_version':SCHEMA, 'rank':rank, 'source':'torch_kineto_selected_plan_only',
                    'path':str(path), 'sha256':file_sha256(path),
                    'trace_bytes':path.stat().st_size,
                    'operator_groups':len(stats), 'activities':['cpu']+(['cuda'] if device.type=='cuda' else []),
                    'gpu_execution':'NATIVE_CUDA_PROFILE' if device.type=='cuda' else 'CPU_TRACE_NOT_GPU_EVIDENCE',
                    'timing_claim':'DIAGNOSTIC_ONLY_NOT_CALIBRATED_SPEEDUP',
                    'claimable_speedup':False}
