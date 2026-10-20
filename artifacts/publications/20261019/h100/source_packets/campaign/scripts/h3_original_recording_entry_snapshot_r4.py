#!/usr/bin/env python3
"""Record the unchanged H3 entrypoint; never a substitute model/trainer.

This explicit campaign entry is used identically by both arms. CUDA reads,
finite checks, JSON and hashes happen only after the existing cycle timer.
Inside the timer the observer retains existing detached scalar references
and increments Python optimizer/scheduler counters. It adds no tensor copy,
synchronization, collective, RNG call, sampler call or model invocation.
"""
from __future__ import annotations

import contextlib
import copy
import datetime
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(function):
    return {'file': inspect.getsourcefile(function),
            'line': inspect.getsourcelines(function)[1],
            'sha256': hashlib.sha256(inspect.getsource(function).encode()).hexdigest()}


class Observer:
    def __init__(self):
        self.trainer = None
        self.timers = []
        self.pending_start = None
        self.scalars = {}
        self.adapter_loads = []
        self.optimizers = {'student': 0, 'fake': 0}
        self.schedulers = {'student': 0, 'fake': 0}
        self.completed_gates = []
        self.sources = {}
        self.setup_memory = None
        self.timed_peak_memory = None

    def remember_scalars(self, role, values):
        if role in self.scalars:
            raise RuntimeError('observer admits exactly one Student1+Fake5 cycle')
        # Existing detached tensors are retained, not copied/read or recomputed.
        self.scalars[role] = list(values)

    def completed_step(self, family, role):
        getattr(self, family)[role] += 1

    def start(self, value):
        if value is None or self.pending_start is not None or self.timers:
            raise RuntimeError('missing, repeated or overlapping original H100 timer')
        self.pending_start = float(value)

    def stop(self, seconds):
        if self.pending_start is None or not math.isfinite(float(seconds)) or seconds <= 0:
            raise RuntimeError('original H100 timer returned an invalid duration')
        self.timers.append({'start_perf_seconds': self.pending_start,
                            'stop_perf_seconds': self.pending_start + float(seconds),
                            'rank_wall_seconds': float(seconds)})
        self.pending_start = None

    def validate_bookkeeping(self):
        if len(self.timers) != 1 or self.pending_start is not None:
            raise RuntimeError('exactly one original timer must complete')
        if self.optimizers != {'student': 1, 'fake': 5}:
            raise RuntimeError('actual optimizer commit counts are not Student1/Fake5')
        if self.schedulers != {'student': 1, 'fake': 5}:
            raise RuntimeError('actual scheduler commit counts are not Student1/Fake5')
        if set(self.scalars) != {'student', 'fake'} or len(self.scalars['student']) != 2 or len(self.scalars['fake']) != 10:
            raise RuntimeError('physical8/GAS2 detached loss inventory mismatch')
        if sorted(self.completed_gates) != sorted(GATES):
            raise RuntimeError('not all original runtime validators returned normally')


GATES = ('_validate_matched_cycle', '_validate_boundary_offload_cycle',
         '_validate_fused_pointwise_cycle', '_validate_fused_rotary_cycle',
         '_validate_fused_qk_cycle', '_validate_fused_swiglu_cycle',
         '_validate_fa3_nograd_split_cycle', '_validate_fa3_replay_cache_cycle',
         '_validate_lora_scale1_cycle')


@contextlib.contextmanager
def observe_training(trainer_class, trainer_module, observer, *, memory_snapshot=None):
    """Delegate every original call once, preserving returns/errors and order."""
    restored = []

    def replace(owner, name, callback, *, source_key=None):
        original = getattr(owner, name)
        observer.sources[source_key or name] = source_identity(original)
        had_local = name in owner.__dict__
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            callback(args, result)
            return result
        setattr(owner, name, wrapped)
        restored.append((owner, name, original, had_local))

    def setup_done(args, _result):
        self = args[0]
        if observer.trainer is not None:
            raise RuntimeError('one real trainer instance required')
        observer.trainer = self
        if memory_snapshot is not None:
            observer.setup_memory = memory_snapshot()
        for family, names in (('optimizers', ('optimizer', 'fake_optimizer')),
                              ('schedulers', ('lr_scheduler', 'fake_lr_scheduler'))):
            for role, name in zip(('student', 'fake'), names):
                target = getattr(self, name)
                replace(target, 'step', lambda _args, _result, family=family, role=role:
                        observer.completed_step(family, role), source_key=f'{family}.{role}.step')

    def stopped(args, seconds):
        observer.stop(seconds)
        if memory_snapshot is not None:
            # The original stop() has ALREADY returned the timed duration.
            observer.timed_peak_memory = memory_snapshot()

    try:
        replace(trainer_class, 'setup', setup_done)
        replace(trainer_class, '_start_hardware_cycle_timer', lambda args, result: observer.start(result))
        replace(trainer_class, '_stop_hardware_cycle_timer', stopped)
        replace(trainer_class, '_student_step', lambda args, result: observer.remember_scalars('student', result))
        replace(trainer_class, '_fake_steps', lambda args, result: observer.remember_scalars('fake', result))
        for name in GATES:
            replace(trainer_class, name, lambda args, result, name=name: observer.completed_gates.append(name))
        replace(trainer_module, 'load_initial_adapter_exact',
                lambda args, result: observer.adapter_loads.append(copy.deepcopy(result)))
        yield
    finally:
        for owner, name, original, had_local in reversed(restored):
            if had_local:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


def memory_snapshot(torch):
    free, total = torch.cuda.mem_get_info()
    return {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
            'peak_allocated': torch.cuda.max_memory_allocated(),
            'peak_reserved': torch.cuda.max_memory_reserved(),
            'driver_free_snapshot': free, 'driver_total': total}


def finite_local_tensors(values, torch):
    """Post-timer local-shard check, never a DTensor collective or payload dump."""
    flags_by_device = {}
    count = 0
    for value in values:
        if not torch.is_tensor(value):
            continue
        local = value.to_local() if hasattr(value, 'to_local') else value
        if not local.is_floating_point():
            continue
        local = local.detach()
        flags_by_device.setdefault(str(local.device), []).append(torch.isfinite(local).all())
        count += 1
    if not count or not all(bool(torch.stack(flags).all().item()) for flags in flags_by_device.values()):
        raise RuntimeError('post-timer local trainable/Adam finite gate failed')
    return {'status': 'PASS_ACTUAL_LOCAL_SHARDS_FINITE', 'floating_tensor_count': count,
            'payload_exported': False, 'inside_timer': False}


def final_receipt(observer, torch):
    # Called ONLY after original entry.main() and its distributed cleanup return.
    observer.validate_bookkeeping()
    trainer = observer.trainer
    if trainer is None:
        raise RuntimeError('no real H3 trainer observed')
    rank = int(os.environ['RANK'])
    if (int(os.environ['WORLD_SIZE']) != 8 or int(os.environ['LOCAL_RANK']) != rank
            or torch.cuda.current_device() != rank or not trainer.hardware_contract.enabled):
        raise RuntimeError('physical8/logical16 device identity mismatch')
    if trainer.logical_samples_per_rank != 2:
        raise RuntimeError('actual GAS/census multiplicity changed')
    if sorted(row['role'] for row in observer.adapter_loads) != ['fake', 'student'] or any(
            row.get('status') != 'PASS_ACTUAL_CPU_TENSOR_EQUALITY' for row in observer.adapter_loads):
        raise RuntimeError('both initial adapters must actually load/compare exactly')
    losses = {role: [float(value.detach().cpu()) for value in values]
              for role, values in observer.scalars.items()}
    if not all(math.isfinite(value) for values in losses.values() for value in values):
        raise RuntimeError('nonfinite real losses; post-timer gate failed')
    finite = {}
    for role, parameters, optimizer in (
            ('student', trainer.trainable_params, trainer.optimizer),
            ('fake', trainer.fake_trainable_params, trainer.fake_optimizer)):
        finite[role + '_parameters'] = finite_local_tensors(parameters, torch)
        finite[role + '_Adam'] = finite_local_tensors(
            (value for state in optimizer.state.values() for value in state.values()), torch)
    grid = trainer.grid_replay_registration
    start = trainer._grid_replay_cycle_start
    delta = {key: int(grid.stats[key]) - int(start.get(key, 0)) for key in
             ('checkpoint_wrap_count', 'scoped_execution_count', 'cache_hit_count',
              'cache_miss_count', 'missing_key_count')}
    expected = {'checkpoint_wrap_count': 600, 'scoped_execution_count': 1200,
                'cache_hit_count': 1200, 'cache_miss_count': 0, 'missing_key_count': 0}
    if delta != expected or trainer.activation_policy != 'none' or trainer.boundary_offload_registration is not None:
        raise RuntimeError('actual Grid replay / disabled checkpoint-boundary policy changed')
    controller = trainer.shared_model.adaln_cache()
    remaining = sum('.adaln_proj.' in name for name, _ in trainer.shared_model.denoiser_module().named_parameters())
    dropped = int(controller.dropped_parameter_numel)
    if remaining or dropped < 12_000_000_000:
        raise RuntimeError('Grid parameter elimination not actually active')
    mechanisms = {}
    for name in ('fused_block_pointwise_registration', 'fused_rotary_registration',
                 'fused_qk_rmsnorm_rotary_registration', 'fused_swiglu_registration',
                 'fa3_nograd_split_registration', 'fa3_replay_cache_registration',
                 'lora_scale1_elision_registration'):
        registration = getattr(trainer, name, None)
        mechanisms[name] = {'enabled': bool(registration is not None and registration.enabled),
                            'actual_snapshot': None if registration is None else registration.snapshot()}
    if observer.setup_memory is None or observer.timed_peak_memory is None:
        raise RuntimeError('actual cold-setup and timed-peak memory snapshots are required')
    return {'status': 'PASS_REAL_WORKER_CENSUS_REPLAY_UPDATES_FINITE', 'rank': rank,
            'local_rank': int(os.environ['LOCAL_RANK']), 'logical_ranks': [rank, rank + 8],
            'actual_census': trainer.matched_cycle_census.snapshot(), 'timer': observer.timers[0],
            'actual_optimizer_commits': observer.optimizers, 'actual_scheduler_commits': observer.schedulers,
            'original_validators_returned': observer.completed_gates,
            'initial_adapter_actual_load_receipts': observer.adapter_loads,
            'losses_post_timer': losses, 'Grid_replay_delta': delta,
            'local_state_finite_post_timer': finite,
            'registered_adaln_parameters': remaining, 'dropped_adaln_parameter_numel': dropped,
            'activation_policy': trainer.activation_policy, 'checkpoint_boundary_cpu_copies': 0,
            'boundary_copy_zero_evidence': 'activation_policy none and no boundary registration; NOT total D2H attribution',
            'mechanisms': mechanisms, 'memory_after_setup': observer.setup_memory,
            'memory_after_timer_before_extra_checks': observer.timed_peak_memory,
            'strict_cross_arm_numerical_parity': 'NOT_RUN',
            'instrumentation_source': observer.sources,
            'extra_sync_collective_or_tensor_copy_inside_timer': False}


def main():
    run_id = os.environ['H3_RECEIPT_RUN_ID']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', run_id):
        raise ValueError('invalid immutable arm run ID')
    if os.environ.get('H3_RUN_GPU_LOCK_HELD') != '1':
        raise RuntimeError('campaign GPU ownership guard required')
    output = Path(os.environ['H3_RECEIPT_OUTPUT']).resolve()
    owned = Path('/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930').resolve()
    if not output.is_relative_to(owned):
        raise ValueError('worker evidence must remain in the owned persistent campaign')
    rank = int(os.environ['RANK'])
    path = output / f'rank_{rank:03d}.json'
    if path.exists():
        raise FileExistsError('old worker receipt cannot be overwritten')
    output.mkdir(exist_ok=True)
    stat = Path('/proc/self/stat').read_text().split(') ', 1)[1].split()
    initial = {'run_id': run_id, 'rank': rank, 'pid': os.getpid(), 'start_ticks': stat[19],
               'wrapper_sha256': file_sha(__file__), 'written_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'stage': 'REAL_ENTRY_IMPORT_PENDING', 'status': 'NOT_YET_COMPLETE'}
    start_path = output / f'rank_{rank:03d}.start.json'
    with start_path.open('x') as handle:
        json.dump(initial, handle, indent=2); handle.write('\n')
    observer = Observer()
    result = dict(initial)
    exit_code = 1
    try:
        # Original model registration, grid patches, trainer factory, sampler,
        # distributed init and training are all owned by this exact entrypoint.
        import torch
        from h3_a100 import grid_entrypoint as entry
        from h3_a100 import trainer as trainer_module
        with observe_training(trainer_module.MiniMaxH3A100DmdTrainer, trainer_module, observer,
                              memory_snapshot=lambda: memory_snapshot(torch)):
            entry.main()
        result.update(final_receipt(observer, torch), stage='ORIGINAL_ENTRY_AND_GATES_COMPLETE')
        exit_code = 0
    except BaseException as error:
        result.update(status='FAIL_REAL_WORKER', stage='ORIGINAL_ENTRY_OR_RECEIPT_GATE_FAILED',
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      observed_timer=observer.timers, observed_optimizer_commits=observer.optimizers,
                      original_validators_returned=observer.completed_gates)
        traceback.print_exc()
    finally:
        with path.open('x') as handle:
            json.dump(result, handle, indent=2, sort_keys=True); handle.write('\n')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
