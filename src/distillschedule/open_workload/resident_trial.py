"""Private, capacity-admitted serial trials on one already resident owner.

This is not checkpoint loading or a generic state_dict fallback. An explicit
source-bound adapter describes a complete boundary and restores its native
metadata after the common in-place tensor/Adam rollback. CUDA is a private,
single-device FP32 opt-in with explicit host/device allowances; it does not
admit model construction or public CUDA planning. CPU snapshots and streamed
comparison never create a second CUDA owner. Budgets are managed envelopes,
not total-process RSS or a promise that an allocator/OOM is recoverable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import os
from pathlib import Path
import traceback
from threading import Lock
from collections.abc import Mapping

import torch

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint, file_sha256

SCHEMA = 'distillschedule.resident_trial.cpu.v1'
CUDA_SCHEMA = 'distillschedule.resident_trial.cuda.v1'
_METHODS = ('_resident_trial_adapter', '_resident_trial_restore',
            '_resident_trial_cleanup', '_resident_trial_validate')
_RNG_ALLOWANCE = 65536
_CUDA_TRIALS = {}
_CUDA_TRIAL_LOCK = Lock()


def _require(condition, message):
    require(condition, message, path='open.resident_trial')


@dataclass(frozen=True)
class TrialBudget:
    host_limit_bytes: int
    snapshot_limit_bytes: int
    result_limit_bytes: int
    scratch_bytes: int
    reserve_bytes: int

    def validate(self):
        _require(all(type(value) is int and value > 0 for value in vars(self).values()),
                 'explicit positive host, snapshot, result, scratch and reserve budgets required')


@dataclass(frozen=True)
class CudaTrialBudget(TrialBudget):
    """Caller-declared allowances, not an empirical training-capacity claim.

    Device limits include the allocator's existing residency, not just this
    owner's tensors. All growth/gradient/activation/workspace/restore allowances
    are additive; reserve is unavailable for normal execution. Host growth
    includes host Adam steps and native EMA, separately from serialized results.
    """
    device_limit_bytes: int
    host_growth_bytes: int
    device_state_growth_bytes: int
    device_gradient_bytes: int
    device_activation_bytes: int
    device_workspace_bytes: int
    device_restore_bytes: int
    device_reserve_bytes: int


@dataclass(frozen=True)
class ResidentOwner:
    """Zero-copy inventory. Growth counts serialized copies, not unique storage."""
    model: torch.nn.Module
    optimizers: Mapping
    state: dict
    aliases: dict
    identity: dict
    growth_bytes: int
    output_bytes: int
    max_new_tensor_bytes: int = 0
    physical: tuple = ()
    retained: dict = field(default_factory=dict)
    schema_version: str = SCHEMA
    cuda_capable: bool = False
    host_growth_bytes: int = 0
    device_growth_bytes: int = 0
    gradient_bytes: int = 0


class RollbackError(RuntimeError):
    def __init__(self, original_error, rollback_error):
        super().__init__(f'resident owner poisoned: rollback failed ({rollback_error}); '
                         f'original failure: {original_error!r}')
        self.original_error, self.rollback_error = original_error, rollback_error


def native_state_view(model, optimizers):
    """Common zero-copy fields; owners must add *all* native/plain/host state."""
    return dict(model=model.state_dict(),
        optimizers={name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        training={name: module.training for name, module in model.named_modules()},
        requires_grad={name: parameter.requires_grad for name, parameter in model.named_parameters()})


def restore_native_state(model, optimizers, state):
    """Reattach original Adam slots, never cast/load a second optimizer state.

    Tensor contents have already been copied directly into retained destinations.
    Deleting lazy slots restores absence; existing moments keep their storage.
    """
    for name, module in model.named_modules():
        module.training = state['training'][name]
    for name, parameter in model.named_parameters():
        parameter.grad = None
        parameter.requires_grad_(state['requires_grad'][name])
    for name, optimizer in optimizers.items():
        saved = state['optimizers'][name]
        parameters = {}
        for group, original in zip(optimizer.param_groups, saved['param_groups'], strict=True):
            parameters.update(zip(original['params'], group['params'], strict=True))
            for key in tuple(group):
                if key != 'params': del group[key]
            group.update({key: value for key, value in original.items() if key != 'params'})
        optimizer.state.clear()
        optimizer.state.update({parameters[key]: value for key, value in saved['state'].items()})


def adam_growth(optimizers):
    """Conservative full AdamW moments and CPU step growth, even when warmed."""
    return sum(2 * p.numel() * p.element_size() + 4 for optimizer in optimizers.values()
               for group in optimizer.param_groups for p in group['params'])


def cuda_growth(optimizers, *, host_bytes=0, device_bytes=0):
    """Full lazy Adam/gradient allowance; warmed slots are not subtracted."""
    parameters = [p for opt in optimizers.values() for group in opt.param_groups for p in group['params']]
    size = sum(p.numel() * p.element_size() for p in parameters)
    return dict(cuda_capable=True, host_growth_bytes=host_bytes + 4 * len(parameters),
                device_growth_bytes=device_bytes + 2 * size, gradient_bytes=size)


def cuda_trial_active(owner):
    """Only this capacity-admitted transaction may expose private CUDA options.

    The marker alone is not a capability. Admission sets the flag after all
    capacity/layout checks, before source validation and snapshot allocation;
    failures close it without ever making a runnable trial available.
    """
    trial = getattr(owner, '_resident_trial_active', None)
    return (type(trial) is ResidentTrial and type(trial.budget) is CudaTrialBudget
            and trial._capacity_admitted and not trial.receipt['closed']
            and trial.owner is owner and not hasattr(owner, '_resident_trial_poisoned'))


def physical_attributes(modules, names):
    """Retain scoped flags/callbacks by reference, never serialize Python code."""
    return tuple((module, name, name in vars(module), vars(module).get(name))
                 for module in modules for name in names)


def _map(value, tensor, *, wire=False):
    if isinstance(value, torch.Tensor):
        _require(type(value) in (torch.Tensor, torch.nn.Parameter)
                 and value.device.type in ('cpu', 'cuda') and value.layout == torch.strided
                 and value.is_contiguous() and not value.is_conj() and not value.is_neg(),
                 'plain contiguous CPU/owned-CUDA tensors required; no DTensor/sparse/view-bit state')
        _require(not value.is_inference(), 'inference storage cannot be restored in place')
        return tensor(value)
    if isinstance(value, Mapping):
        _require(all(type(key) in (str, int) for key in value), 'only string/integer state keys supported')
        return {str(key) if wire else key: _map(item, tensor, wire=wire) for key, item in value.items()}
    if type(value) in (list, tuple):
        items = [_map(item, tensor, wire=wire) for item in value]
        return tuple(items) if type(value) is tuple and not wire else items
    _require(value is None or type(value) in (str, int, float, bool), 'opaque state is not a resident-trial capability')
    return value


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def _memory_inventory(value):
    leaves = []
    _map(value, lambda t: leaves.append(t))
    sizes = [t.numel() * t.element_size() for t in leaves]
    unique = {_storage_key(t): t.untyped_storage().nbytes() for t in leaves}
    return dict(serialized_bytes=sum(sizes), largest_tensor_bytes=max(sizes, default=0),
        host_storage_bytes=sum(size for (device, *_), size in unique.items() if device == 'cpu'),
        device_storage_bytes=sum(size for (device, *_), size in unique.items() if device != 'cpu'))


def _sizes(value):
    row = _memory_inventory(value)
    return (row['serialized_bytes'], row['largest_tensor_bytes'],
            row['host_storage_bytes'] + row['device_storage_bytes'])


def _copy(value, limit):
    # Admission is whole-tree, *before* allocating even the first leaf.
    _require(_sizes(value)[0] <= limit, 'snapshot/result tensor copy exceeds admitted bound')
    return _map(value, lambda t: t.detach().to(device='cpu', copy=True))


def _restore_tensors(original, saved):
    if isinstance(original, torch.Tensor):
        _require(original.shape == saved.shape and original.dtype == saved.dtype,
                 'snapshot layout changed')
        with torch.no_grad(): original.copy_(saved)
    elif isinstance(original, dict):
        _require(original.keys() == saved.keys(), 'snapshot fields changed')
        for key in original: _restore_tensors(original[key], saved[key])
    elif isinstance(original, (tuple, list)):
        _require(len(original) == len(saved), 'snapshot sequence changed')
        for a, b in zip(original, saved): _restore_tensors(a, b)


def _digest(value, *, scratch_bytes=65536):
    """Stream CUDA chunks to CPU; CPU leaves use zero-copy byte views."""
    def tensor(t):
        raw = t.detach().reshape(-1).view(torch.uint8)
        digest = hashlib.sha256()
        for start in range(0, raw.numel(), scratch_bytes):
            chunk = raw[start:start + scratch_bytes].cpu()
            digest.update(memoryview(chunk.numpy()))
            del chunk
        return {'shape': list(t.shape), 'dtype': str(t.dtype),
                'sha256': digest.hexdigest()}
    _require(type(scratch_bytes) is int and scratch_bytes > 0, 'positive hashing scratch required')
    return fingerprint(_map(value, tensor))


def streaming_compare(expected, actual, *, scratch_bytes, rtol=1e-5, atol=1e-7):
    """Legacy result_state/compare semantics, bounded tensor comparison scratch.

    Mapping keys and tuples use result_state's wire normalization; complex RoPE
    retains its dtype. Each mismatching tensor contributes one path, as before.
    """
    expected = _map(expected, lambda t: t, wire=True)
    actual = _map(actual, lambda t: t, wire=True)
    errors = []
    def visit(x, y, path):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            mismatch = x.shape != y.shape or x.dtype != y.dtype
            if not mismatch:
                # Two host transfer chunks plus allclose/isclose intermediates.
                factor = 8 if x.device.type == y.device.type == 'cpu' else 10
                count = scratch_bytes // max(factor * x.element_size(), 1)
                _require(count > 0, 'comparison scratch cannot hold one element')
                a, b = x.detach().reshape(-1), y.detach().reshape(-1)
                mismatch = any(not torch.allclose(a[i:i+count].cpu(), b[i:i+count].cpu(), rtol=rtol, atol=atol)
                               for i in range(0, a.numel(), count))
            if mismatch: errors.append(path)
        elif isinstance(x, dict) and isinstance(y, dict):
            if x.keys() != y.keys(): errors.append(path)
            else:
                for key in x: visit(x[key], y[key], path + '/' + key)
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for index, (a, b) in enumerate(zip(x, y)): visit(a, b, path + '/' + str(index))
        elif type(x) != type(y) or x != y:
            errors.append(path)
    visit(expected, actual, '')
    return dict(status='PASS' if not errors else 'REVIEW', mismatches=errors, rtol=rtol, atol=atol)


def _host_available():
    """Fresh physical and accessible cgroup headroom; an observation, not RSS."""
    available = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    roots = [Path('/sys/fs/cgroup'), Path('/sys/fs/cgroup/memory')]
    membership = Path('/proc/self/cgroup')
    if membership.exists():
        for line in membership.read_text().splitlines():
            _, controllers, path = line.split(':', 2)
            root = Path('/sys/fs/cgroup/memory' if 'memory' in controllers.split(',') else '/sys/fs/cgroup')
            candidate = root / path.lstrip('/')
            if candidate.is_relative_to(root):
                roots.extend([candidate, *[p for p in candidate.parents if p.is_relative_to(root)]])
    for root in set(roots):
        for limit, current in (('memory.max', 'memory.current'),
                               ('memory.limit_in_bytes', 'memory.usage_in_bytes')):
            if (root / limit).is_file():
                maximum = (root / limit).read_text().strip()
                if maximum != 'max':
                    available = min(available, max(0, int(maximum) - int((root / current).read_text())))
    return available


def _cuda_memory(device):
    """Fresh owned-device observation; never query or initialize other devices."""
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return dict(free_bytes=free, total_bytes=total,
        allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def _ownership(adapter):
    def tensor(t):
        return (id(t), _storage_key(t), tuple(t.shape), tuple(t.stride()), str(t.dtype))
    return dict(model=id(adapter.model),
        modules={name: id(module) for name, module in adapter.model.named_modules(remove_duplicate=False)},
        hooks={name: (tuple(module._forward_pre_hooks.items()), tuple(module._forward_hooks.items()))
               for name, module in adapter.model.named_modules()},
        parameters={name: tensor(p) for name, p in adapter.model.named_parameters(remove_duplicate=False)},
        buffers={name: tensor(t) for name, t in adapter.model.named_buffers(remove_duplicate=False)},
        optimizers={name: (id(opt), id(opt.state), [(id(group), tuple(id(p) for p in group['params']))
                    for group in opt.param_groups]) for name, opt in adapter.optimizers.items()},
        aliases=adapter.aliases)


class ResidentTrial:
    def __init__(self, backend, budget):
        self.backend, self.workload, self.budget = backend, backend.workload, budget
        self._entered = self._ready = self._adopt = self._dirty = False
        self._capacity_admitted = False
        self._cuda_claimed = False
        self._device = None
        self._session = self._outputs = self._expected = None
        self._publication = None
        self._backup = self._original = self._outer = self._recovery_error = None
        self.receipt = dict(schema_version=SCHEMA, closed=False, adopted=False,
            scope='CPU managed state/snapshot/result/scratch; not RSS, training activations or CUDA capacity')

    def _provider(self):
        from .owner_execution import _method_signature
        for name in _METHODS:
            method = getattr(self.owner, name, None)
            _require(inspect.ismethod(method) and method.__func__ is getattr(type(self.owner), name),
                     'explicit source-bound resident owner adapter required')
        from .torch_backend import code_signature
        return dict(owner={cls.__module__ + '.' + cls.__qualname__: code_signature(cls)
                           for cls in type(self.owner).__mro__ if cls is not object},
                    methods={name: _method_signature(self.owner, name) for name in _METHODS},
                    helper=file_sha256(Path(__file__)))

    def _adapter(self):
        adapter = self.owner._resident_trial_adapter()
        _require(type(adapter) is ResidentOwner and adapter.schema_version == SCHEMA,
                 'unknown resident owner schema')
        _require(isinstance(adapter.model, torch.nn.Module) and bool(adapter.optimizers)
                 and all(type(opt) is torch.optim.AdamW for opt in adapter.optimizers.values()),
                 'resident rollback supports explicit dense AdamW owners only')
        _require(all(type(n) is int and n >= 0 for n in
                     (adapter.growth_bytes, adapter.output_bytes, adapter.max_new_tensor_bytes,
                      adapter.host_growth_bytes, adapter.device_growth_bytes, adapter.gradient_bytes)),
                 'invalid owner growth/result bounds')
        return adapter

    def __enter__(self):
        _require(not self._entered, 'transaction is one-shot')
        self._entered = True
        try:
            return self._admit_snapshot()
        except BaseException:
            self._close()
            raise

    def _admit_snapshot(self):
        from distillschedule.training.checkpoint import rng_state
        _require(type(self.budget) in (TrialBudget, CudaTrialBudget), 'explicit typed resident trial budget required')
        self.budget.validate()
        _require(not self.backend.distributed and not torch.distributed.is_initialized(),
                 'distributed resident trials are not qualified')
        self._device = self.workload.device
        cuda = type(self.budget) is CudaTrialBudget
        _require(self._device.type == ('cuda' if cuda else 'cpu'),
                 'CUDA budget requires one CUDA resident owner; CUDA requires an explicit CudaTrialBudget')
        _require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('cuda'),
                 'AMP resident trials are not qualified')
        _require(not torch.is_inference_mode_enabled(), 'inference mode cannot create writable rollback snapshots')
        _require(type(self.workload.state) is dict and len(self.workload.state) == 1
                 and type(self.workload.feeds) is dict and not self.workload.immutable_modules,
                 'exactly one resident owner and explicit feed mapping required; no external sharing')
        self.key, self.owner = next(iter(self.workload.state.items()))
        _require(not hasattr(self.owner, '_resident_trial_poisoned'), 'resident owner is poisoned')
        _require(getattr(self.owner, '_resident_trial_active', None) is None,
                 'owner already has an active resident transaction')
        self._source = self._provider()
        adapter = self._adapter()
        _require(not cuda or adapter.cuda_capable is True, 'owner has no explicit CUDA resident capability')
        self._identity = fingerprint(adapter.identity)
        self._structure = _ownership(adapter)
        self._physical = adapter.physical
        self._retained = adapter.retained
        self._feed_object = self.workload.feeds
        self._program = self.workload.program.digest
        self._slots = [(optimizer, p, key, tensor) for optimizer in adapter.optimizers.values()
                       for p, slots in optimizer.state.items() for key, tensor in slots.items()]
        _require(all(p.dtype == torch.float32 for p in adapter.model.parameters())
                 and all(not t.is_floating_point() or t.dtype == torch.float32
                         for t in adapter.model.buffers()), 'resident modules must remain FP32')
        _require(all(t.device == self._device for t in (*adapter.model.parameters(), *adapter.model.buffers())),
                 'resident modules must remain on the single owned device')
        _require(all(p.grad is None for p in adapter.model.parameters()), 'resident boundary requires cleared gradients')
        if cuda:
            for optimizer in adapter.optimizers.values():
                _require(all(not group.get(name, False) for group in optimizer.param_groups
                             for name in ('capturable', 'differentiable', 'fused', 'amsgrad')),
                         'CUDA resident trials require native noncapturable unfused AdamW')
                for parameter, slots in optimizer.state.items():
                    _require(set(slots) == {'step', 'exp_avg', 'exp_avg_sq'}
                             and type(slots['step']) is torch.Tensor and slots['step'].device.type == 'cpu'
                             and slots['step'].dtype == torch.float32 and slots['step'].ndim == 0
                             and all(type(slots[name]) is torch.Tensor and slots[name].device == self._device
                                     and slots[name].dtype == parameter.dtype and slots[name].shape == parameter.shape
                                     for name in ('exp_avg', 'exp_avg_sq')),
                             'CUDA resident AdamW slot layout is not qualified')
        feed_tensors = []
        _map(self.workload.feeds, lambda t: feed_tensors.append(t))
        _require(all(t.device == self._device and not t.requires_grad and t.grad_fn is None
                     and (not t.is_floating_point() or t.dtype == torch.float32) for t in feed_tensors),
                 'resident feeds must be detached FP32/integer inputs')
        # Retain zero-copy destinations independently of dictionaries the native
        # call may replace. Aliased serialized leaves are deliberately counted.
        original = _map(dict(owner=adapter.state, feeds=self.workload.feeds), lambda t: t)
        _map(original, lambda t: _require(t.device.type == 'cpu' or (cuda and t.device == self._device),
                                         'state contains an unowned device'))
        inventory = _memory_inventory(original)
        snapshot_bytes = inventory['serialized_bytes']
        largest = inventory['largest_tensor_bytes']
        owner_bytes = _sizes(adapter.state)[0]
        snapshot_bound = snapshot_bytes + _RNG_ALLOWANCE
        result_bound = owner_bytes + adapter.growth_bytes + adapter.output_bytes
        scratch_bound = 8 * max(largest, adapter.max_new_tensor_bytes, adapter.output_bytes, _RNG_ALLOWANCE // 8)
        _require(snapshot_bound <= self.budget.snapshot_limit_bytes, 'initial snapshot exceeds snapshot budget')
        _require(result_bound <= self.budget.result_limit_bytes, 'forecast reference result exceeds result budget')
        _require(scratch_bound <= self.budget.scratch_bytes, 'identity/comparison scratch exceeds scratch budget')
        host_growth = self.budget.host_growth_bytes if cuda else adapter.growth_bytes
        managed = inventory['host_storage_bytes'] + host_growth + snapshot_bound + result_bound + self.budget.scratch_bytes
        available = _host_available()
        _require(managed + self.budget.reserve_bytes <= min(self.budget.host_limit_bytes, available),
                 'managed host transaction plus reserve exceeds host/cgroup headroom')
        self.receipt.update(existing_host_tensor_bytes=inventory['host_storage_bytes'],
            existing_device_tensor_bytes=inventory['device_storage_bytes'], growth_bound_bytes=adapter.growth_bytes,
            snapshot_bound_bytes=snapshot_bound, result_bound_bytes=result_bound,
            scratch_bytes=self.budget.scratch_bytes, reserve_bytes=self.budget.reserve_bytes,
            managed_host_bound_bytes=managed, available_host_bytes=available, budgets=vars(self.budget),
            provider=self._source)
        self._result_bound = result_bound
        if cuda:
            self._claim_cuda()
            self._admit_cuda(adapter, inventory)
        # Full normal boundary/descriptor validation is only after capacity
        # admission; their largest identity serialization leaf is in scratch.
        self.owner._resident_trial_active = self
        self._capacity_admitted = True
        self.owner._resident_trial_validate()
        self._outer = rng_state(device=self._device)
        _require(_sizes(self._outer)[0] <= _RNG_ALLOWANCE, 'outer RNG exceeds admitted allowance')
        original['outer_rng'] = self._outer
        self._original = original
        self._backup = _copy(original, self.budget.snapshot_limit_bytes)
        self._backup_hash = _digest(self._backup)
        self._feed_hash = _digest(original['feeds'])
        self.receipt['snapshot_tensor_bytes'] = _sizes(self._backup)[0]
        self._guard()
        if cuda:
            self._observe_cuda('snapshot')
        self._ready = True
        return self

    def _claim_cuda(self):
        # Allocator peak counters are device-global within this process. Two
        # overlapping transactions must not erase each other's measurements.
        with _CUDA_TRIAL_LOCK:
            key = str(self._device)
            _require(key not in _CUDA_TRIALS, 'CUDA device already has an active resident transaction')
            _CUDA_TRIALS[key] = self
            self._cuda_claimed = True

    def _admit_cuda(self, adapter, inventory):
        budget = self.budget
        _require(adapter.host_growth_bytes <= budget.host_growth_bytes
                 and adapter.device_growth_bytes <= budget.device_state_growth_bytes
                 and adapter.gradient_bytes <= budget.device_gradient_bytes,
                 'native Adam/EMA/cache/gradient growth exceeds explicit CUDA budgets')
        row = _cuda_memory(self._device)
        _require(inventory['device_storage_bytes'] <= row['allocated_bytes'],
                 'CUDA owner storage is not covered by the observed native allocator')
        extra = sum((budget.device_state_growth_bytes, budget.device_gradient_bytes,
                     budget.device_activation_bytes, budget.device_workspace_bytes, budget.device_restore_bytes))
        self._allocated_bound = row['allocated_bytes'] + extra
        self._reserved_bound = row['reserved_bytes'] + extra
        _require(self._reserved_bound + budget.device_reserve_bytes <= budget.device_limit_bytes
                 and extra + budget.device_reserve_bytes <= row['free_bytes'],
                 'CUDA transaction plus reserve exceeds fresh device headroom/budget')
        self.receipt.update(schema_version=CUDA_SCHEMA,
            scope='PRIVATE CUDA1 FP32; managed host state and declared device allowances; not construction/RSS/pretrained capacity',
            identity_scratch='8x largest serialized/new leaf covers unchanged native/generic full-leaf CPU hashes; resident hashes/comparison stream',
            host_growth_bound_bytes=budget.host_growth_bytes, cuda_admission=row,
            device_additional_bound_bytes=extra, device_allocated_bound_bytes=self._allocated_bound,
            device_reserved_bound_bytes=self._reserved_bound, device_reserve_bytes=budget.device_reserve_bytes,
            device_owned_storage_bytes=inventory['device_storage_bytes'], cuda_observations=[],
            peak_scope='owned-device process allocator; peak counters reset at private trial phase boundaries')
        self._phase_start = row
        torch.cuda.reset_peak_memory_stats(self._device)

    def _begin_cuda(self, phase):
        if type(self.budget) is not CudaTrialBudget: return
        row = _cuda_memory(self._device)
        # Rollback is always attempted, even after headroom or allocator errors.
        if phase != 'rollback':
            remaining = max(0, self._reserved_bound - row['reserved_bytes'])
            _require(remaining + self.budget.device_reserve_bytes <= row['free_bytes'],
                     'CUDA headroom changed before trial update')
            _require(self.budget.reserve_bytes + self.budget.scratch_bytes + self.budget.host_growth_bytes
                     + (self._result_bound if self._expected is None else 0) <= _host_available(),
                     'host/cgroup headroom changed before trial update')
        self._phase_start = row
        torch.cuda.reset_peak_memory_stats(self._device)

    def _observe_cuda(self, phase):
        if type(self.budget) is not CudaTrialBudget: return
        row = _cuda_memory(self._device)
        free_consumption = max(0, self.receipt['cuda_admission']['free_bytes'] - row['free_bytes'])
        self.receipt['cuda_observations'].append(dict(phase=phase, free_consumption_bytes=free_consumption, **row))
        budget = self.budget
        if phase == 'rollback':
            # Exact state restoration is checked before this observation. Cached
            # blocks or external headroom loss can persist after a refused trial;
            # retain those observations without misclassifying them as corruption.
            _require(row['peak_allocated_bytes'] <= self._phase_start['allocated_bytes'] + budget.device_restore_bytes,
                     'observed CUDA rollback peak exceeded restore allowance')
        else:
            _require(row['peak_allocated_bytes'] <= self._allocated_bound
                     and row['peak_reserved_bytes'] <= self._reserved_bound
                     and row['peak_reserved_bytes'] + budget.device_reserve_bytes <= budget.device_limit_bytes
                     and row['free_bytes'] >= budget.device_reserve_bytes
                     and free_consumption <= self.receipt['device_additional_bound_bytes'],
                     'observed CUDA allocator peak exceeded declared admission')
        # Count retained original destinations as well as newly created state.
        host = _memory_inventory((self._original, self._adapter().state, self.workload.feeds,
                                  self._backup, self._expected, self._outputs))['host_storage_bytes']
        self.receipt['observed_managed_host_tensor_bytes'] = max(host,
            self.receipt.get('observed_managed_host_tensor_bytes', 0))
        if phase != 'rollback':
            _require(host + budget.scratch_bytes + budget.reserve_bytes <= budget.host_limit_bytes
                     and budget.reserve_bytes <= _host_available(),
                     'observed managed host state/reserve exceeded admission')

    def _require_active_snapshot(self):
        _require(self._ready and not self.receipt['closed'] and self._backup is not None
                 and getattr(self.owner, '_resident_trial_active', None) is self,
                 'active admitted snapshot required')
        _require(not hasattr(self.owner, '_resident_trial_poisoned'), 'resident owner is poisoned')

    def _close(self):
        # No fallible RNG operation belongs in this unconditional release path.
        self._ready = False
        self._capacity_admitted = False
        if self._cuda_claimed:
            with _CUDA_TRIAL_LOCK:
                if _CUDA_TRIALS.get(str(self._device)) is self:
                    del _CUDA_TRIALS[str(self._device)]
            self._cuda_claimed = False
        self._backup = self._original = self._outer = self._expected = self._session = self._outputs = None
        self._publication = None
        self._slots = []; self._physical = (); self._retained = {}
        owner = getattr(self, 'owner', None)
        if getattr(owner, '_resident_trial_active', None) is self:
            del owner._resident_trial_active
        self.receipt['closed'] = True

    def _guard(self):
        _require(self._source == self._provider(), 'resident trial provider source changed')
        _require(getattr(self.owner, '_resident_trial_active', None) is self,
                 'resident transaction ownership changed')
        adapter = self._adapter()
        _require(fingerprint(adapter.identity) == self._identity, 'resident owner identity changed')
        _require(_ownership(adapter) == self._structure
                 and self.workload.state == {self.key: self.owner}
                 and self.workload.feeds is self._feed_object
                 and self.workload.program.digest == self._program,
                 'resident owner/parameter/optimizer/feed ownership changed')
        _require(all(opt.state.get(p, {}).get(key) is tensor for opt, p, key, tensor in self._slots),
                 'existing optimizer slot identity changed')

    def _result(self, outputs):
        return dict(state={self.key: self._adapter().state}, outputs=outputs, grads={})

    def _restore_outer_rng(self):
        from distillschedule.training.checkpoint import restore_rng
        from distillschedule.training.single_device import device_scope
        # validate_rng checks current device; preserve the caller selection.
        with device_scope(self._device):
            restore_rng(self._outer)

    def _rollback(self):
        self._adopt = False
        self._publication = None
        self._session = self._outputs = None
        try:
            self.owner._resident_trial_cleanup()
            self._begin_cuda('rollback')
            self._guard()
            _require(_digest(self._backup) == self._backup_hash, 'resident snapshot integrity changed')
            _restore_tensors(self._original, self._backup)
            self.owner._resident_trial_restore(_map(self._original['owner'], lambda t: t), self._retained)
            for module, name, present, value in self._physical:
                if present: setattr(module, name, value)
                elif name in vars(module): delattr(module, name)
            self.workload.feeds.clear()
            self.workload.feeds.update(_map(self._original['feeds'], lambda t: t))
            self.owner._resident_trial_validate()
            self._guard()
            _require(_digest(self._adapter().state) == _digest(self._backup['owner'])
                     and _digest(self.workload.feeds) == self._feed_hash, 'resident rollback was not exact')
            self._observe_cuda('rollback')
        except BaseException as state_error:
            # Even an unrestorable model must not leak native/planning RNG when
            # the independent caller stream can still be restored.
            try:
                self._restore_outer_rng()
            except BaseException as rng_error:
                raise RollbackError(state_error, rng_error) from rng_error
            raise
        self._restore_outer_rng()
        self._dirty = False

    def _recover(self, error):
        # Dead native stack frames can otherwise retain graphs after an OOM.
        if error is not None:
            traceback.clear_frames(error.__traceback__)
            if type(self.budget) is CudaTrialBudget:
                # Preserve the failed execution peak before rollback resets it.
                # A poisoned CUDA context can make measurement itself fail.
                try:
                    self.receipt['cuda_observations'].append(dict(phase='failed', **_cuda_memory(self._device)))
                except BaseException as observation_error:
                    self.receipt['failed_observation_error'] = repr(observation_error)
        try:
            self._rollback()
        except BaseException as rollback:
            traceback.clear_frames(rollback.__traceback__)
            self.owner._resident_trial_poisoned = True
            self.owner._resident_trial_poison_reason = repr(rollback)
            self._ready = False
            self._recovery_error = RollbackError(error, rollback)
            raise self._recovery_error from rollback

    def reference(self, implementation=None):
        self._require_active_snapshot()
        _require(self._expected is None and not self.receipt['closed'], 'reference already captured or trial closed')
        try:
            self._begin_cuda('reference')
            self._dirty = True
            if implementation is None: implementation = self.backend.inventory(1)['candidates'][0]
            session = self.backend.session(implementation, copy_state=False)
            outputs = session.run()
            self._observe_cuda('reference')
            self.owner._resident_trial_validate(); self._guard()
            _require(_digest(self.workload.feeds) == self._feed_hash, 'reference mutated input feeds')
            result = _map(self._result(outputs), lambda t: t, wire=True)
            self._expected = _copy(result, self._result_bound)
            self._expected_hash = _digest(self._expected)
            self.receipt['reference_result_tensor_bytes'] = _sizes(self._expected)[0]
            del outputs, session, result
            self._rollback()
        except BaseException as error:
            session = outputs = result = None
            self._recover(error)
            raise

    def candidate(self, implementation, order=None, *, measure=False, execute=None,
                  persistent=False, rtol=1e-5, atol=1e-7, report_inactive=False):
        from .checkpoint_policy import native_effect_evidence
        self._require_active_snapshot()
        _require(self._expected is not None and not self.receipt['closed'], 'capture a reference first')
        _require(type(report_inactive) is bool, 'explicit inactive-candidate reporting flag required')
        self.receipt.pop('candidate_admission', None)
        try:
            if self._dirty: self._rollback()
            self._guard()
            _require(_digest(self._backup) == self._backup_hash, 'resident snapshot integrity changed')
            _require(_digest(self._expected) == self._expected_hash, 'reference result integrity changed')
            self._begin_cuda('candidate')
            self._dirty = True
            self._session = self.backend.session(implementation, copy_state=False, persistent=persistent)
            self._outputs = (execute(self._session) if execute is not None else
                             self._session.run(order, measure=measure))
            self._observe_cuda('candidate')
            self.owner._resident_trial_validate(); self._guard()
            _require(_digest(self.workload.feeds) == self._feed_hash, 'candidate mutated input feeds')
            result = self._result(self._outputs)
            _require(_sizes(result)[0] <= self._result_bound, 'candidate result exceeds admitted bound')
            gate = streaming_compare(self._expected, result, scratch_bytes=self.budget.scratch_bytes,
                                     rtol=rtol, atol=atol)
            native = [native_effect_evidence(implementation, self._session.effects)]
            # Preserve actual rejection details even when inactive effects raise
            # before the candidate can return them to the planning caller.
            self.receipt['candidate_admission'] = dict(reference_gate=gate,
                                                       native_capability_evidence=native)
            if report_inactive and not all(row['active'] for row in native):
                self._rollback()
                return None, None, gate, native
            _require(all(row['active'] for row in native), 'selected native capability was not exercised')
            if gate['status'] != 'PASS':
                self._rollback()
                return None, None, gate, native
            self._candidate_hash = _digest(result)
            return self._session, self._outputs, gate, native
        except BaseException as error:
            result = None
            self._recover(error)
            raise

    def restore(self):
        """Restore between calibration samples before reading initial identity."""
        self._require_active_snapshot()
        if self._dirty: self._recover(None)

    def accept(self, *, publish=None):
        """Request adoption on successful context exit, *after* publication.

        Merely returning a candidate never adopts it. Any subsequent exception
        inside the context (including publication failure) still rolls back.
        """
        self._require_active_snapshot()
        _require(publish is None or callable(publish), 'publication callback must be callable')
        _require(self._session is not None and self._dirty and not self.receipt['closed'],
                 'only a numerically/effect-admitted candidate may be adopted')
        self._adopt = True
        self._publication = publish

    def __exit__(self, kind, error, tb):
        _require(self._ready or self._recovery_error is not None, 'active admitted snapshot required')
        try:
            if error is not None and error is self._recovery_error:
                pass  # Only this transaction's already-handled failure may skip recovery.
            elif self._recovery_error is not None:
                if error is None: raise self._recovery_error
                raise RollbackError(error, self._recovery_error) from self._recovery_error
            elif error is not None:
                self._recover(error)
            elif self._adopt:
                try:
                    self._guard(); self.owner._resident_trial_validate()
                    _require(_digest(self._backup) == self._backup_hash, 'resident snapshot integrity changed')
                    _require(_digest(self._expected) == self._expected_hash, 'reference result integrity changed')
                    _require(_digest(self._result(self._outputs)) == self._candidate_hash
                             and _digest(self.workload.feeds) == self._feed_hash,
                             'candidate changed after numerical/effect admission')
                    # A failed RNG restore is a failed transaction, not a
                    # successfully adopted owner with unfinished finalization.
                    self._observe_cuda('adopt')
                    self._restore_outer_rng()
                    if self._publication is not None:
                        # Last fallible boundary. A publisher receives the exact
                        # success receipt that becomes true immediately after it
                        # returns; a publication exception still rolls back.
                        self._publication(dict(self.receipt, closed=True, adopted=True,
                            publication_order='final_gates_then_publication_then_adoption'))
                except BaseException as exc:
                    self._recover(exc)
                    raise
                self.receipt['adopted'] = True
            else:
                self._recover(None)
        finally:
            self._close()
        return False
