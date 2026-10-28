"""Explicit construction and lifetime admission for single native owners.

Factories own source/architecture qualification. The common lifecycle owns the
capability lifetime; declaring a budget alone does not grant a model capability.
Host accounting covers managed tensors/staging, not total Python process RSS.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from distillschedule.workloads.best_known.registry import fingerprint, file_sha256

SCHEMA = 'distillschedule.training_resident.v1'
_ADMISSIONS = WeakKeyDictionary()


@dataclass(frozen=True)
class ResidentTrainingBudget:
    construction_limit_bytes: int
    host_staging_limit_bytes: int
    trial: object

    @classmethod
    def parse(cls, value, *, device):
        from distillschedule.open_workload.resident_trial import TrialBudget, CudaTrialBudget
        from dataclasses import fields
        if (type(value) is not dict or set(value) != {'schema_version', 'construction_limit_bytes',
                'host_staging_limit_bytes', 'trial'} or value['schema_version'] != SCHEMA):
            raise ValueError('resident budget needs the complete training_resident.v1 schema')
        for name in ('construction_limit_bytes', 'host_staging_limit_bytes'):
            if type(value[name]) is not int or value[name] <= 0:
                raise ValueError('resident construction/staging budgets must be positive integers')
        kind = CudaTrialBudget if str(device).split(':')[0] == 'cuda' else TrialBudget
        if str(device).split(':')[0] not in ('cpu', 'cuda'):
            raise ValueError('resident budget supports CPU or one CUDA device')
        trial = value['trial']
        if type(trial) is not dict or set(trial) != {field.name for field in fields(kind)}:
            raise ValueError('resident trial budget fields do not match device')
        if any(type(val) is not int or val <= 0 for val in trial.values()):
            raise ValueError('resident trial budgets must be positive integers')
        result = cls(value['construction_limit_bytes'], value['host_staging_limit_bytes'], kind(**trial))
        result.trial.validate()
        return result

    def to_dict(self):
        return dict(schema_version=SCHEMA, construction_limit_bytes=self.construction_limit_bytes,
                    host_staging_limit_bytes=self.host_staging_limit_bytes, trial=vars(self.trial).copy())


def active(owner):
    """An owner-bound, live constructor admission, not a mutable boolean flag."""
    admission = _ADMISSIONS.get(owner)
    return (type(admission) is ResidentAdmission and not admission.closed
            and admission.owner is owner and not hasattr(owner, '_resident_trial_poisoned'))


def for_workload(workload, budget):
    if len(workload.state) != 1:
        raise ValueError('resident lifecycle requires exactly one owned native state')
    owner = next(iter(workload.state.values()))
    admission = _ADMISSIONS.get(owner)
    if not active(owner) or admission.budget != budget:
        raise ValueError('resident construction admission missing, expired or budget changed')
    admission.validate()
    return admission


class ResidentAdmission:
    """Created by a qualified factory before loading weights, closed by caller."""
    def __init__(self, *, budget, device, resident_bytes, staging_bytes, identity, files):
        import torch
        from distillschedule.open_workload.resident_trial import _host_available, _cuda_memory
        self.budget, self.device = budget, torch.device(device)
        self.owner, self.closed = None, False
        self.files = dict(files)
        if (torch.distributed.is_initialized() or torch.is_inference_mode_enabled()
                or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda')
                or torch.get_default_dtype() != torch.float32):
            raise ValueError('resident lifecycle requires one process, FP32 and no inference/autocast')
        import os
        if any(int(os.environ.get(key, val)) != val for key, val in
               (('WORLD_SIZE', 1), ('LOCAL_WORLD_SIZE', 1), ('RANK', 0), ('LOCAL_RANK', 0))):
            raise ValueError('resident lifecycle requires one process/rank')
        if not (0 < resident_bytes <= budget.construction_limit_bytes
                and 0 < staging_bytes <= budget.host_staging_limit_bytes):
            raise ValueError('resident construction or host staging forecast exceeds explicit budget')
        if budget.host_staging_limit_bytes + budget.trial.reserve_bytes > min(
                _host_available(), budget.trial.host_limit_bytes):
            raise ValueError('resident construction host staging exceeds host/cgroup headroom')
        self.receipt = dict(schema_version=SCHEMA, identity=identity, budgets=budget.to_dict(),
            forecast_resident_bytes=resident_bytes, forecast_host_staging_bytes=staging_bytes,
            scope='single-device FP32 native tiny profiles; managed tensor budgets, not RSS',
            closed=False)
        self.initial_cuda = None
        if self.device.type == 'cuda':
            self.initial_cuda = _cuda_memory(self.device)
            row, trial = self.initial_cuda, budget.trial
            if (row['reserved_bytes'] + budget.construction_limit_bytes + trial.device_reserve_bytes
                    > trial.device_limit_bytes or budget.construction_limit_bytes
                    + trial.device_reserve_bytes > row['free_bytes']):
                raise ValueError('resident constructor exceeds fresh CUDA capacity/reserve')
            torch.cuda.reset_peak_memory_stats(self.device)
            self.receipt['cuda_before_construction'] = row

    def validate(self):
        if self.closed:
            raise ValueError('resident construction admission is closed')
        if any(file_sha256(Path(path)) != sha for path, sha in self.files.items()):
            raise ValueError('resident admitted source/assets changed')
        if self.owner is not None:
            if not active(self.owner) or fingerprint(self.owner.identity) != self.owner_identity:
                raise ValueError('resident owner identity or lifetime changed')

    def attach(self, components):
        from distillschedule.open_workload.resident_trial import _memory_inventory, _cuda_memory
        self.validate()
        if self.owner is not None or len(components.state) != 1:
            raise ValueError('resident construction admission can bind exactly one owner once')
        owner = next(iter(components.state.values()))
        if not callable(getattr(owner, '_resident_trial_adapter', None)) or owner.device != self.device:
            raise ValueError('factory returned an unsupported resident owner/device')
        adapter = owner._resident_trial_adapter()
        row = _memory_inventory(adapter.state)
        if row['host_storage_bytes'] + row['device_storage_bytes'] > self.budget.construction_limit_bytes:
            raise ValueError('observed native construction storage exceeds budget')
        if owner in _ADMISSIONS:
            raise ValueError('native owner already has a construction admission')
        self.owner, self.owner_identity = owner, fingerprint(owner.identity)
        _ADMISSIONS[owner] = self
        try:
            if len(owner.execution_options()['options']) < 2:
                raise ValueError('resident native checkpoint capability is not qualified for this source/profile')
            if self.initial_cuda is not None:
                observed = _cuda_memory(self.device)
                if (observed['peak_allocated_bytes'] > self.initial_cuda['allocated_bytes'] + self.budget.construction_limit_bytes
                        or observed['peak_reserved_bytes'] > self.initial_cuda['reserved_bytes'] + self.budget.construction_limit_bytes
                        or observed['free_bytes'] < self.budget.trial.device_reserve_bytes):
                    raise ValueError('observed CUDA constructor peak exceeds budget/reserve')
                self.receipt['cuda_after_construction'] = observed
            self.receipt['observed_managed_storage'] = row
            components.construction_receipt = self.receipt
        except BaseException:
            self.close()
            raise

    @contextmanager
    def continuation(self):
        """Recheck fresh headroom; no replay, sample draw or update on admission."""
        from distillschedule.open_workload.resident_trial import _host_available, _cuda_memory, _memory_inventory
        import torch
        self.validate()
        self.owner._resident_trial_validate()
        trial = self.budget.trial
        adapter = self.owner._resident_trial_adapter()
        inventory = _memory_inventory(adapter.state)
        growth = trial.host_growth_bytes if self.device.type == 'cuda' else adapter.growth_bytes
        host_bound = inventory['host_storage_bytes'] + growth + trial.scratch_bytes
        if host_bound + trial.reserve_bytes > min(_host_available(), trial.host_limit_bytes):
            raise ValueError('resident continuation host state/growth exceeds budget or reserve')
        if self.device.type == 'cuda':
            if (adapter.host_growth_bytes > trial.host_growth_bytes
                    or adapter.device_growth_bytes > trial.device_state_growth_bytes
                    or adapter.gradient_bytes > trial.device_gradient_bytes):
                raise ValueError('native continuation Adam/EMA/cache/gradient growth exceeds CUDA budgets')
            before = _cuda_memory(self.device)
            extra = sum(getattr(trial, key) for key in ('device_state_growth_bytes', 'device_gradient_bytes',
                'device_activation_bytes', 'device_workspace_bytes', 'device_restore_bytes'))
            if (before['reserved_bytes'] + extra + trial.device_reserve_bytes > trial.device_limit_bytes
                    or extra + trial.device_reserve_bytes > before['free_bytes']):
                raise ValueError('resident continuation exceeds fresh CUDA budget/reserve')
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            yield
            self.validate()
            managed = _memory_inventory(self.owner._resident_trial_adapter().state)
            if managed['host_storage_bytes'] > host_bound:
                raise ValueError('observed resident continuation exceeded managed host budget')
            self.receipt['last_managed_storage'] = managed
            if self.device.type == 'cuda':
                after = _cuda_memory(self.device)
                if (after['peak_reserved_bytes'] + trial.device_reserve_bytes > trial.device_limit_bytes
                        or after['free_bytes'] < trial.device_reserve_bytes
                        or after['peak_allocated_bytes'] > before['allocated_bytes'] + extra):
                    raise ValueError('observed resident continuation exceeded budget')
                self.receipt['last_continuation'] = after
        except BaseException:
            # Ordinary continuation is not a speculative paired trial. No next
            # checkpoint is published, and an interrupted owner cannot be reused.
            self.owner._resident_trial_poisoned = True
            raise

    def close(self):
        if self.owner is not None and _ADMISSIONS.get(self.owner) is self:
            del _ADMISSIONS[self.owner]
        self.closed = True
        self.receipt['closed'] = True
        self.owner = None
