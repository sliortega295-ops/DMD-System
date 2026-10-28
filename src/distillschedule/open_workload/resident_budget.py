"""Resident trial budget metadata, independent of the optional Torch runtime.

The runtime re-exports these same types. Inspecting a configuration does not
admit a model or replace the live capacity checks performed by its owner.
"""
from __future__ import annotations

from dataclasses import dataclass

from distillschedule.core.contracts import require


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
