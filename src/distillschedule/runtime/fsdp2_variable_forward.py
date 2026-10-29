"""Explicit FSDP2 phase for unequal *non-updating* forward prefixes.

All ranks materialize the same units, in order, before entering rank-local
control flow. Gradients still use FSDP's original reduction and engine; full
parameters are released before the original clipping/optimizer step. This is
not permission for unequal backward/commit counts or arbitrary graph changes.
The caller owns collective agreement, capacity admission and the call census.
"""
from __future__ import annotations

from contextlib import contextmanager


def _require(ok, message):
    if not ok:
        raise ValueError('FSDP2 variable-forward phase: ' + message)


def _group(module):
    # Read-only, version-pinned audit, as in fsdp2_nograd_persistence. Policy
    # changes and parameter ownership always use public FSDP APIs.
    return module._get_fsdp_state()._fsdp_param_group


def _ownership(engine):
    return tuple(tuple(id(p) for p in group['params']) for group in engine.optimizer.param_groups)


class _PhaseEngine:
    def __init__(self, engine, units, with_grad):
        self._engine, self._units, self._with_grad = engine, units, with_grad
        self._owners = _ownership(engine)
        self._live = False
        self.receipt = dict(enabled=True, with_grad=with_grad, unit_count=len(units),
            unshard_count=0, reshard_count=0, engine_forward_count=0,
            backward_count=0, commit_count=0, restored=False)

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def __call__(self, *args, **kwargs):
        _require(self._live, 'forward after phase released')
        self.receipt['engine_forward_count'] += 1
        return self._engine(*args, **kwargs)

    def backward(self, loss):
        _require(self._with_grad, 'backward in no-grad phase')
        _require(self._live and self.receipt['backward_count'] == 0,
                 'expected one backward in an active phase')
        self._engine.backward(loss)
        self.receipt['backward_count'] += 1

    def step(self):
        _require(self._with_grad, 'commit in no-grad phase')
        _require(self._live and self.receipt['backward_count'] == 1,
                 'commit requires one backward in an active phase')
        _require(_ownership(self._engine) == self._owners, 'optimizer ownership changed')
        self.release()
        # Do not implement an alternate norm, reduction, optimizer, scheduler
        # or zero_grad here. The original engine now sees sharded FP32 masters.
        self._engine.step()
        self.receipt['commit_count'] += 1

    def materialize(self):
        self._live = True  # Cleanup also covers partially completed materialization.
        for unit in self._units:
            unit.set_reshard_after_forward(False, recurse=False)
            unit.set_reshard_after_backward(False, recurse=False)
        for unit in self._units:
            unit.unshard(async_op=False)
            self.receipt['unshard_count'] += 1

    def release(self):
        if not self._live:
            return
        errors = []
        for unit in self._units:
            try:
                unit.set_reshard_after_forward(True, recurse=False)
                unit.set_reshard_after_backward(True, recurse=False)
            except Exception as error:
                errors.append(error)
        for unit in reversed(self._units):
            try:
                unit.reshard()
                self.receipt['reshard_count'] += 1
            except Exception as error:
                errors.append(error)
        self._live = False
        self.receipt['restored'] = not errors and not any(
            group.is_unsharded for unit in self._units if (group := _group(unit)) is not None)
        if errors:
            raise RuntimeError('FSDP2 phase cleanup failed') from errors[0]
        _require(self.receipt['restored'], 'full parameters remain unsharded after release')


@contextmanager
def fsdp2_variable_forward_phase(engine, *, enabled, with_grad, expected_units,
                                checkpoint_block_type=None, module_type=None):
    """Yield ``(engine_view, receipt)``; disabled leaves the engine untouched.

    The explicit enabled path requires the already-sharded, reshard=True
    baseline and one backward/commit for a grad phase. A no-grad phase may
    contain a caller-agreed group of rollout calls and never updates weights.
    Grad callers must bind the model's actual whole-checkpoint block class:
    interior Linear/LoRA sharding is not interchangeable with that boundary.
    This is a caller-owned model capability, not inferred from an equal unit
    count. The view must not escape the scope. No RNG, mode or dtype is changed.
    """
    _require(type(enabled) is bool and type(with_grad) is bool, 'explicit boolean policies required')
    if not enabled:
        yield engine, dict(enabled=False, restored=True)
        return
    _require(type(expected_units) is int and expected_units > 0, 'positive explicit unit count required')
    if module_type is None:
        from torch.distributed.fsdp import FSDPModule
        module_type = FSDPModule
    _require(isinstance(engine.module, module_type), 'requires the actual FSDP root owner')
    units = [module for module in engine.module.modules() if isinstance(module, module_type)]
    _require(len(units) == expected_units, 'FSDP unit count differs from admitted topology')
    if with_grad:
        _require(isinstance(checkpoint_block_type, type) and checkpoint_block_type is not module_type,
                 'actual whole checkpoint block class must be explicitly bound')
        blocks = [module for module in engine.module.modules()
                  if module is not engine.module and isinstance(module, checkpoint_block_type)]
        _require(blocks and {id(module) for module in units if module is not engine.module}
                 == {id(module) for module in blocks}, 'FSDP units differ from whole checkpoint blocks')
    groups = [group for unit in units if (group := _group(unit)) is not None]
    _require(groups and not any(group.is_unsharded for group in groups), 'entered unsharded or nested phase')
    _require(all(group._reshard_after_forward is True and group.reshard_after_backward is True
                 for group in groups), 'requires original reshard-after-forward/backward policy')
    phase = _PhaseEngine(engine, units, with_grad)
    completed = False
    try:
        phase.materialize()
        yield phase, phase.receipt
        completed = True
    finally:
        phase.release()
    if completed and with_grad:
        _require(phase.receipt['backward_count'] == phase.receipt['commit_count'] == 1,
                 'grad phase ended without its original backward/commit')
