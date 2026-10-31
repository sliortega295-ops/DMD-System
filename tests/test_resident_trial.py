"""Private resident transactions: real Session calls, not a second trainer."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from distillschedule.open_workload.execution import ExecutionBackend
from distillschedule.open_workload.resident_trial import (
    ResidentOwner, TrialBudget, native_state_view, restore_native_state,
)
from distillschedule.open_workload.torch_backend import Workload, Session, compare, signature
from distillschedule.training.checkpoint import rng_state, restore_rng
from distillschedule.workloads.program import Call, Program


class TrialOwner:
    """Explicit test provider; generic state_dict objects are not opt-ins."""
    def __init__(self):
        self.model = torch.nn.Linear(2, 1)
        self.model.register_buffer('frozen', torch.tensor([2., 3.]))
        self.optimizers = {'g': torch.optim.AdamW([self.model.weight], lr=.001),
                           'c': torch.optim.AdamW([self.model.bias], lr=.001)}
        self.clock = 0
        self._rng = rng_state(device='cpu')
        self.cache = self.ema = None
        self.enabled = False
        self.effects = None
        self.fault = None
        self.exercise = True
        self.version = 1

    def __deepcopy__(self, memo):
        raise AssertionError('resident owners must never be deep-copied')

    def runtime_tensors(self):
        yield from self.model.parameters()
        yield from self.model.buffers()
        for optimizer in self.optimizers.values():
            for state in optimizer.state.values():
                yield from state.values()

    def _validate(self):
        assert not getattr(self, '_resident_trial_poisoned', None)
        assert all(p.grad is None for p in self.model.parameters())
        assert self.clock % 2 == 0
        assert torch.equal(self.model.frozen, torch.tensor([2., 3.]))

    def _resident_trial_adapter(self):
        return ResidentOwner(self.model, self.optimizers,
            dict(native_state_view(self.model, self.optimizers), clock=self.clock,
                 rng=self._rng, cache=self.cache, ema=self.ema),
            aliases={'model': id(self.model)}, identity={'version': self.version},
            growth_bytes=128, output_bytes=4)

    def _resident_trial_restore(self, state, retained):
        restore_native_state(self.model, self.optimizers, state)
        self.clock, self._rng = state['clock'], state['rng']
        self.cache, self.ema = state['cache'], state['ema']

    def _resident_trial_cleanup(self):
        for p in self.model.parameters():
            p.grad = None

    def _resident_trial_validate(self):
        self._validate()

    def state_dict(self):
        self._validate()
        return self._resident_trial_adapter().state

    def load_state_dict(self, state):
        raise AssertionError('ordinary checkpoint restore is not failure rollback')

    def execution_options(self):
        return dict(schema_version='distillschedule.owner_execution.v1',
            baseline={'gradient_checkpointing': False},
            options=[{'gradient_checkpointing': False}, {'gradient_checkpointing': True}],
            effect=dict(kind='native_checkpoint', enabled_key='gradient_checkpointing', components=['native']),
            identity={'version': self.version})

    @contextmanager
    def execution_scope(self, option, effects):
        self.enabled = option['gradient_checkpointing']; self.effects = effects
        effects['components'] = {'native': dict(checkpoint_requests=0, recompute_calls=0,
                                                disabled_verified=False)}
        try:
            yield
        finally:
            effects['components']['native']['disabled_verified'] = not self.enabled
            self.enabled = False; self.effects = None

    def run(self, x):
        self._validate()
        outer = rng_state(device='cpu'); restore_rng(self._rng)
        try:
            total = None
            for role in ('g', 'c'):
                def forward(value):
                    return (self.model(value) + torch.rand(())).sin().square().sum()
                if self.enabled and self.exercise:
                    row = self.effects['components']['native']; row['checkpoint_requests'] += 1
                    count = 0
                    def counted(value):
                        nonlocal count
                        count += 1
                        if count > 1: row['recompute_calls'] += 1
                        return forward(value)
                    loss = torch.utils.checkpoint.checkpoint(counted, x, use_reentrant=False)
                else:
                    loss = forward(x)
                loss.backward(); self.optimizers[role].step()
                self.clock += 1
                self.cache = {'value': x.detach().clone()}
                if self.ema is None: self.ema = self.model.weight.detach().clone()
                self.ema.mul_(.9).add_(self.model.weight.detach(), alpha=.1)
                if self.fault is not None: self.fault(self, role, x)
                for p in self.model.parameters(): p.grad = None
                total = loss.detach()
            self._validate()
            return total
        finally:
            self._rng = rng_state(device='cpu'); restore_rng(outer)


def invoke(ctx):
    return ctx.state['owner'].run(ctx.inputs['x'])


def build(owner=None):
    return Workload(Program((Call('native', 'native', ('x',), ('owner',), ('owner',),
        grad='external_grad', replay_safe=False),), {'owner': 0}, ('x',), ('native',)),
        {'native': invoke}, {'owner': owner or TrialOwner()}, {'x': torch.tensor([[.7, -.5]])})


class UnknownSchemaOwner(TrialOwner):
    def _resident_trial_adapter(self):
        return replace(super()._resident_trial_adapter(), schema_version='unknown')


def option():
    return dict(backend='torch-callable-resident-v1', share_immutable=False, checkpoint_bindings=[],
                owner_execution={'owner': {'gradient_checkpointing': True}})


def budget(**changes):
    return TrialBudget(**dict(dict(host_limit_bytes=8_000_000, snapshot_limit_bytes=500_000,
        result_limit_bytes=500_000, scratch_bytes=200_000, reserve_bytes=100_000), **changes))


def test_real_no_copy_reference_candidate_adopts_exactly_once():
    workload = build(); owner = workload.state['owner']; backend = ExecutionBackend(workload)
    before = signature(owner); outer = signature(rng_state(device='cpu'))
    objects = [id(owner), id(owner.model), *map(id, owner.model.parameters()),
               *map(id, owner.optimizers.values())]
    with backend.resident_trial(budget()) as trial:
        trial.reference()
        assert signature(owner) == before
        session, out, gate, native = trial.candidate(option())
        assert gate['status'] == 'PASS' and all(row['active'] for row in native)
        assert session.state['owner'] is owner and owner.clock == 2
        trial.accept()
    assert owner.clock == 2 and signature(owner) != before
    assert objects == [id(owner), id(owner.model), *map(id, owner.model.parameters()),
                       *map(id, owner.optimizers.values())]
    assert signature(rng_state(device='cpu')) == outer
    assert trial.receipt['closed'] and trial.receipt['adopted']


def real_reference(workload):
    backend = ExecutionBackend(workload)
    session = backend.session(backend.inventory(1)['candidates'][0], copy_state=False)
    return session, session.run()


@pytest.mark.parametrize('warmed', [False, True])
def test_calibration_never_adopts_and_preserves_existing_moment_storage(warmed):
    workload = build(); owner = workload.state['owner']
    if warmed: real_reference(workload)
    before = signature(owner)
    slots = [value for opt in owner.optimizers.values() for row in opt.state.values() for value in row.values()]
    pointers = [value.data_ptr() for value in slots]
    with ExecutionBackend(workload).resident_trial(budget()) as trial:
        trial.reference()
        for _ in range(3):
            session, _, gate, _ = trial.candidate(option(), measure=True)
            assert session.state['owner'] is owner and gate['status'] == 'PASS'
    assert signature(owner) == before and not trial.receipt['adopted']
    assert pointers == [value.data_ptr() for opt in owner.optimizers.values()
                        for row in opt.state.values() for value in row.values()]
    if not warmed:
        assert owner.cache is owner.ema is None and all(not opt.state for opt in owner.optimizers.values())
    real_reference(workload)


@pytest.mark.parametrize('role', ['g', 'c'])
@pytest.mark.parametrize('exception', [RuntimeError, KeyboardInterrupt, torch.OutOfMemoryError])
@pytest.mark.parametrize('warmed', [False, True])
def test_partial_update_fault_restores_everything_and_remains_usable(role, exception, warmed):
    workload = build(); owner = workload.state['owner']
    if warmed: real_reference(workload)
    before = signature(owner); rng = signature(rng_state(device='cpu')); feeds = signature(workload.feeds)
    def fail(active, stage, x):
        if active.enabled and stage == role:
            active.model.frozen.add_(10)
            x.add_(10)
            active.model.training = not active.model.training
            active.model.weight.requires_grad_(False)
            raise exception('partial native commit')
    owner.fault = fail
    with pytest.raises(exception, match='partial native commit'):
        with ExecutionBackend(workload).resident_trial(budget()) as trial:
            trial.reference(); trial.candidate(option())
    assert signature(owner) == before and signature(workload.feeds) == feeds
    assert signature(rng_state(device='cpu')) == rng and trial.receipt['closed']
    owner.fault = None
    real_reference(workload)


def test_reference_failure_is_inside_transaction_and_releases_resources():
    workload = build(); owner = workload.state['owner']; before = signature(owner)
    def fail(active, stage, x):
        raise RuntimeError('reference failed')
    owner.fault = fail
    with pytest.raises(RuntimeError, match='reference failed'):
        with ExecutionBackend(workload).resident_trial(budget()) as trial:
            trial.reference()
    assert signature(owner) == before and trial._backup is None and trial.receipt['closed']
    owner.fault = None; real_reference(workload)


def test_review_returns_no_advanced_session_and_calibration_can_continue():
    workload = build(); before = signature(workload.state)
    with ExecutionBackend(workload).resident_trial(budget()) as trial:
        trial.reference()
        def changed(session):
            outputs = session.run(); outputs['native'].add_(1); return outputs
        session, outputs, gate, _ = trial.candidate(option(), execute=changed)
        assert gate['status'] == 'REVIEW' and session is outputs is None
        assert signature(workload.state) == before
        with pytest.raises(ValueError, match='admitted candidate'): trial.accept()
        assert trial.candidate(option())[2]['status'] == 'PASS'
    assert signature(workload.state) == before


def test_inactive_effects_and_publication_failure_do_not_adopt():
    workload = build(); owner = workload.state['owner']; before = signature(owner)
    owner.exercise = False
    with pytest.raises(ValueError, match='not exercised'):
        with ExecutionBackend(workload).resident_trial(budget()) as trial:
            trial.reference(); trial.candidate(option())
    assert signature(owner) == before
    owner.exercise = True
    with pytest.raises(OSError, match='publication failed'):
        with ExecutionBackend(workload).resident_trial(budget()) as trial:
            trial.reference(); trial.candidate(option()); trial.accept()
            raise OSError('publication failed')
    assert signature(owner) == before
    real_reference(workload)


@pytest.mark.parametrize('phase', ['copy', 'compare'])
def test_result_copy_and_comparison_interruptions_rollback(monkeypatch, phase):
    from distillschedule.open_workload import resident_trial as module
    workload = build(); before = signature(workload.state)
    with pytest.raises(KeyboardInterrupt, match='interrupted'):
        with ExecutionBackend(workload).resident_trial(budget()) as trial:
            if phase == 'compare': trial.reference()
            def fail(*args, **kwargs): raise KeyboardInterrupt('interrupted')
            monkeypatch.setattr(module, '_copy' if phase == 'copy' else 'streaming_compare', fail)
            if phase == 'copy': trial.reference()
            else: trial.candidate(option())
    assert signature(workload.state) == before
    real_reference(workload)


@pytest.mark.parametrize('field', ['snapshot_limit_bytes', 'result_limit_bytes', 'scratch_bytes', 'host_limit_bytes'])
def test_capacity_refusal_precedes_tensor_copy_or_reference(monkeypatch, field):
    from distillschedule.open_workload import resident_trial as module
    workload = build(); before = signature(workload.state)
    def unexpected(*args, **kwargs): raise AssertionError('copied before capacity admission')
    monkeypatch.setattr(module, '_copy', unexpected)
    with pytest.raises(ValueError, match='budget|headroom'):
        with ExecutionBackend(workload).resident_trial(budget(**{field: 1})):
            pytest.fail('admitted insufficient budget')
    assert signature(workload.state) == before


def test_fresh_host_headroom_and_alias_copies_are_counted(monkeypatch):
    from distillschedule.open_workload import resident_trial as module
    workload = build(); tensor = workload.feeds['x']; workload.feeds['x'] = tensor
    logical, _, unique = module._sizes({'a': tensor, 'b': tensor})
    assert logical == 2 * unique
    monkeypatch.setattr(module, '_host_available', lambda: 1)
    with pytest.raises(ValueError, match='headroom'):
        with ExecutionBackend(workload).resident_trial(budget()): pass


@pytest.mark.parametrize('corruption', ['snapshot', 'identity', 'parameter', 'optimizer_slot'])
def test_unrestorable_corruption_poison_preserves_both_errors(corruption):
    from distillschedule.open_workload.resident_trial import RollbackError
    workload = build(); owner = workload.state['owner']
    if corruption == 'optimizer_slot': real_reference(workload)
    with pytest.raises(RollbackError) as captured:
        with ExecutionBackend(workload).resident_trial(budget()) as trial:
            trial.reference(); trial.candidate(option())
            if corruption == 'snapshot': trial._backup['owner']['model']['weight'].add_(1)
            elif corruption == 'identity': owner.version += 1
            elif corruption == 'parameter': owner.model.weight = torch.nn.Parameter(owner.model.weight.detach().clone())
            else:
                row = next(iter(owner.optimizers['g'].state.values())); row['exp_avg'] = row['exp_avg'].clone()
            raise RuntimeError('original failure')
    assert isinstance(captured.value.original_error, RuntimeError)
    assert captured.value.rollback_error is not None and owner._resident_trial_poisoned
    with pytest.raises(ValueError, match='poisoned'):
        with ExecutionBackend(workload).resident_trial(budget()): pass


def test_post_gate_mutation_rolls_back_not_adopts():
    workload = build(); owner = workload.state['owner']; before = signature(owner)
    with pytest.raises(ValueError, match='changed after'):
        with ExecutionBackend(workload).resident_trial(budget()) as trial:
            trial.reference(); trial.candidate(option()); trial.accept()
            with torch.no_grad(): owner.model.weight.add_(1)
    assert signature(owner) == before


def test_streaming_comparison_matches_legacy_wire_semantics():
    from distillschedule.open_workload.resident_trial import streaming_compare, _map
    a = {1: (torch.tensor([1., float('inf')]), torch.tensor([1+2j], dtype=torch.complex128)),
         'metadata': [None, True, 3, 4., 'a'], 'integer': torch.arange(50)}
    b = deepcopy(a); b[1][0][0] += 1.; b['integer'][49] += 1
    expected = compare(_map(a, lambda t: t, wire=True), _map(b, lambda t: t, wire=True))
    assert streaming_compare(a, b, scratch_bytes=128) == expected
    assert streaming_compare(a, a, scratch_bytes=128)['status'] == 'PASS'


@pytest.mark.parametrize('fault', ['schema', 'opaque', 'amp', 'distributed', 'layout', 'dtype'])
def test_unsupported_scope_is_explicitly_refused(monkeypatch, fault):
    workload = build(); owner = workload.state['owner']; backend = ExecutionBackend(workload)
    if fault == 'schema':
        backend = ExecutionBackend(build(UnknownSchemaOwner()))
    elif fault == 'opaque': owner._resident_trial_adapter = None
    elif fault == 'distributed': backend.distributed = True
    elif fault == 'layout': workload.feeds['x'] = torch.ones(2, 2).t()
    elif fault == 'dtype': owner.model.double()
    scope = torch.autocast('cpu') if fault == 'amp' else __import__('contextlib').nullcontext()
    with scope, pytest.raises(ValueError):
        with backend.resident_trial(budget()): pass


def test_feed_mutation_without_exception_is_detected_and_restored():
    workload = build(); before = signature((workload.state, workload.feeds))
    with pytest.raises(ValueError, match='mutated input feeds'):
        with ExecutionBackend(workload).resident_trial(budget()) as trial:
            trial.reference()
            def mutate(session):
                outputs = session.run(); session.feeds['x'].add_(1); return outputs
            trial.candidate(option(), execute=mutate)
    assert signature((workload.state, workload.feeds)) == before
    real_reference(workload)


def test_initial_copy_failure_and_nested_trial_release_the_owner(monkeypatch):
    from distillschedule.open_workload import resident_trial as module
    workload = build(); owner = workload.state['owner']; before = signature(owner)
    copy = module._copy
    def fail(*args, **kwargs): raise MemoryError('snapshot allocation failed')
    monkeypatch.setattr(module, '_copy', fail)
    trial = ExecutionBackend(workload).resident_trial(budget())
    with pytest.raises(MemoryError, match='snapshot allocation'):
        with trial: pass
    assert trial.receipt['closed'] and not hasattr(owner, '_resident_trial_active')
    assert signature(owner) == before
    monkeypatch.setattr(module, '_copy', copy)
    with ExecutionBackend(workload).resident_trial(budget()) as trial:
        with pytest.raises(ValueError, match='already has an active'):
            with ExecutionBackend(workload).resident_trial(budget()): pass
        trial.reference(); assert trial.candidate(option())[2]['status'] == 'PASS'
    assert signature(owner) == before and not hasattr(owner, '_resident_trial_active')


def test_corrupt_reference_result_is_never_used_for_admission():
    workload = build(); before = signature(workload.state)
    with pytest.raises(ValueError, match='reference result integrity'):
        with ExecutionBackend(workload).resident_trial(budget()) as trial:
            trial.reference()
            trial._expected['state']['owner']['model']['weight'].add_(1)
            trial.candidate(option())
    assert signature(workload.state) == before


def test_review_r2_foreign_nested_rollback_error_restores_outer_owner():
    from distillschedule.open_workload.resident_trial import RollbackError
    outer_workload, inner_workload = build(), build()
    outer_owner, inner_owner = outer_workload.state['owner'], inner_workload.state['owner']
    before = signature(outer_owner); outer_rng = signature(rng_state(device='cpu'))
    with pytest.raises(RollbackError) as captured:
        with ExecutionBackend(outer_workload).resident_trial(budget()) as outer:
            outer.reference(); outer.candidate(option())
            with ExecutionBackend(inner_workload).resident_trial(budget()) as inner:
                inner.reference(); inner.candidate(option())
                inner._backup['owner']['model']['weight'].add_(1)
                raise RuntimeError('inner publication failure')
    assert str(captured.value.original_error) == 'inner publication failure'
    assert inner_owner._resident_trial_poisoned
    assert outer_owner.clock == 0 and signature(outer_owner) == before
    assert not hasattr(outer_owner, '_resident_trial_poisoned')
    assert outer.receipt['closed'] and not outer.receipt['adopted']
    assert inner.receipt['closed'] and signature(rng_state(device='cpu')) == outer_rng
    real_reference(outer_workload)


@pytest.mark.parametrize('original_error', [False, True])
@pytest.mark.parametrize('failure', [KeyboardInterrupt, RuntimeError])
def test_review_r2_empty_rollback_error_still_poisons(monkeypatch, original_error, failure):
    from distillschedule.open_workload import resident_trial as module
    workload = build(); owner = workload.state['owner']
    original = RuntimeError('publication failed') if original_error else None
    calls = []
    def fail(*args):
        calls.append(1); raise failure()
    with monkeypatch.context() as patch:
        with pytest.raises(module.RollbackError) as captured:
            with ExecutionBackend(workload).resident_trial(budget()) as trial:
                trial.reference(); trial.candidate(option())
                patch.setattr(module, '_restore_tensors', fail)
                if original is not None: raise original
    assert captured.value.original_error is original
    assert type(captured.value.rollback_error) is failure and calls == [1]
    assert owner._resident_trial_poisoned is True
    assert trial.receipt['closed'] and not trial.receipt['adopted']
    assert trial._backup is None and not hasattr(owner, '_resident_trial_active')
    with pytest.raises(ValueError, match='poisoned'):
        with ExecutionBackend(workload).resident_trial(budget()): pass


@pytest.mark.parametrize('state', ['never_entered', 'budget_rejected', 'adapter_rejected', 'copy_failed', 'closed'])
@pytest.mark.parametrize('action', ['reference', 'candidate', 'accept'])
def test_review_r2_actions_require_successful_active_snapshot(monkeypatch, state, action):
    from distillschedule.open_workload import resident_trial as module
    workload = build(); owner = workload.state['owner']; before = signature(owner)
    selected_budget = budget(host_limit_bytes=1) if state == 'budget_rejected' else budget()
    trial = ExecutionBackend(workload).resident_trial(selected_budget)
    with monkeypatch.context() as patch:
        if state == 'adapter_rejected':
            def fail_adapter(): raise ValueError('adapter refused')
            patch.setattr(trial, '_adapter', fail_adapter)
        if state == 'copy_failed':
            def fail_copy(*args): raise MemoryError('snapshot failed')
            patch.setattr(module, '_copy', fail_copy)
        if state in ('budget_rejected', 'adapter_rejected', 'copy_failed'):
            with pytest.raises((ValueError, MemoryError)): trial.__enter__()
        elif state == 'closed':
            with trial: pass
    with pytest.raises(ValueError, match='active.*snapshot'):
        if action == 'candidate': trial.candidate(option())
        else: getattr(trial, action)()
    assert owner.clock == 0 and signature(owner) == before
    assert not hasattr(owner, '_resident_trial_poisoned')
    assert not hasattr(owner, '_resident_trial_active')


@pytest.mark.parametrize('persistent_failure', [False, True])
def test_review_r2_rng_finalization_precedes_adoption_and_always_closes(monkeypatch, persistent_failure):
    from distillschedule.open_workload.resident_trial import RollbackError
    from distillschedule.training import checkpoint
    workload = build(); owner = workload.state['owner']; before = signature(owner)
    outer_rng = signature(rng_state(device='cpu')); restore = checkpoint.restore_rng
    calls = []
    def fail_rng(value):
        calls.append(trial.receipt['adopted'])
        if persistent_failure or len(calls) == 1: raise MemoryError('RNG finalization failed')
        return restore(value)
    with monkeypatch.context() as patch:
        with pytest.raises(RollbackError if persistent_failure else MemoryError) as captured:
            with ExecutionBackend(workload).resident_trial(budget()) as trial:
                trial.reference(); trial.candidate(option()); trial.accept()
                # Model already passed admission; finalization must still be fallible.
                patch.setattr(checkpoint, 'restore_rng', fail_rng)
    assert calls and not any(calls), 'no adoption may precede fallible RNG finalization'
    assert trial.receipt['closed'] and not trial.receipt['adopted']
    assert trial._backup is trial._expected is trial._original is None
    assert not hasattr(owner, '_resident_trial_active')
    if persistent_failure:
        assert owner._resident_trial_poisoned is True
        assert isinstance(captured.value.original_error, MemoryError)
        assert isinstance(captured.value.rollback_error, MemoryError)
    else:
        assert signature(owner) == before and not hasattr(owner, '_resident_trial_poisoned')
        assert signature(rng_state(device='cpu')) == outer_rng
        real_reference(workload)


def test_review_r2_publication_and_rng_failures_preserve_both_errors(monkeypatch):
    from distillschedule.open_workload.resident_trial import RollbackError
    from distillschedule.training import checkpoint
    workload = build(); owner = workload.state['owner']; original = OSError('publication failed')
    def fail_rng(value): raise MemoryError()
    with monkeypatch.context() as patch:
        with pytest.raises(RollbackError) as captured:
            with ExecutionBackend(workload).resident_trial(budget()) as trial:
                trial.reference(); trial.candidate(option()); trial.accept()
                patch.setattr(checkpoint, 'restore_rng', fail_rng)
                raise original
    assert captured.value.original_error is original
    assert type(captured.value.rollback_error) is MemoryError
    assert owner._resident_trial_poisoned is True
    assert trial.receipt['closed'] and not trial.receipt['adopted']
    assert trial._backup is None and not hasattr(owner, '_resident_trial_active')


@pytest.mark.parametrize('caught_inside', [False, True])
def test_review_r2_own_failed_recovery_is_not_retried_or_hidden(monkeypatch, caught_inside):
    import weakref
    from distillschedule.open_workload import resident_trial as module
    workload = build(); calls = []; handled = None
    def fail_compare(*args, **kwargs): raise RuntimeError('comparison failed')
    def fail_restore(*args):
        calls.append(1); raise KeyboardInterrupt()
    with monkeypatch.context() as patch:
        with pytest.raises(module.RollbackError) as captured:
            with ExecutionBackend(workload).resident_trial(budget()) as trial:
                trial.reference()
                snapshot = weakref.ref(trial._backup['owner']['model']['weight'])
                patch.setattr(module, 'streaming_compare', fail_compare)
                patch.setattr(module, '_restore_tensors', fail_restore)
                if caught_inside:
                    try: trial.candidate(option())
                    except module.RollbackError as error: handled = error
                else:
                    trial.candidate(option())
    assert calls == [1] and str(captured.value.original_error) == 'comparison failed'
    if caught_inside: assert captured.value is handled
    assert snapshot() is None, 'failure traceback must not retain the released snapshot'
    assert workload.state['owner']._resident_trial_poisoned is True
    assert trial.receipt['closed'] and not trial.receipt['adopted']
    assert not hasattr(workload.state['owner'], '_resident_trial_active')


def cuda_budget(**changes):
    from distillschedule.open_workload.resident_trial import CudaTrialBudget
    return CudaTrialBudget(**dict(dict(vars(budget()), device_limit_bytes=64_000_000,
        host_growth_bytes=1_000_000, device_state_growth_bytes=1_000_000,
        device_gradient_bytes=1_000_000, device_activation_bytes=1_000_000,
        device_workspace_bytes=1_000_000, device_restore_bytes=1_000_000,
        device_reserve_bytes=1_000_000), **changes))


def test_cuda_budget_explicit_positive_allowances_and_cpu_budget_unchanged():
    from distillschedule.open_workload.resident_trial import CudaTrialBudget
    assert type(cuda_budget()) is CudaTrialBudget
    cuda_budget().validate(); budget().validate()
    for name in vars(cuda_budget()):
        for invalid in (0, -1, True, 1.0):
            with pytest.raises(ValueError, match='positive'):
                cuda_budget(**{name: invalid}).validate()


def test_device_aware_inventory_distinguishes_serialized_leaves_and_live_aliases():
    from distillschedule.open_workload.resident_trial import _memory_inventory, _storage_key
    value = torch.arange(24, dtype=torch.float32)
    data = {'full': value, 'alias': value, 'slice': value[4:8]}
    row = _memory_inventory(data)
    assert _storage_key(value) == ('cpu', value.untyped_storage().data_ptr(), 96)
    assert row == dict(serialized_bytes=208, largest_tensor_bytes=96,
                      host_storage_bytes=96, device_storage_bytes=0)


def test_cuda_budget_requires_actual_cuda_owner_before_copy(monkeypatch):
    from distillschedule.open_workload import resident_trial as module
    workload = build(); before = signature(workload.state)
    def forbidden(*args, **kwargs): raise AssertionError('copied before device admission')
    monkeypatch.setattr(module, '_copy', forbidden)
    with pytest.raises(ValueError, match='requires one CUDA'):
        with ExecutionBackend(workload).resident_trial(cuda_budget()): pass
    assert signature(workload.state) == before


@pytest.mark.parametrize('inference_context', [False, True])
def test_inference_storage_rejected_before_snapshot(monkeypatch, inference_context):
    from distillschedule.open_workload import resident_trial as module
    workload = build()
    if not inference_context:
        with torch.inference_mode(): workload.feeds['x'] = torch.ones(1, 2)
    def forbidden(*args, **kwargs): raise AssertionError('copied before writable storage admission')
    monkeypatch.setattr(module, '_copy', forbidden)
    scope = torch.inference_mode() if inference_context else __import__('contextlib').nullcontext()
    with scope, pytest.raises(ValueError, match='inference'):
        with ExecutionBackend(workload).resident_trial(budget()): pass


def test_private_cuda_capability_cannot_be_forged_by_marker():
    from distillschedule.open_workload.resident_trial import cuda_trial_active
    from types import SimpleNamespace
    owner = SimpleNamespace(device=torch.device('cuda:0'))
    assert not cuda_trial_active(owner)
    for marker in (True, SimpleNamespace(_ready=True, _capacity_admitted=True, owner=owner), object()):
        owner._resident_trial_active = marker
        assert not cuda_trial_active(owner)


@pytest.mark.parametrize('fault', ['device_limit', 'free', 'host_growth', 'device_growth', 'gradient'])
def test_cuda_capacity_arithmetic_cpu_only_rejects_before_peak_reset(monkeypatch, fault):
    """No CUDA execution evidence: exercise declared-budget arithmetic only."""
    from distillschedule.open_workload import resident_trial as module
    workload = build(); trial = ExecutionBackend(workload).resident_trial(cuda_budget())
    adapter = replace(workload.state['owner']._resident_trial_adapter(), cuda_capable=True,
                      host_growth_bytes=200, device_growth_bytes=400, gradient_bytes=200)
    row = dict(free_bytes=32_000_000, total_bytes=64_000_000, allocated_bytes=1_000_000,
               reserved_bytes=2_000_000, peak_allocated_bytes=1_000_000, peak_reserved_bytes=2_000_000)
    if fault == 'free': row['free_bytes'] = 1
    else:
        field = {'device_limit': 'device_limit_bytes', 'host_growth': 'host_growth_bytes',
                 'device_growth': 'device_state_growth_bytes', 'gradient': 'device_gradient_bytes'}[fault]
        trial.budget = cuda_budget(**{field: 1})
    monkeypatch.setattr(module, '_cuda_memory', lambda device: row)
    def forbidden(*args): raise AssertionError('reset before capacity admission')
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', forbidden)
    with pytest.raises(ValueError, match='headroom|budgets'):
        trial._admit_cuda(adapter, dict(device_storage_bytes=1000))
    assert not trial._capacity_admitted


def test_streamed_digest_preserves_cpu_complex_and_integer_bytes_with_tiny_scratch():
    import hashlib
    from distillschedule.open_workload.resident_trial import _digest
    from distillschedule.workloads.best_known.registry import fingerprint
    for tensor in (torch.arange(50), torch.arange(21).to(torch.complex128) * (1 + 2j)):
        expected = fingerprint(dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
            sha256=hashlib.sha256(memoryview(tensor.reshape(-1).view(torch.uint8).numpy())).hexdigest()))
        assert _digest(tensor, scratch_bytes=7) == expected


def test_cuda_peak_counter_exclusivity_cpu_guard_only():
    """Registry test without CUDA API calls, allocations or execution claims."""
    first = ExecutionBackend(build()).resident_trial(cuda_budget())
    second = ExecutionBackend(build()).resident_trial(cuda_budget())
    first._device = second._device = torch.device('cuda:0')
    try:
        first._claim_cuda()
        with pytest.raises(ValueError, match='already has an active'): second._claim_cuda()
        second._close()  # A refused transaction must not release another's claim.
        with pytest.raises(ValueError, match='already has an active'): second._claim_cuda()
        first._close()
        second._claim_cuda()
    finally:
        first._close(); second._close()


def synthetic_cuda_accounting_on_cpu_trial(trial, monkeypatch):
    """CPU-only recovery integration, NOT CUDA placement or execution proof.

    Start from the real CPU owner's admitted snapshot, then admit the CUDA
    accounting layer against synthetic observations. No tensor changes device
    and no CUDA API executes. The real Session, rollback and claim machinery
    remain in use; only device observations/peak resets and host availability
    are injected.
    """
    from distillschedule.open_workload import resident_trial as module
    row = dict(free_bytes=32_000_000, total_bytes=64_000_000,
               allocated_bytes=1_000_000, reserved_bytes=2_000_000,
               peak_allocated_bytes=1_000_000, peak_reserved_bytes=2_000_000)
    observed = dict(row=row, host_available=32_000_000, restore_overrun=False, reset_count=0)

    def reset(device):
        assert device.type == 'cpu', 'synthetic test must not access a CUDA device'
        observed['reset_count'] += 1
        row['peak_reserved_bytes'] = row['reserved_bytes']
        row['peak_allocated_bytes'] = row['allocated_bytes'] + (
            trial.budget.device_restore_bytes + 1 if observed['restore_overrun'] else 0)

    monkeypatch.setattr(module, '_cuda_memory', lambda device: dict(row))
    monkeypatch.setattr(module, '_host_available', lambda: observed['host_available'])
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', reset)
    trial.budget = cuda_budget()
    trial._claim_cuda()
    trial._admit_cuda(trial._adapter(), module._memory_inventory(trial._original))
    trial._observe_cuda('snapshot')
    assert trial._ready and trial._capacity_admitted
    assert module._CUDA_TRIALS[str(trial._device)] is trial
    return observed


@pytest.mark.parametrize('warmed', [False, True])
@pytest.mark.parametrize('phase', ['reference', 'candidate'])
@pytest.mark.parametrize('breach', ['allocated_peak', 'cached_reservation', 'device_headroom',
                                   'device_free_consumption', 'host_growth', 'host_headroom'])
def test_fix_r1_cpu_synthetic_cuda_execution_breach_restores_without_poison(monkeypatch, warmed, phase, breach):
    """Admitted synthetic observations, real CPU state/Adam/feed/RNG rollback."""
    from distillschedule.open_workload import resident_trial as module
    workload = build(); owner = workload.state['owner']
    if warmed: real_reference(workload)
    before = signature(owner); feeds = signature(workload.feeds); outer = signature(rng_state(device='cpu'))
    parameters = [(id(p), p.data_ptr()) for p in owner.model.parameters()]
    slots = [(id(t), t.data_ptr()) for opt in owner.optimizers.values() for row in opt.state.values() for t in row.values()]
    with pytest.raises(ValueError, match='observed .*exceeded'):
        with ExecutionBackend(workload).resident_trial(budget()) as trial:
            observed = synthetic_cuda_accounting_on_cpu_trial(trial, monkeypatch)
            if phase == 'candidate': trial.reference()

            def exceed(active, role, x):
                if role != 'c': return
                # These state mutations must be undone even when the failure
                # is the capacity observer, not a native execution exception.
                x.add_(1); active.model.training = not active.model.training
                torch.rand(3)
                row = observed['row']
                if breach == 'allocated_peak': row['peak_allocated_bytes'] = trial._allocated_bound + 1
                elif breach == 'cached_reservation':
                    row['reserved_bytes'] = row['peak_reserved_bytes'] = trial._reserved_bound + 1
                elif breach == 'device_headroom': row['free_bytes'] = trial.budget.device_reserve_bytes - 1
                elif breach == 'device_free_consumption':
                    row['free_bytes'] -= trial.receipt['device_additional_bound_bytes'] + 1
                elif breach == 'host_growth':
                    active.cache = {'value': torch.zeros(trial.budget.host_limit_bytes // 4 + 1)}
                else: observed['host_available'] = trial.budget.reserve_bytes - 1

            owner.fault = exceed
            if phase == 'reference': trial.reference()
            else: trial.candidate(option())
    owner.fault = None
    assert signature(owner) == before and signature(workload.feeds) == feeds
    assert signature(rng_state(device='cpu')) == outer
    assert parameters == [(id(p), p.data_ptr()) for p in owner.model.parameters()]
    assert slots == [(id(t), t.data_ptr()) for opt in owner.optimizers.values() for row in opt.state.values() for t in row.values()]
    assert not hasattr(owner, '_resident_trial_poisoned')
    assert not hasattr(owner, '_resident_trial_active') and not trial._cuda_claimed
    assert str(trial._device) not in module._CUDA_TRIALS
    assert trial.receipt['closed'] and not trial.receipt['adopted']
    rows = trial.receipt['cuda_observations']
    assert any(row['phase'] == phase for row in rows)
    assert any(row['phase'] == 'failed' for row in rows)
    assert rows[-1]['phase'] == 'rollback'
    if breach == 'cached_reservation':
        assert rows[-1]['reserved_bytes'] > trial._reserved_bound, 'cached blocks must remain observed, not released/hidden'
    if breach == 'device_headroom': assert rows[-1]['free_bytes'] < trial.budget.device_reserve_bytes
    real_reference(workload)  # Exact restoration must leave the owner usable.


@pytest.mark.parametrize('warmed', [False, True])
def test_fix_r1_cpu_synthetic_cuda_incremental_restore_overrun_still_poisons(monkeypatch, warmed):
    """No GPU execution: only a rollback's additional allocated peak is injected."""
    from distillschedule.open_workload import resident_trial as module
    workload = build(); owner = workload.state['owner']
    if warmed: real_reference(workload)
    before = module._digest(owner._resident_trial_adapter().state)
    feeds = signature(workload.feeds); outer = signature(rng_state(device='cpu'))
    original = OSError('publication failed')
    with pytest.raises(module.RollbackError) as captured:
        with ExecutionBackend(workload).resident_trial(budget()) as trial:
            observed = synthetic_cuda_accounting_on_cpu_trial(trial, monkeypatch)
            trial.reference(); trial.candidate(option()); trial.accept()
            # The next peak reset belongs to rollback. This increase remains
            # below the absolute execution bound but exceeds restore allowance.
            observed['restore_overrun'] = True
            raise original
    assert captured.value.original_error is original
    assert isinstance(captured.value.rollback_error, ValueError)
    assert 'rollback peak exceeded restore allowance' in str(captured.value.rollback_error)
    assert module._digest(owner._resident_trial_adapter().state) == before
    assert signature(workload.feeds) == feeds and signature(rng_state(device='cpu')) == outer
    assert owner._resident_trial_poisoned and trial.receipt['closed'] and not trial.receipt['adopted']
    assert not hasattr(owner, '_resident_trial_active') and str(trial._device) not in module._CUDA_TRIALS
    rollback = trial.receipt['cuda_observations'][-1]
    assert rollback['phase'] == 'rollback' and rollback['peak_allocated_bytes'] < trial._allocated_bound
    with pytest.raises(ValueError, match='poisoned'):
        with ExecutionBackend(workload).resident_trial(budget()): pass
