"""Opt-in actual native CUDA resident trials; NOT_RUN is not acceptance.

Pinned DMD2 SD1.5/SDXL and official Self-Forcing random tiny native classes only.
The independent source oracle, uninterrupted reference and transaction each own
one CUDA model, serially; all cross-run evidence is CPU. No pretrained capacity,
AMP/distributed, speedup, public auto/plan or long-training claim is made here.
Run only under the root-owned GPU schedule and existing Torch2.10 lane.
"""
from copy import deepcopy
import gc
import json
import os
from pathlib import Path

import pytest
import torch
from torch.utils._pytree import tree_leaves

from distillschedule.open_workload.execution import ExecutionBackend
from distillschedule.open_workload.resident_trial import (
    CudaTrialBudget, RollbackError, _copy, _digest, _memory_inventory, streaming_compare,
)
from distillschedule.open_workload.torch_backend import Session
from distillschedule.training.checkpoint import cpu_tree, rng_state, restore_rng
from distillschedule.training.config import load_training_config
from distillschedule.training.session import construct, build_cycle
from distillschedule.training.single_device import device_scope
from test_training_dmd2_cpu import _oracle as dmd2_oracle
from test_training_self_forcing_cpu import _native_trainer_oracle as sf_oracle
from test_training_end_to_end import equal
from cuda_test_policy import cuda_bitwise_policy

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(),
    reason='actual CUDA resident native trials NOT_RUN: CUDA-capable Torch/device required'),
    pytest.mark.usefixtures('cuda_bitwise_policy')]

BASELINE = dict(backend='torch-callable-resident-v1', share_immutable=False, checkpoint_bindings=[])
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(params=('sd15', 'sdxl', 'self_forcing'))
def native_case(tmp_path, request):
    from distillschedule.training.plugins import load_callable
    family = request.param
    env = 'SELF_FORCING_SOURCE_ROOT' if family == 'self_forcing' else 'DMD2_SOURCE_ROOT'
    root = os.environ.get(env)
    if not root: pytest.skip('optional pinned source NOT_RUN: set ' + env)
    root = Path(root).resolve(strict=True)
    recipe = 'self_forcing_official' if family == 'self_forcing' else 'dmd2_sd15'
    builder, _ = load_callable(str(REPO / 'recipes' / recipe / 'create_cpu_fixture.py') + ':build_fixture')
    kwargs = {} if family == 'self_forcing' else dict(variant=family, ratio=2)
    files = builder(tmp_path / family, root, cycles=3, **kwargs)
    path = Path(files['config']); value = json.loads(path.read_text())
    value['trainer']['device'] = 'cuda:' + str(torch.cuda.device_count() - 1)
    if family == 'self_forcing':
        # G is immutable ON; only the existing qualified Fake switch is varied.
        value['model']['parameters']['settings']['gradient_checkpointing'] = True
    path.write_text(json.dumps(value))
    return family, load_training_config(path)


def budget(**changes):
    return CudaTrialBudget(**dict(dict(host_limit_bytes=2**31,
        snapshot_limit_bytes=2**29, result_limit_bytes=2**29, scratch_bytes=2**27,
        reserve_bytes=2**27, host_growth_bytes=2**27, device_limit_bytes=2**32,
        device_state_growth_bytes=2**27, device_gradient_bytes=2**26,
        device_activation_bytes=2**29, device_workspace_bytes=2**27,
        device_restore_bytes=2**26, device_reserve_bytes=2**27), **changes))


def option(family, enabled=None):
    key = 'fake_gradient_checkpointing' if family == 'self_forcing' else 'gradient_checkpointing'
    if enabled is None: enabled = family != 'self_forcing'
    return dict(BASELINE, owner_execution={'native_cycle': {key: enabled}})


def pointers(owner):
    adapter = owner._resident_trial_adapter()
    ema = getattr(owner, 'generator_ema', None)
    return dict(model=id(owner.model), parameters=[(id(p), p.data_ptr()) for p in owner.model.parameters()],
        optimizers=[id(opt) for opt in owner.optimizers.values()],
        slots=[(id(t), t.data_ptr()) for opt in owner.optimizers.values() for row in opt.state.values() for t in row.values()],
        native_aliases=adapter.aliases,
        ema=None if ema is None else (id(ema), [(id(t), t.data_ptr()) for t in ema.shadow.values()]))


def physical(owner):
    return [(id(module), name, name in vars(module), vars(module).get(name))
            for module in owner.model.modules() for name in
            ('gradient_checkpointing', '_gradient_checkpointing_func', 'enable_gradient_checkpointing')]


def public_refused(owner, family):
    for mode, implementation in (('auto', {}), ('plan', {}), ('manual', option(family))):
        with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
            owner.admit_execution(mode, implementation)
    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'): deepcopy(owner)


def _uninterrupted(config, count):
    """Independent normal native execution; return only CPU artifacts."""
    components, data, algorithm, _ = construct(config)
    owner = components.state['native_cycle']; initial = cpu_tree(owner.state_dict())
    states, outputs, events = [], [], []
    for cycle in range(count):
        rows = []; owner.observer = lambda name, value: rows.append((name, cpu_tree(value)))
        workload = build_cycle(config, components, data, algorithm, cycle=cycle, cursor=cycle)
        outputs.append(cpu_tree(owner.run(workload.feeds, cycle=cycle)))
        states.append(cpu_tree(owner.state_dict())); events.append(rows)
    owner.observer = None
    del owner, workload, components, data, algorithm
    gc.collect()
    return initial, states, outputs, events


def _source_oracle(config, family, initial, count):
    """Actual pinned upstream trainer statements, never another live owner."""
    components, data, algorithm, _ = construct(config)
    owner = components.state['native_cycle']; owner.load_state_dict(initial)
    cycles = [build_cycle(config, components, data, algorithm, cycle=i, cursor=i).feeds for i in range(count)]
    outer = rng_state(device=owner.device)
    with device_scope(owner.device):
        if family == 'self_forcing':
            trainer, events, native_rng = sf_oracle(owner, cycles)
            extra = dict(ema=cpu_tree(trainer.generator_ema.state_dict()),
                         unconditional_dict=cpu_tree(trainer.unconditional_dict), events=events,
                         native_step=trainer.step, rng=native_rng)
            del trainer
        else:
            # Existing exact-source oracle owns a single complete DMD2 cycle.
            result = cpu_tree(dmd2_oracle(owner, cycles[0], owner.source.root))
            extra = dict(result, schedulers={k: v.state_dict() for k, v in owner.schedulers.items()})
        restore_rng(outer)
    result = dict(extra, model=cpu_tree(owner.model.state_dict()),
                  optimizers={name: cpu_tree(opt.state_dict()) for name, opt in owner.optimizers.items()})
    del owner, components, data, algorithm, cycles
    gc.collect()
    return result


def _assert_oracle(family, oracle, states, outputs, events):
    index = -1 if family == 'self_forcing' else 0
    for name in ('model', 'optimizers', 'rng'):
        assert equal(oracle[name], states[index][name]), name
    if family == 'self_forcing':
        assert equal(oracle['ema'], states[-1]['ema'])
        assert equal(oracle['unconditional_dict'], states[-1]['unconditional_dict'])
        assert oracle['native_step'] == states[-1]['native_step']
        actual = [(name, value) for rows in events for name, value in rows
                  if name.endswith(('_gradients', '_rollout_cache'))]
        assert equal(actual, [(name, value) for name, value in oracle['events'] if name != 'loss'])
        losses = []
        for output in outputs:
            for row in output['iterations']:
                if row['generator_turn']:
                    losses.append(dict(generator_loss=row['losses']['generator_loss'],
                                       generator_grad_norm=row['generator_grad_norm']))
                losses.append(dict(critic_loss=row['losses']['critic_loss'], critic_grad_norm=row['critic_grad_norm']))
        assert losses == [value for name, value in oracle['events'] if name == 'loss']
    else:
        assert equal(oracle['schedulers'], states[0]['schedulers'])
        assert [row['losses'] for row in outputs[0]['iterations']] == oracle['losses']
        assert equal([value for name, value in events[0] if name.endswith('_gradients')], oracle['gradients'])
        assert equal([value for name, value in events[0] if name == 'guidance_input_cache'], oracle['caches'])


@pytest.mark.parametrize('warmed', [False, True])
def test_actual_native_cuda_serial_trials_match_source_and_continue(native_case, warmed, tmp_path):
    family, config = native_case; first = int(warmed)
    initial, expected, outputs, expected_events = _uninterrupted(config, first + 2)
    oracle = _source_oracle(config, family, initial, first + 2)
    _assert_oracle(family, oracle, expected, outputs, expected_events)
    del oracle
    # Both prior CUDA owners have been released before this owner is constructed.
    components, data, algorithm, _ = construct(config)
    owner = components.state['native_cycle']; owner.load_state_dict(initial)
    if warmed:
        warm = build_cycle(config, components, data, algorithm, cycle=0, cursor=0)
        owner.run(warm.feeds, cycle=0)
    workload = build_cycle(config, components, data, algorithm, cycle=first, cursor=first)
    before = cpu_tree(owner.state_dict()); feeds = _digest(workload.feeds)
    outer = rng_state(device=owner.device); selected_device = torch.cuda.current_device()
    other = next((i for i in range(torch.cuda.device_count()) if i != owner.device.index), None)
    other_rng = None if other is None else torch.cuda.get_rng_state(other).clone()
    original = pointers(owner); flags = physical(owner)
    public_refused(owner, family)
    assert len(owner.execution_options()['options']) == 1
    receipts = []
    with ExecutionBackend(workload).resident_trial(budget()) as trial:
        assert all(t.device.type == 'cpu' for t in tree_leaves(trial._backup) if isinstance(t, torch.Tensor))
        public_refused(owner, family)  # Public gates stay closed even during admission.
        assert len(owner.execution_options()['options']) == 2
        trial.reference()
        assert equal(before, cpu_tree(owner.state_dict())) and pointers(owner) == original
        for enabled in ((True, False) if family == 'self_forcing' else (False, True)):
            events = []; owner.observer = lambda name, value: events.append((name, cpu_tree(value)))
            session, output, gate, evidence = trial.candidate(option(family, enabled))
            assert gate['status'] == 'PASS' and evidence[0]['active']
            assert session.state['native_cycle'] is owner
            assert equal(cpu_tree(owner.state_dict()), expected[first])
            assert equal(events, expected_events[first])
            assert equal(cpu_tree(next(iter(output.values()))), outputs[first])
            assert session.receipt['dispatcher']['final_versions']['native_cycle'] == first + 1
            receipts.append(cpu_tree(session.effects))
            assert _digest(workload.feeds) == feeds and physical(owner) == flags
            assert all(p.grad is None for p in owner.model.parameters())
        owner.observer = None
        trial.accept()
    assert trial.receipt['adopted'] and trial.receipt['closed']
    assert owner.cycle == first + 1 and equal(outer, rng_state(device=owner.device))
    assert torch.cuda.current_device() == selected_device
    if other is not None: assert torch.equal(other_rng, torch.cuda.get_rng_state(other))
    assert len(owner.execution_options()['options']) == 1
    public_refused(owner, family)
    current = pointers(owner)
    for key in ('model', 'parameters', 'optimizers'): assert current[key] == original[key]
    if warmed: assert current['slots'] == original['slots']
    continuation = build_cycle(config, components, data, algorithm, cycle=first + 1, cursor=first + 1)
    # A selected option cannot escape its budget; normal native reference can continue.
    with pytest.raises(ValueError, match='owner'):
        Session(continuation, option(family), copy_state=False)
    resumed = Session(continuation, BASELINE, copy_state=False); result = resumed.run()
    assert equal(cpu_tree(owner.state_dict()), expected[first + 1])
    assert equal(cpu_tree(next(iter(result.values()))), outputs[first + 1])
    proof = dict(family=family, warmed=warmed, evidence='ACTUAL_RANDOM_TINY_NATIVE_CUDA_NOT_PRETRAINED',
        source=owner.source.identity, budget=vars(budget()), transaction=trial.receipt,
        native_effects=receipts, source_oracle='EXACT_PASS', state_grad_adam_scheduler_ema_rng='EXACT_PASS',
        accepted_cycles=1, continued_cycles=1, public_cuda='REFUSED', unowned_rng='NOT_VISIBLE' if other is None else 'UNCHANGED')
    (tmp_path / 'resident-trial-proof.json').write_text(json.dumps(proof, indent=2))
    print('CUDA_RESIDENT_TRIAL_PROOF ' + json.dumps(proof, sort_keys=True))


@pytest.mark.parametrize('warmed', [False, True])
def test_actual_native_cuda_failure_refusal_and_publication_restore(native_case, warmed, monkeypatch, tmp_path):
    family, config = native_case
    components, data, algorithm, _ = construct(config); owner = components.state['native_cycle']
    if warmed:
        owner.run(build_cycle(config, components, data, algorithm, cycle=0, cursor=0).feeds, cycle=0)
    cycle = owner.cycle
    workload = build_cycle(config, components, data, algorithm, cycle=cycle, cursor=cycle)
    before = cpu_tree(owner.state_dict()); pointers_before = pointers(owner); flags = physical(owner)
    outer = rng_state(device=owner.device); feeds = _digest(workload.feeds)
    def unchanged():
        assert equal(before, cpu_tree(owner.state_dict())) and pointers(owner) == pointers_before
        assert equal(outer, rng_state(device=owner.device)) and _digest(workload.feeds) == feeds
        assert physical(owner) == flags and all(p.grad is None for p in owner.model.parameters())
        if family == 'self_forcing' and not warmed:
            assert owner.generator_ema is owner.unconditional_dict is None
        if not warmed: assert all(not opt.state for opt in owner.optimizers.values())
    def forbidden(*args, **kwargs): raise AssertionError('second owner/optimizer copy or checkpoint rollback')
    monkeypatch.setattr(owner, 'load_state_dict', forbidden)
    for optimizer in owner.optimizers.values(): monkeypatch.setattr(optimizer, 'load_state_dict', forbidden)
    evidence = []
    for phase in ('reference', 'partial', 'publication', 'option', 'effect', 'comparison'):
        from distillschedule.open_workload import resident_trial as module
        exception = {'reference': RuntimeError, 'partial': RuntimeError, 'publication': OSError, 'option': ValueError,
                     'effect': ValueError, 'comparison': KeyboardInterrupt}[phase]
        with monkeypatch.context() as patch, pytest.raises(exception):
            with ExecutionBackend(workload).resident_trial(budget()) as trial:
                if phase == 'reference':
                    def failed_reference(name, value):
                        if name == 'generator_committed': raise RuntimeError('partial reference commit')
                    owner.observer = failed_reference
                trial.reference()
                if phase == 'partial':
                    def fail(name, value):
                        if name == 'generator_committed':
                            with torch.no_grad():
                                frozen = next(iter(owner.frozen_hashes))
                                owner.model.state_dict()[frozen].add_(1)
                                next(t for t in tree_leaves(workload.feeds) if isinstance(t, torch.Tensor)).add_(1)
                                next(iter(owner.model.parameters())).requires_grad_(False)
                            owner.model.training = not owner.model.training
                            raise RuntimeError('partial native commit')
                    owner.observer = fail
                if phase == 'comparison':
                    def interrupted(*args, **kwargs): raise KeyboardInterrupt('comparison interrupted')
                    patch.setattr(module, 'streaming_compare', interrupted)
                def execute(session):
                    output = session.run()
                    if phase == 'effect': session.effects['owner_execution'] = {}
                    return output
                selected = option(family)
                if phase == 'option': selected['owner_execution']['native_cycle']['unknown'] = True
                trial.candidate(selected, execute=execute)
                if phase == 'publication':
                    trial.accept(); raise OSError('publication failed')
        owner.observer = None; unchanged()
        assert not trial.receipt['adopted'] and not hasattr(owner, '_resident_trial_poisoned')
        evidence.append(dict(phase=phase, receipt=trial.receipt, restored='EXACT_PASS'))
    for field in ('host_limit_bytes', 'snapshot_limit_bytes', 'device_limit_bytes',
                  'device_state_growth_bytes', 'device_gradient_bytes'):
        with monkeypatch.context() as patch:
            patch.setattr(module, '_copy', forbidden)
            with pytest.raises(ValueError):
                with ExecutionBackend(workload).resident_trial(budget(**{field: 1})): pass
        unchanged()
    owner.run(workload.feeds, cycle=cycle)
    assert owner.cycle == cycle + 1
    proof = dict(family=family, warmed=warmed, failures=evidence,
                 pre_snapshot_capacity_refusals=5, usable_native_continuation='PASS')
    (tmp_path / 'resident-rollback-proof.json').write_text(json.dumps(proof, indent=2))
    print('CUDA_RESIDENT_ROLLBACK_PROOF ' + json.dumps(proof, sort_keys=True))


def test_actual_native_cuda_irrecoverable_snapshot_preserves_both_errors(native_case, tmp_path):
    family, config = native_case
    components, data, algorithm, _ = construct(config); owner = components.state['native_cycle']
    workload = build_cycle(config, components, data, algorithm, cycle=0, cursor=0)
    outer = rng_state(device=owner.device); original = OSError('publication failed')
    with pytest.raises(RollbackError) as captured:
        with ExecutionBackend(workload).resident_trial(budget()) as trial:
            trial.reference(); trial.candidate(option(family))
            next(iter(trial._backup['owner']['model'].values())).add_(1)
            raise original
    assert captured.value.original_error is original and captured.value.rollback_error is not None
    assert owner._resident_trial_poisoned and not trial.receipt['adopted']
    assert trial._backup is None and not hasattr(owner, '_resident_trial_active')
    assert equal(outer, rng_state(device=owner.device))
    with pytest.raises(ValueError, match='poisoned'): owner.run(workload.feeds, cycle=owner.cycle)
    with pytest.raises(ValueError, match='poisoned'):
        with ExecutionBackend(workload).resident_trial(budget()): pass
    proof = dict(family=family, outcome='POISONED_AS_REQUIRED', original=repr(captured.value.original_error),
                 restore=repr(captured.value.rollback_error), transaction=trial.receipt)
    (tmp_path / 'resident-poison-proof.json').write_text(json.dumps(proof, indent=2))
    print('CUDA_RESIDENT_POISON_PROOF ' + json.dumps(proof, sort_keys=True))


def test_actual_cuda_streamed_hash_compare_and_storage_accounting(monkeypatch):
    cpu = torch.arange(257, dtype=torch.float64).to(torch.complex128) * (1 + 2j)
    device = cpu.to('cuda:' + str(torch.cuda.device_count() - 1))
    count = []
    real_cpu = torch.Tensor.cpu
    def copied(tensor, *args, **kwargs):
        if tensor.device.type == 'cuda': count.append(tensor.numel() * tensor.element_size())
        return real_cpu(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', copied)
    assert _digest(device, scratch_bytes=128) == _digest(cpu, scratch_bytes=128)
    assert max(count) <= 128
    count.clear()
    assert streaming_compare(cpu, device, scratch_bytes=320)['status'] == 'PASS'
    assert max(count) <= 32
    row = _memory_inventory((cpu, device, device))
    assert row['host_storage_bytes'] == row['device_storage_bytes'] == cpu.numel() * cpu.element_size()
    assert row['serialized_bytes'] == 3 * row['host_storage_bytes']
    snapshot = _copy(device, row['serialized_bytes'])
    assert snapshot.device.type == 'cpu' and snapshot.dtype == cpu.dtype and torch.equal(snapshot, cpu)
