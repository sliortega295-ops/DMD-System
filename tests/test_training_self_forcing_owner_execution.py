"""Actual source tiny CPU checkpoint policy; never CUDA/pretrained evidence."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from distillschedule.open_workload.checkpoint_policy import native_effect_evidence
from distillschedule.open_workload.torch_backend import Session, candidates, compare, signature
from distillschedule.training.checkpoint import cpu_tree, rng_state, restore_rng
from distillschedule.training.self_forcing import build_self_forcing
from test_training_end_to_end import equal, run_cli
from test_training_self_forcing_cpu import (
    fixture_assets, source_root, model_and_data, _native_trainer_oracle, _rollout_trace,
)


@pytest.fixture
def checkpoint_assets(fixture_assets):
    path = Path(fixture_assets['config']); value = json.loads(path.read_text())
    value['model']['parameters']['settings']['gradient_checkpointing'] = True
    path.write_text(json.dumps(value))
    return fixture_assets


def option(enabled=False):
    return dict(backend='torch-callable-resident-v1', share_immutable=False, checkpoint_bindings=[],
                owner_execution={'native_cycle': {'fake_gradient_checkpointing': enabled}})


def physical(owner):
    return {name: {key: vars(module).get(key, 'ABSENT') for key in
                  ('gradient_checkpointing', 'enable_gradient_checkpointing', '_set_gradient_checkpointing')}
            for name, module in owner.model.named_modules()}


def test_explicit_native_on_reference_and_pure_fake_only_descriptor(checkpoint_assets, source_root):
    from distillschedule.training.self_forcing import original
    api = original(source_root)
    checkpoint = torch.utils.checkpoint.checkpoint
    setter = api.wan.WanModel.enable_gradient_checkpointing
    _, components, data = model_and_data(checkpoint_assets)
    outer = rng_state()
    owner = components.state['native_cycle']; before = signature(owner); flags = physical(owner)
    assert owner.args.gradient_checkpointing is True
    assert owner.model.generator.model.gradient_checkpointing is True
    assert owner.model.fake_score.model.gradient_checkpointing is True
    assert owner.model.real_score.model.gradient_checkpointing is False
    descriptor = owner.execution_options()
    assert descriptor['baseline'] == {'fake_gradient_checkpointing': True}
    assert descriptor['options'] == [{'fake_gradient_checkpointing': True}, {'fake_gradient_checkpointing': False}]
    assert descriptor['effect']['components'] == ['fake']
    assert descriptor['identity']['capability']['status'] == 'AVAILABLE_TINY_CPU'
    assert signature(owner) == before and flags == physical(owner) and equal(outer, rng_state())
    assert api.wan.WanModel.enable_gradient_checkpointing is setter
    assert torch.utils.checkpoint.checkpoint is checkpoint
    for root in (owner.model.generator.model, owner.model.fake_score.model):
        assert 'enable_gradient_checkpointing' not in vars(root)
    workload = build_self_forcing(components, data.cycle_sample(0), {}, 0, 17)
    assert not workload.program.calls[0].replay_safe
    assert candidates(workload)['candidates'] == [dict(backend='torch-callable-resident-v1',
        share_immutable=False, checkpoint_bindings=[]), option()]


def test_legacy_reference_never_searches_generator_checkpoint(fixture_assets):
    _, components, data = model_and_data(fixture_assets)
    owner = components.state['native_cycle']
    assert owner.args.gradient_checkpointing is False
    assert owner.execution_options()['options'] == [{'fake_gradient_checkpointing': False}]
    assert owner.execution_options()['identity']['capability']['status'] == 'UNAVAILABLE_REFERENCE_POLICY'
    workload = build_self_forcing(components, data.cycle_sample(0), {}, 0, 17)
    assert candidates(workload)['total_candidates'] == 1
    with pytest.raises(ValueError, match='owner'):
        Session(workload, option(True))


def test_three_native_cycles_source_trainer_oracle_and_real_counters(checkpoint_assets):
    _, components, data = model_and_data(checkpoint_assets)
    owner = components.state['native_cycle']; before = signature(owner); flags = physical(owner)
    feeds = [data.cycle_sample(index) for index in range(3)]
    outer = rng_state(); reference = deepcopy(owner)
    expected_trace, hook = _rollout_trace(reference)
    trainer, expected_events, expected_rng = _native_trainer_oracle(reference, feeds)
    hook.remove(); restore_rng(outer)
    sessions = [Session(build_self_forcing(components, feeds[0], {}, 0, 17), option(enabled), persistent=True)
                for enabled in (True, False)]
    event_lists, traces, hooks, receipts, results = [], [], [], [], []
    for session in sessions:
        events = []; event_lists.append(events)
        session.state['native_cycle'].observer = lambda name, value, target=events: target.append((name, cpu_tree(value)))
        trace, hook = _rollout_trace(session.state['native_cycle']); traces.append(trace); hooks.append(hook)
    for cycle in range(3):
        outputs = []
        for session in sessions:
            if cycle:
                workload = build_self_forcing(replace(components, state=session.state), feeds[cycle], {}, cycle, 17)
                session.bind_cycle(workload)
            outputs.append(session.run())
            active = session.state['native_cycle']
            assert active.native_step == (cycle + 1) * 5 and active.generator_updates == cycle + 1
            assert active.critic_updates == (cycle + 1) * 5 and active.ema_updates == cycle
            assert active.model.inference_pipeline is None
            assert active.args.gradient_checkpointing is True and physical(active) == flags
        assert compare(sessions[0].result_state(outputs[0]), sessions[1].result_state(outputs[1]))['status'] == 'PASS'
        assert equal(outputs[0], outputs[1]) and equal(event_lists[0], event_lists[1])
        for enabled, session in zip((True, False), sessions):
            assert native_effect_evidence(option(enabled), session.effects)['active']
        on, off = [s.effects['owner_execution']['native_cycle']['components']['fake'] for s in sessions]
        assert on['checkpoint_requests'] == on['recompute_calls'] == 5
        assert on['direct_grad_calls'] == 0 and not on['disabled_verified']
        assert off['checkpoint_requests'] == off['recompute_calls'] == 0
        assert off['direct_grad_calls'] == 5 and off['disabled_verified']
        receipts.append(dict(cycle=cycle + 1, on=deepcopy(on), off=deepcopy(off)))
        results.append(outputs[1]['self_forcing.native_cycle'])
    for hook in hooks:
        hook.remove()
    expected_filtered = [(name, value) for name, value in expected_events if name != 'loss']
    for active, trace, events in zip((s.state['native_cycle'] for s in sessions), traces, event_lists):
        assert equal(trace, expected_trace)
        assert equal([(name, value) for name, value in events if name.endswith(('_gradients', '_rollout_cache'))],
                     expected_filtered)
        assert equal(active.model.state_dict(), reference.model.state_dict())
        assert equal({name: opt.state_dict() for name, opt in active.optimizers.items()},
                     {name: opt.state_dict() for name, opt in reference.optimizers.items()})
        assert equal(active.generator_ema.state_dict(), trainer.generator_ema.state_dict())
        assert equal(active._rng, expected_rng) and equal(active.unconditional_dict, trainer.unconditional_dict)
    losses = []
    for result in results:
        for row in result['iterations']:
            if row['generator_turn']:
                losses.append(dict(generator_loss=row['losses']['generator_loss'], generator_grad_norm=row['generator_grad_norm']))
            losses.append(dict(critic_loss=row['losses']['critic_loss'], critic_grad_norm=row['critic_grad_norm']))
    assert losses == [value for name, value in expected_events if name == 'loss']
    assert signature(owner) == before and physical(owner) == flags and equal(outer, rng_state())
    assert not torch.cuda.is_initialized()
    print('SELF_FORCING_CHECKPOINT_DIFFERENTIAL ' + json.dumps(dict(cycles=3, native_steps=15,
        generator_checkpoint='FIXED_TRUE', fake_counts=receipts, gradients_losses_adam_ema_cache_rng='EXACT_PASS',
        source_trainer='EXACT_PASS', gpu='NOT_RUN'), sort_keys=True))


@pytest.mark.parametrize('enabled', [True, False])
def test_scope_exception_restores_flags_and_removes_only_its_hooks(checkpoint_assets, enabled):
    _, components, data = model_and_data(checkpoint_assets)
    owner = components.state['native_cycle']; flags = physical(owner); before = signature(owner)
    session = Session(build_self_forcing(components, data.cycle_sample(0), {}, 0, 17), option(enabled))
    active = session.state['native_cycle']
    sentinel = active.model.fake_score.model.blocks[0].register_forward_pre_hook(lambda *args: None)
    hooks = {name: dict(module._forward_pre_hooks) for name, module in active.model.named_modules()}
    def fail(event, value):
        if event == 'critic_gradients':
            raise RuntimeError('injected after real native backward')
    active.observer = fail
    with pytest.raises(RuntimeError, match='injected after real native backward'):
        session.run()
    assert physical(active) == flags and active._active_fake_checkpoint is None
    assert {name: dict(module._forward_pre_hooks) for name, module in active.model.named_modules()} == hooks
    sentinel.remove()
    assert signature(owner) == before


@pytest.mark.parametrize('invalid', [{}, {'fake_gradient_checkpointing': 1},
    {'gradient_checkpointing': False}, {'generator_gradient_checkpointing': False},
    {'fake_gradient_checkpointing': False, 'generator_gradient_checkpointing': False}])
def test_generator_unknown_and_nonboolean_overrides_reject_without_mutation(checkpoint_assets, invalid):
    _, components, data = model_and_data(checkpoint_assets)
    owner = components.state['native_cycle']; before = signature(owner); flags = physical(owner)
    with pytest.raises(ValueError, match='option'):
        with owner.execution_scope(invalid, {}):
            pytest.fail('unsupported native option entered')
    assert signature(owner) == before and physical(owner) == flags
    owner.model.generator.model.gradient_checkpointing = False
    with pytest.raises(ValueError, match='checkpoint flags'):
        owner.execution_options()
    owner.model.generator.model.gradient_checkpointing = True


def test_unqualified_runtime_retains_on_reference_and_rejects_override(checkpoint_assets, monkeypatch):
    from distillschedule.training import self_forcing
    actual = self_forcing.runtime_identity()
    unqualified = dict(actual, diffusers='UNQUALIFIED_ADMISSION_SIMULATION')
    monkeypatch.setattr(self_forcing, 'runtime_identity', lambda: deepcopy(unqualified))
    _, components, data = model_and_data(checkpoint_assets)
    owner = components.state['native_cycle']; before = signature(owner)
    descriptor = owner.execution_options()
    assert descriptor['options'] == [descriptor['baseline']]
    assert descriptor['identity']['capability']['status'] == 'UNAVAILABLE_RUNTIME'
    workload = build_self_forcing(components, data.cycle_sample(0), {}, 0, 17)
    assert candidates(workload)['total_candidates'] == 1
    with pytest.raises(ValueError, match='owner'):
        Session(workload, option())
    baseline = Session(workload, candidates(workload)['candidates'][0]); baseline.run()
    assert baseline.state['native_cycle'].native_step == 5 and signature(owner) == before
    monkeypatch.setattr(self_forcing, 'runtime_identity', lambda: deepcopy(actual))
    with pytest.raises(ValueError, match='runtime'):
        candidates(workload)


def test_unqualified_native_settings_keep_reference_not_alternative(checkpoint_assets):
    path = Path(checkpoint_assets['config']); value = json.loads(path.read_text())
    value['model']['parameters']['settings']['context_noise'] = 12
    path.write_text(json.dumps(value))
    _, components, data = model_and_data(checkpoint_assets)
    owner = components.state['native_cycle']
    assert owner.execution_options()['identity']['capability']['status'] == 'UNAVAILABLE_TINY_PROFILE'
    workload = build_self_forcing(components, data.cycle_sample(0), {}, 0, 17)
    assert candidates(workload)['total_candidates'] == 1
    with pytest.raises(ValueError, match='owner'):
        Session(workload, option())


def test_empty_scope_does_not_invent_effects_and_rejects_nested_scope(checkpoint_assets):
    _, components, _ = model_and_data(checkpoint_assets)
    owner = components.state['native_cycle']; before = signature(owner); outer = rng_state()
    effects = {}
    with owner.execution_scope({'fake_gradient_checkpointing': False}, effects):
        assert owner.model.generator.model.gradient_checkpointing is True
        assert owner.model.fake_score.model.gradient_checkpointing is False
        assert signature(owner) == before and equal(outer, rng_state())
        with pytest.raises(ValueError, match='option'):
            with owner.execution_scope({'fake_gradient_checkpointing': True}, {}):
                pytest.fail('nested checkpoint scope')
    row = effects['components']['fake']
    assert row['checkpoint_requests'] == row['recompute_calls'] == row['direct_grad_calls'] == 0
    assert row['disabled_verified'] is False
    assert owner.model.fake_score.model.gradient_checkpointing is True and signature(owner) == before


def test_source_lock_and_generator_reference_identity_reject_before_mutation(checkpoint_assets, monkeypatch):
    from distillschedule.training.self_forcing import build_model
    config, components, _ = model_and_data(checkpoint_assets)
    owner = components.state['native_cycle']; before = signature(owner)
    # Corrupt only this scoped lock value, not any upstream source file.
    with monkeypatch.context() as patch:
        patch.setitem(owner.source.files, 'wan/modules/model.py', '0' * 64)
        with pytest.raises(ValueError, match='SHA256'):
            owner.execution_options()
    parameters = deepcopy(config.model['parameters'])
    parameters['settings']['gradient_checkpointing'] = False
    legacy = build_model(parameters).state['native_cycle']
    with pytest.raises(ValueError, match='identity'):
        owner.load_state_dict(cpu_tree(legacy.state_dict()))
    assert signature(owner) == before


def test_public_manual_fresh_process_resume_exact(checkpoint_assets, tmp_path):
    path = Path(checkpoint_assets['config']); value = json.loads(path.read_text())
    value['execution'] = dict(adapter='program', mode='manual', implementation=option())
    path.write_text(json.dumps(value))
    partial = run_cli('train', path, '--cycles', 1, '--output', tmp_path / 'partial')
    resumed = run_cli('train', path, '--cycles', 3, '--resume', partial['checkpoint'], '--output', tmp_path / 'resumed')
    full = run_cli('train', path, '--cycles', 3, '--output', tmp_path / 'full')
    assert equal(torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True),
                 torch.load(Path(full['checkpoint']) / 'state.pt', weights_only=True))
    policy = json.loads((tmp_path / 'full/effective-policy.json').read_text())
    assert policy['implementation'] == option() and policy['initial_reference_gate']['status'] == 'PASS'
    metrics = [json.loads(row) for row in (tmp_path / 'resumed/metrics.jsonl').read_text().splitlines()]
    assert metrics[0]['execution']['owner_resume_reference_admission']
    assert metrics[0]['execution']['reference_gate']['status'] == 'PASS'
    assert 'owner_resume_reference_admission' not in metrics[1]['execution']
    assert all(native_effect_evidence(option(), row['execution']['effects'])['active'] for row in metrics)
    assert all(row['execution']['dispatcher']['final_versions']['native_cycle'] == row['cycle'] for row in metrics)
    print('SELF_FORCING_CHECKPOINT_RESUME ' + json.dumps(dict(fresh_processes=3, complete_cycles=3,
        native_steps=15, state_optimizer_ema_rng_data_versions='EXACT_PASS', resume_pair='PASS', gpu='NOT_RUN')))


def test_measured_calibration_auto_saved_plan_and_drift(checkpoint_assets, tmp_path):
    from distillschedule.open_workload.planning import calibrate, plan, run
    from distillschedule.training.policy import CyclePolicy
    from distillschedule.training.self_forcing_owner import SelfForcingOwner
    from distillschedule.workloads.best_known.registry import write_new_json
    _, components, data = model_and_data(checkpoint_assets)
    feeds = data.cycle_sample(0)
    workload = build_self_forcing(components, feeds, {}, 0, 17)
    before = signature(workload.state); outer = rng_state(); input_signature = signature(data.cycle_sample(0))
    manifest = calibrate(workload, tmp_path / 'cal', max_candidates=2, samples=2,
                         max_program_runs=5, profile_hardware=False)
    assert len(manifest['variants']) == 2 and not manifest['failed']
    assert signature(workload.state) == before and equal(outer, rng_state())
    assert signature(data.cycle_sample(0)) == input_signature
    document = plan(tmp_path / 'cal/calibration.json', memory_limit_bytes=100_000_000)
    assert len(document['search']['ranked']) == 2
    assert any(row['implementation'] == option() for row in document['search']['ranked'])
    selected = document['selected']['implementation']
    # Selection is from actual native measurements, never replaced with a
    # preferred constant. A measured reference winner is an honest outcome.
    write_new_json(tmp_path / 'selected.json', document)
    policies = [CyclePolicy(dict(mode='manual', implementation=option())),
                CyclePolicy(dict(mode='plan', plan=str(tmp_path / 'selected.json'))),
                CyclePolicy(dict(mode='auto', memory_limit_bytes=100_000_000, max_candidates=2,
                                 samples=2, max_program_runs=5))]
    runs = [policy.run(deepcopy(workload), tmp_path / name)
            for policy, name in zip(policies, ('manual', 'saved', 'auto'))]
    for policy, (session, outputs) in zip(policies, runs):
        assert policy.bound['initial_reference_gate']['status'] == 'PASS'
        assert native_effect_evidence(policy.bound['implementation'], session.effects)['active']
        assert compare(runs[0][0].result_state(runs[0][1]), session.result_state(outputs))['status'] == 'PASS'
    assert policies[0].bound['implementation'] == option()
    assert policies[1].bound['implementation'] == selected
    assert signature(workload.state) == before and equal(outer, rng_state())
    assert signature(data.cycle_sample(0)) == input_signature
    original = SelfForcingOwner.execution_options
    def changed(self, original=original):
        descriptor = original(self)
        descriptor['identity']['provider_file_sha256'] = '0' * 64
        return descriptor
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SelfForcingOwner, 'execution_options', changed)
        with pytest.raises(ValueError, match='changed'):
            run(workload, tmp_path / 'selected.json', tmp_path / 'stale-rejected')
    assert not (tmp_path / 'stale-rejected').exists() and signature(workload.state) == before
    print('SELF_FORCING_CHECKPOINT_PLANNING ' + json.dumps(dict(candidates=2,
        selected=selected, auto_selected=policies[2].bound['implementation'], saved_plan='PASS',
        state_rng_data_unchanged=True, stale_provider='REJECTED', measurements='CPU_DIAGNOSTIC_ONLY', gpu='NOT_RUN')))


@pytest.mark.parametrize('warmed', [False, True])
def test_resident_native_partial_cycle_restores_adam_ema_cache_rng(checkpoint_assets, monkeypatch, warmed):
    """Actual native G1/C5 plus lazy native EMA; no substitute training math."""
    from distillschedule.open_workload.execution import ExecutionBackend
    from test_training_dmd2_owner_execution import resident_budget
    _, components, data = model_and_data(checkpoint_assets)
    owner = components.state['native_cycle']
    if warmed:
        owner.run(data.cycle_sample(0), cycle=0)
    cycle = owner.cycle; feeds = data.cycle_sample(cycle)
    workload = build_self_forcing(components, feeds, {}, cycle, 17)
    before = signature(owner); flags = physical(owner); outer = signature(rng_state(device='cpu'))
    ema = owner.generator_ema
    objects = [id(owner.model), *map(id, owner.model.parameters()), *map(id, owner.optimizers.values())]
    moments = [t.data_ptr() for opt in owner.optimizers.values() for row in opt.state.values() for t in row.values()]
    def forbidden(*args, **kwargs):
        raise AssertionError('resident rollback must not deepcopy/load another model or optimizer')
    monkeypatch.setattr(type(owner), '__deepcopy__', forbidden)
    monkeypatch.setattr(owner, 'load_state_dict', forbidden)
    for optimizer in owner.optimizers.values(): monkeypatch.setattr(optimizer, 'load_state_dict', forbidden)
    with pytest.raises(KeyboardInterrupt, match='partial native critic cycle'):
        with ExecutionBackend(workload).resident_trial(resident_budget()) as trial:
            trial.reference()
            def fail(event, value):
                if event == 'iteration_complete' and owner.native_step == cycle * owner.ratio + 2:
                    assert owner.model.inference_pipeline is not None
                    raise KeyboardInterrupt('partial native critic cycle')
            owner.observer = fail
            trial.candidate(option())
    owner.observer = None
    assert signature(owner) == before and physical(owner) == flags and signature(rng_state(device='cpu')) == outer
    assert owner.generator_ema is ema and owner.model.inference_pipeline is None
    assert objects == [id(owner.model), *map(id, owner.model.parameters()), *map(id, owner.optimizers.values())]
    assert moments == [t.data_ptr() for opt in owner.optimizers.values() for row in opt.state.values() for t in row.values()]
    with ExecutionBackend(workload).resident_trial(resident_budget()) as trial:
        trial.reference()
        session, outputs, gate, native = trial.candidate(option(), persistent=True)
        assert gate['status'] == 'PASS' and native[0]['active'] and session.state['native_cycle'] is owner
        trial.accept()
    assert owner.cycle == cycle + 1 and trial.receipt['adopted']
    if not warmed:
        session.bind_cycle(build_self_forcing(components, data.cycle_sample(cycle + 1), {}, cycle + 1, 17))
        session.run()
        assert owner.cycle == cycle + 2


@pytest.mark.parametrize('marker', ['', None, False, True])
def test_review_r2_self_forcing_poison_guard_precedes_native_access(marker, monkeypatch):
    """Guard-only CPU proof, not native math qualification or a model fixture."""
    from distillschedule.training.self_forcing_owner import SelfForcingOwner
    owner = SelfForcingOwner.__new__(SelfForcingOwner)
    owner._resident_trial_poisoned = marker
    def forbidden(): raise AssertionError('poisoned owner reached native validation')
    monkeypatch.setattr(owner, '_validate_live', forbidden)
    for action in (lambda: owner.run({}, cycle=0), owner.state_dict, owner._resident_trial_validate):
        with pytest.raises(ValueError, match='poisoned'): action()


def test_private_cuda_transaction_does_not_relax_public_self_forcing_admission():
    """CPU guard proof only; the physical G/F policy is not relaxed here."""
    from distillschedule.training.self_forcing_owner import SelfForcingOwner
    owner = SelfForcingOwner.__new__(SelfForcingOwner); owner.device = torch.device('cuda:0')
    owner._resident_trial_active = True
    for mode, implementation in (('auto', {}), ('plan', {}), ('manual', option())):
        with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
            owner.admit_execution(mode, implementation)
    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'): deepcopy(owner)
    owner.admit_execution('reference', {})
