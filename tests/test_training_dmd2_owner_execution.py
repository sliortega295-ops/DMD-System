"""Actual pinned DMD2 / Diffusers CPU native checkpoint capability acceptance."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from distillschedule.open_workload.checkpoint_policy import native_effect_evidence
from distillschedule.open_workload.torch_backend import Session, candidates, compare, signature
from distillschedule.training.checkpoint import cpu_tree, rng_state, restore_rng
from distillschedule.training.dmd2 import build_dmd2
from test_training_dmd2_cpu import source_root, model_and_feeds, _oracle, REPO
from test_training_end_to_end import equal, run_cli


@pytest.fixture(params=('sd15', 'sdxl'))
def native_assets(tmp_path, source_root, request):
    from distillschedule.training.plugins import load_callable
    builder, _ = load_callable(str(REPO / 'recipes/dmd2_sd15/create_cpu_fixture.py') + ':build_fixture')
    return builder(tmp_path / request.param, source_root, variant=request.param, cycles=3, ratio=2)


def option(enabled=True):
    return dict(backend='torch-callable-resident-v1', share_immutable=False, checkpoint_bindings=[],
                owner_execution={'native_cycle': {'gradient_checkpointing': enabled}})


def physical(owner):
    result = [owner.model.gradient_checkpointing, owner.model.guidance_model.gradient_checkpointing]
    for root in (owner.model.feedforward_model, owner.model.guidance_model.fake_unet):
        result.append({name: {key: vars(module).get(key, 'ABSENT') for key in
                             ('gradient_checkpointing', '_gradient_checkpointing_func',
                              'enable_gradient_checkpointing', 'disable_gradient_checkpointing')}
                       for name, module in root.named_modules()})
    return result


def next_cycle(workload, session, cycle):
    return replace(workload, state=session.state,
                   program=replace(workload.program,
                       calls=tuple(replace(call, attributes=dict(call.attributes, cycle=cycle))
                                   for call in workload.program.calls),
                       initial_versions=session.receipt['dispatcher']['final_versions']))


def test_actual_native_checkpoint_is_pure_scoped_and_differential_twice(native_assets, source_root):
    config, components, feeds = model_and_feeds(native_assets)
    owner = components.state['native_cycle']; before = signature(owner); flags = physical(owner)
    descriptor = owner.execution_options()
    assert descriptor['baseline'] == {'gradient_checkpointing': False}
    assert signature(owner) == before and physical(owner) == flags
    unaffected = [(module, {key: vars(module).get(key, 'ABSENT') for key in
                           ('gradient_checkpointing', '_gradient_checkpointing_func')})
                  for root in (owner.model.guidance_model.real_unet, owner.model.text_encoder, owner.model.vae)
                  for module in root.modules()]
    with owner.execution_scope({'gradient_checkpointing': True}, {}):
        assert signature(owner) == before
        assert all(values == {key: vars(module).get(key, 'ABSENT') for key in values}
                   for module, values in unaffected)
    assert signature(owner) == before and physical(owner) == flags
    oracle = deepcopy(owner); outer = rng_state(); expected_oracle = _oracle(oracle, feeds, source_root); restore_rng(outer)
    workload = build_dmd2(components, feeds, {}, 0, 17)
    assert not workload.program.calls[0].replay_safe
    assert candidates(workload)['candidates'] == [dict(backend='torch-callable-resident-v1',
        share_immutable=False, checkpoint_bindings=[]), option()]
    baseline = Session(workload, candidates(workload)['candidates'][0], persistent=True)
    selected = Session(workload, option(), persistent=True)
    all_counts = []
    for cycle in range(2):
        gradients = []; actual_gradients = []
        baseline.state['native_cycle'].observer = lambda event, value: gradients.append((event, cpu_tree(value)))
        selected.state['native_cycle'].observer = lambda event, value: actual_gradients.append((event, cpu_tree(value)))
        a = baseline.run(); b = selected.run()
        assert compare(baseline.result_state(a), selected.result_state(b))['status'] == 'PASS'
        assert equal(gradients, actual_gradients)
        active = selected.state['native_cycle']
        assert active.native_step == (cycle + 1) * 2
        assert active.generator_updates == cycle + 1 and active.guidance_updates == (cycle + 1) * 2
        assert active.args.gradient_checkpointing is False and active.identity == owner.identity
        assert physical(active) == flags and equal(outer, rng_state())
        evidence = native_effect_evidence(option(), selected.effects)
        assert evidence['active'], evidence
        counts = selected.effects['owner_execution']['native_cycle']['components']
        assert set(counts) == {'generator', 'fake'}
        assert all(row['checkpoint_requests'] > 0 and row['recompute_calls'] > 0 for row in counts.values())
        all_counts.append(deepcopy(counts))
        if cycle == 0:
            assert [row['losses'] for row in b['dmd2.native_cycle']['iterations']] == expected_oracle['losses']
            assert equal(active.model.state_dict(), oracle.model.state_dict())
            assert equal({name: opt.state_dict() for name, opt in active.optimizers.items()},
                         {name: opt.state_dict() for name, opt in oracle.optimizers.items()})
            assert equal(active._rng, expected_oracle['rng'])
            baseline.bind_cycle(next_cycle(workload, baseline, 1)); selected.bind_cycle(next_cycle(workload, selected, 1))
    assert all_counts[0] == all_counts[1], 'native request/replay counters must reset each run'
    assert signature(owner) == before and physical(owner) == flags
    print('NATIVE_OWNER_EFFECT_RECEIPT ' + json.dumps(dict(variant=owner.identity['variant'],
          native_ratio=owner.ratio, cycles=2, counts=all_counts, reference='PASS', gpu='NOT_RUN'), sort_keys=True))


def test_scope_restores_missing_callbacks_after_native_exception(native_assets):
    _, components, feeds = model_and_feeds(native_assets)
    owner = components.state['native_cycle']; flags = physical(owner); before = signature(owner)
    session = Session(build_dmd2(components, feeds, {}, 0, 17), option())
    def fail(event, value):
        if event == 'pre_generator_commit_cache':
            raise RuntimeError('injected native failure')
    session.state['native_cycle'].observer = fail
    with pytest.raises(RuntimeError, match='injected native failure'):
        session.run()
    assert physical(session.state['native_cycle']) == flags
    assert owner.args.gradient_checkpointing is False and signature(owner) == before


def test_explicit_disabled_option_observes_zero_requests(native_assets):
    _, components, feeds = model_and_feeds(native_assets)
    session = Session(build_dmd2(components, feeds, {}, 0, 17), option(False)); session.run()
    evidence = native_effect_evidence(option(False), session.effects)
    assert evidence['active'], evidence
    assert all(row['checkpoint_requests'] == row['recompute_calls'] == 0 and row['disabled_verified']
               for row in session.effects['owner_execution']['native_cycle']['components'].values())


def test_bound_checkpoint_true_can_be_physically_disabled_without_changing_args(native_assets):
    path = Path(native_assets['config']); value = json.loads(path.read_text())
    value['model']['parameters']['settings']['gradient_checkpointing'] = True
    path.write_text(json.dumps(value))
    _, components, feeds = model_and_feeds(native_assets)
    owner = components.state['native_cycle']; flags = physical(owner)
    workload = build_dmd2(components, feeds, {}, 0, 17)
    assert candidates(workload)['candidates'][1] == option(False)
    a = Session(workload, candidates(workload)['candidates'][0]); oa = a.run()
    b = Session(workload, option(False)); ob = b.run()
    assert compare(a.result_state(oa), b.result_state(ob))['status'] == 'PASS'
    assert native_effect_evidence(option(False), b.effects)['active']
    assert b.state['native_cycle'].args.gradient_checkpointing is True
    assert physical(b.state['native_cycle']) == flags and physical(owner) == flags


def test_unqualified_checkpoint_stack_keeps_actual_native_reference_available(native_assets, monkeypatch):
    from distillschedule.training import sd
    actual_runtime = sd.runtime_identity()
    # Admission-only simulation, not a claim to have validated another library.
    unqualified = dict(actual_runtime, diffusers='UNQUALIFIED_CHECKPOINT_RUNTIME_TEST')
    monkeypatch.setattr(sd, 'runtime_identity', lambda: deepcopy(unqualified))
    _, components, feeds = model_and_feeds(native_assets)
    owner = components.state['native_cycle']; before = signature(owner)
    descriptor = owner.execution_options()
    assert descriptor['options'] == [descriptor['baseline']]
    assert descriptor['identity']['capability']['status'] == 'UNAVAILABLE_RUNTIME'
    workload = build_dmd2(components, feeds, {}, 0, 17)
    inventory = candidates(workload)
    assert inventory['total_candidates'] == 1 and 'owner_execution' not in inventory['candidates'][0]
    session = Session(workload, inventory['candidates'][0]); session.run()
    assert session.state['native_cycle'].native_step == 2 and signature(owner) == before
    with pytest.raises(ValueError, match='owner'):
        Session(workload, option())
    assert signature(owner) == before
    # An actual change relative to the constructed owner's runtime identity is
    # still drift, not an excuse to silently downgrade an optimized checkpoint.
    monkeypatch.setattr(sd, 'runtime_identity', lambda: deepcopy(actual_runtime))
    with pytest.raises(ValueError, match='runtime|dependency'):
        candidates(workload)


def test_native_manual_fresh_process_resume_matches_uninterrupted(native_assets, tmp_path):
    path = Path(native_assets['config']); value = json.loads(path.read_text())
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
    print('NATIVE_OWNER_RESUME_RECEIPT ' + json.dumps(dict(variant=value['model']['name'],
          fresh_processes=3, complete_cycles=3, native_ratio=2, manual='PASS', resume_pair='PASS',
          uninterrupted_state_adam_scheduler_rng='EXACT_PASS', gpu='NOT_RUN'), sort_keys=True))


def test_native_measured_auto_and_saved_plan_bind_same_real_option(native_assets, tmp_path):
    from distillschedule.open_workload.planning import calibrate, plan, run
    from distillschedule.training.policy import CyclePolicy
    from distillschedule.workloads.best_known.registry import write_new_json
    _, components, feeds = model_and_feeds(native_assets)
    workload = build_dmd2(components, feeds, {}, 0, 17); before = signature(workload.state)
    manifest = calibrate(workload, tmp_path / 'cal', max_candidates=2, samples=1,
                         max_program_runs=3, profile_hardware=False)
    assert len(manifest['variants']) == 2 and not manifest['failed']
    document = plan(tmp_path / 'cal/calibration.json', memory_limit_bytes=100_000_000)
    baseline = next(row for row in document['search']['ranked'] if 'owner_execution' not in row['implementation'])
    optimized = next(row for row in document['search']['ranked'] if row['implementation'] == option())
    baseline_bytes = baseline['memory'][0]['peak_bytes']; optimized_bytes = optimized['memory'][0]['peak_bytes']
    assert optimized_bytes < baseline_bytes, 'tracked CPU storage only, not RSS/GPU or a speedup claim'
    limit = (baseline_bytes + optimized_bytes) // 2
    constrained = plan(tmp_path / 'cal/calibration.json', memory_limit_bytes=limit)
    assert constrained['selected']['implementation'] == option()
    write_new_json(tmp_path / 'selected.json', constrained)
    manual = CyclePolicy(dict(mode='manual', implementation=option()))
    a, oa = manual.run(workload, tmp_path / 'manual')
    saved = CyclePolicy(dict(mode='plan', plan=str(tmp_path / 'selected.json')))
    b, ob = saved.run(workload, tmp_path / 'saved')
    auto = CyclePolicy(dict(mode='auto', memory_limit_bytes=limit, max_candidates=2,
                            samples=1, max_program_runs=3))
    c, oc = auto.run(workload, tmp_path / 'auto')
    for policy, session, outputs in ((manual, a, oa), (saved, b, ob), (auto, c, oc)):
        assert policy.bound['implementation'] == option() and policy.bound['initial_reference_gate']['status'] == 'PASS'
        assert native_effect_evidence(option(), session.effects)['active']
        assert compare(a.result_state(oa), session.result_state(outputs))['status'] == 'PASS'
    assert signature(workload.state) == before
    # A changed provider descriptor/helper identity invalidates a common saved
    # plan even when model contents, Program and option spelling are unchanged.
    from distillschedule.training.dmd2_owner import DMD2Owner
    original = DMD2Owner.execution_options
    def changed(self, original=original):
        descriptor = original(self)
        descriptor['identity']['provider_file_sha256'] = '0' * 64
        return descriptor
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DMD2Owner, 'execution_options', changed)
        with pytest.raises(ValueError, match='changed'):
            run(workload, tmp_path / 'selected.json', tmp_path / 'stale-rejected')
    assert not (tmp_path / 'stale-rejected').exists() and signature(workload.state) == before
    print('NATIVE_OWNER_PLANNING_RECEIPT ' + json.dumps(dict(variant=components.state['native_cycle'].identity['variant'],
          manual='PASS', auto='PASS', saved_plan='PASS', stale_provider='REJECTED_BEFORE_EXECUTION',
          baseline_tracked_bytes=baseline_bytes, optimized_tracked_bytes=optimized_bytes,
          capacity_limit=limit, measurements='CPU_TRACKED_STORAGE_DIAGNOSTIC_ONLY', gpu='NOT_RUN'), sort_keys=True))


def resident_budget():
    from distillschedule.open_workload.resident_trial import TrialBudget
    return TrialBudget(host_limit_bytes=2**31, snapshot_limit_bytes=2**29,
        result_limit_bytes=2**29, scratch_bytes=2**29, reserve_bytes=2**26)


@pytest.mark.parametrize('warmed', [False, True])
def test_resident_native_partial_commit_rolls_back_without_checkpoint_load(native_assets, monkeypatch, warmed):
    """Actual native math; a partial G commit is not a checkpoint boundary."""
    from distillschedule.open_workload.execution import ExecutionBackend
    _, components, feeds = model_and_feeds(native_assets)
    owner = components.state['native_cycle']
    if warmed:
        owner.run(feeds, cycle=0)
    cycle = owner.cycle
    workload = build_dmd2(components, feeds, {}, cycle, 17)
    before = signature(owner); plain = signature(owner._plain_tensors()); flags = physical(owner)
    outer = signature(rng_state(device='cpu'))
    objects = [id(owner.model), *map(id, owner.model.parameters()), *map(id, owner.optimizers.values())]
    moments = [t.data_ptr() for opt in owner.optimizers.values() for row in opt.state.values() for t in row.values()]
    def forbidden(*args, **kwargs):
        raise AssertionError('resident rollback must not deepcopy/load another model or optimizer')
    monkeypatch.setattr(type(owner), '__deepcopy__', forbidden)
    monkeypatch.setattr(owner, 'load_state_dict', forbidden)
    for optimizer in owner.optimizers.values(): monkeypatch.setattr(optimizer, 'load_state_dict', forbidden)
    with pytest.raises(RuntimeError, match='partial native generator commit'):
        with ExecutionBackend(workload).resident_trial(resident_budget()) as trial:
            trial.reference()
            def fail(event, value):
                if event == 'generator_committed':
                    raise RuntimeError('partial native generator commit')
            owner.observer = fail
            trial.candidate(option())
    owner.observer = None
    assert signature(owner) == before and signature(owner._plain_tensors()) == plain and physical(owner) == flags
    assert signature(rng_state(device='cpu')) == outer
    assert objects == [id(owner.model), *map(id, owner.model.parameters()), *map(id, owner.optimizers.values())]
    assert moments == [t.data_ptr() for opt in owner.optimizers.values() for row in opt.state.values() for t in row.values()]
    # Reference/selected really execute the existing Session on this same owner.
    with ExecutionBackend(workload).resident_trial(resident_budget()) as trial:
        trial.reference()
        session, outputs, gate, native = trial.candidate(option(), persistent=True)
        assert gate['status'] == 'PASS' and native[0]['active'] and session.state['native_cycle'] is owner
        trial.accept()
    assert owner.cycle == cycle + 1 and trial.receipt['adopted']
    if not warmed:
        session.bind_cycle(next_cycle(workload, session, cycle + 1)); session.run()
        assert owner.cycle == cycle + 2


@pytest.mark.parametrize('marker', ['', None, False, True])
def test_review_r2_dmd2_poison_guard_precedes_native_access(marker, monkeypatch):
    """Guard-only CPU proof, not native math qualification or a model fixture."""
    from distillschedule.training.dmd2_owner import DMD2Owner
    owner = DMD2Owner.__new__(DMD2Owner)
    owner._resident_trial_poisoned = marker
    def forbidden(): raise AssertionError('poisoned owner reached native validation')
    monkeypatch.setattr(owner, '_validate_live', forbidden)
    for action in (lambda: owner.run({}, cycle=0), owner.state_dict, owner._resident_trial_validate):
        with pytest.raises(ValueError, match='poisoned'): action()


def test_private_cuda_transaction_does_not_relax_public_dmd2_admission():
    """CPU guard proof only; no CUDA device or native model is constructed."""
    from distillschedule.training.dmd2_owner import DMD2Owner
    owner = DMD2Owner.__new__(DMD2Owner); owner.device = torch.device('cuda:0')
    owner._resident_trial_active = True  # A marker must never bypass public gates.
    for mode, implementation in (('auto', {}), ('plan', {}), ('manual', option())):
        with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
            owner.admit_execution(mode, implementation)
    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'): deepcopy(owner)
    owner.admit_execution('reference', {})
