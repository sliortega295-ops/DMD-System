"""Public bounded-owner lifecycle. Generic CPU checks plus actual native fixtures.

The latter use the pinned target dependencies; synthetic providers only test
the preconstruction hook, never native numerical or capacity qualification.
"""
from dataclasses import replace
import json
from pathlib import Path
import os

import pytest

from distillschedule.training.config import TrainingConfig
from distillschedule.training import session
from test_training_end_to_end import fixture


def limits():
    return dict(schema_version='distillschedule.training_resident.v1',
        construction_limit_bytes=128 << 20, host_staging_limit_bytes=128 << 20,
        trial=dict(host_limit_bytes=2 << 30, snapshot_limit_bytes=512 << 20,
                   result_limit_bytes=512 << 20, scratch_bytes=256 << 20, reserve_bytes=32 << 20))


def test_explicit_budget_roundtrip(tmp_path):
    _, config = fixture(tmp_path)
    raw = config.to_dict(); raw['execution']['resident_budget'] = limits()
    parsed = TrainingConfig.from_dict(raw, source=config.source)
    assert parsed.to_dict()['execution']['resident_budget'] == limits()
    assert parsed.identity != config.identity


@pytest.mark.parametrize('bad', [None, {}, {'trial': {}},
    {**limits(), 'construction_limit_bytes': True}, {**limits(), 'host_staging_limit_bytes': 0},
    {**limits(), 'allow_unqualified': True}, {**limits(), 'trial': {**limits()['trial'], 'reserve_bytes': -1}}])
def test_budget_schema_rejects_before_factory(tmp_path, bad):
    _, config = fixture(tmp_path)
    raw = config.to_dict(); raw['execution']['resident_budget'] = bad
    with pytest.raises(ValueError, match='resident|budget'):
        TrainingConfig.from_dict(raw, source=config.source)


def test_no_construction_provider_is_rejected_before_factory(tmp_path, monkeypatch):
    _, config = fixture(tmp_path)
    config = replace(config, execution={**config.execution, 'resident_budget': limits()})
    events = []
    def factory(parameters):
        events.append('allocated')
        raise AssertionError('unadmitted factory executed')
    monkeypatch.setattr(session, 'load_callable', lambda ref: (factory, {'reference': ref, 'sha256': 'test'}))
    with pytest.raises(ValueError, match='construction.*provider|preconstruction'):
        session._construct_model(config)
    assert events == []


def test_declined_construction_provider_cannot_fall_back_to_allocating(tmp_path, monkeypatch):
    _, config = fixture(tmp_path)
    config = replace(config, execution={**config.execution, 'resident_budget': limits()})
    def factory(parameters):
        raise AssertionError('declined resident request allocated a model')
    factory.admit_construction = lambda **kwargs: None
    monkeypatch.setattr(session, 'load_callable', lambda ref: (factory, {'reference': ref, 'sha256': 'test'}))
    with pytest.raises(ValueError, match='declined.*preconstruction'):
        session._construct_model(config)


@pytest.mark.parametrize('fail', ['preflight', 'factory', 'attach', None])
def test_provider_lifetime_and_order(tmp_path, monkeypatch, fail):
    _, config = fixture(tmp_path)
    config = replace(config, execution={**config.execution, 'resident_budget': limits()})
    original, source = session.load_callable(config.model['factory'])
    events = []
    class Admission:
        def attach(self, components):
            events.append('attach')
            if fail == 'attach': raise ValueError('injected attach')
        def close(self): events.append('close')
    def preflight(*, parameters, request, budget):
        events.append('preflight')
        if fail == 'preflight': raise ValueError('injected preflight')
        return Admission()
    def factory(parameters):
        events.append('factory')
        if fail == 'factory': raise ValueError('injected factory')
        components = original(parameters)
        components.close = lambda: events.append('provider_close')
        return components
    factory.admit_construction = preflight
    monkeypatch.setattr(session, 'load_callable', lambda ref: (factory, source))
    if fail:
        with pytest.raises(ValueError, match='injected ' + fail): session._construct_model(config)
        expected = {'preflight': ['preflight'], 'factory': ['preflight', 'factory', 'close'],
                    'attach': ['preflight', 'factory', 'attach', 'provider_close', 'close']}[fail]
    else:
        components, _ = session._construct_model(config)
        assert events == ['preflight', 'factory', 'attach']
        with components.managed(): pass
        expected = ['preflight', 'factory', 'attach', 'provider_close', 'close']
    assert events == expected


@pytest.mark.parametrize('factory', ['sd:build_sd15', 'sd:build_sdxl', 'self_forcing:build_model'])
@pytest.mark.parametrize('mode', ['manual', 'auto', 'plan'])
def test_no_budget_cuda_optimization_refused_before_assets(tmp_path, factory, mode):
    import torch
    _, config = fixture(tmp_path)
    execution = dict(mode=mode)
    if mode == 'manual': execution['implementation'] = implementation('sd15')
    config = replace(config, execution=execution,
        model={'factory': 'distillschedule.training.' + factory, 'parameters': {}},
        trainer=dict(config.trainer, device='cuda:0'))
    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
        session._construct_model(config)
    assert not torch.cuda.is_initialized()


@pytest.fixture(params=['sd15', 'sdxl', 'self_forcing'])
def native_config(tmp_path, request):
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.config import load_training_config
    family = request.param
    key = 'SELF_FORCING_SOURCE_ROOT' if family == 'self_forcing' else 'DMD2_SOURCE_ROOT'
    if key not in os.environ:
        pytest.skip('actual native source required: ' + key)
    folder = 'self_forcing_official' if family == 'self_forcing' else 'dmd2_sd15'
    repo = Path(__file__).resolve().parents[1]
    builder, _ = load_callable(str(repo / 'recipes' / folder / 'create_cpu_fixture.py') + ':build_fixture')
    kwargs = {} if family == 'self_forcing' else dict(variant=family, ratio=2)
    files = builder(tmp_path / 'native', os.environ[key], cycles=3, **kwargs)
    raw = json.loads(Path(files['config']).read_text())
    if family == 'self_forcing':
        raw['model']['parameters']['settings']['gradient_checkpointing'] = True
    raw['execution']['resident_budget'] = limits()
    return family, TrainingConfig.from_dict(raw, source=files['config'])


def implementation(family):
    option = {'fake_gradient_checkpointing': False} if family == 'self_forcing' else {'gradient_checkpointing': True}
    return dict(backend='torch-callable-resident-v1', share_immutable=False, checkpoint_bindings=[],
                owner_execution={'native_cycle': option})


@pytest.mark.parametrize('mode', ['manual', 'auto'])
def test_real_native_public_budget_policy_three_inputs(native_config, tmp_path, mode):
    import torch
    from distillschedule.training.resident import active
    from distillschedule.training.checkpoint import cpu_tree
    from distillschedule.training.policy import CyclePolicy
    from test_training_end_to_end import equal
    family, config = native_config
    control_config = replace(config, execution={'adapter': 'program', 'mode': 'reference'})
    reference, data, algorithm, _ = session.construct(control_config)
    owner = reference.state['native_cycle']
    expected = []
    for cycle in range(3):
        workload = session.build_cycle(control_config, reference, data, algorithm, cycle=cycle, cursor=cycle)
        owner.run(workload.feeds, cycle=cycle)
        expected.append(cpu_tree(owner.state_dict()))
    if reference.close is not None: reference.close()
    del reference, owner, data, algorithm, workload
    execution = dict(config.execution, mode=mode)
    if mode == 'manual': execution['implementation'] = implementation(family)
    else: execution.update(memory_limit_bytes=2 << 30, max_candidates=2, samples=1, max_program_runs=3)
    config = replace(config, execution=execution)
    components, data, algorithm, _ = session.construct(config)
    owner = components.state['native_cycle']; policy = CyclePolicy(execution)
    assert active(owner)
    assert components.construction_receipt['observed_managed_storage']['device_storage_bytes'] == 0
    versions = None; output = tmp_path / 'policy'; output.mkdir()
    try:
        for cycle in range(3):
            workload = session.build_cycle(config, components, data, algorithm, cycle=cycle, cursor=cycle, versions=versions)
            current, _ = policy.run(workload, output)
            assert current.state['native_cycle'] is owner, 'public budget path cloned native owner'
            assert equal(cpu_tree(owner.state_dict()), expected[cycle])
            assert owner.cycle == cycle + 1
            assert active(owner) and not hasattr(owner, '_resident_trial_active')
            versions = current.receipt['dispatcher']['final_versions']
            if mode == 'manual':
                assert all(row['active'] for row in current.receipt['native_capability_evidence'])
        assert policy.bound['resident_budget'] == limits()
        if mode == 'auto':
            assert (output / 'planning/calibration/calibration.json').is_file()
            assert (output / 'plan-admission/execution.json').is_file()
        else:
            assert json.loads((output / 'owner-admission.json').read_text())['resident_trial']['adopted']
        assert not torch.cuda.is_initialized()
    finally:
        components.close()
    assert not active(owner)
    with pytest.raises(ValueError, match='expired'):
        policy.run(workload, output)


@pytest.mark.parametrize('bad', ['budget', 'geometry', 'dtype', 'source'])
def test_native_preflight_rejection_precedes_model_construction(native_config, monkeypatch, bad):
    import torch
    family, config = native_config
    raw = config.to_dict()
    if bad == 'budget': raw['execution']['resident_budget']['construction_limit_bytes'] = 1
    elif bad == 'geometry':
        settings = raw['model']['parameters']['settings']
        if family == 'self_forcing': settings['image_or_video_shape'] = [1, 21, 16, 8, 8]
        else: settings['latent_resolution'] = 16
    elif bad == 'source': raw['model']['parameters']['source_root'] = '/missing-native-source'
    original, identity = session.load_callable(config.model['factory'])
    def never(_): raise AssertionError('weights/model construction occurred before rejection')
    never.admit_construction = original.admit_construction
    monkeypatch.setattr(session, 'load_callable', lambda ref: (never, identity))
    config = TrainingConfig.from_dict(raw, source=config.source)
    old = torch.get_default_dtype()
    try:
        if bad == 'dtype': torch.set_default_dtype(torch.float64)
        with pytest.raises((ValueError, FileNotFoundError)):
            session._construct_model(config)
    finally:
        torch.set_default_dtype(old)


def test_native_public_manual_fresh_process_checkpoint_resume(native_config, tmp_path):
    import torch
    from test_training_end_to_end import equal, run_cli
    family, config = native_config
    raw = config.to_dict()
    raw['execution'].update(mode='manual', implementation=implementation(family))
    path = tmp_path / 'public.json'; path.write_text(json.dumps(raw))
    partial = run_cli('train', path, '--cycles', 1, '--output', tmp_path / 'partial')
    resumed = run_cli('train', path, '--cycles', 3, '--resume', partial['checkpoint'], '--output', tmp_path / 'resumed')
    full = run_cli('train', path, '--cycles', 3, '--output', tmp_path / 'full')
    state = lambda result: torch.load(Path(result['checkpoint']) / 'state.pt', weights_only=True)
    assert equal(state(resumed), state(full))
    assert resumed['cursor'] == full['cursor'] == 3 and resumed['cycles_this_process'] == 2
    metrics = [json.loads(row) for row in (tmp_path / 'resumed/metrics.jsonl').read_text().splitlines()]
    assert metrics[0]['execution']['owner_resume_reference_admission']
    assert metrics[0]['execution']['resident_trial']['adopted']
    assert metrics[0]['sample_id'] != metrics[1]['sample_id']
    report = json.loads((tmp_path / 'resumed/resume-owner-admission.json').read_text())
    assert report['resumed'] and report['resident_trial']['adopted']
    # Reuse this completed training: export the current Student and execute the
    # existing decoder, rather than launching another training-only proof.
    exported = run_cli('export', path, '--checkpoint', resumed['checkpoint'], '--output', tmp_path / 'export')
    assert exported['checkpoint_cycle'] == 3
    verify_current_export(family, path, resumed['checkpoint'], tmp_path / 'export')


def verify_current_export(family, config_path, checkpoint, export_path):
    """Check completed training outputs without repeating any training steps."""
    import torch
    from test_training_end_to_end import equal
    from distillschedule.training.checkpoint import cpu_tree
    from distillschedule.training.session import _load_for_output, _generate_values
    from distillschedule.training.config import load_training_config
    config = load_training_config(config_path)
    trained, data, _, _, _ = _load_for_output(config, checkpoint)
    try:
        owner = trained.state['native_cycle']
        weights = torch.load(Path(export_path) / 'weights.pt', map_location='cpu', weights_only=True)
        generator = owner.model.generator if family == 'self_forcing' else owner.model.feedforward_model
        assert equal(weights['generator'], cpu_tree(generator.state_dict()))
        stream = config.data['streams']['generator']['prepared']
        from distillschedule.data import PreparedDataset
        features = PreparedDataset(stream).sample(0)
        key = 'video' if family == 'self_forcing' else 'image'
        decoded = _generate_values(config, trained, features, {}, output_kind=key)['decoded']
        assert decoded[key].dtype == torch.uint8 and decoded[key].numel() > 0
    finally:
        trained.close()


def test_native_manual_publication_failure_is_not_a_commit(native_config, monkeypatch, tmp_path):
    from distillschedule.training.checkpoint import cpu_tree, rng_state
    from distillschedule.training.policy import CyclePolicy
    from distillschedule.open_workload import planning
    from test_training_end_to_end import equal
    family, config = native_config
    config = replace(config, execution=dict(config.execution, mode='manual', implementation=implementation(family)))
    components, data, algorithm, _ = session.construct(config)
    owner = components.state['native_cycle']; before = cpu_tree(owner.state_dict()); rng = rng_state()
    pointers = [p.data_ptr() for p in owner.model.parameters()]
    output = tmp_path / 'failed'; output.mkdir()
    workload = session.build_cycle(config, components, data, algorithm, cycle=0, cursor=0)
    policy = CyclePolicy(config.execution)
    def fail(path, value):
        assert owner.cycle == 1
        raise OSError('injected public publication')
    try:
        monkeypatch.setattr(planning, '_publish_resident_success', fail)
        with pytest.raises(OSError, match='injected public publication'):
            policy.run(workload, output)
        assert equal(cpu_tree(owner.state_dict()), before) and equal(rng_state(), rng)
        assert [p.data_ptr() for p in owner.model.parameters()] == pointers
        assert policy.bound is None and policy.session is None
        assert not (output / 'owner-admission.json').exists()
    finally:
        components.close()


@pytest.mark.parametrize('native_config', ['sd15'], indirect=True)
def test_continuation_checks_known_host_growth_before_update(native_config, monkeypatch):
    from distillschedule.training.resident import for_workload, ResidentTrainingBudget
    from distillschedule.open_workload import resident_trial
    from distillschedule.training.checkpoint import cpu_tree
    from test_training_end_to_end import equal
    _, config = native_config
    components, data, algorithm, _ = session.construct(config)
    owner = components.state['native_cycle']
    try:
        workload = session.build_cycle(config, components, data, algorithm, cycle=0, cursor=0)
        admission = for_workload(workload, ResidentTrainingBudget.parse(limits(), device='cpu'))
        adapter = owner._resident_trial_adapter()
        inventory = resident_trial._memory_inventory(adapter.state)
        trial = admission.budget.trial
        before = cpu_tree(owner.state_dict())
        # More than the bare reserve, but one byte short of actual native state,
        # Adam growth and scratch. The update must never be entered.
        available = inventory['host_storage_bytes'] + adapter.growth_bytes + trial.scratch_bytes + trial.reserve_bytes - 1
        monkeypatch.setattr(resident_trial, '_host_available', lambda: available)
        with pytest.raises(ValueError, match='host state/growth'):
            with admission.continuation():
                raise AssertionError('capacity refusal happened after starting the update')
        assert equal(before, cpu_tree(owner.state_dict())) and owner.cycle == 0
    finally:
        components.close()


@pytest.mark.parametrize('native_config', ['sd15'], indirect=True)
def test_public_auto_saved_plan_and_resume_never_search_again(native_config, tmp_path, monkeypatch):
    # One family suffices for the shared saved-plan binding; manual continuation
    # and native checkpoints are independently exercised above for all three.
    family, config = native_config
    import torch
    from test_training_end_to_end import run_cli, equal
    from distillschedule.open_workload import planning
    raw = config.to_dict()
    raw['execution'].update(mode='auto', memory_limit_bytes=2 << 30, max_candidates=2, samples=1, max_program_runs=3)
    path = tmp_path / 'auto.json'; path.write_text(json.dumps(raw))
    auto = run_cli('train', path, '--cycles', 1, '--output', tmp_path / 'auto')
    plan_path = tmp_path / 'auto/planning/selected-plan.json'
    planned = config.to_dict()
    planned['execution'].update(mode='plan', plan=str(plan_path))
    planned_path = tmp_path / 'plan.json'; planned_path.write_text(json.dumps(planned))
    partial = run_cli('train', planned_path, '--cycles', 1, '--output', tmp_path / 'saved')
    automatic = torch.load(Path(auto['checkpoint'])/'state.pt', weights_only=True)
    saved = torch.load(Path(partial['checkpoint'])/'state.pt', weights_only=True)
    assert automatic['policy']['mode'] == 'auto' and saved['policy']['mode'] == 'plan'
    assert equal({key: value for key, value in automatic.items() if key != 'policy'},
                 {key: value for key, value in saved.items() if key != 'policy'})
    assert equal({key: value for key, value in automatic['policy'].items() if key != 'mode'},
                 {key: value for key, value in saved['policy'].items() if key != 'mode'})
    resumed = run_cli('train', planned_path, '--cycles', 3, '--resume', partial['checkpoint'], '--output', tmp_path / 'resumed')
    full = run_cli('train', planned_path, '--cycles', 3, '--output', tmp_path / 'full')
    assert equal(torch.load(Path(resumed['checkpoint'])/'state.pt', weights_only=True),
                 torch.load(Path(full['checkpoint'])/'state.pt', weights_only=True))
    assert not (tmp_path / 'saved/planning').exists() and not (tmp_path / 'resumed/planning').exists()
    # A changed real saved plan is rejected, never replaced by a hand command.
    document = json.loads(plan_path.read_text()); document['selected']['order'] = ['bogus']
    plan_path.write_text(json.dumps(document))
    bad = TrainingConfig.from_dict(planned, source=planned_path)
    bad = replace(bad, trainer=dict(bad.trainer, cycles=1, output_dir=str(tmp_path / 'bad-plan')))
    def no_search(*args, **kwargs): raise AssertionError('saved-plan fallback searched')
    monkeypatch.setattr(planning, 'calibrate', no_search)
    with pytest.raises(ValueError, match='changed plan'):
        session.train(bad)
