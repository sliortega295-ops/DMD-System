"""Actual tiny native models, measured common planner; SYNTHETIC_TEST_ONLY."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('DMD_MODERN_FLOW_CPU') != '1',
    reason='requires opt-in isolated modern-flow native CPU stack')


@pytest.fixture(scope='module', params=['flux2-klein-base', 'qwen-image', 'wan22-ti2v-5b', 'hunyuan-video-1.5'])
def native_assets(request, tmp_path_factory):
    from distillschedule.training.plugins import load_callable
    repo = Path(__file__).resolve().parents[1]
    video_recipes = {'wan22-ti2v-5b': 'wan22_dmd', 'hunyuan-video-1.5': 'hunyuan_video_dmd'}
    image = request.param not in video_recipes
    recipe = 'flow_image' if image else video_recipes[request.param]
    build, _ = load_callable(str(repo/'recipes'/recipe/'create_cpu_fixture.py')+':build_fixture')
    options = {'family': request.param} if image else {}
    result = build(tmp_path_factory.mktemp(request.param)/'fixture', **options)
    # Native Hunyuan's crop1000 + glyph256 attention has a larger *tracked*
    # CPU fixture peak. This explicit test budget is not a production/RSS gate.
    return dict(result, family=json.loads(Path(result['assets']).read_text())['family'],
                planning_limit_bytes=200_000_000 if request.param == 'hunyuan-video-1.5' else 100_000_000)


def owned(assets):
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import construct
    config = load_training_config(assets['config'])
    return config, *construct(config)


def automatic(assets):
    return dict(mode='auto', memory_limit_bytes=assets.get('planning_limit_bytes', 100_000_000),
                max_candidates=4, samples=1, max_program_runs=5)


def context_key(feeds):
    """Image/video single text context or actual AV modality contexts."""
    return 'context' if 'context' in feeds else 'contexts'


def test_real_measured_policy_preserves_every_update_and_persistent_owner(native_assets, tmp_path, monkeypatch):
    import torch
    from distillschedule.open_workload.torch_backend import Session, resolve_module, sharing_inventory
    from distillschedule.training.checkpoint import pack_state, rng_state
    from distillschedule.training.policy import CyclePolicy
    from distillschedule.training.session import build_cycle
    from test_training_end_to_end import equal

    # Read-only gradient observers run on the actual candidate/selected objects,
    # not the originals that the paired native path correctly deep-copies.
    observed = {}; runs = []
    original_run = Session.run
    def capture(session, *args, **kwargs):
        gradients = {role: {} for role in ('student', 'fake')}
        handles = []
        for role, values in gradients.items():
            for name, parameter in session.state[role].named_parameters():
                if parameter.requires_grad:
                    values[name] = []
                    handles.append(parameter.register_hook(
                        lambda grad, values=values, name=name: values[name].append(grad.detach().clone())))
        try:
            result = original_run(session, *args, **kwargs)
            observed[id(session)] = gradients
            runs.append(dict(implementation=deepcopy(session.implementation), measured=kwargs.get('measure', False)))
            return result
        finally:
            for handle in handles: handle.remove()
    monkeypatch.setattr(Session, 'run', capture)

    def cycles(execution, output):
        config, components, dataset, algorithm, _ = owned(native_assets)
        run_offset = len(runs)
        policy = CyclePolicy(execution); versions = None; identity = None; rows = []
        initial_teacher = deepcopy(components.state['teacher'].state_dict())
        pair = None
        for cycle in range(3):
            workload = build_cycle(config, components, dataset, algorithm, cycle=cycle, cursor=cycle, versions=versions)
            if pair is None: pair = sharing_inventory(workload)[0]
            session, outputs = policy.run(workload, output)
            components.state.update(session.state)
            versions = session.receipt['dispatcher']['final_versions']
            owners = (id(session), {key: id(value) for key, value in session.state.items()},
                {role: [id(p) for p in session.state[role].parameters()] for role in ('student', 'fake')},
                {role: [id(p) for group in session.state[role+'_optimizer'].param_groups for p in group['params']]
                 for role in ('student', 'fake')})
            if identity is None: identity = owners
            assert owners == identity, 'later cycles must keep the very same Session/modules/Adam owners'
            assert equal(initial_teacher, session.state['teacher'].state_dict())
            assert session.state['dmd_counters'].tolist() == [cycle+1, cycle+1, (cycle+1)*5]
            for role, count in (('student', cycle+1), ('fake', (cycle+1)*5)):
                parameters = {id(p) for p in session.state[role].parameters() if p.requires_grad}
                optimizer = session.state[role+'_optimizer']
                assert parameters == {id(p) for group in optimizer.param_groups for p in group['params']}
                assert all(float(value['step']) == count for value in optimizer.state.values())
                assert session.state[role+'_scheduler'].last_epoch == count
                assert all(p.grad is None for p in session.state[role].parameters())
                assert any(g.any() for values in observed[id(session)][role].values() for g in values)
            pairs = sharing_inventory(session.workload, include_existing=True)
            aliases = sum(resolve_module(session.state, a[0]).get_parameter(a[1]) is
                          resolve_module(session.state, b[0]).get_parameter(b[1]) for a, b, _ in pairs)
            implementation = policy.bound['implementation']
            assert bool(session.effects['shared_parameter_aliases']) == implementation['share_immutable']
            a, b, _ = pair
            source = resolve_module(session.state, a[0]).get_parameter(a[1])
            target = resolve_module(session.state, b[0]).get_parameter(b[1])
            assert (source is target) == implementation['share_immutable']
            assert (source.data_ptr() == target.data_ptr()) == implementation['share_immutable']
            assert bool(session.effects['checkpoint_calls']) == bool(implementation['checkpoint_bindings'])
            rows.append(dict(state=pack_state(session.state), outputs={k:v.detach().clone() for k,v in outputs.items()},
                gradients=observed[id(session)], rng=rng_state(device='cpu'), versions=versions,
                sample_id=dataset.sample_id(cycle), context=deepcopy(workload.feeds[context_key(workload.feeds)]),
                effects=deepcopy(session.effects), aliases=aliases, storage=session.initial_storage,
                storage_witness=dict(first=list(a), second=list(b), same_parameter=source is target,
                    source_data_ptr=source.data_ptr(), target_data_ptr=target.data_ptr()),
                program_runs=len(runs)-run_offset))
        assert len({row['sample_id'] for row in rows}) == 3
        assert not equal(rows[0]['context'], rows[2]['context']), 'later cycles need genuinely new text features'
        return rows, policy

    reference, _ = cycles(dict(mode='reference'), tmp_path/'reference')
    measured, policy = cycles(automatic(native_assets), tmp_path/'auto')
    plan_path = tmp_path/'auto/planning/selected-plan.json'
    document = json.loads(plan_path.read_text())
    calibration = json.loads((plan_path.parent/'calibration/calibration.json').read_text())
    saved, saved_policy = cycles(dict(mode='plan', plan=str(plan_path)), tmp_path/'saved')
    for left, right in ((reference, measured), (reference, saved)):
        for a, b in zip(left, right):
            for key in ('state', 'outputs', 'gradients', 'rng', 'versions', 'sample_id', 'context'):
                assert equal(a[key], b[key]), key
    assert policy.bound['initial_reference_gate']['status'] == 'PASS'
    assert saved_policy.bound['plan_sha256'] == policy.bound['plan_sha256'] == document['plan_sha256']
    count = len(calibration['inventory']['candidates'])
    assert calibration['budget'] == dict(maximum=5, actual=1+count)
    assert count == len(calibration['variants']) == 2 and calibration['inventory']['exhaustive']
    assert not calibration['failed']
    assert calibration['hardware_probe']['status'] == 'NOT_REQUESTED'
    assert not calibration['claimable_speedup'] and calibration['gpu_execution'] == 'NOT_RUN'
    assert all(row['reference_gate']['status'] == 'PASS' for row in calibration['variants'])
    assert (reference[-1]['program_runs'], measured[-1]['program_runs'], saved[-1]['program_runs']) == (3, 7, 4)
    for variant in calibration['variants']:
        costs = json.loads(Path(variant['cost_model']).read_text())
        assert costs['records']
    (tmp_path/'policy-proof.json').write_text(json.dumps(dict(evidence_kind='SYNTHETIC_TEST_ONLY',
        family=native_assets['family'], calibration_runs=calibration['budget']['actual'], selected_admission_runs=2, formal_cycles=3,
        selected_admission_includes_first_formal_cycle=True,
        total_program_runs=dict(reference=3, auto=measured[-1]['program_runs'], saved_plan=saved[-1]['program_runs']),
        selected=document['selected']['implementation'], selected_plan_sha256=document['plan_sha256'],
        cycle_effects=[row['effects'] for row in measured], alias_counts=[row['aliases'] for row in measured],
        storage_witnesses=[row['storage_witness'] for row in measured],
        selected_initial_tracked_storage_bytes=measured[0]['storage'],
        exact_loss_gradient_parameter_adam_scheduler_ema_rng_data_clocks=True,
        capacity_scope='selected tracked tensors only; not construction/clone staging or host RSS',
        pretrained_execution='NOT_RUN', gpu_execution='NOT_RUN', quality_validation='NOT_RUN'), indent=2)+'\n')
    assert not torch.cuda.is_initialized()


def test_public_auto_saved_plan_and_new_process_resume(native_assets, tmp_path):
    import sys
    import torch
    from test_training_end_to_end import equal, run_cli
    commands = []
    def cli(*args):
        result = run_cli(*args)
        commands.append(dict(argv=[sys.executable, '-m', 'distillschedule', *map(str, args)],
            status='EXIT_0_JSON_ACCEPTED_BY_REAL_SUBPROCESS_HELPER',
            result={key:result[key] for key in ('status','checkpoint','completed_cycles','cycles_this_process',
                'resume_cycle','cursor','final_versions','numerical_status','gpu_execution')}))
        return result
    config = json.loads(Path(native_assets['config']).read_text())
    config['execution'] = automatic(native_assets)
    auto_config = tmp_path/'auto.json'; auto_config.write_text(json.dumps(config))
    first = cli('train', auto_config, '--cycles', 1, '--output', tmp_path/'auto-first')
    continued = cli('train', auto_config, '--cycles', 3, '--resume', first['checkpoint'], '--output', tmp_path/'auto-resumed')
    plan_path = tmp_path/'auto-first/planning/selected-plan.json'
    config['execution'] = dict(mode='plan', plan=str(plan_path))
    saved_config = tmp_path/'saved.json'; saved_config.write_text(json.dumps(config))
    whole = cli('train', saved_config, '--cycles', 3, '--output', tmp_path/'saved-whole')
    saved_first = cli('train', saved_config, '--cycles', 1, '--output', tmp_path/'saved-first')
    saved_resume = cli('train', saved_config, '--cycles', 3, '--resume', saved_first['checkpoint'], '--output', tmp_path/'saved-resumed')
    reference = cli('train', native_assets['config'], '--cycles', 3, '--output', tmp_path/'reference')
    def payload(result):
        return torch.load(Path(result['checkpoint'])/'state.pt', weights_only=True)
    expected = payload(reference)
    for result in (continued, whole, saved_resume):
        actual = payload(result)
        for key in ('state', 'rng', 'cycle', 'cursor', 'versions'):
            assert equal(actual[key], expected[key]), key
        assert actual['state']['dmd_counters']['value']['counters'].tolist() == [3,3,15]
        assert actual['cycle'] == actual['cursor'] == 3
        assert result['final_versions']['student'] == 3 and result['final_versions']['fake'] == 15
    assert equal(payload(whole), payload(saved_resume)), 'same saved policy must restart exactly, not only parameters'
    assert equal(payload(first)['policy'], payload(continued)['policy']), 'auto resume must reuse, not recalibrate'
    assert not (tmp_path/'auto-resumed/planning').exists() and not (tmp_path/'saved-resumed/planning').exists()
    rows = [json.loads(row) for row in (tmp_path/'auto-resumed/metrics.jsonl').read_text().splitlines()]
    assert [row['sample_index'] for row in rows] == [1,2]
    assert len({row['sample_id'] for row in rows}) == 2
    for row in rows:
        assert row['execution']['dispatcher']['executed_operations'] == json.loads(plan_path.read_text())['selected']['order']
    (tmp_path/'cli-proof.json').write_text(json.dumps(dict(evidence_kind='SYNTHETIC_TEST_ONLY',
        family=native_assets['family'], fresh_process_commands=6,
        commands=commands,
        calibration_runs=json.loads((plan_path.parent/'calibration/calibration.json').read_text())['budget']['actual'],
        exact_reference_auto_saved_resume_state_rng_data_versions=True, auto_resume_recalibrated=False,
        selected=continued['policy']['implementation'], samples=[row['sample_id'] for row in rows],
        gpu_execution='NOT_RUN', pretrained_execution='NOT_RUN', quality_validation='NOT_RUN'), indent=2)+'\n')
    assert not torch.cuda.is_initialized()


def test_trial_budget_refusal_and_trial_exception_preserve_live_native_state(native_assets, tmp_path, monkeypatch):
    import torch
    from distillschedule.open_workload.planning import calibrate
    from distillschedule.open_workload.torch_backend import Session
    from distillschedule.training.checkpoint import pack_state, rng_state
    from distillschedule.training.session import build_cycle
    from test_training_end_to_end import equal
    config, components, dataset, algorithm, _ = owned(native_assets)
    workload = build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
    before = dict(state=pack_state(components.state), rng=rng_state(device='cpu'),
                  feeds=deepcopy(workload.feeds), identity=workload.identity,
                  samples=[dataset.sample_id(i) for i in range(3)])
    with pytest.raises(ValueError, match='budget'):
        calibrate(workload, tmp_path/'budget', samples=1, max_candidates=4, max_program_runs=1, profile_hardware=False)
    assert not (tmp_path/'budget').exists()
    original = Session.run
    count = 0
    def failing_trial(session, *args, **kwargs):
        nonlocal count
        count += 1
        result = original(session, *args, **kwargs)
        if count == 2:
            raise RuntimeError('SYNTHETIC_TEST_ONLY failure after actual candidate update')
        return result
    monkeypatch.setattr(Session, 'run', failing_trial)
    with pytest.raises(RuntimeError, match='actual candidate update'):
        calibrate(workload, tmp_path/'failure', samples=1, max_candidates=2, max_program_runs=3, profile_hardware=False)
    assert json.loads((tmp_path/'failure/failure.json').read_text())['program_runs'] == 2
    after = dict(state=pack_state(components.state), rng=rng_state(device='cpu'),
                 feeds=workload.feeds, identity=workload.identity,
                 samples=[dataset.sample_id(i) for i in range(3)])
    assert equal(before, after) and components.state['dmd_counters'].tolist() == [0,0,0]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('layout', ['bf16', 'meta'])
def test_fixture_label_does_not_admit_unsupported_native_clone_layout(native_assets, monkeypatch, layout):
    import torch
    _, components, _, _, _ = owned(native_assets)
    role = components.state['student']
    role.bfloat16() if layout == 'bf16' else role.to('meta')
    def forbidden(*args, **kwargs): raise AssertionError('unsupported native tensors reached model graph copy')
    monkeypatch.setattr(type(role.native), '__deepcopy__', forbidden, raising=False)
    with pytest.raises(ValueError, match='CPU1 FP32'):
        deepcopy(role)
    assert not torch.cuda.is_initialized()


def test_qualified_clone_preserves_parameter_optimizer_aliases_and_native_module_state(native_assets):
    from distillschedule.training.checkpoint import pack_state
    from test_training_end_to_end import equal
    _, components, _, _, _ = owned(native_assets)
    state = components.state
    token = object(); state['student']._compiled_call_impl = token
    copied = deepcopy(state)
    assert state['student']._compiled_call_impl is token
    assert copied['student']._compiled_call_impl is None, 'native Module getstate excludes compiled call implementation'
    assert equal(pack_state(state), pack_state(copied))
    for role in ('student', 'fake'):
        assert copied[role] is not state[role]
        parameters = {id(p) for p in copied[role].parameters() if p.requires_grad}
        assert parameters == {id(p) for group in copied[role+'_optimizer'].param_groups for p in group['params']}
        assert not parameters & {id(p) for p in state[role].parameters()}
        assert copied[role+'_scheduler'].optimizer is copied[role+'_optimizer']
        original = list(state[role].named_parameters(remove_duplicate=False))
        cloned = list(copied[role].named_parameters(remove_duplicate=False))
        assert [name for name,_ in original] == [name for name,_ in cloned]
        assert [[i for i,(_,p) in enumerate(original) if p is value] for _,value in original] == [
            [i for i,(_,p) in enumerate(cloned) if p is value] for _,value in cloned]


@pytest.fixture(scope='module')
def native_plan(native_assets, tmp_path_factory):
    return measured_plan(native_assets, tmp_path_factory.mktemp('modern-native-plan'))


def measured_plan(native_assets, root):
    from distillschedule.open_workload.planning import calibrate, plan
    from distillschedule.training.session import build_cycle
    config, components, dataset, algorithm, _ = owned(native_assets)
    workload = build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
    calibrate(workload, root/'calibration', max_candidates=2, samples=1, max_program_runs=3, profile_hardware=False)
    path = root/'plan.json'
    path.write_text(json.dumps(plan(root/'calibration/calibration.json', memory_limit_bytes=native_assets['planning_limit_bytes'])))
    return path


@pytest.mark.parametrize('field', ['source', 'shape', 'dtype', 'model-parameter', 'algorithm', 'hardware',
                                  'cache', 'plan-hash', 'unmeasured'])
def test_saved_native_plan_rejects_drift_before_any_update(native_assets, native_plan, tmp_path, monkeypatch, field):
    from dataclasses import replace
    import torch
    from distillschedule.open_workload import planning
    from distillschedule.open_workload.torch_backend import Session
    from distillschedule.training.checkpoint import pack_state, rng_state
    from distillschedule.training.session import build_cycle
    from distillschedule.workloads.best_known.registry import fingerprint
    from test_training_end_to_end import equal
    config, components, dataset, algorithm, _ = owned(native_assets)
    workload = build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
    path = native_plan
    if field == 'source': monkeypatch.setattr(planning, 'source_identity', lambda: '0'*64)
    elif field == 'shape':
        modality = next(iter(workload.feeds['latents']))
        workload.feeds['latents'][modality] = workload.feeds['latents'][modality][..., :-1]
    elif field == 'dtype':
        key = context_key(workload.feeds)
        value = workload.feeds[key]
        workload.feeds[key] = {k:v.double() for k,v in value.items()} if isinstance(value, dict) else value.double()
    elif field == 'model-parameter':
        with torch.no_grad(): next(components.state['student'].parameters()).add_(.01)
    elif field == 'algorithm':
        workload.program = replace(workload.program, parameters={**workload.program.parameters, 'semantic_drift': True})
    elif field == 'hardware':
        original = planning.hardware_identity
        monkeypatch.setattr(planning, 'hardware_identity', lambda device: {**original(device), 'threads': 999})
    elif field == 'cache': workload.feeds['kv_cache'] = torch.zeros(1)
    else:
        document = json.loads(native_plan.read_text())
        document['selected']['implementation']['backend'] = 'unmeasured-native-backend'
        if field == 'unmeasured':
            document.pop('plan_sha256'); document['plan_sha256'] = fingerprint(document)
        path = tmp_path/'changed-plan.json'; path.write_text(json.dumps(document))
    before = dict(state=pack_state(components.state), rng=rng_state(device='cpu'), feeds=deepcopy(workload.feeds))
    def forbidden(*args, **kwargs): raise AssertionError('drift reached a reference/formal update')
    monkeypatch.setattr(Session, 'run', forbidden)
    with pytest.raises(ValueError): planning.run(workload, path, tmp_path/'rejected')
    after = dict(state=pack_state(components.state), rng=rng_state(device='cpu'), feeds=workload.feeds)
    assert equal(before, after) and not (tmp_path/'rejected').exists()


def assert_unqualified_trials_refused(assets, tmp_path, monkeypatch):
    """Real admitted native state; a trust label is NOT proof of memory safety."""
    import shutil
    import torch
    from distillschedule.open_workload.planning import calibrate
    from distillschedule.open_workload.torch_backend import Session
    from distillschedule.training.checkpoint import pack_state, rng_state
    from distillschedule.training.config import TrainingConfig, load_training_config
    from distillschedule.training.policy import CyclePolicy
    from distillschedule.training.session import build_cycle, construct
    from test_training_end_to_end import equal
    root = tmp_path/'unqualified-assets'; shutil.copytree(assets['root'], root)
    manifest = json.loads((root/'assets.json').read_text())
    manifest['evidence_kind'] = 'LOCAL_PRETRAINED_UNVERIFIED'
    (root/'assets.json').write_text(json.dumps(manifest))
    raw = load_training_config(assets['config']).to_dict()
    raw['model']['parameters']['assets'] = str(root/'assets.json')
    config = TrainingConfig.from_dict(raw, source=assets['config'])
    components, dataset, algorithm, _ = construct(config)  # non-clone reference remains supported
    workload = build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
    before = dict(state=pack_state(components.state), rng=rng_state(device='cpu'),
                  identity=workload.identity, feeds=deepcopy(workload.feeds),
                  samples=[dataset.sample_id(i) for i in range(3)])
    def forbidden(*args, **kwargs):
        raise AssertionError('native role graph was copied or executed before unsupported trial refusal')
    with monkeypatch.context() as patch:
        patch.setattr(Session, 'run', forbidden)
        patch.setattr(type(components.state['student'].native), '__deepcopy__', forbidden, raising=False)
        with pytest.raises(ValueError, match='pretrained.*clone trials.*not admitted'):
            calibrate(workload, tmp_path/'direct', samples=1, max_candidates=1, max_program_runs=2, profile_hardware=False)
        with pytest.raises(ValueError, match='pretrained.*clone trials.*not admitted'):
            CyclePolicy(automatic(assets)).run(workload, tmp_path/'public-auto')
    after = dict(state=pack_state(components.state), rng=rng_state(device='cpu'),
                 identity=workload.identity, feeds=workload.feeds,
                 samples=[dataset.sample_id(i) for i in range(3)])
    assert equal(before, after)
    # Actual parameter-budget-admitted non-clone reference/manual update still
    # works; do not claim the relabeled tiny fixture contains pretrained weights.
    for mode in ('reference', 'manual'):
        model, data, algorithm, _ = construct(config)
        current = build_cycle(config, model, data, algorithm, cycle=0, cursor=0)
        execution = dict(mode=mode)
        if mode == 'manual':
            execution['implementation'] = dict(backend='torch-callable-resident-v1', share_immutable=True, checkpoint_bindings=[])
        session, _ = CyclePolicy(execution).run(current, tmp_path/mode)
        assert session.state['dmd_counters'].tolist() == [1,1,5]
    assert not torch.cuda.is_initialized()


def test_unqualified_pretrained_trials_refused_before_copy(native_assets, tmp_path, monkeypatch):
    assert_unqualified_trials_refused(native_assets, tmp_path, monkeypatch)
