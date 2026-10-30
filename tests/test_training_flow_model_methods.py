"""The real flow-entry seam, not qualification of a new sharding backend.

Local tests exercise the boundary without native optional dependencies. Tests
named ``native`` reuse the actual tiny FLUX/LTX fixtures in their pinned lane:
DMD_MODERN_FLOW_CPU=1 LTX_SOURCE_ROOT=... pytest ... -k native
These fixtures are synthetic weights, never a pretrained/22B/GPU claim.
"""
from copy import deepcopy
from dataclasses import replace
import inspect
import os
from pathlib import Path
import random

import pytest
import torch

from distillschedule.open_workload.replica_contract import module_method_binding
from distillschedule.open_workload.torch_backend import Context, Session, random_scope
from distillschedule.training import dmd
from distillschedule.training.adapters import ModelComponents
from distillschedule.training.checkpoint import cpu_tree, pack_state, rng_state
from distillschedule.training.prediction import PredictionSpec
from test_training_dmd_flow import OtherFlow, components, policy, _change_clock
from test_training_end_to_end import equal


REFERENCE = dict(backend='torch-callable-resident-v1', share_immutable=False, checkpoint_bindings=[])
FORWARDS = dict(student_forward='student', teacher_forward='teacher', fake_forward='fake')


def _owned_components(*, ema=True):
    class OwnedFlow(dmd.FlowDMDMethods, OtherFlow):
        pass
    roles = [OwnedFlow(value) for value in (.2, .3, .6)]
    roles[2].requires_grad_(False)
    return ModelComponents(state=dmd.training_state(*roles, student_lr=.01, fake_lr=.02,
        scheduler_gamma=.95, ema_decay=.9 if ema else None), prediction=PredictionSpec('flow'),
        capabilities=('dmd_flow_v1',)).validate()


def _feeds():
    return dict(latent=torch.zeros(1, 3), condition=torch.full((1, 3), .15),
                null_condition=torch.full((1, 3), -.25))


@pytest.mark.parametrize('admission', ['saved-plan', 'bound-policy'])
def test_extracted_helper_source_drift_rejected_by_real_policy(tmp_path, monkeypatch, admission):
    from distillschedule.open_workload.planning import calibrate, plan, run
    from distillschedule.training.policy import CyclePolicy
    from distillschedule.workloads.best_known.registry import write_new_json
    source = tmp_path / 'algorithm.py'
    source.write_text(Path(dmd.__file__).read_text())
    monkeypatch.setattr(dmd, '__file__', str(source))
    model = _owned_components()
    before = dmd.build_algorithm(model, _feeds(), policy(), 0, 91)
    if admission == 'saved-plan':
        calibrate(before, tmp_path/'costs', samples=1, max_candidates=1, max_program_runs=2)
        document = plan(tmp_path/'costs/calibration.json', memory_limit_bytes=10**8)
        write_new_json(tmp_path/'plan.json', document)
        assert run(before, tmp_path/'plan.json', tmp_path/'valid')[0]['reference_gate']['status'] == 'PASS'
    else:
        retained = CyclePolicy({'mode': 'reference'})
        session, _ = retained.run(before, tmp_path/'first')
    # Change only a helper body in the source identity, not the registered
    # wrapper's source or live tensors. Admission must precede execution.
    text = source.read_text()
    old = "    scale = policy['teacher_cfg']\n"
    assert text.count(old) == 1
    source.write_text(text.replace(old, "    scale = policy['teacher_cfg'] + 1.\n"))
    after = dmd.build_algorithm(model if admission == 'bound-policy' else _owned_components(),
                                _feeds(), policy(), int(admission == 'bound-policy'), 91)
    assert before.identity['bindings'] == after.identity['bindings']
    if admission == 'bound-policy':
        after.program = replace(after.program, initial_versions=session.receipt['dispatcher']['final_versions'])
    snapshot = pack_state(after.state)
    with pytest.raises(ValueError, match='code or state changed|structure changed'):
        if admission == 'saved-plan':
            run(after, tmp_path/'plan.json', tmp_path/'rejected')
        else:
            retained.run(after, tmp_path/'rejected')
    assert equal(snapshot, pack_state(after.state))


def _closure(graph, ids):
    parents = {op.operation_id: set() for op in graph.operations}
    for edge in graph.dependencies:
        parents[edge.target].add(edge.source)
    def ancestors(key):
        result, pending = set(), list(parents[key])
        while pending:
            current = pending.pop()
            if current not in result:
                result.add(current); pending.extend(parents[current])
        return result & ids
    return {key: ancestors(key) for key in ids}


def test_real_role_classes_explicitly_own_one_outer_flow_entry():
    from distillschedule.training.flow_image import ImageRole
    from distillschedule.training.ltx import LTXRole
    assert hasattr(dmd, 'FlowDMDMethods'), 'model-owned flow entry has not been implemented'
    for role in (ImageRole, LTXRole):
        assert issubclass(role, dmd.FlowDMDMethods)
        assert role.dmd_forward is dmd.FlowDMDMethods.dmd_forward
    # A method registration is not a public FSDP capability or a loader change.
    from distillschedule.training.catalog import resolve
    for family in ('flux2-klein-base', 'qwen-image', 'ltx25'):
        assert 'registered_methods' not in str(resolve('model', family))


@pytest.mark.parametrize('updates', [1, 5, 32])
@pytest.mark.parametrize('sample_policy', ['reuse_student', 'fresh_student'])
@pytest.mark.parametrize('ema', [False, True])
def test_lowering_preserves_original_versions_dependencies_and_random_call_ids(updates, sample_policy, ema):
    settings = policy(); settings.update(fake_updates=updates, fake_sample_policy=sample_policy)
    old = components()
    if not ema:
        old.state.pop('student_ema'); old.state['dmd_lifecycle']['ema_decay'] = None
    before = dmd.build_algorithm(old, _feeds(), settings, 0, 91)
    after = dmd.build_algorithm(_owned_components(ema=ema), _feeds(), settings, 0, 91)
    old_calls = {c.id: c for c in before.program.calls}
    new_calls = {c.id: c for c in after.program.calls}
    assert before.program.outputs == after.program.outputs
    assert old_calls.keys() <= new_calls.keys()
    prepare = [c for c in after.program.calls if c.binding == 'prepare_flow_forward']
    assert len(prepare) == 3 + updates + (updates if sample_policy == 'fresh_student' else 0)
    for key, old_call in old_calls.items():
        new_call = new_calls[key]
        if old_call.binding in FORWARDS:
            ready = new_calls[key + '.prepare']
            assert ready.inputs == old_call.inputs and ready.after == old_call.after
            assert ready.grad == 'no_grad' and not ready.writes and not ready.replay_safe
            assert ready.reads == tuple(name for name in old_call.reads if name not in FORWARDS.values())
            assert ready.attributes == dict(old_call.attributes, flow_role=FORWARDS[old_call.binding])
            assert new_call.inputs == (ready.id,)
            assert new_call.attributes == {key: value for key, value in ready.attributes.items()
                if key in ('flow_role', 'phase', 'cycle', 'fake_index', 'flow_policy')}
            assert replace(new_call, inputs=old_call.inputs, attributes=old_call.attributes) == old_call
            expected = module_method_binding(FORWARDS[old_call.binding], 'dmd_forward')
            actual = after.bindings[old_call.binding]
            assert actual.__code__ is expected.__code__
            assert inspect.getclosurevars(actual).nonlocals == inspect.getclosurevars(expected).nonlocals
        elif key == 'dmd.commit_cycle':
            assert set(new_call.after) == set(new_calls) - {key}
            assert replace(new_call, after=old_call.after) == old_call
        else:
            assert new_call == old_call
            assert after.bindings[new_call.binding] is before.bindings[old_call.binding]
    old_graph, new_graph = before.program.compile(), after.program.compile()
    old_ops, new_ops = ({op.operation_id: op for op in graph.operations} for graph in (old_graph, new_graph))
    for key in old_ops:
        assert old_ops[key].required_versions == new_ops[key].required_versions
        assert old_ops[key].produced_versions == new_ops[key].produced_versions
    assert old_graph.metadata['final_versions'] == new_graph.metadata['final_versions']
    assert _closure(old_graph, set(old_calls)) == _closure(new_graph, set(old_calls))
    assert after.replica_semantics is None


def test_partial_optin_and_a_same_named_third_party_method_keep_legacy_wrappers(monkeypatch):
    monkeypatch.setattr(OtherFlow, 'dmd_forward', lambda *a, **k: pytest.fail('not an explicit opt-in'), raising=False)
    legacy = components()
    owned = _owned_components()
    for model in (legacy, replace(owned, state={**owned.state, 'teacher': legacy.state['teacher']})):
        workload = dmd.build_algorithm(model, _feeds(), policy(), 0, 19)
        assert not any(c.id.endswith('.prepare') for c in workload.program.calls)
        assert workload.bindings['student_forward'] is dmd._student
        Session(workload, REFERENCE, copy_state=False).run()
        assert model.state['dmd_counters'].tolist() == [1, 1, 2]


def test_prepare_is_rng_free_reference_only_and_binding_resolves_current_session_owner(monkeypatch):
    model = _owned_components(); feeds = _feeds()
    before = rng_state()
    objects = {key: id(value) for key, value in model.state.items()}
    parameters = {role: tuple(id(p) for p in model.state[role].parameters()) for role in FORWARDS.values()}
    workload = dmd.build_algorithm(model, feeds, policy(), 0, 51)
    session = Session(workload, REFERENCE, copy_state=False)
    assert equal(before, rng_state())
    call = workload.program.calls[0]
    assert call.id == 'dmd.c0.student_forward.prepare'
    context = Context(feeds, session.state, call.attributes)
    def no_draw(*args, **kwargs):
        pytest.fail('pure preparation must not draw random numbers')
    with monkeypatch.context() as patch:
        for key in ('rand', 'randn', 'randn_like'):
            patch.setattr(torch, key, no_draw)
        patch.setattr(random, 'random', no_draw)
        payload = workload.bindings[call.binding](context)
    assert equal(before, rng_state())
    assert all(payload['feeds'][key] is value for key, value in feeds.items())
    assert objects == {key: id(value) for key, value in model.state.items()}
    assert parameters == {role: tuple(id(p) for p in session.state[role].parameters()) for role in FORWARDS.values()}
    forward = workload.program.calls[1]
    current = session.state['fake']
    session.state['student'] = current  # no update: exercise the actual binding's late owner lookup
    ctx = Context({call.id: payload}, {'student': session.state['student']}, forward.attributes)
    with random_scope(workload.seed, forward.id, session.device):
        actual = workload.bindings[forward.binding](ctx)
    with random_scope(workload.seed, forward.id, session.device):
        expected = current.rollout(feeds, policy()['rollout_sigmas'], policy()['gradient_policy'])
    assert equal(actual, expected)
    assert equal(before, rng_state())


@pytest.mark.parametrize('fault', ['cycle_counter', 'student_counter', 'fake_counter', 'student_last_epoch', 'fake_step_count'])
def test_model_owned_clock_fault_rejected_before_any_forward_or_update(monkeypatch, fault):
    model = _owned_components()
    workload = dmd.build_algorithm(model, _feeds(), policy(), 0, 3)
    def forbidden(*a, **k):
        pytest.fail('clock fault reached model computation or optimizer update')
    monkeypatch.setattr(dmd.FlowDMDMethods, 'dmd_forward', forbidden)
    for role in ('student', 'fake'):
        monkeypatch.setattr(model.state[role + '_optimizer'], 'step', forbidden)
    _change_clock(model.state, {}, fault)
    before = pack_state(model.state)
    with pytest.raises(ValueError, match='clock|counter|scheduler'):
        Session(workload, REFERENCE, copy_state=False).run()
    assert equal(before, pack_state(model.state))


def _forward_id(binding, attributes):
    if binding == 'student_forward':
        return ('dmd.c0.student_forward' if attributes['phase'] == 'student'
                else f"dmd.c0.f{attributes['fake_index']}.rollout")
    if binding == 'teacher_forward':
        return 'dmd.c0.teacher'
    if binding == 'fake_forward':
        return ('dmd.c0.fake' if attributes['phase'] == 'student'
                else f"dmd.c0.f{attributes['fake_index']}.forward")
    return ('dmd.c0.noise' if attributes['phase'] == 'student'
            else f"dmd.c0.f{attributes['fake_index']}.noise")


def _gradients(state, role):
    return cpu_tree({name: p.grad for name, p in state[role].named_parameters() if p.requires_grad})


def _update_snapshot(state, role):
    return cpu_tree(dict(role=role, parameters=dict(state[role].named_parameters()),
        optimizer=state[role + '_optimizer'].state_dict(), scheduler=state[role + '_scheduler'].state_dict(),
        counters=state['dmd_counters'].state_dict(), ema=state.get('student_ema')))


def _sequential_single(state, feeds, settings, seed):
    """Direct native-method oracle for the fresh/CFG branches absent in the B1 oracle.

    Like flow_replica_oracle, this calls no Program, DMD wrapper or new method.
    Existing objectives and native AdamW are retained; ordering is independent.
    """
    p = dmd._policy(settings)
    student, fake, teacher = (state[name] for name in ('student', 'fake', 'teacher'))
    device = next(student.parameters()).device
    trace = dict(values={}, losses={}, gradients=[], updates=[])
    def expand(sigma, value):
        return sigma.reshape((-1,) + (1,) * (value.ndim - 1))
    def remember(key, value):
        trace['values'][key] = cpu_tree(value)
        return value
    def noised(sample, call):
        with torch.no_grad(), random_scope(seed, call, device):
            first = next(iter(sample.values()))
            sigma = torch.rand(first.shape[0], device=device, dtype=first.dtype)
            sigma = p['sigma_min'] + (p['sigma_max'] - p['sigma_min']) * sigma
            epsilon = {name: torch.randn_like(value) for name, value in sample.items()}
            noisy = {name: (1 - expand(sigma, value)) * value + expand(sigma, value) * epsilon[name]
                     for name, value in sample.items()}
            return remember(call, dict(sample=student.apply_conditioning(noisy, feeds), sigma=sigma,
                masks=student.loss_masks(feeds, sample), target={name: epsilon[name] - value for name, value in sample.items()}))
    def update(role, loss):
        loss.backward()
        row = dict(role=role, mean=_gradients(state, role))
        parameters = [p for p in state[role].parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(parameters, p['max_grad_norm'], error_if_nonfinite=True)
        row['clipped'] = _gradients(state, role); trace['gradients'].append(row)
        state[role + '_optimizer'].step()
        state[role + '_optimizer'].zero_grad(set_to_none=True)
        state[role + '_scheduler'].step()
        state['dmd_counters'][1 if role == 'student' else 2] += 1
        if role == 'student' and 'student_ema' in state:
            for name, parameter in student.named_parameters():
                state['student_ema'][name].lerp_(parameter.detach(), 1 - state['dmd_lifecycle']['ema_decay'])
        trace['updates'].append(_update_snapshot(state, role))
    with random_scope(seed, 'dmd.c0.student_forward', device):
        sample = remember('dmd.c0.student_forward', student.rollout(feeds, p['rollout_sigmas'], p['gradient_policy']))
    detached = {name: value.detach() for name, value in sample.items()}
    noise = noised(detached, 'dmd.c0.noise')
    with torch.no_grad(), random_scope(seed, 'dmd.c0.teacher', device):
        velocity = teacher.denoise(noise['sample'], noise['sigma'], feeds, conditional=True)
        if p['teacher_cfg'] != 1:
            unconditional = teacher.denoise(noise['sample'], noise['sigma'], feeds, conditional=False)
            velocity = {name: unconditional[name] + p['teacher_cfg'] * (value - unconditional[name])
                        for name, value in velocity.items()}
        teacher_x0 = remember('dmd.c0.teacher', {name: value - expand(noise['sigma'], value) * velocity[name]
            for name, value in noise['sample'].items()})
    with torch.no_grad(), random_scope(seed, 'dmd.c0.fake', device):
        velocity = fake.denoise(noise['sample'], noise['sigma'], feeds, conditional=True)
        fake_x0 = remember('dmd.c0.fake', {name: value - expand(noise['sigma'], value) * velocity[name]
            for name, value in noise['sample'].items()})
    loss = dmd.student_objective(sample, teacher_x0, fake_x0, noise['masks'], weights=p['modality_weights'],
        normalizer=p['normalizer'], epsilon=p['normalizer_epsilon'])
    trace['losses']['dmd.c0.student_loss'] = cpu_tree(loss)
    update('student', loss)
    for index in range(p['fake_updates']):
        prefix = f'dmd.c0.f{index}'
        current = detached
        if p['fake_sample_policy'] == 'fresh_student':
            with torch.no_grad(), random_scope(seed, prefix + '.rollout', device):
                current = remember(prefix + '.rollout', student.rollout(feeds, p['rollout_sigmas'], p['gradient_policy']))
        noise = noised(current, prefix + '.noise')
        with random_scope(seed, prefix + '.forward', device):
            prediction = remember(prefix + '.forward', fake.denoise(noise['sample'], noise['sigma'], feeds, conditional=True))
        loss = dmd.fake_objective(prediction, noise['target'], noise['masks'], weights=p['modality_weights'])
        trace['losses'][prefix + '.loss'] = cpu_tree(loss)
        update('fake', loss)
    state['dmd_counters'][0] += 1
    return trace


def _observed_session(workload):
    """Observe without replacing the actual forward binding with another owner."""
    trace = dict(values={}, losses={}, gradients=[], updates=[])
    session = Session(workload, REFERENCE, copy_state=False)
    originals = dict(workload.bindings)
    def observe(binding):
        def invoke(ctx):
            result = originals[binding](ctx)
            role = 'student' if ctx.attributes['phase'] == 'student' else 'fake'
            if binding in FORWARDS or binding == 'noise':
                trace['values'][_forward_id(binding, ctx.attributes)] = cpu_tree(result)
            elif binding.endswith('_backward'):
                trace['gradients'].append(dict(role=role, mean=_gradients(session.state, role)))
            elif binding.endswith('_update'):
                trace['updates'].append(_update_snapshot(session.state, role))
            return result
        return invoke
    workload.bindings = {key: observe(key) if key in FORWARDS or key == 'noise'
        or key.endswith(('_backward', '_update')) else value for key, value in originals.items()}
    handles = []
    for role in ('student', 'fake'):
        def before_step(optimizer, args, kwargs, role=role):
            assert trace['gradients'][-1]['role'] == role
            trace['gradients'][-1]['clipped'] = _gradients(session.state, role)
        handles.append(session.state[role + '_optimizer'].register_step_pre_hook(before_step))
    try:
        trace['losses'] = cpu_tree(session.run())
    finally:
        workload.bindings = originals
        for handle in handles: handle.remove()
    return trace, session


def _random_trace(monkeypatch, function):
    # The complete rand/randn_like sequence includes initial rollout noise and
    # epsilon, not only inferred targets. Preparing the added calls must add none.
    draws = []
    def observe(function):
        def invoke(*args, **kwargs):
            value = function(*args, **kwargs)
            draws.append(cpu_tree(value))
            return value
        return invoke
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'rand', observe(torch.rand))
        patch.setattr(torch, 'randn_like', observe(torch.randn_like))
        result = function()
    return result, draws


def _compare_oracle(make, settings, monkeypatch, *, feed_transform=lambda feeds: feeds):
    expected_model, feeds = make()
    feeds = feed_transform(feeds)
    # Bind the existing clock schema without using the new forward methods.
    dmd.build_algorithm(expected_model, feeds, settings, 0, 53)
    initial, before_rng = pack_state(expected_model.state), rng_state()
    expected, expected_random = _random_trace(monkeypatch,
        lambda: _sequential_single(expected_model.state, feeds, settings, 53))
    assert equal(before_rng, rng_state())
    expected_state = pack_state(expected_model.state)
    del expected_model
    actual_model, actual_feeds = make()
    actual_feeds = feed_transform(actual_feeds)
    assert equal(initial, pack_state(actual_model.state))
    assert equal(feeds, actual_feeds)
    before_rng = rng_state()
    workload = dmd.build_algorithm(actual_model, actual_feeds, settings, 0, 53)
    assert equal(before_rng, rng_state())
    (actual, session), actual_random = _random_trace(monkeypatch, lambda: _observed_session(workload))
    assert equal(expected_random, actual_random), 'initial noise, sigma and epsilon draws/ordering changed'
    assert equal(expected, actual), 'sample, scores, losses, pre/post-clip gradients or per-update state changed'
    assert equal(expected_state, pack_state(actual_model.state)), 'module/Adam/scheduler/EMA/counter state changed'
    assert equal(before_rng, rng_state()), 'outer Python/Torch RNG must not advance'
    assert any(not equal(row['mean'], row['clipped']) for row in actual['gradients']), 'active clipping required'
    assert actual_model.state['dmd_counters'].tolist() == [1, 1, settings['fake_updates']]
    assert all(parameter.grad is None for role in FORWARDS.values() for parameter in actual_model.state[role].parameters())
    assert session.receipt['dispatcher']['final_versions']['teacher'] == 0
    return actual


@pytest.mark.parametrize('sample_policy', ['reuse_student', 'fresh_student'])
@pytest.mark.parametrize('gradient_policy', ['all_steps', 'last_step'])
def test_owned_entry_exact_sequential_trace(monkeypatch, sample_policy, gradient_policy):
    settings = policy(); settings.update(fake_updates=5, fake_sample_policy=sample_policy,
        gradient_policy=gradient_policy, max_grad_norm=1e-5)
    _compare_oracle(lambda: (_owned_components(), _feeds()), settings, monkeypatch)


def _local_cycle(model, cycle, versions=None):
    workload = dmd.build_algorithm(model, _feeds(), policy(), cycle, 91 + cycle)
    if versions is not None:
        workload.program = replace(workload.program, initial_versions=versions)
    session = Session(workload, REFERENCE, copy_state=False)
    session.run()
    return session.receipt['dispatcher']['final_versions']


def _resume_owned(checkpoint, output):
    from distillschedule.training.checkpoint import restore_checkpoint
    model = _owned_components()
    dmd.build_algorithm(model, _feeds(), policy(), 0, 91)
    restored = restore_checkpoint(checkpoint, model.state, identity={})
    assert restored['cycle'] == restored['cursor'] == 1
    versions = _local_cycle(model, 1, restored['versions'])
    torch.save(dict(state=pack_state(model.state), rng=rng_state(), versions=versions,
        cycle=2, cursor=restored['cursor'] + 1), output)


def test_owned_entry_complete_cycle_resume_in_a_new_process_and_version_refusal(tmp_path):
    import subprocess
    import sys
    from distillschedule.training.checkpoint import save_checkpoint
    model = _owned_components()
    versions = _local_cycle(model, 0)
    before = pack_state(model.state)
    wrong = {**versions, 'student': versions['student'] + 1}
    with pytest.raises(ValueError, match='logical.*version'):
        save_checkpoint(tmp_path/'rejected', model.state, identity={}, cycle=1, cursor=1, versions=wrong)
    assert equal(before, pack_state(model.state)) and not (tmp_path/'rejected').exists()
    checkpoint = save_checkpoint(tmp_path/'cycle-one', model.state, identity={}, cycle=1, cursor=1, versions=versions)
    versions = _local_cycle(model, 1, versions)
    expected = dict(state=pack_state(model.state), rng=rng_state(), versions=versions, cycle=2, cursor=2)
    output = tmp_path/'resumed.pt'
    command = ('from test_training_flow_model_methods import _resume_owned; import sys; '
               '_resume_owned(sys.argv[1], sys.argv[2])')
    paths = [str(Path(__file__).resolve().parent)]
    if os.environ.get('PYTHONPATH'):
        paths.append(os.environ['PYTHONPATH'])
    process = subprocess.run([sys.executable, '-c', command, str(checkpoint), str(output)],
        capture_output=True, text=True, env={**os.environ, 'OMP_NUM_THREADS': '1',
                                           'PYTHONPATH': os.pathsep.join(paths)})
    assert process.returncode == 0, process.stdout + process.stderr
    assert equal(expected, torch.load(output, weights_only=True))


@pytest.fixture(scope='module', params=['flux2-klein-base', 'ltx25'])
def native_config(request, tmp_path_factory):
    if os.environ.get('DMD_MODERN_FLOW_CPU') != '1':
        pytest.skip('target-only actual-native flow boundary; set DMD_MODERN_FLOW_CPU=1 in pinned lane')
    # Opted-in missing dependencies or source must fail, never become a PASS/skip.
    assert torch.__version__.split('+')[0] == '2.10.0', 'actual-native acceptance needs the pinned target lane'
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.config import load_training_config
    family = request.param
    external = os.environ.get('DMD_FLOW_METHOD_IMAGE_CONFIG' if family != 'ltx25' else 'DMD_FLOW_METHOD_LTX_CONFIG')
    if external:
        return family, load_training_config(external)
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp('flow-method-' + family) / 'fixture'
    if family == 'ltx25':
        assert os.environ.get('LTX_SOURCE_ROOT'), 'native LTX target requires existing pinned source'
        builder, _ = load_callable(str(root/'recipes/ltx25/create_cpu_fixture.py') + ':build_fixture')
        assets = builder(output, Path(os.environ['LTX_SOURCE_ROOT']).resolve(strict=True), cycles=2)
        config = assets['lora_config']
    else:
        builder, _ = load_callable(str(root/'recipes/flow_image/create_cpu_fixture.py') + ':build_fixture')
        assets = builder(output, family, cycles=2)
        config = assets['config']
    return family, load_training_config(config)


def _native_owned(native_config):
    from distillschedule.training.session import construct
    from distillschedule.training.flow_image import ImageRole
    from distillschedule.training.ltx import LTXRole
    family, config = native_config
    model, dataset, _, _ = construct(config)
    for role in FORWARDS.values():
        assert isinstance(model.state[role], LTXRole if family == 'ltx25' else ImageRole)
        assert isinstance(model.state[role], dmd.FlowDMDMethods)
        native = model.state[role].native
        if family == 'ltx25':
            # PinnedSource deliberately isolates upstream imports; verify the
            # exact original class rather than assuming a global package name.
            source = model.state[role].source
            source.validate()
            assert type(native) is source.import_module('ltx_core.model.transformer.model').LTXModel
        else:
            assert type(native).__module__.startswith('diffusers.')
    assert not torch.cuda.is_initialized()
    return model, dataset.sample(0)


def test_native_direct_entry_current_owner_no_copy_and_exact_census(native_config, monkeypatch):
    from distillschedule.training.flow_image import QualifiedNativeFlowTrials
    model, feeds = _native_owned(native_config)
    settings = dict(native_config[1].algorithm['parameters'])
    before_rng = rng_state()
    parameters = {role: tuple(id(p) for p in model.state[role].parameters()) for role in FORWARDS.values()}
    def no_copy(*args, **kwargs):
        pytest.fail('model-owned entry must not copy the role')
    monkeypatch.setattr(QualifiedNativeFlowTrials, '__deepcopy__', no_copy)
    workload = dmd.build_algorithm(model, feeds, settings, 0, 17)
    session = Session(workload, REFERENCE, copy_state=False)
    assert equal(before_rng, rng_state())
    method = dmd.FlowDMDMethods.dmd_forward
    entries, denoise = [], []
    def observe(self, payload, **kwargs):
        entries.append((id(self), kwargs['flow_role'], kwargs['phase']))
        return method(self, payload, **kwargs)
    monkeypatch.setattr(dmd.FlowDMDMethods, 'dmd_forward', observe)
    handles = [model.state[role].native.register_forward_pre_hook(
        lambda module, args, role=role: denoise.append(role)) for role in FORWARDS.values()]
    try:
        prepared, forward = workload.program.calls[:2]
        assert prepared.id == 'dmd.c0.student_forward.prepare'
        payload = workload.bindings[prepared.binding](Context(session.feeds, session.state, prepared.attributes))
        for key in feeds:
            assert payload['feeds'][key] is session.feeds[key]
        original = session.state['student']
        session.state['student'] = session.state['fake']
        with random_scope(workload.seed, forward.id, session.device):
            result = workload.bindings[forward.binding](Context({prepared.id: payload}, session.state, forward.attributes))
        assert entries[-1][0] == id(session.state['fake'])
        with random_scope(workload.seed, forward.id, session.device):
            expected = session.state['fake'].rollout(feeds, settings['rollout_sigmas'], settings['gradient_policy'])
        assert equal(result, expected)
        session.state['student'] = original
        entries.clear(); denoise.clear()
        session.run()
    finally:
        for handle in handles: handle.remove()
    assert [role for _, role, _ in entries].count('student') == 1
    assert [role for _, role, _ in entries].count('fake') == 6
    assert [role for _, role, _ in entries].count('teacher') == 1
    assert denoise.count('student') == 4 and denoise.count('fake') == 6
    assert denoise.count('teacher') == (1 if settings['teacher_cfg'] == 1 else 2)
    assert len([call for call in workload.program.calls if call.binding == 'prepare_flow_forward']) == 8
    assert parameters == {role: tuple(id(p) for p in session.state[role].parameters()) for role in FORWARDS.values()}
    for role in ('student', 'fake'):
        owned = {id(p) for group in session.state[role + '_optimizer'].param_groups for p in group['params']}
        assert owned == {id(p) for p in session.state[role].parameters() if p.requires_grad}
    assert equal(before_rng, rng_state()) and workload.replica_semantics is None
    assert model.state['dmd_counters'].tolist() == [1, 1, 5]
    versions = session.receipt['dispatcher']['final_versions']
    assert versions['student_ema'] == versions['student_scheduler'] == 1
    assert versions['fake_scheduler'] == 5 and versions['teacher'] == 0


@pytest.mark.parametrize('sample_policy,gradient_policy,full_audio,cfg_one', [
    ('reuse_student', 'all_steps', False, False),
    ('reuse_student', 'last_step', True, True),
    ('fresh_student', 'all_steps', True, False),
    ('fresh_student', 'last_step', False, False)])
def test_native_exact_sequential_trace_at_new_entry(native_config, monkeypatch,
        sample_policy, gradient_policy, full_audio, cfg_one):
    family, config = native_config
    settings = dict(config.algorithm['parameters'])
    settings.update(fake_sample_policy=sample_policy, gradient_policy=gradient_policy, max_grad_norm=1e-5)
    if cfg_one: settings['teacher_cfg'] = 1.
    if family == 'ltx25': settings['modality_weights'] = dict(video=1., audio=2.)
    def condition(feeds):
        if family == 'ltx25' and full_audio:
            feeds = {**feeds, 'conditioning_mask': {**feeds['conditioning_mask'],
                'audio': torch.ones_like(feeds['conditioning_mask']['audio'])}}
        return feeds
    trace = _compare_oracle(lambda: _native_owned(native_config), settings, monkeypatch, feed_transform=condition)
    if family == 'ltx25' and full_audio:
        assert not trace['values']['dmd.c0.noise']['masks']['audio'].any()
        assert not trace['values']['dmd.c0.f0.noise']['masks']['audio'].any()


def test_native_rank_namespaces_and_local_gradients_match_existing_b1_oracle(native_config, monkeypatch):
    # This is a namespace/binding test, NOT a two-process or sharding claim.
    from flow_replica_oracle import sequential_b1_oracle
    import distillschedule.open_workload.torch_backend as backend
    model, feeds = _native_owned(native_config)
    settings = dict(native_config[1].algorithm['parameters'])
    settings.update(teacher_cfg=1., fake_updates=1, fake_sample_policy='reuse_student')
    before = rng_state()
    expected = sequential_b1_oracle(model.state, [feeds, feeds], settings, 67)
    assert equal(before, rng_state())
    assert not equal(expected['stochastic'][0][3], expected['stochastic'][1][3])
    assert not equal(expected['gradients'][0]['local'][0], expected['gradients'][0]['local'][1])
    actual = []
    for rank in range(2):
        current, row = _native_owned(native_config)
        workload = dmd.build_algorithm(current, row, settings, 0, 67)
        calls = list(workload.program.calls)
        stop = next(index for index, call in enumerate(calls) if call.binding == 'student_backward') + 1
        workload.program = replace(workload.program, calls=tuple(calls[:stop]), outputs=('dmd.c0.student_loss',))
        workload.bindings = {key: value for key, value in workload.bindings.items()
                             if key in {call.binding for call in workload.program.calls}}
        before = rng_state()
        with monkeypatch.context() as patch:
            # Inject only the documented namespace; no process group or
            # reducer is fabricated by this single-process boundary test.
            original_scope = random_scope
            patch.setattr(backend, 'random_scope', lambda seed, key, device, **kwargs:
                original_scope(seed, key, device, namespace=dict(schema='rank_independent_v1', rank=rank, world_size=2)))
            output = Session(workload, REFERENCE, copy_state=False).run()
        assert equal(before, rng_state())
        actual.append(_gradients(current.state, 'student'))
        assert equal(actual[-1], expected['gradients'][0]['local'][rank])
        assert equal(output, {'dmd.c0.student_loss': expected['losses'][rank]['dmd.c0.student_loss']})
    assert not equal(actual[0], actual[1])


def test_native_clock_refusal_precedes_outer_and_native_forward(native_config, monkeypatch):
    model, feeds = _native_owned(native_config)
    workload = dmd.build_algorithm(model, feeds, dict(native_config[1].algorithm['parameters']), 0, 17)
    _change_clock(model.state, {}, 'fake_step_count')
    before = pack_state(model.state)
    def forbidden(*args, **kwargs):
        pytest.fail('clock fault reached a native forward/update')
    monkeypatch.setattr(dmd.FlowDMDMethods, 'dmd_forward', forbidden)
    for role in FORWARDS.values(): monkeypatch.setattr(model.state[role].native, 'forward', forbidden)
    for role in ('student', 'fake'): monkeypatch.setattr(model.state[role + '_optimizer'], 'step', forbidden)
    with pytest.raises(ValueError, match='scheduler clock'):
        Session(workload, REFERENCE, copy_state=False).run()
    assert equal(before, pack_state(model.state))


def test_native_complete_cycle_new_process_resume_is_exact(native_config, tmp_path):
    from test_training_end_to_end import run_cli
    config = native_config[1]
    # Reuse the fixture's immutable assets and real prepared dataset. Only the
    # output/cycle override varies; every invocation is a separate Python process.
    path = tmp_path/'request.json'
    import json
    path.write_text(json.dumps(config.to_dict()))
    first = run_cli('train', path, '--cycles', 1, '--output', tmp_path/'first')
    resumed = run_cli('train', path, '--cycles', 2, '--resume', first['checkpoint'], '--output', tmp_path/'resumed')
    whole = run_cli('train', path, '--cycles', 2, '--output', tmp_path/'whole')
    actual = torch.load(Path(resumed['checkpoint'])/'state.pt', weights_only=True)
    expected = torch.load(Path(whole['checkpoint'])/'state.pt', weights_only=True)
    assert equal(expected, actual), 'parameters/Adam/schedulers/EMA/RNG/data cursor/versions must match'
    assert actual['cycle'] == actual['cursor'] == 2
    assert actual['state']['dmd_counters']['value']['counters'].tolist() == [2, 2, 10]
    assert actual['versions']['teacher'] == 0 and actual['versions']['student_ema'] == 2


def test_optin_does_not_promote_private_replication_or_undeclared_models(monkeypatch):
    from distillschedule.training import distributed
    from distillschedule.open_workload.replica_contract import execution_graph
    monkeypatch.setattr(distributed, 'rank_world', lambda: (0, 2))
    for model in (components(), _owned_components()):
        with pytest.raises(ValueError, match='explicit private-method B1 replication'):
            dmd.build_algorithm(model, _feeds(), policy(), 0, 31)
    model = _owned_components()
    model.capabilities = (*model.capabilities, 'dmd_flow_private_replicated_b1_v1')
    workload = dmd.build_algorithm(model, _feeds(), policy(), 0, 31)
    assert workload.replica_semantics.forward_interface == 'private_methods'
    assert not workload.replica_semantics.forward_methods
    for reduction in ('fsdp_forward', 'ddp_overlap'):
        with pytest.raises(ValueError, match='private methods require post_backward'):
            execution_graph(workload.program, workload.replica_semantics, reduction)


def test_optin_keeps_unqualified_role_clone_refusal_before_state_traversal(monkeypatch):
    from distillschedule.training.flow_image import ImageRole
    from distillschedule.training.ltx import LTXRole
    for role_class in (ImageRole, LTXRole):
        # A shell witnesses only MRO/admission; no substituted model is run.
        role = role_class.__new__(role_class)
        torch.nn.Module.__init__(role)
        role.native = torch.nn.Linear(1, 1)
        if role_class is LTXRole: role.device = torch.device('cpu')
        role._trial_evidence_kind = 'PRETRAINED_NOT_QUALIFIED'
        with monkeypatch.context() as patch:
            patch.setattr(torch.nn.Module, '__getstate__', lambda self: pytest.fail('clone traversed native state'))
            with pytest.raises(ValueError, match='pretrained or unqualified.*clone trials are not admitted'):
                deepcopy(role)
