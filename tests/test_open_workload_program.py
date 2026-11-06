"""Unregistered workload semantics; no model-specific planner or fake timing."""
import pytest
import torch
from distillschedule.workloads.program import Call, Program
from distillschedule.search.plan_ir import validate_versions


def make(n):
    calls = [Call('s', 'predict', ('x',), ('student',), grad='grad', replay_safe=True)]
    calls += [Call('t', 'teacher', ('x',), ('teacher',)),
              Call('loss', 'loss', ('s','t'), grad='grad'),
              Call('bw', 'backward', ('loss',), ('grad',), ('grad',), grad='grad', backward_of=('loss',)),
              Call('update', 'step', ('bw',), ('student','grad'), ('student','grad'))]
    for i in range(n):
        calls += [Call(f'read{i}', 'score', ('x',), ('student',))]
    return Program(tuple(calls), {'student':0,'teacher':0,'grad':0}, ('x',), (f'read{n-1}',), {'n':n})


@pytest.mark.parametrize('n',[1,2,5,8])
def test_arbitrary_frequency_and_roundtrip(n):
    p=make(n); g=p.compile(); validate_versions(g)
    assert Program.from_dict(p.to_dict()).digest==p.digest
    bw=next(o for o in g.operations if o.operation_id=='bw')
    assert bw.required_versions['student']==0
    assert {'s','t','loss'}<=set(bw.metadata['inputs'])
    assert g.metadata['final_versions']=={'student':1,'teacher':0,'grad':2}
    assert len(g.operations)==6+n


def test_late_backward_cannot_read_mutated_parameters():
    with pytest.raises(ValueError,match='version changed'):
        Program((Call('f','f',reads=('w',),grad='grad'),
                 Call('u','u',reads=('w',),writes=('w',)),
                 Call('b','b',inputs=('f',),grad='grad',backward_of=('f',))),{'w':0})


def test_independent_calls_are_not_forced_to_source_order():
    g=Program((Call('a','f',reads=('a',)),Call('b','f',reads=('b',))),{'a':0,'b':0}).compile()
    assert all(e.target=="__open_finish__" for e in g.dependencies)


def test_raw_and_war_added_automatically():
    g=Program((Call('r','f',reads=('w',)),Call('u','f',reads=('w',),writes=('w',)),
               Call('r2','f',reads=('w',))),{'w':0}).compile()
    assert {('r','u'),('u','r2')}<={(e.source,e.target) for e in g.dependencies}


@pytest.mark.parametrize('bad',[
    {'id':'a','binding':'f','replay_safe':True},
    {'id':'a','binding':'f','grad':'grad','replay_safe':True,'reads':('w',),'writes':('w',)},
    {'id':'a','binding':'f','backward_of':('unknown',)},
])
def test_illegal_replay_or_backward_rejected(bad):
    with pytest.raises(ValueError): Call(**bad)


def test_no_name_whitelist_or_fixed_five_contract():
    for name in ('unseen_model_2026','my_loss','totally_new_teacher'):
        p=Program((Call('work',name,('input',)),),{},('input',),('work',))
        assert p.compile().operations[0].metadata['binding']==name


def _saved_forward(ctx):
    model = ctx.state['model']
    value = model(ctx.inputs['x']) if ctx.attributes['linear'] else model.weight * 2
    if ctx.attributes['view']:
        value = value.view_as(value)
    loss = value.square().sum()
    with torch.no_grad():
        if ctx.attributes['mutation'] == 'direct':
            value.add_(10)
        elif ctx.attributes['mutation'] == 'detached_alias':
            value.detach().view_as(value).add_(10)
        elif ctx.attributes['mutation'] == 'feed_alias':
            ctx.inputs['x'].detach().view_as(ctx.inputs['x']).add_(10)
    return loss


def _saved_backward(ctx):
    ctx.inputs['forward'].backward()
    return ctx.state['model'].weight.grad.detach().clone()


def _saved_step(ctx):
    ctx.state['optimizer'].step()
    return ctx.state['model'].weight.detach().clone()


def _saved_session(*, mutation='none', view=False, checkpoint_policy=None, update=False):
    from distillschedule.open_workload.torch_backend import Session, Workload
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2)
    state = {'model': model, 'gradient': 0}
    calls = [Call('forward', 'forward', inputs=('x',), reads=('model',), grad='grad',
                  replay_safe=mutation == 'none' or checkpoint_policy is not None,
                  attributes={'mutation': mutation, 'view': view, 'linear': update}),
             Call('backward', 'backward', inputs=('forward',), reads=('model', 'gradient'),
                  writes=('gradient',), grad='grad', backward_of=('forward',))]
    bindings = {'forward': _saved_forward, 'backward': _saved_backward}
    outputs = ('backward',)
    if update:
        state['optimizer'] = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        calls.append(Call('update', 'step', inputs=('backward',),
                          reads=('model', 'gradient', 'optimizer'),
                          writes=('model', 'optimizer')))
        bindings['step'] = _saved_step
        outputs += ('update',)
    workload = Workload(Program(tuple(calls), {key: 0 for key in state}, ('x',), outputs),
                        bindings, state, {'x': torch.full((1, 1), 2.)})
    option = {'backend': 'torch-callable-resident-v1', 'share_immutable': False,
              'checkpoint_bindings': ['forward'] if checkpoint_policy is not None else []}
    if checkpoint_policy not in (None, 'recompute'):
        option['checkpoint_policy'] = checkpoint_policy
    return Session(workload, option)


@pytest.mark.parametrize('measure', [False, True])
@pytest.mark.parametrize('view', [False, True])
@pytest.mark.parametrize('mutation', ['direct', 'detached_alias'])
def test_session_rejects_saved_intermediate_mutation(measure, view, mutation):
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        _saved_session(mutation=mutation, view=view).run(measure=measure)


@pytest.mark.parametrize('measure', [False, True])
@pytest.mark.parametrize('view', [False, True])
def test_session_preserves_gradient_and_measured_activation_accounting(measure, view):
    session = _saved_session(view=view)
    assert session.run(measure=measure)['backward'].item() == 16.
    if measure:
        # One saved intermediate plus the scalar output, not just output bytes.
        assert session.profile['forward']['retained_bytes'] == 8


def test_unmeasured_session_does_not_install_observation_hooks(monkeypatch):
    session = _saved_session()
    def forbidden(*args, **kwargs):
        raise AssertionError('unmeasured Session installed saved-tensor observation hooks')
    monkeypatch.setattr(torch.autograd.graph, 'saved_tensors_hooks', forbidden)
    assert session.run()['backward'].item() == 16.


def test_observation_guard_keeps_storage_and_alias_version_without_grad_fn():
    from distillschedule.runtime.saved_tensors import pack_saved_tensor, unpack_saved_tensor
    value = torch.ones(2, requires_grad=True) * 2
    view = value.view(1, 2)
    packed = pack_saved_tensor(view)
    saved = unpack_saved_tensor(packed)
    assert saved is not view and saved.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
    assert saved.shape == view.shape and saved.stride() == view.stride()
    assert saved.grad_fn is None and not saved.requires_grad
    value.detach().add_(1)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        unpack_saved_tensor(packed)


def test_cpu_native_calibration_keeps_saved_tensor_mutation_check(monkeypatch):
    from distillschedule.native.engine import NativeEngine
    from distillschedule.native.task import NativeTask
    engine = NativeEngine(NativeTask(fake_updates=1, denoising_steps=1,
                                     video_height=8, video_width=8, video_frames=1))
    def invalid_update(cycle):
        weight = next(iter(engine.adapter.trainables('generator').values()))
        value = weight * 2
        loss = value.square().sum()
        value.detach().add_(10)
        loss.backward()
    monkeypatch.setattr(engine, 'generator_update', invalid_update)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        engine.run_cycle(calibration=True)


@pytest.mark.parametrize('measure', [False, True])
@pytest.mark.parametrize('policy', ['recompute', 'save_matmul'])
def test_checkpoint_session_preserves_gradient_optimizer_and_effects(measure, policy):
    reference = _saved_session(update=True)
    expected = reference.run(measure=measure)
    selected = _saved_session(update=True, checkpoint_policy=policy)
    actual = selected.run(measure=measure)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual['backward'].item() == 16.
    assert actual['update'].item() == pytest.approx(1.84)
    torch.testing.assert_close(selected.state['optimizer'].state_dict()['state'][0]['momentum_buffer'],
                               reference.state['optimizer'].state_dict()['state'][0]['momentum_buffer'],
                               rtol=0, atol=0)
    assert selected.effects['checkpoint_calls'] == 1
    assert selected.effects['recompute_calls'] == 1
    if policy == 'save_matmul':
        assert selected.effects['selective_save_requests'] == 1
        assert selected.effects['selective_native_cache_bytes'] == 4


@pytest.mark.parametrize('measure', [False, True])
@pytest.mark.parametrize('policy', ['recompute', 'save_matmul'])
def test_checkpoint_session_rejects_input_alias_mutation(measure, policy):
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        _saved_session(mutation='feed_alias', checkpoint_policy=policy, update=True).run(measure=measure)


@pytest.mark.parametrize('measure', [False, True])
def test_selective_checkpoint_keeps_native_cache_mutation_check(measure):
    with pytest.raises(RuntimeError, match='Tensor cached during selective activation checkpoint has been mutated'):
        _saved_session(mutation='detached_alias', checkpoint_policy='save_matmul', update=True).run(measure=measure)
