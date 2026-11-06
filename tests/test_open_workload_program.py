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


def _immutable_read(ctx):
    from distillschedule.open_workload.torch_backend import physical_tensor
    return ctx.inputs['x'] + sum(physical_tensor(p).sum() for p in ctx.state['teacher'].parameters())


def _immutable_session(*, persistent=True, lifecycle=None, teacher=None):
    from distillschedule.open_workload.torch_backend import Session, Workload
    if teacher is None:
        teacher = torch.nn.Module()
        teacher.register_parameter('weight', torch.nn.Parameter(torch.ones(2, 2), requires_grad=False))
    workload = Workload(Program((Call('read', 'read', inputs=('x',), reads=('teacher',)),),
                                {'teacher': 0}, ('x',), ('read',)),
                        {'read': _immutable_read}, {'teacher': teacher}, {'x': torch.tensor(1.)},
                        immutable_modules=('teacher',))
    option = {'backend': 'torch-callable-resident-v1', 'share_immutable': False,
              'checkpoint_bindings': []}
    return Session(workload, option, copy_state=not persistent, persistent=persistent,
                   lifecycle=lifecycle), workload


def _count_content_hashes(monkeypatch):
    from distillschedule.open_workload import torch_backend
    original = torch_backend.tensor_signature
    calls = []
    def counted(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch_backend, 'tensor_signature', counted)
    return calls


def test_persistent_session_does_not_rehash_unchanged_frozen_parameters(monkeypatch):
    calls = _count_content_hashes(monkeypatch)
    session, workload = _immutable_session()
    assert len(calls) == 1  # Initial admission still binds the actual contents.
    for cycle in range(3):
        if cycle:
            session.bind_cycle(workload)
        assert session.run()['read'].item() == 5.
        assert len(calls) == 1


def test_frozen_version_bump_revalidates_once_without_rejecting_equal_values(monkeypatch):
    calls = _count_content_hashes(monkeypatch)
    session, workload = _immutable_session()
    parameter = session.state['teacher'].weight
    before = parameter._version
    parameter.copy_(parameter.clone())
    assert parameter._version > before
    session.run()
    assert len(calls) == 2
    session.bind_cycle(workload)
    session.run()
    assert len(calls) == 2


def test_persistent_session_rejects_inplace_frozen_parameter_mutation():
    session, _ = _immutable_session()
    session.state['teacher'].weight.add_(1)
    with pytest.raises(ValueError, match='declared immutable parameters changed'):
        session.run()


@pytest.mark.parametrize('change', ['owner', 'name', 'requires_grad', 'shape', 'stride', 'dtype'])
def test_persistent_session_rejects_frozen_parameter_ownership_or_layout_changes(change):
    session, _ = _immutable_session()
    teacher = session.state['teacher']
    parameter = teacher.weight
    if change == 'owner':
        teacher.weight = torch.nn.Parameter(parameter.clone(), requires_grad=False)
    elif change == 'name':
        del teacher.weight
        teacher.register_parameter('renamed', parameter)
    elif change == 'requires_grad':
        parameter.requires_grad_(True)
    elif change == 'shape':
        parameter.data = parameter.data.reshape(4)
    elif change == 'stride':
        parameter.data = parameter.data.t()
    else:
        parameter.data = parameter.data.double()
    with pytest.raises(ValueError, match='declared immutable parameters changed'):
        session.run()


def test_explicit_frozen_content_audit_catches_unsafe_data_writes():
    session, _ = _immutable_session()
    parameter = session.state['teacher'].weight
    before = parameter._version
    parameter.data.add_(1)  # Unsupported raw writes bypass ordinary version guards.
    assert parameter._version == before
    with pytest.raises(ValueError, match='declared immutable parameters changed'):
        session.audit_immutable()


@pytest.mark.parametrize('measure,persistent', [(True, True), (False, False)])
def test_measured_and_trial_sessions_always_audit_frozen_contents(measure, persistent):
    session, _ = _immutable_session(persistent=persistent)
    session.state['teacher'].weight.data.add_(1)
    with pytest.raises(ValueError, match='declared immutable parameters changed'):
        session.run(measure=measure)


def test_frozen_admission_uses_the_final_lifecycle_owned_parameters():
    import weakref
    class Lifecycle:
        def prepare(self, session):
            teacher = session.state['teacher']
            self.replaced = weakref.ref(teacher.weight)
            teacher.weight = torch.nn.Parameter(teacher.weight.clone(), requires_grad=False)
        def before_run(self, session, order): pass
        def before_call(self, session, call): pass
        def after_call(self, session, call): pass
        def finish(self, session): pass
    lifecycle = Lifecycle()
    session, _ = _immutable_session(lifecycle=lifecycle)
    assert lifecycle.replaced() is None
    assert session.run()['read'].item() == 5.


def test_counterless_frozen_parameters_keep_content_validation():
    with torch.inference_mode():
        teacher = torch.nn.Module()
        teacher.register_parameter('weight', torch.nn.Parameter(torch.ones(2, 2), requires_grad=False))
    session, workload = _immutable_session(teacher=teacher)
    assert session.run()['read'].item() == 5.
    session.bind_cycle(workload)
    with torch.inference_mode():
        teacher.weight.add_(1)
    with pytest.raises(ValueError, match='declared immutable parameters changed'):
        session.run()


def _local_immutable_worker(rank, rendezvous):
    from datetime import timedelta
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        mesh = init_device_mesh('cpu', (2,), mesh_dim_names=('data',))
        teacher = torch.nn.Module()
        local = DTensor.from_local(torch.ones(1, 2), mesh, [Shard(0)],
                                   shape=torch.Size((2, 2)), stride=(2, 1))
        teacher.register_parameter('weight', torch.nn.Parameter(local, requires_grad=False))
        def forbidden(*args, **kwargs):
            raise AssertionError('immutable validation must not gather a full DTensor')
        DTensor.full_tensor = forbidden
        session, workload = _immutable_session(teacher=teacher)
        if rank == 1:
            teacher.weight.to_local().copy_(teacher.weight.to_local().clone())
        assert session.run()['read'].item() == 3.
        session.bind_cycle(workload)
        session.audit_immutable()
        if rank == 1:
            teacher.weight.to_local().add_(1)
            with pytest.raises(ValueError, match='declared immutable parameters changed'):
                session.run()
        else:
            assert session.run()['read'].item() == 3.
    finally:
        dist.destroy_process_group()


def test_dtensor_immutable_checks_are_local_even_when_only_one_rank_changes(tmp_path):
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('local shard validation requires Gloo')
    torch.multiprocessing.spawn(_local_immutable_worker, args=(str(tmp_path / 'gloo'),),
                                nprocs=2, join=True)
