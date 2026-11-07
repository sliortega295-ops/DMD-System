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


@pytest.mark.parametrize('saved_kind', ['parameter', 'buffer', 'other_device'])
def test_activation_offload_checks_versions_for_tensors_not_copied(saved_kind):
    from distillschedule.open_workload.role_offload import ActivationOffloader, HostBudget
    model = torch.nn.Module()
    value = torch.tensor([2.])
    if saved_kind == 'parameter':
        model.register_parameter('weight', torch.nn.Parameter(value, requires_grad=False))
        value = model.weight
    elif saved_kind == 'buffer':
        model.register_buffer('weight', value)
    # A CPU saved tensor is outside this target; no accelerator is allocated.
    device = 'meta' if saved_kind == 'other_device' else 'cpu'
    budget = HostBudget(1024)
    offloader = ActivationOffloader(device, model, budget)
    try:
        source = torch.ones(1, requires_grad=True)
        with offloader.hooks():
            loss = (source * value).sum()
        assert offloader.packed == 0 and budget.used == 0
        value.add_(1)
        with pytest.raises(RuntimeError, match='modified by an inplace operation'):
            loss.backward()
    finally:
        offloader.close()
    assert budget.used == 0


@pytest.mark.parametrize('prefetch', [False, True])
def test_activation_offload_checks_alias_version_after_original_tensor_is_gone(prefetch):
    import gc
    import weakref
    from distillschedule.open_workload.role_offload import ActivationOffloader, HostBudget
    budget = HostBudget(1024)
    offloader = ActivationOffloader('cpu', torch.nn.Identity(), budget)
    try:
        source = torch.tensor([2.], requires_grad=True)
        with offloader.hooks():
            value = source * 2
            loss = value.square().sum()
        alias = value.detach().view_as(value)
        original = weakref.ref(value)
        del value
        gc.collect()
        assert original() is None
        if prefetch:
            offloader.prefetch()
        alias.add_(1)
        with pytest.raises(RuntimeError, match='modified by an inplace operation'):
            loss.backward()
    finally:
        offloader.close()
    assert budget.used == 0


@pytest.mark.parametrize('prefetch', [False, True])
def test_activation_offload_preserves_unmodified_noncontiguous_backward(prefetch):
    from distillschedule.open_workload.role_offload import ActivationOffloader, HostBudget
    source = torch.arange(6., requires_grad=True)
    budget = HostBudget(1024)
    offloader = ActivationOffloader('cpu', torch.nn.Identity(), budget)
    try:
        with offloader.hooks():
            value = (source * 2).view(3, 2).t()
            loss = value.square().sum()
        if prefetch:
            offloader.prefetch()
        loss.backward()
        torch.testing.assert_close(source.grad, source.detach() * 8, rtol=0, atol=0)
        assert offloader.packed_bytes == 24
        assert offloader.unpacked == 1
    finally:
        offloader.close()
    assert budget.used == 0


def test_activation_offload_version_guard_does_not_retain_original_storage():
    import gc
    import weakref
    from distillschedule.open_workload.role_offload import ActivationOffloader, HostBudget
    budget = HostBudget(1024)
    offloader = ActivationOffloader('cpu', torch.nn.Identity(), budget)
    try:
        value = torch.ones(16)
        storage = value.untyped_storage()
        original_storage = weakref.ref(storage)
        packed = offloader.pack(value)
        del value, storage
        gc.collect()
        assert original_storage() is None
        torch.testing.assert_close(offloader.unpack(packed), torch.ones(16))
    finally:
        offloader.close()
    assert budget.used == 0


def test_activation_offload_versionless_tensor_failure_releases_host_budget():
    from distillschedule.open_workload.role_offload import ActivationOffloader, HostBudget
    with torch.inference_mode():
        value = torch.ones(1)
    budget = HostBudget(1024)
    offloader = ActivationOffloader('cpu', torch.nn.Identity(), budget)
    try:
        with pytest.raises(RuntimeError, match='version counter'):
            offloader.pack(value)
    finally:
        offloader.close()
    assert budget.used == 0


def test_component_saved_tensor_observation_does_not_retain_output_grad_fn():
    import gc
    import weakref
    from distillschedule.open_workload.role_profiling import ComponentRecorder
    recorder = ComponentRecorder('cpu', enabled=True)
    source = torch.ones(1, requires_grad=True)
    with recorder.saved(torch.nn.Identity(), 0):
        value = source.sigmoid()
        loss = value.sum()
    output = weakref.ref(value)
    del value
    gc.collect()
    assert output() is None
    loss.backward()
    assert torch.isfinite(source.grad).all()
    assert recorder.saved_bytes == {'0': 4}


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


def _numpy_saved_forward(ctx):
    import numpy as np
    scale=float(np.random.uniform(.5,1.5))
    return (ctx.state['model'](ctx.inputs['x'])*scale).sin().square().sum()


def _numpy_saved_session(policy=None):
    from dataclasses import replace
    from distillschedule.open_workload.torch_backend import Session
    base=_saved_session(update=True,checkpoint_policy=policy)
    workload=replace(base.workload,bindings={**base.workload.bindings,'forward':_numpy_saved_forward})
    return Session(workload,base.implementation)


@pytest.mark.parametrize('measure',[False,True])
@pytest.mark.parametrize('policy',['recompute','save_matmul'])
def test_checkpoint_numpy_random_binding_keeps_gradient_and_optimizer(measure,policy):
    np=pytest.importorskip('numpy')
    np.random.seed(123)
    reference=_numpy_saved_session()
    expected=reference.run(measure=measure)
    np.random.seed(123)
    selected=_numpy_saved_session(policy)
    actual=selected.run(measure=measure)
    for key in expected:
        torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    torch.testing.assert_close(selected.state['optimizer'].state_dict()['state'][0]['momentum_buffer'],
                               reference.state['optimizer'].state_dict()['state'][0]['momentum_buffer'],
                               rtol=0,atol=0)
    assert selected.effects['recompute_calls']==1
    if policy=='save_matmul':
        assert selected.effects['selective_save_requests']==1


@pytest.mark.parametrize('fail',[False,True])
def test_logical_random_scope_preserves_numpy_caller_state(fail):
    import random
    np=pytest.importorskip('numpy')
    from distillschedule.open_workload.torch_backend import random_scope
    np.random.seed(234)
    before=np.random.get_state()
    python_before=random.getstate()
    torch_before=torch.get_rng_state().clone()
    def invoke():
        with random_scope(5,'numpy-call',torch.device('cpu')):
            draws=(np.random.rand(),random.random(),torch.rand(()).item())
            if fail:raise ValueError('rng-failure-sentinel')
            return draws
    if fail:
        with pytest.raises(ValueError,match='rng-failure-sentinel'):invoke()
    else:
        first=invoke()
        current=np.random.get_state()
        assert current[0]==before[0] and np.array_equal(current[1],before[1]) and current[2:]==before[2:]
        np.random.seed(999)
        other=np.random.get_state()
        assert invoke()==first
        current=np.random.get_state()
        assert current[0]==other[0] and np.array_equal(current[1],other[1]) and current[2:]==other[2:]
        np.random.set_state(before)
    after=np.random.get_state()
    assert after[0]==before[0] and np.array_equal(after[1],before[1]) and after[2:]==before[2:]
    assert random.getstate()==python_before
    assert torch.equal(torch.get_rng_state(),torch_before)


def test_logical_random_scope_works_without_numpy(monkeypatch):
    import builtins
    from distillschedule.open_workload.torch_backend import random_scope
    original=builtins.__import__
    def without_numpy(name,*args,**kwargs):
        if name=='numpy':raise ImportError('numpy is optional')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',without_numpy)
    before=torch.get_rng_state().clone()
    with random_scope(5,'optional-numpy',torch.device('cpu')):
        first=torch.rand(2)
    with random_scope(5,'optional-numpy',torch.device('cpu')):
        torch.testing.assert_close(torch.rand(2),first,rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),before)


def _immutable_read(ctx):
    from distillschedule.open_workload.torch_backend import physical_tensor
    return ctx.inputs['x'] + sum(physical_tensor(p).sum() for p in ctx.state['teacher'].parameters())


class _StatefulTeacher(torch.nn.Module):
    def __init__(self, mutation=None):
        super().__init__()
        self.register_buffer('running', torch.zeros(()))
        self.register_buffer('temporary', torch.zeros(()), persistent=False)
        self.mutation = mutation

    def forward(self, value):
        if self.mutation == 'buffer':
            self.running.add_(1)
        elif self.mutation == 'nonpersistent':
            self.temporary.add_(1)
        elif self.mutation == 'mode':
            self.train()
        return value + self.running


def _teacher_forward(ctx):
    return ctx.state['teacher'](ctx.inputs['x'])


def _teacher_commit(ctx):
    ctx.state['commits'].append(True)


def _teacher_state_session(teacher, *, lifecycle=None):
    from distillschedule.open_workload.torch_backend import Session, Workload
    workload = Workload(Program((
        Call('teacher', 'forward', inputs=('x',), reads=('teacher',)),
        Call('commit', 'commit', inputs=('teacher',), reads=('commits',), writes=('commits',))),
        {'teacher': 0, 'commits': 0}, ('x',), ('teacher',)),
        {'forward': _teacher_forward, 'commit': _teacher_commit},
        {'teacher': teacher, 'commits': []}, {'x': torch.ones(8, 4)},
        immutable_modules=('teacher',), immutable_state_modules=('teacher',))
    option = {'backend': 'torch-callable-resident-v1', 'share_immutable': False,
              'checkpoint_bindings': []}
    return Session(workload, option, copy_state=False, persistent=True, lifecycle=lifecycle), workload


def test_dmd_builder_rejects_training_teacher_without_changing_its_mode():
    from distillschedule.open_workload.dmd import DMDHooks, build_dmd_workload
    teacher = torch.nn.BatchNorm1d(4).requires_grad_(False)
    student, fake = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match='Teacher.*eval'):
        build_dmd_workload(student=student, fake=fake, teacher=teacher,
            student_optimizer=torch.optim.SGD(student.parameters(), lr=.01),
            fake_optimizer=torch.optim.SGD(fake.parameters(), lr=.01),
            hooks=DMDHooks(*([_immutable_read] * 6)), feeds={'x': torch.ones(8, 4)},
            cycles=1, fake_updates=1, fake_sample_policy='reuse_student')
    assert teacher.training and teacher.num_batches_tracked.item() == 0


@pytest.mark.parametrize('mutation', ['buffer', 'nonpersistent', 'mode'])
def test_immutable_teacher_state_mutation_is_rejected_before_later_update(mutation):
    session, _ = _teacher_state_session(_StatefulTeacher(mutation).eval())
    with pytest.raises(ValueError, match='immutable model state'):
        session.run()
    assert session.state['commits'] == []


def test_teacher_buffer_audit_catches_raw_writes_and_does_not_rehash_unchanged_buffers(monkeypatch):
    from dataclasses import replace
    calls = _count_content_hashes(monkeypatch)
    session, workload = _teacher_state_session(_StatefulTeacher().eval())
    session.run()
    admitted = len(calls)
    session.bind_cycle(replace(workload, program=replace(workload.program,
        initial_versions=session.receipt['dispatcher']['final_versions'])))
    session.run()
    assert len(calls) == admitted
    session.state['teacher'].temporary.data.add_(1)
    with pytest.raises(ValueError, match='immutable model state'):
        session.audit_immutable()


def test_teacher_state_contract_cannot_be_removed_between_cycles():
    from dataclasses import replace
    session, workload = _teacher_state_session(_StatefulTeacher().eval())
    session.run()
    with pytest.raises(ValueError, match='immutable model state'):
        session.bind_cycle(replace(workload, immutable_state_modules=()))


def _distributed_teacher_guard_worker(rank, rendezvous, mutation):
    from datetime import timedelta
    import torch.distributed as dist
    class Lifecycle:
        def prepare(self, session): pass
        def before_run(self, session, order): pass
        def before_call(self, session, call): pass
        def after_call(self, session, call): pass
        def finish(self, session): pass
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=10))
    try:
        teacher = _StatefulTeacher(mutation if rank == 1 else None).eval()
        session, _ = _teacher_state_session(teacher, lifecycle=Lifecycle())
        if mutation is None:
            session.run()
            assert session.state['commits'] == [True]
            return
        with pytest.raises(ValueError, match='immutable model state') as error:
            session.run()
        errors = [None, None]
        dist.all_gather_object(errors, str(error.value))
        assert errors[0] == errors[1] and 'Timed out' not in errors[0]
        assert session.state['commits'] == []
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mutation', ['buffer', 'mode', None])
def test_rank_local_teacher_state_change_is_rejected_on_all_lifecycle_ranks(tmp_path, mutation):
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    torch.multiprocessing.spawn(_distributed_teacher_guard_worker,
        args=(str(tmp_path / 'teacher-init'), mutation), nprocs=2, join=True)


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
        assert len(calls) == 2  # C1 content audit, then metadata-only ordinary cycles.


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


def test_state_audit_cadence_and_descriptor_purity_without_payload_hashes(monkeypatch):
    from distillschedule.open_workload.state_audit import audit_scope, mutation_snapshot, content_due
    import distillschedule.open_workload.torch_backend as backend
    class Owner:
        def __init__(self):
            self.weight = torch.ones(3)
            self.clock = 0
    owner = Owner()
    def forbidden(*args, **kwargs):
        raise AssertionError('ordinary purity checks must not hash tensor payloads')
    monkeypatch.setattr(backend, 'tensor_signature', forbidden)
    before = mutation_snapshot(owner)
    assert before == mutation_snapshot(owner)
    owner.clock += 1
    assert before != mutation_snapshot(owner)
    owner.clock -= 1
    owner.weight.add_(1)
    assert before != mutation_snapshot(owner)
    with audit_scope(cycle=2, every=50, persistent=True):
        assert content_due() is False
    for cycle in (1, 50, 100):
        with audit_scope(cycle=cycle, every=50, persistent=True):
            assert content_due() is True
    with pytest.raises(ValueError, match='positive integer'):
        with audit_scope(cycle=1, every=True, persistent=True):
            pass


def _audit_cycle_step(ctx):
    from distillschedule.open_workload.torch_backend import unwrap_module
    ctx.state['optimizer'].step()
    ctx.state['optimizer'].zero_grad(set_to_none=True)
    return unwrap_module(ctx.state['model']).weight.detach().clone()


def _audit_cycle_backward(ctx):
    from distillschedule.open_workload.torch_backend import unwrap_module
    ctx.inputs['forward'].backward()
    return unwrap_module(ctx.state['model']).weight.grad.detach().clone()


def _audit_replica_worker(rank, rendezvous, mode):
    from dataclasses import replace
    from datetime import timedelta
    import torch.distributed as dist
    from distillschedule.open_workload.execution import ExecutionBackend
    from distillschedule.open_workload.replica_contract import ReplicaSemantics
    from distillschedule.open_workload.torch_backend import Workload
    import distillschedule.open_workload.torch_backend as runtime
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(17)
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        calls = (Call('forward', 'forward', ('x',), ('model',), grad='grad',
                      attributes={'linear': True, 'view': False, 'mutation': 'none'}),
                 Call('backward', 'backward', ('forward',), ('model',), grad='grad',
                      backward_of=('forward',)),
                 Call('step', 'step', ('backward',), ('model', 'optimizer'), ('model', 'optimizer')))
        workload = Workload(Program(calls, {'model': 0, 'optimizer': 0}, ('x',), ('step',)),
                            {'forward': _saved_forward, 'backward': _audit_cycle_backward, 'step': _audit_cycle_step},
                            {'model': model, 'optimizer': optimizer}, {'x': torch.full((1, 1), rank + 1.)},
                            replica_semantics=ReplicaSemantics({'forward': 'model'}, ('backward',), 1))
        backend = ExecutionBackend(workload, state_audit_every_cycles=3)
        option = {'backend': 'torch-callable-replicated-v1',
                  'local': {'backend': 'torch-callable-resident-v1', 'share_immutable': False,
                            'checkpoint_bindings': []}, 'reduction': mode,
                  'bucket_cap_mb': 1 if mode == 'ddp_overlap' else 0}
        session = backend.session(option, copy_state=False, persistent=True)
        session.run()
        assert session.effects['state_audit']['kind'] == 'content_audit'
        assert session.effects['replica_state_consensus'] is True
        initial = session.receipt['dispatcher']['final_versions']
        workload = replace(workload, state=session.state,
                           program=replace(workload.program, initial_versions=initial))
        session.bind_cycle(workload)
        original = runtime.tensor_signature
        def forbidden(*args, **kwargs):
            raise AssertionError('ordinary metadata cycle must not hash tensor payloads')
        runtime.tensor_signature = forbidden
        session.run()
        assert session.effects['state_audit']['kind'] == 'metadata'
        assert session.effects['replica_state_consensus'] is False
        runtime.tensor_signature = original
        initial = session.receipt['dispatcher']['final_versions']
        workload = replace(workload, state=session.state,
                           program=replace(workload.program, initial_versions=initial))
        session.bind_cycle(workload)
        session.run()
        assert session.effects['state_audit']['kind'] == 'content_audit'
        if rank == 1:
            model.weight.data.add_(1)
        with pytest.raises(ValueError, match='state.*diverged'):
            session.audit_state()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', ['post_backward', 'ddp_overlap'])
def test_persistent_distributed_cycles_only_hash_at_admitted_audit_cadence(tmp_path, mode):
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    torch.multiprocessing.spawn(_audit_replica_worker,
        args=(str(tmp_path / ('audit-' + mode)), mode), nprocs=2, join=True)


def test_frozen_state_guard_rechecks_mutations_and_does_not_retain_tensor():
    import gc
    import weakref
    from distillschedule.open_workload.state_audit import FrozenGuard, audit_scope
    from distillschedule.open_workload.torch_backend import tensor_signature
    tensor = torch.ones(4)
    expected = {'weight': tensor_signature(tensor)}
    calls = []
    def signature(value):
        calls.append(None)
        return tensor_signature(value)
    guard = FrozenGuard()
    guard.check({'weight': tensor}, expected, signature)
    for _ in range(3):
        guard.check({'weight': tensor}, expected, signature)
    assert len(calls) == 1
    tensor.copy_(tensor.clone())
    guard.check({'weight': tensor}, expected, signature)
    assert len(calls) == 2
    with audit_scope(cycle=50):
        guard.check({'weight': tensor}, expected, signature)
    assert len(calls) == 3
    tensor.data.add_(1)
    with audit_scope(cycle=50), pytest.raises(ValueError, match='frozen state changed'):
        guard.check({'weight': tensor}, expected, signature)
    reference = weakref.ref(tensor)
    del tensor
    gc.collect()
    assert reference() is None


class _AuditDescriptorOwner:
    def __init__(self, action='none'):
        self.weight = torch.ones(8192)
        self.clock = 0
        self.action = action

    def state_dict(self):
        raise AssertionError('descriptor inspection must not stage a state_dict')

    def load_state_dict(self, value):
        pass

    def execution_scope(self, option, effects):
        from contextlib import nullcontext
        return nullcontext()

    def execution_options(self):
        if self.action == 'clock':
            self.clock += 1
        if self.action == 'weight':
            self.weight.add_(1)
        if self.action == 'raw_weight':
            self.weight.data.add_(1)
        if self.action == 'rng':
            torch.rand(1)
        if self.action == 'numpy_rng':
            import numpy as np
            np.random.rand()
        if self.action == 'python_rng':
            import random
            random.random()
        return dict(schema_version='distillschedule.owner_execution.v1', baseline={'checkpoint': False},
                    options=[{'checkpoint': False}, {'checkpoint': True}],
                    effect=dict(kind='native_checkpoint', enabled_key='checkpoint', components=['model']),
                    identity={'source': 'test'})


@pytest.mark.parametrize('action', ['none', 'clock', 'weight', 'raw_weight', 'rng', 'numpy_rng', 'python_rng'])
def test_owner_descriptor_purity_keeps_clock_weight_and_all_rng_guards_without_state_hash(action, monkeypatch):
    import distillschedule.open_workload.torch_backend as runtime
    from distillschedule.open_workload.owner_execution import inventory
    owner = _AuditDescriptorOwner(action)
    if action != 'none':
        from distillschedule.open_workload.collective_owner import _rng
        rng = runtime.signature(_rng())
        cuda_initialized = torch.cuda.is_initialized()
        with pytest.raises(ValueError, match='pure and stable'):
            inventory({'owner': owner})
        assert rng == runtime.signature(_rng())
        assert torch.cuda.is_initialized() is cuda_initialized
        return
    before = inventory({'owner': owner})  # Actual first/source admission keeps full checks.
    original = runtime.tensor_signature
    def signature(tensor, **kwargs):
        if tensor.numel() == 8192:
            raise AssertionError('ordinary owner descriptor must not hash model bytes')
        return original(tensor, **kwargs)
    monkeypatch.setattr(runtime, 'tensor_signature', signature)
    rng = torch.random.get_rng_state()
    assert inventory({'owner': owner}) == before
    before['owner']['descriptor']['identity']['source'] = 'changed cached JSON'
    assert inventory({'owner': owner})['owner']['descriptor']['identity']['source'] == 'test'
    assert torch.equal(rng, torch.random.get_rng_state())


def _local_state_audit_worker(rank, rendezvous):
    from types import SimpleNamespace
    from datetime import timedelta
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, Partial
    from distillschedule.open_workload.state_audit import audit_scope, distributed_audit
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        def forbidden(*args, **kwargs):
            raise AssertionError('cycle auditing must not gather a full DTensor')
        DTensor.full_tensor = forbidden
        mesh = init_device_mesh('cpu', (2,), mesh_dim_names=('shard',))
        for length in (1, 3):
            width = (length + 1) // 2
            size = max(0, min(width, length - rank * width))
            tensor = DTensor.from_local(torch.full((size,), rank + 1.), mesh, [Shard(0)],
                                        shape=torch.Size((length,)), stride=(1,))
            session = SimpleNamespace(state={'weight': tensor}, _audit_cycle=1,
                                      state_audit_every_cycles=50, effects={})
            with audit_scope(cycle=1):
                assert distributed_audit(session)['kind'] == 'content_audit'
        session.state = {'weight': torch.ones(2 if rank else 1)}
        with audit_scope(cycle=2), pytest.raises(ValueError, match='metadata diverged'):
            distributed_audit(session)
        session.state = {'weight': torch.ones(1)}
        session.state_audit_every_cycles = rank + 1
        with audit_scope(cycle=2), pytest.raises(ValueError, match='cadence'):
            distributed_audit(session)
        session.state_audit_every_cycles = 50
        from distillschedule.workloads.program import Program, Call
        session.workload = SimpleNamespace(program=Program((Call('read', 'read', reads=('weight',)),),
                                                           {'weight': rank}))
        with audit_scope(cycle=2), pytest.raises(ValueError, match='logical program clocks'):
            distributed_audit(session)
        del session.workload
        session.state = {'weight': DTensor.from_local(torch.ones(1), mesh,
                         [Partial() if rank else Shard(0)], shape=torch.Size((2,)), stride=(1,))}
        with audit_scope(cycle=2), pytest.raises(ValueError, match='rank failures'):
            distributed_audit(session)
    finally:
        dist.destroy_process_group()


def test_local_shard_audits_cover_empty_uneven_and_collective_failures(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    torch.multiprocessing.spawn(_local_state_audit_worker, args=(str(tmp_path / 'local-audit'),),
                                nprocs=2, join=True)


def _hsdp_state_audit_worker(rank, rendezvous):
    from types import SimpleNamespace
    from datetime import timedelta
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard
    from distillschedule.open_workload.state_audit import audit_scope, distributed_audit
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=30))
    try:
        mesh = init_device_mesh('cpu', (2, 2), mesh_dim_names=('replicate', 'shard'))
        local = torch.tensor([float(rank % 2)])
        tensor = DTensor.from_local(local, mesh, [Replicate(), Shard(0)],
                                    shape=torch.Size((2,)), stride=(1,))
        def forbidden(*args, **kwargs):
            raise AssertionError('HSDP audits must compare matching local shards')
        DTensor.full_tensor = forbidden
        session = SimpleNamespace(state={'weight': tensor}, _audit_cycle=1,
                                  state_audit_every_cycles=50, effects={})
        with audit_scope(cycle=1):
            distributed_audit(session)  # Distinct shard columns legitimately differ.
        if rank == 2:
            tensor.to_local().data.add_(1)
        with audit_scope(cycle=50), pytest.raises(ValueError, match='contents diverged'):
            distributed_audit(session)
    finally:
        dist.destroy_process_group()


def test_hsdp_content_audit_compares_only_corresponding_replica_shards(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    torch.multiprocessing.spawn(_hsdp_state_audit_worker, args=(str(tmp_path / 'hsdp-audit'),),
                                nprocs=4, join=True)


class _AuditCollectiveOwner:
    def __init__(self, rank):
        self.weight = torch.nn.Parameter(torch.ones(8192))
        self.local_rng = torch.tensor([rank, rank + 1], dtype=torch.int64)
        self.clock = 0
        self.prepared = False

    def state_dict(self):
        return {'weight': self.weight, 'local_rng': self.local_rng, 'clock': self.clock}

    def load_state_dict(self, value):
        pass

    def runtime_tensors(self):
        yield self.weight
        yield self.local_rng

    def collective_execution_spec(self):
        return dict(schema_version='distillschedule.native_collective_owner.v1', backend='ddp',
                    world_size=torch.distributed.get_world_size(), device_type='cpu',
                    shared_identity={'math': 'test-native-clock-v1'})

    def prepare_collective_execution(self):
        assert self.prepared is False
        self.prepared = True

    def validate_collective_execution(self):
        return dict(world_size=torch.distributed.get_world_size(), rank=torch.distributed.get_rank(),
                    bucket_storage_bound_bytes=0, roles={'native': {'updates': self.clock}}, zero_gradients=True)


def _audit_native_call(ctx):
    owner = ctx.state['owner']
    with torch.no_grad():
        owner.weight.add_(.01)
    owner.clock += 1
    return owner.clock


def _native_state_audit_worker(rank, rendezvous):
    from dataclasses import replace
    from datetime import timedelta
    import torch.distributed as dist
    from distillschedule.open_workload.execution import ExecutionBackend
    from distillschedule.open_workload.torch_backend import Workload
    import distillschedule.open_workload.torch_backend as runtime
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        owner = _AuditCollectiveOwner(rank)
        program = Program((Call('native', 'native', reads=('owner',), writes=('owner',),
                                grad='external_grad'),), {'owner': 0}, outputs=('native',))
        workload = Workload(program, {'native': _audit_native_call}, {'owner': owner}, {})
        backend = ExecutionBackend(workload, state_audit_every_cycles=3)
        session = backend.session(backend.inventory(1)['candidates'][0], copy_state=False, persistent=True)
        for cycle in range(3):
            if cycle:
                workload = replace(workload, program=replace(program,
                    initial_versions=session.receipt['dispatcher']['final_versions']))
                session.bind_cycle(workload)
            original = runtime.tensor_signature
            if cycle == 1:
                def forbidden(tensor, **kwargs):
                    if tensor.numel() == 8192:
                        raise AssertionError('ordinary native cycle must not hash weight payloads')
                    return original(tensor, **kwargs)
                runtime.tensor_signature = forbidden
            assert session.run()['native'] == cycle + 1
            runtime.tensor_signature = original
            assert session.effects['state_audit']['replica_content_consensus'] is False
            assert session.effects['state_audit']['kind'] == ('metadata' if cycle == 1 else 'content_audit')
        assert owner.prepared and owner.clock == 3
        assert session.audit_state()['scope'] == 'native_owner_rank_local'
    finally:
        dist.destroy_process_group()


def test_native_owner_purity_preflight_is_local_and_does_not_rehash_each_cycle(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    torch.multiprocessing.spawn(_native_state_audit_worker, args=(str(tmp_path / 'native-audit'),),
                                nprocs=2, join=True)


def _audit_role_model(role, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3 if role == 'generator' else 7)
        return torch.nn.Linear(1, 1).to(device)


def _audit_role_optimizer(role, model):
    return torch.optim.SGD(model.parameters(), lr=.01, momentum=.9) if role == 'generator' else None


def _audit_role_sample(index, role, device):
    return torch.ones(1, 1, device=device)


def _audit_role_sample_at(cycle, index, role, device):
    return torch.full((1, 1), 1. + cycle * .1, device=device)


def _audit_role_sample_identity(cycle, index, role, actual):
    # This unconditioned fixture has role-private numeric targets, no shared
    # prompt/conditioning; both roles explicitly consume the same logical item.
    return {'sample_id': f'{cycle}:{index}', 'conditions': {}}


def _scaled_role_bindings(scale, mode='closure'):
    from distillschedule.open_workload.role_transport import RoleFunctions
    if mode == 'closure':
        def loss(model, source, sample):
            return scale * _audit_role_loss(model, source, sample)
    elif mode == 'default':
        def loss(model, source, sample, factor=scale):
            return factor * _audit_role_loss(model, source, sample)
    else:
        def loss(model, source, sample, *, factor=scale):
            return factor * _audit_role_loss(model, source, sample)
    return RoleFunctions(_audit_role_model, _audit_role_optimizer, _audit_role_sample,
        _audit_role_forward, loss, data_identity='d' * 64, sample_identity=_audit_role_sample_identity)


@pytest.mark.parametrize('mode', ['closure', 'default', 'keyword_default'])
def test_role_plan_identity_changes_with_actual_callable_configuration(mode):
    from distillschedule.open_workload.role_planning import code_identity
    assert code_identity(_scaled_role_bindings(2, mode)) != code_identity(_scaled_role_bindings(7, mode))
    assert code_identity(_scaled_role_bindings(2, mode)) == code_identity(_scaled_role_bindings(2, mode))


def test_role_plan_identity_rejects_opaque_dynamic_capture_without_reading_tensor_payload(monkeypatch):
    from distillschedule.open_workload.role_planning import code_identity
    def forbidden(*args, **kwargs):
        raise AssertionError('identity must not copy tensor payload')
    monkeypatch.setattr(torch.Tensor, 'cpu', forbidden)
    with pytest.raises(ValueError, match='dynamic|capture|configuration'):
        code_identity(_scaled_role_bindings(torch.tensor(2.)))


_ROLE_LOSS_CONFIGURATION = {'scale': 2}


def _configured_global_role_loss(model, source, sample):
    return _ROLE_LOSS_CONFIGURATION['scale'] * _audit_role_loss(model, source, sample)


def _nested_global_role_loss(model, source, sample):
    return sum(_ROLE_LOSS_CONFIGURATION['scale'] * item.square().sum() for item in (source,))


class _ConfiguredRoleModel(torch.nn.Linear):
    def __init__(self):
        super().__init__(1, 1)

    def forward(self, value):
        return super().forward(value) * _ROLE_LOSS_CONFIGURATION['scale']


class _PropertyRoleModel(torch.nn.Linear):
    @property
    def scale(self):
        return _ROLE_LOSS_CONFIGURATION['scale']

    def forward(self, value):
        return super().forward(value) * self.scale


class _PlainRoleConfiguration:
    scale = 2


def _configured_role_model(role, device):
    return _ConfiguredRoleModel().to(device)


def _property_role_model(role, device):
    return _PropertyRoleModel(1, 1).to(device)


def _plain_class_role_loss(model, source, sample):
    return _PlainRoleConfiguration.scale * source.square().sum()


def _static_config_role_loss(model, source, sample):
    return _ROLE_CONFIG_MODULE.scale * source.sum()


def _dynamic_config_role_loss(model, source, sample):
    return getattr(_ROLE_CONFIG_MODULE, 'scale') * source.sum()


@pytest.mark.parametrize('callback', ['nested', 'class', 'property'])
def test_role_identity_binds_nested_code_globals_and_local_model_methods(callback):
    from dataclasses import replace
    from distillschedule.open_workload.role_planning import code_identity
    bindings = replace(_scaled_role_bindings(2), **(
        {'score_loss': _nested_global_role_loss} if callback == 'nested'
        else {'make_model': _configured_role_model if callback == 'class' else _property_role_model}))
    before = code_identity(bindings)
    try:
        _ROLE_LOSS_CONFIGURATION['scale'] = 7
        assert code_identity(bindings) != before
    finally:
        _ROLE_LOSS_CONFIGURATION['scale'] = 2


def test_role_identity_accepts_plain_local_classes_and_binds_actual_constants(monkeypatch):
    from dataclasses import replace
    from distillschedule.open_workload.role_planning import code_identity
    bindings = replace(_scaled_role_bindings(2), score_loss=_plain_class_role_loss)
    before = code_identity(bindings)
    monkeypatch.setattr(_PlainRoleConfiguration, 'scale', 7)
    assert code_identity(bindings) != before


def test_role_identity_binds_actual_static_module_attributes(monkeypatch):
    from dataclasses import replace
    from types import ModuleType
    from distillschedule.open_workload.role_planning import code_identity
    settings = ModuleType('role_static_config_fixture')
    settings.scale = 2
    monkeypatch.setitem(globals(), '_ROLE_CONFIG_MODULE', settings)
    bindings = replace(_scaled_role_bindings(2), score_loss=_static_config_role_loss)
    before = code_identity(bindings)
    settings.scale = 7
    assert code_identity(bindings) != before


def test_role_identity_rejects_dynamic_module_configuration_lookup(monkeypatch):
    from dataclasses import replace
    from types import ModuleType
    from distillschedule.open_workload.role_planning import code_identity
    settings = ModuleType('role_config_fixture')
    settings.scale = 2
    monkeypatch.setitem(globals(), '_ROLE_CONFIG_MODULE', settings)
    with pytest.raises(ValueError, match='dynamic.*configuration'):
        code_identity(replace(_scaled_role_bindings(2), score_loss=_dynamic_config_role_loss))


def test_role_plan_identity_binds_actual_global_configuration():
    from dataclasses import replace
    from distillschedule.open_workload.role_planning import code_identity
    bindings = replace(_scaled_role_bindings(2), score_loss=_configured_global_role_loss)
    before = code_identity(bindings)
    try:
        _ROLE_LOSS_CONFIGURATION['scale'] = 7
        assert code_identity(bindings) != before
    finally:
        _ROLE_LOSS_CONFIGURATION['scale'] = 2


_ROLE_FACTORY_SOURCE = '''import torch
from torch import nn
from distillschedule.open_workload.role_program import RoleTask
from distillschedule.open_workload.role_transport import RoleFunctions

def make_model(role,device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3 if role=='generator' else 7)
        return nn.Linear(1,1).to(device)

def make_optimizer(role,module):
    return torch.optim.Adam(module.parameters(),lr=.001) if role=='generator' else None

def sample(index,role,device):
    return {'sample_id':str(index),'value':torch.ones(1,1,device=device)}

def sample_at(cycle,index,role,device):
    return {'sample_id':f'{cycle}:{index}','value':torch.full((1,1),1.+.1*cycle,device=device)}

def identity(cycle,index,role,actual):
    return {'sample_id':actual['sample_id'],'conditions':{}}

def forward(module,actual):return module(actual['value'])

def build(parameters):
    scale=parameters['scale']
    parameters.pop('note',None)
    def loss(module,latent,actual):
        return scale*(module(latent)-actual['value']).square().mean()
    return RoleTask(3,(1,1)),RoleFunctions(make_model,make_optimizer,sample,forward,loss,
        sample_at=sample_at,sample_identity=identity,data_identity='d'*64,checkpoint_safe=True)
'''


def test_role_factory_binds_all_requested_parameters_without_mutating_caller(tmp_path):
    from distillschedule.open_workload.role_cli import load_factory
    from distillschedule.open_workload.role_planning import code_identity
    from distillschedule.workloads.best_known.registry import fingerprint
    source = tmp_path / 'factory.py'
    source.write_text(_ROLE_FACTORY_SOURCE)
    parameters = {'scale': 2, 'note': 'original'}
    _, first, metadata = load_factory(str(source) + ':build', parameters)
    _, second, _ = load_factory(str(source) + ':build', {'scale': 2, 'note': 'changed'})
    assert parameters == {'scale': 2, 'note': 'original'}
    assert metadata['parameters_sha256'] == fingerprint(parameters)
    assert first.factory_identity == metadata
    assert code_identity(first) != code_identity(second)


def _role_plan_restart_worker(rank, rendezvous, output, mode):
    from datetime import timedelta
    from pathlib import Path
    import os
    import random
    import numpy as np
    import torch.distributed as dist
    from distillschedule.open_workload.role_cli import load_factory
    from distillschedule.open_workload.role_planning import calibrate, search, run_selected, _run_identity
    from distillschedule.open_workload.role_program import AllocatedWorld
    from distillschedule.training.checkpoint import pack_state, rng_state
    from distillschedule.workloads.best_known.registry import write_new_json
    os.environ.update(RANK=str(rank), WORLD_SIZE='2', LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE='2', GROUP_RANK='0')
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        root = Path(output)
        scale = 7 if mode == 'changed' or mode == 'ranks' and rank == 1 else 2
        task, bindings, _ = load_factory(str(root / 'factory.py') + ':build', {'scale': scale})
        hardware = AllocatedWorld((1 << 30,) * 2)
        random.seed(23); np.random.seed(23); torch.manual_seed(23)
        if mode == 'ranks':
            with pytest.raises(ValueError, match='configuration identity'):
                _run_identity(task, hardware, bindings, 'cpu')
            return
        if mode == 'calibrate':
            calibrate(task, hardware, bindings, root / 'calibration', generator_bytes=10000,
                      score_bytes=10000, max_candidates=1, samples=1, max_native_runs=2)
            if rank == 0:
                write_new_json(root / 'plan.json', search(root / 'calibration/calibration.json'))
            dist.barrier()
            return
        if mode == 'changed':
            with pytest.raises(ValueError, match='identity changed'):
                run_selected(task, hardware, bindings, root / 'plan.json', root / mode)
            assert not (root / mode).exists()
            return
        import json
        previous = json.loads((root / 'segment/execution.json').read_text()) if mode == 'resumed' else None
        receipt, model, optimizer = run_selected(task, hardware, bindings, root / 'plan.json', root / mode,
            cycles=1 if mode == 'segment' else 2,
            checkpoint_policy={'checkpoint_every': 1} if mode == 'segment' else None,
            resume_from=previous['checkpoint_path'] if previous else None)
        torch.save({'state': pack_state({'model': model, 'optimizer': optimizer}), 'rng': rng_state(),
                    'cycle': receipt['completed_cycles']}, root / f'{mode}-{rank}.pt')
        if mode == 'resumed':
            assert receipt['resumed_from_cycle'] == 1 and receipt['cycles_executed'] == 1
    finally:
        dist.destroy_process_group()


def test_saved_role_plan_rejects_changed_config_and_resumes_in_fresh_processes(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    pytest.importorskip('numpy')
    (tmp_path / 'factory.py').write_text(_ROLE_FACTORY_SOURCE)
    for mode in ('ranks', 'calibrate', 'changed', 'segment', 'resumed', 'reference'):
        torch.multiprocessing.spawn(_role_plan_restart_worker,
            args=(str(tmp_path / (mode + '-init')), str(tmp_path), mode), nprocs=2, join=True)
    for rank in range(2):
        expected = torch.load(tmp_path / f'reference-{rank}.pt', weights_only=True)
        resumed = torch.load(tmp_path / f'resumed-{rank}.pt', weights_only=True)
        _assert_role_tree_equal(expected, resumed)


def _audit_role_forward(model, sample):
    return model(sample)


def _audit_role_loss(model, source, sample):
    return (model(source) - sample).square().mean()


def _role_pairing_worker(rank, rendezvous, fault):
    from datetime import timedelta
    import torch.distributed as dist
    from distillschedule.open_workload.role_program import RoleTask, PoolLayout, AllocatedWorld
    from distillschedule.open_workload.role_transport import RoleFunctions, RolePoolExecutor
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=10))
    executor = None
    try:
        calls, identities = [], []
        def model(role, device):
            result = torch.nn.Linear(1, 1, bias=False).to(device)
            with torch.no_grad():
                result.weight.fill_(1)
            return result
        def optimizer(role, module):
            return torch.optim.SGD(module.parameters(), lr=.1)
        def sample(index, role, device):
            calls.append((index, role))
            # The second Score sample is for its independent parameter update.
            update = role == 'score' and len(calls) > 3
            prompt = 'B' if role == 'score' and fault == 'prompt' else ('C' if update else 'A')
            return {'sample_id': str(index + (10 if role == 'score' and fault == 'sample' else 0)),
                    'prompt_id': prompt, 'value': torch.ones(1, 1, device=device),
                    'target': torch.full((1, 1), 2. if prompt == 'B' else 1., device=device)}
        def identity(cycle, index, role, actual):
            identities.append((cycle, index, role))
            if role == 'score' and fault == 'callback':
                raise RuntimeError('identity-sentinel')
            if role == 'score' and fault == 'missing':
                return {'sample_id': actual['sample_id']}
            return {'sample_id': actual['sample_id'], 'conditions': {'prompt': actual['prompt_id']}}
        def forward(module, actual):
            return module(actual['value'])
        def loss(module, latent, actual):
            return (module(latent) - actual['target']).square().mean()
        bindings = RoleFunctions(model, optimizer, sample, forward, loss, loss, sample_identity=identity)
        executor = RolePoolExecutor(RoleTask(3, (1, 1), score_updates=1),
            PoolLayout((0,), (1,), 2), AllocatedWorld((1 << 30,) * 2), bindings, device='cpu')
        if fault is not None:
            with pytest.raises(ValueError, match='sample pairing') as error:
                executor.run()
            errors = [None, None]
            dist.all_gather_object(errors, str(error.value))
            assert errors[0] == errors[1] and 'Timed out' not in errors[0]
            assert executor.module.weight.item() == 1 and executor._cycle == 0 and executor._failed
            if fault == 'callback':
                assert 'identity-sentinel' in errors[0]
            return
        receipt, _, _ = executor.run()
        assert len(identities) == 3 and len(calls) == (3 if rank == 0 else 6)
        assert len(receipt['sample_pairing']['pairs']) == 3
        assert receipt['sample_pairing']['checked_before_updates'] is True
    finally:
        if executor is not None:
            executor.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('fault', ['prompt', 'sample', 'callback', 'missing', None])
def test_role_pool_pairs_actual_sample_conditions_before_any_update(tmp_path, fault):
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    torch.multiprocessing.spawn(_role_pairing_worker,
        args=(str(tmp_path / 'pairing-init'), fault), nprocs=2, join=True)


def _role_state_audit_worker(rank, rendezvous):
    from datetime import timedelta
    import torch.distributed as dist
    import distillschedule.open_workload.torch_backend as runtime
    from distillschedule.open_workload.role_program import RoleTask, PoolLayout, AllocatedWorld
    from distillschedule.open_workload.role_transport import RoleFunctions, RolePoolExecutor
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=30))
    executor = None
    try:
        task = RoleTask(2, (1, 1))
        layout = PoolLayout((0, 1), (2, 3), 1)
        bindings = RoleFunctions(_audit_role_model, _audit_role_optimizer, _audit_role_sample,
                                _audit_role_forward, _audit_role_loss, sample_at=_audit_role_sample_at,
                                sample_identity=_audit_role_sample_identity)
        executor = RolePoolExecutor(task, layout, AllocatedWorld((1 << 30,) * 4), bindings,
                                    device='cpu', state_audit_every_cycles=3)
        for cycle in range(3):
            original = runtime.tensor_signature
            if cycle == 1:
                def forbidden(*args, **kwargs):
                    raise AssertionError('ordinary role cycle must not hash tensors')
                runtime.tensor_signature = forbidden
            receipt, _, _ = executor.run()
            runtime.tensor_signature = original
            for row in receipt['ranks']:
                assert row['state_audit']['kind'] == ('metadata' if cycle == 1 else 'content_audit')
                assert (row['model_sha256'] is None) is (cycle == 1)
        assert executor._cycle == 3
    finally:
        if executor is not None:
            executor.close()
        dist.destroy_process_group()


def test_role_pool_persistent_cycles_do_not_hash_full_model_each_time(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    torch.multiprocessing.spawn(_role_state_audit_worker, args=(str(tmp_path / 'role-audit'),),
                                nprocs=4, join=True)


class _RoleFaultModel(torch.nn.Module):
    def __init__(self, role, fault):
        super().__init__()
        self.role, self.fault = role, fault
        initial = 0. if fault == role + '_gradient' or (role == 'generator' and fault == 'reduction_overflow') else 1.
        self.weight = torch.nn.Parameter(torch.tensor([initial]))

    def to(self, *args, **kwargs):
        if self.role == 'generator' and self.fault == 'construction_move':
            raise ValueError('local-construction-sentinel')
        return super().to(*args, **kwargs)


class _RoleFaultOptimizer(torch.optim.SGD):
    def __init__(self, model):
        super().__init__(model.parameters(), lr=.01, momentum=.9)
        self.role, self.fault, self.steps = model.role, model.fault, 0

    def step(self, *args, **kwargs):
        value = super().step(*args, **kwargs)
        self.steps += 1
        parameter = self.param_groups[0]['params'][0]
        with torch.no_grad():
            if self.fault == self.role + '_parameter':
                parameter.fill_(float('inf'))
            elif self.fault == self.role + '_optimizer':
                self.state[parameter]['invalid_slot'] = torch.full_like(parameter, float('inf'))
        return value


def _role_fault_worker(rank, rendezvous, output, fault):
    import json
    from datetime import timedelta
    from pathlib import Path
    import torch.distributed as dist
    from distillschedule.open_workload.role_program import RoleTask, PoolLayout, AllocatedWorld
    from distillschedule.open_workload.role_transport import RoleFunctions, RolePoolExecutor
    world = 3 if fault == 'reduction_overflow' else 2
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(seconds=8))
    executor = None
    try:
        def make_model(role, device):
            if role == 'generator' and fault == 'construction_model':
                raise ValueError('local-construction-sentinel')
            return _RoleFaultModel(role, fault)
        def make_optimizer(role, model):
            if role == 'generator' and fault == 'construction_optimizer':
                raise ValueError('local-construction-sentinel')
            return _RoleFaultOptimizer(model)
        def forward(model, sample):
            if fault == 'reduction_overflow':
                return (model.weight.double().reshape(1, 1) * 3.6e38).float()
            value = model.weight.sqrt() if fault == 'generator_gradient' else model.weight
            return value.reshape(1, 1) * sample
        def score_loss(model, source, sample):
            return (source * model.weight).sum()
        def train_loss(model, source, sample):
            return model.weight.sqrt().sum() if fault == 'score_gradient' else (model.weight - 2).square().sum()
        score_updates = int(fault.startswith('score_'))
        bindings = RoleFunctions(make_model, make_optimizer, _audit_role_sample, forward,
                                score_loss, train_loss if score_updates else None,
                                sample_identity=_audit_role_sample_identity)
        layout = PoolLayout((0, 1), (2,), 1) if world == 3 else PoolLayout((0,), (1,), 1)
        executor = RolePoolExecutor(RoleTask(world - 1, (1, 1), score_updates=score_updates),
                                    layout, AllocatedWorld((1 << 30,) * world),
                                    bindings, device='cpu')
        try:
            executor.run()
            error = None
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        result = dict(error=error, cycle=executor._cycle, failed=executor._failed,
                      versions=executor._versions,
                      steps=None if executor.optimizer is None else executor.optimizer.steps)
        (Path(output) / f'{rank}.json').write_text(json.dumps(result))
    finally:
        if executor is not None:
            executor.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('fault', ['generator_gradient', 'score_gradient',
                                 'generator_parameter', 'generator_optimizer',
                                 'score_parameter', 'score_optimizer'])
def test_role_pool_nonfinite_gradients_or_committed_state_fail_on_all_ranks(tmp_path, fault):
    import json
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    torch.multiprocessing.spawn(_role_fault_worker,
        args=(str(tmp_path / 'fault-init'), str(tmp_path), fault), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(2)]
    assert all(row['error'] is not None and 'non-finite' in row['error'] for row in rows), rows
    assert rows[0]['error'] == rows[1]['error']
    assert all(row['cycle'] == 0 and row['failed'] for row in rows)
    assert all(not any(row['versions'].values()) for row in rows)
    if fault == 'generator_gradient':
        assert rows[0]['steps'] == 0
    elif fault == 'score_gradient':
        assert rows[1]['steps'] == 0


@pytest.mark.parametrize('fault', ['construction_model', 'construction_move', 'construction_optimizer'])
def test_role_pool_local_construction_failure_is_exchanged_before_collectives(tmp_path, fault):
    import json
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    torch.multiprocessing.spawn(_role_fault_worker,
        args=(str(tmp_path / 'construction-init'), str(tmp_path), fault), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(2)]
    assert all(row['error'] is not None and 'role local construction' in row['error']
               and 'local-construction-sentinel' in row['error'] for row in rows), rows
    assert rows[0]['error'] == rows[1]['error']
    assert all(row['cycle'] == 0 and row['failed'] for row in rows)


def test_role_pool_rejects_overflow_in_gradient_sum_before_any_step(tmp_path):
    import json
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    torch.multiprocessing.spawn(_role_fault_worker,
        args=(str(tmp_path / 'overflow-init'), str(tmp_path), 'reduction_overflow'), nprocs=3, join=True)
    rows = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(3)]
    assert len({row['error'] for row in rows}) == 1
    assert all('reduced parameter gradient admission' in row['error'] and row['cycle'] == 0
               and row['failed'] for row in rows)
    assert all(row['steps'] == 0 for row in rows[:2])


def _role_update_reference_worker(rank, rendezvous):
    from datetime import timedelta
    import torch.distributed as dist
    from distillschedule.open_workload.role_program import RoleTask, PoolLayout, AllocatedWorld
    from distillschedule.open_workload.role_transport import RoleFunctions, RolePoolExecutor
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=3,
                            timeout=timedelta(seconds=20))
    executor = None
    try:
        def optimizer(role, model):
            return torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        def update_input(context):
            return context.retained_latent, context.conditions
        bindings = RoleFunctions(_audit_role_model, optimizer, _audit_role_sample, _audit_role_forward,
            _audit_role_loss, _audit_role_loss, sample_at=_audit_role_sample_at,
            score_update_input=update_input, sample_identity=_audit_role_sample_identity)
        executor = RolePoolExecutor(RoleTask(3, (1, 1), score_updates=2),
                                    PoolLayout((0, 1), (2,), 2), AllocatedWorld((1 << 30,) * 3),
                                    bindings, device='cpu')
        generator = _audit_role_model('generator', torch.device('cpu'))
        score = _audit_role_model('score', torch.device('cpu'))
        optimizers = {role: optimizer(role, model) for role, model in
                      (('generator', generator), ('score', score))}
        for cycle in range(2):
            retained = []
            for index in range(3):
                latent = generator(_audit_role_sample_at(cycle, index, 'generator', 'cpu'))
                condition = _audit_role_sample_at(cycle, index, 'score', 'cpu')
                vjp, = torch.autograd.grad(_audit_role_loss(score, latent, condition) / 3, latent)
                latent.backward(vjp)
                retained.append((latent.detach(), condition))
            optimizers['generator'].step(); optimizers['generator'].zero_grad(set_to_none=True)
            for _ in range(2):
                for latent, condition in retained:
                    (_audit_role_loss(score, latent, condition) / 3).backward()
                optimizers['score'].step(); optimizers['score'].zero_grad(set_to_none=True)
            executor.run()
            reference = generator if executor.role == 'generator' else score
            for actual, expected in zip(executor.module.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            actual_slots = executor.optimizer.state_dict()['state']
            expected_slots = optimizers[executor.role].state_dict()['state']
            for key in actual_slots:
                torch.testing.assert_close(actual_slots[key]['momentum_buffer'],
                                           expected_slots[key]['momentum_buffer'], rtol=1e-6, atol=1e-7)
    finally:
        if executor is not None:
            executor.close()
        dist.destroy_process_group()


def test_role_pool_uneven_replicas_repeated_score_updates_keep_original_math(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    torch.multiprocessing.spawn(_role_update_reference_worker,
        args=(str(tmp_path / 'update-reference-init'),), nprocs=3, join=True)


@pytest.mark.parametrize('updating', [True, False])
def test_role_gradient_admission_does_not_discard_frozen_finite_cache(monkeypatch, updating):
    from distillschedule.open_workload.role_transport import RolePoolExecutor
    from distillschedule.training.finite import FiniteCheck
    model = torch.nn.Linear(1, 1)
    model.register_parameter('base', torch.nn.Parameter(torch.ones(4), requires_grad=False))
    optimizer = torch.optim.SGD(model.parameters(), lr=.01) if updating else None
    executor = object.__new__(RolePoolExecutor)
    executor.module = model
    executor._finite_state, executor._finite_gradients = FiniteCheck(), FiniteCheck()
    original = torch.isfinite
    base_reads, weight_reads = [], []
    def observed(value, *args, **kwargs):
        if value is model.base:
            base_reads.append(1)
        if value is model.weight:
            weight_reads.append(1)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch, 'isfinite', observed)
    executor._check_state(model, optimizer)
    model(torch.ones(1, 1)).sum().backward()
    executor._check_gradients()
    if optimizer is not None:
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    executor._check_state(model, optimizer)
    assert len(base_reads) == 1
    assert len(weight_reads) == (2 if updating else 1)
    with torch.no_grad():
        model.base.add_(1)
    executor._check_state(model, optimizer)
    assert len(base_reads) == 2


@pytest.mark.parametrize('flag,value', [('_busy', True), ('_failed', True), ('_closed', True),
                                        ('_prepared', False), ('_active_offloaders', [object()])])
def test_role_checkpoint_rejects_nonboundary_owner(flag, value):
    from types import SimpleNamespace
    from distillschedule.open_workload.role_checkpoint import _Progress
    owner = SimpleNamespace(_busy=False, _failed=False, _closed=False,
                            _prepared=True, _active_offloaders=[])
    setattr(owner, flag, value)
    with pytest.raises(ValueError, match='boundary|leases'):
        _Progress(owner).validate_checkpoint_boundary()


class _RoleDataStream:
    """Actual private sample RNG/cursor used by the CPU restart fixture."""
    def __init__(self):
        self.rng = torch.Generator().manual_seed(123)
        self.cursor = 0
        self.fail_load = False

    def state_dict(self):
        return {'rng': self.rng.get_state(), 'cursor': self.cursor}

    def validate_state_dict(self, state):
        assert set(state) == {'rng', 'cursor'} and type(state['cursor']) is int and state['cursor'] >= 0
        torch.Generator().set_state(state['rng'])

    def load_state_dict(self, state):
        if self.fail_load:
            raise RuntimeError('owner-load-sentinel')
        self.validate_state_dict(state)
        self.rng.set_state(state['rng'])
        self.cursor = state['cursor']


class _RoleTrainingExtras:
    def __init__(self, module, optimizer):
        self.module = module
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.97)
        self.ema = {name: p.detach().clone() for name, p in module.named_parameters()}
        self.steps = 0

    def advance(self):
        self.scheduler.step()
        with torch.no_grad():
            for name, parameter in self.module.named_parameters():
                self.ema[name].lerp_(parameter, .1)
        self.steps += 1

    def state_dict(self):
        return dict(scheduler=self.scheduler.state_dict(), ema=self.ema, steps=self.steps)

    def validate_state_dict(self, state):
        assert set(state) == {'scheduler', 'ema', 'steps'} and type(state['steps']) is int and state['steps'] >= 0
        assert state['ema'].keys() == self.ema.keys()
        for name, tensor in state['ema'].items():
            assert tensor.shape == self.ema[name].shape and tensor.dtype == self.ema[name].dtype

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.scheduler.load_state_dict(state['scheduler'])
        for name, tensor in state['ema'].items():
            self.ema[name].copy_(tensor)
        self.steps = state['steps']


def _role_restart_worker(rank, rendezvous, output, mode):
    import os
    import random
    from datetime import timedelta
    from pathlib import Path
    import numpy as np
    import torch.distributed as dist
    from distillschedule.open_workload.role_program import RoleTask, PoolLayout, AllocatedWorld
    from distillschedule.open_workload.role_transport import RoleFunctions, RolePoolExecutor
    from distillschedule.open_workload.role_checkpoint import RoleCheckpoint
    from distillschedule.training.checkpoint import pack_state, rng_state
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE='2', GROUP_RANK='0')
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    executor = None
    try:
        stream = _RoleDataStream()
        training = {}
        def sample_at(cycle, index, role, device):
            stream.cursor += 1
            value = random.random() + float(np.random.rand()) + torch.rand(()).item()
            value += torch.rand((), generator=stream.rng).item()
            return torch.full((1, 1), .1 * value + .01 * (cycle + index), device=device)
        def make_optimizer(role, model):
            optimizer = torch.optim.Adam(model.parameters(), lr=.002)
            owner = _RoleTrainingExtras(model, optimizer)
            training['owner'] = owner
            optimizer.register_step_post_hook(lambda *args: owner.advance())
            return optimizer
        def owners(role, module, optimizer):
            return {'data': stream, 'training': training['owner']}
        def score_input(context):
            return context.retained_latent, context.conditions
        bindings = RoleFunctions(_audit_role_model, make_optimizer, _audit_role_sample,
            _audit_role_forward, _audit_role_loss, _audit_role_loss,
            data_identity='d' * 64, sample_at=sample_at, score_update_input=score_input,
            checkpoint_safe=True, state_owners=owners, sample_identity=_audit_role_sample_identity)
        task = RoleTask(3, (1, 1), score_updates=2)
        layout = PoolLayout((0,), (1,), 2)
        if mode == 'reject-data':
            from dataclasses import replace
            bindings = replace(bindings, data_identity='a' * 64)
        elif mode == 'reject-layout':
            layout = PoolLayout((1,), (0,), 2)
        executor = RolePoolExecutor(task, layout,
                                    AllocatedWorld((1 << 30,) * 2), bindings, device='cpu')
        checkpoint = RoleCheckpoint(executor, {'plan_sha256': 'e' * 64,
                                               'identity': {'source': 'SYNTHETIC_TEST_ONLY'}})
        root = Path(output)
        random.seed(400 + rank); np.random.seed(500 + rank); torch.manual_seed(600 + rank)
        if mode.startswith('reject-'):
            if mode == 'reject-load':
                stream.fail_load = rank == 1
            if mode == 'reject-rng':
                from distillschedule.training import checkpoint as checkpoint_module
                restore_rng = checkpoint_module.restore_rng
                def fail_rng(value):
                    if rank == 1:
                        raise RuntimeError('rng-install-sentinel')
                    restore_rng(value)
                checkpoint_module.restore_rng = fail_rng
            before = pack_state(checkpoint.state)
            before_rng = rng_state()
            with pytest.raises(ValueError) as error:
                checkpoint.restore(root / 'segment-checkpoint')
            if mode in ('reject-load', 'reject-rng'):
                label, sentinel = (('checkpoint state restore', 'owner-load-sentinel') if mode == 'reject-load'
                                   else ('checkpoint RNG restore', 'rng-install-sentinel'))
                assert label in str(error.value) and sentinel in str(error.value)
                assert executor._failed
                with pytest.raises(ValueError, match='failed'):
                    executor.run()
                return
            _assert_role_tree_equal(before, pack_state(checkpoint.state))
            _assert_role_tree_equal(before_rng, rng_state())
            assert executor._cycle == 0 and stream.cursor == 0
            return
        if mode == 'resume':
            checkpoint.restore(root / 'segment-checkpoint')
            assert executor._cycle == 1
        target = 1 if mode == 'segment' else 3
        while executor._cycle < target:
            executor.run()
        checkpoint.save(root / (mode + '-checkpoint'))
        torch.save({'state': pack_state(checkpoint.state), 'rng': rng_state(),
                    'cycle': executor._cycle}, root / f'{mode}-{rank}.pt')
    finally:
        if executor is not None:
            executor.close()
        dist.destroy_process_group()


def _assert_role_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_role_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_role_tree_equal(a, b)
    else:
        assert left == right


def test_role_pool_fresh_process_resume_keeps_models_adam_rng_data_and_versions(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    pytest.importorskip('numpy')
    # Three genuinely independent process lifetimes, not in-process state reuse.
    for mode in ('reference', 'segment', 'resume'):
        torch.multiprocessing.spawn(_role_restart_worker,
            args=(str(tmp_path / (mode + '-init')), str(tmp_path), mode), nprocs=2, join=True)
    for rank in range(2):
        reference = torch.load(tmp_path / f'reference-{rank}.pt', weights_only=True)
        resumed = torch.load(tmp_path / f'resume-{rank}.pt', weights_only=True)
        _assert_role_tree_equal(reference, resumed)
        assert resumed['cycle'] == 3
        assert resumed['state']['extra_training']['value']['steps'] == (3 if rank == 0 else 6)
        progress = resumed['state']['progress']['value']
        assert progress['versions'] == dict(generator=3, optimizer_g=3, gradient_g=12,
                                            score=6, optimizer_s=6)
    import json
    manifest = json.loads((tmp_path / 'resume-checkpoint/manifest.json').read_text())
    assert manifest['cursor'] == 9
    assert manifest['data_progress']['next_cycle'] == 3
    assert (tmp_path / 'resume-checkpoint/_SUCCESS').is_file()


def test_role_checkpoint_rejects_corrupt_or_mismatched_restart_before_mutation(tmp_path):
    import hashlib
    import json
    import shutil
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo required')
    pytest.importorskip('numpy')
    torch.multiprocessing.spawn(_role_restart_worker,
        args=(str(tmp_path / 'seed-init'), str(tmp_path), 'segment'), nprocs=2, join=True)
    original = tmp_path / 'segment-checkpoint'
    for fault in ('missing', 'corrupt', 'progress', 'data', 'layout', 'load', 'rng'):
        root = tmp_path / fault
        copied = root / 'segment-checkpoint'
        shutil.copytree(original, copied)
        shard = copied / 'rank-00001.pt'
        if fault == 'missing':
            shard.unlink()
        elif fault == 'corrupt':
            shard.write_bytes(shard.read_bytes() + b'corrupt')
        elif fault == 'progress':
            payload = torch.load(shard, weights_only=True)
            payload['state']['progress']['value']['versions']['score'] = 999
            torch.save(payload, shard)
            manifest_path = copied / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['files'][shard.name] = hashlib.sha256(shard.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            (copied / '_SUCCESS').write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        torch.multiprocessing.spawn(_role_restart_worker,
            args=(str(root / 'reject-init'), str(root), 'reject-' + fault), nprocs=2, join=True)


def test_distributed_audit_retains_module_extra_state_consensus():
    from distillschedule.open_workload.state_audit import state_record, _compare_records
    class ExtraState(torch.nn.Linear):
        def __init__(self):
            super().__init__(1, 1)
            self.clock = 0
        def get_extra_state(self):
            return {'clock': self.clock}
        def set_extra_state(self, value):
            self.clock = value['clock']
    module = ExtraState()
    before = state_record(module, content=False)
    module.clock = 1
    after = state_record(module, content=False)
    with pytest.raises(ValueError, match='state scalar metadata diverged'):
        _compare_records([before, after], [0, 1], content=False)


def test_audit_reuses_mesh_coordinates_but_keeps_every_tensor_validation(monkeypatch):
    from copy import deepcopy
    from distillschedule.open_workload.state_audit import _TensorRecord, _compare_records
    # CPU metadata fixture; actual HSDP tensors and replica domains are tested
    # by the multi-process audit tests above.
    mesh = [[0, 1], [2, 3]]
    rows = []
    for rank in range(4):
        record = _TensorRecord(tensor=dict(shape=[4], dtype='torch.float32', content=str(rank % 2),
            distributed=dict(mesh=mesh, shape=[4], local_shape=[2],
                             placements=[dict(kind='replicate'), dict(kind='shard', dim=0)])))
        rows.append({'parameter': record, 'moment': deepcopy(record)})
    calls = []
    original = torch.tensor
    def observed(value, *args, **kwargs):
        if isinstance(value, list) and value == mesh: calls.append(value)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch, 'tensor', observed)
    _compare_records(rows, list(range(4)), content=True)
    assert len(calls) == 1
    wrong = deepcopy(rows)
    wrong[3]['moment']['tensor']['distributed']['local_shape'] = [1]
    with pytest.raises(ValueError, match='local shard shape'):
        _compare_records(wrong, list(range(4)), content=True)
    wrong = deepcopy(rows)
    wrong[2]['moment']['tensor']['content'] = 'different'
    with pytest.raises(ValueError, match='state contents diverged'):
        _compare_records(wrong, list(range(4)), content=True)
