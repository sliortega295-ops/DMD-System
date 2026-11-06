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


def _audit_role_forward(model, sample):
    return model(sample)


def _audit_role_loss(model, source, sample):
    return (model(source) - sample).square().mean()


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
                                _audit_role_forward, _audit_role_loss, sample_at=_audit_role_sample_at)
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
