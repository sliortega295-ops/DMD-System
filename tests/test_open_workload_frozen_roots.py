"""Explicit frozen roots: real CPU/Gloo native FSDP, not a GPU admission.

The independent oracle runs ordinary eager modules and averages only parameter
gradients. Input VJPs remain rank-local, including the input-only grad call.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import json
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillschedule.open_workload.replica_contract import (
    ReplicaSemantics, module_method_binding, ddp_completion_edges, execution_graph)
from distillschedule.open_workload.torch_backend import Workload, candidates, full_tensor, compare
from distillschedule.workloads.program import Call, Program
from distillschedule.training.tensor_state import ParameterShadows
from test_training_flow_model_methods import native_config


LOCAL = dict(backend='torch-callable-resident-v1', share_immutable=False, checkpoint_bindings=[])
CPU_FSDP = pytest.mark.skipif(tuple(int(x) for x in torch.__version__.split('.')[:2]) < (2, 10),
                             reason='native CPU FSDP requires Torch >= 2.10')


class FrozenRoot(torch.nn.Module):
    def __init__(self, *, frozen=False):
        super().__init__()
        self.inner = torch.nn.Linear(4, 4)
        self.register_buffer('offset', torch.tensor(.03))
        self.requires_grad_(not frozen)

    def denoise(self, value):
        from torch.distributed.tensor import DTensor
        assert all(not isinstance(p, DTensor) for p in self.parameters()), 'unhooked outer entry'
        return torch.sin(self.inner(value)) + self.offset

    def rollout(self, value):
        first = self.denoise(value)
        second = self.denoise(first)
        return second + .1 * torch.nn.functional.linear(first, self.inner.weight)


def leaf(ctx):
    return (next(iter(ctx.inputs.values())).detach().clone() + .02).requires_grad_(True)


def objective(ctx):
    value, target = ctx.inputs.values()
    return (value - target).square().mean()


def input_vjp(ctx):
    loss, value = ctx.inputs.values()
    return torch.autograd.grad(loss, value)[0]


def backward(ctx):
    next(iter(ctx.inputs.values())).backward()


def update(ctx):
    ctx.state['adam'].step()
    ctx.state['adam'].zero_grad(set_to_none=True)
    ctx.state['scheduler'].step()
    for name, parameter in ctx.state['learner'].named_parameters():
        ctx.state['ema'][name].lerp_(parameter.detach(), .1)
    ctx.state['clock'].add_(1)


def frozen_workload(rank=0, *, cycle=0, state=None, versions=None):
    if state is None:
        with torch.random.fork_rng():
            torch.manual_seed(317)
            learner = FrozenRoot()
            teacher = FrozenRoot(frozen=True)
            optimizer = torch.optim.AdamW(learner.parameters(), lr=.003)
            # Nonzero initialized moments must survive native installation.
            learner.rollout(torch.ones(1, 4)).square().mean().backward()
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        state = dict(learner=learner, teacher=teacher, adam=optimizer,
                     scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.97),
                     ema=ParameterShadows('learner', {n: p.detach().clone() + .01
                                                     for n, p in learner.named_parameters()}),
                     clock=torch.tensor(0), gradient=None)
    inputs = torch.tensor([[.1, -.3, .2, .4]]) + .15*rank + .07*cycle
    calls = (
        Call('teacher_pre', 'teacher', ('x',), ('teacher',)),
        Call('leaf', 'leaf', ('x',), grad='grad'),
        Call('teacher_input', 'teacher', ('leaf',), ('teacher',), grad='grad'),
        Call('input_loss', 'loss', ('teacher_input', 'teacher_pre'), grad='grad'),
        Call('input_vjp', 'input_vjp', ('input_loss', 'leaf'), grad='grad', backward_of=('input_loss',)),
        Call('student', 'student', ('x',), ('learner',), grad='grad'),
        Call('teacher_train', 'teacher', ('student',), ('teacher',), grad='grad'),
        Call('loss', 'loss', ('teacher_train', 'teacher_pre'), grad='grad'),
        Call('backward', 'backward', ('loss',), ('gradient',), ('gradient',), grad='grad', backward_of=('loss',)),
        Call('update', 'update', ('backward',), ('learner', 'adam', 'scheduler', 'ema', 'clock', 'gradient'),
             ('learner', 'adam', 'scheduler', 'ema', 'clock', 'gradient')),
        Call('teacher_post', 'teacher', ('x',), ('teacher',)),
    )
    semantics = ReplicaSemantics({'student': 'learner'}, ('input_vjp', 'backward'), 1,
        forward_interface='registered_methods', forward_methods={'student': 'rollout', 'teacher': 'rollout'},
        parameter_shadows={'ema': 'learner'}, frozen_module_bindings={'teacher': 'teacher'})
    bindings = dict(student=module_method_binding('learner', 'rollout'),
                    teacher=module_method_binding('teacher', 'rollout'), leaf=leaf, loss=objective,
                    input_vjp=input_vjp, backward=backward, update=update)
    return Workload(Program(calls, versions or dict.fromkeys(state, 0), ('x',),
                           ('input_vjp', 'loss', 'teacher_pre', 'teacher_post')),
                    bindings, state, {'x': inputs}, immutable_modules=('teacher',),
                    replica_semantics=semantics).validate()


def test_frozen_forward_ownership_changes_identity_but_empty_keeps_legacy():
    w = frozen_workload()
    assert w.replica_semantics.forward_module_bindings == {'student': 'learner', 'teacher': 'teacher'}
    assert w.replica_semantics.module_bindings == {'student': 'learner'}
    assert w.replica_semantics.to_dict()['frozen_module_bindings'] == {'teacher': 'teacher'}
    old = ReplicaSemantics({'forward': 'learner'}, ('backward',), 1)
    assert old.to_dict() == dict(module_bindings={'forward': 'learner'}, backward_calls=['backward'],
                                 local_batch_size=1, objective='equal_rank_mean')


def test_frozen_context_completion_contains_input_only_backward():
    w = frozen_workload()
    edges = ddp_completion_edges(w.program, w.replica_semantics)
    assert ('teacher_pre', 'teacher_input') in edges
    assert ('input_vjp', 'teacher_train') in edges
    assert ('backward', 'teacher_post') in edges
    native = execution_graph(w.program, w.replica_semantics, 'fsdp_forward')
    assert native.operations == w.program.compile().operations
    assert native.metadata['final_versions']['teacher'] == 0
    from distillschedule.open_workload.replicated import forward_order_admitted
    order = [c.id for c in w.program.calls]
    assert forward_order_admitted(w, order)
    order.remove('teacher_post'); order.insert(order.index('backward'), 'teacher_post')
    assert not forward_order_admitted(w, order)
    missing = replace(w.replica_semantics, backward_calls=('backward',))
    with pytest.raises(ValueError, match='outstanding forward'):
        ddp_completion_edges(w.program, missing)
    projected = ddp_completion_edges(w.program, w.replica_semantics, {'learner'})
    assert not any(a.startswith('teacher_') or b.startswith('teacher_') for a, b in projected)


def test_frozen_native_api_alias_and_other_root_storage_are_rejected():
    from distillschedule.open_workload.sharded import rejection
    from types import MethodType
    w = frozen_workload()
    w.state['teacher'].reshard = MethodType(FrozenRoot.rollout, w.state['teacher'])
    w.bindings['teacher'] = module_method_binding('teacher', 'reshard')
    w.replica_semantics = replace(w.replica_semantics,
        forward_methods={'student': 'rollout', 'teacher': 'reshard'})
    assert rejection(w) == 'FORWARD_METHOD_COLLIDES_WITH_FSDP_API'
    w = frozen_workload()
    w.state['other'] = FrozenRoot(frozen=True)
    w.state['other'].inner.weight = w.state['teacher'].inner.weight
    w.program = replace(w.program, initial_versions={**w.program.initial_versions, 'other': 0})
    w.validate()
    assert rejection(w) == 'CROSS_ROOT_ALIASES_NEED_A_SHARED_FSDP_GROUP'


@pytest.mark.parametrize('kind', ['mapping', 'empty_name', 'unknown_root', 'non_module', 'empty_root',
    'binding_overlap', 'root_overlap', 'unknown_binding', 'trainable_teacher', 'partial_immutable',
    'no_method', 'opaque_method', 'second_method', 'write', 'hidden_no_grad', 'hidden_grad',
    'frozen_optimizer', 'module_interface', 'batchnorm', 'hidden_backward_read'])
def test_frozen_ownership_rejects_incomplete_or_unsafe_declarations(kind):
    w = frozen_workload()
    semantic = w.replica_semantics
    if kind == 'mapping': semantic = replace(semantic, frozen_module_bindings=[])
    elif kind == 'empty_name': semantic = replace(semantic, frozen_module_bindings={'teacher': ''})
    elif kind == 'unknown_root': semantic = replace(semantic, frozen_module_bindings={'teacher': 'missing'})
    elif kind == 'non_module': w.state['teacher'] = torch.tensor(1.)
    elif kind == 'empty_root': w.state['teacher'] = torch.nn.Identity()
    elif kind == 'binding_overlap': semantic = replace(semantic, frozen_module_bindings={'student': 'teacher'})
    elif kind == 'root_overlap': semantic = replace(semantic, frozen_module_bindings={'teacher': 'learner'})
    elif kind == 'unknown_binding': semantic = replace(semantic, frozen_module_bindings={'unknown': 'teacher'})
    elif kind == 'trainable_teacher': w.state['teacher'].inner.weight.requires_grad_(True)
    elif kind == 'partial_immutable': w.immutable_modules = ('teacher.inner',)
    elif kind == 'no_method': semantic = replace(semantic, forward_methods={'student': 'rollout'})
    elif kind == 'opaque_method': w.bindings['teacher'] = leaf
    elif kind == 'second_method':
        w.program = replace(w.program, calls=tuple(replace(c, binding='other')
                            if c.id == 'teacher_post' else c for c in w.program.calls))
        w.bindings['other'] = module_method_binding('teacher', 'denoise')
        semantic = replace(semantic, frozen_module_bindings={'teacher': 'teacher', 'other': 'teacher'},
                           forward_methods={**semantic.forward_methods, 'other': 'denoise'})
    elif kind == 'write':
        w.program = replace(w.program, calls=tuple(replace(c, writes=('teacher',))
                            if c.id == 'teacher_post' else c for c in w.program.calls))
    elif kind in ('hidden_no_grad', 'hidden_grad'):
        w.program = replace(w.program, calls=tuple(replace(c, binding='hidden',
            grad='grad' if kind == 'hidden_grad' else 'no_grad') if c.id == 'teacher_post' else c
            for c in w.program.calls))
        w.bindings['hidden'] = module_method_binding('teacher', 'rollout')
    elif kind == 'frozen_optimizer': w.state['adam'].param_groups[0]['params'].append(w.state['teacher'].inner.weight)
    elif kind == 'module_interface': semantic = replace(semantic, forward_interface='module_forward')
    elif kind == 'batchnorm': w.state['teacher'].norm = torch.nn.BatchNorm1d(4).requires_grad_(False)
    elif kind == 'hidden_backward_read':
        w.program = replace(w.program, calls=tuple(replace(c, reads=(*c.reads, 'teacher'))
                            if c.id == 'backward' else c for c in w.program.calls))
    w.replica_semantics = semantic
    with pytest.raises(ValueError): w.validate()


def oracle_cycles(count):
    """Independent eager global-mean training, with separate rank-local VJPs."""
    model = frozen_workload().state
    expected = []
    for cycle in range(count):
        values = [frozen_workload(rank, cycle=cycle, state=model).feeds['x'] for rank in range(2)]
        local_vjp = []
        for value in values:
            with torch.no_grad(): target = model['teacher'].rollout(value)
            value = (value.clone() + .02).requires_grad_(True)
            loss = (model['teacher'].rollout(value) - target).square().mean()
            local_vjp.append(torch.autograd.grad(loss, value)[0].detach())
        assert torch.count_nonzero(local_vjp[0]) > 0
        assert not torch.equal(local_vjp[0], local_vjp[1]), 'VJP must expose accidental rank averaging'
        x = torch.cat(values)
        with torch.no_grad(): target = model['teacher'].rollout(x)
        prediction = model['teacher'].rollout(model['learner'].rollout(x))
        loss = (prediction - target).square().mean()
        loss.backward()
        gradients = {name: p.grad.detach().clone() for name, p in model['learner'].named_parameters()}
        model['adam'].step(); model['adam'].zero_grad(set_to_none=True)
        model['scheduler'].step()
        for name, p in model['learner'].named_parameters(): model['ema'][name].lerp_(p.detach(), .1)
        model['clock'].add_(1)
        expected.append(dict(parameters={name: p.detach().clone() for name, p in model['learner'].named_parameters()},
                             teacher={name: p.detach().clone() for name, p in model['teacher'].named_parameters()},
                             gradient=gradients, vjp=local_vjp, loss=loss.detach()))
    return expected


def _init(rank, rendezvous):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE='2', GROUP_RANK='0')
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))


def native_worker(rank, rendezvous, directory, mode, stage):
    _init(rank, rendezvous)
    try:
        from distillschedule.open_workload.replicated import make_session
        from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint, pack_state, rng_state
        from test_training_end_to_end import equal
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor
        root = Path(directory)
        expected = oracle_cycles(3)
        option = dict(backend='torch-callable-replicated-v1', local=LOCAL, reduction=mode, bucket_cap_mb=0)
        identity = dict(fixture='frozen-root-v1', option=option,
                        replica_semantics=frozen_workload(rank).replica_semantics.to_dict())

        def build():
            w = frozen_workload(rank)
            session = make_session(w, option, copy_state=False, persistent=True)
            assert session.state['teacher'] is w.state['teacher']
            assert isinstance(session.state['teacher'], FSDPModule)
            assert all(isinstance(p, DTensor) for p in session.state['teacher'].parameters())
            assert all(not p.requires_grad and p.grad is None for p in session.state['teacher'].parameters())
            assert session.effects['frozen_only_models'] == {'sharded': ['teacher'], 'resident': []}
            assert session.effects['fsdp_forward_methods'] == {'learner': 'rollout', 'teacher': 'rollout'}
            assert session.state['teacher'].rollout.__wrapped__.__func__ is FrozenRoot.rollout
            assert session.state['teacher'].denoise.__func__ is FrozenRoot.denoise
            assert session.effects['optimizer_moments_migrated'] == 4
            assert session.lifecycle.full_parameter_bytes == 2 * 20 * 4
            assert sum(p.to_local().numel() for p in session.state['teacher'].parameters()) < 20
            assert set(id(p) for g in session.state['adam'].param_groups for p in g['params']) == {
                id(p) for p in session.state['learner'].parameters()}
            return w, session

        def cycles(w, session, start, stop):
            rows = []
            observed = []
            original = session.lifecycle.after_call
            def after(current, call):
                original(current, call)
                if call.id.startswith('teacher_'):
                    assert all(isinstance(p, DTensor) == mode.endswith('_reshard')
                               for p in current.state['teacher'].parameters())
                if call.id == 'backward':
                    observed.append({n: full_tensor(p.grad).clone() for n, p in current.state['learner'].named_parameters()})
            session.lifecycle.after_call = after
            for cycle in range(start, stop):
                if cycle or session._ran:
                    current = frozen_workload(rank, cycle=cycle, state=session.state,
                        versions=session.receipt['dispatcher']['final_versions'] if session._ran else w.program.initial_versions)
                    session.bind_cycle(current, graph=execution_graph(current.program, current.replica_semantics, 'fsdp_forward'))
                outputs = session.run()
                assert session.receipt['dispatcher']['final_versions']['teacher'] == 0
                assert all(isinstance(p, DTensor) for p in session.state['teacher'].parameters())
                for name, p in session.state['learner'].named_parameters():
                    torch.testing.assert_close(full_tensor(p), expected[cycle]['parameters'][name], rtol=1e-5, atol=1e-7)
                    torch.testing.assert_close(observed[-1][name], expected[cycle]['gradient'][name], rtol=1e-5, atol=1e-7)
                for name, p in session.state['teacher'].named_parameters():
                    torch.testing.assert_close(full_tensor(p), expected[cycle]['teacher'][name], rtol=0, atol=0)
                torch.testing.assert_close(outputs['input_vjp'], expected[cycle]['vjp'][rank], rtol=1e-5, atol=1e-7)
                mean = outputs['loss'].detach().clone(); dist.all_reduce(mean); mean.div_(2)
                torch.testing.assert_close(mean, expected[cycle]['loss'], rtol=1e-5, atol=1e-7)
                rows.append(dict(packed=pack_state(session.state), versions=session.receipt['dispatcher']['final_versions'],
                                 rng=rng_state(), outputs={k: v.detach().clone() for k, v in outputs.items()}))
            assert session.effects['lifecycle_installations'] == 1
            return rows

        if stage == 'write':
            w, session = build()
            whole = cycles(w, session, 0, 3)
            torch.save(whole, root / f'whole-{rank}.pt')
            w, session = build()
            split = cycles(w, session, 0, 1)
            assert equal(whole[0], split[0])
            # Packing Teacher uses local shards. Signatures elsewhere in this
            # lifecycle still intentionally materialize full tensors.
            teacher = session.state['teacher']
            original_full = DTensor.full_tensor
            def forbidden(*args, **kwargs): raise AssertionError('checkpoint packing materialized a shard')
            DTensor.full_tensor = forbidden
            try: pack_state({'teacher': teacher})
            finally: DTensor.full_tensor = original_full
            save_checkpoint(root / 'checkpoint', session.state, identity=identity, cycle=1, cursor=1,
                            versions=session.receipt['dispatcher']['final_versions'])
        else:
            w, session = build()
            payload = restore_checkpoint(root / 'checkpoint', session.state, identity=identity)
            assert payload['cycle'] == payload['cursor'] == 1
            assert equal(payload['rng'], rng_state())
            saved = torch.load(root / f'whole-{rank}.pt', weights_only=True)
            assert equal(saved[0]['packed'], pack_state(session.state))
            current = frozen_workload(rank, cycle=1, state=session.state, versions=payload['versions'])
            session.bind_cycle(current)
            for wanted, actual in zip(saved[1:], cycles(current, session, 1, 3)):
                assert equal(wanted, actual)
            Path(root, f'pass-{rank}.json').write_text(json.dumps(dict(cycles=3, cuda_initialized=torch.cuda.is_initialized(),
                fresh_process=True, teacher_gradients=False, teacher_optimizer=False)))
    finally:
        dist.destroy_process_group()


@CPU_FSDP
@pytest.mark.parametrize('mode', ['fsdp2_reshard', 'fsdp2_keep_forward'])
def test_frozen_native_input_gradients_and_fresh_process_checkpoint(tmp_path, mode):
    for stage in ('write', 'restore'):
        mp.spawn(native_worker, args=(str(tmp_path / ('rdzv-' + stage)), str(tmp_path), mode, stage), nprocs=2, join=True)
    for rank in range(2):
        assert json.loads((tmp_path / f'pass-{rank}.json').read_text())['fresh_process']


def rejection_worker(rank, rendezvous, directory):
    _init(rank, rendezvous)
    try:
        from distillschedule.open_workload.replicated import make_session
        from distillschedule.training.checkpoint import pack_state
        from test_training_end_to_end import equal
        option = dict(backend='torch-callable-replicated-v1', local=LOCAL,
                      reduction='fsdp2_reshard', bucket_cap_mb=0)
        w = frozen_workload(rank)
        session = make_session(w, option, copy_state=False, persistent=True)
        before = pack_state(session.state)
        # Removal is valid as a resident declaration, but must not replace a
        # previously installed native Teacher ownership, even on one rank.
        semantic = replace(w.replica_semantics, frozen_module_bindings={},
            forward_methods={'student': 'rollout'}) if rank == 0 else w.replica_semantics
        session.bind_cycle(replace(w, state=session.state, replica_semantics=semantic))
        with pytest.raises(ValueError, match='FSDP cycle admission.*installed FSDP method contract changed'):
            session.run()
        assert equal(before, pack_state(session.state))
        assert session.profile == {} and session.lifecycle.backward_count == 0
        dist.barrier()
        # Existing staged restore rejects a Teacher shard or flag mismatch on
        # any rank before mutating live parameters, optimizer, or shadow maps.
        from distillschedule.training.checkpoint import unpack_state
        for kind in ('requires_grad', 'layout'):
            changed = deepcopy(before)
            if rank == 0:
                if kind == 'requires_grad': changed['teacher']['requires_grad']['inner.weight'] = True
                else:
                    tensor = changed['teacher']['value']['inner.weight']
                    tensor['layout']['dtype'] = 'torch.float64'
                    tensor['local'] = tensor['local'].double()
            with pytest.raises(ValueError, match='checkpoint state admission'):
                unpack_state(changed, session.state)
            assert equal(before, pack_state(session.state))
            dist.barrier()
        # An undeclared frozen root stays resident in the existing native
        # reference option; opting in is never inferred from immutability.
        legacy = frozen_workload(rank)
        legacy.replica_semantics = replace(legacy.replica_semantics,
            frozen_module_bindings={}, forward_methods={'student': 'rollout'})
        resident = make_session(legacy, option, copy_state=False)
        from torch.distributed.tensor import DTensor
        assert not any(isinstance(p, DTensor) for p in resident.state['teacher'].parameters())
        assert resident.effects['frozen_only_models'] == 'resident'
        assert resident.lifecycle.full_parameter_bytes == 20 * 4
        resident.run()
        # Validate the actual cloned owner, not the still-dense factory state.
        copied = make_session(frozen_workload(rank), option)
        copy_before = pack_state(copied.state)
        parameter = next(copied.state['teacher'].parameters())
        if rank == 0: parameter.requires_grad_(True)
        with pytest.raises(ValueError, match='FSDP cycle admission.*all trainable modules'):
            copied.run()
        if rank == 0: parameter.requires_grad_(False)
        assert equal(copy_before, pack_state(copied.state))
        assert copied.profile == {} and copied.lifecycle.backward_count == 0
        dist.barrier()
        Path(directory, f'rejected-{rank}.json').write_text(json.dumps(dict(rejected_before_forward=True)))
    finally:
        dist.destroy_process_group()


@CPU_FSDP
def test_frozen_owner_rebind_failure_is_collective_before_forward(tmp_path):
    mp.spawn(rejection_worker, args=(str(tmp_path / 'rdzv'), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        assert json.loads((tmp_path / f'rejected-{rank}.json').read_text())['rejected_before_forward']


def common_flow_worker(rank, rendezvous, directory, mode, native=None):
    from distillschedule.training.adapters import ModelComponents
    from test_open_workload_parameter_shadows import (
        matrix_components, matrix_feeds, flow_settings, flow_workload, _snapshot, _assert_close)
    from test_training_flow_model_methods import _native_owned
    from flow_replica_oracle import sequential_b1_oracle, observe_gradients
    from distillschedule.open_workload.replicated import make_session
    from torch.distributed.tensor import DTensor
    # The public native factory still refuses distributed construction; the
    # fixture constructs before process-group setup and qualifies only the
    # explicit owner mapping added below.
    torch.set_num_threads(1)
    models = []
    for _ in range(2):
        if native is None:
            model, feeds = matrix_components(), matrix_feeds(rank)
            feeds_by_rank = [matrix_feeds(r) for r in range(2)]
        else:
            model, feeds = _native_owned(native)
            model = ModelComponents(state=model.state, immutable_modules=model.immutable_modules,
                prediction=model.prediction,
                capabilities=(*model.capabilities, 'dmd_flow_private_replicated_b1_v1'))
            feeds_by_rank = [feeds, feeds]
        models.append((model, feeds, feeds_by_rank))
    settings = flow_settings() if native is None else dict(native[1].algorithm['parameters'])
    settings.update(fake_updates=5, teacher_cfg=1., fake_cfg=1., max_grad_norm=1e-4)
    reference, _, feeds_by_rank = models[0]
    expected_trace = sequential_b1_oracle(reference.state, feeds_by_rank, settings, 53)
    expected_state = _snapshot(reference.state)
    _init(rank, rendezvous)
    try:
        model, feeds, _ = models[1]
        w = flow_workload(model, feeds, settings)
        w.replica_semantics = replace(w.replica_semantics,
            frozen_module_bindings={'teacher_forward': 'teacher'},
            forward_methods={**w.replica_semantics.forward_methods, 'teacher_forward': 'dmd_forward'})
        w.validate()
        option = dict(backend='torch-callable-replicated-v1', local=LOCAL, reduction=mode, bucket_cap_mb=0)
        session = make_session(w, option, copy_state=False, persistent=True)
        assert all(isinstance(p, DTensor) for p in session.state['teacher'].parameters())
        assert session.effects['fsdp_forward_methods']['teacher'] == 'dmd_forward'
        assert all(c.grad == 'no_grad' for c in w.program.calls if c.binding == 'teacher_forward')
        gradients, handles = observe_gradients(session)
        try: outputs = session.run()
        finally:
            for handle in handles: handle.remove()
        _assert_close(expected_state, _snapshot(session.state))
        _assert_close(expected_trace['losses'][rank], outputs)
        _assert_close([{k: v for k, v in row.items() if k != 'local'}
                       for row in expected_trace['gradients']], gradients)
        assert session.state['dmd_counters'].tolist() == [1, 1, 5]
        assert session.receipt['dispatcher']['final_versions']['teacher'] == 0
        assert all(p.grad is None for p in session.state['teacher'].parameters())
        assert len([call for call in w.program.calls if call.binding == 'teacher_forward']) == 1
        Path(directory, f'flow-{rank}.json').write_text(json.dumps(dict(G=1, F=5, teacher_grad=False,
            native_family=native[0] if native else 'synthetic_matrix', public_factory_admitted=False,
            cuda_initialized=torch.cuda.is_initialized())))
    finally:
        dist.destroy_process_group()


@CPU_FSDP
@pytest.mark.parametrize('mode', ['fsdp2_reshard', 'fsdp2_keep_forward'])
def test_common_flow_teacher_stays_nograd_with_real_G1_F5_update(tmp_path, mode):
    mp.spawn(common_flow_worker, args=(str(tmp_path / 'rdzv'), str(tmp_path), mode), nprocs=2, join=True)
    assert json.loads((tmp_path / 'flow-0.json').read_text())['teacher_grad'] is False


@CPU_FSDP
@pytest.mark.parametrize('mode', ['fsdp2_reshard', 'fsdp2_keep_forward'])
def test_private_native_flow_frozen_teacher_G1_F5(tmp_path, native_config, mode):
    mp.spawn(common_flow_worker, args=(str(tmp_path / 'rdzv'), str(tmp_path), mode, native_config), nprocs=2, join=True)
    assert json.loads((tmp_path / 'flow-0.json').read_text())['native_family'] == native_config[0]
