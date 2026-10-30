"""Explicit full parameter shadows through native FSDP and common flow updates.

Matrix fixtures are synthetic CPU evidence, not native LTX or GPU qualification.
The optional native fixture reuses the pinned tiny ImageRole/LTXRole lane.
"""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import json
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillschedule.open_workload.execution import ExecutionBackend
from distillschedule.training import dmd
from distillschedule.training.adapters import ModelComponents
from distillschedule.training.checkpoint import cpu_tree, pack_state, rng_state
from distillschedule.training.prediction import PredictionSpec
from flow_replica_oracle import observe_gradients, sequential_b1_oracle
from test_training_dmd_flow import OtherFlow, policy
from test_training_end_to_end import equal
from test_training_flow_model_methods import native_config, _native_owned


CPU_FSDP = pytest.mark.skipif(tuple(int(x) for x in torch.__version__.split('.')[:2]) < (2, 10),
                             reason='actual CPU FSDP2 requires Torch >= 2.10')
LOCAL = dict(backend='torch-callable-resident-v1', share_immutable=False, checkpoint_bindings=[])


class MatrixFlow(dmd.FlowDMDMethods, OtherFlow):
    """Actual nested flow methods with frozen matrix base and trainable adapter."""
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.base = torch.nn.Linear(4, 4).requires_grad_(False)
        self.adapter = torch.nn.Linear(4, 4, bias=False)

    def denoise(self, sample, sigma, feeds, *, conditional):
        condition = feeds['condition' if conditional else 'null_condition']
        value = sample['payload']
        return {'payload': torch.sin(self.base(value) + self.adapter(value)) + condition}


def matrix_components():
    with torch.random.fork_rng():
        torch.manual_seed(52)
        roles = [MatrixFlow() for _ in range(3)]
    roles[-1].requires_grad_(False)
    state = dmd.training_state(*roles, student_lr=.01, fake_lr=.02,
                               scheduler_gamma=.95, ema_decay=.9)
    # Frozen shadows intentionally differ from their owner: skipping their
    # lerp would be observable, not hidden by an already-equal initialization.
    for shadow in state['student_ema'].values():
        shadow.add_(.1)
    return ModelComponents(state=state, prediction=PredictionSpec('flow'),
        capabilities=('dmd_flow_v1', 'dmd_flow_private_replicated_b1_v1')).validate()


def matrix_feeds(rank):
    return dict(latent=torch.zeros(1, 4), condition=torch.full((1, 4), .15 + .05*rank),
                null_condition=torch.full((1, 4), -.25))


def flow_settings():
    settings = policy()
    settings.update(teacher_cfg=1., max_grad_norm=1e-4)
    return settings


def flow_workload(model, feeds, settings, cycle=0, *, shadows=True, versions=None):
    if shadows:
        from distillschedule.training.tensor_state import ParameterShadows
        if type(model.state['student_ema']) is dict:
            model.state['student_ema'] = ParameterShadows('student', model.state['student_ema'])
    workload = dmd.build_algorithm(model, feeds, settings, cycle, 53 + cycle)
    semantics = workload.replica_semantics
    if semantics is None:
        from distillschedule.open_workload.replica_contract import ReplicaSemantics
        semantics = ReplicaSemantics({'student_forward':'student', 'fake_forward':'fake'},
            tuple(call.id for call in workload.program.calls if call.backward_of), 1)
    extra = {'parameter_shadows': {'student_ema':'student'}} if shadows else {}
    workload.replica_semantics = replace(semantics, forward_interface='registered_methods',
        random_streams='rank_independent_v1',
        forward_methods={'student_forward':'dmd_forward', 'fake_forward':'dmd_forward'}, **extra)
    if versions is not None:
        workload.program = replace(workload.program, initial_versions=versions)
    return workload.validate()


def implementation(mode='fsdp2_reshard'):
    return dict(backend='torch-callable-replicated-v1', local=LOCAL,
                reduction=mode, bucket_cap_mb=0)


def _init(rank, rendezvous):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE='2', GROUP_RANK='0')
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))


def _red_worker(rank, rendezvous, output, declared):
    _init(rank, rendezvous)
    try:
        model = matrix_components()
        w = flow_workload(model, matrix_feeds(rank), flow_settings(), shadows=declared)
        session = ExecutionBackend(w).session(implementation(), copy_state=False, persistent=True)
        if declared:
            session.run()
            from torch.distributed.tensor import DTensor
            assert session.state['student_ema'] is w.state['student_ema'] is model.state['student_ema']
            assert all(isinstance(value, DTensor) for value in session.state['student_ema'].values())
            assert session.state['dmd_counters'].tolist() == [1, 1, 2]
        else:
            with pytest.raises(RuntimeError, match='mixed torch.Tensor and DTensor'):
                session.run()
            assert session.state['dmd_counters'].tolist() == [0, 1, 0]
            assert all(type(value) is torch.Tensor for value in session.state['student_ema'].values())
        Path(output, f'red-{rank}.json').write_text(json.dumps(dict(declared=declared,
            counters=session.state['dmd_counters'].tolist(), cuda_initialized=torch.cuda.is_initialized())))
    finally:
        dist.destroy_process_group()


@CPU_FSDP
def test_undeclared_dense_ema_reproduces_mixed_layout_after_real_common_update(tmp_path):
    mp.spawn(_red_worker, args=(str(tmp_path/'rdzv'), str(tmp_path), False), nprocs=2, join=True)


@CPU_FSDP
def test_declared_parameter_shadows_complete_real_common_flow_update(tmp_path):
    mp.spawn(_red_worker, args=(str(tmp_path/'rdzv'), str(tmp_path), True), nprocs=2, join=True)


def _snapshot(state):
    return cpu_tree(dict(
        modules={key: state[key].state_dict() for key in ('student', 'fake', 'teacher')},
        optimizers={key: state[key + '_optimizer'].state_dict() for key in ('student', 'fake')},
        schedulers={key: state[key + '_scheduler'].state_dict() for key in ('student', 'fake')},
        counters=state['dmd_counters'].state_dict(), ema=dict(state['student_ema'])))


def _assert_close(expected, actual):
    # The generic comparison path formats string Program keys; native Adam
    # state_dict uses integer slot IDs. Keep its full structured tensor tree.
    from torch.utils._pytree import tree_flatten
    left, structure = tree_flatten(expected)
    right, actual_structure = tree_flatten(actual)
    assert structure == actual_structure
    for a, b in zip(left, right):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=4e-5, atol=4e-7)
        else:
            assert type(a) is type(b) and a == b


def _prepared_model(native, rank):
    if native is None:
        model, feeds, settings = matrix_components(), matrix_feeds(rank), flow_settings()
        feeds_by_rank = [matrix_feeds(i) for i in range(2)]
    else:
        model, feeds = _native_owned(native)
        # Test-only capability binding; the public factory/catalog remains
        # gated. Frozen Teacher stays resident in this bounded layout proof.
        model = ModelComponents(state=model.state, immutable_modules=model.immutable_modules,
            prediction=model.prediction,
            capabilities=(*model.capabilities, 'dmd_flow_private_replicated_b1_v1'))
        settings = dict(native[1].algorithm['parameters'])
        settings.update(teacher_cfg=1., max_grad_norm=1e-4)
        feeds_by_rank = [feeds, feeds]
        for value in model.state['student_ema'].values(): value.add_(.01)
    initial = flow_workload(model, feeds, settings)
    # Real pointwise AdamW moments and nontrivial EMA already exist on entry;
    # native installation must migrate them, never reset or reinitialize.
    sequential_b1_oracle(model.state, feeds_by_rank, settings, 53)
    versions = initial.program.compile().metadata['final_versions']
    return model, feeds, settings, feeds_by_rank, versions


def _run_cycles(session, model, feeds, settings, start, stop, versions, expected=None):
    from torch.distributed.tensor import DTensor
    rows, handles = observe_gradients(session)
    result = []
    try:
        for cycle in range(start, stop):
            if cycle != start or session._ran:
                current = flow_workload(model, feeds, settings, cycle, versions=versions)
                session.bind_cycle(current, graph=ExecutionBackend(current).graph(session.lifecycle.option))
            before_rng = rng_state()
            outputs = session.run()
            assert equal(before_rng, rng_state())
            actual = _snapshot(session.state)
            versions = session.receipt['dispatcher']['final_versions']
            if expected is not None:
                record = expected[cycle]
                _assert_close(record['state'], actual)
                _assert_close(record['trace']['losses'][dist.get_rank()], cpu_tree(outputs))
                offset = (cycle-start) * (1+settings['fake_updates'])
                observed = rows[offset:]
                _assert_close([{k:v for k,v in row.items() if k != 'local'}
                               for row in record['trace']['gradients']], observed)
                assert any(not equal(row['mean'], row['clipped']) for row in observed)
            for role in ('student', 'fake'):
                parameters = {id(p) for p in session.state[role].parameters() if p.requires_grad}
                optimizer = session.state[role + '_optimizer']
                assert {id(p) for group in optimizer.param_groups for p in group['params']} == parameters
                assert all(isinstance(p, DTensor) for p in session.state[role].parameters())
            assert all(p.grad is None for role in ('student', 'fake', 'teacher')
                       for p in session.state[role].parameters())
            assert session.state['student_ema'] is model.state['student_ema'] is session.workload.state['student_ema']
            assert set(session.state['student_ema']) == dict(session.state['student'].named_parameters()).keys()
            assert all(isinstance(p, DTensor) for p in session.state['student_ema'].values())
            assert session.state['dmd_counters'].tolist() == [cycle+1, cycle+1, (cycle+1)*settings['fake_updates']]
            assert versions['teacher'] == 0 and versions['student_ema'] == cycle+1
            result.append(dict(packed=pack_state(session.state), versions=versions, outputs=cpu_tree(outputs)))
    finally:
        for handle in handles: handle.remove()
    return result


def _lifecycle_worker(rank, rendezvous, output, mode, stage, native=None):
    import faulthandler
    faulthandler.dump_traceback_later(120, repeat=False)
    try:
        # Native factories are constructed before the process group, rather
        # than bypassing their still-closed public construction guard.
        torch.set_num_threads(1)
        models = [_prepared_model(native, rank) for _ in range(3 if stage == 'write' else 1)]
        _init(rank, rendezvous)
        from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint
        from distillschedule.training.tensor_state import layout
        root = Path(output)
        option = implementation(mode)
        identity = dict(fixture='parameter-shadows-cpu2', implementation=option,
                        parameter_shadows={'student_ema':'student'}, native=native[0] if native else None)
        if stage == 'write':
            reference, _, settings, feeds_by_rank, _ = models.pop(0)
            expected = {}
            teacher = cpu_tree(reference.state['teacher'].state_dict())
            for cycle in range(1, 4):
                trace = sequential_b1_oracle(reference.state, feeds_by_rank, settings, 53+cycle)
                expected[cycle] = dict(trace=trace, state=_snapshot(reference.state))
            del reference
            for stop, prepared in zip((4, 2), models):
                model, feeds, settings, _, versions = prepared
                w = flow_workload(model, feeds, settings, 1, versions=versions)
                shadows = model.state['student_ema']
                initial = _snapshot(model.state)
                session = ExecutionBackend(w).session(option, copy_state=False, persistent=True)
                assert session.state['student_ema'] is shadows is w.state['student_ema']
                _assert_close(initial, _snapshot(session.state))
                assert session.effects['optimizer_moments_migrated'] == 2 * sum(
                    p.requires_grad for role in ('student', 'fake') for p in session.state[role].parameters())
                for name, parameter in session.state['student'].named_parameters():
                    assert layout(shadows[name]) == layout(parameter)
                    assert shadows[name].to_local().numel() <= shadows[name].numel()
                full_bytes = sum(value.numel()*value.element_size() for value in shadows.values())
                local_bytes = sum(value.to_local().numel()*value.element_size() for value in shadows.values())
                assert local_bytes < full_bytes, 'aggregate shadows must really be sharded'
                total_bytes = torch.tensor(local_bytes, dtype=torch.int64)
                dist.all_reduce(total_bytes)
                assert int(total_bytes) == full_bytes, 'all local shards must cover the full parameter shadow'
                rows = _run_cycles(session, model, feeds, settings, 1, stop, versions, expected)
                assert equal(teacher, cpu_tree(session.state['teacher'].state_dict()))
                assert session.effects['lifecycle_installations'] == 1
                if stop == 4:
                    torch.save(rows, root/f'whole-{rank}.pt')
                else:
                    save_checkpoint(root/'checkpoint', session.state, identity=identity, cycle=2, cursor=2,
                                    versions=rows[-1]['versions'])
        else:
            model, feeds, settings, _, versions = models[0]
            w = flow_workload(model, feeds, settings, 1, versions=versions)
            session = ExecutionBackend(w).session(option, copy_state=False, persistent=True)
            shadows = session.state['student_ema']
            value = restore_checkpoint(root/'checkpoint', session.state, identity=identity)
            assert value['cycle'] == value['cursor'] == 2
            assert equal(value['rng'], rng_state())
            assert session.state['student_ema'] is shadows is model.state['student_ema']
            # The common checkpoint loader replaces ordinary value trees such
            # as dmd_lifecycle; adopt the restored session just as the trainer.
            model.state = session.state
            whole = torch.load(root/f'whole-{rank}.pt', weights_only=True)
            assert equal(whole[0]['packed'], pack_state(session.state))
            current = flow_workload(model, feeds, settings, 2, versions=value['versions'])
            session.bind_cycle(current, graph=ExecutionBackend(current).graph(option))
            resumed = _run_cycles(session, model, feeds, settings, 2, 4, value['versions'])
            assert equal(whole[1:], resumed), 'fresh-process continuation must be bitwise exact'
            Path(root, f'restored-{rank}.json').write_text(json.dumps(dict(cycles=4,
                fresh_process=True, native=native[0] if native else None, gpu_execution='NOT_RUN')))
        assert not torch.cuda.is_initialized()
    finally:
        faulthandler.cancel_dump_traceback_later()
        if dist.is_initialized():
            dist.destroy_process_group()


@CPU_FSDP
@pytest.mark.parametrize('mode', ['fsdp2_reshard', 'fsdp2_keep_forward'])
def test_full_shadow_oracle_and_fresh_process_checkpoint(tmp_path, mode):
    for stage in ('write', 'restore'):
        mp.spawn(_lifecycle_worker, args=(str(tmp_path/('rdzv-'+stage)), str(tmp_path), mode, stage),
                 nprocs=2, join=True)


@pytest.mark.parametrize('mode', ['fsdp2_reshard', 'fsdp2_keep_forward'])
def test_native_full_shadow_oracle_and_fresh_process_checkpoint(tmp_path, native_config, mode):
    for stage in ('write', 'restore'):
        mp.spawn(_lifecycle_worker,
            args=(str(tmp_path/('rdzv-'+stage)), str(tmp_path), mode, stage, native_config), nprocs=2, join=True)


@pytest.mark.parametrize('fault', ['missing_frozen', 'extra', 'shape', 'dtype', 'stride', 'grad',
    'parameter', 'meta', 'nonfinite', 'alias_parameter', 'alias_shadow', 'alias_feed', 'owner_alias',
    'untyped', 'owner', 'unknown_state', 'unknown_root', 'frozen_root', 'binding_type'])
def test_parameter_shadow_binding_rejects_malformed_state_locally(fault):
    model = matrix_components()
    w = flow_workload(model, matrix_feeds(0), flow_settings())
    shadows = w.state['student_ema']
    name = 'base.weight'
    if fault == 'missing_frozen': del shadows[name]
    elif fault == 'extra': shadows['unknown'] = shadows[name].clone()
    elif fault == 'shape': shadows[name] = torch.zeros(1)
    elif fault == 'dtype': shadows[name] = shadows[name].double()
    elif fault == 'stride': shadows[name] = shadows[name].t()
    elif fault == 'grad': shadows[name].requires_grad_()
    elif fault == 'parameter': shadows[name] = torch.nn.Parameter(shadows[name], requires_grad=False)
    elif fault == 'meta': shadows[name] = torch.empty_like(shadows[name], device='meta')
    elif fault == 'nonfinite': shadows[name][0, 0] = float('nan')
    elif fault == 'alias_parameter': shadows[name] = w.state['student'].base.weight.detach()
    elif fault == 'alias_shadow': shadows[name] = shadows['adapter.weight']
    elif fault == 'alias_feed': w.feeds['latent'] = shadows[name][:1]
    elif fault == 'owner_alias': w.state['student'].base.weight_alias = w.state['student'].base.weight
    elif fault == 'untyped': w.state['student_ema'] = dict(shadows)
    elif fault == 'owner': shadows.owner = 'fake'
    else:
        bindings = {'unknown_state': {'unknown':'student'}, 'unknown_root': {'student_ema':'unknown'},
            'frozen_root': {'student_ema':'teacher'}, 'binding_type': None}[fault]
        w.replica_semantics = replace(w.replica_semantics, parameter_shadows=bindings)
    with pytest.raises(ValueError):
        w.validate()
    from torch.distributed.fsdp import FSDPModule
    assert not isinstance(w.state['student'], FSDPModule)


def test_empty_shadows_preserve_legacy_identity_without_storage_scan(monkeypatch):
    from distillschedule.training.tensor_state import validate_parameter_shadows
    import distillschedule.open_workload.torch_backend as backend
    from test_open_workload_replica_contract import with_replica
    w = with_replica()
    assert 'parameter_shadows' not in w.replica_semantics.to_dict()
    def forbidden(*a, **k): pytest.fail('empty shadow contract must not scan tensor state')
    monkeypatch.setattr(backend, 'tensors', forbidden)
    validate_parameter_shadows(w.state, {}, ())


def test_shadow_checkpoint_boundary_rejects_nonfinite_before_publication(tmp_path):
    from distillschedule.training.checkpoint import save_checkpoint
    from distillschedule.training.tensor_state import ParameterShadows
    state = {'shadow': ParameterShadows('root', {'weight':torch.tensor([float('nan')])})}
    with pytest.raises(ValueError, match='finite detached'):
        save_checkpoint(tmp_path/'rejected', state, identity={}, cycle=0, cursor=0, versions={'shadow':0})
    assert not (tmp_path/'rejected').exists()


def test_partial_root_sharding_refused_before_native_mutation(monkeypatch):
    from types import SimpleNamespace
    from distillschedule.open_workload.sharded import shard_roots
    import torch.distributed.fsdp as fsdp
    model = matrix_components()
    w = flow_workload(model, matrix_feeds(0), flow_settings())
    state = dict(w.state, partial=w.state['student'].base)
    def forbidden(*a, **k): pytest.fail('partial root reached native mutation')
    monkeypatch.setattr(fsdp, 'fully_shard', forbidden)
    with pytest.raises(ValueError, match='whole-root'):
        shard_roots(SimpleNamespace(state=state), ('partial',), None, reshard_after_forward=True,
                    parameter_shadows=w.replica_semantics.parameter_shadows)


def _negative_worker(rank, rendezvous, output):
    from copy import deepcopy
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor
    from distillschedule.training.checkpoint import unpack_state
    _init(rank, rendezvous)
    try:
        # Asymmetric validation must complete collectively before any FSDP
        # construction or data collective; a healthy peer must not time out.
        for fault in ('shape', 'binding'):
            model = matrix_components()
            w = flow_workload(model, matrix_feeds(rank), flow_settings())
            if rank == 0:
                if fault == 'shape': w.state['student_ema']['base.weight'] = torch.zeros(1)
                else:
                    w.state['student_ema'].owner = 'fake'
                    w.replica_semantics = replace(w.replica_semantics, parameter_shadows={'student_ema':'fake'})
            before = pack_state(w.state)
            with pytest.raises(ValueError, match='session preflight|replica state, program'):
                ExecutionBackend(w).session(implementation(), copy_state=False, persistent=True)
            assert equal(before, pack_state(w.state))
            assert not isinstance(w.state['student'], FSDPModule)
            dist.barrier()
        # Explicit trial cloning keeps the original typed dense map untouched;
        # persistent no-copy sessions preserve the very same mapping instead.
        model = matrix_components()
        w = flow_workload(model, matrix_feeds(rank), flow_settings())
        before = pack_state(w.state)
        session = ExecutionBackend(w).session(implementation(), copy_state=True, persistent=True)
        assert session.state['student_ema'] is not w.state['student_ema']
        session.run()
        assert equal(before, pack_state(w.state))
        assert all(type(value) is torch.Tensor for value in w.state['student_ema'].values())
        assert all(isinstance(value, DTensor) for value in session.state['student_ema'].values())
        progress = dict(cycle=1, versions=session.receipt['dispatcher']['final_versions'])
        packed = pack_state(session.state)
        faults = ('owner', 'schema', 'kind', 'missing_frozen', 'dense', 'dtype', 'shape', 'placement',
                  'mesh', 'nonfinite', 'grad')
        for fault in faults:
            bad = deepcopy(packed)
            if rank == 0:
                row = bad['student_ema']
                value = row['value']
                parameter = value['parameters']['base.weight']
                if fault == 'owner': value['owner'] = 'fake'
                elif fault == 'schema': value['schema_version'] = 'unknown'
                elif fault == 'kind': row['kind'] = 'value'
                elif fault == 'missing_frozen': del value['parameters']['base.weight']
                elif fault == 'dense': value['parameters']['base.weight'] = torch.zeros(4, 4)
                elif fault == 'dtype': parameter['local'] = parameter['local'].double()
                elif fault == 'shape': parameter['layout']['shape'] = [8, 4]
                elif fault == 'placement': parameter['layout']['placements'] = [{'kind':'replicate'}]
                elif fault == 'mesh': parameter['layout']['mesh'] = [1, 0]
                elif fault == 'nonfinite': parameter['local'][0, 0] = float('nan')
                elif fault == 'grad': parameter['local'].requires_grad_()
            before_rng = rng_state()
            with pytest.raises(ValueError, match='checkpoint state admission'):
                unpack_state(bad, session.state, progress=progress)
            assert equal(packed, pack_state(session.state))
            assert equal(before_rng, rng_state())
            dist.barrier()
        # Declarations belong to the whole installed method contract. The
        # separate installed binding snapshot must also reject local drift.
        for fault, message in (
                ('declaration', 'installed FSDP method contract changed between cycles'),
                ('installed_binding', 'installed FSDP parameter shadow bindings changed between cycles')):
            changed = replace(session.workload, state=session.state, replica_semantics=w.replica_semantics,
                program=replace(session.workload.program, initial_versions=progress['versions']))
            if rank == 0:
                if fault == 'declaration':
                    changed.replica_semantics = replace(changed.replica_semantics, parameter_shadows={})
                else:
                    session.lifecycle.parameter_shadows = {}
            session.bind_cycle(changed, graph=ExecutionBackend(changed).graph(implementation()))
            before_rng = rng_state()
            with pytest.raises(ValueError, match='FSDP cycle admission.*' + message):
                session.run()
            assert equal(packed, pack_state(session.state))
            assert equal(before_rng, rng_state())
            assert session.profile == {}
            dist.barrier()
        Path(output, f'negative-{rank}.json').write_text(json.dumps(dict(checkpoint_faults=list(faults),
            asymmetric_preflight=True, unchanged_state=True, cuda_initialized=torch.cuda.is_initialized())))
    finally:
        dist.destroy_process_group()


@CPU_FSDP
def test_collective_binding_refusal_trial_isolation_and_checkpoint_layout_refusal(tmp_path):
    mp.spawn(_negative_worker, args=(str(tmp_path/'rdzv'), str(tmp_path)), nprocs=2, join=True)


def _mixed_worker(rank, rendezvous):
    from distillschedule.training.tensor_state import ParameterShadows
    from torch.distributed.tensor import DTensor
    from test_open_workload_multirole_replica import build_workload
    _init(rank, rendezvous)
    try:
        for sharded in ('role_a', 'role_b'):
            w = build_workload(1, 4, 3, rank=rank)
            bindings = {'shadow_a':'role_a', 'second_a':'role_a', 'shadow_b':'role_b'}
            for index, (key, root) in enumerate(bindings.items()):
                w.state[key] = ParameterShadows(root, {
                    name: parameter.detach().clone() + index*.1 for name, parameter in w.state[root].named_parameters()})
            before = {key: cpu_tree(dict(w.state[key])) for key in bindings}
            # Binding insertion order is not a distributed communication order.
            declared = dict(reversed(tuple(bindings.items()))) if rank else bindings
            w.replica_semantics = replace(w.replica_semantics, parameter_shadows=declared)
            w.program = replace(w.program, initial_versions={**w.program.initial_versions,
                **{key:0 for key in bindings}})
            option = dict(backend='torch-callable-replicated-v1', local=LOCAL,
                reduction='per_module', bucket_cap_mb=128,
                module_layouts={key:'fsdp2_reshard' if key == sharded else 'post_backward'
                                for key in ('role_a', 'role_b')})
            session = ExecutionBackend(w).session(option, copy_state=False, persistent=True)
            session.run()
            for key, root in bindings.items():
                assert session.state[key] is w.state[key]
                assert all(isinstance(value, DTensor) == (root == sharded) for value in session.state[key].values())
                assert equal(before[key], cpu_tree(dict(session.state[key])))
            snapshot = pack_state(session.state)
            current = replace(w, state=session.state,
                program=replace(w.program, initial_versions=session.receipt['dispatcher']['final_versions']))
            if rank == 0:
                current.replica_semantics = replace(current.replica_semantics, parameter_shadows={})
            session.bind_cycle(current, graph=ExecutionBackend(current).graph(option))
            with pytest.raises(ValueError, match='mixed parameter shadow admission.*shadow bindings changed'):
                session.run()
            assert equal(snapshot, pack_state(session.state))
            dist.barrier()
    finally:
        dist.destroy_process_group()


@CPU_FSDP
def test_multiple_shadows_follow_only_their_mixed_roots(tmp_path):
    mp.spawn(_mixed_worker, args=(str(tmp_path/'rdzv'),), nprocs=2, join=True)


def _identity_workload(rank, lane):
    from distillschedule.training.tensor_state import ParameterShadows
    if lane == 'fsdp':
        model = matrix_components()
        return flow_workload(model, matrix_feeds(rank), flow_settings()), implementation()
    from test_open_workload_multirole_replica import build_workload
    w = build_workload(1, 4, 3, rank=rank)
    bindings = {'shadow_a':'role_a', 'shadow_b':'role_b'}
    for key, root in bindings.items():
        w.state[key] = ParameterShadows(root, {
            name: parameter.detach().clone() for name, parameter in w.state[root].named_parameters()})
    w.replica_semantics = replace(w.replica_semantics, parameter_shadows=bindings)
    w.program = replace(w.program, initial_versions={**w.program.initial_versions,
        **{key:0 for key in bindings}})
    return w, dict(backend='torch-callable-replicated-v1', local=LOCAL,
        reduction='per_module', bucket_cap_mb=128,
        module_layouts={'role_a':'fsdp2_reshard', 'role_b':'post_backward'})


def _identity_worker(rank, rendezvous, output, lane, copy_state):
    from distillschedule.open_workload.torch_backend import unwrap_module
    from distillschedule.training.tensor_state import ParameterShadows
    from distillschedule.training.checkpoint import unpack_state
    _init(rank, rendezvous)
    try:
        observations = []
        w, option = _identity_workload(rank, lane)
        bindings = w.replica_semantics.parameter_shadows
        for key, root in bindings.items():
            for fault in ('session_map', 'workload_map', 'session_parameter', 'workload_parameter'):
                w, option = _identity_workload(rank, lane)
                session = ExecutionBackend(w).session(option, copy_state=copy_state, persistent=True)
                if rank == 0:
                    state = session.state if fault.startswith('session') else w.state
                    if fault.endswith('map'):
                        state[key] = ParameterShadows(root, {name:value.clone() for name,value in state[key].items()})
                    else:
                        model = unwrap_module(state[root])
                        parameter = model.base.weight
                        assert not parameter.requires_grad
                        model.base.weight = torch.nn.Parameter(parameter.detach().clone(), requires_grad=False)
                before, before_rng = pack_state(session.state), rng_state()
                try:
                    session.run()
                except ValueError as exc:
                    assert 'parameter shadow' in str(exc) and 'identity changed' in str(exc)
                    assert session.profile == {}
                    assert equal(before, pack_state(session.state))
                    assert equal(before_rng, rng_state())
                    refused = True
                else:
                    refused = False
                observations.append(dict(shadow=key, fault=fault, refused=refused))
                dist.barrier()
        Path(output, f'identity-{rank}.json').write_text(json.dumps(observations))
        assert all(row['refused'] for row in observations), observations

        # In-place checkpoint copies and an explicit adoption of a cloned
        # session remain valid; ownership checks must not require original
        # dense trial maps to be the installed distributed maps.
        w, option = _identity_workload(rank, lane)
        original = pack_state(w.state)
        session = ExecutionBackend(w).session(option, copy_state=copy_state, persistent=True)
        owners = {key: (session.state[key], dict(unwrap_module(session.state[root]).named_parameters()))
                  for key, root in bindings.items()}
        session.run()
        versions = session.receipt['dispatcher']['final_versions']
        packed = pack_state(session.state)
        unpack_state(packed, session.state, progress=dict(cycle=1, versions=versions))
        for key, root in bindings.items():
            assert session.state[key] is owners[key][0]
            assert all(parameter is owners[key][1][name]
                       for name, parameter in unwrap_module(session.state[root]).named_parameters())
        if copy_state:
            assert equal(original, pack_state(w.state))
        if lane == 'fsdp':
            model = ModelComponents(state=session.state, prediction=PredictionSpec('flow'),
                capabilities=('dmd_flow_v1', 'dmd_flow_private_replicated_b1_v1'))
            current = flow_workload(model, matrix_feeds(rank), flow_settings(), 1, versions=versions)
        else:
            current = replace(w, state=session.state, program=replace(w.program, initial_versions=versions))
        session.bind_cycle(current, graph=ExecutionBackend(current).graph(option))
        session.run()
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@CPU_FSDP
@pytest.mark.parametrize('lane', ['fsdp', 'mixed'])
@pytest.mark.parametrize('copy_state', [False, True], ids=['owned', 'copied'])
def test_installed_shadow_and_parameter_identities_are_bound(tmp_path, lane, copy_state):
    mp.spawn(_identity_worker, args=(str(tmp_path/'rdzv'), str(tmp_path), lane, copy_state),
             nprocs=2, join=True)
