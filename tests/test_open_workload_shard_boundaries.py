"""Model-owned nested native FSDP groups, CPU/Gloo correctness evidence only."""
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

from distillschedule.open_workload.execution import ExecutionBackend
from distillschedule.open_workload.torch_backend import Session, physical_tensor, storage_bytes
from distillschedule.training import dmd
from distillschedule.training.adapters import ModelComponents
from distillschedule.training.checkpoint import cpu_tree, pack_state, rng_state
from distillschedule.training.prediction import PredictionSpec
from flow_replica_oracle import observe_gradients, sequential_b1_oracle
from test_open_workload_parameter_shadows import (
    CPU_FSDP, _assert_close, _snapshot, flow_settings, flow_workload, implementation, matrix_feeds)
from test_training_dmd_flow import OtherFlow
from test_training_end_to_end import equal


class BoundaryBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(4, 4).requires_grad_(False)
        self.adapter = torch.nn.Linear(4, 4, bias=False)

    def forward(self, value):
        return torch.sin(self.base(value) + self.adapter(value))


class BlockFlow(dmd.FlowDMDMethods, OtherFlow):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.blocks = torch.nn.ModuleList([BoundaryBlock() for _ in range(3)])
        self.head = torch.nn.Linear(4, 4, bias=False)

    def denoise(self, sample, sigma, feeds, *, conditional):
        value = sample['payload']
        for block in self.blocks:
            value = block(value)
        condition = feeds['condition' if conditional else 'null_condition']
        return {'payload': self.head(value) + condition}


def _components():
    with torch.random.fork_rng():
        torch.manual_seed(113)
        roles = [BlockFlow() for _ in range(3)]
    roles[-1].requires_grad_(False)
    state = dmd.training_state(*roles, student_lr=.01, fake_lr=.02,
                               scheduler_gamma=.95, ema_decay=.9)
    for value in state['student_ema'].values():
        value.add_(.01)
    return ModelComponents(state=state, prediction=PredictionSpec('flow'),
        capabilities=('dmd_flow_v1', 'dmd_flow_private_replicated_b1_v1')).validate()


def _workload(model=None, rank=0, cycle=0, *, versions=None, boundaries=None, frozen=False):
    model = _components() if model is None else model
    value = flow_workload(model, matrix_feeds(rank), flow_settings(), cycle, versions=versions)
    mapping = {role: tuple(f'blocks.{i}' for i in range(3)) for role in ('student', 'fake')}
    if frozen:
        mapping['teacher'] = tuple(f'blocks.{i}' for i in range(3))
        value.replica_semantics = replace(value.replica_semantics,
            frozen_module_bindings={'teacher_forward': 'teacher'},
            forward_methods={**value.replica_semantics.forward_methods, 'teacher_forward': 'dmd_forward'})
    value.replica_semantics = replace(value.replica_semantics,
        shard_boundaries=mapping if boundaries is None else boundaries)
    return value.validate()


def test_explicit_boundary_declaration_changes_identity_and_covers_real_children():
    value = _workload()
    plain = replace(value, replica_semantics=replace(value.replica_semantics, shard_boundaries={}))
    assert plain.identity != value.identity
    assert value.replica_semantics.to_dict()['shard_boundaries'] == {
        'student': ['blocks.0', 'blocks.1', 'blocks.2'],
        'fake': ['blocks.0', 'blocks.1', 'blocks.2']}
    assert 'shard_boundaries' not in plain.replica_semantics.to_dict()
    assert value.replica_semantics.forward_methods == {'student_forward': 'dmd_forward',
                                                      'fake_forward': 'dmd_forward'}


@pytest.mark.parametrize('mapping', [None, [], {'student': ()}, {'unknown': ('blocks.0',)},
    {'student': ('',)}, {'student': ('missing',)}, {'student': ('blocks.0', 'blocks.0')},
    {'student': ('blocks', 'blocks.0')}, {'student': 'blocks.0'}, {'student': ('blocks',)},
    {'student': ('head.weight',)}])
def test_invalid_boundary_paths_rejected_before_native_install(mapping):
    value = _workload(boundaries={})
    value.replica_semantics = replace(value.replica_semantics, shard_boundaries=mapping)
    with pytest.raises(ValueError, match='shard boundar|child|module'):
        value.validate()


def _init(rank, rendezvous):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE='2', GROUP_RANK='0')
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))


def _observe_storage(session):
    """Sample real physical parameter storage inside native child forwards."""
    rows, handles = [], []
    for role in ('student', 'fake', 'teacher'):
        model = session.state[role]
        def record(module, arguments, *, role=role, model=model):
            from torch.distributed.tensor import DTensor
            rows.append(dict(role=role, physical_parameter_bytes=storage_bytes(tuple(model.parameters())),
                dense_parameter_numel=sum(p.numel() for p in model.parameters() if not isinstance(p, DTensor))))
        for block in model.blocks:
            handles.append(block.register_forward_pre_hook(record))
    return rows, handles


def _assert_layout(session):
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor
    from distillschedule.training.tensor_state import layout
    for role in ('student', 'fake', 'teacher'):
        model = session.state[role]
        assert isinstance(model, FSDPModule)
        assert all(isinstance(block, FSDPModule) for block in model.blocks)
        assert all(isinstance(parameter, DTensor) for parameter in model.parameters())
    for role in ('student', 'fake'):
        assert {id(parameter) for parameter in session.state[role].parameters() if parameter.requires_grad} == {
            id(parameter) for group in session.state[role + '_optimizer'].param_groups for parameter in group['params']}
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in session.state['teacher'].parameters())
    for name, parameter in session.state['student'].named_parameters():
        assert layout(parameter) == layout(session.state['student_ema'][name])
    assert session.effects['fsdp_forward_methods'] == {role: 'dmd_forward' for role in ('student', 'fake', 'teacher')}
    assert session.effects['fsdp_shard_boundaries'] == {
        role: [f'blocks.{i}' for i in range(3)] for role in ('student', 'fake', 'teacher')}
    assert session.state['student'].dmd_forward.__wrapped__.__func__ is dmd.FlowDMDMethods.dmd_forward
    assert session.state['teacher'].denoise.__func__ is BlockFlow.denoise


def _prepared(rank):
    model = _components()
    workload = _workload(model, rank, frozen=True)
    sequential_b1_oracle(model.state, [matrix_feeds(i) for i in range(2)], flow_settings(), 53)
    return model, workload.program.compile().metadata['final_versions']


def _cycles(session, model, rank, start, stop, versions, expected):
    gradients, gradient_handles = observe_gradients(session)
    storage, storage_handles = _observe_storage(session)
    rows = []
    try:
        for cycle in range(start, stop):
            if session._ran:
                current = _workload(model, rank, cycle, versions=versions, frozen=True)
                session.bind_cycle(current, graph=ExecutionBackend(current).graph(session.lifecycle.option))
            before_rng = rng_state()
            outputs = session.run()
            assert equal(before_rng, rng_state())
            versions = session.receipt['dispatcher']['final_versions']
            _assert_close(expected[cycle]['state'], _snapshot(session.state))
            _assert_close(expected[cycle]['trace']['losses'][rank], cpu_tree(outputs))
            first = (cycle - start) * (1 + flow_settings()['fake_updates'])
            _assert_close([{key: value for key, value in row.items() if key != 'local'}
                for row in expected[cycle]['trace']['gradients']], gradients[first:])
            _assert_layout(session)
            assert versions['teacher'] == 0
            assert session.state['dmd_counters'].tolist() == [cycle + 1, cycle + 1, 2 * (cycle + 1)]
            rows.append(dict(packed=pack_state(session.state), outputs=cpu_tree(outputs),
                             versions=versions, rng=rng_state()))
    finally:
        for handle in gradient_handles + storage_handles:
            handle.remove()
    assert session.effects['lifecycle_installations'] == 1
    return rows, storage


def _lifecycle_worker(rank, rendezvous, directory, mode, stage):
    _init(rank, rendezvous)
    try:
        from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint
        root = Path(directory)
        reference, _ = _prepared(rank)
        expected = {}
        for cycle in range(1, 4):
            trace = sequential_b1_oracle(reference.state,
                [matrix_feeds(i) for i in range(2)], flow_settings(), 53 + cycle)
            expected[cycle] = dict(trace=trace, state=_snapshot(reference.state))
        option = implementation(mode)
        identity = dict(fixture='nested-fsdp-cpu2', mode=mode,
                        semantics=_workload(frozen=True).replica_semantics.to_dict())
        if stage == 'write':
            for length in (4, 2):
                model, versions = _prepared(rank)
                workload = _workload(model, rank, 1, versions=versions, frozen=True)
                optimizer_ids = {role: id(model.state[role + '_optimizer']) for role in ('student', 'fake')}
                session = ExecutionBackend(workload).session(option, copy_state=False, persistent=True)
                _assert_layout(session)
                assert all(id(session.state[role + '_optimizer']) == optimizer_ids[role]
                           for role in ('student', 'fake'))
                assert session.effects['optimizer_moments_migrated'] == 2 * sum(
                    parameter.requires_grad for role in ('student', 'fake') for parameter in session.state[role].parameters())
                rows, storage = _cycles(session, model, rank, 1, length, versions, expected)
                if length == 4:
                    torch.save(rows, root / f'whole-{rank}.pt')
                    dense = sum(parameter.numel() * parameter.element_size() for parameter in session.state['teacher'].parameters())
                    peak_dense = max(row['dense_parameter_numel'] * 4 for row in storage if row['role'] == 'teacher')
                    # Keep-forward intentionally permits all visited children to
                    # stay full until finish; only reshard bounds visited groups.
                    assert peak_dense < dense if mode.endswith('_reshard') else peak_dense == dense
                    Path(root, f'storage-{rank}.json').write_text(json.dumps(dict(
                        scope='CPU_PHYSICAL_PARAMETER_STORAGE_ONLY', mode=mode,
                        teacher_full_parameter_bytes=dense, teacher_peak_dense_parameter_bytes=peak_dense,
                        teacher_peak_physical_parameter_bytes=max(row['physical_parameter_bytes'] for row in storage if row['role'] == 'teacher'),
                        samples=len(storage), gpu_execution='NOT_RUN')))
                else:
                    saved = torch.load(root / f'whole-{rank}.pt', weights_only=True)
                    assert equal(saved[0], rows[0])
                    save_checkpoint(root/'checkpoint', session.state, identity=identity, cycle=2, cursor=2,
                                    versions=session.receipt['dispatcher']['final_versions'])
        else:
            model, versions = _prepared(rank)
            workload = _workload(model, rank, 1, versions=versions, frozen=True)
            session = ExecutionBackend(workload).session(option, copy_state=False, persistent=True)
            mismatched = deepcopy(identity)
            mismatched['semantics']['shard_boundaries']['student'].reverse()
            before_restore = pack_state(session.state)
            with pytest.raises(ValueError, match='identity'):
                restore_checkpoint(root/'checkpoint', session.state, identity=mismatched)
            assert equal(before_restore, pack_state(session.state))
            payload = restore_checkpoint(root/'checkpoint', session.state, identity=identity)
            # The public lifecycle adopts ordinary restored value trees too.
            model.state = session.state
            assert payload['cycle'] == payload['cursor'] == 2
            assert equal(payload['rng'], rng_state())
            saved = torch.load(root / f'whole-{rank}.pt', weights_only=True)
            assert equal(saved[0]['packed'], pack_state(session.state))
            current = _workload(model, rank, 2, versions=payload['versions'], frozen=True)
            session.bind_cycle(current, graph=ExecutionBackend(current).graph(option))
            rows, _ = _cycles(session, model, rank, 2, 4, payload['versions'], expected)
            assert equal(saved[1:], rows)
            Path(root, f'pass-{rank}.json').write_text(json.dumps(dict(
                cycles=3, fresh_process=True, blocks_per_root=3, frozen_teacher=True,
                cuda_initialized=torch.cuda.is_initialized(), gpu_execution='NOT_RUN')))
    finally:
        dist.destroy_process_group()


@CPU_FSDP
@pytest.mark.parametrize('mode', ['fsdp2_reshard', 'fsdp2_keep_forward'])
def test_nested_fsdp_common_flow_oracle_and_fresh_process_checkpoint(tmp_path, mode):
    for stage in ('write', 'restore'):
        mp.spawn(_lifecycle_worker,
                 args=(str(tmp_path/('rdzv-' + stage)), str(tmp_path), mode, stage), nprocs=2, join=True)
    for rank in range(2):
        assert json.loads((tmp_path/f'pass-{rank}.json').read_text())['fresh_process']


def test_boundary_group_parameter_and_module_aliases_are_rejected():
    value = _workload(boundaries={})
    model = value.state['student']
    model.blocks[1] = model.blocks[0]
    value.replica_semantics = replace(value.replica_semantics,
        shard_boundaries={'student': ('blocks.0', 'blocks.1')})
    with pytest.raises(ValueError, match='shard boundar.*alias|module.*alias'):
        value.validate()
    value = _workload(boundaries={})
    value.state['student'].blocks[1].adapter.weight = value.state['student'].blocks[0].adapter.weight
    # Repair the optimizer ownership to isolate the cross-group parameter alias.
    value.state['student_optimizer'] = torch.optim.AdamW(
        [parameter for parameter in value.state['student'].parameters() if parameter.requires_grad], lr=.01)
    value.replica_semantics = replace(value.replica_semantics,
        shard_boundaries={'student': ('blocks.0', 'blocks.1')})
    with pytest.raises(ValueError, match='shard boundar.*alias|parameter.*alias'):
        value.validate()


def _rejection_worker(rank, rendezvous, directory):
    _init(rank, rendezvous)
    try:
        from torch.distributed.tensor import DTensor
        # Rank contract includes explicit path order and rejects mismatches
        # before native FSDP has replaced a single parameter.
        initial = _workload(frozen=True)
        if rank == 0:
            initial.replica_semantics = replace(initial.replica_semantics,
                shard_boundaries={**initial.replica_semantics.shard_boundaries,
                    'student': tuple(reversed(initial.replica_semantics.shard_boundaries['student']))})
        before = pack_state(initial.state)
        with pytest.raises(ValueError, match='replica state, program, code, batch or geometry differs'):
            ExecutionBackend(initial).session(implementation(), copy_state=False, persistent=True)
        assert equal(before, pack_state(initial.state))
        assert all(not isinstance(p, DTensor) for p in initial.state['student'].parameters())
        dist.barrier()
        for mutation in ('declaration', 'inplace_declaration', 'child_object', 'child_parameter'):
            model = _components()
            value = _workload(model, rank, frozen=True)
            session = ExecutionBackend(value).session(implementation(), copy_state=False, persistent=True)
            current = replace(value, state=session.state)
            if rank == 0:
                if mutation == 'declaration':
                    current.replica_semantics = replace(current.replica_semantics,
                        shard_boundaries={**current.replica_semantics.shard_boundaries,
                            'student': ('blocks.1', 'blocks.0', 'blocks.2')})
                elif mutation == 'inplace_declaration':
                    current.replica_semantics.shard_boundaries['student'] = ['blocks.1', 'blocks.0', 'blocks.2']
                elif mutation == 'child_parameter':
                    block = session.state['teacher'].blocks[0]
                    block.adapter.weight = torch.nn.Parameter(block.adapter.weight.detach().clone(), requires_grad=False)
                else:
                    old = session.state['teacher'].blocks[0]
                    with torch.random.fork_rng():
                        replacement = BoundaryBlock()
                    replacement.base.weight, replacement.base.bias = old.base.weight, old.base.bias
                    replacement.adapter.weight = old.adapter.weight
                    session.state['teacher'].blocks[0] = replacement
            session.bind_cycle(current)
            before = pack_state(session.state)
            with pytest.raises(ValueError, match='FSDP cycle admission.*installed.*(contract|shard boundar)'):
                session.run()
            assert equal(before, pack_state(session.state))
            assert not session.profile and session.lifecycle.backward_count == 0
            dist.barrier()
        # A pool caller has no registered model-owned entry; it cannot reuse
        # this declaration by silently treating the group as a whole root.
        from distillschedule.open_workload.sharded import shard_roots
        from types import SimpleNamespace
        plain = _components()
        with pytest.raises(ValueError, match='shard boundar.*registered'):
            shard_roots(SimpleNamespace(state=plain.state, device=torch.device('cpu')),
                ('student',), None, reshard_after_forward=True,
                shard_boundaries={'student': ('blocks.0',)})
        Path(directory, f'rejected-{rank}.json').write_text(json.dumps(dict(
            asymmetric_initial=True, asymmetric_dynamic=True, child_identity=True,
            rejected_before_forward=True, gpu_execution='NOT_RUN')))
    finally:
        dist.destroy_process_group()


@CPU_FSDP
def test_nested_boundary_mutation_and_asymmetric_ranks_fail_before_forward(tmp_path):
    mp.spawn(_rejection_worker, args=(str(tmp_path/'rdzv'), str(tmp_path)), nprocs=2, join=True)


def test_boundaries_require_registered_method_ownership_and_uniform_layout():
    value = _workload()
    for mode in ('private_methods', 'module_forward'):
        bad = replace(value, replica_semantics=replace(value.replica_semantics,
            forward_interface=mode, forward_methods={}))
        with pytest.raises(ValueError, match='shard boundar.*registered'):
            bad.validate()
    from distillschedule.open_workload.mixed_layout import domains, graph
    assert domains(value) == {}
    with pytest.raises(ValueError, match='shard boundar.*uniform'):
        graph(value, dict(module_layouts={'student': 'post_backward', 'fake': 'fsdp2_reshard'}))


class NestedVJPRoot(torch.nn.Module):
    def __init__(self, *, frozen=False):
        super().__init__()
        self.blocks = torch.nn.ModuleList([BoundaryBlock() for _ in range(3)])
        self.head = torch.nn.Linear(4, 4, bias=False)
        if frozen:
            self.requires_grad_(False)

    def rollout(self, value):
        for block in self.blocks:
            value = block(value)
        return self.head(value)


def _vjp_state():
    from distillschedule.training.tensor_state import ParameterShadows
    with torch.random.fork_rng():
        torch.manual_seed(319)
        learner, teacher = NestedVJPRoot(), NestedVJPRoot(frozen=True)
        optimizer = torch.optim.AdamW([p for p in learner.parameters() if p.requires_grad], lr=.003)
        learner.rollout(torch.ones(1, 4)).square().mean().backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return dict(learner=learner, teacher=teacher, adam=optimizer,
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.97),
        ema=ParameterShadows('learner', {name: p.detach().clone()+.01 for name, p in learner.named_parameters()}),
        clock=torch.tensor(0), gradient=None)


def _input_vjp_worker(rank, rendezvous, directory, mode):
    from test_open_workload_frozen_roots import frozen_workload
    from distillschedule.open_workload.torch_backend import full_tensor
    reference = _vjp_state()
    expected_vjp, expected_losses = [], []
    for index in range(2):
        inputs = torch.tensor([[.1, -.3, .2, .4]]) + .15*index
        with torch.no_grad(): target = reference['teacher'].rollout(inputs)
        leaf = (inputs.detach().clone() + .02).requires_grad_(True)
        value = reference['teacher'].rollout(leaf)
        loss = (value - target).square().mean()
        expected_vjp.append(torch.autograd.grad(loss, leaf)[0])
        loss = (reference['teacher'].rollout(reference['learner'].rollout(inputs)) - target).square().mean()
        expected_losses.append(loss.detach())
        (loss / 2).backward()
    expected_grads = {name: p.grad.detach().clone() for name, p in reference['learner'].named_parameters() if p.requires_grad}
    reference['adam'].step(); reference['adam'].zero_grad(set_to_none=True)
    expected_parameters = cpu_tree(reference['learner'].state_dict())
    _init(rank, rendezvous)
    try:
        value = frozen_workload(rank, state=_vjp_state())
        value.replica_semantics = replace(value.replica_semantics,
            shard_boundaries={role: tuple(f'blocks.{i}' for i in range(3)) for role in ('learner', 'teacher')})
        session = ExecutionBackend(value).session(implementation(mode), copy_state=False, persistent=True)
        observed = []
        original = session.lifecycle.after_call
        def after(current, call):
            original(current, call)
            if call.id == 'backward':
                observed.append({name: full_tensor(p.grad).detach().clone()
                    for name, p in current.state['learner'].named_parameters() if p.requires_grad})
        session.lifecycle.after_call = after
        outputs = session.run()
        _assert_close(expected_vjp[rank], outputs['input_vjp'])
        assert not equal(expected_vjp[0], expected_vjp[1])
        _assert_close(expected_grads, observed[0])
        _assert_close(expected_parameters, cpu_tree(session.state['learner'].state_dict()))
        _assert_close(expected_losses[rank], outputs['loss'])
        assert all(p.grad is None and not p.requires_grad for p in session.state['teacher'].parameters())
        assert session.receipt['dispatcher']['final_versions']['teacher'] == 0
        from torch.distributed.tensor import DTensor
        assert all(isinstance(p, DTensor) for p in session.state['teacher'].parameters())
        Path(directory, f'vjp-{rank}.json').write_text(json.dumps(dict(
            rank_local_vjp=True, parameter_mean_once=True, teacher_version=0, gpu_execution='NOT_RUN')))
    finally:
        dist.destroy_process_group()


@CPU_FSDP
@pytest.mark.parametrize('mode', ['fsdp2_reshard', 'fsdp2_keep_forward'])
def test_nested_frozen_teacher_preserves_input_vjp_and_parameter_mean(tmp_path, mode):
    mp.spawn(_input_vjp_worker, args=(str(tmp_path/'rdzv'), str(tmp_path), mode), nprocs=2, join=True)


def test_shard_boundary_serialization_snapshot_does_not_alias_mutable_path_lists():
    value = _workload(boundaries={'student': ['blocks.0', 'blocks.1']})
    before = value.replica_semantics.to_dict()
    value.replica_semantics.shard_boundaries['student'].reverse()
    assert before['shard_boundaries']['student'] == ['blocks.0', 'blocks.1']
    assert before != value.replica_semantics.to_dict()
