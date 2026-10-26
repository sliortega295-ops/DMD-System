"""Real CPU/Gloo native macrocalls; this is not CUDA/performance evidence."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from distillschedule.open_workload.execution import ExecutionBackend
from distillschedule.open_workload.torch_backend import Workload, unwrap_module
from distillschedule.workloads.program import Call, Program


BACKEND = 'torch-callable-native-owner-v1'


def option():
    return dict(backend=BACKEND, local=dict(backend='torch-callable-resident-v1',
                share_immutable=False, checkpoint_bindings=[]))


class CollectiveOwner:
    """The owner, not a second runtime reducer, performs the entire update."""
    def __init__(self, rank):
        with torch.random.fork_rng():
            torch.manual_seed(17)
            self.model = torch.nn.Linear(3, 2)
        self.adam = torch.optim.Adam(self.model.parameters(), lr=.01)
        self.rng = torch.Generator().manual_seed(41 + rank).get_state()
        self.steps = self.preparations = self.forwards = self.backwards = 0
        self.hook = None
        self.fault = None

    def __deepcopy__(self, memo):
        raise AssertionError('native collective owners must never be deepcopied')

    def state_dict(self):
        model = unwrap_module(self.model)
        return dict(model=model.state_dict(), adam=self.adam.state_dict(),
                    steps=self.steps, rng=self.rng)

    def load_state_dict(self, value):
        model = unwrap_module(self.model)
        model.load_state_dict(value['model']); self.adam.load_state_dict(value['adam'])
        self.steps, self.rng = value['steps'], value['rng']

    def runtime_tensors(self):
        yield from self.model.parameters()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                yield parameter.grad
        for slots in self.adam.state.values():
            yield from slots.values()

    def collective_execution_spec(self):
        value = dict(schema_version='distillschedule.native_collective_owner.v1',
                     backend='ddp', world_size=2, device_type='cpu',
                     shared_identity={'algorithm': 'linear-square-mean-clip-adam', 'version': 1})
        if self.fault == 'schema': value['schema_version'] = 'unknown'
        if self.fault == 'backend': value['backend'] = 'fsdp2'
        if self.fault == 'world': value['world_size'] = 3
        if self.fault == 'world_bool': value['world_size'] = True
        if self.fault == 'device': value['device_type'] = 'cuda'
        if self.fault == 'shared': value['shared_identity'] = {}
        if self.fault == 'nonfinite': value['shared_identity']['nan'] = float('nan')
        if self.fault == 'rng': torch.rand(1)
        if self.fault == 'numpy_rng':
            import numpy as np
            np.random.random()
        if self.fault == 'impure': self.steps += 1
        if self.fault == 'changed_shared': value['shared_identity']['version'] = 2
        return value

    def prepare_collective_execution(self):
        from distillschedule.runtime.wan13b_fair import _DDPCommState, ddp_sum_then_divide_hook
        assert self.preparations == 0 and not isinstance(self.model, DDP)
        self.model = DDP(self.model, process_group=dist.group.WORLD,
                         broadcast_buffers=False, gradient_as_bucket_view=False)
        self.hook = _DDPCommState(dist.group.WORLD, dist.get_world_size())
        self.model.register_comm_hook(self.hook, ddp_sum_then_divide_hook)
        self.preparations += 1

    def validate_collective_execution(self):
        assert isinstance(self.model, DDP) and self.preparations == 1
        assert self.model.process_group is dist.group.WORLD and dist.get_world_size() == 2
        parameters = tuple(self.model.parameters())
        assert all(p.grad is None for p in parameters)
        assert {id(p) for p in parameters} == {
            id(p) for group in self.adam.param_groups for p in group['params']}
        return dict(bucket_storage_bound_bytes=2 * sum(p.numel() * p.element_size() for p in parameters),
                    roles={'learner': dict(forward_calls=self.forwards, backward_calls=self.backwards,
                         optimizer_steps=self.steps, gradient_collectives=len(self.hook.bucket_bytes))},
                    rank=dist.get_rank(), world_size=dist.get_world_size(), zero_gradients=True)

    def run(self, x):
        value = self.model(x).square().mean(); self.forwards += 1
        value.backward(); self.backwards += 1
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.adam.step(); self.adam.zero_grad(set_to_none=True); self.steps += 1
        return value.detach()


class ReceiptOwner(CollectiveOwner):
    def validate_collective_execution(self):
        value = super().validate_collective_execution()
        if self.receipt_fault == 'bound': value['bucket_storage_bound_bytes'] = -1
        if self.receipt_fault == 'bool': value['bucket_storage_bound_bytes'] = True
        if self.receipt_fault == 'roles': value['roles'] = {}
        if self.receipt_fault == 'counts': value['roles']['learner']['forward_calls'] = True
        if self.receipt_fault == 'nonfinite': value['extra'] = float('inf')
        if self.receipt_fault == 'topology': value['world_size'] = 3
        if self.receipt_fault == 'boundary': value['zero_gradients'] = False
        return value


class LeakyOwner(CollectiveOwner):
    def run(self, x):
        result = super().run(x)
        parameter = next(self.model.parameters())
        parameter.grad = torch.ones_like(parameter)
        return result


class GrowingOwner(CollectiveOwner):
    def runtime_tensors(self):
        yield from super().runtime_tensors()
        if hasattr(self, 'cache'):
            yield self.cache

    def state_dict(self):
        return dict(super().state_dict(), cache=getattr(self, 'cache', None))

    def load_state_dict(self, value):
        super().load_state_dict(value)
        self.cache = value['cache']

    def run(self, x):
        result = super().run(x)
        self.cache = torch.ones(self.steps * 8)
        return result


def changed_collective_spec(self):
    """Same declaration, different bound method source: still must reject."""
    return dict(schema_version='distillschedule.native_collective_owner.v1',
                backend='ddp', world_size=2, device_type='cpu',
                shared_identity={'algorithm': 'linear-square-mean-clip-adam', 'version': 1})


def native_call(ctx):
    return ctx.state['owner'].run(ctx.inputs['x'])


def workload(rank, *, owner=None, cycle=0):
    return Workload(Program((Call('native', 'native', ('x',), ('owner',), ('owner',),
                    grad='external_grad', attributes={'cycle': cycle}),),
                    {'owner': cycle}, ('x',), ('native',)), {'native': native_call},
                    {'owner': owner if owner is not None else CollectiveOwner(rank)},
                    {'x': torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10 + rank * .3 + cycle * .1},
                    seed=29 + rank)


def initialize(rank, filename, *, timeout=40):
    torch.set_num_threads(1)
    torch.manual_seed(71 + rank)
    dist.init_process_group('gloo', init_method='file://' + filename, rank=rank,
                            world_size=2, timeout=timedelta(seconds=timeout))


def happy_worker(rank, filename, output):
    initialize(rank, filename)
    try:
        w = workload(rank); owner = w.state['owner']; backend = ExecutionBackend(w)
        assert backend.key == BACKEND
        identity = backend.identity()
        rows = identity['native_collective_owner']['ranks']
        assert rows[0]['initial_state'] != rows[1]['initial_state']
        assert rows[0]['feeds'] != rows[1]['feeds'] and rows[0]['rng'] != rows[1]['rng']
        assert [row['seed'] for row in rows] == [29, 30]
        assert json.loads(json.dumps(identity, allow_nan=False)) == identity
        inventory = backend.inventory(1)
        assert inventory['total_candidates'] == 1 and inventory['candidates'] == [option()]
        assert owner.preparations == 0
        session = backend.session(option(), copy_state=False, persistent=True)
        assert session.state['owner'] is owner and owner.preparations == 1
        oracle = CollectiveOwner(0)
        receipts = []
        for cycle in range(2):
            if cycle:
                w = workload(rank, owner=owner, cycle=cycle)
                backend = ExecutionBackend(w)
                session.bind_cycle(w, graph=backend.graph(option()))
            session.run(measure=True)
            global_x = torch.cat([workload(r, cycle=cycle).feeds['x'] for r in range(2)])
            oracle.run(global_x)
            for parameter, expected in zip(owner.model.parameters(), oracle.model.parameters()):
                torch.testing.assert_close(parameter, expected, rtol=2e-5, atol=2e-7)
                gathered = [torch.empty_like(parameter) for _ in range(2)]
                dist.all_gather(gathered, parameter)
                torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
            assert all(p.grad is None for p in owner.model.parameters())
            assert owner.preparations == 1 and session.effects['lifecycle_installations'] == 1
            runtime = session.effects['native_collective_owner']['runtime']
            counters = runtime['roles']['learner']
            assert counters['optimizer_steps'] == counters['backward_calls'] == cycle + 1
            assert counters['gradient_collectives'] >= cycle + 1 and runtime['zero_gradients']
            local_static = session.static_bytes
            old_extra = getattr(session, '_native_observation_extra', 0)
            backend.observations(session)
            assert len(backend.last_observations) == 2
            assert all(row['bucket_storage_bound_bytes'] == runtime['bucket_storage_bound_bytes']
                       for row in backend.last_observations)
            assert session.static_bytes >= local_static - old_extra + runtime['bucket_storage_bound_bytes']
            for key in session.profile:
                assert session.profile[key]['duration_ns'] == max(
                    row['profile'][key]['duration_ns'] for row in backend.last_observations)
            observed_static = session.static_bytes
            backend.observations(session)
            assert session.static_bytes == observed_static
            receipts.append(deepcopy(session.receipt))
        with pytest.raises(ValueError, match='retain'):
            session.bind_cycle(workload(rank, cycle=2))
        Path(output, f'rank{rank}.json').write_text(json.dumps(dict(identity=identity,
                    receipts=receipts, gpu_execution='NOT_RUN')))
        # Reference/direct manual use the established trainer lifecycle too.
        # prepare_resume must install wrappers BEFORE checkpoint state is loaded.
        from distillschedule.training.policy import CyclePolicy
        for mode in ('reference', 'manual'):
            execution = dict(mode=mode, **({'implementation': option()} if mode == 'manual' else {}))
            policy = CyclePolicy(execution); owned = CollectiveOwner(rank)
            first, _ = policy.run(workload(rank, owner=owned), output)
            saved, bound = deepcopy(owned.state_dict()), deepcopy(policy.bound)
            second, _ = policy.run(workload(rank, owner=owned, cycle=1), output)
            assert first is second and owned.preparations == 1 and owned.steps == 2
            fresh = CollectiveOwner(rank)
            resumed = CyclePolicy(execution); resumed.restore(bound)
            next_workload = workload(rank, owner=fresh, cycle=1)
            restored = resumed.prepare_resume(next_workload)
            assert fresh.preparations == 1 and fresh.steps == 0
            fresh.load_state_dict(saved)
            continued, _ = resumed.run(next_workload, output)
            assert continued is restored and fresh.preparations == 1 and fresh.steps == 2
            for actual, expected in zip(fresh.model.parameters(), owned.model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def rejection_worker(rank, filename):
    initialize(rank, filename)
    try:
        for fault in ('schema', 'backend', 'world', 'world_bool', 'device', 'shared',
                      'nonfinite', 'rng', 'numpy_rng', 'impure', 'partial', 'mixed', 'unsafe',
                      'replay', 'readonly', 'replica', 'gradient', 'rank_gradient', 'rank_spec',
                      'no_capability', 'rank_capability'):
            w = workload(rank); owner = w.state['owner']
            if fault == 'partial': owner.prepare_collective_execution = None
            elif fault == 'mixed':
                w = replace(w, state={**w.state, 'extra': torch.nn.Linear(3, 2)},
                            program=replace(w.program, initial_versions={'owner': 0, 'extra': 0}))
            elif fault in ('unsafe', 'replay', 'readonly'):
                call = replace(w.program.calls[0],
                    grad='grad' if fault == 'replay' else ('no_grad' if fault == 'unsafe' else 'external_grad'),
                    replay_safe=fault == 'replay', writes=() if fault in ('replay', 'readonly') else ('owner',))
                w = replace(w, program=replace(w.program, calls=(call,)))
            elif fault == 'replica':
                from distillschedule.open_workload.replica_contract import ReplicaSemantics
                w.replica_semantics = ReplicaSemantics({}, (), 4)
            elif fault == 'gradient' or (fault == 'rank_gradient' and rank):
                next(owner.model.parameters()).grad = torch.ones_like(next(owner.model.parameters()))
            elif fault == 'rank_spec': owner.fault = 'changed_shared' if rank else None
            elif fault == 'no_capability' or (fault == 'rank_capability' and rank):
                w = replace(w, state={'owner': torch.nn.Linear(3, 2)})
            else: owner.fault = fault
            # One rank's rejection must be visible everywhere BEFORE any DDP install.
            rng = torch.random.get_rng_state()
            with pytest.raises(ValueError): ExecutionBackend(w)
            assert owner.preparations == 0
            assert torch.equal(torch.random.get_rng_state(), rng)
        for drift in ('feed_schema', 'graph'):
            w = workload(rank); owner = w.state['owner']
            if rank and drift == 'feed_schema': w.feeds['x'] = w.feeds['x'][:3]
            if rank and drift == 'graph': w.program = replace(w.program, parameters={'different': True})
            backend = ExecutionBackend(w)
            with pytest.raises(ValueError, match='graph|schema|layout|source'):
                backend.session(option(), copy_state=False, persistent=True)
            assert owner.preparations == 0
        w = workload(rank); owner = w.state['owner']; backend = ExecutionBackend(w)
        for copy_state, persistent in ((True, False), (True, True), (False, False)):
            with pytest.raises(ValueError, match='persistent|copy|clone'):
                backend.session(option(), copy_state=copy_state, persistent=persistent)
        invalid = [dict(option(), owner_execution={}),
                   dict(option(), local=dict(option()['local'], share_immutable=True)),
                   dict(option(), local=dict(option()['local'], checkpoint_bindings=['native'])),
                   dict(option(), local=dict(option()['local'], owner_execution={'owner': {}}))]
        for implementation in invalid:
            with pytest.raises(ValueError):
                backend.session(implementation, copy_state=False, persistent=True)
        with pytest.raises(ValueError, match='trial|calibration|clone'):
            backend.resident_trial(None)
        from distillschedule.open_workload.planning import calibrate, paired_execution
        with pytest.raises(ValueError, match='persistent|copy|clone|calibration'):
            paired_execution(backend, option(), None, persistent=True)
        calibration_root = Path(filename).with_suffix('.calibration')
        with pytest.raises(ValueError, match='persistent|copy|clone|calibration'):
            calibrate(w, calibration_root, samples=1, max_candidates=1, max_program_runs=2,
                      profile_hardware=False)
        assert not (calibration_root / 'calibration.json').exists()
        assert owner.preparations == 0
        # Source and spec changes after selection cannot execute an update.
        session = backend.session(option(), copy_state=False, persistent=True)
        owner.fault = 'world'
        with pytest.raises(ValueError, match='collective|owner|world|changed'):
            session.run()
        assert owner.steps == 0 and owner.preparations == 1
        w = workload(rank); owner = w.state['owner']
        session = ExecutionBackend(w).session(option(), copy_state=False, persistent=True)
        original = CollectiveOwner.collective_execution_spec
        CollectiveOwner.collective_execution_spec = changed_collective_spec
        try:
            with pytest.raises(ValueError, match='source|spec|changed'):
                session.run()
            assert owner.steps == 0
        finally:
            CollectiveOwner.collective_execution_spec = original
        for fault in ('bound', 'bool', 'roles', 'counts', 'nonfinite', 'topology', 'boundary'):
            owner = ReceiptOwner(rank); owner.receipt_fault = None
            session = ExecutionBackend(workload(rank, owner=owner)).session(
                option(), copy_state=False, persistent=True)
            owner.receipt_fault = fault if rank else None
            with pytest.raises(ValueError, match='runtime|topology|boundary|JSON'):
                session.run()
            assert owner.steps == 0 and owner.preparations == 1
        owner = LeakyOwner(rank)
        session = ExecutionBackend(workload(rank, owner=owner)).session(
            option(), copy_state=False, persistent=True)
        with pytest.raises(ValueError, match='zero-gradient'):
            session.run()
        assert owner.steps == 1 and not hasattr(session, 'receipt')
        from distillschedule.open_workload.torch_backend import storage_bytes
        owner = GrowingOwner(rank); w = workload(rank, owner=owner); backend = ExecutionBackend(w)
        session = backend.session(option(), copy_state=False, persistent=True)
        for cycle in range(2):
            if cycle:
                w = workload(rank, owner=owner, cycle=cycle); backend = ExecutionBackend(w)
                session.bind_cycle(w, graph=backend.graph(option()))
            session.run()
            actual_live = storage_bytes((session.state, session.feeds))
            extra = session.lifecycle.runtime['bucket_storage_bound_bytes']
            backend.observations(session)
            assert session.static_bytes == actual_live + extra
    finally:
        dist.destroy_process_group()


def test_native_owner_real_gloo2_persistent_cycles_and_rank_local_identity(tmp_path):
    mp.spawn(happy_worker, args=(str(tmp_path / 'rdzv'), str(tmp_path)), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f'rank{rank}.json').read_text()) for rank in range(2)]
    assert all(row['gpu_execution'] == 'NOT_RUN' for row in rows)
    assert rows[0]['identity'] == rows[1]['identity']
    assert all(len(row['receipts']) == 2 for row in rows)


def test_native_owner_contract_and_clone_paths_fail_before_collectives(tmp_path):
    mp.spawn(rejection_worker, args=(str(tmp_path / 'rdzv'),), nprocs=2, join=True)


def assert_no_update(owner, *, preparations):
    assert owner.preparations == preparations
    assert owner.steps == owner.forwards == owner.backwards == 0
    assert not owner.adam.state
    assert all(parameter.grad is None for parameter in owner.model.parameters())


def assert_group_usable(rank):
    rows = [None, None]
    dist.all_gather_object(rows, rank)
    assert rows == [0, 1]


def preflight_rejection_worker(rank, filename, fault):
    # Broken coordination must fail this regression in seconds, not strand a
    # test worker for the normal training group's much longer timeout.
    initialize(rank, filename, timeout=8)
    try:
        w = workload(rank); owner = w.state['owner']
        if fault == 'rank_frozen' and rank:
            owner.model.requires_grad_(False)
        elif fault == 'all_frozen':
            owner.model.requires_grad_(False)
        elif fault == 'rank_mask' and rank:
            owner.model.bias.requires_grad_(False)
        backend = ExecutionBackend(w)
        preparations = 0
        if fault.startswith('order_'):
            session = backend.session(option(), copy_state=False, persistent=True)
            preparations = 1
            orders = {'order_missing': ('native',),
                      'order_unknown': ('native', 'missing'),
                      'order_duplicate': ('native', 'native'),
                      'order_dependency': ('__open_finish__', 'native'),
                      'order_type': 7}
            action = lambda: session.run(orders[fault] if rank else None)
            expected = 'native dispatch order'
        elif fault.startswith('limit_'):
            limit = 0 if fault == 'limit_zero' else True
            action = lambda: backend.inventory(limit if rank else 1)
            expected = 'native owner inventory limit'
        else:
            action = lambda: backend.session(option(), copy_state=False, persistent=True)
            expected = 'trainable|layout'
        started = time.monotonic()
        with pytest.raises(ValueError, match=expected):
            action()
        assert time.monotonic() - started < 4
        assert_no_update(owner, preparations=preparations)
        assert_group_usable(rank)
        if fault.startswith('order_'):
            # Rejected pure validation must not consume the persistent cycle.
            assert not session._ran and not hasattr(session, 'receipt')
            session.run(iter(('native', '__open_finish__')))
            assert owner.steps == owner.forwards == owner.backwards == 1
            assert owner.preparations == 1
            assert_group_usable(rank)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('fault', ['rank_frozen', 'all_frozen', 'rank_mask'])
def test_native_preflight_live_parameter_trainability(tmp_path, fault):
    mp.spawn(preflight_rejection_worker, args=(str(tmp_path / 'rdzv'), fault),
             nprocs=2, join=True)


@pytest.mark.parametrize('fault', ['limit_zero', 'limit_bool'])
def test_native_preflight_rank_local_inventory_limit(tmp_path, fault):
    mp.spawn(preflight_rejection_worker, args=(str(tmp_path / 'rdzv'), fault),
             nprocs=2, join=True)


@pytest.mark.parametrize('fault', ['order_missing', 'order_unknown', 'order_duplicate',
                                 'order_dependency', 'order_type'])
def test_native_preflight_rank_local_dispatch_order(tmp_path, fault):
    mp.spawn(preflight_rejection_worker, args=(str(tmp_path / 'rdzv'), fault),
             nprocs=2, join=True)


@pytest.mark.parametrize('group', ['absent', 'world1'])
@pytest.mark.parametrize('declaration', ['complete', 'partial'])
def test_native_preflight_rejects_single_rank_fallback(tmp_path, group, declaration):
    assert not dist.is_initialized()
    if group == 'world1':
        dist.init_process_group('gloo', init_method='file://' + str(tmp_path / 'rdzv'),
                                rank=0, world_size=1, timeout=timedelta(seconds=8))
    try:
        w = workload(0); owner = w.state['owner']
        if declaration == 'partial':
            owner.prepare_collective_execution = None
        with pytest.raises(ValueError, match='multi-rank'):
            ExecutionBackend(w)
        assert_no_update(owner, preparations=0)
        if dist.is_initialized():
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class EvolvingMaskOwner(CollectiveOwner):
    def run(self, x):
        # Native owners can change guidance masks between updates. Only the
        # installation mask is shared; it is not a fixed per-cycle contract.
        self.model.requires_grad_(True)
        result = super().run(x)
        unwrap_module(self.model).bias.requires_grad_(False)
        return result


def evolving_mask_worker(rank, filename):
    initialize(rank, filename, timeout=8)
    try:
        from distillschedule.training.policy import CyclePolicy
        policy = CyclePolicy(dict(mode='reference')); owner = EvolvingMaskOwner(rank)
        for cycle in range(3):
            session, _ = policy.run(workload(rank, owner=owner, cycle=cycle), Path(filename).parent)
            assert owner.steps == owner.forwards == owner.backwards == cycle + 1
            assert owner.preparations == session.effects['lifecycle_installations'] == 1
            assert not unwrap_module(owner.model).bias.requires_grad
            assert all(parameter.grad is None for parameter in owner.model.parameters())
        assert_group_usable(rank)
    finally:
        dist.destroy_process_group()


def test_native_preflight_allows_evolving_native_masks(tmp_path):
    mp.spawn(evolving_mask_worker, args=(str(tmp_path / 'rdzv'),), nprocs=2, join=True)
