"""Role-pool ownership adapter for the common same-topology checkpoint writer."""
from __future__ import annotations

import hashlib
import inspect

import torch

from distillschedule.core.contracts import require
from distillschedule.training import distributed
from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint, read_manifest


class _Progress:
    def __init__(self, executor):
        self.executor = executor

    def state_dict(self):
        owner = self.executor
        return dict(cycle=owner._cycle, versions=dict(owner._versions),
                    rank=owner.rank, role=owner.role)

    def validate_state_dict(self, value):
        owner = self.executor
        require(isinstance(value, dict) and set(value) == {'cycle', 'versions', 'rank', 'role'}
                and type(value['cycle']) is int and value['cycle'] >= 0
                and value['rank'] == owner.rank and value['role'] == owner.role,
                'role checkpoint progress owner changed', path='role.checkpoint')
        cycle = value['cycle']
        expected = dict(generator=cycle, optimizer_g=cycle,
                        gradient_g=cycle * (owner.task.microbatches + 1),
                        score=cycle * owner.task.score_updates,
                        optimizer_s=cycle * owner.task.score_updates)
        require(value['versions'] == expected and
                all(type(v) is int for v in value['versions'].values()),
                'role checkpoint update versions disagree with completed cycles', path='role.checkpoint')

    def validate_checkpoint_progress(self, value, progress):
        self.validate_state_dict(value)
        require(progress is not None and progress == dict(cycle=value['cycle'],
                    cursor=value['cycle'] * self.executor.task.microbatches, versions=value['versions']),
                'role checkpoint header and owner progress disagree', path='role.checkpoint')

    def validate_checkpoint_boundary(self):
        owner = self.executor
        require(owner._prepared and not (owner._busy or owner._failed or owner._closed),
                'role checkpoint requires a drained successful cycle boundary', path='role.checkpoint')
        require(not getattr(owner, '_active_offloaders', ()),
                'role checkpoint has outstanding activation leases', path='role.checkpoint')

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        self.executor._cycle = value['cycle']
        self.executor._versions = dict(value['versions'])


class RoleCheckpoint:
    """Persist the real role model, optimizer, extra owners and cycle clock.

    The binding opts in only after declaring all stateful data/scheduler/EMA
    owners. Process groups, autograd graphs and activation caches are rebuilt.
    """
    def __init__(self, executor, plan):
        self.executor = executor
        def admission():
            require(executor.bindings.checkpoint_safe and executor.bindings.sample_at is not None
                    and bool(executor.bindings.data_identity),
                    'role restart requires checkpoint_safe, sample_at and data_identity',
                    path='role.checkpoint')
            require(not (executor._busy or executor._failed or executor._closed),
                    'cannot attach checkpoint to an active, failed or closed role owner', path='role.checkpoint')
        distributed.checked(admission, 'role checkpoint admission')
        executor._prepare()
        progress = _Progress(executor)
        def owners():
            provider = executor.bindings.state_owners
            extras = {} if provider is None else provider(executor.role, executor.module, executor.optimizer)
            require(isinstance(extras, dict) and all(isinstance(k, str) and k.isidentifier() for k in extras),
                    'role state_owners must return named live owners', path='role.checkpoint')
            require(len({id(v) for v in extras.values()}) == len(extras) and all(
                        not isinstance(v, (torch.nn.Module, torch.optim.Optimizer))
                        and callable(getattr(v, 'state_dict', None))
                        and callable(getattr(v, 'load_state_dict', None)) for v in extras.values()),
                    'extra state owners must be distinct stateful objects, not model/optimizer snapshots',
                    path='role.checkpoint')
            state = {'model': executor.module, 'progress': progress}
            if executor.optimizer is not None:
                state['optimizer'] = executor.optimizer
            state.update({'extra_' + k: v for k, v in extras.items()})
            # Bind custom owner implementations as well as the factory. A
            # scheduler/data class cannot change behind an unchanged provider.
            schemas = {k: dict(cls=type(v).__module__ + '.' + type(v).__qualname__,
                               source=hashlib.sha256(inspect.getsource(type(v)).encode()).hexdigest())
                       for k, v in extras.items()}
            return state, schemas
        self.state, schemas = distributed.checked(owners, 'role checkpoint state owners')
        self.identity = dict(schema_version='distillschedule.role_checkpoint.v1',
            plan_sha256=plan['plan_sha256'], run_identity=plan['identity'],
            task=executor.task.to_dict(), layout=executor.layout.to_dict(),
            producer_dispatch=executor.producer_dispatch,
            state_owners=distributed.gather(dict(rank=executor.rank, role=executor.role, schemas=schemas)))

    def data_progress(self, cycle):
        return dict(schema_version='distillschedule.role_data_progress.v1',
                    data_identity=self.executor.bindings.data_identity, next_cycle=cycle,
                    next_microbatch=0, microbatches_per_cycle=self.executor.task.microbatches,
                    cursor_unit='global_logical_generator_microbatch')

    def save(self, path, *, policy=None):
        owner = self.executor
        return save_checkpoint(path, self.state, identity=self.identity, cycle=owner._cycle,
            cursor=owner._cycle * owner.task.microbatches, versions=owner._versions,
            policy=policy, data_progress=self.data_progress(owner._cycle))

    def restore(self, path):
        owner = self.executor
        def admission():
            self.state['progress'].validate_checkpoint_boundary()
            require(owner._cycle == 0 and not any(owner._versions.values()),
                    'role restore requires a fresh executor', path='role.checkpoint')
        distributed.checked(admission, 'role restart owner admission')
        manifest = distributed.checked(lambda: read_manifest(path, identity=self.identity),
                                       'role restart metadata admission')
        distributed.checked(lambda: require(manifest.get('data_progress') == self.data_progress(manifest['cycle']),
                    'role checkpoint data progress changed', path='role.checkpoint'),
                    'role restart data admission')
        try:
            payload = restore_checkpoint(path, self.state, identity=self.identity)
            distributed.checked(lambda: owner._check_state(owner.module, owner.optimizer, audit=True),
                                'role restored finite state')
            return payload
        except BaseException:
            owner._failed = True
            raise
