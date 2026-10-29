"""Opt-in collectives owned INSIDE a source-bound atomic training macrocall.

This provider installs the owner's own role wrappers once. It neither infers
ReplicaSemantics nor adds a second gradient reducer. Only persistent reference
and direct manual execution are admitted; clone/paired calibration is not.
"""
from __future__ import annotations

import inspect
import json

import torch
import torch.distributed as dist

from distillschedule.core.contracts import require
from distillschedule.runtime.role_runner import StaticRoleRunner
from distillschedule.runtime.version_sync import VersionLedger
from distillschedule.workloads.best_known.registry import fingerprint
from .owner_execution import _method_signature
from .replicated import checked, gather
from .torch_backend import Session, _stateful, code_signature, signature, tensors


BACKEND = 'torch-callable-native-owner-v1'
SCHEMA = 'distillschedule.native_collective_owner.v1'
_METHODS = ('collective_execution_spec', 'prepare_collective_execution',
            'validate_collective_execution')
_LOCAL = dict(backend='torch-callable-resident-v1', share_immutable=False, checkpoint_bindings=[])


def _require(condition, message):
    require(condition, message, path='open.collective_owner')


def present(workload):
    """A partial declaration must fail closed, never fall back to replicas."""
    return any(any(hasattr(owner, name) for name in _METHODS) for owner in workload.state.values())


def _json(value, label):
    try:
        plain = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'native collective {label} must be finite JSON') from exc
    _require(value == plain, f'{label} must be plain finite JSON')
    return plain


def _rng():
    # The checkpoint ABI already covers Python/NumPy/Torch and ONLY the owned
    # current CUDA device; do not initialize unrelated visible GPU contexts.
    from distillschedule.training.checkpoint import rng_state
    return rng_state()


def _restore_rng(value):
    from distillschedule.training.checkpoint import restore_rng
    restore_rng(value)


def _boundary(owner):
    # state_dict may detach Parameters and hide live gradients. Use the existing
    # live-tensor seam instead, before any wrapper constructor can communicate.
    parameters = [tensor for tensor in tensors(owner) if isinstance(tensor, torch.nn.Parameter)]
    _require(bool(parameters) and all(parameter.grad is None for parameter in parameters),
             'native owner requires an explicit live zero-gradient boundary')


def declaration(workload):
    """Qualify the native structural protocol without invoking either descriptor.

    Presence alone is not admission: partial or mixed declarations fail closed.
    Workload validation uses this same predicate, so native admission must not
    recursively enter itself through the local capability inventory.
    """
    if not present(workload):
        return None
    _require(workload.replica_semantics is None,
             'native owner collectives cannot be combined with ReplicaSemantics')
    _require(len(workload.state) == 1 and not workload.immutable_modules,
             'exactly one native owner is required; no mixed top-level modules or sharing')
    key, owner = next(iter(workload.state.items()))
    _require(_stateful(owner) and not isinstance(owner, (torch.nn.Module, torch.optim.Optimizer)),
             'explicit stateful native owner required')
    for name in (*_METHODS, 'runtime_tensors'):
        method = getattr(owner, name, None)
        _require(name not in getattr(owner, '__dict__', {}) and inspect.ismethod(method)
                 and method.__func__ is getattr(type(owner), name, None),
                 'native collective methods must be bound inspectable class methods')
    _require(all(call.grad == 'external_grad' and not call.replay_safe and not call.backward_of
                 and call.reads == (key,) and call.writes == (key,)
                 for call in workload.program.calls),
             'native calls must be non-replayable external_grad macrocalls exclusively reading/writing their owner')
    return key, owner


def admit(workload):
    """Rank-local, collective-free preflight. Caller exchanges ALL failures."""
    _require(dist.is_initialized() and dist.get_world_size() > 1,
             'native collectives require an explicitly initialized multi-rank group')
    declared = declaration(workload)
    _require(declared is not None, 'explicit native collective owner required')
    key, owner = declared
    workload.validate()
    backend, device = str(dist.get_backend()), workload.device
    _require((device.type == 'cpu' and backend == 'gloo') or
             (device.type == 'cuda' and backend == 'nccl'),
             'native owner requires CPU/Gloo or CUDA/NCCL with one owned device per rank')
    if device.type == 'cuda':
        _require(device.index == torch.cuda.current_device(), 'wrong native owner local CUDA device')
    _boundary(owner)
    source = dict(provider_code={cls.__module__ + '.' + cls.__qualname__: code_signature(cls)
                                 for cls in type(owner).__mro__ if cls is not object},
                  protocol_code={name: _method_signature(owner, name) for name in (*_METHODS, 'runtime_tensors')})
    before, rng = signature(owner, _local_shards=True), _rng()
    try:
        spec = owner.collective_execution_spec()
        repeated = owner.collective_execution_spec()
        pure = signature(_rng()) == signature(rng)
        _require(spec == repeated and pure and signature(owner, _local_shards=True) == before,
                 'native collective spec must be pure and stable')
    finally:
        _restore_rng(rng)
    spec = _json(spec, 'spec')
    _require(type(spec) is dict and set(spec) == {
        'schema_version', 'backend', 'world_size', 'device_type', 'shared_identity'}
        and (spec['schema_version'], spec['backend']) in (
            (SCHEMA, 'ddp'), ('distillschedule.native_sharded_owner.v1', 'fsdp2'))
        and type(spec['world_size']) is int and spec['world_size'] == dist.get_world_size()
        and spec['device_type'] == device.type
        and type(spec['shared_identity']) is dict and bool(spec['shared_identity']),
        'invalid native collective schema, world, device or shared source/layout identity')
    if spec['backend'] == 'fsdp2':
        from distillschedule.training.self_forcing_fsdp import SelfForcingFSDP2Owner
        from distillschedule.training.dmd2_fsdp import DMD2FSDP2Owner
        _require(type(owner) in (SelfForcingFSDP2Owner, DMD2FSDP2Owner) and device.type == 'cuda'
                 and dist.get_world_size() == 16,
                 'native FSDP2 protocol requires its exact versioned CUDA world16 owner')
    return dict(owner_key=key, spec=spec, **source)


def contract(workload):
    value = checked(lambda: admit(workload), 'native owner preflight')
    rows = gather(value)
    _require(all(row == rows[0] for row in rows), 'rank native owner source/spec/layout differ')
    return value


def _tensor_layout(value, *, parameter_trainability=False):
    return [dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                 device_type=t.device.type,
                 **({'requires_grad': t.requires_grad}
                    if parameter_trainability and isinstance(t, torch.nn.Parameter) else {}))
            for t in tensors(value)]


def identity(workload):
    binding = contract(workload)
    def local():
        _require(any(t.requires_grad for t in tensors(workload.state[binding['owner_key']])
                     if isinstance(t, torch.nn.Parameter)),
                 'native owner installation requires trainable live Parameters')
        value = workload.identity
        # State and native RNG legitimately differ by rank. Retain every hash,
        # in rank order, instead of demanding whole-state consensus or dropping
        # the state identity. CUDA local indices are not a shared layout fact.
        common = {key: item for key, item in value.items()
                  if key not in ('initial_state', 'feeds', 'device', 'seed')}
        common.update(native_owner=binding, feed_layout=_tensor_layout(workload.feeds),
                      # Compare live masks before wrapper construction only;
                      # native guidance masks may legitimately evolve later.
                      state_layout=_tensor_layout(workload.state, parameter_trainability=True),
                      device_type=workload.device.type)
        return dict(common=common, initial_state=value['initial_state'], feeds=value['feeds'],
                    seed=value['seed'], rng=fingerprint(signature(_rng())),
                    device=str(workload.device), rank=dist.get_rank())
    rows = gather(checked(local, 'native owner identity'))
    _require(all(row['common'] == rows[0]['common'] for row in rows),
             'rank native owner graph, source, feed schema or layout differ')
    native = dict(world_size=dist.get_world_size(), ranks=rows, digest=fingerprint(rows))
    return _json({**rows[0]['common'], 'device': workload.device.type,
                  'initial_state': fingerprint([row['initial_state'] for row in rows]),
                  'feeds': fingerprint([row['feeds'] for row in rows]),
                  'seed': [row['seed'] for row in rows], 'native_collective_owner': native}, 'identity')


def implementation(value):
    _require(type(value) is dict and set(value) == {'backend', 'local'} and value['backend'] == BACKEND
             and type(value['local']) is dict and value['local'] == _LOCAL
             and type(value['local'].get('share_immutable')) is bool,
             'native owner admits only its persistent reference; no owner_execution or local transforms')
    return value


def inventory(workload, *, limit=16):
    checked(lambda: _require(type(limit) is int and limit > 0,
                             'positive native owner inventory limit required'),
            'native owner inventory limit')
    binding = contract(workload)
    backend = binding['spec']['backend']
    return dict(candidates=[dict(backend=BACKEND, local=_json(_LOCAL, 'local implementation'))],
                total_candidates=1, exhaustive=True, capabilities=['native_owner_collectives'],
                layout='native_owner_' + backend, cross_call_overlap=False,
                unsupported=['clone-based calibration', 'paired trials', 'owner_execution overrides',
                             'generic local transforms',
                             *(['native mixed FSDP2'] if backend == 'ddp' else ['other native sharded owners'])])


def _structure(workload):
    program = workload.program.to_dict()
    program['initial_versions'] = dict.fromkeys(program['initial_versions'], 0)
    for call in program['calls']:
        if 'cycle' in call['attributes']:
            call['attributes']['cycle'] = 0
    return fingerprint(dict(program=program, bindings={key: code_signature(fn)
                           for key, fn in workload.bindings.items()}, feeds=_tensor_layout(workload.feeds)))


class NativeOwnerLifecycle:
    def __init__(self, workload, option, binding):
        self.workload, self.option, self.binding = workload, option, binding
        self.key = binding['owner_key']
        self.owner = workload.state[self.key]
        self.structure = _structure(workload)
        self.completed_calls = self.completed_cycles = 0
        self.runtime = None

    def _preflight(self, session):
        _require(session.state[self.key] is self.owner and self.workload.state[self.key] is self.owner,
                 'native cycle must retain its owned object')
        _require(admit(self.workload) == self.binding and _structure(self.workload) == self.structure,
                 'native owner source, spec or cycle structure changed')

    def _runtime(self):
        _boundary(self.owner)
        row = _json(self.owner.validate_collective_execution(), 'runtime receipt')
        _require(type(row) is dict and type(row.get('bucket_storage_bound_bytes')) is int
                 and row['bucket_storage_bound_bytes'] >= 0
                 and type(row.get('roles')) is dict and bool(row['roles'])
                 and all(type(name) is str and name and type(counts) is dict and bool(counts)
                         and all(type(key) is str and key and type(count) is int and count >= 0
                                 for key, count in counts.items()) for name, counts in row['roles'].items()),
                 'native runtime receipt requires a buffer bound and actual role counters')
        for name, expected in (('world_size', dist.get_world_size()), ('rank', dist.get_rank())):
            if name in row:
                _require(type(row[name]) is int and row[name] == expected, 'native runtime topology changed')
        if 'zero_gradients' in row:
            _require(row['zero_gradients'] is True, 'native runtime is not at a zero-gradient boundary')
        self.runtime = row
        return row

    def _effects(self, session):
        session.effects['native_collective_owner'] = dict(owner_key=self.key, spec=self.binding['spec'],
            runtime=self.runtime, completed_calls=self.completed_calls, completed_cycles=self.completed_cycles,
            preparations=1)

    def prepare(self, session):
        options = gather(self.option)
        _require(all(value == options[0] for value in options), 'rank native owner implementations differ')
        checked(lambda: (implementation(self.option), self._preflight(session)), 'native session preflight')
        identity(self.workload)
        # All rank-local preflights have succeeded. Only the source-bound owner
        # knows which internal roles can be wrapped without changing its math.
        self.owner.prepare_collective_execution()
        checked(self._runtime, 'native collective installation')
        self._effects(session)

    def before_run(self, session, order):
        checked(lambda: self._preflight(session), 'native cycle preflight')
        orders = gather(tuple(order))
        _require(all(value == orders[0] for value in orders), 'rank native macrocall orders differ')
        checked(self._runtime, 'native cycle entry boundary')
        # Session tracks live tensor high water during dispatch. A previously
        # aggregated bucket allowance must not mask new Adam/cache growth when
        # its max() runs in the next cycle; observations adds it back afterward.
        session.static_bytes -= getattr(session, '_native_observation_extra', 0)
        session._native_observation_extra = 0
        dist.barrier()

    def before_call(self, session, call):
        _require(session.state[self.key] is self.owner, 'native owner object changed')

    def after_call(self, session, call):
        self.completed_calls += 1

    def finish(self, session):
        session.synchronize()
        checked(lambda: self._preflight(session), 'native completed cycle source/spec')
        checked(self._runtime, 'native cycle completion boundary')
        self.completed_cycles += 1
        self._effects(session)


class _NativeOwnerSession(Session):
    def run(self, order=None, *, measure=False):
        def preflight():
            validate_session(self, self.workload, self.implementation, self.lifecycle,
                             copy_state=False, persistent=self.persistent)
            _require(not self._ran, 'sessions are one-shot; bind the next native cycle before reuse')
            normalized = tuple(order or (*[call.id for call in self.workload.program.calls], '__open_finish__'))
            # Reuse the dispatcher's complete validation, not a second set of
            # ordering rules. These handlers are never executed in preflight.
            runner = StaticRoleRunner('open_runtime', {'call': lambda ctx: None, 'barrier': lambda ctx: None},
                                      versions=VersionLedger(self.graph.metadata['initial_versions']))
            runner.prepare(self.graph, order=normalized, retain_outputs=self.workload.program.outputs)
            return normalized
        normalized = checked(preflight, 'native dispatch order')
        # The existing Session alone owns dispatch and lifecycle collectives.
        return super().run(normalized, measure=measure)


def validate_session(session, workload, local, lifecycle, *, copy_state, persistent):
    """Only the qualified native lifecycle may consume a native declaration."""
    declared = declaration(workload)
    if declared is None:
        _require(type(session) is not _NativeOwnerSession,
                 'native session requires its retained native owner lifecycle')
        return
    key, owner = declared
    _require(type(session) is _NativeOwnerSession and type(lifecycle) is NativeOwnerLifecycle
             and lifecycle.key == key and lifecycle.owner is owner
             and lifecycle.workload.state.get(key) is owner,
             'native owner requires its qualified native session lifecycle')
    implementation(lifecycle.option)
    _require(local == lifecycle.option['local'] and copy_state is False and persistent is True,
             'native lifecycle requires its exact local implementation and persistent owned state')


def make_session(workload, option, *, copy_state=True, persistent=False, binding=None):
    def preflight():
        implementation(option)
        _require(copy_state is False and persistent is True,
                 'native owner requires copy_state=False and persistent=True; clone/calibration/paired trials are not admitted')
        current = admit(workload)
        _require(binding is None or current == binding, 'native owner source/spec changed since selection')
        _structure(workload)  # inspect binding source before any constructor collective
        return current
    binding = checked(preflight, 'native persistent session admission')
    return _NativeOwnerSession(workload, option['local'], lifecycle=NativeOwnerLifecycle(workload, option, binding),
                               copy_state=False, persistent=True)
