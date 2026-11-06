"""Local state inspection with an explicit, collectively agreed audit cadence.

Metadata checks never gather a DTensor or copy payloads to the host. Content
audits compare only real replica domains: different shards normally differ.
No tensor payload or strong model reference is cached by these helpers.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Mapping
import inspect
import math

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint

DEFAULT_EVERY = 50
_AUDIT = ContextVar('distillschedule_state_content_audit', default=False)


class _TensorRecord(dict):
    """Pickle-safe local leaf tag, so user mapping keys cannot impersonate it."""


def validate_every(every):
    require(type(every) is int and every > 0, 'state audit cadence must be a positive integer',
            path='open.state_audit')
    return every


def content_due():
    return _AUDIT.get()


@contextmanager
def audit_scope(*, cycle, every=DEFAULT_EVERY, persistent=True, measure=False, force=False):
    validate_every(every)
    require(type(cycle) is int and cycle > 0, 'positive audit cycle required', path='open.state_audit')
    token = _AUDIT.set(force or measure or not persistent or cycle == 1 or cycle % every == 0)
    try:
        yield content_due()
    finally:
        _AUDIT.reset(token)


def _tensor(value, *, content=False, tokens=False):
    from .torch_backend import physical_tensor, tensor_signature, _immutable_parameter_state
    from torch.distributed.tensor import DTensor
    local = physical_tensor(value)
    info = dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                device_type=value.device.type, layout=str(value.layout), requires_grad=value.requires_grad)
    if isinstance(value, DTensor):
        from distillschedule.training.tensor_state import layout
        info['distributed'] = layout(value)  # Also rejects unresolved Partial placements.
    if tokens:
        _, mutation = _immutable_parameter_state(value)
        info['mutation'] = (id(value), mutation)
    if content or (tokens and any(version is None for version in mutation[1])):
        info['content'] = tensor_signature(local, _local_shards=True)
    return info


def mutation_snapshot(value):
    """Guard descriptor purity using live local mutation tokens and scalar clocks.

Inspect actual object fields rather than state_dict(), which some owners stage
or clone. Scheduled content audits also protect raw writes bypassing versions.
Opaque runtime handles are bound by identity; no global object cache exists.
"""
    import torch
    seen = {}
    def visit(item):
        if isinstance(item, torch.Tensor):
            return _tensor(item, content=content_due(), tokens=True)
        if item is None or type(item) in (str, int, bool, float):
            return repr(item) if type(item) is float and not math.isfinite(item) else item
        if (inspect.ismodule(item) or inspect.isclass(item) or inspect.isfunction(item)
                or inspect.ismethod(item) or inspect.isbuiltin(item)):
            # Source modules contain entire imported package graphs. Their
            # source/method identities are independently admitted; walking all
            # globals is neither training-state inspection nor a bounded check.
            return (type(item).__module__, type(item).__qualname__, id(item),
                    id(getattr(item, '__code__', None)))
        identity = id(item)
        if identity in seen:
            return ('alias', seen[identity])
        seen[identity] = len(seen)
        if isinstance(item, Mapping):
            return [(visit(key), visit(child)) for key, child in item.items()]
        if isinstance(item, (tuple, list)):
            return (type(item).__name__, [visit(child) for child in item])
        if isinstance(item, (set, frozenset)):
            return sorted((visit(child) for child in item), key=repr)
        if hasattr(item, '__dict__') and not isinstance(item, type):
            return (type(item).__module__, type(item).__qualname__,
                    {key: visit(child) for key, child in vars(item).items()
                     if not key.startswith('_state_audit_')})
        return (type(item).__module__, type(item).__qualname__, identity)
    return visit(value)


def pure_descriptor(owner, method_name):
    """First/source-new inspection is full; ordinary repeated calls use tokens.

    Always call the actual descriptor twice and compare live clocks/RNG/state.
    Only the admitted function identity is remembered, never its mutable JSON.
    """
    from .collective_owner import _rng, _restore_rng
    from .torch_backend import signature
    function = getattr(type(owner), method_name)
    key = '_state_audit_descriptor_' + method_name
    initial = getattr(owner, key, None) is not function
    token = _AUDIT.set(content_due() or initial)
    try:
        before, rng = mutation_snapshot(owner), _rng()
        try:
            method = getattr(owner, method_name)
            value, repeated = method(), method()
            pure_rng = signature(_rng()) == signature(rng)
            require(value == repeated and pure_rng and mutation_snapshot(owner) == before,
                    'owner descriptor must be pure and stable', path='open.state_audit')
        finally:
            _restore_rng(rng)
        setattr(owner, key, function)
        return value
    finally:
        _AUDIT.reset(token)


class FrozenGuard:
    """Version/storage guard for existing admitted immutable content signatures."""
    def __init__(self):
        self._tokens = {}

    def __deepcopy__(self, memo):
        # A cloned owner has different storages; weak storage handles must not
        # be copied as if they certified the clone's immutable contents.
        return type(self)()

    def check(self, values, expected, signature, *, full=False, message='frozen state changed'):
        from .torch_backend import _immutable_parameter_state
        from .torch_backend import physical_tensor
        from torch.multiprocessing.reductions import StorageWeakRef
        require(values.keys() == expected.keys(), message, path='open.state_audit')
        for name, tensor in values.items():
            metadata, token = _immutable_parameter_state(tensor)
            current = (metadata, token, fingerprint(expected[name]))
            admitted = self._tokens.get(name)
            storage = physical_tensor(tensor).untyped_storage()
            valid_storage = (admitted is not None and not admitted[1].expired()
                             and admitted[1].cdata == storage._cdata)
            if (full or content_due() or not valid_storage or admitted[0] != current
                    or None in token[1]):
                require(signature(tensor) == expected[name], message, path='open.state_audit')
                self._tokens[name] = (current, StorageWeakRef(storage))


def check_frozen(owner, name, values, expected, signature, *, message):
    key = '_state_audit_' + name
    guard = getattr(owner, key, None)
    if guard is None:
        guard = FrozenGuard()
        setattr(owner, key, guard)
    guard.check(values, expected, signature, message=message)


def state_record(value, *, content):
    """Serializable local view; hashing is allowed only at explicit audit points."""
    import torch
    from .torch_backend import unwrap_module, _stateful
    if isinstance(value, torch.Tensor):
        return _TensorRecord(tensor=_tensor(value, content=content))
    if isinstance(value, torch.nn.Module):
        value = unwrap_module(value)
        return {'module': type(value).__qualname__,
                'training': [(name, module.training) for name, module in value.named_modules()],
                'extra_state': {name: state_record(module.get_extra_state(), content=content)
                                for name, module in value.named_modules()
                                if type(module).get_extra_state is not torch.nn.Module.get_extra_state},
                'parameters': {name: state_record(tensor, content=content)
                               for name, tensor in value.named_parameters(remove_duplicate=False)},
                'buffers': {name: state_record(tensor, content=content)
                            for name, tensor in value.named_buffers(remove_duplicate=False)}}
    if isinstance(value, torch.optim.Optimizer) or _stateful(value):
        return state_record(value.state_dict(), content=content)
    if isinstance(value, Mapping):
        return {str(key): state_record(child, content=content) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [state_record(child, content=content) for child in value]
    require(value is None or type(value) in (str, int, float, bool),
            'state requires explicit scalar or tensor metadata', path='open.state_audit')
    return repr(value) if type(value) is float and not math.isfinite(value) else value


def _compare_records(rows, ranks, *, content):
    """Compare tensor contents within corresponding replica copies, not shards."""
    import torch
    meshes = {}
    def visit(values):
        first = values[0]
        require(all(type(value) is type(first) for value in values), 'state layout diverged', path='open.state_audit')
        if isinstance(first, _TensorRecord):
            infos = [value['tensor'] for value in values]
            dist_info = [info.get('distributed') for info in infos]
            common = [{key: child for key, child in info.items() if key not in ('content', 'distributed')}
                      for info in infos]
            require(all(item == common[0] for item in common), 'state tensor metadata diverged', path='open.state_audit')
            if all(item is None for item in dist_info):
                domains = {(): list(range(len(values)))}
            else:
                require(all(item is not None for item in dist_info), 'state distributed layout diverged', path='open.state_audit')
                topology = [{key: child for key, child in info.items() if key != 'local_shape'} for info in dist_info]
                require(all(item == topology[0] for item in topology), 'state mesh/placements diverged', path='open.state_audit')
                key = fingerprint(dist_info[0]['mesh'])
                if key not in meshes:
                    mesh = torch.tensor(dist_info[0]['mesh'])
                    coordinates = {rank: (mesh == rank).nonzero().flatten().tolist() for rank in ranks}
                    meshes[key] = mesh, coordinates
                mesh, coordinates = meshes[key]
                require(sorted(mesh.flatten().tolist()) == sorted(ranks), 'state mesh does not match owning ranks', path='open.state_audit')
                domains = {}
                for index, rank in enumerate(ranks):
                    coord = coordinates[rank]
                    expected_shape = list(dist_info[index]['shape'])
                    for axis, placement in enumerate(dist_info[index]['placements']):
                        if placement['kind'] == 'shard':
                            dim = placement['dim']
                            width = math.ceil(expected_shape[dim] / mesh.shape[axis])
                            expected_shape[dim] = max(0, min(width, expected_shape[dim] - width * coord[axis]))
                    require(dist_info[index]['local_shape'] == expected_shape,
                            'state local shard shape does not match mesh/placements', path='open.state_audit')
                    shard = tuple(coord[axis] for axis, placement in enumerate(dist_info[index]['placements'])
                                  if placement['kind'] == 'shard')
                    domains.setdefault(shard, []).append(index)
                for indices in domains.values():
                    require(all(dist_info[index]['local_shape'] == dist_info[indices[0]]['local_shape']
                                for index in indices), 'replica local shard shape diverged', path='open.state_audit')
            if content:
                require(all(len({infos[index]['content'] for index in indices}) == 1
                            for indices in domains.values()), 'state contents diverged', path='open.state_audit')
        elif isinstance(first, dict):
            require(all(value.keys() == first.keys() for value in values), 'state keys diverged', path='open.state_audit')
            for key in first:
                visit([value[key] for value in values])
        elif isinstance(first, list):
            require(all(len(value) == len(first) for value in values), 'state length diverged', path='open.state_audit')
            for index in range(len(first)):
                visit([value[index] for value in values])
        else:
            require(all(value == first for value in values), 'state scalar metadata diverged', path='open.state_audit')
    visit(rows)


def distributed_audit(session, *, force=False):
    from .replicated import checked, gather
    import torch.distributed as dist
    full = force or content_due()
    controls = gather(cycle_controls(session, full))
    require(all(value == controls[0] for value in controls), 'state audit cadence or logical program clocks differ across ranks',
            path='open.state_audit')
    row = checked(lambda: state_record(session.state, content=full), 'local state audit')
    rows = gather(row)
    _compare_records(rows, list(range(dist.get_world_size())), content=full)
    receipt = dict(kind='content_audit' if full else 'metadata', cycle=session._audit_cycle,
                   every_cycles=session.state_audit_every_cycles, local_shards=True,
                   replica_content_consensus=full)
    session.effects.update(state_audit=receipt, replica_state_consensus=full)
    return receipt


def cycle_controls(session, full, *, native_owner=False):
    workload = getattr(session, 'workload', None)
    program = workload.program if workload is not None and not native_owner else None
    return (session._audit_cycle, session.state_audit_every_cycles, full,
            (program.digest, tuple(sorted(program.initial_versions.items()))) if program is not None else None)
