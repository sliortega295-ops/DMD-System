"""Reuse verified frozen role weights before copying another role to its device."""
from contextvars import ContextVar
from copy import deepcopy

from distillschedule.workloads.best_known.registry import fingerprint
from .low_rank import TrainableLowRankLinear
from .single_device import (
    FLOW_CUDA_EXECUTION, FLOW_REPLICATED_EXECUTION, execution_policy_matches, place_module,
)

_ACTIVE = ContextVar('dmd_frozen_construction', default=None)


def _require(condition, message):
    if not condition:
        raise ValueError('frozen role construction: ' + message)


def frozen_native_subtrees(module, prefix):
    """Maximal fully frozen subtrees, using the existing immutable-module ABI."""
    result = []

    def visit(child, path):
        parameters = tuple(child.parameters())
        if parameters and all(not p.requires_grad for p in parameters):
            result.append(path)
        else:
            for name, nested in child.named_children():
                visit(nested, '.'.join(part for part in (path, name) if part))

    visit(module, prefix)
    return tuple(result)


def _frozen_slots(native):
    roots = frozen_native_subtrees(native, '')
    wrappers = sorted((name for name, child in native.named_modules()
                       if isinstance(child, TrainableLowRankLinear)), key=len, reverse=True)
    for name, parameter in native.named_parameters(remove_duplicate=False):
        if parameter.requires_grad or not any(not root or name.startswith(root + '.') for root in roots):
            continue
        canonical = name
        for wrapper in wrappers:
            prefix = wrapper + '.base.'
            if name.startswith(prefix):
                canonical = wrapper + '.' + name[len(prefix):]
                break
        yield name, canonical, parameter


class _FrozenRolePool:
    def __init__(self):
        self.parameters = {}
        self.aliases = 0
        self.bytes_reused = 0
        self.roles = 0

    def prepare(self, native, identity):
        from distillschedule.open_workload.torch_backend import tensor_signature
        _require(isinstance(identity, dict) and isinstance(identity.get('config'), dict),
                 'verified native checkpoint and config identity required')
        config_key = (type(native), fingerprint(identity['config']))
        keys = {}
        expected = {}
        counted = set()
        # All signatures are computed on the fresh CPU owner, before aliases
        # or device copies. A different checkpoint can still have identical
        # frozen layers; no GPU-to-host comparison is needed to find them.
        slots = list(_frozen_slots(native))
        signatures = {}
        for _, _, parameter in slots:
            _require(parameter.device.type == 'cpu', 'share native roles before device transfer')
            if id(parameter) not in signatures:
                signatures[id(parameter)] = tensor_signature(parameter)
        for name, canonical, parameter in slots:
            key = (*config_key, canonical, signatures[id(parameter)])
            keys[name] = key
            source = self.parameters.get(key)
            if source is None:
                continue
            _require(not source.requires_grad and source.shape == parameter.shape
                     and source.dtype == parameter.dtype and source.is_contiguous()
                     and parameter.is_contiguous(), 'frozen checkpoint parameter layout changed')
            parent, _, leaf = name.rpartition('.')
            owner = native.get_submodule(parent) if parent else native
            setattr(owner, leaf, source)
            expected[name] = source
            if id(parameter) not in counted and parameter is not source:
                self.aliases += 1
                self.bytes_reused += parameter.numel() * parameter.element_size()
                counted.add(id(parameter))
        return keys, expected

    def publish(self, native, keys, expected):
        for name, source in expected.items():
            _require(native.get_parameter(name) is source and not source.requires_grad,
                     'placement changed a shared frozen Parameter')
        for name, key in keys.items():
            parameter = native.get_parameter(name)
            _require(not parameter.requires_grad, 'placement changed frozen parameter ownership')
            self.parameters.setdefault(key, parameter)
        self.roles += 1


def place_frozen_role(module, device, *, identity, native=None, require=_require):
    """Keep ordinary constructors and LoRA RNG; change only frozen storage placement.

    The caller supplies the config and checkpoint hashes already checked by its
    native loader. Corresponding keys may alias across roles, never within a role.
    Modules, buffers, adapters, optimizers and EMA remain independently owned.
    """
    native = module if native is None else native
    pool = _ACTIVE.get()
    prepared = pool.prepare(native, identity) if pool is not None else None
    place_module(module, device, require=require)
    if pool is not None:
        pool.publish(native, *prepared)
    return module


class _SharedFrozenConstruction:
    def __init__(self, parameters, factory):
        self.parameters = deepcopy(parameters)
        self.factory = factory
        self.pool = _FrozenRolePool()
        self.started = self.closed = False
        self.components = None

    def construct(self, factory, parameters):
        _require(not self.started and not self.closed and factory is self.factory
                 and parameters == self.parameters and _ACTIVE.get() is None,
                 'construction provider is single-use and bound to its factory/parameters')
        self.started = True
        token = _ACTIVE.set(self.pool)
        try:
            self.components = factory(parameters)
            return self.components
        except BaseException:
            self.close()
            raise
        finally:
            _ACTIVE.reset(token)

    def attach(self, components):
        _require(not self.closed and self.started and components is self.components and self.pool.roles > 0,
                 'factory did not publish its actual native role construction')
        components.construction_receipt = {
            'schema_version': 'distillschedule.frozen_role_construction.v1',
            'placement': 'BEFORE_DEVICE_TRANSFER', 'roles': self.pool.roles,
            'aliased_parameters': self.pool.aliases,
            'frozen_parameter_bytes_reused': self.pool.bytes_reused,
        }
        # The actual role modules now own every live alias. Do not retain
        # constructor-only Parameters that runtime sharing may replace later.
        self.pool.parameters.clear()

    def close(self):
        self.pool.parameters.clear()
        self.components = None
        self.closed = True


def admit_frozen_sharing(*, parameters, request, budget, factory):
    """Bind the existing explicit sharing strategy to the preconstruction hook."""
    if (budget is not None or parameters.get('construction') is not None
            or parameters.get('trainable', {}).get('mode') != 'lora'
            or request.execution.get('mode') != 'manual'):
        return None
    if not any(execution_policy_matches(request.execution, constraints)
               for constraints in (FLOW_CUDA_EXECUTION, FLOW_REPLICATED_EXECUTION)):
        return None
    return _SharedFrozenConstruction(parameters, factory)
