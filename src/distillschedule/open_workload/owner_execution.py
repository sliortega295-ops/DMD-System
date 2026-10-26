"""Optional, bounded physical options on explicitly stateful logical owners.

Descriptors are pure JSON inspections, not a second planner or state authority.
The owner's mathematical macrocall remains atomic and need not be replay safe.
"""
from copy import deepcopy
import itertools
import inspect
import json
import random
import torch

from distillschedule.core.contracts import require

SCHEMA = 'distillschedule.owner_execution.v1'


def _component_keys(effect, option):
    """Validate effect bindings, retaining the original shared-toggle ABI.

    A per-role policy cannot use one global boolean as proof that both roles
    executed the chosen strategy. Every physical flag must own exactly one
    component; effect receipts then qualify each component independently.
    """
    if type(effect) is not dict:
        return None
    if (set(effect) == {'kind', 'enabled_key', 'components'}
            and effect['kind'] == 'native_checkpoint'
            and type(effect['enabled_key']) is str and effect['enabled_key'] in option):
        names = effect['components']
        if (type(names) is list and 1 <= len(names) <= 16
                and all(type(name) is str and name for name in names)
                and len(set(names)) == len(names)):
            return dict.fromkeys(names, effect['enabled_key'])
    if (set(effect) == {'kind', 'enabled_keys'}
            and effect['kind'] == 'native_component_checkpoint'):
        keys = effect['enabled_keys']
        if (type(keys) is dict and 1 <= len(keys) <= 16
                and all(type(name) is str and name and type(key) is str and key in option
                        for name, key in keys.items())
                and len(set(keys.values())) == len(keys) and set(keys.values()) == set(option)):
            return keys
    return None


def _method_signature(owner, name):
    from .torch_backend import code_signature, signature
    from distillschedule.workloads.best_known.registry import fingerprint
    method = inspect.unwrap(getattr(type(owner), name))
    closure = dict(inspect.getclosurevars(method).nonlocals)
    intrinsic = closure.pop('__class__', None)
    if intrinsic is None:
        return code_signature(method)
    # Zero-argument super() creates this intrinsic class cell, not external
    # mathematical state. Bind only a class in the already-inspected owner MRO;
    # all remaining closure values retain the normal signature restrictions.
    require(intrinsic in type(owner).__mro__, 'owner method captures an unrelated class', path='open.owner')
    return fingerprint({'source': inspect.getsource(method), 'closure': signature(closure),
                        'class': code_signature(intrinsic)})


def inventory(state):
    from .torch_backend import _stateful, code_signature, signature
    result = {}
    for key, owner in state.items():
        if not any(hasattr(owner, name) for name in ('execution_options', 'execution_scope')):
            continue
        require(_stateful(owner) and callable(getattr(owner, 'execution_options', None))
                and callable(getattr(owner, 'execution_scope', None)),
                'owner execution requires a complete stateful provider', path='open.owner')
        require(all(inspect.ismethod(getattr(owner, name))
                    and getattr(owner, name).__func__ is getattr(type(owner), name)
                    for name in ('execution_options', 'execution_scope')),
                'owner execution methods must be bound inspectable class methods', path='open.owner')
        before = signature(owner)
        py, rng = random.getstate(), torch.random.get_rng_state()
        try:
            descriptor = owner.execution_options()
            repeated = owner.execution_options()
            pure_rng = random.getstate() == py and torch.equal(rng, torch.random.get_rng_state())
        finally:
            random.setstate(py); torch.random.set_rng_state(rng)
        require(descriptor == repeated and signature(owner) == before and pure_rng,
                'owner execution descriptor must be pure and stable', path='open.owner')
        try:
            plain = json.loads(json.dumps(descriptor, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError('owner execution descriptor must be finite JSON') from exc
        require(type(descriptor) is dict and descriptor == plain and set(descriptor) == {
            'schema_version', 'baseline', 'options', 'effect', 'identity'}
            and descriptor['schema_version'] == SCHEMA,
            'invalid owner execution descriptor', path='open.owner')
        baseline, options, effect = (descriptor[name] for name in ('baseline', 'options', 'effect'))
        require(type(baseline) is dict and 1 <= len(baseline) <= 16
                and all(type(k) is str and k and type(v) is bool for k, v in baseline.items())
                and type(options) is list and 1 <= len(options) <= 16
                and all(type(option) is dict and option.keys() == baseline.keys()
                        and all(type(v) is bool for v in option.values()) for option in options)
                and baseline in options and len({json.dumps(o, sort_keys=True) for o in options}) == len(options),
                'owner options must be finite distinct boolean maps including baseline', path='open.owner')
        require(_component_keys(effect, baseline) is not None
                and type(descriptor['identity']) is dict and bool(descriptor['identity']),
                'owner execution requires source identity and native effect predicates', path='open.owner')
        result[key] = {'descriptor': deepcopy(descriptor),
                       'provider_code': {cls.__module__ + '.' + cls.__qualname__: code_signature(cls)
                                         for cls in type(owner).__mro__ if cls is not object},
                       'protocol_code': {name: _method_signature(owner, name)
                                         for name in ('execution_options', 'execution_scope')}}
    require(len(result) <= 16, 'owner provider inventory exceeds bound', path='open.owner')
    return result


def alternatives(providers):
    keys = sorted(providers)
    choices = [[None, *(option for option in providers[key]['descriptor']['options']
                        if option != providers[key]['descriptor']['baseline'])] for key in keys]
    for options in itertools.product(*choices):
        yield {key: deepcopy(option) for key, option in zip(keys, options) if option is not None}


def admit(implementation, providers):
    if 'owner_execution' not in implementation:
        return {}
    selected = implementation['owner_execution']
    require(type(selected) is dict and bool(selected) and set(selected) <= set(providers),
            'unknown or empty owner execution selection', path='open.owner')
    for key, option in selected.items():
        require(len(providers[key]['descriptor']['options']) > 1
                and type(option) is dict and all(type(value) is bool for value in option.values())
                and option in providers[key]['descriptor']['options'],
                'owner execution option not admitted by descriptor', path='open.owner')
    return deepcopy(selected)


def selected(implementation):
    """Local option presence, also for the existing replicated policy envelope."""
    return bool(implementation.get('local', implementation).get('owner_execution'))


def evidence(implementation, rows):
    options = implementation.get('owner_execution', {})
    ranks = []
    for row in rows:
        owners = {}
        for key, option in options.items():
            actual = row.get('owner_execution', {}).get(key, {})
            effect = actual.get('effect', {})
            components = actual.get('components', {})
            keys = _component_keys(effect, option)
            active = (actual.get('option') == option and keys is not None
                      and type(components) is dict and set(components) == set(keys or {}))
            for name, enabled_key in (keys or {}).items():
                item = components.get(name) if type(components) is dict else None
                if type(item) is not dict:
                    active = False
                    continue
                enabled = option[enabled_key]
                requests, replay = item.get('checkpoint_requests'), item.get('recompute_calls')
                counts = type(requests) is int and type(replay) is int and requests >= 0 and replay >= 0
                active = active and type(enabled) is bool and counts and ((requests > 0 and replay > 0) if enabled else
                    (requests == replay == 0 and item.get('disabled_verified') is True))
                if effect['kind'] == 'native_component_checkpoint':
                    forwards, retained = item.get('forward_calls'), item.get('retained_calls')
                    active = (active and type(forwards) is int and forwards > 0
                              and type(retained) is int
                              and retained == (0 if enabled else forwards))
            owners[key] = {'active': bool(active), 'option': deepcopy(option),
                           'components': deepcopy(components)}
        ranks.append(owners)
    active = bool(ranks) and all(owner['active'] for rank in ranks for owner in rank.values())
    return {'active': active, 'rank_evidence': ranks,
            'status': 'ACTIVE_NATIVE_OWNER_EXECUTION' if active else 'INAPPLICABLE_NATIVE_OWNER_EXECUTION'}
