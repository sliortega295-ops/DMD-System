"""Lazy model/workflow registration, including installed trusted plugins.

Discovery reads package metadata only. Factory imports and effective capability
admission happen during construction, never in catalog queries.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.metadata
import json
import re

_MODELS = {}
_ALGORITHMS = {}
_GROUPS = {'model':'distillschedule.models', 'algorithm':'distillschedule.algorithms'}
_NAME = re.compile(r'[a-z0-9][a-z0-9_-]*')
_FACTORY = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*')


def _register(registry, name, *, factory, capabilities=(), description='', **metadata):
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError('registry names must be lowercase identifiers')
    if name in registry:
        raise ValueError(f'already registered: {name}')
    if not isinstance(factory, str) or ':' not in factory or not all(factory.rsplit(':', 1)):
        raise ValueError('registered factory needs an explicit module:callable')
    if (not isinstance(capabilities, (tuple, list)) or
            any(not isinstance(item, str) or not item for item in capabilities) or
            len(set(capabilities)) != len(capabilities) or not isinstance(description, str)):
        raise ValueError('capabilities must be unique nonempty strings with a text description')
    row = dict(factory=factory, capabilities=list(capabilities), description=description, **metadata)
    try:
        # A registry is metadata, not a hidden executable/Tensor state channel.
        json.dumps(row, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('registry metadata must be finite JSON') from exc
    registry[name] = deepcopy(row)


def register_model(name, *, factory, capabilities=(), description='', **metadata):
    _register(_MODELS, name, factory=factory, capabilities=capabilities,
              description=description, **metadata)


def register_algorithm(name, *, factory, capabilities=(), description='', **metadata):
    _register(_ALGORITHMS, name, factory=factory, capabilities=capabilities,
              description=description, **metadata)


def _registry(kind):
    registry = {'model':_MODELS, 'algorithm':_ALGORITHMS}.get(kind)
    if registry is None:
        raise ValueError('unknown registry kind: ' + str(kind))
    result = deepcopy(registry)
    group = _GROUPS[kind]
    # Never call EntryPoint.load(): even broken optional dependencies must not
    # prevent a lightweight description of their advertised factory reference.
    points = importlib.metadata.entry_points(group=group)
    for point in sorted(points, key=lambda p: (str(p.name), str(p.value))):
        if not isinstance(point.value, str) or _FACTORY.fullmatch(point.value) is None:
            raise ValueError('entry point requires a plain package.module:callable, without extras: ' + str(point.name))
        distribution = point.dist
        if distribution is None:
            raise ValueError('entry point must belong to an installed distribution')
        name, version = distribution.metadata.get('Name'), distribution.version
        if not all(isinstance(value, str) and value for value in (name, version)):
            raise ValueError('entry point distribution identity is incomplete')
        _register(result, point.name, factory=point.value,
                  description='Installed trusted ' + kind + ' plugin',
                  provider=dict(distribution=name, version=version, entry_point_group=group),
                  capability_validation='ON_FACTORY_ADMISSION')
    return result


def resolve(kind, name):
    registry = _registry(kind)
    if name not in registry:
        raise ValueError(f'unknown registered {kind}: {name}')
    return deepcopy(registry[name])


def describe():
    return dict(models=_registry('model'), algorithms=_registry('algorithm'))


for _name, _factory, _caps, _description in (
    ('sd15', 'sd:build_sd15', ('image', 'epsilon', 'gan'), 'Stable Diffusion 1.5'),
    ('sdxl', 'sd:build_sdxl', ('image', 'epsilon', 'lora', 'pooled_text'), 'Stable Diffusion XL'),
    ('self-forcing-wan', 'self_forcing:build_model', ('video', 'causal', 'kv_cache', 'flow'), 'Official Self-Forcing Wan'),
    ('ltx25', 'ltx:build_model', ('video', 'audio', 'flow', 'lora'), 'LTX-2.5 audio-video'),
    ('wan13b', 'wan_public:build_model', ('video', 'history', 'lora'), 'Wan1.3B / W1'),
    ('helios-nogan', 'helios_nogan_public:build_model', ('video', 'history', 'lora'), 'Helios No-GAN / W2'),
    ('helios-gan', 'helios_public:build_model', ('video', 'history', 'lora', 'gan'), 'Helios GAN / W3'),
    ('livewan', 'livewan_public:build_model', ('video', 'causal', 'kv_cache'), 'LiveWan cross-scale / W4'),
    ('h3-ar', 'h3_cached_public:build_model', ('video', 'audio', 'history', 'lora'), 'Cached H3 autoregressive / W6'),
):
    register_model(_name, factory='distillschedule.training.' + _factory,
                   capabilities=_caps, description=_description)

for _name, _factory, _description in (
    ('dmd2', 'dmd2:build_dmd2', 'Official SD DMD2 with native iteration and guidance updates'),
    ('self-forcing', 'self_forcing:build_self_forcing', 'Official causal self-rollout DMD'),
    ('dmd', 'dmd:build_algorithm', 'No-GAN DMD with model-owned denoising and predictions'),
):
    register_algorithm(_name, factory='distillschedule.training.' + _factory, description=_description)
