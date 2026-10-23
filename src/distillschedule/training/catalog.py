"""Lazy public model/workflow registry. Discovery imports no model libraries."""
from __future__ import annotations

from copy import deepcopy
import re

_MODELS = {}
_ALGORITHMS = {}


def _register(registry, name, *, factory, capabilities=(), description='', **metadata):
    if not isinstance(name, str) or re.fullmatch(r'[a-z0-9][a-z0-9_-]*', name) is None:
        raise ValueError('registry names must be lowercase identifiers')
    if name in registry:
        raise ValueError(f'already registered: {name}')
    if not isinstance(factory, str) or ':' not in factory:
        raise ValueError('registered factory needs an explicit module:callable')
    registry[name] = dict(factory=factory, capabilities=list(capabilities),
                          description=description, **metadata)


def register_model(name, *, factory, capabilities=(), description='', **metadata):
    _register(_MODELS, name, factory=factory, capabilities=capabilities,
              description=description, **metadata)


def register_algorithm(name, *, factory, capabilities=(), description='', **metadata):
    _register(_ALGORITHMS, name, factory=factory, capabilities=capabilities,
              description=description, **metadata)


def resolve(kind, name):
    registry = {'model': _MODELS, 'algorithm': _ALGORITHMS}.get(kind)
    if registry is None or name not in registry:
        raise ValueError(f'unknown registered {kind}: {name}')
    return deepcopy(registry[name])


def describe():
    return dict(models=deepcopy(_MODELS), algorithms=deepcopy(_ALGORITHMS))


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
