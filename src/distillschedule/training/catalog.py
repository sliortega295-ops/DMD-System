"""Lazy model/workflow registration, including installed trusted plugins.

Discovery reads package metadata only. Factory imports and effective capability
admission happen during construction, never in catalog queries.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.metadata
import json
import re
from .single_device import FLOW_CUDA_EXECUTION, FLOW_REPLICATED_EXECUTION, execution_policy_matches, _exact_value

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


# Static request contracts only. Source/assets, geometry, precision, live world
# size and qualified resident capacity remain the actual native owners' gates.
_NATIVE_SINGLE = dict(devices=['cpu', 'cuda'], distributed=False)
_DMD2_DDP = dict(devices=['cpu', 'cuda'], distributed=True, forbidden_fields=['order', 'plan'],
                 fixed_parameters={'native_checkpoint_policy': 'coupled-v1'})
_DMD2_FSDP = dict(_DMD2_DDP, devices=['cuda'])
_SF_FSDP = dict(devices=['cuda'], distributed=True, fixed_parameters={'native_cache_lifetime': 'native-v1'})
_NATIVE_IMPLEMENTATION = dict(backend='torch-callable-native-owner-v1', local=dict(
    backend='torch-callable-resident-v1', share_immutable=False, checkpoint_bindings=[]))
_OFFICIAL_REQUESTS = {
    'sd15': dict(algorithm='distillschedule.training.dmd2:build_dmd2',
        default_backend=_NATIVE_SINGLE, backends={'single-device-v1': _NATIVE_SINGLE,
                  'ddp-fp32-v1': _DMD2_DDP, 'fsdp2-native-sd15-v1': _DMD2_FSDP},
        profiles={'dmd2-sd15-resident-tree-v1': dict(_DMD2_FSDP, allowed_role_backends=[])},
        resident_forbidden_parameters=['native_role_backend', 'optimization_profile', 'generator_checkpoint']),
    'sdxl': dict(algorithm='distillschedule.training.dmd2:build_dmd2',
        default_backend=_NATIVE_SINGLE, backends={'single-device-v1': _NATIVE_SINGLE, 'ddp-fp32-v1': _DMD2_DDP},
        profiles={}, resident_forbidden_parameters=['native_role_backend', 'optimization_profile', 'generator_checkpoint']),
    'self-forcing-wan': dict(algorithm='distillschedule.training.self_forcing:build_self_forcing',
        default_backend=_NATIVE_SINGLE, backends={'fsdp2-native-bf16-v1': _SF_FSDP},
        profiles={'self-forcing-wan-fsdp2-r6-v1': dict(_SF_FSDP, allowed_role_backends=['fsdp2-native-bf16-v1'])},
        resident_forbidden_parameters=['native_role_backend', 'optimization_profile']),
}


def _validate_native_request(contract, model, algorithm, trainer, execution):
    """Reject known official mismatches without importing a numerical provider."""
    if algorithm.get('factory') != contract['algorithm'] or algorithm.get('parameters'):
        raise ValueError('official native algorithm required without algorithm overrides: ' + model['factory'])
    if execution.get('adapter', 'program') != 'program':
        raise ValueError('official native execution requires the Program adapter')
    parameters = model.get('parameters', {})
    if 'optimization_profile' in parameters:
        profile = parameters['optimization_profile']
        if not isinstance(profile, str) or profile not in contract['profiles']:
            raise ValueError('official native optimization profile is not supported by this model')
        selected = contract['profiles'][profile]
        if ('native_role_backend' in parameters
                and parameters['native_role_backend'] not in selected['allowed_role_backends']):
            raise ValueError('official native profile owns its backend; conflicting backend override')
    elif 'native_role_backend' in parameters:
        backend = parameters['native_role_backend']
        if not isinstance(backend, str) or backend not in contract['backends']:
            raise ValueError('official native role backend is not supported by this model')
        selected = contract['backends'][backend]
    else:
        selected = contract['default_backend']
    devices = (trainer.get('device', 'cpu'), parameters.get('device', trainer.get('device', 'cpu')))
    if any(not isinstance(device, str) or re.fullmatch(r'cpu|cuda(?::[0-9]+)?', device) is None
           or device.split(':')[0] not in selected['devices'] for device in devices):
        raise ValueError('official native device is not supported by the selected backend')
    if (devices[0].split(':')[0] != devices[1].split(':')[0]
            or all(':' in device for device in devices) and int(devices[0].split(':')[1]) != int(devices[1].split(':')[1])):
        raise ValueError('official native trainer/model device mismatch')
    if ('distributed' in trainer) != selected['distributed']:
        raise ValueError('official native backend requires matching explicit trainer.distributed ownership')
    for name, expected in selected.get('fixed_parameters', {}).items():
        if name in parameters and not _exact_value(parameters[name], expected):
            raise ValueError('official native backend owns this policy: ' + name)
    mode, implementation = execution.get('mode', 'reference'), execution.get('implementation', {})
    if selected['distributed']:
        distribution = trainer['distributed']
        if (not isinstance(distribution, dict)
                or distribution.get('backend') != {'cpu': 'gloo', 'cuda': 'nccl'}[devices[0].split(':')[0]]):
            raise ValueError('official native distributed device/backend mismatch')
        if set(execution) & set(selected.get('forbidden_fields', [])):
            raise ValueError('official native execution requires original order without a saved plan')
        if (mode not in ('reference', 'manual') or mode == 'reference' and bool(implementation)
                or mode == 'manual' and not _exact_value(implementation, _NATIVE_IMPLEMENTATION)):
            raise ValueError('official native collective execution requires its reference or exact native-owner policy')
    if 'resident_budget' in execution:
        if selected['distributed'] or set(parameters) & set(contract['resident_forbidden_parameters']):
            raise ValueError('official native resident budget does not admit collective or initialization overrides')
    elif devices[0].startswith('cuda') and not selected['distributed']:
        if mode not in ('reference', 'manual') or 'owner_execution' in implementation:
            raise ValueError('official native CUDA trial execution requires qualified resident admission')


for _name, _factory, _caps, _description in (
    ('sd15', 'sd:build_sd15', ('image', 'epsilon', 'gan'), 'Stable Diffusion 1.5'),
    ('sdxl', 'sd:build_sdxl', ('image', 'epsilon', 'lora', 'pooled_text'), 'Stable Diffusion XL'),
    ('self-forcing-wan', 'self_forcing:build_model', ('video', 'causal', 'kv_cache', 'flow'), 'Official Self-Forcing Wan'),
    ('wan13b', 'wan_public:build_model', ('video', 'history', 'lora'), 'Wan1.3B / W1'),
    ('helios-nogan', 'helios_nogan_public:build_model', ('video', 'history', 'lora'), 'Helios No-GAN / W2'),
    ('helios-gan', 'helios_public:build_model', ('video', 'history', 'lora', 'gan'), 'Helios GAN / W3'),
    ('livewan', 'livewan_public:build_model', ('video', 'causal', 'kv_cache'), 'LiveWan cross-scale / W4'),
    ('h3-ar', 'h3_cached_public:build_model', ('video', 'audio', 'history', 'lora'), 'Cached H3 autoregressive / W6'),
):
    register_model(_name, factory='distillschedule.training.' + _factory,
                   capabilities=_caps, description=_description,
                   **({'native_owner_compatibility': _OFFICIAL_REQUESTS[_name]} if _name in _OFFICIAL_REQUESTS else {}))

register_model('ltx25', factory='distillschedule.training.ltx:build_model',
    capabilities=('video', 'audio', 'flow', 'lora'), description='LTX-2.5 audio-video',
    execution_constraints={'cuda': FLOW_CUDA_EXECUTION})

# Source-qualified image families. Validation levels remain distinct: actual
# tiny native CPU fixtures are not pretrained/GPU/quality acceptance.
for _name, _module, _model_id, _revision in (
    ('flux2-klein-base', 'flux', 'black-forest-labs/FLUX.2-klein-base-4B', 'a3b4f4849157f664bdbc776fd7453c2783562f4d'),
    ('qwen-image', 'qwen_image', 'Qwen/Qwen-Image', '75e0b4be04f60ec59a75f475837eced720f823b6'),
):
    register_model(_name, factory='distillschedule.training.' + _module + ':build_model',
        capabilities=('image', 'flow', 'lora', 'dmd_flow_v1'),
        description='Pinned native T2I FP32; explicit post-backward B1 replication; flow DMD, not official DMD2',
        execution_constraints={'cuda': FLOW_CUDA_EXECUTION, 'replicated': FLOW_REPLICATED_EXECUTION},
        source=dict(model_id=_model_id, revision=_revision, license='Apache-2.0'),
        validation=dict(source='AUDITED', cpu='SYNTHETIC_TEST_ONLY', gpu='NOT_RUN', replicated='NOT_RUN', quality='NOT_RUN'),
        compatibility=dict(algorithm_factories=['distillschedule.training.dmd:build_algorithm'],
            devices=['cpu', 'cuda'], distributed=True, distributed_backends={'cpu': 'gloo', 'cuda': 'nccl'},
            model_parameters=['assets', 'device', 'seed', 'trainable', 'student_lr', 'fake_lr',
                'scheduler_gamma', 'ema_decay', 'generation_sigmas']))


register_model('wan22-ti2v5b-t2v', factory='distillschedule.training.wan22:build_model',
    capabilities=('video', 'flow', 'lora', 'dmd_flow_v1'),
    description='Pinned native Wan2.2 TI2V-5B T2V only, FP32 local B1, explicit post-backward replicas; common flow DMD',
    execution_constraints={'cuda': FLOW_CUDA_EXECUTION, 'replicated': FLOW_REPLICATED_EXECUTION},
    source=dict(model_id='Wan-AI/Wan2.2-TI2V-5B-Diffusers',
        revision='b8fff7315c768468a5333511427288870b2e9635', license='Apache-2.0'),
    validation=dict(source='AUDITED', cpu='SYNTHETIC_TEST_ONLY', gpu='NOT_RUN', replicated='NOT_RUN', quality='NOT_RUN'),
    compatibility=dict(algorithm_factories=['distillschedule.training.dmd:build_algorithm'],
        devices=['cpu', 'cuda'], distributed=True, distributed_backends={'cpu': 'gloo', 'cuda': 'nccl'}, precision='FP32', batch_size=1,
        model_parameters=['assets', 'device', 'seed', 'trainable', 'student_lr', 'fake_lr',
            'scheduler_gamma', 'ema_decay', 'generation_sigmas']))


register_model('hunyuan15-t2v', factory='distillschedule.training.hunyuan_video:build_model',
    capabilities=('video', 'flow', 'lora', 'dmd_flow_v1'),
    description='Pinned native HunyuanVideo1.5 T2V dual text/temporal4 VAE, FP32 local B1, explicit post-backward replicas; common flow DMD',
    execution_constraints={'cuda': FLOW_CUDA_EXECUTION, 'replicated': FLOW_REPLICATED_EXECUTION},
    source=dict(model_id='hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v',
        revision='286be7ce72277246578a3e3cc2487e95ddae5bcf', license='Tencent Hunyuan Community License (model assets; not Apache-2.0)'),
    validation=dict(source='AUDITED', cpu='SYNTHETIC_TEST_ONLY', gpu='NOT_RUN', replicated='NOT_RUN', quality='NOT_RUN'),
    compatibility=dict(algorithm_factories=['distillschedule.training.dmd:build_algorithm'],
        devices=['cpu', 'cuda'], distributed=True, distributed_backends={'cpu': 'gloo', 'cuda': 'nccl'}, precision='FP32', batch_size=1,
        model_parameters=['assets', 'device', 'seed', 'trainable', 'student_lr', 'fake_lr',
            'scheduler_gamma', 'ema_decay', 'generation_sigmas']))


def validate_compatibility(model, algorithm, trainer, execution=None):
    """Optional declarative preconstruction contract; unknown plugins stay open.

    Metadata only: never import a model or allocate/load weights to reject a
    known incompatible request. Explicit factories receive the same gate as
    named models when they match a registered declaration.
    """
    entries = [row for row in _registry('model').values() if row['factory'] == model.get('factory')]
    for entry in entries:
        if 'native_owner_compatibility' in entry:
            _validate_native_request(entry['native_owner_compatibility'], model, algorithm, trainer, execution or {})
        replicated = ('distributed' in trainer and 'replicated' in entry.get('execution_constraints', {}))
        # Device-conditional policy admission is metadata-only. In particular,
        # unsupported trial cloning must fail before model/codec ownership, not
        # after a CUDA request has already materialized all native role weights.
        for device in (trainer.get('device', 'cpu'), model.get('parameters', {}).get('device', 'cpu')):
            family = device.split(':', 1)[0] if isinstance(device, str) else None
            policy_kind = 'replicated' if replicated else family
            policy = entry.get('execution_constraints', {}).get(policy_kind)
            if policy is not None:
                settings = execution or {}
                if not execution_policy_matches(settings, policy):
                    raise ValueError(f'{policy_kind.upper()} execution requires {policy["modes"]} '
                                     f'with the declared exact policy and original order: {model["factory"]}')
        contract = entry.get('compatibility')
        if contract is None:
            continue
        if replicated:
            devices = (trainer.get('device', 'cpu'), model.get('parameters', {}).get('device', trainer.get('device', 'cpu')))
            distribution = trainer['distributed']
            if (not isinstance(distribution, dict) or any(not isinstance(device, str) or
                    contract['distributed_backends'].get(device.split(':', 1)[0]) != distribution.get('backend')
                    for device in devices)):
                raise ValueError('replicated device/backend mismatch: ' + model['factory'])
        def admitted_device(value):
            return value in contract['devices'] or (isinstance(value, str) and 'cuda' in contract['devices']
                and value.startswith('cuda:') and value[5:].isascii() and value[5:].isdecimal())
        if (algorithm.get('factory') not in contract['algorithm_factories']
                or not admitted_device(trainer.get('device', 'cpu'))
                or not contract['distributed'] and 'distributed' in trainer
                or set(model.get('parameters', {})) - set(contract['model_parameters'])
                or not admitted_device(model.get('parameters', {}).get('device', 'cpu'))):
            raise ValueError('incompatible model/algorithm/backend or unsupported model capability: ' + model['factory'])


for _name, _factory, _description in (
    ('dmd2', 'dmd2:build_dmd2', 'Official SD DMD2 with native iteration and guidance updates'),
    ('self-forcing', 'self_forcing:build_self_forcing', 'Official causal self-rollout DMD'),
    ('dmd', 'dmd:build_algorithm', 'No-GAN DMD with model-owned denoising and predictions'),
):
    register_algorithm(_name, factory='distillschedule.training.' + _factory, description=_description)
