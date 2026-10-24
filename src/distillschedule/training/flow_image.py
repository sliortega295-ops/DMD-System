"""Shared local asset, payload and lifecycle plumbing, not a model-name trainer.

Family modules own native forward, text/codec ABI and packing. This module never
changes common DMD math, planner choices, or controller scheduling. Production
loaders require every actual config/safetensor file; random initialization is
allowed only in explicitly labeled fixture creators outside this module.
"""
from contextlib import nullcontext
from copy import deepcopy
import importlib.metadata
import inspect
import json
from pathlib import Path

import torch
from torch import nn

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from .adapters import InferenceProjection, ModelComponents
from .low_rank import TrainableLowRankLinear, trainable_scope
from .prediction import PredictionSpec
from .single_device import admit_fp32, construction_rng, place_module, resident_flow_state

ASSET_SCHEMA = 'distillschedule.flow_image_assets.v1'
INPUT_SCHEMA = 'distillschedule.flow_image_inputs.v1'
CUDA_TRIAL_LIMIT = ('CUDA native image trial cloning requires a capacity-bounded snapshot/restore; '
                    'auto/plan/calibration and non-reference execution are not admitted')


def require(condition, message):
    if not condition:
        raise ValueError('native flow image: ' + message)


def json_object(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON field: ' + key)
            result[key] = value
        return result
    try:
        result = json.loads(Path(path).read_text(), object_pairs_hook=unique,
                            parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid local native JSON: ' + str(path)) from exc
    require(isinstance(result, dict), 'explicit JSON object required')
    return result


def cpu1():
    from .distributed import rank_world
    require(rank_world()[1] == 1, 'only CPU1 admitted; distributed execution NOT_RUN')
    require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('cuda'),
            'FP32 only; AMP/autocast is not admitted')


def runtime_identity(api):
    versions = {k: importlib.metadata.version(k) for k in ('torch', 'diffusers', 'transformers', 'peft', 'safetensors')}
    require(versions['torch'].split('+')[0] == '2.10.0' and versions['diffusers'] == '0.37.1'
            and versions['transformers'] == '5.14.1' and versions['peft'] == '0.18.1',
            'use the isolated modern-flow pinned lane, not the Transformers4/PEFT0.17 lane')
    classes = api.classes()
    config_classes = set()
    def configs(cls):
        config_classes.add(cls)
        for child in getattr(cls, 'sub_configs', {}).values():
            configs(child)
    for cls in classes:
        if hasattr(cls, 'config_class'):
            configs(cls.config_class)
    files = {cls.__module__ + '.' + cls.__name__: file_sha256(Path(inspect.getfile(cls))) for cls in (*classes, *config_classes)}
    from transformers import configuration_utils, conversion_mapping, core_model_loading, modeling_rope_utils, modeling_utils
    from . import flow_text_config
    files.update({module.__name__: file_sha256(Path(module.__file__)) for module in
                  (configuration_utils, conversion_mapping, core_model_loading, modeling_rope_utils, modeling_utils, flow_text_config)})
    files.update({name: file_sha256(path) for name,path in {
        'adapter': Path(api.__file__), 'shared_native_plumbing': Path(__file__),
        'feature_encoder': Path(__file__).parent.parent/'data/flow_image_inputs.py',
        'unmerged_lora': Path(__file__).parent/'low_rank.py',
        'device_ownership': Path(__file__).parent/'single_device.py'}.items()})
    import platform
    return dict(python=platform.python_version(), versions=versions, source_files=files)


def asset_files(base, spec):
    require(isinstance(spec, dict) and {'path', 'files', 'max_parameters'} <= set(spec) and set(spec) <= {'path', 'files', 'max_parameters', 'dtype_policy'}
            and isinstance(spec['path'], str) and spec['path'] and type(spec['max_parameters']) is int and spec['max_parameters'] > 0,
            'each local component needs path, exact SHA256 inventory and CPU max_parameters')
    require(spec.get('dtype_policy', 'strict_fp32') in ('strict_fp32', 'bf16_or_fp32_to_fp32'), 'explicit supported storage dtype policy required')
    root = (Path(base) / spec['path']).resolve(strict=True)
    require(root.is_dir() and root.is_relative_to(Path(base).resolve()), 'component path escapes local bundle')
    require(isinstance(spec['files'], dict) and spec['files'], 'empty native component inventory')
    actual = {str(p.relative_to(root)): p for p in root.rglob('*') if p.is_file()}
    require(set(actual) == set(spec['files']), 'local native file inventory changed or missing shards')
    for name, path in actual.items():
        require(not path.is_symlink() and path.resolve().is_relative_to(root)
                and isinstance(spec['files'][name], str) and file_sha256(path) == spec['files'][name],
                'native asset SHA256/path mismatch: ' + name)
    return root, list(actual.values())


def _text_config(config_class, config):
    from .flow_text_config import text_config
    return text_config(config_class, config)


def architecture(cls, config):
    """Use exact native classes, never AutoModel remote code or ignored config keys."""
    if hasattr(cls, 'config_class'):
        require(config.get('model_type') == cls.config_class.model_type, 'wrong actual text model class/config')
        return _text_config(cls.config_class, config)
    fields = set(inspect.signature(cls.__init__).parameters) - {'self'}
    metadata = {'_class_name', '_diffusers_version', '_name_or_path', '_use_default_values'}
    require(config.get('_class_name') == cls.__name__ and set(config) <= fields | metadata,
            'wrong actual native class or unknown constructor/config field')
    return {k: deepcopy(v) for k, v in config.items() if k in fields}


def shell(cls, config):
    native_config = architecture(cls, config)
    # A meta sizing witness only; no randomly initialized model can escape.
    with torch.device('meta'):
        value = cls(native_config) if hasattr(cls, 'config_class') else cls(**native_config)
    return value


def _parameter_alias_groups(model, layout):
    # Only identical actual Parameter slots are a proof of a missing alias.
    # Equal shapes, values, storage pointers (meta!), or native key hints are not.
    slots = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if name in layout:
            slots.setdefault(id(parameter), set()).add(name)
    groups = list(slots.values())
    covered = set().union(*groups) if groups else set()
    return groups + [{key} for key in layout.keys() - covered]


def load_native(cls, root, max_parameters, *, dtype_policy='strict_fp32', config_transform=None):
    import safetensors
    from safetensors.torch import load_file
    require(dtype_policy in ('strict_fp32', 'bf16_or_fp32_to_fp32'), 'unknown storage dtype policy')
    permitted_float = ('F32',) if dtype_policy == 'strict_fp32' else ('F32', 'BF16')
    config = json_object(root / 'config.json')
    # Trusted family-owned normalization of audited historical serialization
    # fields only. Preserve/return raw config bytes and their asset identity;
    # ordinary callers retain the exact fail-closed constructor-field contract.
    effective_config = config if config_transform is None else config_transform(deepcopy(config))
    model = shell(cls, effective_config).float()
    require(0 < sum(p.numel() for p in model.parameters()) <= max_parameters,
            'native model exceeds explicit CPU parameter budget')
    layout = model.state_dict()
    alias_groups = _parameter_alias_groups(model, layout)
    def complete(keys):
        return keys <= layout.keys() and all(keys & group for group in alias_groups)
    rename = lambda name: name
    if hasattr(cls, 'config_class'):
        # Transformers5's actual Qwen2.5-VL save format uses original checkpoint
        # names. Use the pinned native key operations, never a guessed prefix map.
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import WeightRenaming, rename_source_key
        operations = get_model_conversion_mapping(model)
        require(all(isinstance(op, WeightRenaming) for op in operations),
                'non-renaming native text conversions need an independent audit')
        rename = lambda name: rename_source_key(name, operations, [],
            base_model_prefix=model.base_model_prefix, meta_state_dict=layout)[0]
    indices = list(root.glob('*.safetensors.index.json'))
    weights = sorted(root.glob('*.safetensors'))
    require(weights and len(indices) <= 1 and not list(root.glob('*.bin')),
            'real local safetensors required; no pickle/random fallback')
    weight_map = None
    if indices:
        index = json_object(indices[0]); weight_map = index.get('weight_map')
        require(set(index) <= {'metadata', 'weight_map'} and isinstance(weight_map, dict)
                and len({rename(k) for k in weight_map}) == len(weight_map) and complete({rename(k) for k in weight_map})
                and all(isinstance(value, str) for value in weight_map.values())
                and set(weight_map.values()) == {p.name for p in weights}, 'incomplete safe weight shard index')
    else:
        require(len(weights) == 1, 'multiple safe shards need an explicit complete index')
    state = {}
    for path in weights:
        with safetensors.safe_open(path, framework='pt', device='cpu') as handle:
            for raw_key in handle.keys():
                key = rename(raw_key)
                require(key in layout and key not in state and (weight_map is None or weight_map.get(raw_key) == path.name),
                        'unexpected/duplicate/wrong-shard native safe weight: ' + key)
                header = handle.get_slice(raw_key)
                expected_dtype = layout[key].dtype
                allowed_dtype = permitted_float if expected_dtype == torch.float32 else {
                    torch.int64: ('I64',), torch.bool: ('BOOL',)}.get(expected_dtype, ())
                require(tuple(header.get_shape()) == tuple(layout[key].shape)
                        and header.get_dtype() in allowed_dtype,
                        'native safetensor shape/dtype mismatch; FP32-only qualification: ' + key)
                # Record headers before loading any payload from this shard.
                state[key] = None
    require(complete(set(state)) and (weight_map is None or set(state) == {rename(k) for k in weight_map}),
            'incomplete native weights; random fallback forbidden')
    for path in weights:
        payload = {rename(key): tensor for key, tensor in load_file(path, device='cpu').items()}
        for key, tensor in payload.items():
            if tensor.is_floating_point():
                require(tensor.dtype in ((torch.float32,) if dtype_policy == 'strict_fp32' else (torch.float32, torch.bfloat16))
                        and bool(torch.isfinite(tensor).all()), 'nonfinite or incorrectly typed native weight: ' + key)
                # BF16 -> FP32 is exact, explicitly source/manifest-bound storage
                # conversion only. It is NOT BF16 runtime/GPU qualification.
                payload[key] = tensor.float()
            require(payload[key].dtype == layout[key].dtype, 'native weight dtype differs from FP32 compute layout: ' + key)
        state.update(payload)
    for group in alias_groups:
        supplied = sorted(group & state.keys())
        value = state[supplied[0]]
        require(all(torch.equal(value, state[key]) for key in supplied), 'contradictory native tied weight aliases')
        for key in group:
            state[key] = value
    require(set(state) == set(layout) and all(t is not None for t in state.values()),
            'incomplete native weights; random fallback forbidden')
    # Native nonpersistent RoPE caches must be initialized by the actual CPU
    # constructor, not left on meta or approximated/reimplemented here.
    native_config = architecture(cls, effective_config)
    with torch.device('cpu'):
        model = (cls(native_config) if hasattr(cls, 'config_class') else cls(**native_config)).float()
    require({frozenset(group) for group in _parameter_alias_groups(model, model.state_dict())}
            == {frozenset(group) for group in alias_groups}, 'native constructor parameter aliases changed')
    model.load_state_dict(state, strict=True)
    require(all(not p.is_meta for p in (*model.parameters(), *model.buffers())), 'uninitialized meta native state')
    model.eval()
    return model, config


def tokenizer_state(tokenizer):
    options = {key: getattr(tokenizer, key) for key in ('model_max_length', 'padding_side', 'truncation_side',
        'pad_token', 'bos_token', 'eos_token', 'unk_token', 'chat_template')}
    return {'json': tokenizer.backend_tokenizer.to_str(), 'options': options}


def restore_tokenizer(spec):
    from tokenizers import Tokenizer
    from transformers import Qwen2TokenizerFast
    require(isinstance(spec, dict) and set(spec) == {'json', 'options'}, 'actual serialized native tokenizer required')
    return Qwen2TokenizerFast(tokenizer_object=Tokenizer.from_str(spec['json']), **spec['options'])


class ImageMedia(nn.Module):
    def __init__(self, vae, text_encoder, tokenizer, *, config, binding):
        super().__init__()
        self.vae = vae.eval().requires_grad_(False)
        self.text_encoder = text_encoder.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.architecture = deepcopy(config)
        self.binding = binding

    @property
    def device(self):
        return next(self.parameters()).device

    def image(self, pixels):
        require(type(pixels) is torch.Tensor and pixels.dtype == torch.float32 and pixels.device.type == 'cpu'
                and pixels.ndim == 4 and pixels.shape[:2] == (1, 3) and min(pixels.shape) > 0
                and not pixels.requires_grad and bool(torch.isfinite(pixels).all()) and bool((pixels.abs() <= 1).all()),
                'native codec needs explicit normalized [-1,1] FP32 BCHW RGB; no implicit resize')
        return pixels.to(self.device)

    def decode(self, generated, parameters):
        require(isinstance(parameters, dict) and set(parameters) <= {'rollout_sigmas'}, 'unsupported native decoding option')
        with torch.no_grad():
            pixels = self.decode_tokens(generated)
        require(pixels.ndim == 4 and pixels.shape[:2] == (1, 3) and bool(torch.isfinite(pixels).all()),
                'actual native decoder must return finite batch-one RGB')
        return {'image': ((pixels[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)}


class QualifiedNativeFlowTrials:
    """Qualification refusal, not a parameter-count or host-capacity estimator.

    Whole-role paired clones are currently tested only with explicitly labeled
    synthetic CPU fixtures. A manifest label is trusted provenance, not a
    verified model size/origin or a bound on simultaneous clones and CPU RSS.
    Reference/manual persistent execution does not enter this copy boundary.
    """
    def __deepcopy__(self, memo):
        require(getattr(self, '_trial_evidence_kind', None) == 'SYNTHETIC_TEST_ONLY',
                'pretrained or unqualified native flow clone trials are not admitted; '
                'SYNTHETIC_TEST_ONLY CPU fixtures only; selected plan bytes are not a clone/RSS budget')
        cpu1()
        require(all(t.device.type == 'cpu' and (not t.is_floating_point() or t.dtype == torch.float32)
                    for t in (*self.parameters(), *self.buffers())),
                'native flow clone trials require CPU1 FP32 tensors')
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        # Preserve native Module getstate/setstate behavior (including compiled
        # state handling) and the shared Parameter/optimizer graph's copy memo.
        result.__setstate__(deepcopy(self.__getstate__(), memo))
        return result


def qualify_trial_roles(state, evidence_kind):
    for name in ('student', 'fake', 'teacher'):
        if name in state:
            state[name]._trial_evidence_kind = evidence_kind


class ImageRole(QualifiedNativeFlowTrials, nn.Module):
    def __init__(self, native, *, api, config, media_config, binding, empty, trainable, schedule):
        super().__init__()
        self.native = native
        self.api_entry = api.ENTRY  # deepcopy-safe immutable import identity
        self.architecture = deepcopy(config)
        self.media_config = deepcopy(media_config)
        self.binding = binding
        self.trainable_config = deepcopy(trainable)
        self.generation_sigmas = tuple(schedule)
        self.register_buffer('empty_context', empty[0].detach().clone())
        self.register_buffer('empty_context_mask', empty[1].detach().clone())

    @property
    def device(self):
        return next(self.native.parameters()).device

    def __deepcopy__(self, memo):
        require(self.device.type != 'cuda', CUDA_TRIAL_LIMIT)
        return super().__deepcopy__(memo)

    @property
    def api(self):
        import importlib
        return importlib.import_module(self.api_entry)

    def validate_feeds(self, feeds):
        from distillschedule.data.flow_image_inputs import validate_row
        return validate_row(feeds, api=self.api, config=self.architecture, media_config=self.media_config,
                            binding=self.binding, empty=(self.empty_context, self.empty_context_mask), device=self.device)

    def sample_template(self, feeds):
        return feeds['latents']

    def loss_masks(self, feeds, sample):
        return {'image': torch.ones_like(sample['image'], dtype=torch.bool)}

    def apply_conditioning(self, sample, feeds):
        require(isinstance(sample, dict) and set(sample) == {'image'} and sample['image'].shape == feeds['latents']['image'].shape,
                'only the admitted T2I image payload is supported; edit/causal conditioning rejected')
        return sample

    def check_denoise(self, sample, sigma, feeds):
        cpu1()
        self.apply_conditioning(sample, feeds)
        value = sample['image']
        require(type(value) is torch.Tensor and value.dtype == torch.float32 and value.device == self.device
                and bool(torch.isfinite(value).all()), 'native flow sample must be finite FP32 on the owned device')
        require(type(sigma) is torch.Tensor and sigma.dtype == torch.float32 and sigma.device == self.device
                and sigma.shape == (sample['image'].shape[0],) and bool(torch.isfinite(sigma).all())
                and bool(((sigma >= 0) & (sigma <= 1)).all()), 'normalized per-batch sigma must be FP32 in [0,1]')

    def rollout(self, feeds, rollout_sigmas, gradient_policy):
        from .dmd import sigmas
        self.validate_feeds(feeds)
        schedule = sigmas(rollout_sigmas)
        require(gradient_policy in ('all_steps', 'last_step'), 'explicit Euler gradient policy required')
        value = {'image': torch.randn_like(feeds['latents']['image'])}
        for index, (sigma, following) in enumerate(zip(schedule, schedule[1:])):
            context = torch.no_grad() if gradient_policy == 'last_step' and index < len(schedule) - 2 else nullcontext()
            with context:
                prediction = self.denoise(value, value['image'].new_full((value['image'].shape[0],), sigma), feeds, conditional=True)
                value = {'image': value['image'] + (following - sigma) * prediction['image']}
        return value


def _metadata(api, descriptor):
    return dict(schema_version='distillschedule.native_flow_image.v1', family=api.FAMILY,
        source=dict(model_id=api.MODEL_ID, revision=api.REVISION, license='Apache-2.0'),
        **deepcopy(descriptor), precision='FP32_NATIVE_SINGLE_DEVICE', gpu_execution='NOT_RUN', quality_validation='NOT_RUN',
        prediction='v=epsilon-x0; z=(1-sigma)*x0+sigma*epsilon; x0=z-sigma*v; NO_NEGATION',
        algorithm='EXPLICIT_FLOW_DMD_NOT_OFFICIAL_DMD2_OR_SELF_FORCING',
        generation='CURRENT_STUDENT_NOT_BASE_TEACHER_OR_EMA')


def _components(api, state, descriptor, *, training):
    from distillschedule.data.flow_image_inputs import encode_export_input
    qualify_trial_roles(state, descriptor['evidence_kind'])
    immutable = ['teacher', 'media'] if training else ['media']
    if training:
        # Reuse the existing exact immutable alias mechanism, including actual
        # frozen native Linear bases in LoRA roles; never merge their weights.
        for role in ('student', 'fake'):
            for name, module in state[role].named_modules():
                if isinstance(module, TrainableLowRankLinear):
                    immutable.append(role + '.' + name + '.base')
    return ModelComponents(state=state, immutable_modules=tuple(immutable), metadata=_metadata(api, descriptor),
        prediction=PredictionSpec('flow'), capabilities=('image', 'flow', 'lora', 'dmd_flow_v1'),
        generate=api.generate, decode=state['media'].decode,
        encode_input=None if training else encode_export_input,
        inference_projection=api.project_current_generator if training else None,
        admit_training=api.admit_training if training else None).validate()


def load_media_bundle(assets, api):
    from transformers import Qwen2TokenizerFast
    cpu1()
    runtime = runtime_identity(api)
    path = Path(assets).expanduser().resolve(strict=True)
    manifest = json_object(path)
    require(set(manifest) == {'schema_version', 'family', 'source', 'evidence_kind', 'roles', 'media', 'text_max_length'}
            and manifest['schema_version'] == ASSET_SCHEMA and manifest['family'] == api.FAMILY
            and manifest['source'] == {'model_id': api.MODEL_ID, 'revision': api.REVISION}
            and manifest['evidence_kind'] in ('SYNTHETIC_TEST_ONLY', 'LOCAL_PRETRAINED_UNVERIFIED'),
            'exact family/version/source asset contract required; family variants cannot be substituted')
    require(isinstance(manifest['roles'], dict) and isinstance(manifest['media'], dict)
            and set(manifest['roles']) == {'student', 'fake', 'teacher'} and set(manifest['media']) == {'vae', 'text_encoder', 'tokenizer'},
            'all actual role, VAE, text encoder and tokenizer components required')
    native_cls, vae_cls, text_cls, _ = api.classes()
    documents = [path]; roots = {}
    for name, spec in {**manifest['roles'], **manifest['media']}.items():
        roots[name], files = asset_files(path.parent, spec); documents.extend(files)
    # Config/capability admission precedes loading any role, text or VAE weights.
    preview = dict(vae=json_object(roots['vae']/'config.json'),
        text_encoder=json_object(roots['text_encoder']/'config.json'), text_max_length=manifest['text_max_length'])
    architecture(vae_cls, preview['vae']); architecture(text_cls, preview['text_encoder'])
    role_configs = [json_object(roots[name]/'config.json') for name in ('student', 'fake', 'teacher')]
    require(all(config == role_configs[0] for config in role_configs), 'same-family role architectures must match before loading')
    for config in role_configs:
        api.validate_architecture(config, preview)
    # Refuse every role/media parameter budget before any concrete constructor
    # or safetensor payload load. This is meta parameter sizing, NOT host RSS or
    # a bound on checkpoint staging or paired calibration clones.
    transform = getattr(api, 'transformer_config', None)
    for name, cls in [('student', native_cls), ('fake', native_cls), ('teacher', native_cls),
                      ('vae', vae_cls), ('text_encoder', text_cls)]:
        config = json_object(roots[name]/'config.json')
        if name in manifest['roles'] and transform is not None:
            config = transform(deepcopy(config))
        witness = shell(cls, config)
        count = sum(p.numel() for p in witness.parameters())
        spec = (manifest['roles'] if name in manifest['roles'] else manifest['media'])[name]
        require(0 < count <= spec['max_parameters'], 'native model exceeds explicit CPU parameter budget: ' + name)
        del witness
    binding = fingerprint(dict(family=api.FAMILY, source=manifest['source'], media=manifest['media'],
        text_max_length=manifest['text_max_length'], runtime=runtime, dtype='FP32', codec_policy=api.CODEC_POLICY,
        transformer_config_sha256=file_sha256(roots['student']/'config.json')))
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        vae, vae_config = load_native(vae_cls, roots['vae'], manifest['media']['vae']['max_parameters'],
            dtype_policy=manifest['media']['vae'].get('dtype_policy', 'strict_fp32'))
        text, text_config = load_native(text_cls, roots['text_encoder'], manifest['media']['text_encoder']['max_parameters'],
            dtype_policy=manifest['media']['text_encoder'].get('dtype_policy', 'strict_fp32'))
        tokenizer = Qwen2TokenizerFast.from_pretrained(roots['tokenizer'], local_files_only=True)
        media_config = dict(vae=vae_config, text_encoder=text_config, tokenizer=tokenizer_state(tokenizer),
                            text_max_length=manifest['text_max_length'])
        media = api.Media(vae, text, tokenizer, config=media_config, binding=binding).eval().requires_grad_(False)
        with torch.no_grad():
            empty = media.encode_text([''])
        media_config['empty_context_shape'] = list(empty[0].shape)
        media.architecture = deepcopy(media_config)
    return media, manifest, roots, runtime, binding, media_config, empty, documents


def build_family(parameters, api):
    from .dmd import sigmas, training_state
    allowed = {'assets', 'device', 'seed', 'trainable', 'student_lr', 'fake_lr', 'scheduler_gamma', 'ema_decay', 'generation_sigmas'}
    require(isinstance(parameters, dict) and set(parameters) <= allowed
            and allowed - {'device', 'seed'} <= parameters.keys(), 'complete explicit native image model parameters required; unknown options rejected')
    device = admit_fp32(parameters, require=require)
    require(type(parameters.get('seed', 0)) is int, 'integer construction seed required')
    require(isinstance(parameters['trainable'], dict) and parameters['trainable'].get('mode') == 'lora', 'initial admitted image recipe is same-family unmerged LoRA only')
    schedule = sigmas(parameters['generation_sigmas'])
    media, manifest, roots, runtime, binding, media_config, empty, documents = load_media_bundle(parameters['assets'], api)
    native_cls = api.classes()[0]
    with construction_rng(device, parameters.get('seed', 0)), torch.device('cpu'):
        # Offline preparation owns the exact CPU null-prompt encoding. Move it,
        # rather than substituting nearly-equal CUDA text arithmetic on resume.
        empty = tuple(tensor.to(device) for tensor in empty)
        place_module(media, device, require=require)
        roles = {}; config = None
        for role in ('student', 'fake', 'teacher'):
            native, current = load_native(native_cls, roots[role], manifest['roles'][role]['max_parameters'],
                dtype_policy=manifest['roles'][role].get('dtype_policy', 'strict_fp32'),
                config_transform=getattr(api, 'transformer_config', None))
            api.validate_architecture(current, media_config)
            config = current if config is None else config
            require(current == config, 'first recipe requires exact same-family role architectures')
            if role == 'teacher':
                native.eval().requires_grad_(False)
            else:
                trainable_scope(native, parameters['trainable'])
                native.train()
            place_module(native, device, require=require)
            roles[role] = api.Role(native, api=api, config=config, media_config=media_config,
                binding=binding, empty=empty, trainable=parameters['trainable'], schedule=schedule)
            if role == 'teacher':
                roles[role].eval()
        with torch.device(device):
            state = training_state(roles['student'], roles['fake'], roles['teacher'], student_lr=parameters['student_lr'],
                fake_lr=parameters['fake_lr'], scheduler_gamma=parameters['scheduler_gamma'], ema_decay=parameters['ema_decay'])
        resident_flow_state(state, device)
    state['media'] = media
    descriptor = dict(architecture=config, media_config=media_config, binding=binding, runtime=runtime,
        trainable=deepcopy(parameters['trainable']), generation_sigmas=list(schedule), evidence_kind=manifest['evidence_kind'],
        asset_identity=dict(manifest_sha256=file_sha256(Path(parameters['assets'])),roles=deepcopy(manifest['roles']),media=deepcopy(manifest['media'])))
    # Recheck all immutable files after materialization; no silent cache reuse.
    for name, spec in {**manifest['roles'], **manifest['media']}.items():
        asset_files(Path(parameters['assets']).resolve().parent, spec)
    require(json_object(parameters['assets']) == manifest, 'asset manifest changed during load')
    return _components(api, state, descriptor, training=True)


def admit_training(api, *, state, request, prepared_identity, prepared_manifest):
    device = admit_fp32({'device': request.trainer.get('device', 'cpu')}, require=require)
    require('distributed' not in request.trainer
            and all(state[name].device == device for name in ('student', 'fake', 'teacher', 'media')),
            'matching single-device FP32 ownership required; distributed execution NOT_RUN')
    require(device.type != 'cuda' or (request.execution.get('mode', 'reference') == 'reference'
            and not set(request.execution) & {'implementation', 'order', 'plan'}), CUDA_TRIAL_LIMIT)
    require(request.algorithm['factory'] == 'distillschedule.training.dmd:build_algorithm'
            and request.execution.get('adapter', 'program') == 'program', 'use existing common flow DMD Program, not DMD2/Self-Forcing')
    require('streams' not in request.data and prepared_manifest.get('metadata', {}).get('schema_version') == INPUT_SCHEMA
            and prepared_manifest['metadata'].get('family') == api.FAMILY
            and prepared_manifest['metadata'].get('binding') == state['student'].binding,
            'wrong family/source/codec feature manifest; same shape is insufficient')
    p = request.algorithm.get('parameters', {})
    require(p.get('teacher_cfg') == 1 and p.get('fake_cfg') == 1,
            'first image recipe admits teacher_cfg=fake_cfg=1; native Qwen norm-rescaled CFG is not linear CFG')
    require(p.get('fake_updates') == 5 and p.get('rollout_sigmas') == list(state['student'].generation_sigmas)
            and len(state['student'].generation_sigmas) == 5, 'initial recipe requires G1/F5 and four explicit Euler steps')


def generate(api, state, feeds, parameters):
    from .dmd import sigmas
    require(isinstance(parameters, dict) and set(parameters) <= {'rollout_sigmas'}, 'unsupported native image inference option')
    role = state['student']; role.validate_feeds(feeds)
    schedule = sigmas(parameters.get('rollout_sigmas', role.generation_sigmas))
    tokens = role.rollout(feeds, schedule, 'all_steps')['image']
    latent = api.unpack(tokens, feeds['latent_shape'].tolist())
    return {'image_latents': latent, 'latent_schema': {'image_latents': {
        'layout': 'BCHW', 'shape': list(latent.shape), 'dtype': str(latent.dtype)}}}


def project_current_generator(api, components, effective_training_parameters):
    descriptor = {key: deepcopy(components.metadata[key]) for key in ('architecture', 'media_config', 'binding',
        'runtime', 'trainable', 'generation_sigmas', 'evidence_kind')}
    state = {name: components.state[name] for name in ('student', 'media')}
    return InferenceProjection(components=_components(api, state, descriptor, training=False),
        factory=api.ENTRY + ':build_inference_model', parameters=descriptor)


def build_inference_family(parameters, api):
    from .dmd import sigmas
    keys = {'architecture', 'media_config', 'binding', 'runtime', 'trainable', 'generation_sigmas', 'evidence_kind'}
    require(isinstance(parameters, dict) and keys <= parameters.keys() and set(parameters) <= keys | {'device', 'seed'},
            'export reconstructs only CURRENT Student and codecs, not training/base paths')
    device = admit_fp32(parameters, require=require)
    descriptor = {k: deepcopy(parameters[k]) for k in keys}
    require(descriptor['runtime'] == runtime_identity(api), 'export native source/runtime changed')
    config = descriptor['architecture']; media_config = descriptor['media_config']; api.validate_architecture(config, media_config)
    native_cls, vae_cls, text_cls, _ = api.classes()
    def layout(cls, cfg):
        # Empty concrete layouts only. Common restore_modules strictly replaces
        # EVERY named tensor before inference. This is never an asset fallback.
        native_config = architecture(cls, cfg)
        with torch.device('cpu'):
            return (cls(native_config) if hasattr(cls, 'config_class') else cls(**native_config)).float()
    with construction_rng(device, parameters.get('seed', 0)), torch.device('cpu'):
        transform = getattr(api, 'transformer_config', None)
        native = layout(native_cls, config if transform is None else transform(config))
        trainable_scope(native, descriptor['trainable'])
        place_module(native, device, require=require)
        media = api.Media(layout(vae_cls, media_config['vae']), layout(text_cls, media_config['text_encoder']),
            restore_tokenizer(media_config['tokenizer']), config=media_config, binding=descriptor['binding'])
        place_module(media, device, require=require)
        empty_shape = api.empty_shape(media_config)
        empty = (torch.empty(empty_shape, dtype=torch.float32, device=device),
                 torch.empty(empty_shape[:2], dtype=torch.bool, device=device))
        role = api.Role(native, api=api, config=config, media_config=media_config, binding=descriptor['binding'], empty=empty,
            trainable=descriptor['trainable'], schedule=sigmas(descriptor['generation_sigmas']))
    return _components(api, {'student': role, 'media': media}, descriptor, training=False)
