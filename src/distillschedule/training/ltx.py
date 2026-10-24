"""Actual pinned LTX-2.5 AV models through the common training lifecycle.

Explicit CPU/native-local-state-dict route. No LightX2V, architecture substitute,
22B guessed defaults, online loader, new trainer or model-name solver branch.
All media/text codecs are actual upstream classes; portable exports retain the
CURRENT Student and those owned codecs, not a reloaded base/Teacher/EMA.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path

import torch
from torch import nn

from .adapters import InferenceProjection, ModelComponents
from .flow_image import QualifiedNativeFlowTrials, qualify_trial_roles
from .ltx_source import original, require
from .prediction import PredictionSpec


# Backwards-compatible public aliases; the arithmetic and ownership are unchanged.
from .low_rank import TrainableLowRankLinear, trainable_scope


def _validate_architecture(architecture):
    """One bounded contract for local/original roles and current-G reconstruction."""
    require(isinstance(architecture, dict), 'explicit actual LTXModel constructor config required')
    essential = {'model_type', 'num_attention_heads', 'attention_head_dim', 'audio_num_attention_heads',
        'audio_attention_head_dim', 'in_channels', 'out_channels', 'audio_in_channels', 'audio_out_channels',
        'cross_attention_dim', 'audio_cross_attention_dim', 'num_layers', 'cross_attention_adaln',
        'use_prompt_adaln_single', 'ff_bias', 'audio_ff_bias', 'use_middle_indices_grid'}
    require(essential <= architecture.keys() and not {'attention_ops', 'caption_projection', 'audio_caption_projection'} & architecture.keys(),
            'complete native AV architecture required; callable injections/guessed 22B defaults refused')
    require(architecture['model_type'] == 'ltx av model' and architecture['use_middle_indices_grid'] is True
            and all(architecture[k] == 128 for k in ('in_channels', 'out_channels', 'audio_in_channels', 'audio_out_channels')),
            'this admitted route requires actual AV width128 tokens and native bound coordinates')
    require(all(type(architecture[k]) is int and architecture[k] > 0 for k in (
                'num_layers', 'num_attention_heads', 'attention_head_dim',
                'audio_num_attention_heads', 'audio_attention_head_dim'))
            and architecture['num_layers'] <= 16
            and architecture['num_attention_heads'] * architecture['attention_head_dim'] <= 1024
            and architecture['audio_num_attention_heads'] * architecture['audio_attention_head_dim'] <= 1024,
            'bounded explicit CPU native architecture required; full pretrained 22B route is NOT_RUN')
    return essential


def native_transformer(source, architecture):
    from torch.nn.attention import SDPBackend
    from .ltx_media import constructor
    essential = _validate_architecture(architecture)
    api = source.import_module('ltx_core.model.transformer.model')
    attention = source.import_module('ltx_core.model.transformer.attention')
    config = deepcopy(architecture)
    config['model_type'] = api.LTXModelType(config['model_type'])
    if 'rope_type' in config:
        config['rope_type'] = source.import_module('ltx_core.model.transformer.rope').LTXRopeType(config['rope_type'])
    config['attention_ops'] = attention.AttentionOps(
        attention_function=attention.PytorchAttention(priority=[SDPBackend.MATH]),
        masked_attention_function=attention.PytorchAttention(priority=[SDPBackend.MATH]))
    return constructor(api.LTXModel, config, essential).float()


def _original_architecture(metadata):
    """Mirror the pinned configurator's explicit fields, not invented 22B defaults."""
    require(isinstance(metadata, dict) and isinstance(metadata.get('config'), dict)
            and isinstance(metadata['config'].get('transformer'), dict), 'original checkpoint config.transformer metadata required')
    config = metadata['config']['transformer']
    direct = {'num_attention_heads', 'attention_head_dim', 'in_channels', 'out_channels', 'num_layers',
        'cross_attention_dim', 'norm_eps', 'positional_embedding_theta', 'positional_embedding_max_pos',
        'timestep_scale_multiplier', 'use_middle_indices_grid', 'audio_num_attention_heads', 'audio_attention_head_dim',
        'audio_in_channels', 'audio_out_channels', 'audio_cross_attention_dim', 'audio_positional_embedding_max_pos',
        'av_ca_timestep_scale_multiplier', 'rope_type', 'apply_gated_attention', 'cross_attention_adaln',
        'use_prompt_adaln_single', 'ff_bias', 'audio_ff_bias', 'use_keyframes_abs_pos_embedding'}
    require(direct | {'caption_proj_before_connector', 'frequencies_precision'} <= config.keys(),
            'complete original transformer geometry/behavior required; missing fields never use production defaults')
    require(config['caption_proj_before_connector'] is True and config['ff_bias'] is False
            and config['audio_ff_bias'] is False, 'this original-format route is native 2.5 preprojected/Gemma4 AV, not 19B')
    require(config['frequencies_precision'] in ('float32', 'float64'), 'explicit original RoPE precision required')
    version = str(metadata.get('model_version', ''))
    require(version == '2.5' or version.startswith('2.5.'), 'explicit original LTX-2.5 model_version required')
    architecture = {key: deepcopy(config[key]) for key in direct}
    architecture.update(model_type='ltx av model', double_precision_rope=config['frequencies_precision'] == 'float64')
    require(all(type(config[k]) is int and config[k] > 0 for k in ('num_attention_heads', 'attention_head_dim',
        'audio_num_attention_heads', 'audio_attention_head_dim', 'num_layers', 'cross_attention_dim', 'audio_cross_attention_dim'))
        and all(config[k] == 128 for k in ('in_channels', 'out_channels', 'audio_in_channels', 'audio_out_channels'))
        and config['use_middle_indices_grid'] is True, 'original native AV shape contract is incompatible')
    _validate_architecture(architecture)
    return architecture


def load_original_transformer(source, checkpoint, expected_sha256, *, max_parameters, dtype_policy):
    """Strict ORIGINAL LTX safetensors import via all four actual native owners.

    Metadata/configurator and model.diffusion_model key ops come from upstream.
    We add fail-closed admission because upstream assignment is intentionally
    strict=False and may return an incompletely initialized meta shell. A caller
    must explicitly budget CPU parameters and any native FP16/BF16->FP32 cast.
    This imports transformer weights, not a claim that every monolithic VAE/text
    variant has an integrated CPU pipeline or has pretrained quality validation.
    """
    from .ltx_media import file_sha256
    from .checkpoint import _validate_layout
    from distillschedule.workloads.best_known.registry import fingerprint
    from torch.nn.attention import SDPBackend
    import safetensors
    path = Path(checkpoint).expanduser().resolve(strict=True)
    require(path.is_file() and path.suffix == '.safetensors', 'explicit local original safetensors checkpoint required')
    require(type(max_parameters) is int and max_parameters > 0, 'explicit positive CPU max_parameters required')
    require(dtype_policy in ('strict_fp32', 'cast_native_float_to_fp32'), 'explicit original checkpoint dtype policy required')
    require(file_sha256(path) == expected_sha256, 'original checkpoint SHA256 mismatch')
    source.validate()
    loader_api = source.import_module('ltx_core.loader.sft_loader')
    configurator = source.import_module('ltx_core.model.transformer.model_configurator')
    builder_api = source.import_module('ltx_core.loader.single_gpu_model_builder')
    registry_api = source.import_module('ltx_core.loader.registry')
    attention_api = source.import_module('ltx_core.model.transformer.attention')
    loader = loader_api.SafetensorsModelStateDictLoader()
    metadata = loader.metadata(str(path))  # ORIGINAL JSON metadata parsing.
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('LTX-2.5: nonfinite/non-JSON original metadata') from exc
    architecture = _original_architecture(metadata)
    operations = configurator.LTXV_MODEL_COMFY_RENAMING_MAP
    with safetensors.safe_open(path, framework='pt', device='cpu') as handle:
        raw_keys = list(handle.keys())
        selected = [operations.apply_to_key(name) for name in raw_keys if operations.apply_to_key(name) is not None]
        raw_metadata = handle.metadata() or {}
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate original metadata JSON key: ' + key)
            result[key] = value
        return result
    require(isinstance(raw_metadata.get('config'), str), 'original config must be JSON metadata, not an external sidecar')
    require(json.loads(raw_metadata['config'], object_pairs_hook=unique) == metadata['config'],
            'original metadata parser/config identity disagreement')
    require(selected and len(selected) == len(set(selected)), 'original checkpoint has no model weights or colliding selected keys')
    builder = builder_api.SingleGPUModelBuilder(model_class_configurator=configurator.LTXModelConfigurator,
        model_path=str(path), model_sd_ops=operations, model_loader=loader, loras=(),
        registry=registry_api.ModelRegistry(cache_models=False, cache_weights=False))
    # The native internal meta shell is only a sizing/layout witness; it is
    # NEVER returned as a constructed model or accepted as usable weights.
    shell = builder.meta_model(metadata, ())
    count = sum(p.numel() for p in shell.parameters())
    require(0 < count <= max_parameters, 'original native parameter count exceeds explicit CPU budget')
    expected_layout = shell.state_dict()
    require(set(selected) == set(expected_layout), 'original checkpoint selected keys are incomplete or unexpected')
    permitted_header = {'F32'} if dtype_policy == 'strict_fp32' else {'F32', 'F16', 'BF16'}
    # Check selected headers BEFORE materializing weights, so a bogus shape
    # cannot allocate model-sized data before the strict layout rejection.
    with safetensors.safe_open(path, framework='pt', device='cpu') as handle:
        for raw_name in raw_keys:
            name = operations.apply_to_key(raw_name)
            if name is not None:
                header = handle.get_slice(raw_name)
                require(tuple(header.get_shape()) == tuple(expected_layout[name].shape)
                        and header.get_dtype() in permitted_header, 'original selected weight header shape/dtype mismatch: ' + name)
    state = loader.load(str(path), sd_ops=operations, device=torch.device('cpu')).sd
    require(state and set(state) == set(shell.state_dict()) and len(selected) == len(state),
            'original checkpoint selected keys are incomplete or unexpected')
    allowed = (torch.float32,) if dtype_policy == 'strict_fp32' else (torch.float32, torch.float16, torch.bfloat16)
    require(all(type(t) is torch.Tensor and not t.is_meta and t.numel() > 0 and t.dtype in allowed
                and t.layout == torch.strided and bool(torch.isfinite(t).all()) for t in state.values()),
            'original native weights must be finite nonempty non-meta tensors with explicit permitted precision')
    expected = {name: tensor.float() for name, tensor in state.items()}
    _validate_layout(shell.state_dict(), expected, 'original LTX checkpoint')
    native = builder.build(device=torch.device('cpu'), dtype=torch.float32 if dtype_policy != 'strict_fp32' else None)
    live = native.state_dict()
    require(all(not t.is_meta and t.numel() > 0 and bool(torch.isfinite(t).all()) for t in
                (*native.parameters(), *native.buffers())), 'original native builder returned a meta/uninitialized/nonfinite shell')
    _validate_layout(shell.state_dict(), live, 'original native built model')
    require(all(torch.equal(live[name], expected[name]) for name in expected), 'original builder did not load every exact selected weight')
    for module in native.modules():
        if isinstance(module, attention_api.Attention):
            module.attention_function = attention_api.PytorchAttention(priority=[SDPBackend.MATH])
            module.masked_attention_function = attention_api.PytorchAttention(priority=[SDPBackend.MATH])
    require(file_sha256(path) == expected_sha256, 'original checkpoint changed during native load')
    source.validate()
    receipt = dict(format='ORIGINAL_LTX_2_5_SAFETENSORS', checkpoint_sha256=expected_sha256,
        metadata_sha256=fingerprint(metadata), model_version=metadata['model_version'],
        parameter_count=count, strict_selected_key_count=len(state), unrelated_key_count=len(raw_keys) - len(selected),
        dtype_policy=dtype_policy, cpu_max_parameters=max_parameters, loras=[],
        loader=type(loader).__module__ + '.' + type(loader).__name__,
        configurator=configurator.LTXModelConfigurator.__module__ + '.LTXModelConfigurator',
        builder=type(builder).__module__ + '.' + type(builder).__name__, key_ops=operations.name,
        source='PINNED_NATIVE_CONFIGURATOR_LOADER_KEY_OPS_AND_BUILDER', quality_validation='NOT_RUN', gpu_execution='NOT_RUN')
    return native, architecture, receipt


class LTXRole(QualifiedNativeFlowTrials, nn.Module):
    """Model-owned native ABI, conditioning and differentiable Euler rollout."""
    def __init__(self, native, source, *, architecture, media_architecture, binding, empty_context,
                 generation_sigmas, trainable):
        super().__init__()
        self.native = native
        self.source = source
        self.architecture = deepcopy(architecture)
        self.media_architecture = deepcopy(media_architecture)
        self.binding = binding
        self.generation_sigmas = tuple(generation_sigmas)
        self.trainable_config = deepcopy(trainable)
        self.register_buffer('empty_video_context', empty_context['video'].detach().clone())
        self.register_buffer('empty_audio_context', empty_context['audio'].detach().clone())
        self.register_buffer('empty_context_mask', empty_context['mask'].detach().clone().bool())

    def validate_feeds(self, feeds):
        from distillschedule.data.ltx_inputs import validate_row
        self.source.validate()
        return validate_row(feeds, source=self.source, media_architecture=self.media_architecture, binding=self.binding,
            context_dimensions={'video': self.architecture['cross_attention_dim'],
                                'audio': self.architecture['audio_cross_attention_dim']},
            null_context={'video': self.empty_video_context, 'audio': self.empty_audio_context, 'mask': self.empty_context_mask})

    def sample_template(self, feeds):
        return feeds['latents']

    def loss_masks(self, feeds, sample):
        return {name: (~feeds['conditioning_mask'][name]).unsqueeze(-1).expand_as(value)
                for name, value in sample.items()}

    def apply_conditioning(self, sample, feeds):
        return {name: torch.where(feeds['conditioning_mask'][name].unsqueeze(-1), feeds['conditioning'][name], value)
                for name, value in sample.items()}

    def denoise(self, sample, sigma, feeds, *, conditional):
        api = self.source.import_module('ltx_core.model.transformer.modality')
        require(set(sample) == {'video', 'audio'} and type(sigma) is torch.Tensor and sigma.dtype == torch.float32
                and sigma.shape == (sample['video'].shape[0],) and bool(torch.isfinite(sigma).all())
                and bool(((sigma >= 0) & (sigma <= 1)).all()), 'actual AV flow denoising requires finite per-batch sigma')
        sample = self.apply_conditioning(sample, feeds)
        contexts = feeds['contexts'] if conditional else feeds['null_contexts']
        mask = feeds['context_mask'] if conditional else feeds['null_context_mask']
        # Upstream's binary branch subtracts 1 and does not accept Bool - Int.
        # Preserve bool 1=valid semantics by an explicit integer conversion;
        # native additive B11L 0/-finfo.max floats pass through unchanged.
        native_mask = mask.to(torch.int64) if mask.dtype == torch.bool else mask
        modalities = {}
        for name in ('video', 'audio'):
            times = sigma.unsqueeze(1).expand(sample[name].shape[:2]).clone()
            times.masked_fill_(feeds['conditioning_mask'][name], 0.)
            keyframes = None
            if name == 'video' and self.architecture.get('use_keyframes_abs_pos_embedding', False):
                keyframes = torch.zeros_like(times).unsqueeze(-1)
                shape = feeds['video_shape'].tolist()
                keyframes[:, :shape[3] * shape[4]] = 1
            modalities[name] = api.Modality(latent=sample[name], sigma=sigma,
                timesteps=api.TokenTimesteps(times), positions=feeds['positions'][name],
                context=contexts[name], context_mask=native_mask, keyframes_mask=keyframes)
        video, audio = self.native(video=modalities['video'], audio=modalities['audio'], perturbations=None)
        require(type(video) is torch.Tensor and type(audio) is torch.Tensor
                and video.shape == sample['video'].shape and audio.shape == sample['audio'].shape,
                'native model must return both unmodified AV token layouts')
        return {'video': video, 'audio': audio}

    def rollout(self, feeds, rollout_sigmas, gradient_policy):
        from .dmd import sigmas
        self.validate_feeds(feeds)
        schedule = sigmas(rollout_sigmas)
        require(gradient_policy in ('all_steps', 'last_step'), 'explicit rollout gradient policy required')
        value = self.apply_conditioning({name: torch.randn_like(latent) for name, latent in feeds['latents'].items()}, feeds)
        for index, (sigma, following) in enumerate(zip(schedule, schedule[1:])):
            context = torch.no_grad() if gradient_policy == 'last_step' and index < len(schedule) - 2 else nullcontext()
            with context:
                prediction = self.denoise(value, next(iter(value.values())).new_full((value['video'].shape[0],), sigma),
                                          feeds, conditional=True)
                value = self.apply_conditioning({k: x + (following - sigma) * prediction[k] for k, x in value.items()}, feeds)
        return value


def _runtime():
    return {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'torchaudio')}


def _metadata(api, *, architecture, media_architecture, binding, trainable, generation_sigmas):
    return dict(schema_version='distillschedule.native_ltx25.v1', upstream=api.identity,
        architecture=deepcopy(architecture), media_architecture=deepcopy(media_architecture),
        media_binding=binding, trainable=deepcopy(trainable), generation_sigmas=list(generation_sigmas),
        precision='FP32_CPU_MATH_SDPA', runtime=_runtime(), quality_validation='NOT_RUN', gpu_execution='NOT_RUN',
        pretrained_quality='UNVERIFIED_LOCAL_WEIGHTS_OR_RANDOM_NATIVE_FIXTURE',
        generation='CURRENT_STUDENT_NOT_BASE_TEACHER_OR_EMA',
        scope='CPU1_NATIVE_LOCAL_STATE_DICTS; NO_MONOLITHIC_22B_CONVERSION_OR_GPU_CLAIM')


def build_model(parameters):
    from .dmd import sigmas, training_state
    from .ltx_media import load_media, media_binding, strict_load, asset_file
    allowed = {'source_root', 'assets', 'device', 'seed', 'trainable', 'student_lr', 'fake_lr',
               'scheduler_gamma', 'ema_decay', 'generation_sigmas'}
    require(isinstance(parameters, dict) and set(parameters) <= allowed
            and allowed - {'seed', 'device'} <= parameters.keys(), 'complete explicit native LTX model parameters required')
    require(parameters.get('device', 'cpu') == 'cpu', 'this native implementation is explicitly CPU-only; GPU validation NOT_RUN')
    require(type(parameters.get('seed', 0)) is int, 'explicit integer initialization seed required')
    schedule = sigmas(parameters['generation_sigmas'])
    api = original(parameters['source_root'])
    asset_path = Path(parameters['assets']).expanduser().resolve(strict=True)
    # Factory initialization is seed-isolated; runtime noise belongs to the
    # existing Program RNG and common checkpoint, not an untracked model RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(parameters.get('seed', 0))
        media, manifest, identity = load_media(api.source, asset_path)
        require(set(manifest) == {'schema_version', 'evidence_kind', 'roles', 'media'}
                and isinstance(manifest['roles'], dict) and set(manifest['roles']) == {'student', 'fake', 'teacher'},
                'native local bundle must declare actual Student/Fake/Teacher and media')
        binding = media_binding(manifest, api.source)
        empty = media.encode_text([''])
        empty_context = {'video': empty['video_context'], 'audio': empty['audio_context'], 'mask': empty['context_mask']}
        roles = {}; role_receipts = {}; architecture = None
        for name, spec in manifest['roles'].items():
            require(isinstance(spec, dict), 'native role must be an explicit asset object')
            if spec.get('format') == 'ltx_original_safetensors':
                require(set(spec) == {'format', 'checkpoint', 'cpu_max_parameters', 'dtype_policy'},
                        'original native role requires checkpoint/hash, explicit CPU budget and precision policy')
                checkpoint = asset_file(asset_path.parent, spec['checkpoint'])
                native, role_architecture, receipt = load_original_transformer(api.source, checkpoint,
                    spec['checkpoint']['sha256'], max_parameters=spec['cpu_max_parameters'], dtype_policy=spec['dtype_policy'])
                role_receipts[name] = receipt
            else:
                require(set(spec) == {'config', 'weights'}, 'local native role requires exact config/weights')
                checkpoint = asset_file(asset_path.parent, spec['weights'])
                role_architecture = spec['config']
                native = native_transformer(api.source, role_architecture)
                strict_load(native, checkpoint, spec['weights']['sha256'])
                role_receipts[name] = dict(format='EXPLICIT_NATIVE_STATE_DICT', checkpoint_sha256=spec['weights']['sha256'])
            architecture = role_architecture if architecture is None else architecture
            require(role_architecture == architecture, 'this first recipe requires same-family G/F/Teacher native architecture')
            if name == 'teacher':
                native.eval().requires_grad_(False)
            else:
                trainable_scope(native, parameters['trainable'])
            roles[name] = LTXRole(native, api.source, architecture=architecture,
                media_architecture=media.architecture, binding=binding, empty_context=empty_context,
                generation_sigmas=schedule, trainable=parameters['trainable'])
        state = training_state(roles['student'], roles['fake'], roles['teacher'], student_lr=parameters['student_lr'],
            fake_lr=parameters['fake_lr'], scheduler_gamma=parameters['scheduler_gamma'], ema_decay=parameters['ema_decay'])
    state['media'] = media
    # Keep the original evidence label in public metadata. Whole-role trial
    # copies are qualified only for our explicit random actual-class fixture;
    # this provenance label is not a pretrained/host-RSS capacity guarantee.
    qualify_trial_roles(state, 'SYNTHETIC_TEST_ONLY'
        if manifest['evidence_kind'] == 'RANDOM_TINY_ACTUAL_NATIVE_CLASSES_NOT_PRETRAINED' else None)
    immutable = ['teacher', 'media']
    for role in ('student', 'fake'):
        immutable.extend(role+'.'+path+'.base' for path, module in state[role].named_modules()
                         if isinstance(module, TrainableLowRankLinear))
    metadata = _metadata(api, architecture=architecture, media_architecture=media.architecture, binding=binding,
        trainable=parameters['trainable'], generation_sigmas=schedule)
    metadata.update(asset_identity={**identity, 'roles': role_receipts},
                    evidence_kind=manifest['evidence_kind'],
                    optimizer='AdamW(beta=.9,.999; eps=1e-8; weight_decay=0); scheduler per ACTUAL role update',
                    ema_decay=parameters['ema_decay'])
    return ModelComponents(state=state, immutable_modules=tuple(immutable), metadata=metadata,
        prediction=PredictionSpec('flow'), capabilities=('video', 'audio', 'flow', 'lora', 'dmd_flow_v1'),
        generate=generate, decode=media.decode, inference_projection=project_current_generator,
        admit_training=admit_training).validate()


def admit_training(*, state, request, prepared_identity, prepared_manifest):
    from .distributed import rank_world
    from distillschedule.data.ltx_inputs import SCHEMA
    require(request.trainer.get('device', 'cpu') == 'cpu' and 'distributed' not in request.trainer and rank_world()[1] == 1,
            'LTX native local validation admits explicit CPU1 only; distributed/GPU behavior NOT_RUN')
    require(request.execution.get('adapter', 'program') == 'program'
            and request.algorithm['factory'] == 'distillschedule.training.dmd:build_algorithm',
            'actual shared No-GAN flow DMD factory required, not official DMD2/GAN/Self-Forcing substitution')
    require('streams' not in request.data and prepared_manifest.get('metadata', {}).get('schema_version') == SCHEMA,
            'this LTX recipe consumes one complete native AV record per common cycle')
    require(list(state['student'].generation_sigmas) == request.algorithm.get('parameters', {}).get('rollout_sigmas'),
            'training/generation rollout sigma contracts disagree')


def generate(state, feeds, parameters):
    from .dmd import sigmas
    require(isinstance(parameters, dict) and set(parameters) <= {'rollout_sigmas'}, 'unsupported LTX generation parameter')
    student = state['student']
    schedule = sigmas(parameters.get('rollout_sigmas', student.generation_sigmas))
    sample = student.rollout(feeds, schedule, 'all_steps')
    patch = student.source.import_module('ltx_core.components.patchifiers')
    types = student.source.import_module('ltx_core.types')
    video = patch.VideoLatentPatchifier(1).unpatchify(sample['video'], types.VideoLatentShape(*feeds['video_shape'].tolist()))
    audio = patch.AudioPatchifier(1).unpatchify(sample['audio'], types.AudioLatentShape(*feeds['audio_shape'].tolist()))
    return {'video_latents': video, 'audio_latents': audio, 'latent_schema': {
        'video_latents': {'layout': 'BCFHW', 'shape': list(video.shape), 'dtype': str(video.dtype)},
        'audio_latents': {'layout': 'BCTF', 'shape': list(audio.shape), 'dtype': str(audio.dtype)}}}


def _inference_parameters(student):
    return dict(source_root=str(student.source.root), architecture=deepcopy(student.architecture),
        media_architecture=deepcopy(student.media_architecture), binding=student.binding,
        trainable=deepcopy(student.trainable_config), generation_sigmas=list(student.generation_sigmas))


def _inference_components(api, student, media, parameters):
    return ModelComponents(state={'student': student, 'media': media}, immutable_modules=('media',),
        metadata=_metadata(api, architecture=parameters['architecture'], media_architecture=parameters['media_architecture'],
            binding=parameters['binding'], trainable=parameters['trainable'], generation_sigmas=parameters['generation_sigmas']),
        prediction=PredictionSpec('flow'), capabilities=('video', 'audio', 'flow', 'lora', 'dmd_flow_v1'),
        generate=generate, decode=media.decode).validate()


def project_current_generator(components, effective_training_parameters):
    student = components.state['student']; media = components.state['media']
    _validate_architecture(student.architecture)
    parameters = _inference_parameters(student)
    api = original(parameters['source_root'])
    return InferenceProjection(components=_inference_components(api, student, media, parameters),
        factory='distillschedule.training.ltx:build_inference_model', parameters=parameters)


def build_inference_model(parameters):
    from .dmd import sigmas
    from .ltx_media import NativeMedia
    allowed = {'source_root', 'architecture', 'media_architecture', 'binding', 'trainable', 'generation_sigmas', 'seed', 'device'}
    require(isinstance(parameters, dict) and set(parameters) <= allowed
            and allowed - {'seed', 'device'} <= parameters.keys() and parameters.get('device', 'cpu') == 'cpu',
            'explicit exported native LTX CPU architecture required, no training/base bundle')
    api = original(parameters['source_root']); schedule = sigmas(parameters['generation_sigmas'])
    media = NativeMedia(api.source, parameters['media_architecture'])
    native = native_transformer(api.source, parameters['architecture'])
    trainable_scope(native, parameters['trainable'])
    p = parameters['media_architecture']; length = p['text_encoder']['tokenizer']['max_length']
    empty = {'video': torch.zeros(1, length, p['text_processor']['video_dim']),
             'audio': torch.zeros(1, length, p['text_processor']['audio_dim']), 'mask': torch.zeros(1, length, dtype=torch.bool)}
    student = LTXRole(native, api.source, architecture=parameters['architecture'], media_architecture=p,
        binding=parameters['binding'], empty_context=empty, generation_sigmas=schedule, trainable=parameters['trainable'])
    # Only layout is reconstructed here. The common export loader MUST strictly
    # replace every buffer/parameter before validation or generation can succeed.
    return _inference_components(api, student, media, parameters)
