"""Official external DMD2 SD1.5/SDXL models, explicit local assets only.

The noncommercial upstream is deliberately not vendored or relicensed here.
Both factories instantiate its actual SDUniModel/SDGuidance numerical classes.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from .adapters import ModelComponents
from .prediction import PredictionSpec
from .upstream import PinnedSource
from distillschedule.workloads.best_known.registry import file_sha256

REVISION = '8d8fa55633d47cfb81bbc7a892e7248f9518763f'
REPOSITORY = 'https://github.com/tianweiy/DMD2'
SOURCE_FILES = {
    'main/sd_unified_model.py': 'caaf04746dbc0f79b6ca9de9dc37cb5fcae99d0311d2ee2e17eeb5dbea8a8e36',
    'main/sd_guidance.py': '5e5ec5cb2978f40309d6a89d7e52255d508a7707345291284c2d4969918851fd',
    'main/sd_unet_forward.py': 'c78803275026c1056f62f3aa1e9d379dda9ed0160cc26bb6e13001fecb118483',
    'main/sdxl/sdxl_text_encoder.py': 'd82840a257b465dd44d3694fe90a4c99473c0edd6aa101020aac55737d631c5c',
    'main/utils.py': 'ccac4f287123a93b7d0246b95b0c21da2d031bd7bd06159b0d4422f7ffb128e8',
    'LICENSE.md': '17b2442405956fe8457ef0e230cbc18ed2cb6444bc86c04b5d61b75df3518335',
}


def require(condition, message):
    if not condition:
        raise ValueError('official DMD2: ' + message)


def original(source_root):
    import torch
    source = PinnedSource(source_root, SOURCE_FILES, import_roots=('main',),
        definitions={'main.utils': ('get_x0_from_noise', 'NoOpContext', 'DummyNetwork')},
        definition_globals={'main.utils': {'torch': torch, 'nn': torch.nn}})
    return SimpleNamespace(source=source,
        unified=source.import_module('main.sd_unified_model'),
        guidance=source.import_module('main.sd_guidance'),
        unet=source.import_module('main.sd_unet_forward'),
        text=source.import_module('main.sdxl.sdxl_text_encoder'),
        utils=source.import_module('main.utils'))


def local_bundle_identity(model_id, *, sdxl):
    root = Path(model_id).expanduser().resolve(strict=True)
    require(root.is_dir(), 'model_id must be an explicit local pretrained directory')
    folders = ['unet', 'scheduler', 'vae', 'text_encoder', 'tokenizer']
    if sdxl:
        folders += ['text_encoder_2', 'tokenizer_2']
    files = {}
    for folder in folders:
        directory = root / folder
        require(directory.is_dir(), 'missing local pretrained subfolder: ' + folder)
        entries = sorted(path for path in directory.rglob('*') if path.is_file())
        require(bool(entries), 'empty local pretrained subfolder: ' + folder)
        for path in entries:
            require(path.resolve().is_relative_to(root), 'asset symlink escapes model bundle')
            files[path.relative_to(root).as_posix()] = file_sha256(path)
    return {'root': str(root), 'files': files}


def runtime_identity():
    import importlib.metadata
    import torch
    return {'torch': str(torch.__version__), **{name: importlib.metadata.version(name)
            for name in ('diffusers', 'transformers', 'peft', 'accelerate')}}


class _LocalAccelerator:
    """Single-process placement only; no replacement for upstream math."""
    is_local_main_process = False

    def __init__(self, device):
        self.device = device

    @staticmethod
    def unwrap_model(model):
        return model


def _execution_device(value):
    import torch
    require(isinstance(value, str), 'device must be cpu or a single CUDA device')
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise ValueError('official DMD2: invalid device') from error
    require(device.type in ('cpu', 'cuda') and (device.type != 'cpu' or device.index is None),
            'device must be cpu or a single CUDA device')
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'requested CUDA device is unavailable')
        index = device.index if device.index is not None else torch.cuda.current_device()
        require(0 <= index < torch.cuda.device_count(), 'requested CUDA device index is unavailable')
        device = torch.device('cuda', index)
    return device


def _device_scope(device):
    import torch
    return torch.cuda.device(device) if device.type == 'cuda' else nullcontext()


@contextmanager
def _construction_rng(device, seed):
    """Initialize only the owned generators, leaving caller/unowned RNG intact."""
    from .checkpoint import rng_state, restore_rng
    from .session import _seed
    with _device_scope(device):
        outer = rng_state(device=device)
        try:
            _seed(seed, device=device)
            yield
        finally:
            restore_rng(outer)


def _place_native_tensors(model, device):
    """Module.to does not place the upstream scheduler or plain conditions."""
    import torch
    model.to(device)
    for owner in (model, model.noise_scheduler):
        for name, value in vars(owner).copy().items():
            if isinstance(value, torch.Tensor):
                setattr(owner, name, value.to(device))
    # All three are the same original schedule, not independently mutable copies.
    model.alphas_cumprod = model.noise_scheduler.alphas_cumprod = model.guidance_model.alphas_cumprod


def _arguments(parameters, *, sdxl):
    import torch
    require(isinstance(parameters, dict) and set(parameters) <= {
        'source_root', 'model_id', 'settings', 'generator_checkpoint', 'seed', 'device'},
        'unknown model parameters')
    require(isinstance(parameters.get('source_root'), str)
            and isinstance(parameters.get('model_id'), str), 'source_root and local model_id required')
    _execution_device(parameters.get('device', 'cpu'))
    require(not torch.distributed.is_initialized(), 'distributed ownership is not admitted')
    require(type(parameters.get('seed', 0)) is int and 0 <= parameters.get('seed', 0) < 2**63,
            'invalid model RNG seed')
    defaults = dict(initialie_generator=True, use_fp16=False, gradient_checkpointing=False,
        backward_simulation=sdxl, cls_on_clean_image=False, denoising=sdxl,
        denoising_timestep=1000, conditioning_timestep=999, num_denoising_step=4 if sdxl else 1,
        num_train_timesteps=1000, grid_size=1, tiny_vae=False, sdxl=sdxl, revision=None,
        gan_alone=False, min_step_percent=0.02, max_step_percent=0.98,
        real_guidance_scale=6., fake_guidance_scale=1., gen_cls_loss=False,
        gen_cls_loss_weight=0., guidance_cls_loss_weight=0., dm_loss_weight=1.,
        diffusion_gan=False, diffusion_gan_max_timestep=0, generator_lora=False,
        lora_rank=64, lora_alpha=8., lora_dropout=0.,
        dfake_gen_update_ratio=5 if sdxl else 10, batch_size=1, resolution=1024 if sdxl else 512,
        latent_resolution=128 if sdxl else 64, latent_channel=4, max_grad_norm=10.,
        warmup_step=500, train_iters=1000000, generator_lr=None, guidance_lr=None)
    settings = parameters.get('settings', {})
    require(isinstance(settings, dict) and set(settings) <= set(defaults), 'unknown DMD2 settings')
    defaults.update(settings)
    require(defaults['sdxl'] is sdxl and defaults['use_fp16'] is False
            and defaults['tiny_vae'] is False and defaults['initialie_generator'] is True
            and defaults['revision'] is None, 'unsupported model initialization/precision override')
    for key in ('backward_simulation', 'cls_on_clean_image', 'denoising', 'gradient_checkpointing',
                'gan_alone', 'gen_cls_loss', 'diffusion_gan', 'generator_lora'):
        require(type(defaults[key]) is bool, 'boolean setting required: ' + key)
    for key in ('dfake_gen_update_ratio', 'batch_size', 'resolution', 'latent_resolution',
                'latent_channel', 'num_denoising_step', 'num_train_timesteps', 'denoising_timestep',
                'grid_size', 'train_iters', 'lora_rank'):
        require(type(defaults[key]) is int and defaults[key] > 0, 'positive integer required: ' + key)
    require(defaults['num_train_timesteps'] == 1000 and defaults['denoising_timestep'] == 1000
            and defaults['latent_channel'] == 4,
            'official SD epsilon/timestep/latent contract changed')
    require(defaults['conditioning_timestep'] == 999 or (sdxl and defaults['num_denoising_step'] == 1
            and defaults['conditioning_timestep'] == 399 and parameters.get('generator_checkpoint')),
            'conditioning399 requires the actual SDXL single-step checkpoint')
    require(0 <= defaults['min_step_percent'] < defaults['max_step_percent'] <= 1
            and defaults['fake_guidance_scale'] == 1, 'invalid score timestep or fake CFG policy')
    require(type(defaults['warmup_step']) is int and defaults['warmup_step'] >= 0,
            'nonnegative scheduler warmup required')
    for key in ('generator_lr', 'guidance_lr', 'max_grad_norm'):
        value = defaults[key]
        require(type(value) in (int, float) and 0 < value < float('inf'), 'explicit finite positive ' + key + ' required')
    for key in ('gen_cls_loss_weight', 'guidance_cls_loss_weight', 'dm_loss_weight', 'real_guidance_scale',
                'lora_alpha', 'lora_dropout'):
        require(type(defaults[key]) in (int, float) and 0 <= defaults[key] < float('inf'),
                'finite nonnegative setting required: ' + key)
    require(not defaults['denoising'] or sdxl, 'original denoising is SDXL-only')
    require(not defaults['backward_simulation'] or defaults['denoising'], 'backward simulation needs denoising')
    require(not defaults['generator_lora'] or sdxl, 'original generator LoRA is SDXL-only')
    require(not defaults['gan_alone'] or (defaults['cls_on_clean_image'] and defaults['gen_cls_loss']
            and defaults['gen_cls_loss_weight'] > 0), 'GAN-alone requires an actual weighted generator GAN loss')
    require(not defaults['gen_cls_loss'] or defaults['cls_on_clean_image'], 'GAN loss requires original classifier')
    require(not defaults['diffusion_gan'] or (defaults['cls_on_clean_image']
            and type(defaults['diffusion_gan_max_timestep']) is int
            and 0 < defaults['diffusion_gan_max_timestep'] <= 1000), 'invalid diffusion GAN noise range')
    if sdxl and defaults['num_denoising_step'] == 1:
        require(bool(parameters.get('generator_checkpoint')), 'SDXL 1-step requires an explicit actual generator checkpoint')
    require(defaults['num_denoising_step'] in (1, 2, 4, 5, 10, 20, 25, 50),
            'denoising steps must exactly divide the original 1000-step schedule')
    defaults['model_id'] = str(Path(parameters['model_id']).expanduser().resolve(strict=True))
    return SimpleNamespace(**defaults)


def _build(parameters, *, sdxl):
    args = _arguments(parameters, sdxl=sdxl)
    device = _execution_device(parameters.get('device', 'cpu'))
    with _construction_rng(device, parameters.get('seed', 0)):
        return _build_placed(parameters, args, device, sdxl=sdxl)


def _build_placed(parameters, args, device, *, sdxl):
    import torch
    from transformers import CLIPTokenizer
    from distillschedule.data.sd_inputs import preprocessing_assets
    from .dmd2_owner import DMD2Owner
    api = original(parameters['source_root'])
    bundle = local_bundle_identity(args.model_id, sdxl=sdxl)
    accelerator = _LocalAccelerator(device)
    model = api.unified.SDUniModel(args, accelerator)
    _place_native_tensors(model, device)
    require(model.guidance_model.scheduler.config.prediction_type == 'epsilon'
            and model.guidance_model.scheduler.config.num_train_timesteps == 1000,
            'local scheduler must use the original 1000-step epsilon convention')
    require(model.feedforward_model.config.in_channels == 4
            and model.feedforward_model.config.out_channels == 4, 'native 4-channel UNet required')
    if args.cls_on_clean_image:
        require(model.guidance_model.fake_unet.config.block_out_channels[-1] == 1280,
                'original GAN head requires 1280 bottleneck channels; it is never resized')
    checkpoint_identity = None
    if parameters.get('generator_checkpoint'):
        path = Path(parameters['generator_checkpoint']).expanduser().resolve(strict=True)
        checkpoint_identity = {'path': str(path), 'sha256': file_sha256(path)}
        value = torch.load(path, map_location='cpu', weights_only=True)
        model.feedforward_model.load_state_dict(value, strict=True)
        require(file_sha256(path) == checkpoint_identity['sha256'], 'generator checkpoint changed during load')
    require(local_bundle_identity(args.model_id, sdxl=sdxl) == bundle, 'local model bundle changed during load')
    uncond = None
    if not sdxl:
        tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder='tokenizer', local_files_only=True)
        ids = tokenizer([''], max_length=tokenizer.model_max_length, return_tensors='pt',
                        padding='max_length', truncation=True).input_ids
        with torch.no_grad():
            uncond = model.text_encoder(ids.to(device))[0].detach()
    metadata = dict(schema_version='distillschedule.official_dmd2_sd.v1', variant='sdxl' if sdxl else 'sd15',
        upstream=dict(repository=REPOSITORY, revision=REVISION, license='CC-BY-NC-SA-4.0', **api.source.identity),
        bundle=bundle, preprocessing_assets=preprocessing_assets(bundle['files'],
            variant='sdxl' if sdxl else 'sd15', images=True),
        generator_checkpoint=checkpoint_identity, settings=deepcopy(vars(args)),
        runtime=runtime_identity(), precision='FP32_CPU1' if device.type == 'cpu' else 'FP32_CUDA1',
        placement=dict(device=str(device), layout='SINGLE_DEVICE_RESIDENT', distributed=False, amp=False),
        ema='NONE', quality_validation='NOT_RUN',
        cpu_execution='CPU_IMPLEMENTATION_TESTED_ON_RANDOM_TINY_ASSETS_ONLY', gpu_execution='NOT_RUN',
        placement_limit='SINGLE_DEVICE_FP32_RUNTIME_TENSORS_EXCLUDES_HOST_RNG_AND_ADAM_STEP_METADATA',
        cuda_trial_snapshot='UNAVAILABLE_AUTO_PLAN_AND_OWNER_EXECUTION_REJECTED_BEFORE_TRIAL')
    owner = DMD2Owner(model, args, api.source, uncond, identity=metadata, seed=parameters.get('seed', 0))
    return ModelComponents(state={'native_cycle': owner}, metadata=metadata,
        prediction=PredictionSpec('epsilon'), capabilities=('image', 'epsilon', 'gan', *(('lora',) if sdxl else ())),
        generate=generate, decode=_decoder(model.vae), inference_projection=project_current_generator,
        admit_training=admit_training, collect_metrics=collect_metrics).validate()


def build_sd15(parameters):
    return _build(parameters, sdxl=False)


def build_sdxl(parameters):
    return _build(parameters, sdxl=True)


def admit_training(*, state, request, prepared_identity, prepared_manifest):
    from .dmd2_owner import stream_counts
    from distillschedule.data.sd_inputs import validate_prepared_binding
    owner = state['native_cycle']
    require(request.execution.get('adapter', 'program') == 'program', 'common Program adapter required')
    require(request.algorithm['factory'] == 'distillschedule.training.dmd2:build_dmd2', 'official DMD2 algorithm required')
    require(not request.algorithm.get('parameters'), 'training policy belongs to model settings, not algorithm overrides')
    counts = stream_counts(owner.args)
    streams = request.data.get('streams', {})
    require(set(streams) == set(counts) and all(streams[name].get('samples_per_cycle', 1) == count
            for name, count in counts.items()), 'independent streams/counts must match native iteration consumption')
    require(isinstance(prepared_manifest, dict) and isinstance(prepared_manifest.get('streams'), dict)
            and set(prepared_manifest['streams']) == set(counts), 'prepared preprocessing streams binding required')
    for manifest in prepared_manifest['streams'].values():
        validate_prepared_binding(manifest, owner.identity['preprocessing_assets'])
    require('distributed' not in request.trainer
            and _execution_device(request.trainer.get('device', 'cpu')) == owner.device,
            'single-device trainer/model placement must match; distributed ownership is not admitted')
    owner.admit_execution(request.execution.get('mode', 'reference'), request.execution.get('implementation', {}))
    require(request.trainer['cycles'] * owner.ratio <= owner.args.train_iters, 'native train_iters budget exceeded')


def collect_metrics(outputs):
    rows = outputs['dmd2.native_cycle']['iterations']
    return {f'native_{index}/{name}': value for index, row in enumerate(rows)
            for name, value in row['losses'].items()}


def _generation_row(feeds):
    if 'generator' in feeds:
        return feeds['generator'][0]
    return feeds


def generate(state, feeds, parameters):
    """Current generator, original x0 conversion and ancestral renoising."""
    from distillschedule.data.sd_inputs import validate_row
    owner = state['native_cycle']; model = owner.model
    owner._validate_live()
    validate_row(_generation_row(feeds), sdxl=owner.args.sdxl,
                 binding=owner.identity['preprocessing_assets'], model=model)
    model.eval()
    return _generate_inference(dict(generator=model.feedforward_model, text_encoder=model.text_encoder,
                                   vae=model.vae), feeds, parameters, _inference_parameters(owner))


def _decoder(vae):
    def decode(generated, parameters):
        import torch
        image = vae.decode(generated['latents'] / vae.config.scaling_factor).sample.float()
        require(image.ndim == 4 and image.shape[0] == 1 and image.shape[1] == 3,
                'public image output requires one native BCHW RGB sample')
        return {'image': ((image[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).contiguous()}
    return decode


def _plain(value):
    from enum import Enum
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _plain(sorted(item) if key == '_use_default_values' else item)
                for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_plain(item) for item in (sorted(value) if isinstance(value, set) else value)]
    require(value is None or type(value) in (str, bool, int, float), 'non-JSON inference architecture')
    return value


def _inference_parameters(owner):
    model, args = owner.model, owner.args
    text = model.text_encoder
    architecture = dict(unet=_plain(dict(model.feedforward_model.config)), vae=_plain(dict(model.vae.config)),
        scheduler=_plain(dict(model.noise_scheduler.config)),
        clip_one=_plain((text.text_encoder_one if args.sdxl else text).config.to_dict()))
    architecture['clip_one']['attn_implementation'] = (text.text_encoder_one if args.sdxl else text).config._attn_implementation
    if args.sdxl:
        architecture['clip_two'] = _plain(text.text_encoder_two.config.to_dict())
        architecture['clip_two']['attn_implementation'] = text.text_encoder_two.config._attn_implementation
    for key in ('clip_one', 'clip_two'):
        if key in architecture:
            architecture[key]['_name_or_path'] = ''
    lora = {name: _plain(config.to_dict()) for name, config in
            getattr(model.feedforward_model, 'peft_config', {}).items()}
    return dict(source_root=str(owner.source.root), upstream=deepcopy(owner.source.identity),
        variant='sdxl' if args.sdxl else 'sd15', architecture=architecture, lora=lora, runtime=runtime_identity(),
        preprocessing_assets=deepcopy(owner.identity['preprocessing_assets']),
        geometry=dict(resolution=args.resolution, latent_resolution=args.latent_resolution, latent_channel=4),
        steps=args.num_denoising_step if args.sdxl else 1,
        conditioning_timestep=args.conditioning_timestep, seed=0)


def _inference_components(state, parameters):
    metadata = dict(schema_version='distillschedule.official_sd_inference.v2',
        parameters=deepcopy({key: value for key, value in parameters.items() if key != 'device'}),
        selection='CURRENT_NOT_EMA', quality_validation='NOT_RUN', precision='FP32_SINGLE_DEVICE',
        dependencies='EXTERNAL_PINNED_SOURCE_ONLY_NO_BASE_MODEL_WEIGHTS')

    def generate_current(state, feeds, generation_parameters):
        return _generate_inference(state, feeds, generation_parameters, parameters)

    def refuse_training(**kwargs):
        raise ValueError('official SD inference-only export cannot train')

    return ModelComponents(state=state, metadata=metadata, generate=generate_current,
        decode=_decoder(state['vae']), admit_training=refuse_training,
        prediction=PredictionSpec('epsilon'), capabilities=('image', 'epsilon')).validate()


def project_current_generator(components, effective_training_parameters):
    from .adapters import InferenceProjection
    owner = components.state['native_cycle']
    owner._validate_live()
    parameters = _inference_parameters(owner)
    state = dict(generator=owner.model.feedforward_model, text_encoder=owner.model.text_encoder, vae=owner.model.vae)
    return InferenceProjection(_inference_components(state, parameters),
        'distillschedule.training.sd:build_inference_model', parameters).validate()


def build_inference_model(parameters):
    """Architecture-only G/CLIP/VAE; no base weights, teacher, fake, optimizer."""
    import torch
    from diffusers import UNet2DConditionModel, AutoencoderKL
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
    from peft import LoraConfig
    from distillschedule.data.sd_inputs import validate_preprocessing_assets
    require(set(parameters) <= {'source_root', 'upstream', 'variant', 'architecture', 'lora', 'runtime',
            'preprocessing_assets', 'geometry', 'steps', 'conditioning_timestep', 'seed', 'device'}
            and parameters['variant'] in ('sd15', 'sdxl'),
            'invalid inference-only construction parameters')
    device = _execution_device(parameters.get('device', 'cpu'))
    validate_preprocessing_assets(parameters.get('preprocessing_assets'), variant=parameters['variant'], images=True)
    api = original(parameters['source_root'])
    require(parameters['upstream'] == api.source.identity, 'inference upstream source lock changed')
    require(parameters['runtime'] == runtime_identity(), 'inference numerical dependency versions changed')
    # Some library config constructors consume/pop entries from their mapping.
    # Never let them mutate the saved inference identity or caller parameters.
    architecture = deepcopy(parameters['architecture']); sdxl = parameters['variant'] == 'sdxl'
    def clip_config(name):
        config = architecture[name]
        return CLIPTextConfig.from_dict(config, attn_implementation=config['attn_implementation'])
    generator = UNet2DConditionModel.from_config(architecture['unet'])
    for name, config in parameters['lora'].items():
        generator.add_adapter(LoraConfig(**config), adapter_name=name)
    vae = AutoencoderKL.from_config(architecture['vae'])
    if sdxl:
        # Use the exact original dual-CLIP forward but construct from configs;
        # every weight is restored by the common named-module export loader.
        text = api.text.SDXLTextEncoder.__new__(api.text.SDXLTextEncoder)
        torch.nn.Module.__init__(text)
        text.text_encoder_one = CLIPTextModel(clip_config('clip_one'))
        text.text_encoder_two = CLIPTextModelWithProjection(clip_config('clip_two'))
        text.accelerator = _LocalAccelerator(device)
    else:
        text = CLIPTextModel(clip_config('clip_one'))
    text.requires_grad_(False); vae.requires_grad_(False)
    for module in (generator, text, vae):
        module.to(device=device, dtype=torch.float32)
    return _inference_components(dict(generator=generator, text_encoder=text, vae=vae), parameters)


def _generate_inference(state, feeds, parameters, settings):
    import torch
    from distillschedule.data.sd_inputs import validate_row
    require(set(parameters) <= {'steps'}, 'unknown inference settings')
    row = _generation_row(feeds); sdxl = settings['variant'] == 'sdxl'
    generator, text = state['generator'], state['text_encoder']
    device = next(generator.parameters()).device
    require(not torch.is_autocast_enabled(device.type), 'FP32 inference does not admit AMP')
    require(all(t.device == device and (not t.is_floating_point() or t.dtype == torch.float32)
                for module in state.values() for t in (*module.parameters(), *module.buffers())),
            'inference module device/FP32 placement changed')
    validate_row(row, sdxl=sdxl, binding=settings['preprocessing_assets'], device=device)
    steps = parameters.get('steps', settings['steps'])
    require(type(steps) is int and steps > 0 and 1000 % steps == 0 and (sdxl or steps == 1),
            'invalid inference denoising schedule')
    if sdxl:
        embedding, pooled = text(row)
        resolution = settings['geometry']['resolution']
        time_ids = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]],
                                dtype=torch.float32, device=device)
        conditions = {'time_ids': time_ids.repeat(len(embedding), 1), 'text_embeds': pooled}
    else:
        embedding = text(row['text_input_ids_one'].squeeze(1))[0].float(); conditions = None
    geometry = settings['geometry']
    sample = torch.randn(len(embedding), 4, geometry['latent_resolution'], geometry['latent_resolution'], device=device)
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler.from_config(settings['architecture']['scheduler'])
    for name, value in vars(scheduler).copy().items():
        if isinstance(value, torch.Tensor):
            setattr(scheduler, name, value.to(device))
    get_x0 = original(settings['source_root']).utils.get_x0_from_noise
    schedule = [settings['conditioning_timestep']] if steps == 1 else list(range(999, 0, -(1000 // steps)))
    for index, timestep in enumerate(schedule):
        times = torch.full((len(embedding),), timestep, dtype=torch.long, device=device)
        noise = generator(sample, times, embedding, added_cond_kwargs=conditions).sample
        image = get_x0(sample.double(), noise.double(), scheduler.alphas_cumprod.double(), times).float()
        if index + 1 < len(schedule):
            sample = scheduler.add_noise(image, torch.randn_like(image), torch.full_like(times, schedule[index + 1]))
    return {'latents': image}
