"""Pinned Wan2.2 TI2V-5B T2V through the existing common flow-DMD lifecycle.

CPU1/FP32/B1, four Euler steps, G1/F5, unmerged LoRA. Not Wan2.1,
A14B/high-low experts, I2V or official Self-Forcing/DMD2. Production loads only
hash-bound local native safe assets. Random weights belong to the labeled fixture.
"""
from contextlib import nullcontext
from copy import deepcopy
import importlib.metadata
import inspect
from pathlib import Path
import sys

import torch
from torch import nn

from . import flow_image as safe
from .adapters import InferenceProjection, ModelComponents
from .flow_image import require
from .low_rank import TrainableLowRankLinear, trainable_scope
from .prediction import PredictionSpec
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint

FAMILY = 'wan22-ti2v5b-t2v'
MODEL_ID = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
REVISION = 'b8fff7315c768468a5333511427288870b2e9635'
ENTRY = 'distillschedule.training.wan22'
ASSET_SCHEMA = 'distillschedule.wan22_t2v_assets.v1'
INPUT_SCHEMA = 'distillschedule.wan22_t2v_inputs.v1'
PIPELINE = dict(_class_name='WanPipeline', expand_timesteps=True, transformer_2=None, boundary_ratio=None, task='t2v')
CODEC_POLICY = 'NATIVE_RESIDUAL_WAN_VAE_PATCH2_INTERNAL_SPATIAL16_TEMPORAL4_MODE_MEAN_STD; UMT5_NATIVE226_ZERO_PAD'


def classes():
    from diffusers import WanTransformer3DModel, AutoencoderKLWan, WanPipeline
    from transformers import UMT5EncoderModel
    return WanTransformer3DModel, AutoencoderKLWan, UMT5EncoderModel, WanPipeline


def vae_config(config):
    """Only the audited upstream historical no-op; raw file/config is retained."""
    result = deepcopy(config)
    if 'clip_output' in result:
        require(result.pop('clip_output') is False, 'historical Wan VAE clip_output must be exactly false')
    return result


def validate_architecture(config, media):
    native, vae, text, _ = classes()
    safe.architecture(native, config)
    safe.architecture(vae, vae_config(media['vae']))
    text_config = safe.architecture(text, media['text_encoder'])
    v = media['vae']
    require(config.get('patch_size') in ([1,2,2], (1,2,2))
            and config.get('in_channels') == config.get('out_channels') == v.get('z_dim')
            and config.get('text_dim') == text_config.d_model
            and config.get('image_dim') is None and config.get('added_kv_proj_dim') is None
            and config.get('pos_embed_seq_len') is None,
            'only same-domain native TI2V-5B T2V, no image/experts/causal variant admitted')
    require(v.get('is_residual') is True and v.get('in_channels') == v.get('out_channels') == 12
            and v.get('patch_size') == 2 and v.get('scale_factor_temporal') == 4
            and v.get('scale_factor_spatial') == 16 and len(v.get('dim_mult', [])) == 4
            and v.get('temperal_downsample') == [False,True,True]
            and media.get('text_max_length') == 226,
            'actual residual Wan2.2 spatial16/temporal4/patch2 VAE and native226 text required')
    for key in ('latents_mean', 'latents_std'):
        value = v.get(key)
        require(isinstance(value, list) and len(value) == v['z_dim']
                and all(type(x) in (int,float) and abs(x) < float('inf') for x in value),
                'explicit finite native VAE channel normalization required')
    require(all(x > 0 for x in v['latents_std']), 'native VAE standard deviations must be positive')
    require(type(config.get('rope_max_seq_len')) is int and config['rope_max_seq_len'] > 0,
            'explicit bounded native RoPE geometry required')


def runtime_identity():
    from transformers import T5TokenizerFast
    import ftfy
    import diffusers.pipelines.wan.pipeline_wan as pipeline
    result = safe.runtime_identity(sys.modules[__name__])
    for name, version in (('ftfy','6.3.1'), ('sentencepiece','0.2.2'), ('tokenizers','0.22.2')):
        require(importlib.metadata.version(name) == version, 'native Wan tokenizer/cleaner dependency pin changed: ' + name)
        result['versions'][name] = version
    result['source_files'].update({
        'feature_encoder': file_sha256(Path(__file__).parent.parent/'data/wan22_inputs.py'),
        'shared_tensor_validator': file_sha256(Path(__file__).parent.parent/'data/flow_image_inputs.py'),
        'raw_media_decoder': file_sha256(Path(__file__).parent.parent/'data/decoding.py'),
        'source_identity': file_sha256(Path(__file__).parent.parent/'data/manifest.py'),
        'native_t5_tokenizer': file_sha256(Path(inspect.getfile(T5TokenizerFast))),
        'native_prompt_clean': file_sha256(Path(pipeline.__file__)), 'ftfy': file_sha256(Path(ftfy.__file__))})
    return result


def tokenizer_state(tokenizer):
    from transformers import T5TokenizerFast
    require(type(tokenizer) is T5TokenizerFast and tokenizer.backend == 'tokenizers'
            and tokenizer.padding_side == 'right', 'actual native T5 fast backend/right padding required')
    value = safe.tokenizer_state(tokenizer)
    value['class'] = type(tokenizer).__module__ + '.' + type(tokenizer).__name__
    value['options'].update(extra_ids=tokenizer._extra_ids, backend='tokenizers',
        additional_special_tokens=[str(x) for x in tokenizer._extra_special_tokens],
        clean_up_tokenization_spaces=tokenizer.clean_up_tokenization_spaces,
        split_special_tokens=tokenizer.split_special_tokens, add_prefix_space=tokenizer.add_prefix_space,
        model_input_names=list(tokenizer.model_input_names))
    return value


def restore_tokenizer(value):
    from tokenizers import Tokenizer
    from transformers import T5TokenizerFast
    require(isinstance(value, dict) and set(value) == {'json','options','class'}
            and value['class'] == T5TokenizerFast.__module__ + '.' + T5TokenizerFast.__name__,
            'actual serialized T5 tokenizer class/backend/options required; Qwen reconstruction forbidden')
    tokenizer = T5TokenizerFast(tokenizer_object=Tokenizer.from_str(value['json']), **value['options'])
    require(tokenizer_state(tokenizer) == value, 'native T5 tokenizer options/backend did not restore exactly')
    return tokenizer


class Media(nn.Module):
    def __init__(self, vae, text_encoder, tokenizer, *, config, binding):
        super().__init__()
        self.vae = vae.eval().requires_grad_(False)
        self.text_encoder = text_encoder.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.architecture = deepcopy(config)
        self.binding = binding

    def pipeline(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        return classes()[3](tokenizer=self.tokenizer, text_encoder=self.text_encoder, vae=self.vae,
            scheduler=FlowMatchEulerDiscreteScheduler(), transformer=None, expand_timesteps=True)

    def encode_text(self, prompts):
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        require(isinstance(prompts,list) and prompts and all(isinstance(p,str) for p in prompts), 'explicit native prompt strings required')
        cleaned = [prompt_clean(p) for p in prompts]
        tokens = self.tokenizer(cleaned, truncation=False, add_special_tokens=True, return_attention_mask=True)
        require(all(0 < len(ids) <= 226 for ids in tokens['input_ids']), 'native226 prompt overflow; truncation refused')
        padded = self.tokenizer(cleaned, padding='max_length', max_length=226, truncation=False,
            add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
        with torch.no_grad():
            context = self.pipeline()._get_t5_prompt_embeds(prompts, max_sequence_length=226,
                device=torch.device('cpu'), dtype=torch.float32)
        return context.detach().float(), padded.attention_mask.bool()

    def encode_video(self, frames):
        require(type(frames) is torch.Tensor and frames.dtype == torch.uint8 and frames.device.type == 'cpu'
                and frames.ndim == 4 and frames.shape[-1] == 3 and min(frames.shape) > 0,
                'actual raw RGB THWC uint8 video required')
        f,h,w,_ = frames.shape
        require((f-1) % 4 == 0 and h % 32 == w % 32 == 0,
                'native raw video needs F=(Flatent-1)*4+1 and H/W divisible by32; no crop/resize/resample')
        pixels = frames.permute(3,0,1,2).unsqueeze(0).float()/127.5-1
        with torch.no_grad():
            raw = self.vae.encode(pixels).latent_dist.mode()
            mean = raw.new_tensor(self.vae.config.latents_mean).view(1,-1,1,1,1)
            std = raw.new_tensor(self.vae.config.latents_std).view(1,-1,1,1,1)
            value = (raw-mean)/std  # native VAE patchifies internally, exactly once
        require(list(value.shape) == [1,self.vae.config.z_dim,(f-1)//4+1,h//16,w//16], 'native VAE returned incompatible geometry')
        return value.detach()

    def decode_latents(self, latent):
        mean = latent.new_tensor(self.vae.config.latents_mean).view(1,-1,1,1,1)
        std = latent.new_tensor(self.vae.config.latents_std).view(1,-1,1,1,1)
        return self.vae.decode(latent*std+mean, return_dict=False)[0]

    def decode(self, generated, parameters):
        require(isinstance(parameters,dict) and set(parameters) <= {'rollout_sigmas'}, 'unsupported native video decoding option')
        latent = generated['video_latents']
        require(latent.ndim == 5 and latent.shape[:2] == (1,self.vae.config.z_dim), 'native decoder requires B1/CFHW latents')
        with torch.no_grad():
            pixels = self.decode_latents(latent)
        require(pixels.shape == (1,3,(latent.shape[2]-1)*4+1,latent.shape[3]*16,latent.shape[4]*16)
                and bool(torch.isfinite(pixels).all()), 'native residual VAE returned invalid RGB video')
        return {'video': ((pixels[0].permute(1,2,3,0).clamp(-1,1)+1)*127.5).round().to(torch.uint8).contiguous()}


class Role(nn.Module):
    def __init__(self, native, *, config, media_config, binding, empty, trainable, schedule):
        super().__init__()
        self.native = native
        self.architecture = deepcopy(config)
        self.media_config = deepcopy(media_config)
        self.binding = binding
        self.trainable_config = deepcopy(trainable)
        self.generation_sigmas = tuple(schedule)
        self.register_buffer('empty_context', empty[0].detach().clone())
        self.register_buffer('empty_context_mask', empty[1].detach().clone())

    def validate_feeds(self, feeds):
        from distillschedule.data.wan22_inputs import validate_row
        return validate_row(feeds, config=self.architecture, media_config=self.media_config, binding=self.binding,
            empty=(self.empty_context,self.empty_context_mask))

    def sample_template(self, feeds):
        return feeds['latents']

    def loss_masks(self, feeds, sample):
        return {'video': torch.ones_like(sample['video'],dtype=torch.bool)}

    def apply_conditioning(self, sample, feeds):
        require(isinstance(sample,dict) and set(sample) == {'video'}
                and sample['video'].shape == feeds['latents']['video'].shape,
                'only native T2V video payload admitted; I2V/experts/causal conditioning refused')
        return sample

    def denoise(self, sample, sigma, feeds, *, conditional):
        safe.cpu1()
        self.apply_conditioning(sample,feeds)
        value = sample['video']
        require(type(value) is torch.Tensor and value.dtype == torch.float32 and value.device.type == 'cpu'
                and bool(torch.isfinite(value).all()), 'finite native FP32 CPU video required')
        require(type(sigma) is torch.Tensor and sigma.dtype == torch.float32 and sigma.device.type == 'cpu'
                and sigma.shape == (1,) and bool(torch.isfinite(sigma).all())
                and bool(((sigma>=0)&(sigma<=1)).all()), 'native normalized per-batch sigma required')
        b,_,f,h,w = value.shape
        # TI2V-5B expand_timesteps=True; T2V mask is all ones. Native forward
        # consumes 1000*sigma at every spatial/temporal patch, without sign flip.
        timestep = (1000*sigma[:,None]).expand(b,f*(h//2)*(w//2))
        context = feeds['context'] if conditional else feeds['null_context']
        output = self.native(hidden_states=value,timestep=timestep,encoder_hidden_states=context,return_dict=False)[0]
        require(output.shape == value.shape and bool(torch.isfinite(output).all()), 'native Wan velocity shape/values changed')
        return {'video': output}

    def rollout(self, feeds, rollout_sigmas, gradient_policy):
        from .dmd import sigmas
        self.validate_feeds(feeds)
        schedule = sigmas(rollout_sigmas)
        require(gradient_policy in ('all_steps','last_step'), 'explicit Euler gradient policy required')
        value = {'video': torch.randn_like(feeds['latents']['video'])}
        for index,(sigma,following) in enumerate(zip(schedule,schedule[1:])):
            context = torch.no_grad() if gradient_policy == 'last_step' and index < len(schedule)-2 else nullcontext()
            with context:
                velocity = self.denoise(value,value['video'].new_full((1,),sigma),feeds,conditional=True)['video']
                value = {'video': value['video']+(following-sigma)*velocity}
        return value


def load_media_bundle(assets):
    from transformers import T5TokenizerFast
    safe.cpu1()
    runtime = runtime_identity()
    path = Path(assets).expanduser().resolve(strict=True)
    manifest = safe.json_object(path)
    require(set(manifest) == {'schema_version','family','source','evidence_kind','pipeline','roles','media','text_max_length'}
            and manifest['schema_version'] == ASSET_SCHEMA and manifest['family'] == FAMILY
            and manifest['source'] == dict(model_id=MODEL_ID,revision=REVISION) and manifest['pipeline'] == PIPELINE
            and manifest['evidence_kind'] in ('SYNTHETIC_TEST_ONLY','LOCAL_PRETRAINED_UNVERIFIED'),
            'exact Wan2.2 TI2V-5B T2V source/pipeline/schema required; I2V/A14B/high-low experts rejected')
    require(isinstance(manifest['roles'],dict) and set(manifest['roles']) == {'student','fake','teacher'}
            and isinstance(manifest['media'],dict) and set(manifest['media']) == {'vae','text_encoder','tokenizer'},
            'all actual Wan roles and native codecs/tokenizer required')
    roots,documents = {},[path]
    for name,spec in {**manifest['roles'],**manifest['media']}.items():
        roots[name],files = safe.asset_files(path.parent,spec); documents.extend(files)
    media_config = dict(vae=safe.json_object(roots['vae']/'config.json'),
        text_encoder=safe.json_object(roots['text_encoder']/'config.json'),text_max_length=manifest['text_max_length'])
    configs = [safe.json_object(roots[r]/'config.json') for r in ('student','fake','teacher')]
    require(all(c == configs[0] for c in configs), 'same-family native role architectures must match before loading')
    for config in configs:
        validate_architecture(config,media_config)
    native_cls,vae_cls,text_cls,_ = classes()
    # All constructor budgets precede any concrete model or weight allocation.
    for name,cls in [('student',native_cls),('fake',native_cls),('teacher',native_cls),('vae',vae_cls),('text_encoder',text_cls)]:
        config = safe.json_object(roots[name]/'config.json')
        witness = safe.shell(cls,vae_config(config) if name == 'vae' else config)
        count = sum(p.numel() for p in witness.parameters())
        spec = (manifest['roles'] if name in manifest['roles'] else manifest['media'])[name]
        require(0 < count <= spec['max_parameters'], 'native model exceeds explicit CPU parameter budget: '+name)
    binding = fingerprint(dict(family=FAMILY,source=manifest['source'],pipeline=PIPELINE,media=manifest['media'],
        text_max_length=226,runtime=runtime,dtype='FP32_CPU1_B1',codec_policy=CODEC_POLICY,
        transformer_config_sha256=file_sha256(roots['student']/'config.json')))
    with torch.device('cpu'),torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        def load(name,cls,**kwargs):
            spec = manifest['media'][name]
            return safe.load_native(cls,roots[name],spec['max_parameters'],
                dtype_policy=spec.get('dtype_policy','strict_fp32'),**kwargs)[0]
        vae = load('vae',vae_cls,config_transform=vae_config)
        text = load('text_encoder',text_cls)
        tokenizer = T5TokenizerFast.from_pretrained(roots['tokenizer'],local_files_only=True)
        media_config['tokenizer'] = tokenizer_state(tokenizer)
        media = Media(vae,text,tokenizer,config=media_config,binding=binding).eval().requires_grad_(False)
        empty = media.encode_text([''])
    return media,manifest,roots,runtime,binding,media_config,empty,documents


def _components(state, descriptor, *, training):
    from distillschedule.data.wan22_inputs import encode_export_input
    immutable = ['teacher','media'] if training else ['media']
    if training:
        for role in ('student','fake'):
            immutable += [role+'.'+name+'.base' for name,module in state[role].named_modules()
                          if isinstance(module,TrainableLowRankLinear)]
    metadata = dict(schema_version='distillschedule.native_wan22_t2v.v1',family=FAMILY,
        source=dict(model_id=MODEL_ID,revision=REVISION,license='Apache-2.0'),**deepcopy(descriptor),
        precision='FP32_CPU1_B1',prediction='v=epsilon-x0; z=(1-sigma)*x0+sigma*epsilon; x0=z-sigma*v; NO_NEGATION',
        algorithm='COMMON_FLOW_DMD_NOT_OFFICIAL_DMD2_OR_SELF_FORCING',generation='CURRENT_STUDENT_NOT_BASE_TEACHER_OR_EMA',
        gpu_execution='NOT_RUN',pretrained_quality='NOT_RUN',quality_validation='NOT_RUN')
    return ModelComponents(state=state,immutable_modules=tuple(immutable),metadata=metadata,
        prediction=PredictionSpec('flow'),capabilities=('video','flow','lora','dmd_flow_v1'),
        generate=generate,decode=state['media'].decode,inference_projection=project_current_generator if training else None,
        encode_input=None if training else encode_export_input,
        admit_training=admit_training if training else None).validate()


def build_model(parameters):
    from .dmd import sigmas,training_state
    safe.cpu1()
    allowed = {'assets','device','seed','trainable','student_lr','fake_lr','scheduler_gamma','ema_decay','generation_sigmas'}
    require(isinstance(parameters,dict) and set(parameters) <= allowed and allowed-{'device','seed'} <= parameters.keys(),
            'complete explicit Wan2.2 model parameters required; variant/backend/precision overrides refused')
    require(parameters.get('device','cpu') == 'cpu' and type(parameters.get('seed',0)) is int, 'CPU1 FP32 and integer construction seed required')
    require(isinstance(parameters['trainable'],dict) and parameters['trainable'].get('mode') == 'lora', 'initial native Wan recipe is exact unmerged LoRA only')
    schedule = sigmas(parameters['generation_sigmas'])
    require(len(schedule) == 5, 'initial native Wan recipe requires four Euler steps')
    media,manifest,roots,runtime,binding,media_config,empty,documents = load_media_bundle(parameters['assets'])
    with torch.device('cpu'),torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(parameters.get('seed',0))
        roles = {}
        for name in ('student','fake','teacher'):
            spec = manifest['roles'][name]
            native,config = safe.load_native(classes()[0],roots[name],spec['max_parameters'],dtype_policy=spec.get('dtype_policy','strict_fp32'))
            native.eval().requires_grad_(False) if name == 'teacher' else trainable_scope(native,parameters['trainable'])
            native.train(name != 'teacher')
            roles[name] = Role(native,config=config,media_config=media_config,binding=binding,empty=empty,
                trainable=parameters['trainable'],schedule=schedule)
            roles[name].train(name != 'teacher')
        state = training_state(roles['student'],roles['fake'],roles['teacher'],student_lr=parameters['student_lr'],
            fake_lr=parameters['fake_lr'],scheduler_gamma=parameters['scheduler_gamma'],ema_decay=parameters['ema_decay'])
    state['media'] = media
    descriptor = dict(architecture=config,media_config=media_config,binding=binding,runtime=runtime,
        trainable=deepcopy(parameters['trainable']),generation_sigmas=list(schedule),evidence_kind=manifest['evidence_kind'],
        cpu_parameter_limits=dict(student=manifest['roles']['student']['max_parameters'],
            vae=manifest['media']['vae']['max_parameters'],text_encoder=manifest['media']['text_encoder']['max_parameters']),
        asset_identity=dict(manifest_sha256=file_sha256(Path(parameters['assets'])),roles=deepcopy(manifest['roles']),media=deepcopy(manifest['media'])))
    for name,spec in {**manifest['roles'],**manifest['media']}.items():
        safe.asset_files(Path(parameters['assets']).resolve().parent,spec)
    require(safe.json_object(parameters['assets']) == manifest, 'asset manifest changed during load')
    return _components(state,descriptor,training=True)


def admit_training(*,state,request,prepared_identity,prepared_manifest):
    safe.cpu1()
    require(request.trainer.get('device','cpu') == 'cpu' and 'distributed' not in request.trainer, 'only CPU1 FP32 admitted')
    require(request.algorithm['factory'] == 'distillschedule.training.dmd:build_algorithm'
            and request.execution.get('adapter','program') == 'program', 'use existing common flow DMD Program, not Self-Forcing/DMD2')
    metadata = prepared_manifest.get('metadata',{})
    require('streams' not in request.data and metadata.get('schema_version') == INPUT_SCHEMA and metadata.get('family') == FAMILY
            and metadata.get('binding') == state['student'].binding, 'wrong native family/source/codec binding; stale features refused')
    policy = request.algorithm.get('parameters',{})
    require(policy.get('teacher_cfg') == policy.get('fake_cfg') == 1, 'first native Wan recipe requires teacher_cfg=fake_cfg=1')
    require(policy.get('fake_updates') == 5 and policy.get('rollout_sigmas') == list(state['student'].generation_sigmas),
            'first native Wan recipe requires G1/F5 and four explicit Euler steps')


def generate(state,feeds,parameters):
    from .dmd import sigmas
    require(isinstance(parameters,dict) and set(parameters) <= {'rollout_sigmas'}, 'unsupported native Wan inference option')
    role = state['student']
    schedule = sigmas(parameters.get('rollout_sigmas',role.generation_sigmas))
    require(tuple(schedule) == role.generation_sigmas, 'generation must retain the admitted four-step schedule')
    latent = role.rollout(feeds,schedule,'all_steps')['video']
    return {'video_latents': latent,'latent_schema': {'video_latents': dict(layout='BCFHW',shape=list(latent.shape),dtype=str(latent.dtype))}}


_DESCRIPTOR = {'architecture','media_config','binding','runtime','trainable','generation_sigmas','evidence_kind','cpu_parameter_limits'}


def project_current_generator(components,effective_training_parameters):
    descriptor = {key:deepcopy(components.metadata[key]) for key in _DESCRIPTOR}
    state = {name:components.state[name] for name in ('student','media')}
    return InferenceProjection(components=_components(state,descriptor,training=False),
        factory=ENTRY+':build_inference_model',parameters=descriptor)


def build_inference_model(parameters):
    from .dmd import sigmas
    safe.cpu1()
    require(isinstance(parameters,dict) and _DESCRIPTOR <= parameters.keys() and set(parameters) <= _DESCRIPTOR|{'device','seed'}
            and parameters.get('device','cpu') == 'cpu' and type(parameters.get('seed',0)) is int,
            'export reconstructs CURRENT Student and codecs only; training paths/backend overrides rejected')
    descriptor = {key:deepcopy(parameters[key]) for key in _DESCRIPTOR}
    require(descriptor['runtime'] == runtime_identity(), 'export native source/runtime/cleaner changed')
    config,media_config = descriptor['architecture'],descriptor['media_config']
    validate_architecture(config,media_config)
    require(descriptor['trainable'].get('mode') == 'lora' and len(sigmas(descriptor['generation_sigmas'])) == 5,
            'export requires original unmerged LoRA/four-step recipe')
    native_cls,vae_cls,text_cls,_ = classes()
    limits = descriptor['cpu_parameter_limits']
    require(isinstance(limits,dict) and set(limits) == {'student','vae','text_encoder'}
            and all(type(n) is int and n > 0 for n in limits.values()), 'portable export requires explicit native CPU parameter limits')
    for name,cls,cfg in (('student',native_cls,config),('vae',vae_cls,vae_config(media_config['vae'])),
                         ('text_encoder',text_cls,media_config['text_encoder'])):
        require(0 < sum(p.numel() for p in safe.shell(cls,cfg).parameters()) <= limits[name],
                'portable native model exceeds explicit CPU parameter budget: '+name)
    def layout(cls,cfg):
        cfg = safe.architecture(cls,vae_config(cfg) if cls is vae_cls else cfg)
        return (cls(cfg) if hasattr(cls,'config_class') else cls(**cfg)).float()
    # Concrete layouts only: common restore_modules validates/replaces EVERY
    # named tensor before output. These constructors are not an asset fallback.
    with torch.device('cpu'),torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(parameters.get('seed',0))
        native = layout(native_cls,config); trainable_scope(native,descriptor['trainable'])
        media = Media(layout(vae_cls,media_config['vae']),layout(text_cls,media_config['text_encoder']),
            restore_tokenizer(media_config['tokenizer']),config=media_config,binding=descriptor['binding'])
        empty = (torch.empty(1,226,config['text_dim']),torch.empty(1,226,dtype=torch.bool))
        role = Role(native,config=config,media_config=media_config,binding=descriptor['binding'],empty=empty,
            trainable=descriptor['trainable'],schedule=descriptor['generation_sigmas'])
    return _components({'student':role,'media':media},descriptor,training=False)
