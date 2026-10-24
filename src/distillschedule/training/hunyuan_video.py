"""Pinned HunyuanVideo1.5 T2V through the existing common flow-DMD owner.

FP32 local B1, single-device or explicit post-backward replicas; four Euler steps,
G1/F5 and unmerged same-family LoRA only.
Native Qwen text/crop108 + ByT5 glyphs, temporal4 VAE and native Transformer.
No pretrained/random fallback, I2V, meanflow or distilled-model admission.
Tencent model-asset terms are separate from the Apache-licensed implementation.
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
from .single_device import admit_fp32, construction_rng, place_module, resident_flow_state
from .single_device import FLOW_CUDA_EXECUTION, FLOW_REPLICATED_EXECUTION, execution_policy_matches
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint

FAMILY = 'hunyuan15-t2v'
MODEL_ID = 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v'
REVISION = '286be7ce72277246578a3e3cc2487e95ddae5bcf'
ENTRY = 'distillschedule.training.hunyuan_video'
ASSET_SCHEMA = 'distillschedule.hunyuan15_t2v_assets.v1'
INPUT_SCHEMA = 'distillschedule.hunyuan15_t2v_inputs.v1'
PIPELINE = dict(_class_name='HunyuanVideo15Pipeline', task='t2v', use_meanflow=False, distilled=False)
TEXT_POLICY = dict(primary_max_length=1000, crop_start=108, hidden_layer=-3,
    secondary_max_length=256, image_tokens=729)
CODEC_POLICY = 'NATIVE_HUNYUAN15_TEMPORAL4_VAE_MODE_SCALE; NATIVE_QWEN_TEXT_CROP108_HIDDEN_MINUS3; NATIVE_BYT5_GLYPH_OR_ZERO_SENTINEL'
LICENSE = 'Tencent Hunyuan Community License (model assets; not Apache-2.0)'


def classes():
    from diffusers import HunyuanVideo15Transformer3DModel, AutoencoderKLHunyuanVideo15, HunyuanVideo15Pipeline
    from transformers import Qwen2_5_VLTextModel, T5EncoderModel
    return HunyuanVideo15Transformer3DModel, AutoencoderKLHunyuanVideo15, Qwen2_5_VLTextModel, T5EncoderModel, HunyuanVideo15Pipeline


def validate_architecture(config, media):
    native, vae, text, text2, _ = classes()
    safe.architecture(native, config)
    safe.architecture(vae, media['vae'])
    primary = safe.architecture(text, media['text_encoder'])
    secondary = safe.architecture(text2, media['text_encoder_2'])
    v = media['vae']
    require(type(config.get('out_channels')) is int and config['out_channels'] > 0,
            'explicit positive native output/VAE channels required')
    require(config.get('task_type') == 't2v' and config.get('use_meanflow') is False
            and config.get('out_channels') == v.get('latent_channels')
            and config.get('in_channels') == 2*config['out_channels']+1
            and config.get('patch_size') == config.get('patch_size_t') == 1
            and config.get('text_embed_dim') == primary.hidden_size
            and config.get('text_embed_2_dim') == secondary.d_model,
            'only actual HunyuanVideo1.5 same-domain T2V native dual-text architecture admitted; I2V/meanflow rejected')
    axes = config.get('rope_axes_dim')
    require(isinstance(axes, (list,tuple)) and len(axes) == 3
            and all(type(n) is int and n > 0 and n % 2 == 0 for n in axes)
            and sum(axes) == config.get('attention_head_dim')
            and type(config.get('image_embed_dim')) is int and config['image_embed_dim'] > 0
            and type(config.get('target_size')) is int and config['target_size'] > 0,
            'actual native three-axis RoPE, image embedding width and target geometry required')
    require(v.get('temporal_compression_ratio') == 4 and v.get('spatial_compression_ratio') in (4,8,16)
            and len(v.get('block_out_channels', [])) == {4:3,8:4,16:5}[v['spatial_compression_ratio']]
            and v.get('in_channels') == v.get('out_channels') == 3 and v.get('scaling_factor') == 1.03682
            and media.get('text_policy') == TEXT_POLICY and primary.num_hidden_layers >= 3,
            'native temporal4 RGB VAE/scaling and exact crop108/hidden-3/1000+256 text policy required')
    # Native Qwen text permits padding_idx=None. The actual tokenizer owns pad
    # IDs and the native pipeline passes its explicit attention mask; setting a
    # model pad ID here would alter embedding/gradient semantics.
    require((primary.pad_token_id is None or type(primary.pad_token_id) is int and 0 <= primary.pad_token_id < primary.vocab_size)
            and all(type(n) is int and 0 <= n < primary.vocab_size for n in
                (primary.bos_token_id, primary.eos_token_id)),
            'valid native primary text special-token IDs required')


def runtime_identity():
    from transformers import Qwen2TokenizerFast, ByT5Tokenizer, tokenization_python, tokenization_utils_base
    result = safe.runtime_identity(sys.modules[__name__])
    for name, version in (('tokenizers','0.22.2'), ('sentencepiece','0.2.2')):
        require(importlib.metadata.version(name) == version, 'native dual-tokenizer dependency pin changed: '+name)
        result['versions'][name] = version
    result['source_files'].update({
        'feature_encoder': file_sha256(Path(__file__).parent.parent/'data/hunyuan_video_inputs.py'),
        'shared_video_input_identity': file_sha256(Path(__file__).parent.parent/'data/wan22_inputs.py'),
        'shared_tensor_validator': file_sha256(Path(__file__).parent.parent/'data/flow_image_inputs.py'),
        'raw_media_decoder': file_sha256(Path(__file__).parent.parent/'data/decoding.py'),
        'source_identity': file_sha256(Path(__file__).parent.parent/'data/manifest.py'),
        'qwen_tokenizer': file_sha256(Path(inspect.getfile(Qwen2TokenizerFast))),
        'byt5_tokenizer': file_sha256(Path(inspect.getfile(ByT5Tokenizer))),
        'native_slow_tokenizer': file_sha256(Path(tokenization_python.__file__)),
        'native_tokenizer_base': file_sha256(Path(tokenization_utils_base.__file__))})
    return result


def tokenizer_state(tokenizer, *, secondary=False):
    """ByT5 is the native slow byte tokenizer, not a fast/Qwen backend."""
    from transformers import Qwen2TokenizerFast, ByT5Tokenizer
    cls = ByT5Tokenizer if secondary else Qwen2TokenizerFast
    require(type(tokenizer) is cls and tokenizer.padding_side == tokenizer.truncation_side == 'right'
            and tokenizer.backend == ('custom' if secondary else 'tokenizers'),
            'actual native Qwen-fast/ByT5-slow tokenizer and right padding/truncation required')
    require(type(tokenizer.pad_token_id) is int and tokenizer.pad_token_id >= 0
            and tokenizer.pad_token_id in tokenizer.get_vocab().values(),
            'actual native tokenizer must own a valid padding token; model pad_token_id is not a substitute')
    options = {key: deepcopy(getattr(tokenizer,key)) for key in ('model_max_length','padding_side','truncation_side',
        'pad_token','bos_token','eos_token','unk_token','chat_template','clean_up_tokenization_spaces',
        'split_special_tokens','model_input_names')}
    options['additional_special_tokens'] = [str(x) for x in tokenizer._extra_special_tokens]
    if secondary:
        # ByT5's native constructor stores extra_ids=0 and preserves actual
        # sentinels through its explicit added-token map, including token flags.
        options['extra_ids'] = 0
        value = dict(options=options, added_tokens={str(k):v.__getstate__() for k,v in tokenizer.added_tokens_decoder.items()})
    else:
        require(isinstance(tokenizer.chat_template,str) and tokenizer.chat_template, 'actual primary native chat template required')
        options['add_prefix_space'] = tokenizer.add_prefix_space
        value = dict(options=options, json=tokenizer.backend_tokenizer.to_str())
    return dict(value, **{'class':cls.__module__+'.'+cls.__name__})


def restore_tokenizer(value, *, secondary=False):
    from tokenizers import Tokenizer, AddedToken
    from transformers import Qwen2TokenizerFast, ByT5Tokenizer
    cls = ByT5Tokenizer if secondary else Qwen2TokenizerFast
    require(isinstance(value,dict) and set(value) == {'class','options', 'added_tokens' if secondary else 'json'}
            and value['class'] == cls.__module__+'.'+cls.__name__, 'exact native tokenizer class/backend/state required')
    if secondary:
        require(isinstance(value['added_tokens'],dict), 'native ByT5 added-token state required')
        tokenizer = cls(**value['options'], added_tokens_decoder={int(k):AddedToken(**v) for k,v in value['added_tokens'].items()})
    else:
        tokenizer = cls(tokenizer_object=Tokenizer.from_str(value['json']), **value['options'])
    require(tokenizer_state(tokenizer,secondary=secondary) == value, 'native tokenizer options/added-token state did not restore exactly')
    return tokenizer


class Media(nn.Module):
    def __init__(self, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, *, config, binding):
        super().__init__()
        self.vae = vae.eval().requires_grad_(False)
        self.text_encoder = text_encoder.eval().requires_grad_(False)
        self.text_encoder_2 = text_encoder_2.eval().requires_grad_(False)
        self.tokenizer, self.tokenizer_2 = tokenizer, tokenizer_2
        self.architecture = deepcopy(config)
        self.binding = binding

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_text(self, prompts):
        from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5 import format_text_input, extract_glyph_texts
        require(isinstance(prompts,list) and prompts and all(isinstance(p,str) for p in prompts), 'explicit native prompt strings required')
        pipeline = classes()[4]
        method = pipeline._get_mllm_prompt_embeds
        system = inspect.signature(method).parameters['system_message'].default
        tokens = self.tokenizer.apply_chat_template(format_text_input(prompts,system), add_generation_prompt=True,
            tokenize=True, return_dict=True, padding=False, truncation=False)
        require(all(108 < len(ids) <= 1108 for ids in tokens['input_ids']), 'native primary crop108/1000 overflow or empty context; truncation refused')
        for prompt in prompts:
            glyph = extract_glyph_texts(prompt)
            if glyph is not None:
                require(0 < len(self.tokenizer_2(glyph, truncation=False, add_special_tokens=True)['input_ids']) <= 256,
                        'native quoted-glyph256 overflow; truncation refused')
        with torch.no_grad():
            primary, mask = method(self.text_encoder,self.tokenizer,prompts,device=self.device)
            secondary, mask2 = pipeline._get_byt5_prompt_embeds(self.tokenizer_2,self.text_encoder_2,prompts,device=self.device)
        require(primary.shape == (len(prompts),1000,self.text_encoder.config.hidden_size)
                and secondary.shape == (len(prompts),256,self.text_encoder_2.config.d_model)
                and bool(mask.bool().any(dim=1).all()) and bool(torch.isfinite(primary).all())
                and bool(torch.isfinite(secondary).all()), 'invalid native dual-text outputs; primary empty mask is not a glyph sentinel')
        return primary.detach().float(), mask.bool(), secondary.detach().float(), mask2.bool()

    def encode_video(self, frames):
        require(type(frames) is torch.Tensor and frames.dtype == torch.uint8 and frames.device.type == 'cpu'
                and frames.ndim == 4 and frames.shape[-1] == 3 and min(frames.shape) > 0,
                'actual raw RGB THWC uint8 video required')
        f,h,w,_ = frames.shape
        spatial = self.vae.config.spatial_compression_ratio
        require((f-1) % 4 == 0 and h % spatial == w % spatial == 0,
                'native raw geometry needs F=(Flatent-1)*4+1 and divisible spatial compression; no crop/resize/resample')
        pixels = (frames.permute(3,0,1,2).unsqueeze(0).float()/127.5-1).to(self.device)
        with torch.no_grad():
            latent = self.vae.encode(pixels).latent_dist.mode()*self.vae.config.scaling_factor
        require(list(latent.shape) == [1,self.vae.config.latent_channels,(f-1)//4+1,h//spatial,w//spatial],
                'native temporal4 VAE returned incompatible geometry')
        return latent.detach()

    def decode_latents(self, latent):
        return self.vae.decode(latent/self.vae.config.scaling_factor,return_dict=False)[0]

    def decode(self, generated, parameters):
        require(isinstance(parameters,dict) and set(parameters) <= {'rollout_sigmas'}, 'unsupported native video decoding option')
        latent = generated['video_latents']
        require(latent.ndim == 5 and latent.shape[:2] == (1,self.vae.config.latent_channels), 'native decoder requires B1/CFHW latents')
        with torch.no_grad():
            pixels = self.decode_latents(latent)
        spatial = self.vae.config.spatial_compression_ratio
        require(pixels.shape == (1,3,(latent.shape[2]-1)*4+1,latent.shape[3]*spatial,latent.shape[4]*spatial)
                and bool(torch.isfinite(pixels).all()), 'native temporal4 VAE returned invalid RGB video')
        return {'video': ((pixels[0].permute(1,2,3,0).clamp(-1,1)+1)*127.5).round().to(torch.uint8).contiguous()}


class Role(safe.QualifiedNativeFlowTrials, nn.Module):
    def __init__(self, native, *, config, media_config, binding, empty, trainable, schedule):
        super().__init__()
        self.native = native
        self.architecture, self.media_config = deepcopy(config), deepcopy(media_config)
        self.binding, self.trainable_config = binding, deepcopy(trainable)
        self.generation_sigmas = tuple(schedule)
        for name,value in zip(('empty_context','empty_context_mask','empty_context_2','empty_context_mask_2'),empty):
            self.register_buffer(name,value.detach().clone())

    @property
    def device(self):
        return next(self.native.parameters()).device

    def __deepcopy__(self, memo):
        require(self.device.type != 'cuda', safe.CUDA_TRIAL_LIMIT)
        return super().__deepcopy__(memo)

    def validate_feeds(self, feeds):
        from distillschedule.data.hunyuan_video_inputs import validate_row
        return validate_row(feeds,config=self.architecture,media_config=self.media_config,binding=self.binding,
            empty=(self.empty_context,self.empty_context_mask,self.empty_context_2,self.empty_context_mask_2), device=self.device)

    def sample_template(self, feeds):
        return feeds['latents']

    def loss_masks(self, feeds, sample):
        return {'video':torch.ones_like(sample['video'],dtype=torch.bool)}

    def apply_conditioning(self, sample, feeds):
        require(isinstance(sample,dict) and set(sample) == {'video'}
                and sample['video'].shape == feeds['latents']['video'].shape,
                'only native T2V payload admitted; I2V/reference image/causal conditions refused')
        return sample

    def denoise(self, sample, sigma, feeds, *, conditional):
        admit_fp32({'device': str(self.device)}, require=require, allow_replicated=True)
        self.validate_feeds(feeds)
        self.apply_conditioning(sample,feeds)
        value = sample['video']
        require(type(value) is torch.Tensor and value.dtype == torch.float32 and value.device == self.device
                and bool(torch.isfinite(value).all()), 'finite native FP32 video on the owned device required')
        require(type(sigma) is torch.Tensor and sigma.dtype == torch.float32 and sigma.device == self.device
                and sigma.shape == (1,) and bool(torch.isfinite(sigma).all())
                and bool(((sigma>=0)&(sigma<=1)).all()), 'native normalized per-batch sigma required')
        prefix = '' if conditional else 'null_'
        # Actual T2V native image-input channels and semantic tokens are zeros;
        # they are not an invented reference image or learned condition.
        output = self.native(hidden_states=torch.cat((value,torch.zeros_like(value),torch.zeros_like(value[:,:1])),dim=1),
            timestep=sigma*1000, image_embeds=value.new_zeros((1,729,self.architecture['image_embed_dim'])),
            encoder_hidden_states=feeds[prefix+'context'], encoder_attention_mask=feeds[prefix+'context_mask'],
            encoder_hidden_states_2=feeds[prefix+'context_2'], encoder_attention_mask_2=feeds[prefix+'context_mask_2'],
            return_dict=False)[0]
        require(output.shape == value.shape and bool(torch.isfinite(output).all()), 'native Hunyuan velocity shape/values changed')
        return {'video':output}

    def rollout(self, feeds, rollout_sigmas, gradient_policy):
        from .dmd import sigmas
        self.validate_feeds(feeds)
        schedule = sigmas(rollout_sigmas)
        require(gradient_policy in ('all_steps','last_step'), 'explicit Euler gradient policy required')
        value = {'video':torch.randn_like(feeds['latents']['video'])}
        for index,(sigma,following) in enumerate(zip(schedule,schedule[1:])):
            context = torch.no_grad() if gradient_policy == 'last_step' and index < len(schedule)-2 else nullcontext()
            with context:
                velocity = self.denoise(value,value['video'].new_full((1,),sigma),feeds,conditional=True)['video']
                value = {'video':value['video']+(following-sigma)*velocity}
        return value


def _budget(layouts, limits, *, portable=False):
    require(isinstance(limits,dict) and set(limits) == {name for name,_,_ in layouts}
            and all(type(n) is int and n > 0 for n in limits.values()), 'explicit native CPU parameter limits required')
    for name,cls,config in layouts:
        count = sum(p.numel() for p in safe.shell(cls,config).parameters())
        require(0 < count <= limits[name], ('portable ' if portable else '')+'native model exceeds explicit CPU parameter budget: '+name)


def load_media_bundle(assets, *, device=None):
    from transformers import Qwen2TokenizerFast, ByT5Tokenizer
    if device is None:
        safe.cpu1()
    else:
        admit_fp32({'device': str(device)}, require=require, allow_replicated=True)
    runtime = runtime_identity()
    path = Path(assets).expanduser().resolve(strict=True)
    manifest = safe.json_object(path)
    require(set(manifest) == {'schema_version','family','source','evidence_kind','pipeline','roles','media','text_policy'}
            and manifest['schema_version'] == ASSET_SCHEMA and manifest['family'] == FAMILY
            and manifest['source'] == dict(model_id=MODEL_ID,revision=REVISION) and manifest['pipeline'] == PIPELINE
            and manifest['text_policy'] == TEXT_POLICY
            and manifest['evidence_kind'] in ('SYNTHETIC_TEST_ONLY','LOCAL_PRETRAINED_UNVERIFIED'),
            'exact HunyuanVideo1.5 T2V source/pipeline/schema required; other variants rejected')
    require(isinstance(manifest['roles'],dict) and set(manifest['roles']) == {'student','fake','teacher'}
            and isinstance(manifest['media'],dict) and set(manifest['media']) == {'vae','text_encoder','text_encoder_2','tokenizer','tokenizer_2'},
            'all actual native roles and both text encoders/tokenizers plus VAE required')
    roots,documents = {},[path]
    for name,spec in {**manifest['roles'],**manifest['media']}.items():
        roots[name],files = safe.asset_files(path.parent,spec); documents.extend(files)
    media_config = {name:safe.json_object(roots[name]/'config.json') for name in ('vae','text_encoder','text_encoder_2')}
    media_config['text_policy'] = deepcopy(TEXT_POLICY)
    configs = {name:safe.json_object(roots[name]/'config.json') for name in ('student','fake','teacher')}
    require(all(config == configs['student'] for config in configs.values()), 'same-family native role architectures must match before loading')
    validate_architecture(configs['student'],media_config)
    native,vae,text,text2,_ = classes()
    layouts = [(name,native,config) for name,config in configs.items()]
    layouts += [('vae',vae,media_config['vae']),('text_encoder',text,media_config['text_encoder']),('text_encoder_2',text2,media_config['text_encoder_2'])]
    # Size EVERY native role/codec on meta before ANY concrete model/weight load.
    _budget(layouts,{name:({**manifest['roles'],**manifest['media']})[name]['max_parameters'] for name,_,_ in layouts})
    binding = fingerprint(dict(family=FAMILY,source=manifest['source'],pipeline=PIPELINE,media=manifest['media'],
        text_policy=TEXT_POLICY,runtime=runtime,dtype='FP32_CPU1_B1',codec_policy=CODEC_POLICY,
        transformer_config_sha256=file_sha256(roots['student']/'config.json')))
    with torch.device('cpu'),torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        def load(name,cls):
            spec = manifest['media'][name]
            return safe.load_native(cls,roots[name],spec['max_parameters'],dtype_policy=spec.get('dtype_policy','strict_fp32'))[0]
        tokenizer = Qwen2TokenizerFast.from_pretrained(roots['tokenizer'],local_files_only=True)
        tokenizer2 = ByT5Tokenizer.from_pretrained(roots['tokenizer_2'],local_files_only=True)
        media_config['tokenizer'] = tokenizer_state(tokenizer)
        media_config['tokenizer_2'] = tokenizer_state(tokenizer2,secondary=True)
        for tok,cfg in ((tokenizer,media_config['text_encoder']),(tokenizer2,media_config['text_encoder_2'])):
            require(max(tok.get_vocab().values()) < cfg['vocab_size'], 'native tokenizer IDs exceed source-bound text vocabulary')
        media = Media(load('vae',vae),load('text_encoder',text),load('text_encoder_2',text2),tokenizer,tokenizer2,
            config=media_config,binding=binding).eval().requires_grad_(False)
        empty = media.encode_text([''])
    return media,manifest,roots,runtime,binding,media_config,empty,documents


def _components(state, descriptor, *, training):
    from distillschedule.data.hunyuan_video_inputs import encode_export_input
    from .distributed import rank_world
    safe.qualify_trial_roles(state, descriptor['evidence_kind'])
    immutable = ['teacher','media'] if training else ['media']
    if training:
        for role in ('student','fake'):
            immutable += [role+'.'+name+'.base' for name,module in state[role].named_modules() if isinstance(module,TrainableLowRankLinear)]
    # Single-process and portable-inference identities remain unchanged.
    replica_caps = ('dmd_flow_private_replicated_b1_v1',) if training and rank_world()[1] > 1 else ()
    metadata = dict(schema_version='distillschedule.native_hunyuan15_t2v.v1',family=FAMILY,
        source=dict(model_id=MODEL_ID,revision=REVISION,license=LICENSE),**deepcopy(descriptor),
        **(dict(precision='FP32_NATIVE_RANK_LOCAL_REPLICATED', replication='EXPLICIT_EQUAL_RANK_LOCAL_B1_POST_BACKWARD_MEAN')
           if replica_caps else dict(precision='FP32_NATIVE_SINGLE_DEVICE_B1')),
        prediction='v=epsilon-x0; z=(1-sigma)*x0+sigma*epsilon; x0=z-sigma*v; NO_NEGATION',
        algorithm='COMMON_FLOW_DMD_NOT_OFFICIAL_DMD2_OR_SELF_FORCING',generation='CURRENT_STUDENT_NOT_BASE_TEACHER_OR_EMA',
        gpu_execution='NOT_RUN',pretrained_quality='NOT_RUN',quality_validation='NOT_RUN')
    return ModelComponents(state=state,immutable_modules=tuple(immutable),metadata=metadata,prediction=PredictionSpec('flow'),
        capabilities=('video','flow','lora','dmd_flow_v1',*replica_caps),generate=generate,decode=state['media'].decode,
        encode_input=None if training else encode_export_input,
        inference_projection=project_current_generator if training else None,admit_training=admit_training if training else None).validate()


def build_model(parameters):
    from .dmd import sigmas, training_state
    allowed = {'assets','device','seed','trainable','student_lr','fake_lr','scheduler_gamma','ema_decay','generation_sigmas'}
    require(isinstance(parameters,dict) and set(parameters) <= allowed and allowed-{'device','seed'} <= parameters.keys(),
            'complete explicit HunyuanVideo1.5 parameters required; variant/backend/precision overrides refused')
    device = admit_fp32(parameters, require=require, allow_replicated=True)
    require(type(parameters.get('seed',0)) is int, 'integer construction seed required')
    require(isinstance(parameters['trainable'],dict) and parameters['trainable'].get('mode') == 'lora', 'initial native Hunyuan recipe is unmerged LoRA only')
    schedule = sigmas(parameters['generation_sigmas'])
    require(len(schedule) == 5, 'initial native Hunyuan recipe requires four Euler steps')
    media,manifest,roots,runtime,binding,media_config,empty,_ = load_media_bundle(parameters['assets'],device=device)
    with construction_rng(device, parameters.get('seed',0)), torch.device('cpu'):
        # Preserve exact CPU-prepared null contexts; CUDA encoder arithmetic
        # is not a replacement for the source-bound empty-prompt features.
        empty = tuple(tensor.to(device) for tensor in empty)
        place_module(media, device, require=require)
        roles = {}
        for name in ('student','fake','teacher'):
            spec = manifest['roles'][name]
            native,config = safe.load_native(classes()[0],roots[name],spec['max_parameters'],dtype_policy=spec.get('dtype_policy','strict_fp32'))
            native.eval().requires_grad_(False) if name == 'teacher' else trainable_scope(native,parameters['trainable'])
            roles[name] = Role(native,config=config,media_config=media_config,binding=binding,empty=empty,
                trainable=parameters['trainable'],schedule=schedule)
            place_module(roles[name], device, require=require)
            roles[name].train(name != 'teacher')
        with torch.device(device):
            state = training_state(roles['student'],roles['fake'],roles['teacher'],student_lr=parameters['student_lr'],
                fake_lr=parameters['fake_lr'],scheduler_gamma=parameters['scheduler_gamma'],ema_decay=parameters['ema_decay'])
        resident_flow_state(state, device)
    state['media'] = media
    descriptor = dict(architecture=config,media_config=media_config,binding=binding,runtime=runtime,
        trainable=deepcopy(parameters['trainable']),generation_sigmas=list(schedule),evidence_kind=manifest['evidence_kind'],
        cpu_parameter_limits={name:(manifest['roles'] if name == 'student' else manifest['media'])[name]['max_parameters']
            for name in ('student','vae','text_encoder','text_encoder_2')},
        asset_identity=dict(manifest_sha256=file_sha256(Path(parameters['assets'])),roles=deepcopy(manifest['roles']),media=deepcopy(manifest['media'])))
    for spec in {**manifest['roles'],**manifest['media']}.values():
        safe.asset_files(Path(parameters['assets']).resolve().parent,spec)
    require(safe.json_object(parameters['assets']) == manifest, 'asset manifest changed during load')
    return _components(state,descriptor,training=True)


def admit_training(*, state, request, prepared_identity, prepared_manifest):
    from .distributed import rank_world
    device = admit_fp32({'device':request.trainer.get('device','cpu')}, require=require, allow_replicated=True)
    require(all(state[name].device == device for name in ('student','fake','teacher','media')),
            'matching rank-local FP32 device ownership required')
    if rank_world()[1] > 1:
        require(request.trainer.get('distributed',{}).get('backend') == torch.distributed.get_backend()
                and request.data.get('sharding') == 'rank_strided',
                'explicit distributed backend and rank_strided data required')
        require(execution_policy_matches(request.execution, FLOW_REPLICATED_EXECUTION),
                'replicated private flow requires reference or exact manual post_backward immutable sharing; trials unsupported')
    else:
        require('distributed' not in request.trainer, 'distributed execution requires actual torchrun ranks')
        require(device.type != 'cuda' or execution_policy_matches(request.execution, FLOW_CUDA_EXECUTION), safe.CUDA_TRIAL_LIMIT)
    require(request.algorithm['factory'] == 'distillschedule.training.dmd:build_algorithm'
            and request.execution.get('adapter','program') == 'program', 'use existing common flow DMD Program, not official Self-Forcing/DMD2')
    metadata = prepared_manifest.get('metadata',{})
    require('streams' not in request.data and metadata.get('schema_version') == INPUT_SCHEMA and metadata.get('family') == FAMILY
            and metadata.get('binding') == state['student'].binding, 'wrong native family/source/codec binding; stale features refused')
    policy = request.algorithm.get('parameters',{})
    require(policy.get('teacher_cfg') == policy.get('fake_cfg') == 1, 'first native Hunyuan recipe requires teacher_cfg=fake_cfg=1')
    require(policy.get('fake_updates') == 5 and policy.get('rollout_sigmas') == list(state['student'].generation_sigmas),
            'first native Hunyuan recipe requires G1/F5 and four explicit Euler steps')


def generate(state, feeds, parameters):
    from .dmd import sigmas
    require(isinstance(parameters,dict) and set(parameters) <= {'rollout_sigmas'}, 'unsupported native Hunyuan inference option')
    role = state['student']
    schedule = sigmas(parameters.get('rollout_sigmas',role.generation_sigmas))
    require(tuple(schedule) == role.generation_sigmas, 'generation must retain admitted four-step schedule')
    latent = role.rollout(feeds,schedule,'all_steps')['video']
    return {'video_latents':latent,'latent_schema':{'video_latents':dict(layout='BCFHW',shape=list(latent.shape),dtype=str(latent.dtype))}}


_DESCRIPTOR = {'architecture','media_config','binding','runtime','trainable','generation_sigmas','evidence_kind','cpu_parameter_limits'}


def project_current_generator(components, effective_training_parameters):
    descriptor = {key:deepcopy(components.metadata[key]) for key in _DESCRIPTOR}
    return InferenceProjection(components=_components({name:components.state[name] for name in ('student','media')},descriptor,training=False),
        factory=ENTRY+':build_inference_model',parameters=descriptor)


def build_inference_model(parameters):
    from .dmd import sigmas
    require(isinstance(parameters,dict) and _DESCRIPTOR <= parameters.keys() and set(parameters) <= _DESCRIPTOR|{'device','seed'}
            and type(parameters.get('seed',0)) is int,
            'export reconstructs CURRENT Student and native codecs only; training paths/backend overrides refused')
    device = admit_fp32(parameters, require=require)
    descriptor = {key:deepcopy(parameters[key]) for key in _DESCRIPTOR}
    require(descriptor['runtime'] == runtime_identity(), 'export native source/runtime/tokenizer changed')
    config,media_config = descriptor['architecture'],descriptor['media_config']
    validate_architecture(config,media_config)
    require(descriptor['trainable'].get('mode') == 'lora' and len(sigmas(descriptor['generation_sigmas'])) == 5,
            'export requires original unmerged LoRA/four-step recipe')
    native,vae,text,text2,_ = classes()
    layouts = [('student',native,config),('vae',vae,media_config['vae']),
        ('text_encoder',text,media_config['text_encoder']),('text_encoder_2',text2,media_config['text_encoder_2'])]
    _budget(layouts,descriptor['cpu_parameter_limits'],portable=True)
    tokenizer = restore_tokenizer(media_config['tokenizer'])
    tokenizer2 = restore_tokenizer(media_config['tokenizer_2'],secondary=True)
    def layout(cls,cfg):
        cfg = safe.architecture(cls,cfg)
        return (cls(cfg) if hasattr(cls,'config_class') else cls(**cfg)).float()
    # Concrete native layouts only. Common restore_modules must validate and
    # replace EVERY named tensor before output; no Teacher/Fake/base reload.
    with construction_rng(device, parameters.get('seed',0)), torch.device('cpu'):
        model = layout(native,config); trainable_scope(model,descriptor['trainable'])
        media = Media(layout(vae,media_config['vae']),layout(text,media_config['text_encoder']),layout(text2,media_config['text_encoder_2']),
            tokenizer,tokenizer2,config=media_config,binding=descriptor['binding'])
        place_module(media, device, require=require)
        empty = (torch.empty(1,1000,config['text_embed_dim'],device=device),torch.empty(1,1000,dtype=torch.bool,device=device),
            torch.empty(1,256,config['text_embed_2_dim'],device=device),torch.empty(1,256,dtype=torch.bool,device=device))
        role = Role(model,config=config,media_config=media_config,binding=descriptor['binding'],empty=empty,
            trainable=descriptor['trainable'],schedule=descriptor['generation_sigmas'])
        place_module(role, device, require=require)
    return _components({'student':role,'media':media},descriptor,training=False)
