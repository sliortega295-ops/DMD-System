"""Versioned native W1 CURRENT-G portable output, not a training/resume owner.

The existing world16 worker restores the complete checkpoint. This module
projects its live unmerged G and a pinned, local Wan2.1 VAE into the existing
named-module export format. Only the independent single-process target is
frozen/evaluated. No training assets or implicit model downloads are needed
after publication. CPU tiny/random evidence is never pretrained GPU evidence.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from .adapters import InferenceProjection, ModelComponents
from .helios_inference import _json as plain
from .helios_public_assets import EVIDENCE_KINDS, FIXTURE, checked_pin, read_json
from .wan21_codec import NORMALIZATION
from .wan_inference import _rollout
from distillschedule.workloads.best_known.registry import file_sha256, repository_root

SCHEMA = 'distillschedule.wan_native_generator_inference.v2'
EXPORT_SCHEMA = 'distillschedule.wan_native_output_request.v1'
NOISE_SCHEMA = 'distillschedule.wan_fresh_inference_noise.v1'
NOISE_PREFIX = 'native_inference/new_prompt'
FACTORY = 'distillschedule.training.wan_native_inference:build_inference_model'
PRECISION = 'WAN_NATIVE_MIXED_BF16_BASE_FP32_LORA_UNMERGED'
SCOPE = 'NATIVE_W1_CURRENT_G_SINGLE_PROCESS_DECODED_OUTPUT_NOT_QUALITY'
NATIVE = dict(args=dict(base_seq_len=256,max_seq_len=4096,base_shift=.5,max_shift=1.15),
              scheduler=dict(shift=1.,stages=3,stage_range=[0,1/3,2/3,1],gamma=1/3))
# The source-pinned original cls.from_config retains this raw Wan field in
# model.config, but the complete original __init__ does not consume it.
NON_CONSTRUCTOR_CONFIG_FIELDS = frozenset({'rope_max_seq_len'})


def require(ok, message):
    if not ok:
        raise ValueError('native W1 output: ' + message)


def noise_protocol():
    """Explicit new inference RNG ABI, not old Cholesky bitwise replay.

    gamma=1/3 has covariance (4/3)I-(1/3)11^T, with one ZERO
    eigenvalue. It is PSD, not Cholesky-PD. Use the already audited exact
    projection sampler through original operation inputs, not global helper
    mutation, a compatibility environment flag, jitter or iid replacement.
    """
    return dict(schema_version=NOISE_SCHEMA,operation_prefix=NOISE_PREFIX,gamma=1/3,
        tensor_draw_order=['entry_noise','rollout_noise','block_noise/1','block_noise/2'],
        seed_derivation='wan_capture.input_stream: SHA256 compact JSON [seed,schema_version,tensor_name]; first8 big-endian modulo 2**63',
        iid_distribution='CPU torch.randn full BCTHW BF16, one independent labelled stream per entry/rollout tensor',
        block_distribution='runtime.official_baseline.stable_block_noise: FP32 iid4, subtract per-block mean, multiply sqrt(4/3)',
        block_covariance='(4/3) * (I4 - ones(4,4)/4), rank3 PSD; no Cholesky or diagonal jitter',
        block_layout='native view(B,C,T,H/2,W/2,2,2), permute(0,1,2,3,5,4,6), reshape(B,C,T,H,W)',
        block_dtype='CPU FP32; unchanged native consumer casts to its current noisy_model_input dtype/device',
        exits='fixed last step of each configured original pyramid horizon; no exit RNG or training history',
        portability='noise sampled on CPU independently of inference device and training RNG',
        compatibility='NEW_INFERENCE_RNG_PROTOCOL_NOT_OLD_CHOLESKY_BITWISE_OR_TRAINING_CONTINUATION')


class _FreshNoiseInputs:
    """Ephemeral NEW operation inputs; no training trace/file/history reader."""
    def __init__(self, values, sequence):
        self._values = values
        self._sequence = sequence
        self._used = set()
        self._operation_used = False
        # The original entrypoint uses this argument for shape/dtype only,
        # then obtains the authoritative entry tensor via its operation seam.
        self.shape_noise = values[NOISE_PREFIX+'/entry_noise']

    def tensor(self, name):
        require(name in self._values and name not in self._used, 'unexpected or repeated fresh noise input: '+str(name))
        self._used.add(name)
        return self._values[name]

    def operation(self, name):
        require(name == NOISE_PREFIX+'/rollout' and not self._operation_used,
                'unexpected or repeated fresh inference operation: '+str(name))
        self._operation_used = True
        return dict(control=dict(pyramid_stage_call_sequence=deepcopy(self._sequence)))

    def finish(self):
        require(self._used == set(self._values) and self._operation_used,
                'original native forward did not consume the exact fresh inference inputs/control')


def fresh_noise_inputs(p):
    """Generate four explicit independent streams; preserve caller CPU/Python RNG."""
    from distillschedule.strategies.wan_task import OriginalWanTask
    require(p.get('noise_protocol') == noise_protocol(), 'fresh inference noise protocol changed')
    require(type(p.get('seed')) is int and 0 <= p['seed'] < 2**63, 'invalid fresh inference seed')
    task = OriginalWanTask.from_dict(p['task'])
    require(p['geometry']['noise_shape'] == list(task.latent_shape) and p['native']['rollout'] == _rollout(task),
            'fresh inference shape/one-section/last-step rollout contract changed')
    return _noise_inputs(task, p['seed'])


def _noise_inputs(task, seed):
    """Shared native geometry sampler; each public caller admits its own task."""
    from distillschedule.strategies.wan_capture import input_stream
    from runtime.official_baseline import stable_block_noise
    values = {}
    for name in ('entry_noise','rollout_noise'):
        with input_stream(seed,NOISE_SCHEMA,name):
            values[NOISE_PREFIX+'/'+name] = torch.randn(task.latent_shape,dtype=torch.bfloat16,device='cpu')
    b,c,t,_,_ = task.latent_shape
    for stage in (1,2):
        name = 'block_noise/'+str(stage);h,w = task.stage_spatial_shapes[stage]
        with input_stream(seed,NOISE_SCHEMA,name):
            noise = stable_block_noise(b*c*t*(h//2)*(w//2),device='cpu',dtype=torch.float32)
        values[NOISE_PREFIX+'/'+name] = noise.view(b,c,t,h//2,w//2,2,2).permute(
            0,1,2,3,5,4,6).reshape(b,c,t,h,w)
    sequence = [dict(stage=i,denoising_exit=horizon-1,call_count=horizon,
                     denoising_step_budget=horizon,latent_shape=[b,c,t,*task.stage_spatial_shapes[i]])
                for i,horizon in enumerate(task.stage2_inference_steps)]
    return _FreshNoiseInputs(values,sequence)


def inference_source():
    """Read-only closure identity, legal even while native helios.* is loaded.

    Do not load the source-only helper namespace in a live training worker:
    that namespace correctly refuses an already imported native kernel stack.
    The standalone forward imports that provider only after this byte check.
    """
    from distillschedule.adapters.helios import reference_model, reference_helpers
    source = reference_model.reference_model_identity(backend='torch-sdpa')
    source = {**source, 'source': {k:v for k,v in source['source'].items()
                                  if k not in ('transformer_path','manifest_path')},
              'backend': {k:v for k,v in source['backend'].items() if k != 'path'}}
    root = repository_root()
    files = ['src/distillschedule/training/wan_native_inference.py',
        'src/distillschedule/training/wan_inference.py',
        'src/distillschedule/training/helios_inference.py',
        'src/distillschedule/training/wan21_codec.py',
        'src/distillschedule/training/inference.py',
        'src/distillschedule/training/adapters.py',
        'src/distillschedule/training/checkpoint.py',
        'src/distillschedule/training/distributed.py',
        'src/distillschedule/training/session.py',
        'src/distillschedule/training/plugins.py',
        'src/distillschedule/open_workload/torch_backend.py',
        'src/distillschedule/strategies/wan_capture.py',
        'src/distillschedule/strategies/wan_task.py',
        'runtime/critic_pool_helios.py',
        'runtime/official_baseline.py',
        'third_party/helios_compat/source/helios/utils/utils_helios_post.py',
        'third_party/helios_compat/source/helios/utils/utils_helios_base.py',
        'third_party/helios_compat/source/helios/utils/utils_base.py',
        'third_party/helios_compat/source/helios/scheduler/scheduling_helios.py']
    return dict(model=source, files={name:file_sha256(root/name) for name in files},
                providers={name:file_sha256(Path(api.__file__)) for name,api in
                           (('model',reference_model),('helpers',reference_helpers))})


def runtime_identity():
    """Same dependency bytes on export and inference, without private paths."""
    result = {}
    for name in ('torch','diffusers','peft','transformers','omegaconf','safetensors'):
        api = importlib.import_module(name)
        result[name] = dict(version=importlib.metadata.version(name),
                            entry_sha256=file_sha256(Path(api.__file__).resolve()))
    return result


class CurrentGenerator(torch.nn.Module):
    """Pure view: stable output ABI despite native/FSDP runtime subclasses.

    Wrapping registers the existing module; it never changes its modes,
    trainability, adapters or parameters. The inference target uses this same
    container around the original source-only model for strict named loading.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model


def _codec_class():
    from diffusers import AutoencoderKLWan
    return AutoencoderKLWan


def _codec_runtime():
    from diffusers.configuration_utils import register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils.accelerate_utils import apply_forward_hook
    return dict(diffusers=importlib.metadata.version('diffusers'),
        implementation_sha256=file_sha256(Path(inspect.getfile(_codec_class()))),
        helpers={name:file_sha256(Path(inspect.getfile(api))) for name,api in
                 (('configuration',register_to_config),('model',ModelMixin),('decode_hook',apply_forward_hook))})


def codec_parameters(vae, *, evidence_kind, provenance):
    cls = _codec_class()
    require(type(vae) is cls, 'actual AutoencoderKLWan codec required')
    keys = set(inspect.signature(cls.__init__).parameters)-{'self'}
    architecture = plain({key:dict(vae.config)[key] for key in keys})
    p = _codec_parameters(architecture,str(vae.dtype),evidence_kind,provenance)
    require(vae.spatial_compression_ratio == 8 and 2**sum(architecture['temperal_downsample']) == 4,
            'Wan2.1 spatial8/temporal4 codec required')
    return p


def _codec_parameters(architecture, dtype, evidence_kind, provenance):
    cls = _codec_class()
    p = dict(architecture=architecture,dtype=dtype,evidence_kind=evidence_kind,
        provenance=plain(provenance),runtime=_codec_runtime(),
        normalization='normalized_latents * config.latents_std + config.latents_mean',
        latent_clamp=False,spatial_stride=8,temporal_stride=4)
    _admit_codec(p)
    return p


def _admit_codec(p):
    cls = _codec_class()
    require(isinstance(p,dict) and set(p) == {'architecture','dtype','evidence_kind','provenance','runtime',
        'normalization','latent_clamp','spatial_stride','temporal_stride'}, 'complete portable codec schema required')
    require(p['dtype'] == 'torch.float32' and p['evidence_kind'] in EVIDENCE_KINDS
        and p['runtime'] == _codec_runtime()
        and p['normalization'] == 'normalized_latents * config.latents_std + config.latents_mean'
        and p['latent_clamp'] is False and p['spatial_stride'] == 8 and p['temporal_stride'] == 4,
        'codec runtime/precision/normalization changed')
    provenance = p['provenance']
    require(isinstance(provenance,dict) and set(provenance) == {'status','origin','license'}
        and all(isinstance(v,str) and v.strip() for v in provenance.values())
        and provenance['status'] == ('RANDOM_NOT_PRETRAINED' if p['evidence_kind'] == FIXTURE
                                    else 'USER_SUPPLIED_NOT_QUALITY_VERIFIED'), 'explicit truthful codec provenance required')
    a = p['architecture']
    require(isinstance(a,dict) and set(a) == set(inspect.signature(cls.__init__).parameters)-{'self'},
            'complete actual codec constructor required')
    require(a['z_dim'] == 16 and a['dim_mult'] == [1,2,4,4]
        and a['temperal_downsample'] == [False,True,True] and a['attn_scales'] == [] and a['dropout'] == 0.
        and a['latents_mean'] == NORMALIZATION['mean'] and a['latents_std'] == NORMALIZATION['std']
        and a.get('is_residual',False) is False and a.get('in_channels',3) == a.get('out_channels',3) == 3
        and a.get('patch_size',None) in (None,1), 'Wan2.1 codec required; no residual Wan2.2/patch2/in12 substitute')
    require((p['evidence_kind'] == FIXTURE and type(a['base_dim']) is int and 2 <= a['base_dim'] <= 16
             and a['base_dim']%2 == 0 and a['num_res_blocks'] in (1,2))
            or (p['evidence_kind'] != FIXTURE and a['base_dim'] == 96 and a['num_res_blocks'] == 2),
            'production codec architecture or explicitly bounded random fixture required')
    return cls


def _budget(budget):
    limits = dict(max_parameters=4_000_000_000,max_loaded_bytes=64*2**30,
        max_export_bytes_per_rank=64*2**30,max_export_bytes_all_ranks=1024*2**30,
        max_activation_elements=2_000_000_000,max_attention_elements=2_000_000_000)
    require(isinstance(budget,dict) and set(budget) == set(limits)
            and all(type(budget[k]) is int and 0 < budget[k] <= v for k,v in limits.items()),
            'explicit generator/codec/transient/all-rank capacity budget required')


def _generator(p, *, meta, lora_rank=None, base_initializer=None):
    from peft import LoraConfig
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from runtime.critic_pool_helios import TRANSFORMER_ADDITIONAL_KWARGS, _all_linear_targets
    cls = load_reference_model_class(backend='torch-sdpa')
    a = p['architecture']
    rank = p['task']['lora_rank'] if lora_rank is None else lora_rank
    require(isinstance(a,dict) and set(a) == set(inspect.signature(cls.__init__).parameters)-{'self'},
            'complete original Generator constructor required')
    require(all(a[k] == v for k,v in TRANSFORMER_ADDITIONAL_KWARGS.items())
        and a['text_dim'] == 4096 and a['in_channels'] == a['out_channels'] == 16 and a['patch_size'] == [1,2,2]
        and not any(a[k] for k in ('is_use_gan','is_use_gan_final','is_use_gan_hooks'))
        and a['image_dim'] is None and a['added_kv_proj_dim'] is None
        and all(type(a[k]) is int and 0 < a[k] <= p['budget']['max_parameters']
                for k in ('num_attention_heads','attention_head_dim','ffn_dim','freq_dim','num_layers'))
        and a['num_layers'] <= 128 and len(a['rope_dim']) == 3
        and a['num_attention_heads'] <= 128 and a['attention_head_dim'] <= 256
        and a['ffn_dim'] <= 32768 and a['freq_dim'] <= 1024
        and all(type(n) is int and n > 0 and n%2 == 0 for n in a['rope_dim'])
        and sum(a['rope_dim']) == a['attention_head_dim'], 'original W1 architecture/conditioning changed')
    require(base_initializer is None or callable(base_initializer) and not meta,
            'base initializer is an internal CPU construction seam, never config')
    with torch.device('meta' if meta else 'cpu'):
        with torch.device('meta' if base_initializer is not None else ('meta' if meta else 'cpu')):
            g = cls(**a)
        targets = sorted(set(_all_linear_targets(g)+['patch_short','patch_mid','patch_long']))
        require(p['peft'] == dict(r=rank,lora_alpha=float(rank),
            lora_dropout=0.,init_lora_weights='gaussian',target_modules=targets,exclude_modules=['down','up']),
            'full original LoRA/pyramid targets changed; unmerged FP32 LoRA required')
        if base_initializer is None:
            for name,tensor in g.named_parameters():
                tensor.data = tensor.data.to(torch.float32 if any(k in name for k in cls._keep_in_fp32_modules) else torch.bfloat16)
        else:
            # The original pretrained loader starts FP32 meta and casts only
            # loaded keys. Missing pyramid patches must reset FP32, then BF16.
            base_initializer(g)
        g.requires_grad_(False)
        g.add_adapter(LoraConfig(**p['peft']))
        for name,tensor in g.named_parameters():
            if 'lora_' in name:
                tensor.data = tensor.data.float()
    return g


def _admit(parameters):
    from distillschedule.strategies.wan_task import OriginalWanTask
    from .checkpoint import rng_state, restore_rng
    require(isinstance(parameters,dict) and set(parameters) <= {'schema_version','precision','source','runtime','architecture',
        'peft','native','task','geometry','budget','seed','evidence_kind','codec','noise_protocol','device'}, 'unsupported portable native parameters')
    p = plain({k:v for k,v in parameters.items() if k != 'device'})
    require(set(p) == {'schema_version','precision','source','runtime','architecture','peft','native','task','geometry',
        'budget','seed','evidence_kind','codec','noise_protocol'} and p['schema_version'] == SCHEMA and p['precision'] == PRECISION,
        'explicit versioned native inference ABI required; CPU diagnostic v1 is separate')
    require(p['noise_protocol'] == noise_protocol(), 'explicit fresh inference noise protocol changed')
    require(p['source'] == inference_source() and p['runtime'] == runtime_identity(), 'inference source/runtime changed')
    require(type(p['seed']) is int and 0 <= p['seed'] < 2**63 and p['evidence_kind'] in EVIDENCE_KINDS,
            'explicit valid independent inference seed/evidence required')
    task = OriginalWanTask.from_dict(p['task'])
    geometry = p['geometry']; _budget(p['budget'])
    require(isinstance(geometry,dict) and set(geometry) == {'noise_shape','prompt_shape','video_shape','fps'}
        and geometry['noise_shape'] == list(task.latent_shape)
        and geometry['video_shape'] == [task.video_frames,task.video_height,task.video_width,3]
        and isinstance(geometry['prompt_shape'],list) and len(geometry['prompt_shape']) == 3
        and geometry['prompt_shape'][0] == 1 and geometry['prompt_shape'][2] == 4096
        and type(geometry['prompt_shape'][1]) is int and 0 < geometry['prompt_shape'][1] <= 4096
        and type(geometry['fps']) is int and 1 <= geometry['fps'] <= 120, 'native prompt/video geometry/fps changed')
    require(p['native'] == {**NATIVE,'rollout':_rollout(task)}, 'original three-stage native rollout contract changed')
    cls = _admit_codec(p['codec'])
    require(p['codec']['evidence_kind'] == p['evidence_kind'], 'generator and codec evidence kinds disagree')
    before = rng_state()
    try:
        g = _generator(p,meta=True)
        with torch.device('meta'):
            codec = cls(**p['codec']['architecture'])
        g_state,c_state = g.state_dict(),codec.state_dict()
        tensors = list(g_state.values())+list(c_state.values())
        tensors += [v for name,v in g.named_buffers() if name not in g_state]
        tensors += [v for name,v in codec.named_buffers() if name not in c_state]
        count = sum(t.numel() for t in tensors)
        size = sum(t.numel()*t.element_size() for t in tensors)
        # Constructor FP32 bases, mixed inventory, strict-loaded copy and
        # output staging all count. This is not an HBM performance estimate.
        require(count <= p['budget']['max_parameters'] and max(12*count,3*size) <= p['budget']['max_loaded_bytes'],
                'generator/codec construction and strict-load capacity exceeded')
        latent = math.prod(geometry['noise_shape'])
        h,w = geometry['noise_shape'][-2:]
        # The widest decoded feature is base_dim at full spatial resolution;
        # dim_mult=4 is at spatial /8, not at full RGB resolution. Reserve two
        # full-resolution features, without multiplying RGB channels twice.
        activation = max(8*latent, 2*math.prod(geometry['video_shape'][:-1])*p['codec']['architecture']['base_dim'])
        tokens = math.prod(geometry['noise_shape'][2:])//4
        attention = max(tokens*tokens*p['architecture']['num_attention_heads'], (h*w)**2)
        require(activation <= p['budget']['max_activation_elements'] and attention <= p['budget']['max_attention_elements']
            and max(12*count,3*size)+4*(activation+attention) <= p['budget']['max_loaded_bytes'],
            'native generation/decoder activation or attention capacity exceeded')
    finally:
        restore_rng(before)
    return p,g,codec,size


def _metadata(p):
    return dict(schema_version=SCHEMA,scope=SCOPE,precision=PRECISION,parameters=deepcopy(p),
        generator_selection='CURRENT_FULL_UNMERGED_NOT_EMA',evidence_kind=p['evidence_kind'],
        training_resume='NOT_SUPPORTED_PORTABLE_INFERENCE_ONLY',quality_validation='NOT_RUN',claimable_speedup=False)


def _environment(device):
    require(torch.get_default_dtype() == torch.float32 and int(os.environ.get('WORLD_SIZE','1')) == 1
        and (not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1),
        'portable native inference requires an independent single-process owner')
    require(os.environ.get('DISTILLSCHEDULE_STABLE_BLOCK_NOISE') != '1',
            'native block-noise sampling required; compatibility noise override refused')
    device = torch.device(device)
    require(device.type in ('cpu','cuda') and (device.type != 'cuda' or torch.cuda.is_available()),
            'explicit available CPU or single CUDA inference device required')
    return device


def build_inference_model(parameters):
    device = _environment(parameters.get('device','cpu'))
    p,_,_,_ = _admit(parameters)
    g = _generator(p,meta=False)
    codec = _codec_class()(**p['codec']['architecture'])
    g.requires_grad_(False).eval(); g.disable_gradient_checkpointing()
    codec.requires_grad_(False).eval()
    state = dict(generator=CurrentGenerator(g.to(device)),codec=codec.to(device),inference=p)
    def decode_output(generated, parameters):
        # Program decoder callbacks receive (generated, parameters), not state.
        # Strict loading updates these owned modules in place before decoding.
        require(not parameters, 'codec parameter overrides are not supported')
        return decode(state,generated)
    return ModelComponents(state=state,metadata=_metadata(p),generate=generate,decode=decode_output,
        admit_training=refuse_training).validate()


def refuse_training(*args,**kwargs):
    raise ValueError('native W1 portable inference weights are not a training checkpoint')


def generate(state, feeds, parameters):
    return _generate(state, feeds, parameters, prepare_noise=fresh_noise_inputs)


def _generate(state, feeds, parameters, *, prepare_noise):
    """Same native call/RNG transaction for admitted Wan and Helios tasks."""
    from distillschedule.strategies.wan_capture import native_primitives
    from .checkpoint import rng_state, restore_rng
    p,g = state['inference'],state['generator'].model
    device = _environment(next(g.parameters()).device)
    require(not parameters and isinstance(feeds,dict) and set(feeds) == {'prompt_embeds'},
            'exact NEW preencoded prompt required; no training history/cache/parameter overrides')
    prompt = feeds['prompt_embeds']
    require(isinstance(prompt,torch.Tensor) and list(prompt.shape) == p['geometry']['prompt_shape']
        and prompt.dtype == torch.bfloat16 and prompt.layout == torch.strided and bool(prompt.isfinite().all()),
        'NEW prompt shape/BF16/finiteness changed')
    require(not g.training and not g.gradient_checkpointing and all(not v.requires_grad for v in g.parameters())
        and all(not getattr(m,'merged',False) for m in g.modules()), 'separate frozen unmerged inference G required')
    native = native_primitives(source_only=True)
    # Seeding and checkpoint RNG validation both use the current CUDA device.
    # Own the whole transaction without changing the caller's device selection.
    with torch.cuda.device(device) if device.type == 'cuda' else nullcontext():
        before = rng_state(device=device)
        try:
            import random
            random.seed(p['seed']); torch.random.default_generator.manual_seed(p['seed'])
            if device.type == 'cuda':
                torch.cuda.manual_seed(p['seed'])
            with torch.no_grad():
                # Fresh labelled CPU draws enter the original public operation
                # seam. Native forward/scheduler/flow/history geometry are intact;
                # no helper binding or global compatibility flag is modified.
                inputs = prepare_noise(p)
                result = native.post.run_generator(args=SimpleNamespace(training_config=SimpleNamespace(**p['native']['args'])),
                    accelerator=SimpleNamespace(device=device,is_main_process=True),transformer=g,
                    scheduler=native.scheduler(**p['native']['scheduler']),noise=inputs.shape_noise,
                    prompt_embeds=prompt.to(device),operation_trace=inputs,trace_prefix=NOISE_PREFIX,**p['native']['rollout'])
                inputs.finish()
        finally:
            restore_rng(before)
    latents = result[0]
    require(isinstance(latents,torch.Tensor) and list(latents.shape) == p['geometry']['noise_shape']
        and latents.dtype == torch.float32 and not latents.requires_grad and bool(latents.isfinite().all()),
        'original native generator returned invalid latents')
    return dict(latents=latents,latent_schema=dict(latents=dict(layout='BCTHW',shape=list(latents.shape),dtype=str(latents.dtype))))


def decode(state, generated):
    from .inference import latent_outputs
    p,vae = state['inference'],state['codec']
    require(latent_outputs(generated) == dict(latents=dict(layout='BCTHW',shape=p['geometry']['noise_shape'],dtype='torch.float32')),
            'codec requires exact native BCTHW FP32 latents')
    require(codec_parameters(vae,evidence_kind=p['codec']['evidence_kind'],provenance=p['codec']['provenance']) == p['codec']
        and not vae.training and all(not v.requires_grad for v in vae.parameters()), 'separate frozen real Wan2.1 codec required')
    z = generated['latents'].to(device=vae.device,dtype=vae.dtype)
    mean = torch.tensor(vae.config.latents_mean,device=vae.device,dtype=vae.dtype).view(1,16,1,1,1)
    std = torch.tensor(vae.config.latents_std,device=vae.device,dtype=vae.dtype).view(1,16,1,1,1)
    with torch.no_grad():
        decoded = vae.decode(z*std+mean,return_dict=False)[0]
    frames, h,w,_ = p['geometry']['video_shape']
    require(decoded.shape == (1,3,frames,h,w) and bool(decoded.isfinite().all()), 'actual VAE decoded geometry/finiteness changed')
    rgb = ((decoded[0].permute(1,2,3,0)+1.)/2.*255.).clamp(0,255).to(torch.uint8)
    return dict(video=rgb.contiguous(),decoder=dict(class_name='diffusers.AutoencoderKLWan',
        normalization=p['codec']['normalization'],latent_clamp=False, fps=p['geometry']['fps'],
        evidence_kind=p['codec']['evidence_kind'],provenance=deepcopy(p['codec']['provenance']),quality_validation='NOT_RUN'))


def _load_codec(spec, budget, evidence_kind):
    """Strict local safetensors admission; never from_pretrained/Hub/pickle."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    from .checkpoint import rng_state, restore_rng
    require(isinstance(spec,dict) and set(spec) == {'config','weights','provenance'}, 'pinned local codec config/weights/provenance required')
    raw = read_json(checked_pin(spec['config']))
    cls = _codec_class()
    signature = inspect.signature(cls.__init__).parameters
    require(isinstance(raw,dict) and not {k for k in raw if not k.startswith('_')} - (set(signature)-{'self'}),
            'unknown codec constructor fields')
    architecture = plain({name:raw.get(name,arg.default) for name,arg in signature.items() if name != 'self'})
    before = rng_state()
    try:
        # Validate the exact native codec geometry/branch and bounded width
        # before even the meta constructor can create an unbounded module tree.
        portable = _codec_parameters(architecture,'torch.float32',evidence_kind,spec['provenance'])
        with torch.device('meta'):
            probe = cls(**architecture)
        expected = {k:(list(v.shape),str(v.dtype)) for k,v in probe.state_dict().items()}
        count = sum(v.numel() for v in probe.state_dict().values())
        require(count <= min(200_000_000,budget['max_parameters']) and 12*count <= budget['max_loaded_bytes'],
                'codec construction/strict-load capacity exceeded')
        require(isinstance(spec['weights'],list) and 0 < len(spec['weights']) <= 32, 'bounded complete codec safe shards required')
        paths,shapes,total = [],{},0
        for pin in spec['weights']:
            path = checked_pin(pin)
            require(path.suffix == '.safetensors' and path not in paths, 'unique local codec safetensors required')
            total += pin['bytes']; require(total <= budget['max_loaded_bytes'], 'codec safe shard capacity exceeded')
            paths.append(path)
            with safe_open(path,framework='pt',device='cpu') as stream:
                for name in stream.keys():
                    view = stream.get_slice(name)
                    require(name not in shapes and view.get_dtype() == 'F32', 'codec duplicate state or non-FP32 precision')
                    shapes[name] = (view.get_shape(),'torch.float32')
        require(shapes == expected, 'complete codec safe state layout changed')
        codec = cls(**architecture).requires_grad_(False).eval()
        # Strict complete load, while bounding and releasing every source shard.
        # load_state_dict(strict=False) is only a shard copy; completeness was
        # admitted above and no key is left at random initialization.
        for pin,path in zip(spec['weights'],paths,strict=True):
            state = load_file(str(path),device='cpu'); checked_pin(pin)
            require(all(bool(v.isfinite().all()) for v in state.values()), 'codec nonfinite safe weights')
            result = codec.load_state_dict(state,strict=False)
            require(not result.unexpected_keys, 'unexpected codec safe weights')
            del state
        checked_pin(spec['config'])
    finally:
        restore_rng(before)
    return codec,portable


def project_current_generator(session, codec, request, *, world_size):
    """Export-only live view of the actual restored OriginalWanSession G."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    from distillschedule.open_workload.torch_backend import unwrap_module
    g = unwrap_module(session.g.module)
    require(set(g.peft_config) == {'default'} and all(not getattr(m,'merged',False) for m in g.modules()),
            'CURRENT unmerged default Generator adapters required; never EMA or merged LoRA')
    peft = g.peft_config['default']
    # PEFT may compress config targets to suffixes; only installed default
    # A/B adapters prove the complete full-name inventory for strict admission.
    targets = sorted(name for name, layer in g.named_modules()
        if isinstance(layer, BaseTunerLayer) and 'default' in getattr(layer, 'lora_A', {})
        and 'default' in getattr(layer, 'lora_B', {}))
    task = session.task
    constructor = set(inspect.signature(type(g).__init__).parameters)-{'self'}
    parameters = plain(dict(schema_version=SCHEMA,precision=PRECISION,source=inference_source(),runtime=runtime_identity(),
        architecture={k:v for k,v in dict(g.config).items() if not k.startswith('_')
            and (k in constructor or k not in NON_CONSTRUCTOR_CONFIG_FIELDS)},
        peft=dict(r=peft.r,lora_alpha=peft.lora_alpha,lora_dropout=peft.lora_dropout,init_lora_weights=peft.init_lora_weights,
                  target_modules=targets,exclude_modules=sorted(peft.exclude_modules)),
        native={**NATIVE,'rollout':_rollout(task)},task=task.to_dict(),
        noise_protocol=noise_protocol(),
        geometry=dict(noise_shape=list(task.latent_shape),prompt_shape=request['prompt_shape'],
            video_shape=[task.video_frames,task.video_height,task.video_width,3],fps=request['fps']),
        budget=request['budget'],seed=request['seed'],evidence_kind=request['evidence_kind'],
        codec=codec_parameters(codec,evidence_kind=request['evidence_kind'],provenance=request['codec']['provenance'])))
    p,probe,codec_probe,size = _admit(parameters)
    for actual,expected,label in ((g,probe,'Generator'),(codec,codec_probe,'codec')):
        current,target = actual.state_dict(),expected.state_dict()
        require(current.keys() == target.keys() and all(current[k].shape == target[k].shape
            and current[k].dtype == target[k].dtype for k in target), label+' mixed precision/full state layout changed')
    require(type(world_size) is int and world_size >= 1
        and 3*size <= p['budget']['max_export_bytes_per_rank']
        and 3*size*world_size <= p['budget']['max_export_bytes_all_ranks'],
        'all-owner full-tensor gather and transient host capacity exceeded')
    # No eval/freeze/merge/EMA swap, optimizer/cursor manipulation, or G copy.
    components = ModelComponents(state=dict(generator=CurrentGenerator(g),codec=codec),metadata=_metadata(p)).validate()
    return InferenceProjection(components,FACTORY,p).validate()


def standalone_config(manifest, path):
    """NEW prepared inputs are supplied explicitly to the existing generate CLI."""
    return dict(schema_version='distillschedule.training.v1',
        model={key:manifest['model'][key] for key in ('factory','parameters')},
        algorithm=dict(factory='distillschedule.training.wan_native_inference:refuse_training'),
        data=dict(prepared='NEW_PROMPT_FEATURES_NOT_TRAINING.json'),execution=dict(adapter='program',mode='reference'),
        trainer=dict(cycles=1,seed=manifest['model']['parameters']['seed'],device='cpu',
                     output_dir=str(Path(path).with_suffix('.not-training'))))


class NativeOutputRequest:
    """One bounded callback for the existing native restore owner, not a runner."""
    def __init__(self, value, output):
        self.value = plain(value)
        self.output = Path(output).expanduser().resolve()
        self.config_path = self.output.with_name(self.output.name+'.inference.json')
        self.codec = None

    def admit(self):
        from . import distributed
        def local():
            p = self.value
            require(isinstance(p,dict) and set(p) == {'schema_version','seed','prompt_shape','fps','budget','codec','evidence_kind'}
                and p['schema_version'] == EXPORT_SCHEMA, 'explicit trainer.export native output request required')
            _budget(p['budget'])
            require(type(p['seed']) is int and 0 <= p['seed'] < 2**63 and p['evidence_kind'] in EVIDENCE_KINDS
                and type(p['fps']) is int and 1 <= p['fps'] <= 120
                and isinstance(p['prompt_shape'],list) and len(p['prompt_shape']) == 3
                and p['prompt_shape'][0] == 1 and p['prompt_shape'][2] == 4096
                and type(p['prompt_shape'][1]) is int and 0 < p['prompt_shape'][1] <= 4096,
                'explicit independent inference seed/prompt/fps/evidence required')
            require(not self.output.exists() and not self.config_path.exists(), 'portable output/config already exists')
            self.codec,portable = _load_codec(p['codec'],p['budget'],p['evidence_kind'])
            return dict(request={k:v for k,v in p.items() if k != 'codec'},codec=portable,
                codec_hashes=[v['sha256'] for v in p['codec']['weights']],config_hash=p['codec']['config']['sha256'],
                output=str(self.output),config_path=str(self.config_path))
        value = distributed.checked(local,'native output codec/capacity admission')
        require(all(row == value for row in distributed.gather(value)), 'ranks disagree on native output policy/codec/path')

    def publish(self, session, *, manifest, identity):
        from . import distributed
        from .inference import publish_export
        from .plugins import load_callable
        _,world = distributed.rank_world()
        projection = distributed.checked(lambda: project_current_generator(session,self.codec,self.value,world_size=world),
                                          'native CURRENT Generator projection')
        _,source = distributed.checked(lambda: load_callable(FACTORY),'native inference factory source')
        metadata = dict(projection.components.metadata)
        result = publish_export(projection.components,self.output,
            model=dict(factory=FACTORY,source=source,parameters=projection.parameters,metadata=metadata),
            identity=dict(model=source,model_metadata=metadata,training_provenance=identity),cycle=manifest['cycle'])
        def write_config():
            import uuid
            temporary = self.config_path.with_name('.'+self.config_path.name+'.tmp-'+uuid.uuid4().hex)
            try:
                temporary.write_text(json.dumps(standalone_config(result,self.config_path),indent=2,sort_keys=True)+'\n')
                os.link(temporary,self.config_path)  # no replacement of another user's file
            finally:
                temporary.unlink(missing_ok=True)
        distributed.primary(write_config,'native standalone inference config publication')
        return {**result,'inference_config':str(self.config_path)}
