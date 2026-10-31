"""Portable CURRENT full mixed-precision W1 G, independent of all trainers.

One new-prompt zero-history native latent diagnostic; no decoder, quality,
world16, C, Adam, EMA, base-weight path or training-data dependency.
"""
from copy import deepcopy
import inspect
import math
from types import SimpleNamespace
from pathlib import Path

from .adapters import ModelComponents, InferenceProjection
from .helios_public_assets import require, EVIDENCE_KINDS
from .helios_inference import content_identity, content_runtime, _json, _environment, SHIFT_KEYS, refuse_training

SCHEMA = 'distillschedule.wan_current_generator_inference.v1'
PROJECTION_SCHEMA = 'distillschedule.wan_current_generator_inference.v2'
FACTORY = 'distillschedule.training.wan_inference:build_inference_model'
PRECISION = 'WAN_NATIVE_MIXED_BF16'
SCOPE = 'CPU1_BF16_FP32_CURRENT_W1_NATIVE_LATENT_DIAGNOSTIC_NOT_WORLD16'


def inference_source():
    """All portable helpers, without importing/constructing a training owner."""
    from distillschedule.workloads.best_known.registry import file_sha256
    from distillschedule.data import wan_reference_inputs
    from . import helios_public_assets, helios_inference
    return dict(native=content_identity(),layout=wan_reference_inputs.layout_identity(),portable={name:file_sha256(path) for name,path in
        (('inference',Path(__file__)),('input_admission',Path(wan_reference_inputs.__file__)),
         ('safe_assets',Path(helios_public_assets.__file__)),('shared_inference',Path(helios_inference.__file__)))})


def _metadata(p):
    return dict(schema_version=p['schema_version'],scope=SCOPE,precision=PRECISION,parameters=deepcopy(p),
        generator_selection='CURRENT_FULL_UNMERGED_NOT_EMA',evidence_kind=p['evidence_kind'],
        quality_validation='NOT_RUN',claimable_speedup=False,
        # Original NEW-input (uncaptured) block-noise path returns FP32. Model
        # bases/trainables remain BF16/FP32; never cast the native return value.
        latent_schema={'latents':dict(layout='BCTHW',shape=p['geometry']['noise_shape'],dtype='torch.float32')})


def _admit(parameters):
    import torch
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from distillschedule.strategies.wan_task import OriginalWanTask
    from runtime.critic_pool_helios import TRANSFORMER_ADDITIONAL_KWARGS, _all_linear_targets
    from distillschedule.data.wan_reference_inputs import topology
    require(isinstance(parameters,dict) and set(parameters) <= {'schema_version','precision','source','runtime',
        'architecture','peft','native','geometry','budget','evidence_kind','seed','device','task','noise_protocol'}, 'unsupported W1 inference parameters')
    p = _json({k:v for k,v in parameters.items() if k != 'device'})
    fields = {'schema_version','precision','source','runtime','architecture','peft','native','geometry',
              'budget','evidence_kind','seed','task'}
    projection = p.get('schema_version') == PROJECTION_SCHEMA
    require(set(p) == fields | ({'noise_protocol'} if projection else set())
            and p.get('schema_version') in (SCHEMA,PROJECTION_SCHEMA) and p['precision'] == PRECISION
            and parameters.get('device','cpu') == 'cpu', 'explicit W1 mixed-precision inference ABI required')
    if projection:
        from .reference_inference_noise import protocol
        require(p['noise_protocol'] == protocol('torch.bfloat16'), 'inference noise protocol/source changed')
    topology(); _environment()
    require(p['source'] == inference_source() and p['runtime'] == content_runtime(), 'W1 inference source/runtime changed')
    require(type(p['seed']) is int and 0 <= p['seed'] < 2**63 and p['evidence_kind'] in EVIDENCE_KINDS, 'invalid W1 seed/evidence')
    task = OriginalWanTask.from_dict(p['task'])
    require(task.first_step == 0 and task.fake_updates == 5 and task.stage_schedule_policy == 'source_normal_only', 'wrong W1 inference task')
    budget,geometry = p['budget'],p['geometry']
    require(set(budget) == {'max_parameters','max_loaded_bytes'} and type(budget['max_parameters']) is int
            and 0 < budget['max_parameters'] <= 100000000 and type(budget['max_loaded_bytes']) is int
            and 0 < budget['max_loaded_bytes'] <= 2**31, 'bounded W1 inference budget required')
    require(set(geometry) == {'noise_shape','prompt_shape'} and geometry['noise_shape'] == list(task.latent_shape)
            and len(geometry['prompt_shape']) == 3 and geometry['prompt_shape'][0] == 1
            and geometry['prompt_shape'][-1] == 4096 and type(geometry['prompt_shape'][1]) is int
            and 0 < geometry['prompt_shape'][1] <= 4096
            and 2*(math.prod(geometry['noise_shape'])+math.prod(geometry['prompt_shape'])) <= budget['max_loaded_bytes'], 'native W1 prompt/noise shape changed')
    cls = load_reference_model_class(backend='torch-sdpa')
    architecture = p['architecture']
    require(set(architecture) == set(inspect.signature(cls.__init__).parameters)-{'self'}, 'complete original W1 constructor required')
    require(all(architecture[k] == v for k,v in TRANSFORMER_ADDITIONAL_KWARGS.items())
            and architecture['text_dim'] == 4096 and architecture['in_channels'] == architecture['out_channels'] == 16
            and architecture['patch_size'] == [1,2,2]
            and not any(architecture[k] for k in ('is_use_gan','is_use_gan_final','is_use_gan_hooks'))
            and architecture['image_dim'] is None and architecture['added_kv_proj_dim'] is None
            and all(type(architecture[k]) is int and 0 < architecture[k] <= budget['max_parameters']
                    for k in ('num_attention_heads','attention_head_dim','ffn_dim','freq_dim','num_layers'))
            and architecture['num_layers'] <= 128 and len(architecture['rope_dim']) == 3
            and all(type(n) is int and n > 0 and n%2 == 0 for n in architecture['rope_dim'])
            and sum(architecture['rope_dim']) == architecture['attention_head_dim'], 'unsupported original W1 architecture branch/dimensions')
    with torch.device('meta'):
        probe = cls(**architecture)
    targets = sorted(set(_all_linear_targets(probe)+['patch_short','patch_mid','patch_long']))
    require(p['peft'] == dict(r=task.lora_rank,lora_alpha=float(task.lora_rank),lora_dropout=0.,
        init_lora_weights='gaussian',target_modules=targets,exclude_modules=['down','up']), 'original W1 full LoRA/pyramid layout changed')
    size = sum(t.numel() for t in probe.parameters()) + sum(t.numel() for t in probe.buffers())
    size += sum(task.lora_rank*(layer.weight[0].numel()+layer.weight.shape[0])
                for name,layer in probe.named_modules() if name in targets)
    require(size <= budget['max_parameters'] and 4*size <= budget['max_loaded_bytes'], 'W1 inference construction budget exceeded')
    require(p['native'] == dict(args=dict(base_seq_len=256,max_seq_len=4096,base_shift=.5,max_shift=1.15),
        scheduler=dict(shift=1.,stages=3,stage_range=[0,1/3,2/3,1],gamma=1/3),
        rollout=_rollout(task)), 'unsupported original W1 native rollout/cache/conditioning controls')
    return p, cls


def _rollout(task):
    return dict(is_keep_x0=True,history_sizes=[16,2,1],is_enable_stage2=True,stage2_num_stages=3,
        stage2_num_inference_steps_list=list(task.stage2_inference_steps),denoising_step_list=list(task.denoising_step_list),
        last_step_only=True,last_section_grad_only=False,timestep_shift=5.,use_dynamic_shifting=True,time_shift_type='linear',
        num_critic_input_frames=task.latent_shape[2],num_rollout_sections=1,is_skip_first_section=False,
        is_amplify_first_chunk=False,is_corrupt_history_latents=False,is_add_saturation=False,is_use_gt_history=False,
        is_dmd_vae_decode=False,is_multi_pyramid_stage_backward_simulated=False,use_kv_cache=False)


def project_current_generator(components,effective_training_parameters):
    require(set(components.state) == {'native_cycle'}, 'original W1 native cycle owner required')
    owner = components.state['native_cycle']; owner._validate_live()
    g = owner.generator.module
    peft = g.peft_config['default']
    parameters = _json(dict(schema_version=SCHEMA,precision=PRECISION,source=inference_source(),runtime=content_runtime(),
        architecture={k:v for k,v in dict(g.config).items() if not k.startswith('_')},
        peft=dict(r=peft.r,lora_alpha=peft.lora_alpha,lora_dropout=peft.lora_dropout,init_lora_weights='gaussian',
            target_modules=sorted(peft.target_modules),exclude_modules=['down','up']),
        native=dict(args={k:getattr(owner.args.training_config,k) for k in SHIFT_KEYS},
            scheduler=dict(shift=1.,stages=3,stage_range=[0,1/3,2/3,1],gamma=1/3),rollout=_rollout(owner.task)),
        task=owner.task.to_dict(),geometry=deepcopy(components.metadata['assets']['geometry']),
        budget=deepcopy(components.metadata['assets']['budget']),evidence_kind=components.metadata['evidence_kind'],
        seed=effective_training_parameters['seed']))
    from .reference_inference_noise import enabled, protocol
    if enabled(effective_training_parameters):
        parameters.update(schema_version=PROJECTION_SCHEMA,noise_protocol=protocol('torch.bfloat16'))
    _admit(parameters)
    # Pure live view: do not eval/freeze/merge/change adapters or RNG here.
    view = ModelComponents(state={'generator':g},metadata=_metadata(parameters)).validate()
    return InferenceProjection(view,FACTORY,parameters).validate()


def build_inference_model(parameters):
    import torch
    from peft import LoraConfig
    p,cls = _admit(parameters)
    g = cls(**p['architecture'])
    # Match the native loader's mixed base inventory BEFORE strict export load.
    # This is a bounded target, never pretrained/random inference fallback.
    for name,tensor in g.named_parameters():
        tensor.data = tensor.data.to(torch.float32 if any(k in name for k in cls._keep_in_fp32_modules) else torch.bfloat16)
    g.requires_grad_(False)
    g.add_adapter(LoraConfig(**p['peft']))
    for name,tensor in g.named_parameters():
        if 'lora_' in name:
            tensor.data = tensor.data.float()
    g.requires_grad_(False).eval(); g.disable_gradient_checkpointing()
    return ModelComponents(state={'generator':g,'inference':p},metadata=_metadata(p),
        generate=generate,admit_training=refuse_training).validate()


def generate(state,feeds,parameters):
    import torch
    from distillschedule.strategies.wan_capture import native_primitives
    from distillschedule.data.wan_reference_inputs import topology
    topology(); _environment()
    p,g = state['inference'],state['generator']
    require(not parameters and isinstance(feeds,dict) and set(feeds) == {'prompt_embeds'}, 'exact NEW W1 prompt only; no training rows/override/cache')
    prompt = feeds['prompt_embeds']
    require(isinstance(prompt,torch.Tensor) and list(prompt.shape) == p['geometry']['prompt_shape']
            and prompt.dtype == torch.bfloat16 and prompt.device.type == 'cpu' and prompt.layout == torch.strided
            and bool(prompt.isfinite().all()), 'W1 inference prompt layout/precision/finiteness changed')
    require(not g.training and not g.gradient_checkpointing and all(not v.requires_grad for v in g.parameters()), 'separate frozen inference G required')
    native = native_primitives(source_only=True)
    inputs = None
    kwargs = {}
    noise = torch.zeros(p['geometry']['noise_shape'],dtype=torch.bfloat16)
    if p['schema_version'] == PROJECTION_SCHEMA:
        from .reference_inference_noise import prepare
        inputs = prepare(p,torch.bfloat16)
        noise = inputs.shape_noise
        kwargs = dict(operation_trace=inputs,trace_prefix=p['noise_protocol']['operation_prefix'])
    result = native.post.run_generator(args=SimpleNamespace(training_config=SimpleNamespace(**p['native']['args'])),
        accelerator=SimpleNamespace(device=torch.device('cpu'),is_main_process=True),transformer=g,
        scheduler=native.scheduler(**p['native']['scheduler']),noise=noise,
        prompt_embeds=prompt,**kwargs,**p['native']['rollout'])
    if inputs is not None:
        inputs.finish()
    latents = result[0]
    require(isinstance(latents,torch.Tensor) and list(latents.shape) == p['geometry']['noise_shape']
            and latents.dtype == torch.float32 and not latents.requires_grad and bool(latents.isfinite().all()), 'invalid original W1 latent diagnostic')
    return dict(latents=latents,latent_schema=deepcopy(_metadata(p)['latent_schema']))
