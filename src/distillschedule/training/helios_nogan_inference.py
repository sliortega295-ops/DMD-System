"""Portable CURRENT full unmerged W2 G, independently of warm/training assets.

Original complete Helios constructor and three-stage run_generator only. The
explicit one-section, native zero-history output is an FP32 BCTHW LATENT
DIAGNOSTIC, not decoded video, pretrained quality or distributed validation.
"""
from copy import deepcopy
import inspect
import math
import os
from pathlib import Path
from types import SimpleNamespace

from .adapters import InferenceProjection, ModelComponents
from .helios_public_assets import EVIDENCE_KINDS, require
from .helios_inference import (_environment, _json, content_identity, content_runtime,
                               refuse_training, SHIFT_KEYS)

SCHEMA = 'distillschedule.helios_nogan_current_generator_inference.v1'
FACTORY = 'distillschedule.training.helios_nogan_inference:build_inference_model'
PRECISION = 'WAN_NATIVE_MIXED_BF16'
SCOPE = 'CPU1_BF16_FP32_CURRENT_W2_NATIVE_ZERO_HISTORY_LATENT_DIAGNOSTIC_NOT_WORLD16'


def inference_source():
    """Content pins, never paths to a base, logical package or training input."""
    from distillschedule.workloads.best_known.registry import file_sha256
    from distillschedule.data import helios_nogan_reference_inputs
    from . import helios_public_assets, helios_inference, inference
    return dict(native=content_identity(), layout=helios_nogan_reference_inputs.layout_identity(),
        portable={name:file_sha256(path) for name,path in (
            ('inference',Path(__file__)), ('input_admission',Path(helios_nogan_reference_inputs.__file__)),
            ('safe_assets',Path(helios_public_assets.__file__)),
            ('shared_inference',Path(helios_inference.__file__)), ('named_module_export',Path(inference.__file__)))})


def _cpu1():
    import torch
    _environment()
    require(all(os.environ.get(k,default) == default for k,default in
                (('WORLD_SIZE','1'),('LOCAL_WORLD_SIZE','1'),('RANK','0'),('LOCAL_RANK','0')))
            and not torch.distributed.is_initialized() and not torch.cuda.is_initialized(),
            'W2 inference requires CPU1/rank0 without distributed/CUDA ownership')


def _metadata(p):
    return dict(schema_version=SCHEMA, scope=SCOPE, precision=PRECISION, parameters=deepcopy(p),
        generator_selection='CURRENT_FULL_UNMERGED_NOT_EMA', evidence_kind=p['evidence_kind'],
        conditioning='NEW_PROMPT_NATIVE_ZERO_HISTORY_NOT_TRAINING_HISTORY',
        quality_validation='NOT_RUN', distributed_execution='NOT_RUN', claimable_speedup=False,
        latent_schema={'latents':dict(layout='BCTHW',shape=p['geometry']['noise_shape'],dtype='torch.float32')})


def _rollout(task):
    return dict(is_keep_x0=True, history_sizes=[16,2,1], is_enable_stage2=True, stage2_num_stages=3,
        stage2_num_inference_steps_list=list(task.stage2_inference_steps),
        denoising_step_list=list(task.denoising_step_list), last_step_only=True, last_section_grad_only=False,
        timestep_shift=5., use_dynamic_shifting=True, time_shift_type='linear',
        num_critic_input_frames=task.latent_shape[2], num_rollout_sections=1, is_skip_first_section=False,
        is_amplify_first_chunk=False, is_corrupt_history_latents=False, is_add_saturation=False,
        is_use_gt_history=False, is_dmd_vae_decode=False, is_multi_pyramid_stage_backward_simulated=False,
        use_kv_cache=False)


def _admit(parameters):
    """Fail closed on independent W2 identity before any real model allocation."""
    import torch
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from distillschedule.training.helios_inputs import HeliosCohortTask, MAX_COHORTS
    from runtime.critic_pool_helios import TRANSFORMER_ADDITIONAL_KWARGS, _all_linear_targets
    keys = {'schema_version','scope','precision','source','runtime','task','boundary','architecture',
            'peft','native','geometry','budget','evidence_kind','seed'}
    require(isinstance(parameters,dict) and set(parameters) <= keys | {'device'}, 'unsupported W2 inference parameters')
    p = _json({k:v for k,v in parameters.items() if k != 'device'})
    require(set(p) == keys and p['schema_version'] == SCHEMA and p['scope'] == SCOPE
            and p['precision'] == PRECISION and parameters.get('device','cpu') == 'cpu',
            'explicit independent W2 mixed-precision inference ABI required')
    _cpu1()
    require(p['source'] == inference_source() and p['runtime'] == content_runtime(), 'W2 inference source/runtime/layout changed')
    task = HeliosCohortTask.from_dict(p['task'])
    require(task == HeliosCohortTask(0), 'W2 inference requires original Step10 No-GAN33/222 task')
    require(type(p['seed']) is int and 0 <= p['seed'] < 2**63 and p['evidence_kind'] in EVIDENCE_KINDS,
            'invalid W2 inference seed/evidence')
    boundary = p['boundary']
    require(isinstance(boundary,dict) and set(boundary) == {'cycle','versions','warm_kind'}
            and type(boundary['cycle']) is int and 1 <= boundary['cycle'] <= MAX_COHORTS
            and boundary['versions'] == dict(generator=2+boundary['cycle'],critic=10+5*boundary['cycle'],ema=10+5*boundary['cycle'])
            and all(type(v) is int for v in boundary['versions'].values())
            and boundary['warm_kind'] in ('OPERATOR_SUPPLIED_ORIGINAL_W2_STEP10',
                'ACTUAL_ORIGINAL_W2_CPU_TEST_WARM_NOT_HISTORICAL_STEP10'),
            'W2 export needs a completed current warm-owner boundary, not cold/reader/W1/W3 state')
    budget,geometry = p['budget'],p['geometry']
    require(isinstance(budget,dict) and set(budget) == {'max_parameters','max_loaded_bytes'}
            and type(budget['max_parameters']) is int and 0 < budget['max_parameters'] <= 100000000
            and type(budget['max_loaded_bytes']) is int and 0 < budget['max_loaded_bytes'] <= 2**31,
            'bounded W2 inference CPU parameter/byte budget required')
    require(geometry == dict(noise_shape=list(task.latent_shape),prompt_shape=[1,512,4096])
            and 4*(math.prod(geometry['noise_shape'])+math.prod(geometry['prompt_shape'])) <= budget['max_loaded_bytes'],
            'native W2 prompt/noise geometry changed; no resize or dtype fallback')
    cls = load_reference_model_class(backend='torch-sdpa')
    constructor = inspect.signature(cls.__init__).parameters
    architecture = p['architecture']
    require(isinstance(architecture,dict) and set(architecture) == set(constructor)-{'self'},
            'complete original W2 constructor required')
    dimensions = {'num_attention_heads','attention_head_dim','freq_dim','ffn_dim','num_layers','rope_dim'}
    fixed = _json({k:v.default for k,v in constructor.items() if k != 'self' and k not in dimensions})
    fixed.update(TRANSFORMER_ADDITIONAL_KWARGS)
    # JSON int/float spelling is equivalent only for a native fixed FLOAT
    # option at its exact value. In particular, bool is never numeric here.
    require(all(architecture[k] == v and (type(architecture[k]) in (int,float) if type(v) is float
                else type(architecture[k]) is type(v)) for k,v in fixed.items())
            and all(type(architecture[k]) is int and 0 < architecture[k] <= budget['max_parameters']
                for k in dimensions-{'rope_dim'}) and architecture['num_layers'] <= 128
            and isinstance(architecture['rope_dim'],list) and len(architecture['rope_dim']) == 3
            and all(type(n) is int and n > 0 and n % 2 == 0 for n in architecture['rope_dim'])
            and sum(architecture['rope_dim']) == architecture['attention_head_dim'],
            'unsupported original W2 architecture/conditioning/pyramid dimensions')
    # A meta probe only bounds the ORIGINAL full constructor and targets. The
    # actual CPU constructor below initializes its nonpersistent rotary buffers.
    with torch.device('meta'):
        probe = cls(**architecture)
    targets = sorted(set(_all_linear_targets(probe)+['patch_short','patch_mid','patch_long']))
    require(p['peft'] == dict(r=256,lora_alpha=256.,lora_dropout=0.,init_lora_weights='gaussian',
                target_modules=targets,exclude_modules=['down','up']), 'original W2 full LoRA256/pyramid layout changed')
    size = sum(t.numel() for t in probe.parameters())+sum(t.numel() for t in probe.buffers())
    size += sum(256*(layer.weight[0].numel()+layer.weight.shape[0])
                for name,layer in probe.named_modules() if name in targets)
    require(size <= budget['max_parameters'] and 4*size <= budget['max_loaded_bytes'],
            'W2 inference construction exceeds CPU budget')
    require(p['native'] == dict(args=dict(base_seq_len=256,max_seq_len=4096,base_shift=.5,max_shift=1.15),
                scheduler=dict(shift=1.,stages=3,stage_range=[0,1/3,2/3,1],gamma=1/3),rollout=_rollout(task)),
            'original W2 native scheduler/flow-shift/horizon/cache/zero-history controls changed')
    return p,cls


def project_current_generator(components,effective_training_parameters):
    """A pure live full-G view: never eval/freeze/merge/copy/select EMA."""
    from .helios_nogan_public import HeliosNoGANCycleOwner
    from distillschedule.training.helios_inputs import HeliosCohortTask
    require(set(components.state) == {'native_cycle'} and type(components.state['native_cycle']) is HeliosNoGANCycleOwner,
            'only the independent original W2 owner can project current G')
    owner = components.state['native_cycle']; owner.validate_checkpoint_boundary()
    g,task = owner.generator.module,HeliosCohortTask(0)
    peft,c = g.peft_config['default'],owner.args.training_config
    p = _json(dict(schema_version=SCHEMA,scope=SCOPE,precision=PRECISION,source=inference_source(),runtime=content_runtime(),
        task=task.to_dict(),boundary=dict(cycle=owner.cycle,versions=owner.versions,warm_kind=owner.warm.identity['evidence_kind']),
        architecture={k:v for k,v in dict(g.config).items() if not k.startswith('_')},
        peft=dict(r=peft.r,lora_alpha=peft.lora_alpha,lora_dropout=peft.lora_dropout,init_lora_weights='gaussian',
            target_modules=sorted(peft.target_modules),exclude_modules=['down','up']),
        native=dict(args={k:getattr(c,k) for k in SHIFT_KEYS},
            scheduler=dict(shift=c.stage2_timestep_shift,stages=c.stage2_num_stages,
                stage_range=list(c.stage2_stage_range),gamma=c.stage2_scheduler_gamma),rollout=_rollout(task)),
        geometry=dict(noise_shape=list(task.latent_shape),prompt_shape=[1,512,4096]),
        budget=deepcopy(dict(owner.warm.identity['assets']['budget'])),
        evidence_kind=owner.warm.identity['assets']['base_evidence_kind'],seed=effective_training_parameters['seed']))
    _admit(p)
    view = ModelComponents(state={'generator':g},metadata=_metadata(p)).validate()
    return InferenceProjection(view,FACTORY,p).validate()


def build_inference_model(parameters):
    """A bounded strict-load target, not random/pretrained missing-key fallback."""
    import torch
    from peft import LoraConfig
    p,cls = _admit(parameters)
    # Do not to_empty a meta model: native nonpersistent RoPE is NOT exported.
    with torch.device('cpu'):
        g = cls(**p['architecture'])
        for name,tensor in g.named_parameters():
            tensor.data = tensor.data.to(torch.float32 if any(k in name for k in cls._keep_in_fp32_modules) else torch.bfloat16)
        g.requires_grad_(False)
        g.add_adapter(LoraConfig(**p['peft']))
        for name,tensor in g.named_parameters():
            if 'lora_' in name:
                tensor.data = tensor.data.float()
    g.requires_grad_(False).eval(); g.disable_gradient_checkpointing()
    require(all(t.device.type == 'cpu' and bool(t.isfinite().all()) for t in g.buffers()),
            'original W2 nonpersistent rotary buffers must be initialized by their CPU constructor')
    return ModelComponents(state={'generator':g,'inference':p},metadata=_metadata(p),
        generate=generate,admit_training=refuse_training).validate()


def generate(state,feeds,parameters):
    import torch
    from distillschedule.strategies.wan_capture import native_primitives
    _cpu1()
    p,g = state['inference'],state['generator']
    require(not parameters and isinstance(feeds,dict) and set(feeds) == {'prompt_embeds'},
            'exact NEW W2 prompt only; no training rows/history/override/cache')
    prompt = feeds['prompt_embeds']
    require(isinstance(prompt,torch.Tensor) and list(prompt.shape) == p['geometry']['prompt_shape']
            and prompt.dtype == torch.bfloat16 and prompt.device.type == 'cpu' and prompt.layout == torch.strided
            and bool(prompt.isfinite().all()), 'W2 inference prompt layout/precision/finiteness changed')
    require(not g.training and not g.gradient_checkpointing and all(not v.requires_grad for v in g.parameters())
            and all(not getattr(m,'merged',False) for m in g.modules())
            and all(b.attn1.processor.kv_cache is None and not b.attn1.processor.cache_enabled for b in g.blocks),
            'separate frozen unmerged cache-free W2 inference G required')
    native = native_primitives(source_only=True)
    result = native.post.run_generator(args=SimpleNamespace(training_config=SimpleNamespace(**p['native']['args'])),
        accelerator=SimpleNamespace(device=torch.device('cpu'),is_main_process=True),transformer=g,
        scheduler=native.scheduler(**p['native']['scheduler']),
        noise=torch.zeros(p['geometry']['noise_shape'],dtype=torch.bfloat16,device='cpu'),
        prompt_embeds=prompt,**p['native']['rollout'])
    latents = result[0]
    # Original uncaptured block-noise inference returns FP32, despite BF16
    # bases/prompt and FP32 trainables. Never coerce this native return value.
    require(isinstance(latents,torch.Tensor) and list(latents.shape) == p['geometry']['noise_shape']
            and latents.dtype == torch.float32 and not latents.requires_grad and bool(latents.isfinite().all()),
            'invalid original W2 native latent diagnostic')
    return dict(latents=latents,latent_schema=deepcopy(_metadata(p)['latent_schema']))
