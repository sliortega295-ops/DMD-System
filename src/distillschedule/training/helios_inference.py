"""Inference-only CURRENT full original Helios G, no training/data/base owners.

Reuses the original three-stage ``run_generator``. This first CPU contract is
an explicit one-section zero-history, last-step-per-stage LATENT DIAGNOSTIC.
It has no decoder or media-quality claim, no C/Adam/EMA, and no Hub fallback.
"""
from __future__ import annotations

from copy import deepcopy
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from distillschedule.workloads.best_known.registry import file_sha256
from .adapters import InferenceProjection, ModelComponents
from .helios_public_assets import EVIDENCE_KINDS, require, runtime_identity

SCHEMA = 'distillschedule.helios_current_generator_inference.v1'
FACTORY = 'distillschedule.training.helios_inference:build_inference_model'
SCOPE = 'CPU1_FP32_ORIGINAL_NATIVE_ZERO_HISTORY_LATENT_DIAGNOSTIC'
SHIFT_KEYS = ('base_seq_len', 'max_seq_len', 'base_shift', 'max_shift')


def _json(value):
    """Detach plain configs, with no owner, tensor, path or mutable closure."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def content_identity():
    """Re-admit actual source/runtime bytes, deliberately excluding locations."""
    from distillschedule.adapters.helios.reference_model import reference_model_identity
    from distillschedule.strategies.wan_capture import native_primitives
    from runtime.gan_rolephase_official import common_official_call

    model = reference_model_identity(backend='torch-sdpa')
    model = {**model, 'source': {k:v for k,v in model['source'].items()
                               if k not in ('transformer_path', 'manifest_path')},
             'backend': {k:v for k,v in model['backend'].items() if k != 'path'}}
    native = native_primitives()
    return dict(model=model, helpers={name: file_sha256(Path(inspect.getfile(function))) for name,function in
        (('run_generator', native.post.run_generator), ('history', native.history.prepare_stage1_clean_input_from_latents),
         ('schedule_shift', native.base.apply_schedule_shift), ('scheduler', native.scheduler),
         ('official_call_binding', common_official_call), ('model_provider', reference_model_identity),
         ('native_import_adapter', native_primitives))})


def content_runtime():
    return {name:{k:v for k,v in row.items() if k != 'path'} for name,row in runtime_identity().items()}


def _environment():
    import torch
    require(torch.get_default_dtype() == torch.float32, 'inference requires the explicit FP32 runtime default')
    require(int(os.environ.get('WORLD_SIZE', '1')) == 1
            and (not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1),
            'inference CPU1 only; distributed topology not admitted')
    require(os.environ.get('DISTILLSCHEDULE_STABLE_BLOCK_NOISE') != '1',
            'inference uses original block-noise sampling, not an environment-selected compatibility path')


def _metadata(parameters):
    # No current parameter VALUES: those are bound by the complete export hashes.
    return dict(schema_version=SCHEMA, scope=SCOPE, parameters=deepcopy(parameters),
                generator_selection='CURRENT_FULL_UNMERGED_NOT_EMA',
                evidence_kind=parameters['evidence_kind'], quality_validation='NOT_RUN', claimable_speedup=False,
                latent_schema={'latents': dict(layout='BCTHW', shape=parameters['geometry']['noise_shape'],
                                                dtype='torch.float32')})


def _admit(parameters):
    """Bound shapes/capacity/source/config before materializing any real G."""
    import torch
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    require(isinstance(parameters, dict) and set(parameters) <= {
        'schema_version', 'source', 'runtime', 'architecture', 'peft', 'native', 'geometry',
        'budget', 'evidence_kind', 'seed', 'device'}, 'unsupported inference factory parameters')
    p = _json({k:v for k,v in parameters.items() if k != 'device'})
    require(set(p) == {'schema_version', 'source', 'runtime', 'architecture', 'peft', 'native',
                      'geometry', 'budget', 'evidence_kind', 'seed'} and p['schema_version'] == SCHEMA,
            'complete explicit inference parameter schema required')
    require(parameters.get('device', 'cpu') == 'cpu', 'inference CPU FP32 only; no CUDA substitution')
    _environment()
    require(type(p['seed']) is int and 0 <= p['seed'] < 2**63 and p['evidence_kind'] in EVIDENCE_KINDS,
            'inference seed/evidence kind changed')
    require(p['source'] == content_identity(), 'inference original source/backend/helper content changed')
    require(p['runtime'] == content_runtime(), 'inference runtime content/version changed')
    budget = p['budget']
    require(isinstance(budget, dict) and set(budget) == {'max_parameters', 'max_loaded_bytes'}
            and type(budget['max_parameters']) is int and 0 < budget['max_parameters'] <= 100000000
            and type(budget['max_loaded_bytes']) is int and 0 < budget['max_loaded_bytes'] <= 2**31,
            'explicit bounded inference CPU parameter/byte budget required')
    geometry = p['geometry']
    require(isinstance(geometry, dict) and set(geometry) == {'noise_shape', 'prompt_shape'},
            'explicit native inference geometry required')
    noise, prompt = geometry['noise_shape'], geometry['prompt_shape']
    require(isinstance(noise, list) and len(noise) == 5 and noise[:2] == [1,16]
            and all(type(x) is int and x > 0 for x in noise) and noise[-2] % 16 == noise[-1] % 16 == 0
            and isinstance(prompt, list) and len(prompt) == 3 and prompt[0] == 1
            and all(type(x) is int and x > 0 for x in prompt), 'unsupported native inference geometry')
    require(4 * (math.prod(noise) + math.prod(prompt)) <= budget['max_loaded_bytes'],
            'inference native tensor geometry exceeds explicit CPU byte budget')
    cls = load_reference_model_class(backend='torch-sdpa')
    architecture = p['architecture']
    require(isinstance(architecture, dict)
            and set(architecture) == set(inspect.signature(cls.__init__).parameters) - {'self'},
            'complete original constructor architecture required; no unknown/default config')
    require(all(type(architecture[k]) is int and 0 < architecture[k] <= budget['max_parameters']
                for k in ('num_attention_heads','attention_head_dim','text_dim','freq_dim','ffn_dim','num_layers'))
            and architecture['num_layers'] <= 128
            and isinstance(architecture['rope_dim'], list) and len(architecture['rope_dim']) == 3
            and all(type(n) is int and n > 0 and n % 2 == 0 for n in architecture['rope_dim'])
            and sum(architecture['rope_dim']) == architecture['attention_head_dim']
            and type(architecture['rope_theta']) in (int,float) and architecture['rope_theta'] > 0
            and architecture['image_dim'] is None and architecture['added_kv_proj_dim'] is None,
            'unsupported or unbounded original inference architecture dimensions')
    require(architecture['text_dim'] == prompt[-1] and architecture['in_channels'] == architecture['out_channels'] == 16
            and architecture['patch_size'] == [1,2,2] and architecture['has_multi_term_memory_patch'] is True
            and not any(architecture[key] for key in ('is_use_gan', 'is_use_gan_final', 'is_use_gan_hooks',
                'zero_history_timestep', 'restrict_self_attn', 'guidance_cross_attn', 'is_train_restrict_lora',
                'restrict_lora', 'is_amplify_history')), 'unsupported Generator architecture/conditioning branch')
    peft = p['peft']
    require(isinstance(peft, dict) and set(peft) == {'r', 'lora_alpha', 'lora_dropout',
            'init_lora_weights', 'target_modules', 'exclude_modules'} and type(peft['r']) is int and peft['r'] > 0
            and type(peft['lora_alpha']) in (int,float) and peft['lora_alpha'] > 0
            and type(peft['lora_dropout']) in (int,float) and 0 <= peft['lora_dropout'] < 1
            and peft['init_lora_weights'] == 'gaussian'
            and isinstance(peft['target_modules'], list) and peft['target_modules']
            and all(isinstance(n,str) and n and 'norm' not in n for n in peft['target_modules'])
            and isinstance(peft['exclude_modules'], list)
            and all(isinstance(n,str) and n for n in peft['exclude_modules']), 'unsupported original PEFT config')
    native = p['native']
    require(isinstance(native, dict) and set(native) == {'args', 'scheduler', 'rollout'},
            'explicit native inference Args/scheduler/rollout required')
    args, scheduler, rollout = native['args'], native['scheduler'], native['rollout']
    require(isinstance(args, dict) and set(args) == set(SHIFT_KEYS)
            and type(args['base_seq_len']) is int and type(args['max_seq_len']) is int
            and 0 < args['base_seq_len'] < args['max_seq_len']
            and all(type(args[k]) in (int,float) and args[k] > 0 for k in ('base_shift', 'max_shift')),
            'unsupported native dynamic-shift Args')
    require(isinstance(scheduler, dict) and set(scheduler) == {'shift','stages','stage_range','gamma'}
            and type(scheduler['shift']) in (int,float) and scheduler['shift'] > 0
            and scheduler['stages'] == 3 and scheduler['stage_range'] == [0,1/3,2/3,1]
            and scheduler['gamma'] == 1/3, 'unsupported native inference scheduler')
    fixed = dict(is_keep_x0=True, history_sizes=[16,2,1], is_enable_stage2=True, stage2_num_stages=3,
        last_step_only=True, last_section_grad_only=False, num_rollout_sections=1, is_skip_first_section=False,
        is_amplify_first_chunk=False, is_corrupt_history_latents=False, is_add_saturation=False,
        is_use_gt_history=False, is_dmd_vae_decode=False, is_multi_pyramid_stage_backward_simulated=False,
        use_kv_cache=False)
    require(isinstance(rollout, dict) and set(rollout) == set(fixed) | {
        'stage2_num_inference_steps_list','denoising_step_list','timestep_shift','use_dynamic_shifting',
        'time_shift_type','num_critic_input_frames'} and all(rollout[k] == v for k,v in fixed.items())
        and isinstance(rollout['stage2_num_inference_steps_list'], list)
        and len(rollout['stage2_num_inference_steps_list']) == 3
        and all(type(x) is int and 0 < x <= 1000 for x in rollout['stage2_num_inference_steps_list'])
        and isinstance(rollout['denoising_step_list'], list) and rollout['denoising_step_list']
        and all(type(x) is int and 0 <= x <= 1000 for x in rollout['denoising_step_list'])
        and type(rollout['timestep_shift']) in (int,float) and rollout['timestep_shift'] > 0
        and type(rollout['use_dynamic_shifting']) is bool and rollout['time_shift_type'] in ('linear','exponential')
        and rollout['num_critic_input_frames'] == noise[2], 'unsupported native zero-history diagnostic rollout')
    # The ORIGINAL constructor runs on meta solely to bound architecture and
    # adapters, BEFORE any real allocation. Its deterministic nonpersistent RoPE
    # is later initialized by the original CPU constructor, never to_empty.
    with torch.device('meta'):
        probe = cls(**architecture)
    base_count = sum(t.numel() for t in probe.parameters())
    adapter_count = sum(peft['r'] * (layer.in_features + layer.out_features)
        for name,layer in probe.named_modules() if isinstance(layer,torch.nn.Linear)
        and any(name.endswith(target) for target in peft['target_modules']))
    buffers = sum(t.numel() for t in probe.buffers())
    require(adapter_count > 0 and base_count + adapter_count + buffers <= budget['max_parameters']
            and 4 * (base_count + adapter_count + buffers) <= budget['max_loaded_bytes'],
            'inference original architecture/PEFT/buffer capacity exceeds explicit CPU budget')
    return p, cls


def _components(generator, parameters):
    return ModelComponents(state={'generator':generator, 'inference':deepcopy(parameters)},
        metadata=_metadata(parameters), generate=generate, admit_training=refuse_training).validate()


def project_current_generator(components, effective_training_parameters):
    """Only restored live current G is projected; never EMA or a rebuilt G."""
    from runtime.gan_rolephase_official import common_official_call
    require(set(components.state) == {'native_step'}, 'explicit native_step training owner required')
    owner = components.state['native_step']
    owner._validate_live()
    generator = owner.generator.module
    require(all(not getattr(module, 'merged', False) for module in generator.modules()),
            'inference projection requires current unmerged Generator adapters')
    c, m = owner.args.training_config, owner.args.model_config
    rollout = common_official_call(owner.args)
    rollout.update(is_use_gt_history=False, last_step_only=True, last_section_grad_only=False,
        is_skip_first_section=False, is_corrupt_history_latents=False, is_add_saturation=False, use_kv_cache=False)
    parameters = _json(dict(schema_version=SCHEMA, source=content_identity(), runtime=content_runtime(),
        architecture={k:v for k,v in dict(generator.config).items() if not k.startswith('_')},
        peft=dict(r=m.lora_rank, lora_alpha=m.lora_alpha, lora_dropout=m.lora_dropout,
            init_lora_weights='gaussian', target_modules=list(m.lora_target_modules), exclude_modules=list(m.lora_exclude_modules)),
        native=dict(args={k:getattr(c,k) for k in SHIFT_KEYS},
            scheduler=dict(shift=c.stage2_timestep_shift, stages=c.stage2_num_stages,
                           stage_range=list(c.stage2_stage_range), gamma=c.stage2_scheduler_gamma), rollout=rollout),
        geometry=deepcopy(components.metadata['assets']['geometry']), budget=deepcopy(components.metadata['assets']['budget']),
        evidence_kind=components.metadata['evidence_kind'], seed=effective_training_parameters['seed']))
    _admit(parameters)
    # This view has no generate callback: exporting must never freeze/eval or
    # run the live training G. The separate inference factory installs its own.
    projected = ModelComponents(state={'generator':generator}, metadata=_metadata(parameters)).validate()
    return InferenceProjection(projected, FACTORY, parameters).validate()


def build_inference_model(parameters):
    """Original full G + PEFT, then STRICT exported weights before any forward.

    Constructor initialization is only a bounded strict-load target, not a base
    fallback. No from_pretrained, training assets, datasets, algorithms or Adam.
    The factory does not seed; runtime generation has an explicit separate seed.
    """
    import torch
    from peft import LoraConfig
    parameters, cls = _admit(parameters)
    with torch.device('cpu'):
        generator = cls(**parameters['architecture'])
        generator.requires_grad_(False)
        generator.add_adapter(LoraConfig(**parameters['peft']))
    generator.requires_grad_(False).eval()
    generator.disable_gradient_checkpointing()
    return _components(generator, parameters)


def refuse_training(*args, **kwargs):
    raise ValueError('Helios inference-only factory refuses training/checkpoint evaluation; use generate with exported weights')


def generate(state, feeds, parameters):
    """Only original native rollout, with exact NEW prepared prompt tensors."""
    import torch
    from distillschedule.strategies.wan_capture import native_primitives
    require(not parameters, 'native diagnostic generation has no unbound rollout override parameters')
    p, generator = state['inference'], state['generator']
    _environment()
    require(isinstance(feeds, dict) and set(feeds) == {'prompt_embeds'},
            'native inference prompt requires exactly prompt_embeds, not five training rows')
    prompt = feeds['prompt_embeds']
    require(isinstance(prompt, torch.Tensor) and list(prompt.shape) == p['geometry']['prompt_shape']
            and prompt.dtype == torch.float32 and prompt.device.type == 'cpu' and prompt.layout == torch.strided
            and bool(prompt.isfinite().all()), 'native inference prompt layout/dtype/finiteness changed')
    require(not generator.training and not generator.gradient_checkpointing
            and all(not t.requires_grad for t in generator.parameters()), 'inference G must be separately frozen with checkpointing off')
    native = native_primitives()
    # Original run_generator draws actual noise; this tensor supplies its exact
    # admitted native geometry/dtype, not fabricated GT history or a recurrence.
    noise = torch.zeros(p['geometry']['noise_shape'], dtype=torch.float32, device='cpu')
    args = SimpleNamespace(training_config=SimpleNamespace(**p['native']['args']))
    with torch.device('cpu'):
        scheduler = native.scheduler(**p['native']['scheduler'])
        result = native.post.run_generator(args=args,
            accelerator=SimpleNamespace(device=torch.device('cpu'), is_main_process=True),
            transformer=generator, scheduler=scheduler, noise=noise, prompt_embeds=prompt, **p['native']['rollout'])
    latents = result[0]
    require(isinstance(latents, torch.Tensor) and list(latents.shape) == p['geometry']['noise_shape']
            and latents.dtype == torch.float32 and not latents.requires_grad and bool(latents.isfinite().all()),
            'original native rollout produced invalid latent diagnostic')
    return {'latents':latents, 'latent_schema':deepcopy(_metadata(p)['latent_schema'])}
