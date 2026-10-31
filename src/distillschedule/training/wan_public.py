"""W1 original P0 Full5 as ONE nonreplayable Program call, CPU1 only.

This owns asset/state/admission boundaries. The existing parameterized cycle
owns dispatch; original P0 functions own every model/loss/backward/update.
No native CUDA CLI, numerical world16 or generation-quality claim is made.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import inspect
from pathlib import Path
from types import SimpleNamespace

from .adapters import ModelComponents
from .helios_public_assets import (checked_pin, local_file, read_json, read_args,
    require, runtime_identity, inspect_safe, load_base, EVIDENCE_KINDS)
from .checkpoint import cpu_tree, rng_state, restore_rng, validate_rng, _validate_layout
from .native_state import engine_state, TrainableWeights
from .helios_step_owner import _SparseEMAView, _content, _plain
from distillschedule.data.wan_reference_inputs import SCHEMA as INPUTS, SCOPE, PRECISION, topology, layout_identity
from distillschedule.workloads.best_known.registry import fingerprint, file_sha256

SCHEMA = 'distillschedule.wan_cpu_reference_assets.v1'
OWNER = 'distillschedule.wan_original_full5_cpu_owner.v1'


def _p0():
    """Intact original worker, NOT the hardened benchmark's callback overlays."""
    import importlib.util
    import sys
    from distillschedule.workloads.best_known.registry import repository_root
    from distillschedule.adapters.helios.reference_model import _SourceOnlyLoader
    path = repository_root()/'scripts/distillschedule/probe_static_fsdp2_p0_world16.py'
    name = 'distillschedule.training._wan_original_p0'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name,path,loader=_SourceOnlyLoader(name,path,file_sha256(path)))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name,None)
            raise
    return sys.modules[name]


def source_identity():
    from .helios_inference import content_identity
    from distillschedule.strategies.wan_planning import source_fingerprint
    from distillschedule.data.wan_reference_inputs import layout_identity
    return dict(native=content_identity(), framework=source_fingerprint(),layout=layout_identity())


def _live_bindings():
    """Actual original call aliases AND code, not source-file labels alone."""
    import types
    from distillschedule.workloads.best_known.registry import repository_root
    p0 = _p0()
    paths = {
        '_run_nogan_generator_update_p0':'scripts/distillschedule/probe_static_fsdp2_p0_world16.py',
        '_precompute_critic_rollouts':'scripts/distillschedule/probe_static_fsdp2_p0_world16.py',
        '_run_nogan_critic_precomputed':'scripts/distillschedule/probe_static_fsdp2_p0_world16.py',
        'run_generator_update_trace_rollout':'runtime/gst_pool_helios.py',
        'prepare_dmd_score_batch':'runtime/gst_pool_helios.py', 'score_forward':'runtime/gst_pool_helios.py',
        'construct_dmd_gradient':'runtime/gst_pool_helios.py',
        'run_generator_trace_rollout_batch':'runtime/static_fsdp2_p0_helios.py',
        'prepare_student_batch':'runtime/critic_pool_helios.py', 'student_forward_loss':'runtime/critic_pool_helios.py',
        'detached_dmd_surrogate_loss':'runtime/dmd_boundary.py'}
    def code_rows(code):
        yield code
        for item in code.co_consts:
            if isinstance(item,types.CodeType):
                yield from code_rows(item)
    result, compiled_paths = {}, {}
    for name,relative in paths.items():
        function = getattr(p0,name)
        # no_grad wraps two helpers with a Torch contextmanager; the wrapped
        # original body is still checked against its admitted source code.
        body = inspect.unwrap(function)
        path = repository_root()/relative
        require(inspect.isfunction(body) and Path(body.__code__.co_filename).resolve() == path.resolve(),
                'original P0 numerical binding redirected: '+name)
        if path not in compiled_paths:
            compiled_paths[path] = tuple(code_rows(compile(path.read_bytes(),str(path),'exec',dont_inherit=True)))
        codes = [c for c in compiled_paths[path]
                 if c.co_name == body.__code__.co_name and c.co_firstlineno == body.__code__.co_firstlineno]
        require(len(codes) == 1 and codes[0] == body.__code__,
                'original P0 live code differs from admitted source: '+name)
        result[name] = (function,body.__code__)
    return result


def admit_assets(parameters):
    from distillschedule.adapters.helios.reference_model import reference_model_identity
    from distillschedule.strategies.wan_task import OriginalWanTask
    topology()
    layout_identity()
    require(set(parameters) <= {'asset_manifest', 'seed', 'device','inference_noise_policy'} and parameters.get('device', 'cpu') == 'cpu',
            'W1 CPU reference only; no device/parameter substitution')
    from .reference_inference_noise import enabled
    enabled(parameters)
    path = local_file(parameters.get('asset_manifest'))
    value = read_json(path)
    require(set(value) == {'schema_version','scope','precision','evidence_kind','source','runtime','args','task',
                         'negative','generator','critic','geometry','budget','input_config'}
            and value['schema_version'] == SCHEMA and value['scope'] == SCOPE and value['precision'] == PRECISION
            and value['evidence_kind'] in EVIDENCE_KINDS, 'unsupported W1 assets/precision/evidence')
    require(value['source'] == reference_model_identity(backend='torch-sdpa')
            and value['runtime'] == runtime_identity(), 'W1 source/runtime changed')
    task = OriginalWanTask.from_dict(value['task'])
    require(task.first_step == 0 and task.fake_updates == 5 and task.stage_schedule_policy == 'source_normal_only',
            'W1 cold-zero original normal Full5 required; no Step10 or W2 relabel')
    args = read_args(value['args'])
    c, m = args.training_config, args.model_config
    from distillschedule.strategies.wan_capture import read_training_config
    from distillschedule.data.wan_reference_inputs import input_policy
    from omegaconf import OmegaConf
    require(input_policy(read_training_config(checked_pin(value['input_config']))) == input_policy(OmegaConf.to_container(c,resolve=True)),
            'actual native source history/timestep input policy differs from original Args')
    require(c.mixed_precision == 'bf16' and c.train_batch_size == c.gradient_accumulation_steps == 1
            and c.is_train_dmd and not c.is_use_gan and c.use_ema and c.dfake_gen_update_ratio == 5
            and c.is_enable_stage2 and c.stage2_num_stages == 3 and c.is_use_gt_history
            and c.num_critic_input_frames == task.latent_shape[2] and list(c.history_sizes) == [16,2,1]
            and c.zero_history_timestep and c.guidance_cross_attn and c.has_multi_term_memory_patch
            and c.gradient_checkpointing and c.optimizer in ('adam', 'adamw')
            and not c.use_8bit_adam and not c.scale_lr
            and type(c.max_train_steps) is int and c.max_train_steps > 0 and c.max_train_steps % 5 == 0,
            'W1 original precision/batch/optimizer/branch/geometry changed')
    disabled = ('restrict_self_attn','restrict_lora','is_train_restrict_lora','is_amplify_history',
        'is_use_reward_model','is_use_ode_regression','is_only_ode_regression','is_multi_pyramid_stage_backward_simulated',
        'is_dmd_vae_decode','is_navit_pyramid','dmd_teacher_forcing','is_enable_cold_start','use_error_recycling')
    require(not any(getattr(c,k) for k in disabled) and not args.validation_config.use_kv_cache
            and c.dmd_num_latent_sections_min == c.dmd_num_latent_sections_max == 1
            and list(args.validation_config.stage2_simulated_inference_steps) == list(task.stage2_inference_steps)
            and list(c.dmd_denoising_step_list) == list(task.denoising_step_list)
            and c.stage2_timestep_shift == 1. and list(c.stage2_stage_range) == [0,1/3,2/3,1]
            and c.stage2_scheduler_gamma == 1/3 and c.dmd_timestep_shift == 5.
            and c.use_dynamic_shifting and c.time_shift_type == 'linear'
            and [c.base_seq_len,c.max_seq_len,c.base_shift,c.max_shift] == [256,4096,.5,1.15]
            and c.fake_guidance_scale == 0. and c.real_guidance_scale == 3., 'unsupported P0 numerical controls')
    require(m.lora_rank == m.critic_lora_rank == task.lora_rank
            and m.lora_alpha == m.critic_lora_alpha == task.lora_rank
            and m.lora_dropout == m.critic_lora_dropout == 0. and not m.train_norm_layers,
            'native all-linear/pyramid LoRA ownership changed')
    geometry = value['geometry']
    require(set(geometry) == {'noise_shape','prompt_shape'} and geometry['noise_shape'] == list(task.latent_shape)
            and geometry['prompt_shape'][:1] == [1] and len(geometry['prompt_shape']) == 3
            and geometry['prompt_shape'][-1] == 4096 and geometry['prompt_shape'][1] > 0
            and task.latent_shape[-2] % 16 == task.latent_shape[-1] % 16 == 0, 'native geometry mismatch')
    budget = value['budget']
    require(set(budget) == {'max_parameters','max_loaded_bytes'}
            and type(budget['max_parameters']) is int and 0 < budget['max_parameters'] <= 100000000
            and type(budget['max_loaded_bytes']) is int and 0 < budget['max_loaded_bytes'] <= 2**31,
            'explicit bounded CPU budget required')
    count, loaded, paths = 0, 0, []
    import math
    for role in ('generator','critic'):
        spec = value[role]
        require(set(spec) == {'config','weights'} and isinstance(spec['weights'],list) and spec['weights'], 'all role shards required')
        checked_pin(spec['config'])
        keys = set()
        for pin in spec['weights']:
            weight, layout = inspect_safe(pin, evidence_kind=value['evidence_kind'], precision=PRECISION)
            require(not keys & layout.keys(), 'duplicate native base keys')
            keys.update(layout); count += sum(math.prod(shape) for shape in layout.values())
            loaded += pin['bytes']; paths.append(str(weight))
        require(all(f'{name}.weight' in keys and f'{name}.bias' in keys for name in ('patch_short','patch_mid','patch_long')),
                'missing pyramid weights; random fallback forbidden')
    from safetensors import safe_open
    with safe_open(checked_pin(value['negative']),framework='pt',device='cpu') as handle:
        require((handle.metadata() or {}).get('evidence_kind') == value['evidence_kind'] and list(handle.keys()) == ['negative_prompt_embeds']
                and handle.get_slice('negative_prompt_embeds').get_dtype() == 'BF16'
                and handle.get_slice('negative_prompt_embeds').get_shape() == geometry['prompt_shape'], 'native negative precision/shape changed')
    require(len(paths) == len(set(paths)) and count <= budget['max_parameters']
            and loaded + value['negative']['bytes'] <= budget['max_loaded_bytes'], 'native base budget/path reuse')
    return path, value, args, task


def build_model(parameters):
    from peft import LoraConfig
    from safetensors.torch import load_file
    from diffusers.optimization import get_scheduler
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from distillschedule.strategies.wan_capture import native_primitives
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
    from runtime.sparse_ema import SparseShardEMA
    from runtime.critic_pool_helios import _all_linear_targets, TRANSFORMER_ADDITIONAL_KWARGS
    path, assets, args, task = admit_assets(parameters)
    cls, engines = load_reference_model_class(backend='torch-sdpa'), {}
    remaining = assets['budget']['max_parameters']
    c = args.training_config
    for role in ('generator','critic'):
        model = load_base(cls, assets[role], evidence_kind=assets['evidence_kind'],
                          remaining_parameters=remaining, precision=PRECISION)
        remaining -= sum(p.numel() for p in model.parameters())
        require(all(getattr(model.config,k) == v for k,v in TRANSFORMER_ADDITIONAL_KWARGS.items())
                and model.config.text_dim == 4096 and model.config.in_channels == model.config.out_channels == 16
                and tuple(model.config.patch_size) == (1,2,2)
                and not any(getattr(model.config,k) for k in ('is_use_gan','is_use_gan_final','is_use_gan_hooks')),
                'not the complete original W1 architecture/conditioning branch')
        model.requires_grad_(False)
        targets, exclude = _all_linear_targets(model), ['down','up']
        if role == 'generator':
            targets += ['patch_short','patch_mid','patch_long']
        else:
            patches = ['patch_short','patch_mid','patch_long','patch_embedding']
            targets = [n for n in targets if n not in patches]
            exclude += patches + ['gan_heads','gan_final_head']
        # PEFT Conv3D adapters for all three ORIGINAL G pyramid patches are real
        # trainables, not an omitted/fake projection. Bound before adding them.
        bound = sum(task.lora_rank*(layer.weight[0].numel()+layer.weight.shape[0])
                    for name,layer in model.named_modules() if name in targets)
        require(0 < bound <= remaining, 'native LoRA budget exceeded')
        remaining -= bound
        model.add_adapter(LoraConfig(r=task.lora_rank,lora_alpha=float(task.lora_rank),lora_dropout=0.,
            init_lora_weights='gaussian',target_modules=sorted(set(targets)),exclude_modules=exclude))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        model.enable_gradient_checkpointing()
        model.train()
        rate = c.learning_rate if role == 'generator' else c.critic_learning_rate
        optimizer = native_primitives(source_only=True).base.get_optimizer(args, SimpleNamespace(print=lambda *a,**kw:None),
            [{'params':[p for p in model.parameters() if p.requires_grad], 'lr':rate}], use_deepspeed=False)
        scheduler = get_scheduler(c.lr_scheduler, optimizer=optimizer, num_warmup_steps=c.lr_warmup_steps,
            num_training_steps=c.max_train_steps,num_cycles=c.lr_num_cycles,power=c.lr_power)
        engines[role] = FSDP2EngineAdapter(model,optimizer,scheduler,physical_step=0,global_batch=1,dp_world_size=1,
            gradient_clip=c.max_grad_norm if role == 'generator' else c.max_grad_norm_critic)
    g = engines['generator']
    ema = SparseShardEMA(g.module.named_parameters(),decay=c.ema_decay,update_after_step=c.ema_start_step,
                        base_model=assets['generator']['weights'][0]['sha256'])
    negative = load_file(checked_pin(assets['negative']), device='cpu')['negative_prompt_embeds']
    require(bool(negative.isfinite().all()), 'nonfinite negative prompt')
    owner = WanCycleOwner(engines,ema,args,task,negative)
    metadata = dict(schema_version=SCHEMA,scope=SCOPE,precision=PRECISION,assets=deepcopy(assets),
        asset_manifest=dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha256(path)),
        owner_identity=deepcopy(owner.identity),evidence_kind=assets['evidence_kind'],quality_validation='NOT_RUN',claimable_speedup=False)
    from .wan_inference import project_current_generator
    return ModelComponents(state={'native_cycle':owner}, metadata=metadata, admit_training=admit_training,
        collect_metrics=collect_metrics,inference_projection=project_current_generator).validate()


class _BoundEMA:
    """Only bind the existing sparse-EMA parameters to the cycle's no-arg ABI."""
    def __init__(self, ema, generator):
        self.ema, self.generator = ema, generator
    @property
    def optimization_step(self):
        return self.ema.optimization_step
    def step(self):
        return self.ema.step(self.generator.module.parameters())


class _WanWeights(TrainableWeights):
    """Stage canonical boundary flags even if P0 failed during Teacher mode.

PEFT's ORIGINAL disable_adapters temporarily masks C's LoRA requires_grad.
The shared serialization view intentionally refuses mask changes; only this
W1 owner admits that source-defined partial-cycle state for checkpoint recovery.
"""
    def __init__(self,engine):
        super().__init__(engine)
        value = self.state_dict()
        self.flags, self.modes = value['requires_grad'], value['training']

    def validate_state_dict(self,value):
        flags = value['requires_grad']
        require(isinstance(flags,dict) and flags.keys() == self.flags.keys()
                and all(type(flag) is bool for flag in flags.values()),
                'W1 checkpoint trainability flags must be canonical booleans')
        require(value['requires_grad'] == self.flags and value['training'] == self.modes,
                'W1 checkpoint trainability/module modes are not a complete original boundary')
        super().validate_state_dict({**value,'requires_grad':self.state_dict()['requires_grad']})

    def load_state_dict(self,value):
        super().load_state_dict(value)
        for name,parameter in self.engine.module.named_parameters():
            parameter.requires_grad_(value['requires_grad'][name])


class WanCycleOwner:
    def __init__(self, engines, ema, args, task, negative):
        from omegaconf import OmegaConf
        self.generator, self.critic = engines['generator'], engines['critic']
        self._engines = engines
        self.ema, self.args, self.task, self.negative = ema, args, task, negative
        self.cycle, self.poisoned, self.consumed = 0, False, []
        self._stream = None
        self._rng = cpu_tree(rng_state())
        self._bindings = _live_bindings()
        self._views = engine_state(engines)
        for role,engine in engines.items():
            self._views[role+'.weights'] = _WanWeights(engine)
        self._views['ema'] = _SparseEMAView(ema,self.generator)
        self._ownership = {role:{name:id(p) for name,p in engine.module.named_parameters()} for role,engine in engines.items()}
        self.identity = dict(source=source_identity(), args=OmegaConf.to_container(args,resolve=True), task=task.to_dict(),negative=_content(negative),
            models={role:dict(config=_plain(dict(engine.module.config)),
                frozen=_content({n:p.detach() for n,p in engine.module.named_parameters() if not p.requires_grad}),
                trainables={n:dict(shape=list(p.shape),dtype=str(p.dtype)) for n,p in engine.module.named_parameters() if p.requires_grad},
                buffers=_content(dict(engine.module.named_buffers()))) for role,engine in engines.items()})
        self._policies = {role:self._policy(engine) for role,engine in engines.items()}
        self._adapter_policy = self._adapters(policy=True)
        self._calls = self._engine_calls()
        require(self.versions == dict(generator=0,critic=0,ema=0)
                and not self.generator.optimizer.state and not self.critic.optimizer.state, 'true zero Adam/EMA start required')
        self._validate_live()

    @property
    def capacity(self):
        return self.args.training_config.max_train_steps // 5

    @property
    def versions(self):
        return dict(generator=self.generator.global_steps,critic=self.critic.global_steps,ema=self.ema.optimization_step)

    def expected(self, cycle):
        return dict(generator=cycle,critic=5*cycle,ema=sum(len(replace(self.task,first_step=5*i).ema_after) for i in range(cycle)))

    @staticmethod
    def _policy(engine):
        return dict(optimizer=type(engine.optimizer).__module__+'.'+type(engine.optimizer).__qualname__,
            groups=[{k:v for k,v in group.items() if k not in ('params','lr')} for group in engine.optimizer.param_groups],
            scheduler={k:v for k,v in engine.lr_scheduler.state_dict().items()
                       if k not in ('last_epoch','_step_count','_last_lr')})

    def _engine_calls(self):
        result = {}
        for role,engine in self._engines.items():
            for name,function in (('backward',engine.backward),('commit',engine.step),
                ('adam',engine.optimizer.step),('lr',engine.lr_scheduler.step)):
                body = getattr(function,'__func__',function)
                result[role+'.'+name] = (body,getattr(body,'__code__',None))
        return result

    def _adapters(self, *, policy=False):
        from peft.tuners.tuners_utils import BaseTunerLayer
        result = {}
        for role,engine in self._engines.items():
            model = engine.module
            layers = {name:dict(active=list(layer.active_adapters),disabled=layer.disable_adapters,
                scaling=_plain(layer.scaling),rank=_plain(layer.r),alpha=_plain(layer.lora_alpha))
                for name,layer in model.named_modules() if isinstance(layer,BaseTunerLayer)}
            require(layers and set(model.peft_config) == {'default'}, 'original single PEFT adapter required')
            if not policy:
                require(list(model.active_adapters()) == ['default'] and all(v['active'] == ['default']
                        and v['disabled'] is False for v in layers.values()), 'native adapter enabled/selection boundary changed')
            result[role] = dict(config={n:_plain(c.to_dict()) for n,c in model.peft_config.items()},
                layers={n:{k:v for k,v in item.items() if k not in ('active','disabled')}
                        for n,item in layers.items()})
        return result

    def _validate_identity(self):
        import torch
        from omegaconf import OmegaConf
        from distillschedule.adapters.helios.reference_model import load_reference_model_class
        topology()
        require(self._bindings == _live_bindings(), 'original P0 live callback binding changed')
        require(self._calls == self._engine_calls() and self._adapter_policy == self._adapters(policy=True),
                'original engine callback/PEFT configuration/scaling changed')
        require(self.identity['source'] == source_identity() and self.identity['args'] == OmegaConf.to_container(self.args,resolve=True)
                and self.identity['task'] == self.task.to_dict() and self.identity['negative'] == _content(self.negative),
                'original W1 source/config/negative binding changed')
        cls = load_reference_model_class(backend='torch-sdpa')
        for role,engine in (('generator',self.generator),('critic',self.critic)):
            model = engine.module; identity = self.identity['models'][role]
            selected = set(identity['trainables'])
            require(type(model) is cls and engine.dp_world_size == engine.train_batch_size() == 1
                    and self._ownership[role] == {n:id(p) for n,p in model.named_parameters()}
                    and identity['config'] == _plain(dict(model.config))
                    and identity['frozen'] == _content({n:p.detach() for n,p in model.named_parameters() if n not in selected})
                    and identity['buffers'] == _content(dict(model.named_buffers()))
                    and identity['trainables'] == {n:dict(shape=list(p.shape),dtype=str(p.dtype)) for n,p in model.named_parameters() if n in selected}
                    and all(not p.requires_grad for n,p in model.named_parameters() if n not in selected)
                    and (self.poisoned or all(p.requires_grad for n,p in model.named_parameters() if n in selected))
                    and all(p.device.type == 'cpu' for p in model.parameters())
                    and all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
                    and all(not getattr(m,'merged',False) for m in model.modules())
                    and self._policies[role] == self._policy(engine)
                    and 'forward' not in vars(model) and model.gradient_checkpointing
                    and all(m.training for m in model.modules()), 'W1 live model/parameter/precision/optimizer ownership changed')
            for block in model.blocks:
                require(block.attn1.processor.kv_cache is None and not block.attn1.processor.cache_enabled,
                        'W1 KV/cache boundary changed')

    def _validate_native(self, native, cycle):
        import torch
        expected = self.expected(cycle)
        require(native.keys() == self._views.keys(), 'native owner inventory changed')
        for key,view in self._views.items():
            check = getattr(view,'validate_state_dict',None)
            if check:
                check(native[key])
            else:
                _validate_layout(view.state_dict(), native[key], key)
        for role,engine in (('generator',self.generator),('critic',self.critic)):
            step = expected[role]; clocks = native[role+'.clocks']
            require(all(value == step for value in clocks['counters'].values())
                    and native[role+'.scheduler']['last_epoch'] == step
                    and native[role+'.scheduler']['_step_count'] == step+1, 'native engine/scheduler clock changed')
            optimizer = native[role+'.optimizer']['optimizer']
            require([{k:v for k,v in group.items() if k not in ('params','lr')} for group in optimizer['param_groups']]
                    == self._policies[role]['groups'], 'native Adam policy changed')
            require({k:v for k,v in native[role+'.scheduler'].items() if k not in ('last_epoch','_step_count','_last_lr')}
                    == self._policies[role]['scheduler'], 'native LR scheduler policy changed')
            require([g['lr'] for g in optimizer['param_groups']] == native[role+'.scheduler']['_last_lr'], 'Adam/LR state differs')
            states = optimizer['state']
            names = native[role+'.optimizer']['parameters'][0]
            require((not states and step == 0) or set(states) == set(optimizer['param_groups'][0]['params']), 'missing native Adam state')
            for state in states.values():
                require(set(state) == {'step','exp_avg','exp_avg_sq'} and state['step'].item() == step
                        and state['step'].dtype == torch.float32 and state['step'].ndim == 0
                        and all(v.isfinite().all().item() for v in state.values()), 'Adam clock/moment state changed')
            require(len(names) > 0 and all(v.isfinite().all().item() for v in native[role+'.weights']['tensors'].values())
                    and _content(native[role+'.weights']['buffers']) == self.identity['models'][role]['buffers'],
                    'nonfinite trainables or changed immutable native buffers')
        require(native['ema']['optimization_step'] == expected['ema'], 'native EMA clock changed')

    def _validate_live(self):
        require(not self.poisoned, 'incomplete native cycle poisoned; restore a complete checkpoint before continuing')
        self._validate_identity()
        self._adapters()
        require(self.versions == self.expected(self.cycle), 'native cycle versions changed')
        require(self._stream is None or self.consumed == self._stream[:self.cycle], 'consumed original stream prefix changed')
        for role in ('generator','critic'):
            self._views[role+'.clocks'].validate_checkpoint_boundary()
        self._validate_native({k:v.state_dict() for k,v in self._views.items()}, self.cycle)

    @contextmanager
    def _random_scope(self):
        outer = rng_state()
        try:
            restore_rng(self._rng)
            yield
        finally:
            self._rng = cpu_tree(rng_state())
            restore_rng(outer)

    def run(self, trace, descriptor):
        from distillschedule.runtime.wan_cycle_plan import prepare_parameterized_wan_cycle
        from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
        from distillschedule.profiling.static_profile_diagnostics import StaticProfileDiagnostics
        self._validate_live()
        require(self.cycle < self.capacity and trace.logical_steps == replace(self.task,first_step=5*self.cycle).source_steps
                and descriptor not in self.consumed
                and (self._stream is None or descriptor == self._stream[self.cycle])
                and all(row['content'] != descriptor['content'] for row in self.consumed), 'trace/source replay or cycle reordered')
        trace.validate_cpu_reference()
        task = replace(self.task,first_step=5*self.cycle)
        prepared = prepare_parameterized_wan_cycle(p0=_p0(),source_steps=task.source_steps,ema_after=task.ema_after,
            trace_contract=task.trace_contract(),max_live_rollouts=1,rank=0,device=__import__('torch').device('cpu'),
            generator_engine=self.generator,critic_engine=self.critic,ema=_BoundEMA(self.ema,self.generator),trace=trace,
            negative=self.negative,diagnostics=StaticProfileDiagnostics(enabled=False,rank=0),policy=StaticFSDP2P0Policy(),
            clean_runtime=False,runtime_cache=None,deferred_finite_checks=None,collect_scalars=True)
        # Once any numerical action can start, failures poison ALL owners. No
        # rollback fiction, no retry/replay of an already committed G or C.
        self.poisoned = True
        with self._random_scope():
            result = prepared.run()
        self.cycle += 1; self.consumed.append(deepcopy(descriptor)); self.poisoned = False
        self._validate_live()
        return dict(cycle=self.cycle,versions=self.versions,native=result,dispatch=prepared.receipt())

    def state_dict(self):
        self._validate_live()
        return dict(schema=OWNER,identity=deepcopy(self.identity),cycle=self.cycle,versions=self.versions,adapters=deepcopy(self._adapter_policy),
                    consumed=deepcopy(self.consumed),rng=cpu_tree(self._rng),native={k:cpu_tree(v.state_dict()) for k,v in self._views.items()})

    def validate_checkpoint_boundary(self):
        self._validate_live()
        require(self.cycle > 0, 'complete original Full5 cycle required')

    def validate_state_dict(self, value):
        # Identity/layout check deliberately also works while a partial cycle is
        # poisoned. ALL decoded owners are staged before the first copy below.
        self._validate_identity()
        require(set(value) == {'schema','identity','cycle','versions','consumed','rng','native','adapters'}
                and value['schema'] == OWNER and value['identity'] == self.identity
                and value['adapters'] == self._adapter_policy
                and type(value['cycle']) is int and 0 < value['cycle'] <= self.capacity
                and value['versions'] == self.expected(value['cycle'])
                and all(type(v) is int for v in value['versions'].values()), 'source/boundary/version checkpoint mismatch')
        consumed = value['consumed']
        require(isinstance(consumed,list) and len(consumed) == value['cycle']
                and len({r['control_hash'] for r in consumed}) == len(consumed)
                and len({fingerprint(r['content']) for r in consumed}) == len(consumed), 'native cursor/source replay checkpoint mismatch')
        require(self._stream is None or consumed == self._stream[:value['cycle']], 'checkpoint consumed original source prefix changed')
        validate_rng(value['rng'])
        self._validate_native(value['native'],value['cycle'])

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        # Torch can retain decoded CPU Adam tensors by reference. A restored
        # optimizer must never mutate the caller's preceding complete snapshot
        # when the next native cycle commits (including another failed cycle).
        value = cpu_tree(value)
        self.poisoned = True
        for key,view in self._views.items():
            view.load_state_dict(value['native'][key])
        for engine in self._engines.values():
            engine.module.set_adapter('default')
            engine.module.enable_adapters()
            engine.optimizer.zero_grad(set_to_none=True)
        self.cycle, self.consumed, self._rng = value['cycle'],deepcopy(value['consumed']),cpu_tree(value['rng'])
        self.poisoned = False
        self._validate_live()

    def validate_checkpoint_progress(self, value, progress):
        self.validate_state_dict(value)
        require(progress.get('cycle') == progress.get('cursor') == value['cycle']
                and type(progress.get('cycle')) is type(progress.get('cursor')) is int
                and progress.get('versions') == {'native_cycle':value['cycle']}
                and type(progress['versions']['native_cycle']) is int, 'inner/outer W1 cursor changed')


def admit_training(*,state,request,prepared_identity,prepared_manifest):
    owner = state['native_cycle']; owner._validate_live()
    require(dict(request.execution) == dict(adapter='program',mode='reference')
            and request.trainer.get('device','cpu') == 'cpu' and not request.trainer.get('distributed')
            and not request.trainer.get('validation') and request.data.get('sharding','single') == 'single'
            and request.data.get('sampling',dict(kind='finite',shuffle=False)) == dict(kind='finite',shuffle=False),
            'W1 CPU1 finite reference only; no auto/plan/distributed/validation')
    require(request.algorithm['factory'] == 'distillschedule.training.wan_public:build_algorithm'
            and not request.algorithm.get('parameters'), 'original single atomic cycle required')
    _,assets,_,_ = admit_assets(request.model.get('parameters',{}))
    meta = prepared_manifest['metadata']
    require(meta == dict(schema_version=INPUTS,scope=SCOPE,precision=PRECISION,
        source=owner.identity['source']['framework'],task=owner.task.to_dict(),evidence_kind=assets['evidence_kind'],
        input_config=assets['input_config'],layout=owner.identity['source']['layout']),
        'W1 prepared source/task/precision/evidence changed')
    from distillschedule.data.manifest import validate_source
    for pin in prepared_manifest['provenance']['documents']:
        validate_source(pin)
    records = prepared_manifest['records']
    require(request.trainer['cycles'] <= len(records) <= owner.capacity, 'finite trace inventory/horizon exhausted')
    hashes, contents = set(), set()
    for index,record in enumerate(records):
        metadata = record['metadata']; row = metadata['trace']
        require(metadata['cycle'] == index and row['control_hash'] not in hashes and fingerprint(row['content']) not in contents,
                'prepared trace/control/source reused or reordered')
        hashes.add(row['control_hash']); contents.add(fingerprint(row['content']))
    expected = [deepcopy(row['metadata']['trace']) for row in records]
    require(owner._stream is None and isinstance(prepared_identity,str) and len(prepared_identity) == 64,
            'finite original prepared stream must be admitted exactly once')
    owner._stream = expected
    owner.identity['stream'] = dict(prepared_identity=prepared_identity,cohorts=expected)


def native_cycle(ctx):
    from distillschedule.data.wan_reference_inputs import decode, descriptor
    owner = ctx.state['native_cycle']
    trace = decode(ctx.inputs,replace(owner.task,first_step=5*ctx.attributes['cycle']))
    return owner.run(trace,descriptor(trace.reference_document))


def build_algorithm(components,feeds,parameters,cycle,seed):
    from distillschedule.open_workload.torch_backend import Workload
    from distillschedule.workloads.program import Program,Call
    from distillschedule.data.wan_reference_inputs import decode,descriptor
    require(not parameters and type(cycle) is int and cycle >= 0 and set(components.state) == {'native_cycle'}, 'original atomic W1 owner required')
    owner = components.state['native_cycle']; owner._validate_live()
    require(owner.cycle == cycle, 'original W1 cycle reordered')
    checked_pin(components.metadata['asset_manifest'])
    assets = components.metadata['assets']
    pins = [assets['args'],assets['negative'],assets['input_config']]
    for role in ('generator','critic'):
        pins.extend([assets[role]['config'],*assets[role]['weights']])
    for pin in pins:
        checked_pin(pin)
    trace = decode(feeds,replace(owner.task,first_step=5*cycle))
    row = descriptor(trace.reference_document)
    require(row not in owner.consumed and all(row['content'] != old['content'] for old in owner.consumed), 'original source/trace replay')
    call = Call(id='wan.native_cycle',binding='original_full5',inputs=('control','control_length','tensors'),reads=('native_cycle',),
        writes=('native_cycle',),grad='external_grad',replay_safe=False,attributes=dict(cycle=cycle))
    return Workload(program=Program((call,),initial_versions={'native_cycle':cycle},feeds=('control','control_length','tensors'),outputs=(call.id,)),
        bindings={'original_full5':native_cycle},state=components.state,feeds=feeds,seed=seed,identity_extra={'scope':SCOPE})


def collect_metrics(outputs):
    require(set(outputs) == {'wan.native_cycle'}, 'one complete atomic native cycle required')
    result = outputs['wan.native_cycle']['native']
    return {'generator/loss':result['generator']['surrogate_loss'],
            **{f'critic/loss_{i+1}':row['loss'] for i,row in enumerate(result['critic'])}}
