"""Independent W2 warm No-GAN owner over the ORIGINAL complete P0 cycle.

CPU1/source-rank0 diagnostics only. One external-grad Program call owns G1,
five independent post-G B1 rollouts, C5 and native sparse EMA5. Current full-G
export is a separate inference projection, not a distributed/quality claim.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

from .adapters import ModelComponents
from .checkpoint import cpu_tree, rng_state, restore_rng, validate_rng, _validate_layout
from .helios_public_assets import checked_pin, local_file, read_json, read_args, require
from .helios_step_owner import _SparseEMAView, _content, _plain
from .native_state import engine_state, TrainableWeights
from . import helios_nogan_bootstrap as bootstrap
from distillschedule.data.helios_nogan_reference_inputs import (SCHEMA as INPUTS, SCOPE, PRECISION,
    HeliosCohortTask, decode, descriptor, conditioning_fingerprint, topology, layout_identity, source_fingerprint)
from distillschedule.workloads.best_known.registry import file_sha256

SCHEMA = 'distillschedule.helios_nogan_cpu_reference_assets.v1'
OWNER = 'distillschedule.helios_nogan_original_warm_cpu_owner.v1'


def plain(value):
    """D1 identity is recursively frozen, not directly JSON serializable."""
    if isinstance(value,Mapping):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [plain(v) for v in value]
    return _plain(value)


def _p0():
    # Source-only original helper loading/auditing is shared, NOT W1's owner,
    # cold clocks, task admission or its parameterized numerical dispatcher.
    from .wan_public import _p0 as source_module
    return source_module()


def _live_bindings():
    import inspect
    from pathlib import Path
    import types
    from .wan_public import _live_bindings as audit_original_helpers
    from distillschedule.runtime.helios_planner_cycle import _Observer, _prepare_cycle
    result = audit_original_helpers()
    def codes(code):
        yield code
        for value in code.co_consts:
            if isinstance(value,types.CodeType):
                yield from codes(value)
    functions = {'prepare':_prepare_cycle,**{name:getattr(_Observer,name) for name in
        ('__init__','installed','action','finish','entry','versions','version_snapshot','memory','produced','consumed')}}
    compiled = {}
    for name,function in functions.items():
        body = inspect.unwrap(function)
        path = Path(__file__).resolve().parents[1]/'runtime/helios_planner_cycle.py'
        require(inspect.isfunction(body) and Path(body.__code__.co_filename).resolve() == path,
                'W2 original observer binding redirected: '+name)
        if path not in compiled:
            compiled[path] = tuple(codes(compile(path.read_bytes(),str(path),'exec',dont_inherit=True)))
        matches = [c for c in compiled[path] if c.co_name == body.__code__.co_name and c.co_firstlineno == body.__code__.co_firstlineno]
        require(len(matches) == 1 and matches[0] == body.__code__, 'W2 observer live code differs from source: '+name)
        result['observer.'+name] = (function,body.__code__)
    return result


def source_identity():
    return dict(native=bootstrap.source_identity(),framework=source_fingerprint(),layout=layout_identity())


def admit_assets(parameters):
    from omegaconf import OmegaConf
    from distillschedule.strategies.wan_capture import read_training_config
    from distillschedule.data.wan_reference_inputs import input_policy
    topology(); layout_identity()
    require(set(parameters) <= {'asset_manifest','seed','device'} and parameters.get('device','cpu') == 'cpu',
            'W2 only admits explicit CPU1 assets')
    path = local_file(parameters.get('asset_manifest'))
    value = read_json(path)
    require(set(value) == {'schema_version','scope','precision','bootstrap','args','input_config'}
            and value['schema_version'] == SCHEMA and value['scope'] == SCOPE and value['precision'] == PRECISION,
            'exact independent W2 public asset envelope required')
    warm = read_json(checked_pin(value['bootstrap']))
    proof = read_json(checked_pin(warm['provenance']))
    require(proof['evidence_kind'] in (bootstrap.ORIGINAL,bootstrap.CPU_WARM)
            and proof['update_source'] == 'ORIGINAL_W2_HELPERS', 'state-reader fixture is not original W2 warm update evidence')
    args = read_args(value['args']); c,m = args.training_config,args.model_config
    require(input_policy(read_training_config(checked_pin(value['input_config']))) ==
            input_policy(OmegaConf.to_container(c,resolve=True)), 'W2 input capture policy differs from original Args')
    require(c.mixed_precision == 'bf16' and c.train_batch_size == c.gradient_accumulation_steps == 1
            and c.is_train_dmd and not c.is_use_gan and c.use_ema and c.dfake_gen_update_ratio == 5
            and c.is_enable_stage2 and c.stage2_num_stages == 3 and c.is_use_gt_history
            and c.num_critic_input_frames == 9 and list(c.history_sizes) == [16,2,1]
            and c.zero_history_timestep and c.guidance_cross_attn and c.has_multi_term_memory_patch
            and c.gradient_checkpointing and c.optimizer == 'adamw' and not c.use_8bit_adam and not c.scale_lr
            and [c.learning_rate,c.critic_learning_rate,c.adam_beta1,c.adam_beta2,c.adam_epsilon,c.adam_weight_decay]
                == [2e-6,4e-7,0.,.999,1e-8,1e-3]
            and c.max_grad_norm == c.max_grad_norm_critic == 1. and c.lr_scheduler == 'constant'
            and c.ema_decay == .99 and c.ema_start_step == 750
            and type(c.max_train_steps) is int and 20 <= c.max_train_steps <= 330 and c.max_train_steps % 5 == 0,
            'W2 original optimizer/precision/batch/EMA/horizon controls changed')
    disabled = ('restrict_self_attn','restrict_lora','is_train_restrict_lora','is_amplify_history',
        'is_use_reward_model','is_use_ode_regression','is_only_ode_regression','is_multi_pyramid_stage_backward_simulated',
        'is_dmd_vae_decode','is_navit_pyramid','dmd_teacher_forcing','is_enable_cold_start','use_error_recycling')
    require(not any(getattr(c,k) for k in disabled) and not args.validation_config.use_kv_cache
            and c.dmd_num_latent_sections_min == c.dmd_num_latent_sections_max == 1
            and list(args.validation_config.stage2_simulated_inference_steps) == [2,2,2]
            and list(c.dmd_denoising_step_list) == [1000,750,500,250]
            and c.stage2_timestep_shift == 1. and list(c.stage2_stage_range) == [0,1/3,2/3,1]
            and c.stage2_scheduler_gamma == 1/3 and c.dmd_timestep_shift == 5.
            and c.use_dynamic_shifting and c.time_shift_type == 'linear'
            and [c.base_seq_len,c.max_seq_len,c.base_shift,c.max_shift] == [256,4096,.5,1.15]
            and c.fake_guidance_scale == 0. and c.real_guidance_scale == 3.
            and m.lora_rank == m.critic_lora_rank == 256 and m.lora_alpha == m.critic_lora_alpha == 256.
            and m.lora_dropout == m.critic_lora_dropout == 0. and not m.train_norm_layers,
            'W2 original No-GAN normal33/222/LoRA256 branch changed')
    return path,value,args


def build_model(parameters):
    from .helios_nogan_inference import project_current_generator
    from runtime.gst_pool_contract import NegativePromptAsset
    path,assets,args = admit_assets(parameters)
    warm = bootstrap.materialize_bootstrap(checked_pin(assets['bootstrap']))
    identity = plain(warm.identity)
    negative = NegativePromptAsset.load(identity['assets']['native_assets']['assets']['negative_prompt']['path']).embedding
    owner = HeliosNoGANCycleOwner(warm,args,negative)
    metadata = dict(schema_version=SCHEMA,scope=SCOPE,precision=PRECISION,assets=assets,
        asset_manifest=dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha256(path)),
        bootstrap=identity,owner_identity=deepcopy(owner.identity),
        quality_validation='NOT_RUN',distributed_execution='NOT_RUN',claimable_speedup=False,
        inference_export='CURRENT_FULL_UNMERGED_G_NATIVE_ZERO_HISTORY_LATENTS_ONLY')
    return ModelComponents(state={'native_cycle':owner},metadata=metadata,admit_training=admit_training,
                           collect_metrics=collect_metrics,inference_projection=project_current_generator).validate()


class _BoundEMA:
    def __init__(self,warm):
        self.warm = warm
    @property
    def optimization_step(self):
        return self.warm.ema.optimization_step
    def step(self):
        return self.warm.step_ema()


class _BoundaryWeights(TrainableWeights):
    """Canonical boundary flags can be restored after a failed Teacher call."""
    def __init__(self,engine):
        super().__init__(engine)
        value = self.state_dict()
        self.flags,self.modes = value['requires_grad'],value['training']
    def validate_state_dict(self,value):
        require(value.keys() == self.state_dict().keys()
                and value['requires_grad'] == self.flags and value['training'] == self.modes
                and all(type(v) is bool for v in value['requires_grad'].values())
                and all(type(v) is bool for v in value['training'].values()), 'W2 complete boundary flags/modes required')
        _validate_layout(self.state_dict()['tensors'],value['tensors'],'W2 trainables')
        _validate_layout(self.state_dict()['buffers'],value['buffers'],'W2 buffers')
    def load_state_dict(self,value):
        super().load_state_dict(value)
        for name,p in self.engine.module.named_parameters():
            p.requires_grad_(value['requires_grad'][name])


class HeliosNoGANCycleOwner:
    def __init__(self,warm,args,negative):
        from omegaconf import OmegaConf
        require(type(warm) is bootstrap.WarmBootstrap
                and warm.identity['evidence_kind'] in (bootstrap.ORIGINAL,bootstrap.CPU_WARM),
                'actual W2 warm owner required; no cold W1/reader fixture')
        self.warm,self.args,self.negative = warm,args,negative
        self._engines = dict(warm.engines)
        self.generator,self.critic,self.ema = self._engines['generator'],self._engines['critic'],warm.ema
        self.cycle,self.poisoned,self.consumed,self._stream = 0,False,[],None
        self._rng = cpu_tree(rng_state())
        self._views = engine_state(self._engines)
        for role,engine in self._engines.items():
            self._views[role+'.weights'] = _BoundaryWeights(engine)
        self._views['ema'] = _SparseEMAView(self.ema,self.generator)
        self._bindings = _live_bindings()
        self._ownership = {role:{n:id(p) for n,p in e.module.named_parameters()} for role,e in self._engines.items()}
        self.identity = dict(source=source_identity(),bootstrap=plain(warm.identity),args=OmegaConf.to_container(args,resolve=True),
            negative=_content(negative),models={role:dict(config=_plain(dict(e.module.config)),
                frozen=_content({n:p.detach() for n,p in e.module.named_parameters() if not p.requires_grad}),
                trainables={n:dict(shape=list(p.shape),dtype=str(p.dtype)) for n,p in e.module.named_parameters() if p.requires_grad},
                buffers=_content(dict(e.module.named_buffers()))) for role,e in self._engines.items()})
        self._policies = {role:self._policy(e) for role,e in self._engines.items()}
        self._adapter_policy,self._calls = self._adapters(policy=True),self._engine_calls()
        self._validate_live()

    @property
    def capacity(self):
        return (self.args.training_config.max_train_steps-10)//5
    @property
    def versions(self):
        return dict(generator=self.generator.global_steps,critic=self.critic.global_steps,ema=self.ema.optimization_step)
    @staticmethod
    def expected(cycle):
        return dict(generator=2+cycle,critic=10+5*cycle,ema=10+5*cycle)
    @staticmethod
    def _policy(engine):
        return dict(optimizer=type(engine.optimizer).__module__+'.'+type(engine.optimizer).__qualname__,
            groups=[{k:v for k,v in g.items() if k not in ('params','lr')} for g in engine.optimizer.param_groups],
            scheduler={k:v for k,v in engine.lr_scheduler.state_dict().items()
                       if k not in ('last_epoch','_step_count','_last_lr')})
    def _engine_calls(self):
        return {role+'.'+key:(getattr(fn,'__func__',fn),getattr(getattr(fn,'__func__',fn),'__code__',None))
            for role,e in self._engines.items() for key,fn in (('backward',e.backward),('commit',e.step),
                ('adam',e.optimizer.step),('lr',e.lr_scheduler.step))}
    def _adapters(self,*,policy=False):
        from peft.tuners.tuners_utils import BaseTunerLayer
        result = {}
        for role,engine in self._engines.items():
            model = engine.module
            layers = {n:dict(active=list(m.active_adapters),disabled=m.disable_adapters,
                scaling=_plain(m.scaling),rank=_plain(m.r),alpha=_plain(m.lora_alpha))
                for n,m in model.named_modules() if isinstance(m,BaseTunerLayer)}
            require(layers and set(model.peft_config) == {'default'}, 'W2 single original PEFT adapter required')
            if not policy:
                require(list(model.active_adapters()) == ['default'] and all(v['active'] == ['default']
                        and v['disabled'] is False for v in layers.values()), 'W2 adapter enabled/selection boundary changed')
            result[role] = dict(config={n:_plain(c.to_dict()) for n,c in model.peft_config.items()},
                layers={n:{k:v for k,v in item.items() if k not in ('active','disabled')} for n,item in layers.items()})
        return result
    def _validate_identity(self):
        import torch
        from omegaconf import OmegaConf
        from distillschedule.adapters.helios.reference_model import load_reference_model_class
        topology()
        require(self._bindings == _live_bindings() and self._calls == self._engine_calls()
                and self._adapter_policy == self._adapters(policy=True), 'W2 original live helper/engine/PEFT binding changed')
        require(self.identity['source'] == source_identity() and self.identity['bootstrap'] == plain(self.warm.identity)
                and self.identity['args'] == OmegaConf.to_container(self.args,resolve=True)
                and self.identity['negative'] == _content(self.negative), 'W2 source/config/negative binding changed')
        cls = load_reference_model_class(backend='torch-sdpa')
        for role,e in self._engines.items():
            model,identity = e.module,self.identity['models'][role]; selected = set(identity['trainables'])
            require(type(model) is cls and e.dp_world_size == e.train_batch_size() == 1
                    and self._ownership[role] == {n:id(p) for n,p in model.named_parameters()}
                    and identity['config'] == _plain(dict(model.config))
                    and identity['frozen'] == _content({n:p.detach() for n,p in model.named_parameters() if n not in selected})
                    and identity['buffers'] == _content(dict(model.named_buffers()))
                    and identity['trainables'] == {n:dict(shape=list(p.shape),dtype=str(p.dtype)) for n,p in model.named_parameters() if n in selected}
                    and all(not p.requires_grad for n,p in model.named_parameters() if n not in selected)
                    and (self.poisoned or all(p.requires_grad for n,p in model.named_parameters() if n in selected))
                    and all(p.device.type == 'cpu' for p in model.parameters())
                    and all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
                    and all(not getattr(m,'merged',False) and m.training for m in model.modules())
                    and self._policies[role] == self._policy(e) and e._gradient_clip == 1.
                    and 'forward' not in vars(model) and model.gradient_checkpointing,
                    'W2 live model/precision/parameter/optimizer ownership changed')
            require(all(b.attn1.processor.kv_cache is None and not b.attn1.processor.cache_enabled for b in model.blocks),
                    'W2 KV cache policy changed')
        # A failed Teacher call may mask adapter flags. Fixed EMA ownership can
        # still be staged without consulting any current requires_grad mask.
        # Do this BEFORE any G/C restore, not only when the EMA view copies last.
        view,e = self._views['ema'],self.ema
        parameters = list(self.generator.module.named_parameters())
        require(view.ema is e and self.warm.ema is e and view.generator is self.generator
                and view._policy() == view.policy and e.parameter_count == len(parameters)
                and view.parameters == self._ownership['generator'] == {n:id(p) for n,p in parameters}
                and len(e.names) == len(e.indices) == len(e.shadows)
                and [parameters[i][0] for i in e.indices] == e.names,
                'W2 live EMA policy/name/index/Parameter ownership changed')
        require(all(torch.is_tensor(shadow) and shadow.device.type == 'cpu' and not shadow.requires_grad
                    and shadow.dtype == parameters[index][1].dtype and shadow.shape == parameters[index][1].shape
                    for shadow,index in zip(e.shadows,e.indices,strict=True)), 'W2 live EMA shadow layout changed')
    def _validate_native(self,native,cycle):
        import torch
        require(native.keys() == self._views.keys(), 'W2 named owner inventory changed')
        for key,view in self._views.items():
            if key == 'ema' and self.poisoned:
                continue  # Fixed ownership validated below without reading masked Teacher flags.
            if hasattr(view,'validate_state_dict'):
                view.validate_state_dict(native[key])
            else:
                _validate_layout(view.state_dict(),native[key],key)
        for role in ('generator','critic'):
            step = self.expected(cycle)[role]
            clocks,scheduler = native[role+'.clocks'],native[role+'.scheduler']
            # _step_calls is intentionally local to this newly materialized
            # engine instance; native global_steps/samples retain warm clocks.
            require(clocks['counters'] == dict(global_steps=step,global_samples=step,
                        _step_calls=cycle*(1 if role == 'generator' else 5))
                    and scheduler['last_epoch'] == step and scheduler['_step_count'] == step+1,
                    'W2 native engine/scheduler clocks changed')
            opt = native[role+'.optimizer']['optimizer']
            require([{k:v for k,v in g.items() if k not in ('params','lr')} for g in opt['param_groups']]
                    == self._policies[role]['groups'] and [g['lr'] for g in opt['param_groups']] == [bootstrap.POLICY[role+'_lr']]
                    and scheduler['_last_lr'] == [bootstrap.POLICY[role+'_lr']]
                    and {k:v for k,v in scheduler.items() if k not in ('last_epoch','_step_count','_last_lr')}
                        == self._policies[role]['scheduler'], 'W2 native Adam/LR policy changed')
            require(set(opt['state']) == set(opt['param_groups'][0]['params']), 'W2 complete warm named Adam slots required')
            for state in opt['state'].values():
                require(set(state) == {'step','exp_avg','exp_avg_sq'} and state['step'].item() == step
                        and state['step'].dtype == torch.float32 and state['step'].ndim == 0
                        and all(v.isfinite().all().item() for v in state.values())
                        and bool((state['exp_avg_sq'] >= 0).all()), 'W2 Adam clock/nonfinite/negative moment changed')
            weights = native[role+'.weights']
            require(all(v.isfinite().all().item() for v in weights['tensors'].values())
                    and _content(weights['buffers']) == self.identity['models'][role]['buffers'], 'W2 native weight/buffer corruption')
        ema = native['ema']; step = self.expected(cycle)['ema']
        require(ema.keys() == {'policy','shadows','optimization_step','cur_decay_value'}
                and ema['policy'] == self._views['ema'].policy and ema['optimization_step'] == step
                and type(ema['optimization_step']) is int and ema['cur_decay_value'] == self.ema.get_decay(step),
                'W2 sparse EMA policy/clock changed')
        _validate_layout(self.ema.shadows,ema['shadows'],'W2 sparse EMA')
        require(all(s.isfinite().all().item() for s in ema['shadows']), 'nonfinite W2 EMA shadow')
    def _validate_live(self):
        require(not self.poisoned, 'incomplete W2 cycle poisoned; restore a complete all-owner checkpoint')
        self._validate_identity(); self._adapters()
        require(self.versions == self.expected(self.cycle) and (self._stream is None or self.consumed == self._stream[:self.cycle]),
                'W2 native versions/consumed stream prefix changed')
        for role in ('generator','critic'):
            self._views[role+'.clocks'].validate_checkpoint_boundary()
        self._validate_native({k:v.state_dict() for k,v in self._views.items()},self.cycle)
    @contextmanager
    def _random_scope(self):
        outer = rng_state()
        try:
            restore_rng(self._rng)
            yield
        finally:
            self._rng = cpu_tree(rng_state())
            restore_rng(outer)
    def run(self,trace,row):
        import torch
        from distillschedule.runtime.helios_planner_cycle import _Observer,_prepare_cycle
        from distillschedule.search.helios_implementation import HeliosImplementation
        from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
        from distillschedule.profiling.static_profile_diagnostics import StaticProfileDiagnostics
        self._validate_live()
        require(self.cycle < self.capacity and trace.task == HeliosCohortTask(self.cycle)
                and row == descriptor(trace.reference_document) and row not in self.consumed
                and (self._stream is None or row == self._stream[self.cycle])
                and all(not set(row['sources']) & set(old['sources'])
                    and conditioning_fingerprint(row['content']) != conditioning_fingerprint(old['content']) for old in self.consumed),
                'W2 source/trace replay or reordered cohort')
        trace.validate_cpu_reference()
        observer = _Observer(torch=torch,device=torch.device('cpu'),rank=0,
            impl=HeliosImplementation('no-gan','world16-fsdp',512,1,False),generator=self.generator,critic=self.critic,
            ema=_BoundEMA(self.warm),mode='cpu-diagnostic',version_offsets=dict(generator=0,critic=0,ema=0),cpu_trace=trace)
        prepared,graph,order = _prepare_cycle(observer=observer,p0=_p0(),
            args=SimpleNamespace(timing_mode=False,profile_mode=False,profile_diagnostics=False,ema_device='cpu'),
            policy=StaticFSDP2P0Policy(),steps=trace.task.source_steps,trace=trace,negative=self.negative,cfg=None,
            diagnostics=StaticProfileDiagnostics(enabled=False,rank=0),generator_free_memory=None)
        self.poisoned = True
        with self._random_scope(),observer.installed():
            prepared.run()
        validation = observer.finish()
        require(self.versions == graph.metadata['final_versions'], 'W2 actual native versions differ from graph')
        self.cycle += 1; self.consumed.append(deepcopy(row))
        # Failed postconditions remain poisoned too; clearing is transactional.
        self.poisoned = False
        try:
            self._validate_live()
        except BaseException:
            self.poisoned = True
            raise
        return dict(cycle=self.cycle,versions=self.versions,dispatch=prepared.receipt(),order=list(order),
            actions=observer.actions,semantic_validation=validation,source_steps=list(trace.task.source_steps),
            telemetry=observer.memory(),scope=SCOPE,claimable_speedup=False)
    def state_dict(self):
        self._validate_live()
        return dict(schema=OWNER,identity=deepcopy(self.identity),cycle=self.cycle,versions=self.versions,
            consumed=deepcopy(self.consumed),rng=cpu_tree(self._rng),adapters=deepcopy(self._adapter_policy),
            native={k:cpu_tree(v.state_dict()) for k,v in self._views.items()})
    def validate_checkpoint_boundary(self):
        self._validate_live()
        require(self.cycle > 0, 'complete original W2 cycle required')
    def validate_state_dict(self,value):
        self._validate_identity()
        require(set(value) == {'schema','identity','cycle','versions','consumed','rng','native','adapters'}
                and value['schema'] == OWNER and value['identity'] == self.identity and value['adapters'] == self._adapter_policy
                and type(value['cycle']) is int and 0 <= value['cycle'] <= self.capacity
                and value['versions'] == self.expected(value['cycle'])
                and all(type(v) is int for v in value['versions'].values()), 'W2 source/boundary/version checkpoint mismatch')
        rows = value['consumed']
        require(isinstance(rows,list) and len(rows) == value['cycle']
                and len({r['control_hash'] for r in rows}) == len(rows)
                and len({conditioning_fingerprint(r['content']) for r in rows}) == len(rows)
                and len({s for r in rows for s in r['sources']}) == 80*len(rows)
                and (self._stream is None or rows == self._stream[:value['cycle']]), 'W2 checkpoint cursor/source prefix changed')
        validate_rng(value['rng']); self._validate_native(value['native'],value['cycle'])
    def load_state_dict(self,value):
        self.validate_state_dict(value)
        value = cpu_tree(value)
        self.poisoned = True
        for key,view in self._views.items():
            view.load_state_dict(value['native'][key])
        for e in self._engines.values():
            e.module.set_adapter('default'); e.module.enable_adapters(); e.optimizer.zero_grad(set_to_none=True)
        self.cycle,self.consumed,self._rng = value['cycle'],deepcopy(value['consumed']),cpu_tree(value['rng'])
        self.poisoned = False
        try:
            self._validate_live()
        except BaseException:
            self.poisoned = True
            raise
    def validate_checkpoint_progress(self,value,progress):
        self.validate_state_dict(value)
        require(progress.get('cycle') == progress.get('cursor') == value['cycle']
                and type(progress.get('cycle')) is type(progress.get('cursor')) is int
                and progress.get('versions') == {'native_cycle':value['cycle']}
                and type(progress['versions']['native_cycle']) is int, 'W2 inner/outer cursor changed')


def admit_training(*,state,request,prepared_identity,prepared_manifest):
    from distillschedule.data.manifest import validate_source
    owner = state['native_cycle']; owner._validate_live()
    require(dict(request.execution) == dict(adapter='program',mode='reference')
            and request.trainer.get('device','cpu') == 'cpu' and not request.trainer.get('distributed')
            and not request.trainer.get('validation') and request.data.get('sharding','single') == 'single'
            and request.data.get('sampling',dict(kind='finite',shuffle=False)) == dict(kind='finite',shuffle=False)
            and request.algorithm['factory'] == 'distillschedule.training.helios_nogan_public:build_algorithm'
            and not request.algorithm.get('parameters'), 'W2 CPU1 finite single atomic original cycle required')
    _,assets,_ = admit_assets(request.model.get('parameters',{}))
    meta = prepared_manifest['metadata']
    require(set(meta) == {'schema_version','scope','precision','task','source','stream','input_config','evidence_kind','layout'}
            and meta['schema_version'] == INPUTS and meta['scope'] == SCOPE and meta['precision'] == PRECISION
            and meta['task'] == HeliosCohortTask(0).to_dict() and meta['source'] == source_fingerprint()
            and meta['input_config'] == assets['input_config'] and meta['layout'] == layout_identity()
            and meta['evidence_kind'] == owner.warm.identity['assets']['base_evidence_kind'], 'W2 prepared source/task/precision/evidence changed')
    for pin in prepared_manifest['provenance']['documents']:
        validate_source(pin)
    records = prepared_manifest['records']
    require(2 <= len(records) <= min(64,owner.capacity) and request.trainer['cycles'] <= len(records), 'finite W2 cohort inventory exhausted')
    hashes,contents,sources = set(),set(),set()
    for i,record in enumerate(records):
        row = record['metadata']['trace']; content = conditioning_fingerprint(row['content'])
        require(record['metadata']['cycle'] == i and row['control_hash'] not in hashes and content not in contents
                and len(row['sources']) == len(set(row['sources'])) == 80 and not set(row['sources']) & sources,
                'W2 prepared source/control/conditioning reused or reordered')
        hashes.add(row['control_hash']); contents.add(content); sources.update(row['sources'])
    if owner.warm.identity['evidence_kind'] == bootstrap.CPU_WARM:
        proof = read_json(checked_pin(plain(owner.warm.identity)['assets']['provenance']))
        receipt = read_json(checked_pin(proof['inputs'][0]))
        warm_sources = {s for row in receipt['cycles'] for s in row['descriptor']['sources']}
        require(not sources & warm_sources, 'W2 training stream reuses warm input source bytes')
    require(owner._stream is None and isinstance(prepared_identity,str) and len(prepared_identity) == 64,
            'W2 original finite stream must be admitted once')
    owner._stream = [deepcopy(r['metadata']['trace']) for r in records]
    owner.identity['stream'] = dict(prepared_identity=prepared_identity,cohorts=deepcopy(owner._stream))


def native_cycle(ctx):
    owner = ctx.state['native_cycle']
    trace = decode(ctx.inputs,HeliosCohortTask(ctx.attributes['cycle']))
    return owner.run(trace,descriptor(trace.reference_document))


def build_algorithm(components,feeds,parameters,cycle,seed):
    from distillschedule.open_workload.torch_backend import Workload
    from distillschedule.workloads.program import Program,Call
    require(not parameters and type(cycle) is int and 0 <= cycle < 64 and set(components.state) == {'native_cycle'},
            'independent original W2 atomic owner required')
    owner = components.state['native_cycle']; require(type(owner) is HeliosNoGANCycleOwner, 'foreign W1/W3 owner')
    owner._validate_live(); require(owner.cycle == cycle, 'W2 cycle reordered')
    checked_pin(components.metadata['asset_manifest'])
    for key in ('bootstrap','args','input_config'):
        checked_pin(components.metadata['assets'][key])
    trace = decode(feeds,HeliosCohortTask(cycle))
    row = descriptor(trace.reference_document)
    require(owner._stream is not None and row == owner._stream[cycle] and row not in owner.consumed, 'W2 source/trace replay')
    call = Call(id='helios_nogan.native_cycle',binding='original_nogan_cycle',inputs=('control','control_length','tensors'),
        reads=('native_cycle',),writes=('native_cycle',),grad='external_grad',replay_safe=False,attributes=dict(cycle=cycle))
    return Workload(program=Program((call,),initial_versions={'native_cycle':cycle},feeds=('control','control_length','tensors'),
        outputs=(call.id,)),bindings={'original_nogan_cycle':native_cycle},state=components.state,feeds=feeds,seed=seed,
        identity_extra={'scope':SCOPE})


def collect_metrics(outputs):
    require(set(outputs) == {'helios_nogan.native_cycle'}, 'one complete original W2 cycle required')
    actions = outputs['helios_nogan.native_cycle']['actions']
    return {f'{role}/loss_{i}':loss['value'] for i,row in enumerate(actions) for loss in row['losses']
            for role in (loss['role'],)}
