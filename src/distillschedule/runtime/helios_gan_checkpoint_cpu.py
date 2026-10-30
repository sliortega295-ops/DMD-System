"""Narrow W3 adapter: true original G + installed original C-SAC composition.

No unwrapping: authenticate the precise functions/closures owned by the real
NativeDownStreamSAC.bind installation. Default W2 and W3 validators stay strict.
"""
from contextlib import ExitStack
from contextvars import ContextVar
from functools import partial
from pathlib import Path
from types import FunctionType, CodeType

import torch

from distillschedule.core.contracts import require
from distillschedule.training import helios_gan_checkpoint_cpu as policy
from distillschedule.workloads.best_known.registry import file_sha256, repository_root

_SOURCE={
    'src/distillschedule/runtime/fsdp2_static.py':'dd2fd0861f058b731145b9868524c13542d38c9ad19973de90b99088aa4a5d50',
    'src/distillschedule/runtime/helios_gan_manual/down_replay.py':'4cedeb51c5e142025cc2bc2321f4c357ca9cea6b8425c35df31a1db8e680ae5a',
    'src/distillschedule/runtime/helios_gan_manual/stream.py':'bfe444330d8953bd4183936f28ef0d3b77e01067bee9fd55182a0813483821ed',
    'src/distillschedule/runtime/helios_lora_epilogue.py':'9e2ac9a5e381fdea1d17598bb9970448e8a82c0ea66e0c60857fb8af3dc45151',
}


def _closure(fn):
    require(type(fn) is FunctionType,'W3 original owned function type changed',path='helios.gan.checkpoint_cpu')
    return {name:cell.cell_contents for name,cell in zip(fn.__code__.co_freevars,fn.__closure__ or (),strict=True)}


def _code(function,*path):
    result=function.__code__
    for name in path:
        candidates=[c for c in result.co_consts if isinstance(c,CodeType) and c.co_name==name]
        require(len(candidates)==1,'W3 original checkpoint code boundary changed',path='helios.gan.checkpoint_cpu')
        result=candidates[0]
    return result


def _source_code(module,*path):
    # Compile the authenticated file WITHOUT executing/importing it. Compare
    # actual loaded code, not __wrapped__ or user-supplied provenance labels.
    result=compile(Path(module.__file__).read_text(),module.__file__,'exec',dont_inherit=True)
    for name in path:
        candidates=[c for c in result.co_consts if isinstance(c,CodeType) and c.co_name==name]
        require(len(candidates)==1,'W3 source callable code missing',path='helios.gan.checkpoint_cpu')
        result=candidates[0]
    return result


def authenticate_callables(core,*,require_cohort=True):
    from . import fsdp2_static as fsdp
    from .helios_gan_manual.stream import NativeDownStreamSAC
    from .helios_gan_manual.down_replay import NativeDownSAC
    from .helios_gan_manual import down_replay
    require(type(core) is NativeDownStreamSAC and core.bound and core.enabled
            and core.current is None and not core.stores and (not require_cohort or core.cohort is not None),
            'W3 needs the actual bound drained fresh-cohort C-SAC owner',path='helios.gan.checkpoint_cpu')
    require(all(file_sha256(repository_root()/p)==sha for p,sha in _SOURCE.items()),
            'W3 original checkpoint/SAC source changed',path='helios.gan.checkpoint_cpu')
    require(NativeDownSAC.bind.__code__==_source_code(down_replay,'NativeDownSAC','bind')
            and fsdp.configure_fsdp2_checkpointing.__code__==_source_code(fsdp,'configure_fsdp2_checkpointing'),
            'W3 loaded checkpoint/SAC implementation changed',path='helios.gan.checkpoint_cpu')
    originals={}
    for name,code_path in (('fsdp2_metadata_tolerant_checkpoint',('dispatch','call')),
                           ('_selective_flash_checkpoint_contexts',('factory','call'))):
        rows=[r for r in core._restore_targets if r[0] is fsdp and r[1]==name]
        fn=getattr(fsdp,name)
        require(len(rows)==1 and type(fn) is FunctionType and fn.__code__ is _code(NativeDownSAC.bind,*code_path),
                'W3 original installed SAC callable boundary changed',path='helios.gan.checkpoint_cpu')
        cells=_closure(fn);original=rows[0][4]
        require(cells.get('self') is core and cells.get('original') is original
                and type(original) is FunctionType and original.__module__==fsdp.__name__
                and original.__globals__ is vars(fsdp) and original.__name__==name
                and original.__code__==_source_code(fsdp,name),
                'W3 installed SAC closure ownership changed',path='helios.gan.checkpoint_cpu')
        originals[name]=original
    g,c=core.generator.module,core.critic.module
    from . import helios_lora_epilogue as lora
    require(lora.install_generator_nograd_lora_fused_epilogue.__code__==_source_code(lora,'install_generator_nograd_lora_fused_epilogue'),
            'W3 loaded original LoRA replay implementation changed',path='helios.gan.checkpoint_cpu')
    for role,root in (('generator',g),('critic',c)):
        wrapper=getattr(root,'_distillschedule_lora_checkpoint_function_wrapper',None)
        cells=_closure(wrapper)
        receipt=getattr(root,'_distillschedule_helios_nograd_lora_epilogue',{})
        require(wrapper.__code__ is _code(lora.install_generator_nograd_lora_fused_epilogue,'wrap_checkpoint_function')
                and wrapper.__globals__ is vars(lora) and cells.get('receipt') is receipt
                and type(cells.get('active_call')) is ContextVar and receipt.get('role')==role
                and receipt.get('enabled') is receipt.get('checkpoint_replay_scope') is True
                and receipt.get('scope')=='ALL_ROLE_SCALE1_MODEL_FORWARD',
                'W3 exact original LoRA replay wrapper ownership changed',path='helios.gan.checkpoint_cpu')
    fn=original_g=g._gradient_checkpointing_func;options={}
    while isinstance(fn,partial):
        require(not fn.args and not set(fn.keywords)&set(options),'W3 ambiguous G checkpoint partial',path='helios.gan.checkpoint_cpu')
        options.update(fn.keywords);fn=fn.func
    require(fn is originals['fsdp2_metadata_tolerant_checkpoint']
            and set(options)=={'function_wrapper'}
            and callable(options['function_wrapper'])
            and options['function_wrapper'] is getattr(g,'_distillschedule_lora_checkpoint_function_wrapper',None),
            'W3 exact captured original G checkpoint/replay callable changed',path='helios.gan.checkpoint_cpu')
    selector=c._gradient_checkpointing_func;cells=_closure(selector)
    receipt=getattr(c,'_distillschedule_checkpoint_policy_receipt',{})
    require(selector.__code__ is _code(fsdp.configure_fsdp2_checkpointing,'selective_checkpoint')
            and selector.__globals__ is vars(fsdp) and cells.get('receipt') is receipt
            and cells.get('block_index_by_id')=={id(b):i for i,b in enumerate(c.blocks)}
            and cells.get('module_name_by_id')=={id(m):str(n) for n,m in c.named_modules()}
            and cells.get('selected_set')==frozenset(range(0,40,2))
            and cells.get('extra')=={'function_wrapper':getattr(c,'_distillschedule_lora_checkpoint_function_wrapper',None)}
            and callable(getattr(c,'_distillschedule_lora_checkpoint_function_wrapper',None))
            and receipt.get('selective_flash_stride')==2 and receipt.get('operation_source') is False,
            'W3 exact original C selective checkpoint/replay owner changed',path='helios.gan.checkpoint_cpu')
    require(g.gradient_checkpointing is c.gradient_checkpointing is True and len(g.blocks)==len(c.blocks)==40
            and len({id(b) for b in (*g.blocks,*c.blocks)})==80,
            'W3 disjoint original G/C whole checkpoints changed',path='helios.gan.checkpoint_cpu')
    return dict(generator=original_g,critic=selector,original=originals['fsdp2_metadata_tolerant_checkpoint'])


def admit_engine(engine,*,task=None):
    """Run before complete-G materialization swaps FP32 optimizer slots."""
    from torch.distributed.fsdp import FSDPModule
    from helios.modules import transformer_helios as transformer
    from helios.modules.transformer_helios import HeliosTransformerBlock, Discriminator3DHead
    from .helios_gan_native_history import require_task
    from .helios_gan_manual.down_contract import FA3
    from . import fsdp2_static as fsdp
    from distillschedule.training.helios_gan_training import owner_layout
    from distillschedule.training.native_state import _named
    binding=policy.require_binding();core=binding.get('gan_core')
    require(core is not None and core.generator is engine,'W3 exact original G role owner missing',path='helios.gan.checkpoint_cpu')
    callables=authenticate_callables(core,require_cohort=task is not None)
    from .helios_gan_manual.runtime import TRANSFORMER_SHA
    require(file_sha256(transformer.__file__)==TRANSFORMER_SHA,
            'W3 original whole-block/head class source changed',path='helios.gan.checkpoint_cpu')
    if task is not None:
        require_task(task)
        require(core.cohort==task.index and int(engine.global_steps)==2+task.index
                and int(core.critic.global_steps)==10+5*task.index,
                'W3 fresh G action version/cohort changed',path='helios.gan.checkpoint_cpu')
    require(fsdp.SELECTIVE_FLASH_ATTN_OP==FA3,'W3 original FA3 policy changed',path='helios.gan.checkpoint_cpu')
    require(core.torch is torch and torch.__version__=='2.10.0+cu128' and torch.version.cuda=='12.8'
            and core.rank==binding['checkpoint_input_startup']['rank'],
            'W3 original Torch/CUDA/rank owner changed',path='helios.gan.checkpoint_cpu')
    for role,root,extra in (('G',engine.module,()),('C',core.critic.module,
            (*core.critic.module.gan_heads.values(),core.critic.module.gan_final_head))):
        blocks=tuple(root.blocks)
        require(isinstance(root,FSDPModule) and len(blocks)==40
                and all(isinstance(b,HeliosTransformerBlock) for b in blocks)
                and (not extra or len(extra)==5 and all(isinstance(h,Discriminator3DHead) for h in extra))
                and {id(m) for m in root.modules() if isinstance(m,FSDPModule)}=={id(root),*(id(b) for b in blocks),*(id(h) for h in extra)},
                'W3 original '+role+' whole-block/head FSDP ownership changed',path='helios.gan.checkpoint_cpu')
        owners=[m for m in root.modules() if '_gradient_checkpointing_func' in vars(m)]
        require(owners==[root],'W3 unique original checkpoint owner changed',path='helios.gan.checkpoint_cpu')
    layout=owner_layout(engine,core.critic)
    saved=binding['identity'].get('native_owner_layout')
    require((saved is None or saved==layout) and len(_named(engine)[1])==818
            and len(_named(core.critic)[1])==942,
            'W3 original FP32 joint G/C/head named-owner layout changed',path='helios.gan.checkpoint_cpu')
    require(next(engine.module.parameters()).device.type=='cuda',
            'W3 native checkpoint CPU policy cannot use a CPU fake owner',path='helios.gan.checkpoint_cpu')
    return dict(schema_version=policy.SESSION_SCHEMA,checkpoint_blocks=40,fsdp_units=41,
        critic_fsdp_units=46,topology=layout['topology'],original_callable_composition='EXACT_BOUND_SAC_CAPTURED_ORIGINAL_G',
        callables=callables)


class _CriticBoundary:
    """Observe, never offload/rewrite, the original GAN-G C checkpoint/SAC path."""
    def __init__(self,core,callables,identity,device):
        self.core,self.callables,self.identity=core,callables,identity
        self.device=torch.device(device);self.capacity_observations=[]
        self.spec=policy.critic_boundary_contract();self.active=False;self.checkpoint_active=False
        self.seen=set();self.heads=set();self.patches=[];self.stores=[]
        self.counts=dict(forwards=0,patch_calls=0,block_calls=0,head_calls=0,selective_contexts=0,
                         input_derivatives=0,g_backwards=0,drained_before_g_backward=False)
        self.versions=(int(core.generator.global_steps),int(core.critic.global_steps))
        self.before={k:core.critic.module._distillschedule_checkpoint_policy_receipt[k] for k in
            ('selective_flash_original_save_count','selective_flash_recompute_reuse_count')}
        self.gradients={n:id(p.grad) for n,p in core.critic.module.named_parameters() if p.grad is not None}

    def _check(self):
        require(self.core.current is None and not self.core.stores and self.core.cohort==self.identity['cohort_index']
                and (int(self.core.generator.global_steps),int(self.core.critic.global_steps))==self.versions,
                'W3 G/C window or version crossed',path='helios.gan.checkpoint_cpu')

    def install(self,stack):
        from . import fsdp2_static as fsdp
        from .helios_gan_manual.down_replay import replace
        root=self.core.critic.module;block_ids={id(b):i for i,b in enumerate(root.blocks)}
        head_ids={id(h) for h in (*root.gan_heads.values(),root.gan_final_head)}
        patch_ids={id(getattr(root,name)):name for name in self.spec['patch_modules']}
        require(len(patch_ids)==4 and not set(patch_ids)&(set(block_ids)|head_ids),
                'W3 distinct original patch checkpoint owners changed',path='helios.gan.checkpoint_cpu')
        def before(module,args,kwargs):
            self._check()
            if not torch.is_grad_enabled():return
            require(not self.active and not args and kwargs.get('gan_mode') is True and self.counts['forwards']==0,
                    'W3 original cached GAN-G C role/call changed',path='helios.gan.checkpoint_cpu')
            hidden=kwargs.get('hidden_states')
            require(type(hidden) is torch.Tensor and list(hidden.shape)==self.spec['latent_shape']
                    and str(hidden.dtype)==self.spec['dtype'] and hidden.device==self.device
                    and hidden.layout==torch.strided and hidden.is_contiguous() and hidden.requires_grad and hidden.is_leaf,
                    'W3 original cropped C latent shape/dtype/leaf changed',path='helios.gan.checkpoint_cpu')
            if self.device.type=='cuda':
                free,total=torch.cuda.mem_get_info(self.device)
                needed=self.spec['hidden_input_bytes']+policy.GPU_RESERVE
                require(free>=needed,'W3 unchanged C hidden-input subset exceeds GPU floor before GAN-G',path='helios.gan.checkpoint_cpu')
                self.capacity_observations.append(dict(free_bytes=free,total_bytes=total,required_subset_plus_floor=needed,
                    scope='KNOWN_C_CHECKPOINT_SUBSET_ONLY_NOT_TOTAL_FIT_PROOF_NOT_RESERVATION'))
            self.active=True;self.counts['forwards']+=1
            def derivative(gradient):
                self._check();self.counts['input_derivatives']+=1
                return gradient
            stack.callback(hidden.register_hook(derivative).remove)
        def after(module,args,kwargs,output):
            self.active=False
        stack.callback(root.register_forward_pre_hook(before,with_kwargs=True).remove)
        stack.callback(root.register_forward_hook(after,with_kwargs=True,always_call=True).remove)
        def observe(original):
            require(original is self.callables['critic'],'W3 C original callable changed before observer',path='helios.gan.checkpoint_cpu')
            def call(module,*args,**kwargs):
                self._check()
                require(self.active and torch.is_grad_enabled(),'W3 checkpoint outside original GAN-G C forward',path='helios.gan.checkpoint_cpu')
                if id(module) in patch_ids:
                    name=patch_ids[id(module)]
                    require(not self.seen and not self.heads
                            and len(self.patches)<len(self.spec['patch_modules'])
                            and name==self.spec['patch_modules'][len(self.patches)],
                            'W3 original patch checkpoint order/count changed',path='helios.gan.checkpoint_cpu')
                    self.patches.append(name);self.counts['patch_calls']+=1
                elif id(module) in block_ids:
                    require(self.patches==self.spec['patch_modules']
                            and id(module) not in self.seen and args and type(args[0]) is torch.Tensor
                            and list(args[0].shape)==self.spec['hidden_shape'] and str(args[0].dtype)==self.spec['dtype'],
                            'W3 original GAN-G C checkpoint shape/dtype/order changed',path='helios.gan.checkpoint_cpu')
                    self.seen.add(id(module));self.counts['block_calls']+=1
                else:
                    require(self.patches==self.spec['patch_modules']
                            and id(module) in head_ids and id(module) not in self.heads,
                            'W3 auxiliary GAN-G checkpoint owner changed',path='helios.gan.checkpoint_cpu')
                    self.heads.add(id(module));self.counts['head_calls']+=1
                self.checkpoint_active=True
                try:return original(module,*args,**kwargs)
                finally:self.checkpoint_active=False
            self.observed=call
            return call
        replace(stack,root,'_gradient_checkpointing_func',observe)
        def contexts(original):
            def call(receipt):
                self._check()
                require(self.active and self.checkpoint_active and receipt is root._distillschedule_checkpoint_policy_receipt,
                        'W3 selective C context outside original checkpoint',path='helios.gan.checkpoint_cpu')
                f,r=original(receipt)
                require(f.storage is r.storage,'W3 original C SAC store changed',path='helios.gan.checkpoint_cpu')
                self.stores.append(f.storage);self.counts['selective_contexts']+=1
                return f,r
            return call
        replace(stack,fsdp,'_selective_flash_checkpoint_contexts',contexts)
        def backward(original):
            def call(*args,**kwargs):
                self.validate_drained();self.counts['g_backwards']+=1
                require(self.counts['g_backwards']==1,'W3 extra G backward',path='helios.gan.checkpoint_cpu')
                return original(*args,**kwargs)
            return call
        replace(stack,self.core.generator,'backward',backward)

    def validate_drained(self):
        self._check();root=self.core.critic.module
        receipt=root._distillschedule_checkpoint_policy_receipt
        delta=[receipt[k]-self.before[k] for k in self.before]
        require(not self.active and root._gradient_checkpointing_func is self.observed
                and self.counts['forwards']==self.counts['input_derivatives']==1
                and self.patches==self.spec['patch_modules']
                and self.counts['patch_calls']==len(self.spec['patch_modules'])
                and self.counts['block_calls']==self.spec['checkpoint_blocks']
                and self.counts['head_calls']==self.spec['auxiliary_heads']
                and self.counts['selective_contexts']==self.spec['selective_contexts']
                and _selective_drained(self.stores,delta,self.spec['selective_FA3_results'])
                and {n:id(p.grad) for n,p in root.named_parameters() if p.grad is not None}==self.gradients,
                'W3 original GAN-G C backward/FA3 cache did not drain before G backward: '+str(dict(
                    counts=self.counts,selective_delta=delta,queues=[sum(len(q) for q in store.values()) for store in self.stores])),
                path='helios.gan.checkpoint_cpu')
        self.counts['drained_before_g_backward']=True
        self.stores.clear()

    def receipt(self):
        return dict(schema_version='distillschedule.helios_gan_g_c_boundary.v1',contract=dict(self.spec),
            identity=dict(self.identity),counts=dict(self.counts),live_selective_stores=len(self.stores),
            cpu_copies=0,capacity_observations=list(self.capacity_observations),
            original_callable_restored=self.core.critic.module._gradient_checkpointing_func is self.callables['critic'])


def _selective_drained(stores,delta,expected):
    """Pinned Torch2.10 FIFO/counter contract. Never infer a newer SAC ABI."""
    return delta==[expected]*2 and all(type(q) is list and not q for store in stores for q in store.values())


class GAction:
    """Component construction permits CPU; only generator_scope admits native owners."""
    def __init__(self,core,*,capacity,identity,device):
        from .helios_checkpoint_inputs_cpu import GCheckpointInputSession
        self.callables=authenticate_callables(core)
        self.core=core;self.boundary=_CriticBoundary(core,self.callables,identity,device)
        def original(owner):
            require(owner is core.generator.module and owner._gradient_checkpointing_func is self.callables['generator'],
                    'W3 captured original G callable replaced',path='helios.gan.checkpoint_cpu')
            return self.callables['generator']
        self.session=GCheckpointInputSession(owner=core.generator.module,blocks=core.generator.module.blocks,
            capacity=capacity,identity=identity,device=device,_policy=policy,_original_validator=original)
        self.stack=ExitStack()

    def __enter__(self):
        try:
            self.stack.enter_context(self.session)
            self.boundary.install(self.stack)
            return self
        except BaseException:
            self.stack.close();raise

    def __exit__(self,kind,error,traceback):
        try:
            require(self.core.critic.module._gradient_checkpointing_func is self.boundary.observed,
                    'W3 C observer replaced during G action',path='helios.gan.checkpoint_cpu')
            if kind is None:
                require(self.boundary.counts['g_backwards']==1 and self.boundary.counts['drained_before_g_backward'],
                        'W3 original G action did not execute drained C derivative and G backward',path='helios.gan.checkpoint_cpu')
        finally:
            self.boundary.stores.clear()
            self.stack.__exit__(kind,error,traceback)


def generator_scope(engine,*,task):
    binding=policy.require_binding()
    capability=admit_engine(engine,task=task)
    registration=GAction(binding['gan_core'],capacity=binding['identity']['checkpoint_input_capacity'],
        device=next(engine.module.parameters()).device,identity=dict(
            source=binding['identity']['source_head'],source_fingerprint=binding['identity']['source_fingerprint'],
            stream_sha256=binding['identity']['stream_sha256'],task_sha256=task.digest,cohort_index=task.index,
            generator_version=int(engine.global_steps),critic_version=int(binding['gan_core'].critic.global_steps),
            rank=binding['checkpoint_input_startup']['rank']))
    return registration,{k:v for k,v in capability.items() if k!='callables'}


def validate_action(row):
    from distillschedule.training.helios_lifecycle import current_training
    from .helios_checkpoint_inputs_cpu import validate_phase_receipt
    binding=current_training();phase=row.get('native_history_phase',{})
    selected=binding is not None and binding['config'].execution.get('preset')==policy.PRESET_ID
    if not selected and 'checkpoint_input_cpu' not in phase:return
    require(row['action_id']=='g' and len(row['trace_entries'])==1,
            'W3 G-only checkpoint CPU action/trace changed',path='helios.gan.checkpoint_cpu')
    if selected:policy.require_binding()
    validate_phase_receipt(phase.get('checkpoint_input_cpu'),with_grad=True,task_sha256=phase.get('task_sha256'),
        generator_version=row.get('physical_versions_before',{}).get('generator'),rank=row.get('source_rank'),
        stage_schedule=row['trace_entries'][0]['stage_schedule'],_policy=policy)
    c=phase.get('gan_g_checkpoint_boundary',{});identity=phase['checkpoint_input_cpu']['identity']
    observations=c.get('capacity_observations',[])
    required=policy.critic_boundary_contract()['hidden_input_bytes']+policy.GPU_RESERVE
    require(c.get('schema_version')=='distillschedule.helios_gan_g_c_boundary.v1'
            and c.get('identity')==identity and c.get('contract')==policy.critic_boundary_contract()
            and c.get('counts')==dict(forwards=1,patch_calls=4,block_calls=40,head_calls=5,selective_contexts=20,
                input_derivatives=1,g_backwards=1,drained_before_g_backward=True)
            and c.get('live_selective_stores')==c.get('cpu_copies')==0 and c.get('original_callable_restored') is True
            and type(observations) is list and len(observations)==1
            and all(type(observations[0].get(k)) is int for k in ('free_bytes','total_bytes','required_subset_plus_floor'))
            and observations[0]['required_subset_plus_floor']==required
            and required<=observations[0]['free_bytes']<=observations[0]['total_bytes']
            and observations[0].get('scope')=='KNOWN_C_CHECKPOINT_SUBSET_ONLY_NOT_TOTAL_FIT_PROOF_NOT_RESERVATION'
            and identity.get('critic_version')==row.get('physical_versions_before',{}).get('critic')
            and type(identity.get('cohort_index')) is int and identity['generator_version']==2+identity['cohort_index']
            and identity['critic_version']==10+5*identity['cohort_index'],
            'W3 original GAN-G C checkpoint/SAC/role/version census changed',path='helios.gan.checkpoint_cpu')
