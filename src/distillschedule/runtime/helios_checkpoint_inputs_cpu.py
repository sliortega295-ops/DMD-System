"""Bounded hidden-input leases around the unchanged original G checkpoint.

Like the audited H3 boundary hook, hooks surround only the checkpoint CALL,
not its recomputation. Unlike shape-only staging, only the exact hidden input
object is copied. The saved closures retain neither that tensor nor its args.
Synchronous copies make lifetime/overflow refusal explicit; no async prefetch.
"""
from contextvars import ContextVar
from functools import partial
import weakref

import torch

from distillschedule.core.contracts import require
from distillschedule.open_workload.role_offload import HostBudget
from distillschedule.training import helios_checkpoint_cpu as policy
from distillschedule.workloads.best_known.registry import fingerprint

SCHEMA = 'distillschedule.helios_g_checkpoint_hidden_cpu_execution.v1'

def _storage_key(tensor):
    from torch.distributed.tensor import DTensor
    local = tensor.to_local() if isinstance(tensor,DTensor) else tensor
    return str(local.device),local.untyped_storage().data_ptr()

_ACTIVE = ContextVar('helios_g_checkpoint_hidden_cpu',default=None)


def _original_checkpoint(owner):
    from .fsdp2_static import fsdp2_metadata_tolerant_checkpoint
    fn = original = owner._gradient_checkpointing_func
    options = {}
    while isinstance(fn,partial):
        require(not fn.args and not (set(fn.keywords)&set(options)),
                'ambiguous original checkpoint partial',path='helios.checkpoint_cpu')
        options.update(fn.keywords);fn=fn.func
    require(fn is fsdp2_metadata_tolerant_checkpoint and not set(options)-{'operation_source','function_wrapper'}
            and options.get('operation_source',False) is False
            and ('function_wrapper' not in options or options['function_wrapper'] is
                 getattr(owner,'_distillschedule_lora_checkpoint_function_wrapper',None)),
            'checkpoint CPU requires exact original checkpoint callable and replay wrapper',path='helios.checkpoint_cpu')
    return original


class _Lease:
    def __init__(self,host,guard,version,device,metadata,session):
        self.host,self.guard,self.version,self.device = host,guard,version,device
        self.metadata,self.session = metadata,session
        self.finalizer=weakref.finalize(self,session._release,host.numel()*host.element_size(),metadata)

    def cancel(self):
        self.metadata['release_outcome']='cancelled'
        self.host=self.guard=None
        self.finalizer()


class _KeptTensor:
    """Keep original storage and its SavedVariable mutation check, no copy."""
    def __init__(self, tensor):
        self.tensor = tensor.detach()
        self.version = int(tensor._version)


class _BoundaryScope:
    def __init__(self,session,target,metadata):
        self.session,self.metadata = session,metadata
        self.target_id=int(target._cdata)
        self.target_shape,self.target_dtype = tuple(target.shape),target.dtype
        self.copies=0

    def pack(self,tensor):
        session=self.session;session.stats['pack_count']+=1
        if int(tensor._cdata)!=self.target_id:
            session.stats['kept_other_input_count']+=1
            return _KeptTensor(tensor)
        require(tuple(tensor.shape)==self.target_shape and tensor.dtype==self.target_dtype
                and tensor.device==session.device and tensor.is_contiguous(),
                'exact checkpoint input identity/shape/dtype/device changed',path='helios.checkpoint_cpu')
        require(self.copies==0,'duplicate exact checkpoint hidden input save',path='helios.checkpoint_cpu')
        size=tensor.numel()*tensor.element_size()
        session._reserve(size)
        host=guard=None
        try:
            host=torch.empty_like(tensor,device='cpu',pin_memory=session.pin_memory)
            host.copy_(tensor.detach(),non_blocking=False)
            # saved_tensors_hooks bypass PyTorch's SavedVariable version check.
            # ONLY this new detached sentinel has its data replaced. It shares
            # the original version counter but owns ZERO original storage,
            # including when the source dies and a detached alias survives.
            version=int(tensor._version)
            require(version==self.metadata['input_version'],
                    'checkpoint input modified inplace during CPU copy',path='helios.checkpoint_cpu')
            guard=tensor.detach()
            guard.data=torch.empty(0,device=tensor.device,dtype=tensor.dtype)
            require(guard.untyped_storage().nbytes()==0 and tensor._version==version,
                    'empty checkpoint version sentinel changed original input',path='helios.checkpoint_cpu')
            lease=_Lease(host,guard,version,tensor.device,self.metadata,session)
        except BaseException:
            host=guard=None
            session.budget.release(size)
            session.stats['live_cpu_bytes']=session.budget.used
            session.stats['live_leases']-=1
            session.stats['failed_pack_count']+=1
            raise
        session.leases.add(lease)
        session.stats['cpu_copy_count']+=1
        session.stats['copied_bytes']+=size
        self.copies+=1
        return lease

    def unpack(self,item):
        session=self.session;session.stats['unpack_count']+=1
        if isinstance(item, _KeptTensor):
            if item.tensor._version != item.version:
                raise RuntimeError('checkpoint shared input modified inplace before original backward/recompute')
            return item.tensor
        if not isinstance(item,_Lease):return item
        if item.host is None or session.closed:
            raise RuntimeError('checkpoint input lease cancelled or closed')
        if item.guard._version!=item.version:
            raise RuntimeError('checkpoint input modified inplace before original backward/recompute')
        if session.device.type=='cuda':
            free,_=torch.cuda.mem_get_info(session.device)
            require(free>=item.host.numel()*item.host.element_size()+session.policy.GPU_RESERVE,
                    'checkpoint restore exceeds unchanged GPU reserve',path='helios.checkpoint_cpu')
        result=item.host.to(item.device,non_blocking=False,copy=True)
        require(tuple(result.shape)==tuple(item.metadata['shape']) and str(result.dtype)==item.metadata['dtype']
                and str(result.device)==item.metadata['device'],
                'checkpoint restore metadata changed',path='helios.checkpoint_cpu')
        session.stats['restore_count']+=1
        session.stats['restored_bytes']+=item.host.numel()*item.host.element_size()
        item.metadata['restore_count']+=1
        return result


class GCheckpointInputSession:
    """One admitted G action, with graph-owned weakly tracked CPU leases.

    Native construction uses the W2 factory below or the independently admitted
    W3 adapter, AFTER whole-block/FSDP/callable checks. CPU direct construction is only
    a component fixture, never native physical admission.
    """
    def __init__(self,*,owner,blocks,capacity,identity,device,_policy=None,_original_validator=None):
        self.owner,self.blocks=owner,tuple(blocks)
        self.policy=policy if _policy is None else _policy
        self.original=(_original_checkpoint if _original_validator is None else _original_validator)(owner)
        self.spec=self.policy.native_contract()
        require(owner.gradient_checkpointing is True and len(self.blocks)==self.spec['checkpoint_blocks']
                and len({id(b) for b in self.blocks})==len(self.blocks),
                'checkpoint CPU requires all original whole blocks',path='helios.checkpoint_cpu')
        self.capacity=self.policy.validate_capacity(capacity)
        require(isinstance(identity,dict) and type(identity.get('generator_version')) is int
                and identity['generator_version']>=2 and isinstance(identity.get('source'),str) and bool(identity['source'])
                and isinstance(identity.get('task_sha256'),str) and len(identity['task_sha256'])==64,
                'checkpoint input source/task/generator version identity missing',path='helios.checkpoint_cpu')
        self.identity=dict(identity);self.identity_sha256=fingerprint(self.identity)
        self.device=torch.device(device);self.pin_memory=capacity['storage']=='pinned'
        self.budget=HostBudget(capacity['rank_pinned_limit_bytes']+capacity['rank_pageable_limit_bytes'])
        self.block_ids={id(block):index for index,block in enumerate(self.blocks)}
        self.leases=weakref.WeakSet();self.boundaries=[];self.seen=set()
        self.closed=False;self.entered=False;self.scope_active=False;self.token=None
        self.stats={name:0 for name in ('checkpoint_calls','auxiliary_checkpoint_calls','pack_count','unpack_count',
            'kept_other_input_count','cpu_copy_count','restore_count','release_count','cancel_count',
            'restored_release_count','unused_release_count',
            'copied_bytes','restored_bytes','failed_pack_count','live_cpu_bytes','peak_live_cpu_bytes','live_leases','peak_live_leases')}

    def _reserve(self,size):
        require(self.stats['live_leases']<self.spec['checkpoint_blocks']*self.spec['max_connected_grad_stages'],
                'checkpoint input live count exceeds original three-stage bound',path='helios.checkpoint_cpu')
        if self.device.type=='cuda':
            from distillschedule.training.host_memory import admit_host_memory
            admit_host_memory(self.capacity['node_pinned_limit_bytes']+self.capacity['node_pageable_limit_bytes'],
                              self.policy.RUNNING_RESERVE)
        self.budget.reserve(size)
        self.stats['live_cpu_bytes']=self.budget.used
        self.stats['peak_live_cpu_bytes']=self.budget.peak
        self.stats['live_leases']+=1
        self.stats['peak_live_leases']=max(self.stats['peak_live_leases'],self.stats['live_leases'])

    def _release(self,size,metadata):
        self.budget.release(size)
        self.stats['release_count']+=1;self.stats['live_leases']-=1
        self.stats['live_cpu_bytes']=self.budget.used
        metadata['release_count']+=1
        if metadata['release_outcome']!='cancelled':
            outcome='restored' if metadata['restore_count'] else 'unused'
            metadata['release_outcome']=outcome
            self.stats[outcome+'_release_count']+=1

    def __call__(self,block,*args,**kwargs):
        require(self.entered and not self.closed and self.owner._gradient_checkpointing_func is self,
                'checkpoint input wrapper ownership changed',path='helios.checkpoint_cpu')
        if id(block) not in self.block_ids or not torch.is_grad_enabled():
            self.stats['auxiliary_checkpoint_calls']+=1
            return self.original(block,*args,**kwargs)
        require(not self.scope_active,'nested checkpoint hidden-input hooks forbidden',path='helios.checkpoint_cpu')
        require(bool(args) and torch.is_tensor(args[0]),'original block checkpoint hidden tensor missing',path='helios.checkpoint_cpu')
        target=args[0];shape=tuple(target.shape)
        require(type(target) is torch.Tensor and list(shape) in self.spec['shapes'] and str(target.dtype)==self.spec['dtype']
                and target.device==self.device and target.layout==torch.strided and target.is_contiguous(),
                'original B1 checkpoint hidden shape/dtype/device changed',path='helios.checkpoint_cpu')
        # Reject parameter/buffer storage even if a malicious owner supplies an
        # input with an allowed shape. Never offload the shared conditioning.
        storage=_storage_key(target)
        require(all(storage!=_storage_key(p)
                    for p in (*self.owner.parameters(),*self.owner.buffers())),
                'checkpoint hidden input aliases parameter/buffer storage',path='helios.checkpoint_cpu')
        key=(self.block_ids[id(block)],shape)
        require(key not in self.seen,'extra differentiable stage/block checkpoint input exceeds original bound',path='helios.checkpoint_cpu')
        self.seen.add(key)
        metadata={**self.identity,'block_index':key[0],'stage_index':self.spec['shapes'].index(list(shape)),
            'shape':list(shape),'dtype':str(target.dtype),'device':str(target.device),
            'input_version':int(target._version),'identity_sha256':self.identity_sha256,
            'restore_count':0,'release_count':0,'release_outcome':None}
        scope=_BoundaryScope(self,target,metadata)
        del target
        self.scope_active=True;self.stats['checkpoint_calls']+=1
        try:
            with torch.autograd.graph.saved_tensors_hooks(scope.pack,scope.unpack):
                result=self.original(block,*args,**kwargs)
            require(scope.copies==1,'original non-reentrant checkpoint did not save exactly one hidden input',path='helios.checkpoint_cpu')
            self.boundaries.append(metadata)
            return result
        finally:
            self.scope_active=False

    def __enter__(self):
        require(not self.entered and not self.closed and _ACTIVE.get() is None,
                'nested/reused G checkpoint input context forbidden',path='helios.checkpoint_cpu')
        require(self.owner._gradient_checkpointing_func is self.original,
                'original checkpoint callable changed before wrapper install',path='helios.checkpoint_cpu')
        self.entered=True;self.token=_ACTIVE.set(self)
        self.owner._gradient_checkpointing_func=self
        return self

    def __exit__(self,kind,error,traceback):
        changed=self.owner._gradient_checkpointing_func is not self
        self.owner._gradient_checkpointing_func=self.original
        _ACTIVE.reset(self.token);self.closed=True
        live=list(self.leases)
        for lease in live:
            if lease.host is not None:
                self.stats['cancel_count']+=1;lease.cancel()
        require(kind is not None or not live,
                'G action retained checkpoint input graph after original backward',path='helios.checkpoint_cpu')
        require(not changed,'checkpoint input wrapper replaced during G action',path='helios.checkpoint_cpu')

    def receipt(self):
        return dict(schema_version=getattr(self.policy,'SESSION_SCHEMA',SCHEMA),policy=self.policy.PRESET_ID,enabled=True,identity=dict(self.identity),
            identity_sha256=self.identity_sha256,capacity=dict(self.capacity),native_contract=self.spec,
            stats=dict(self.stats),boundaries=[dict(row) for row in self.boundaries],closed=self.closed,
            original_callable_restored=self.owner._gradient_checkpointing_func is self.original,
            cache_restore='REBUILT_PER_G_ACTION_NO_SAVED_TENSOR_CACHE_IN_CHECKPOINT',
            node_accounting='EIGHT_DISJOINT_RANK_PARTITIONS_NO_SHARED_OVERSUBSCRIPTION',
            evidence_scope='CPU_SYNTHETIC_TEST_ONLY' if self.device.type=='cpu' else 'ACTUAL_EXECUTED_BOUNDARY_CENSUS_NOT_NUMERICAL_OR_CAPACITY_PROMOTION')


def generator_scope(engine,*,task):
    """Mint the narrow wrapper only after original native physical admission."""
    binding=policy.require_binding()
    from .helios_native_safe_history import admit_engine
    admit_engine(engine)  # Unchanged validator must see the ORIGINAL callable.
    owners=[m for m in engine.module.modules() if '_gradient_checkpointing_func' in vars(m)]
    from helios.modules.transformer_helios import HeliosTransformerBlock
    blocks=[m for m in engine.module.modules() if m is not engine.module and isinstance(m,HeliosTransformerBlock)]
    device=next(engine.module.parameters()).device
    require(device.type=='cuda','native checkpoint CPU policy cannot use a CPU fake owner',path='helios.checkpoint_cpu')
    return GCheckpointInputSession(owner=owners[0],blocks=blocks,
        capacity=binding['identity']['checkpoint_input_capacity'],device=device,
        identity=dict(source=binding['identity']['source_head'], source_fingerprint=binding['identity']['source_fingerprint'],
            stream_sha256=binding['identity']['stream_sha256'],task_sha256=task.digest,
            generator_version=int(engine.global_steps),rank=binding['checkpoint_input_startup']['rank']))


def validate_phase_receipt(receipt,*,with_grad,task_sha256,generator_version=None,rank=None,stage_schedule=None,_policy=None):
    resolved_policy=policy if _policy is None else _policy
    require(isinstance(receipt,dict) and receipt.get('schema_version')==getattr(resolved_policy,'SESSION_SCHEMA',SCHEMA)
            and receipt.get('policy')==resolved_policy.PRESET_ID and receipt.get('enabled') is with_grad,
            'checkpoint CPU executed phase receipt missing/changed',path='helios.checkpoint_cpu')
    if not with_grad:return
    # This is the original executed/source-validated row trace, never a chosen
    # count or a config-derived exit. Later no-grad prefixes cut the sole final
    # x0 loss graph: the last such stage and every following stage must replay.
    require(isinstance(stage_schedule,list) and len(stage_schedule)==3
            and all(isinstance(item,dict) and item.get('stage')==stage
                and type(item.get('denoising_exit')) is int and item['denoising_exit']>=0
                and type(item.get('denoising_step_budget')) is int
                and item['denoising_exit']<item['denoising_step_budget']
                and item.get('call_count')==item['denoising_exit']+1
                for stage,item in enumerate(stage_schedule)),
            'checkpoint CPU original executed stage schedule missing/changed',path='helios.checkpoint_cpu')
    first_live=max([0]+[stage for stage in (1,2) if stage_schedule[stage]['denoising_exit']>0])
    expected_restores=40*(3-first_live)
    spec=resolved_policy.native_contract()
    expected_restored_bytes=40*sum(a*b*c*2 for a,b,c in spec['shapes'][first_live:])
    stats=receipt.get('stats',{})
    require(receipt.get('closed') is True and receipt.get('original_callable_restored') is True
            and receipt.get('identity',{}).get('task_sha256')==task_sha256
            and stats.get('checkpoint_calls')==stats.get('cpu_copy_count')==120
            and stats.get('restore_count')==stats.get('restored_release_count')==expected_restores
            and stats.get('unused_release_count')==120-expected_restores
            and stats.get('release_count')==120 and stats.get('cancel_count')==0
            and stats.get('live_cpu_bytes')==stats.get('live_leases')==0,
            'checkpoint CPU exact executed/copy/restore/lifetime census changed',path='helios.checkpoint_cpu')
    capacity=resolved_policy.validate_capacity(receipt.get('capacity'))
    rank_bytes=capacity['rank_pageable_limit_bytes']+capacity['rank_pinned_limit_bytes']
    identity=receipt['identity']
    expected={(stage,block,tuple(shape)) for stage,shape in enumerate(spec['shapes']) for block in range(40)}
    boundaries=receipt.get('boundaries',[])
    require(receipt.get('native_contract')==spec
            and receipt.get('identity_sha256')==(identity_digest := fingerprint(identity))
            and (generator_version is None or identity.get('generator_version')==generator_version)
            and type(identity.get('rank')) is int and 0<=identity['rank']<16
            and (rank is None or identity['rank']==rank)
            and len(boundaries)==120
            and all(isinstance(r,dict) and type(r.get('stage_index')) is int
                    and type(r.get('block_index')) is int and isinstance(r.get('shape'),list) for r in boundaries)
            and {(r['stage_index'],r['block_index'],tuple(r['shape'])) for r in boundaries}==expected
            and all(r.get('identity_sha256')==identity_digest
                    and all(r.get(key)==value for key,value in identity.items())
                    and r.get('dtype')=='torch.bfloat16' and r.get('device','').startswith('cuda:')
                    and type(r.get('input_version')) is int and r['input_version']>=0
                    and type(r.get('restore_count')) is int and r['restore_count']==int(r['stage_index']>=first_live)
                    and type(r.get('release_count')) is int and r['release_count']==1
                    and r.get('release_outcome')==('restored' if r['stage_index']>=first_live else 'unused')
                    for r in boundaries)
            and stats.get('copied_bytes')==rank_bytes and stats.get('restored_bytes')==expected_restored_bytes
            and 0<stats.get('peak_live_cpu_bytes',0)<=rank_bytes
            and 0<stats.get('peak_live_leases',0)<=120 and stats.get('failed_pack_count')==0,
            'checkpoint CPU shape/dtype/version/source/capacity census changed',path='helios.checkpoint_cpu')
    from distillschedule.training.helios_lifecycle import current_training
    binding=current_training()
    if binding is not None:
        wanted=binding['identity']
        require(capacity==wanted['checkpoint_input_capacity'] and identity.get('source')==wanted['source_head']
                and identity.get('source_fingerprint')==wanted['source_fingerprint']
                and identity.get('stream_sha256')==wanted['stream_sha256'],
                'checkpoint CPU actual source/stream binding changed',path='helios.checkpoint_cpu')
