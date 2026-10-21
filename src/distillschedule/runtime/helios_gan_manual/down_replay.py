"""Default-OFF source-seam frozen-down SAC for unchanged native two-cycle body.

No Torch import at module scope. No prefix inheritance, GPU owner or admission.
Only this opt-in runtime policy owns the added result caches. Original stable
FA3 scope/checkpoint receipts, LoRA wrapper, update/EMA/timer bodies are delegated.
"""
from __future__ import annotations
import contextlib,contextvars,functools,hashlib,inspect,json,os,threading
from pathlib import Path
from .down_contract import HEAD,FA3,GRAPHS,MIN_DRIVER_FREE,ScopeKey,TensorMeta,Leaf,require,select_down

FSDP_SHA='f9af12208185d9891f94e107fbfe7b0fc12c750115d82d9d9f0b4cda8757e4ff'
CHECKPOINT_SHA='b6ac0745d58fc891f0f741b433e4fdd59efb1e9fe991c6b3f2b8fc460f7d5d7a'
CRITIC_BODY_SHA='7078d12bb1c59913bf5f8438785fe097cf22114796f53a489394d3645db4c282'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def meta(t):return TensorMeta(tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device),bool(t.requires_grad),int(t.untyped_storage().data_ptr()),int(t.storage_offset()))
def verify_store(store,key,tensor_type):
    wanted={'aten.addmm.default'}|({FA3}if key.block%2==0 else set())
    require({str(op)for op in store}==wanted,'ONLY_ORIGINAL_EVEN_FA3_PLUS_FROZEN_DOWN_STORE')
    for op,queue in store.items():
        attention=str(op)==FA3
        require(type(queue)is list and len(queue)==(2 if attention else 1),'EXACT_NATIVE_CACHE_FIFO_LENGTH')
        for value in queue:
            whole=value if attention else(value,)
            require(type(whole)in(tuple,list)and len(whole)==(4 if attention else 1),'WHOLE4_ORIGINAL_FA3_RETURN_ABI')
            for index,wrapped in enumerate(whole):
                require(hasattr(wrapped,'val')and hasattr(wrapped,'version')and isinstance(wrapped.val,tensor_type)
                    and bool(wrapped.val.is_cuda)and str(wrapped.val.device)==f'cuda:{key.rank%8}','ACTUAL_LOCAL_GPU_VERSION_WRAPPER')
                if attention and index>=2:require(wrapped.val.numel()==0,'ACTUAL_WHOLE4_EMPTY_ACCUM')
def replace(stack,obj,name,factory):
    old=getattr(obj,name);own=name in vars(obj);raw=vars(obj).get(name)
    setattr(obj,name,factory(old));stack.callback(setattr,obj,name,raw)if own else stack.callback(delattr,obj,name)

class NativeDownSAC:
    def __init__(self,*,torch,rank,enabled,rows,diagnostic_capacity=False,snapshot_dir=None,run_id=None):
        require(type(enabled)is bool and type(diagnostic_capacity)is bool,'EXPLICIT_PLAIN_BOOL_GATES')
        require(type(rank)is int and 0<=rank<16,'EXACT_WORLD16_RANK')
        require(not diagnostic_capacity or enabled and snapshot_dir is not None,'DIAGNOSTIC_CAPACITY_BOUND_OUTPUT_REQUIRED')
        self.torch,self.rank,self.enabled,self.rows=torch,rank,enabled,dict(rows)
        self.diagnostic_capacity,self.snapshot_dir,self.run_id=diagnostic_capacity,snapshot_dir,run_id
        self.active=contextvars.ContextVar('native_C_down_execution',default=None)
        self.leaf=contextvars.ContextVar('native_C_actual_named_down',default=None)
        self.creation=contextvars.ContextVar('native_C_checkpoint_creation',default=None)
        self.local=threading.local();self.current=None;self.stores={};self.calls=0;self.bound=False
        self.state=dict(schema='gan.native_down_sac.v1',source_head=HEAD,rank=rank,enabled=enabled,
            combined_context_count=0,extra_FA3=dict(original=0,replay=0),extra_down=dict(original=0,replay=0),
            graph_drains=[],updates=[],audit_errors=[],capacity_samples=[],additional_CPU_copies=0,
            additional_device_fences=0,additional_collectives=0,G_policy_changed=False,restored=False,
            formal_eligible=False,numerical_parity='NOT_RUN',raw_counts='NOT_MEASURED_UNPROFILED')
        self._restore_targets=[]

    def replace_owned(self,stack,obj,name,factory):
        self._restore_targets.append((obj,name,name in vars(obj),vars(obj).get(name),getattr(obj,name)))
        replace(stack,obj,name,factory)

    def check_key(self,key):
        key.validate();c=self.current
        require(c is not None and not c['closed']and key.rank==self.rank
            and tuple((key.cycle,key.index,key.logical_step,key.version))==c['scope']
            and key.graph==c['graph']and int(self.critic.global_steps)==key.version,'CAPTURED_ACTIVE_C_SCOPE_VERSION')

    def contexts(self,key,old_policy):
        self.check_key(key);cp=self.checkpoint
        require(key not in self.stores,'FRESH_STORE_PER_COMPLETE_KEY')
        signatures=[];position=[0]
        def policy(context,op,*args,**kwargs):
            label='replay'if context.is_recompute else'original'
            self.check_key(key);require(self.active.get()==(key,label),'EXPLICIT_AUTOGRAD_ISSUING_THREAD_SCOPE')
            if str(op)==FA3:
                if old_policy is None:return cp.CheckpointPolicy.PREFER_RECOMPUTE
                decision=old_policy(context,op,*args,**kwargs)
                require(decision is cp.CheckpointPolicy.MUST_SAVE,'UNCHANGED_ORIGINAL_EVEN_FA3_POLICY')
                return decision
            leaf=self.leaf.get()
            if str(op)=='aten.addmm.default'and leaf is not None:
                require(leaf[0]==key,'ACTUAL_NAMED_LINEAR_SCOPE_KEY')
                values=tuple(meta(t)for t in args)
                if select_down(key,leaf[1],values,kwargs,rows=self.rows[(key.logical_step,key.graph)]):
                    signature=tuple((m.shape,m.stride,m.dtype,m.device)for m in values)
                    if context.is_recompute:
                        require(position[0]<len(signatures)and signatures[position[0]]==signature,'DOWN_REPLAY_FIFO_ABI')
                        position[0]+=1
                    else:signatures.append(signature)
                    self.state['extra_down'][label]+=1
                    return cp.CheckpointPolicy.MUST_SAVE
            return old_policy(context,op,*args,**kwargs)if old_policy is not None else cp.CheckpointPolicy.PREFER_RECOMPUTE
        f,r=cp.create_selective_checkpoint_contexts(policy,allow_cache_entry_mutation=False)
        require(f.storage is r.storage,'ONE_FRESH_PUBLIC_SAC_STORE')
        self.stores[key]=f.storage;self.state['combined_context_count']+=1
        @contextlib.contextmanager
        def execution(mode,label):
            self.check_key(key);token=self.active.set((key,label));primary=None
            try:
                with mode:yield
            except BaseException as error:primary=error;raise
            finally:
                try:
                    self.check_key(key)
                    if label=='original':
                        require(len(signatures)==1,'ONE_ORIGINAL_DOWN_RESULT')
                        verify_store(f.storage,key,self.torch.Tensor)
                    else:require(position[0]==len(signatures)==1 and all(not q for q in r.storage.values()),'WHOLE_STORE_FIFO_DRAINED_PER_BLOCK')
                    if self.diagnostic_capacity:self.capacity_boundary(key,label,f.storage)
                except BaseException as error:
                    self.state['audit_errors'].append(dict(key=key.__dict__,execution=label,error_type=type(error).__name__,error=str(error)[:400]))
                    if primary is None:raise
                finally:self.active.reset(token)
        return execution(f,'original'),execution(r,'replay')

    def capacity_boundary(self,key,label,store):
        cuda=self.torch.cuda;device=self.rank%8;free,total=cuda.mem_get_info(device)
        sample=dict(key=key.__dict__,execution=label,free_bytes=int(free),total_bytes=int(total))
        self.state['capacity_samples'].append(sample)
        if free<MIN_DRIVER_FREE:
            try:
                sample.update(allocated_bytes=int(cuda.memory_allocated(device)),reserved_bytes=int(cuda.memory_reserved(device)),
                    max_allocated_bytes=int(cuda.max_memory_allocated(device)),max_reserved_bytes=int(cuda.max_memory_reserved(device)))
                # Only scalar storage metadata: no values/copies/fences/PG ops.
                pointers={};graph_stores=[v for k,v in self.stores.items()if k.graph==key.graph]
                for stored in graph_stores:
                    for values in stored.values():
                        for wrapped in values:
                            for w in wrapped if isinstance(wrapped,(tuple,list))else(wrapped,):
                                s=w.val.untyped_storage();p=(str(w.val.device),int(s.data_ptr()));n=int(s.nbytes())
                                if n:
                                    require(p not in pointers or pointers[p]==n,'GRAPH_ALIAS_STORAGE_SIZE_AGREEMENT');pointers[p]=n
                sample.update(graph_store_unique_bytes=sum(pointers.values()),graph_stored_blocks=len(graph_stores),
                    source_additional_down_payload_NOT_FIT_PROOF=self.rows[(key.logical_step,key.graph)]*5120*2*40)
                payload=dict(schema='gan.native_down_sac.primary_capacity.v1',run_id=self.run_id,source_head=HEAD,
                    rank=self.rank,error='C_CACHE_CAPACITY_MIN_2GIB_DRIVER_FREE',min_driver_free_bytes=MIN_DRIVER_FREE,
                    sample=sample,stage='BEFORE_PRIMARY_RAISE_BEFORE_TEARDOWN',source_sha256=sha(__file__),formal_eligible=False)
                directory=Path(self.snapshot_dir);require(directory.is_dir()and not directory.is_symlink(),'BOUND_SCALAR_OUTPUT')
                target=directory/f'native_down_primary_rank{self.rank:02d}_cycle{key.cycle}_index{key.index}_{key.graph}_block{key.block}.json'
                raw=(json.dumps(payload,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<=65536,'SCALAR_FAILURE_64KiB_BOUND')
                tmp=target.with_name(target.name+'.tmp-'+str(os.getpid()))
                try:
                    with tmp.open('xb')as out:out.write(raw);out.flush();os.fsync(out.fileno())
                    os.link(tmp,target);fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
                    try:os.fsync(fd)
                    finally:os.close(fd)
                finally:
                    if tmp.exists():tmp.unlink()
            except BaseException as error:self.state['audit_errors'].append(dict(snapshot_error=type(error).__name__,error=str(error)[:400]))
            raise ValueError('C_CACHE_CAPACITY_MIN_2GIB_DRIVER_FREE')

    def bind(self,stack,*,p0,native,fsdp,checkpoint,generator,critic,ema):
        if not self.enabled:return
        require(not self.bound and sha(fsdp.__file__)==FSDP_SHA and sha(checkpoint.__file__)==CHECKPOINT_SHA,'EXACT_NATIVE_PUBLIC_SAC_SOURCE')
        require(self.torch.__version__=='2.10.0+cu128'and fsdp.SELECTIVE_FLASH_ATTN_OP==FA3,'PINNED_TORCH_STABLE_RAW_FA3')
        require((int(generator.global_steps),int(critic.global_steps),int(ema.optimization_step))==(2,10,800),'EXACT_NATIVE_INITIAL_VERSIONS')
        require(set(self.rows)=={(s,g)for s in range(1000,1005)for g in GRAPHS}and all(type(x)is int and x>0 for x in self.rows.values()),'ALL5_SOURCE_ATTESTED_ROW_TABLE')
        from helios.utils.utils_helios_post import _critic_loss
        body=inspect.getsource(_critic_loss).encode()
        require(hashlib.sha256(body).hexdigest()==CRITIC_BODY_SHA,'EXACT_ORIGINAL_CRITIC_BODY')
        self.generator,self.critic,self.ema,self.checkpoint=generator,critic,ema,checkpoint;self.bound=True
        blocks=tuple(critic.module.blocks);require(len(blocks)==40 and len({id(b)for b in blocks})==40,'UNIQUE_ACTUAL_C_BLOCK_OWNERS')
        self.block_ids={id(b):i for i,b in enumerate(blocks)}
        named=dict(critic.module.named_modules())
        for i in range(40):
            name=f'blocks.{i}.ffn.net.2.base_layer';module=named.get(name)
            require(isinstance(module,self.torch.nn.Linear)and tuple(module.weight.shape)==(5120,13824)
                and module.bias is not None and tuple(module.bias.shape)==(5120,),'ALL40_SOURCE_NAMED_DOWN_LINEAR')
            require(not module.weight.requires_grad and not module.bias.requires_grad,'ALL40_FROZEN_DOWN_BASE')
            def before(module,args,kwargs,index=i,name=name):
                active=self.active.get()
                if self.current is None:return
                require(active is not None and active[0].block==index and not kwargs and len(args)==1,'NAMED_DOWN_CAPTURED_SCOPE_ARGS')
                key,label=active;self.check_key(key)
                token=self.leaf.set((key,Leaf('critic.'+name,meta(args[0]),meta(module.weight),meta(module.bias))))
                tokens=getattr(self.local,'tokens',None)
                if tokens is None:tokens={};self.local.tokens=tokens
                tokens.setdefault(id(module),[]).append(token)
            def after(module,args,kwargs,output):
                tokens=getattr(self.local,'tokens',{}).get(id(module),[])
                if tokens:self.leaf.reset(tokens.pop())
            stack.callback(module.register_forward_pre_hook(before,with_kwargs=True).remove)
            stack.callback(module.register_forward_hook(after,with_kwargs=True,always_call=True).remove)
        def root_before(module,args,kwargs):
            c=self.current
            if c is None:return
            require(not args and c['forwards']==c['backwards']and c['forwards']<2,'EXACT_C_FORWARD_BACKWARD_GRAPH_ORDER')
            hidden=kwargs.get('hidden_states');ordinal=c['forwards'];graph=GRAPHS[ordinal]
            require(hidden is not None and hidden.ndim==5 and str(hidden.dtype)=='torch.bfloat16'
                and hidden.shape[0]==(1 if ordinal==0 else 3)and bool(kwargs.get('gan_mode',False))==(ordinal==1),'SOURCE_B1_B3_ROOT_GRAPH_BINDING')
            c['graph']=graph;c['forwards']+=1
        stack.callback(critic.module.register_forward_pre_hook(root_before,with_kwargs=True).remove)
        def dispatch(original):
            @functools.wraps(original)
            def call(module,*args,**kwargs):
                if self.current is None or id(module)not in self.block_ids:return original(module,*args,**kwargs)
                c=self.current;cycle,index,step,version=c['scope']
                key=ScopeKey(self.rank,cycle,index,step,version,c['graph'],self.block_ids[id(module)]).validate();self.check_key(key)
                require(not kwargs.get('operation_source',False)and 'context_fn'not in kwargs,'UNCHANGED_NATIVE_CHECKPOINT_DOMAIN')
                selected=kwargs.get('selective_flash_receipt')
                require((selected is not None)==(key.block%2==0),'ORIGINAL_EVEN20_SELECTION')
                token=self.creation.set(key)
                try:
                    if selected is None:return original(module,*args,**dict(kwargs,context_fn=lambda:self.contexts(key,None)))
                    return original(module,*args,**kwargs)
                finally:self.creation.reset(token)
            return call
        self.replace_owned(stack,fsdp,'fsdp2_metadata_tolerant_checkpoint',dispatch)
        def factory(original):
            @functools.wraps(original)
            def call(receipt):
                key=self.creation.get()
                if key is None:return original(receipt)
                f,r=original(receipt);require(f.policy_fn is r.policy_fn,'ORIGINAL_POLICY_PAIR')
                return self.contexts(key,f.policy_fn)
            return call
        self.replace_owned(stack,fsdp,'_selective_flash_checkpoint_contexts',factory)
        def backward(original):
            @functools.wraps(original)
            def call(*args,**kwargs):
                c=self.current
                if c is None:return original(*args,**kwargs)
                require(c['forwards']==c['backwards']+1,'EXACT_TWO_GRAPH_BACKWARD_ORDER')
                result=original(*args,**kwargs)
                keys=[k for k in self.stores if k.graph==c['graph']]
                require(len(keys)==40 and {k.block for k in keys}==set(range(40))
                    and all(all(not q for q in self.stores[k].values())for k in keys),'ALL40_FRESH_GRAPH_STORES_DRAINED')
                for k in keys:del self.stores[k]
                self.state['graph_drains'].append(dict(scope=list(c['scope']),graph=c['graph']))
                c['backwards']+=1;return result
            return call
        self.replace_owned(stack,critic,'backward',backward)
        def commit(original):
            @functools.wraps(original)
            def call(*args,**kwargs):
                c=self.current
                if c is None:return original(*args,**kwargs)
                require(c['forwards']==c['backwards']==2 and c['commits']==0 and not self.stores,'DRAIN_BEFORE_ORIGINAL_C_COMMIT')
                require(int(critic.global_steps)==c['scope'][3],'ORIGINAL_PRECOMMIT_VERSION')
                result=original(*args,**kwargs);require(int(critic.global_steps)==c['scope'][3]+1,'ORIGINAL_POSTCOMMIT_VERSION')
                c['commits']+=1;return result
            return call
        self.replace_owned(stack,critic,'step',commit)
        def run_critic(original):
            @functools.wraps(original)
            def call(*args,**kwargs):
                require(not args and kwargs.get('critic_engine')is critic and self.current is None and not self.stores,'EXACT_C_HELPER_OWNED_ENGINE_ONCE')
                call_index=self.calls;require(call_index<10,'EXACT10_NATIVE_C_UPDATES')
                cycle,index=1+call_index//5,call_index%5;step=1000+index;version=10+call_index
                require(kwargs.get('step')==step and kwargs.get('rank')==self.rank and int(critic.global_steps)==version
                    and int(generator.global_steps)==2+cycle and int(ema.optimization_step)==800+cycle-1+(index>0),'SEQUENTIAL_NATIVE_CYCLE_SOURCE_VERSION_HISTORY')
                c=dict(scope=(cycle,index,step,version),graph=None,forwards=0,backwards=0,commits=0,closed=False)
                self.current=c;primary=None
                try:
                    result=original(*args,**kwargs)
                    require(c['forwards']==c['backwards']==2 and c['commits']==1 and not self.stores,'NATIVE_C_HELPER_TWO_GRAPHS_ONE_COMMIT')
                    self.state['updates'].append(dict(cycle=cycle,index=index,logical_step=step,before=version,after=int(critic.global_steps)))
                    self.calls+=1;return result
                except BaseException as error:primary=error;raise
                finally:
                    c['closed']=True;self.current=None
                    if primary is not None:
                        self.state['audit_errors'].append(dict(primary_type=type(primary).__name__,primary=str(primary)[:400],scope=list(c['scope'])))
                        self.stores.clear() # Drop only policy-owned refs; never original autograd/FSDP owners.
            return call
        self.replace_owned(stack,p0,'_run_gan_critic_precomputed',run_critic)

    def finish(self):
        if not self.enabled:return
        require(self.calls==10 and self.current is None and not self.stores and len(self.state['graph_drains'])==20,'COMPLETE10_UPDATES20_DRAINED_GRAPHS')
        require(self.state['combined_context_count']==800 and self.state['extra_FA3']==dict(original=0,replay=0)
            and self.state['extra_down']==dict(original=800,replay=800)and not self.state['audit_errors'],'EXACT_DOWNONLY_TRAINING_COUNTERS')
        require((int(self.generator.global_steps),int(self.critic.global_steps),int(self.ema.optimization_step))==(4,20,802),'FINAL_NATIVE_G4C20EMA802')

    def restored(self):
        require(self.current is None and not self.stores,'NO_LIVE_POLICY_CACHE_ON_RESTORE')
        for obj,name,own,raw,old in self._restore_targets:
            require((name in vars(obj))==own and (vars(obj).get(name)is raw if own else getattr(obj,name)==old),'ORIGINAL_NATIVE_CALLABLE_OWNERSHIP_RESTORED')
        self.state['restored']=True
