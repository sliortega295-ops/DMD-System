"""Generic callable backend: workload code describes math, not optimization.

State/feeds are cloned together for paired calibration, preserving optimizer
parameter identity. Current lowering is serial, single-device resident. It
supports arbitrary DAGs and automatic immutable-parameter aliasing/checkpoint
capabilities, not arbitrary FSDP or cross-pool dispatch. Original distributed
backends are admitted separately by the common planning service.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import inspect
import itertools
import math
import random
import time
from typing import Any, Callable, Mapping

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from distillschedule.core.contracts import require
from distillschedule.runtime.role_runner import StaticRoleRunner
from distillschedule.runtime.version_sync import VersionLedger
from distillschedule.workloads.program import Program
from distillschedule.workloads.best_known.registry import fingerprint


def _stateful(value):
    return callable(getattr(value, 'state_dict', None)) and callable(getattr(value, 'load_state_dict', None))


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, nn.Module):
        yield from value.parameters(); yield from value.buffers()
        for p in value.parameters():
            if p.grad is not None: yield p.grad
    elif isinstance(value, torch.optim.Optimizer):
        yield from tensors(value.state)
    elif _stateful(value):
        live = getattr(value, 'runtime_tensors', None)
        if live is None:
            yield from tensors(value.state_dict())
        else:
            require(callable(live), 'runtime_tensors must be callable', path='open.state')
            for tensor in live():
                require(isinstance(tensor, torch.Tensor),
                        'runtime_tensors must yield live tensors', path='open.state')
                yield tensor
    elif isinstance(value, Mapping):
        for v in value.values(): yield from tensors(v)
    elif isinstance(value, (tuple, list)):
        for v in value: yield from tensors(v)


def physical_tensor(t):
    from torch.distributed.tensor import DTensor
    return t.to_local() if isinstance(t, DTensor) else t


def full_tensor(t):
    from torch.distributed.tensor import DTensor
    return t.full_tensor() if isinstance(t, DTensor) else t


def storage_key(t):
    local = physical_tensor(t)
    return (str(local.device), local.untyped_storage().data_ptr(), local.untyped_storage().nbytes())


def storage_bytes(value):
    return sum({storage_key(t): physical_tensor(t).untyped_storage().nbytes() for t in tensors(value)}.values())


def tensor_signature(t):
    t=full_tensor(t)
    v=t.detach().contiguous().cpu()
    return fingerprint({'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),
        'bytes':hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()})


def signature(value, *, _metadata=False):
    if isinstance(value, torch.Tensor): return {'tensor':tensor_signature(value)}
    if isinstance(value, nn.Module):
        return {'class':type(value).__module__+'.'+type(value).__qualname__,
                'training':[ (n,m.training) for n,m in value.named_modules()],
                'tensors':{k:signature(v) for k,v in value.state_dict().items()},
                'requires_grad':{k:p.requires_grad for k,p in value.named_parameters(remove_duplicate=False)}}
    if isinstance(value, torch.optim.Optimizer): return {'optimizer':type(value).__qualname__, 'state':signature(value.state_dict())}
    if _stateful(value):
        return {'stateful':type(value).__module__+'.'+type(value).__qualname__,
                'state':signature(value.state_dict(), _metadata=True)}
    if isinstance(value, Mapping): return {str(k):signature(v, _metadata=_metadata) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [signature(v, _metadata=_metadata) for v in value]
    # E.g. ReduceLROnPlateau deliberately starts with best=+/-inf. Bind
    # those scheduler sentinels explicitly without admitting nonfinite inputs.
    if _metadata and type(value) is float and math.isinf(value):
        return {'stateful_float_sentinel':repr(value)}
    require(value is None or type(value) in (int,float,bool,str),'state must have an explicit serializable signature',path='open.state')
    return value


def code_signature(fn):
    """Bind real source, not a user assertion that a named workload is unchanged."""
    obj=fn if inspect.isfunction(fn) or inspect.isclass(fn) else type(fn)
    try: text=inspect.getsource(obj)
    except (OSError,TypeError):
        raise ValueError('bind inspectable Python functions/classes, not opaque or interactive closures')
    closure = inspect.getclosurevars(fn).nonlocals if inspect.isfunction(fn) else {}
    # Immutable scalar closure values are part of the actual code binding.
    return fingerprint({'source':text,'closure':signature(closure)})


@dataclass
class Workload:
    program: Program
    bindings: Mapping[str, Callable]
    state: Mapping[str, Any]
    feeds: Mapping[str, Any]
    seed: int = 0
    # Semantic declaration: these model subtree parameters are immutable.
    # The backend checks requires_grad=False and validates contents after execution.
    # This is not an instruction to share them; the capability library decides.
    immutable_modules: tuple[str, ...] = ()
    identity_extra: Mapping[str, Any] = field(default_factory=dict)
    replica_semantics: Any = None

    def validate(self):
        require(isinstance(self.program,Program),'Program required',path='open.workload')
        require(set(self.bindings)=={c.binding for c in self.program.calls},'exact callable bindings required',path='open.workload')
        require(all(callable(f) for f in self.bindings.values()),'non-callable binding',path='open.workload')
        require(set(self.state)==set(self.program.initial_versions),'all logical states must be bound',path='open.workload')
        require(set(self.feeds)==set(self.program.feeds),'feed binding mismatch',path='open.workload')
        require(type(self.seed) is int and 0<=self.seed<2**63,'invalid seed',path='open.workload')
        require(all(t.grad_fn is None for t in tensors(self.feeds)),'feeds must not carry external autograd graphs',path='open.workload')
        device={str(t.device) for t in tensors((self.state,self.feeds))}
        require(len(device)<=1,'generic resident backend currently requires one device',path='open.workload')
        for path in self.immutable_modules:
            module=resolve_module(self.state,path)
            require(all(not p.requires_grad for p in module.parameters()),'immutable module has trainable parameters',path='open.workload')
        require(all(p.grad is None for obj in self.state.values() if isinstance(obj,nn.Module)
                    for p in obj.parameters()), 'entry must be an explicit zero-gradient boundary',path='open.workload')
        owners={}
        for key,obj in self.state.items():
            if not isinstance(obj,nn.Module): continue
            for p in obj.parameters():
                if p.requires_grad:
                    storage=storage_key(p)
                    require(storage not in owners or owners[storage]==key,
                            'trainable storage aliased across logical models',path='open.workload')
                    owners[storage]=key
        # Replayed closures must use ctx.state, not capture a second model/optimizer.
        for fn in self.bindings.values():
            require(inspect.isfunction(fn),'bindings must be inspectable functions using ctx.state',path='open.workload')
            captured=inspect.getclosurevars(fn)
            require(not any(isinstance(v,(nn.Module,torch.optim.Optimizer,torch.Tensor))
                            for v in (*captured.nonlocals.values(),*captured.globals.values())),
                    'model/tensor state captured outside ctx.state or ctx.inputs',path='open.workload')
        if self.replica_semantics is not None:
            from .replica_contract import ReplicaSemantics
            require(isinstance(self.replica_semantics, ReplicaSemantics), 'typed replica semantics required', path='open.workload')
            self.replica_semantics.validate(self)
        from .owner_execution import inventory
        inventory(self.state)
        return self

    @property
    def device(self):
        return next((t.device for t in tensors((self.state,self.feeds))),torch.device('cpu'))

    @property
    def identity(self):
        self.validate()
        classes={type(m) for obj in self.state.values() if isinstance(obj,nn.Module) for m in obj.modules()}
        from .owner_execution import inventory
        owners = inventory(self.state)
        return {'program':self.program.digest,'initial_state':fingerprint(signature(self.state)),
                'feeds':fingerprint(signature(self.feeds)), 'seed':self.seed,
                'bindings':{k:code_signature(f) for k,f in self.bindings.items()},
                'model_code':{k.__module__+'.'+k.__qualname__:code_signature(k) for k in classes},
                'immutable_modules':list(self.immutable_modules), 'extra':dict(self.identity_extra),
                'torch':torch.__version__,'device':str(self.device),
                **({'owner_execution': owners} if owners else {}),
                **({'replica_semantics':self.replica_semantics.to_dict()} if self.replica_semantics is not None else {})}


def unwrap_module(value):
    # Preserve semantic parameter names after a framework-owned DDP wrapper.
    from torch.nn.parallel import DistributedDataParallel
    return value.module if isinstance(value, DistributedDataParallel) else value


def resolve_module(state,path):
    root,_,rest=path.partition('.')
    require(root in state and isinstance(state[root],nn.Module),'immutable module root missing',path='open.state')
    module = unwrap_module(state[root])
    return module.get_submodule(rest) if rest else module


def sharing_inventory(workload, *, include_existing=False):
    # Inspect every named immutable slot, even when two slots already alias.
    # Parameter ownership/optimizers still use unique Parameter objects. Without
    # slot-complete snapshots, equal DiT norm/bias aliases erase verification
    # keys and can make a valid immutable share look like a value mutation.
    seen={}; pairs=[]
    for path in workload.immutable_modules:
        for name,p in resolve_module(workload.state,path).named_parameters(remove_duplicate=False):
            require(p.is_contiguous() and p.layout==torch.strided,'shared parameter must be contiguous strided storage',path='open.state')
            key=tensor_signature(p)
            item=(path,name,p)
            if key in seen and (include_existing or seen[key][2] is not p):
                pairs.append((seen[key][:2],item[:2],p.numel()*p.element_size()))
            else: seen[key]=item
    return pairs


def candidates(workload, *, limit=16):
    """Global capabilities: match facts, never branch on workload/model names."""
    workload.validate()
    require(type(limit) is int and limit>=1,'candidate budget required',path='open.capabilities')
    groups=sorted({c.binding for c in workload.program.calls if c.replay_safe})
    shares=(False,True) if sharing_inventory(workload) else (False,)
    from .checkpoint_policy import POLICIES, operations
    policies=tuple(p for p in POLICIES if operations(p))
    from .owner_execution import inventory, alternatives
    owners = inventory(workload.state)
    total=len(shares)*(1+(1+len(policies))*(2**len(groups)-1))
    total *= math.prod(len(row['descriptor']['options']) for row in owners.values())
    result=[]
    # Baseline and ordinary singleton/full checkpoint choices first; then give
    # selective policies budget before lower-priority combinatorial subsets.
    primary=[()] + [(g,) for g in groups]
    if len(groups)>1: primary.append(tuple(groups))
    rest=itertools.chain.from_iterable(itertools.combinations(groups,n) for n in range(2,len(groups)))
    visited=set()
    def append(selected,policy):
        key=(selected,policy)
        if key in visited: return
        visited.add(key)
        for share in shares:
            for owner_options in alternatives(owners):
                if len(result)>=limit: return
                option={'share_immutable':share,'checkpoint_bindings':list(selected),'backend':'torch-callable-resident-v1'}
                if policy!='recompute': option['checkpoint_policy']=policy
                if owner_options: option['owner_execution']=owner_options
                result.append(option)
    for selected in primary: append(selected,'recompute')
    for policy in policies:
        for selected in primary:
            if selected: append(selected,policy)
    for selected in rest:
        if len(result)>=limit: break
        for policy in ('recompute',*policies): append(selected,policy)
    return {'candidates':result,'total_candidates':total,'exhaustive':len(result)==total,
            'capabilities':['reference','nonreentrant_checkpoint','immutable_parameter_alias',*policies,
                            *(['stateful_owner_execution'] if any(len(row['descriptor']['options']) > 1
                                                                  for row in owners.values()) else [])],
            'layout':'single-device-resident','cross_call_overlap':False}



@contextmanager
def random_scope(seed,key,device, *, namespace=None):
    # Namespace is pure hash input, not another RNG draw. Omitting it preserves
    # every historical logical-call seed byte. All ranks share the semantic
    # namespace schema; only execution injects the actual owned rank/world.
    binding = {'seed':seed,'logical_call':key}
    if namespace is not None: binding['namespace'] = namespace
    derived=int(fingerprint(binding)[:16],16)%(2**63)
    py=random.getstate()
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(derived); random.seed(derived)
            if devices:
                with torch.cuda.device(devices[0]): torch.cuda.manual_seed(derived)
            yield
    finally: random.setstate(py)


@dataclass(frozen=True)
class Context:
    inputs: Mapping[str,Any]
    state: Mapping[str,Any]
    attributes: Mapping[str,Any]


class Session:
    def __init__(self, workload: Workload, implementation, *, lifecycle=None, copy_state=True, persistent=False):
        workload.validate()
        from .checkpoint_policy import POLICIES, operations
        from .owner_execution import admit, inventory
        self.owner_inventory = inventory(workload.state)
        self.owner_options = admit(implementation, self.owner_inventory)
        policy=implementation.get('checkpoint_policy','recompute')
        require(set(implementation) - {'owner_execution'} in ({'share_immutable','checkpoint_bindings','backend'},
                    {'share_immutable','checkpoint_bindings','backend','checkpoint_policy'})
                and (policy=='recompute' or (policy in POLICIES and operations(policy)
                     and bool(implementation['checkpoint_bindings'])))
                and implementation['backend']=='torch-callable-resident-v1'
                and type(implementation['share_immutable']) is bool
                and (not implementation['share_immutable'] or bool(sharing_inventory(workload, include_existing=True)))
                and isinstance(implementation['checkpoint_bindings'],list)
                and len(set(implementation['checkpoint_bindings']))==len(implementation['checkpoint_bindings'])
                and set(implementation['checkpoint_bindings']) <= {c.binding for c in workload.program.calls if c.replay_safe},
                'implementation not admitted by capability predicates',path='open.runtime')
        self.workload=workload; self.implementation=deepcopy(implementation)
        require(type(copy_state) is bool and type(persistent) is bool,
                'explicit state ownership required', path='open.runtime')
        self.persistent = persistent or not copy_state
        # Paired trials keep the original clone semantics. A persistent trainer
        # explicitly owns its modules/optimizers and must not clone them at each
        # update boundary (including preserving optimizer Parameter ownership).
        self.state,self.feeds=(deepcopy((dict(workload.state),dict(workload.feeds)))
                               if copy_state else (dict(workload.state),dict(workload.feeds)))
        self.device=workload.device; self.graph=workload.program.compile()
        self.profile={}; self.effects={'shared_parameter_aliases':0,'checkpoint_calls':0,'recompute_calls':0}
        self._immutable={path:{n:tensor_signature(p) for n,p in resolve_module(self.state,path).named_parameters(remove_duplicate=False)}
                         for path in workload.immutable_modules}
        if implementation['share_immutable']:
            for first,second,_ in sharing_inventory(workload):
                source=resolve_module(self.state,first[0]).get_parameter(first[1])
                target=resolve_module(self.state,second[0]); parent,_,leaf=second[1].rpartition('.')
                setattr(target.get_submodule(parent) if parent else target,leaf,source)
                self.effects['shared_parameter_aliases']+=1
        self.initial_storage=storage_bytes((self.state,self.feeds))
        self.static_bytes=self.initial_storage
        self._ran=False
        self.lifecycle = lifecycle
        if lifecycle is not None:
            lifecycle.prepare(self)
            self.effects['lifecycle_installations'] = 1
            self.initial_storage = storage_bytes((self.state, self.feeds))
            self.static_bytes = self.initial_storage

    def bind_cycle(self, workload, *, graph=None):
        """Rebind an owned complete-cycle session; never reinstall its layout.

        The public trainer verifies structural/source identity. This final
        ownership check prevents a rebuilt optimizer or model from silently
        replacing the parameters retained by DDP/FSDP and their hooks.
        Trial sessions remain one-shot unless explicitly adopted as persistent.
        """
        require(self.persistent, 'trial sessions are one-shot', path='open.runtime')
        workload.validate()
        from .owner_execution import inventory
        require(inventory(workload.state) == self.owner_inventory,
                'owner execution descriptor or source changed', path='open.runtime')
        require(set(workload.state) == set(self.state) and all(
            workload.state[key] is self.state[key] for key in self.state),
            'cycle must retain the owned state objects', path='open.runtime')
        if self._ran:
            require(workload.program.initial_versions == self.receipt['dispatcher']['final_versions'],
                    'cycle must continue the last committed versions', path='open.runtime')
        self.workload = workload
        self.feeds = dict(workload.feeds)
        self.graph = graph if graph is not None else workload.program.compile()
        self.profile = {}
        self._ran = False
        if self.lifecycle is not None:
            self.lifecycle.workload = workload

    def synchronize(self):
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)

    def run(self, order=None, *, measure=False):
        require(not self._ran,'sessions are one-shot; clone original initial state for each trial',path='open.runtime')
        self._ran=True
        resident = None
        if measure and self.device.type == 'cuda':
            from .resident_trial import cuda_trial_active
            resident = next((owner._resident_trial_active for owner in self.state.values()
                             if cuda_trial_active(owner)), None)
        by_id={c.id:c for c in self.workload.program.calls}
        def dispatch(ctx):
            c=by_id[ctx.operation.operation_id]
            if self.lifecycle is not None: self.lifecycle.before_call(self, c)
            args={k:self.feeds[k] if k in self.feeds else ctx.values[k] for k in c.inputs}
            states={k:self.state[k] for k in ctx.operation.required_versions}
            inputs=tuple(args.values()); input_keys=tuple(args)
            invocations=0
            def invoke(*values):
                nonlocal invocations
                invocations+=1
                if invocations>1: self.effects['recompute_calls']+=1
                trace_context = (torch.profiler.record_function('distillschedule.call.'+c.id)
                                 if getattr(self,'deep_trace_labels',False) else nullcontext())
                namespace = None
                semantics = self.workload.replica_semantics
                if semantics is not None and semantics.random_streams == 'rank_independent_v1':
                    import torch.distributed as dist
                    require(dist.is_initialized() and dist.get_world_size() > 1,
                            'rank-independent streams require an actual replica group', path='open.runtime')
                    namespace = dict(schema=semantics.random_streams, rank=dist.get_rank(), world_size=dist.get_world_size())
                with trace_context, random_scope(self.workload.seed,c.id,self.device,namespace=namespace):
                    return self.workload.bindings[c.binding](Context(dict(zip(input_keys,values)),states,c.attributes))
            saved={}; excluded={storage_key(t) for t in tensors((self.state,self.feeds))}
            def pack(t):
                k=storage_key(t)
                if k not in excluded: saved[k]=physical_tensor(t).untyped_storage().nbytes()
                return t
            if measure:
                self.synchronize()
            if measure and self.device.type=='cuda':
                before=torch.cuda.memory_allocated(self.device)
                # ResidentTrial owns device-global counters across the complete
                # atomic native phase, including setup and comparison. Never
                # erase its high-water evidence at a nested call boundary.
                if resident is None: torch.cuda.reset_peak_memory_stats(self.device)
            start_event = end_event = None
            if measure and self.device.type == 'cuda':
                # GPU device time is a diagnostic; conservative scheduling cost
                # remains host wall time through the synchronization boundary.
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(torch.cuda.current_stream(self.device))
            start=time.perf_counter_ns()
            with torch.set_grad_enabled(c.grad!='no_grad'):
                with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
                    if c.replay_safe and c.binding in self.implementation['checkpoint_bindings']:
                        self.effects['checkpoint_calls']+=1
                        policy=self.implementation.get('checkpoint_policy','recompute')
                        options={}
                        if policy!='recompute':
                            from .checkpoint_policy import contexts
                            options['context_fn']=lambda:contexts(policy,saved,excluded,self.effects)
                        value=checkpoint(invoke,*inputs,use_reentrant=False,preserve_rng_state=True,**options)
                    else: value=invoke(*inputs)
            if c.grad=='no_grad':
                from torch.utils._pytree import tree_map
                value=tree_map(lambda v:v.detach() if isinstance(v,torch.Tensor) else v,value)
            if self.lifecycle is not None: self.lifecycle.after_call(self, c)
            if end_event is not None:
                end_event.record(torch.cuda.current_stream(self.device))
            if measure:
                self.synchronize()
            duration=max(1,time.perf_counter_ns()-start)
            gpu_event_ns = (max(1,int(round(start_event.elapsed_time(end_event)*1_000_000)))
                            if start_event is not None else None)
            for t in tensors(value):
                k=storage_key(t)
                if k not in excluded: saved[k]=physical_tensor(t).untyped_storage().nbytes()
            retained=sum(saved.values())
            workspace=0
            if measure and self.device.type=='cuda':
                after=torch.cuda.memory_allocated(self.device)
                retained=max(retained,after-before,0)
                peak=torch.cuda.max_memory_allocated(self.device)
                workspace=(max(0,peak-before-retained) if resident is None else
                           max(0,peak-resident._phase_start['allocated_bytes']))
            self.static_bytes=max(self.static_bytes,storage_bytes((self.state,self.feeds)))
            self.profile[c.id]={'duration_ns':duration,'retained_bytes':retained,'workspace_bytes':workspace,
                'timing_method':('synchronized_component_host_wall' if measure
                                 else 'host_dispatch_wall_not_gpu_completion')}
            if gpu_event_ns is not None:
                self.profile[c.id].update(cuda_event_ns=gpu_event_ns,
                    timing_method='synchronized_cuda_event_and_host_wall',
                    capacity_measurement=('isolated_operation_allocator_peak' if resident is None
                                          else 'resident_phase_high_water_bound'),
                    allocator_peak_counter_reset=resident is None,
                    peak_allocated_bytes=peak,
                    # Host wall time, which includes NCCL completion at this
                    # call boundary, is the optimization objective.
                    scheduling_time_source='host_wall_ns')
            return value
        def finish(ctx):
            self.profile['__open_finish__']={'duration_ns':0,'retained_bytes':0,'workspace_bytes':0}
        runner=StaticRoleRunner('open_runtime',{'call':dispatch,'barrier':finish},versions=VersionLedger(self.graph.metadata['initial_versions']))
        prepared=runner.prepare(self.graph,order=tuple(order or (*[c.id for c in self.workload.program.calls],'__open_finish__')),retain_outputs=self.workload.program.outputs)
        if self.lifecycle is not None: self.lifecycle.before_run(self, prepared.order if hasattr(prepared, 'order') else tuple(order or (*[c.id for c in self.workload.program.calls], '__open_finish__')))
        started=time.perf_counter_ns()
        with ExitStack() as scopes:
            if self.owner_options:
                from .owner_execution import inventory
                require(inventory(self.state) == self.owner_inventory,
                        'owner execution descriptor or source changed', path='open.runtime')
                # Per-run observations, not cumulative success from an earlier cycle.
                self.effects['owner_execution'] = {}
                for key, option in self.owner_options.items():
                    row = {'option': deepcopy(option),
                           'effect': deepcopy(self.owner_inventory[key]['descriptor']['effect'])}
                    self.effects['owner_execution'][key] = row
                    scopes.enter_context(self.state[key].execution_scope(deepcopy(option), row))
            outputs=prepared.run();self.synchronize()
            if self.lifecycle is not None: self.lifecycle.finish(self)
        for key, option in self.owner_options.items():
            row = self.effects['owner_execution'][key]
            require(row.get('option') == option
                    and row.get('effect') == self.owner_inventory[key]['descriptor']['effect'],
                    'owner changed admitted execution effect predicates', path='open.runtime')
        for path,values in self._immutable.items():
            require({n:tensor_signature(p) for n,p in resolve_module(self.state,path).named_parameters(remove_duplicate=False)}==values,
                    'declared immutable parameters changed',path='open.runtime')
        self.receipt={'status':'EXECUTED_OPEN_PROGRAM','program':self.workload.program.digest,
            'dispatcher':prepared.receipt(),'effects':deepcopy(self.effects),'wall_ns':time.perf_counter_ns()-started,
            'device':str(self.device),'gpu_execution':'NOT_RUN' if self.device.type=='cpu' else 'NATIVE_CUDA_RUN',
            'claimable_speedup':False}
        return outputs

    def result_state(self, outputs):
        def clone(v):
            if isinstance(v,torch.Tensor): return full_tensor(v).detach().cpu().clone()
            if isinstance(v,nn.Module): return clone(unwrap_module(v).state_dict())
            if isinstance(v,torch.optim.Optimizer): return clone(v.state_dict())
            if _stateful(v): return clone(v.state_dict())
            if isinstance(v,Mapping): return {str(k):clone(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)): return [clone(x) for x in v]
            return v
        return clone({'state':self.state,'outputs':outputs,'grads':{k:{n:p.grad for n,p in unwrap_module(v).named_parameters(remove_duplicate=False)}
                     for k,v in self.state.items() if isinstance(v,nn.Module)}})


def compare(a,b,*,rtol=1e-5,atol=1e-7):
    errors=[]
    def visit(x,y,path):
        if isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor):
            if x.shape!=y.shape or x.dtype!=y.dtype or not torch.allclose(x,y,rtol=rtol,atol=atol): errors.append(path)
        elif isinstance(x,dict) and isinstance(y,dict):
            if x.keys()!=y.keys(): errors.append(path)
            else:
                for k in x: visit(x[k],y[k],path+'/'+k)
        elif isinstance(x,list) and isinstance(y,list) and len(x)==len(y):
            for i,(xx,yy) in enumerate(zip(x,y)): visit(xx,yy,path+'/'+str(i))
        elif type(x)!=type(y) or x!=y: errors.append(path)
    visit(a,b,'')
    return {'status':'PASS' if not errors else 'REVIEW','mismatches':errors,'rtol':rtol,'atol':atol}
