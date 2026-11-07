"""Measured, capability-bounded G/Score pool search over the shared frontier solver.

Only role splits and FIFO in-flight windows with an actual native lowering are
admitted. The internal G/S overlap is measured as a WHOLE cycle; a single
resource envelope is passed to the common DP to avoid predicting an arbitrary
interleaving which the native runtime cannot execute. This is a deliberately
conservative but honest first compositional pool backend.
"""
from __future__ import annotations
import hashlib
import ast
import builtins
import dis
import inspect
import json
from pathlib import Path
import statistics
import sysconfig
import time
import textwrap
from dataclasses import asdict

import torch
import torch.distributed as dist
from distillschedule.core.contracts import require
from distillschedule.core.enums import OperationKind, GradientMode
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation
from distillschedule.search.cost_model import CostModel, OperationCost, ComponentContract
from distillschedule.search.plan_ir import BindingRule, StateAllocation
from distillschedule.search.service import BoundCandidate, select
from distillschedule.workloads.best_known.registry import fingerprint,file_sha256,write_new_json
from .role_program import RoleTask, PoolLayout, AllocatedWorld, enumerate_layouts, transport_bytes_by_rank
from .role_transport import RoleFunctions, RolePoolExecutor, RoleNativeCapabilityRejected

SCHEMA='distillschedule.measured_role_pool_calibration.v2'
PLAN='distillschedule.measured_role_pool_plan.v2'


def code_identity(bindings:RoleFunctions):
    """Source plus actual immutable configuration; never hash captured tensors.

    Mutable runtime owners must not masquerade as constants. Opaque captures
    are rejected at plan admission, not pickled or reduced to repr/object IDs.
    """
    seen={}
    file_hashes={}
    dependency_roots=tuple(Path(sysconfig.get_path(key)).resolve() for key in ('stdlib','purelib','platlib'))
    dynamic_lookups=(getattr,setattr,delattr,vars,globals,locals,eval,exec,__import__)
    def source_target(value):
        # Inspect the implementation that actually runs, not an arbitrary
        # functools.wraps target (a user decorator can change the mathematics).
        return value.__code__ if inspect.isfunction(value) else value
    def installed_dependency(value):
        try:path=inspect.getsourcefile(source_target(value))
        except (OSError,TypeError):return False
        return path is not None and any(Path(path).resolve().is_relative_to(root) for root in dependency_roots)
    def source_record(value):
        target=source_target(value)
        path=inspect.getsourcefile(target)
        if path is not None:
            path=Path(path)
            if path not in file_hashes:file_hashes[path]=file_sha256(path)
        try:source=inspect.getsource(target)
        except (OSError,TypeError) as exc:raise ValueError('uninspectable role callable configuration') from exc
        return dict(callable=hashlib.sha256(source.encode()).hexdigest(),
                    module=file_hashes.get(path)),source
    def bind(value, *, dependency=False, depth=0):
        require(depth<=24,'cyclic or excessively nested role configuration',path='role.identity')
        if value is None or type(value) in (bool,int,float,str):
            # fingerprint rejects nonfinite floats without reading tensor data.
            fingerprint(value)
            return value
        if type(value) in (list,tuple):
            return {type(value).__name__:[bind(item,depth=depth+1) for item in value]}
        if type(value) is dict:
            require(all(type(key) is str for key in value),'role configuration keys must be strings',path='role.identity')
            return {'dict':{key:bind(value[key],depth=depth+1) for key in sorted(value)}}
        if isinstance(value,(torch.dtype,torch.device)):
            return {type(value).__name__:str(value)}
        if inspect.ismodule(value):
            path=getattr(value,'__file__',None)
            if path and Path(path).is_file():
                path=Path(path)
                if path not in file_hashes:file_hashes[path]=file_sha256(path)
            return dict(imported_module=value.__name__,version=str(getattr(value,'__version__','')),
                        source=file_hashes.get(path))
        if inspect.isbuiltin(value) or (inspect.isclass(value) and value.__module__=='builtins'):
            require(not any(value is function for function in dynamic_lookups),
                    'dynamic role configuration lookup must be explicit',path='role.identity')
            return dict(imported_symbol=value.__module__+'.'+value.__qualname__)
        require(inspect.isfunction(value) or inspect.isclass(value),
                'dynamic role capture requires explicit configuration and state ownership: '+type(value).__qualname__,
                path='role.identity')
        if dependency and inspect.isfunction(value):
            # Torch's first optimizer import lazily installs this transparent
            # compiler-disable wrapper. Normalize only that stock dependency
            # wrapper; never unwrap arbitrary user decorators or their config.
            code=value.__code__
            disable=getattr(torch,'_disable_dynamo',None)
            if (Path(code.co_filename).resolve()==Path(torch.__file__).parent/'_compile.py'
                    and inspect.isfunction(disable)
                    and any(code is child for child in disable.__code__.co_consts if inspect.iscode(child))
                    and value.__globals__.get('__name__')=='torch._compile'):
                original=inspect.getclosurevars(value).nonlocals.get('fn')
                if (inspect.isfunction(original) and installed_dependency(original)
                        and getattr(value,'__wrapped__',None) is original):
                    value=original
        key=(id(value),dependency)
        if key in seen:return {'code_reference':seen[key]}
        seen[key]=len(seen)
        record,source=source_record(value)
        # Imported dependency implementation is bound by source; its mutable
        # internal caches/RNG are not the caller's constant configuration.
        if dependency:return record
        if inspect.isclass(value):
            record['constants']={key:bind(item,depth=depth+1) for key,item in sorted(vars(value).items())
                if not key.startswith('__') and not (inspect.isroutine(item) or isinstance(item,(property,staticmethod,classmethod)))}
            record['methods']={key:bind(item.__func__ if isinstance(item,(staticmethod,classmethod)) else item,
                                       depth=depth+1) for key,item in sorted(vars(value).items())
                if inspect.isfunction(item) or isinstance(item,(staticmethod,classmethod))}
            record['properties']={key:{accessor:bind(getattr(item,accessor),depth=depth+1)
                                       for accessor in ('fget','fset','fdel')}
                for key,item in sorted(vars(value).items()) if isinstance(item,property)}
            record['bases']=[bind(base,dependency=installed_dependency(base),depth=depth+1)
                             for base in value.__bases__]
            return record
        captures=inspect.getclosurevars(value)
        # Comprehensions/generators have separate code objects: outer co_names
        # alone omits globals used by their actual computation.
        names=set();pending=[value.__code__]
        while pending:
            code=pending.pop()
            names.update(item.argval for item in dis.get_instructions(code)
                         if item.opname in ('LOAD_GLOBAL','LOAD_NAME'))
            pending.extend(item for item in code.co_consts if inspect.iscode(item))
        referenced={key:value.__globals__[key] for key in names if key in value.__globals__}
        require(not any(getattr(builtins,key,None) in dynamic_lookups for key in names if key not in referenced),
                'dynamic role configuration lookup must be explicit',path='role.identity')
        record.update(defaults=bind(value.__defaults__,depth=depth+1),
                      kwdefaults=bind(value.__kwdefaults__,depth=depth+1),
                      closure={key:bind(item,depth=depth+1) for key,item in sorted(captures.nonlocals.items())})
        record['globals']={}
        for key,item in sorted(referenced.items()):
            imported=(inspect.isfunction(item) or inspect.isclass(item)) and installed_dependency(item)
            record['globals'][key]=bind(item,dependency=imported,depth=depth+1)
        # Bind actual scalar module attributes too (e.g. settings.loss_scale),
        # not just unchanged settings.py bytes. Do not invoke dynamic getters.
        attributes={}
        tree=ast.parse(textwrap.dedent(source))
        parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load):
                namespace=referenced.get(node.id,captures.nonlocals.get(node.id))
                if inspect.ismodule(namespace):
                    require(isinstance(parents.get(node),ast.Attribute) and parents[node].value is node,
                            'dynamic module configuration must use explicit attributes',path='role.identity')
            chain=[]
            while isinstance(node,ast.Attribute):
                chain.insert(0,node.attr);node=node.value
            if not chain or not isinstance(node,ast.Name):continue
            item=referenced.get(node.id,captures.nonlocals.get(node.id))
            if not inspect.ismodule(item):continue
            for part in chain:
                try:item=inspect.getattr_static(item,part)
                except AttributeError as exc:raise ValueError('dynamic module configuration must be explicit: '+node.id+'.'+'.'.join(chain)) from exc
            imported=(inspect.isfunction(item) or inspect.isclass(item)) and installed_dependency(item)
            attributes[node.id+'.'+'.'.join(chain)]=bind(item,dependency=imported,depth=depth+1)
        record['attributes']=attributes
        return record
    files={}
    for name in ('make_model','make_optimizer','sample','generator_forward','score_loss','score_train_loss',
                 'score_input_gradient','sample_at','score_update_input','state_owners','sample_identity'):
        fn=getattr(bindings,name)
        if fn is None:continue
        require(inspect.isfunction(fn),'role factory functions must be inspectable Python functions',path='role.identity')
        files[name]=bind(fn)
    return fingerprint(dict(schema_version='distillschedule.role_callable_configuration.v2',
                            factory=bindings.factory_identity,functions=files))


def source_identity():
    root=Path(__file__).resolve().parent
    files=tuple(root/name for name in
        ('role_program.py','role_transport.py','role_planning.py','role_offload.py','role_sharding.py','role_cli.py',
         'role_score.py','role_profiling.py','role_prediction.py','role_dispatch.py','role_checkpoint.py',
         'torch_backend.py','planning.py','hardware_probe.py','replicated.py'))
    files+=tuple(root.parent/name for name in ('runtime/saved_tensors.py','training/checkpoint.py',
        'training/tensor_state.py','training/distributed.py','training/finite.py','workloads/program.py'))
    return fingerprint({str(p.relative_to(root.parent)):file_sha256(p) for p in files})


def _run_identity(task,hardware,bindings,device):
    from .replicated import checked
    from .planning import hardware_identity, source_revision
    def local():
        bindings.validate(task)
        require(bool(getattr(bindings,'data_identity','')),
                'explicit immutable data/cohort identity required for role calibration',path='role.identity')
        props={**hardware_identity(torch.device(device)),
               'backend':str(dist.get_backend()),'world_size':dist.get_world_size(),
               'node_ids':list(hardware.node_ids),'cuda_version':torch.version.cuda}
        value={'source_head':source_revision(),'source_fingerprint':source_identity(),
               'task_sha256':task.digest,'semantic_graph_sha256':task.program().digest,
               'source_functions_sha256':code_identity(bindings),
               'factory':bindings.factory_identity,
               'score_parameters_frozen':bindings.score_parameters_frozen,
               'checkpoint_safe':bindings.checkpoint_safe,'data_identity':bindings.data_identity}
        return props,value
    props,value=checked(local,'role source/data/device identity')
    entries=[None]*hardware.world
    dist.all_gather_object(entries,dict(hardware=props,identity=value))
    require(all(row['identity']==value for row in entries),
            'ranks disagree on role source/configuration identity',path='role.identity')
    value['hardware']=[row['hardware'] for row in entries]
    return value


def _binding(identity,task):
    return {'source_head':identity['source_head'],
        'hardware_fingerprint':fingerprint(identity['hardware']),
        'environment_fingerprint':fingerprint({'code':identity['source_fingerprint'],
                                             'torch':[r['torch'] for r in identity['hardware']]}),
        'asset_fingerprint':fingerprint({'model_code':identity['source_functions_sha256'],
                                        'data':identity['data_identity']}),
        'semantic_fingerprint':task.digest,
        'precision_fingerprint':fingerprint({'dtype':task.dtype,'devices':identity['hardware']})}


def _plan_resources(selected, hardware, host_bytes=None):
    rows=selected['memory']
    require(len(rows)==hardware.world and {r['rank'] for r in rows}==set(range(hardware.world)),
            'selected memory must cover every allocated rank',path='role.capacity')
    device=[None]*hardware.world
    for row in rows:
        device[row['rank']]=row['peak_bytes']
    value=dict(schema_version='distillschedule.role_plan_resources.v1', device_bytes=device,
               host_bytes=list(host_bytes) if host_bytes is not None else [0]*hardware.world)
    _admit_plan_resources(value,hardware)
    return value


def _admit_plan_resources(value, hardware):
    require(type(value) is dict and set(value)=={'schema_version','device_bytes','host_bytes'}
            and value['schema_version']=='distillschedule.role_plan_resources.v1',
            'missing or invalid role resource requirements',path='role.capacity')
    for name,available in (('device_bytes',hardware.capacities),('host_bytes',hardware.host_capacities)):
        needs=value[name]
        require(type(needs) is list and len(needs)==hardware.world
                and all(type(n) is int and n>=0 for n in needs),
                'invalid per-rank '+name+' requirements',path='role.capacity')
        require(all(current>=needed for current,needed in zip(available,needs)),
                'available '+name+' capacity below selected plan requirements',path='role.capacity')


def _measured_host_requirements(task, layout, costs, world):
    if layout.activation_policy=='resident':return [0]*world
    cost=costs.get_component(_contract(task,layout,'measured_native_cycle'))
    raw=_load(cost.evidence['source_receipt'])
    observations=raw['rank_observations']
    require(observations,'offload plan requires actual host observations',path='role.capacity')
    peaks=[0]*world
    for observation in observations:
        rows=observation['ranks']
        require(observation['layout']==layout.to_dict() and len(rows)==world
                and [row['rank'] for row in rows]==list(range(world)),
                'offload host observations changed ownership',path='role.capacity')
        for rank,row in enumerate(rows):
            peak=row['offload']['peak_host_bytes']
            require(type(peak) is int and peak>=0,'invalid observed host peak',path='role.capacity')
            peaks[rank]=max(peaks[rank],peak)
    for rank in layout.generator:
        peaks[rank]=max(peaks[rank],2*layout.window*task.bytes_per_latent)
    return peaks


def cycle_graph(task):
    return ExecutionGraph((
        Operation('prepare_g',0,'generator',OperationKind.CALL,GradientMode.NO_GRAD),
        Operation('prepare_s',1,'score',OperationKind.CALL,GradientMode.NO_GRAD),
        Operation('measured_native_cycle',2,'pool_cycle',OperationKind.CALL,GradientMode.NO_GRAD,
                  metadata={'semantic_program_sha256':task.program().digest}),
        Operation('finish',3,'pool_cycle',OperationKind.BARRIER,GradientMode.NO_GRAD)),
        (Dependency('prepare_g','measured_native_cycle'),
         Dependency('prepare_s','measured_native_cycle'),
         Dependency('measured_native_cycle','finish')), 
        {'initial_versions':{},'final_versions':{},
         'semantic_program_sha256':task.program().digest,
         'lowering':'bounded_fifo_role_vjp_pipeline_v1'})


def _contract(task,layout,phase):
    return ComponentContract('open.role_pool_cycle',
        {'task_sha256':task.digest,'semantic_program_sha256':task.program().digest,'phase':phase},
        {'dtype':task.dtype},
        {'generator':list(layout.generator),'score':list(layout.score)},
        {'layout':layout.to_dict()},
        {'protocol':'explicit_score_input_vjp','order':'bounded_pairwise_fifo','optimizer_commit':'all_vjp_first'})


def calibrate(task:RoleTask,hardware:AllocatedWorld,bindings:RoleFunctions,output,*,
              device='cpu',generator_bytes=0,score_bytes=0,safety_bytes=0,
              max_candidates=4,samples=1,max_native_runs=10,allow_offload=False,
              allow_generator_fsdp=False,cost_mode='measured_cycle'):
    """Each candidate executes actual torch autograd/P2P/optimizer, not a mock."""
    require(cost_mode in ('measured_cycle','components'),'unknown role cost mode',path='role.calibrate')
    if cost_mode=='components':
        require(not allow_generator_fsdp,'FSDP needs measured_cycle cost mode in this release',path='role.calibrate')
        from .role_prediction import calibrate as component_calibrate
        return component_calibrate(task,hardware,bindings,output,device=device,
            generator_bytes=generator_bytes,score_bytes=score_bytes,safety_bytes=safety_bytes,
            max_candidates=max_candidates,samples=samples,max_native_runs=max_native_runs)
    require(dist.is_initialized() and dist.get_world_size()==hardware.world,
            'process group must be allocated before calibrating pools',path='role.calibrate')
    require(type(samples) is int and 1<=samples<=8 and type(max_native_runs) is int,
            'invalid native calibration budget',path='role.calibrate')
    inventory=enumerate_layouts(task,hardware,generator_bytes=generator_bytes,
        score_bytes=score_bytes,safety_bytes=safety_bytes,max_candidates=max_candidates,
        allow_offload=allow_offload,allow_generator_fsdp=allow_generator_fsdp,
        allow_frozen_score_fsdp=bindings.score_parameters_frozen,
        allow_trainable_score_fsdp=(task.score_updates>0 and bindings.score_input_gradient is not None))
    require(samples*len(inventory['candidates'])+1<=max_native_runs,
            'all measured variants plus confirmation must fit the execution budget',path='role.calibrate')
    identity=_run_identity(task,hardware,bindings,device)
    root=Path(output).resolve()
    if dist.get_rank()==0:root.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    models=[];native_rejected=[];attempts=0
    for idx,layout in enumerate(inventory['candidates']):
        measured=[]
        rejected_reason=None
        for _ in range(samples):
            attempts+=1
            executor=RolePoolExecutor(task,layout,hardware,bindings,device=device)
            try:
                result,model,optimizer=executor.run()
            except RoleNativeCapabilityRejected as exc:
                # Only a WORLD-synchronized native capability rejection is safe
                # to skip. Never swallow native loss, CUDA, transport or OOM errors.
                rejected_reason=str(exc)
            finally:executor.close()
            if rejected_reason is not None:break
            measured.append(result)
            # Do not keep a previous candidate's optimizer/weights alive while
            # constructing the next candidate (notably on 40GB devices).
            del model,optimizer
        if rejected_reason is not None:
            native_rejected.append({'layout':layout.to_dict(),
                                    'reason':'NATIVE_CAPABILITY_REJECTED',
                                    'detail':rejected_reason})
            continue
        # Exact-rerun state invariance within the SAME layout; cross-layout
        # numerical equivalence and generated video quality remain separate gates.
        if layout.activation_policy=='cpu_offload_prefetch':
            require(all(r['offload']['packed_tensors']>0 and
                        r['offload']['prefetch_tensors']>0 and
                        r['offload']['unpacked_tensors']>0 and r['host_leases_closed']
                        for x in measured for r in x['ranks'] if r['role']=='generator'),
                    'no actual saved-activation offload/prefetch effect',path='role.calibrate')
        roles=[[(r['role'],r['model_sha256']) for r in x['ranks']]
               for x in measured]
        require(all(x==roles[0] for x in roles),'nondeterministic calibration outcomes',path='role.calibrate')
        binding=_binding(identity,task)
        directory=root/f'candidate-{idx}'
        if dist.get_rank()==0:directory.mkdir()
        dist.barrier()
        rows=[]
        for phase in ('prepare_g','prepare_s','measured_native_cycle','finish'):
            contract=_contract(task,layout,phase)
            samples_ns=[x['wall_ns'] if phase=='measured_native_cycle' else 0 for x in measured]
            raw={'schema_version':'distillschedule.operation_calibration.v1',
                 'binding':binding,'key':contract.digest,'samples_ns':samples_ns,
                 'workspace_bytes_per_rank':0,'status':'PASS','profiled':False,
                 'kind':'native_operation','operation_contract_sha256':contract.digest,
                 'component_contract':contract.to_dict(),'rank_observations':measured,
                 'device':str(device),'aggregate':'max_rank_full_cycle_wall_ns',
                 'capacity_scope':'known_transport_bound_and_observed_cuda_peak',
                 'claimable_speedup':False,'quality_validation':'NOT_RUN'}
            path=directory/f'{phase}.json'
            if dist.get_rank()==0:write_new_json(path,raw)
            dist.barrier()
            rows.append(OperationCost(contract.digest,tuple(samples_ns),0,
                {**{k:raw[k] for k in ('status','profiled','kind','operation_contract_sha256','component_contract')},
                 'source_receipt':str(path),'source_receipt_sha256':file_sha256(path)}))
        costs=CostModel(binding,tuple(rows));costs.verify_receipts()
        cost_path=directory/'costs.json'
        if dist.get_rank()==0:write_new_json(cost_path,costs.to_dict())
        dist.barrier()
        static={role:max(max(r['initial_storage_bytes'],r['final_storage_bytes'])
            for x in measured for r in x['ranks'] if r['role']==role) for role in ('generator','score')}
        models.append({'layout':layout.to_dict(),'cost_path':str(cost_path),
                       'cost_sha256':file_sha256(cost_path),
                       'memory_floor_bytes':{str(r):max(
                           x['ranks'][r]['cuda_peak_allocated_bytes'] or 0 for x in measured)
                           for r in range(hardware.world)},
                       'static_bytes':static,'samples_ns':[x['wall_ns'] for x in measured],
                       'outcome_hash':fingerprint(roles[0]),
                       'numerical_reference':'REQUIRES_INDEPENDENT_BASELINE'})
    require(models,'no native role-pool candidate passed capability admission',path='role.calibrate')
    manifest={'schema_version':SCHEMA,'task':task.to_dict(),'identity':identity,
              'hardware':{'capacities':list(hardware.capacities),'node_ids':list(hardware.node_ids),
                          'host_capacities':list(hardware.host_capacities)},
              'candidates':models,'candidate_inventory':{
                  'rejected':list(inventory['rejected'])+native_rejected,
                  'enumeration_complete':inventory['enumeration_complete'],
                  'family':inventory['family']},
              'budget':{'maximum':max_native_runs,'calibration_runs':attempts,
                        'remaining_for_confirmation':max_native_runs-attempts},
              'evidence_scope':'MEASURED_GLOO' if torch.device(device).type=='cpu' else 'MEASURED_NCCL_NOT_INDEPENDENTLY_VALIDATED',
              'gpu_execution':'NOT_RUN' if torch.device(device).type=='cpu' else 'NATIVE_CUDA_CALIBRATION',
              'claimable_speedup':False}
    if dist.get_rank()==0:write_new_json(root/'calibration.json',manifest)
    dist.barrier()
    return manifest


def _load(path):
    return json.loads(Path(path).read_text())


def search(calibration_path,*,max_states=50000,beam_width=0,timeout_seconds=60):
    manifest=_load(calibration_path)
    from .role_prediction import SCHEMA as COMPONENT_SCHEMA
    if manifest.get('schema_version')==COMPONENT_SCHEMA:
        from .role_prediction import search as component_search
        return component_search(calibration_path,max_states=max_states,
                                beam_width=beam_width,timeout_seconds=timeout_seconds)
    require(manifest.get('schema_version')==SCHEMA and manifest['identity']['source_fingerprint']==source_identity(),
            'changed role calibration/source',path='role.plan')
    task=RoleTask.from_dict(manifest['task'])
    hw=AllocatedWorld(tuple(manifest['hardware']['capacities']),
                      tuple(manifest['hardware']['node_ids']),
                      tuple(manifest['hardware']['host_capacities']))
    candidates=[];host_requirements={}
    for row in manifest['candidates']:
        layout=PoolLayout.from_dict(row['layout']).validate(task,hw)
        require(file_sha256(Path(row['cost_path']))==row['cost_sha256'],
                'role measured costs changed',path='role.plan')
        costs=CostModel.from_dict(_load(row['cost_path']));costs.verify_receipts()
        require(costs.binding==_binding(manifest['identity'],task),'mixed calibration identity',path='role.plan')
        host_requirements[fingerprint(layout.to_dict())]=_measured_host_requirements(task,layout,costs,hw.world)
        graph=cycle_graph(task);ranks=tuple(range(hw.world))
        transport=transport_bytes_by_rank(task,layout)
        # Rank-specific queues matter for asymmetric pools. Never give every
        # Score rank a constant three-latent allowance independent of its work.
        states=[]
        for role,pool in (('generator',layout.generator),('score',layout.score)):
            amounts=[]
            for rank in pool:
                known=row['static_bytes'][role]+transport[rank]
                observed=int(row['memory_floor_bytes'][str(rank)])
                require(observed>=0,'invalid observed GPU peak',path='role.plan')
                amounts.append(max(known,observed))
            # PlanIR uses uniform bytes per rank inside one StateAllocation.
            # Keep the pool maximum, never an average that hides rank skew.
            states.append(StateAllocation('memory_'+role,pool,max(amounts),
                (role,),False,manifest['identity']['source_functions_sha256']))
        states=tuple(states)
        rules={}
        for key in ('prepare_g','prepare_s','measured_native_cycle','finish'):
            contract=_contract(task,layout,key)
            costs.get_component(contract)
            occupied=layout.generator if key=='prepare_g' else layout.score if key=='prepare_s' else ranks
            active=('memory_generator',) if key=='prepare_g' else (
                ('memory_score',) if key=='prepare_s' else ())
            rules[key]=BindingRule(key,{},contract.digest,occupied,
                tuple('gpu:'+str(r) for r in occupied),active,contract.digest)
        candidates.append(BoundCandidate(fingerprint(layout.to_dict()),layout.to_dict(),graph,rules,costs,states,
            (),{r:hw.capacities[r] for r in ranks},'real_pairwise_P2P_role_pool_FIFO'))
    prediction=select(candidates,max_states=max_states,beam_width=beam_width,
                      timeout_seconds=timeout_seconds)
    require(prediction['ranked'],'no feasible measured role-pool plan',path='role.plan')
    selected=prediction['ranked'][0]
    result={'schema_version':PLAN,'calibration':str(Path(calibration_path).resolve()),
            'calibration_sha256':file_sha256(Path(calibration_path)),
            'identity':manifest['identity'],'task':task.to_dict(),
            'selected':selected,'search':prediction,
            'resource_requirements':_plan_resources(selected,hw,host_requirements[selected['candidate_id']]),
            'execution_model':'FULL_CYCLE_MEASURED_FIFO_WINDOW_NOT_UNLOWERED_DP_ORDER',
            'numerical_reference':'REQUIRES_INDEPENDENT_BASELINE',
            'claimable_speedup':False,'gpu_execution':'NOT_RUN_BY_SEARCH'}
    result['plan_sha256']=fingerprint(result)
    return result


def run_selected(task,hardware,bindings,plan_path,output,*,device='cpu',cycles=1,state_audit_every_cycles=50,
                 checkpoint_policy=None,resume_from=None):
    from distillschedule.training import distributed
    from distillschedule.training.checkpoint import CheckpointPolicy, prune_checkpoints
    from .state_audit import validate_every
    require(dist.is_initialized() and dist.get_world_size()==hardware.world,
            'selected role pool needs allocated process group',path='role.run')
    def local_controls():
        bindings.validate(task)
        require(type(cycles) is int and 1<=cycles<=1024 and
                (cycles==1 or bindings.sample_at is not None),
                'explicit sample_at is required for bounded continuous execution',path='role.run')
        validate_every(state_audit_every_cycles)
        policy=(CheckpointPolicy.from_trainer(checkpoint_policy or {})
                if checkpoint_policy is not None or resume_from is not None else None)
        resume_sha=None
        if resume_from is not None:
            resume_sha=file_sha256(Path(resume_from)/'manifest.json')
            require((Path(resume_from)/'_SUCCESS').read_text().strip()==resume_sha,
                    'resume checkpoint is incomplete or changed',path='role.run')
        return dict(cycles=cycles,policy=asdict(policy) if policy else None,
                    resume_manifest=resume_sha,output=str(Path(output).resolve()),
                    state_audit_every_cycles=state_audit_every_cycles)
    controls=distributed.checked(local_controls,'role run/restart controls')
    require(all(row==controls for row in distributed.gather(controls)),
            'ranks disagree on role run/restart controls',path='role.run')
    def load_plan():
        p=_load(plan_path);signed=dict(p);sha=signed.pop('plan_sha256',None)
        require(p.get('schema_version')==PLAN and sha==fingerprint(signed),
                'changed role plan',path='role.run')
        require(p['task']==task.to_dict() and file_sha256(Path(p['calibration']))==p['calibration_sha256'],
                'role task/calibration changed',path='role.run')
        calibration=_load(p['calibration'])
        require(calibration['identity']==p['identity'],'mixed role identities',path='role.run')
        expected=search(p['calibration'],**p.get('search_settings',{}))
        required=('candidate_id','implementation','predicted_makespan_ns','graph_sha256',
                  'cost_model_sha256','order','timeline','memory')
        require(all(p['selected'].get(key)==expected['selected'].get(key) for key in required),
                'selected role pool differs from source-bound measured search',path='role.run')
        require(p.get('producer_dispatch')==expected.get('producer_dispatch'),
                'selected producer dispatch differs from source-bound search',path='role.run')
        require(p.get('resource_requirements')==expected['resource_requirements'],
                'selected role resource requirements changed',path='role.run')
        return p,sha,PoolLayout.from_dict(p['selected']['implementation']).validate(task,hardware)
    p,sha,layout=distributed.checked(load_plan,'role selected plan admission')
    require(len(set(distributed.gather(sha)))==1,'ranks loaded different role plans',path='role.run')
    identity=_run_identity(task,hardware,bindings,device)
    distributed.checked(lambda:require(p['identity']==identity,
                'role source/hardware/data identity changed',path='role.run'),'role runtime identity')
    distributed.checked(lambda:_admit_plan_resources(p['resource_requirements'],hardware),
                        'role available resource admission')
    policy=CheckpointPolicy(**controls['policy']) if controls['policy'] is not None else None
    root=Path(controls['output'])
    distributed.primary(lambda:root.mkdir(parents=True,exist_ok=False),'role output directory')
    executor=RolePoolExecutor(task,layout,hardware,bindings,device=device,
        producer_dispatch=p.get('producer_dispatch'),state_audit_every_cycles=state_audit_every_cycles)
    summaries=[]
    checkpoint=None;latest_checkpoint=None;start=0
    try:
        if policy is not None:
            from .role_checkpoint import RoleCheckpoint
            checkpoint=RoleCheckpoint(executor,p)
        if resume_from is not None:
            start=checkpoint.restore(resume_from)['cycle']
            require(start<cycles,'cycles must exceed the restored completed cycle',path='role.run')
        last_saved=time.monotonic()
        for cycle in range(start,cycles):
            receipt,module,optimizer=executor.run()
            if p.get('cost_mode')=='components':
                from .role_prediction import validate_execution,confirm_state
                gate=validate_execution(task,layout,receipt,dispatch=p.get('producer_dispatch'))
                predicted=p['selected']['predicted_makespan_ns']
                observed=receipt['compute_transport_wall_ns']
                gate.update(predicted_ns=predicted,observed_ns=observed,
                            relative_error=(abs(observed-predicted)/max(1,observed)),
                            measured_before_selection=p['prediction_provenance'][
                                p['selected']['candidate_id']]['was_full_cycle_measured'])
                gate['model_check']='PASS_25_PERCENT' if gate['relative_error']<=.25 else 'REVIEW_MODEL_ERROR'
                gate['state_check']=confirm_state(p,receipt,module,optimizer,cycle)
                receipt['component_prediction_check']=gate
                if gate['state_check']['status']=='REVIEW_ANCHOR_STATE_MISMATCH':
                    if dist.get_rank()==0:write_new_json(root/'review.json',receipt)
                    raise ValueError('component plan state differs from measured anchor; REVIEW required')
            receipt.update({'selected_plan_sha256':sha,'measured_calibration':p['calibration'],
                            'numerical_reference':'REQUIRES_INDEPENDENT_BASELINE',
                            'claimable_speedup':False})
            distributed.primary(lambda:write_new_json(root/f'cycle-{cycle:06d}.json',receipt),
                                'role cycle receipt')
            summaries.append({'cycle':cycle,'wall_ns':receipt['wall_ns'],
                              'initial_versions':receipt['initial_versions'],
                              'final_versions':receipt['final_versions']})
            if checkpoint is not None:
                due=policy.due(cycle=executor._cycle,total=cycles,elapsed=time.monotonic()-last_saved)
                if distributed.gather(due if dist.get_rank()==0 else None)[0]:
                    latest_checkpoint=checkpoint.save(root/'checkpoints'/f'cycle-{executor._cycle:06d}',
                                                       policy=asdict(policy))
                    distributed.primary(lambda:prune_checkpoints(root/'checkpoints',
                        identity=checkpoint.identity,keep_last=policy.keep_last),'role checkpoint retention')
                    last_saved=time.monotonic()
    finally:executor.close()
    receipt.update({'cycles_executed':len(summaries),'completed_cycles':cycles,'resumed_from_cycle':start,
                    'cycle_summaries':summaries,'checkpoint_path':str(latest_checkpoint) if latest_checkpoint else None,
                    'training_state':('RESUMABLE_COMPLETE_CYCLE' if latest_checkpoint else
                                      'PERSISTED_IN_PROCESS_NOT_CHECKPOINT_RESTART')})
    distributed.primary(lambda:write_new_json(root/'execution.json',receipt),'role execution receipt')
    return receipt,module,optimizer
