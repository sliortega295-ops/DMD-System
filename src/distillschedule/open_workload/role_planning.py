"""Measured, capability-bounded G/Score pool search over the shared frontier solver.

Only role splits and FIFO in-flight windows with an actual native lowering are
admitted. The internal G/S overlap is measured as a WHOLE cycle; a single
resource envelope is passed to the common DP to avoid predicting an arbitrary
interleaving which the native runtime cannot execute. This is a deliberately
conservative but honest first compositional pool backend.
"""
from __future__ import annotations
import hashlib
import inspect
import json
import platform
from pathlib import Path
import statistics
import subprocess

import torch
import torch.distributed as dist
from distillschedule.core.contracts import require
from distillschedule.core.enums import OperationKind, GradientMode
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation
from distillschedule.search.cost_model import CostModel, OperationCost, ComponentContract
from distillschedule.search.plan_ir import BindingRule, StateAllocation
from distillschedule.search.service import BoundCandidate, select
from distillschedule.workloads.best_known.registry import fingerprint,file_sha256,repository_root,write_new_json
from .role_program import RoleTask, PoolLayout, AllocatedWorld, enumerate_layouts, transport_bytes_by_rank
from .role_transport import RoleFunctions, RolePoolExecutor

SCHEMA='distillschedule.measured_role_pool_calibration.v1'
PLAN='distillschedule.measured_role_pool_plan.v1'


def code_identity(bindings:RoleFunctions):
    """Callable source fingerprints, not user supplied winner/recipe aliases."""
    files={}
    for name in ('make_model','make_optimizer','sample','generator_forward','score_loss','score_train_loss',
                 'score_input_gradient','sample_at'):
        fn=getattr(bindings,name)
        if fn is None:continue
        require(inspect.isfunction(fn),'role factory functions must be inspectable Python functions',path='role.identity')
        try:source=inspect.getsource(fn)
        except (OSError,TypeError) as exc:raise ValueError(f'uninspectable role callable: {name}') from exc
        files[name]=hashlib.sha256(source.encode()).hexdigest()
    return fingerprint(files)


def source_identity():
    root=repository_root()
    files=tuple(root/'src/distillschedule/open_workload'/name for name in
        ('role_program.py','role_transport.py','role_planning.py','role_offload.py','role_sharding.py'))
    return fingerprint({p.name:file_sha256(p) for p in files})


def _run_identity(task,hardware,bindings,device):
    bindings.validate(task)
    require(bool(getattr(bindings,'data_identity','')),
            'explicit immutable data/cohort identity required for role calibration',path='role.identity')
    props={'device':str(torch.device(device)),'torch':torch.__version__,
           'backend':str(dist.get_backend()),'world_size':dist.get_world_size(),
           'capacity_bytes':list(hardware.capacities),'node_ids':list(hardware.node_ids),
           'host_capacity_bytes':list(hardware.host_capacities),
           'platform':platform.platform(),'cuda_version':torch.version.cuda}
    if torch.device(device).type=='cuda':
        p=torch.cuda.get_device_properties(device)
        props.update({'gpu_name':p.name,'gpu_capability':[p.major,p.minor],
                      'gpu_total_memory':p.total_memory,'gpu_uuid':str(getattr(p,'uuid','UNAVAILABLE'))})
    identities=[None]*hardware.world
    dist.all_gather_object(identities,props)
    # On multi-node A100/H100 ranks may differ in UUID but must agree on source.
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=repository_root(),text=True,capture_output=True)
    revision=head.stdout.strip() if head.returncode==0 else '0'*40
    value={'source_head':revision,'source_fingerprint':source_identity(),
           'task_sha256':task.digest,'semantic_graph_sha256':task.program().digest,
           'source_functions_sha256':code_identity(bindings),
           'score_parameters_frozen':bindings.score_parameters_frozen,
           'data_identity':bindings.data_identity,'hardware':identities}
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
              allow_generator_fsdp=False):
    """Each candidate executes actual torch autograd/P2P/optimizer, not a mock."""
    require(dist.is_initialized() and dist.get_world_size()==hardware.world,
            'process group must be allocated before calibrating pools',path='role.calibrate')
    require(type(samples) is int and 1<=samples<=8 and type(max_native_runs) is int,
            'invalid native calibration budget',path='role.calibrate')
    inventory=enumerate_layouts(task,hardware,generator_bytes=generator_bytes,
        score_bytes=score_bytes,safety_bytes=safety_bytes,max_candidates=max_candidates,
        allow_offload=allow_offload,allow_generator_fsdp=allow_generator_fsdp,
        allow_frozen_score_fsdp=bindings.score_parameters_frozen)
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
            except ValueError as exc:
                # Only a WORLD-synchronized native capability rejection is safe
                # to skip. Never swallow native loss, CUDA, transport or OOM errors.
                if 'role native layout admission' not in str(exc):raise
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
    require(manifest.get('schema_version')==SCHEMA and manifest['identity']['source_fingerprint']==source_identity(),
            'changed role calibration/source',path='role.plan')
    task=RoleTask.from_dict(manifest['task'])
    hw=AllocatedWorld(tuple(manifest['hardware']['capacities']),
                      tuple(manifest['hardware']['node_ids']),
                      tuple(manifest['hardware']['host_capacities']))
    candidates=[]
    for row in manifest['candidates']:
        layout=PoolLayout.from_dict(row['layout']).validate(task,hw)
        require(file_sha256(Path(row['cost_path']))==row['cost_sha256'],
                'role measured costs changed',path='role.plan')
        costs=CostModel.from_dict(_load(row['cost_path']));costs.verify_receipts()
        require(costs.binding==_binding(manifest['identity'],task),'mixed calibration identity',path='role.plan')
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
    result={'schema_version':PLAN,'calibration':str(Path(calibration_path).resolve()),
            'calibration_sha256':file_sha256(Path(calibration_path)),
            'identity':manifest['identity'],'task':task.to_dict(),
            'selected':prediction['ranked'][0],'search':prediction,
            'execution_model':'FULL_CYCLE_MEASURED_FIFO_WINDOW_NOT_UNLOWERED_DP_ORDER',
            'numerical_reference':'REQUIRES_INDEPENDENT_BASELINE',
            'claimable_speedup':False,'gpu_execution':'NOT_RUN_BY_SEARCH'}
    result['plan_sha256']=fingerprint(result)
    return result


def run_selected(task,hardware,bindings,plan_path,output,*,device='cpu',cycles=1):
    require(dist.is_initialized() and dist.get_world_size()==hardware.world,
            'selected role pool needs allocated process group',path='role.run')
    require(type(cycles) is int and 1<=cycles<=1024 and
            (cycles==1 or bindings.sample_at is not None),
            'explicit sample_at is required for bounded continuous execution',path='role.run')
    p=_load(plan_path);signed=dict(p);sha=signed.pop('plan_sha256',None)
    require(p.get('schema_version')==PLAN and sha==fingerprint(signed),
            'changed role plan',path='role.run')
    require(p['task']==task.to_dict() and file_sha256(Path(p['calibration']))==p['calibration_sha256'],
            'role task/calibration changed',path='role.run')
    calibration=_load(p['calibration'])
    require(calibration['identity']==p['identity']==_run_identity(task,hardware,bindings,device),
            'role source/hardware/data identity changed',path='role.run')
    expected=search(p['calibration'])
    required=('candidate_id','implementation','predicted_makespan_ns','graph_sha256',
              'cost_model_sha256','order','timeline','memory')
    require(all(p['selected'].get(key)==expected['selected'].get(key) for key in required),
            'selected role pool differs from source-bound measured search',path='role.run')
    layout=PoolLayout.from_dict(p['selected']['implementation']).validate(task,hardware)
    root=Path(output).resolve()
    if dist.get_rank()==0:root.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    executor=RolePoolExecutor(task,layout,hardware,bindings,device=device)
    summaries=[]
    try:
        for cycle in range(cycles):
            receipt,module,optimizer=executor.run()
            receipt.update({'selected_plan_sha256':sha,'measured_calibration':p['calibration'],
                            'numerical_reference':'REQUIRES_INDEPENDENT_BASELINE',
                            'claimable_speedup':False})
            if dist.get_rank()==0:write_new_json(root/f'cycle-{cycle:06d}.json',receipt)
            summaries.append({'cycle':cycle,'wall_ns':receipt['wall_ns'],
                              'initial_versions':receipt['initial_versions'],
                              'final_versions':receipt['final_versions']})
    finally:executor.close()
    receipt.update({'cycles_executed':cycles,'cycle_summaries':summaries,
                    'training_state':'PERSISTED_IN_PROCESS_NOT_CHECKPOINT_RESTART'})
    if dist.get_rank()==0:write_new_json(root/'execution.json',receipt)
    dist.barrier()
    return receipt,module,optimizer
