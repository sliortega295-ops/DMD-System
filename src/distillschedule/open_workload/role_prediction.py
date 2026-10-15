"""Component-cost prediction of executable FIFO windows, not full-cycle ranking.

An anchor measures native calls at window=1 once per admitted physical pool.
The unchanged native FIFO backend lowers other windows. No component is reused
across different models, inputs, precision, hardware, pools or parameter layout.
The first admitted family is balanced replicated/resident pools. Isolated link
costs do not describe NIC contention; predictions MUST be confirmed separately.
"""
from dataclasses import replace
from pathlib import Path
import json

from distillschedule.core.contracts import require
from distillschedule.core.graph import Operation, Dependency, ExecutionGraph
from distillschedule.core.enums import OperationKind, GradientMode
from distillschedule.search.cost_model import CostModel, OperationCost, ComponentContract
from distillschedule.search.plan_ir import BindingRule, StateAllocation, Lifetime
from distillschedule.search.service import BoundCandidate, select
from distillschedule.workloads.best_known.registry import fingerprint, file_sha256, write_new_json
from .role_program import RoleTask, PoolLayout, AllocatedWorld, enumerate_layouts, transport_bytes_by_rank

SCHEMA = 'distillschedule.role_component_calibration.v1'
BACKEND = 'native_fifo_from_component_costs_v1'


def admission(task, layout, hardware):
    layout.validate(task, hardware)
    require(layout.activation_policy=='resident' and
            layout.generator_parallelism==layout.score_parallelism=='replicated' and
            len(layout.generator)==len(layout.score) and
            task.microbatches>=len(layout.generator) and task.microbatches<=48,
            'component prediction admits balanced replicated resident pools with 1..48 microbatches',
            path='role.predict')


def projection(layout):
    result=layout.to_dict()
    result.pop('window')
    return result


def generator_order(indices, window):
    """Exactly the producer loop's fill / flush-oldest / refill / drain order."""
    pending=[];order=[]
    for i in indices:
        if len(pending)>=window:
            order.append(f'b{pending.pop(0)}')
        order.append(f'g{i}');pending.append(i)
    order.extend(f'b{i}' for i in pending)
    return order


def contract(task, layout, key):
    return ComponentContract('open.role_fifo_component',
        {'task':task.digest,'operation':key}, {'dtype':task.dtype},
        {'generator':list(layout.generator),'score':list(layout.score)},
        {'resident_replicated':True},
        {'anchor_window':1,'transferability':'same_pool_same_inputs_different_fifo_window',
         'network':'isolated_directional_cost_no_contention_discount'})


def graph_for(task, layout, hardware):
    admission(task,layout,hardware)
    specs={};edges=set()
    def add(key, cost_key, ranks, resources):
        specs[key]=(cost_key,tuple(ranks),tuple(resources))
    for i in range(task.microbatches):
        g,s=layout.owners(i)
        # Transfer order on a pair is fixed, not chosen by the DP and ignored
        # by the backend. Serialize return before the next receive on that pair.
        link=(f'pair:{g}:{s}',f'dma:{g}',f'dma:{s}')
        add(f'g{i}',f'forward:{i}',(g,),(f'gpu:{g}',))
        add(f'a{i}',f'link:{g}:{s}',(g,s),link)
        add(f's{i}',f'score:{i}',(s,),(f'gpu:{s}',))
        add(f'r{i}',f'link:{s}:{g}',(s,g),link)
        add(f'b{i}',f'backward:{i}',(g,),(f'gpu:{g}',))
        chain=[f'{p}{i}' for p in 'gasrb']
        edges.update(zip(chain,chain[1:]))
    for g in layout.generator:
        indices=[i for i in range(task.microbatches) if layout.owners(i)[0]==g]
        chain=generator_order(indices,layout.window)
        edges.update(zip(chain,chain[1:]))
        for left,right in zip(indices,indices[1:]):
            edges.add((f'r{left}',f'a{right}'))
    add('commit_g','generator_commit',layout.generator,
        [f'gpu:{r}' for r in layout.generator])
    edges.update((f'b{i}','commit_g') for i in range(task.microbatches))
    if task.score_updates:
        add('updates_s','score_updates',layout.score,[f'gpu:{r}' for r in layout.score])
        edges.add(('commit_g','updates_s'))
    operations=tuple(Operation(k,n,'generator' if k.startswith(('g','b','commit')) else 'score',
                               OperationKind.TENSOR_TRANSFER if k.startswith(('a','r')) else OperationKind.CALL,
                               GradientMode.NO_GRAD,
                               metadata={'component_key':v[0]})
                     for n,(k,v) in enumerate(specs.items()))
    graph=ExecutionGraph(operations,tuple(Dependency(a,b) for a,b in sorted(edges)),
        {'semantic_program_sha256':task.program().digest,'lowering':BACKEND,
         'initial_versions':{},'final_versions':{},
         'timing_scope':'predicted_native_fifo_precedence_not_arbitrary_dispatch'})
    return graph,specs


def component_values(task, layout, anchor):
    """Derive costs only from the original per-rank/native-link observations."""
    _,specs=graph_for(task,layout,AllocatedWorld(tuple(anchor['capacities'])))
    values={}
    for key in sorted({spec[0] for spec in specs.values()}):
        if key.startswith('link:'):
            durations=anchor['links']['samples_ns'][key.removeprefix('link:')]
            workspace=0;kind='isolated_microbenchmark'
        else:
            action,sep,index=key.partition(':')
            rows=[]
            for run in anchor['runs']:
                matches=[row for rank in run['ranks'] for row in rank['components']['records']
                         if row['action']==action and row['microbatch']==(int(index) if sep else None)]
                require(matches,'missing actual component observations: '+key,path='role.predict')
                rows.append(matches)
            durations=[max(row['duration_ns'] for row in group) for group in rows]
            workspace=max(row['extra_allocated_bytes'] for group in rows for row in group)
            kind='native_operation'
        require(durations and all(type(t) is int and t>0 for t in durations),
                'invalid native component samples',path='role.predict')
        values[key]=(tuple(durations),workspace,kind)
    return values


def _save_costs(task, layout, anchor, raw_path, output, binding):
    """All compute samples come from anchor spans; no candidate cycle time."""
    for key,(values,workspace,kind) in component_values(task,layout,anchor).items():
        c=contract(task,layout,key)
        raw={'schema_version':'distillschedule.operation_calibration.v1',
             'binding':binding,'key':c.digest,'samples_ns':list(values),
             'workspace_bytes_per_rank':workspace,'status':'PASS','profiled':False,'kind':kind,
             'operation_contract_sha256':c.digest,'component_contract':c.to_dict(),
             'anchor_receipt':str(raw_path),'anchor_receipt_sha256':file_sha256(raw_path),
             'timing_method':'explicit_synchronized_component_span_or_isolated_P2P',
             'claimable_speedup':False}
        path=output/(c.digest+'.json');write_new_json(path,raw)
        evidence={k:raw[k] for k in ('status','profiled','kind','operation_contract_sha256','component_contract')}
        evidence.update(source_receipt=str(path),source_receipt_sha256=file_sha256(path))
        yield OperationCost(c.digest,values,workspace,evidence)


def calibrate(task,hardware,bindings,output,*,device='cpu',generator_bytes=0,score_bytes=0,
              safety_bytes=0,max_candidates=8,samples=2,max_native_runs=10,**unused):
    import torch
    import torch.distributed as dist
    from .role_transport import RolePoolExecutor
    from .role_profiling import profile_links
    from .role_planning import _run_identity,_binding
    from .torch_backend import signature
    require(type(samples) is int and 1<=samples<=8 and type(max_candidates) is int
            and 1<=max_candidates<=128 and type(max_native_runs) is int,
            'invalid component calibration budget',path='role.predict')
    # Enumeration is shared, but this cost projection does NOT transfer from
    # replication to FSDP/offload. Those remain on the measured-cycle route.
    inventory=enumerate_layouts(task,hardware,generator_bytes=generator_bytes,
        score_bytes=score_bytes,safety_bytes=safety_bytes,max_candidates=1024)
    admitted=[];rejected=list(inventory['rejected'])
    for layout in inventory['candidates']:
        try:admission(task,layout,hardware)
        except ValueError as exc:
            rejected.append({'layout':layout.to_dict(),'reason':str(exc)});continue
        if layout not in admitted:admitted.append(layout)
    require(admitted,'no component-predictable native layouts',path='role.predict')
    # Cover each physical pool once before adding further windows. This bounds
    # search while allowing one anchor to evaluate several unseen windows.
    admitted.sort(key=lambda x:(x.window,tuple(x.generator)))
    chosen=admitted[:max_candidates]
    groups={}
    for c in chosen:groups.setdefault(fingerprint(projection(c)),[]).append(c)
    require((samples+1)*len(groups)+1<=max_native_runs,
            'reference, anchor profiles and confirmation exceed budget',path='role.predict')
    identity=_run_identity(task,hardware,bindings,device)
    root=Path(output).resolve()
    if dist.get_rank()==0:root.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    anchors=[];trials=0
    for number,(group,targets) in enumerate(groups.items()):
        layout=replace(targets[0],window=1)
        links=profile_links(task,layout,device,samples=samples)
        # Explicit unprofiled reference/warmup at the SAME initial weights and
        # input cohort. It is charged to the budget and never a training update
        # subsequently reused by the selected executor.
        reference=RolePoolExecutor(task,layout,hardware,bindings,device=device)
        trials+=1
        try:
            reference_receipt,model,optimizer=reference.run()
            value=fingerprint(signature((model,optimizer)))
            reference_outcomes=[None]*hardware.world
            dist.all_gather_object(reference_outcomes,value)
        finally:reference.close()
        del reference,model,optimizer
        runs=[];outcomes=[]
        for repeat in range(samples):
            trials+=1
            executor=RolePoolExecutor(task,layout,hardware,bindings,device=device,profile_components=True)
            try:
                receipt,model,optimizer=executor.run()
                outcome=fingerprint(signature((model,optimizer)))
                peers=[None]*hardware.world;dist.all_gather_object(peers,outcome)
                runs.append(receipt);outcomes.append(peers)
            finally:executor.close()
            del model,optimizer,executor
        require(all(v==reference_outcomes for v in outcomes),'profile differs from unprofiled anchor state',path='role.predict')
        directory=root/f'anchor-{number}'
        if dist.get_rank()==0:directory.mkdir()
        dist.barrier()
        raw_path=directory/'native.json'
        raw={'layout':layout.to_dict(),'runs':runs,'links':links,'outcomes':outcomes,
             'identity':identity,'capacities':list(hardware.capacities),
             'unprofiled_reference':reference_receipt,'reference_outcomes':reference_outcomes}
        if dist.get_rank()==0:
            write_new_json(raw_path,raw)
            costs=CostModel(_binding(identity,task),tuple(_save_costs(
                task,layout,raw,raw_path,directory,_binding(identity,task))))
            write_new_json(directory/'costs.json',costs.to_dict())
        dist.barrier()
        anchors.append({'native_path':str(raw_path),'native_sha256':file_sha256(raw_path),
                        'cost_path':str(directory/'costs.json'),
                        'cost_sha256':file_sha256(directory/'costs.json'),
                        'targets':[c.to_dict() for c in targets]})
    manifest={'schema_version':SCHEMA,'identity':identity,'task':task.to_dict(),
              'hardware':{'capacities':list(hardware.capacities),'node_ids':list(hardware.node_ids),
                          'host_capacities':list(hardware.host_capacities)},'anchors':anchors,
              'budget':{'maximum':max_native_runs,'calibration_runs':trials,
                        'remaining_for_confirmation':max_native_runs-trials,
                        'predicted_candidates':len(chosen),'fully_executed_layouts':len(groups),
                        'unprofiled_reference_runs':len(groups), 'component_profile_runs':samples*len(groups)},
              'candidate_inventory':{'rejected':rejected,'budget_omitted':len(admitted)-len(chosen)},
              'cost_mode':'components','claimable_speedup':False,
              'capability_scope':'replicated_resident_windows_only_not_FSDP_or_offload',
              'extrapolation':'same_shape_same_inputs_same_pools_FIFO_windows_only'}
    if dist.get_rank()==0:write_new_json(root/'calibration.json',manifest)
    dist.barrier()
    return manifest


def _load(path):
    return json.loads(Path(path).read_text())


def search(calibration_path,*,max_states=50000,beam_width=0,timeout_seconds=60):
    from .role_planning import source_identity,_binding,PLAN
    m=_load(calibration_path)
    require(m.get('schema_version')==SCHEMA and m['identity']['source_fingerprint']==source_identity(),
            'changed component calibration/source',path='role.predict')
    task=RoleTask.from_dict(m['task']);h=m['hardware']
    hardware=AllocatedWorld(tuple(h['capacities']),tuple(h['node_ids']),tuple(h['host_capacities']))
    candidates=[];provenance={}
    for anchor in m['anchors']:
        for kind in ('native','cost'):
            require(file_sha256(Path(anchor[kind+'_path']))==anchor[kind+'_sha256'],
                    'changed component '+kind+' receipt',path='role.predict')
        raw=_load(anchor['native_path'])
        require(raw['identity']==m['identity'],'mixed component identity',path='role.predict')
        costs=CostModel.from_dict(_load(anchor['cost_path']));costs.verify_receipts()
        require(costs.binding==_binding(m['identity'],task),'mixed component cost binding',path='role.predict')
        original=PoolLayout.from_dict(raw['layout'])
        for key,(values,workspace,kind) in component_values(task,original,raw).items():
            c=contract(task,original,key);observed=costs.get_component(c)
            require(observed.samples_ns==values and observed.workspace_bytes_per_rank==workspace
                    and observed.evidence['kind']==kind,
                    'costs differ from original native component observations',path='role.predict')
        for d in anchor['targets']:
            layout=PoolLayout.from_dict(d)
            require(projection(layout)==projection(original) and original.window==1,
                    'cannot project component costs across pools or implementations',path='role.predict')
            graph,specs=graph_for(task,layout,hardware)
            states=[];lifetimes=[];rules={}
            transport=transport_bytes_by_rank(task,layout)
            for rank in range(hardware.world):
                observations=[run['ranks'][rank] for run in raw['runs']]
                static=max(max(v['initial_storage_bytes'],v['final_storage_bytes']) for v in observations)
                # Preserve the measured allocator floor. Plus lifetimes/workspace
                # is deliberately conservative; this is not an unseen-shape bound.
                floor=max(v['cuda_peak_allocated_bytes'] or 0 for v in observations)
                if rank in layout.score:static+=transport[rank]
                states.append(StateAllocation(f'state{rank}',(rank,),max(static,floor),
                    ('generator' if rank in layout.generator else 'score',),False,task.digest))
            for i in range(task.microbatches):
                g,_=layout.owners(i)
                saved=max(int(run['ranks'][g]['components']['saved_bytes_by_microbatch'][str(i)])
                          for run in raw['runs'])
                lifetimes.append(Lifetime(f'activation{i}',f'g{i}',(f'b{i}',),(g,),
                                          saved+3*task.bytes_per_latent,'producer_entry'))
            for key,(component,ranks,resources) in specs.items():
                c=contract(task,layout,component);costs.get_component(c)
                rules[key]=BindingRule(key,{},c.digest,ranks,resources,
                                       tuple(f'state{r}' for r in ranks) if len(ranks)==1 else (),c.digest)
            candidate_id=fingerprint(layout.to_dict())
            provenance[candidate_id]={'anchor_window':1,'native_receipt':anchor['native_path'],
                                      'was_full_cycle_measured':layout.window==1}
            candidates.append(BoundCandidate(candidate_id,layout.to_dict(),graph,rules,costs,
                              tuple(states),tuple(lifetimes),dict(enumerate(hardware.capacities)),BACKEND))
    prediction=select(candidates,max_states=max_states,beam_width=beam_width,timeout_seconds=timeout_seconds)
    require(prediction['ranked'],'no feasible component-predicted FIFO plan',path='role.predict')
    selected=prediction['ranked'][0]
    result={'schema_version':PLAN,'calibration':str(Path(calibration_path).resolve()),
            'calibration_sha256':file_sha256(Path(calibration_path)),
            'identity':m['identity'],'task':m['task'],'selected':selected,'search':prediction,
            'cost_mode':'components','execution_model':BACKEND,
            'search_settings':{'max_states':max_states,'beam_width':beam_width,'timeout_seconds':timeout_seconds},
            'prediction_provenance':provenance,'numerical_reference':'REQUIRES_INDEPENDENT_BASELINE',
            'assumptions':['same indexed compute costs across FIFO windows',
                           'isolated links serialize each pair; shared-NIC contention not modeled',
                           'native FIFO precedences, not arbitrary optimized event dispatch'],
            'claimable_speedup':False,'gpu_execution':'NOT_RUN_BY_SEARCH'}
    result['plan_sha256']=fingerprint(result)
    return result


def validate_execution(task, layout, receipt):
    """Prove the runtime followed the graph's producer/Score order, not timings."""
    for row in receipt['ranks']:
        if row['role']=='generator':
            expected=generator_order([i for i in range(task.microbatches)
                                      if layout.owners(i)[0]==row['rank']],layout.window)
            mapping={'generator_forward_send':'g','generator_vjp_backward':'b'}
            observed=[mapping[e['action']]+str(e['microbatch']) for e in row['events']
                      if e['action'] in mapping]
        else:
            expected=[i for i in range(task.microbatches) if layout.owners(i)[1]==row['rank']]
            observed=[e['microbatch'] for e in row['events'] if e['action']=='score_receive_vjp_send']
        require(expected==observed and not row['components']['enabled'],
                'native FIFO projection differs or confirmation was profiled',path='role.predict')
    return {'status':'PASS_NATIVE_FIFO_PROJECTION','timing_validation':'SEPARATE_OBSERVATION',
            'numerical_reference':'REQUIRES_INDEPENDENT_BASELINE'}


def confirm_state(plan, receipt, model, optimizer, cycle):
    """Exact first-cycle anchor state check, separate from prediction accuracy.

    This is a same-layout numerical consistency gate, NOT an independent
    original-DMD oracle. Later fresh-input cycles need their own reference.
    """
    import torch.distributed as dist
    from .torch_backend import signature
    if cycle:
        return {'status':'NOT_COMPARED_FRESH_CYCLE','independent_reference':'NOT_RUN'}
    path=plan['prediction_provenance'][plan['selected']['candidate_id']]['native_receipt']
    raw=_load(path)
    value=fingerprint(signature((model,optimizer)))
    peers=[None]*dist.get_world_size();dist.all_gather_object(peers,value)
    match=peers==raw['outcomes'][0]
    return {'status':'PASS_SAME_ANCHOR_STATE' if match else 'REVIEW_ANCHOR_STATE_MISMATCH',
            'independent_reference':'NOT_RUN','per_rank_match':[
                a==b for a,b in zip(peers,raw['outcomes'][0])]}
