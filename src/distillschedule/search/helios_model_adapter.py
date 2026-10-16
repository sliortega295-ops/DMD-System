"""Measured Helios components -> common plan search -> typed native lowering.

No fallback to a historical winner. Internal CUDA overlap is already inside
each fenced calibration envelope; serial model search invents no overlap gain.
GAN persistence includes acquire/all five rollouts/release as one component.
"""
from collections import defaultdict
from dataclasses import replace
from itertools import product
from pathlib import Path

from distillschedule.workloads.best_known.registry import (
    _object,file_sha256,fingerprint,git_head,repository_root,load_profile,write_new_json,
)
from .contracts import require,semantic_contract
from .helios_implementation import HeliosImplementation,NO_GAN,GAN
from .helios_request import OriginalHeliosRequest,TRACE_KEY
from .helios_safe_bundle import BUNDLE_ID,HeliosSafeBundle
from .helios_action_costs import ADAPTER,PRECISION,RANKS,MEMORY,anchors,action_graph,component_contract,observed_work
from .cost_model import CostModel,OperationCost
from .plan_ir import PlanningProblem,BindingRule,StateAllocation,Lifetime,validate_binding
from .helios_native import bound,ref,summarize,launch_evidence,validate_admission

SCHEMA='distillschedule.helios_component_layout.v1'
FRAGMENT='distillschedule.helios_component_fragment.v1'


def capability_report(workload, *, include_safe_incumbent=False, request=None):
    require(workload in (NO_GAN,GAN),'unknown Helios workload')
    require(type(include_safe_incumbent) is bool,'safe incumbent opt-in must be boolean')
    require(not include_safe_incumbent or workload==NO_GAN,'safe incumbent is No-GAN only; GAN is not migrated')
    require(request is None or (type(request) is OriginalHeliosRequest
            and request.variant==('no-gan' if workload==NO_GAN else 'gan')), 'typed matching Helios request required')
    report=dict(workload=workload,adapter=ADAPTER,arbitrary_pool_lowering=False,
        placement=['world16-fsdp','node-hsdp'] if workload==NO_GAN else ['world16-fsdp'],
        modulation_chunk_tokens=[512,2048],rollout_window=list(range(1,6)) if workload==NO_GAN else [5],
        persistent_generator_unshard=[False] if workload==NO_GAN else [False,True],
        sharing='EXISTING_SCORE_TEACHER_ADAPTER_ALIAS_ONLY; NO_NEW_BACKBONE_ALIAS',
        cost_policy='MISSING_NATIVE_COMPONENTS_UNCALIBRATED',
        gan_window_restriction='R_ALL_OWNS_ACQUIRE_FIVE_B1_RELEASE; SPLIT_SCOPE_NOT_ADMITTED',
        gpu_execution='NOT_RUN_BY_CAPABILITY_INVENTORY',historical_timings_used_as_costs=False)
    if include_safe_incumbent:
        report['candidate_bundles']={'none':'LEGACY_UNCHANGED',BUNDLE_ID:dict(
            assignment=safe_incumbent(request=request).assignment,fields=HeliosSafeBundle.safe().cost_identity(),
            calibration='MATCHED_FSDP_C512_K0_AND_HSDP_C512_K1_REQUIRED')}
    if request is not None:
        report[TRACE_KEY]=request.to_dict()
        report['request_scope']='FRESH_EXACT_HASH_QUALIFIED_TRACE_AND_COMPONENT_CALIBRATION_REQUIRED'
    return report


def domains(workload, *, include_safe_incumbent=False, request=None):
    c=capability_report(workload,include_safe_incumbent=include_safe_incumbent,request=request)
    result=dict(variant=('no-gan',) if workload==NO_GAN else ('gan',),mesh=tuple(c['placement']),
        chunk_tokens=('512','2048'),rollout_window=tuple(map(str,c['rollout_window'])),
        persistent_unshard=('off',) if workload==NO_GAN else ('off','on'))
    if include_safe_incumbent:result['candidate_bundle']=('none',BUNDLE_ID)
    if request is not None:
        result['original_request']=(anchors(workload,request=request)[0].assignment['original_request'],)
    return result


def safe_incumbent(*, request=None):
    return HeliosImplementation('no-gan','node-hsdp',512,5,False,HeliosSafeBundle.safe(),request)


def implementations(workload, *, include_safe_incumbent=False, request=None):
    require(type(include_safe_incumbent) is bool,'safe incumbent opt-in must be boolean')
    require(not include_safe_incumbent or workload==NO_GAN,'safe incumbent is No-GAN only; GAN is not migrated')
    d=domains(workload,request=request)
    values=tuple(HeliosImplementation.from_assignment(dict(zip(d,values))) for values in product(*d.values()))
    return (*values,safe_incumbent(request=request)) if include_safe_incumbent else values


def _read_fragment(reference,root):
    from .execution import validated_profile
    from .runtime_identity import verify_measurement_runtime
    f=bound(reference);require(f.get('schema_version')==FRAGMENT,'not a Helios component fragment')
    b=f['binding'];validate_binding(b)
    require(b['source_head']==git_head(root) and b['precision_fingerprint']==PRECISION,'Helios component source/precision differs')
    impl=HeliosImplementation.parse(f['implementation_id']);w=impl.workload
    require(impl.assignment in [i.assignment for i in implementations(w,include_safe_incumbent=w==NO_GAN,request=impl.request)],
            'Helios graph family not admitted')
    semantics=impl.lower_profile() if impl.request is not None else load_profile(w)
    require(b['semantic_fingerprint']==fingerprint(semantic_contract(semantics)),'Helios workload contract differs')
    plan=bound(f['execution_receipt']);validated_profile(plan,root=root)
    require(plan['mode']=='calibration' and plan['search_candidate']['id']==impl.id,'not this native calibration')
    for field,key in (('source_head','integration_head'),('asset_fingerprint','asset_fingerprint'),
                      ('semantic_fingerprint','workload_semantics_hash')):
        require(b[field]==plan[key],'native calibration binding differs')
    runtime=bound(f['runtime_identity_evidence']);verify_measurement_runtime({'runtime_identity_evidence':runtime},b)
    out=Path(plan['asset_config']['run_root'])/plan['run_id']
    require({Path(r['path']).resolve() for r in f['rank_receipts']}==
            {(out/'execution/ranks'/f'rank_{i:02d}.json').resolve() for i in RANKS},'rank sources do not belong to launch')
    rows=sorted([bound(r) for r in f['rank_receipts']],key=lambda r:r['rank'])
    s=summarize(rows,expected_head=b['source_head'],candidate_id=impl.id)
    require(s['calibration_only'],'native timing is not component calibration')
    validate_admission(bound(plan['asset_config']['correctness']),plan)
    launch_evidence(plan,out)
    for row in rows:
        env=next(n['environment'] for n in runtime['before'] if n['node_rank']==row['rank']//8)
        actual=row['runtime_identity']
        require(actual['torch_version']==env['torch'] and actual['cuda_version']==env['torch_cuda']
                and Path(actual['python_executable']).resolve()==Path(env['executable']).resolve(),
                'native worker environment differs from observed environment')
    return f,impl,rows


def _component_samples(impl,rows):
    work=observed_work(rows);graph=action_graph(impl)
    by_rank=[{f'cycle{c["cycle"]}.{a["action_id"]}':a for c in r['cycles'] for a in c['actions']} for r in rows]
    entry=[]
    for r in rows:
        values={}
        for cycle in r['cycles']:
            prior=0
            for a in cycle['actions']:
                values[f'cycle{cycle["cycle"]}.{a["action_id"]}']=prior
                prior=a['retained_after']['bytes']
        entry.append(values)
    grouped={}
    for op in graph.operations:
        contract=component_contract(op,impl,work[op.operation_id]);key=contract.digest
        g=grouped.setdefault(key,dict(contract=contract,operation_ids=[],samples_ns=[],workspace_bytes_per_rank=0))
        g['operation_ids'].append(op.operation_id)
        g['samples_ns'].append(max(round(m[op.operation_id]['calibration_seconds']*1e9) for m in by_rank))
        g['workspace_bytes_per_rank']=max(g['workspace_bytes_per_rank'],max(max(0,
            m[op.operation_id]['memory_after']['max_allocated_bytes']-r['post_setup_memory']['allocated_bytes']-
            e[op.operation_id]) for r,m,e in zip(rows,by_rank,entry,strict=True)))
    return grouped


def build_problem(layout_path,*,root=None):
    root=root or repository_root();layout=_object(layout_path)
    require(layout.get('schema_version')==SCHEMA and layout.get('fragments'),'Helios native component layout required')
    include_safe=layout.get('include_safe_incumbent_bundle',False)
    request=(OriginalHeliosRequest.from_dict(layout[TRACE_KEY]) if TRACE_KEY in layout else None)
    require(type(include_safe) is bool,'safe incumbent layout opt-in must be boolean')
    measured_ids=set()
    identity=work=workload=limits=None;static={};outputs={};sources=[]
    for reference in layout['fragments']:
        f,impl,rows=_read_fragment(reference,root)
        require(impl.request==request,'Helios fragment request differs from explicit layout request')
        require(not impl.candidate_bundle.enabled or include_safe,'safe incumbent fragment requires explicit layout opt-in')
        require(not include_safe or impl.workload==NO_GAN,'safe incumbent is No-GAN only; GAN is not migrated')
        measured_ids.add(impl.id)
        require(identity is None or identity==f['binding'],'Helios calibration identities differ')
        require(workload is None or workload==impl.workload,'cannot mix GAN and No-GAN components')
        identity,workload=f['binding'],impl.workload
        observed=observed_work(rows);require(work is None or work==observed,'physical shape/call work differs')
        work=observed;static[impl.mesh]=max(static.get(impl.mesh,0),max(r['post_setup_memory']['allocated_bytes'] for r in rows))
        if limits is None:limits={}
        for r in rows:
            memories=[r['post_setup_memory'],*[a[k] for c in r['cycles'] for a in c['actions'] for k in ('memory_before','memory_after')]]
            overhead=max(max(0,m['driver_total_bytes']-m['driver_free_bytes']-m['reserved_bytes']) for m in memories)
            limit=r['post_setup_memory']['driver_total_bytes']-overhead-2*2**30
            require(limit>0,'invalid observed capacity')
            limits[r['rank']]=min(limits.get(r['rank'],limit),limit)
            for c in r['cycles']:
                for life in c['retained_lifetimes']:
                    key=(c['cycle'],life['source_step'])
                    outputs[key]=max(outputs.get(key,0),life['bytes'])
        sources.append(reference)
    baseline,incumbent=anchors(workload,request=request)
    if include_safe:
        # Two measured arms suffice: the explicit k1 arm is the mandatory
        # incumbent, not an uncalibrated old-HSDP k0 seed disguised as measured.
        incumbent=safe_incumbent(request=request)
        require({baseline.id,incumbent.id}<=measured_ids,
                'safe incumbent problem requires measured FSDP C512 k0 control and HSDP C512 k1 incumbent')
    require({i.mesh for i in (baseline,incumbent)}<=set(static),'matched mesh construction layouts missing')
    graph=action_graph(baseline,selected=False)
    states=tuple(StateAllocation('native:'+mesh,RANKS,size,('native_runtime',),False,fingerprint(sources))
                 for mesh,size in sorted(static.items()))
    steps=load_profile(workload)['contract']['logical_steps']
    lives=tuple(Lifetime(f'cycle{cycle}.latent{steps.index(step)}',
        f'cycle{cycle}.r_all' if workload==GAN else f'cycle{cycle}.r{steps.index(step)}',
        (f'cycle{cycle}.c{steps.index(step)}',),RANKS,size,'producer_exit') for (cycle,step),size in sorted(outputs.items()))
    require(len(lives)==10,'ten measured retained output lifetimes required')
    rules=[];edges=[]
    def assignment(impl):
        return impl.assignment|({'candidate_bundle':impl.candidate_bundle.id} if include_safe else {})
    for impl in implementations(workload,include_safe_incumbent=include_safe,request=request):
        if impl.mesh not in static:continue
        for op in graph.operations:
            contract=component_contract(op,impl,work[op.operation_id])
            rules.append(BindingRule(op.operation_id,assignment(impl),contract.digest,RANKS,
                tuple(f'gpu:{r}' for r in RANKS),('native:'+impl.mesh,),contract.digest))
    for w in domains(workload,request=request)['rollout_window']:
        order=action_graph(replace(baseline,rollout_window=int(w))).topological_order()
        edges.extend(dict(when={'rollout_window':w},source=a,target=b) for a,b in zip(order,order[1:]))
    decisions=domains(workload,include_safe_incumbent=include_safe,request=request)
    forbidden=tuple({'candidate_bundle':BUNDLE_ID,key:value}
        for key,values in decisions.items() if key!='candidate_bundle'
        for value in values if value!=safe_incumbent(request=request).assignment[key]) if include_safe else ()
    return PlanningProblem(workload,identity,graph,decisions,tuple(rules),states,lives,limits,
        (assignment(baseline),assignment(incumbent)),forbidden=forbidden,
        conditional_dependencies=tuple(edges),evidence_scope='MEASURED',
        lowering=dict(adapter=ADAPTER,layout_receipt=str(Path(layout_path).resolve()),layout_sha256=file_sha256(layout_path),
            memory_scope=MEMORY+'; TWO_GIB_MARGIN; NATIVE_CAPACITY_CONFIRMATION_REQUIRED',
            resource_scope='SERIAL_ACTIONS_NO_INVENTED_COMM_OVERLAP',component_sources=sources))


def validate_native_problem(problem):
    require(problem.evidence_scope=='MEASURED' and problem.lowering.get('adapter')==ADAPTER,'not measured Helios actions')
    path=Path(problem.lowering['layout_receipt'])
    require(file_sha256(path)==problem.lowering['layout_sha256'],'Helios layout hash changed')
    require(build_problem(path).digest==problem.digest,'Helios native graph/lowering changed')


def lower_prediction(result,problem,costs,*,run_id,assets_path=None,mode='correctness'):
    from .model_search import validate_saved_prediction
    from .execution import resolve_candidate
    validate_native_problem(problem);costs.verify_receipts()
    p=validate_saved_prediction(result,problem,costs)
    require(p['status']=='PREDICTED_FEASIBLE','cannot lower an uncalibrated/infeasible plan')
    impl=HeliosImplementation.from_assignment(p['assignment'])
    receipt=resolve_candidate(problem.workload,impl.id,assets_path=assets_path,run_id=run_id,mode=mode)
    receipt['model_plan']=dict(problem=problem.to_dict(),cost_model=costs.to_dict(),search_result=dict(result))
    validate_lowered_plan(receipt);return receipt


def validate_lowered_plan(receipt):
    from .model_search import validate_saved_prediction
    p=PlanningProblem.from_dict(receipt['model_plan']['problem']);validate_native_problem(p)
    costs=CostModel.from_dict(receipt['model_plan']['cost_model']);costs.verify_receipts()
    chosen=validate_saved_prediction(receipt['model_plan']['search_result'],p,costs)
    impl=HeliosImplementation.from_assignment(chosen['assignment'])
    require(receipt['search_candidate']==impl.candidate().to_dict()
            and receipt['integration_head']==p.binding['source_head']
            and receipt['workload_semantics_hash']==p.binding['semantic_fingerprint'],'searched and launched choices differ')
    if receipt.get('asset_config'):require(receipt['asset_fingerprint']==p.binding['asset_fingerprint'],'native assets differ')


def assemble_native(receipt,preparation,identity,runtime_identity_evidence):
    from .helios_native import checked
    from .runtime_identity import verify_measurement_runtime
    validate_binding(identity);verify_measurement_runtime({'runtime_identity_evidence':runtime_identity_evidence},identity)
    out,summary,rows,refs=checked(receipt)
    require(receipt['mode']=='calibration' and summary['calibration_only'],'not Helios component calibration')
    require(preparation['candidate_id']==receipt['search_candidate']['id'],'calibration preparation differs')
    target=out/'component-costs';target.mkdir(exist_ok=False)
    p=target/'execution-plan.json';write_new_json(p,receipt)
    rt=target/'runtime-identity.json';write_new_json(rt,runtime_identity_evidence)
    fragment=dict(schema_version=FRAGMENT,binding=identity,implementation_id=receipt['search_candidate']['id'],
        execution_receipt=ref(p),runtime_identity_evidence=ref(rt),rank_receipts=refs)
    path=target/'fragment.json';write_new_json(path,fragment)
    _,impl,rows=_read_fragment(ref(path),repository_root())
    records=[]
    for key,row in _component_samples(impl,rows).items():
        raw=dict(schema_version='distillschedule.operation_calibration.v1',binding=identity,key=key,
            samples_ns=row['samples_ns'],workspace_bytes_per_rank=row['workspace_bytes_per_rank'],
            status='PASS',profiled=False,kind='native_operation',operation_contract_sha256=key,
            component_contract=row['contract'].to_dict(),operation_ids=row['operation_ids'],
            helios_native_fragment=ref(path),calibration_only=True,formal_timing_eligible=False,
            duration_definition='MAX_RANK_LOCAL_FENCED_ACTION_NOT_FORMAL_WALL')
        f=target/'components'/f'{key}.json';write_new_json(f,raw)
        evidence={k:raw[k] for k in ('status','profiled','kind','operation_contract_sha256','component_contract')}
        evidence.update(source_receipt=str(f),source_receipt_sha256=file_sha256(f))
        records.append(OperationCost(key,tuple(raw['samples_ns']),raw['workspace_bytes_per_rank'],evidence))
    costs=CostModel(identity,tuple(records));costs.verify_receipts()
    p=target/'cost-model.json';write_new_json(p,costs.to_dict())
    return dict(status='PASS_NATIVE_COMPONENT_CALIBRATION',source_head=identity['source_head'],
        implementation_id=impl.id,rank_count=16,layout_fragment=ref(path),operation_costs=ref(p),
        formal_timing_eligible=False,cycle_wall_seconds=summary['cycle_wall_seconds'])


def validate_component_receipt(raw,cache):
    reference=raw.get('helios_native_fragment');require(isinstance(reference,dict),'Helios component provenance missing')
    key=(reference['path'],reference['sha256'])
    if key not in cache:
        f,impl,rows=_read_fragment(reference,repository_root())
        cache[key]=(f['binding'],_component_samples(impl,rows))
    binding,samples=cache[key];row=samples.get(raw['key'])
    require(row is not None and raw['binding']==binding and raw['samples_ns']==row['samples_ns']
            and raw['workspace_bytes_per_rank']==row['workspace_bytes_per_rank']
            and raw['component_contract']==row['contract'].to_dict() and raw['operation_ids']==row['operation_ids'],
            'Helios component cost differs from native rank observations')


def assemble_calibrations(results,*,output,expected_head,root=None,include_safe_incumbent=False,request=None):
    from .component_calibration import pool_component_models
    root=root or repository_root();require(git_head(root)==expected_head,'assembly source changed')
    require(type(include_safe_incumbent) is bool,'safe incumbent assembly opt-in must be boolean')
    require(request is None or type(request) is OriginalHeliosRequest,'typed Helios assembly request required')
    fragments=[];models=[]
    for path in results:
        row=_object(path)
        require(row['status']=='PASS_NATIVE_COMPONENT_CALIBRATION' and row['source_head']==expected_head,
                'invalid Helios calibration result')
        _,impl,_=_read_fragment(row['layout_fragment'],root)
        require(impl.request==request,'Helios assembly request differs from calibrated fragment')
        model=CostModel.from_dict(bound(row['operation_costs']));model.verify_receipts()
        fragments.append(row['layout_fragment']);models.append(model)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    data=dict(schema_version=SCHEMA,fragments=fragments)
    if request is not None:data[TRACE_KEY]=request.to_dict()
    if include_safe_incumbent:data['include_safe_incumbent_bundle']=True
    layout=output/'layout.json';write_new_json(layout,data)
    problem=build_problem(layout,root=root);costs=pool_component_models(models,output=output/'components')
    costs.bind(problem.binding,problem_scope=problem.evidence_scope)
    write_new_json(output/'problem.json',problem.to_dict());write_new_json(output/'cost-model.json',costs.to_dict())
    return problem,costs
