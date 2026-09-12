"""Native H3 action evidence -> reusable components -> the common simulator.

No recipe timing table, hypothetical overlap, or cost inferred from kernel
coverage. Windows reuse source-qualified rank layouts and measured storage.
Unmeasured depth/tile costs remain explicit calibration requests.
"""
from collections import defaultdict
from dataclasses import replace
from itertools import product
from pathlib import Path

from distillschedule.workloads.best_known.registry import (
    _object, file_sha256, fingerprint, git_head, load_profile, repository_root, write_new_json,
)
from .contracts import require, semantic_contract
from .cost_model import CostModel, OperationCost
from .h3_action_calibration import validate_calls
from .h3_action_costs import ADAPTER, H3, PRECISION, RANKS, action_graph, component_contract
from .h3_implementation import H3Implementation
from .h3_native import summarize_native_two_cycle
from .plan_ir import BindingRule, Lifetime, PlanningProblem, StateAllocation, validate_binding
from .wan_action_calibration import validate_lifetimes

SCHEMA = "distillschedule.h3_action_layout.v1"
DOMAINS = dict(prefetch_depth=("1","2"), lora_epilogue_block_n=("64","128"),
               rollout_window=("1","2","3","4","5"))


def _bound_object(reference):
    path=Path(reference['path'])
    require(path.is_file() and file_sha256(path)==reference['sha256'],"H3 source evidence hash changed")
    return _object(path)


def _verify_run_binding(layout, native, summary, root):
    from .execution import validated_profile
    from .h3_native_bridge import _unprofiled_launch_evidence
    from .runtime_identity import verify_measurement_runtime
    plan=_bound_object(layout['execution_receipt'])
    validated_profile(plan,root=root)
    require(plan['mode']=='calibration' and plan['search_candidate']['id']==layout['implementation_id'],
            "H3 effective calibration launcher differs")
    for field,name in (('source_head','integration_head'),('asset_fingerprint','asset_fingerprint'),
                       ('semantic_fingerprint','workload_semantics_hash')):
        require(layout['binding'][field]==plan[name],"H3 source/asset/semantic binding differs from actual launch")
    runtime=_bound_object(layout['runtime_identity_evidence'])
    verify_measurement_runtime({'runtime_identity_evidence':runtime},layout['binding'])
    out=Path(plan['asset_config']['run_root'])/plan['run_id']
    expected=[out/'execution/ranks'/f'rank_{rank:03d}.json' for rank in RANKS]
    require({Path(r['path']).resolve() for r in layout['rank_receipts']}=={p.resolve() for p in expected},
            "H3 calibration ranks do not belong to the bound run")
    _unprofiled_launch_evidence(plan,out,summary,mode='calibration')
    for row in native:
        observed=next(n['environment'] for n in runtime['before'] if n['node_rank']==row['rank']//8)
        actual=row['runtime_identity']
        require(actual['torch_version']==observed['torch'] and actual['torch_cuda_version']==observed['torch_cuda']
                and Path(actual['python_executable']).resolve()==Path(observed['executable']).resolve(),
                "H3 rank Python/Torch differs from observed calibration environment")
    for node in (0,1):
        resource=_object(out/f'search-resources.node{node}.json')
        require(resource['status']=='PASS' and resource['run_id']==plan['run_id']
                and resource['node_rank']==node and len(resource['samples'])>=2,"H3 resource observation missing")
        for event in ('oom','oom_kill'):
            counts=[s['cgroup_events'][event] for s in resource['samples']]
            require(max(counts)==min(counts),'H3 cgroup OOM invalidates calibration')
        flags=dict(line.split('=',1) for line in (out/'launcher'/f'node_{node}'/'teardown_status.txt').read_text().splitlines())
        require(flags==dict(status='PASS',gpu_process_count='0',owner_run_id=plan['run_id']),"H3 teardown incomplete")


def _read(path, root):
    layout=_object(path)
    require(layout.get('schema_version')==SCHEMA,"not an H3 action layout")
    binding=layout['binding']; validate_binding(binding)
    require(binding['source_head']==git_head(root),"H3 action calibration source differs")
    require(binding['precision_fingerprint']==PRECISION and binding['semantic_fingerprint']==
            fingerprint(semantic_contract(load_profile(H3))),"H3 precision/semantics differs")
    refs=layout['rank_receipts']
    require(len(refs)==16 and len({r['path'] for r in refs})==16,"H3 needs sixteen calibration sources")
    native=[]
    for ref in refs:
        p=Path(ref['path'])
        require(file_sha256(p)==ref['sha256'],"H3 calibration rank hash changed")
        native.append(_object(p))
    native.sort(key=lambda r:r['rank'])
    impl=H3Implementation.parse(layout['implementation_id'])
    summary=summarize_native_two_cycle(native,expected_head=binding['source_head'],candidate_id=impl.id)
    require(summary['calibration_only'] and summary['run_kind']=='calibration',"H3 native timing is not a component calibration")
    _verify_run_binding(layout,native,summary,root)
    ranks=[r['stage_calibration'] for r in native]
    graph=action_graph(impl.rollout_window)
    for r in ranks:
        require(r['schema_version']=='distillschedule.h3_action_rank_calibration.v1' and r['status']=='PASS'
                and r['evidence_scope']=='MEASURED' and r['source_head']==binding['source_head']
                and r['implementation_id']==impl.id and r['adapter']==ADAPTER
                and r['precision_fingerprint']==PRECISION and r['profiled'] is False,
                "H3 action collector identity/status differs")
        require(r['native_graph_sha256']==fingerprint(graph.to_dict()) and
                r['completed_operation_ids']==[o.operation_id for o in graph.operations] and
                not r['missing_operation_ids'] and not r['errors'],"H3 action stages incomplete")
        validate_lifetimes(r)
        for op,s in zip(graph.operations,r['stages']):
            work=r['work_signatures'][op.operation_id]
            validate_calls(op.metadata['action'],work['application_calls'])
            contract=component_contract(op,impl,work)
            require(s['status']=='PASS' and s['operation_id']==op.operation_id
                    and s['key']==s['operation_contract_sha256']==contract.digest
                    and s['component_contract']==s['operation_contract']==contract.to_dict()
                    and type(s['duration_ns']) is int and s['duration_ns']>0
                    and s['duration_ns']==s['stop_ns']-s['start_ns'],"H3 component observation differs")
            require(s['workspace_bytes_per_rank']==max(0,s['memory_after']['peak_allocated_bytes']-
                    r['post_setup_static_bytes']-s['entry_retained_input_bytes']),"H3 allocation envelope differs")
    require({r['rank'] for r in ranks}==set(RANKS),"H3 calibration rank ownership differs")
    return layout,impl,ranks


def _work(ranks):
    # All rank-qualified physical layouts, including variable text lengths,
    # enter the cost key. Same shapes may share a component; source assets bind values.
    return {op.operation_id:dict(physical_batch=1,logical_batch=16,
                rank_layouts=[r['work_signatures'][op.operation_id] for r in ranks])
            for op in action_graph().operations}


def build_problem(layout_path, *, root=None):
    root=root or repository_root()
    layout,impl,ranks=_read(layout_path,root)
    binding,work=layout['binding'],_work(ranks)
    static=max(r['post_setup_static_bytes'] for r in ranks)
    outputs={key:max(next(l['bytes'] for l in r['output_lifetimes'] if l['producer']==key) for r in ranks)
             for key in (o.operation_id for o in action_graph().operations if o.metadata['action'].startswith('r'))}
    # Allocator-reserved unused pages are reclaimable; non-allocator occupancy
    # at setup is excluded. Keep 2 GiB headroom in the predictive allocation gate.
    limits={r['rank']:r['post_setup_memory']['driver_free_bytes']+
            r['post_setup_memory']['reserved_bytes']-2*1024**3 for r in ranks}
    require(all(v>0 for v in limits.values()),"H3 insufficient observed driver capacity")
    graph=action_graph()
    graph=replace(graph,metadata={**graph.metadata,'canonical_profile_sha256':fingerprint(load_profile(H3))})
    state=StateAllocation('native:shared',RANKS,static,('native_runtime',),False,
                          fingerprint(layout['rank_receipts']))
    lives=tuple(Lifetime(k,k,(k.replace('.r','.c'),),RANKS,v,'producer_exit') for k,v in sorted(outputs.items()))
    rules=[];edges=[]
    for d,n,w in product((1,2),(64,128),range(1,6)):
        candidate=H3Implementation(d,n,w)
        for op in graph.operations:
            c=component_contract(op,candidate,work[op.operation_id])
            rules.append(BindingRule(op.operation_id,candidate.assignment,c.digest,RANKS,
                tuple(f'gpu:{r}' for r in RANKS),('native:shared',),c.digest))
    for w in range(1,6):
        order=action_graph(w).topological_order()
        edges.extend(dict(when={'rollout_window':str(w)},source=a,target=b) for a,b in zip(order,order[1:]))
    return PlanningProblem(H3,binding,graph,DOMAINS,tuple(rules),(state,),lives,limits,
        (H3Implementation(impl.prefetch_depth,impl.lora_epilogue_block_n,5).assignment,impl.assignment),
        conditional_dependencies=tuple(edges),evidence_scope='MEASURED',
        lowering=dict(adapter=ADAPTER,layout_receipt=str(layout_path.resolve()),layout_sha256=file_sha256(layout_path),
            memory_scope='CONSERVATIVE_PERSISTENT_GROWTH_ENVELOPE_PLUS_RETAINED_STORAGE; NATIVE_CONFIRMATION_REQUIRED',
            static_alternative_policy='UNMEASURED_DEPTH_TILE_MUST_NOT_BE_ADMITTED_FROM_MISSING_COSTS'))


def validate_native_problem(problem):
    require(problem.evidence_scope=='MEASURED' and problem.lowering.get('adapter')==ADAPTER,"not measured H3 actions")
    path=Path(problem.lowering['layout_receipt'])
    require(file_sha256(path)==problem.lowering['layout_sha256'],"H3 layout hash changed")
    require(build_problem(path).digest==problem.digest,"H3 generated graph/layout changed")


def assemble_calibration(layout_path, *, output, root=None):
    root=root or repository_root()
    layout,impl,ranks=_read(layout_path,root)
    work=_work(ranks); grouped=defaultdict(list); contracts={}
    by_rank=[{s['operation_id']:s for s in r['stages']} for r in ranks]
    for op in action_graph().operations:
        contract=component_contract(op,impl,work[op.operation_id])
        grouped[contract.digest].append(op.operation_id);contracts[contract.digest]=contract
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for key,ids in sorted(grouped.items()):
        samples=tuple(max(rows[i]['duration_ns'] for rows in by_rank) for i in ids)
        memory=max(rows[i]['workspace_bytes_per_rank'] for rows in by_rank for i in ids)
        raw=dict(schema_version='distillschedule.operation_calibration.v1',binding=layout['binding'],key=key,
            samples_ns=list(samples),workspace_bytes_per_rank=memory,status='PASS',profiled=False,
            kind='native_operation',operation_contract_sha256=key,component_contract=contracts[key].to_dict(),
            source_rank_receipts=layout['rank_receipts'],operation_ids=ids,
            h3_action_layout=dict(path=str(layout_path.resolve()),sha256=file_sha256(layout_path)),
            duration_definition='MAX_RANK_FENCED_NATIVE_ACTION; NOT_NATIVE_CYCLE_WALL',
            allocation_limitations=ranks[0]['allocation_limitations'])
        path=output/'components'/f'{key}.json';write_new_json(path,raw)
        evidence={k:raw[k] for k in ('status','profiled','kind','operation_contract_sha256','component_contract')}
        evidence.update(source_receipt=str(path.resolve()),source_receipt_sha256=file_sha256(path))
        records.append(OperationCost(key,samples,memory,evidence))
    problem=build_problem(layout_path,root=root)
    costs=CostModel(layout['binding'],tuple(records));costs.verify_receipts()
    write_new_json(output/'problem.json',problem.to_dict());write_new_json(output/'cost-model.json',costs.to_dict())
    return problem,costs


def validate_component_receipt(raw, cache):
    """Re-derive compiled costs, so rehashing fabricated costs cannot admit them."""
    ref=raw.get('h3_action_layout')
    require(isinstance(ref,dict),'H3 component lacks native action provenance')
    key=(ref['path'],ref['sha256'])
    if key not in cache:
        path=Path(ref['path'])
        require(file_sha256(path)==ref['sha256'],'H3 component layout hash changed')
        layout,impl,ranks=_read(path,repository_root())
        works=_work(ranks);grouped=defaultdict(list)
        by_rank=[{s['operation_id']:s for s in r['stages']} for r in ranks]
        contracts={}
        for op in action_graph().operations:
            c=component_contract(op,impl,works[op.operation_id]);contracts[c.digest]=c.to_dict()
            grouped[c.digest].append(op.operation_id)
        cache[key]={k:dict(binding=layout['binding'],component_contract=contracts[k],operation_ids=ids,
            source_rank_receipts=layout['rank_receipts'],samples_ns=[max(rows[i]['duration_ns'] for rows in by_rank) for i in ids],
            workspace_bytes_per_rank=max(rows[i]['workspace_bytes_per_rank'] for rows in by_rank for i in ids))
            for k,ids in grouped.items()}
    expected=cache[key].get(raw['key'])
    require(expected is not None and all(raw.get(k)==v for k,v in expected.items()),
            'H3 compiled cost differs from native rank observations')
