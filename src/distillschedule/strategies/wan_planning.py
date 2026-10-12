"""Original native backend costs -> layout enumeration + existing event DP.

Only observed/admitted implementations are ranked. No embedded speedups, model
name winner map, synthetic GPU timings, or Euler-protocol substitution.
"""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from distillschedule.core.contracts import require
from distillschedule.runtime.parameterized_cycle import build_parameterized_cycle_graph
from distillschedule.search.cost_model import ComponentContract,CostModel,OperationCost
from distillschedule.search.frontier_schedule import schedule_frontier
from distillschedule.search.plan_ir import BindingRule,StateAllocation,Lifetime
from distillschedule.workloads.best_known.registry import fingerprint,file_sha256,write_new_json,repository_root
from .wan_strategy import WanStrategy,implementations
from .wan_task import OriginalWanTask
from .wan_calibration import component

COLLECTION='distillschedule.original_wan_calibration_collection.v1'
PLAN='distillschedule.original_wan_plan.v1'


def source_fingerprint():
    root=repository_root()
    # Include transitive runtime/helper code, not only the front-door module.
    paths=[]
    for directory in ('src/distillschedule','runtime','third_party/helios_compat/source'):
        paths.extend((root/directory).rglob('*.py'))
    paths.extend((root/'scripts/distillschedule').glob('probe_*.py'))
    return fingerprint({str(p.relative_to(root)):file_sha256(p) for p in sorted(set(paths))})


def _read(path):
    value=json.loads(Path(path).read_text())
    require(isinstance(value,dict),'expected JSON object',path='wan.plan')
    return value


def assemble(task, rank_paths, output, *, reserve_bytes=2**31):
    """One completed native calibration arm, all 16 ranks; no new model work."""
    require(type(reserve_bytes) is int and reserve_bytes>=0,'invalid reserve',path='wan.calibration')
    rows=[_read(p) for p in rank_paths]
    require(len(rows)==16 and {r.get('rank') for r in rows}==set(range(16)),
            'complete rank16 calibration required',path='wan.calibration')
    rows.sort(key=lambda r:r['rank']);first=rows[0]
    strategy=WanStrategy.from_dict(first['strategy']);strategy.validate_for(task)
    require(strategy.rollout_window==1,'calibration must use one retained rollout',path='wan.calibration')
    identity=first['identity']
    require(identity['source_fingerprint']==source_fingerprint() and identity['task_sha256']==task.digest,
            'stale source/task calibration',path='wan.calibration')
    for row in rows:
        require(row.get('status')=='COMPLETED_ORIGINAL_WAN_WORK' and row.get('mode')=='calibrate'
                and row['task']==task.to_dict() and row['identity']==identity
                and row['strategy']==strategy.to_dict(), 'mismatched native calibration',path='wan.calibration')
        require(row.get('finite') is True and row.get('gpu_execution')=='NATIVE_CUDA_RUN',
                'missing native execution/finite checks',path='wan.calibration')
        c=row['calibration']
        require(c['status']=='PASS' and c['evidence_scope']=='MEASURED'
                and c['pending_output'] is None and c['retained_input_bytes_at_finish']==0,
                'incomplete original action observation',path='wan.calibration')
        require(row['final_versions']==first['final_versions'] and row['initial_versions']==first['initial_versions'],
                'rank update counts diverge',path='wan.calibration')
    graph=build_parameterized_cycle_graph(source_steps=task.source_steps,
        initial_versions=first['warm_initial_versions'],ema_after=task.ema_after)
    work=first['calibration']['work_signatures']
    require(all(r['calibration']['work_signatures']==work for r in rows),'different physical work',path='wan.calibration')
    binding={'source_head':identity['source_head'], 'hardware_fingerprint':fingerprint([r['hardware'] for r in rows]),
        'environment_fingerprint':fingerprint({'source':identity['source_fingerprint'],'runtime':[r['runtime'] for r in rows]}),
        'asset_fingerprint':fingerprint(identity['assets']),'semantic_fingerprint':task.digest,
        'precision_fingerprint':fingerprint(first['runtime'])}
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    raw_sources=[{'path':str(Path(p).resolve()),'sha256':file_sha256(Path(p))} for p in rank_paths]
    costs=[];output_sizes={}
    for op in graph.operations:
        action=op.operation_id
        observed=[]
        for r in rows:
            found=[v for v in r['calibration']['stages'] if v['operation_id']=='cycle2.'+action]
            require(len(found)==1 and found[0]['status']=='PASS','missing/duplicate warm action',path='wan.calibration')
            observed.append(found[0])
        contract=component(task,action,strategy,cycle=2,work=work)
        require(all(o['operation_contract']==contract.to_dict() for o in observed),
                'action does not match current work contract',path='wan.calibration')
        samples=[max(o['duration_ns'] for o in observed)]
        workspace=max(o['workspace_bytes_per_rank'] for o in observed)
        if action.startswith('r'):
            output_sizes[action]=max(o['produced_output_bytes'] for o in observed)
            require(output_sizes[action]>0,'missing retained storage',path='wan.calibration')
        raw={'schema_version':'distillschedule.operation_calibration.v1','binding':binding,
            'key':contract.digest,'samples_ns':samples,'workspace_bytes_per_rank':workspace,
            'status':'PASS','profiled':False,'kind':'native_operation',
            'operation_contract_sha256':contract.digest,'component_contract':contract.to_dict(),
            'native_rank_sources':raw_sources,'aggregation':'max_rank_duration_and_workspace',
            'claimable_speedup':False}
        p=root/f'{action}.json';write_new_json(p,raw)
        evidence={k:raw[k] for k in ('status','profiled','kind','operation_contract_sha256','component_contract')}
        evidence.update(source_receipt=str(p),source_receipt_sha256=file_sha256(p))
        costs.append(OperationCost(contract.digest,tuple(samples),workspace,evidence))
    model=CostModel(binding,tuple(costs));model.verify_receipts()
    cost_path=root/'costs.json';write_new_json(cost_path,model.to_dict())
    manifest={'schema_version':COLLECTION,'task':task.to_dict(),'identity':identity,
        'warm_initial_versions':first['warm_initial_versions'],'work':work,
        'variants':[{'strategy':strategy.to_dict(),'cost_model':str(cost_path),'cost_sha256':file_sha256(cost_path),
            'output_bytes':output_sizes,'static_bytes':max(r['calibration']['post_setup_static_bytes'] for r in rows),
            'memory_limit_bytes':min(r['hardware']['total_memory']-reserve_bytes for r in rows),
            'rank_sources':raw_sources}], 'evidence_scope':'MEASURED','gpu_execution':'CALIBRATION_ONLY',
        'claimable_speedup':False,'numerical_comparison':'REQUIRES_MATCHED_REFERENCE'}
    write_new_json(root/'collection.json',manifest)
    return manifest


def load_collections(task, paths):
    variants=[];first=None
    for path in paths:
        doc=_read(path)
        require(doc.get('schema_version')==COLLECTION and doc['task']==task.to_dict(),
                'collection task/schema mismatch',path='wan.plan')
        require(doc['evidence_scope'] in {'MEASURED','CPU_FIXTURE_ONLY'},'unknown cost evidence',path='wan.plan')
        if doc['evidence_scope']=='MEASURED':
            require(doc['identity']['source_fingerprint']==source_fingerprint(),'source changed: recalibrate',path='wan.plan')
        if first is None:first=doc
        else:
            require(all(doc[k]==first[k] for k in ('identity','work','warm_initial_versions','evidence_scope')),
                    'incompatible calibration cohorts',path='wan.plan')
        for v in doc['variants']:
            candidate=WanStrategy.from_dict(v['strategy']);candidate.validate_for(task)
            require(candidate.rollout_window==1,'costs must use reference retention',path='wan.plan')
            if doc['evidence_scope']=='MEASURED':
                for source in v['rank_sources']:
                    require(file_sha256(Path(source['path']))==source['sha256'],'native rank receipt changed',path='wan.plan')
            require(file_sha256(Path(v['cost_model']))==v['cost_sha256'],'cost bytes changed',path='wan.plan')
            costs=CostModel.from_dict(_read(v['cost_model']));costs.verify_receipts()
            require(costs.scope==doc['evidence_scope'],'fixture costs cannot be used as GPU costs',path='wan.plan')
            variants.append((candidate,v,costs))
    require(first is not None and variants,'no observed candidates',path='wan.plan')
    require(len({s.backend.id for s,_,_ in variants})==len(variants),'duplicate implementation measurements',path='wan.plan')
    return first,variants


def plan(task, paths, *, max_states=50000,beam_width=0,timeout_seconds=60,memory_limit_bytes=None):
    doc,variants=load_collections(task,paths)
    graph=build_parameterized_cycle_graph(source_steps=task.source_steps,
        initial_versions=doc['warm_initial_versions'],ema_after=task.ema_after)
    ranks=tuple(range(16));ranked=[];rejected=[]
    for strategy,v,costs in variants:
        limit=v['memory_limit_bytes']
        if memory_limit_bytes is not None:
            require(type(memory_limit_bytes) is int and memory_limit_bytes>0,'invalid memory limit',path='wan.plan')
            limit=min(limit,memory_limit_bytes)
        rules={}
        for op in graph.operations:
            contract=component(task,op.operation_id,strategy,cycle=2,work=doc['work'])
            costs.get_component(contract)
            rules[op.operation_id]=BindingRule(op.operation_id,{},contract.digest,ranks,
                tuple(f'gpu:{r}' for r in ranks),('state',),contract.digest)
        lives=tuple(Lifetime(f'output-r{i}',f'r{i}',(f'c{i}',),ranks,v['output_bytes'][f'r{i}'],'producer_exit')
                    for i in range(task.fake_updates))
        state=(StateAllocation('state',ranks,v['static_bytes'],('native_runtime',),False,'original-native-state'),)
        result=schedule_frontier(graph,rules,costs,states=state,lifetimes=lives,
            memory_limits={r:limit for r in ranks},max_states=max_states,beam_width=beam_width,
            timeout_seconds=timeout_seconds)
        if result['status']!='PASS_CPU_SCHEDULE':
            rejected.append({'strategy':strategy.to_dict(),'schedule':result});continue
        live=peak=0
        for key in result['order']:
            live+=1 if key.startswith('r') else (-1 if key.startswith('c') else 0);peak=max(peak,live)
        result['strategy']=replace(strategy,rollout_window=peak).to_dict()
        result['model_binding']=dict(costs.binding)
        ranked.append(result)
    require(ranked,'no feasible plan within observed candidates and search budget',path='wan.plan')
    ranked.sort(key=lambda x:(x['predicted_makespan_ns'],max(r['peak_bytes'] for r in x['memory']),fingerprint(x['strategy'])))
    measured={s.backend.id for s,_,_ in variants}
    missing=[s.to_dict() for s in implementations(task,triton_available=True) if s.backend.id not in measured]
    result={'schema_version':PLAN,'task':task.to_dict(),'identity':doc['identity'],
        'collections':[{'path':str(Path(p).resolve()),'sha256':file_sha256(Path(p))} for p in paths],
        'selected':ranked[0],'ranked':ranked,'rejected':rejected,'uncalibrated':missing,
        'algorithm':'layout_enumeration_then_event_frontier_DP_or_beam',
        'evidence_scope':doc['evidence_scope'],'status':'PLAN_READY_FOR_ORIGINAL_NATIVE_WORKER',
        'native_lowering':'same OriginalWanSession and old Wan handlers',
        'gpu_execution':'NOT_RUN_BY_PLANNER','claimable_speedup':False}
    result['plan_sha256']=fingerprint(result)
    return result


def validate_plan(task, path):
    result=_read(path);unsigned=dict(result);sha=unsigned.pop('plan_sha256',None)
    require(result.get('schema_version')==PLAN and sha==fingerprint(unsigned)
            and result['task']==task.to_dict(),'changed plan/task',path='wan.run')
    require(result['evidence_scope']=='MEASURED','CPU fixture plans cannot launch a GPU worker',path='wan.run')
    paths=[]
    for item in result['collections']:
        p=Path(item['path']);require(file_sha256(p)==item['sha256'],'collection changed',path='wan.run');paths.append(p)
    doc,variants=load_collections(task,paths)
    require(result['identity']==doc['identity'],'plan identity changed',path='wan.run')
    selected=result['selected'];strategy=WanStrategy.from_dict(selected['strategy']);strategy.validate_for(task)
    matches=[(v,costs) for s,v,costs in variants if s.backend==strategy.backend]
    require(len(matches)==1,'selected backend not calibrated',path='wan.run')
    v,costs=matches[0]
    require(selected['cost_model_sha256']==costs.digest and selected['model_binding']==dict(costs.binding),
            'selected cost model changed',path='wan.run')
    graph=build_parameterized_cycle_graph(source_steps=task.source_steps,
        initial_versions=doc['warm_initial_versions'],ema_after=task.ema_after)
    from distillschedule.runtime.role_runner import StaticRoleRunner
    from distillschedule.runtime.version_sync import VersionLedger
    handlers={op.kind.value:lambda ctx:None for op in graph.operations}
    StaticRoleRunner('native_runtime',handlers,versions=VersionLedger(doc['warm_initial_versions'])).prepare(
        graph,order=tuple(selected['order']))
    require(selected['graph_sha256']==fingerprint(graph.to_dict()),'selected graph changed',path='wan.run')
    # Independently recompute the serial full-world schedule, not just its hash.
    live={};peak=v['static_bytes'];elapsed=0;peak_count=0;timeline=[]
    for key in selected['order']:
        c=costs.get_component(component(task,key,strategy,cycle=2,work=doc['work']))
        peak=max(peak,v['static_bytes']+sum(live.values())+c.workspace_bytes_per_rank)
        timeline.append({'operation':key,'start_ns':elapsed,'end_ns':elapsed+c.duration_ns,
                         'resources':[f'gpu:{r}' for r in range(16)]})
        elapsed+=c.duration_ns
        if key.startswith('r'):live[key]=v['output_bytes'][key]
        elif key.startswith('c'):live.pop('r'+key[1:])
        peak=max(peak,v['static_bytes']+sum(live.values()));peak_count=max(peak_count,len(live))
    require(not live and peak_count==strategy.rollout_window and elapsed==selected['predicted_makespan_ns'],
            'selected sequence does not reconcile with costs/lifetimes',path='wan.run')
    # Resource labels are sorted lexically by the frontier (0,1,10,...,2).
    for row in timeline:row['resources'].sort()
    require(timeline==selected['timeline'],'selected timeline changed',path='wan.run')
    require(len(selected['memory'])==16 and {r['rank'] for r in selected['memory']}==set(range(16))
            and all(r['static_bytes']==v['static_bytes'] and r['peak_bytes']==peak
                    and peak<=r['limit_bytes']<=v['memory_limit_bytes'] for r in selected['memory']),
            'selected capacity estimate changed',path='wan.run')
    return result
