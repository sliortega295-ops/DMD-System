"""Completion-fenced execution of an explicitly bound event plan.

The executor does not sleep for predicted durations. It preserves every modeled
completion-before-start relation, checks resource leases and declared memory at
runtime, and waits for each native completion fence. This is single-process
host dispatch; collective/distributed workers require their own bound actions.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import time

from distillschedule.core.contracts import require
from distillschedule.runtime.role_runner import OperationContext
from distillschedule.runtime.version_sync import VersionLedger
from distillschedule.search.plan_ir import validate_versions
from distillschedule.workloads.best_known.registry import fingerprint


def validate_event_schedule(case,schedule):
    validate_versions(case.graph)
    require(schedule.get('graph_sha256')==fingerprint(case.graph.to_dict())
            and schedule.get('cost_model_sha256')==case.costs.digest,
            'event plan graph/cost mismatch',path='program.events')
    rows=schedule.get('timeline',[]);keys={op.operation_id for op in case.graph.operations}
    require(len(rows)==len(keys) and {r['operation'] for r in rows}==keys
            and schedule.get('order')==[r['operation'] for r in rows],
            'event plan must execute every action exactly once',path='program.events')
    entries={r['operation']:r for r in rows};claims={};durations={}
    for row in rows:
        key=row['operation'];binding=case.bindings[key];cost=case.costs.get(binding.cost_key)
        require(type(row['start_ns']) is int and row['start_ns']>=0
                and type(row['end_ns']) is int and row['end_ns']-row['start_ns']==cost.duration_ns
                and cost.duration_ns>0,'invalid event duration',path='program.events')
        claim=set(binding.resources)
        claim.update('adapter_scope:'+s.id for s in case.states if s.adapter_scoped and s.id in binding.states)
        require(row['resources']==sorted(claim),'event resources differ from binding',path='program.events')
        claims[key]=claim;durations[key]=cost
    require([r['start_ns'] for r in rows]==sorted(r['start_ns'] for r in rows),
            'event dispatch order is not chronological',path='program.events')
    require(all(entries[e.source]['end_ns']<=entries[e.target]['start_ns'] for e in case.graph.dependencies),
            'event plan violates a data/version dependency',path='program.events')
    for i,left in enumerate(rows):
        for right in rows[i+1:]:
            overlap=max(left['start_ns'],right['start_ns'])<min(left['end_ns'],right['end_ns'])
            require(not overlap or not claims[left['operation']]&claims[right['operation']],
                    'event plan overlaps an exclusive resource',path='program.events')
    require(all(v.allocation_point=='producer_entry' for v in case.lifetimes),
            'parallel event executor requires producer-entry output reservations',path='program.events')
    static={r:sum(s.bytes_per_rank for s in case.states if r in s.ranks
                and any(s.id in b.states for b in case.bindings.values())) for r in case.memory_limits}
    def footprint(started,done,running):
        memory=dict(static)
        for key in running:
            for rank in case.bindings[key].ranks:memory[rank]+=durations[key].workspace_bytes_per_rank
        for lease in case.lifetimes:
            if lease.producer in started and not set(lease.consumers)<=done:
                for rank in lease.ranks:memory[rank]+=lease.bytes_per_rank
        return memory
    peaks=dict(static)
    for now in sorted({r[k] for r in rows for k in ('start_ns','end_ns')}):
        started={r['operation'] for r in rows if r['start_ns']<=now}
        done={r['operation'] for r in rows if r['end_ns']<=now}
        memory=footprint(started,done,started-done)
        for rank in memory:peaks[rank]=max(peaks[rank],memory[rank])
    expected=[{'rank':r,'static_bytes':static[r],'peak_bytes':peaks[r],'limit_bytes':case.memory_limits[r]}
              for r in sorted(case.memory_limits)]
    require(schedule.get('memory')==expected and all(peaks[r]<=case.memory_limits[r] for r in peaks)
            and schedule.get('predicted_makespan_ns')==max(r['end_ns'] for r in rows),
            'event memory or makespan does not reconcile',path='program.events')
    fences={key:{other for other,row in entries.items()
                 if row['end_ns']<=entries[key]['start_ns']} for key in keys}
    for edge in case.graph.dependencies:fences[edge.target].add(edge.source)
    return claims,fences,footprint


def run_events(program,selected,case,schedule,*,max_workers=32):
    require(type(max_workers) is int and 1<=max_workers<=128,'invalid worker limit',path='program.events')
    claims,fences,footprint=validate_event_schedule(case,schedule)
    ops={o.operation_id:o for o in program.graph.operations}
    require(set(selected)==set(ops),'incomplete event handlers',path='program.events')
    versions=VersionLedger(program.graph.metadata.get('initial_versions',{}))
    consumers={key:{o.operation_id for o in ops.values() if key in o.metadata.get('inputs',())} for key in ops}
    leaves={k for k in ops if not any(e.source==k for e in program.graph.dependencies)}
    started=set();done=set();active={};values={};timings=[];executed=[]
    def call(key,context):
        option=selected[key];option.synchronize();begin=time.perf_counter_ns()
        result=option.invoke(context);option.synchronize()
        return result,{'operation':key,'implementation':option.key,'start_ns':begin,
                       'end_ns':time.perf_counter_ns()}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while len(done)<len(ops):
            occupied=set().union(*(claims[k] for k in active.values())) if active else set()
            for key in schedule['order']:
                if key in started or not fences[key]<=done or claims[key]&occupied:continue
                running=set(active.values())|{key}
                memory=footprint(started|{key},done,running)
                if any(memory[r]>case.memory_limits[r] for r in memory):continue
                op=ops[key]
                for component,version in op.required_versions.items():versions.require(component,version)
                inputs={k:values[k] for k in op.metadata.get('inputs',())}
                context=OperationContext(op,VersionLedger(versions.snapshot().values),inputs)
                future=pool.submit(call,key,context)
                active[future]=key;started.add(key);occupied.update(claims[key])
            require(active,'runtime event plan is blocked by dependencies or capacity',path='program.events')
            finished,_=wait(tuple(active),return_when=FIRST_COMPLETED)
            for future in finished:
                key=active.pop(future);result,timing=future.result();op=ops[key]
                for component,version in op.required_versions.items():versions.require(component,version)
                for component,version in op.produced_versions.items():versions.set(component,version)
                values[key]=result;done.add(key);executed.append(key);timings.append(timing)
            for key in list(values):
                if key not in leaves and consumers[key]<=done:del values[key]
    final=dict(versions.snapshot().values)
    require(final==dict(program.graph.metadata.get('final_versions',final)),
            'native final versions differ',path='program.events')
    return {'outputs':values,'timings':timings,'receipt':{'status':'PASS',
            'executed_operations':executed,'final_versions':final,
            'lowering':'completion_fences_not_predicted_wallclock_sleeps',
            'native_correctness':'REQUIRES_BOUND_FUNCTION_CONTRACTS'}}
