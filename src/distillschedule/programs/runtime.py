"""Execute a selected graph through the existing PreparedRolePlan dispatcher.

Serial native calls must complete asynchronous device work before returning.
This lowerer refuses overlapping plans instead of silently serializing them.
Different physical executors may be registered by backend integrations.
"""
from __future__ import annotations
from dataclasses import replace
import time
from typing import Any, Mapping

from distillschedule.core.contracts import require
from distillschedule.core.graph import ExecutionGraph
from distillschedule.runtime.role_runner import StaticRoleRunner, OperationContext
from distillschedule.runtime.version_sync import VersionLedger
from .capabilities import ExecutableOption


def run_serial(program, selected: Mapping[str,ExecutableOption], *, order=None, observe=None):
    """Existing PreparedRolePlan owns input lifetimes and version commits.

    Normalize the DISPATCH role only; handlers still see the original logical
    role/operation. No model or autograd logic is rewritten by this lowering.
    """
    source={op.operation_id:op for op in program.graph.operations}
    require(set(selected)==set(source),"incomplete executable selection",path="program.run")
    graph=replace(program.graph,operations=tuple(replace(op,role="program_dispatch") for op in program.graph.operations))
    ledger=VersionLedger(graph.metadata.get("initial_versions",{}))
    timings=[]
    def invoke(context):
        key=context.operation.operation_id;option=selected[key]
        original=OperationContext(source[key],context.versions,context.values)
        option.synchronize()
        start=time.perf_counter_ns()
        result=option.invoke(original)
        option.synchronize()
        elapsed=time.perf_counter_ns()-start
        timings.append({"operation":key,"implementation":option.key,"duration_ns":elapsed})
        if observe is not None:observe(original,option,result,elapsed)
        return result
    handlers={op.kind.value:invoke for op in graph.operations}
    runner=StaticRoleRunner("program_dispatch",handlers,versions=ledger)
    chosen=tuple(order or graph.topological_order())
    leaves=tuple(op.operation_id for op in graph.operations
                 if not any(edge.source==op.operation_id for edge in graph.dependencies))
    prepared=runner.prepare(graph,order=chosen,retain_outputs=leaves)
    values=prepared.run()
    return {"outputs":values,"receipt":prepared.receipt(),"timings":timings}


def validate_serial_schedule(case, schedule):
    """Reconcile selected graph, timings/resources and storage before execution.

    Serial lowering deliberately refuses an overlapping plan. Backends able to
    execute multi-pool event graphs can bind a different executor; a plan never
    becomes executable merely because the CPU scheduler found it feasible.
    """
    from distillschedule.workloads.best_known.registry import fingerprint
    from distillschedule.search.plan_ir import validate_versions
    require(schedule.get("graph_sha256")==fingerprint(case.graph.to_dict())
            and schedule.get("cost_model_sha256")==case.costs.digest,
            "selected graph/cost identity differs",path="program.run")
    validate_versions(case.graph)
    keys={o.operation_id for o in case.graph.operations};order=schedule.get("order",[])
    require(len(order)==len(keys) and set(order)==keys,"selected order not a permutation",path="program.run")
    positions={k:i for i,k in enumerate(order)}
    require(all(positions[e.source]<positions[e.target] for e in case.graph.dependencies),
            "selected dependency order differs",path="program.run")
    elapsed=0;timeline=[]
    static={r:sum(s.bytes_per_rank for s in case.states
                  if r in s.ranks and any(s.id in b.states for b in case.bindings.values())) for r in case.memory_limits}
    peaks=dict(static);done=set()
    for key in order:
        b=case.bindings[key];cost=case.costs.get(b.cost_key)
        resources=set(b.resources)
        resources.update("adapter_scope:"+s.id for s in case.states if s.adapter_scoped and s.id in b.states)
        timeline.append({"operation":key,"start_ns":elapsed,"end_ns":elapsed+cost.duration_ns,
                         "resources":sorted(resources)})
        for phase in ("entry","exit"):
            completed=done | ({key} if phase=="exit" else set())
            live=dict(static)
            for lease in case.lifetimes:
                allocated=lease.producer in done or (lease.producer==key and
                    (lease.allocation_point=="producer_entry" or phase=="exit"))
                if allocated and not set(lease.consumers)<=completed:
                    for r in lease.ranks:live[r]+=lease.bytes_per_rank
            if phase=="entry":
                for r in b.ranks:live[r]+=cost.workspace_bytes_per_rank
            for r in live:peaks[r]=max(peaks[r],live[r])
        done.add(key);elapsed+=cost.duration_ns
    require(schedule.get("timeline")==timeline and schedule.get("predicted_makespan_ns")==elapsed,
            "serial executor cannot implement altered or overlapping schedule",path="program.run")
    expected=[{"rank":r,"static_bytes":static[r],"peak_bytes":peaks[r],"limit_bytes":case.memory_limits[r]}
              for r in sorted(case.memory_limits)]
    require(schedule.get("memory")==expected and all(peaks[r]<=case.memory_limits[r] for r in peaks),
            "selected memory does not reconcile",path="program.run")
