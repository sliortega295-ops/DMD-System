"""Shared bounded selection over backend-admitted candidates.

No model names, optimizer recipes, training math or GPU launch logic. Every
candidate must already have executable bindings and matched component costs.
Used by open programs and the original Wan adapter, not a second DP solver.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import time

from distillschedule.core.contracts import require
from distillschedule.core.graph import ExecutionGraph
from distillschedule.search.cost_model import CostModel
from distillschedule.search.frontier_schedule import schedule_frontier
from distillschedule.search.plan_ir import BindingRule, Lifetime, StateAllocation
from distillschedule.workloads.best_known.registry import fingerprint


@dataclass(frozen=True)
class BoundCandidate:
    id: str
    implementation: Mapping[str, Any]
    graph: ExecutionGraph
    rules: Mapping[str, BindingRule]
    costs: CostModel
    states: Sequence[StateAllocation]
    lifetimes: Sequence[Lifetime]
    memory_limits: Mapping[int, int]
    executable_backend: str

    def __post_init__(self):
        require(bool(self.id) and bool(self.executable_backend), 'candidate must have an executable backend', path='search.service')
        require(set(self.rules)=={o.operation_id for o in self.graph.operations},'incomplete action lowering',path='search.service')
        for rule in self.rules.values(): self.costs.get(rule.cost_key)
        fingerprint(dict(self.implementation))


def select(candidates, *, max_states=50000, beam_width=0, timeout_seconds=60, prune=True):
    require(candidates and len({c.id for c in candidates})==len(candidates),'nonempty unique candidates required',path='search.service')
    require(type(timeout_seconds) in (float,int) and 0<timeout_seconds<=600,'invalid total search budget',path='search.service')
    ranked=[];rejected=[];pending=[]
    started=time.monotonic()
    for candidate in candidates:
        remaining=timeout_seconds-(time.monotonic()-started)
        if remaining<=0:
            pending.append(candidate.id);continue
        c=candidate
        result=schedule_frontier(c.graph,c.rules,c.costs,states=c.states,lifetimes=c.lifetimes,
            memory_limits=c.memory_limits,max_states=max_states,beam_width=beam_width,
            timeout_seconds=remaining,prune=prune)
        row={**result,'candidate_id':c.id,'implementation':dict(c.implementation),
             'executable_backend':c.executable_backend}
        (ranked if result['status']=='PASS_CPU_SCHEDULE' else rejected).append(row)
    ranked.sort(key=lambda r:(r['predicted_makespan_ns'],max(x['peak_bytes'] for x in r['memory']),r['candidate_id']))
    return {'status':'PLAN_FOUND' if ranked else 'NO_ADMITTED_PLAN', 'ranked':ranked,'rejected':rejected,
            'pending':pending,'candidate_count':len(candidates),'search_wall_seconds':time.monotonic()-started,
            'scope':'backend_admitted_discrete_candidates','gpu_execution':'NOT_RUN_BY_SEARCH'}
