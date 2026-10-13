"""Common bound-case planner for arbitrary graphs and existing native adapters.

A case is a lowering-supported physical structure, not a model name or winner.
The existing event-frontier solver owns scheduling and memory feasibility.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Iterable, Mapping

from distillschedule.core.contracts import require
from distillschedule.core.graph import ExecutionGraph
from distillschedule.search.cost_model import CostModel
from distillschedule.search.frontier_schedule import schedule_frontier
from distillschedule.search.plan_ir import BindingRule, Lifetime, StateAllocation
from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import json_copy


@dataclass(frozen=True)
class BoundCase:
    key: str
    graph: ExecutionGraph
    bindings: Mapping[str, BindingRule]
    costs: CostModel
    states: tuple[StateAllocation, ...]
    lifetimes: tuple[Lifetime, ...]
    memory_limits: Mapping[int, int]
    selection: Mapping[str, Any]

    @property
    def digest(self):
        return fingerprint({"key":self.key,"graph":self.graph.to_dict(),
            "bindings":{k:asdict(v) for k,v in self.bindings.items()},
            "costs":self.costs.digest,"states":[asdict(s) for s in self.states],
            "lifetimes":[asdict(s) for s in self.lifetimes],
            "memory_limits":dict(self.memory_limits),"selection":json_copy(self.selection)})


def solve_cases(cases: Iterable[BoundCase], *, max_cases=256, max_states=50000,
                beam_width=0, timeout_seconds=60, total_timeout_seconds=600):
    """Enumerate admitted physical structures; schedule every retained structure.

    Independent adapter/capability layers own lowering. No layout is committed
    before lifecycle/resource scheduling. A budget cutoff is not infeasibility
    and does not imply optimality across the unvisited cases.
    """
    require(type(max_cases) is int and 1<=max_cases<=100000,"invalid case budget",path="program.solve")
    require(type(total_timeout_seconds) in (int,float) and math.isfinite(total_timeout_seconds)
            and 0<total_timeout_seconds<=86400,"invalid total time budget",path="program.solve")
    started=time.monotonic();ranked=[];rejected=[];seen=set();complete=True
    for index,case in enumerate(cases):
        if index>=max_cases or time.monotonic()-started>=total_timeout_seconds:
            complete=False;break
        require(case.key not in seen,"duplicate physical case identity",path="program.solve")
        seen.add(case.key)
        case.costs.verify_receipts()
        remaining=total_timeout_seconds-(time.monotonic()-started)
        if remaining<=0:complete=False;break
        result=schedule_frontier(case.graph,case.bindings,case.costs,states=case.states,
            lifetimes=case.lifetimes,memory_limits=case.memory_limits,max_states=max_states,
            beam_width=beam_width,timeout_seconds=min(timeout_seconds,remaining))
        row={"case_key":case.key,"case_sha256":case.digest,
             "selection":json_copy(case.selection),"schedule":result}
        if result["status"]=="PASS_CPU_SCHEDULE":ranked.append(row)
        else:rejected.append(row)
    ranked.sort(key=lambda r:(r["schedule"]["predicted_makespan_ns"],
                max(v["peak_bytes"] for v in r["schedule"]["memory"]),r["case_key"]))
    return {"status":"PLAN_FOUND" if ranked else "NO_PLAN_FOUND",
            "ranked":ranked,"rejected":rejected,"cases_evaluated":len(seen),
            "case_enumeration_complete":complete,
            "search_seconds":time.monotonic()-started,
            "global_native_optimality":"NOT_CLAIMED"}
