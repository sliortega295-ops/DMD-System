"""Hierarchical search over the existing *executable* adapter decision space.

Structural choices are expanded first, never fixed greedily.  The inner dynamic
program memoizes identical residual binding/constraint domains.  Final costs,
versions, resource overlap and tensor lifetimes still come from ``simulate``;
there is no second interpretation of native plans here.

Bounds are lower bounds of the declared deterministic simulator, not confidence
bounds on GPU time.  In particular, cycle2 = completion2 - completion1: a
whole-DAG makespan bound is UNSAFE when next-cycle work starts early.  Without
a proven cycle fence the cycle2 time bound is deliberately zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
import time
from typing import Any, Mapping, Sequence

from distillschedule.core.graph import Dependency, ExecutionGraph
from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import require
from .cost_model import CostModel, MissingCalibration
from .plan_ir import BindingRule, PlanningProblem, ancestors
from .simulator import simulate


STRUCTURAL_NAMES = ("placement", "mesh", "sharing", "pool", "layout", "execution")


def _compatible(condition: Mapping[str, str], partial: Mapping[str, str]) -> bool:
    return all(k not in partial or partial[k] == v for k, v in condition.items())


def _residual(condition: Mapping[str, str], partial: Mapping[str, str]) -> tuple:
    return tuple(sorted((k, v) for k, v in condition.items() if k not in partial))


@dataclass(frozen=True)
class PartialBound:
    cycle2_ns: int
    memory_bytes: tuple[tuple[int, int], ...]
    future_key: tuple
    reason: str | None = None
    fenced: bool = False


class _DomainModel:
    def __init__(self, problem: PlanningProblem, costs: CostModel) -> None:
        self.problem = problem
        self.costs = {r.key: r for r in costs.records}
        self.rules = {op.operation_id: [] for op in problem.graph.operations}
        for rule in problem.rules:
            self.rules[rule.operation].append(rule)
        self.states = {s.id: s for s in problem.states}
        self.rule_keys = {id(r): (r.operation, r.cost_key, r.ranks, r.resources,
                                  r.states, r.operation_contract_sha256)
                          for r in problem.rules}

    def analyze(self, partial: Mapping[str, str]) -> PartialBound:
        p = self.problem
        forbidden = []
        for row in p.forbidden:
            if _compatible(row, partial):
                rest = _residual(row, partial)
                if not rest:
                    return PartialBound(0, (), (), "FORBIDDEN")
                forbidden.append(rest)
        domains: dict[str, list[BindingRule]] = {}
        domain_key = []
        for op, rows in self.rules.items():
            candidates = [r for r in rows if _compatible(r.when, partial)]
            if not candidates:
                return PartialBound(0, (), (), "NO_IMPLEMENTATION")
            domains[op] = candidates
            # Preserve multiplicity: two simultaneously matching rules remain
            # ambiguous. Source/contract/state identity is part of the key.
            domain_key.append(tuple(sorted(
                ((_residual(r.when, partial), self.rule_keys[id(r)]) for r in candidates),
                key=repr)))
        edge_key, extra = [], []
        for index, row in enumerate(p.conditional_dependencies):
            if _compatible(row["when"], partial):
                rest = _residual(row["when"], partial)
                # Retain original edge identity/order for deterministic lowering.
                edge_key.append((index, rest))
                if not rest:
                    extra.append(Dependency(row["source"], row["target"], "plan_control"))
        key = (tuple(domain_key), tuple(forbidden), tuple(edge_key))
        try:
            graph = ExecutionGraph(p.graph.operations, p.graph.dependencies + tuple(extra), p.graph.metadata)
            before = ancestors(graph)
        except ValueError:
            return PartialBound(0, (), key, "DEPENDENCY_CYCLE")

        # Only storage present in *every* possible binding of at least one op
        # is unavoidable. Do not add per-role replicas for a shared allocation.
        forced = set()
        for rows in domains.values():
            forced.update(set.intersection(*(set(r.states) for r in rows)))
        static = {rank: sum(self.states[s].bytes_per_rank for s in forced
                            if rank in self.states[s].ranks) for rank in p.memory_limits}
        durations, resources = {}, {}
        peak_lb = dict(static)
        for op, rows in domains.items():
            records = [self.costs.get(r.cost_key) for r in rows]
            # Missing measurement is optimistic zero, NEVER infinity or a
            # historical speedup. It remains a calibration request in results.
            durations[op] = min(c.duration_ns if c else 0 for c in records)
            claims = []
            for r in rows:
                claims.append(set(r.resources) | {
                    "adapter_scope:" + s for s in r.states if self.states[s].adapter_scoped})
            resources[op] = set.intersection(*claims)
            for rank in peak_lb:
                scratch = min(c.workspace_bytes_per_rank if c and c.duration_ns > 0 and rank in r.ranks
                              else 0 for r, c in zip(rows, records))
                peak_lb[rank] = max(peak_lb[rank], static[rank] + scratch)
        memory = tuple(sorted(peak_lb.items()))
        if any(value > p.memory_limits[rank] for rank, value in memory):
            return PartialBound(0, memory, key, "UNAVOIDABLE_CAPACITY")

        c1 = graph.metadata.get("cycle_ends", {}).get("1")
        second = {o.operation_id for o in graph.operations if o.metadata.get("cycle") == 2}
        fenced = bool(second) and all(c1 in before[k] for k in second)
        if not fenced:
            return PartialBound(0, memory, key, fenced=False)
        parents = {k: [] for k in second}
        for e in graph.dependencies:
            if e.source in second and e.target in second:
                parents[e.target].append(e.source)
        finish = {}
        for k in graph.topological_order():
            if k in second:
                finish[k] = durations[k] + max((finish[a] for a in parents[k]), default=0)
        resource_work: dict[str, int] = {}
        for k in second:
            for resource in resources[k]:
                resource_work[resource] = resource_work.get(resource, 0) + durations[k]
        lower = max(max(finish.values(), default=0), max(resource_work.values(), default=0))
        return PartialBound(lower, memory, key, fenced=True)


def bound_partial(problem: PlanningProblem, costs: CostModel,
                  assignment: Mapping[str, str]) -> PartialBound:
    """Public CPU lower-bound hook for brute-force bound verification tests."""
    costs.bind(problem.binding, problem_scope=problem.evidence_scope)
    require(set(assignment) <= set(problem.decisions), "unknown partial decision")
    require(all(v in problem.decisions[k] for k, v in assignment.items()), "unknown partial option")
    return _DomainModel(problem, costs).analyze(assignment)


def search_hierarchical(problem: PlanningProblem, costs: CostModel, *,
                        max_proposals: int = 10000, timeout_seconds: float = 600,
                        beam_width: int = 0,
                        structural_decisions: Sequence[str] | None = None,
                        prune: bool = True) -> dict[str, Any]:
    """Search existing native decisions with residual-domain DP and safe bounds.

    ``beam_width=0`` imposes no heuristic frontier truncation. Exactness is only
    certified when the queue drains, budget/time limits did not truncate it,
    and all admitted bindings have costs. No human incumbent is required.
    The returned schema is consumed by the unchanged native lowerers/validator.
    """
    require(type(max_proposals) is int and 1 <= max_proposals <= 100000, "proposal budget must be 1..100000")
    require(type(beam_width) is int and 0 <= beam_width <= 100000, "invalid beam width")
    require(type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds)
            and 0 < timeout_seconds <= 600, "search deadline must be <=600 seconds")
    require(type(prune) is bool, "prune must be boolean")
    costs.bind(problem.binding, problem_scope=problem.evidence_scope)
    structural = (tuple(k for k in STRUCTURAL_NAMES if k in problem.decisions)
                  if structural_decisions is None else tuple(structural_decisions))
    require(len(set(structural)) == len(structural) and set(structural) <= set(problem.decisions),
            "invalid structural decision list")
    names = (*structural, *sorted(set(problem.decisions) - set(structural)))
    started = time.monotonic()
    model = _DomainModel(problem, costs)
    cache: dict[str, dict[str, Any]] = {}
    journal = []
    pruned_branches = []
    stats = {k: 0 for k in ("expanded", "generated", "equivalent_pruned", "contract_pruned",
                            "capacity_pruned", "bound_pruned", "beam_dropped", "unfenced_bounds")}
    best = math.inf

    def record_rejection(partial, bound):
        if len(pruned_branches) < 128:
            pruned_branches.append({"status": "PRUNED_CAPACITY_BOUND" if bound.reason == "UNAVOIDABLE_CAPACITY"
                                    else "PRUNED_CONTRACT", "partial_assignment": dict(partial),
                                    "reason": bound.reason, "gpu_execution": "NOT_RUN"})

    def evaluate(assignment: Mapping[str, str], reason: str) -> None:
        nonlocal best
        key = fingerprint(dict(assignment))
        if key not in cache:
            try:
                result = simulate(problem, assignment, costs)
            except MissingCalibration as error:
                result = {"status": "UNCALIBRATED", "detail": str(error), "assignment": dict(assignment)}
            except ValueError as error:
                result = {"status": "REJECTED_CONTRACT", "detail": str(error), "assignment": dict(assignment)}
            cache[key] = result
            if result["status"] == "PREDICTED_FEASIBLE":
                best = min(best, result["predicted_cycle2_ns"])
        journal.append({"proposal": len(journal), "assignment_sha256": key,
                        "reason": reason, "status": cache[key]["status"]})

    # Optional hints are evaluated once. They are not prerequisites or historical
    # costs; the ordinary queue can discover the first feasible assignment.
    for assignment in problem.seeds:
        if len(cache) >= max_proposals or time.monotonic() - started >= timeout_seconds:
            break
        evaluate(assignment, "OPTIONAL_SEED")
    counter = itertools.count()
    queue = [(0, next(counter), {}, None)]
    memo: set[tuple] = set()
    stop = "COMPLETE"
    while queue:
        if stats["expanded"] >= max_proposals:
            stop = "STATE_BUDGET"
            break
        if time.monotonic() - started >= timeout_seconds:
            stop = "TIME_BUDGET"
            break
        lower, _, partial, bound = heapq.heappop(queue)
        stats["expanded"] += 1
        if bound is None:
            bound = model.analyze(partial)
        if bound.reason:
            record_rejection(partial, bound)
            stats["capacity_pruned" if bound.reason == "UNAVOIDABLE_CAPACITY" else "contract_pruned"] += 1
            # Invalid states cannot be lowered even for --no-pruning ablations.
            continue
        if prune and bound.cycle2_ns > best:
            stats["bound_pruned"] += 1
            continue
        depth = len(partial)
        future = (depth, bound.future_key)
        if prune and future in memo:
            stats["equivalent_pruned"] += 1
            continue
        memo.add(future)
        if depth == len(names):
            if len(cache) >= max_proposals and fingerprint(partial) not in cache:
                stop = "EVALUATION_BUDGET"
                break
            evaluate(partial, "HIERARCHICAL_LEAF")
            continue
        name = names[depth]
        for value in problem.decisions[name]:
            if stats["generated"] >= max_proposals:
                stop = "GENERATED_STATE_BUDGET"
                break
            if time.monotonic() - started >= timeout_seconds:
                stop = "TIME_BUDGET"
                break
            child = {**partial, name: value}
            child_bound = model.analyze(child)
            stats["generated"] += 1
            if child_bound.reason:
                record_rejection(child, child_bound)
                stats["capacity_pruned" if child_bound.reason == "UNAVOIDABLE_CAPACITY" else "contract_pruned"] += 1
                continue
            if not child_bound.fenced:
                stats["unfenced_bounds"] += 1
            if prune and child_bound.cycle2_ns > best:
                stats["bound_pruned"] += 1
                continue
            heapq.heappush(queue, (child_bound.cycle2_ns, next(counter), child, child_bound))
        if stop != "COMPLETE":
            break
        if beam_width and len(queue) > beam_width:
            stats["beam_dropped"] += len(queue) - beam_width
            queue = heapq.nsmallest(beam_width, queue)
            heapq.heapify(queue)
    ranked = sorted((r for r in cache.values() if r["status"] == "PREDICTED_FEASIBLE"),
                    key=lambda r: (r["predicted_cycle2_ns"], r["predicted_total_ns"], fingerprint(r["assignment"])))
    missing = sorted({r.cost_key for r in problem.rules if r.cost_key not in model.costs})
    complete = stop == "COMPLETE" and not stats["beam_dropped"]
    return {"schema_version": "distillschedule.model_search_result.v1",
            "status": "PASS_CPU_SEARCH" if ranked else "NO_CALIBRATED_FEASIBLE_PLAN",
            "workload": problem.workload, "binding": dict(problem.binding),
            "problem_sha256": problem.digest, "cost_model_sha256": costs.digest,
            "algorithm": "hierarchical", "seed": None,
            "space_size": math.prod(len(problem.decisions[k]) for k in names),
            "unique_evaluated": len(cache), "search_seconds": time.monotonic() - started,
            "exhaustive": False, "evidence_scope": problem.evidence_scope,
            "gpu_execution": "NOT_RUN", "strict_winner": None, "claimable_speedup": False,
            "predicted_best": ranked[0] if ranked else None, "ranked": ranked,
            "calibration_requests": ["UNCALIBRATED: " + key for key in missing],
            "rejections": [r for r in cache.values() if r["status"] != "PREDICTED_FEASIBLE"] + pruned_branches,
            "journal": journal,
            "search_diagnostics": {**stats, "termination": stop, "remaining_frontier": len(queue),
                "decision_order": list(names), "structural_decisions": list(structural),
                "beam_width": beam_width, "safe_pruning": prune,
                "dp_scope": "EQUIVALENT_RESIDUAL_BINDING_DOMAINS_NOT_NATIVE_EVENT_REORDERING",
                "optimality": ("DISCRETE_SIMULATOR_OPTIMAL" if complete and not missing else
                               "CALIBRATED_SUBSET_OPTIMAL" if complete else "BEST_FOUND_ONLY"),
                "native_optimality": "NOT_CLAIMED",
                "bound_scope": "DECLARED_COST_MODEL_ONLY; EARLY_CYCLE2_USES_ZERO_TIME_BOUND"}}
