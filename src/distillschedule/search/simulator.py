"""Dependency/resource list scheduling and per-rank live-memory accounting.

An operation cost is a calibrated envelope, including its internal overlap.
Only disjoint declared resources overlap here. Unsupported resource sharing is
not granted a hypothetical free speedup. All output times are predictions.
"""
from __future__ import annotations

from typing import Any, Mapping

from .contracts import require
from .cost_model import CostModel
from .plan_ir import PlanningProblem


def simulate(problem: PlanningProblem, assignment: Mapping[str, str], costs: CostModel) -> dict[str, Any]:
    costs.bind(problem.binding, problem_scope=problem.evidence_scope)
    graph, bindings = problem.resolve(assignment)
    operations = {o.operation_id: o for o in graph.operations}
    order = {key: i for i, key in enumerate(graph.topological_order())}
    parents: dict[str, set[str]] = {key: set() for key in operations}
    for edge in graph.dependencies:
        parents[edge.target].add(edge.source)
    states = {s.id: s for s in problem.states}
    allocation_ids = {sid for r in bindings.values() for sid in r.states}
    static = {rank: 0 for rank in problem.memory_limits}
    for sid in allocation_ids:
        for rank in states[sid].ranks:
            static[rank] += states[sid].bytes_per_rank

    resource_last: dict[str, tuple[int, str]] = {}
    done: dict[str, dict[str, Any]] = {}
    pending = set(operations)
    peak_events: dict[int, list[tuple[int, int, str]]] = {r: [] for r in static}
    while pending:
        ready = []
        for key in pending:
            if not parents[key] <= done.keys():
                continue
            rule = bindings[key]
            resources = set(rule.resources)
            # Adapter state cannot be switched concurrently during an in-flight
            # original forward or checkpoint replay on a shared physical object.
            resources.update("adapter_scope:" + sid for sid in rule.states if states[sid].adapter_scoped)
            blockers = [(done[p]["end_ns"], p, "dependency") for p in parents[key]]
            blockers += [(resource_last[r][0], resource_last[r][1], "resource:" + r)
                         for r in sorted(resources) if r in resource_last]
            start = max((x[0] for x in blockers), default=0)
            ready.append((start, order[key], key, resources, blockers))
        require(bool(ready), "unschedulable graph")
        start, _, key, resources, blockers = min(ready, key=lambda x: (x[0], x[1]))
        rule, op = bindings[key], operations[key]
        cost = costs.get(rule.cost_key)
        if costs.scope == "MEASURED":
            require(rule.operation_contract_sha256 == cost.evidence["operation_contract_sha256"],
                    "operation calibration contract does not match selected binding")
        end = start + cost.duration_ns
        done[key] = {"operation": key, "role": op.role, "cycle": op.metadata.get("cycle"),
                     "kind": op.kind.value, "ranks": list(rule.ranks), "resources": sorted(resources),
                     "start_ns": start, "end_ns": end, "duration_ns": cost.duration_ns,
                     "cost_key": rule.cost_key,
                     "critical_predecessors": [{"operation": p, "reason": reason}
                         for when, p, reason in blockers if when == start and start > 0]}
        for resource in resources:
            resource_last[resource] = (end, key)
        if end > start:
            for rank in rule.ranks:
                peak_events[rank].extend([(start, cost.workspace_bytes_per_rank, key + ":workspace"),
                                          (end, -cost.workspace_bytes_per_rank, key + ":workspace")])
        pending.remove(key)

    for value in problem.lifetimes:
        # Legacy costs exclude retained outputs. Native action calibration can
        # instead include output creation in producer workspace and start the
        # separate retained lifetime at return, without double counting.
        start = done[value.producer]["end_ns" if value.allocation_point == "producer_exit" else "start_ns"]
        end = max(done[c]["end_ns"] for c in value.consumers)
        if end > start:
            for rank in value.ranks:
                peak_events[rank].extend([(start, value.bytes_per_rank, value.id),
                                          (end, -value.bytes_per_rank, value.id)])
    memory = []
    for rank, events in peak_events.items():
        live, peak, peak_at = static[rank], static[rank], 0
        # Intervals are half-open: release before allocate at a common boundary.
        for when, delta, _ in sorted(events, key=lambda e: (e[0], e[1])):
            live += delta
            require(live >= static[rank], "memory lifetime reconciliation failed")
            if live > peak:
                peak, peak_at = live, when
        require(live == static[rank], "tensor/workspace memory not released")
        memory.append({"rank": rank, "static_bytes": static[rank], "peak_bytes": peak,
                       "peak_at_ns": peak_at, "limit_bytes": problem.memory_limits[rank],
                       "predicted_headroom_bytes": problem.memory_limits[rank] - peak})
    wall = max(r["end_ns"] for r in done.values())
    cycle_ends = graph.metadata.get("cycle_ends", {})
    require(set(cycle_ends) == {"1", "2"}, "two-cycle completion boundaries required")
    require(all(k in done for k in cycle_ends.values()), "unknown cycle completion operation")
    c1, c2 = (done[cycle_ends[str(i)]]["end_ns"] for i in (1, 2))
    require(0 < c1 < c2 == wall, "invalid two-cycle/drain timer boundary")
    for key, row in done.items():
        require(row["cycle"] in {1, 2}, "operation missing cycle identity")
        require(row["end_ns"] <= (c1 if row["cycle"] == 1 else c2), "cycle completion precedes its work")
    early = [row for row in done.values() if row["cycle"] == 2 and row["start_ns"] < c1]
    return {"status": "PREDICTED_FEASIBLE" if all(r["predicted_headroom_bytes"] >= 0 for r in memory)
            else "PREDICTED_CAPACITY_FAIL", "evidence_scope": problem.evidence_scope,
            "gpu_execution": "NOT_RUN", "claimable_speedup": False,
            "problem_sha256": problem.digest, "cost_model_sha256": costs.digest,
            "assignment": dict(assignment), "predicted_cycle1_ns": c1,
            "predicted_cycle2_ns": c2 - c1, "predicted_total_ns": wall,
            "cycle2_work_started_before_cycle1_commit": [r["operation"] for r in early],
            "cycle2_early_activity_sum_ns_not_wall": sum(min(c1, r["end_ns"]) - r["start_ns"] for r in early),
            "timeline": sorted(done.values(), key=lambda r: (r["start_ns"], order[r["operation"]])),
            "memory": memory, "memory_reconciliation": "PASS",
            "critical_path_scope": "MODEL_PREDICTION_NOT_NSYS_DEPENDENCY_PROOF"}
