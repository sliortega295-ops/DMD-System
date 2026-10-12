"""Event-frontier DP/beam scheduling of a fixed, explicitly bound operation DAG.

This is the finer-grained CPU solver, not permission to send arbitrary event
orders to existing model workers. Current native adapters still admit only
their declared orders. All numerical costs and physical state effects must be
supplied; the solver does not infer gradient or cache correctness from names.

The finite search family permits dispatch at zero or a preceding completion
only. This is not an unrestricted continuous-time scheduling oracle: with
producer-exit allocations, an off-event delayed start can be better. Tasks have
no external release dates, dynamic branches or preemption. Explicit wait
transitions are essential even when work is ready.
The DP state retains completed work and remaining running tasks (including
resource/adapter leases); live tensor storage and versions are determined by
that state and the validated DAG. Equal bytes alone never merge two states.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
import time
from typing import Any, Mapping, Sequence

from distillschedule.core.enums import OperationKind
from distillschedule.core.graph import ExecutionGraph
from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import require
from .cost_model import CostModel
from .plan_ir import BindingRule, Lifetime, StateAllocation, ancestors, validate_versions


@dataclass(frozen=True)
class _Label:
    now: int
    done: int
    running: tuple[tuple[int, int], ...]
    peaks: tuple[int, ...]
    history: tuple[tuple[int, int, int], ...]

    @property
    def state(self) -> tuple:
        return (self.done, tuple((i, end - self.now) for i, end in self.running))

    @property
    def value(self) -> tuple:
        return self.now, self.peaks


def schedule_frontier(graph: ExecutionGraph, bindings: Mapping[str, BindingRule],
                      costs: CostModel, *, states: Sequence[StateAllocation] = (),
                      lifetimes: Sequence[Lifetime] = (), memory_limits: Mapping[int, int],
                      max_states: int = 50000, beam_width: int = 0,
                      timeout_seconds: float = 60, prune: bool = True,
                      retained_limits: Mapping[int, int] | None = None) -> dict[str, Any]:
    """Minimize one bound DAG's makespan; ties retain lower peak-memory labels.

    Costs/resource reservations include internal compute/communication overlap.
    Between-task overlap is possible ONLY on disjoint declared resources.
    Different layouts/sharing/bindings are separate outer-search problems.
    Native numerical correctness and measured speedup are never inferred here.
    """
    require(type(max_states) is int and 1 <= max_states <= 1000000, "invalid state budget")
    require(type(beam_width) is int and 0 <= beam_width <= max_states, "invalid beam width")
    require(type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds)
            and 0 < timeout_seconds <= 600, "invalid timeout")
    require(type(prune) is bool, "prune must be boolean")
    require(memory_limits and all(type(r) is int and r >= 0 and type(b) is int and b > 0
                                 for r, b in memory_limits.items()), "invalid memory limits")
    retained_limits = {} if retained_limits is None else dict(retained_limits)
    require(set(retained_limits) <= set(memory_limits) and
            all(type(b) is int and b >= 0 for b in retained_limits.values()), "invalid retained limits")
    ops = graph.operations
    require(0 < len(ops) <= 512, "frontier solver requires 1..512 explicit operations")
    require(all(o.condition is None for o in ops), "resolve branch predicates before scheduling")
    validate_versions(graph)
    ids = {o.operation_id: i for i, o in enumerate(ops)}
    require(set(bindings) == set(ids), "each operation requires one explicit binding")
    before = ancestors(graph)
    ranks = tuple(sorted(memory_limits))
    state_map = {s.id: s for s in states}
    require(len(state_map) == len(states), "duplicate physical storage identity")
    for s in states:
        require(s.ranks and len(set(s.ranks)) == len(s.ranks) and set(s.ranks) <= set(ranks), "invalid state ranks")
        require(type(s.bytes_per_rank) is int and s.bytes_per_rank >= 0, "invalid storage bytes")
        require(s.roles and (len(s.roles) == 1 or s.frozen), "mutable role state cannot be aliased")
    parents = [0 for _ in ops]
    for e in graph.dependencies:
        parents[ids[e.target]] |= 1 << ids[e.source]
    durations, scratch, resources = [], [], []
    used_storage = set()
    for op in ops:
        b = bindings[op.operation_id]
        require(b.operation == op.operation_id and not b.when, "bindings must be fully resolved (empty when)")
        require(b.ranks and len(set(b.ranks)) == len(b.ranks) and set(b.ranks) <= set(ranks), "invalid execution ranks")
        require(b.resources and len(set(b.resources)) == len(b.resources), "invalid resource reservations")
        if op.kind not in {OperationKind.VERSION_SYNC, OperationKind.BARRIER}:
            require({f"gpu:{r}" for r in b.ranks} <= set(b.resources), "model work must reserve its GPUs")
        require(set(b.states) <= set(state_map), "unknown physical storage")
        claim = set(b.resources)
        for sid in b.states:
            s = state_map[sid]
            require(set(b.ranks) <= set(s.ranks) and op.role in s.roles, "state unavailable or unauthorized")
            if s.adapter_scoped:
                require(bool(op.adapter_mode), "shared adapter operation needs explicit role context")
                claim.add("adapter_scope:" + sid)
        used_storage.update(b.states)
        c = costs.get(b.cost_key)
        if costs.scope == "MEASURED":
            require(b.operation_contract_sha256 == c.evidence.get("operation_contract_sha256"),
                    "operation calibration contract mismatch")
        durations.append(c.duration_ns)
        scratch.append(tuple(c.workspace_bytes_per_rank if r in b.ranks else 0 for r in ranks))
        resources.append(frozenset(claim))
    static = tuple(sum(state_map[s].bytes_per_rank for s in used_storage if r in state_map[s].ranks)
                   for r in ranks)
    limit = tuple(memory_limits[r] for r in ranks)
    require(len({v.id for v in lifetimes}) == len(lifetimes), "duplicate retained storage identity")
    leases = []
    for v in lifetimes:
        require(v.producer in ids and v.consumers and all(c in ids for c in v.consumers), "unknown lifetime endpoint")
        require(all(v.producer in before[c] for c in v.consumers), "retained consumer lacks producer dependency")
        require(v.ranks and len(set(v.ranks)) == len(v.ranks) and set(v.ranks) <= set(ranks), "invalid lifetime ranks")
        require(type(v.bytes_per_rank) is int and v.bytes_per_rank >= 0, "invalid lifetime bytes")
        require(v.allocation_point in {"producer_entry", "producer_exit"}, "invalid allocation boundary")
        leases.append((1 << ids[v.producer], sum(1 << ids[c] for c in set(v.consumers)),
                       tuple(v.bytes_per_rank if r in v.ranks else 0 for r in ranks),
                       v.allocation_point == "producer_entry"))
    order = tuple(ids[k] for k in graph.topological_order())
    full = (1 << len(ops)) - 1
    stats = {k: 0 for k in ("expanded", "generated", "dominance_pruned", "capacity_pruned",
                            "bound_pruned", "beam_dropped", "wait_transitions")}
    started = time.monotonic()
    counter = itertools.count()
    queue = []
    memo: dict[tuple, list[tuple]] = {}
    best: _Label | None = None
    admission_truncated = False

    def lower_bound(label: _Label) -> int:
        running = dict(label.running)
        residual = [0 if label.done & (1 << i) else
                    running[i] - label.now if i in running else durations[i] for i in range(len(ops))]
        finish = [0 for _ in ops]
        for i in order:
            if not label.done & (1 << i):
                finish[i] = residual[i] + max((finish[j] for j in range(len(ops))
                                               if parents[i] & (1 << j)), default=0)
        resource_work: dict[str, int] = {}
        for i, duration in enumerate(residual):
            for resource in resources[i]:
                resource_work[resource] = resource_work.get(resource, 0) + duration
        return label.now + max(max(finish, default=0), max(resource_work.values(), default=0))

    def admit(label: _Label) -> None:
        nonlocal best, admission_truncated
        if stats["generated"] >= max_states:
            admission_truncated = True
            return
        stats["generated"] += 1
        done = label.done
        running = []
        for i, end in label.running:
            if end <= label.now:
                done |= 1 << i
            else:
                running.append((i, end))
        running = tuple(sorted(running))
        active = done
        live = list(static)
        retained_live = [0 for _ in ranks]
        for i, _ in running:
            active |= 1 << i
            live = [a + b for a, b in zip(live, scratch[i])]
        for producer, consumers, size, entry in leases:
            if ((active if entry else done) & producer) and done & consumers != consumers:
                live = [a + b for a, b in zip(live, size)]
                retained_live = [a + b for a, b in zip(retained_live, size)]
        if any(a > b for a, b in zip(live, limit)) or any(
                retained_live[i] > retained_limits[r] for i, r in enumerate(ranks) if r in retained_limits):
            stats["capacity_pruned"] += 1
            return
        label = _Label(label.now, done, running, tuple(max(a, b) for a, b in zip(label.peaks, live)), label.history)
        if done == full:
            if best is None or (label.now, label.peaks) < (best.now, best.peaks):
                best = label
            return
        bound = lower_bound(label)
        if prune and best is not None and bound > best.now:
            stats["bound_pruned"] += 1
            return
        labels = memo.setdefault(label.state, [])
        if prune:
            if any(t <= label.now and all(a <= b for a, b in zip(peak, label.peaks)) for t, peak in labels):
                stats["dominance_pruned"] += 1
                return
            labels[:] = [(t, peak) for t, peak in labels
                         if not (label.now <= t and all(a <= b for a, b in zip(label.peaks, peak)))]
        labels.append(label.value)
        heapq.heappush(queue, (bound, next(counter), label))

    admit(_Label(0, 0, (), static, ()))
    stop = "COMPLETE"
    while queue:
        if stats["expanded"] >= max_states:
            stop = "STATE_BUDGET"
            break
        if time.monotonic() - started >= timeout_seconds:
            stop = "TIME_BUDGET"
            break
        bound, _, label = heapq.heappop(queue)
        if prune and label.value not in memo.get(label.state, ()):
            continue
        if prune and best is not None and bound > best.now:
            stats["bound_pruned"] += 1
            continue
        stats["expanded"] += 1
        active = label.done
        occupied = set()
        for i, _ in label.running:
            active |= 1 << i
            occupied.update(resources[i])
        for i in order:
            if time.monotonic() - started >= timeout_seconds:
                stop = "TIME_BUDGET"
                break
            if active & (1 << i) or parents[i] & label.done != parents[i] or resources[i] & occupied:
                continue
            end = label.now + durations[i]
            admit(_Label(label.now, label.done, (*label.running, (i, end)), label.peaks,
                         (*label.history, (i, label.now, end))))
        if stop != "COMPLETE":
            break
        # Intentional waiting is a genuine action, even when another task is
        # ready. Greedy non-delay scheduling can lose the optimal schedule.
        if label.running:
            stats["wait_transitions"] += 1
            admit(_Label(min(end for _, end in label.running), label.done, label.running,
                         label.peaks, label.history))
        if beam_width and len(queue) > beam_width:
            stats["beam_dropped"] += len(queue) - beam_width
            queue = heapq.nsmallest(beam_width, queue)
            heapq.heapify(queue)
    if admission_truncated and stop == "COMPLETE":
        stop = "GENERATED_STATE_BUDGET"
    complete = stop == "COMPLETE" and not stats["beam_dropped"]
    return {"schema_version": "distillschedule.frontier_schedule.v1",
            "status": "PASS_CPU_SCHEDULE" if best else
                      "NO_FEASIBLE_SCHEDULE" if complete else "NO_SCHEDULE_WITHIN_BUDGET",
            "graph_sha256": fingerprint(graph.to_dict()), "cost_model_sha256": costs.digest,
            "evidence_scope": costs.scope, "gpu_execution": "NOT_RUN", "claimable_speedup": False,
            "native_lowering": "REQUIRES_EXPLICIT_ACTION_ADAPTER",
            "predicted_makespan_ns": best.now if best else None,
            "order": [ops[i].operation_id for i, _, _ in best.history] if best else [],
            "timeline": [{"operation": ops[i].operation_id, "start_ns": begin, "end_ns": end,
                          "resources": sorted(resources[i])} for i, begin, end in best.history] if best else [],
            "memory": [{"rank": r, "static_bytes": static[j], "peak_bytes": best.peaks[j] if best else None,
                        "limit_bytes": limit[j]} for j, r in enumerate(ranks)],
            "search_diagnostics": {**stats, "termination": stop, "beam_width": beam_width,
                "remaining_frontier": len(queue), "search_seconds": time.monotonic() - started,
                "optimality": "FIXED_BINDING_EVENT_MODEL_OPTIMAL" if complete else "BEST_FOUND_ONLY",
                "dispatch_model": "ZERO_OR_COMPLETION_EVENTS_ONLY",
                "native_optimality": "NOT_CLAIMED"}}
