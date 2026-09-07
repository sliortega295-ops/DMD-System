"""CPU-only physical choices over the existing versioned ExecutionGraph.

This is a planning IR, not a second training executor. Adapters must explicitly
lower a selected plan before it can be admitted to native execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from distillschedule.core.enums import GradientMode, OperationKind
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation
from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import require

SCHEMA = "distillschedule.model_planning.v1"
BINDING_KEYS = ("source_head", "hardware_fingerprint", "environment_fingerprint",
                "asset_fingerprint", "semantic_fingerprint", "precision_fingerprint")


def nonnegative(value: Any, name: str) -> int:
    require(type(value) is int and value >= 0, f"{name}: expected nonnegative integer")
    return value


def validate_binding(binding: Mapping[str, str]) -> None:
    require(set(binding) == set(BINDING_KEYS), "incomplete planning identity (including precision/environment)")
    for key, size in [(k, 40 if k == "source_head" else 64) for k in BINDING_KEYS]:
        value = binding[key]
        require(isinstance(value, str) and len(value) == size
                and all(c in "0123456789abcdef" for c in value), f"invalid {key}")


def graph_from_dict(value: Mapping[str, Any]) -> ExecutionGraph:
    return ExecutionGraph(
        operations=tuple(Operation(
            operation_id=r["operation_id"], logical_step=r["logical_step"], role=r["role"],
            kind=OperationKind.parse(r["kind"], path="operation.kind"),
            grad_mode=GradientMode.parse(r["grad_mode"], path="operation.grad_mode"),
            physical_state=r.get("physical_state"), adapter_mode=r.get("adapter_mode"),
            required_versions=r.get("required_versions", {}),
            produced_versions=r.get("produced_versions", {}),
            condition=r.get("condition"), metadata=r.get("metadata", {}),
        ) for r in value["operations"]),
        dependencies=tuple(Dependency(**r) for r in value["dependencies"]),
        metadata=value.get("metadata", {}),
    )


def ancestors(graph: ExecutionGraph) -> dict[str, set[str]]:
    parents = {op.operation_id: set() for op in graph.operations}
    for e in graph.dependencies:
        parents[e.target].add(e.source)
    result: dict[str, set[str]] = {}
    for key in graph.topological_order():
        result[key] = set(parents[key])
        for parent in parents[key]:
            result[key].update(result[parent])
    return result


def validate_versions(graph: ExecutionGraph) -> None:
    """Check RAW and WAR edges, including readers held through recompute.

No implicit snapshotting: a write may not overtake any reader of the old
version. Adapters express checkpoint replay reads on the backward operation.
"""
    before = ancestors(graph)
    initial = graph.metadata.get("initial_versions", {})
    producers: dict[tuple[str, int], str] = {}
    for role, version in initial.items():
        nonnegative(version, role)
    for op in graph.operations:
        for role, version in op.produced_versions.items():
            nonnegative(version, role)
            require(role in initial, f"missing initial version for {role}")
            require(version > initial[role], f"write does not advance {role}")
            key = (role, version)
            require(key not in producers, f"duplicate version producer {key}")
            producers[key] = op.operation_id
    for role in initial:
        versions = sorted(v for (r, v) in producers if r == role)
        require(versions == list(range(initial[role] + 1, initial[role] + 1 + len(versions))),
                f"noncontiguous updates for {role}")
    for op in graph.operations:
        for role, version in op.required_versions.items():
            nonnegative(version, role)
            require(role in initial and version >= initial[role], f"invalid version read {role}")
            if version != initial[role]:
                producer = producers.get((role, version))
                require(producer in before[op.operation_id], f"missing RAW dependency: {op.operation_id}/{role}{version}")
            next_writer = producers.get((role, version + 1))
            if next_writer is not None and next_writer != op.operation_id:
                require(op.operation_id in before[next_writer], f"stale/recompute reader overtaken: {op.operation_id}/{role}")
        for role, version in op.produced_versions.items():
            require(op.required_versions.get(role) == version - 1,
                    f"commit must read preceding version: {op.operation_id}/{role}")


@dataclass(frozen=True)
class StateAllocation:
    """One physical allocation, counted once even when several roles share it."""
    id: str
    ranks: tuple[int, ...]
    bytes_per_rank: int
    roles: tuple[str, ...]
    frozen: bool
    content_hash: str
    adapter_scoped: bool = False


@dataclass(frozen=True)
class BindingRule:
    operation: str
    when: Mapping[str, str]
    cost_key: str
    ranks: tuple[int, ...]
    resources: tuple[str, ...]
    states: tuple[str, ...] = ()
    # Explicit staging/transport implementations are operations, not hidden
    # duration discounts. Backend lowering owns physical placement validation.


@dataclass(frozen=True)
class Lifetime:
    id: str
    producer: str
    consumers: tuple[str, ...]
    ranks: tuple[int, ...]
    bytes_per_rank: int


@dataclass(frozen=True)
class PlanningProblem:
    workload: str
    binding: Mapping[str, str]
    graph: ExecutionGraph
    decisions: Mapping[str, tuple[str, ...]]
    rules: tuple[BindingRule, ...]
    states: tuple[StateAllocation, ...]
    lifetimes: tuple[Lifetime, ...]
    memory_limits: Mapping[int, int]
    seeds: tuple[Mapping[str, str], ...]
    forbidden: tuple[Mapping[str, str], ...] = ()
    conditional_dependencies: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = SCHEMA
    evidence_scope: str = "UNCALIBRATED"
    # Native lowering is explicitly separate from planning/model feasibility.
    lowering: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(self.schema_version == SCHEMA, "planning schema mismatch")
        validate_binding(self.binding)
        require(self.evidence_scope in {"UNCALIBRATED", "MEASURED", "CPU_FIXTURE_ONLY"}, "invalid evidence scope")
        require(self.decisions and self.memory_limits, "empty planning domain/hardware")
        for name, options in self.decisions.items():
            require(options and len(set(options)) == len(options) and all(isinstance(x, str) and x for x in options),
                    f"invalid decision domain: {name}")
        for rank, limit in self.memory_limits.items():
            nonnegative(rank, "rank")
            require(nonnegative(limit, "memory limit") > 0, "empty GPU capacity")
        ids = {op.operation_id for op in self.graph.operations}
        require(ids, "empty operation graph")
        for op in self.graph.operations:
            require(op.condition is None, "resolve workload branch predicates before planning")
            require(type(op.metadata.get("cycle")) is int and op.metadata["cycle"] in {1, 2},
                    "two-cycle operation identity required")
        state_ids = {s.id for s in self.states}
        require(len(state_ids) == len(self.states), "duplicate physical allocation")
        for s in self.states:
            self._ranks(s.ranks)
            nonnegative(s.bytes_per_rank, s.id)
            require(s.roles and len(set(s.roles)) == len(s.roles), "empty/duplicate state roles")
            require(bool(s.content_hash), "missing immutable/state identity")
            require(len(s.roles) == 1 or s.frozen, "trainable state cannot be shared across roles")
        for rule in self.rules:
            require(rule.operation in ids and rule.cost_key, "binding references unknown operation/cost")
            self._condition(rule.when)
            self._ranks(rule.ranks)
            require(rule.resources and len(set(rule.resources)) == len(rule.resources), "empty/duplicate resource reservations")
            require(set(rule.states) <= state_ids, "unknown state allocation")
        require({r.operation for r in self.rules} == ids, "missing operation implementations")
        require(len({v.id for v in self.lifetimes}) == len(self.lifetimes), "duplicate tensor lifetime")
        for v in self.lifetimes:
            require(v.producer in ids and v.consumers and set(v.consumers) <= ids, "invalid tensor lifetime")
            require(v.producer not in v.consumers, "self-consuming lifetime")
            self._ranks(v.ranks)
            nonnegative(v.bytes_per_rank, v.id)
        for condition in self.forbidden:
            require(bool(condition), "empty forbidden condition rejects every plan")
            self._condition(condition)
        for row in self.conditional_dependencies:
            require(set(row) == {"when", "source", "target"}, "invalid conditional dependency")
            self._condition(row["when"])
            require(row["source"] in ids and row["target"] in ids, "unknown conditional edge endpoint")
        for seed in self.seeds:
            self.check_assignment(seed)

    def _ranks(self, ranks: tuple[int, ...]) -> None:
        require(ranks and len(set(ranks)) == len(ranks) and set(ranks) <= set(self.memory_limits), "invalid GPU mapping")

    def _condition(self, condition: Mapping[str, str]) -> None:
        require(all(k in self.decisions and v in self.decisions[k] for k, v in condition.items()), "unknown decision option")

    def check_assignment(self, assignment: Mapping[str, str]) -> None:
        require(set(assignment) == set(self.decisions), "plan must bind every decision exactly once")
        self._condition(assignment)
        require(not any(matches(x, assignment) for x in self.forbidden), "forbidden capability/semantic combination")

    def resolve(self, assignment: Mapping[str, str]) -> tuple[ExecutionGraph, dict[str, BindingRule]]:
        self.check_assignment(assignment)
        rules = {}
        for op in self.graph.operations:
            choices = [r for r in self.rules if r.operation == op.operation_id and matches(r.when, assignment)]
            require(len(choices) == 1, f"ambiguous or missing lowering rule: {op.operation_id}")
            rules[op.operation_id] = choices[0]
        extra = tuple(Dependency(r["source"], r["target"], "plan_control")
                      for r in self.conditional_dependencies if matches(r["when"], assignment))
        graph = ExecutionGraph(self.graph.operations, self.graph.dependencies + extra, self.graph.metadata)
        validate_versions(graph)
        before = ancestors(graph)
        for v in self.lifetimes:
            require(all(v.producer in before[c] for c in v.consumers), f"tensor consumer not dependent on producer: {v.id}")
        state_map = {s.id: s for s in self.states}
        op_map = {o.operation_id: o for o in graph.operations}
        # Equal content is not enough to share: the selected physical object
        # must advertise each reader role and cover the operation's ranks.
        for key, rule in rules.items():
            if op_map[key].kind not in {OperationKind.VERSION_SYNC, OperationKind.BARRIER}:
                require({f"gpu:{r}" for r in rule.ranks} <= set(rule.resources),
                        "model operation must reserve each executing GPU")
            for sid in rule.states:
                state = state_map[sid]
                require(set(rule.ranks) <= set(state.ranks), "state unavailable on selected ranks")
                require(op_map[key].role in state.roles, "role not authorized to consume shared state")
                if state.adapter_scoped:
                    require(bool(op_map[key].adapter_mode), "shared adapter execution/replay has no role scope")
        return graph, rules

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["graph"] = self.graph.to_dict()
        value["memory_limits"] = {str(k): v for k, v in self.memory_limits.items()}
        return value

    @property
    def digest(self) -> str:
        return fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanningProblem":
        fields = set(cls.__dataclass_fields__)
        require(set(value) <= fields, "unknown planning problem field")
        d = dict(value)
        d["graph"] = graph_from_dict(d["graph"])
        d["decisions"] = {k: tuple(v) for k, v in d["decisions"].items()}
        d["rules"] = tuple(BindingRule(**{**r, "ranks": tuple(r["ranks"]), "resources": tuple(r["resources"]),
                                         "states": tuple(r.get("states", ()))}) for r in d["rules"])
        d["states"] = tuple(StateAllocation(**{**s, "ranks": tuple(s["ranks"]), "roles": tuple(s["roles"])}) for s in d["states"])
        d["lifetimes"] = tuple(Lifetime(**{**v, "ranks": tuple(v["ranks"]), "consumers": tuple(v["consumers"])}) for v in d["lifetimes"])
        d["memory_limits"] = {int(k): v for k, v in d["memory_limits"].items()}
        for k in ("seeds", "forbidden", "conditional_dependencies"):
            d[k] = tuple(d.get(k, ()))
        return cls(**d)


def matches(condition: Mapping[str, str], assignment: Mapping[str, str]) -> bool:
    return all(assignment.get(k) == v for k, v in condition.items())
