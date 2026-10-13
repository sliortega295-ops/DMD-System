"""Open workload semantics, independent of model names and optimization recipes.

A workload supplies calls, model bindings and real training dependencies. It does
not supply a winner or schedule. This frontend compiles to the EXISTING versioned
ExecutionGraph. Backend capabilities and the existing solver choose execution.
Python loops build parameterized programs; unresolved data-dependent branches
must be instantiated by the workload before compilation, never guessed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from distillschedule.core.contracts import require
from distillschedule.core.enums import GradientMode, OperationKind
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation
from distillschedule.search.plan_ir import validate_versions
from distillschedule.workloads.best_known.registry import fingerprint


@dataclass(frozen=True)
class Call:
    """Training semantics only. Names are arbitrary, not a workload whitelist.

    inputs refer to earlier call IDs or external feed IDs. reads/writes describe
    ALL mutable logical state (parameters, gradient buffers, optimizer, EMA).
    backward_of identifies forward-derived values differentiated by this call;
    the compiler extends their transitive state reads and tensor lifetimes.
    replay_safe is a user assertion about purity/RNG/state of a forward region,
    not permission to checkpoint an optimizer or arbitrary training helper.
    """
    id: str
    binding: str
    inputs: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    grad: str = "no_grad"
    backward_of: tuple[str, ...] = ()
    replay_safe: bool = False
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        require(isinstance(self.id, str) and bool(self.id) and isinstance(self.binding, str)
                and bool(self.binding), "call and binding IDs are required", path="program")
        GradientMode.parse(self.grad, path="program.grad")
        for name in ("inputs", "reads", "writes", "after", "backward_of"):
            values = getattr(self, name)
            require(isinstance(values, (tuple, list)) and len(set(values)) == len(values)
                    and all(isinstance(x, str) and x for x in values),
                    f"invalid {name}", path="program")
            object.__setattr__(self, name, tuple(values))
        require(set(self.writes) <= set(self.reads), "writes must read the preceding state", path="program")
        require(type(self.replay_safe) is bool, "replay_safe must be boolean", path="program")
        require(not self.replay_safe or (self.grad == "grad" and not self.writes and not self.backward_of),
                "only pure gradient-bearing forward calls may be replayed", path="program")
        require(set(self.backward_of) <= set(self.inputs), "backward targets must be inputs", path="program")
        fingerprint(dict(self.attributes))

    def to_dict(self):
        return {"id": self.id, "binding": self.binding,
                **{k: list(getattr(self, k)) for k in ("inputs", "reads", "writes", "after", "backward_of")},
                "grad": self.grad, "replay_safe": self.replay_safe, "attributes": dict(self.attributes)}


@dataclass(frozen=True)
class Program:
    calls: tuple[Call, ...]
    initial_versions: Mapping[str, int]
    feeds: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "distillschedule.open_program.v1"

    def __post_init__(self):
        require(self.schema_version == "distillschedule.open_program.v1", "unknown program schema", path="program")
        require(0 < len(self.calls) <= 512, "program requires 1..512 instantiated calls", path="program")
        require(all(isinstance(c, Call) for c in self.calls), "typed calls required", path="program")
        ids = [c.id for c in self.calls]
        require(len(set(ids)) == len(ids), "duplicate call IDs", path="program")
        require(len(set(self.feeds)) == len(self.feeds) and not set(self.feeds) & set(ids),
                "feeds must be unique and disjoint from calls", path="program")
        require(len(set(self.outputs)) == len(self.outputs) and set(self.outputs) <= set(ids),
                "unknown or duplicate program outputs", path="program")
        require(all(isinstance(k, str) and k and type(v) is int and v >= 0
                    for k, v in self.initial_versions.items()), "invalid initial state versions", path="program")
        fingerprint(dict(self.parameters))
        self.compile()

    def compile(self) -> ExecutionGraph:
        current = dict(self.initial_versions)
        writers: dict[str, str] = {}
        readers = {state: set() for state in current}
        operations = []
        edges = set()
        seen = {}
        provenance: dict[str, dict[str, int]] = {}
        ancestry: dict[str, set[str]] = {}
        feeds = set(self.feeds)
        for call in self.calls:
            require(set(call.reads) <= set(current), "undeclared logical state", path=f"program.{call.id}")
            require(set(call.inputs) <= set(seen) | feeds, "input must be an earlier call or feed", path=f"program.{call.id}")
            require(set(call.after) <= set(seen), "control dependency must be an earlier call", path=f"program.{call.id}")
            direct = set(call.inputs) & set(seen)
            needed = {state: current[state] for state in call.reads}
            retained = set(direct)
            for key in call.backward_of:
                require(key in seen, "backward target must be produced by the program", path="program")
                for state, version in provenance[key].items():
                    require(current[state] == version, "backward/replay after parameter version changed", path=f"program.{call.id}")
                    needed[state] = version
                retained.update(ancestry[key] | {key})
            parents = set(call.after) | retained
            for state in needed:
                if state in writers:
                    parents.add(writers[state])
                readers[state].add(call.id)
            for state in call.writes:
                parents.update(readers[state] - {call.id})
            edges.update((source, call.id) for source in parents)
            produced = {state: current[state] + 1 for state in call.writes}
            operations.append(Operation(call.id, len(operations), "open_runtime", OperationKind.CALL,
                GradientMode.parse(call.grad, path="program.grad"),
                required_versions=needed, produced_versions=produced,
                metadata={"inputs": sorted(retained), "binding": call.binding,
                          "declared_inputs": list(call.inputs), "replay_safe": call.replay_safe,
                          "attributes": dict(call.attributes)}))
            ancestry[call.id] = set(direct)
            for key in direct:
                ancestry[call.id].update(ancestry[key])
            provenance[call.id] = {}
            if call.grad != "no_grad" and not call.backward_of:
                for key in direct:
                    for state, version in provenance[key].items():
                        require(state not in provenance[call.id] or provenance[call.id][state] == version,
                                "mixed parameter versions in one differentiable value", path="program")
                        provenance[call.id][state] = version
                provenance[call.id].update(needed)
            for state in call.writes:
                current[state] = produced[state]
                writers[state] = call.id
                readers[state] = set()
            seen[call.id] = call
        # Returned values must survive to the end, not be released by the runner.
        graph = ExecutionGraph(tuple(operations), tuple(Dependency(a, b) for a, b in sorted(edges)),
            {"initial_versions": dict(self.initial_versions), "final_versions": current,
             "program_schema": self.schema_version, "parameters": dict(self.parameters)})
        validate_versions(graph)
        return graph

    def to_dict(self):
        return {"schema_version": self.schema_version, "calls": [c.to_dict() for c in self.calls],
                "initial_versions": dict(self.initial_versions), "feeds": list(self.feeds),
                "outputs": list(self.outputs), "parameters": dict(self.parameters)}

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, Mapping) and set(value) <= set(cls.__dataclass_fields__),
                "unknown program fields", path="program")
        return cls(**{**value, "calls": tuple(Call(**x) for x in value["calls"]),
                      "feeds": tuple(value.get("feeds", ())), "outputs": tuple(value.get("outputs", ()))})
