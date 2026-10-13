"""Workload facts, separated from optimization choices and physical execution.

A new program supplies an existing ExecutionGraph and per-call properties. It
never supplies a winning recipe. Operator/gradient/version semantics remain the
responsibility of its model interface, rather than being guessed by the planner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

from distillschedule.core.contracts import require
from distillschedule.core.graph import ExecutionGraph
from distillschedule.search.plan_ir import ancestors, graph_from_dict, validate_versions
from distillschedule.workloads.best_known.registry import fingerprint


def json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


@dataclass(frozen=True)
class CallContract:
    """Algorithm facts: operator interface and shape/precision/gradient context."""
    operator: str
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        require(isinstance(self.operator, str) and bool(self.operator.strip()),
                "operator interface must be nonempty", path="program.call")
        object.__setattr__(self, "properties", json_copy(dict(self.properties)))

    def to_dict(self):
        return {"operator": self.operator, "properties": json_copy(self.properties)}


@dataclass(frozen=True)
class WorkloadProgram:
    """Resolved logical graph; no model-name dispatch or hidden fixed frequency.

    Every side effect and delayed backward/replay read must be represented in the
    supplied graph. Branches must be resolved before planning. Physical choices are
    produced by capabilities, not inserted into the workload's parameter contract.
    """
    graph: ExecutionGraph
    calls: Mapping[str, CallContract]
    parameters: Mapping[str, Any]
    identity: Mapping[str, Any]
    name: str = "unnamed"

    def __post_init__(self):
        object.__setattr__(self, "graph", graph_from_dict(json_copy(self.graph.to_dict())))
        require(bool(self.graph.operations) and len(self.graph.operations) <= 512,
                "program requires 1..512 resolved calls", path="program")
        require(set(self.calls) == {op.operation_id for op in self.graph.operations},
                "each graph operation requires one call contract", path="program")
        require(all(isinstance(c, CallContract) for c in self.calls.values()),
                "typed call contracts required", path="program")
        object.__setattr__(self, "calls", {k: CallContract(**v.to_dict()) for k, v in self.calls.items()})
        object.__setattr__(self, "parameters", json_copy(dict(self.parameters)))
        object.__setattr__(self, "identity", json_copy(dict(self.identity)))
        require(bool(self.identity), "source/model/input identity required", path="program")
        validate_versions(self.graph)
        before = ancestors(self.graph)
        for op in self.graph.operations:
            require(op.condition is None, "resolve conditional branches before planning", path="program")
            inputs = op.metadata.get("inputs", ())
            require(isinstance(inputs, (list, tuple)) and all(isinstance(i, str) for i in inputs),
                    "invalid call input references", path="program")
            require(set(inputs) <= before[op.operation_id],
                    "input producers must be dependency ancestors", path="program")

    def to_dict(self):
        return {"schema_version": "distillschedule.workload_program.v1", "name": self.name,
                "graph": self.graph.to_dict(), "calls": {k: v.to_dict() for k, v in self.calls.items()},
                "parameters": json_copy(self.parameters), "identity": json_copy(self.identity)}

    @property
    def digest(self):
        value = self.to_dict()
        value.pop("name")
        return fingerprint(value)

    @classmethod
    def from_dict(cls, value):
        require(set(value) == {"schema_version", "name", "graph", "calls", "parameters", "identity"}
                and value["schema_version"] == "distillschedule.workload_program.v1",
                "unknown program fields/schema", path="program")
        return cls(graph_from_dict(value["graph"]),
                   {k: CallContract(**v) for k, v in value["calls"].items()},
                   value["parameters"], value["identity"], value["name"])
