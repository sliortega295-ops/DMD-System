"""Small logical execution graph used by adapters and static role runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import ensure_unique, require
from .enums import GradientMode, OperationKind


@dataclass(frozen=True)
class Operation:
    operation_id: str
    logical_step: int
    role: str
    kind: OperationKind
    grad_mode: GradientMode
    physical_state: str | None = None
    adapter_mode: str | None = None
    required_versions: Mapping[str, int] = field(default_factory=dict)
    produced_versions: Mapping[str, int] = field(default_factory=dict)
    condition: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Dependency:
    source: str
    target: str
    kind: str = "control"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionGraph:
    operations: tuple[Operation, ...]
    dependencies: tuple[Dependency, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = tuple(operation.operation_id for operation in self.operations)
        ensure_unique(ids, path="execution_graph.operations")
        known = set(ids)
        for dependency in self.dependencies:
            require(dependency.source in known, f"unknown dependency source {dependency.source}", path="execution_graph.dependencies")
            require(dependency.target in known, f"unknown dependency target {dependency.target}", path="execution_graph.dependencies")
        self.topological_order()

    def topological_order(self) -> tuple[str, ...]:
        indegree = {operation.operation_id: 0 for operation in self.operations}
        outgoing = {operation.operation_id: [] for operation in self.operations}
        for dependency in self.dependencies:
            indegree[dependency.target] += 1
            outgoing[dependency.source].append(dependency.target)
        ready = [operation.operation_id for operation in self.operations if indegree[operation.operation_id] == 0]
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for target in outgoing[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        require(len(order) == len(self.operations), "execution graph contains a cycle", path="execution_graph")
        return tuple(order)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "operations": [
                {
                    "operation_id": operation.operation_id,
                    "logical_step": operation.logical_step,
                    "role": operation.role,
                    "kind": operation.kind.value,
                    "grad_mode": operation.grad_mode.value,
                    "physical_state": operation.physical_state,
                    "adapter_mode": operation.adapter_mode,
                    "required_versions": dict(operation.required_versions),
                    "produced_versions": dict(operation.produced_versions),
                    "condition": operation.condition,
                    "metadata": dict(operation.metadata),
                }
                for operation in self.operations
            ],
            "dependencies": [
                {"source": edge.source, "target": edge.target, "kind": edge.kind, "metadata": dict(edge.metadata)}
                for edge in self.dependencies
            ],
        }
