"""Static role runner for gradually replacing the legacy Helios entrypoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from distillschedule.core.contracts import ContractViolation
from distillschedule.core.graph import ExecutionGraph, Operation

from .version_sync import VersionLedger


@dataclass
class OperationContext:
    operation: Operation
    versions: VersionLedger
    values: dict[str, Any] = field(default_factory=dict)


OperationHandler = Callable[[OperationContext], Any]


class StaticRoleRunner:
    """Execute only operations owned by one role in a precompiled graph.

    It is intentionally small: the first refactor version still routes Helios to
    its source-faithful entrypoint. Model adapters can migrate one operation at a
    time by registering handlers here without changing core scheduling types.
    """

    def __init__(
        self,
        role: str,
        handlers: Mapping[str, OperationHandler],
        *,
        versions: VersionLedger | None = None,
    ) -> None:
        self.role = role
        self.handlers = dict(handlers)
        self.versions = versions or VersionLedger()

    def run(self, graph: ExecutionGraph) -> dict[str, Any]:
        operations = {operation.operation_id: operation for operation in graph.operations}
        results: dict[str, Any] = {}
        for operation_id in graph.topological_order():
            operation = operations[operation_id]
            if operation.role != self.role:
                continue
            handler = self.handlers.get(operation.kind.value)
            if handler is None:
                raise ContractViolation(
                    f"role {self.role!r} has no handler for {operation.kind.value!r}",
                    path=f"runtime.role_runner.{operation_id}",
                )
            for component, expected in operation.required_versions.items():
                self.versions.require(component, expected)
            context = OperationContext(operation=operation, versions=self.versions, values=results)
            results[operation_id] = handler(context)
            for component, produced in operation.produced_versions.items():
                self.versions.set(component, produced)
        return results
