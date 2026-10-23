"""Static role runner for gradually replacing the legacy Helios entrypoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any, Callable, Mapping

from distillschedule.core.contracts import ContractViolation, require
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

    def prepare(self, graph: ExecutionGraph, *, order: tuple[str, ...],
                retain_outputs: tuple[str, ...] = ()) -> PreparedRolePlan:
        """Validate and bind a selected schedule before the training timer.

        Unlike the legacy role-filtered ``run``, this path consumes *every*
        action exactly once. Cross-pool transport must be lowered explicitly;
        silently skipping another role would not implement a distributed plan.
        """
        return PreparedRolePlan(self, graph, order, retain_outputs)

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


class PreparedRolePlan:
    """One-shot native dispatcher with last-consumer tensor lifetime release.

    No Torch import, synchronization, tensor inspection, copies, or collectives
    are performed here. Handlers retain the native loss/backward/optimizer code.
    Receipts describe dispatch, not proof of native numerical equivalence.
    """

    def __init__(self, runner: StaticRoleRunner, graph: ExecutionGraph,
                 order: tuple[str, ...], retain_outputs: tuple[str, ...]) -> None:
        path = "runtime.prepared_plan"
        graph = deepcopy(graph)
        operations = {op.operation_id: op for op in graph.operations}
        require(len(order) == len(operations) and set(order) == set(operations),
                "selected order must execute every operation exactly once", path=path)
        position = {key: i for i, key in enumerate(order)}
        ancestors: dict[str, set[str]] = {key: set() for key in order}
        for edge in graph.dependencies:
            require(position[edge.source] < position[edge.target],
                    "selected order violates dependency", path=path)
        for key in order:
            for edge in graph.dependencies:
                if edge.target == key:
                    ancestors[key].update({edge.source} | ancestors[edge.source])
        require(len(set(retain_outputs)) == len(retain_outputs)
                and set(retain_outputs) <= set(operations), "unknown retained output", path=path)
        initial = dict(graph.metadata.get("initial_versions", runner.versions.snapshot().values))
        versions = VersionLedger(initial)
        self._last_use = {key: position[key] for key in order}
        self._inputs: set[str] = set()
        bound = []
        for key in order:
            op = operations[key]
            require(op.role == runner.role, "cross-role actions require explicit transport lowering", path=path)
            require(op.condition is None, "conditional action must be resolved before execution", path=path)
            handler = runner.handlers.get(op.kind.value)
            require(callable(handler), f"missing handler for {op.kind.value}", path=path)
            for component, expected in op.required_versions.items():
                versions.require(component, expected)
            for component, version in op.produced_versions.items():
                versions.set(component, version)
            inputs = op.metadata.get("inputs", ())
            require(isinstance(inputs, (tuple, list)) and all(isinstance(v, str) for v in inputs),
                    "invalid input references", path=path)
            for source in inputs:
                require(source in ancestors[key], "input producer must be a dependency ancestor", path=path)
                self._inputs.add(source)
                self._last_use[source] = max(self._last_use[source], position[key])
            bound.append((op, handler))
        expected_final = graph.metadata.get("final_versions", versions.snapshot().values)
        require(versions.snapshot().values == expected_final, "final version contract mismatch", path=path)
        self._bound = tuple(bound)
        self._versions = runner.versions
        self._initial = initial
        self._final = dict(expected_final)
        self._retain = frozenset(retain_outputs)
        self._release = tuple(tuple(key for key in order if self._last_use[key] == i and key not in self._retain)
                              for i in range(len(order)))
        self._status, self._failed = "NOT_RUN", None
        self._executed: list[str] = []
        self._peak_inputs = 0
        self._terminal_versions: dict[str, int] | None = None

    def run(self) -> dict[str, Any]:
        require(self._status == "NOT_RUN", "prepared plans are one-shot", path="runtime.prepared_plan")
        self._status = "RUNNING"
        values: dict[str, Any] = {}
        try:
            for component, expected in self._initial.items():
                self._versions.require(component, expected)
            for i, (op, handler) in enumerate(self._bound):
                self._failed = op.operation_id
                for component, expected in op.required_versions.items():
                    self._versions.require(component, expected)
                values[op.operation_id] = handler(OperationContext(op, self._versions, values))
                for component, version in op.produced_versions.items():
                    self._versions.set(component, version)
                self._executed.append(op.operation_id)
                self._peak_inputs = max(self._peak_inputs, len(self._inputs.intersection(values)))
                for key in self._release[i]:
                    del values[key]
            require(self._versions.snapshot().values == self._final,
                    "executed final version mismatch", path="runtime.prepared_plan")
            self._status, self._failed = "PASS", None
            return values
        except BaseException:
            self._status = "FAIL"
            values.clear()
            raise
        finally:
            self._terminal_versions = dict(self._versions.snapshot().values)

    def receipt(self) -> dict[str, Any]:
        return {"schema_version": "distillschedule.prepared_role_execution.v1",
                "status": self._status, "executed_operations": list(self._executed),
                "failed_operation": self._failed, "initial_versions": dict(self._initial),
                "final_versions": dict(self._terminal_versions if self._terminal_versions is not None
                                       else self._versions.snapshot().values),
                "peak_live_inputs": self._peak_inputs, "native_correctness": "REQUIRES_NATIVE_RECEIPTS"}
