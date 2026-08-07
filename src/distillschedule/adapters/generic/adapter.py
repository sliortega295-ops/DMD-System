"""Declarative adapter for new DMD families before a native implementation exists."""

from __future__ import annotations

from pathlib import Path

from distillschedule.adapters.base import CompiledAdapterPlan, DMDModelAdapter
from distillschedule.core.capabilities import AdapterCapabilities
from distillschedule.core.contracts import ContractViolation
from distillschedule.core.enums import RolloutEngineKind, TrainingBackendKind
from distillschedule.core.plan import ExecutionPlan
from distillschedule.workloads.dmd import DMDProtocolCompiler


class GenericDMDAdapter(DMDModelAdapter):
    name = "generic"

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            training_backends=(
                TrainingBackendKind.NATIVE,
                TrainingBackendKind.DEEPSPEED,
                TrainingBackendKind.FSDP,
                TrainingBackendKind.HSDP,
            ),
            rollout_engines=(
                RolloutEngineKind.NATIVE,
                RolloutEngineKind.VLLM,
                RolloutEngineKind.SGLANG,
                RolloutEngineKind.REMOTE,
            ),
            supports_dynamic_shape=False,
            supports_exact_version_sync=True,
            supports_external_rollout=True,
        )

    def validate(self, plan: ExecutionPlan) -> None:
        for component, backend in plan.training_backends.items():
            if not self.capabilities.supports_training(backend.kind):
                raise ContractViolation(
                    f"backend {backend.kind.value!r} is unsupported for {component}",
                    path=f"training_backends.{component}.kind",
                )
        if not self.capabilities.supports_rollout(plan.rollout.engine):
            raise ContractViolation(
                f"rollout engine {plan.rollout.engine.value!r} is unsupported",
                path="rollout.engine",
            )

    def compile(self, plan: ExecutionPlan, *, source_dir: Path, repo_root: Path) -> CompiledAdapterPlan:
        self.validate(plan)
        protocol = DMDProtocolCompiler().compile(plan)
        return CompiledAdapterPlan(
            protocol=protocol,
            invocation=None,
            warnings=(
                "generic adapter is plan-only; add a native adapter or legacy invocation before execution",
            ),
        )
