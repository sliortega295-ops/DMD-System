"""Compile typed plans and execute compatibility invocations without importing models."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from distillschedule.adapters import AdapterRegistry, CompiledAdapterPlan, default_registry
from distillschedule.core.config import LoadedPlan
from distillschedule.core.contracts import ContractViolation


class CommandRunner(Protocol):
    def run(self, command: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> int:
        ...


class SubprocessCommandRunner:
    def run(self, command: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> int:
        completed = subprocess.run(list(command), cwd=str(cwd), env=dict(env), check=False)
        return int(completed.returncode)


@dataclass(frozen=True)
class CompiledRun:
    loaded_plan: LoadedPlan
    adapter_plan: CompiledAdapterPlan
    repo_root: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.loaded_plan.plan.plan_id,
            "plan_digest": self.loaded_plan.digest,
            "adapter": self.loaded_plan.plan.workload.adapter,
            "profile_key": self.loaded_plan.plan.profile_key,
            "source_files": [str(path) for path in self.loaded_plan.source_files],
            "warnings": list(self.adapter_plan.warnings),
            "protocol": {
                "steps": [
                    {
                        "logical_step": step.intent.logical_step,
                        "kind": step.intent.kind.value,
                        "generator_update": step.intent.generator_update,
                        "critic_update": step.intent.critic_update,
                        "ema_update": step.intent.ema_update,
                        "operation_ids": list(step.operation_ids),
                    }
                    for step in self.adapter_plan.protocol.steps
                ],
                "final_versions": dict(self.adapter_plan.protocol.final_versions),
                "graph": self.adapter_plan.protocol.graph.to_dict(),
            },
            "invocation": (
                self.adapter_plan.invocation.to_dict()
                if self.adapter_plan.invocation is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ExecutionResult:
    plan_id: str
    return_code: int | None
    dry_run: bool
    output_dir: Path
    command: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class DistillExecutor:
    def __init__(
        self,
        *,
        registry: AdapterRegistry | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.runner = runner or SubprocessCommandRunner()

    def compile(self, loaded_plan: LoadedPlan, *, repo_root: str | Path) -> CompiledRun:
        root = Path(repo_root).expanduser().resolve()
        if not root.is_dir():
            raise ContractViolation("repository root does not exist", path="repo_root")
        adapter = self.registry.get(loaded_plan.plan.workload.adapter)
        adapter_plan = adapter.compile(
            loaded_plan.plan,
            source_dir=loaded_plan.base_dir,
            repo_root=root,
        )
        return CompiledRun(loaded_plan=loaded_plan, adapter_plan=adapter_plan, repo_root=root)

    def run(
        self,
        loaded_plan: LoadedPlan,
        *,
        repo_root: str | Path,
        dry_run: bool = False,
        check: bool = True,
    ) -> ExecutionResult:
        compiled = self.compile(loaded_plan, repo_root=repo_root)
        plan = loaded_plan.plan
        output_dir = Path(plan.evidence.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (loaded_plan.base_dir / output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "compiled_run.json").write_text(
            json.dumps(compiled.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        invocation = compiled.adapter_plan.invocation
        if invocation is not None:
            for generated_path, content in invocation.generated_files.items():
                generated_path.parent.mkdir(parents=True, exist_ok=True)
                generated_path.write_text(content, encoding="utf-8")
        if dry_run:
            return ExecutionResult(
                plan_id=plan.plan_id,
                return_code=None,
                dry_run=True,
                output_dir=output_dir,
                command=invocation.command if invocation is not None else (),
                warnings=compiled.adapter_plan.warnings,
            )
        if invocation is None:
            raise ContractViolation(
                "compiled plan has no executable backend; use --dry-run or implement the adapter",
                path="execution",
            )
        environment = dict(os.environ)
        environment.update(invocation.environment)
        return_code = self.runner.run(
            invocation.command,
            cwd=invocation.cwd,
            env=environment,
        )
        if check and return_code != 0:
            raise RuntimeError(
                f"plan {plan.plan_id!r} failed with return code {return_code}"
            )
        return ExecutionResult(
            plan_id=plan.plan_id,
            return_code=return_code,
            dry_run=False,
            output_dir=output_dir,
            command=invocation.command,
            warnings=compiled.adapter_plan.warnings,
        )
