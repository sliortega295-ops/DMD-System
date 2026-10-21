"""A user training request, separate from a physical ExecutionPlan."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from distillschedule.core.config import load_document
from distillschedule.workloads.best_known.registry import fingerprint as digest
from .plugins import resolve_reference

SCHEMA = "distillschedule.training.v1"


@dataclass(frozen=True)
class TrainingConfig:
    model: Mapping[str, Any]
    data: Mapping[str, Any]
    algorithm: Mapping[str, Any]
    execution: Mapping[str, Any]
    trainer: Mapping[str, Any]
    source: Path

    @classmethod
    def from_dict(cls, value, *, source):
        source = Path(source).expanduser().resolve()
        if set(value) != {"schema_version", "model", "data", "algorithm", "execution", "trainer"}:
            raise ValueError("training config needs model/data/algorithm/execution/trainer sections")
        if value["schema_version"] != SCHEMA:
            raise ValueError("unsupported training config schema")
        parts = {}
        for key in ("model", "data", "algorithm", "execution", "trainer"):
            if not isinstance(value[key], dict):
                raise ValueError(f"{key} must be an object")
            parts[key] = deepcopy(value[key])
        execution, trainer = parts["execution"], parts["trainer"]
        if execution.get("adapter", "program") not in ("program", "native"):
            raise ValueError("execution.adapter must be program or native")
        if execution.get("mode", "reference") not in ("reference", "manual", "plan", "auto"):
            raise ValueError("unsupported execution mode")
        if execution.get("mode") == "plan" and not execution.get("plan"):
            raise ValueError("saved-plan mode requires execution.plan")
        if type(trainer.get("cycles")) is not int or trainer["cycles"] < 1:
            raise ValueError("trainer.cycles must be positive")
        if type(trainer.get("seed", 0)) is not int or not 0 <= trainer.get("seed", 0) < 2**63:
            raise ValueError("trainer.seed must be an integer in [0, 2**63)")
        if not trainer.get("output_dir"):
            raise ValueError("trainer.output_dir is required")
        if type(trainer.get("checkpoint_every", 1)) is not int or trainer.get("checkpoint_every", 1) < 0:
            raise ValueError("checkpoint_every must be a nonnegative integer")
        if execution.get("adapter", "program") == "program":
            allowed = {
                "model": {"factory", "parameters"}, "data": {"prepared"},
                "algorithm": {"factory", "parameters"},
                "execution": {"adapter", "mode", "implementation", "order", "plan", "memory_limit_bytes",
                              "max_candidates", "samples", "max_program_runs", "max_states"},
                "trainer": {"cycles", "seed", "output_dir", "checkpoint_every", "resume_from", "device"},
            }
            for key, names in allowed.items():
                if set(parts[key]) - names:
                    raise ValueError(f"unsupported {key} fields: {sorted(set(parts[key]) - names)}")
            mode = execution.get("mode", "reference")
            if mode != "manual" and ("implementation" in execution or "order" in execution):
                raise ValueError("implementation/order overrides require manual mode")
            if mode != "plan" and "plan" in execution:
                raise ValueError("execution.plan requires saved-plan mode")
            if mode == "manual" and not isinstance(execution.get("implementation"), dict):
                raise ValueError("manual mode requires an explicit implementation object")
            for key in ("model", "algorithm"):
                if not isinstance(parts[key].get("factory"), str):
                    raise ValueError(f"{key}.factory is required")
                parts[key]["factory"] = resolve_reference(parts[key]["factory"], source.parent)
                if not isinstance(parts[key].get("parameters", {}), dict):
                    raise ValueError(f"{key}.parameters must be an object")
            if not parts["data"].get("prepared"):
                raise ValueError("data.prepared is required; prepare raw media before training")
        else:
            if parts["model"].get("workload") not in tuple(f"W{i}" for i in range(1, 7)):
                raise ValueError("native model.workload must be W1-W6")
        for section, keys in (("data", ("prepared",)), ("trainer", ("output_dir", "resume_from")),
                              ("execution", ("plan", "task", "assets", "capacity_policy", "transport"))):
            for key in keys:
                if parts[section].get(key):
                    parts[section][key] = str((source.parent / Path(parts[section][key]).expanduser()).resolve())
        return cls(**parts, source=source)

    def to_dict(self):
        return {"schema_version": SCHEMA, **{key: deepcopy(dict(getattr(self, key)))
                for key in ("model", "data", "algorithm", "execution", "trainer")}}

    @property
    def identity(self):
        # Output, checkpoint frequency and target cycle count do not change a
        # complete-cycle restart's mathematical state. Everything else does.
        value = self.to_dict()
        value["trainer"] = {key: val for key, val in value["trainer"].items()
                            if key not in ("output_dir", "resume_from", "cycles", "checkpoint_every")}
        return digest(value)


def load_training_config(path):
    value, _ = load_document(path)
    return TrainingConfig.from_dict(value, source=path)
