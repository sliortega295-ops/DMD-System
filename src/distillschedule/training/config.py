"""A user training request, separate from a physical ExecutionPlan."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from distillschedule.core.config import load_document
from distillschedule.workloads.best_known.registry import fingerprint as digest

from .plugins import resolve_reference
from .sampling import sampling_policy

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
                "model": {"name", "factory", "parameters"}, "data": {"prepared", "streams", "sharding", "validation_prepared", "sampling"},
                "algorithm": {"name", "factory", "parameters"},
                "execution": {"adapter", "mode", "implementation", "order", "plan", "memory_limit_bytes",
                              "max_candidates", "samples", "max_program_runs", "max_states"},
                "trainer": {"cycles", "seed", "output_dir", "checkpoint_every", "resume_from", "device", "distributed", "validation"},
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
                if 'name' in parts[key]:
                    from .catalog import resolve
                    entry = resolve(key, parts[key]['name'])
                    if 'factory' in parts[key] and parts[key]['factory'] != entry['factory']:
                        raise ValueError(f'{key}.name and factory disagree')
                    parts[key]['factory'] = entry['factory']
                if not isinstance(parts[key].get("factory"), str):
                    raise ValueError(f"{key}.factory is required")
                parts[key]["factory"] = resolve_reference(parts[key]["factory"], source.parent)
                if not isinstance(parts[key].get("parameters", {}), dict):
                    raise ValueError(f"{key}.parameters must be an object")
            if bool(parts['data'].get('prepared')) == bool(parts['data'].get('streams')):
                raise ValueError('exactly one of data.prepared or data.streams is required')
            if 'streams' in parts['data']:
                if 'sampling' in parts['data']:
                    raise ValueError('streams use per-stream sampling, not data.sampling')
                streams = parts['data']['streams']
                if not isinstance(streams, dict) or not streams:
                    raise ValueError('data.streams must be a nonempty object')
                for name, stream in streams.items():
                    if (not isinstance(name, str) or not name or not isinstance(stream, dict)
                            or set(stream) - {'prepared', 'samples_per_cycle', 'sampling'}
                            or not isinstance(stream.get('prepared'), str) or not stream['prepared']
                            or type(stream.get('samples_per_cycle', 1)) is not int
                            or stream.get('samples_per_cycle', 1) < 1):
                        raise ValueError('each stream needs prepared and positive samples_per_cycle')
                    if 'sampling' in stream:
                        stream['sampling'] = sampling_policy(stream['sampling'])
                    stream['prepared'] = str((source.parent / Path(stream['prepared']).expanduser()).resolve())
            if parts["data"].get("sharding", "single") not in ("single", "rank_strided"):
                raise ValueError("data.sharding must be single or rank_strided")
            if "sampling" in parts["data"]:
                parts["data"]["sampling"] = sampling_policy(parts["data"]["sampling"])
            if ("validation" in trainer) != ("validation_prepared" in parts["data"]):
                raise ValueError("trainer.validation and data.validation_prepared must be supplied together")
            if "validation" in trainer:
                settings = trainer["validation"]
                if (not isinstance(settings, dict) or set(settings) - {"every_cycles", "samples", "parameters"}
                        or any(type(settings.get(key)) is not int or settings[key] < 1
                               for key in ("every_cycles", "samples"))):
                    raise ValueError("validation requires positive every_cycles/samples and no unknown fields")
                if (not isinstance(parts["data"]["validation_prepared"], str)
                        or not parts["data"]["validation_prepared"]):
                    raise ValueError("data.validation_prepared must be an explicit prepared manifest path")
                def json_keys(value):
                    if isinstance(value, dict):
                        return all(isinstance(key, str) and json_keys(item) for key, item in value.items())
                    if isinstance(value, (list, tuple)):
                        return all(json_keys(item) for item in value)
                    return value is None or type(value) in (str, int, float, bool)
                parameters = settings.get("parameters", {})
                try:
                    canonical = json.dumps(parameters, allow_nan=False, sort_keys=True)
                    if not isinstance(parameters, dict) or not json_keys(parameters):
                        raise ValueError("not a JSON mapping")
                    settings["parameters"] = json.loads(canonical)
                except (TypeError, ValueError) as exc:
                    raise ValueError("validation parameters must be a finite JSON mapping with string keys") from exc
            if "distributed" in trainer:
                distributed = trainer["distributed"]
                if (not isinstance(distributed, dict) or set(distributed) - {"backend", "timeout_seconds"}
                        or distributed.get("backend") not in ("gloo", "nccl")
                        or type(distributed.get("timeout_seconds", 180)) is not int
                        or distributed.get("timeout_seconds", 180) <= 0):
                    raise ValueError("trainer.distributed requires an explicit gloo/nccl backend and positive timeout")
        else:
            if "sampling" in parts["data"]:
                raise ValueError("data.sampling is supported only by the Program adapter")
            if "validation" in trainer or "validation_prepared" in parts["data"]:
                raise ValueError("periodic validation is supported only by the Program adapter")
            if parts["model"].get("workload") not in tuple(f"W{i}" for i in range(1, 7)):
                raise ValueError("native model.workload must be W1-W6")
        for section, keys in (("data", ("prepared", "validation_prepared")), ("trainer", ("output_dir", "resume_from")),
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
