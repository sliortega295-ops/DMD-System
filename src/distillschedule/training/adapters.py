"""Model-owned functions; execution placement is supplied separately.

An algorithm factory accepts ``(components, feeds, parameters, cycle, seed)``
and returns the existing ``open_workload.torch_backend.Workload`` for ONE cycle.
A model factory accepts a parameter mapping and returns ModelComponents.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelComponents:
    state: dict[str, Any]
    immutable_modules: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    generate: Callable | None = None
    decode: Callable | None = None
    export: Callable | None = None
    load_export: Callable | None = None
    validate_batch: Callable | None = None
    admit_training: Callable | None = None
    collect_metrics: Callable | None = None
    inference_projection: Callable | None = None
    prediction: Any = None
    capabilities: tuple[str, ...] = ()

    def validate(self):
        from .prediction import PredictionSpec
        if self.prediction is not None and not isinstance(self.prediction, PredictionSpec):
            raise ValueError('prediction must be an explicit PredictionSpec')
        if (not isinstance(self.capabilities, tuple)
                or any(not isinstance(item, str) or not item for item in self.capabilities)
                or len(set(self.capabilities)) != len(self.capabilities)):
            raise ValueError('capabilities must be unique nonempty strings')
        if self.prediction is not None or self.capabilities:
            contract = {'prediction': self.prediction.kind if self.prediction is not None else None,
                        'capabilities': sorted(self.capabilities)}
            metadata = dict(self.metadata)
            key = '_distillschedule_model_contract'
            if key in metadata and metadata[key] != contract:
                raise ValueError('model metadata disagrees with effective prediction/capabilities')
            metadata[key] = contract
            self.metadata = metadata
        if not self.state or not all(isinstance(key, str) and key for key in self.state):
            raise ValueError("model state must have explicit nonempty names")
        if self.immutable_modules:
            from distillschedule.open_workload.torch_backend import resolve_module
            for path in self.immutable_modules:
                if not isinstance(path, str) or not path:
                    raise ValueError("immutable_modules must name model subtrees")
                module = resolve_module(self.state, path)
                if any(p.requires_grad for p in module.parameters()):
                    raise ValueError(f"immutable subtree has trainable parameters: {path}")
        for name in ("generate", "decode", "export", "load_export", "validate_batch", "admit_training", "collect_metrics",
                     "inference_projection"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise ValueError(f"model {name} must be callable")
        return self


@dataclass(frozen=True)
class InferenceProjection:
    """Optional export-only view of a RESTORED model's current owned modules.

    ``inference_projection(components, effective_training_parameters)`` must
    not mutate the training owner. The caller resolves the new factory source;
    ordinary named-module export and strict inference identity still apply.
    """
    components: ModelComponents
    factory: str
    parameters: Mapping[str, Any]

    def validate(self):
        import json
        from .plugins import resolve_reference

        if not isinstance(self.components, ModelComponents):
            raise ValueError("inference projection must return ModelComponents")
        self.components.validate()
        from pathlib import Path
        if not isinstance(self.factory, str) or resolve_reference(self.factory, Path.cwd()) != self.factory:
            raise ValueError("inference projection needs an explicit resolved factory reference")
        def keys(value):
            if isinstance(value, Mapping):
                return all(isinstance(key, str) and keys(item) for key, item in value.items())
            if isinstance(value, (list, tuple)):
                return all(keys(item) for item in value)
            return True
        try:
            if not isinstance(self.parameters, Mapping) or not keys(self.parameters) or "device" in self.parameters:
                raise ValueError("invalid projection parameters")
            json.dumps(dict(self.parameters), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("inference projection parameters must be finite JSON without device") from exc
        return self
