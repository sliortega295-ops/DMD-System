"""Model-owned functions; execution placement is supplied separately.

An algorithm factory accepts ``(components, feeds, parameters, cycle, seed)``
and returns the existing ``open_workload.torch_backend.Workload`` for ONE cycle.
A model factory accepts a parameter mapping and returns ModelComponents.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass
class ModelComponents:
    state: dict[str, Any]
    immutable_modules: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    generate: Callable | None = None
    decode: Callable | None = None
    export: Callable | None = None

    def validate(self):
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
        for name in ("generate", "decode", "export"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise ValueError(f"model {name} must be callable")
        return self
