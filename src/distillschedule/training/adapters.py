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
        if not set(self.immutable_modules) <= set(self.state):
            raise ValueError("immutable_modules must name model state")
        for name in ("generate", "decode", "export"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise ValueError(f"model {name} must be callable")
        return self
