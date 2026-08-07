"""Typed DMD runtime facade introduced by the first DistillSchedule refactor."""

from .adapters import default_registry
from .core.config import LoadedPlan, load_plan
from .core.plan import ExecutionPlan
from .runtime import DistillExecutor

__version__ = "0.1.0"

__all__ = [
    "DistillExecutor",
    "ExecutionPlan",
    "LoadedPlan",
    "default_registry",
    "load_plan",
]
