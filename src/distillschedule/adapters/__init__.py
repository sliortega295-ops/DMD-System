"""Model-family adapters."""

from .base import AdapterRegistry, CompiledAdapterPlan, DMDModelAdapter, LegacyInvocation
from .generic import GenericDMDAdapter
from .livewan import LiveWanSFDMDAdapter


def default_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register(GenericDMDAdapter())
    registry.register(LiveWanSFDMDAdapter())
    return registry


__all__ = [
    "AdapterRegistry",
    "CompiledAdapterPlan",
    "DMDModelAdapter",
    "GenericDMDAdapter",
    "LiveWanSFDMDAdapter",
    "LegacyInvocation",
    "default_registry",
]
