"""Model-family adapters."""

from .base import AdapterRegistry, CompiledAdapterPlan, DMDModelAdapter, LegacyInvocation
from .generic import GenericDMDAdapter
from .helios import HeliosDMDAdapter


def default_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register(GenericDMDAdapter())
    registry.register(HeliosDMDAdapter())
    return registry


__all__ = [
    "AdapterRegistry",
    "CompiledAdapterPlan",
    "DMDModelAdapter",
    "GenericDMDAdapter",
    "HeliosDMDAdapter",
    "LegacyInvocation",
    "default_registry",
]
