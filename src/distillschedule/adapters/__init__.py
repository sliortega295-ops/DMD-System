"""Model-family adapters."""

from .base import AdapterRegistry, CompiledAdapterPlan, DMDModelAdapter, LegacyInvocation
from .generic import GenericDMDAdapter
from .helios import HeliosDMDAdapter
from .livewan import LiveWanSFDMDAdapter


def default_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register(GenericDMDAdapter())
    registry.register(HeliosDMDAdapter())
    registry.register(LiveWanSFDMDAdapter())
    return registry


__all__ = [
    "AdapterRegistry",
    "CompiledAdapterPlan",
    "DMDModelAdapter",
    "GenericDMDAdapter",
    "HeliosDMDAdapter",
    "LiveWanSFDMDAdapter",
    "LegacyInvocation",
    "default_registry",
]
