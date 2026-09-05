"""Canonical, provenance-backed workload profiles. No model import side effects."""
from .registry import load_profile, list_profiles, resolve_profile

__all__ = ["load_profile", "list_profiles", "resolve_profile"]
