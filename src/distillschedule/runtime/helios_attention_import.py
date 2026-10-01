"""Bind the registered H100 backend before frozen Helios auto-detection.

The pinned dispatcher otherwise selects FA3 on Hopper even for a registered
SDPA/FA2 run. Only its startup kernel lookup is scoped; attention functions,
CUDA capability, external source files and shared environments are unchanged.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys
from importlib.machinery import PathFinder
from pathlib import Path
from types import ModuleType
from typing import Any

FA2 = "kernels-community/flash-attn2"
FA3 = "kernels-community/flash-attn3"
DISPATCH_MODULE = "helios.modules.helios_kernels.attention_dispatch"
DISPATCH_SHA256 = "090e0cf5894eb3d57d6bbb609b5346ba909ea3a9422be8e3abc49f89518f6daf"


def import_registered_attention_backend(expected: str) -> tuple[ModuleType, dict[str, Any]]:
    """Import once, fail closed, and restore the original loader in finally."""
    if expected not in ("sdpa", "local-fa2"):
        raise ValueError(f"unsupported registered attention backend: {expected}")
    mapping = os.environ.get("LOCAL_KERNELS", "")
    if (expected == "sdpa" and mapping) or (
        expected == "local-fa2" and (not mapping.startswith(FA2 + "=") or ":" in mapping)
    ):
        raise ValueError("registered attention backend requires its exact LOCAL_KERNELS policy")
    if DISPATCH_MODULE in sys.modules:
        raise RuntimeError("Helios attention dispatcher already imported; cannot switch a live backend")
    spec = PathFinder.find_spec("helios", sys.path)
    roots = list(spec.submodule_search_locations or ()) if spec is not None else []
    if len(roots) != 1:
        raise RuntimeError("exactly one frozen Helios package root is required")
    source = Path(roots[0]) / "modules/helios_kernels/attention_dispatch.py"
    observed = hashlib.sha256(source.read_bytes()).hexdigest()
    if observed != DISPATCH_SHA256:
        raise RuntimeError(f"Helios attention dispatcher source SHA changed: {observed}")

    import kernels

    original = kernels.get_kernel
    declined: list[str] = []
    loaded: list[str] = []

    def selected_loader(repo: str, *args: Any, **kwargs: Any) -> ModuleType:
        if repo == FA3 or (expected == "sdpa" and repo == FA2):
            # Enter the existing native ImportError fallback, without a Hub
            # request or disguising a real failure of the selected FA2 loader.
            declined.append(repo)
            raise ImportError(f"not the registered {expected} backend: {repo}")
        if repo != FA2:
            raise RuntimeError(f"unexpected kernel lookup in frozen dispatcher: {repo}")
        loaded.append(repo)
        return original(repo, *args, **kwargs)

    dispatcher = None
    try:
        kernels.get_kernel = selected_loader
        dispatcher = importlib.import_module(DISPATCH_MODULE)
    finally:
        kernels.get_kernel = original
        if dispatcher is not None and getattr(dispatcher, "get_kernel", None) is selected_loader:
            dispatcher.get_kernel = original

    if expected == "sdpa":
        if loaded or any(
            getattr(dispatcher, name, None) is not None
            for name in ("flash_attn_func", "flash_attn_varlen_func", "sageattn", "sageattn_varlen", "xformers_attn_func")
        ):
            raise RuntimeError("registered SDPA imported an external attention callable")
    elif loaded != [FA2] or any(
        not callable(getattr(dispatcher, name, None))
        or "flash_attn2" not in str(getattr(getattr(dispatcher, name, None), "__module__", ""))
        for name in ("flash_attn_func", "flash_attn_varlen_func")
    ):
        raise RuntimeError("registered FA2 did not load both pinned FA2 callables")
    return dispatcher, {
        "status": "PASS", "scope": "H100_REGISTERED_BACKEND_IMPORT_ONLY",
        "requested": expected, "source_path": str(source.resolve()), "source_sha256": observed,
        "declined_repositories": declined, "loaded_repositories": loaded,
        "loader_restored": kernels.get_kernel is original,
        "frozen_source_modified": False, "hardware_capability_modified": False,
    }
