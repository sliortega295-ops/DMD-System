"""Explicit source-only admission of the original public CPU helper closure.

The native ``helios`` package keeps its ordinary names and relative imports.
Only this opt-in path bypasses bytecode; legacy native adapters are unchanged.
Installed dependencies remain trusted Python, not a sandboxed substitute.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
from importlib.abc import MetaPathFinder
import json
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType, SimpleNamespace

from .reference_model import _SourceOnlyLoader, reference_model_identity
from distillschedule.workloads.best_known.registry import file_sha256


_LOCK = RLock()
_MODULES = ("helios", "helios.scheduler", "helios.scheduler.scheduling_helios",
            "helios.utils", "helios.utils.utils_base", "helios.utils.utils_recycle_batch",
            "helios.utils.utils_helios_base", "helios.utils.utils_helios_post",
            "helios.utils.train_config", "helios.dataset", "helios.dataset.dataloader_dmd")
_PACKAGES = {"helios", "helios.scheduler", "helios.utils", "helios.dataset"}
_loaded_modules = None
_loaded_bindings = None


def _require(condition, message):
    if not condition:
        raise ValueError("Helios reference helpers: " + message)


def _cached_modules():
    return {name: module for name, module in sys.modules.copy().items()
            if name == "helios" or name.startswith("helios.")}


def _sources(source):
    root = Path(source["transformer_path"]).parents[2]
    return {name: root / (name.replace(".", "/") + ("/__init__.py" if name in _PACKAGES else ".py"))
            for name in _MODULES}


class _HelperSourceFinder(MetaPathFinder):
    def __init__(self, sources, digests):
        self.sources = {name: (path, digests[name]) for name, path in sources.items()}

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "helios" and not fullname.startswith("helios."):
            return None
        _require(fullname in self.sources, "unadmitted original helper import: " + fullname)
        source, digest = self.sources[fullname]
        options = {"submodule_search_locations": [str(source.parent)]} if fullname in _PACKAGES else {}
        return importlib.util.spec_from_file_location(fullname, source,
            loader=_SourceOnlyLoader(fullname, source, digest), **options)


@contextmanager
def _source_imports(sources, digests):
    finder = _HelperSourceFinder(sources, digests)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path[:] = [entry for entry in sys.meta_path if entry is not finder]


def _bindings(modules):
    # As in the cached-AV provider, include cross-module aliases and class
    # methods/code, not just the four top-level loss/rollout entrypoints.
    result = {}
    for module_name, module in modules.items():
        for name, value in vars(module).items():
            owner = getattr(value, "__module__", "")
            own = isinstance(owner, str) and owner.startswith("helios.")
            if ((callable(value) and own) or
                    (isinstance(value, ModuleType) and value.__name__.startswith("helios."))):
                result[(module_name, name)] = value
                if isinstance(value, type) and own:
                    for member, item in vars(value).items():
                        if isinstance(item, (staticmethod, classmethod)):
                            item = item.__func__
                        if callable(item):
                            result[(module_name, name, member)] = item
                            if hasattr(item, "__code__"):
                                result[(module_name, name, member, "__code__")] = item.__code__
                elif hasattr(value, "__code__"):
                    result[(module_name, name, "__code__")] = value.__code__
    return result


def _validate_modules(modules, bindings, sources):
    _require(_cached_modules() == modules and set(modules) == set(sources), "namespace cache changed")
    for name, path in sources.items():
        module = modules[name]
        spec = getattr(module, "__spec__", None)
        _require(getattr(module, "__name__", None) == name
                 and getattr(module, "__file__", None) == str(path)
                 and spec is not None and spec.name == name and spec.origin == str(path),
                 "namespace origin changed: " + name)
        if name != "helios":
            parent, _, child = name.rpartition(".")
            _require(getattr(sys.modules.get(parent), child, None) is module,
                     "namespace parent binding changed: " + name)
        if name in _PACKAGES:
            _require(list(module.__path__) == [str(path.parent)], "package search path changed: " + name)
    current = _bindings(modules)
    _require(current.keys() == bindings.keys() and all(current[key] is value for key, value in bindings.items()),
             "live numerical binding changed")


def _identity(backend):
    source = reference_model_identity(backend=backend)["source"]
    path = Path(source["manifest_path"])
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == source["manifest_sha256"],
             "original helper manifest changed during admission")
    # Compile against the pinned manifest, not hashes sampled again between
    # admission and execution. A concurrent source change must fail closed.
    digests = {row["path"]: row["sha256"] for row in json.loads(data)["files"]}
    return dict(schema_version="distillschedule.helios_reference_helpers.v1", source=source,
                modules={name: digests[file.relative_to(path.parent).as_posix()]
                         for name, file in _sources(source).items()},
                provider_sha256=file_sha256(Path(__file__)))


def reference_helper_identity(*, backend: str):
    """Re-admit pinned source and any cache without importing Torch/helpers."""
    with _LOCK:
        identity = _identity(backend)
        if _loaded_modules is None:
            _require(not _cached_modules(), "unadmitted original helper namespace cache already exists")
        else:
            _validate_modules(_loaded_modules, _loaded_bindings, _sources(identity["source"]))
        return identity


def load_reference_primitives(*, backend: str):
    """Load intact pinned .py modules, refusing unadmitted preloaded helpers."""
    global _loaded_modules, _loaded_bindings
    with _LOCK:
        identity = reference_helper_identity(backend=backend)
        if _loaded_modules is None:
            try:
                sources = _sources(identity["source"])
                with _source_imports(sources, identity["modules"]):
                    for name in _MODULES:
                        importlib.import_module(name)
                modules = _cached_modules()
                bindings = _bindings(modules)
                _require(_identity(backend) == identity,
                         "original helper source changed during import")
                _validate_modules(modules, bindings, sources)
            except BaseException:
                # Entry required an empty native namespace. Do not erase a
                # caller's previously loaded code or dependency modules.
                for name, module in sorted(_cached_modules().items(), key=lambda row: len(row[0]), reverse=True):
                    parent, _, child = name.rpartition(".")
                    owner = sys.modules.get(parent)
                    if owner is not None and getattr(owner, child, None) is module:
                        delattr(owner, child)
                    sys.modules.pop(name, None)
                raise
            _loaded_modules, _loaded_bindings = modules, bindings
        modules = _loaded_modules
        return SimpleNamespace(scheduler=modules["helios.scheduler.scheduling_helios"].HeliosScheduler,
            history=modules["helios.utils.utils_helios_base"], post=modules["helios.utils.utils_helios_post"],
            base=modules["helios.utils.utils_base"], dataset=modules["helios.dataset.dataloader_dmd"].BucketedFeatureDataset)
