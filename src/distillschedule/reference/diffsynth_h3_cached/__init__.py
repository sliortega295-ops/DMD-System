"""Lazy admission of a maintained, cached-AV-only DiffSynth H3 derivative.

This is not a complete or pristine DiffSynth distribution, the frozen W6 native
benchmark, or a public training lifecycle. No model/download is performed here.
Call ``load_cpu(backend="torch-dense-cpu")`` to import the numerical closure.
Execution requires the optional ``h3-cached-reference`` dependency extra.
The original trusted-pickle checkpoint API is deliberately not made default-safe.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
from importlib.abc import MetaPathFinder
from importlib.machinery import SourceFileLoader
import json
import os
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType, SimpleNamespace


_ROOT = Path(__file__).resolve().parent
_MANIFEST_SHA256 = "527591fce99ba50753db9e148b9bc07029c04ebd417c0e262bbf6d6211234089"
_NAMESPACE = __name__
_LOCK = RLock()
_importing = False
_loaded_modules = None
_loaded_bindings = None
_api = None
_API_MODULES = dict(model="models.minimax_h3_dit", attention="core.attention.attention",
                    dmd2="h3_dmd2", packing="packing", cache="cache", weights="weights",
                    training="training_api", support="support", zero3="dmd2_zero3")
_OPTIMIZED_MODULES = ("flash_attn", "flash_attn_interface", "sageattention", "xformers")


def _require(condition, message):
    if not condition:
        raise ValueError("diffsynth_h3_cached: " + message)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cached_modules():
    return {name: module for name, module in sys.modules.copy().items()
            if name.startswith(_NAMESPACE + ".")}


def _source_manifest():
    path = _ROOT / "source_manifest.json"
    _require(path.is_file() and path.resolve() == path and _hash(path) == _MANIFEST_SHA256,
             "source manifest identity changed")
    manifest = json.loads(path.read_text())
    files = {row["path"] for row in manifest["files"]}
    actual = {p.relative_to(_ROOT).as_posix() for p in _ROOT.rglob("*.py")
              if p != _ROOT / "__init__.py"}
    _require(actual == files, "source inventory identity changed")
    for row in manifest["files"]:
        source = _ROOT / row["path"]
        _require(source.is_file() and source.resolve() == source
                 and source.stat().st_size == row["bytes"] and _hash(source) == row["sha256"],
                 "source identity mismatch: " + row["path"])
    return manifest


def _module_paths(manifest):
    result = {}
    for row in manifest["files"]:
        path = row["path"]
        module = path[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[:-9]
        result[_NAMESPACE + "." + module] = (_ROOT / path, row["sha256"])
    return result


def _bindings(modules):
    """Capture live source-owned/imported numerical bindings, not model state."""
    result = {}
    for module_name, module in modules.items():
        for name, value in vars(module).items():
            owner = getattr(value, "__module__", "")
            own = isinstance(owner, str) and owner.startswith(_NAMESPACE + ".")
            if (callable(value) and own) or (isinstance(value, ModuleType) and value.__name__.startswith(_NAMESPACE + ".")):
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


def _validate_modules(manifest):
    modules = _cached_modules()
    _require(modules == _loaded_modules, "namespace cache binding changed")
    expected = _module_paths(manifest)
    _require(set(modules) == set(expected), "namespace inventory binding changed")
    for name, (path, _) in expected.items():
        module = modules[name]
        spec = getattr(module, "__spec__", None)
        _require(getattr(module, "__name__", None) == name
                 and getattr(module, "__file__", None) == str(path)
                 and spec is not None and spec.name == name and spec.origin == str(path),
                 "namespace origin binding changed: " + name)
        parent, _, child = name.rpartition(".")
        _require(getattr(sys.modules.get(parent), child, None) is module,
                 "namespace parent binding changed: " + name)
        if path.name == "__init__.py":
            _require(list(module.__path__) == [str(path.parent)], "package path binding changed: " + name)
    bindings = _bindings(modules)
    _require(bindings.keys() == _loaded_bindings.keys()
             and all(bindings[key] is value for key, value in _loaded_bindings.items()),
             "live numerical binding changed")
    _require(vars(_api).keys() == _API_MODULES.keys()
             and all(getattr(_api, key) is modules[_NAMESPACE + "." + value]
                     for key, value in _API_MODULES.items()), "provider API binding changed")
    _require(_api.attention.ATTENTION_IMPLEMENTATION == "torch", "attention backend binding changed")


def source_identity() -> dict:
    """Validate the packaged source manifest/bytes with no Torch/CUDA import.

    Original byte/AST identities and every maintained transformation are shipped
    in source_manifest.json; private archive paths are not required by CI.
    """
    with _LOCK:
        manifest = _source_manifest()
        if _loaded_modules is not None:
            _validate_modules(manifest)
        else:
            _require(not _cached_modules(), "unadmitted namespace cache already exists")
        return dict(schema_version=manifest["schema_version"], scope=manifest["scope"],
                    manifest_sha256=_MANIFEST_SHA256, files_checked=len(manifest["files"]),
                    canonical_gan_start_step=500, qualified_world_size=1,
                    checkpoint_trust=manifest["checkpoint_trust"],
                    not_run=manifest["not_run"], qualified_protocol=manifest["qualified_protocol"].copy())


def _require_execution():
    _require(_importing or _loaded_modules is not None,
             "numerical imports require load_cpu(backend='torch-dense-cpu') admission")


def _attention_environment():
    _require(not any(key.startswith("DIFFSYNTH_FLASH_ATTN_KERNEL") for key in os.environ),
             "remote attention kernel settings are not admitted")
    _require(os.environ.get("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch").lower() == "torch",
             "only explicit dense Torch attention is admitted")
    # The unchanged attention module probes optional implementations at import.
    # Reject their presence instead of importing them or quietly using whatever
    # happened to be installed. kernels is never imported without a remote id.
    for name in _OPTIMIZED_MODULES:
        _require(not any(key == name or key.startswith(name + ".") for key in sys.modules),
                 "optimized attention module is already loaded: " + name)
        _require(importlib.util.find_spec(name) is None,
                 "optimized attention installation is not admitted: " + name)


class _SourceOnlyLoader(SourceFileLoader):
    """Use normal module/spec/package execution, but never unverified .pyc."""

    def __init__(self, name, path, sha256):
        super().__init__(name, str(path))
        self.source_sha256 = sha256

    def get_code(self, fullname):
        data = self.get_data(self.path)
        _require(fullname == self.name and hashlib.sha256(data).hexdigest() == self.source_sha256,
                 "import source identity changed: " + fullname)
        return self.source_to_code(data, self.path)


class _SourceFinder(MetaPathFinder):
    def __init__(self, manifest):
        self.sources = _module_paths(manifest)

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_NAMESPACE + "."):
            return None
        _require(fullname in self.sources, "unadmitted numerical import: " + fullname)
        source, digest = self.sources[fullname]
        options = {"submodule_search_locations": [str(source.parent)]} if source.name == "__init__.py" else {}
        return importlib.util.spec_from_file_location(fullname, source,
            loader=_SourceOnlyLoader(fullname, source, digest), **options)


@contextmanager
def _source_imports(manifest):
    finder = _SourceFinder(manifest)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path[:] = [entry for entry in sys.meta_path if entry is not finder]


def _discard_failed_import():
    for name, module in sorted(_cached_modules().items(), key=lambda pair: len(pair[0]), reverse=True):
        parent, _, child = name.rpartition(".")
        owner = sys.modules.get(parent)
        if owner is not None and getattr(owner, child, None) is module:
            delattr(owner, child)
        sys.modules.pop(name, None)


def load_cpu(*, backend: str) -> SimpleNamespace:
    """Import actual original numerical modules after explicit CPU admission.

    The returned modules construct nothing. Consumers must supply their own tiny
    architecture, tensors and trusted checkpoints. The exact original config has
    critic_updates=5 by default; qualified cached W6 explicitly supplies 1.
    """
    global _importing, _loaded_modules, _loaded_bindings, _api
    with _LOCK:
        _require(backend == "torch-dense-cpu", "explicit backend='torch-dense-cpu' required")
        source_identity()
        _attention_environment()
        if _loaded_modules is not None:
            _require(not sys.modules["torch"].cuda.is_initialized(), "CPU reference cannot admit initialized CUDA")
            return _api
        manifest = _source_manifest()
        previous = os.environ.get("DIFFSYNTH_ATTENTION_IMPLEMENTATION")
        os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"
        _importing = True
        try:
            with _source_imports(manifest):
                api = SimpleNamespace(**{key: importlib.import_module(_NAMESPACE + "." + value)
                                         for key, value in _API_MODULES.items()})
                _require(not sys.modules["torch"].cuda.is_initialized(), "CPU reference cannot admit initialized CUDA")
                _require(_source_manifest() == manifest, "source identity changed during import")
                _loaded_modules, _loaded_bindings, _api = _cached_modules(), _bindings(_cached_modules()), api
                _validate_modules(manifest)
        except BaseException:
            _loaded_modules, _loaded_bindings, _api = None, None, None
            _discard_failed_import()
            raise
        finally:
            _importing = False
            if previous is None:
                os.environ.pop("DIFFSYNTH_ATTENTION_IMPLEMENTATION", None)
            else:
                os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = previous
        return _api
