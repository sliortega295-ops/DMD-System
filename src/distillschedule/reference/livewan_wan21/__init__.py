"""Lazy, licensed full Wan2.1/LiveWan numerical source, explicit CPU FP32 only.

This is NOT native BF16/FlashAttention qualification, a trainer, pretrained
weights, world16, or C51 restart. Original constructors and numerical bodies are
retained. Consumers must explicitly use CPU/float32, T2V, cuda_graphs=False and
``generate_block`` rather than the original CUDA-timed ``stream`` convenience.
The original TeacherLatents pickle API is trusted-input-only, not safe ingestion.
Discovery imports only stdlib; load_cpu imports classes but constructs nothing.
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
_NAMESPACE = __name__
_MANIFEST_SHA256 = "8803891e7c20a6cc40ce4d17a650877fceefb722be8063bde56427ee418869cf"
_LOCK = RLock()
_importing = False
_loaded_modules = None
_loaded_bindings = None
_api = None
_API_MODULES = dict(model="wan.modules.model", attention="wan.modules.attention",
                    blockcausal="wanstreamer.blockcausal", stream="wanstreamer.stream",
                    core="wanstreamer.core", rope="wanstreamer.rope", kvcache="wanstreamer.kvcache",
                    dmd="wanstreamer.dmd", lora="wanstreamer.lora", data="wanstreamer.data",
                    graphrunner="wanstreamer.graphrunner")


def _require(condition, message):
    if not condition:
        raise ValueError("livewan_wan21: " + message)


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
    actual = {p.relative_to(_ROOT).as_posix() for p in _ROOT.rglob("*")
              if p.is_file() and "__pycache__" not in p.relative_to(_ROOT).parts
              and p.suffix != ".pyc" and p != _ROOT / "__init__.py"}
    rows = manifest["files"] + manifest["licenses"]
    _require(actual == {r["path"] for r in rows} | {"source_manifest.json"},
             "source inventory identity changed")
    for row in rows:
        source = _ROOT / row["path"]
        _require(source.is_file() and source.resolve() == source
                 and source.stat().st_size == row["bytes"] and _hash(source) == row["sha256"],
                 "source/license identity mismatch: " + row["path"])
    return manifest


def _module_paths(manifest):
    result = {}
    for row in manifest["files"]:
        module = row["path"][:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[:-9]
        result[_NAMESPACE + "." + module] = (_ROOT / row["path"], row["sha256"])
    return result


def _bindings(modules):
    """Snapshot source-owned exports, methods/code, wrappers and dispatch maps."""
    result = {}

    def capture(key, value):
        owner = getattr(value, "__module__", "")
        own = isinstance(owner, str) and owner.startswith(_NAMESPACE + ".")
        if (callable(value) and own) or (isinstance(value, ModuleType)
                                        and value.__name__.startswith(_NAMESPACE + ".")):
            result[key] = value
            if isinstance(value, type):
                for name, item in vars(value).items():
                    if isinstance(item, (staticmethod, classmethod)):
                        item = item.__func__
                    if callable(item):
                        capture(key + (name,), item)
            elif hasattr(value, "__code__"):
                result[key + ("__code__",)] = value.__code__
                if hasattr(value, "__wrapped__"):
                    capture(key + ("__wrapped__",), value.__wrapped__)
        elif isinstance(value, dict):
            for name, item in value.items():
                if isinstance(name, str):
                    capture(key + (name,), item)

    for module_name, module in modules.items():
        for name, value in vars(module).items():
            if not name.startswith("__"):
                capture((module_name, name), value)
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
    _api.attention._validate_backend()


def source_identity() -> dict:
    """Admit exact packaged source/licenses and any live numerical bindings.

    No external checkout, Torch import, model construction or weight read. This
    API records source admission, not numerical/pretrained/native qualification.
    """
    with _LOCK:
        manifest = _source_manifest()
        if _loaded_modules is None:
            _require(not _cached_modules(), "unadmitted namespace cache already exists")
        else:
            _validate_modules(manifest)
        return dict(schema_version=manifest["schema_version"], scope=manifest["scope"],
                    backend="torch-sdpa-cpu", precision="float32", manifest_sha256=_MANIFEST_SHA256,
                    files_checked=len(manifest["files"]), licenses_checked=len(manifest["licenses"]),
                    origins=manifest["origins"], unsupported=manifest["unsupported"],
                    checkpoint_trust=manifest["checkpoint_trust"])


def _require_execution():
    _require(_importing or _loaded_modules is not None,
             "numerical imports require load_cpu(backend='torch-sdpa-cpu', precision='float32') admission")


def _environment():
    _require(os.environ.get("WAN_SDPA_BACKEND", "").strip().upper() in ("", "MATH"),
             "only CPU MATH attention backend is admitted; WAN_SDPA_BACKEND override refused")


class _SourceOnlyLoader(SourceFileLoader):
    """Normal packages and source execution, never timestamp/hash-valid pyc."""

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


def load_cpu(*, backend: str, precision: str) -> SimpleNamespace:
    """Return complete original model/streaming/DMD/LoRA source module API.

    This constructs no models or weights. Only the explicit FP32 CPU contract
    is admitted. Raw original APIs retain their upstream defaults and unsafe
    conveniences; pass dtype=torch.float32 and cuda_graphs=False explicitly.
    """
    global _importing, _loaded_modules, _loaded_bindings, _api
    with _LOCK:
        _require(backend == "torch-sdpa-cpu", "explicit backend='torch-sdpa-cpu' required")
        _require(precision == "float32", "explicit precision='float32' required")
        source_identity()
        _environment()
        torch = sys.modules.get("torch")
        _require(torch is None or not torch.cuda.is_initialized(), "CPU reference cannot admit initialized CUDA")
        if _loaded_modules is not None:
            return _api
        manifest = _source_manifest()
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
        return _api
