"""Lazy, source-admitted full original Helios model with explicit dense SDPA.

Discovery uses only the standard library. Execution imports the complete pinned
transformer in an ordinary private package, never as ``helios.*`` and never via
AST extraction. This provider is not the public W3 trainer, a pretrained model,
or an admission of any native optimized/CUDA attention backend.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
from importlib.abc import MetaPathFinder
from importlib.machinery import SourceFileLoader
import json
from pathlib import Path
import sys
from threading import RLock

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import HELIOS_SHA, file_sha256, repository_root


_NAMESPACE = "distillschedule.adapters.helios._reference"
_MODEL_MODULE = _NAMESPACE + ".modules.transformer_helios"
_BACKEND_MODULE = _NAMESPACE + ".modules.helios_kernels"
_REFERENCE_PATH = Path(__file__).resolve().with_name("_reference")
_BACKEND_PATH = _REFERENCE_PATH / "modules/helios_kernels.py"
_BACKEND_SHA256 = "8965435b63b9712305b2e2d40d53311d33db1682f0ff2c255ae6dccd82d0c71d"
_MANIFEST_SHA256 = "a3d98faab5787f4f9f4e1155eb577235a885f14babc677004affb42877118bd1"
_LOCK = RLock()
_loaded_modules = None
_loaded_objects = None


def _require(condition, message):
    require(condition, message, path="helios.reference_model")


def _source_identity():
    root = repository_root().resolve() / "third_party/helios_compat"
    manifest_path = root / "source_manifest.json"
    _require(file_sha256(manifest_path) == _MANIFEST_SHA256,
             "Helios reference source manifest identity changed")
    manifest = json.loads(manifest_path.read_text())
    _require(manifest["expected_transformer_sha256"] == HELIOS_SHA and manifest["file_count"] == 83,
             "Helios reference original source pin changed")
    for row in manifest["files"]:
        path = root / row["path"]
        _require(path.resolve() == path and path.is_file()
                 and path.stat().st_size == row["bytes"] and file_sha256(path) == row["sha256"],
                 f"Helios reference source identity mismatch: {row['path']}")
    actual = {p.relative_to(root).as_posix() for name in ("source", "reference")
              for p in (root / name).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"} | {"LICENSE.txt"}
    _require(actual == {r["path"] for r in manifest["files"]},
             "Helios reference source inventory changed")
    transformer = root / "source/helios/modules/transformer_helios.py"
    _require(file_sha256(transformer) == HELIOS_SHA, "Helios reference transformer source pin changed")
    return {"transformer_path": str(transformer), "transformer_sha256": HELIOS_SHA,
            "manifest_path": str(manifest_path), "manifest_sha256": _MANIFEST_SHA256,
            "content_tree_sha256": manifest["content_tree_sha256"],
            "source_git_commit": manifest["source_git_commit"], "files_checked": len(manifest["files"])}


def _cached_modules():
    return {name: module for name, module in sys.modules.copy().items()
            if name == _NAMESPACE or name.startswith(_NAMESPACE + ".")}


def _module_paths(source):
    return {_NAMESPACE: _REFERENCE_PATH / "__init__.py",
            _NAMESPACE + ".modules": _REFERENCE_PATH / "modules/__init__.py",
            _BACKEND_MODULE: _BACKEND_PATH, _MODEL_MODULE: Path(source["transformer_path"])}


class _SourceOnlyLoader(SourceFileLoader):
    """Normal module execution, but compile admitted source, never cached code."""

    def __init__(self, name, path, sha256):
        super().__init__(name, str(path))
        self.source_sha256 = sha256

    def get_code(self, fullname):
        data = self.get_data(self.path)
        _require(fullname == self.name and hashlib.sha256(data).hexdigest() == self.source_sha256,
                 f"Helios reference import source changed: {fullname}")
        # The regular loader may accept timestamp-valid or unchecked-hash .pyc
        # with different code despite a valid source-file digest. Bypass only
        # that bytecode path; preserve the normal spec/module/package protocol.
        return self.source_to_code(data, self.path)


class _ReferenceSourceFinder(MetaPathFinder):
    def __init__(self, source):
        paths = _module_paths(source)
        self.sources = {name: (path, file_sha256(path)) for name, path in paths.items()}
        self.sources[_MODEL_MODULE] = (paths[_MODEL_MODULE], source["transformer_sha256"])
        self.sources[_BACKEND_MODULE] = (paths[_BACKEND_MODULE], _BACKEND_SHA256)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _NAMESPACE and not fullname.startswith(_NAMESPACE + "."):
            return None
        _require(fullname in self.sources, f"unadmitted Helios reference import: {fullname}")
        source, sha256 = self.sources[fullname]
        options = {"submodule_search_locations": [str(source.parent)]} if source.name == "__init__.py" else {}
        return importlib.util.spec_from_file_location(fullname, source,
            loader=_SourceOnlyLoader(fullname, source, sha256), **options)


@contextmanager
def _source_imports(source):
    finder = _ReferenceSourceFinder(source)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        # Remove only our exact temporary hook, not other importers installed
        # by dependencies or another thread during the normal model import.
        sys.meta_path[:] = [entry for entry in sys.meta_path if entry is not finder]


def _validate_modules(modules, objects, source):
    transformer = Path(source["transformer_path"])
    expected = _module_paths(source)
    _require(set(modules) == set(expected) and _cached_modules() == modules,
             "Helios reference namespace cache changed")
    for name, path in expected.items():
        module = modules[name]
        spec = getattr(module, "__spec__", None)
        _require(isinstance(getattr(module, "__file__", None), str)
                 and Path(module.__file__).resolve() == path
                 and spec is not None and isinstance(getattr(spec, "origin", None), str)
                 and Path(spec.origin).resolve() == path
                 and getattr(module, "__name__", None) == name and getattr(spec, "name", None) == name,
                 f"Helios reference namespace origin changed: {name}")
        parent, _, child = name.rpartition(".")
        _require(getattr(sys.modules.get(parent), child, None) is module,
                 f"Helios reference namespace parent binding changed: {name}")
    for name, paths in ((_NAMESPACE, [_REFERENCE_PATH]),
                        (_NAMESPACE + ".modules", [_REFERENCE_PATH / "modules", transformer.parent])):
        found = getattr(modules[name], "__path__", None)
        _require(isinstance(found, (list, tuple)) and [Path(p).resolve() for p in found] == paths,
                 f"Helios reference namespace search path changed: {name}")
    model, backend = modules[_MODEL_MODULE], modules[_BACKEND_MODULE]
    _require(getattr(model, "HeliosTransformer3DModel", None) is objects["class"]
             and objects["class"].__module__ == _MODEL_MODULE
             and getattr(objects["class"], "forward", None) is objects["forward"]
             and getattr(model, "attn_varlen_func", None) is objects["attention"]
             and getattr(backend, "attn_varlen_func", None) is objects["attention"]
             and getattr(model, "create_navit_attention_masks", None) is objects["navit"]
             and getattr(backend, "create_navit_attention_masks", None) is objects["navit"],
             "Helios reference model/backend binding changed")


def _admit(backend):
    _require(backend == "torch-sdpa", "explicit backend='torch-sdpa' required; no optimized-backend fallback")
    source = _source_identity()
    _require(_BACKEND_PATH.is_file() and file_sha256(_BACKEND_PATH) == _BACKEND_SHA256,
             "Helios reference backend source identity changed")
    if _loaded_modules is None:
        _require(not _cached_modules(), "unadmitted Helios reference namespace cache already exists")
    else:
        _validate_modules(_loaded_modules, _loaded_objects, source)
    return {"schema_version": "distillschedule.helios_reference_model.v1",
            "scope": "FULL_ORIGINAL_SOURCE_EXPLICIT_DENSE_SDPA_NOT_NATIVE_W3_LIFECYCLE",
            "namespace": _NAMESPACE, "model_module": _MODEL_MODULE,
            "class_name": "HeliosTransformer3DModel", "source": source,
            "backend": {"name": "torch-sdpa", "module": _BACKEND_MODULE,
                        "path": str(_BACKEND_PATH), "sha256": _BACKEND_SHA256, "varlen": False}}


def reference_model_identity(*, backend: str) -> dict:
    """Recheck all original bytes, backend bytes and any admitted import cache.

    Does not import Torch, the private namespace, or the model. A source checkout
    is required, just like the existing original-native adapters.
    """
    with _LOCK:
        return _admit(backend)


def _discard_failed_import():
    # Admission required an empty namespace on entry. Only this failed import's
    # private modules/parent attributes are removed; native helios.* is untouched.
    for name, module in sorted(_cached_modules().items(), key=lambda row: len(row[0]), reverse=True):
        parent, _, child = name.rpartition(".")
        owner = sys.modules.get(parent)
        if owner is not None and getattr(owner, child, None) is module:
            delattr(owner, child)
        sys.modules.pop(name, None)


def load_reference_model_class(*, backend: str) -> type:
    """Return the actual original class, with explicit opt-in to dense SDPA.

    Constructor, PEFT, GAN head and save_pretrained/from_pretrained remain owned
    by the original class. No model, weights or optimizer are created here.
    Unsupported NaViT/varlen execution raises, never substitutes another route.
    """
    global _loaded_modules, _loaded_objects
    with _LOCK:
        identity = _admit(backend)
        if _loaded_modules is None:
            try:
                with _source_imports(identity["source"]):
                    model = importlib.import_module(_MODEL_MODULE)
                    backend_module = sys.modules[_BACKEND_MODULE]
                    modules = _cached_modules()
                    objects = {"class": model.HeliosTransformer3DModel,
                               "forward": model.HeliosTransformer3DModel.forward,
                               "attention": backend_module.attn_varlen_func,
                               "navit": backend_module.create_navit_attention_masks}
                    source = _source_identity()
                    _require(source == identity["source"] and file_sha256(_BACKEND_PATH) == _BACKEND_SHA256,
                             "Helios reference source/backend identity changed during import")
                    _validate_modules(modules, objects, source)
            except BaseException:
                _discard_failed_import()
                raise
            _loaded_modules, _loaded_objects = modules, objects
        return _loaded_objects["class"]
