"""Trusted local plugins execute their recorded source, not stale bytecode."""
from __future__ import annotations

import hashlib
import importlib.util
from importlib._bootstrap_external import _code_to_hash_pyc, _code_to_timestamp_pyc
from pathlib import Path
import sys
from types import ModuleType

import pytest

from distillschedule.training.plugins import load_callable


def module_name(path):
    return "dmd_user_" + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("invalidation", ["timestamp", "unchecked-hash"])
def test_local_plugin_executes_recorded_source_not_cached_payload(tmp_path, invalidation):
    source = tmp_path / "adapter.py"
    source.write_text("def build(value):\n    return 'recorded source'\n")
    data, stat = source.read_bytes(), source.stat()
    code = compile("def build(value):\n    return 'unrecorded bytecode'\n", str(source), "exec")
    payload = (_code_to_timestamp_pyc(code, int(stat.st_mtime), len(data))
               if invalidation == "timestamp" else
               _code_to_hash_pyc(code, importlib.util.source_hash(data), checked=False))
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir()
    cache.write_bytes(payload)
    name = module_name(source)
    try:
        function, receipt = load_callable(str(source) + ":build")
        assert function(None) == "recorded source", "plugin executed unrecorded cached code"
        assert receipt == {"reference": str(source) + ":build", "sha256": hashlib.sha256(data).hexdigest()}
        assert source.read_bytes() == data and cache.read_bytes() == payload
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("failure", ["raise", "missing", "change-source"])
def test_local_plugin_failure_discards_only_its_partial_module(tmp_path, failure):
    source = tmp_path / "adapter.py"
    content = {
        "raise": "raise RuntimeError('explicit plugin failure')\n",
        "missing": "build = None\n",
        "change-source": ("from pathlib import Path\n"
                          "Path(__file__).write_text('changed during import')\n"
                          "def build(value): return value\n"),
    }[failure]
    source.write_text(content)
    name = module_name(source)
    unrelated = ModuleType("unrelated_plugin_fixture")
    sys.modules[unrelated.__name__] = unrelated
    try:
        with pytest.raises(RuntimeError if failure == "raise" else ValueError):
            load_callable(str(source) + ":build")
        assert name not in sys.modules, "failed plugin left a partial hashed module"
        assert sys.modules[unrelated.__name__] is unrelated
        source.write_text("def build(value): return value + 1\n")
        function, _ = load_callable(str(source) + ":build")
        assert function(2) == 3
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop(module_name(source), None)
        sys.modules.pop(unrelated.__name__, None)


def test_local_plugin_failure_restores_prior_module_binding(tmp_path):
    source = tmp_path / "adapter.py"
    source.write_text("raise RuntimeError('explicit plugin failure')\n")
    name = module_name(source)
    previous = ModuleType(name)
    previous.marker = "previous admitted object"
    sys.modules[name] = previous
    try:
        with pytest.raises(RuntimeError, match="explicit plugin failure"):
            load_callable(str(source) + ":build")
        assert sys.modules.get(name) is previous
    finally:
        sys.modules.pop(name, None)


def test_local_plugin_preserves_normal_spec_dataclass_and_self_import(tmp_path):
    source = tmp_path / "adapter.py"
    source.write_text("from dataclasses import dataclass\nimport sys\n"
                      "@dataclass\nclass Value:\n    number: int\n"
                      "def build(value):\n    return Value(value), sys.modules[__name__]\n")
    name = module_name(source)
    cache = Path(importlib.util.cache_from_source(str(source)))
    try:
        function, receipt = load_callable(str(source) + ":build")
        value, module = function(7)
        assert value.number == 7 and value.__class__ is module.Value
        assert module is sys.modules[name] and module.__spec__.name == name
        assert module.__file__ == str(source) and module.__spec__.origin == str(source)
        assert receipt["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert not cache.exists(), "source execution must not create bytecode"
    finally:
        sys.modules.pop(name, None)


def test_package_module_plugin_path_and_receipt_are_unchanged():
    from distillschedule.training import plugins
    function, receipt = load_callable("distillschedule.training.plugins:resolve_reference")
    assert function is plugins.resolve_reference
    assert receipt == {"reference": "distillschedule.training.plugins:resolve_reference",
                       "sha256": hashlib.sha256(Path(plugins.__file__).read_bytes()).hexdigest()}
