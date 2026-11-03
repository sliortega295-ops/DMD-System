"""The fast suite and source identities used by public native entry points."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from distillschedule.evidence.source_layout import reviewed_imports

ROOT = Path(__file__).resolve().parents[1]


def runner():
    spec = importlib.util.spec_from_file_location(
        "core_test_runner", ROOT / "scripts/run_framework_cpu_tests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_core_selection_is_complete_and_ignores_extra_files(tmp_path):
    module = runner()
    assert set(module.test_files(ROOT)) == {
        str(path.relative_to(ROOT)) for path in (ROOT / "tests").glob("test_*.py")}
    tests = tmp_path / "tests"
    tests.mkdir()
    for name in module.CORE_TESTS:
        (tests / name).touch()
    (tests / "test_optional_model.py").touch()
    assert len(module.test_files(tmp_path)) == len(module.CORE_TESTS)
    (tests / module.CORE_TESTS[0]).unlink()
    with pytest.raises(ValueError, match="missing core tests"):
        module.test_files(tmp_path)


def test_native_source_inventory_resolves_to_unchanged_bytes():
    rows = reviewed_imports(ROOT)
    assert rows
    for row in rows:
        path = ROOT / row["target"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["integrated_sha256"], path


def test_retired_imports_are_absent_and_still_hash_bound(tmp_path):
    source = tmp_path / "docs/workloads"
    source.mkdir(parents=True)
    path = "tests/retired.py"
    digest = hashlib.sha256(b"original source").hexdigest()
    (source / "source_import_map.json").write_text(json.dumps({
        "files": [{"target": path, "integrated_sha256": digest}]}))
    lock = {"schema_version": "distillschedule.source_layout_lock.v1", "files": {},
            "retired_imports": {path: digest}}
    lockpath = source / "source_layout.lock.json"
    lockpath.write_text(json.dumps(lock))
    assert reviewed_imports(tmp_path) == []
    (tmp_path / "tests").mkdir()
    (tmp_path / path).write_text("original source")
    with pytest.raises(ValueError, match="retired import hash changed or file is still active"):
        reviewed_imports(tmp_path)
    (tmp_path / path).unlink()
    lock["retired_imports"][path] = "0" * 64
    lockpath.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="retired import hash changed or file is still active"):
        reviewed_imports(tmp_path)
