"""The fast suite and source identities used by public native entry points."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

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
    result = subprocess.run([sys.executable, "-m", "distillschedule.evidence.source_layout",
                             "--repo", str(ROOT), "--check"],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"status": "PASS", "files_checked": len(rows)}


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


def test_compact_source_layout_resolves_current_bytes_and_rejects_bad_paths(tmp_path):
    directory = tmp_path / "docs/workloads"
    directory.mkdir(parents=True)
    original = {"target": "old/helper.py", "integrated_sha256": "a" * 64,
                "source_repository": "owner/project", "source_commit": "1" * 40,
                "source_path": "helper.py"}
    (directory / "source_import_map.json").write_text(json.dumps({"files": [original]}))
    lock = {"schema_version": "distillschedule.source_layout_lock.v2",
            "files": {"old/helper.py": {"target": "runtime/helper.py",
                       "integrated_sha256": "b" * 64, "source_commit": "2" * 40}},
            "retired_imports": []}
    path = directory / "source_layout.lock.json"
    path.write_text(json.dumps(lock))
    assert reviewed_imports(tmp_path) == [{**original, "target": "runtime/helper.py",
                                         "integrated_sha256": "b" * 64}]
    for target, digest, revision in (("../escape.py", "b" * 64, "2" * 40),
                                     ("runtime/helper.py", "not-a-hash", "2" * 40),
                                     ("runtime/helper.py", "b" * 64, "0" * 40)):
        lock["files"]["old/helper.py"].update(target=target, integrated_sha256=digest,
                                             source_commit=revision)
        path.write_text(json.dumps(lock))
        with pytest.raises(ValueError):
            reviewed_imports(tmp_path)
    lock["files"] = {"old/helper.py": None}
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="invalid compact source override"):
        reviewed_imports(tmp_path)
    lock["files"] = {}
    lock["retired_imports"] = ["old/helper.py"]
    path.write_text(json.dumps(lock))
    assert reviewed_imports(tmp_path) == []
    (tmp_path / "old").mkdir()
    (tmp_path / "old/helper.py").touch()
    with pytest.raises(ValueError, match="retired import"):
        reviewed_imports(tmp_path)


def test_publication_verifier_uses_the_current_result_directory():
    path = ROOT / "scripts/distillschedule/check_source_publication.py"
    spec = importlib.util.spec_from_file_location("publication_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.verify(ROOT)["status"] == "PASS_PUBLICATION_INTEGRITY_ONLY"


def test_helios_source_inventory_excludes_archived_runtime():
    root = ROOT / "third_party/helios_compat"
    assert not (root / "reference/runtime").exists()
    spec = importlib.util.spec_from_file_location(
        "helios_source_check", ROOT / "scripts/distillschedule/verify_helios_compat_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.verify_source(ROOT)
    assert result["status"] == "PASS"
    from distillschedule.adapters.helios.reference_model import _source_identity
    identity = _source_identity()
    assert identity["transformer_sha256"] == module.EXPECTED_TRANSFORMER_SHA256
    assert identity["files_checked"] == result["files_checked"]


def test_source_layout_update_is_explicit_and_keeps_original_receipts(tmp_path, monkeypatch):
    from distillschedule.evidence.source_layout import update_layout
    directory = tmp_path / "docs/workloads"
    directory.mkdir(parents=True)
    source = tmp_path / "runtime/helper.py"
    source.parent.mkdir()
    source.write_text("current bytes\n")
    original = json.dumps({"files": [{"target": "runtime/helper.py",
                                       "integrated_sha256": "a" * 64}]})
    (directory / "source_import_map.json").write_text(original)
    path = directory / "source_layout.lock.json"
    path.write_text(json.dumps({"schema_version": "distillschedule.source_layout_lock.v2",
                                "files": {}, "retired_imports": []}))
    update_layout(tmp_path, ["runtime/helper.py"], "2" * 40)
    assert reviewed_imports(tmp_path)[0]["integrated_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (directory / "source_import_map.json").read_text() == original
    unchanged = path.read_bytes()
    for paths, revision in ((["unknown.py"], "2" * 40),
                            (["../escape.py"], "2" * 40),
                            (["runtime/helper.py"], "0" * 40)):
        with pytest.raises(ValueError):
            update_layout(tmp_path, paths, revision)
        assert path.read_bytes() == unchanged
    def fail_replace(*args):
        raise OSError("injected replacement failure")
    monkeypatch.setattr("distillschedule.evidence.source_layout.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected replacement failure"):
        update_layout(tmp_path, ["runtime/helper.py"], "3" * 40)
    assert path.read_bytes() == unchanged
    assert {p.name for p in directory.iterdir()} == {"source_import_map.json", "source_layout.lock.json"}
