from pathlib import Path
R=ROOT
def write(p,s):
    target=R/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s.strip()+'\n')
p=R/'src/distillschedule/__init__.py';p.write_text(p.read_text().replace('__version__ = "0.1.0"','__version__ = "0.2.0"'))
write('tests/test_repository_layout.py',r'''
"""Layout checks are source-only; they never authorize new GPU results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.core.config import _load_with_includes
from distillschedule.workloads.best_known.registry import CANONICAL_IDS, fingerprint, load_profile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "docs/architecture/repository_cleanup.json").read_text())


def test_exactly_five_canonical_profiles_and_original_bytes():
    directory = ROOT / "configs/workloads"
    assert {p.stem for p in directory.glob("*_best.json")} == set(CANONICAL_IDS)
    for path, expected in MANIFEST["canonical_profile_sha256"].items():
        assert hashlib.sha256((directory / path).read_bytes()).hexdigest() == expected, path
    assert not (ROOT / "configs/distillschedule/best_known").exists()


def test_legacy_namespace_is_installed_from_src_not_root():
    import runtime
    import runtime.host_admission
    assert Path(runtime.__file__).resolve() == ROOT / "src/runtime/__init__.py"
    assert not (ROOT / "runtime").exists()
    assert '"runtime*"' in (ROOT / "pyproject.toml").read_text()


def test_current_and_historical_livewan_resolve_identically():
    profile = load_profile("livewan_crossscale_pipeline_best", root=ROOT)
    path = ROOT / profile["runtime"]["typed_plan"]
    assert "include" not in json.loads(path.read_text())
    resolved = _load_with_includes(path, stack=(), sources=[])
    original = _load_with_includes(ROOT / MANIFEST["livewan_original_final_fixture"], stack=(), sources=[])
    assert resolved == original
    assert fingerprint(resolved) == profile["runtime"]["typed_plan_template_sha256"]
    assert sorted(p.name for p in path.parent.glob("livewan*.json")) == [path.name]


def test_relocations_have_one_source_of_truth_and_no_data_loss():
    destinations = set()
    for row in MANIFEST["relocations"]:
        assert row["to"] not in destinations
        destinations.add(row["to"])
        assert not (ROOT / row["from"]).exists(), row["from"]
        target = ROOT / row["to"]
        assert target.is_file(), row["to"]
        # Historic result payloads are immutable, even though their folder moved.
        if row["to"].startswith("benchmarks/fixtures/"):
            assert hashlib.sha256(target.read_bytes()).hexdigest() == row["original_sha256"]


def test_preserved_canonical_launcher_dependency_closure():
    for path in MANIFEST["canonical_launch_closure"]:
        assert (ROOT / path).is_file(), path
    for profile_id in CANONICAL_IDS:
        assert load_profile(profile_id, root=ROOT)["runtime"]["launcher"] in MANIFEST["canonical_launch_closure"]


def test_import_provenance_hashes_match_reorganized_sources():
    for row in json.loads((ROOT / "docs/workloads/source_import_map.json").read_text())["files"]:
        assert hashlib.sha256((ROOT / row["target"]).read_bytes()).hexdigest() == row["integrated_sha256"], row["target"]


def test_root_detection_works_below_checkout(monkeypatch, tmp_path):
    from distillschedule.cli import _default_repo_root
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    nested = tmp_path / "src/distillschedule/subdir"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert _default_repo_root() == tmp_path


@pytest.mark.parametrize("arguments, expected", [
    (["--help"], "workloads"),
    (["workloads", "--help"], "native program"),
    (["workloads", "list"], "livewan_crossscale_pipeline_best"),
])
def test_public_commands_are_discoverable_without_model_execution(arguments, expected):
    completed = subprocess.run([sys.executable, "scripts/distillctl.py", *arguments],
                               cwd=ROOT, capture_output=True, text=True, check=True)
    assert expected in completed.stdout


def test_package_version_matches_declared_project_version():
    import distillschedule
    assert f'version = "{distillschedule.__version__}"' in (ROOT / "pyproject.toml").read_text()
''')
write('.github/workflows/repository-layout.yml',r'''
name: repository-layout
on:
  pull_request:
  push:
    branches: [main, agent/general-workload-planning-20261012]
    paths:
      - src/**
      - configs/**
      - scripts/**
      - tools/**
      - benchmarks/**
      - tests/test_repository_layout.py
      - pyproject.toml
      - docs/architecture/repository_cleanup.json
      - docs/workloads/source_import_map.json
      - .github/workflows/repository-layout.yml
permissions:
  contents: read
jobs:
  source-and-package:
    runs-on: ubuntu-latest
    timeout-minutes: 12
    strategy:
      fail-fast: false
      matrix:
        python: ['3.10', '3.11']
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha || github.sha }}
          persist-credentials: false
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - name: Lightweight developer dependencies
        run: python -m pip install -e '.[test]' wheel
      - name: Verify source closure, vendor pins and locked profiles
        run: python scripts/distillschedule/check_best_known_integration.py
      - name: Layout and canonical regressions
        run: python -m pytest -q tests/test_repository_layout.py tests/test_refactor_best_known*.py
      - name: Build wheel and test imports outside the checkout
        shell: bash
        run: |
          set -euo pipefail
          python -m pip wheel . --no-deps --no-build-isolation --wheel-dir /tmp/dmd-wheel
          python -m venv /tmp/dmd-package-test
          /tmp/dmd-package-test/bin/python -m pip install --no-deps /tmp/dmd-wheel/*.whl
          cd /tmp
          /tmp/dmd-package-test/bin/python -I -c "import distillschedule, runtime, runtime.host_admission; assert '/site-packages/' in distillschedule.__file__; assert '/site-packages/' in runtime.__file__"
''')
p=R/'scripts/distillschedule/check_best_known_integration.py';s=p.read_text().replace('''    for p in (ROOT/"scripts").rglob("*.sh"):
        subprocess.run(["bash","-n",str(p)],check=True)''','''    for folder in ("scripts", "tools", "benchmarks"):
        for p in (ROOT / folder).rglob("*.sh"):
            subprocess.run(["bash", "-n", str(p)], check=True)''');p.write_text(s)
# Canonical migration keys include relative subdirectories (asset examples may
# have the same basenames as profiles and must not overwrite their hashes).
import json,hashlib
p=R/'docs/architecture/repository_cleanup.json'; m=json.loads(p.read_text())
m['canonical_profile_sha256']={name.removeprefix('configs/distillschedule/best_known/'):hashlib.sha256(data).hexdigest() for name,data in sorted(profiles.items())}
p.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
