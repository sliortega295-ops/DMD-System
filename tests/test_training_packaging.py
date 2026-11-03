"""Wheel metadata and the real, offline installed-package lifecycle acceptance."""
import hashlib
import json
import os
import re
import subprocess
import sys
import venv
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
LICENSE_SHA256 = {
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "NOTICE": "bc15142557d3efcc5a4bc60eef15e8d58a718953e36f231110ac87ac0631bad4",
    "THIRD_PARTY.md": "95595d8f994a10d713f7f7e47a1cde7103a3a9a45f44267a6367003ca1746ca7",
    "third_party/helios_compat/NOTICE": "731209f5f2fc28cc824102244eb2a2962e5d37d4befd7de802f82f3b0bf732f1",
    "third_party/helios_compat/LICENSE.txt": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "third_party/licenses/H3-DMD-for-A100-LICENSE": "ef5dc286188f40073d466b3ad5c3a9b7728a7874766850d29d74d6b58c704115",
    "licenses/diffsynth-h3-cached/LICENSE.Apache-2.0": "d43b96112d60d83595395649e61c6bc215f568052a979320489556c6c1262eb1",
    "licenses/diffsynth-h3-cached/ORIGIN_AND_CHANGES.md": "ebdee8bac7e36ed9b18d5d323c12c7b98bad5ab5de3dede4b2679cba688cf4c1",
    "licenses/livewan-wan21-reference/Wan2.1-LICENSE.txt": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "licenses/livewan-wan21-reference/LiveWan-LICENSE.txt": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "licenses/livewan-wan21-reference/ORIGIN_AND_CHANGES.md": "e63d33cbe36c4dcc34faa1446bb66d02c09316c41163b3c8737b25320cbd2724",
}


def test_public_document_dependencies_have_installed_data_file_destinations():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10: pytest's TOML dependency.
        import tomli as tomllib

    data_files = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["data-files"]
    installed = {destination: {path for pattern in patterns for path in ROOT.glob(pattern)}
                 for destination, patterns in data_files.items()}
    guides = {path for paths in installed.values() for path in paths
              if path.suffix == ".md" and path.relative_to(ROOT).parts[0] in {"recipes", "docs", "artifacts"}}
    assert guides, "public recipes and documentation must be declared as package data"
    dependencies = set()
    for guide in sorted(guides):
        for href in re.findall(r"\]\(([^)\s]+)\)", guide.read_text()):
            link = urlsplit(href)
            if link.scheme or link.netloc or not link.path:
                continue
            target = (guide.parent / unquote(link.path)).resolve()
            if not target.is_relative_to(ROOT):
                continue
            source = target.relative_to(ROOT)
            assert target.is_file(), f"{guide.relative_to(ROOT)} links to missing {source}"
            dependencies.add(source)
    # Independent regression: the packaged H3 continuation recipe needs this policy.
    assert Path("docs/contracts/h3_native.md") in dependencies
    # Consolidation must ship the actual maintained pages, not stale paths or
    # guide-only coverage that leaves normative contracts/evidence behind.
    packaged = {path for paths in installed.values() for path in paths}
    assert set((ROOT / 'docs').rglob('*.md')) <= packaged
    assert {ROOT / 'docs/contracts' / name for name in (
        'native_output.md', 'helios_native.md', 'h3_native.md',
        'livewan_native.md', 'state_retention.md', 'python_dependency_overlays_v1.md',
    )} <= packaged
    assert all(list(ROOT.glob(pattern))
               for patterns in data_files.values() for pattern in patterns)
    missing = [str(source) for source in sorted(dependencies)
               if ROOT / source not in installed.get(
                   (Path("share/distillschedule") / source.parent).as_posix(), set())]
    assert not missing, f"public document dependencies lack their installed data-file destination: {missing}"


@pytest.fixture(scope="module")
def framework_wheel(tmp_path_factory):
    output = tmp_path_factory.mktemp("framework-wheel-metadata")
    result = subprocess.run(
        [sys.executable, "-c", ("import sys; from setuptools.build_meta import build_wheel; "
         "build_wheel(sys.argv[1])"), str(output)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(output.glob("distillschedule-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


@pytest.fixture(scope="module")
def wheel_metadata(framework_wheel):
    with zipfile.ZipFile(framework_wheel) as wheel:
        metadata = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        assert len(metadata) == 1
        return BytesParser().parsebytes(wheel.read(metadata[0]))


def test_wheel_preserves_original_licenses_and_notice_paths(framework_wheel, wheel_metadata):
    with zipfile.ZipFile(framework_wheel) as wheel:
        metadata = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        prefix = metadata.removesuffix("METADATA") + "licenses/"
        for name, expected in LICENSE_SHA256.items():
            archive_path = prefix + name
            assert archive_path in wheel.namelist(), archive_path
            original = (ROOT / name).read_bytes()
            assert hashlib.sha256(original).hexdigest() == expected, name
            assert wheel.read(archive_path) == original, name
    assert tuple(map(int, wheel_metadata["Metadata-Version"].split("."))) >= (2, 4)
    assert wheel_metadata["License-Expression"] == "Apache-2.0"
    assert set(wheel_metadata.get_all("License-File")) == set(LICENSE_SHA256)


def test_installed_wheel_preserves_original_licenses_and_notice_paths(framework_wheel, tmp_path):
    target = tmp_path / "installed"
    venv.EnvBuilder(with_pip=True).create(target)
    python = target / "bin/python"
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    result = subprocess.run(
        [str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", str(framework_wheel)],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    script = '''
import hashlib, importlib.metadata, json, sys, sysconfig
from pathlib import Path
expected = json.loads(sys.argv[1])
distribution = importlib.metadata.distribution("distillschedule")
files = {str(item): item for item in distribution.files}
metadata = next(name for name in files if name.endswith(".dist-info/METADATA"))
prefix = metadata.removesuffix("METADATA") + "licenses/"
site = Path(sysconfig.get_paths()["purelib"]).resolve()
for name, sha in expected.items():
    record = files[prefix + name]
    path = Path(distribution.locate_file(record)).resolve()
    assert path.is_relative_to(site), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, name
assert distribution.metadata["License-Expression"] == "Apache-2.0"
assert set(distribution.metadata.get_all("License-File")) == set(expected)
'''
    result = subprocess.run(
        [str(python), "-I", "-c", script, json.dumps(LICENSE_SHA256)],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_train_extra_declares_the_public_fsdp_api_minimum(wheel_metadata):
    assert wheel_metadata["Requires-Python"] == ">=3.10"
    requirements = {value.replace(" ", "") for value in wheel_metadata.get_all("Requires-Dist")}
    assert 'torch>=2.10;extra=="train"' in requirements
