"""Wheel metadata and the real, offline installed-package lifecycle acceptance."""
import hashlib
import json
import os
import subprocess
import sys
import venv
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LICENSE_SHA256 = {
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "NOTICE": "9247ab500279666c898a4ac7e943989ea8c4316c3211d7fda88ed5a28151366b",
    "THIRD_PARTY.md": "b688c5e7b7084bb8fef5547b2fab2f3b381844b3ebcdc0cfd190d1cb53654e18",
    "third_party/helios_compat/NOTICE": "463dd51a8fc29fcc19861287194922f076b5ff53aa2da26085830ebd3a38b4b9",
    "third_party/helios_compat/LICENSE.txt": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "third_party/licenses/H3-DMD-for-A100-LICENSE": "ef5dc286188f40073d466b3ad5c3a9b7728a7874766850d29d74d6b58c704115",
}


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


def test_native_extras_keep_their_independent_dependency_floor(wheel_metadata):
    requirements = {value.replace(" ", "") for value in wheel_metadata.get_all("Requires-Dist")}
    for extra in ("native", "native-wan"):
        assert f'torch>=2.5;extra=="{extra}"' in requirements


def test_original_helios_cpu_helpers_have_an_explicit_test_environment(wheel_metadata):
    requirements = {value.replace(" ", "") for value in wheel_metadata.get_all("Requires-Dist")}
    for package in ("diffusers==0.34.0", "transformers==4.51.3", "einops==0.8.1", "ftfy==6.3.1"):
        assert f'{package};extra=="test-helios"' in requirements
    import yaml
    workflow = yaml.safe_load((ROOT / ".github/workflows/framework-cpu.yml").read_text())
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["lifecycle"]["steps"])
    assert ".[train,test,test-helios]" in commands


@pytest.mark.parametrize("workflow", ["ci.yml", "hierarchical-planner.yml"])
def test_full_refactor_ci_provisions_the_tensor_and_asset_dependencies(workflow):
    import yaml

    config = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    for job in config["jobs"].values():
        commands = "\n".join(step.get("run", "") for step in job["steps"])
        assert "tests/test_refactor_*.py" in commands
        assert "torch==2.10.0" in commands, "real tensor regressions must execute, not import-skip"
        assert ".[test,native]" in commands, "native bootstrap collection requires NumPy and safetensors"


@pytest.fixture(scope="module")
def installed_report(tmp_path_factory):
    output = tmp_path_factory.mktemp("framework-wheel-acceptance") / "report.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_framework_install.py"), "--output", str(output)],
        cwd=output.parent, env={**os.environ, "PIP_NO_INDEX": "1"},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "PASS_INSTALLED_FRAMEWORK_CPU"
    return report


def test_installed_wheel_generates_on_new_inputs_without_training_artifacts(installed_report):
    for field in ("exported_weights_new_inputs", "training_artifacts_unavailable",
                  "fresh_process_resume", "fresh_process_generation", "saved_plan_executed"):
        assert installed_report.get(field) is True, field
    inference = installed_report["inference"]
    assert inference["kind"] == "INFERENCE_FROM_EXPORTED_WEIGHTS"
    assert inference["sample_id"] == "new-input-000004"
    assert inference["input_identity"] != inference["training_input_identity"]
    assert inference["prepared_features_disjoint"] is True
    assert inference["encoded_feeds_disjoint"] is True
    assert inference["video_layout"] == "THWC"
    assert inference["source_bound"] is True
    assert inference["module_layout_bound"] is True


def test_installed_acceptance_records_package_source_and_quality_boundaries(installed_report):
    source = installed_report["source"]
    assert source["working_directory_outside_checkout"] is True
    package_site = Path(source["site_packages"])
    assert set(source["module_paths"]) == {"distillschedule", "runtime"}
    assert all(Path(path).is_relative_to(package_site) for path in source["module_paths"].values())
    for digest in (source["wheel_sha256"], installed_report["inference"]["model_source_sha256"],
                   installed_report["inference"]["export_manifest_sha256"]):
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
    assert installed_report["scope"] == "LOCAL_RANDOM_TINY_MEDIA_FIXTURE"
    for field in ("gpu_execution", "pretrained_quality", "quality_validation", "native_pretrained_lifecycles"):
        assert installed_report[field] == "NOT_RUN"
