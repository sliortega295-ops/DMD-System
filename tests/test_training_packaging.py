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
    "THIRD_PARTY.md": "ccad27b7badea5d6f8bd37b8b08164dd603a27e8fbab30d8ded4c5a2b2347eb6",
    "third_party/helios_compat/NOTICE": "463dd51a8fc29fcc19861287194922f076b5ff53aa2da26085830ebd3a38b4b9",
    "third_party/helios_compat/LICENSE.txt": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "third_party/licenses/H3-DMD-for-A100-LICENSE": "ef5dc286188f40073d466b3ad5c3a9b7728a7874766850d29d74d6b58c704115",
    "licenses/diffsynth-h3-cached/LICENSE.Apache-2.0": "d43b96112d60d83595395649e61c6bc215f568052a979320489556c6c1262eb1",
    "licenses/diffsynth-h3-cached/ORIGIN_AND_CHANGES.md": "ebdee8bac7e36ed9b18d5d323c12c7b98bad5ab5de3dede4b2679cba688cf4c1",
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


def test_wheel_includes_cached_h3_source_manifest_and_unchanged_numerical_files(framework_wheel):
    package = "distillschedule/reference/diffsynth_h3_cached/"
    with zipfile.ZipFile(framework_wheel) as wheel:
        manifest = json.loads(wheel.read(package + "source_manifest.json"))
        assert manifest["scope"] == "MAINTAINED_CACHED_AV_DERIVATIVE_CPU_SOURCE_ONLY"
        for row in manifest["files"]:
            assert hashlib.sha256(wheel.read(package + row["path"])).hexdigest() == row["sha256"]
        assert package + "__init__.py" in wheel.namelist()


def test_cached_h3_reference_has_its_own_small_dependency_closure(wheel_metadata):
    requirements = {value.replace(" ", "") for value in wheel_metadata.get_all("Requires-Dist")}
    for package in ("torch>=2.10", "numpy>=1.24", "safetensors>=0.4", "einops==0.8.1"):
        assert f'{package};extra=="h3-cached-reference"' in requirements


def test_native_extras_keep_their_independent_dependency_floor(wheel_metadata):
    requirements = {value.replace(" ", "") for value in wheel_metadata.get_all("Requires-Dist")}
    for extra in ("native", "native-wan"):
        assert f'torch>=2.5;extra=="{extra}"' in requirements


def test_original_helios_cpu_helpers_have_an_explicit_test_environment(wheel_metadata):
    requirements = {value.replace(" ", "") for value in wheel_metadata.get_all("Requires-Dist")}
    for package in ("diffusers==0.34.0", "transformers==4.51.3", "einops==0.8.1", "ftfy==6.3.1", "kernels==0.9.0"):
        assert f'{package};extra=="test-helios"' in requirements
    from packaging.requirements import Requirement
    triton = [Requirement(value) for value in wheel_metadata.get_all("Requires-Dist")
              if Requirement(value).name == "triton"]
    assert len(triton) == 1 and str(triton[0].specifier) == "==3.6.0"
    assert triton[0].marker.evaluate({"extra": "test-helios", "platform_system": "Linux", "platform_machine": "x86_64"})
    assert not triton[0].marker.evaluate({"extra": "train", "platform_system": "Linux", "platform_machine": "x86_64"})
    assert not triton[0].marker.evaluate({"extra": "test-helios", "platform_system": "Darwin", "platform_machine": "arm64"})
    import yaml
    workflow = yaml.safe_load((ROOT / ".github/workflows/framework-cpu.yml").read_text())
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["lifecycle"]["steps"])
    assert ".[train,test,test-helios]" in commands


def test_original_helios_ci_requires_actual_public_admission_and_restart():
    import yaml
    workflow = yaml.safe_load((ROOT / ".github/workflows/framework-cpu.yml").read_text())
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["lifecycle"]["steps"])
    assert "test_actual_original_helpers_three_cycles_checkpoint_resume_equal" in commands
    assert "test_public_to_original_p0_collectively_admits_before_model" in commands
    assert "find('skipped')" in commands


def test_original_h3_cpu_job_uses_real_pinned_sources_and_dependencies(wheel_metadata):
    import yaml
    requirements = {value.replace(" ", "") for value in wheel_metadata.get_all("Requires-Dist")}
    for package in ("loguru==0.7.3", "peft==0.15.2", "imageio==2.37.4"):
        assert f'{package};extra=="test-h3"' in requirements
    workflow = yaml.safe_load((ROOT / ".github/workflows/h3-grid-native-cpu.yml").read_text())
    job = workflow["jobs"]["original-source-state"]
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "git submodule update --init --depth 1 third_party/lightx2v" in commands
    sources = [step["with"] for step in job["steps"]
               if step.get("with", {}).get("repository") == "huggingface/diffusers"]
    assert len(sources) == 1 and sources[0]["ref"] == "9284607295a09f759aadd65ed08f48b35feea6d9"
    assert ".[train,test,test-helios,test-h3]" in commands
    assert "get('skipped', '0')" in commands


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
