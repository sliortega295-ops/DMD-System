"""Wheel metadata and the real, offline installed-package lifecycle acceptance."""
import json
import os
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wheel_metadata(tmp_path_factory):
    output = tmp_path_factory.mktemp("framework-wheel-metadata")
    result = subprocess.run(
        [sys.executable, "-c", ("import sys; from setuptools.build_meta import build_wheel; "
         "build_wheel(sys.argv[1])"), str(output)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(output.glob("distillschedule-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        assert len(metadata) == 1
        return BytesParser().parsebytes(wheel.read(metadata[0]))


def test_train_extra_declares_the_public_fsdp_api_minimum(wheel_metadata):
    assert wheel_metadata["Requires-Python"] == ">=3.10"
    requirements = {value.replace(" ", "") for value in wheel_metadata.get_all("Requires-Dist")}
    assert 'torch>=2.10;extra=="train"' in requirements


def test_native_extras_keep_their_independent_dependency_floor(wheel_metadata):
    requirements = {value.replace(" ", "") for value in wheel_metadata.get_all("Requires-Dist")}
    for extra in ("native", "native-wan"):
        assert f'torch>=2.5;extra=="{extra}"' in requirements


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
