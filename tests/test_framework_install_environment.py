"""Installed qualification must not silently swap a validated numerical stack."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

def tool():
    spec = importlib.util.spec_from_file_location('check_framework_install', ROOT/'scripts/check_framework_install.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_roots_preserve_overlay_order_without_project_leaks(tmp_path):
    checkout = tmp_path/'checkout'; (checkout/'src').mkdir(parents=True)
    overlay = tmp_path/'torch210'; (overlay/'torch-2.10.dist-info').mkdir(parents=True)
    base = tmp_path/'base'/'site-packages'; base.mkdir(parents=True)
    foreign = tmp_path/'foreign'/'site-packages'; (foreign/'distillschedule').mkdir(parents=True)
    compatibility = tmp_path/'compat'/'site-packages'; (compatibility/'runtime').mkdir(parents=True)
    roots = tool().dependency_roots(checkout, paths=[str(checkout/'src'),str(overlay),str(base),
        str(overlay),str(foreign),str(compatibility),'','import evil',str(base)+'\nimport evil'])
    assert roots == [str(overlay.resolve()), str(base.resolve())]


def test_actual_dependency_origins_must_be_in_the_admitted_roots():
    with pytest.raises(RuntimeError, match='dependency origin'):
        tool().qualified_dependencies([])


def test_installed_recipe_accessor_uses_distribution_inventory_not_cwd(tmp_path, monkeypatch):
    from distillschedule import recipes
    root = tmp_path/'install/share/distillschedule/recipes'
    root.mkdir(parents=True); (root/'README.md').write_text('recipes')
    class Distribution:
        files = ['share/distillschedule/recipes/README.md']
        def locate_file(self, path):
            return tmp_path/'install'/path
    monkeypatch.setattr(recipes.metadata, 'distribution', lambda name: Distribution())
    monkeypatch.chdir(tmp_path)
    assert recipes.directory() == root
    Distribution.files = []
    with pytest.raises(RuntimeError, match='recipe resources'):
        recipes.directory()


def test_modern_qualification_rejects_different_dependency_versions(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace
    pins = dict(torch='2.10.0+cpu', numpy='2.5.3', PIL='12.3.0', safetensors='0.8.0',
                diffusers='0.37.1', transformers='5.14.1', peft='0.17.1',
                tokenizers='0.22.2', accelerate='1.14.0', huggingface_hub='1.7.1')
    monkeypatch.setattr(importlib, 'import_module', lambda name: SimpleNamespace(
        __file__=str(tmp_path/name/'__init__.py'), __version__=pins[name]))
    with pytest.raises(RuntimeError, match='modern-flow.*peft'):
        tool().qualified_dependencies([str(tmp_path)], modern=True)


@pytest.mark.parametrize('inside_checkout', [False, True])
def test_installed_work_directory_refuses_overwrite_and_checkout(tmp_path, monkeypatch, inside_checkout):
    module = tool()
    monkeypatch.setattr(module, 'dependency_roots', lambda checkout: [])
    monkeypatch.setattr(module, 'qualified_dependencies', lambda roots, modern: {})
    existing = ROOT if inside_checkout else tmp_path
    monkeypatch.setattr(sys, 'argv', ['check_framework_install.py', '--work-dir', str(existing)])
    with pytest.raises(ValueError if inside_checkout else FileExistsError):
        module.main()


@pytest.mark.skipif(os.environ.get('DMD_MODERN_FLOW_CPU') != '1',
                    reason='separate mandatory modern-flow installed-wheel lane')
def test_installed_modern_wheel_actual_image_video_restart_and_export_only_inputs(tmp_path):
    output = tmp_path/'installed-modern.json'
    result = subprocess.run([sys.executable, str(ROOT/'scripts/check_framework_install.py'),
        '--modern-flow', '--work-dir', str(tmp_path/'acceptance'), '--output', str(output)],
        cwd=tmp_path, env={**os.environ, 'PIP_NO_INDEX': '1'}, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report['status'] == 'PASS_INSTALLED_MODERN_FLOW_CPU'
    assert report['queries_without_torch_cuda_or_weights'] is True
    assert set(report['families']) == {'flux2-klein-base', 'wan22-ti2v5b-t2v'}
    for family in report['families'].values():
        for key in ('complete_state_rng_data_equal', 'fresh_process_resume',
                    'training_assets_checkpoints_deleted', 'public_export_only_new_prompt',
                    'public_export_only_new_raw_media', 'exact_current_student_decode'):
            assert family[key] is True, key
        assert family['evidence_kind'] == 'SYNTHETIC_TEST_ONLY'
    assert report['gpu_execution'] == report['pretrained_quality'] == 'NOT_RUN'
