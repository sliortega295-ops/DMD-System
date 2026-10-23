"""Exercise the exact common-CI producer, not a mirrored --ignore list."""
from pathlib import Path
import os
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _api():
    return runpy.run_path(str(ROOT / 'scripts/run_framework_cpu_tests.py'))


def test_collection_excludes_provider_files_before_passing_explicit_pytest_args(tmp_path):
    source = tmp_path / 'tests'; source.mkdir()
    names = ('test_data_real.py', 'test_training_real.py', 'test_open_workload_real.py',
             'test_framework_real.py', 'test_cpu_receipt_gate.py',
             'test_training_livewan_reference_inputs.py', 'test_training_livewan_public_cpu.py',
             'test_training_livewan_inference_cpu.py', 'unrelated.py')
    for name in names:
        (source / name).write_text('')
    selected = _api()['test_files'](tmp_path)
    assert selected == ['tests/' + name for name in sorted(names[:5])]
    with pytest.raises(ValueError, match='no common CPU tests'):
        _api()['test_files'](tmp_path / 'empty')


def test_actual_launcher_executes_the_selected_inventory_and_propagates_failure(monkeypatch):
    api = _api(); observed = []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs:
        (observed.append((command, kwargs)) or SimpleNamespace(returncode=7)))
    assert api['main'](['--junitxml=unit-test.xml']) == 7
    command, kwargs = observed[0]
    assert command == [sys.executable, '-m', 'pytest', '-q', *api['test_files'](ROOT),
                       '--junitxml=unit-test.xml']
    assert kwargs['cwd'] == ROOT and kwargs['check'] is False
    assert kwargs['env'] == dict(os.environ, CUDA_VISIBLE_DEVICES='',
                                 HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    assert not any(name in arg for name in api['EXCLUDED'] for arg in command)
    workflow = (ROOT / '.github/workflows/framework-cpu.yml').read_text()
    assert 'python scripts/run_framework_cpu_tests.py --junitxml=framework-cpu.xml' in workflow


def test_collection_is_import_light_in_a_fresh_process():
    script = ("import runpy, sys; value = runpy.run_path(sys.argv[1]); "
              "assert value['test_files'](sys.argv[2]); assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, '-c', script,
        str(ROOT / 'scripts/run_framework_cpu_tests.py'), str(ROOT)], text=True,
        capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
