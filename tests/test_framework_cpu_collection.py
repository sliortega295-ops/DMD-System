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
             'test_framework_real.py', 'test_cpu_receipt_gate.py', 'test_runtime_real.py',
             'test_training_livewan_reference_inputs.py', 'test_training_livewan_public_cpu.py',
             'test_training_livewan_inference_cpu.py', 'unrelated.py')
    for name in names:
        (source / name).write_text('')
    selected = _api()['test_files'](tmp_path)
    assert selected == ['tests/' + name for name in sorted(names[:6])]
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


def test_native_phase_consumer_is_in_actual_common_lane_and_triggers_ci():
    import yaml
    assert 'tests/test_evidence_livewan_c51_phases.py' in _api()['test_files'](ROOT)
    workflow = yaml.load((ROOT / '.github/workflows/framework-cpu.yml').read_text(), Loader=yaml.BaseLoader)
    assert 'tests/test_evidence_livewan_c51_phases.py' in workflow['on']['pull_request']['paths']


def test_variable_forward_phase_proofs_are_in_actual_common_lane_and_cannot_silently_skip():
    import yaml
    files = _api()['test_files'](ROOT)
    assert 'tests/test_runtime_fsdp2_variable_forward.py' in files
    assert 'tests/test_runtime_fsdp2_variable_forward_cpu.py' in files
    assert 'tests/test_training_helios_gan_native_history.py' in files
    workflow = yaml.load((ROOT / '.github/workflows/framework-cpu.yml').read_text(), Loader=yaml.BaseLoader)
    assert 'tests/test_runtime_*.py' in workflow['on']['pull_request']['paths']
    job = workflow['jobs']['lifecycle']
    assert job['strategy']['matrix']['python'] == ['3.10', '3.12']
    gate = next(step['run'] for step in job['steps'] if step.get('name') ==
                'Require original Helios startup and persistent state proof')
    assert 'test_actual_hsdp_peft_unequal_prefix_original_adam_and_rng' in gate
    assert 'test_target_original_gan_zero_history_capture_exact_rng_crops_and_r1' in gate
    # Exercise the actual CI gate with marked synthetic XML, not a second
    # implementation of its required-case or no-skip rules.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cases = ['test_actual_original_helpers_three_cycles_checkpoint_resume_equal',
                 *[f'test_public_to_original_p0_collectively_admits_before_model[{n}]' for n in range(13)],
                 'test_actual_hsdp_peft_unequal_prefix_original_adam_and_rng',
                 'test_target_original_gan_zero_history_capture_exact_rng_crops_and_r1']
        for selected in (-2, -1):
            for mutation in ('valid', 'missing', 'skipped', 'failure'):
                rows = ['<testcase name="' + name + '"/>' for name in cases]
                if mutation == 'missing': rows.pop(selected)
                elif mutation != 'valid':
                    rows[selected] = '<testcase name="' + cases[selected] + '"><' + mutation + '/></testcase>'
                (root / 'framework-cpu.xml').write_text('<testsuite>' + ''.join(rows) + '</testsuite>')
                # Execute the production here-doc; XML is SYNTHETIC_TEST_ONLY.
                source = gate.split("python - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
                result = subprocess.run([sys.executable, '-c', source], cwd=root, capture_output=True, text=True)
                assert (result.returncode == 0) is (mutation == 'valid'), (selected, mutation, result.stderr)
