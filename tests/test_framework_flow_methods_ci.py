"""Provider-separated flow CI selection and synthetic no-skip receipt guards.

Collection and fabricated XML exercise CI wiring only, not native model math.
"""
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'tests/test_training_flow_model_methods.py'
LANES = {
    'flux2-klein-base': ('modern-flow-cpu.yml', 'native-media-lifecycle',
                         'Actual FLUX model-owned flow boundary', 'flux-flow-methods-cpu.xml'),
    'ltx25': ('official-models-cpu.yml', 'actual-model-classes',
              'Actual LTX model-owned flow boundary', 'ltx-flow-methods-cpu.xml'),
}


def _proofs(family):
    return [
        f'test_native_direct_entry_current_owner_no_copy_and_exact_census[{family}]',
        *[f'test_native_exact_sequential_trace_at_new_entry[{family}-{policy}]' for policy in (
            'reuse_student-all_steps-False-False', 'reuse_student-last_step-True-True',
            'fresh_student-all_steps-True-False', 'fresh_student-last_step-False-False')],
        f'test_native_rank_namespaces_and_local_gradients_match_existing_b1_oracle[{family}]',
        f'test_native_clock_refusal_precedes_outer_and_native_forward[{family}]',
        f'test_native_complete_cycle_new_process_resume_is_exact[{family}]',
    ]


def _lane(family):
    filename, job_name, step_name, receipt = LANES[family]
    workflow = yaml.load((ROOT / '.github/workflows' / filename).read_text(), Loader=yaml.BaseLoader)
    job = workflow['jobs'][job_name]
    step = next((item for item in job['steps'] if item.get('name') == step_name), None)
    assert step is not None, f'{family} actual-native flow-method CI step is missing'
    return workflow, job, step, receipt


def _commands(step):
    commands = [shlex.split(line) for line in step['run'].replace('\\\n', ' ').splitlines() if line.strip()]
    assert len(commands) == 2, 'run the selected proofs, then require their exact no-skip receipt'
    return commands


@pytest.mark.parametrize('family', LANES)
def test_flow_methods_ci_runs_only_its_exact_provider_and_requires_all_eight_proofs(family):
    workflow, job, step, receipt = _lane(family)
    environment = {**job['env'], **step.get('env', {})}
    assert environment['DMD_MODERN_FLOW_CPU'] == '1'
    assert environment['CUDA_VISIBLE_DEVICES'] == ''
    assert environment['HF_HUB_OFFLINE'] == environment['TRANSFORMERS_OFFLINE'] == '1'
    assert not {'DMD_FLOW_METHOD_IMAGE_CONFIG', 'DMD_FLOW_METHOD_LTX_CONFIG'} & environment.keys()
    assert 'continue-on-error' not in step and 'continue-on-error' not in job
    assert 'if' not in job
    assert workflow['permissions'] == {'contents': 'read'}
    paths = workflow['on']['pull_request']['paths']
    assert 'tests/test_training_*.py' in paths and 'tests/flow_replica_oracle.py' in paths
    assert any(Path(__file__).match(pattern) for pattern in paths)
    commands = '\n'.join(item.get('run', '') for item in job['steps'])
    if family == 'ltx25':
        assert step['if'] == "${{ matrix.family == 'ltx25' }}"
        assert step['env']['DMD_MODERN_FLOW_CPU'] == '1'
        assert 'DMD_MODERN_FLOW_CPU' not in job['env']
        assert {'family': 'ltx25', 'extra': 'train,test,ltx25'} in job['strategy']['matrix']['include']
        assert environment['LTX_SOURCE_ROOT'] == '${{ github.workspace }}/.framework-test-deps/ltx25'
        assert 'ltx25) python recipes/ltx25/fetch_source.py "$LTX_SOURCE_ROOT" ;;' in commands
        assert '.[${{ matrix.extra }}]' in commands and 'modern-flow]' not in commands
    else:
        assert 'if' not in step and 'LTX_SOURCE_ROOT' not in environment
        assert '.[test,modern-flow]' in commands and 'ltx25' not in commands
    assert 'torch==2.10.0' in commands and 'https://download.pytorch.org/whl/cpu' in commands
    proof_command, gate_command = _commands(step)
    proofs = _proofs(family)
    assert proof_command == ['python', '-m', 'pytest', '-q',
                             *[f'{TARGET}::{name}' for name in proofs], f'--junitxml={receipt}']
    assert gate_command == ['python', 'scripts/check_cpu_receipt.py', receipt,
                            *[word for name in proofs for word in ('--require', name)]]
    upload = next(item for item in job['steps'] if item.get('uses') == 'actions/upload-artifact@v4'
                  and receipt in item['with']['path'].splitlines())
    assert 'always()' in upload['if'] and upload['with']['if-no-files-found'] == 'error'
    if family == 'ltx25':
        assert upload['if'] == "${{ always() && matrix.family == 'ltx25' }}"


@pytest.mark.parametrize('family', LANES)
def test_flow_methods_ci_exact_node_ids_are_actually_collected(family):
    _, _, step, _ = _lane(family)
    proof_command, _ = _commands(step)
    # No fixture runs: this stays valid in the common CPU lane without either
    # native optional stack or source. The hosted provider steps do run fixtures.
    arguments = [arg for arg in proof_command[1:] if not arg.startswith('--junitxml=')]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
                       PYTHONPATH=os.pathsep.join(str(ROOT / part) for part in ('src', '.', 'tests')))
    result = subprocess.run([sys.executable, *arguments, '--collect-only', '-p', 'no:cacheprovider'],
                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    collected = [line for line in result.stdout.splitlines() if line.startswith(TARGET + '::')]
    assert sorted(collected) == sorted(f'{TARGET}::{name}' for name in _proofs(family))


@pytest.mark.parametrize('family', LANES)
@pytest.mark.parametrize('mutation', [None, 'missing-case', 'other-provider', 'skipped',
                                     'failure', 'error', 'empty', 'missing-file'])
def test_flow_methods_ci_actual_receipt_gate_rejects_false_green(tmp_path, family, mutation):
    """SYNTHETIC_TEST_ONLY: invoke the configured gate, not model qualification."""
    _, _, step, receipt = _lane(family)
    _, gate_command = _commands(step)
    root = ET.Element('testsuite')
    for name in _proofs(family):
        ET.SubElement(root, 'testcase', name=name)
    if mutation == 'missing-case':
        root.remove(root[-1])
    elif mutation == 'other-provider':
        other = next(name for name in LANES if name != family)
        for case, name in zip(root, _proofs(other)):
            case.set('name', name)
    elif mutation in ('skipped', 'failure', 'error'):
        ET.SubElement(root[-1], mutation)
    elif mutation == 'empty':
        root.clear()
    path = tmp_path / receipt
    if mutation != 'missing-file':
        ET.ElementTree(root).write(path)
    result = subprocess.run([sys.executable, str(ROOT / gate_command[1]), str(path), *gate_command[3:]],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) is (mutation is None), result.stdout + result.stderr


def test_common_cpu_collector_runs_the_flow_methods_ci_regressions():
    api = runpy.run_path(str(ROOT / 'scripts/run_framework_cpu_tests.py'))
    assert 'tests/test_framework_flow_methods_ci.py' in api['test_files'](ROOT)
