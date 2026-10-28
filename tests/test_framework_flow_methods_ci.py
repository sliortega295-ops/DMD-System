"""Provider-separated native CI selection and synthetic no-skip receipt guards.

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


def _lane(family, lanes=LANES):
    filename, job_name, step_name, receipt = lanes[family]
    workflow = yaml.load((ROOT / '.github/workflows' / filename).read_text(), Loader=yaml.BaseLoader)
    job = workflow['jobs'][job_name]
    step = next((item for item in job['steps'] if item.get('name') == step_name), None)
    assert step is not None, f'{family} actual-native CI step is missing: {step_name}'
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
    # Collection checks the real node IDs without running optional model fixtures.
    assert sorted(_collect_command(proof_command)) == sorted(
        f'{TARGET}::{name}' for name in _proofs(family))


@pytest.mark.parametrize('family', LANES)
@pytest.mark.parametrize('mutation', [None, 'missing-case', 'other-provider', 'skipped',
                                     'failure', 'error', 'empty', 'missing-file'])
def test_flow_methods_ci_actual_receipt_gate_rejects_false_green(tmp_path, family, mutation):
    """SYNTHETIC_TEST_ONLY: invoke the configured gate, not model qualification."""
    _, _, step, receipt = _lane(family)
    other = next(name for name in LANES if name != family)
    _assert_receipt_gate(tmp_path, step, receipt, _proofs(family), mutation, _proofs(other))


def test_common_cpu_collector_runs_the_flow_methods_ci_regressions():
    api = runpy.run_path(str(ROOT / 'scripts/run_framework_cpu_tests.py'))
    assert 'tests/test_framework_flow_methods_ci.py' in api['test_files'](ROOT)


SHARDING_LANES = {
    'flux2-klein-base': ('modern-flow-cpu.yml', 'native-media-lifecycle',
                         'Actual FLUX CPU FSDP flow ownership', 'flux-flow-sharding-cpu.xml'),
    'ltx25': ('official-models-cpu.yml', 'actual-model-classes',
              'Actual LTX CPU FSDP flow ownership', 'ltx-flow-sharding-cpu.xml'),
}
H3_FILES = ['tests/test_training_h3_grid.py', 'tests/test_training_h3_grid_distributed.py',
            'tests/test_training_h3_grid_stop.py', 'tests/test_training_native_state.py']
H3_PROOFS = [
    'test_real_original_grid_loop_three_cycles_exact_interrupted_continuation',
    'test_actual_gloo_rank_failure_and_complete_original_state[exact_resume]',
    'test_full_lazy_cpu_admission_with_real_pinned_sources_and_identity_only_assets[None]',
    'test_public_horizon50_stop25_keeps_original_identity',
    'test_real_loop_seam_stops_only_after_durable_publication_and_defaults_unchanged[1-1-1-saved0]',
    'test_real_loop_seam_stops_only_after_durable_publication_and_defaults_unchanged[None-0-3-saved3]',
    'test_real_gloo_loop_stop_waits_for_all_rank_checkpoint_publication[False]',
    'test_real_gloo_loop_stop_waits_for_all_rank_checkpoint_publication[True]',
]


def _sharding_proofs(family):
    return [
        (file, f'{name}[{family}-{mode}]')
        for file, name in (
            ('tests/test_open_workload_parameter_shadows.py',
             'test_native_full_shadow_oracle_and_fresh_process_checkpoint'),
            ('tests/test_open_workload_frozen_roots.py',
             'test_private_native_flow_frozen_teacher_G1_F5'),
            ('tests/test_training_flow_owned_generation.py',
             'test_native_owned_generation_restored_current_student_and_closed_groups'),
        )
        for mode in ('fsdp2_reshard', 'fsdp2_keep_forward')
    ]


def _assert_receipt_gate(tmp_path, step, receipt, names, mutation, other_names=()):
    """SYNTHETIC_TEST_ONLY: producer names test the real configured XML consumer."""
    _, gate_command = _commands(step)
    root = ET.Element('testsuite')
    for name in names:
        ET.SubElement(root, 'testcase', name=name)
    if mutation == 'missing-case':
        root.remove(root[-1])
    elif mutation == 'other-provider':
        assert len(other_names) == len(names)
        for case, name in zip(root, other_names):
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


def _collect_command(proof_command):
    arguments = [arg for arg in proof_command[1:] if not arg.startswith('--junitxml=')]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
                       PYTHONPATH=os.pathsep.join(str(ROOT / part) for part in ('src', '.', 'tests')))
    result = subprocess.run([sys.executable, *arguments, '--collect-only', '-p', 'no:cacheprovider'],
                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return [line for line in result.stdout.splitlines() if line.startswith('tests/') and '::' in line]


@pytest.mark.parametrize('family', SHARDING_LANES)
def test_flow_sharding_ci_keeps_exact_provider_dependencies_and_requires_six_native_proofs(family):
    workflow, job, step, receipt = _lane(family, SHARDING_LANES)
    _, old_job, old_step, _ = _lane(family)
    assert job == old_job  # Reuse the original provider lane and its pinned stack.
    environment = {**job['env'], **step.get('env', {})}
    assert environment['DMD_MODERN_FLOW_CPU'] == '1' and environment['CUDA_VISIBLE_DEVICES'] == ''
    assert environment['HF_HUB_OFFLINE'] == environment['TRANSFORMERS_OFFLINE'] == '1'
    assert step.get('if') == old_step.get('if')
    assert 'continue-on-error' not in step and 'continue-on-error' not in job
    assert 'if' not in job and workflow['permissions'] == {'contents': 'read'}
    assert not {'DMD_FLOW_METHOD_IMAGE_CONFIG', 'DMD_FLOW_METHOD_LTX_CONFIG'} & environment.keys()
    proofs = _sharding_proofs(family)
    proof_command, gate_command = _commands(step)
    assert proof_command == ['python', '-m', 'pytest', '-q',
        *[f'{file}::{name}' for file, name in proofs], f'--junitxml={receipt}']
    assert gate_command == ['python', 'scripts/check_cpu_receipt.py', receipt,
        *[word for _, name in proofs for word in ('--require', name)]]
    paths = workflow['on']['pull_request']['paths']
    assert 'tests/test_open_workload_*.py' in paths
    assert all(any(Path(file).match(pattern) for pattern in paths) for file, _ in proofs)
    upload = next(item for item in job['steps'] if item.get('uses') == 'actions/upload-artifact@v4'
                  and receipt in item['with']['path'].splitlines())
    assert 'always()' in upload['if'] and upload['with']['if-no-files-found'] == 'error'
    if family == 'ltx25':
        assert step['if'] == "${{ matrix.family == 'ltx25' }}"
        assert upload['if'] == "${{ always() && matrix.family == 'ltx25' }}"
    else:
        assert 'if' not in step and 'LTX_SOURCE_ROOT' not in environment


@pytest.mark.parametrize('family', SHARDING_LANES)
def test_flow_sharding_ci_exact_native_node_ids_are_actually_collected(family):
    _, _, step, _ = _lane(family, SHARDING_LANES)
    command, _ = _commands(step)
    assert sorted(_collect_command(command)) == sorted(
        f'{file}::{name}' for file, name in _sharding_proofs(family))


@pytest.mark.parametrize('family', SHARDING_LANES)
@pytest.mark.parametrize('mutation', [None, 'missing-case', 'other-provider', 'skipped',
                                     'failure', 'error', 'empty', 'missing-file'])
def test_flow_sharding_ci_actual_receipt_gate_rejects_false_green(tmp_path, family, mutation):
    _, _, step, receipt = _lane(family, SHARDING_LANES)
    other = next(name for name in SHARDING_LANES if name != family)
    _assert_receipt_gate(tmp_path, step, receipt, [name for _, name in _sharding_proofs(family)],
                         mutation, [name for _, name in _sharding_proofs(other)])


def _h3_stop_lane():
    workflow = yaml.load((ROOT / '.github/workflows/h3-grid-native-cpu.yml').read_text(), Loader=yaml.BaseLoader)
    job = workflow['jobs']['original-source-state']
    proof_step = next(step for step in job['steps'] if step.get('name') ==
        'Actual original loop, data cursor, atomic state and rejection tests')
    gate_step = next(step for step in job['steps'] if step.get('name') ==
        'Do not silently skip the original-source qualification')
    assert 'if' not in job and 'continue-on-error' not in job
    for step in (proof_step, gate_step):
        assert 'if' not in step and 'continue-on-error' not in step
    return workflow, job, {'run': proof_step['run'] + '\n' + gate_step['run']}


def test_h3_ci_requires_durable_stop_and_current_original_restart_without_skips():
    workflow, job, step = _h3_stop_lane()
    assert job['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert job['env']['HF_HUB_OFFLINE'] == job['env']['TRANSFORMERS_OFFLINE'] == '1'
    assert job['env']['DMD_H3_TEST_UPSTREAM'] == '${{ github.workspace }}/third_party/lightx2v'
    assert workflow['permissions'] == {'contents': 'read'}
    paths = workflow['on']['pull_request']['paths']
    assert 'tests/test_framework_flow_methods_ci.py' in paths and 'scripts/check_cpu_receipt.py' in paths
    assert all(any(Path(file).match(pattern) for pattern in paths) for file in H3_FILES)
    proof_command, gate_command = _commands(step)
    assert proof_command == ['python', '-m', 'pytest', '-q', *H3_FILES, '--junitxml=h3-grid-cpu.xml']
    assert gate_command == ['python', 'scripts/check_cpu_receipt.py', 'h3-grid-cpu.xml',
        *[word for name in H3_PROOFS for word in ('--require', name)]]
    upload = next(item for item in job['steps'] if item.get('uses') == 'actions/upload-artifact@v4')
    assert upload['if'] == 'always()' and upload['with']['path'] == 'h3-grid-cpu.xml'
    assert upload['with']['if-no-files-found'] == 'error'


def test_h3_ci_required_stop_and_restart_names_are_actually_collected():
    _, _, step = _h3_stop_lane()
    command, _ = _commands(step)
    names = {node.split('::', 1)[1] for node in _collect_command(command)}
    assert set(H3_PROOFS) <= names


@pytest.mark.parametrize('mutation', [None, 'missing-case', 'skipped', 'failure', 'error', 'empty', 'missing-file'])
def test_h3_ci_actual_stop_receipt_gate_rejects_false_green(tmp_path, mutation):
    _, _, step = _h3_stop_lane()
    _assert_receipt_gate(tmp_path, step, 'h3-grid-cpu.xml', H3_PROOFS, mutation)


def test_common_cpu_collector_and_triggers_include_sharding_and_ci_regressions():
    api = runpy.run_path(str(ROOT / 'scripts/run_framework_cpu_tests.py'))
    files = api['test_files'](ROOT)
    assert 'tests/test_framework_flow_methods_ci.py' in files
    assert {file for family in SHARDING_LANES for file, _ in _sharding_proofs(family)} <= set(files)
    workflow = yaml.load((ROOT / '.github/workflows/framework-cpu.yml').read_text(), Loader=yaml.BaseLoader)
    paths = workflow['on']['pull_request']['paths']
    assert 'tests/test_open_workload_*.py' in paths


NATIVE_LOADER_LANE = {
    'frozen-native-shards': ('modern-flow-cpu.yml', 'native-media-lifecycle',
        'Actual frozen native local-shard construction', 'native-shard-loading-cpu.xml'),
}
NATIVE_LOADER_FILE = 'tests/test_training_flow_sharded_weight_loading.py'
NATIVE_LOADER_PROOFS = [
    'test_actual_native_frozen_shard_loader_cpu2[flux2-float32]',
    'test_class_and_inherited_tensor_constants_rejected_without_descriptors_cpu2',
]


def test_native_shard_loader_ci_reuses_modern_lane_and_requires_both_actual_proofs():
    workflow, job, step, receipt = _lane('frozen-native-shards', NATIVE_LOADER_LANE)
    _, original_job, _, _ = _lane('flux2-klein-base')
    assert job == original_job
    assert job['env']['DMD_MODERN_FLOW_CPU'] == '1'
    assert job['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert job['env']['HF_HUB_OFFLINE'] == job['env']['TRANSFORMERS_OFFLINE'] == '1'
    assert 'if' not in step and 'continue-on-error' not in step and 'continue-on-error' not in job
    assert workflow['permissions'] == {'contents': 'read'}
    proof_command, gate_command = _commands(step)
    assert proof_command == ['python', '-m', 'pytest', '-q',
        *[f'{NATIVE_LOADER_FILE}::{name}' for name in NATIVE_LOADER_PROOFS], f'--junitxml={receipt}']
    assert gate_command == ['python', 'scripts/check_cpu_receipt.py', receipt,
        *[word for name in NATIVE_LOADER_PROOFS for word in ('--require', name)]]
    paths = workflow['on']['pull_request']['paths']
    assert 'src/distillschedule/**' in paths
    assert all(any(Path(name).match(pattern) for pattern in paths) for name in
               (NATIVE_LOADER_FILE, 'tests/test_framework_flow_methods_ci.py'))
    upload = next(item for item in job['steps'] if item.get('uses') == 'actions/upload-artifact@v4'
                  and receipt in item['with']['path'].splitlines())
    assert upload['if'] == 'always()' and upload['with']['if-no-files-found'] == 'error'


def test_native_shard_loader_ci_exact_nodes_are_actually_collected():
    _, _, step, _ = _lane('frozen-native-shards', NATIVE_LOADER_LANE)
    proof_command, _ = _commands(step)
    assert sorted(_collect_command(proof_command)) == sorted(
        f'{NATIVE_LOADER_FILE}::{name}' for name in NATIVE_LOADER_PROOFS)


@pytest.mark.parametrize('mutation', [None, 'missing-case', 'other-provider', 'skipped',
                                     'failure', 'error', 'empty', 'missing-file'])
def test_native_shard_loader_ci_real_receipt_gate_refuses_false_green(tmp_path, mutation):
    """SYNTHETIC_TEST_ONLY metadata; native model proofs have separate receipts."""
    _, _, step, receipt = _lane('frozen-native-shards', NATIVE_LOADER_LANE)
    _assert_receipt_gate(tmp_path, step, receipt, NATIVE_LOADER_PROOFS, mutation,
                         ['unqualified-' + name for name in NATIVE_LOADER_PROOFS])


NATIVE_CONSTRUCTION_HANDOFF_LANE = {
    'flux2': ('modern-flow-cpu.yml', 'native-media-lifecycle',
        'Actual FLUX2 construction-provider lifecycle and fresh-process restore',
        'flux2-construction-handoff-cpu.xml'),
}
NATIVE_CONSTRUCTION_HANDOFF_FILE = 'tests/test_training_sharded_construction.py'
NATIVE_CONSTRUCTION_HANDOFF_PROOFS = [
    'test_caller_pre_sharded_actual_native_root_still_rejected[flux2]',
    'test_public_provider_actual_native_teacher_adopt_update_and_new_process_restore[flux2]',
    'test_real_frozen_handoff_refusals_are_collective_before_gather_or_update[flux2]',
    'test_actual_flux2_public_factory_train_and_new_process_resume',
    'test_failed_new_construction_expires_unpublished_root_without_rebinding_live_state[flux2]',
    'test_issued_native_and_child_forward_rebinding_rejected_before_signatures[native_forward-flux2]',
    'test_issued_native_and_child_forward_rebinding_rejected_before_signatures[child_forward-flux2]',
    'test_asymmetric_cached_construction_mesh_rejected_before_new_cohort[flux2]',
]


def test_native_construction_handoff_ci_reuses_modern_lane_and_requires_actual_flux2_proofs():
    workflow, job, step, receipt = _lane('flux2', NATIVE_CONSTRUCTION_HANDOFF_LANE)
    _, original_job, _, _ = _lane('flux2-klein-base')
    assert job == original_job
    environment = {**job['env'], **step.get('env', {})}
    assert environment['DMD_MODERN_FLOW_CPU'] == '1' and environment['CUDA_VISIBLE_DEVICES'] == ''
    assert environment['HF_HUB_OFFLINE'] == environment['TRANSFORMERS_OFFLINE'] == '1'
    assert not {'DMD_FLOW_METHOD_IMAGE_CONFIG', 'DMD_FLOW_METHOD_LTX_CONFIG', 'LTX_SOURCE_ROOT'} & environment.keys()
    assert 'if' not in step and 'if' not in job
    assert 'continue-on-error' not in step and 'continue-on-error' not in job
    assert workflow['permissions'] == {'contents': 'read'}
    commands = '\n'.join(item.get('run', '') for item in job['steps'])
    assert 'torch==2.10.0' in commands and 'https://download.pytorch.org/whl/cpu' in commands
    assert '.[test,modern-flow]' in commands and 'ltx25' not in commands
    proof_command, gate_command = _commands(step)
    assert proof_command == ['python', '-m', 'pytest', '-q',
        *[f'{NATIVE_CONSTRUCTION_HANDOFF_FILE}::{name}' for name in NATIVE_CONSTRUCTION_HANDOFF_PROOFS],
        f'--junitxml={receipt}']
    assert gate_command == ['python', 'scripts/check_cpu_receipt.py', receipt,
        *[word for name in NATIVE_CONSTRUCTION_HANDOFF_PROOFS for word in ('--require', name)]]
    paths = workflow['on']['pull_request']['paths']
    assert 'src/distillschedule/**' in paths and 'scripts/check_cpu_receipt.py' in paths
    assert all(any(Path(name).match(pattern) for pattern in paths) for name in
               (NATIVE_CONSTRUCTION_HANDOFF_FILE, 'tests/test_framework_flow_methods_ci.py'))
    upload = next(item for item in job['steps'] if item.get('uses') == 'actions/upload-artifact@v4'
                  and receipt in item['with']['path'].splitlines())
    assert upload['if'] == 'always()' and upload['with']['if-no-files-found'] == 'error'


def test_native_construction_handoff_ci_exact_nodes_are_actually_collected():
    _, _, step, _ = _lane('flux2', NATIVE_CONSTRUCTION_HANDOFF_LANE)
    proof_command, _ = _commands(step)
    assert sorted(_collect_command(proof_command)) == sorted(
        f'{NATIVE_CONSTRUCTION_HANDOFF_FILE}::{name}' for name in NATIVE_CONSTRUCTION_HANDOFF_PROOFS)


@pytest.mark.parametrize('mutation', [None, 'missing-case', 'other-provider', 'skipped',
                                     'failure', 'error', 'empty', 'missing-file'])
def test_native_construction_handoff_ci_real_receipt_gate_refuses_false_green(tmp_path, mutation):
    """SYNTHETIC_TEST_ONLY metadata; not a new native model qualification."""
    _, _, step, receipt = _lane('flux2', NATIVE_CONSTRUCTION_HANDOFF_LANE)
    _assert_receipt_gate(tmp_path, step, receipt, NATIVE_CONSTRUCTION_HANDOFF_PROOFS, mutation,
                         [name.replace('flux2', 'flux1') for name in NATIVE_CONSTRUCTION_HANDOFF_PROOFS])
