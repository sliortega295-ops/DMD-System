"""Dependency-separated CI must actually execute the native CPU proofs."""
from pathlib import Path
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def job(name, job_id=None):
    workflow = yaml.safe_load((ROOT / '.github/workflows' / name).read_text())
    assert workflow['permissions'] == {'contents': 'read'}
    if job_id is not None:
        return workflow['jobs'][job_id]
    assert len(workflow['jobs']) == 1
    return next(iter(workflow['jobs'].values()))


def test_cached_h3_ci_uses_its_actual_small_source_and_public_lifecycle():
    value = job('h3-cached-reference-cpu.yml', 'original-cached-av')
    commands = '\n'.join(step.get('run', '') for step in value['steps'])
    assert "'.[train,test,h3-cached-reference]'" in commands
    assert 'torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu' in commands
    assert all(name in commands for name in (
        'tests/test_h3_cached_source_provider.py', 'tests/test_h3_cached_original_cpu.py',
        'tests/test_data_h3_cached_inputs.py', 'tests/test_training_h3_cached_public_cpu.py',
        'test_common_checkpoint_fresh_os_process_exact_gan_transition[False]',
        'test_common_checkpoint_fresh_os_process_exact_gan_transition[True]'))
    assert 'test-helios' not in commands and 'native-wan' not in commands
    assert value['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert value['strategy']['matrix']['python'] == ['3.10', '3.12']
    assert "('skipped', 'failures', 'errors')" in commands


def test_cached_h3_ci_requires_current_student_export_and_fresh_AV_inference():
    value = job('h3-cached-reference-cpu.yml', 'original-cached-av')
    commands = '\n'.join(step.get('run', '') for step in value['steps'])
    assert 'tests/test_training_h3_cached_inference_cpu.py' in commands
    assert all(name in commands for name in (
        'test_projection_is_student_only_nonmutating_and_next_update_exact',
        'test_common_export_actual_current_student_and_not_optimizer',
        'test_inference_factory_is_separate_and_torch_free_at_discovery',
        'test_fresh_process_only_export_and_NEW_conditions_no_C_Adam_old_reads[flat]',
        'test_fresh_process_only_export_and_NEW_conditions_no_C_Adam_old_reads[packed]'))


def test_full_original_helios_ci_requires_public_training_and_portable_current_g():
    value = job('helios-reference-model-cpu.yml')
    commands = '\n'.join(step.get('run', '') for step in value['steps'])
    assert "'.[train,test,test-helios-model]'" in commands
    assert value['env']['HELIOS_PUBLIC_REQUIRE_STACK'] == '1'
    assert 'tests' in value['env']['PYTHONPATH'].split(':')
    assert all(name in commands for name in (
        'tests/test_training_helios_public_cpu.py', 'tests/test_data_helios_native_inputs.py',
        'tests/test_training_helios_public_inference_cpu.py',
        'test_public_fresh_os_process_four_vs_three_resume_exact[True]',
        'test_public_fresh_os_process_four_vs_three_resume_exact[False]',
        'test_full_chain_fresh_process_uses_only_export_source_runtime_and_NEW_prompt',
        'tests/test_training_helios_native_safe_stream.py',
        'test_real_native_stream_producer_consumer_lowering_and_separate_checkpoint_identity',
        'test_actual_checkpoint_publication_and_old_policy_refusal_for_new_identity',
        'test_planner_consumer_cannot_treat_full_safe_action_as_old_reference_phase',
        'test_native_safe_recipe_cold_discovery_avoids_torch_cuda_and_model_assets',
        'tests/test_training_helios_checkpoint_inputs_cpu.py',
        'test_native_shape_contract_is_exact_three_original_stages_and_partitioned_node_bound',
        'test_native_receipt_refuses_wrong_source_version_shape_dtype_or_lifetime[version]',
        'test_shared_checkpoint_input_mutation_still_refuses_without_a_cpu_copy[direct-True]',
        'test_shared_checkpoint_input_mutation_still_refuses_without_a_cpu_copy[detached_alias_after_gc-True]',
        'recipes/helios_nogan/verify_checkpoint_inputs_target_cpu.py'))
    text = (ROOT / '.github/workflows/helios-reference-model-cpu.yml').read_text()
    assert "- 'tests/test_training_helios_native_safe_stream.py'" in text
    assert "- 'recipes/helios_nogan/training.native-history-safe.json'" in text
    assert "- 'tests/test_training_helios_checkpoint_inputs_cpu.py'" in text
    assert "- 'recipes/helios_nogan/**'" in text
    assert 'native-wan' not in commands
    assert "('skipped', 'failures', 'errors')" in commands


def test_original_helper_ci_is_not_silently_repinned_to_full_model_extra():
    value = job('framework-cpu.yml')
    commands = '\n'.join(step.get('run', '') for step in value['steps'])
    assert "'.[train,test,test-helios]'" in commands
    assert 'test-helios-model' not in commands and 'native-wan' not in commands


def test_original_helios_ci_requires_checkpoint_liveness_and_cross_section_gradients():
    value = job('helios-reference-model-cpu.yml')
    commands = '\n'.join(step.get('run', '') for step in value['steps'])
    assert 'tests/test_helios_rollout_liveness_cpu.py' in commands
    assert 'DMD_TEST_REARRANGE_PROVIDER' not in value['env']
    assert '-k ' not in commands and '--deselect' not in commands
    for name in (
        'test_liveness_fixture_restores_rng_threads_and_import_namespaces',
        'test_candidate_changes_only_two_stored_aliases',
        'test_default_checkpoint_cpp_storage_lifetime_without_saved_tensor_hooks[exits0]',
    ):
        assert repr(name) in commands
    for hooks in (False, True):
        for sections, frames in ((2, 4), (3, 4), (3, 6)):
            name = ('test_cross_section_generated_history_keeps_required_grads_and_only_frees_dead_state'
                    f'[{hooks}-{sections}-{frames}]')
            assert repr(name) in commands


def test_livewan_native_ci_requires_payload_lifetime_and_public_binding_proofs():
    value = job('livewan-native-inputs-cpu.yml')
    commands = '\n'.join(step.get('run', '') for step in value['steps'])
    assert 'tests/test_training_livewan_text_kv_retirement.py' in commands
    assert 'tests/test_training_livewan_c51_diagnostics.py' in commands
    assert value['env']['DMD_LIVEWAN_TEST_UPSTREAM'].endswith('/third_party/livewan')
    for name in (
        'test_real_nonreentrant_checkpoint_output_all_grads_adam_rng_and_counters_equal',
        'test_actual_prepare_rollout_runahead_and_fresh_resume_prime_retire_same_payload',
        'test_typed_public_config_native_runner_reaches_actual_diagnostic_setup[self_forcing.c51_stream.v1]',
        'test_typed_public_config_native_runner_reaches_actual_diagnostic_setup[self_forcing.c51_stream_retire_textkv.v1]',
    ):
        assert repr(name) in commands


def test_original_wan_ci_requires_mixed_precision_update_restart_and_current_g():
    value = job('helios-reference-model-cpu.yml')
    commands = '\n'.join(step.get('run', '') for step in value['steps'])
    assert value['env']['WAN_PUBLIC_REQUIRE_STACK'] == '1'
    assert all(name in commands for name in (
        'tests/test_training_wan_public_cpu.py',
        'tests/test_training_wan_inference_cpu.py',
        'test_real_original_bf16_program_two_distinct_cohorts',
        'test_fresh_os_process_full4_vs3_plus_exact_restart',
        'test_partial_cycle_poison_and_safe_all_owner_restore',
        'test_fresh_process_new_prompt_only_no_old_assets_training_or_optimizer'))
    assert "('skipped', 'failures', 'errors')" in commands


def test_independent_nogan_ci_requires_real_warm_restart_and_new_prompt_output():
    value = job('helios-nogan-reference-cpu.yml')
    commands = '\n'.join(step.get('run', '') for step in value['steps'])
    assert value['env']['HELIOS_NOGAN_PUBLIC_REQUIRE_STACK'] == '1'
    assert value['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert value['strategy']['matrix']['python'] == ['3.10', '3.12']
    assert 'tests' in value['env']['PYTHONPATH'].split(':')
    assert 'runner.temp' in value['env']['HELIOS_NOGAN_PUBLIC_FIXTURE']
    assert "'.[train,test,helios-reference]'" in commands
    assert 'torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu' in commands
    assert all(name in commands for name in (
        'tests/test_training_helios_nogan_bootstrap_cpu.py',
        'tests/test_training_helios_nogan_public_cpu.py',
        'tests/test_data_helios_nogan_reference_inputs.py',
        'tests/test_training_helios_nogan_inference_cpu.py',
        'test_actual_original_warm_public_two_fresh_cycles',
        'test_fresh_process_exact_uninterrupted3_vs1_plus2',
        'test_outer_zero_actual_warm_boundary_recovers_first_partial_cycle',
        'test_actual_restored_full_current_generator_projection_and_export',
        'test_fresh_process_new_prompt_without_old_paths_owners_or_optimizer'))
    assert "('skipped', 'failures', 'errors')" in commands
    assert 'native-wan' not in commands and 'test-helios]' not in commands


def test_nogan_ci_releases_its_large_training_fixture_before_independent_output():
    value = job('helios-nogan-reference-cpu.yml')
    runs = [step.get('run', '') for step in value['steps']]
    train = next(i for i, text in enumerate(runs)
                 if 'pytest' in text and 'test_training_helios_nogan_public_cpu.py' in text)
    output = next(i for i, text in enumerate(runs)
                  if 'pytest' in text and 'test_training_helios_nogan_inference_cpu.py' in text)
    cleanup = next(i for i, text in enumerate(runs) if 'shutil.rmtree' in text)
    assert train < cleanup < output
    assert 'helios-nogan-training.xml' in runs[train]
    assert 'helios-nogan-inference.xml' in runs[output]
    assert 'RUNNER_TEMP' in runs[cleanup] and 'resolve() == path' in runs[cleanup]
    assert 'test_training_helios_nogan_inference_cpu.py' not in runs[train]


@pytest.mark.parametrize('mutation', [None, 'suite-skip', 'suite-failure', 'suite-error',
                                     'case-skip', 'missing-proof', 'missing-file', 'empty'])
def test_nogan_actual_xml_gate_rejects_false_green_reports(tmp_path, mutation):
    """SYNTHETIC_TEST_ONLY: run the real YAML consumer, not model qualification."""
    value = job('helios-nogan-reference-cpu.yml')
    command = next(step['run'] for step in value['steps']
                   if step.get('name') == 'Require real full-source proofs without skips')
    names = [
        'test_complete_original_layout_real_adam_reader_state_and_next_commit',
        'test_actual_original_warm_public_two_fresh_cycles',
        'test_fresh_process_exact_uninterrupted3_vs1_plus2',
        'test_partial_g_teacher_c_and_restore_failures_poison_all_owners',
        'test_outer_zero_actual_warm_boundary_recovers_first_partial_cycle',
        'test_actual_restored_full_current_generator_projection_and_export',
        'test_fresh_process_new_prompt_without_old_paths_owners_or_optimizer',
    ]
    for index, filename in enumerate(('helios-nogan-training.xml', 'helios-nogan-inference.xml')):
        root = ET.Element('testsuites')
        suite = ET.SubElement(root, 'testsuite', tests=str(5 if index == 0 else 2),
                              failures='0', errors='0', skipped='0')
        for name in (names[:5] if index == 0 else names[5:]):
            ET.SubElement(suite, 'testcase', name=name)
        if index == 1:
            if mutation in ('suite-skip', 'suite-failure', 'suite-error'):
                suite.set({'suite-skip':'skipped', 'suite-failure':'failures',
                           'suite-error':'errors'}[mutation], '1')
            elif mutation == 'case-skip':
                ET.SubElement(suite[0], 'skipped')
            elif mutation == 'missing-proof':
                suite.remove(suite[-1])
            elif mutation == 'empty':
                root.remove(suite)
            elif mutation == 'missing-file':
                continue
        ET.ElementTree(root).write(tmp_path / filename)
    binary = tmp_path / 'bin'
    binary.mkdir()
    (binary / 'python').symlink_to(sys.executable)
    result = subprocess.run(['bash', '-e', '-c', command], cwd=tmp_path,
        env={**os.environ, 'PATH':str(binary) + os.pathsep + os.environ.get('PATH', '')},
        capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) is (mutation is None), result.stdout + result.stderr


@pytest.mark.parametrize('alias', [False, True])
def test_nogan_ci_cleanup_is_owned_and_refuses_symlink_targets(tmp_path, alias):
    value = job('helios-nogan-reference-cpu.yml')
    command = next(step['run'] for step in value['steps']
                   if step.get('name') == "Release only this job's completed training fixture")
    root, outside = tmp_path / 'runner', tmp_path / 'not-this-jobs-data'
    root.mkdir()
    outside.mkdir()
    (outside / 'keep').write_text('unrelated input')
    fixture = root / 'helios-nogan-reference'
    if alias:
        fixture.symlink_to(outside, target_is_directory=True)
    else:
        fixture.mkdir()
    temporary = root / 'helios-nogan-training-pytest'
    temporary.mkdir()
    binary = tmp_path / 'bin'
    binary.mkdir()
    (binary / 'python').symlink_to(sys.executable)
    result = subprocess.run(['bash', '-e', '-c', command], cwd=tmp_path,
        env={**os.environ, 'RUNNER_TEMP':str(root),
             'PATH':str(binary) + os.pathsep + os.environ.get('PATH', '')},
        capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) is (not alias), result.stdout + result.stderr
    assert (outside / 'keep').read_text() == 'unrelated input'
    assert temporary.exists() is alias
    assert fixture.exists() is alias
