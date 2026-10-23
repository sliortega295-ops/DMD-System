"""Dependency-separated CI must actually execute the native CPU proofs."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def job(name):
    workflow = yaml.safe_load((ROOT / '.github/workflows' / name).read_text())
    assert workflow['permissions'] == {'contents': 'read'}
    assert len(workflow['jobs']) == 1
    return next(iter(workflow['jobs'].values()))


def test_cached_h3_ci_uses_its_actual_small_source_and_public_lifecycle():
    value = job('h3-cached-reference-cpu.yml')
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
    value = job('h3-cached-reference-cpu.yml')
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
        'test_full_chain_fresh_process_uses_only_export_source_runtime_and_NEW_prompt'))
    assert 'native-wan' not in commands
    assert "('skipped', 'failures', 'errors')" in commands


def test_original_helper_ci_is_not_silently_repinned_to_full_model_extra():
    value = job('framework-cpu.yml')
    commands = '\n'.join(step.get('run', '') for step in value['steps'])
    assert "'.[train,test,test-helios]'" in commands
    assert 'test-helios-model' not in commands and 'native-wan' not in commands


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
