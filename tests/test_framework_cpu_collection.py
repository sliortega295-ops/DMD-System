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
    for name in (*names, *_api()['EXCLUDED']):
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


def test_native_attention_capture_seam_is_in_actual_common_lane_and_triggers_ci():
    import yaml
    name = 'tests/test_refactor_h3_attention_capture_entrypoint.py'
    assert name in _api()['test_files'](ROOT)
    workflow = yaml.load((ROOT / '.github/workflows/framework-cpu.yml').read_text(), Loader=yaml.BaseLoader)
    assert name in workflow['on']['pull_request']['paths']


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


H3_LANES = {
    'grid-output': (
        'h3-grid-output-cpu.yml', 'original-grid-output',
        'Actual converted model, original sampler and current Student outputs',
        'h3-grid-output-cpu.xml',
        ['tests/test_training_h3_grid_outputs.py', 'tests/test_training_h3_grid_native_outputs.py'],
        ['test_grid_original_source_schedule_and_projection_identity',
         'test_actual_grid_export_restore_four_step_rollout_matches_original_projection',
         'test_grid_public_export_and_new_process_generate',
         'test_real_current_grid_callback_common_export_and_source_guard',
         'test_actual_checkpoint_restore_precedes_output_and_performs_zero_updates',
         'test_real_gloo_dtensor_export_gathers_all_current_weight_shards']),
    'consolidation': (
        'h3-cached-reference-cpu.yml', 'original-cached-av',
        'Actual installed DeepSpeed Student consolidation on CPU',
        'h3-native-consolidation-cpu.xml',
        ['tests/test_training_h3_native_consolidation.py'],
        ['test_current_unmerged_student_uses_real_installed_method_and_keeps_owner',
         'test_invalid_owner_or_capacity_fails_before_native_gather[method]',
         'test_nonprimary_returns_no_cpu_snapshot',
         'test_backend_failure_restores_rng_and_preserves_error']),
}


def _h3_lane(name):
    import shlex
    import yaml
    filename, owner, step_name, receipt, files, proofs = H3_LANES[name]
    workflow = yaml.load((ROOT / '.github/workflows' / filename).read_text(), Loader=yaml.BaseLoader)
    job = workflow['jobs'][owner]
    step = next(step for step in job['steps'] if step.get('name') == step_name)
    commands = [shlex.split(line) for line in step['run'].replace('\\\n', ' ').splitlines() if line.strip()]
    return workflow, job, commands, receipt, files, proofs


@pytest.mark.parametrize('name', H3_LANES)
def test_h3_provider_collection_has_actual_dependency_lane_and_mandatory_receipt(name):
    workflow, job, commands, receipt, files, proofs = _h3_lane(name)
    assert {Path(path).name for path in files} <= _api()['EXCLUDED']
    assert not set(files) & set(_api()['test_files'](ROOT))
    assert commands == [
        ['python', '-m', 'pytest', '-q', *files, f'--junitxml={receipt}'],
        ['python', 'scripts/check_cpu_receipt.py', receipt,
         *[word for proof in proofs for word in ('--require', proof)]],
    ]
    assert job['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert job['env']['HF_HUB_OFFLINE'] == job['env']['TRANSFORMERS_OFFLINE'] == '1'
    assert 'tests' in job['env']['PYTHONPATH'].split(':')
    assert 'if' not in job and 'continue-on-error' not in job
    assert workflow['permissions'] == {'contents': 'read'}
    paths = workflow['on']['pull_request']['paths']
    assert all(any(Path(path).match(pattern) for pattern in paths) for path in files)
    assert 'tests/test_framework_cpu_collection.py' in paths
    upload = next(step for step in job['steps'] if step.get('uses') == 'actions/upload-artifact@v4')
    assert upload['if'] == 'always()' and receipt in upload['with']['path'].splitlines()
    assert upload['with']['if-no-files-found'] == 'error'
    installs = '\n'.join(step.get('run', '') for step in job['steps'])
    if name == 'consolidation':
        assert "'.[test-h3-consolidation]'" in installs
        assert job['env']['DS_ACCELERATOR'] == 'cpu' and job['env']['DS_BUILD_OPS'] == '0'
    else:
        assert "'.[train,test,test-h3-output]'" in installs and 'pip check' in installs
        assert 'scripts/prepare_h3_grid_output_test_source.py' in installs
        sources = {step['with']['repository']: step['with']['ref'] for step in job['steps']
                   if 'repository' in step.get('with', {})}
        assert sources == {'ModelTC/LightX2V': 'd034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be',
                           'huggingface/diffusers': '9284607295a09f759aadd65ed08f48b35feea6d9'}


@pytest.mark.parametrize('name', H3_LANES)
def test_migrated_h3_provider_proofs_are_collected_not_just_named(name):
    from test_framework_flow_methods_ci import _collect_command
    _, _, commands, _, _, proofs = _h3_lane(name)
    names = {node.split('::', 1)[1] for node in _collect_command(commands[0])}
    assert set(proofs) <= names


@pytest.mark.parametrize('name', H3_LANES)
@pytest.mark.parametrize('mutation', [None, 'missing-proof', 'skipped', 'failure', 'error', 'empty', 'missing-file'])
def test_migrated_h3_actual_receipt_gate_rejects_false_green(tmp_path, name, mutation):
    import xml.etree.ElementTree as ET
    _, _, commands, receipt, _, proofs = _h3_lane(name)
    root = ET.Element('testsuite')
    for proof in proofs:
        ET.SubElement(root, 'testcase', name=proof)
    if mutation == 'missing-proof': root.remove(root[-1])
    elif mutation in ('skipped', 'failure', 'error'): ET.SubElement(root[-1], mutation)
    elif mutation == 'empty': root.clear()
    if mutation != 'missing-file': ET.ElementTree(root).write(tmp_path / receipt)
    gate = commands[1]
    result = subprocess.run([sys.executable, str(ROOT / gate[1]), *gate[2:]],
                            cwd=tmp_path, capture_output=True, text=True)
    assert (result.returncode == 0) is (mutation is None), result.stdout + result.stderr


def test_h3_grid_review_restores_existing_source_identity_without_relabeling_it():
    import hashlib
    import json
    from distillschedule.training.h3_grid_inference import MODEL_SHA, SOURCE_FILES, TRAINER
    fixture = ROOT / 'tests/fixtures/h3_grid_output_source.json'
    review = json.loads(fixture.read_text())
    assert review['trainer'] == TRAINER
    assert review['reviewed_trainer_sha256'] == SOURCE_FILES[TRAINER]
    assert review['diffusers']['model_sha256'] == MODEL_SHA
    assert review['upstream_trainer_sha256'] != review['reviewed_trainer_sha256']
    assert hashlib.sha256(fixture.with_suffix('.patch').read_bytes()).hexdigest() == review['patch_sha256']


MODEL_LANES = {
    'native-topology': (
        'helios-native-topology-cpu.yml', 'original-native-topology',
        'Actual original native topology in a separate namespace process',
        'helios-native-topology-cpu.xml',
        ['tests/test_helios_native_topology_cpu.py',
         'tests/test_training_helios_native_history.py::test_cpu_topology_import_probe_calls_original_and_preserves_success',
         'tests/test_training_helios_native_history.py::test_cpu_topology_import_probe_only_normalizes_missing_cpu_cuda'],
        ['test_target_actual_helios_block_class_checkpoint_and_cpu_fsdp_ownership',
         'test_cpu_topology_import_probe_calls_original_and_preserves_success',
         *[f'test_cpu_topology_import_probe_only_normalizes_missing_cpu_cuda[{case}]' for case in (
             'None-error0-True', '12.8-error1-False', 'None-error2-False', 'None-error3-False', 'None-error4-False')]]),
    'wan-import': (
        'helios-reference-model-cpu.yml', 'original-full-model',
        'Original Wan Generator import, export and fresh-process inference',
        'wan-generator-import-upstream-cpu.xml',
        ['tests/test_training_wan_generator_import_upstream_cpu.py'],
        [*[f'test_actual_upstream_import_cli_export_and_new_process_new_prompt[{case}]'
           for case in ('False-False', 'True-False', 'True-True')],
         *[f'test_actual_upstream_low_export_budget_refuses_before_publication[{case}]'
           for case in ('max_export_bytes_per_rank', 'max_export_bytes_all_ranks')],
         'test_actual_from_config_retains_rope_metadata_without_changing_constructor_state_or_dense_output',
         *[f'test_actual_upstream_nonconstructor_config_projection_and_public_import_are_exact[{case}]'
           for case in ('rope_max_seq_len', 'unknown_native_config_field')],
         'test_actual_upstream_compacted_peft_targets_project_import_and_strict_reload_are_exact',
         *[f'test_actual_upstream_compacted_peft_projection_still_refuses_invalid_adapter_inventory[{case}]'
           for case in ('missing_adapter', 'extra_adapter', 'wrong_default', 'merged')]]),
}


def _model_lane(name):
    import shlex
    import yaml
    filename, owner, step_name, receipt, files, proofs = MODEL_LANES[name]
    workflow = yaml.load((ROOT / '.github/workflows' / filename).read_text(), Loader=yaml.BaseLoader)
    job = workflow['jobs'][owner]
    step = next(step for step in job['steps'] if step.get('name') == step_name)
    commands = [shlex.split(line) for line in step['run'].replace('\\\n', ' ').splitlines() if line.strip()]
    return workflow, job, step, commands, receipt, files, proofs


@pytest.mark.parametrize('name', MODEL_LANES)
def test_original_model_provider_proofs_keep_explicit_isolated_processes_and_receipts(name):
    from test_framework_flow_methods_ci import _collect_command
    workflow, job, step, commands, receipt, files, proofs = _model_lane(name)
    assert commands == [
        ['python', '-m', 'pytest', '-q', *files, f'--junitxml={receipt}'],
        ['python', 'scripts/check_cpu_receipt.py', receipt,
         *[word for proof in proofs for word in ('--require', proof)]],
    ]
    assert set(proofs) == {node.split('::', 1)[1] for node in _collect_command(commands[0])}
    assert workflow['permissions'] == {'contents': 'read'}
    assert job['strategy']['matrix']['python'] == ['3.10', '3.12']
    assert job['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert job['env']['HF_HUB_OFFLINE'] == job['env']['TRANSFORMERS_OFFLINE'] == '1'
    assert 'tests' in job['env']['PYTHONPATH'].split(':')
    assert 'if' not in job and 'continue-on-error' not in job
    assert 'if' not in step and 'continue-on-error' not in step
    assert 'tests/test_framework_cpu_collection.py' in workflow['on']['pull_request']['paths']
    upload = next(value for value in job['steps'] if value.get('uses') == 'actions/upload-artifact@v4')
    assert upload['if'] == 'always()' and receipt in upload['with']['path'].splitlines()
    assert upload['with']['if-no-files-found'] == 'error'
    installs = '\n'.join(value.get('run', '') for value in job['steps'])
    if name == 'wan-import':
        assert "'.[train,test,test-helios-model]'" in installs
        assert step['env']['R217_REQUIRE_UPSTREAM'] == '1'
        assert Path(files[0]).name in _api()['EXCLUDED'] and files[0] not in _api()['test_files'](ROOT)
        full = next(value for value in job['steps'] if value.get('name', '').startswith('Full original class'))
        assert job['steps'].index(step) < job['steps'].index(full)
        assert files[0] not in full['run']  # Separate OS process; no extra imports in the full-model proof.
    else:
        assert "'.[train,test,test-helios-model,test-helios-native-history]'" in installs and 'pip check' in installs
        assert files[0] not in _api()['test_files'](ROOT)
        assert 'tests/test_training_helios_native_history.py' in _api()['test_files'](ROOT)
        assert 'test_target_actual_helios_block_class_checkpoint_and_cpu_fsdp_ownership' not in (
            ROOT / 'tests/test_helios_reference_model_cpu.py').read_text()
        full_workflow = (ROOT / '.github/workflows/helios-reference-model-cpu.yml').read_text()
        assert 'test-helios-native-history' not in full_workflow and files[0] not in full_workflow


@pytest.mark.parametrize('name', MODEL_LANES)
@pytest.mark.parametrize('mutation', [None, 'missing-proof', 'skipped', 'failure', 'error', 'empty', 'missing-file'])
def test_original_model_provider_receipt_gate_rejects_false_green(tmp_path, name, mutation):
    import xml.etree.ElementTree as ET
    _, _, _, commands, receipt, _, proofs = _model_lane(name)
    root = ET.Element('testsuite')
    for proof in proofs:
        ET.SubElement(root, 'testcase', name=proof)
    if mutation == 'missing-proof': root.remove(root[-1])
    elif mutation in ('skipped', 'failure', 'error'): ET.SubElement(root[-1], mutation)
    elif mutation == 'empty': root.clear()
    if mutation != 'missing-file': ET.ElementTree(root).write(tmp_path / receipt)
    gate = commands[1]
    result = subprocess.run([sys.executable, str(ROOT / gate[1]), *gate[2:]],
                            cwd=tmp_path, capture_output=True, text=True)
    assert (result.returncode == 0) is (mutation is None), result.stdout + result.stderr


def test_native_cpu_fixture_preloads_admit_the_real_source_helper_namespace():
    import ast
    for filename in ('test_training_helios_gan_native_history.py', 'test_training_helios_gan_state.py',
                     'test_training_helios_native_history.py', 'test_training_helios_original.py'):
        calls = [node for node in ast.walk(ast.parse((ROOT / 'tests' / filename).read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == 'native_primitives']
        assert len(calls) == 1
        assert not calls[0].args and len(calls[0].keywords) == 1
        keyword = calls[0].keywords[0]
        assert keyword.arg == 'source_only' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True


def test_operator_grid_codec_target_retains_all_four_actual_source_tests():
    import hashlib
    from test_framework_flow_methods_ci import _collect_command
    target = 'tests/test_h3_grid_codec_target_cpu.py'
    text = (ROOT / target).read_text()
    marker = 'from test_training_h3_grid_outputs import actual_grid_case'
    retained = text[text.index(marker):]
    # Exact pre-migration fixture and assertions; no substitute codec or skip.
    assert hashlib.sha256(retained.encode()).hexdigest() == '5a52ac84e7dd570af713f67a25d9a3a523fbef8efbd1838617c5b8652975c082'
    assert 'pytest.skip' not in retained and 'importorskip' not in retained
    expected = [f'{target}::test_incompatible_codec_domain_rejected_before_loading_models[{fault}]'
                for fault in ('frames', 'fps', 'audio_length')]
    expected.append(f'{target}::test_original_full_grid_sampler_current_student_decodes_in_new_public_process')
    assert sorted(_collect_command(['python', '-m', 'pytest', '-q', target])) == sorted(expected)
    common = _api()['test_files'](ROOT)
    assert target not in common and 'tests/test_training_h3_grid_codec.py' in common


def test_operator_grid_codec_missing_manifest_is_an_error_not_a_skip():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / path) for path in ('src', '.', 'tests')))
    env.pop('DMD_H3_GRID_CODEC_TEST_MANIFEST', None)
    script = ('from test_h3_grid_codec_target_cpu import reused_codec; '
              'reused_codec.__wrapped__()')
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "KeyError: 'DMD_H3_GRID_CODEC_TEST_MANIFEST'" in result.stderr


def test_original_cpu_noise_policies_are_scoped_to_their_actual_runtime_contracts():
    import yaml
    job=yaml.load((ROOT/'.github/workflows/helios-reference-model-cpu.yml').read_text(),Loader=yaml.BaseLoader)['jobs']['original-full-model']
    flag='DISTILLSCHEDULE_STABLE_BLOCK_NOISE'
    assert flag not in job['env']
    owner=next(s for s in job['steps'] if s.get('name','').startswith('Full original class'))
    native=next(s for s in job['steps'] if s.get('name')=='Explicit outputs and native contracts without inherited sampler override')
    assert owner['env'][flag]=='1' and flag not in native.get('env',{})
    for name in ('test_training_helios_public_inference_cpu.py','test_training_wan_inference_cpu.py',
                 'test_training_helios_optimized_stream.py','test_training_helios_native_safe_stream.py'):
        assert name in native['run'] and name not in owner['run']
    default=next(s for s in job['steps'] if s.get('name')=='Original default sampler and explicit PSD-policy contracts')
    output=next(s for s in job['steps'] if s.get('name')=='Explicit v2 output protocol and exact restart without inherited environment')
    assert flag not in default.get('env',{}) and flag not in output.get('env',{})
    assert 'tests/test_helios_reference_inference_noise_cpu.py' in output['run']
    assert 'tests/test_helios_reference_inference_noise_cpu.py' not in _api()['test_files'](ROOT)
    assert '--require test_native_gamma_one_third_noise_is_singular_psd_not_cholesky_pd' in default['run']
    for family in ('helios','wan'):
        for name in ('test_fresh_process_v2_export_and_generate_uses_real_operation_inputs_without_env',
                     'test_resume_identity_binds_explicit_output_policy','test_unselected_v1_keeps_original_call_and_environment_rejection'):
            assert "--require '"+name+'['+family+"]'" in output['run']


@pytest.mark.parametrize('mutation',[None,'owner-missing','native-missing','native-empty','missing-proof','skipped','failure','error'])
def test_scoped_owner_output_native_junit_consumer_requires_both_real_suites(tmp_path,mutation):
    """Actual CI consumer with SYNTHETIC_TEST_ONLY XML; not GPU evidence."""
    import ast,yaml
    import xml.etree.ElementTree as ET
    job=yaml.load((ROOT/'.github/workflows/helios-reference-model-cpu.yml').read_text(),Loader=yaml.BaseLoader)['jobs']['original-full-model']
    step=next(s for s in job['steps'] if s.get('name')=='Require the full-source proofs without skips')
    script=step['run'].split("python - <<'PY'\n",1)[1].rsplit('\nPY',1)[0]
    required=ast.literal_eval(next(n.value for n in ast.parse(script).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='required' for t in n.targets)))
    for index,path in enumerate(('helios-reference-model-cpu.xml','helios-native-policy-cpu.xml')):
        root=ET.Element('testsuite',name='SYNTHETIC_TEST_ONLY')
        for name in sorted(required)[index::2]:ET.SubElement(root,'testcase',name=name,classname='SYNTHETIC_TEST_ONLY')
        if index==1:
            if mutation=='native-empty':root.clear()
            elif mutation=='missing-proof':root.remove(root[-1])
            elif mutation in ('skipped','failure','error'):
                ET.SubElement(root[-1],mutation)
                root.set({'skipped':'skipped','failure':'failures','error':'errors'}[mutation],'1')
        if mutation!=('owner-missing' if index==0 else 'native-missing'):ET.ElementTree(root).write(tmp_path/path)
    result=subprocess.run([sys.executable,'-c',script],cwd=tmp_path,capture_output=True,text=True)
    assert (result.returncode==0) is (mutation is None),result.stdout+result.stderr
