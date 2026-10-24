"""Public optional stacks and mandatory real-upstream CPU qualification lanes."""
import json
from pathlib import Path
import tarfile
import subprocess
import sys
import zipfile

from test_training_packaging import framework_wheel, wheel_metadata

ROOT = Path(__file__).resolve().parents[1]


def test_model_extras_are_separate_and_keep_native_environment_pins(wheel_metadata):
    extras = set(wheel_metadata.get_all('Provides-Extra'))
    assert {'dmd2', 'self-forcing-official', 'ltx25', 'modern-flow'} <= extras
    requirements = {v.replace(' ', '') for v in wheel_metadata.get_all('Requires-Dist')}
    for extra, pins in {
        'test': ['setuptools>=77.0.0', 'wheel'],
        'test-helios': ['diffusers==0.34.0'],
        'native-wan': ['diffusers==0.35.2'],
        'helios-reference': ['diffusers==0.37.1'],
        'dmd2': ['torch==2.10.0', 'diffusers==0.37.1', 'transformers==4.51.3', 'peft==0.17.1', 'ftfy==6.3.1'],
        'self-forcing-official': ['torch==2.10.0', 'diffusers==0.37.1', 'transformers==4.51.3', 'sentencepiece==0.2.2'],
        'ltx25': ['torch==2.10.0', 'torchaudio==2.10.0', 'transformers==5.14.1',
                  'tokenizers==0.22.2', 'scipy==1.16.3', 'colour-science==0.4.7'],
        'livewan-wan21-reference': ['diffusers==0.35.2', 'einops==0.8.1'],
        'modern-flow': ['torch==2.10.0', 'diffusers==0.37.1', 'transformers==5.14.1',
                        'peft==0.18.1', 'tokenizers==0.22.2', 'accelerate==1.14.0'],
    }.items():
        for pin in pins:
            assert f'{pin};extra=="{extra}"' in requirements
    assert wheel_metadata['Requires-Python'] == '>=3.10'


def test_dedicated_provider_lanes_fetch_source_only_and_forbid_acceptance_skips():
    workflow = (ROOT/'.github/workflows/official-models-cpu.yml').read_text()
    for family in ('dmd2', 'official-self-forcing', 'ltx25', 'livewan-public'):
        assert f'family: {family}' in workflow
    assert "python-version: '3.12'" in workflow
    assert '--with-test-oracle' in workflow
    assert 'tests/test_training_sd_binding.py' in workflow
    assert workflow.count('scripts/check_cpu_receipt.py provider-cpu.xml') == 4
    assert 'test_fresh_process_exact_resume_and_current_generator_av_export[lora_config]' in workflow
    assert 'test_source_trainer_oracle_loss_gradient_adam_ema_cache_rng_and_native_clocks' in workflow
    assert 'test_cli_actual_original_optional_video_parameter_and_no_fallback' in workflow
    assert 'tests/gpu/test_training_dmd2_lifecycle.py --junitxml=optional-cuda.xml' in workflow
    assert 'tests/gpu/test_training_self_forcing_lifecycle.py --junitxml=optional-cuda.xml' in workflow
    assert 'scripts/check_cpu_receipt.py optional-cuda.xml' not in workflow
    assert '--deselect' not in workflow, 'optional GPU files are outside the mandatory root globs'
    generic = (ROOT/'.github/workflows/framework-cpu.yml').read_text()
    # W4 full-class .35 tests run in their own job, not a conflicting .34 lane.
    assert 'python scripts/run_framework_cpu_tests.py --junitxml=framework-cpu.xml' in generic
    # Exact selection/real subprocess argv are exercised by the collector tests;
    # pytest --ignore does not exclude explicitly shell-expanded file arguments.


def test_official_self_forcing_static_config_uses_registered_native_workflow():
    from distillschedule.training.config import load_training_config
    config = load_training_config(ROOT/'recipes/self_forcing_official/training.json')
    assert config.model['factory'] == 'distillschedule.training.self_forcing:build_model'
    assert config.algorithm['factory'] == 'distillschedule.training.self_forcing:build_self_forcing'
    assert config.execution['mode'] == 'reference'
    assert config.data['streams']['generator']['samples_per_cycle'] == 1
    assert config.data['streams']['critic']['samples_per_cycle'] == 5
    for mode in ('prompts', 'preencoded'):
        value = json.loads((ROOT/f'recipes/self_forcing_official/inputs.{mode}.template.json').read_text())
        assert value['mode'] == mode and value['schema_version'] == 'distillschedule.official_self_forcing_inputs.v1'


def test_modern_flow_ci_is_isolated_mandatory_and_runs_the_real_installed_path():
    import yaml
    workflow = yaml.safe_load((ROOT/'.github/workflows/modern-flow-cpu.yml').read_text())
    job = workflow['jobs']['native-media-lifecycle']
    assert job['env']['DMD_MODERN_FLOW_CPU'] == '1'
    commands = '\n'.join(step.get('run', '') for step in job['steps'])
    for value in ('torch==2.10.0', '.[test,modern-flow]', 'ffmpeg', 'ffprobe',
                  'tests/test_training_flow_image_device.py',
                  'tests/test_training_flow_video_device.py',
                  'tests/test_training_flow_images_cpu.py', 'tests/test_training_wan22_cpu.py',
                  'tests/test_training_hunyuan_video_cpu.py', 'tests/test_training_modern_flow_planning_cpu.py',
                  'tests/test_training_native_text_assets_cpu.py',
                  'tests/test_data_export_inputs_native_cpu.py', 'scripts/check_cpu_receipt.py modern-flow-cpu.xml',
                  'test_actual_lossless_video_file_and_raw_rgba_refusal',
                  'test_public_auto_saved_plan_and_new_process_resume[wan22-ti2v-5b]',
                  "--require 'test_legacy_native_effective_config_forward_gradient_raw_restore_and_safe_reload[flux]'",
                  "--require 'test_alias_shard_index_remains_exact_and_header_failures_precede_payload[wrong-shard]'",
                  'scripts/check_framework_install.py --modern-flow'):
        assert value in commands, value
    assert '--deselect' not in commands and '-k ' not in commands
    assert ('tests/gpu/test_training_flow_image_lifecycle.py tests/gpu/test_training_flow_video_lifecycle.py '
            '--junitxml=optional-modern-cuda.xml') in commands
    assert 'scripts/check_cpu_receipt.py optional-modern-cuda.xml' not in commands
    assert '--require test_native_tensor_placement_preserves_complex_rope_and_parameter_owners' in commands
    assert '--require test_ema_restoration_validates_every_tensor_before_mutation' in commands
    for family in ('wan22-ti2v5b-t2v-wan22', 'hunyuan15-t2v-hunyuan_video'):
        for name in ('test_catalog_admits_only_single_cpu_cuda_reference_or_exact_manual',
                     'test_prepared_feature_device_contract_and_owned_native_tensors'):
            assert f"--require '{name}[{family}]'" in commands


def test_w2_safe_stream_is_mandatory_in_original_tf4_lane_not_modern():
    original = (ROOT/'.github/workflows/helios-reference-model-cpu.yml').read_text()
    modern = (ROOT/'.github/workflows/modern-flow-cpu.yml').read_text()
    assert 'tests/test_training_helios_optimized_stream.py' in original
    for name in ('test_actual_producer_worker_and_original_helpers_four_cycles_resume_at_two',
                 'test_two_rank_gloo_local_native_state_resume_and_failure_boundaries'):
        assert name in original
    assert 'test_training_helios_optimized_stream.py' not in modern


def test_wheel_and_sdist_ship_exact_curated_recipes_not_model_payloads(framework_wheel, tmp_path):
    expected = {str(path.relative_to(ROOT)): path for path in (ROOT/'recipes').rglob('*')
                if path.is_file() and path.suffix in ('.py', '.json', '.md', '.yaml', '.yml')}
    assert {'recipes/wan_ar/training.incumbent-stream.json', 'recipes/wan_ar/INCUMBENT_STREAM.md'} <= expected.keys()
    with zipfile.ZipFile(framework_wheel) as wheel:
        prefix = next(name.split('.data/', 1)[0] + '.data/data/share/distillschedule/'
                      for name in wheel.namelist() if '.data/data/share/distillschedule/recipes/' in name)
        for relative, path in expected.items():
            assert wheel.read(prefix + relative) == path.read_bytes(), relative
            if path.suffix in ('.json', '.yaml', '.yml'):
                for private in (b'/home/', b'/tmp/lyy-', b'/share/project/', b'api_key', b'password'):
                    assert private not in wheel.read(prefix + relative), (relative, private)
        assert not any(name.endswith(('.pt', '.pth', '.safetensors', '.nsys-rep', '.sqlite', '.env'))
                       for name in wheel.namelist())
    result = subprocess.run([sys.executable, '-B', '-c',
        'import sys; from setuptools.build_meta import build_sdist; build_sdist(sys.argv[1])', str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    with tarfile.open(next(tmp_path.glob('distillschedule-*.tar.gz'))) as archive:
        prefix = archive.getnames()[0] + '/'
        for relative, path in expected.items():
            assert archive.extractfile(prefix + relative).read() == path.read_bytes(), relative
        assert not any('/docs/plans/' in name or name.endswith(('.pt', '.safetensors', '.nsys-rep', '.sqlite', '.env'))
                       for name in archive.getnames())
