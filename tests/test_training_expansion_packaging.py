"""Public optional stacks and mandatory real-upstream CPU qualification lanes."""
import json
from pathlib import Path

from test_training_packaging import framework_wheel, wheel_metadata

ROOT = Path(__file__).resolve().parents[1]


def test_model_extras_are_separate_and_keep_native_environment_pins(wheel_metadata):
    extras = set(wheel_metadata.get_all('Provides-Extra'))
    assert {'dmd2', 'self-forcing-official', 'ltx25'} <= extras
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
