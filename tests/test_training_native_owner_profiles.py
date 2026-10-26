"""Public templates and lazy discovery, not model/GPU acceptance."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.training.config import load_training_config


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['training.ddp-fp32.json', 'training.component-checkpoint.json'])
def test_native_owner_templates_preserve_original_workload(name):
    config = load_training_config(REPO / 'recipes/dmd2_sd15' / name)
    reference = load_training_config(REPO / 'recipes/dmd2_sd15/training.json')
    assert config.algorithm == reference.algorithm
    assert config.data['streams'] == reference.data['streams']
    params = config.model['parameters']
    settings = {k: v for k, v in params['settings'].items() if k != 'gradient_checkpointing'}
    assert settings == reference.model['parameters']['settings']
    if name == 'training.ddp-fp32.json':
        assert params['native_role_backend'] == 'ddp-fp32-v1'
        assert config.execution == reference.execution
        assert config.data['sharding'] == 'rank_strided'
        assert config.trainer['distributed']['backend'] == 'gloo'
    else:
        assert params['native_checkpoint_policy'] == 'component-v1'
        assert params['settings']['gradient_checkpointing'] is True
        assert config.execution['implementation']['owner_execution']['native_cycle'] == {
            'generator_checkpointing': True, 'fake_checkpointing': False}


def test_native_owner_discovery_is_lazy_and_honest():
    program = '''import json, sys
from distillschedule.training.strategy_catalog import describe
value = describe()
assert "torch" not in sys.modules and "diffusers" not in sys.modules
assert value["strategies"]["native-owner-ddp"]["backend"] == "torch-callable-native-owner-v1"
assert "NOT_RUN" in value["strategies"]["native-owner-ddp"]["validation"]
assert not value["capabilities"]["native-component-checkpointing"]["automatic_enable"]
assert not value["capabilities"]["native-consumed-kv-release"]["automatic_enable"]
print(json.dumps(value))
'''
    result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['strategies']['native-owner-ddp']


def test_self_forcing_consumed_cache_template_changes_only_bound_policy():
    folder = REPO / 'recipes/self_forcing_official'
    reference = json.loads((folder / 'training.json').read_text())
    value = json.loads((folder / 'training.consumed-kv.json').read_text())
    assert value['model']['parameters'].pop('native_cache_lifetime') == 'release-after-commit-v1'
    assert value == reference
    config = load_training_config(folder / 'training.consumed-kv.json')
    assert config.execution == {'adapter': 'program', 'mode': 'reference'}


def test_official_cpu_lane_includes_component_and_collective_rejection_coverage():
    workflow = (REPO / '.github/workflows/official-models-cpu.yml').read_text()
    command = next(line for line in workflow.splitlines() if 'pytest -q tests/test_training_dmd2_*.py' in line)
    for name in ('test_owner_component_checkpoint.py', 'test_collective_owner_execution.py',
                 'test_training_native_owner_profiles.py'):
        assert 'tests/' + name in command
