import json
import os
import subprocess
import sys

import pytest

from distillschedule.training.config import TrainingConfig


def test_catalog_discovery_is_lightweight_and_lists_real_model_families():
    result = subprocess.run([sys.executable, '-c', """
import json, sys
from distillschedule.training.catalog import describe
value = describe()
assert not any(name in sys.modules for name in ('torch', 'diffusers', 'transformers', 'ltx_core'))
print(json.dumps(value))
"""], capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert {'sd15', 'sdxl', 'self-forcing-wan', 'ltx25'} <= set(value['models'])
    assert {'dmd2', 'self-forcing', 'dmd'} <= set(value['algorithms'])


def test_registered_model_and_algorithm_resolve_without_importing_models(tmp_path):
    config = TrainingConfig.from_dict({
        'schema_version': 'distillschedule.training.v1',
        'model': {'name': 'sd15', 'parameters': {}},
        'data': {'streams': {'generator': {'prepared': 'g/manifest.json', 'samples_per_cycle': 1},
                             'critic': {'prepared': 'c/manifest.json', 'samples_per_cycle': 10}}},
        'algorithm': {'name': 'dmd2', 'parameters': {}},
        'execution': {'mode': 'reference'},
        'trainer': {'cycles': 2, 'output_dir': 'run'}}, source=tmp_path / 'train.json')
    assert config.model['factory'].endswith(':build_sd15')
    assert config.algorithm['factory'].endswith(':build_dmd2')
    assert config.data['streams']['critic']['prepared'] == str(tmp_path / 'c/manifest.json')
    assert TrainingConfig.from_dict(config.to_dict(), source=config.source).identity == config.identity


def test_registry_can_add_external_model_without_solver_changes(tmp_path):
    from distillschedule.training.catalog import register_model, resolve
    register_model('external-unit-model', factory='external_package.adapter:build',
                   capabilities=('image',), description='external model')
    assert resolve('model', 'external-unit-model')['factory'] == 'external_package.adapter:build'
    with pytest.raises(ValueError, match='already registered'):
        register_model('external-unit-model', factory='other:build')


@pytest.mark.parametrize('kind', ['epsilon', 'velocity', 'flow', 'x0'])
def test_prediction_conversions_recover_same_clean_latent(kind):
    import torch
    from distillschedule.training.prediction import PredictionSpec
    clean, noise = torch.tensor([1., -2.]), torch.tensor([.5, 3.])
    sigma = torch.tensor(.6)
    alpha = 1 - sigma if kind == 'flow' else torch.tensor(.8)
    x = alpha * clean + sigma * noise
    pred = {'epsilon': noise, 'velocity': alpha * noise - sigma * clean,
            'flow': noise - clean, 'x0': clean}[kind]
    recovered = PredictionSpec(kind).to_x0(pred, x, sigma=sigma, alpha=alpha)
    torch.testing.assert_close(recovered, clean)


def test_prediction_conversion_rejects_missing_diffusion_coefficient():
    from distillschedule.training.prediction import PredictionSpec
    with pytest.raises(ValueError, match='alpha'):
        PredictionSpec('epsilon').to_x0(1., 2., sigma=.5)


def test_effective_prediction_and_capabilities_are_bound_to_checkpoint_identity(tmp_path, monkeypatch):
    from distillschedule.examples.tiny_av import build_fixture
    from distillschedule.data import prepare
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import train
    root = tmp_path / 'media'; build_fixture(root, samples=4)
    prepare(root / 'raw/manifest.json', root / 'prepared')
    plugin = tmp_path / 'model.py'
    plugin.write_text('''import os
from distillschedule.examples.tiny_av import build_model as tiny
from distillschedule.training.prediction import PredictionSpec
def build(parameters):
    result = tiny(parameters)
    result.prediction = PredictionSpec(os.environ.get('TEST_PRED_KIND', 'flow'))
    result.capabilities = (os.environ.get('TEST_MODEL_CAP', 'video'),)
    return result
''')
    value = json.loads((root / 'train.json').read_text())
    value['model']['factory'] = str(plugin) + ':build'
    value['trainer'].update(cycles=1, output_dir=str(tmp_path / 'first'))
    config = TrainingConfig.from_dict(value, source=root / 'training.json')
    first = train(config)
    value['trainer'].update(cycles=2, resume_from=first['checkpoint'], output_dir=str(tmp_path / 'wrong'))
    monkeypatch.setenv('TEST_PRED_KIND', 'x0')
    with pytest.raises(ValueError, match='identity'):
        train(TrainingConfig.from_dict(value, source=config.source))
    monkeypatch.setenv('TEST_PRED_KIND', 'flow'); monkeypatch.setenv('TEST_MODEL_CAP', 'audio')
    with pytest.raises(ValueError, match='identity'):
        train(TrainingConfig.from_dict(value, source=config.source))


def test_official_self_forcing_registered_workflow_resolves_to_actual_factory():
    from distillschedule.training.catalog import resolve
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.self_forcing import build_self_forcing
    factory, _ = load_callable(resolve('algorithm', 'self-forcing')['factory'])
    assert factory is build_self_forcing
