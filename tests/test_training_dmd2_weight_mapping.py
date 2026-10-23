"""Native DMD2 initializer mapping is not a resumable training checkpoint."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from distillschedule.training.config import load_training_config
from distillschedule.training.plugins import load_callable
from distillschedule.training.sd import build_sd15, build_sdxl
from test_training_dmd2_cpu import REPO, source_root
from test_training_end_to_end import equal, run_cli


@pytest.mark.parametrize('variant,lora', [('sd15', False), ('sdxl', False), ('sdxl', True)])
def test_current_export_maps_exactly_to_native_generator_initializer(source_root, tmp_path, variant, lora):
    builder, _ = load_callable(str(REPO / 'recipes/dmd2_sd15/create_cpu_fixture.py') + ':build_fixture')
    files = builder(tmp_path / 'actual', source_root, variant=variant, cycles=1, lora=lora)
    config = Path(files['config'])
    trained = run_cli('train', config, '--output', tmp_path / 'trained')
    exported = run_cli('export', config, '--checkpoint', trained['checkpoint'], '--output', tmp_path / 'export')
    weights = torch.load(tmp_path / 'export/weights.pt', map_location='cpu', weights_only=True)
    generator = weights['generator']
    assert set(weights) == {'generator', 'text_encoder', 'vae'}
    assert set(exported['model']['modules']) == set(weights)
    # This is the exact upstream feedforward_model.state_dict() key space:
    # no framework prefix, adapter merging, dtype cast or tensor rewriting.
    checkpoint = tmp_path / 'native_generator.pt'
    torch.save(generator, checkpoint)
    parameters = deepcopy(load_training_config(config).model['parameters'])
    parameters['generator_checkpoint'] = str(checkpoint)
    rebuilt = (build_sdxl if variant == 'sdxl' else build_sd15)(parameters).state['native_cycle']
    assert equal(rebuilt.model.feedforward_model.state_dict(), generator)
    assert rebuilt.native_step == rebuilt.generator_updates == rebuilt.guidance_updates == 0
    assert all(not optimizer.state for optimizer in rebuilt.optimizers.values())
    assert rebuilt.identity['generator_checkpoint']['path'] == str(checkpoint)
    if lora:
        assert any('lora_A' in name for name in generator)
        assert any('lora_B' in name for name in generator)
        assert rebuilt.args.generator_lora
    # A bare generator initializer deliberately cannot stand in for the
    # complete-cycle owner/optimizer/RNG/data-state needed by train --resume.
    payload = torch.load(Path(trained['checkpoint']) / 'state.pt', weights_only=True)
    assert set(payload) != set(generator)
