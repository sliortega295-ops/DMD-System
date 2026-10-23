"""Optional actual original tiny DMD2 CUDA FP32, never pretrained/quality proof.

Run explicitly in the existing dmd2 dependency lane with DMD2_SOURCE_ROOT.
CPU-only Torch must skip honestly; no substitute modules or CUDA simulations.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch.utils._pytree import tree_map

from distillschedule.open_workload.torch_backend import Session, candidates
from distillschedule.training.checkpoint import cpu_tree, rng_state, restore_rng
from distillschedule.training.config import load_training_config
from distillschedule.training.dmd2 import build_dmd2
from distillschedule.training.session import construct, build_cycle
from test_training_dmd2_cpu import source_root, _oracle
from test_training_dmd2_owner_execution import native_assets
from test_training_end_to_end import equal, run_cli

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
    reason='actual CUDA DMD2 lifecycle NOT_RUN: CUDA-capable Torch/device required')


def _cuda_config(files):
    path = Path(files['config']); value = json.loads(path.read_text())
    value['trainer']['device'] = 'cuda:' + str(torch.cuda.current_device())
    path.write_text(json.dumps(value))
    return load_training_config(path)


def test_actual_cuda_native_oracle_restore_rng_and_preallocation_refusals(native_assets, source_root, tmp_path):
    config = _cuda_config(native_assets)
    components, data, algorithm, _ = construct(config)
    owner = components.state['native_cycle']; device = owner.device
    assert device.type == 'cuda'
    assert all(t.device == device for t in owner.runtime_tensors())
    assert all(not t.is_floating_point() or t.dtype == torch.float32 for t in owner.runtime_tensors())
    teacher = cpu_tree(owner.model.guidance_model.real_unet.state_dict())
    initial = cpu_tree(owner.state_dict())
    oracle = construct(config)[0].state['native_cycle']; oracle.load_state_dict(initial)
    workload = build_cycle(config, components, data, algorithm, cycle=0, cursor=0)
    assert all(t.device == device for t in owner._plain_tensors().values())
    assert owner.model.alphas_cumprod is owner.model.guidance_model.alphas_cumprod
    assert owner.model.noise_scheduler.alphas_cumprod is owner.model.guidance_model.alphas_cumprod
    assert all(t.device == device for t in owner.model.guidance_model.buffers())
    outer = rng_state(device=device)
    expected = _oracle(oracle, workload.feeds, source_root)
    restore_rng(outer)
    actual = owner.run(workload.feeds, cycle=0)
    assert [row['losses'] for row in actual['iterations']] == expected['losses']
    assert equal(cpu_tree(owner.model.state_dict()), cpu_tree(oracle.model.state_dict()))
    assert equal(owner._rng, expected['rng']) and equal(outer, rng_state(device=device))
    assert equal(teacher, cpu_tree(owner.model.guidance_model.real_unet.state_dict()))
    assert all(p.grad is None for p in owner.model.parameters())
    saved = cpu_tree(owner.state_dict())
    restored = construct(config)[0].state['native_cycle']; restored.load_state_dict(saved)
    assert equal(saved, cpu_tree(restored.state_dict()))
    named = {id(p): name for name, p in restored.model.named_parameters()}
    for role, optimizer in restored.optimizers.items():
        assert [[named[id(p)] for p in group['params']] for group in optimizer.param_groups] == restored.ownership[role]
        for parameter, slots in optimizer.state.items():
            assert parameter.device == slots['exp_avg'].device == slots['exp_avg_sq'].device == device
            assert slots['step'].device.type == 'cpu'
    if restored.uncond_embedding is not None:
        assert restored.uncond_embedding.device == device
    bad = deepcopy(saved)
    next(iter(bad['optimizers']['guidance']['state'].values()))['exp_avg'] = next(
        iter(bad['optimizers']['guidance']['state'].values()))['exp_avg'].double()
    with pytest.raises(ValueError):
        restored.load_state_dict(bad)
    assert equal(saved, cpu_tree(restored.state_dict()))
    # No paired trial may allocate another live CUDA owner or mutate state.
    next_feeds = tree_map(lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, data.cycle_sample(1))
    next_workload = build_dmd2(components, next_feeds, {}, 1, 29)
    before = cpu_tree(owner.state_dict()); outer = rng_state(device=device)
    for mode, implementation in (('auto', {}), ('plan', {}), ('manual', {'owner_execution': {'native_cycle': {}}})):
        with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
            owner.admit_execution(mode, implementation)
    baseline = candidates(next_workload)['candidates'][0]
    assert owner.execution_options()['identity']['capability']['status'] == 'UNAVAILABLE_CUDA_TRIAL_SNAPSHOT'
    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
        Session(next_workload, baseline)
    from distillschedule.open_workload.planning import calibrate
    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
        calibrate(next_workload, tmp_path / 'rejected-trial', max_candidates=1,
                  samples=1, max_program_runs=2, profile_hardware=False)
    assert equal(before, cpu_tree(owner.state_dict())) and equal(outer, rng_state(device=device))
    # CPU rows must not be silently copied by an upstream CLIP .to call.
    with pytest.raises(ValueError, match='device|placement'):
        owner.run(data.cycle_sample(1), cycle=1)
    owner.run(next_feeds, cycle=1); restored.run(next_feeds, cycle=1)
    assert equal(cpu_tree(owner.state_dict()), cpu_tree(restored.state_dict()))


def test_actual_cuda_fresh_process_restart_and_portable_current_generator_export(native_assets, tmp_path):
    config = _cuda_config(native_assets); path = config.source
    partial = run_cli('train', path, '--cycles', 1, '--output', tmp_path / 'partial')
    resumed = run_cli('train', path, '--cycles', 3, '--resume', partial['checkpoint'], '--output', tmp_path / 'resumed')
    full = run_cli('train', path, '--cycles', 3, '--output', tmp_path / 'full')
    assert resumed['gpu_execution'] == full['gpu_execution'] == 'NATIVE_CUDA_RUN'
    assert equal(torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True),
                 torch.load(Path(full['checkpoint']) / 'state.pt', weights_only=True))
    exported = run_cli('export', path, '--checkpoint', resumed['checkpoint'], '--output', tmp_path / 'export')
    assert set(exported['model']['modules']) == {'generator', 'text_encoder', 'vae'}
    assert 'device' not in exported['model']['parameters'] and 'model_id' not in exported['model']['parameters']
    from distillschedule.data.sd_inputs import SCHEMA, prepare_inputs
    source = tmp_path / 'new-prompt.json'
    source.write_text(json.dumps(dict(schema_version=SCHEMA, variant=exported['model']['parameters']['variant'],
        model_id=native_assets['assets'], records=[dict(id='new-boat', prompt='new boat in a lavender sea')])))
    prepared = prepare_inputs(source, tmp_path / 'new-prepared')
    from distillschedule.training.session import _load_for_output, _generate_values
    from distillschedule.data import PreparedDataset
    trained = _load_for_output(config, resumed['checkpoint'])[0]
    expected = cpu_tree(_generate_values(config, trained, PreparedDataset(prepared).sample(0), {}, output_kind='image'))
    # The inference artifact must not need the initial UNet/Teacher/Fake bundle.
    assets = Path(native_assets['assets']); assets.rename(assets.with_name('REMOVED_BASE_BUNDLE'))
    inference = dict(schema_version='distillschedule.training.v1',
        model=dict(factory=exported['model']['factory'], parameters=exported['model']['parameters']),
        algorithm=dict(name='dmd2'), data=dict(prepared=str(prepared)),
        execution=dict(adapter='program', mode='reference'),
        trainer=dict(cycles=1, seed=17, device=config.trainer['device'], output_dir=str(tmp_path / 'unused')))
    inference_path = tmp_path / 'inference.json'; inference_path.write_text(json.dumps(inference))
    result = run_cli('generate', inference_path, '--weights', tmp_path / 'export', '--prepared', prepared,
                     '--output', tmp_path / 'gpu-image', '--output-kind', 'image')
    assert equal(torch.load(result['artifact'], weights_only=True), expected)
    # Placement is relocatable for inference, not for resumable training state.
    inference['trainer']['device'] = 'cpu'; inference_path.write_text(json.dumps(inference))
    cpu = run_cli('generate', inference_path, '--weights', tmp_path / 'export', '--prepared', prepared,
                  '--output', tmp_path / 'cpu-image', '--output-kind', 'image')
    decoded = torch.load(cpu['artifact'], weights_only=True)['decoded']['image']
    assert decoded.shape == (16, 16, 3) and decoded.dtype == torch.uint8
