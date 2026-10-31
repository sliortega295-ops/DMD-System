"""Optional actual tiny official Self-Forcing CUDA FP32 MATH lifecycle.

CPU-only Torch skips honestly. These tests are not pretrained, capacity,
BF16/FlashAttention, distributed, quality or full-U6 acceptance evidence.
Whole CUDA owners are never copied: native/Program comparisons and restart use
sequential resident owners with explicit CPU snapshots.
Exact comparisons use deterministic CUDA kernels in parent and CLI processes;
production defaults and numerical thresholds are unchanged.
"""
from copy import deepcopy
import gc
import json
from pathlib import Path

import pytest
import torch

from distillschedule.training.checkpoint import cpu_tree, rng_state, restore_rng
from distillschedule.training.config import load_training_config
from distillschedule.training.policy import CyclePolicy
from distillschedule.training.self_forcing_owner import plain_tensors
from distillschedule.training.session import construct, build_cycle
from distillschedule.training.single_device import device_scope
from test_training_end_to_end import equal
from cuda_test_policy import cuda_bitwise_policy, run_cli
from test_training_self_forcing_cpu import fixture_assets, source_root, _native_trainer_oracle

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(),
    reason='actual CUDA official Self-Forcing NOT_RUN: CUDA-capable Torch/device required'),
    pytest.mark.usefixtures('cuda_bitwise_policy')]


def _cuda_config(files):
    path = Path(files['config']); value = json.loads(path.read_text())
    value['trainer']['device'] = 'cuda:' + str(torch.cuda.device_count() - 1)
    path.write_text(json.dumps(value))
    return load_training_config(path)


def test_actual_cuda_native_oracle_program_and_host_checkpoint_sequentially(fixture_assets, tmp_path):
    config = _cuda_config(fixture_assets)
    previous = torch.cuda.current_device()
    components, data, algorithm, _ = construct(config)
    reference = components.state['native_cycle']; device = reference.device
    assert torch.cuda.current_device() == previous
    initial = cpu_tree(reference.state_dict())
    workload = build_cycle(config, components, data, algorithm, cycle=0, cursor=0)
    feeds = workload.feeds
    outer = rng_state(device=device)
    with device_scope(device):
        trainer, expected_events, expected_rng = _native_trainer_oracle(reference, [feeds])
        restore_rng(outer)
    expected_model = cpu_tree(reference.model.state_dict())
    expected_optimizers = {name: cpu_tree(opt.state_dict()) for name, opt in reference.optimizers.items()}
    expected_ema = cpu_tree(trainer.generator_ema.state_dict())
    expected_condition = cpu_tree(trainer.unconditional_dict)
    # Release the source oracle before constructing the resident Program owner.
    del trainer, reference, components, workload
    gc.collect()

    components, data, algorithm, _ = construct(config)
    owner = components.state['native_cycle']; owner.load_state_dict(initial)
    workload = build_cycle(config, components, data, algorithm, cycle=0, cursor=0)
    assert workload.device == device and workload.identity_extra['placement']['device'] == str(device)
    assert all(t.device == device for t in owner.runtime_tensors())
    assert owner.model.scheduler is owner.model.generator.scheduler
    assert all(t.device == device for t in plain_tensors(owner.model).values())
    assert owner.model.generator.model.freqs.dtype == torch.complex128
    events = []
    owner.observer = lambda name, value: events.append((name, cpu_tree(value)))
    outer = rng_state(device=device)
    unowned = next((index for index in range(torch.cuda.device_count()) if index != device.index), None)
    other_rng = None if unowned is None else torch.cuda.get_rng_state(unowned).clone()
    session, outputs = CyclePolicy(config.execution).run(workload, tmp_path / 'reference')
    assert session.state['native_cycle'] is owner
    actual = outputs['self_forcing.native_cycle']
    assert equal(outer, rng_state(device=device)) and equal(owner._rng, expected_rng)
    assert torch.cuda.current_device() == previous
    if unowned is not None:
        assert torch.equal(other_rng, torch.cuda.get_rng_state(unowned))
    assert equal(cpu_tree(owner.model.state_dict()), expected_model)
    assert equal({name: cpu_tree(opt.state_dict()) for name, opt in owner.optimizers.items()}, expected_optimizers)
    assert equal(cpu_tree(owner.generator_ema.state_dict()), expected_ema)
    assert equal(cpu_tree(owner.unconditional_dict), expected_condition)
    assert all(t.device.type == 'cpu' and t.dtype == torch.float32 for t in owner.generator_ema.shadow.values())
    assert equal([(name, value) for name, value in events if name.endswith(('_gradients', '_rollout_cache'))],
                 [(name, value) for name, value in expected_events if name != 'loss'])
    losses = []
    for row in actual['iterations']:
        if row['generator_turn']:
            losses.append(dict(generator_loss=row['losses']['generator_loss'], generator_grad_norm=row['generator_grad_norm']))
        losses.append(dict(critic_loss=row['losses']['critic_loss'], critic_grad_norm=row['critic_grad_norm']))
    assert losses == [value for name, value in expected_events if name == 'loss']
    assert actual['consumed'] == {'generator': 1, 'critic': 5}
    saved = cpu_tree(owner.state_dict()); owner.observer = None
    for mode, implementation in (('auto', {}), ('plan', {}),
                                 ('manual', {'owner_execution': {'native_cycle': {}}})):
        with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
            owner.admit_execution(mode, implementation)
    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
        deepcopy(owner)
    from distillschedule.open_workload.torch_backend import Session, candidates
    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
        Session(workload, candidates(workload)['candidates'][0])
    with pytest.raises(ValueError, match='device'):
        owner.run(data.cycle_sample(1), cycle=1)
    assert equal(saved, cpu_tree(owner.state_dict()))
    next_feeds = build_cycle(config, components, data, algorithm, cycle=1, cursor=1).feeds
    owner.run(next_feeds, cycle=1)
    expected_next = cpu_tree(owner.state_dict())
    del session, workload, components, owner
    gc.collect()

    components, _, _, _ = construct(config)
    restored = components.state['native_cycle']; restored.load_state_dict(saved)
    assert equal(saved, cpu_tree(restored.state_dict()))
    assert restored.model.inference_pipeline is None
    for optimizer in restored.optimizers.values():
        for parameter, slots in optimizer.state.items():
            assert parameter.device == slots['exp_avg'].device == slots['exp_avg_sq'].device == device
            assert slots['step'].device.type == 'cpu'
    bad = deepcopy(saved); bad['identity']['placement']['device'] = 'cpu'
    with pytest.raises(ValueError, match='identity'):
        restored.load_state_dict(bad)
    assert equal(saved, cpu_tree(restored.state_dict()))
    restored.run(next_feeds, cycle=1)
    assert equal(expected_next, cpu_tree(restored.state_dict()))


def test_actual_cuda_fresh_process_restart_current_g_export_and_cpu_relocation(fixture_assets, tmp_path):
    config = _cuda_config(fixture_assets); path = config.source
    partial = run_cli('train', path, '--cycles', 1, '--output', tmp_path / 'partial')
    resumed = run_cli('train', path, '--cycles', 3, '--resume', partial['checkpoint'], '--output', tmp_path / 'resumed')
    full = run_cli('train', path, '--cycles', 3, '--output', tmp_path / 'full')
    assert resumed['gpu_execution'] == full['gpu_execution'] == 'NATIVE_CUDA_RUN'
    assert equal(torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True),
                 torch.load(Path(full['checkpoint']) / 'state.pt', weights_only=True))
    exported = run_cli('export', path, '--checkpoint', resumed['checkpoint'], '--output', tmp_path / 'export')
    assert set(exported['model']['modules']) == {'generator', 'vae'}
    assert 'device' not in exported['model']['parameters'] and 'assets' not in exported['model']['parameters']
    from distillschedule.data import PreparedDataset
    from distillschedule.data.self_forcing_inputs import prepare_inputs, SCHEMA
    from distillschedule.training.session import _load_for_output, _generate_values
    from distillschedule.workloads.best_known.registry import file_sha256
    tensor = tmp_path / 'new-condition.pt'
    torch.save(torch.linspace(-.8, .9, 128).reshape(1, 8, 16), tensor)
    source = tmp_path / 'new-condition.json'
    source.write_text(json.dumps(dict(schema_version=SCHEMA, mode='preencoded', records=[dict(
        id='new-native-condition', embedding=dict(path=tensor.name, sha256=file_sha256(tensor)))])))
    prepared = prepare_inputs(source, tmp_path / 'new-prepared')
    trained = _load_for_output(config, resumed['checkpoint'])[0]
    expected = cpu_tree(_generate_values(config, trained, PreparedDataset(prepared).sample(0), {}, output_kind='video'))
    del trained
    gc.collect()
    assets = Path(fixture_assets['assets']); assets.rename(assets.with_name('REMOVED_TRAINING_ASSETS'))
    inference = dict(schema_version='distillschedule.training.v1',
        model=dict(factory=exported['model']['factory'], parameters=exported['model']['parameters']),
        algorithm=dict(name='self-forcing'), data=dict(prepared=str(prepared)),
        execution=dict(adapter='program', mode='reference'),
        trainer=dict(cycles=1, seed=17, device=config.trainer['device'], output_dir=str(tmp_path / 'unused')))
    request = tmp_path / 'inference.json'; request.write_text(json.dumps(inference))
    generated = run_cli('generate', request, '--weights', tmp_path / 'export', '--prepared', prepared,
                        '--output', tmp_path / 'gpu-video', '--output-kind', 'video')
    assert equal(torch.load(generated['artifact'], weights_only=True), expected)
    # Only inference may relocate device; a training checkpoint remains exact.
    inference['trainer']['device'] = 'cpu'; request.write_text(json.dumps(inference))
    generated_cpu = run_cli('generate', request, '--weights', tmp_path / 'export', '--prepared', prepared,
                            '--output', tmp_path / 'cpu-video', '--output-kind', 'video')
    video = torch.load(generated_cpu['artifact'], weights_only=True)['decoded']['video']
    assert video.shape == (81, 32, 32, 3) and video.dtype == torch.uint8


def test_actual_cuda_math_attention_varlen_local_bottom_right_and_cache_backward():
    from distillschedule.training.self_forcing_source import fp32_attention
    device = torch.device('cuda', torch.cuda.device_count() - 1)
    q = torch.linspace(-.4, .8, 24, device=device).reshape(1, 3, 2, 4).requires_grad_(True)
    k = torch.linspace(.1, .9, 40, device=device).reshape(1, 5, 2, 4)
    v = k.flip(1).clone()
    lengths = dict(q_lens=torch.tensor([2], device=device), k_lens=torch.tensor([4], device=device))
    actual = fp32_attention(q, k, v, causal=True, window_size=(2, 0), **lengths)
    expected = fp32_attention(q.detach().cpu(), k.cpu(), v.cpu(), causal=True, window_size=(2, 0),
                             q_lens=torch.tensor([2]), k_lens=torch.tensor([4]))
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
    expected_grad, = torch.autograd.grad(actual.sum(), (q,), retain_graph=True)
    k.add_(2); v.mul_(3)
    actual_grad, = torch.autograd.grad(actual.sum(), (q,))
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
