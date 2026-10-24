"""Actual tiny pinned LTX AV CUDA FP32 reference, not pretrained/22B evidence.

The raw-codec fixture, flow-DMD mathematics and Program are shared with CPU.
CUDA uses native AdamW(capturable=True) for resident step tensors; exact oracle
and restart checks compare that SAME CUDA implementation, not CPU arithmetic.
No whole CUDA model is cloned and no tolerance or production math is changed.
"""
from copy import deepcopy
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from distillschedule.training.checkpoint import cpu_tree, pack_state, rng_state
from distillschedule.training.config import load_training_config
from distillschedule.training.policy import CyclePolicy
from distillschedule.training.session import construct, build_cycle
from distillschedule.training.single_device import device_scope
from test_training_end_to_end import equal
from test_training_ltx_cpu import source_root
from cuda_test_policy import cuda_bitwise_policy, run_cli

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(),
    reason='actual native LTX CUDA NOT_RUN: CUDA-capable Torch/device required'),
    pytest.mark.usefixtures('cuda_bitwise_policy')]


@pytest.fixture(scope='module')
def native_assets(tmp_path_factory, source_root):
    # The shared fixture deliberately asserts no CUDA initialization. Keep its
    # CPU-only preparation in a fresh process, even in a combined GPU test run.
    recipe = Path(__file__).resolve().parents[2] / 'recipes/ltx25/create_cpu_fixture.py'
    result = subprocess.run([sys.executable, str(recipe),
        str(tmp_path_factory.mktemp('cuda-ltx-assets') / 'fixture'), '--source-root', str(source_root)],
        capture_output=True, text=True, env={**os.environ, 'OMP_NUM_THREADS': '1'})
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def _cuda_config(files, tmp_path, key='lora_config'):
    value = json.loads(Path(files[key]).read_text())
    value['trainer']['device'] = 'cuda:' + str(torch.cuda.device_count() - 1)
    path = tmp_path / 'training-cuda.json'; path.write_text(json.dumps(value))
    return load_training_config(path)


from native_flow_oracle import _native_flow_oracle


@pytest.mark.parametrize('config_key', ['config', 'lora_config'])
def test_actual_cuda_native_flow_oracle_and_complete_resident_state(native_assets, tmp_path, config_key):
    from distillschedule.open_workload.torch_backend import tensors, Session, candidates
    from distillschedule.open_workload.planning import calibrate
    config = _cuda_config(native_assets, tmp_path, config_key)
    previous = torch.cuda.current_device()
    components, data, algorithm, _ = construct(config)
    device = components.state['student'].device
    assert torch.cuda.current_device() == previous
    workload = build_cycle(config, components, data, algorithm, cycle=0, cursor=0)
    assert workload.device == device and all(t.device == device for t in tensors((workload.state, workload.feeds)))
    initial = pack_state(components.state)
    teacher = cpu_tree(components.state['teacher'].state_dict())
    outer = rng_state(device=device)
    expected_losses, expected_gradients = _native_flow_oracle(
        components.state, workload.feeds, config.algorithm['parameters'], workload.seed)
    expected = pack_state(components.state)
    assert equal(outer, rng_state(device=device))
    del components, workload, algorithm, data
    gc.collect()

    # Only one resident set of native Student/Fake/Teacher/codecs at a time.
    components, data, algorithm, identity = construct(config)
    assert equal(initial, pack_state(components.state))
    workload = build_cycle(config, components, data, algorithm, cycle=0, cursor=0)
    gradients = []; handles = []
    def observe(role):
        def before_step(optimizer, args, kwargs):
            gradients.append((role, cpu_tree({name: p.grad for name, p in components.state[role].named_parameters()
                                             if p.requires_grad})))
        return before_step
    for role in ('student', 'fake'):
        handles.append(components.state[role + '_optimizer'].register_step_pre_hook(observe(role)))
    outer = rng_state(device=device)
    policy = CyclePolicy(config.execution)
    session, actual = policy.run(workload, tmp_path / 'reference')
    for handle in handles:
        handle.remove()
    assert equal(expected_losses, cpu_tree(actual)) and equal(expected_gradients, gradients)
    assert equal(expected, pack_state(components.state)) and equal(outer, rng_state(device=device))
    assert equal(teacher, cpu_tree(components.state['teacher'].state_dict()))
    assert components.state['dmd_counters'].tolist() == [1, 1, 5]
    assert all(p.grad is None and not p.requires_grad for p in components.state['teacher'].parameters())
    assert all(t.device == device for t in tensors(components.state))
    for role in ('student', 'fake'):
        optimizer = components.state[role + '_optimizer']
        assert all(group['capturable'] for group in optimizer.param_groups)
        assert {id(p) for group in optimizer.param_groups for p in group['params']} == {
            id(p) for p in components.state[role].parameters() if p.requires_grad}
        assert all(value['step'].device == value['exp_avg'].device == value['exp_avg_sq'].device == device
                   for value in optimizer.state.values())
    for module in (components.state[name] for name in ('student', 'fake', 'teacher', 'media')):
        for child in module.modules():
            assert not any(isinstance(value, torch.Tensor) for value in vars(child).values()), 'unowned plain tensor'
    # Actual codec execution is CUDA, not CPU decoding of CUDA-trained weights.
    media = components.state['media']
    with torch.no_grad():
        encoded = media.encode_text(['a dog rests'])
        assert encoded['video_context'].device == encoded['audio_context'].device == device
        # Offline preparation normally owns CPU codecs. Exercise the separate
        # public raw-media boundary with CUDA-owned native codecs as well: the
        # raw CPU payload must move, not an implicit CPU fallback of the model.
        raw_video = torch.arange(3 * 8 * 8 * 3).reshape(3, 8, 8, 3).to(torch.uint8)
        raw_audio = {'samples': torch.zeros(6400, 2), 'sample_rate': 16000}
        video_latents = media.encode_video(raw_video)
        audio_latents = media.encode_audio(raw_audio)
        assert raw_video.device.type == raw_audio['samples'].device.type == 'cpu'
        assert video_latents.device == audio_latents.device == device
        assert video_latents.shape == (1, 128, 2, 2, 2)
        assert audio_latents.shape == (1, 8, 11, 16)
        assert torch.isfinite(video_latents).all() and torch.isfinite(audio_latents).all()
        generated = components.generate(components.state, workload.feeds, {})
        decoded = components.decode(generated, {})
    assert decoded['video'].shape == (3, 8, 8, 3)
    assert decoded['audio']['samples'].shape == (9840, 2)
    assert decoded['video'].device == decoded['audio']['samples'].device == device

    before = pack_state(components.state); outer = rng_state(device=device)
    from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint, unpack_state
    versions = session.receipt['dispatcher']['final_versions']
    with device_scope(device):
        checkpoint = save_checkpoint(tmp_path / 'complete', components.state, identity=identity,
            cycle=1, cursor=1, versions=versions, policy=policy.bound)
        with pytest.raises(ValueError, match='identity'):
            restore_checkpoint(checkpoint, components.state, identity={**identity, 'stale': True})
    for fault in ('ema', 'optimizer', 'clock'):
        bad = deepcopy(before)
        if fault == 'ema':
            next(iter(bad['student_ema']['value'].values())).reshape(-1)[0] = float('nan')
        elif fault == 'optimizer':
            moment = next(iter(bad['fake_optimizer']['value']['state'].values()))
            moment['exp_avg'] = moment['exp_avg'].double()
        else:
            bad['dmd_counters']['value']['counters'][2] = 4
        with pytest.raises(ValueError):
            unpack_state(bad, components.state, progress={'cycle': 1, 'cursor': 1, 'versions': versions})
        assert equal(before, pack_state(components.state)), 'bad state must not partially restore live roles'
    for mode in ('auto', 'plan', 'manual'):
        request = replace(config, execution={'mode': mode})
        with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
            components.admit_training(state=components.state, request=request,
                prepared_identity=data.identity, prepared_manifest=data.manifest)
    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
        deepcopy(components.state)
    next_workload = build_cycle(config, components, data, algorithm, cycle=1, cursor=1,
        versions=session.receipt['dispatcher']['final_versions'])
    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
        Session(next_workload, candidates(next_workload)['candidates'][0])
    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
        calibrate(next_workload, tmp_path / 'forbidden-trials', max_candidates=1,
                  samples=1, max_program_runs=2, profile_hardware=False)
    with pytest.raises(ValueError, match='device'):
        components.state['student'].validate_feeds(data.sample(0))
    for fault in ('missing', 'identity', 'mask', 'wrong_device'):
        feeds = dict(next_workload.feeds)
        if fault == 'missing':
            del feeds['null_contexts']
        elif fault == 'identity':
            feeds['encoder_identity'] = feeds['encoder_identity'].clone(); feeds['encoder_identity'][0] ^= 1
        elif fault == 'mask':
            feeds['null_context_mask'] = torch.zeros_like(feeds['null_context_mask'])
        else:
            feeds['fps'] = feeds['fps'].cpu()
        with pytest.raises(ValueError):
            components.state['student'].validate_feeds(feeds)
    with torch.autocast('cuda', dtype=torch.bfloat16), pytest.raises(ValueError, match='autocast'):
        components.state['student'].rollout(next_workload.feeds, [1., 0.], 'all_steps')
    assert equal(before, pack_state(components.state)) and equal(outer, rng_state(device=device))
    policy.run(next_workload, tmp_path / 'reference')
    assert components.state['dmd_counters'].tolist() == [2, 2, 10]


@pytest.mark.parametrize('config_key', ['config', 'lora_config'])
def test_actual_cuda_new_process_resume_export_and_new_input_av(native_assets, tmp_path, config_key):
    from distillschedule.data import PreparedDataset
    from distillschedule.data.ltx_inputs import prepare_inputs, SCHEMA
    from distillschedule.training.session import _load_for_output, _generate_values
    config = _cuda_config(native_assets, tmp_path, config_key); path = config.source
    partial = run_cli('train', path, '--cycles', 1, '--output', tmp_path / 'partial')
    resumed = run_cli('train', path, '--cycles', 3, '--resume', partial['checkpoint'], '--output', tmp_path / 'resumed')
    full = run_cli('train', path, '--cycles', 3, '--output', tmp_path / 'full')
    assert resumed['gpu_execution'] == full['gpu_execution'] == 'NATIVE_CUDA_RUN'
    a = torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(full['checkpoint']) / 'state.pt', weights_only=True)
    assert equal(a, b), 'all roles, named Adam slots, clocks, EMA, RNG, data cursor and versions'
    assert a['cycle'] == a['cursor'] == 3 and a['state']['dmd_counters']['value']['counters'].tolist() == [3, 3, 15]
    exported = run_cli('export', path, '--checkpoint', resumed['checkpoint'], '--output', tmp_path / 'export')
    assert set(exported['model']['modules']) == {'student', 'media'}
    assert 'device' not in exported['model']['parameters'] and 'assets' not in exported['model']['parameters']
    source = tmp_path / 'new-prompts.json'
    source.write_text(json.dumps(dict(schema_version=SCHEMA, mode='prompts',
        source_root=os.environ['LTX_SOURCE_ROOT'],
        assets=str(Path(native_assets['assets']) / 'assets.json'), records=[dict(id='new-native-prompt',
            prompt='a dog rests', geometry=dict(fps=8., video_shape=[1, 128, 2, 2, 2], audio_shape=[1, 8, 11, 16],
                conditioning=dict(video_prefix_latent_frames=0, audio_prefix_latent_frames=0)))])))
    prepared = prepare_inputs(source, tmp_path / 'new-prepared')
    with device_scope(torch.device(config.trainer['device'])):
        trained = _load_for_output(config, resumed['checkpoint'])[0]
        expected = cpu_tree(_generate_values(config, trained, PreparedDataset(prepared).sample(0), {},
                                            output_kind='audio_video'))
    del trained
    gc.collect()
    inference = dict(schema_version='distillschedule.training.v1',
        model=dict(factory=exported['model']['factory'], parameters=exported['model']['parameters']),
        algorithm=dict(name='dmd'), data=dict(prepared=str(prepared)), execution=dict(adapter='program', mode='reference'),
        trainer=dict(cycles=1, seed=config.trainer['seed'], device=config.trainer['device'], output_dir=str(tmp_path / 'unused')))
    request = tmp_path / 'inference.json'; request.write_text(json.dumps(inference))
    assets = Path(native_assets['assets']); hidden = assets.with_name('REMOVED_TRAINING_ASSETS')
    assets.rename(hidden)
    try:
        generated = run_cli('generate', request, '--weights', tmp_path / 'export', '--prepared', prepared,
                            '--output', tmp_path / 'new-av', '--output-kind', 'audio_video')
        assert equal(torch.load(generated['artifact'], weights_only=True), expected)
        assert Path(generated['audio']['path']).is_file()
        assert len(list(Path(generated['video']['frames']).glob('*.png'))) == 3
        inference['trainer']['device'] = 'cpu'; request.write_text(json.dumps(inference))
        relocated = run_cli('generate', request, '--weights', tmp_path / 'export', '--prepared', prepared,
                            '--output', tmp_path / 'cpu-av', '--output-kind', 'audio_video')
        assert torch.load(relocated['artifact'], weights_only=True)['decoded']['video'].shape == (3, 8, 8, 3)
    finally:
        hidden.rename(assets)
