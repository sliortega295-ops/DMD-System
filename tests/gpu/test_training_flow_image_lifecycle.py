"""Actual tiny FLUX/Qwen CUDA ownership, reference math and portable lifecycle.

This is functional FP32 qualification, not pretrained capacity, quality, a
performance result or permission to clone CUDA roles for planning trials.
All native fixtures are SYNTHETIC_TEST_ONLY, including explicit manual sharing.
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

from distillschedule.data import PreparedDataset
from distillschedule.training.checkpoint import cpu_tree, pack_state, rng_state, unpack_state
from distillschedule.training.config import load_training_config
from distillschedule.training.policy import CyclePolicy
from distillschedule.training.session import construct, build_cycle
from test_training_end_to_end import equal
from cuda_test_policy import cuda_bitwise_policy, run_cli
from native_flow_oracle import (_native_flow_oracle, assert_manual_cycles, assert_manual_restart_policy,
                                manual_config, manual_sharing_only)

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason='native image CUDA NOT_RUN: no CUDA'),
              pytest.mark.usefixtures('cuda_bitwise_policy')]


@pytest.fixture(scope='module', params=['flux2-klein-base', 'qwen-image'])
def native_assets(request, tmp_path_factory):
    recipe = Path(__file__).resolve().parents[2] / 'recipes/flow_image/create_cpu_fixture.py'
    result = subprocess.run([sys.executable, str(recipe),
        str(tmp_path_factory.mktemp('cuda-' + request.param) / 'fixture'), '--family', request.param],
        capture_output=True, text=True, env={**os.environ, 'OMP_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''})
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def _config(files, tmp_path):
    value = json.loads(Path(files['config']).read_text())
    value['trainer']['device'] = 'cuda:' + str(torch.cuda.device_count() - 1)
    path = tmp_path / 'cuda.json'; path.write_text(json.dumps(value))
    return load_training_config(path)


def test_native_cuda_roles_gradients_adam_ema_and_refusals(native_assets, tmp_path):
    from distillschedule.open_workload.torch_backend import tensors
    config = _config(native_assets, tmp_path)
    previous = torch.cuda.current_device()
    components, data, algorithm, _ = construct(config)
    assert torch.cuda.current_device() == previous
    device = components.state['student'].device
    workload = build_cycle(config, components, data, algorithm, cycle=0, cursor=0)
    assert all(t.device == device for t in tensors((workload.state, workload.feeds)))
    initial = pack_state(components.state)
    frozen = cpu_tree(components.state['teacher'].state_dict())
    outer = rng_state(device=device)
    expected_loss, expected_grad = _native_flow_oracle(components.state, workload.feeds,
                                                      config.algorithm['parameters'], workload.seed)
    expected_state = pack_state(components.state)
    assert equal(outer, rng_state(device=device))
    del components, data, algorithm, workload
    gc.collect()
    # No live CUDA owner is deep-copied: recreate once from the original assets.
    components, data, algorithm, _ = construct(config)
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
    policy = CyclePolicy(config.execution)
    outer = rng_state(device=device)
    session, losses = policy.run(workload, tmp_path / 'reference')
    for handle in handles: handle.remove()
    assert equal(expected_loss, cpu_tree(losses)) and equal(expected_grad, gradients)
    assert equal(expected_state, pack_state(components.state)) and equal(outer, rng_state(device=device))
    assert equal(frozen, cpu_tree(components.state['teacher'].state_dict()))
    assert components.state['dmd_counters'].tolist() == [1, 1, 5]
    assert all(t.device == device for t in tensors(components.state))
    for role in ('student', 'fake'):
        optimizer = components.state[role + '_optimizer']
        assert {id(p) for group in optimizer.param_groups for p in group['params']} == {
            id(p) for p in components.state[role].parameters() if p.requires_grad}
        assert all(group['capturable'] for group in optimizer.param_groups)
        assert all(value['step'].device == value['exp_avg'].device == value['exp_avg_sq'].device == device
                   for value in optimizer.state.values())
    if native_assets['family'] == 'qwen-image':
        rope = components.state['student'].native.pos_embed
        assert rope.pos_freqs.is_complex() and bool(rope.pos_freqs.imag.any())
        assert rope.pos_freqs.device == rope.neg_freqs.device == device
    media = components.state['media']
    with torch.no_grad():
        context, mask = media.encode_text(['a new real prompt'])
        assert context.device == mask.device == device
        pixels = torch.linspace(-1., 1., 1 * 3 * 8 * 8).reshape(1, 3, 8, 8)
        tokens, shape = media.encode_image(pixels)
        assert tokens.device == device and torch.isfinite(tokens).all()
        generated = components.generate(components.state, workload.feeds, {})
        decoded = components.decode(generated, {})
        assert decoded['image'].device == device and decoded['image'].dtype == torch.uint8
    before = pack_state(components.state)
    versions = session.receipt['dispatcher']['final_versions']
    for fault in ('ema', 'adam', 'clock'):
        bad = deepcopy(before)
        if fault == 'ema':
            next(iter(bad['student_ema']['value'].values())).reshape(-1)[0] = float('nan')
        elif fault == 'adam':
            moment = next(iter(bad['fake_optimizer']['value']['state'].values()))
            moment['exp_avg'] = moment['exp_avg'].double()
        else:
            bad['dmd_counters']['value']['counters'][2] = 4
        with pytest.raises(ValueError):
            unpack_state(bad, components.state, progress={'cycle': 1, 'cursor': 1, 'versions': versions})
        assert equal(before, pack_state(components.state))
    for mode in ('auto', 'plan', 'manual'):
        with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
            components.admit_training(state=components.state, request=replace(config, execution={'mode': mode}),
                prepared_identity=data.identity, prepared_manifest=data.manifest)
    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
        deepcopy(components.state)
    with pytest.raises(ValueError, match='device'):
        components.state['student'].validate_feeds(data.sample(0))
    changed = dict(workload.feeds, encoder_identity=workload.feeds['encoder_identity'].clone())
    changed['encoder_identity'][0] ^= 1
    with pytest.raises(ValueError, match='binding'):
        components.state['student'].validate_feeds(changed)
    next_cycle = build_cycle(config, components, data, algorithm, cycle=1, cursor=1, versions=versions)
    policy.run(next_cycle, tmp_path / 'reference')
    assert components.state['dmd_counters'].tolist() == [2, 2, 10]


@manual_sharing_only
def test_native_cuda_manual_sharing_matches_reference(native_assets, tmp_path, monkeypatch):
    assert_manual_cycles(_config(native_assets, tmp_path), tmp_path, monkeypatch)


@pytest.mark.parametrize('mode', ['reference', pytest.param('manual', marks=manual_sharing_only)])
def test_native_cuda_fresh_process_resume_export_and_new_prompt(native_assets, tmp_path, mode):
    from distillschedule.training.session import _load_for_output, _generate_values
    from distillschedule.training.single_device import device_scope
    config = manual_config(_config(native_assets, tmp_path), mode)
    partial = run_cli('train', config.source, '--cycles', 1, '--output', tmp_path / 'partial')
    resumed = run_cli('train', config.source, '--cycles', 3, '--resume', partial['checkpoint'], '--output', tmp_path / 'resumed')
    whole = run_cli('train', config.source, '--cycles', 3, '--output', tmp_path / 'whole')
    assert resumed['gpu_execution'] == whole['gpu_execution'] == 'NATIVE_CUDA_RUN'
    a = torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(whole['checkpoint']) / 'state.pt', weights_only=True)
    assert equal(a, b), 'roles, Adam, scheduler, EMA, RNG, cursor and clocks'
    assert a['cycle'] == a['cursor'] == 3 and a['state']['dmd_counters']['value']['counters'].tolist() == [3, 3, 15]
    assert_manual_restart_policy(config, partial, resumed, whole, tmp_path)
    exported = run_cli('export', config.source, '--checkpoint', resumed['checkpoint'], '--output', tmp_path / 'export')
    assert set(exported['model']['modules']) == {'student', 'media'} and 'assets' not in exported['model']['parameters']
    shape = PreparedDataset(native_assets['prepared']).sample(0)['latent_shape'].tolist()
    source = tmp_path / 'new-input.json'
    source.write_text(json.dumps(dict(schema_version='distillschedule.export_inputs.v1', weights=str(tmp_path / 'export'),
        mode='prompts', records=[dict(id='NEW_PROMPT', prompt='a new real prompt', latent_shape=shape)])))
    dataset = run_cli('data', 'prepare', source, '--output', tmp_path / 'new-features',
                     '--preparer', 'distillschedule.data.export_inputs:prepare_inputs')
    prepared = dataset['manifest']
    assert not torch.equal(PreparedDataset(prepared).sample(0)['context'],
                           PreparedDataset(native_assets['prepared']).sample(0)['context'])
    with device_scope(torch.device(config.trainer['device'])):
        owner = _load_for_output(config, resumed['checkpoint'])[0]
        expected = cpu_tree(_generate_values(config, owner, PreparedDataset(prepared).sample(0), {}, output_kind='image'))
    del owner
    gc.collect()
    value = dict(schema_version='distillschedule.training.v1',
        model=dict(factory=exported['model']['factory'], parameters=exported['model']['parameters']),
        algorithm=dict(name='dmd'), data=dict(prepared=prepared), execution=dict(adapter='program', mode='reference'),
        trainer=dict(cycles=1, seed=config.trainer['seed'], device=config.trainer['device'], output_dir=str(tmp_path / 'unused')))
    request = tmp_path / 'inference.json'; request.write_text(json.dumps(value))
    original_assets = Path(native_assets['root']); retired = original_assets.with_name(original_assets.name + '-retired')
    original_assets.rename(retired)
    try:
        generated = run_cli('generate', request, '--weights', tmp_path / 'export', '--prepared', prepared,
                            '--output', tmp_path / 'cuda-image', '--output-kind', 'image')
        assert equal(expected, torch.load(generated['artifact'], weights_only=True))
        assert Path(generated['image']['path']).is_file()
        value['trainer']['device'] = 'cpu'; request.write_text(json.dumps(value))
        relocated = run_cli('generate', request, '--weights', tmp_path / 'export', '--prepared', prepared,
                            '--output', tmp_path / 'cpu-image', '--output-kind', 'image')
        assert Path(relocated['image']['path']).is_file()
    finally:
        retired.rename(original_assets)
    (tmp_path / 'acceptance.json').write_text(json.dumps(dict(family=native_assets['family'],
        evidence_kind='SYNTHETIC_TEST_ONLY', execution_mode=mode,
        scope='TINY_NATIVE_CUDA_FP32_NOT_PRETRAINED_CAPACITY_OR_QUALITY',
        exact_restart=True, cycles=3, counts=[3, 3, 15], new_prompt_current_student=True,
        training_assets_absent_at_generation=True, portable_cpu_output=True), indent=2))
