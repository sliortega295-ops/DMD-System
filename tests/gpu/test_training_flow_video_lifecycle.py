"""Actual tiny Wan2.2/HunyuanVideo1.5 CUDA reference ownership and lifecycle.

Functional FP32 only: not pretrained capacity/quality, long-run qualification,
performance evidence, or permission to clone resident CUDA owners for trials.
"""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import gc
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from distillschedule.data import PreparedDataset
from distillschedule.training.checkpoint import cpu_tree, pack_state, rng_state, unpack_state
from distillschedule.training.config import load_training_config
from distillschedule.training.policy import CyclePolicy
from distillschedule.training.session import construct, build_cycle
from distillschedule.training.single_device import device_scope, validate_placement
from test_training_end_to_end import equal
from cuda_test_policy import cuda_bitwise_policy, run_cli
from native_flow_oracle import _native_flow_oracle

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason='native video CUDA NOT_RUN: no CUDA'),
              pytest.mark.usefixtures('cuda_bitwise_policy')]


@pytest.fixture(scope='module', params=['wan22', 'hunyuan_video'])
def native_assets(request, tmp_path_factory):
    recipe = Path(__file__).resolve().parents[2] / ('recipes/' + request.param + '_dmd/create_cpu_fixture.py')
    result = subprocess.run([sys.executable, str(recipe),
        str(tmp_path_factory.mktemp('cuda-' + request.param) / 'fixture')], capture_output=True, text=True,
        env={**os.environ, 'OMP_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''})
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(json.loads(result.stdout), module=request.param)


def _config(files, tmp_path):
    value = json.loads(Path(files['config']).read_text())
    # Exercise an explicitly non-default visible device when available.
    value['trainer']['device'] = 'cuda:' + str(torch.cuda.device_count() - 1)
    path = tmp_path / 'cuda.json'; path.write_text(json.dumps(value))
    return load_training_config(path)


def _native_forward(module, role, feeds, value, sigma, *, conditional=True):
    """Independent native ABI call: never obtains arguments from Role.denoise."""
    prefix = '' if conditional else 'null_'
    if module == 'wan22':
        b, _, f, h, w = value.shape
        return role.native(hidden_states=value, timestep=(1000 * sigma[:, None]).expand(b, f * (h // 2) * (w // 2)),
            encoder_hidden_states=feeds[prefix + 'context'], return_dict=False)[0]
    return role.native(hidden_states=torch.cat((value, torch.zeros_like(value), torch.zeros_like(value[:, :1])), dim=1),
        timestep=1000 * sigma, image_embeds=value.new_zeros((1, 729, role.architecture['image_embed_dim'])),
        encoder_hidden_states=feeds[prefix + 'context'], encoder_attention_mask=feeds[prefix + 'context_mask'],
        encoder_hidden_states_2=feeds[prefix + 'context_2'], encoder_attention_mask_2=feeds[prefix + 'context_mask_2'],
        return_dict=False)[0]


def _placement(state, device):
    from distillschedule.open_workload.torch_backend import tensors
    assert all(t.device == device and not t.is_meta and (not t.is_floating_point() or t.dtype == torch.float32)
               for t in tensors(state))
    # Also inspect plain/native cache tensors that the regular tensor tree does
    # not own; their appearance after a real forward is not silently accepted.
    for value in state.values():
        if isinstance(value, torch.nn.Module):
            validate_placement(value, device)


def test_native_cuda_video_forward_gradient_euler_text_and_codec(native_assets, tmp_path):
    from distillschedule.data.decoding import _video
    from distillschedule.open_workload.torch_backend import random_scope
    from distillschedule.training.session import _construct_model, _seed
    config = _config(native_assets, tmp_path)
    device = torch.device(config.trainer['device'])
    # The factory preserves its caller; the public session intentionally seeds
    # training. Distinguish these boundaries and avoid a previous test masking
    # an invalid arbitrary-caller RNG comparison around construct().
    _seed(913, device=device)
    outer = rng_state(device=device); current_device = torch.cuda.current_device()
    factory_owner, _ = _construct_model(config)
    assert equal(outer, rng_state(device=device)) and torch.cuda.current_device() == current_device
    del factory_owner
    gc.collect()
    components, data, algorithm, _ = construct(config)
    assert not equal(outer, rng_state(device=device)) and torch.cuda.current_device() == current_device
    outer = rng_state(device=device)
    role, media = components.state['student'], components.state['media']
    api = importlib.import_module('distillschedule.training.' + native_assets['module'])
    from distillschedule.workloads.best_known.registry import file_sha256
    import distillschedule.training.single_device as ownership
    assert components.metadata['runtime']['source_files']['device_ownership'] == file_sha256(Path(ownership.__file__))
    assert type(role.native) is api.classes()[0] and type(media.vae) is api.classes()[1]
    assert type(media.text_encoder) is api.classes()[2]
    if native_assets['module'] == 'hunyuan_video':
        assert type(media.text_encoder_2) is api.classes()[3]
    feeds = build_cycle(config, components, data, algorithm, cycle=0, cursor=0).feeds
    _placement(components.state, device)
    original = data.sample(0)
    fields = ('context', 'context_mask') + (('context_2', 'context_mask_2') if native_assets['module'] == 'hunyuan_video' else ())
    for key in fields:
        assert torch.equal(getattr(role, 'empty_' + key), original['null_' + key].to(device))
    value = feeds['latents']['video']; sigma = value.new_full((1,), .625)
    params = tuple(p for p in role.parameters() if p.requires_grad)
    for conditional in (True, False):
        actual = role.denoise({'video': value}, sigma, feeds, conditional=conditional)['video']
        native = _native_forward(native_assets['module'], role, feeds, value, sigma, conditional=conditional)
        assert torch.equal(actual, native), 'native timestep/sign/channel/text ABI'
        a = torch.autograd.grad(actual.square().mean(), params)
        b = torch.autograd.grad(native.square().mean(), params)
        assert all(torch.equal(x, y) for x, y in zip(a, b)) and any(bool(g.any()) for g in a)
    for gradient_policy in ('all_steps', 'last_step'):
        with random_scope(991, 'video-native-euler', device):
            actual = role.rollout(feeds, role.generation_sigmas, gradient_policy)['video']
        with random_scope(991, 'video-native-euler', device):
            manual = torch.randn_like(value)
            for index, (sigma_value, following) in enumerate(zip(role.generation_sigmas, role.generation_sigmas[1:])):
                context = torch.no_grad() if gradient_policy == 'last_step' and index < 3 else nullcontext()
                with context:
                    velocity = _native_forward(native_assets['module'], role, feeds, manual,
                                               manual.new_full((1,), sigma_value))
                    manual = manual + (following - sigma_value) * velocity
        assert torch.equal(actual, manual), 'native four-step Euler'
        a = torch.autograd.grad(actual.square().mean(), params)
        b = torch.autograd.grad(manual.square().mean(), params)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert equal(outer, rng_state(device=device)), 'logical calls must not consume caller RNG'
    with torch.no_grad():
        if native_assets['module'] == 'wan22':
            prompt = ' a &amp;amp; new bird '
            context, mask = media.encode_text([prompt])
            native_context = media.pipeline()._get_t5_prompt_embeds([prompt], max_sequence_length=226,
                                                                   device=device, dtype=torch.float32)
            assert torch.equal(context, native_context) and context.device == mask.device == device
            assert not bool(context[~mask].any())
        else:
            pipeline = api.classes()[4]
            for prompt, has_glyph in [('a new bird', False), ('a sign says "NEW"', True)]:
                encoded = media.encode_text([prompt])
                native = (*pipeline._get_mllm_prompt_embeds(media.text_encoder, media.tokenizer, [prompt], device=device),
                          *pipeline._get_byt5_prompt_embeds(media.tokenizer_2, media.text_encoder_2, [prompt], device=device))
                assert all(torch.equal(x, y) and x.device == device for x, y in zip(encoded, native))
                assert bool(encoded[3].any()) == has_glyph
                if not has_glyph:
                    assert not bool(encoded[2].any()), 'native no-glyph zero/allfalse sentinel'
        frames = _video(Path(native_assets['root']) / 'frames')
        latent = media.encode_video(frames)
        pixels = (frames.permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1).to(device)
        raw = media.vae.encode(pixels).latent_dist.mode()
        if native_assets['module'] == 'wan22':
            mean = raw.new_tensor(media.vae.config.latents_mean).view(1, -1, 1, 1, 1)
            std = raw.new_tensor(media.vae.config.latents_std).view(1, -1, 1, 1, 1)
            assert torch.equal(latent, (raw - mean) / std)
            inverse = latent * std + mean
        else:
            assert torch.equal(latent, raw * media.vae.config.scaling_factor)
            inverse = latent / media.vae.config.scaling_factor
        assert torch.equal(media.decode_latents(latent), media.vae.decode(inverse, return_dict=False)[0])
        decoded = media.decode({'video_latents': latent}, {})['video']
        assert decoded.device == device and decoded.dtype == torch.uint8 and decoded.shape == frames.shape
    _placement(components.state, device)
    assert torch.cuda.current_device() == current_device


def test_native_cuda_video_dmd_adam_ema_restore_and_refusals(native_assets, tmp_path):
    config = _config(native_assets, tmp_path)
    components, data, algorithm, _ = construct(config)
    device = components.state['student'].device
    workload = build_cycle(config, components, data, algorithm, cycle=0, cursor=0)
    _placement(components.state, device)
    initial = pack_state(components.state)
    frozen = {role: cpu_tree({name: p for name, p in components.state[role].named_parameters() if not p.requires_grad})
              for role in ('student', 'fake', 'teacher', 'media')}
    frozen_modules = {role: cpu_tree(components.state[role].state_dict()) for role in ('teacher', 'media')}
    outer = rng_state(device=device)
    expected_loss, expected_grad = _native_flow_oracle(components.state, workload.feeds,
                                                      config.algorithm['parameters'], workload.seed)
    expected_state = pack_state(components.state)
    assert equal(outer, rng_state(device=device))
    del components, data, algorithm, workload
    gc.collect()
    # Reconstruct sequentially; never deepcopy a live GPU owner to get an oracle.
    components, data, algorithm, _ = construct(config)
    assert equal(initial, pack_state(components.state))
    workload = build_cycle(config, components, data, algorithm, cycle=0, cursor=0)
    gradients = []; handles = []
    def observe(role):
        def before_step(optimizer, args, kwargs):
            gradients.append((role, cpu_tree({name: p.grad for name, p in components.state[role].named_parameters()
                                             if p.requires_grad})))
        return before_step
    owners = {role: {id(p) for p in components.state[role].parameters() if p.requires_grad} for role in ('student', 'fake')}
    assert owners['student'] and owners['fake'] and not owners['student'] & owners['fake']
    for role in owners:
        handles.append(components.state[role + '_optimizer'].register_step_pre_hook(observe(role)))
    outer = rng_state(device=device)
    policy = CyclePolicy(config.execution)
    session, losses = policy.run(workload, tmp_path / 'reference')
    for handle in handles:
        handle.remove()
    assert equal(expected_loss, cpu_tree(losses)) and equal(expected_grad, gradients)
    assert equal(expected_state, pack_state(components.state)) and equal(outer, rng_state(device=device))
    assert components.state['dmd_counters'].tolist() == [1, 1, 5]
    _placement(components.state, device)
    for role in frozen:
        assert equal(frozen[role], cpu_tree({name: p for name, p in components.state[role].named_parameters() if not p.requires_grad}))
    for role in frozen_modules:
        assert equal(frozen_modules[role], cpu_tree(components.state[role].state_dict()))
    for role, count in (('student', 1), ('fake', 5)):
        optimizer = components.state[role + '_optimizer']
        assert {id(p) for group in optimizer.param_groups for p in group['params']} == owners[role]
        assert all(group['capturable'] for group in optimizer.param_groups)
        assert optimizer.state and all(value['step'].device == value['exp_avg'].device == value['exp_avg_sq'].device == device
                                       and value['step'].item() == count for value in optimizer.state.values())
        assert components.state[role + '_scheduler'].last_epoch == count
    before = pack_state(components.state)
    versions = session.receipt['dispatcher']['final_versions']
    for fault in ('ema', 'adam', 'clock'):
        bad = deepcopy(before)  # Detached CPU checkpoint, not CUDA state.
        if fault == 'ema':
            next(iter(bad['student_ema']['value'].values())).reshape(-1)[0] = float('nan')
        elif fault == 'adam':
            next(iter(bad['fake_optimizer']['value']['state'].values()))['exp_avg'] = torch.zeros(1, dtype=torch.float64)
        else:
            bad['dmd_counters']['value']['counters'][2] = 4
        with pytest.raises(ValueError):
            unpack_state(bad, components.state, progress={'cycle': 1, 'cursor': 1, 'versions': versions})
        assert equal(before, pack_state(components.state)), 'failed restore must be atomic'
    for execution in ({'mode': 'auto'}, {'mode': 'plan'}, {'mode': 'manual'},
                      {'mode': 'reference', 'implementation': {}}, {'mode': 'reference', 'order': []},
                      {'mode': 'reference', 'plan': 'MISSING'}):
        with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
            components.admit_training(state=components.state, request=replace(config, execution=execution),
                prepared_identity=data.identity, prepared_manifest=data.manifest)
    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
        deepcopy(components.state)
    with pytest.raises(ValueError, match='device'):
        components.state['student'].validate_feeds(data.sample(0))
    for field in ('context_mask', 'latent_shape', 'encoder_identity'):
        with pytest.raises(ValueError, match='device'):
            components.state['student'].validate_feeds(dict(workload.feeds, **{field: workload.feeds[field].cpu()}))
    changed = dict(workload.feeds, encoder_identity=workload.feeds['encoder_identity'].clone())
    changed['encoder_identity'][0] ^= 1
    with pytest.raises(ValueError, match='binding'):
        components.state['student'].validate_feeds(changed)
    with torch.autocast('cuda', dtype=torch.bfloat16), pytest.raises(ValueError, match='FP32'):
        components.state['student'].denoise(workload.feeds['latents'], workload.feeds['latents']['video'].new_full((1,), .5),
                                           workload.feeds, conditional=True)
    next_cycle = build_cycle(config, components, data, algorithm, cycle=1, cursor=1, versions=versions)
    policy.run(next_cycle, tmp_path / 'reference')
    assert components.state['dmd_counters'].tolist() == [2, 2, 10]
    _placement(components.state, device)


def test_native_cuda_video_new_process_resume_export_new_inputs_and_cpu_relocation(native_assets, tmp_path):
    from distillschedule.training.session import _load_for_output, _generate_values, _construct_model
    from distillschedule.data.manifest import source_identity
    config = _config(native_assets, tmp_path)
    partial = run_cli('train', config.source, '--cycles', 1, '--output', tmp_path / 'partial')
    resumed = run_cli('train', config.source, '--cycles', 3, '--resume', partial['checkpoint'], '--output', tmp_path / 'resumed')
    whole = run_cli('train', config.source, '--cycles', 3, '--output', tmp_path / 'whole')
    assert resumed['gpu_execution'] == whole['gpu_execution'] == 'NATIVE_CUDA_RUN'
    a = torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(whole['checkpoint']) / 'state.pt', weights_only=True)
    assert equal(a, b), 'complete roles/optimizer/scheduler/EMA/RNG/data/cursor/version restart'
    assert a['cycle'] == a['cursor'] == 3 and a['state']['dmd_counters']['value']['counters'].tolist() == [3, 3, 15]
    assert a['state']['student_scheduler']['value']['last_epoch'] == 3 and a['state']['fake_scheduler']['value']['last_epoch'] == 15
    def read_metrics(name):
        return [json.loads(line) for line in (tmp_path / name / 'metrics.jsonl').read_text().splitlines()]
    rows, resumed_rows = read_metrics('whole'), read_metrics('resumed')
    assert [row['sample_id'] for row in rows] == ['native-video-0', 'native-video-1', 'native-video-2']
    assert [row['sample_id'] for row in resumed_rows] == ['native-video-1', 'native-video-2']
    for left, right in zip(rows[1:], resumed_rows):
        for key in ('cycle', 'cursor', 'sample_id', 'sample_index', 'epoch', 'epoch_offset', 'metrics', 'versions'):
            assert left[key] == right[key], key
    exported = run_cli('export', config.source, '--checkpoint', resumed['checkpoint'], '--output', tmp_path / 'export')
    assert set(exported['model']['modules']) == {'student', 'media'} and 'assets' not in exported['model']['parameters']
    original_row = PreparedDataset(native_assets['prepared']).sample(0)
    shape = original_row['latent_shape'].tolist()
    novel_frames = tmp_path / 'novel-frames'
    from PIL import Image, ImageOps
    novel_frames.mkdir()
    for frame in sorted((Path(native_assets['root']) / 'frames').glob('*.png')):
        with Image.open(frame) as image:
            ImageOps.mirror(image).save(novel_frames / frame.name)
    sources = {}
    for mode, record in (
        ('prompts', dict(id='NEW_PROMPT', prompt='a new sign says "NEW" by a river', latent_shape=shape)),
        ('raw', dict(id='NEW_RAW', prompt='a new bird on a river', video=dict(path=str(novel_frames),
                     sha256=source_identity(novel_frames, frames=True)['sha256']))),
    ):
        source = tmp_path / (mode + '-input.json')
        source.write_text(json.dumps(dict(schema_version='distillschedule.export_inputs.v1', weights=str(tmp_path / 'export'),
                                         mode=mode, records=[record])))
        sources[mode] = source
    # Load CURRENT Student once before removing source assets/checkpoints. Hold
    # only this original owner, not a duplicate CUDA trial or inference owner.
    with device_scope(torch.device(config.trainer['device'])):
        owner = _load_for_output(config, resumed['checkpoint'])[0]
    assert equal(a['state']['student']['value'], cpu_tree(owner.state['student'].state_dict()))
    for name in ('partial', 'resumed', 'whole'):
        shutil.copyfile(tmp_path / name / 'metrics.jsonl', tmp_path / (name + '-metrics.jsonl'))
        shutil.rmtree(tmp_path / name)
    original_assets = Path(native_assets['root']); retired = original_assets.with_name(original_assets.name + '-retired')
    original_assets.rename(retired)
    try:
        prepared = {}
        for mode, source in sources.items():
            result = run_cli('data', 'prepare', source, '--output', tmp_path / (mode + '-features'),
                             '--preparer', 'distillschedule.data.export_inputs:prepare_inputs')
            prepared[mode] = result['manifest']
        assert not torch.equal(PreparedDataset(prepared['prompts']).sample(0)['context'], original_row['context'])
        assert not torch.equal(PreparedDataset(prepared['raw']).sample(0)['latents']['video'], original_row['latents']['video'])
        with device_scope(torch.device(config.trainer['device'])):
            expected = {mode: cpu_tree(_generate_values(config, owner, PreparedDataset(path).sample(0), {}, output_kind='video'))
                        for mode, path in prepared.items()}
        del owner
        gc.collect()
        for mode, path in prepared.items():
            value = dict(schema_version='distillschedule.training.v1',
                model=dict(factory=exported['model']['factory'], parameters=exported['model']['parameters']),
                algorithm=dict(name='dmd'), data=dict(prepared=path), execution=dict(adapter='program', mode='reference'),
                trainer=dict(cycles=1, seed=config.trainer['seed'], device=config.trainer['device'], output_dir=str(tmp_path / 'unused')))
            request = tmp_path / (mode + '-inference.json'); request.write_text(json.dumps(value))
            generated = run_cli('generate', request, '--weights', tmp_path / 'export', '--prepared', path,
                                '--output', tmp_path / ('cuda-' + mode), '--output-kind', 'video')
            actual = torch.load(generated['artifact'], weights_only=True)
            assert equal(expected[mode], actual), 'export must reproduce CURRENT Student, not Teacher/base/EMA'
            assert actual['decoded']['video'].dtype == torch.uint8
            assert len(list(Path(generated['video']['frames']).glob('*.png'))) == 5
            value['trainer']['device'] = 'cpu'; request.write_text(json.dumps(value))
            # Independent CPU current-Student reference from the saved training
            # tensors, not the export loader used by the fresh CLI. Do not
            # impose bitwise equality between CPU and CUDA kernels.
            cpu_config = load_training_config(request)
            cpu_owner, _ = _construct_model(cpu_config)
            for role in ('student', 'media'):
                cpu_owner.state[role].load_state_dict(a['state'][role]['value'], strict=True)
                assert equal(a['state'][role]['value'], cpu_tree(cpu_owner.state[role].state_dict()))
            _placement(cpu_owner.state, torch.device('cpu'))
            expected_cpu = cpu_tree(_generate_values(cpu_config, cpu_owner, PreparedDataset(path).sample(0), {},
                                                    output_kind='video'))
            del cpu_owner
            relocated = run_cli('generate', request, '--weights', tmp_path / 'export', '--prepared', path,
                                '--output', tmp_path / ('cpu-' + mode), '--output-kind', 'video')
            cpu_actual = torch.load(relocated['artifact'], weights_only=True)
            assert equal(expected_cpu, cpu_actual), 'relocation must retain CURRENT Student and native media'
            assert cpu_actual['decoded']['video'].dtype == torch.uint8
            assert cpu_actual['decoded']['video'].shape == actual['decoded']['video'].shape
            assert len(list(Path(relocated['video']['frames']).glob('*.png'))) == 5
    finally:
        retired.rename(original_assets)
    (tmp_path / 'acceptance.json').write_text(json.dumps(dict(module=native_assets['module'],
        scope='TINY_NATIVE_CUDA_FP32_NOT_PRETRAINED_CAPACITY_LONG_RUN_OR_QUALITY', exact_restart=True,
        cycles=3, counts=[3, 3, 15], sample_ids=[row['sample_id'] for row in rows], versions=whole['final_versions'],
        new_prompt_and_raw_current_student=True, source_assets_and_checkpoints_absent=True, portable_cpu_output=True), indent=2))
