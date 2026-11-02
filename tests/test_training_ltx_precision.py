"""LTX native mixed compute with FP32 flow, codecs and persistent training state."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_training_ltx_cpu import assets, source_root
from test_training_end_to_end import equal, run_cli


def amp_config(assets, path, *, config_key='lora_config'):
    from distillschedule.training.config import load_training_config
    value = json.loads(Path(assets[config_key]).read_text())
    value['model']['parameters']['compute_precision'] = 'bf16-mixed'
    path.write_text(json.dumps(value))
    return load_training_config(path)


def test_ltx_default_fp32_characterization(assets):
    """Preserve the exact default identity envelope, not an implicit new policy."""
    from distillschedule.training.session import construct, build_cycle
    from distillschedule.training.config import load_training_config
    config = load_training_config(assets['config'])
    components, dataset, algorithm, _ = construct(config)
    assert components.metadata['precision'] == 'FP32_MATH_SDPA'
    assert components.metadata['placement'] == {'device': 'cpu', 'world_size': 1, 'precision': 'FP32'}
    assert components.metadata['scope'] == 'BOUNDED_CPU1_OR_CUDA1_REFERENCE; NO_AMP_DISTRIBUTED_OR_FULL_PRETRAINED_22B_CLAIM'
    assert 'compute_precision' not in components.metadata and 'flow_compute_precision' not in components.state
    projected = components.inference_projection(components, config.model['parameters'])
    assert 'compute_precision' not in projected.parameters and 'compute_precision' not in projected.components.metadata
    row = dataset.sample(0)
    prediction = components.state['student'].denoise(row['latents'], torch.tensor([.625]), row, conditional=True)
    assert all(value.dtype == torch.float32 for value in prediction.values())
    assert build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0).identity
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('precision', ['', 'bf16', 'float16', True, None, []])
def test_ltx_compute_precision_catalog_rejects_before_imports(precision):
    from distillschedule.training.catalog import resolve, validate_compatibility
    entry = resolve('model', 'ltx25')
    with pytest.raises(ValueError, match='compute precision'):
        validate_compatibility(dict(factory=entry['factory'], parameters=dict(compute_precision=precision)),
            dict(factory='distillschedule.training.dmd:build_algorithm'), dict(device='cpu'))


def test_ltx_compute_precision_discovery_remains_torch_free():
    result = subprocess.run([sys.executable, '-c', '''
import sys
from distillschedule.training.catalog import resolve, validate_compatibility
entry = resolve('model', 'ltx25')
assert entry['compute_precisions'] == ['fp32', 'bf16-mixed']
for precision in entry['compute_precisions']:
    validate_compatibility(dict(factory=entry['factory'], parameters=dict(compute_precision=precision)),
        dict(factory='distillschedule.training.dmd:build_algorithm'), dict(device='cpu'))
assert not any(name in sys.modules for name in ('torch', 'transformers', 'ltx_core'))
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('precision', ['fp32', 'bf16-mixed'])
def test_ltx_cold_and_warm_factory_preserve_all_caller_rng(assets, tmp_path, precision):
    value = json.loads(Path(assets['lora_config']).read_text())
    value['model']['parameters']['compute_precision'] = precision
    config = tmp_path / 'cold-owner.json'; config.write_text(json.dumps(value))
    result = subprocess.run([sys.executable, '-c', '''
import sys
from distillschedule.training.config import load_training_config
from distillschedule.training.session import _construct_model, _seed
from distillschedule.training.checkpoint import pack_state, rng_state
from test_training_end_to_end import equal
config = load_training_config(sys.argv[1])
_seed(917, device='cpu')
before = rng_state(device='cpu')
first, _ = _construct_model(config)
assert equal(before, rng_state(device='cpu')), 'cold native construction changed caller RNG'
second, _ = _construct_model(config)
assert equal(before, rng_state(device='cpu')), 'warm native construction changed caller RNG'
assert equal(pack_state(first.state), pack_state(second.state)), 'cold/warm native state differs'
''', str(config)], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('stage', ['training', 'inference'])
@pytest.mark.parametrize('precision', ['', 'bf16', True, None, []])
def test_ltx_invalid_precision_rejects_before_source_assets_or_model(stage, precision, monkeypatch):
    import distillschedule.training.ltx as ltx
    from test_training_ltx_device import _parameters
    def forbidden(*args, **kwargs):
        pytest.fail('invalid precision reached original source/assets/model loading')
    monkeypatch.setattr(ltx, 'original', forbidden)
    parameters = _parameters()
    build = ltx.build_model
    if stage == 'inference':
        parameters = dict(source_root='must-not-load', architecture={}, media_architecture={}, binding='00'*32,
            trainable=dict(mode='full'), generation_sigmas=[1., 0.])
        build = ltx.build_inference_model
    with pytest.raises(ValueError, match='compute precision'):
        build(dict(parameters, compute_precision=precision))
    assert not torch.cuda.is_initialized()


def native_prediction(role, row, sigma, *, conditional=True):
    """Invoke the actual upstream AV ABI under caller-owned native autocast."""
    api = role.source.import_module('ltx_core.model.transformer.modality')
    values = role.apply_conditioning(row['latents'], row)
    contexts = row['contexts'] if conditional else row['null_contexts']
    mask = row['context_mask'] if conditional else row['null_context_mask']
    mask = mask.to(torch.int64) if mask.dtype == torch.bool else mask
    modalities = {}
    for name in ('video', 'audio'):
        times = sigma[:, None].expand(values[name].shape[:2]).clone()
        times.masked_fill_(row['conditioning_mask'][name], 0.)
        modalities[name] = api.Modality(latent=values[name], sigma=sigma, timesteps=api.TokenTimesteps(times),
            positions=row['positions'][name], context=contexts[name], context_mask=mask, keyframes_mask=None)
    return role.native(video=modalities['video'], audio=modalities['audio'], perturbations=None)


@pytest.mark.parametrize('config_key', ['config', 'lora_config'])
def test_ltx_native_bf16_forward_gradient_updates_and_sharing(assets, tmp_path, config_key):
    from distillschedule.training.session import construct, build_cycle
    from distillschedule.open_workload.torch_backend import Session, resolve_module, sharing_inventory
    from distillschedule.training.checkpoint import pack_state
    config = amp_config(assets, tmp_path / 'amp.json', config_key=config_key)
    runs = []
    for sharing in (False, True):
        components, dataset, algorithm, _ = construct(config)
        role = components.state['student']
        assert type(role.native).__name__ == 'LTXModel'
        assert all(components.state[name].compute_precision == 'bf16-mixed' for name in ('student', 'fake', 'teacher'))
        assert components.metadata['compute_precision'] == components.state['flow_compute_precision'] == 'bf16-mixed'
        assert components.metadata['precision'] == 'BF16_AUTOCAST_FP32_STATE_SINGLE_DEVICE'
        assert all(value.dtype == torch.float32 for name in ('student', 'fake', 'teacher', 'media')
                   for value in components.state[name].parameters())
        row = dataset.sample(0); sigma = torch.tensor([.625])
        seen = []
        handles = [module.register_forward_hook(lambda _, __, value:
                   seen.append((torch.is_autocast_enabled('cpu'), value.dtype)))
                   for module in role.native.modules() if type(module) is torch.nn.Linear]
        actual = role.denoise(row['latents'], sigma, row, conditional=True)
        assert seen and all(enabled and dtype == torch.bfloat16 for enabled, dtype in seen)
        assert all(value.dtype == torch.float32 and value.requires_grad for value in actual.values())
        assert not torch.is_autocast_enabled('cpu')
        with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
            expected = native_prediction(role, row, sigma)
        assert equal(actual, {name: value.float() for name, value in zip(('video', 'audio'), expected)})
        actual_loss = sum(value.square().mean() for value in actual.values())
        native_loss = sum(value.float().square().mean() for value in expected)
        parameters = tuple(p for p in role.parameters() if p.requires_grad)
        actual_gradients = torch.autograd.grad(actual_loss, parameters, retain_graph=True, allow_unused=True)
        native_gradients = torch.autograd.grad(native_loss, parameters, allow_unused=True)
        assert equal(actual_gradients, native_gradients), 'native BF16 backward must remain the upstream computation'
        actual_loss.backward()
        assert any(p.grad is not None and p.grad.count_nonzero() for p in role.parameters() if p.requires_grad)
        assert all(p.grad is None or p.grad.dtype == torch.float32 for p in role.parameters())
        parameter = next(p for p in role.parameters() if p.requires_grad and p.grad is not None and p.grad.count_nonzero())
        initial, version = parameter.detach().clone(), parameter._version
        unchanged = role.denoise(row['latents'], sigma, row, conditional=False)
        with torch.no_grad():
            parameter.add_(.125)
        assert parameter._version > version
        changed = role.denoise(row['latents'], sigma, row, conditional=False)
        assert any(not torch.equal(changed[name], unchanged[name]) for name in changed)
        with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
            expected = native_prediction(role, row, sigma, conditional=False)
        assert equal(changed, {name: value.float() for name, value in zip(('video', 'audio'), expected)})
        with torch.no_grad():
            parameter.copy_(initial)
        for handle in handles:
            handle.remove()
        def failure(*args):
            raise RuntimeError('native scope witness')
        handle = role.native.register_forward_hook(failure)
        try:
            with pytest.raises(RuntimeError, match='native scope witness'):
                role.denoise(row['latents'], sigma, row, conditional=True)
        finally:
            handle.remove()
        assert not torch.is_autocast_enabled('cpu')
        for name in ('student', 'fake'):
            components.state[name + '_optimizer'].zero_grad(set_to_none=True)
        frozen = {name: deepcopy(resolve_module(components.state, name).state_dict()) for name in components.immutable_modules}
        before = {name: {key: p.detach().clone() for key, p in components.state[name].named_parameters() if p.requires_grad}
                  for name in ('student', 'fake')}
        workload = build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
        pairs = sharing_inventory(workload)
        assert pairs, 'actual equal native frozen parameters must exercise sharing'
        session = Session(workload, dict(backend='torch-callable-resident-v1', share_immutable=sharing,
            checkpoint_bindings=[]), copy_state=False)
        outputs = session.run()
        assert bool(session.effects['shared_parameter_aliases']) == sharing
        first, second, _ = pairs[0]
        source = resolve_module(session.state, first[0]).get_parameter(first[1])
        target = resolve_module(session.state, second[0]).get_parameter(second[1])
        assert (source is target) == (source.data_ptr() == target.data_ptr()) == sharing
        assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in outputs.values())
        assert components.state['dmd_counters'].tolist() == [1, 1, 5]
        for path, saved in frozen.items():
            assert equal(saved, resolve_module(components.state, path).state_dict())
        for name, steps in (('student', 1), ('fake', 5)):
            owned = {id(p) for p in components.state[name].parameters() if p.requires_grad}
            assert owned == {id(p) for group in components.state[name + '_optimizer'].param_groups for p in group['params']}
            slots = components.state[name + '_optimizer'].state.values()
            assert slots and all(float(slot['step']) == steps and slot['exp_avg'].dtype == slot['exp_avg_sq'].dtype == torch.float32 for slot in slots)
            assert any(slot['exp_avg'].count_nonzero() for slot in slots)
            assert any(not torch.equal(p, before[name][key]) for key, p in components.state[name].named_parameters() if p.requires_grad)
        assert not ({id(p) for p in components.state['student'].parameters() if p.requires_grad} &
                    {id(p) for p in components.state['fake'].parameters() if p.requires_grad})
        assert all(value.dtype == torch.float32 for value in components.state['student_ema'].values())
        assert not torch.is_autocast_enabled('cpu')
        runs.append(dict(outputs={key: value.detach().clone() for key, value in outputs.items()}, state=deepcopy(pack_state(components.state))))
    assert equal(*runs), 'reference and exact immutable sharing must commit the same AMP updates'
    assert not torch.cuda.is_initialized()


def test_ltx_default_and_explicit_fp32_preserve_descriptor_and_workload(assets):
    from dataclasses import replace
    from distillschedule.training.session import construct, build_cycle
    from distillschedule.training.config import load_training_config
    from distillschedule.training.checkpoint import pack_state
    config = load_training_config(assets['config'])
    explicit = replace(config, model={**config.model,
        'parameters': {**config.model['parameters'], 'compute_precision': 'fp32'}})
    snapshots = []
    for request in (config, explicit):
        model, data, algorithm, _ = construct(request)
        assert all(model.state[name].compute_precision == 'fp32' for name in ('student', 'fake', 'teacher'))
        projection = model.inference_projection(model, request.model['parameters'])
        workload = build_cycle(request, model, data, algorithm, cycle=0, cursor=0)
        snapshots.append(dict(metadata=model.metadata, state=pack_state(model.state),
            inference=projection.parameters, workload_identity=workload.identity))
    assert equal(*snapshots)


def test_ltx_compute_policy_binds_program_resume_and_live_admission(assets, tmp_path, monkeypatch):
    from dataclasses import replace
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import construct, build_cycle
    from distillschedule.training.policy import CyclePolicy, cycle_structure
    from distillschedule.training.checkpoint import pack_state
    control = load_training_config(assets['lora_config'])
    amp = amp_config(assets, tmp_path / 'amp.json')
    workloads = []
    for config in (control, amp):
        model, data, algorithm, _ = construct(config)
        workloads.append(build_cycle(config, model, data, algorithm, cycle=0, cursor=0))
    assert workloads[0].identity != workloads[1].identity
    assert cycle_structure(workloads[0]) != cycle_structure(workloads[1])
    policy = CyclePolicy({'mode': 'reference'})
    policy.run(workloads[0], tmp_path / 'control-policy')
    resumed = CyclePolicy({'mode': 'reference'}); resumed.restore(policy.bound)
    before = deepcopy(pack_state(model.state))
    with pytest.raises(ValueError, match='source or structure changed'):
        resumed.prepare_resume(workloads[1])
    with pytest.raises(ValueError, match='compute precision'):
        model.admit_training(state=model.state, request=replace(amp, model=control.model),
            prepared_identity=data.identity, prepared_manifest=data.manifest)
    for name in ('student', 'fake', 'teacher'):
        with monkeypatch.context() as patch:
            patch.setattr(model.state[name], 'compute_precision', 'fp32')
            with pytest.raises(ValueError, match='role compute precision'):
                model.admit_training(state=model.state, request=amp,
                    prepared_identity=data.identity, prepared_manifest=data.manifest)
    with monkeypatch.context() as patch:
        patch.setitem(model.state, 'flow_compute_precision', 'fp32')
        with pytest.raises(ValueError, match='owned compute precision'):
            model.admit_training(state=model.state, request=amp,
                prepared_identity=data.identity, prepared_manifest=data.manifest)
    assert equal(before, pack_state(model.state)), 'failed precision admission must precede live state mutation'


def test_ltx_bf16_keeps_cuda_trial_and_distributed_guards():
    from distillschedule.training.catalog import resolve, validate_compatibility
    from distillschedule.training.ltx import _admit_device
    from test_training_flow_image_device import CUDA_UNSUPPORTED
    entry = resolve('model', 'ltx25')
    for execution in CUDA_UNSUPPORTED:
        with pytest.raises(ValueError, match='CUDA execution requires'):
            validate_compatibility(dict(factory=entry['factory'], parameters=dict(
                device='cuda:0', compute_precision='bf16-mixed')), {}, dict(device='cuda:0'), execution)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('WORLD_SIZE', '2')
        with pytest.raises(ValueError, match='one actual process/rank'):
            _admit_device(dict(device='cpu', compute_precision='bf16-mixed'))
    assert not torch.cuda.is_initialized()


def test_ltx_bf16_public_auto_and_saved_plan_execute_same_updates(assets, tmp_path):
    from distillschedule.training.policy import cycle_structure
    from distillschedule.training.session import construct, build_cycle
    auto = tmp_path / 'auto.json'
    config = amp_config(assets, auto)
    value = json.loads(auto.read_text())
    value['execution'] = dict(mode='auto', memory_limit_bytes=100_000_000,
        max_candidates=3, samples=1, max_program_runs=5)
    auto.write_text(json.dumps(value))
    selected = run_cli('train', auto, '--cycles', 1, '--output', tmp_path / 'auto')
    saved_plan = tmp_path / 'auto/planning/selected-plan.json'
    plan = json.loads(saved_plan.read_text())
    value['execution'] = dict(mode='plan', plan=str(saved_plan))
    saved = tmp_path / 'saved.json'; saved.write_text(json.dumps(value))
    executed = run_cli('train', saved, '--cycles', 1, '--output', tmp_path / 'saved')
    a = torch.load(Path(selected['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(executed['checkpoint']) / 'state.pt', weights_only=True)
    for field in ('state', 'rng', 'cycle', 'cursor', 'versions'):
        assert equal(a[field], b[field]), field
    assert a['state']['flow_compute_precision']['value'] == 'bf16-mixed'
    assert a['state']['dmd_counters']['value']['counters'].tolist() == [1, 1, 5]
    row = json.loads((tmp_path / 'saved/metrics.jsonl').read_text().splitlines()[0])
    assert row['execution']['dispatcher']['executed_operations'] == plan['selected']['order']
    model, dataset, algorithm, _ = construct(config)
    workload = build_cycle(config, model, dataset, algorithm, cycle=0, cursor=0)
    assert selected['policy']['structure'] == executed['policy']['structure'] == cycle_structure(workload)
    assert not torch.cuda.is_initialized()


def test_ltx_bf16_new_process_resume_and_current_student_av_export(assets, tmp_path):
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.inference import read_export, restore_modules
    from distillschedule.data import PreparedDataset
    config = tmp_path / 'amp.json'; amp_config(assets, config)
    first = run_cli('train', config, '--cycles', 1, '--output', tmp_path / 'first')
    wrong_output = tmp_path / 'wrong-precision'
    rejected = subprocess.run([sys.executable, '-m', 'distillschedule', 'train', assets['lora_config'],
        '--cycles', '2', '--resume', first['checkpoint'], '--output', str(wrong_output)],
        capture_output=True, text=True, timeout=180)
    assert rejected.returncode != 0 and 'identity' in rejected.stdout + rejected.stderr
    assert not wrong_output.exists()
    resumed = run_cli('train', config, '--cycles', 2, '--resume', first['checkpoint'], '--output', tmp_path / 'resumed')
    whole = run_cli('train', config, '--cycles', 2, '--output', tmp_path / 'whole')
    a = torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(whole['checkpoint']) / 'state.pt', weights_only=True)
    assert equal(a, b), 'parameters, Adam, schedulers, EMA, RNG, data and native clocks must resume exactly'
    assert a['cycle'] == a['cursor'] == 2
    assert a['state']['dmd_counters']['value']['counters'].tolist() == [2, 2, 10]
    assert a['state']['student_scheduler']['value']['last_epoch'] == 2
    assert a['state']['fake_scheduler']['value']['last_epoch'] == 10
    rows = [json.loads(line) for line in (tmp_path / 'resumed/metrics.jsonl').read_text().splitlines()]
    assert [row['sample_index'] for row in rows] == [1]
    exported = run_cli('export', config, '--checkpoint', resumed['checkpoint'], '--output', tmp_path / 'export')
    assert set(exported['model']['modules']) == {'student', 'media'}
    assert exported['model']['parameters']['compute_precision'] == 'bf16-mixed'
    assert 'assets' not in exported['model']['parameters']
    factory, _ = load_callable(exported['model']['factory'])
    current = factory(exported['model']['parameters'])
    root, manifest = read_export(tmp_path / 'export')
    restore_modules(current, root, manifest)
    assert current.state['student'].compute_precision == 'bf16-mixed'
    assert set(current.state) == {'student', 'media'}
    source = tmp_path / 'new-input.json'
    source.write_text(json.dumps(dict(schema_version='distillschedule.export_inputs.v1', weights=str(root),
        mode='prompts', records=[dict(id='new-amp-prompt', prompt='a dog rests', geometry=dict(
            fps=8., video_shape=[1, 128, 2, 2, 2], audio_shape=[1, 8, 11, 16],
            conditioning=dict(video_prefix_latent_frames=0, audio_prefix_latent_frames=0)))])))
    inference = dict(schema_version='distillschedule.training.v1',
        model=dict(factory=exported['model']['factory'], parameters=exported['model']['parameters']),
        algorithm=dict(name='dmd'), data={}, execution=dict(mode='reference'),
        trainer=dict(cycles=1, seed=909, device='cpu', output_dir=str(tmp_path / 'unused')))
    assets_path = Path(assets['assets']); hidden = assets_path.with_name(assets_path.name + '-retired')
    assets_path.rename(hidden)
    try:
        prepared = run_cli('data', 'prepare', source, '--output', tmp_path / 'new-prepared',
            '--preparer', 'distillschedule.data.export_inputs:prepare_inputs')['manifest']
        inference['data']['prepared'] = prepared
        request = tmp_path / 'inference.json'; request.write_text(json.dumps(inference))
        generated = run_cli('generate', request, '--weights', root, '--prepared', prepared,
            '--output', tmp_path / 'new-av', '--output-kind', 'audio_video')
    finally:
        hidden.rename(assets_path)
    for module in current.state.values():
        module.eval()
    torch.random.default_generator.manual_seed(909)
    with torch.no_grad():
        expected = current.generate(current.state, PreparedDataset(prepared).sample(0), {})
        expected_media = current.decode(expected, {})
    actual = torch.load(generated['artifact'], weights_only=True)
    assert equal(actual['generated'], expected) and equal(actual['decoded'], expected_media)
    assert all(actual['generated'][name].dtype == torch.float32 for name in ('video_latents', 'audio_latents'))
    assert actual['decoded']['video'].dtype == torch.uint8 and actual['decoded']['video'].shape == (3, 8, 8, 3)
    assert actual['decoded']['audio']['samples'].dtype == torch.float32
    assert actual['decoded']['audio']['samples'].shape == (9840, 2)
    assert Path(generated['audio']['path']).is_file() and len(list(Path(generated['video']['frames']).glob('*.png'))) == 3
    assert not torch.cuda.is_initialized()
