"""Explicit native AMP compute, preserving FP32 flow math and owned state."""
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

FAMILIES = ('flux2-klein-base', 'qwen-image', 'wan22-ti2v5b-t2v', 'hunyuan15-t2v')


@pytest.mark.parametrize('precision', ['', 'bf16', 'float16', True, None, []])
def test_flow_compute_precision_rejects_unknown_values_without_loading_models(precision):
    from distillschedule.training.catalog import describe, validate_compatibility
    for name in FAMILIES:
        factory = describe()['models'][name]['factory']
        with pytest.raises(ValueError, match='compute precision'):
            validate_compatibility(dict(factory=factory, parameters={'compute_precision': precision}),
                dict(factory='distillschedule.training.dmd:build_algorithm'), {'device': 'cpu'})
    assert not torch.cuda.is_initialized()


def test_flow_compute_precision_catalog_is_lazy_in_a_fresh_process():
    script = '''
import sys
from distillschedule.training.catalog import describe, validate_compatibility
families = ('flux2-klein-base', 'qwen-image', 'wan22-ti2v5b-t2v', 'hunyuan15-t2v')
for name in families:
    entry = describe()['models'][name]
    assert entry['compute_precisions'] == ['fp32', 'bf16-mixed']
    for precision in entry['compute_precisions']:
        validate_compatibility(dict(factory=entry['factory'], parameters=dict(compute_precision=precision)),
            dict(factory='distillschedule.training.dmd:build_algorithm'), dict(device='cpu'))
assert not any(name in sys.modules for name in ('torch', 'diffusers', 'transformers'))
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('module', ['wan22', 'hunyuan_video'])
@pytest.mark.parametrize('precision', ['', 'bf16', True, None, []])
@pytest.mark.parametrize('stage', ['training', 'inference'])
def test_video_compute_precision_rejects_before_assets_or_native_construction(module, precision, stage, monkeypatch):
    api = importlib.import_module('distillschedule.training.' + module)
    def forbidden(*args, **kwargs):
        pytest.fail('invalid compute precision reached native assets or constructors')
    monkeypatch.setattr(api, 'load_media_bundle', forbidden)
    monkeypatch.setattr(api, 'classes', forbidden)
    monkeypatch.setattr(api, 'runtime_identity', forbidden)
    parameters = dict(assets='MISSING', device='cpu', trainable=dict(mode='lora'), student_lr=.003,
        fake_lr=.004, scheduler_gamma=.95, ema_decay=.9, generation_sigmas=[1., .75, .5, .25, 0.])
    build = api.build_model
    if stage == 'inference':
        parameters = {key: {} for key in api._DESCRIPTOR}
        build = api.build_inference_model
    with pytest.raises(ValueError, match='compute precision'):
        build(dict(parameters, compute_precision=precision))
    assert not torch.cuda.is_initialized()


def test_native_amp_scope_has_no_persistent_or_ambient_autocast():
    from distillschedule.training.single_device import native_compute_context
    module = torch.nn.Linear(4, 3)
    value = torch.ones(2, 4, requires_grad=True)
    with native_compute_context('bf16-mixed', torch.device('cpu')):
        assert torch.is_autocast_enabled('cpu')
        result = module(value)
        assert result.dtype == torch.bfloat16
    assert not torch.is_autocast_enabled('cpu')
    result.float().sum().backward()
    assert module.weight.grad.dtype == value.grad.dtype == torch.float32
    with pytest.raises(RuntimeError, match='test exit'):
        with native_compute_context('bf16-mixed', torch.device('cpu')):
            raise RuntimeError('test exit')
    assert not torch.is_autocast_enabled('cpu')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with pytest.raises(ValueError, match='ambient'):
            with native_compute_context('bf16-mixed', torch.device('cpu')):
                pass
    for bad in ('fp16', None, []):
        with pytest.raises(ValueError, match='compute precision'):
            native_compute_context(bad, torch.device('cpu'))
    assert not torch.cuda.is_initialized()


def test_native_bf16_capability_rejection_does_not_initialize_cuda(monkeypatch):
    from contextlib import nullcontext
    from distillschedule.training.single_device import native_compute_context
    monkeypatch.setattr(torch.amp.autocast_mode, 'is_autocast_available', lambda _: False)
    with pytest.raises(ValueError, match='autocast is unavailable'):
        native_compute_context('bf16-mixed', torch.device('cpu'))
    monkeypatch.setattr(torch.amp.autocast_mode, 'is_autocast_available', lambda _: True)
    monkeypatch.setattr(torch.cuda, 'device', lambda _: nullcontext())
    checks = []
    def unsupported(*, including_emulation):
        checks.append(including_emulation)
        return False
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', unsupported)
    with pytest.raises(ValueError, match='native device support'):
        native_compute_context('bf16-mixed', torch.device('cuda:0'))
    assert checks == [False]
    with native_compute_context('fp32', torch.device('cpu')):
        assert not torch.is_autocast_enabled('cpu')
        assert torch.nn.Linear(2, 2)(torch.ones(1, 2)).dtype == torch.float32
    assert not torch.cuda.is_initialized()


def test_bf16_does_not_silently_enable_frozen_teacher_sharding():
    from distillschedule.training.catalog import describe, validate_compatibility
    from distillschedule.training.single_device import FROZEN_TEACHER_CONSTRUCTION, FLOW_FROZEN_TEACHER_EXECUTION
    entry = describe()['models']['flux2-klein-base']
    with pytest.raises(ValueError, match='unsupported construction'):
        validate_compatibility(dict(factory=entry['factory'], parameters=dict(device='cpu',
            compute_precision='bf16-mixed', construction=FROZEN_TEACHER_CONSTRUCTION)),
            dict(factory='distillschedule.training.dmd:build_algorithm'),
            dict(device='cpu', distributed=dict(backend='gloo')),
            dict(mode='manual', implementation=FLOW_FROZEN_TEACHER_EXECUTION['manual_implementation']))
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope='module', params=FAMILIES)
def native_assets(request, tmp_path_factory):
    if os.environ.get('DMD_MODERN_FLOW_CPU') != '1':
        pytest.skip('isolated modern-flow lane required for actual upstream model classes')
    from distillschedule.training.plugins import load_callable
    root = tmp_path_factory.mktemp('amp-' + request.param) / 'fixture'
    recipes = {'wan22-ti2v5b-t2v': 'wan22_dmd', 'hunyuan15-t2v': 'hunyuan_video_dmd'}
    recipe = recipes.get(request.param, 'flow_image')
    create, _ = load_callable(str(Path(__file__).resolve().parents[1] /
        f'recipes/{recipe}/create_cpu_fixture.py') + ':build_fixture')
    assets = create(root) if request.param in recipes else create(root, request.param)
    # Reuse the existing modern-flow fixture budget for Hunyuan's longer
    # primary/glyph attention. This bounds tracked CPU test tensors only.
    return dict(assets, family=request.param,
        planning_limit_bytes=200_000_000 if request.param == 'hunyuan15-t2v' else 100_000_000)


def amp_config(assets, path):
    value = json.loads(assets['config'].read_text())
    value['model']['parameters']['compute_precision'] = 'bf16-mixed'
    path.write_text(json.dumps(value))
    from distillschedule.training.config import load_training_config
    return load_training_config(path)


@pytest.mark.parametrize('precision', ['fp32', 'bf16-mixed'])
def test_cold_native_factory_preserves_all_caller_rng(native_assets, tmp_path, precision):
    value = json.loads(native_assets['config'].read_text())
    value['model']['parameters']['compute_precision'] = precision
    config = tmp_path / 'cold-owner.json'
    config.write_text(json.dumps(value))
    script = '''
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
assert equal(pack_state(first.state), pack_state(second.state)), 'cold/warm native owners differ'
'''
    result = subprocess.run([sys.executable, '-c', script, str(config)],
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def native_prediction(role, row, sigma, family, *, conditional=True):
    """The pinned native ABI, shared by the AMP witness and no-cache check."""
    prefix = '' if conditional else 'null_'
    context = row[prefix + 'context']
    if family == 'flux2-klein-base':
        return role.native(hidden_states=row['latents']['image'], encoder_hidden_states=context,
            timestep=sigma, img_ids=row['img_ids'], txt_ids=row[prefix + 'txt_ids'], guidance=None, return_dict=False)[0]
    if family == 'qwen-image':
        return role.native(hidden_states=row['latents']['image'], encoder_hidden_states=context,
            encoder_hidden_states_mask=row[prefix + 'context_mask'], timestep=sigma,
            img_shapes=[[tuple(value)] for value in row['img_shapes'].tolist()], guidance=None, return_dict=False)[0]
    value = row['latents']['video']
    if family == 'wan22-ti2v5b-t2v':
        b, _, f, h, w = value.shape
        return role.native(hidden_states=value, timestep=(1000 * sigma[:, None]).expand(b, f * (h // 2) * (w // 2)),
            encoder_hidden_states=context, return_dict=False)[0]
    assert family == 'hunyuan15-t2v'
    return role.native(hidden_states=torch.cat((value, torch.zeros_like(value), torch.zeros_like(value[:, :1])), dim=1),
        timestep=sigma * 1000, image_embeds=value.new_zeros((1, 729, role.architecture['image_embed_dim'])),
        encoder_hidden_states=context, encoder_attention_mask=row[prefix + 'context_mask'],
        encoder_hidden_states_2=row[prefix + 'context_2'], encoder_attention_mask_2=row[prefix + 'context_mask_2'],
        return_dict=False)[0]


def test_native_bf16_forward_gradient_updates_and_immutable_sharing(native_assets, tmp_path):
    from distillschedule.training.session import construct, build_cycle
    from distillschedule.open_workload.torch_backend import Session
    from distillschedule.training.checkpoint import pack_state
    from test_training_end_to_end import equal

    config = amp_config(native_assets, tmp_path / 'amp.json')
    modality = 'video' if native_assets['family'] in ('wan22-ti2v5b-t2v', 'hunyuan15-t2v') else 'image'
    runs = []
    for sharing in (False, True):
        components, dataset, algorithm, _ = construct(config)
        role = components.state['student']
        assert all(components.state[r].compute_precision == 'bf16-mixed'
                   for r in ('student', 'fake', 'teacher'))
        assert components.metadata['compute_precision'] == 'bf16-mixed'
        assert components.metadata['precision'] == 'BF16_AUTOCAST_FP32_STATE_SINGLE_DEVICE'
        assert all(p.dtype == torch.float32 for r in ('student', 'fake', 'teacher')
                   for p in components.state[r].parameters())
        assert all(p.dtype == torch.float32 for p in components.state['media'].parameters())
        complex_before = {name: value.clone() for name, value in role.native.named_buffers() if value.is_complex()}
        if native_assets['family'] == 'qwen-image':
            assert complex_before, 'the real Qwen complex rotary tensors must remain owned and intact'
        row = dataset.sample(0)
        sigma = torch.tensor([.625])
        linear_calls = []
        handles = [module.register_forward_hook(lambda _, __, out:
                   linear_calls.append((torch.is_autocast_enabled('cpu'), out.dtype)))
                   for module in role.native.modules() if type(module) is torch.nn.Linear]
        native_outputs = []
        handles.append(role.native.register_forward_hook(lambda _, __, out: native_outputs.append(out[0].dtype)))
        actual = role.denoise(row['latents'], sigma, row, conditional=True)[modality]
        assert actual.dtype == torch.float32 and actual.requires_grad
        assert linear_calls and all(enabled and dtype == torch.bfloat16 for enabled, dtype in linear_calls)
        assert native_outputs == [torch.bfloat16]
        assert not torch.is_autocast_enabled('cpu')
        with torch.autocast('cpu', dtype=torch.bfloat16):
            expected = native_prediction(role, row, sigma, native_assets['family'])
        assert torch.equal(actual, expected.float()), 'the adapter must use the actual native AMP forward unchanged'
        actual.square().mean().backward()
        assert any(p.grad is not None and p.grad.count_nonzero() for p in role.parameters() if p.requires_grad)
        assert all(p.grad is None or p.grad.dtype == torch.float32 for p in role.parameters())
        # An in-place LoRA update must be observed by the next native call;
        # no cached BF16 view may survive a Parameter version increment.
        parameter = next(p for p in role.parameters() if p.requires_grad and p.grad is not None and p.grad.count_nonzero())
        original, version = parameter.detach().clone(), parameter._version
        unchanged = role.denoise(row['latents'], sigma, row, conditional=False)[modality].detach().clone()
        with torch.no_grad():
            parameter.add_(.125)
        assert parameter._version > version
        updated = role.denoise(row['latents'], sigma, row, conditional=False)[modality]
        assert not torch.equal(updated, unchanged), 'the version-change witness must change the native result'
        with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
            updated_native = native_prediction(role, row, sigma, native_assets['family'], conditional=False)
        assert torch.equal(updated, updated_native.float())
        assert all(p.dtype == torch.float32 for p in role.parameters())
        with torch.no_grad():
            parameter.copy_(original)
        for handle in handles:
            handle.remove()
        def fail_after_native(*args):
            raise RuntimeError('native forward witness failure')
        handle = role.native.register_forward_hook(fail_after_native)
        try:
            with pytest.raises(RuntimeError, match='native forward witness failure'):
                role.denoise(row['latents'], sigma, row, conditional=True)
        finally:
            handle.remove()
        assert not torch.is_autocast_enabled('cpu')
        for r in ('student', 'fake'):
            components.state[r + '_optimizer'].zero_grad(set_to_none=True)
        frozen = deepcopy(components.state['teacher'].state_dict())
        trainable_before = {r: {name: p.detach().clone() for name, p in components.state[r].named_parameters() if p.requires_grad}
                            for r in ('student', 'fake')}
        workload = build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
        session = Session(workload, dict(backend='torch-callable-resident-v1', share_immutable=sharing,
                                       checkpoint_bindings=[]), copy_state=False)
        output = session.run()
        assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in output.values())
        assert components.state['dmd_counters'].tolist() == [1, 1, 5]
        assert equal(frozen, components.state['teacher'].state_dict())
        assert all(torch.equal(value, dict(role.native.named_buffers())[name]) for name, value in complex_before.items())
        for r, steps in (('student', 1), ('fake', 5)):
            slots = components.state[r + '_optimizer'].state.values()
            assert slots and all(float(slot['step']) == steps and
                slot['exp_avg'].dtype == slot['exp_avg_sq'].dtype == torch.float32 for slot in slots)
            assert any(slot['exp_avg'].count_nonzero() for slot in slots), 'the actual cycle must have nonzero update gradients'
            assert any(not torch.equal(p, trainable_before[r][name]) for name, p in components.state[r].named_parameters() if p.requires_grad)
            assert all(p.dtype == torch.float32 for p in components.state[r].parameters())
        assert all(value.dtype == torch.float32 for value in components.state['student_ema'].values())
        assert not torch.is_autocast_enabled('cpu')
        runs.append({'output': {key: value.detach().clone() for key, value in output.items()},
                     'state': deepcopy(pack_state(components.state))})
    assert equal(*runs), 'AMP must retain exact reference/manual-sharing state and outputs'
    assert not torch.cuda.is_initialized()


def test_native_precision_changes_actual_workload_and_rejects_control_request(native_assets, tmp_path, monkeypatch):
    from dataclasses import replace
    from distillschedule.training.session import construct, build_cycle
    from distillschedule.training.config import load_training_config
    from distillschedule.training.policy import CyclePolicy, cycle_structure
    from distillschedule.training.checkpoint import pack_state
    from test_training_end_to_end import equal
    control = load_training_config(native_assets['config'])
    amp = amp_config(native_assets, tmp_path / 'amp.json')
    workloads = []
    for config in (control, amp):
        components, dataset, algorithm, _ = construct(config)
        workloads.append(build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0))
    assert workloads[0].identity != workloads[1].identity
    assert cycle_structure(workloads[0]) != cycle_structure(workloads[1])
    policy = CyclePolicy({'mode': 'reference'})
    policy.run(workloads[0], tmp_path / 'control-policy')
    restored = CyclePolicy({'mode': 'reference'})
    restored.restore(policy.bound)
    before = deepcopy(pack_state(components.state))
    with pytest.raises(ValueError, match='source or structure changed'):
        restored.prepare_resume(workloads[1])
    assert equal(before, pack_state(components.state)), 'wrong-precision saved policy must reject before state mutation'
    with pytest.raises(ValueError, match='compute precision'):
        components.admit_training(state=components.state, request=replace(amp, model=control.model),
            prepared_identity=dataset.identity, prepared_manifest=dataset.manifest)
    for role in ('student', 'fake', 'teacher'):
        with monkeypatch.context() as patch:
            patch.setattr(components.state[role], 'compute_precision', 'fp32')
            with pytest.raises(ValueError, match='role compute precision'):
                components.admit_training(state=components.state, request=amp,
                    prepared_identity=dataset.identity, prepared_manifest=dataset.manifest)
    with monkeypatch.context() as patch:
        patch.setitem(components.state, 'flow_compute_precision', 'fp32')
        with pytest.raises(ValueError, match='owned compute precision'):
            components.admit_training(state=components.state, request=amp,
                prepared_identity=dataset.identity, prepared_manifest=dataset.manifest)
    assert equal(before, pack_state(components.state)), 'failed precision admission must not mutate owned state'


def test_native_default_fp32_descriptor_is_unchanged(native_assets):
    from dataclasses import replace
    from distillschedule.training.session import construct, build_cycle
    from distillschedule.training.config import load_training_config
    from distillschedule.training.checkpoint import pack_state
    from test_training_end_to_end import equal
    control = load_training_config(native_assets['config'])
    explicit = replace(control, model={**control.model,
        'parameters': {**control.model['parameters'], 'compute_precision': 'fp32'}})
    snapshots = []
    for config in (control, explicit):
        components, dataset, algorithm, _ = construct(config)
        assert 'compute_precision' not in components.metadata
        assert 'flow_compute_precision' not in components.state
        assert all(components.state[role].compute_precision == 'fp32' for role in ('student', 'fake', 'teacher'))
        projected = components.inference_projection(components, config.model['parameters'])
        assert 'compute_precision' not in projected.parameters
        workload = build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
        snapshots.append(dict(metadata=components.metadata, parameters=projected.parameters,
            state=pack_state(components.state), workload_identity=workload.identity))
    assert equal(*snapshots), 'explicit fp32 must preserve the default descriptor and Program identity'


def test_native_bf16_calibration_saved_plan_and_real_execution(native_assets, tmp_path):
    from distillschedule.training.session import construct, build_cycle
    from distillschedule.training.checkpoint import pack_state
    from distillschedule.open_workload.planning import calibrate, plan, run
    from test_training_end_to_end import equal

    config = amp_config(native_assets, tmp_path / 'amp.json')
    components, dataset, algorithm, _ = construct(config)
    workload = build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
    initial = deepcopy(pack_state(components.state))
    calibrate(workload, tmp_path / 'calibration', samples=1, max_candidates=3, max_program_runs=5)
    assert equal(initial, pack_state(components.state)), 'calibration must not commit any training update'
    document = plan(tmp_path / 'calibration/calibration.json', memory_limit_bytes=native_assets['planning_limit_bytes'])
    saved = tmp_path / 'plan.json'
    saved.write_text(json.dumps(document))
    receipt, session, _ = run(workload, saved, tmp_path / 'executed')
    assert receipt['reference_gate']['status'] == 'PASS'
    assert session.state['flow_compute_precision'] == 'bf16-mixed'
    assert all(session.state[role].compute_precision == 'bf16-mixed' for role in ('student', 'fake', 'teacher'))
    assert session.state['dmd_counters'].tolist() == [1, 1, 5]
    assert not torch.cuda.is_initialized()


def test_native_bf16_complete_checkpoint_resume_and_portable_export(native_assets, tmp_path):
    from test_training_end_to_end import run_cli, equal
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.inference import read_export, restore_modules

    config = tmp_path / 'amp.json'
    amp_config(native_assets, config)
    first = run_cli('train', config, '--cycles', 1, '--output', tmp_path / 'first')
    wrong_output = tmp_path / 'wrong-precision'
    refused = subprocess.run([sys.executable, '-m', 'distillschedule', 'train', str(native_assets['config']),
        '--cycles', '2', '--resume', first['checkpoint'], '--output', str(wrong_output)],
        capture_output=True, text=True, env={**os.environ, 'OMP_NUM_THREADS': '1'})
    assert refused.returncode != 0 and 'identity' in refused.stdout + refused.stderr
    assert not wrong_output.exists(), 'wrong-precision checkpoint must reject before output publication'
    resumed = run_cli('train', config, '--cycles', 2, '--resume', first['checkpoint'], '--output', tmp_path / 'resumed')
    whole = run_cli('train', config, '--cycles', 2, '--output', tmp_path / 'whole')
    a = torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(whole['checkpoint']) / 'state.pt', weights_only=True)
    assert equal(a, b), 'FP32 master/Adam/EMA, RNG, cursor and update clocks must survive fresh-process AMP resume'
    exported = run_cli('export', config, '--checkpoint', resumed['checkpoint'], '--output', tmp_path / 'export')
    assert exported['model']['parameters']['compute_precision'] == 'bf16-mixed'
    factory, _ = load_callable(exported['model']['factory'])
    current = factory(exported['model']['parameters'])
    root, manifest = read_export(tmp_path / 'export')
    restore_modules(current, root, manifest)
    assert current.state['student'].compute_precision == 'bf16-mixed'
    assert set(current.state) == {'student', 'media'}
    from native_flow_replicated import _portable_case
    output_kind, shape, decoded_shape, prompt = _portable_case(native_assets['family'])
    new_input = tmp_path / 'new-input.json'
    new_input.write_text(json.dumps(dict(schema_version='distillschedule.export_inputs.v1',
        weights=str(root), mode='prompts', records=[dict(id='unseen-amp-prompt',
            prompt=prompt, latent_shape=shape)])))
    inference = dict(schema_version='distillschedule.training.v1', model=dict(
        factory=exported['model']['factory'], parameters=exported['model']['parameters']),
        algorithm=dict(name='dmd'), data={}, execution=dict(mode='reference'),
        trainer=dict(cycles=1, seed=909, device='cpu', output_dir=str(tmp_path / 'unused')))
    hidden = native_assets['root'].with_name(native_assets['root'].name + '-hidden')
    native_assets['root'].rename(hidden)
    try:
        prepared = run_cli('data', 'prepare', new_input, '--output', tmp_path / 'new-prepared',
            '--preparer', 'distillschedule.data.export_inputs:prepare_inputs')['manifest']
        inference['data']['prepared'] = prepared
        (tmp_path / 'inference.json').write_text(json.dumps(inference))
        generated = run_cli('generate', tmp_path / 'inference.json', '--weights', root,
            '--prepared', prepared, '--output', tmp_path / 'generation', '--output-kind', output_kind)
    finally:
        hidden.rename(native_assets['root'])
    from distillschedule.data import PreparedDataset
    for module in current.state.values():
        module.eval()
    torch.random.default_generator.manual_seed(909)
    with torch.no_grad():
        expected = current.generate(current.state, PreparedDataset(prepared).sample(0), {})
    artifact = torch.load(generated['artifact'], weights_only=True)
    assert equal(artifact['generated'], expected)
    assert artifact['generated'][output_kind + '_latents'].dtype == torch.float32
    assert artifact['decoded'][output_kind].shape == decoded_shape
    if output_kind == 'image':
        assert Path(generated['image']['path']).is_file()
    else:
        from distillschedule.data.decoding import _video
        frames = artifact['decoded']['video']
        assert frames.dtype == torch.uint8
        assert torch.equal(_video(Path(generated['video']['frames'])), frames)
    assert not torch.cuda.is_initialized()
