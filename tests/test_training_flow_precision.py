"""Explicit native AMP compute, preserving FP32 flow math and owned state."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize('precision', ['', 'bf16', 'float16', True, None, []])
def test_flow_compute_precision_rejects_unknown_values_without_loading_models(precision):
    from distillschedule.training.catalog import describe, validate_compatibility
    for name in ('flux2-klein-base', 'qwen-image'):
        factory = describe()['models'][name]['factory']
        with pytest.raises(ValueError, match='compute precision'):
            validate_compatibility(dict(factory=factory, parameters={'compute_precision': precision}),
                dict(factory='distillschedule.training.dmd:build_algorithm'), {'device': 'cpu'})
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


@pytest.fixture(scope='module', params=['flux2-klein-base', 'qwen-image'])
def native_assets(request, tmp_path_factory):
    if os.environ.get('DMD_MODERN_FLOW_CPU') != '1':
        pytest.skip('isolated modern-flow lane required for actual upstream model classes')
    from distillschedule.training.plugins import load_callable
    create, _ = load_callable(str(Path(__file__).resolve().parents[1] /
        'recipes/flow_image/create_cpu_fixture.py') + ':build_fixture')
    return create(tmp_path_factory.mktemp('amp-' + request.param) / 'fixture', request.param)


def amp_config(assets, path):
    value = json.loads(assets['config'].read_text())
    value['model']['parameters']['compute_precision'] = 'bf16-mixed'
    path.write_text(json.dumps(value))
    from distillschedule.training.config import load_training_config
    return load_training_config(path)


def test_native_bf16_forward_gradient_updates_and_immutable_sharing(native_assets, tmp_path):
    from distillschedule.training.session import construct, build_cycle
    from distillschedule.open_workload.torch_backend import Session
    from distillschedule.training.checkpoint import pack_state
    from test_training_end_to_end import equal

    config = amp_config(native_assets, tmp_path / 'amp.json')
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
        complex_before = {name: value.clone() for name, value in role.native.named_buffers() if value.is_complex()}
        if native_assets['family'] == 'qwen-image':
            assert complex_before, 'the real Qwen complex rotary tensors must remain owned and intact'
        row = dataset.sample(0)
        sigma = torch.tensor([.625])
        linear_calls = []
        handles = [module.register_forward_hook(lambda _, __, out:
                   linear_calls.append((torch.is_autocast_enabled('cpu'), out.dtype)))
                   for module in role.native.modules() if type(module) is torch.nn.Linear]
        actual = role.denoise(row['latents'], sigma, row, conditional=True)['image']
        assert actual.dtype == torch.float32 and actual.requires_grad
        assert linear_calls and all(enabled and dtype == torch.bfloat16 for enabled, dtype in linear_calls)
        assert not torch.is_autocast_enabled('cpu')
        with torch.autocast('cpu', dtype=torch.bfloat16):
            if native_assets['family'] == 'flux2-klein-base':
                expected = role.native(hidden_states=row['latents']['image'], encoder_hidden_states=row['context'],
                    timestep=sigma, img_ids=row['img_ids'], txt_ids=row['txt_ids'], guidance=None, return_dict=False)[0]
            else:
                expected = role.native(hidden_states=row['latents']['image'], encoder_hidden_states=row['context'],
                    encoder_hidden_states_mask=row['context_mask'], timestep=sigma,
                    img_shapes=[[tuple(value)] for value in row['img_shapes'].tolist()], guidance=None, return_dict=False)[0]
        assert torch.equal(actual, expected.float()), 'the adapter must use the actual native AMP forward unchanged'
        actual.square().mean().backward()
        assert any(p.grad is not None and p.grad.count_nonzero() for p in role.parameters() if p.requires_grad)
        assert all(p.grad is None or p.grad.dtype == torch.float32 for p in role.parameters())
        for handle in handles:
            handle.remove()
        for r in ('student', 'fake'):
            components.state[r + '_optimizer'].zero_grad(set_to_none=True)
        frozen = deepcopy(components.state['teacher'].state_dict())
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
            assert slots and all(float(slot['step']) == steps and slot['exp_avg'].dtype == torch.float32 for slot in slots)
        assert not torch.is_autocast_enabled('cpu')
        runs.append({'output': {key: value.detach().clone() for key, value in output.items()},
                     'state': deepcopy(pack_state(components.state))})
    assert equal(*runs), 'AMP must retain exact reference/manual-sharing state and outputs'
    assert not torch.cuda.is_initialized()


def test_native_precision_changes_actual_workload_and_rejects_control_request(native_assets, tmp_path):
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
    document = plan(tmp_path / 'calibration/calibration.json', memory_limit_bytes=100_000_000)
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
    new_input = tmp_path / 'new-input.json'
    new_input.write_text(json.dumps(dict(schema_version='distillschedule.export_inputs.v1',
        weights=str(root), mode='prompts', records=[dict(id='unseen-amp-prompt',
            prompt='a blue boat', latent_shape=[1, 4, 4, 4])])))
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
            '--prepared', prepared, '--output', tmp_path / 'generation', '--output-kind', 'image')
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
    assert artifact['generated']['image_latents'].dtype == torch.float32
    assert Path(generated['image']['path']).is_file()
    assert not torch.cuda.is_initialized()
