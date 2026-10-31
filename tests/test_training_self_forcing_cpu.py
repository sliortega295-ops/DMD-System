"""Opt-in actual pinned Self-Forcing CPU acceptance (no model substitutes).

Missing optional source skips broad CI; SELF_FORCING_SOURCE_ROOT makes missing
dependencies or native incompatibilities errors, never acceptance skips.
"""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import torch

from distillschedule.training.checkpoint import cpu_tree, rng_state, restore_rng
from test_training_end_to_end import equal, run_cli

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def source_root():
    value = os.environ.get('SELF_FORCING_SOURCE_ROOT')
    if not value:
        pytest.skip('optional official source: set SELF_FORCING_SOURCE_ROOT')
    return Path(value).resolve(strict=True)


@pytest.fixture
def fixture_assets(tmp_path, source_root):
    from distillschedule.training.plugins import load_callable
    builder, _ = load_callable(str(REPO / 'recipes/self_forcing_official/create_cpu_fixture.py') + ':build_fixture')
    return builder(tmp_path / 'actual-native', source_root, cycles=3)


def model_and_data(files):
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import construct
    config = load_training_config(files['config'])
    components, data, _, _ = construct(config)
    return config, components, data


def test_actual_native_classes_shapes_and_no_livewan(fixture_assets, source_root):
    from distillschedule.training.self_forcing import original
    api = original(source_root)
    _, components, data = model_and_data(fixture_assets)
    owner = components.state['native_cycle']
    assert isinstance(owner.model, api.dmd.DMD)
    assert type(owner.model.generator.model) is api.causal.CausalWanModel
    assert type(owner.model.real_score.model) is api.wan.WanModel
    assert type(owner.model.fake_score.model) is api.wan.WanModel
    assert type(owner.model.vae.model) is api.vae.WanVAE_
    assert owner.args.image_or_video_shape == [1, 21, 16, 4, 4]
    assert components.prediction.kind == 'flow'
    assert set(data.cycle_sample(0)) == {'generator', 'critic'}
    assert len(data.cycle_sample(0)['critic']) == 5
    assert owner.model.generator_loss.__func__ is api.dmd.DMD.generator_loss
    assert owner.model.critic_loss.__func__ is api.dmd.DMD.critic_loss
    assert owner.model._run_generator.__func__ is api.base.SelfForcingModel._run_generator


def test_native_training_changes_generator_critic_not_teacher_and_strict_restore(fixture_assets):
    config, components, data = model_and_data(fixture_assets)
    owner = components.state['native_cycle']
    before = cpu_tree(owner.model.state_dict())
    outer_rng = rng_state()
    result = owner.run(data.cycle_sample(0), cycle=0)
    assert equal(outer_rng, rng_state())
    assert owner.native_step == 5 and owner.generator_updates == 1 and owner.critic_updates == 5
    assert result['consumed'] == {'generator': 1, 'critic': 5}
    assert [row['generator_turn'] for row in result['iterations']] == [True, False, False, False, False]
    assert all(torch.isfinite(torch.tensor(list(row['losses'].values()))).all() for row in result['iterations'])
    assert not equal(before['generator.model.head.head.weight'], owner.model.state_dict()['generator.model.head.head.weight'])
    assert not equal(before['fake_score.model.head.head.weight'], owner.model.state_dict()['fake_score.model.head.head.weight'])
    assert all(equal(value, owner.model.state_dict()[name]) for name, value in before.items() if name.startswith(('real_score.', 'vae.')))
    assert all(p.grad is None for p in owner.model.parameters())
    _, other, _ = model_and_data(fixture_assets)
    other.state['native_cycle'].load_state_dict(deepcopy(owner.state_dict()))
    assert equal(owner.state_dict(), other.state['native_cycle'].state_dict())


@pytest.mark.parametrize('bad_field', ['clock', 'teacher', 'ownership', 'moment', 'rng', 'ema', 'condition',
                                      'moment_missing', 'moment_dtype', 'mask', 'mode', 'plain', 'source_identity'])
def test_checkpoint_rejection_is_transactional(fixture_assets, bad_field):
    _, components, data = model_and_data(fixture_assets)
    owner = components.state['native_cycle']; owner.run(data.cycle_sample(0), cycle=0)
    before = cpu_tree(owner.state_dict()); bad = deepcopy(before)
    if bad_field == 'clock':
        bad['native_step'] = 4
    elif bad_field == 'teacher':
        name = next(name for name in bad['model'] if name.startswith('real_score.'))
        bad['model'][name].add_(1)
    elif bad_field == 'ownership':
        bad['optimizer_ownership']['critic'][0][0] = 'real_score.teacher'
    elif bad_field == 'moment':
        next(iter(bad['optimizers']['critic']['state'].values()))['exp_avg'] = torch.zeros(1)
    elif bad_field == 'rng':
        bad['rng'] = {}
    elif bad_field == 'ema':
        bad['ema_updates'] += 1
    elif bad_field == 'condition':
        bad['unconditional_dict']['prompt_embeds'].add_(1)
    elif bad_field == 'moment_missing':
        bad['optimizers']['critic']['state'].pop(next(iter(bad['optimizers']['critic']['state'])))
    elif bad_field == 'moment_dtype':
        slot = next(iter(bad['optimizers']['critic']['state'].values()))
        slot['exp_avg'] = slot['exp_avg'].double()
    elif bad_field == 'mask':
        name = next(name for name in bad['requires_grad'] if name.startswith('real_score.'))
        bad['requires_grad'][name] = True
    elif bad_field == 'mode':
        bad['training']['generator'] = True
    elif bad_field == 'plain':
        bad['plain']['generator.model.freqs'].add_(1)
    elif bad_field == 'source_identity':
        bad['identity']['upstream']['revision'] = '0' * 40
    with pytest.raises(ValueError):
        owner.load_state_dict(bad)
    assert equal(before, owner.state_dict())
    with pytest.raises(ValueError, match='outer Program'):
        owner.validate_checkpoint_progress(before, dict(cycle=1, cursor=2, versions={'native_cycle': 1}))


def test_fresh_process_common_program_restart_current_g_export(fixture_assets, tmp_path):
    config = Path(fixture_assets['config'])
    partial = run_cli('train', config, '--cycles', 1, '--output', tmp_path / 'partial')
    resumed = run_cli('train', config, '--cycles', 3, '--resume', partial['checkpoint'], '--output', tmp_path / 'resumed')
    full = run_cli('train', config, '--cycles', 3, '--output', tmp_path / 'full')
    assert equal(torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True),
                 torch.load(Path(full['checkpoint']) / 'state.pt', weights_only=True))
    exported = run_cli('export', config, '--checkpoint', resumed['checkpoint'], '--output', tmp_path / 'export')
    assert exported['model']['factory'] == 'distillschedule.training.self_forcing:build_inference_model'
    assert set(exported['model']['modules']) == {'generator', 'vae'}
    weights = torch.load(tmp_path / 'export/weights.pt', weights_only=True)
    assert set(weights) == {'generator', 'vae'}
    assert 'assets' not in exported['model']['parameters']
    # New native condition, never replay a saved training batch or saved sample.
    from distillschedule.data import PreparedDataset
    from distillschedule.data.self_forcing_inputs import prepare_inputs, SCHEMA
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import _load_for_output
    from distillschedule.workloads.best_known.registry import file_sha256
    fresh = tmp_path / 'fresh-inputs'; fresh.mkdir()
    tensor = fresh / 'new-embedding.pt'
    torch.save(torch.linspace(-.8, .9, 128).reshape(1, 8, 16), tensor)
    manifest = fresh / 'new-inputs.json'
    manifest.write_text(json.dumps(dict(schema_version=SCHEMA, mode='preencoded', records=[dict(
        id='brand-new-native-condition', embedding=dict(path=tensor.name, sha256=file_sha256(tensor)))])))
    prepared = prepare_inputs(manifest, fresh / 'prepared')
    trained = _load_for_output(load_training_config(config), resumed['checkpoint'])[0]
    torch.manual_seed(909)
    inference_trace, inference_hook = _rollout_trace(trained.state['native_cycle'])
    with torch.no_grad():
        expected = trained.generate(trained.state, PreparedDataset(prepared).sample(0), {})
        expected = trained.decode(expected, {})
    inference_hook.remove()
    # Full native inference: four denoising calls AND the clean context write
    # for every block, not the random training exit or a replayed latent.
    from distillschedule.training.self_forcing import original
    api = original(trained.state['native_cycle'].source.root)
    assert api.InferencePipeline.inference is api.inference.CausalInferencePipeline.inference
    assert len(inference_trace) == 7 * 5
    assert all(not row['grad_enabled'] for row in inference_trace)
    assert expected['video'].shape == (81, 32, 32, 3)
    inference = dict(schema_version='distillschedule.training.v1',
        model=dict(factory=exported['model']['factory'], parameters=exported['model']['parameters']),
        algorithm=dict(factory='distillschedule.training.self_forcing:build_self_forcing'),
        data=dict(prepared=str(prepared)), execution=dict(adapter='program', mode='reference'),
        trainer=dict(cycles=1, seed=909, device='cpu', output_dir=str(tmp_path / 'unused')))
    inference_path = tmp_path / 'inference.json'; inference_path.write_text(json.dumps(inference))
    import shutil
    shutil.rmtree(fixture_assets['root'])
    for directory in ('partial', 'resumed', 'full'):
        shutil.rmtree(tmp_path / directory)
    generated = run_cli('generate', inference_path, '--weights', tmp_path / 'export', '--prepared', prepared,
                        '--output', tmp_path / 'new-video', '--output-kind', 'video')
    actual = torch.load(generated['artifact'], weights_only=True)
    assert equal(actual['decoded'], expected)
    assert actual['decoded']['video'].dtype == torch.uint8


def _native_trainer_oracle(reference, cycles):
    """Execute the SOURCE's actual Trainer.train, not owner.run copied math."""
    from types import MethodType
    from distillschedule.training.self_forcing import original
    api = original(reference.source.root)
    trainer = api.trainer.Trainer.__new__(api.trainer.Trainer)
    trainer.model = reference.model
    trainer.config = deepcopy(reference.config)
    trainer.config.no_save = True
    trainer.config.gc_interval = 10**9
    trainer.step = 0
    trainer.is_main_process = False
    trainer.disable_wandb = True
    trainer.device, trainer.dtype = reference.device, reference.dtype
    trainer.max_grad_norm_generator = reference.max_grad_norm_generator
    trainer.max_grad_norm_critic = reference.max_grad_norm_critic
    trainer.generator_optimizer = reference.optimizers['generator']
    trainer.critic_optimizer = reference.optimizers['critic']
    trainer.generator_ema = deepcopy(reference.generator_ema)
    trainer.unconditional_dict = None
    # The real trainer uses an FSDP wrapper's `.module` spelling. This
    # nonregistered self-alias supplies only that CPU1 shell; native model
    # forward/backward/state_dict and ORIGINAL EMA_FSDP methods are unchanged.
    object.__setattr__(trainer.model.generator, 'module', trainer.model.generator)
    batches = []
    for feeds in cycles:
        for row in [feeds['generator'][0], *feeds['critic']]:
            batches.append({'prompts': list(row['prompt_embeds'].unbind(0))})
    trainer.dataloader = iter(batches)
    events = []
    native = api.trainer.Trainer.fwdbwd_one_step

    def capture(self, batch, train_generator):
        result = native(self, batch, train_generator)
        role = 'generator' if train_generator else 'critic'
        module = self.model.generator if train_generator else self.model.fake_score
        events.append((role + '_gradients', cpu_tree({name: p.grad for name, p in module.named_parameters()})))
        pipeline = self.model.inference_pipeline
        events.append((role + '_rollout_cache', cpu_tree(dict(kv=pipeline.kv_cache1, crossattn=pipeline.crossattn_cache))))
        events.append(('loss', {key: float(value.detach()) for key, value in result.items()
                               if key in ('generator_loss', 'critic_loss', 'generator_grad_norm', 'critic_grad_norm')}))
        return result

    trainer.fwdbwd_one_step = MethodType(capture, trainer)
    restore_rng(reference._rng)
    with pytest.raises(StopIteration):
        trainer.train()
    assert trainer.step == len(cycles) * reference.ratio
    return trainer, events, cpu_tree(rng_state())


def _rollout_trace(owner):
    records = []
    def capture(module, args, kwargs):
        records.append(dict(noise=cpu_tree(kwargs['noisy_image_or_video']),
            condition=cpu_tree(kwargs['conditional_dict']), timestep=cpu_tree(kwargs['timestep']),
            current_start=kwargs['current_start'], grad_enabled=torch.is_grad_enabled()))
    hook = owner.model.generator.register_forward_pre_hook(capture, with_kwargs=True)
    return records, hook


def test_source_trainer_oracle_loss_gradient_adam_ema_cache_rng_and_native_clocks(fixture_assets):
    _, components, data = model_and_data(fixture_assets)
    owner = components.state['native_cycle']; reference = deepcopy(owner)
    cycles = [data.cycle_sample(index) for index in range(2)]
    outer_rng = rng_state()
    expected_trace, expected_hook = _rollout_trace(reference)
    trainer, expected_events, expected_rng = _native_trainer_oracle(reference, cycles)
    expected_hook.remove(); restore_rng(outer_rng)
    events = []
    owner.observer = lambda name, value: events.append((name, cpu_tree(value)))
    actual_trace, actual_hook = _rollout_trace(owner)
    results = [owner.run(feeds, cycle=index) for index, feeds in enumerate(cycles)]
    actual_hook.remove()
    assert equal(rng_state(), outer_rng)
    assert equal(owner._rng, expected_rng)
    assert equal(actual_trace, expected_trace), 'native prefix/exit/context draw order or independent C rollout changed'
    assert any(row['grad_enabled'] for row in actual_trace)
    assert any(not row['grad_enabled'] and int(row['timestep'][0, 0]) > 0 for row in actual_trace)
    assert sorted(set(row['current_start'] for row in actual_trace)) == list(range(0, 84, 12))
    filtered = [(name, value) for name, value in events if name.endswith(('_gradients', '_rollout_cache'))]
    assert equal(filtered, [(name, value) for name, value in expected_events if name != 'loss'])
    losses = []
    for result in results:
        for row in result['iterations']:
            if row['generator_turn']:
                losses.append(dict(generator_loss=row['losses']['generator_loss'], generator_grad_norm=row['generator_grad_norm']))
            losses.append(dict(critic_loss=row['losses']['critic_loss'], critic_grad_norm=row['critic_grad_norm']))
    assert losses == [value for name, value in expected_events if name == 'loss']
    assert equal(owner.model.state_dict(), reference.model.state_dict())
    assert equal({key: value.state_dict() for key, value in owner.optimizers.items()},
                 {key: value.state_dict() for key, value in reference.optimizers.items()})
    assert equal(owner.generator_ema.state_dict(), trainer.generator_ema.state_dict())
    assert equal(owner.unconditional_dict, trainer.unconditional_dict)
    assert owner.native_step == 10 and owner.generator_updates == 2 and owner.critic_updates == 10
    assert owner.ema_initialized_step == 5 and owner.ema_updates == 1
    # G/C have different input records AND G->C independently samples noise.
    first_g = actual_trace[0]
    first_c = next(row for row in actual_trace if not equal(row['condition'], first_g['condition']))
    assert not torch.equal(first_g['noise'], first_c['noise'])
    assert not torch.equal(first_g['condition']['prompt_embeds'], first_c['condition']['prompt_embeds'])


@pytest.mark.parametrize('start,weight', [(0, .9), (1, .9), (5, .9), (6, .9), (200, .9), (0, 0.)])
def test_native_ema_start_including_zero_and_post_increment_transition(fixture_assets, start, weight):
    path = Path(fixture_assets['config'])
    value = json.loads(path.read_text())
    value['model']['parameters']['settings'].update(ema_start_step=start, ema_weight=weight)
    path.write_text(json.dumps(value))
    _, components, data = model_and_data(fixture_assets)
    owner = components.state['native_cycle']; reference = deepcopy(owner)
    cycles = [data.cycle_sample(index) for index in range(2)]
    outer = rng_state()
    trainer, _, expected_rng = _native_trainer_oracle(reference, cycles)
    restore_rng(outer)
    owner.run(cycles[0], cycle=0)
    if weight > 0 and start <= 5:
        assert owner.generator_ema is not None and owner.ema_initialized_step == start
        assert owner.ema_updates == int(start == 0)
    else:
        assert owner.generator_ema is None
    owner.run(cycles[1], cycle=1)
    expected_updates = sum(int(weight > 0 and step >= start) for step in (0, 5))
    assert owner.ema_updates == expected_updates
    assert equal(owner._rng, expected_rng)
    assert equal(None if owner.generator_ema is None else owner.generator_ema.state_dict(),
                 None if trainer.generator_ema is None else trainer.generator_ema.state_dict())
    saved = cpu_tree(owner.state_dict())
    _, restored, _ = model_and_data(fixture_assets)
    restored.state['native_cycle'].load_state_dict(saved)
    assert equal(saved, restored.state['native_cycle'].state_dict())


def test_pinned_original_recipe_defaults_distinguish_both_shifts(fixture_assets):
    from distillschedule.training.self_forcing import _arguments, official_recipe_settings
    value = json.loads(Path(fixture_assets['config']).read_text())['model']['parameters']
    value.pop('settings')
    args = _arguments(value)
    source = official_recipe_settings(value['source_root'])
    assert args.timestep_shift == args.scheduler_shift == source['model_kwargs']['timestep_shift'] == 5.
    assert args.ts_schedule is False and args.lr == 2e-6 and args.lr_critic == 4e-7
    assert args.ema_start_step == 200 and args.dfake_gen_update_ratio == 5
    assert args.num_frame_per_block == 3 and args.image_or_video_shape == [1, 21, 16, 60, 104]
    assert args.mixed_precision is False and args.gradient_checkpointing is False
    assert source['mixed_precision'] is True and source['gradient_checkpointing'] is True


@pytest.mark.parametrize('setting,value', [('num_training_frames', 24), ('independent_first_frame', True),
    ('gradient_checkpointing', 1), ('i2v', True), ('mixed_precision', True), ('num_frame_per_block', 4),
    ('denoising_loss_type', 'noise'), ('image_or_video_shape', [1, 3, 16, 4, 4]), ('denoising_step_list', [1000, 0])])
def test_unsupported_native_scope_rejects_before_model_or_training(fixture_assets, setting, value):
    from distillschedule.training.self_forcing import _arguments
    parameters = json.loads(Path(fixture_assets['config']).read_text())['model']['parameters']
    parameters['settings'][setting] = value
    with pytest.raises(ValueError):
        _arguments(parameters)


def test_common_program_builder_does_not_presample_native_rng_and_stream_admission(fixture_assets):
    from distillschedule.training.session import build_cycle
    from distillschedule.training.config import load_training_config
    from distillschedule.training.self_forcing import build_self_forcing
    config, components, data = model_and_data(fixture_assets)
    owner = components.state['native_cycle']
    before = cpu_tree(owner._rng); outer = cpu_tree(rng_state())
    workload = build_cycle(config, components, data, build_self_forcing, cycle=0, cursor=0)
    assert workload.program.calls[0].grad == 'external_grad'
    assert equal(owner._rng, before) and equal(rng_state(), outer)
    assert not any(value.is_cuda for value in owner.runtime_tensors())
    value = json.loads(Path(fixture_assets['config']).read_text())
    value['data']['streams']['critic']['samples_per_cycle'] = 4
    invalid = Path(fixture_assets['root']) / 'bad-stream.json'; invalid.write_text(json.dumps(value))
    from distillschedule.training.session import construct
    with pytest.raises(ValueError, match='prompt draws'):
        construct(load_training_config(invalid))


@pytest.mark.parametrize('settings', [dict(same_step_across_blocks=False, context_noise=12),
    dict(last_step_only=True, fake_guidance_scale=1.2, ts_schedule=True, ts_schedule_max=True)])
def test_native_exit_context_noise_fake_cfg_and_timestep_schedule_variants(fixture_assets, settings):
    path = Path(fixture_assets['config'])
    value = json.loads(path.read_text()); value['model']['parameters']['settings'].update(settings)
    path.write_text(json.dumps(value))
    _, components, data = model_and_data(fixture_assets)
    owner = components.state['native_cycle']; reference = deepcopy(owner)
    outer = rng_state()
    trainer, _, expected_rng = _native_trainer_oracle(reference, [data.cycle_sample(0)])
    restore_rng(outer)
    score_calls = []
    def capture(role):
        def observe(module, args, kwargs):
            score_calls.append(dict(role=role, condition=cpu_tree(kwargs['conditional_dict']),
                noise=cpu_tree(kwargs['noisy_image_or_video']), timestep=cpu_tree(kwargs['timestep']),
                grad_enabled=torch.is_grad_enabled()))
        return observe
    hooks = [module.register_forward_pre_hook(capture(role), with_kwargs=True)
             for role, module in [('fake', owner.model.fake_score), ('real', owner.model.real_score)]]
    owner.run(data.cycle_sample(0), cycle=0)
    for hook in hooks:
        hook.remove()
    assert equal(owner.model.state_dict(), reference.model.state_dict())
    assert equal(owner._rng, expected_rng)
    assert equal(owner.optimizers['critic'].state_dict(), reference.optimizers['critic'].state_dict())
    fake_count = 2 if settings.get('fake_guidance_scale', 0.) else 1
    assert [row['role'] for row in score_calls] == ['fake'] * fake_count + ['real', 'real'] + ['fake'] * 5
    assert all(not row['grad_enabled'] for row in score_calls[:fake_count + 2])
    assert all(row['grad_enabled'] for row in score_calls[fake_count + 2:])
    assert torch.equal(score_calls[fake_count]['noise'], score_calls[fake_count + 1]['noise'])
    assert torch.equal(score_calls[fake_count + 1]['condition']['prompt_embeds'], owner.negative_embedding)


def test_native_local_kv_roll_and_sink_preserve_original_cache_body(fixture_assets):
    path = Path(fixture_assets['assets']) / 'assets.json'
    value = json.loads(path.read_text())
    value['generator']['config'].update(local_attn_size=6, sink_size=3)
    path.write_text(json.dumps(value))
    _, components, data = model_and_data(fixture_assets)
    owner = components.state['native_cycle']; reference = deepcopy(owner)
    outer = rng_state()
    trainer, expected, expected_rng = _native_trainer_oracle(reference, [data.cycle_sample(0)])
    restore_rng(outer)
    observed = []
    owner.observer = lambda name, value: observed.append((name, cpu_tree(value)))
    owner.run(data.cycle_sample(0), cycle=0)
    expected_caches = [(name, value) for name, value in expected if name.endswith('_rollout_cache')]
    actual_caches = [(name, value) for name, value in observed if name.endswith('_rollout_cache')]
    assert equal(expected_caches, actual_caches)
    assert equal(owner._rng, expected_rng)
    assert equal(owner.model.state_dict(), reference.model.state_dict())
    assert all(value['kv'][0]['k'].shape == (1, 24, 2, 16) for _, value in actual_caches)
    assert all(value['kv'][0]['global_end_index'].item() == 84
               and value['kv'][0]['local_end_index'].item() == 24 for _, value in actual_caches)
