"""Input capture tests; native replay uses a test boundary, not trained weights."""
from dataclasses import replace
import json
import random
from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
from distillschedule.strategies.wan_capture import input_stream, read_training_config
from distillschedule.strategies.wan_inputs import WanInputSources
from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.workloads.best_known.registry import repository_root


def test_rng_scope_restores_both_generators():
    before = torch.get_rng_state().clone(); python_before = random.getstate()
    with input_stream(17, 1, 'generator'):
        a = torch.rand(4); b = random.random()
    with input_stream(17, 1, 'generator'):
        assert torch.equal(a, torch.rand(4)) and b == random.random()
    with input_stream(17, 2, 'generator'):
        assert not torch.equal(a, torch.rand(4))
    assert torch.equal(before, torch.get_rng_state()) and python_before == random.getstate()


@pytest.fixture
def native():
    pytest.importorskip('diffusers'); pytest.importorskip('accelerate'); pytest.importorskip('ftfy')
    from distillschedule.strategies.wan_capture import native_primitives
    return native_primitives()


def make_sources(tmp_path, task):
    paths = []
    for rank in range(16):
        path = tmp_path / f'feature-{rank}.pt'
        torch.save({'vae_latent': torch.ones((50, *task.latent_shape[1:]))*(rank+1)/32,
                    'prompt_embed': torch.ones(2, 4096), 'attention_mask': torch.tensor([1, 0])}, path)
        paths.append(str(path))
    return WanInputSources(tuple(paths), seed=17)


def config_file(tmp_path):
    import yaml
    value = yaml.safe_load((repository_root()/'configs/distillschedule/templates/helios_stage3_384x640.yaml').read_text())
    # Test a DECLARED no-drop configuration; production must not silently disable dropout.
    value['training_config']['is_random_drop'] = False
    path = tmp_path/'training.yaml'; path.write_text(yaml.safe_dump(value))
    return path


@pytest.mark.parametrize('updates,width,frames,horizons', [(1,64,5,(1,2,3)),(2,128,9,(2,2,2))])
def test_native_capture_new_cohort_replays_original_schedule(tmp_path, native, updates, width, frames, horizons):
    from distillschedule.strategies.wan_capture import capture
    from distillschedule.strategies.wan_trace import load_original_trace
    task = OriginalWanTask(video_height=64, video_width=width, video_frames=frames,
                           fake_updates=updates, first_step=2*updates, stage2_inference_steps=horizons)
    source = make_sources(tmp_path, task)
    report = capture(task, source, config_file(tmp_path), tmp_path/'capture')
    trace = load_original_trace(task, report['trace_bundle'], expected_hash=report['expected_trace_hash'])
    assert trace.logical_steps == task.source_steps
    assert report['rank_validation']['cross_rank_call_count_parity']
    loaded = trace._load(0)
    assert loaded.tensor('sample/0/primary/target').shape == task.latent_shape[1:]
    prefixes = [f'step/{task.first_step}/generator', f'step/{task.first_step}/critic']
    for prefix in prefixes:
        calls = []
        def boundary(**kw):
            calls.append((list(kw['hidden_states'].shape), kw['timestep'].detach().cpu().tolist()))
            return (kw['hidden_states'] * .1,)
        history_names = ('indices_hidden_states','indices_history_short','indices_history_mid','indices_history_long',
                         'history_short','history_mid','history_long','continuous_history')
        entry = loaded.tensor(prefix+'/entry_noise')
        gt = (entry, *(loaded.tensor(prefix+'/'+k) for k in history_names))
        native.post.run_generator(args=SimpleNamespace(training_config=SimpleNamespace(
            base_seq_len=256,max_seq_len=4096,base_shift=.5,max_shift=1.15)),
            accelerator=SimpleNamespace(device='cpu',is_main_process=True), transformer=boundary,
            scheduler=native.scheduler(shift=1.,stages=3,stage_range=[0,1/3,2/3,1],gamma=1/3),
            noise=entry,prompt_embeds=loaded.tensor('sample/0/text_embedding').unsqueeze(0),
            is_enable_stage2=True,stage2_num_stages=3,stage2_num_inference_steps_list=list(horizons),
            sigmas=torch.linspace(1.,.001,1000,dtype=torch.float64),timesteps=torch.linspace(1000.,1.,1000,dtype=torch.float64),
            is_keep_x0=True,history_sizes=[16,2,1],num_critic_input_frames=task.latent_shape[2],
            num_rollout_sections=1,is_use_gt_history=True,gt_all_data=gt,is_amplify_first_chunk=True,
            use_dynamic_shifting=True,time_shift_type='linear',use_kv_cache=False,
            operation_trace=loaded,trace_prefix=prefix)
        expected = [(stage['latent_shape'], [t]) for stage in loaded.operation(prefix+'/rollout')['control']['pyramid_stage_call_sequence']
                    for t in stage['timestep']]
        assert calls == expected
    assert report['gpu_execution'] == 'NOT_RUN' and not report['claimable_speedup']
    with pytest.raises(FileExistsError):
        capture(task, source, config_file(tmp_path), tmp_path/'capture')


def test_amplified_zero_history_is_not_resampled(tmp_path, native):
    from distillschedule.strategies.wan_capture import prepare_history
    task = OriginalWanTask(video_height=64,video_width=64,video_frames=5)
    cfg = read_training_config(config_file(tmp_path))
    sample = (torch.zeros(16,1,8,8),torch.zeros(16,19,8,8),torch.zeros(16,2,8,8))
    with input_stream(17, 'zero'):
        with pytest.raises(ValueError, match='all-zero'):
            prepare_history(native, task, sample, cfg)


def test_config_cannot_silently_change_loss_or_optimizer(tmp_path):
    import yaml
    path = config_file(tmp_path)
    assert read_training_config(path)['is_random_drop'] is False
    value = yaml.safe_load(path.read_text())
    value['training_config']['learning_rate'] *= 10
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match='non-input training changes'):
        read_training_config(path)


def test_gt_history_does_not_apply_non_gt_saturation(tmp_path, native, monkeypatch):
    from distillschedule.strategies.wan_capture import prepare_history
    task = OriginalWanTask(video_height=64, video_width=64, video_frames=5)
    cfg = read_training_config(config_file(tmp_path))
    assert cfg['is_add_saturation'] is True
    seen = []
    def forbidden(*args, **kwargs):
        seen.append(True)
        raise AssertionError('native GT-history must not apply non-GT saturation')
    monkeypatch.setattr(native.history, 'add_saturation_to_history_latents', forbidden)
    sample = (torch.ones(16,1,8,8),torch.ones(16,19,8,8),torch.ones(16,2,8,8))
    with input_stream(17, 'native-gt'):
        result = prepare_history(native, task, sample, cfg)
    assert not seen and result['history_short'].shape == (1,16,2,8,8)
