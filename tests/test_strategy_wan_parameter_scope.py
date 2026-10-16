"""Explicit historical benchmark branch scope; no sampling, model or CUDA mocks."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from distillschedule.strategies.wan_task import OriginalWanTask


def test_historical_global_scope_is_explicit_and_task_qualified():
    task = OriginalWanTask.from_dict({'stage_schedule_policy': 'benchmark_v4_strict_global'})
    assert task.stage_schedule_policy == 'benchmark_v4_strict_global'
    assert task.digest != OriginalWanTask().digest
    assert task.to_dict()['stage_schedule_policy'] == 'benchmark_v4_strict_global'
    assert 'stage_schedule_policy' not in OriginalWanTask().to_dict()
    with pytest.raises(ValueError, match='schedule policy'):
        OriginalWanTask.from_dict({'stage_schedule_policy': 'silently_drop_history'})


def test_default_capture_still_rejects_zero_history(native):
    from distillschedule.strategies.wan_capture import prepare_history
    from distillschedule.strategies.wan_capture import read_training_config
    from distillschedule.workloads.best_known.registry import repository_root
    torch = pytest.importorskip('torch')
    task = OriginalWanTask(video_height=64, video_width=64, video_frames=5)
    cfg = read_training_config(repository_root()/'configs/distillschedule/templates/helios_stage3_384x640.yaml')
    cfg = dict(cfg, is_random_drop=False)
    sample = (torch.zeros(16,1,8,8),torch.zeros(16,19,8,8),torch.zeros(16,2,8,8))
    with pytest.raises(ValueError, match='all-zero'):
        prepare_history(native, task, sample, cfg)
    scoped = replace(task, stage_schedule_policy='benchmark_v4_strict_global')
    output = prepare_history(native, scoped, sample, cfg)
    assert all(torch.count_nonzero(output[k]) == 0 for k in ('history_short','history_mid','history_long'))


@pytest.fixture
def native():
    pytest.importorskip('torch'); pytest.importorskip('diffusers'); pytest.importorskip('accelerate'); pytest.importorskip('ftfy')
    from distillschedule.strategies.wan_capture import native_primitives
    return native_primitives()


def test_native_schedule_replay_budget_matches_all_zero_history(native):
    from distillschedule.strategies.wan_capture import stage_control
    task = OriginalWanTask(stage_schedule_policy='benchmark_v4_strict_global')
    normal = stage_control(native, task, 49, 0, 10, 'generator', all_history_zero=False)
    zero = stage_control(native, task, 49, 0, 10, 'generator', all_history_zero=True)
    assert normal['initial_synchronized_pyramid_stage_exit'] == zero['initial_synchronized_pyramid_stage_exit']
    assert normal['call_count'] == zero['call_count']
    assert [s['denoising_step_budget'] for s in zero['pyramid_stage_call_sequence']] == [4,4,4]
    assert [s['denoising_step_budget'] for s in normal['pyramid_stage_call_sequence']] == [2,2,2]
    assert zero['all_history_zero'] is True
    assert zero['stage_schedule_policy'] == 'benchmark_v4_strict_global'
    assert 'local_exception_seed' not in zero
