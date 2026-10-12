"""Original protocol parameterization; CPU contract tests, not model/GPU timing."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from distillschedule.strategies.wan_task import OriginalWanTask, rollout_geometry, TRACE_KEY


def test_default_matches_original_geometry():
    t=OriginalWanTask()
    assert t.latent_shape == (1,16,9,48,80)
    assert t.source_steps == (10,11,12,13,14)
    assert t.ema_after == (0,1,2,3,4)
    assert t.stage_spatial_shapes == ((12,20),(24,40),(48,80))
    assert OriginalWanTask.from_dict(t.to_dict()) == t


@pytest.mark.parametrize('n',[1,2,3,5,8,64])
def test_ratio_is_a_real_protocol_parameter(n):
    t=OriginalWanTask(fake_updates=n, first_step=2*n)
    assert len(t.source_steps)==n and t.ema_after==tuple(range(n))
    assert t.trace_contract()['generator_source_step']==2*n


@pytest.mark.parametrize('h,w,f',[(256,384,17),(512,768,49),(64,128,1)])
def test_shape_propagation(h,w,f):
    t=OriginalWanTask(video_height=h,video_width=w,video_frames=f)
    assert t.latent_shape == (1,16,(f-1)//4+1,h//8,w//8)
    assert rollout_geometry(SimpleNamespace(manifest={'contract':{TRACE_KEY:t.to_dict()}}))==t


@pytest.mark.parametrize('changes',[{'world_size':8},{'physical_batch':2},{'fake_updates':3},
    {'lora_rank':True},{'video_height':400},{'video_frames':32},{'ema_every':-1},
    {'denoising_step_list':(1000,500,500)},{'denoising_step_list':(500,250)},
    {'stage2_inference_steps':(2,2)},{'model_family':'h3'},{'schema_version':'euler'}])
def test_unsupported_contract_not_ignored(changes):
    with pytest.raises(ValueError): replace(OriginalWanTask(),**changes)


def test_no_unknown_parameters_or_legacy_drift():
    with pytest.raises(ValueError): OriginalWanTask.from_dict({'gan':True})
    assert rollout_geometry(SimpleNamespace(manifest={'contract':{}})) == OriginalWanTask()
    assert replace(OriginalWanTask(),ema_every=0).ema_after==()
    assert replace(OriginalWanTask(),ema_every=5).ema_after==(4,)


def test_pyramid_horizon_not_ignored_ordinary_list():
    t=OriginalWanTask(stage2_inference_steps=(4,8,4))
    assert OriginalWanTask.from_dict(t.to_dict()).stage2_inference_steps==(4,8,4)
    with pytest.raises(ValueError,match='active pyramid'):
        OriginalWanTask(denoising_step_list=(1000,500))
