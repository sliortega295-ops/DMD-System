from dataclasses import replace
import pytest
from distillschedule.strategies.wan_spec import WanTask,WanStrategy,strategies_for


def test_original_optimized_combination_is_a_real_candidate():
    t=WanTask('original')
    rows=strategies_for(t)
    assert len(rows)==35 and len({x.id for x in rows})==35
    s=next(s for s in rows if s.placement=='resident' and s.chunk_tokens==2048 and s.lora_epilogue and s.rollout_window==5)
    assert s.implementation.native_placement=='resident-async128-tree-chunk2048'
    e=s.environment()
    assert e['NCCL_ALGO']=='Tree'
    assert e['DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE']=='1'
    assert e['DISTILLSCHEDULE_SEARCH_CANDIDATE_ID']=='joint-wan1-resident-c2048-e1'
    assert 'DISTILLSCHEDULE_WAN_ROLLOUT_WINDOW' not in e


@pytest.mark.parametrize('n',[1,2,3,5,8])
@pytest.mark.parametrize('res,frames', [((256,384),17),((384,640),33),((512,768),49)])
def test_resolution_and_ratio_change_contract_and_domain(n,res,frames):
    t=WanTask('different',resolution=res,video_frames=frames,source_steps=tuple(range(20,20+n)),ema_after=())
    assert t.rollout_spec.latent_shape==(1,16,(frames-1)//4+1,res[0]//8,res[1]//8)
    assert len(strategies_for(t))==7*n
    assert all(s.rollout_window<=n for s in strategies_for(t))
    assert WanTask.from_dict(t.to_dict())==t
    assert replace(t,stage_steps=(4,4,4)).digest!=t.digest


@pytest.mark.parametrize('bad',[dict(world_size=8),dict(lora_rank=128),dict(resolution=(255,384)),
                               dict(video_frames=34),dict(stage_steps=(0,2,2)),dict(ema_after=(5,)),
                               dict(denoising_timesteps=(750,1000)),dict(source_steps=(10,12))])
def test_unsupported_or_malformed_tasks_fail_before_model_construction(bad):
    with pytest.raises(ValueError): WanTask('invalid',**bad)


def test_no_fake_fsdp_kernel_admission():
    with pytest.raises(ValueError): WanStrategy('fsdp',2048,True)
    with pytest.raises(ValueError): WanTask.from_dict({'task_id':'bad','gan':True})
