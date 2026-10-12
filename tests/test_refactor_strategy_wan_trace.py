import copy
from types import SimpleNamespace as NS
import pytest
from distillschedule.strategies.wan_spec import WanTask,PyramidRolloutSpec
from distillschedule.strategies.wan_trace import rollout_spec_for,validate_rank_manifest,TASK_KEY


def manifest(task):
    spec=task.rollout_spec
    c={TASK_KEY:task.to_dict(),'resolution':list(task.resolution),'full_latent_shape':list(spec.latent_shape),
       'stage_spatial_shapes':spec.spatial_shapes,'dtype':'bfloat16','dfake_gen_update_ratio':len(task.source_steps)}
    ops=[]
    for step in task.source_steps:
        first=step==task.source_steps[0]
        ops.append(dict(key=f'step/{step}/branch',control=dict(global_step=step,dfake_gen_update_ratio=len(task.source_steps),
                       train_generator=first,train_critic=True,gan=False,reward=False,vae_decode=False)))
        for role in (('generator','critic') if first else ('critic',)):
            prefix=f'step/{step}/{role}'
            ops.append(dict(key=prefix+('/dmd' if role=='generator' else '/train')))
            ops.append(dict(key=prefix+'/rollout',control=dict(call_count=3,pyramid_stage_call_sequence=[
                dict(stage=i,latent_shape=[1,16,spec.latent_shape[2],*spec.spatial_shapes[i]],
                     call_count=1,denoising_exit=0) for i in range(3)])))
    return dict(contract=c,operations=ops)


@pytest.mark.parametrize('n',[1,3,5])
@pytest.mark.parametrize('res,frames,steps',[((256,384),17,(4,4,4)),((384,640),33,(2,2,2)),((512,768),49,(8,8,8))])
def test_task_is_bound_to_trace_shape_ratio_and_sampler(n,res,frames,steps):
    t=WanTask('trace',resolution=res,video_frames=frames,stage_steps=steps,source_steps=tuple(range(10,10+n)),ema_after=())
    m=manifest(t)
    validate_rank_manifest(t,m,rank=0)
    assert rollout_spec_for(NS(manifest=m))==t.rollout_spec
    for key in ('resolution','full_latent_shape','dfake_gen_update_ratio'):
        bad=copy.deepcopy(m);bad['contract'][key]=None
        with pytest.raises(ValueError):validate_rank_manifest(t,bad,rank=0)


def test_legacy_defaults_unchanged():
    assert rollout_spec_for(NS(manifest={'contract':{}}))==PyramidRolloutSpec()


def test_legacy_manifest_cannot_be_called_parameterized():
    t=WanTask('old');m=manifest(t);del m['contract'][TASK_KEY]
    with pytest.raises(ValueError,match='recorded'):validate_rank_manifest(t,m,rank=0)


def test_bad_branch_and_stage_shape_rejected():
    t=WanTask('bad');m=manifest(t)
    m['operations'][0]['control']['train_generator']=False
    with pytest.raises(ValueError):validate_rank_manifest(t,m,rank=0)
    m=manifest(t);m['operations'][2]['control']['pyramid_stage_call_sequence'][0]['latent_shape']=[1,16,9,12,20]
    t2=WanTask('other',resolution=(512,768))
    with pytest.raises(ValueError):validate_rank_manifest(t2,m,rank=0)
