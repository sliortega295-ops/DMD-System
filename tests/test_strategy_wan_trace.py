"""Trace admission uses original cross-rank checks; no mocked performance."""
import copy
from types import SimpleNamespace as NS
import pytest
pytest.importorskip('torch')
from distillschedule.strategies.wan_task import OriginalWanTask, TRACE_KEY
from distillschedule.strategies.wan_trace import validate_rank_controls
from runtime.static_world16_trace import _generator_keys, _critic_keys


def controls(task):
    ops=[];tensors={}
    for step in task.source_steps:
        ops.append(dict(key=f'step/{step}/branch',control=dict(global_step=step,
            dfake_gen_update_ratio=task.fake_updates,train_generator=step==task.first_step,
            train_critic=True,gan=False,reward=False,vae_decode=False)))
        phases=[('critic',_critic_keys(step,gan=False))]
        if step==task.first_step: phases.append(('generator',_generator_keys(step,gan=False)))
        for phase,keys in phases:
            for key in keys.values(): tensors[key]={'shape':list(task.latent_shape),'dtype':'bfloat16'}
            sequence=[dict(stage=i,call_count=1,denoising_exit=0,timestep=[1000],
                      latent_shape=[*task.latent_shape[:3],*hw]) for i,hw in enumerate(task.stage_spatial_shapes)]
            ops.append(dict(key=f'step/{step}/{phase}/rollout',control=dict(call_count=3,
                pyramid_stage_call_sequence=sequence,initial_synchronized_pyramid_stage_exit=[0,0,0],
                stage_schedule_scope='global_rank0_broadcast_equivalent')))
            ops.append(dict(key=f'step/{step}/{phase}/{"train" if phase=="critic" else "dmd"}',control={}))
    contract={TRACE_KEY:task.to_dict(),'resolution':[task.video_height,task.video_width],
        'stage_spatial_shapes':[list(x) for x in task.stage_spatial_shapes],
        'full_latent_shape':list(task.latent_shape),'dtype':'bfloat16','dfake_gen_update_ratio':task.fake_updates}
    result={}
    for rank in range(16):
        m=copy.deepcopy(dict(contract=contract,operations=ops,tensors=tensors))
        index={o['key']:o for o in m['operations']}
        result[rank]=NS(manifest=m,operation=lambda k,rows=index:rows[k])
    return result


@pytest.mark.parametrize('n,h,w,f',[(1,256,384,17),(3,384,640,33),(8,512,768,49)])
def test_original_controls_variable_shape_ratio(n,h,w,f):
    task=OriginalWanTask(fake_updates=n,first_step=2*n,video_height=h,video_width=w,video_frames=f)
    result=validate_rank_controls(task,controls(task))
    assert result['status']=='PASS'


@pytest.mark.parametrize('mutation',['rank','ratio','extra','duplicate','shape','stage','exit','missing_task','horizon'])
def test_bad_controls_rejected(mutation):
    task=OriginalWanTask(fake_updates=2,first_step=4);rows=controls(task)
    m=rows[1].manifest
    if mutation=='rank':rows.pop(2)
    if mutation=='ratio':m['contract']['dfake_gen_update_ratio']=5
    if mutation=='extra':m['operations'].append(dict(key='step/6/branch',control={}))
    if mutation=='duplicate':m['operations'].append(m['operations'][0])
    if mutation=='shape':m['tensors']['step/4/critic/entry_noise']['shape']=[1,16,9,12,20]
    op=rows[1].operation('step/4/critic/rollout')['control']
    if mutation=='stage':op['pyramid_stage_call_sequence'][0]['latent_shape']=[1,16,9,1,1]
    if mutation=='exit':op['initial_synchronized_pyramid_stage_exit']=[1,0,0]
    if mutation=='missing_task':m['contract'].pop(TRACE_KEY)
    if mutation=='horizon':op['pyramid_stage_call_sequence'][0]['call_count']=3
    with pytest.raises(ValueError):validate_rank_controls(task,rows)
