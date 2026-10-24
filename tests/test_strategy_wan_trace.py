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


def native_controls(task):
    rows = controls(task)
    for rank, trace in rows.items():
        for op in trace.manifest['operations']:
            if not op['key'].endswith('/rollout'):
                continue
            c = op['control']
            zero = rank % 2 == 0
            c.update(stage_schedule_policy=task.stage_schedule_policy, rank=rank,
                     stage_schedule_scope='native_rank0_initial_with_rank_local_amplified', all_history_zero=zero)
            for stage, s in enumerate(c['pyramid_stage_call_sequence']):
                calls = (rank+stage) % 4+1 if zero else 1
                s.update(denoising_step_budget=4 if zero else 2, call_count=calls,
                         denoising_exit=calls-1, timestep=[1000-i for i in range(calls)])
            c['call_count'] = sum(s['call_count'] for s in c['pyramid_stage_call_sequence'])
    return rows


def test_native_history_rank_census_not_padding_or_cross_rank_equality():
    from distillschedule.strategies.wan_trace import expected_rank_rollout_inventory
    task = OriginalWanTask(stage_schedule_policy='source_native_amplified_v1')
    rows = native_controls(task)
    checked = validate_rank_controls(task, rows)
    assert checked['status'] == 'PASS' and not checked['cross_rank_call_count_parity']
    assert checked['local_exception_entry_count'] == 8*6
    trace = NS(rank_control_validation=checked)
    key = 'step/10/generator/rollout'
    for rank in range(16):
        census = expected_rank_rollout_inventory(trace, operation=key, rank=rank)
        c = rows[rank].operation(key)['control']
        assert census['call_count'] == census['logical_call_count'] == c['call_count']
        assert census['dummy_call_count'] == 0
        assert list(census['stage_call_count'].values()) == [s['call_count'] for s in c['pyramid_stage_call_sequence']]
    with pytest.raises(ValueError, match='executing rank'):
        expected_rank_rollout_inventory(trace, operation=key)


@pytest.mark.parametrize('mutation', ['missing_rank', 'rank_identity', 'scope', 'initial', 'budget',
    'normal_exit', 'exit_count', 'timestep_count', 'timestep_nan', 'total', 'order', 'predicate'])
def test_native_history_invalid_controls_fail_closed(mutation):
    task = OriginalWanTask(stage_schedule_policy='source_native_amplified_v1')
    rows = native_controls(task)
    c = rows[1].operation('step/10/generator/rollout')['control']
    if mutation == 'missing_rank': rows.pop(15)
    if mutation == 'rank_identity': c['rank'] = 0
    if mutation == 'scope': c['stage_schedule_scope'] = 'global_rank0_broadcast_equivalent'
    if mutation == 'initial': c['initial_synchronized_pyramid_stage_exit'] = [1,0,0]
    if mutation == 'budget': c['pyramid_stage_call_sequence'][0]['denoising_step_budget'] = 4
    if mutation == 'normal_exit':
        c['pyramid_stage_call_sequence'][0].update(denoising_exit=1,call_count=2,timestep=[1000,900]); c['call_count']=4
    if mutation == 'exit_count': c['pyramid_stage_call_sequence'][0]['denoising_exit'] = 1
    if mutation == 'timestep_count': c['pyramid_stage_call_sequence'][0]['timestep'] = []
    if mutation == 'timestep_nan': c['pyramid_stage_call_sequence'][0]['timestep'] = [float('nan')]
    if mutation == 'total': c['call_count'] = 4
    if mutation == 'order': c['pyramid_stage_call_sequence'].reverse()
    if mutation == 'predicate': c['all_history_zero'] = 0
    with pytest.raises(ValueError): validate_rank_controls(task,rows)


def test_native_history_payload_predicate_is_verified():
    import torch
    from distillschedule.strategies.wan_trace import validate_loaded_history
    task = OriginalWanTask(stage_schedule_policy='source_native_amplified_v1')
    value = torch.zeros(1)
    loaded = NS(tensor=lambda key:value,operation=lambda key:dict(control=dict(all_history_zero=True)))
    validate_loaded_history(task,loaded)
    value.fill_(1)
    with pytest.raises(ValueError,match='payload'):
        validate_loaded_history(task,loaded)
