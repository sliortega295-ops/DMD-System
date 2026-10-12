from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from distillschedule.strategies.wan import OriginalWanTask as Task, OriginalWanStrategy as Strategy, candidates, capabilities, rollout_parameters
from distillschedule.runtime.wan13b_fair import configure_modulation_chunk_tokens
from distillschedule.search.wan_implementation import current_implementation

@pytest.mark.parametrize('n',[1,2,3,5,8])
def test_strategies_contain_actual_winners_without_frequency_lock(n,monkeypatch):
    task=Task(fake_updates=n)
    rows=candidates(task)
    assert len(rows)==7*n
    assert Strategy('resident',2048,True,n) in rows
    for s in rows:
        s.validate(task)
        for k,v in s.environment(task).items(): monkeypatch.setenv(k,v)
        assert current_implementation()==s.implementation
        model=NS(blocks=[NS() for _ in range(30)])
        result=configure_modulation_chunk_tokens(model,placement=s.implementation.native_placement)
        assert result['effective_tokens']==[s.chunk_tokens]
        order=s.order(task,{'generator':2,'critic':10,'ema':10})
        assert sum(x.startswith('c') for x in order)==n
        assert next(i for i,x in enumerate(order) if x.startswith('c'))==s.rollout_window+1

@pytest.mark.parametrize('resolution,frames,n',[((256,384),17,1),((384,640),33,5),((512,768),65,8)])
def test_shapes_frequency_change_real_invocation_arguments(resolution,frames,n):
    t=Task(resolution=resolution,video_frames=frames,fake_updates=n,stage2_steps=(3,3,2))
    params=rollout_parameters(NS(manifest={'contract':t.trace_contract()}))
    assert params['num_critic_input_frames']==(frames-1)//4+1
    assert (params['base_height'],params['base_width'])==tuple(x//8 for x in resolution)
    assert params['stage2_num_inference_steps_list']==[3,3,2]
    assert capabilities(t)['latent_shape']==list(t.latent_shape)
    assert Task.from_dict(t.to_dict())==t

def test_legacy_invocation_defaults_unchanged():
    p=rollout_parameters(NS(manifest={}))
    assert p['latent_shape']==(1,16,9,48,80)
    assert p['denoising_step_list']==[1000,750,500,250]
    assert p['stage2_num_inference_steps_list']==[2,2,2]

@pytest.mark.parametrize('kwargs',[{'resolution':(384,641)},{'world_size':8},{'physical_batch':2},
 {'lora_rank':128},{'fake_updates':True},{'fake_updates':0},{'video_frames':32},
 {'denoising_step_list':(250,1000)},{'stage2_steps':(2,2)},{'ema_after':(5,)},{'protocol':'euler'}])
def test_unsupported_parameters_are_not_silently_substituted(kwargs):
    with pytest.raises(ValueError):Task(**kwargs)

def test_frozen_backend_capabilities_not_relaxed():
    for s in [('fsdp',2048,False),('fsdp',512,True),('resident',8192,False)]:
        with pytest.raises(ValueError):Strategy(*s)
    with pytest.raises(ValueError): Strategy(rollout_window=5).validate(Task(fake_updates=1))
    with pytest.raises(ValueError):Task.from_dict({'gan':True})
    t=Task();c=t.trace_contract();c['full_latent_shape']=[1,16,9,12,20]
    with pytest.raises(ValueError):rollout_parameters(NS(manifest={'contract':c}))
