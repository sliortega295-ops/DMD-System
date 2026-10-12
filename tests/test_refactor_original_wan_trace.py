import copy
import pytest
from distillschedule.strategies.wan import OriginalWanTask as Task
from distillschedule.strategies.wan_trace import validate_manifest


def manifest(task):
    ops=[];tensors={}
    for i,s in enumerate(task.source_steps):
        ops.extend([{'key':f'step/{s}/branch','control':dict(global_step=s,dfake_gen_update_ratio=task.fake_updates,
                     train_generator=i==0,train_critic=True,gan=False,reward=False,vae_decode=False)},
                    {'key':f'step/{s}/critic/train','control':{}}])
    ops.append({'key':'step/10/generator/dmd','control':{}})
    for prefix in ['step/10/generator',*[f'step/{s}/critic' for s in task.source_steps]]:
        ops.append({'key':prefix+'/rollout','control':{'call_count':3,'pyramid_stage_call_sequence':[
            {'stage':i,'call_count':1,'denoising_exit':0,'latent_shape':[*task.latent_shape[:3],*shape]}
            for i,shape in enumerate(task.stage_shapes)]}})
        for name in ('entry_noise','rollout_noise','dmd_noise' if 'generator' in prefix else 'critic_noise'):
            tensors[prefix+'/'+name]={'shape':list(task.latent_shape),'dtype':'bfloat16'}
    return {'contract':task.trace_contract(),'operations':ops,'tensors':tensors}

@pytest.mark.parametrize('n',[1,2,5,8])
@pytest.mark.parametrize('resolution,frames',[((256,384),17),((384,640),33),((512,768),65)])
def test_parameterized_control_checks(n,resolution,frames):
    t=Task(fake_updates=n,resolution=resolution,video_frames=frames)
    assert len(validate_manifest(manifest(t),t)['rollout_operations'])==n+1

@pytest.mark.parametrize('kind',['shape','frequency','extra_rollout','missing_branch','wrong_g','noise','dtype','stage','calls'])
def test_trace_mismatch_not_resized_or_reinterpreted(kind):
    t=Task(fake_updates=2);m=manifest(t)
    if kind=='shape':m['contract']['resolution']=[64,64]
    if kind=='frequency':m['contract']['dfake_gen_update_ratio']=5
    if kind=='extra_rollout':m['operations'].append({'key':'step/14/critic/rollout'})
    if kind=='missing_branch':m['operations']=m['operations'][1:]
    if kind=='wrong_g':m['operations'][0]['control']['train_generator']=False
    if kind=='noise':m['tensors']['step/10/critic/entry_noise']['shape']=[1,16,1,48,80]
    if kind=='dtype':m['tensors']['step/10/critic/entry_noise']['dtype']='float32'
    if kind=='stage':m['operations'][-1]['control']['pyramid_stage_call_sequence'][0]['latent_shape']=[1,16,9,48,80]
    if kind=='calls':m['operations'][-1]['control']['call_count']=100
    with pytest.raises(ValueError):validate_manifest(m,t)

def test_legacy_contract_requires_explicit_admission_and_exact_default_task():
    t=Task();m=manifest(t);m['contract'].pop('parameterized_wan_task')
    with pytest.raises(ValueError):validate_manifest(m,t)
    assert validate_manifest(m,t,allow_legacy=True)['legacy_exact_contract']
    with pytest.raises(ValueError):validate_manifest(m,Task(fake_updates=1),allow_legacy=True)
