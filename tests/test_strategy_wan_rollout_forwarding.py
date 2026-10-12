"""Exercise the original wrapper with a CPU sampler double, not GPU/model evidence."""
import sys
from types import SimpleNamespace as NS, ModuleType
import pytest
torch=pytest.importorskip('torch')
from distillschedule.strategies.wan_spec import WanTask
from distillschedule.strategies.wan_trace import TASK_KEY


@pytest.mark.parametrize('resolution,frames,budgets',[((256,384),17,(4,4,4)),((384,640),33,(2,2,2))])
def test_original_rollout_receives_task_parameters(monkeypatch,resolution,frames,budgets):
    import runtime.gst_pool_helios as native
    task=WanTask('forwarding',resolution=resolution,video_frames=frames,stage_steps=budgets)
    spec=task.rollout_spec
    scheduler=ModuleType('helios.scheduler.scheduling_helios')
    scheduler.HeliosScheduler=lambda **kw:NS()
    upstream=ModuleType('helios.utils.utils_helios_post');observed={}
    def run_generator(**kw):
        observed.update(kw)
        kw['transformer'].calls=[dict(stage=i,call_index=i,grad_enabled=True,output_requires_grad=True) for i in range(3)]
        out=kw['noise']*torch.tensor(1.,requires_grad=True)
        return out,None,torch.tensor(1.),torch.tensor(0.),torch.tensor(0.)
    upstream.run_generator=run_generator
    monkeypatch.setitem(sys.modules,scheduler.__name__,scheduler)
    monkeypatch.setitem(sys.modules,upstream.__name__,upstream)
    monkeypatch.setattr(native,'LightweightNestedCallAudit',lambda *a,**kw:NS(calls=[],**kw))
    monkeypatch.setattr(native,'_memory_snapshot',lambda dev:{})
    model=torch.nn.Linear(1,1);model.enable_adapters=lambda:None
    keys=('entry_noise','indices_hidden_states','indices_history_short','indices_history_mid','indices_history_long',
          'history_short','history_mid','history_long','continuous_history','text_embedding')
    tensors={k:torch.zeros(1) for k in keys}
    tensors['entry_noise']=torch.zeros(spec.latent_shape,dtype=torch.bfloat16)
    control=dict(call_count=3,pyramid_stage_call_sequence=[dict(stage=i,call_count=1) for i in range(3)])
    contract={TASK_KEY:task.to_dict(),'resolution':list(resolution),'full_latent_shape':list(spec.latent_shape),
              'stage_spatial_shapes':spec.spatial_shapes,'dtype':'bfloat16','dfake_gen_update_ratio':5}
    trace=NS(manifest={'contract':contract},operation=lambda key:{'control':control},trace_hash='cpu-fixture')
    entry=NS(tensors=tensors,operation_trace=trace,rollout_operation='step/10/generator/rollout',
             logical_step=10,identity={},identity_sha256='cpu-fixture')
    out,_,_=native.run_generator_update_trace_rollout(engine=model,entry=entry,device=torch.device('cpu'),
                      detailed_profile=False,collect_diagnostics=False,clean_runtime=True)
    assert out.shape==spec.latent_shape and out.requires_grad
    assert observed['stage2_num_inference_steps_list']==list(budgets)
    assert observed['num_critic_input_frames']==spec.latent_shape[2]
    assert observed['transformer'].base_height==spec.latent_shape[-2]
    assert observed['denoising_step_list']==list(task.denoising_timesteps)
