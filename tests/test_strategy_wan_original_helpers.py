"""Run existing Generator helper with tensor fixtures, not a replacement trainer."""
import copy
import sys
from types import SimpleNamespace as NS, ModuleType
from unittest.mock import patch
import pytest

torch=pytest.importorskip('torch')
from distillschedule.strategies.wan_task import OriginalWanTask,TRACE_KEY


@pytest.mark.parametrize('h,w,frames,horizon',[(384,640,33,(2,2,2)),(64,128,5,(4,8,4))])
def test_original_generator_passes_actual_geometry_and_stage_horizons(monkeypatch,h,w,frames,horizon):
    import runtime.gst_pool_helios as module
    task=OriginalWanTask(video_height=h,video_width=w,video_frames=frames,stage2_inference_steps=horizon)
    captured={};parameter=torch.nn.Parameter(torch.tensor(0.25))
    class Audit:
        def __init__(self,*a,**kw):
            captured['audit']=kw
            self.calls=[dict(stage=i,call_index=i,grad_enabled=True,output_requires_grad=True) for i in range(3)]
    scheduler=ModuleType('helios.scheduler.scheduling_helios')
    scheduler.HeliosScheduler=lambda **kw:NS()
    utils=ModuleType('helios.utils.utils_helios_post')
    def generator(**kw):
        captured['generator']=kw
        return parameter.expand(task.latent_shape).to(torch.bfloat16),None,1,0,0
    utils.run_generator=generator
    monkeypatch.setitem(sys.modules,scheduler.__name__,scheduler)
    monkeypatch.setitem(sys.modules,utils.__name__,utils)
    monkeypatch.setattr(module,'LightweightNestedCallAudit',Audit)
    monkeypatch.setattr(module,'_memory_snapshot',lambda device:{})
    controls=dict(call_count=3,pyramid_stage_call_sequence=[dict(stage=i,call_count=1) for i in range(3)])
    trace=NS(manifest={'contract':{TRACE_KEY:task.to_dict()}},operation=lambda key:dict(control=controls),trace_hash='fixture')
    names=('entry_noise','indices_hidden_states','indices_history_short','indices_history_mid',
           'indices_history_long','history_short','history_mid','history_long','continuous_history','text_embedding')
    entry=NS(tensors={name:torch.zeros(1) for name in names},operation_trace=trace,
             rollout_operation=f'step/{task.first_step}/generator/rollout',logical_step=task.first_step,
             identity={},identity_sha256='fixture')
    model=torch.nn.Module()
    model.enable_adapters=lambda:None
    x,_,_=module.run_generator_update_trace_rollout(engine=model,entry=entry,device=torch.device('cpu'),
                detailed_profile=False,collect_diagnostics=False,empty_cache_before_rollout=False,
                synchronize_after_rollout=False,clean_runtime=True)
    assert captured['audit']['base_height']==h//8 and captured['audit']['base_width']==w//8
    assert captured['generator']['num_critic_input_frames']==task.latent_shape[2]
    assert captured['generator']['stage2_num_inference_steps_list']==list(horizon)
    assert x.shape==task.latent_shape and x.requires_grad
    x.float().mean().backward();assert parameter.grad is not None


def test_bootstrap_rank_is_explicit_and_legacy_default_unchanged():
    import inspect
    from distillschedule.runtime.wan13b_upper_bound import Wan13BBootstrap
    assert inspect.signature(Wan13BBootstrap.load).parameters['expected_lora_rank'].default==256


def test_generator_record_not_still_anchored_at_10():
    import ast
    from pathlib import Path
    source=Path('scripts/distillschedule/probe_static_fsdp2_p0_world16.py').read_text()
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)
                  and n.name=='_run_nogan_generator_update_p0')
    text=ast.unparse(function)
    assert 'generator_source_step = int(entry.logical_step)' in text
    assert "'logical_step': generator_source_step" in text
