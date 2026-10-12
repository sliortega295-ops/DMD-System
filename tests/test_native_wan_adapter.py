"""Real diffusers WanTransformer3DModel tests at tiny random scale (CPU, no download).

These are architecture/gradient and local checkpoint-loading tests, NOT trained
Wan1.3B/14B quality, old canonical workload parity, or GPU performance evidence.
"""
from dataclasses import replace
from pathlib import Path
import json
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("diffusers")
from diffusers import WanTransformer3DModel
from safetensors.torch import save_file
from distillschedule.native.task import NativeTask
from distillschedule.native.engine import NativeEngine, compare_states
from distillschedule.native.planning import calibrate,plan,run
from distillschedule.runtime.parameterized_cycle import parameterized_window_order
from distillschedule.workloads.best_known.registry import write_new_json

CONFIG={"patch_size":[1,2,2],"num_attention_heads":2,"attention_head_dim":12,
        "in_channels":4,"out_channels":4,"text_dim":16,"freq_dim":32,
        "ffn_dim":32,"num_layers":1,"cross_attn_norm":True,
        "qk_norm":"rms_norm_across_heads","rope_max_seq_len":32}


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def task(**kwargs):
    return NativeTask(backend="diffusers-wan",text_dim=16,model_config=dict(CONFIG),
                      target_modules=("to_q","to_k","to_v","to_out.0"),**kwargs)


@pytest.mark.parametrize("shape,n,steps,rank",[
    ((16,32,1),1,4,1),((32,32,5),2,4,2),((32,48,9),1,8,3)])
def test_actual_wan_forward_backward_parameterization(shape,n,steps,rank):
    h,w,f=shape
    t=task(video_height=h,video_width=w,video_frames=f,fake_updates=n,denoising_steps=steps,lora_rank=rank)
    e=NativeEngine(t);initial=e.snapshot()
    assert isinstance(e.adapter.models['generator'],WanTransformer3DModel)
    e.run_reference_cycle();reference=e.signature_tensors();e.restore(initial)
    e.adapter.set_implementation(checkpoint_enabled=True,epilogue="reference")
    result=e.run_cycle(parameterized_window_order(e.graph(),max_live_rollouts=n))
    assert result['calls']['generator']==(n+1)*steps
    assert compare_states(reference,e.signature_tensors())["status"]=="PASS"
    assert e.adapter.base_digest()==e.adapter.base_sha256


def test_real_wan_measured_frontier_native_loop(tmp_path):
    t=task(fake_updates=2,denoising_steps=2,ema_after=(0,1))
    m=calibrate(t,tmp_path/'cal')
    assert m['model_sources']['diffusers_version']=='0.35.2'
    p=plan(t,tmp_path/'cal/calibration.json');write_new_json(tmp_path/'plan.json',p)
    result=run(t,tmp_path/'plan.json',tmp_path/'run')
    assert result['status']=='PASS_NATIVE_EXECUTION'
    assert result['reference_gate']['status']=='PASS'
    assert result['final_versions']=={'generator':2,'critic':4,'ema':4}
    assert result['fixture'] is True and result['gpu_execution']=='NOT_RUN'


def test_local_checkpoint_and_data_path_not_random_fallback(tmp_path):
    model=WanTransformer3DModel(**CONFIG)
    model.save_pretrained(tmp_path/'model',safe_serialization=True)
    save_file({'prompt_embeds':torch.randn(8,4,16)},tmp_path/'conditions.safetensors')
    t=replace(task(fake_updates=1,denoising_steps=2),fixture=False,model_config=None,
              checkpoints={r:str(tmp_path/'model') for r in ('generator','critic','teacher')},
              conditions_path=str(tmp_path/'conditions.safetensors'))
    e=NativeEngine(t)
    assert 'config.json' in e.adapter.sources['generator']
    assert e.conditions.identity!='SYNTHETIC_CONDITIONS_EXPLICIT_FIXTURE'
    a=e.run_cycle();b=e.run_cycle()
    assert b['versions']=={'generator':2,'critic':2,'ema':0}
    assert a['fixture'] is False
    with pytest.raises(ValueError,match='cohort exhausted'):
        e.inputs(100,0,0)


def test_native_geometry_failure_not_silent_resize():
    with pytest.raises(ValueError,match='patch lattice'):
        NativeEngine(task(video_height=24,fake_updates=1))


def test_wan_channel_mismatch_rejected():
    with pytest.raises(ValueError,match='channels mismatch'):
        NativeEngine(task(channels=8,fake_updates=1))
