"""Actual Tensor/Adam tests; not GPU or pretrained model performance evidence."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

torch=pytest.importorskip("torch")
from distillschedule.native.task import NativeTask
from distillschedule.native.engine import NativeEngine, compare_states
from distillschedule.native.planning import calibrate, plan, run, capabilities
from distillschedule.native.models import LoRALinear
from distillschedule.runtime.parameterized_cycle import parameterized_window_order
from distillschedule.workloads.best_known.registry import write_new_json, fingerprint


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("n",[1,2,5])
@pytest.mark.parametrize("steps",[1,4,8])
@pytest.mark.parametrize("rank",[1,3])
@pytest.mark.parametrize("shape",[(16,32,1),(32,48,9)])
def test_parameter_changes_execute_actual_math(n,steps,rank,shape):
    h,w,f=shape
    task=NativeTask(fake_updates=n,denoising_steps=steps,lora_rank=rank,
                    video_height=h,video_width=w,video_frames=f,ema_after=tuple(range(n)))
    engine=NativeEngine(task);initial=engine.snapshot()
    before={k:v.clone() for k,v in initial["params"]["critic"].items()}
    for _ in range(2):engine.run_reference_cycle()
    reference=engine.signature_tensors()
    engine.restore(initial)
    engine.adapter.set_implementation(checkpoint_enabled=True,epilogue="reference")
    for _ in range(2):
        result=engine.run_cycle(parameterized_window_order(engine.graph(),max_live_rollouts=n))
        assert result["calls"]["generator"]==(n+1)*steps
        assert result["retained_peak_bytes"]==n*task.rollout_bytes
    assert engine.versions=={"generator":2,"critic":2*n,"ema":2*n}
    assert compare_states(reference,engine.signature_tensors())["status"]=="PASS"
    assert any(not torch.equal(p,before[name]) for name,p in engine.adapter.trainables("critic").items())
    assert engine.adapter.base_digest()==engine.adapter.base_sha256
    assert all(p.grad is None for p in engine.adapter.models["teacher"].parameters())


@pytest.mark.parametrize("micro,accum",[(1,2),(2,2),(3,1)])
def test_batch_and_accumulation(micro,accum):
    t=NativeTask(microbatch=micro,accumulation=accum,fake_updates=2,ema_after=(0,))
    e=NativeEngine(t);initial=e.snapshot();e.run_reference_cycle();expected=e.signature_tensors();e.restore(initial)
    r=e.run_cycle(parameterized_window_order(e.graph(),max_live_rollouts=2))
    assert r["calls"]["generator"]==3*4*accum
    assert compare_states(expected,e.signature_tensors())["status"]=="PASS"


def test_rng_inputs_bound_by_logical_work():
    e=NativeEngine(NativeTask())
    before=torch.get_rng_state().clone()
    a=e.inputs(2,1,0);e.inputs(1,3,0);b=e.inputs(2,1,0)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert torch.equal(before,torch.get_rng_state())


def test_full_native_calibration_search_run(tmp_path):
    t=NativeTask(ema_after=(0,2),max_retained_bytes=512)
    m=calibrate(t,tmp_path/"cal")
    assert len(m["variants"])==2 and m["gpu_execution"]=="NOT_RUN"
    assert all(v["reference_gate"]["status"]=="PASS" for v in m["variants"])
    p=plan(t,tmp_path/"cal/calibration.json");write_new_json(tmp_path/"plan.json",p)
    assert p["selected"]["search_diagnostics"]["capacity_pruned"]>0
    out=run(t,tmp_path/"plan.json",tmp_path/"execution",cycles=3)
    assert out["final_versions"]=={"generator":3,"critic":9,"ema":6}
    assert out["reference_gate"]["status"]=="PASS" and out["gpu_execution"]=="NOT_RUN"
    assert all(r["retained_peak_bytes"]<=512 for r in out["cycles"])
    state=torch.load(tmp_path/"execution/final-state.pt",weights_only=True)
    assert state and all(isinstance(v,torch.Tensor) for v in state.values())


@pytest.mark.parametrize("field,value",[("video_width",48),("video_frames",9),("fake_updates",1),
    ("denoising_steps",8),("lora_rank",3),("lora_alpha",4.0),("microbatch",2),("seed",18)])
def test_changed_task_rejects_old_costs(tmp_path,field,value):
    t=NativeTask(fake_updates=2)
    calibrate(t,tmp_path/"cal")
    with pytest.raises(ValueError,match="task changed"):
        plan(replace(t,**{field:value}),tmp_path/"cal/calibration.json")


def test_offline_plan_never_constructs_model(tmp_path,monkeypatch):
    t=NativeTask(fake_updates=1)
    calibrate(t,tmp_path/"cal")
    import distillschedule.native.engine as engine_module
    def reject(*args,**kwargs):raise AssertionError("planning must be CPU metadata-only")
    monkeypatch.setattr(engine_module,"NativeEngine",reject)
    assert plan(t,tmp_path/"cal/calibration.json")["status"]=="PASS_NATIVE_PLAN_LOWERING"


def test_changed_cost_file_rejected(tmp_path):
    t=NativeTask(fake_updates=1);m=calibrate(t,tmp_path/"cal")
    p=Path(m["variants"][0]["cost_model"]);v=json.loads(p.read_text());v["records"][0]["samples_ns"]=[1];p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match="artifact changed"):plan(t,tmp_path/"cal/calibration.json")


@pytest.mark.parametrize("kind",["order","prediction","memory","implementation","graph"])
def test_tampered_native_plan_rejected_even_if_rehashed(tmp_path,kind):
    t=NativeTask(fake_updates=2);calibrate(t,tmp_path/"cal");p=plan(t,tmp_path/"cal/calibration.json")
    if kind=="order":p["lowered_order"][0],p["lowered_order"][1]=p["lowered_order"][1],p["lowered_order"][0]
    if kind=="prediction":p["selected"]["predicted_makespan_ns"]=1
    if kind=="memory":p["selected"]["memory"][0]["peak_bytes"]=0
    if kind=="implementation":p["selected"]["implementation"]["epilogue"]="fake-faster-kernel"
    if kind=="graph":p["graph"]["metadata"]["final_versions"]["critic"]=999
    p.pop("plan_sha256");p["plan_sha256"]=fingerprint(p);write_new_json(tmp_path/"plan.json",p)
    with pytest.raises(ValueError):run(t,tmp_path/"plan.json",tmp_path/"out")


def test_retained_limit_is_not_a_second_gpu(tmp_path):
    t=NativeTask(fake_updates=3,max_retained_bytes=512)
    calibrate(t,tmp_path/"cal");p=plan(t,tmp_path/"cal/calibration.json")
    assert [r["rank"] for r in p["selected"]["memory"]]==[0]
    assert all(r["resources"]==["gpu:0"] for r in p["selected"]["timeline"])


def test_no_silent_distributed_execution(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE","16")
    with pytest.raises(ValueError,match="one process"):NativeEngine(NativeTask())


def test_no_silent_fused_cpu_path():
    t=NativeTask();cap=capabilities(t)
    assert all(v["epilogue"]=="reference" for v in cap["variants"])
    assert cap["fallbacks"][0]["selected_reference"]=="torch-pointwise"
    e=NativeEngine(t)
    with pytest.raises(ValueError,match="CUDA/BF16"):
        e.adapter.set_implementation(checkpoint_enabled=False,epilogue="wan-fused")


def test_failed_capability_not_zero_trainables():
    with pytest.raises(ValueError,match="LoRA targets"):NativeEngine(NativeTask(target_modules=("does_not_exist",)))


def test_bf16_reference_and_checkpoint():
    e=NativeEngine(NativeTask(dtype="bfloat16",fake_updates=1))
    assert all(p.dtype==torch.float32 for p in e.adapter.trainables("generator").values())
    initial=e.snapshot();e.run_reference_cycle();reference=e.signature_tensors();e.restore(initial)
    e.adapter.set_implementation(checkpoint_enabled=True,epilogue="reference");e.run_cycle()
    assert compare_states(reference,e.signature_tensors())["status"]=="PASS"


def test_source_mutation_invalidates_costs(tmp_path,monkeypatch):
    t=NativeTask(fake_updates=1);calibrate(t,tmp_path/"cal")
    import distillschedule.native.planning as module
    monkeypatch.setattr(module,"source_fingerprint",lambda:"0"*64)
    with pytest.raises(ValueError,match="source changed"):plan(t,tmp_path/"cal/calibration.json")


def test_output_never_overwritten(tmp_path):
    t=NativeTask(fake_updates=1);calibrate(t,tmp_path/"cal")
    with pytest.raises(FileExistsError):calibrate(t,tmp_path/"cal")


def test_cli_requires_explicit_execution(tmp_path,capsys):
    from distillschedule.cli import main
    p=tmp_path/"task.json";p.write_text(json.dumps(NativeTask().to_dict()))
    rc=main(["workloads","native","calibrate","--task",str(p),"--output",str(tmp_path/"out")])
    assert rc==0 and not (tmp_path/"out").exists()
    assert "NOT_RUN_REQUIRES_EXPLICIT_EXECUTE" in capsys.readouterr().out
