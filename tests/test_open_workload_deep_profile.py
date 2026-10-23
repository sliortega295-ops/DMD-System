"""One selected native program emits a trace without contaminating cost timing."""
import json
from pathlib import Path
import pytest
import torch
from distillschedule.open_workload.deep_profile import selected_plan_trace
from distillschedule.workloads.best_known.registry import write_new_json,file_sha256
from distillschedule.open_workload.planning import calibrate,plan,run
from test_open_workload_execution import build


def test_selected_trace_is_real_unprofiled_reference_outside_profiling(tmp_path):
    w=build(n=1,width=4)
    m=calibrate(w,tmp_path/'cal',samples=1,max_candidates=1,max_program_runs=2)
    p=plan(tmp_path/'cal/calibration.json',memory_limit_bytes=1000000)
    write_new_json(tmp_path/'plan.json',p)
    result,session,_=run(w,tmp_path/'plan.json',tmp_path/'run',deep_profile=True)
    assert result['status']=='PASS_SELECTED_PROGRAM'
    assert result['reference_gate']['status']=='PASS'
    assert result['timing_claim']=='PROFILER_TRACE_NOT_BENCHMARK'
    assert len(result['profiler_traces'])==1
    evidence=result['profiler_traces'][0]
    trace=Path(evidence['path'])
    assert trace.exists() and file_sha256(trace)==evidence['sha256']
    assert evidence['gpu_execution']=='CPU_TRACE_NOT_GPU_EVIDENCE'
    assert trace.stat().st_size>200
    assert 'distillschedule.call.' in trace.read_text()
    assert m['hardware_probe']['status']=='NOT_REQUESTED'
    assert all(r['reference_gate']['status']=='PASS' for r in m['variants'])


def test_deep_trace_no_overwrite_and_no_fake_cuda(tmp_path):
    path=tmp_path/'trace';path.mkdir()
    result,evidence=selected_plan_trace(lambda:torch.ones(3)+torch.ones(3),path,rank=0,device='cpu')
    assert result.shape==(3,) and evidence['claimable_speedup'] is False
    with pytest.raises(ValueError,match='overwrite'):
        selected_plan_trace(lambda:torch.zeros(1),path,rank=0,device='cpu')
