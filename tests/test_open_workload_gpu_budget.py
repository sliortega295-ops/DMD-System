"""Live-memory admission checks independently of GPU hardware availability."""
from types import SimpleNamespace
import pytest
import torch
from distillschedule.open_workload.hardware_probe import plan_memory_limit


def test_cpu_requires_real_user_memory_bound():
    with pytest.raises(ValueError,match='must be explicit'):
        plan_memory_limit('cpu',distributed=False,consensus=lambda _: None)
    budget=plan_memory_limit('cpu',distributed=False,consensus=lambda _: None,requested=5000)
    assert budget['limit_bytes']==5000 and budget['rank_budgets']==[]


def test_gpu_size_is_observed_not_hardcoded(monkeypatch):
    # Mock only the CUDA driver observations; not a CUDA correctness test.
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    memory={'free':30*2**30, 'total':40*2**30, 'allocated':2*2**30}
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(memory['free'],memory['total']))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device:memory['allocated'])
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda device:memory['allocated'])
    a=plan_memory_limit('cuda:0',distributed=False,consensus=lambda _: None)
    assert a['limit_bytes']==30*2**30 and a['source'].startswith('automatic')
    memory.update(free=72*2**30,total=80*2**30)
    h=plan_memory_limit('cuda:0',distributed=False,consensus=lambda _: None)
    assert h['limit_bytes']==72*2**30 and h['limit_bytes']>a['limit_bytes']
    assert h['rank_budgets'][0]['total_bytes']==80*2**30
    with pytest.raises(ValueError,match='exceeds observed'):
        plan_memory_limit('cuda:0',distributed=False,consensus=lambda _: None,requested=90*2**30)


def test_distributed_capacity_is_minimum_rank(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(20*2**30,40*2**30))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device:3*2**30)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda device:3*2**30)
    def peer(value):
        return [value,{**value,'free_bytes':10*2**30,'conservative_limit_bytes':11*2**30}]
    b=plan_memory_limit('cuda:0',distributed=True,consensus=peer,requested=None)
    assert b['limit_bytes']==11*2**30 and len(b['rank_budgets'])==2
    with pytest.raises(ValueError,match='exceeds observed'):
        plan_memory_limit('cuda:0',distributed=True,consensus=peer,requested=12*2**30)


def test_explicit_negative_or_unsafe_reserve_rejected():
    for reserve in [-1,17*2**30]:
        with pytest.raises(ValueError,match='reserve'):
            plan_memory_limit('cpu',distributed=False,consensus=lambda _:None,requested=1,reserve_bytes=reserve)


def test_hardware_mismatch_rejects_reused_plan_before_native_work(tmp_path,monkeypatch):
    from test_open_workload_execution import build
    from distillschedule.open_workload.planning import calibrate,plan,run
    from distillschedule.open_workload.execution import ExecutionBackend
    from distillschedule.workloads.best_known.registry import write_new_json
    w=build(n=1)
    cal=calibrate(w,tmp_path/'cal',max_candidates=1,samples=1,max_program_runs=2)
    p=plan(tmp_path/'cal/calibration.json',memory_limit_bytes=10000000)
    write_new_json(tmp_path/'plan.json',p)
    old=ExecutionBackend.hardware
    def different(self):return {**old(self),'accelerator':{'gpu_family':'NVIDIA_H100_SM90'}}
    monkeypatch.setattr(ExecutionBackend,'hardware',different)
    with pytest.raises(ValueError,match='runtime environment changed'):
        run(w,tmp_path/'plan.json',tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_allocator_reserved_cache_is_counted_without_double_counting(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(16*2**30,40*2**30))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device:4*2**30)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda device:8*2**30)
    b=plan_memory_limit('cuda:0',distributed=False,consensus=lambda _:None)
    assert b['limit_bytes']==22*2**30
    assert b['rank_budgets'][0]['torch_allocated_bytes']==4*2**30
    assert b['rank_budgets'][0]['torch_reserved_bytes']==8*2**30


def test_cpu_plan_still_explicit_after_gpu_autoadmission(tmp_path):
    import json
    from distillschedule.open_workload.cli import main
    from distillschedule.workloads.best_known.registry import repository_root
    root=repository_root()
    code=main(['auto','--factory',str(root/'examples/open_workloads/video_input_gradient.py')+':build',
               '--parameters',str(root/'examples/open_workloads/video_parameters.json'),
               '--output',str(tmp_path/'not-created'),'--samples','1',
               '--max-candidates','1','--max-program-runs','4','--execute'])
    # The failure is preflight, not a CPU 'GPU auto-memory' guess.
    assert code==2 and not (tmp_path/'not-created').exists()
