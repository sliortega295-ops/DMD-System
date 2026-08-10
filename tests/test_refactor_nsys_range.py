from __future__ import annotations
import json
import pytest
from distillschedule.runtime.nsys_range import GanFull5NsightRange,validate_nsys_receipts

class FakeProfiler:
    def __init__(self):self.calls=[]
    def start(self):self.calls.append("start")
    def stop(self):self.calls.append("stop")

def test_local_main_controls_profiler(tmp_path):
    api=FakeProfiler(); gate=GanFull5NsightRange(api=api,receipt_path=tmp_path/"rank_000.json",rank=0,local_rank=0)
    gate.before_model_compute(1000); gate.after_critic_commit(1004); gate.assert_complete()
    assert api.calls==["start","stop"]
    assert json.loads((tmp_path/"rank_000.json").read_text())["diagnostic_only"] is True

def test_other_rank_records_without_profiler_api(tmp_path):
    api=FakeProfiler(); gate=GanFull5NsightRange(api=api,receipt_path=tmp_path/"rank_001.json",rank=1,local_rank=1)
    gate.before_model_compute(1000); gate.after_critic_commit(1004)
    assert api.calls==[]

def test_bundle_requires_one_controller_per_node(tmp_path):
    for rank in range(4):
        lr=rank%2; api=FakeProfiler(); gate=GanFull5NsightRange(api=api,receipt_path=tmp_path/f"rank_{rank:03d}.json",rank=rank,local_rank=lr)
        gate.before_model_compute(1000); gate.after_critic_commit(1004)
    result=validate_nsys_receipts(tmp_path,expected_world_size=4,expected_local_world_size=2)
    assert result["controller_ranks"]==[0,2]

def test_bad_range_order_fails():
    gate=GanFull5NsightRange(api=FakeProfiler(),rank=0,local_rank=0)
    with pytest.raises(RuntimeError,match="before it started"):gate.after_critic_commit(1004)
