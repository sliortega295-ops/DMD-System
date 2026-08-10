from __future__ import annotations
import json,time
import pytest
from distillschedule.runtime.gan_timing_range import GanFull5TimingRange,validate_timing_receipts

def test_exact_unprofiled_range_has_no_extra_sync_contract(tmp_path):
    gate=GanFull5TimingRange(receipt_path=tmp_path/"rank_000.json",rank=0)
    gate.before_model_compute(1000); time.sleep(0.001); gate.after_critic_commit(1004); gate.assert_complete()
    value=json.loads((tmp_path/"rank_000.json").read_text())
    assert value["profiled"] is False
    assert value["diagnostic_only"] is False
    assert value["logical_window"]==[1000,1004]

def test_global_window_uses_earliest_start_and_latest_stop(tmp_path):
    values=[(100,500),(120,480)]
    for rank,(start,stop) in enumerate(values):
        (tmp_path/f"rank_{rank:03d}.json").write_text(json.dumps({"schema_version":"distillschedule.gan_full5_timing_rank.v1","status":"PASS","profiled":False,"diagnostic_only":False,"rank":rank,"logical_window":[1000,1004],"start_point":"before_first_model_compute","stop_point":"after_step1004_critic_commit","start_unix_ns":start,"stop_unix_ns":stop,"local_elapsed_seconds":(stop-start)/1e9,"host":"h"})+"\n")
    result=validate_timing_receipts(tmp_path,expected_world_size=2)
    assert result["global_training_window_seconds"]==(500-100)/1e9

def test_incomplete_range_fails():
    gate=GanFull5TimingRange(rank=0)
    gate.before_model_compute(1000)
    with pytest.raises(AssertionError,match="both boundaries"):gate.assert_complete()
