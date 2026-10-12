from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pytest
from distillschedule.runtime.wan13b_shadow_trajectory_study import (
    ENV, ROUNDING_ENV, anchors, study_requested, stochastic_bf16,
    rounding_moments, metrics, rounding_seed, within_noise, noise_ratio,
    sustained_growth, accepted_timing_cycles,
)


def test_opt_in_and_anchors(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert not study_requested()
    monkeypatch.setenv(ENV,'yes')
    with pytest.raises(ValueError): study_requested()
    assert anchors(30)==(1,5,10,20,30)
    assert anchors(20)==(1,5,10,20)
    assert anchors(1)==(1,)
    assert anchors(2)==(1,2)
    assert anchors(5)==(1,5)
    assert anchors(50)==(1,10,25,50)


def test_stochastic_rounding_reproducible_unbiased_and_training_rng_untouched():
    t=pytest.importorskip('torch')
    values=t.tensor([1.001, -1.001, 0.501, -0.501, 1.0, 0.0],dtype=t.float32).repeat(40000,1)
    before=t.get_rng_state().clone()
    def run():
        g=t.Generator(device='cpu').manual_seed(rounding_seed(seed=42,rank=0,version=3,layer='x'))
        return stochastic_bf16(values,generator=g)
    a,b=run(),run()
    assert t.equal(a,b) and t.equal(before,t.get_rng_state())
    assert t.equal(a[:,4:].float(),values[:,4:])
    assert (a.float().mean(0)-values[0]).abs().max().item()<0.00007
    assert a.dtype==t.bfloat16
    assert rounding_seed(seed=42,rank=0,version=4,layer='x') != rounding_seed(seed=42,rank=0,version=3,layer='x')


def test_rounding_stats_detect_swallowed_update():
    t=pytest.importorskip('torch')
    base=t.tensor([1.,-1.],dtype=t.bfloat16)
    delta=t.tensor([0.001,-0.001])
    row=metrics(rounding_moments(base,delta,(base.float()+delta).to(t.bfloat16)).tolist())
    assert row['lost_ratio']==pytest.approx(1,abs=0.0001)
    assert row['bias_cos']==pytest.approx(1,abs=0.0001)
    assert row['sub_half_bf16_ulp_fraction']==1
    zero=metrics(rounding_moments(base,t.zeros_like(delta),base).tolist())
    assert zero['lost_ratio'] is None


def test_no_posthoc_noise_floor_or_trend_relaxation():
    assert within_noise(0,0) and not within_noise(1e-10,0)
    assert within_noise(.02,.01) and not within_noise(.021,.01)
    assert noise_ratio(1,0) is None
    assert sustained_growth([.01,.02,.03,.04])
    assert not sustained_growth([.01,.02,.019,.04])


def test_timing_gate_fail_closed(tmp_path,monkeypatch):
    assert accepted_timing_cycles({},expected_head='h',arm_id='A')==2
    p=tmp_path/'manifest.json';p.write_text('{}')
    gate={'schema_version':'wan13b.shadow_noise_acceptance.v1','status':'PASS',
          'expected_head':'h','cycles':30,'accepted_arm_ids':['A','B'],
          'preregistration_path':str(p),'preregistration_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    x={'shadow_noise_acceptance':gate}
    with pytest.raises(ValueError): accepted_timing_cycles(x,expected_head='h',arm_id='A')
    monkeypatch.setenv(ENV,'1')
    assert accepted_timing_cycles(x,expected_head='h',arm_id='A')==30
    for key,val in [('status','FAIL'),('expected_head','bad'),('cycles',2),('accepted_arm_ids',['C'])]:
        bad={'shadow_noise_acceptance':{**gate,key:val}}
        with pytest.raises(ValueError): accepted_timing_cycles(bad,expected_head='h',arm_id='A')
    p.write_text('changed')
    with pytest.raises(ValueError): accepted_timing_cycles(x,expected_head='h',arm_id='A')


def test_probe_is_unmerged_and_outside_timer():
    root=Path(__file__).resolve().parents[1]
    helper=(root/'src/distillschedule/runtime/wan13b_shadow_trajectory_study.py').read_text()
    worker=(root/'scripts/distillschedule/probe_wan13b_trajectory_world16.py').read_text()
    assert 'model=self.generator.module' in helper
    assert 'study consumed CUDA training RNG' in helper
    assert 'merged_shadow_parity_check=shadow_enabled and not study' in worker
    assert worker.index('training_window_wall =') < worker.index('observer.after_cycle(')
