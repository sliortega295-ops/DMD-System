from __future__ import annotations
import pytest
from distillschedule.runtime.gan_fairness import CRITIC_RANDOM_ROLES,FULL5_STEPS,GENERATOR_RANDOM_ROLES,LOGICAL_SAMPLE_SLOTS,GanLogicalSampleAddress,compare_gan_fairness_receipts,normalize_gan_fairness_receipt

def _sha(seed:int)->str:return f"{seed:064x}"[-64:]
def _phase(seed:int,roles:tuple[str,...],exits=(1,0,1))->dict:
    return {"trace_operation_sha256":_sha(seed),"dit":{"stage_schedule_scope":"global_rank0_broadcast_equivalent","stage_exits":list(exits),"stage_call_counts":[x+1 for x in exits],"total_dit_forwards":sum(x+1 for x in exits)},"consumed":{role:_sha(seed*100+i+1) for i,role in enumerate(roles)}}
def _receipt(family:str)->dict:
    entries=[]
    for step in FULL5_STEPS:
        for slot in LOGICAL_SAMPLE_SLOTS:
            a=GanLogicalSampleAddress.for_slot(slot)
            physical={"rank":a.baseline_rank} if family=="baseline" else {"wave":a.pipeline_wave,"generator_rank":a.pipeline_generator_rank,"score_rank":a.pipeline_score_rank}
            seed=step*1000+slot
            entries.append({"logical_step":step,"logical_sample_slot":slot,"physical":physical,"generator":_phase(seed+1,GENERATOR_RANDOM_ROLES) if step==1000 else None,"critic":_phase(seed+2,CRITIC_RANDOM_ROLES)})
    return {"schema_version":"distillschedule.gan_fairness_receipt.v1","family":family,"status":"PASS","trace_bundle_hash":"a"*64,"r2_enabled":False,"entries":entries}

def test_world16_to_two_wave_mapping():
    assert GanLogicalSampleAddress.for_slot(15).pipeline_wave==1
    assert GanLogicalSampleAddress.for_slot(15).pipeline_generator_rank==7
    assert GanLogicalSampleAddress.for_slot(15).pipeline_score_rank==15

def test_exact_consumed_random_and_dit_parity_passes():
    assert compare_gan_fairness_receipts(_receipt("baseline"),_receipt("pipeline"))["status"]=="PASS"

def test_changed_noise_fails_even_if_seed_would_match():
    a,b=_receipt("baseline"),_receipt("pipeline")
    b["entries"][0]["generator"]["consumed"]["generator_dmd_noise"]="b"*64
    with pytest.raises(ValueError,match="stochastic tensor or DiT schedule"):compare_gan_fairness_receipts(a,b)

def test_changed_dit_exit_fails():
    a,b=_receipt("baseline"),_receipt("pipeline")
    b["entries"][16]["critic"]["dit"].update(stage_exits=[1,1,1],stage_call_counts=[2,2,2],total_dit_forwards=6)
    with pytest.raises(ValueError,match="stochastic tensor or DiT schedule"):compare_gan_fairness_receipts(a,b)

def test_wrong_wave_mapping_fails():
    value=_receipt("pipeline"); value["entries"][8]["physical"]["wave"]=0
    with pytest.raises(ValueError,match="pipeline sample mapping"):normalize_gan_fairness_receipt(value)

def test_rank_local_exception_is_out_of_scope():
    value=_receipt("baseline"); value["entries"][0]["critic"]["dit"]["stage_schedule_scope"]="rank_local_amplified_gt_zero_history_exception"
    with pytest.raises(ValueError,match="forbids rank-local"):normalize_gan_fairness_receipt(value)
