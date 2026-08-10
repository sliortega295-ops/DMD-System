from __future__ import annotations
import pytest
from distillschedule.runtime.gan_compute_inventory import CRITIC_FIXED_CALLS,GENERATOR_FIXED_CALLS,compare_gan_compute_receipts,validate_compute_against_fairness
from distillschedule.runtime.gan_fairness import CRITIC_RANDOM_ROLES,FULL5_STEPS,GENERATOR_RANDOM_ROLES,LOGICAL_SAMPLE_SLOTS,GanLogicalSampleAddress

def _sha(n:int)->str:return f"{n:064x}"[-64:]
def _phase(seed:int,roles:tuple[str,...],exits=(1,0,1)):
    return {"trace_operation_sha256":_sha(seed),"dit":{"stage_schedule_scope":"global_rank0_broadcast_equivalent","stage_exits":list(exits),"stage_call_counts":[x+1 for x in exits],"total_dit_forwards":sum(x+1 for x in exits)},"consumed":{r:_sha(seed*100+i+1) for i,r in enumerate(roles)}}
def _fairness(family:str):
    entries=[]
    for step in FULL5_STEPS:
        for slot in LOGICAL_SAMPLE_SLOTS:
            a=GanLogicalSampleAddress.for_slot(slot); p={"rank":slot} if family=="baseline" else {"wave":a.pipeline_wave,"generator_rank":a.pipeline_generator_rank,"score_rank":a.pipeline_score_rank}; seed=step*100+slot
            entries.append({"logical_step":step,"logical_sample_slot":slot,"physical":p,"generator":_phase(seed+1,GENERATOR_RANDOM_ROLES) if step==1000 else None,"critic":_phase(seed+2,CRITIC_RANDOM_ROLES)})
    return {"schema_version":"distillschedule.gan_fairness_receipt.v1","family":family,"status":"PASS","trace_bundle_hash":"a"*64,"r2_enabled":False,"entries":entries}
def _calls(total:int,fixed:dict[str,int]):return {"rollout_generator_dit_forwards":total,**fixed}
def _compute(family:str,fairness:dict):
    entries=[]
    for item in fairness["entries"]:
        g=item["generator"]; c=item["critic"]
        entries.append({"logical_step":item["logical_step"],"logical_sample_slot":item["logical_sample_slot"],"generator":_calls(g["dit"]["total_dit_forwards"],GENERATOR_FIXED_CALLS) if g else None,"critic":_calls(c["dit"]["total_dit_forwards"],CRITIC_FIXED_CALLS)})
    return {"schema_version":"distillschedule.gan_compute_receipt.v1","family":family,"status":"PASS","entries":entries}

def test_exact_fixed_score_and_rollout_counts_pass():
    fa,fb=_fairness("baseline"),_fairness("pipeline"); ca,cb=_compute("baseline",fa),_compute("pipeline",fb)
    assert validate_compute_against_fairness(ca,fa)["status"]=="PASS"
    result=compare_gan_compute_receipts(ca,cb)
    assert result["fixed_generator_score_forwards_per_sample"]==4
    assert result["fixed_critic_score_forwards_per_sample"]==2
    assert result["gan_d_batched_items_per_sample"]==3

def test_skipping_real_uncond_is_rejected():
    f=_fairness("pipeline"); c=_compute("pipeline",f); c["entries"][0]["generator"]["dmd_real_uncond_score_forwards"]=0
    with pytest.raises(ValueError,match="expected official value 1"):validate_compute_against_fairness(c,f)

def test_gan_d_batch_must_include_fake_real_r1():
    f=_fairness("baseline"); c=_compute("baseline",f); c["entries"][0]["critic"]["gan_d_score_batch_items"]=2
    with pytest.raises(ValueError,match="expected official value 3"):validate_compute_against_fairness(c,f)

def test_rollout_count_must_match_stage_exit_trace():
    f=_fairness("baseline"); c=_compute("baseline",f); c["entries"][0]["generator"]["rollout_generator_dit_forwards"]+=1
    with pytest.raises(ValueError,match="Generator rollout DiT count mismatch"):validate_compute_against_fairness(c,f)
