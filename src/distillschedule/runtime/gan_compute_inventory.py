"""Exact semantic model-call inventory for GAN baseline/pipeline fairness.

Random-tensor parity is necessary but not sufficient: a scheduler could still
silently omit a fixed Score/DiT call. This contract binds the complete logical
model work per sample for the frozen Helios GAN configuration at steps1000..1004.
"""
from __future__ import annotations
import hashlib,json
from typing import Any,Mapping
from .gan_fairness import FULL5_STEPS,LOGICAL_SAMPLE_SLOTS,normalize_gan_fairness_receipt

SCHEMA_VERSION="distillschedule.gan_compute_receipt.v1"
GENERATOR_FIXED_CALLS={
 "dmd_fake_cond_score_forwards":1,
 "dmd_fake_uncond_score_forwards":0,
 "dmd_real_cond_score_forwards":1,
 "dmd_real_uncond_score_forwards":1,
 "gan_g_score_forwards":1,
 "gan_g_score_batch_items":1,
}
CRITIC_FIXED_CALLS={
 "critic_denoise_score_forwards":1,
 "critic_denoise_score_batch_items":1,
 "gan_d_score_forwards":1,
 "gan_d_score_batch_items":3,
}

def _digest(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def _int(v:Any,p:str)->int:
    if isinstance(v,bool):raise ValueError(f"{p} must be an integer")
    try:r=int(v)
    except (TypeError,ValueError) as e:raise ValueError(f"{p} must be an integer") from e
    if r<0:raise ValueError(f"{p} must be non-negative")
    return r
def _calls(raw:Any,*,fixed:Mapping[str,int],path:str)->dict[str,int]:
    if not isinstance(raw,Mapping):raise ValueError(f"{path} must be a mapping")
    expected={"rollout_generator_dit_forwards",*fixed.keys()}; observed=set(map(str,raw))
    if observed!=expected:raise ValueError(f"{path} call inventory drift: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
    out={k:_int(raw[k],f"{path}.{k}") for k in sorted(expected)}
    for k,v in fixed.items():
        if out[k]!=v:raise ValueError(f"{path}.{k}={out[k]}, expected official value {v}")
    return out

def normalize_gan_compute_receipt(value:Mapping[str,Any])->dict[str,Any]:
    if value.get("schema_version")!=SCHEMA_VERSION or value.get("status")!="PASS":raise ValueError("GAN compute receipt schema/status mismatch")
    if value.get("family") not in {"baseline","pipeline"}:raise ValueError("GAN compute receipt family must be baseline or pipeline")
    entries=value.get("entries")
    if not isinstance(entries,list):raise ValueError("GAN compute receipt entries must be a list")
    expected={(s,l) for s in FULL5_STEPS for l in LOGICAL_SAMPLE_SLOTS}; seen=set(); norm=[]
    for i,raw in enumerate(entries):
        p=f"entries[{i}]"
        if not isinstance(raw,Mapping):raise ValueError(f"{p} must be a mapping")
        step,slot=_int(raw.get("logical_step"),f"{p}.logical_step"),_int(raw.get("logical_sample_slot"),f"{p}.logical_sample_slot")
        pair=(step,slot)
        if pair not in expected or pair in seen:raise ValueError(f"{p} invalid/duplicate step-slot {pair}")
        seen.add(pair); gr=raw.get("generator")
        if step==1000:g=_calls(gr,fixed=GENERATOR_FIXED_CALLS,path=f"{p}.generator")
        else:
            if gr not in (None,{}):raise ValueError(f"{p}.generator must be absent on Critic-only step {step}")
            g=None
        c=_calls(raw.get("critic"),fixed=CRITIC_FIXED_CALLS,path=f"{p}.critic")
        norm.append({"logical_step":step,"logical_sample_slot":slot,"generator":g,"critic":c})
    if seen!=expected:raise ValueError(f"GAN compute receipt misses logical work: {sorted(expected-seen)}")
    norm.sort(key=lambda x:(x["logical_step"],x["logical_sample_slot"]))
    canonical={"schema_version":SCHEMA_VERSION,"status":"PASS","logical_steps":list(FULL5_STEPS),"logical_sample_slots":list(LOGICAL_SAMPLE_SLOTS),"entries":norm}
    canonical["compute_contract_digest"]=_digest(canonical); return canonical

def validate_compute_against_fairness(compute:Mapping[str,Any],fairness:Mapping[str,Any])->dict[str,Any]:
    c,f=normalize_gan_compute_receipt(compute),normalize_gan_fairness_receipt(fairness); ce={(x["logical_step"],x["logical_sample_slot"]):x for x in c["entries"]}
    for item in f["entries"]:
        pair=(item["logical_step"],item["logical_sample_slot"]); comp=ce[pair]
        if item["generator"] is not None and comp["generator"]["rollout_generator_dit_forwards"]!=item["generator"]["dit"]["total_dit_forwards"]:raise ValueError(f"Generator rollout DiT count mismatch at {pair}")
        if comp["critic"]["rollout_generator_dit_forwards"]!=item["critic"]["dit"]["total_dit_forwards"]:raise ValueError(f"Critic rollout DiT count mismatch at {pair}")
    return {"status":"PASS","compute_contract_digest":c["compute_contract_digest"],"fairness_contract_digest":f["fairness_contract_digest"]}
def compare_gan_compute_receipts(a:Mapping[str,Any],b:Mapping[str,Any])->dict[str,Any]:
    x,y=normalize_gan_compute_receipt(a),normalize_gan_compute_receipt(b)
    if x!=y:raise ValueError("GAN baseline/pipeline model-call inventory mismatch")
    return {"schema_version":"distillschedule.gan_compute_pair.v1","status":"PASS","compute_contract_digest":x["compute_contract_digest"],"fixed_generator_score_forwards_per_sample":4,"fixed_critic_score_forwards_per_sample":2,"gan_d_batched_items_per_sample":3}
__all__=["CRITIC_FIXED_CALLS","GENERATOR_FIXED_CALLS","SCHEMA_VERSION","compare_gan_compute_receipts","normalize_gan_compute_receipt","validate_compute_against_fairness"]
