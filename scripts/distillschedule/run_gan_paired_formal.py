#!/usr/bin/env python3
"""Run matched unprofiled GAN baseline/pipeline Full5 timing pairs.

Formal speedup exists only when baseline and pipeline prove the same logical
steps1000..1004, consumed stochastic values, DiT exits/model-call inventory,
and exact low-overhead Full5 wall-clock boundaries. Nsight/profiled runs are
separate diagnostics and are rejected here.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,math,os,shlex,statistics,subprocess
from pathlib import Path
from typing import Any,Mapping,Sequence
from distillschedule.runtime.gan_compute_inventory import compare_gan_compute_receipts,validate_compute_against_fairness
from distillschedule.runtime.gan_fairness import FULL5_STEPS,compare_gan_fairness_receipts
from distillschedule.runtime.gan_timing_range import validate_timing_receipts

SCHEMA="distillschedule.gan_paired_formal_timing.v2"

def load(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text())
    if not isinstance(value,dict):raise RuntimeError(f"expected JSON object: {path}")
    return value
def atomic(path:Path,value:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
def digest(value:Any)->str:return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def resolve_path(result:Mapping[str,Any],result_path:Path,key:str,*,directory:bool=False)->Path:
    raw=result.get(key)
    if not isinstance(raw,str) or not raw.strip():raise RuntimeError(f"{result_path}: missing {key}")
    path=Path(raw).expanduser(); path=path if path.is_absolute() else result_path.parent/path; path=path.resolve()
    if directory and not path.is_dir():raise RuntimeError(f"{result_path}: {key} missing directory {path}")
    if not directory and not path.is_file():raise RuntimeError(f"{result_path}: {key} missing file {path}")
    return path
def observed_workload_digest(result:Mapping[str,Any])->str|None:
    value=result.get("workload_contract_digest")
    if value is None and isinstance(result.get("evidence_contract"),Mapping):value=result["evidence_contract"].get("shared_digest")
    return value if isinstance(value,str) else None
def validate_result(path:Path,expected_digest:str)->dict[str,Any]:
    result=load(path)
    if result.get("status")!="PASS":raise RuntimeError(f"{path}: result is not PASS")
    if result.get("profiled") is True or result.get("diagnostic_only") is True:raise RuntimeError(f"{path}: profiled/diagnostic run cannot enter formal timing")
    if observed_workload_digest(result)!=expected_digest:raise RuntimeError(f"{path}: workload contract digest drift")
    fairness_path=resolve_path(result,path,"gan_fairness_receipt"); compute_path=resolve_path(result,path,"gan_compute_receipt"); timing_dir=resolve_path(result,path,"gan_timing_receipt_dir",directory=True)
    fair_value,compute_value=load(fairness_path),load(compute_path); validate_compute_against_fairness(compute_value,fair_value)
    timing=validate_timing_receipts(timing_dir,expected_world_size=16)
    if timing.get("logical_steps")!=list(FULL5_STEPS) or timing.get("profiled") is not False:raise RuntimeError(f"{path}: exact timing receipt contract drift")
    seconds=float(timing["global_training_window_seconds"])
    if not math.isfinite(seconds) or seconds<=0:raise RuntimeError(f"{path}: invalid exact Full5 timing")
    return {"seconds":seconds,"fairness":fair_value,"compute":compute_value,"fairness_path":fairness_path,"compute_path":compute_path,"timing_dir":timing_dir,"timing":timing}
def run_one(*,family:str,pair:int,parent:str,root:Path,template:str,expected_digest:str,resume:bool)->dict[str,Any]:
    run_id=f"{parent}-pair{pair:02d}-{family}"; run_dir=root/"trials"/run_id; result_path=run_dir/"result.json"
    if result_path.exists():
        if not resume:raise RuntimeError(f"immutable trial exists: {run_id}")
    else:
        run_dir.mkdir(parents=True,exist_ok=False); command=shlex.split(template.format(run_id=run_id,result_path=str(result_path)))
        env=dict(os.environ); env["DISTILLSCHEDULE_GAN_NSYS_PROFILE"]="0"; env["DISTILLSCHEDULE_GAN_EXACT_TIMING"]="1"
        with (run_dir/"console.log").open("wb") as log:rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False,env=env).returncode
        if rc!=0:raise RuntimeError(f"{run_id}: command failed rc={rc}")
        if not result_path.is_file():raise RuntimeError(f"{run_id}: no result manifest")
    checked=validate_result(result_path,expected_digest)
    return {"family":family,"run_id":run_id,"seconds":checked["seconds"],"result_path":str(result_path),"result_sha256":sha(result_path),"fairness_path":str(checked["fairness_path"]),"fairness_sha256":sha(checked["fairness_path"]),"compute_path":str(checked["compute_path"]),"compute_sha256":sha(checked["compute_path"]),"timing_receipt_dir":str(checked["timing_dir"]),"timing":checked["timing"],"_fairness":checked["fairness"],"_compute":checked["compute"]}
def summary(values:Sequence[float])->dict[str,float|int]:
    x=list(map(float,values)); mean=statistics.fmean(x); med=statistics.median(x); sd=statistics.stdev(x) if len(x)>1 else 0.0
    return {"count":len(x),"mean":mean,"median":med,"min":min(x),"max":max(x),"sample_stdev":sd,"coefficient_of_variation":sd/mean if mean else 0.0}
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--run-id",required=True); p.add_argument("--output-root",required=True,type=Path); p.add_argument("--workload-contract",required=True,type=Path); p.add_argument("--baseline-command-template",required=True); p.add_argument("--pipeline-command-template",required=True); p.add_argument("--trials",type=int,default=3); p.add_argument("--resume",action="store_true"); a=p.parse_args()
    if a.trials<3:raise SystemExit("formal GAN timing requires at least 3 paired trials")
    contract=load(a.workload_contract.resolve()); expected=contract.get("workload_contract_digest") or contract.get("shared_digest")
    if not isinstance(expected,str) or len(expected)!=64:raise SystemExit("workload contract lacks 64-char digest")
    root=(a.output_root/a.run_id).resolve(); prereg=root/"preregistration.json"
    frozen={"schema_version":SCHEMA,"run_id":a.run_id,"workload_contract_sha256":sha(a.workload_contract.resolve()),"workload_contract_digest":expected,"trials":a.trials,"order_policy":"alternate baseline/pipeline then pipeline/baseline","baseline_command_digest":digest(a.baseline_command_template),"pipeline_command_digest":digest(a.pipeline_command_template),"formal_window":list(FULL5_STEPS),"profiled_runs_allowed":False,"required_evidence":["gan_fairness_receipt","gan_compute_receipt","gan_timing_receipt_dir"]}
    if prereg.exists():
        old=load(prereg); comparable=dict(old); comparable.pop("created_at",None)
        if comparable!=frozen:raise SystemExit("immutable GAN formal run collision")
    else:
        root.mkdir(parents=True,exist_ok=False); atomic(prereg,{**frozen,"created_at":dt.datetime.now(dt.timezone.utc).isoformat()})
    pairs=[]
    try:
        for i in range(a.trials):
            order=("baseline","pipeline") if i%2==0 else ("pipeline","baseline"); got={}
            for family in order:got[family]=run_one(family=family,pair=i,parent=a.run_id,root=root,template=a.baseline_command_template if family=="baseline" else a.pipeline_command_template,expected_digest=expected,resume=a.resume)
            fair=compare_gan_fairness_receipts(got["baseline"]["_fairness"],got["pipeline"]["_fairness"]); comp=compare_gan_compute_receipts(got["baseline"]["_compute"],got["pipeline"]["_compute"])
            for family in ("baseline","pipeline"):got[family].pop("_fairness"); got[family].pop("_compute")
            pairs.append({"pair_index":i,"launch_order":list(order),"baseline":got["baseline"],"pipeline":got["pipeline"],"fairness":fair,"compute_inventory":comp})
        base=[x["baseline"]["seconds"] for x in pairs]; pipe=[x["pipeline"]["seconds"] for x in pairs]; ratios=[b/p for b,p in zip(base,pipe)]
        result={"schema_version":SCHEMA,"status":"PASS","claimable":True,"profiled":False,"logical_steps":list(FULL5_STEPS),"workload_contract_digest":expected,"pair_count":len(pairs),"pairs":pairs,"baseline_seconds":base,"pipeline_seconds":pipe,"paired_speedup_baseline_over_pipeline":ratios,"median_paired_speedup":statistics.median(ratios),"geometric_mean_paired_speedup":math.exp(statistics.fmean(math.log(x) for x in ratios)),"distributions":{"baseline":summary(base),"pipeline":summary(pipe),"speedup":summary(ratios)},"claim_boundary":"Exact unprofiled model-compute Full5 windows only; stochastic/DiT/model-call parity required."}
        atomic(root/"result.json",result); print(json.dumps({"status":"PASS","result":str(root/"result.json")},sort_keys=True)); return 0
    except Exception as exc:
        failure={"schema_version":SCHEMA,"status":"FAIL","claimable":False,"error":str(exc),"completed_pairs":pairs}; atomic(root/"result.json",failure); print(json.dumps(failure,sort_keys=True)); return 1
if __name__=="__main__":raise SystemExit(main())
