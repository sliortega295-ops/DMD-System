#!/usr/bin/env python3
"""Run matched unprofiled GAN baseline/pipeline Full5 timing pairs.

A timing result is claimable only when each pair proves:
- the same steps1000..1004 logical window and workload digest;
- exact consumed stochastic/source/DiT-exit parity;
- exact semantic model-call inventory, including variable rollout DiT counts;
- no Nsight/profiled/diagnostic instrumentation in the timed runs.

Nsight Systems runs are deliberately separate diagnostics and cannot enter this
formal speedup table.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,math,os,shlex,statistics,subprocess
from pathlib import Path
from typing import Any,Mapping,Sequence
from distillschedule.runtime.gan_compute_inventory import compare_gan_compute_receipts,validate_compute_against_fairness
from distillschedule.runtime.gan_fairness import FULL5_STEPS,compare_gan_fairness_receipts

SCHEMA="distillschedule.gan_paired_formal_timing.v1"

def load(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text())
    if not isinstance(value,dict):raise RuntimeError(f"expected JSON object: {path}")
    return value
def atomic(path:Path,value:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
def sha(path:Path)->str:
    h=hashlib.sha256();
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
def digest(value:Any)->str:return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def resolve_receipt(result:Mapping[str,Any],result_path:Path,key:str)->Path:
    raw=result.get(key)
    if not isinstance(raw,str) or not raw.strip():raise RuntimeError(f"{result_path}: missing {key}")
    path=Path(raw).expanduser(); path=path if path.is_absolute() else result_path.parent/path
    path=path.resolve()
    if not path.is_file():raise RuntimeError(f"{result_path}: {key} missing file {path}")
    return path
def logical_steps(result:Mapping[str,Any])->list[int]|None:
    raw=result.get("logical_steps")
    if raw is None and isinstance(result.get("official_semantics"),Mapping):raw=result["official_semantics"].get("logical_steps")
    if raw is None and isinstance(result.get("timing"),Mapping):
        window=result["timing"].get("logical_window")
        if window==[1000,1004]:raw=list(FULL5_STEPS)
    if not isinstance(raw,list):return None
    try:return [int(x) for x in raw]
    except (TypeError,ValueError):return None
def validate_result(path:Path,expected_digest:str)->dict[str,Any]:
    result=load(path)
    if result.get("status")!="PASS" or result.get("claimable") is not True:raise RuntimeError(f"{path}: result is not claimable PASS")
    if result.get("profiled") is True or result.get("diagnostic_only") is True:raise RuntimeError(f"{path}: profiled/diagnostic run cannot enter formal timing")
    observed=result.get("workload_contract_digest")
    if observed is None and isinstance(result.get("evidence_contract"),Mapping):observed=result["evidence_contract"].get("shared_digest")
    if observed!=expected_digest:raise RuntimeError(f"{path}: workload contract digest drift")
    if logical_steps(result)!=list(FULL5_STEPS):raise RuntimeError(f"{path}: formal GAN timing must cover exactly steps1000..1004")
    timing=result.get("timing")
    if not isinstance(timing,Mapping):raise RuntimeError(f"{path}: timing object missing")
    seconds=float(timing.get("global_training_window_seconds",0))
    if not math.isfinite(seconds) or seconds<=0:raise RuntimeError(f"{path}: invalid five-step timing window")
    fairness=resolve_receipt(result,path,"gan_fairness_receipt")
    compute=resolve_receipt(result,path,"gan_compute_receipt")
    fair_value,compute_value=load(fairness),load(compute)
    validate_compute_against_fairness(compute_value,fair_value)
    return {"result":result,"seconds":seconds,"fairness_path":fairness,"compute_path":compute,"fairness":fair_value,"compute":compute_value}
def run_one(*,family:str,pair:int,parent:str,root:Path,template:str,expected_digest:str,resume:bool)->dict[str,Any]:
    run_id=f"{parent}-pair{pair:02d}-{family.lower()}"; run_dir=root/"trials"/run_id; result_path=run_dir/"result.json"
    if result_path.exists():
        if not resume:raise RuntimeError(f"immutable trial exists: {run_id}")
        checked=validate_result(result_path,expected_digest)
    else:
        run_dir.mkdir(parents=True,exist_ok=False); command=shlex.split(template.format(run_id=run_id,result_path=str(result_path)))
        with (run_dir/"console.log").open("wb") as log:
            rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
        if rc!=0:raise RuntimeError(f"{run_id}: command failed rc={rc}")
        if not result_path.is_file():raise RuntimeError(f"{run_id}: no result manifest")
        checked=validate_result(result_path,expected_digest)
    return {"family":family,"run_id":run_id,"seconds":checked["seconds"],"result_path":str(result_path),"result_sha256":sha(result_path),"fairness_path":str(checked["fairness_path"]),"fairness_sha256":sha(checked["fairness_path"]),"compute_path":str(checked["compute_path"]),"compute_sha256":sha(checked["compute_path"]),"_fairness":checked["fairness"],"_compute":checked["compute"]}
def summarize(values:Sequence[float])->dict[str,float|int]:
    x=list(map(float,values)); mean=statistics.fmean(x); med=statistics.median(x); sd=statistics.stdev(x) if len(x)>1 else 0.0
    return {"count":len(x),"mean":mean,"median":med,"min":min(x),"max":max(x),"sample_stdev":sd,"coefficient_of_variation":sd/mean if mean else 0.0}
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--run-id",required=True); p.add_argument("--output-root",required=True,type=Path); p.add_argument("--workload-contract",required=True,type=Path); p.add_argument("--baseline-command-template",required=True); p.add_argument("--pipeline-command-template",required=True); p.add_argument("--trials",type=int,default=3); p.add_argument("--resume",action="store_true"); a=p.parse_args()
    if a.trials<3:raise SystemExit("formal GAN timing requires at least 3 paired trials")
    contract=load(a.workload_contract.resolve()); expected=contract.get("workload_contract_digest") or contract.get("shared_digest")
    if not isinstance(expected,str) or len(expected)!=64:raise SystemExit("workload contract lacks 64-char digest")
    root=(a.output_root/a.run_id).resolve(); prereg=root/"preregistration.json"
    frozen={"schema_version":SCHEMA,"run_id":a.run_id,"workload_contract":str(a.workload_contract.resolve()),"workload_contract_sha256":sha(a.workload_contract.resolve()),"workload_contract_digest":expected,"trials":a.trials,"order_policy":"alternate baseline/pipeline then pipeline/baseline","baseline_command_digest":digest(a.baseline_command_template),"pipeline_command_digest":digest(a.pipeline_command_template),"formal_window":list(FULL5_STEPS),"profiled_runs_allowed":False,"required_evidence":["gan_fairness_receipt","gan_compute_receipt"]}
    if prereg.exists():
        if load(prereg)!={**frozen,"created_at":load(prereg).get("created_at")}:raise SystemExit("immutable GAN formal run collision")
    else:
        root.mkdir(parents=True,exist_ok=False); atomic(prereg,{**frozen,"created_at":dt.datetime.now(dt.timezone.utc).isoformat()})
    pairs=[]
    try:
        for i in range(a.trials):
            order=("baseline","pipeline") if i%2==0 else ("pipeline","baseline"); got={}
            for family in order:
                got[family]=run_one(family=family,pair=i,parent=a.run_id,root=root,template=a.baseline_command_template if family=="baseline" else a.pipeline_command_template,expected_digest=expected,resume=a.resume)
            fair=compare_gan_fairness_receipts(got["baseline"]["_fairness"],got["pipeline"]["_fairness"])
            comp=compare_gan_compute_receipts(got["baseline"]["_compute"],got["pipeline"]["_compute"])
            for family in ("baseline","pipeline"):
                got[family].pop("_fairness"); got[family].pop("_compute")
            pairs.append({"pair_index":i,"launch_order":list(order),"baseline":got["baseline"],"pipeline":got["pipeline"],"fairness":fair,"compute_inventory":comp})
        base=[x["baseline"]["seconds"] for x in pairs]; pipe=[x["pipeline"]["seconds"] for x in pairs]; ratios=[b/p for b,p in zip(base,pipe)]
        result={"schema_version":SCHEMA,"status":"PASS","claimable":True,"profiled":False,"logical_steps":list(FULL5_STEPS),"workload_contract_digest":expected,"pair_count":len(pairs),"pairs":pairs,"baseline_seconds":base,"pipeline_seconds":pipe,"paired_speedup_baseline_over_pipeline":ratios,"median_paired_speedup":statistics.median(ratios),"geometric_mean_paired_speedup":math.exp(statistics.fmean(math.log(x) for x in ratios)),"distributions":{"baseline":summarize(base),"pipeline":summarize(pipe),"speedup":summarize(ratios)},"claim_boundary":"Only unprofiled steps1000..1004 pairs with exact stochastic, DiT-exit, and model-call parity."}
        atomic(root/"result.json",result); print(json.dumps({"status":"PASS","result":str(root/"result.json")},sort_keys=True)); return 0
    except Exception as exc:
        failure={"schema_version":SCHEMA,"status":"FAIL","claimable":False,"error":str(exc),"completed_pairs":pairs}; atomic(root/"result.json",failure); print(json.dumps(failure,sort_keys=True)); return 1
if __name__=="__main__":raise SystemExit(main())
