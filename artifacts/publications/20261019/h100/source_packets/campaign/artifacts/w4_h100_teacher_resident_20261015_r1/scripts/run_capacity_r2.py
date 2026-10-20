"""Bounded W4 placement capacity only; reuse existing native launcher/lifecycle."""
import datetime,json,os,pathlib,signal,subprocess,sys,traceback,types
ROOT=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100")
O=ROOT/"research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1"
P=O/"artifacts/w4_h100_teacher_resident_20261015_r1"
W=ROOT/"repos/DMD-System-h100-livewan-teacher-resident-20261015-r1"
HEAD="c7590a3e45abd5770e4069cbf26fe3ceba2dcf1b"
sys.path.insert(0,str(O/"scripts"));import run_livewan_resident_matched_r2 as base
base.REPO=W;base.HEAD=HEAD

def main():
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,lambda signum,frame:(_ for _ in ()).throw(InterruptedError("bounded ownedW4runner:"+str(signum))))
 signal.alarm(1800)
 assert os.environ.get("DISTILLSCHEDULE_INHERITED_GPU_LOCKS")=="1"
 assert subprocess.check_output(["git","-C",str(W),"rev-parse","HEAD"],text=True).strip()==HEAD
 assert not subprocess.check_output(["git","-C",str(W),"status","--porcelain"],text=True).strip()
 cpu=json.load(open(P/"manifests/CPU_tests_r1.json"));assert cpu["status"]=="PASS" and not cpu["CUDA_initialized"]
 out=P/"artifacts/capacity_r2";out.mkdir(parents=True,exist_ok=False);(out/"logs").mkdir()
 runner=object.__new__(base.Runner);runner.workload="W4_teacher_resident_capacity";runner.out=out;runner.ordinal=0;runner.records=[]
 def status(self,phase,state,**fields):base.dump(self.out/"status.json",{"run_id":"livewan_h100_teacher_resident_capacity_20261015_r2","phase":phase,"status":state,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"head":HEAD,**fields},immutable=False)
 runner.status=types.MethodType(status,runner)
 r={"status":"FAIL","head":HEAD,"workload":"W4 C34 Teacherfullresident only","physical_ranks":8,"logical_batch":16,"formal_timing":"NOT_RUN","strict_full_state_equivalence":"NOT_RUN","new_controller":False}
 try:
  runner.invoke("verify_inherited_lease",[base.PYTHON,W/"scripts/distillschedule/verify_inherited_gpu_locks.py","--project-root",ROOT,"--asset-fd","7"],deadline=30)
  loader=base.RUNS/"scripts/livewan_h100_environment_r1.sh"
  code="import os,json;print(json.dumps({k:v for k,v in os.environ.items() if k.startswith(('LIVEWAN_','DISTILLSCHEDULE_')) or k in ('PATH','PYTHONPATH','HF_HOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','PYTHONDONTWRITEBYTECODE','TOKENIZERS_PARALLELISM')}))"
  values=subprocess.check_output(["bash","-c",'source "$1"; "$2" -c "$3"',"_",str(loader),str(base.PYTHON),code],text=True,timeout=30)
  env=base.clean_env();env.update(json.loads(values))
  relocation=P/"manifests/livewan_relocation_r2.json"
  env.update(DMD_SYSTEM_ROOT=str(W),DISTILLSCHEDULE_INHERITED_GPU_LOCKS="1",DISTILLSCHEDULE_ASSET_RELOCATION_RECEIPT=str(relocation),DISTILLSCHEDULE_ASSET_RELOCATION_SHA256=base.sha(relocation),DISTILLSCHEDULE_CAMPAIGN_ROOT=str(ROOT),PYTHONPATH=f'{env["LIVEWAN_PYTHON_OVERLAY"]}:{W}/src:{W}:{W}/third_party/livewan:{env["LIVEWAN_WAN21_SOURCE"]}',LIVEWAN_PIPELINE_VARIANT="c34-continuous-b16-residual-layout",LIVEWAN_BASELINE_PLACEMENT="legacy-hybrid",LIVEWAN_PIPELINE_TEACHER_PLACEMENT="full-resident",LIVEWAN_TEACHER_PARAMETER_OFFLOAD="off",LIVEWAN_TRAJECTORY_CYCLES="2",LIVEWAN_TRAJECTORY_SEED="20260830",LIVEWAN_TRAJECTORY_ARM_ID="h100-c34-teacher-fullresident",LIVEWAN_TWO_CYCLE_SEED="20260830",LIVEWAN_EXPECTED_HEAD=HEAD)
  env["LIVEWAN_TWO_CYCLE_ARM_ID"]=env["LIVEWAN_TRAJECTORY_ARM_ID"]
  base.dump(out/"asset_identity.json",{k:{"path":env[k],"sha256":base.sha(pathlib.Path(env[k]))} for k in ("LIVEWAN_OPERATION_TRACE","LIVEWAN_SYSTEMS_BOOTSTRAP","LIVEWAN_SOURCE_RECEIPT")})
  launcher=W/"scripts/distillschedule/launch_livewan_crossscale_world16_node.sh"
  validate=out/"validate";rid="livewan_h100_teacher_resident_validate_20261015_r2"
  runner.invoke(rid,["bash",launcher,"0",rid,"validate","pipeline",HEAD,"127.0.0.1","29734"],dict(env,DISTILLSCHEDULE_OUTPUT_DIR=str(validate)),deadline=300)
  x=json.load(open(validate/"validate_only.node0.json"));assert x["status"]=="PASS" and x["gpu_workload"]=="NOT_RUN"
  run=out/"trajectory";rid="livewan_h100_teacher_resident_capacity_20261015_r2"
  runner.invoke(rid,["bash",launcher,"0",rid,"trajectory","pipeline",HEAD,"127.0.0.1","29734"],dict(env,DISTILLSCHEDULE_OUTPUT_DIR=str(run)),deadline=1200,gpu=True)
  summary=run/"trajectory_summary.json"
  runner.invoke("summarize_native_trajectory",[base.PYTHON,W/"scripts/distillschedule/summarize_livewan_trajectory_world16.py","--run-root",run,"--expected-head",HEAD,"--hardware-profile","h100-8-b16","--cycles","2","--output",summary],env,deadline=180)
  x=json.load(open(summary));assert x["status"]=="PASS" and x["final_versions"]=={"generator":2,"critic":10,"ema":0}
  sys.path.insert(0,str(W/"src"));os.environ["DISTILLSCHEDULE_HARDWARE_PROFILE"]="h100-8-b16"
  from distillschedule.runtime.livewan_pipeline_placement import audit_run
  from distillschedule.runtime.livewan_hardware import livewan_hardware
  binding=audit_run(run,mode="pipeline",hardware=livewan_hardware())
  assert len(binding["rank_bindings"])==8
  base.dump(out/"teacher_storage_binding.json",binding)
  r.update(status="PASS_W4_TEACHER_RESIDENT_CAPACITY_ONLY",summary=str(summary),summary_sha256=base.sha(summary),final_versions=x["final_versions"],teacher_storage_binding=str(out/"teacher_storage_binding.json"),mathematical_and_numerical_comparison="NOT_RUN_NEXT_FRESH_SOURCE_MATCHED_CONTROLS_REQUIRED")
 except BaseException as exc:r.update(error=repr(exc),traceback=traceback.format_exc())
 finally:signal.alarm(0)
 r["utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat();base.dump(out/"result.json",r);print(json.dumps(r),flush=True)
 return 0 if r["status"].startswith("PASS_") else 1
if __name__=="__main__":raise SystemExit(main())
