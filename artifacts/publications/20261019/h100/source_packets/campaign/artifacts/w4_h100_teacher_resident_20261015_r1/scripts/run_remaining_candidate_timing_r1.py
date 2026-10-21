"""Resume only the missing W4 candidate; preserve completed baseline and gates."""
import datetime,hashlib,json,os,pathlib,signal,socket,subprocess,sys,traceback,types
ROOT=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100");O=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1");P=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/artifacts/w4_h100_teacher_resident_20261015_r1");W=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/repos/DMD-System-h100-livewan-teacher-resident-20261015-r1")
TRAIN_HEAD='4d7c589acd01a4a6c90ca141da266b3090df2b68';ANALYZER_COMMIT='812236caf73163824e20aa055dda0326fdfec258'
sys.path.insert(0,str(O/'scripts'));import run_livewan_resident_matched_r2 as base
base.REPO=W;base.HEAD=TRAIN_HEAD
def source_admit():
 proof=json.load(open(P/'manifests/timing_audit_committed_source_r1.json'))
 assert proof['status']=='PASS_ANALYZER_ONLY_COMMIT_AND_REAL_CPU_REAUDIT' and proof['analyzer_commit']==ANALYZER_COMMIT and proof['training_head']==TRAIN_HEAD
 assert base.sha(P/'scripts/summarize_livewan_two_cycle_timing_hw_audit_r1.py')==proof['analyzer_sha256']
 assert base.sha(P/'reports/timing_baseline_HW_reaudit_r1.json')==proof['baseline_reaudit_sha256']
 assert subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip()==TRAIN_HEAD
 assert not subprocess.check_output(['git','-C',str(W),'status','--porcelain'],text=True).strip()
 old=P/'artifacts/matched_resident_teacher_r1';baseline=json.load(open(P/'reports/timing_baseline_HW_reaudit_r1.json'))
 assert baseline['status']=='PASS' and baseline['head']==TRAIN_HEAD and baseline['rank_count']==8
 assert len(baseline['correctness_hardware_binding_audit'])==8 and all(x['source']=='HASH_VERIFIED_ORIGINAL_RESIDENT_GATE' for x in baseline['correctness_hardware_binding_audit'])
 for name in ('resident_timing_gate.json','timing_gate.json'):
  gate=json.load(open(old/name));assert gate['status']=='PASS'
  actual=gate.get('expected_head');assert actual==TRAIN_HEAD
 assert json.load(open(old/'trajectory_comparison.json'))['status']=='TRAJECTORY_REVIEW_50C_COMPLETED'
 return dict(status='PASS_ACTUAL_EXISTING_GATES_AND_BASELINE_AUDIT',GPU='NOT_RUN',training_head=TRAIN_HEAD,analyzer_commit=ANALYZER_COMMIT)
def main():
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,lambda s,f:(_ for _ in ()).throw(InterruptedError('ownedW4resume'+str(s))))
 signal.alarm(3600);assert os.environ.get('DISTILLSCHEDULE_INHERITED_GPU_LOCKS')=='1';source_admit()
 out=P/'artifacts/matched_candidate_resume_r1';out.mkdir(exist_ok=False);(out/'logs').mkdir()
 runner=object.__new__(base.Runner);runner.workload='livewan';runner.out=out;runner.ordinal=0;runner.records=[]
 def status(self,phase,state,**kw):base.dump(out/'status.json',dict(run_id='livewan_h100_teacher_resident_candidate_timing_resume_20261015_r1',phase=phase,status=state,training_head=TRAIN_HEAD,strict_claimable=False,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),immutable=False)
 runner.status=types.MethodType(status,runner)
 r=dict(status='FAIL_PRESERVED',training_head=TRAIN_HEAD,analyzer_commit=ANALYZER_COMMIT,new_GPU_arms=1,baseline_rerun=False,profiled=False,replications=1,strict_full_state_equivalence='REVIEW_UNCHANGED_NOT_STRICT_PASS')
 try:
  runner.invoke('inherited_lease',[base.PYTHON,W/'scripts/distillschedule/verify_inherited_gpu_locks.py','--project-root',ROOT,'--asset-fd','7'],deadline=30)
  loader=base.RUNS/'scripts/livewan_h100_environment_r1.sh'
  code="import os,json;print(json.dumps({k:v for k,v in os.environ.items() if k.startswith(('LIVEWAN_','DISTILLSCHEDULE_')) or k in ('PATH','PYTHONPATH','HF_HOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','PYTHONDONTWRITEBYTECODE','TOKENIZERS_PARALLELISM')}))"
  values=subprocess.check_output(['bash','-c','source "$1"; "$2" -c "$3"','_',str(loader),str(base.PYTHON),code],text=True,timeout=30)
  env=base.clean_env();env.update(json.loads(values));relocation=P/'manifests/livewan_relocation_r2.json';gate=P/'artifacts/matched_resident_teacher_r1/timing_gate.json'
  env.update(DMD_SYSTEM_ROOT=str(W),DISTILLSCHEDULE_INHERITED_GPU_LOCKS='1',DISTILLSCHEDULE_ASSET_RELOCATION_RECEIPT=str(relocation),DISTILLSCHEDULE_ASSET_RELOCATION_SHA256=base.sha(relocation),DISTILLSCHEDULE_CAMPAIGN_ROOT=str(ROOT),DISTILLSCHEDULE_HARDWARE_PROFILE='h100-8-b16',PYTHONPATH=f'{env["LIVEWAN_PYTHON_OVERLAY"]}:{W}/src:{W}:{W}/third_party/livewan:{env["LIVEWAN_WAN21_SOURCE"]}',LIVEWAN_PIPELINE_VARIANT='c34-continuous-b16-residual-layout',LIVEWAN_BASELINE_PLACEMENT='legacy-hybrid',LIVEWAN_PIPELINE_TEACHER_PLACEMENT='full-resident',LIVEWAN_TEACHER_PARAMETER_OFFLOAD='off',LIVEWAN_TRAJECTORY_CYCLES='2',LIVEWAN_TRAJECTORY_SEED='20260830',LIVEWAN_TRAJECTORY_ARM_ID='h100-pipeline-c34-continuous-b16-residual-layout',LIVEWAN_TWO_CYCLE_SEED='20260830',LIVEWAN_TWO_CYCLE_ARM_ID='h100-pipeline-c34-continuous-b16-residual-layout',LIVEWAN_EXPECTED_HEAD=TRAIN_HEAD,LIVEWAN_PAIR_CORRECTNESS_RECEIPT=str(gate),LIVEWAN_PAIR_CORRECTNESS_SHA256=base.sha(gate),DISTILLSCHEDULE_OUTPUT_DIR=str(out/'candidate_timing'))
  with socket.socket() as sock:sock.bind(('127.0.0.1',29735))
  rid='livewan_h100_teacher_resident_candidate_timing_resume_20261015_r1'
  runner.invoke(rid,['bash',W/'scripts/distillschedule/launch_livewan_crossscale_world16_node.sh','0',rid,'two-cycle-timing','pipeline',TRAIN_HEAD,'127.0.0.1','29735'],env,deadline=2400,gpu=True)
  analyzer=P/'scripts/summarize_livewan_two_cycle_timing_hw_audit_r1.py';summary=out/'candidate_timing_audited_summary.json'
  runner.invoke('summarize_actual_remaining_candidate',[base.PYTHON,analyzer,'--run-root',out/'candidate_timing','--expected-head',TRAIN_HEAD,'--hardware-profile','h100-8-b16','--execution-mode','pipeline','--pipeline-variant','c34-continuous-b16-residual-layout','--trace',env['LIVEWAN_OPERATION_TRACE'],'--output',summary],dict(env,CUDA_VISIBLE_DEVICES=''),deadline=300)
  a=json.load(open(P/'reports/timing_baseline_HW_reaudit_r1.json'));b=json.load(open(summary))
  assert b['status']=='PASS' and b['rank_count']==8 and b['final_versions']==dict(generator=2,critic=10,ema=0)
  for key in ('head','identity','matched_workload_fingerprint','final_versions','hardware_profile'):assert a[key]==b[key],'MATCHED_CONTRACT_'+key
  x=a['cycle_2']['seconds'];y=b['cycle_2']['seconds']
  r.update(status='PASS_W4_SAME_SOURCE_SINGLE_DIAGNOSTIC_PAIR',baseline_seconds=x,candidate_seconds=y,descriptive_ratio=x/y,baseline_summary=str(P/'reports/timing_baseline_HW_reaudit_r1.json'),candidate_summary=str(summary),claim_boundary='originalrawbaselinesummaryFAILpreserved; originalsame8ranktimer recoveredonlythroughhashboundHWmetadata. ExistingtrajectoryREVIEWkept; no strictfullstate claim, no profiledtime')
 except BaseException as exc:r.update(error=repr(exc),traceback=traceback.format_exc())
 finally:signal.alarm(0)
 r['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();base.dump(out/'result.json',r);print(json.dumps(r),flush=True)
 return 0 if r['status'].startswith('PASS_') else 1
if __name__=='__main__':raise SystemExit(main())
