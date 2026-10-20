"""Current-source named Resident capacity fallback. Inherit R6 locks, never preempt."""
import datetime,json,os,pathlib,signal,subprocess,sys,traceback
O=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1")
P=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/artifacts/w6_resident_adaln16_staging_20261015_r1")
W=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/repos/DMD-System-h100-w6-lora-unit-inplace-20261015-r1")
HEAD="0419b96b9ed9a8dd76737042e6c5020c80f6673a"
sys.path.insert(0,str(O/'scripts'))
import run_w6_lora_unit_capacity_r1 as old
base=old.base
base.REPO=W;base.HEAD=HEAD;base.WORKER=W/'scripts/distillschedule/probe_h3_diffsynth_h100_world8.py'
base.ACTIVATE=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/artifacts/w6_resident_adaln16_staging_20261015_r1/scripts/environment_r1.sh")
sys.path.insert(0,str(O/'artifacts/w6_h100_native_cache_residency_20261015_r1/scripts'))
from observe_gpu_lease_r1 import LeaseObserver
def validate(out,tiny):
 for rank in range(8):
  row=json.load(open(out/f'rank_receipts/rank_{rank:02d}.json'))
  assert row['named_capacity_fallback']=='resident_boundary_cpu_frozen_adaln16_pinned_staging'
  assert row['activation_offload'] is True and row['parameter_offload'] is True and not row['model_swap']
  assert row['precision']==dict(base_storage='BF16',LoRA_storage='FP32',autocast='BF16_cache_enabled_false',reduction='FP32_SUM_then_div8')
  for cycle in row['cycles']:
   b=cycle['checkpoint_boundary'];c=b['counters'];n=4 if tiny else 100
   assert b['enabled'] and b['recompute_hook_scope']=='outside_offload_wrapper'
   assert b['calls_by_role']==dict(student=n,fake_score=n) and c['checkpoint_calls']==2*n
   assert c['D2H_count']==c['H2D_count']>=2*n
   assert c['D2H_bytes']==c['H2D_bytes']>0
   cast=cycle['lora_input_cast_elision'];d=cast['cycle_delta']
   assert d['cuda_grad_elided']>0 and d['cuda_nograd_elided']>0 and not cast['parameter_dtype_changed']
   assert not any(k.startswith('cpu_') and v for k,v in d.items())
   unit=cycle['lora_unit_sum'];u=unit['counters']
   assert unit['enabled'] and unit['input_selector_composed'] and unit['A_B_module_calls_preserved']
   assert u['grad_forward']>0 and u['nograd_forward']>0
   assert u['physical_dual_lora_forward']==u['elided_unit_multiply']==u['inplace_base_add']
   assert not unit['parameter_dtype_changed'] and not unit['optimizer_or_loss_changed']
   a=cycle['resident_adaln_CPU'];k=2 if tiny else 16
   assert a['selected_blocks']==k and a['parameter_offload'] is True
   assert a['calls_by_role']==dict(student=20*k,fake_score=6*k,teacher=2*k)
   assert a['counters']['linear_calls']==28*k and a['counters']['H2D_count']==56*k
   assert a['counters']['H2D_bytes']==28*a['canonical_CPU_bytes']>0
   assert not a['optimizer_or_loss_changed'] and not a['parameter_values_or_dtype_changed']
 return dict(status='PASS_ACTUAL_NATIVE_BOUNDARY_LORA_ADALN_STAGING_COUNTERS',ranks=8,scope='runtime/census, not full-model strict cross-arm parity')
def stop(sig,frame):raise InterruptedError('bounded Resident '+str(sig))
def cpu_source_gate():
 base.source_gate()
 proof=json.load(open(P/'manifests/committed_source_r1.json'))
 assert proof['status']=='PASS_EXACT_COMMITTED_CPU_TESTED_SOURCE' and proof['head']==HEAD
 assert proof['tests']==dict(tests=100,errors=0,failures=0,skipped=0)
 for name,digest in proof['files'].items():assert base.sha(W/name)==digest
 return dict(status='PASS_ACTUAL_CPU_SOURCE_ADMISSION_ONLY',head=HEAD,tests=proof['tests'],GPU='NOT_RUN')
def main():
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
 signal.alarm(7200)
 os.environ['SUDO_UID']=str(W.stat().st_uid)
 assert os.environ.get('DISTILLSCHEDULE_INHERITED_GPU_LOCKS')=='1'
 subprocess.run([str(base.PYTHON),str(W/'scripts/distillschedule/verify_inherited_gpu_locks.py'),'--project-root',str(base.ROOT),'--asset-fd','7'],check=True,pass_fds=(7,8,9))
 out=P/'capacity_and_single_timing_r1';out.mkdir(exist_ok=False)
 report=dict(status='FAIL_PRESERVED',head=HEAD,run_id=out.name,arms=[],physical_GPUs=8,logical_slots=16,sigma='native continuous shifted logit-normal',Grid1000='NOT_APPLICABLE',named_fallback='resident_checkpoint_boundary_CPU_plus_frozen_AdaLN16_parameter_CPU_staging',full_model_strict_cross_arm_parity='NOT_RUN',formal_speedup_eligibility=False,profiled=False,max_real_attempts=1)
 try:
  cpu_source_gate()
  import shutil
  assert shutil.disk_usage(P).free>32*(1<<30),'PERSISTENT_STORAGE_UNDER32GiB'
  assert not base.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
  primitive=P/'manifests/GPU_frozen_adaln_primitive_r1.json'
  old.own_command(['bash',str(base.ACTIVATE),str(P/'scripts/GPU_frozen_adaln_primitive_r1.py'),str(primitive)],out/'GPU_primitive',gpu=True)
  report['GPU_primitive']=json.load(open(primitive))
  assert report['GPU_primitive']['status']=='PASS_ACTUAL_FULL_SHAPE_FROZEN_ADALN_CUDA_STAGED_BITWISE'
  inputs=json.load(open(base.INPUT))
  assert inputs['status']=='PASS_CPU_NATIVE_16_REAL_ROWS_MODEL_LAYOUT'
  assert base.sha(inputs['cache_manifest'])==inputs['cache_manifest_sha256'] and base.sha(inputs['config_path'])==inputs['config_sha256']
  base.WORKER_EXTRA_ARGS=('--checkpoint-boundary-cpu','--lora-input-cast-elision','--lora-unit-inplace','--resident-adaln-cpu-blocks','16')
  tiny,path=base.arm(out,'resident','correctness',inputs,True)
  report['arms'].append(dict(name='tiny_native_boundary_composition',path=str(path),**tiny))
  assert tiny['status']=='PASS_TINY_8_RANK_INTEGRATION_ONLY','TINY_RUNTIME_FAIL_NO_REAL_MODEL'
  report['tiny_policy_gate']=validate(path,True)
  old.compare_tiny(out,path)
  report['tiny_state_parity']=json.load(open(out/'tiny_state_parity.json'))
  assert report['tiny_state_parity']['status']=='PASS_TINY_NATIVE_PARAMETERS_ADAM','TINY_STRICT_FAIL_NO_REAL_MODEL'
  for mode in ('correctness','timing'):
   observer=LeaseObserver(out/(mode+'_GPU_lease_samples.jsonl'));observer.start()
   try:summary,path=base.arm(out,'resident',mode,inputs,False)
   finally:observer.stop()
   report['arms'].append(dict(name='resident_'+mode,path=str(path),**summary))
   assert summary['status']=='PASS_REAL_8_RANK_16_LOGICAL_SLOT_NATIVE_GATE',mode+'_CAPACITY_RUNTIME_FAIL'
   gate=validate(path,False);base.write(path/'manifests/boundary_lora_gate.json',gate)
   observed=observer.receipt(path/'rank_receipts');base.write(path/'manifests/GPU_lease_gate.json',observed)
   assert observed['status']=='PASS_SAMPLED_8_STABLE_WORKER_PID_SET_8_GPUS'
  report.update(status='PASS_RESIDENT_ADALN_CPU_NAMED_CAPACITY_FALLBACK_SINGLE_DIAGNOSTIC',single_C2_seconds=report['arms'][-1]['timed_C2_seconds'],claim_boundary='one excluded C1warmup plus C2; runtime/census and actual tiny parameter/Adam parity only, strict full-model cross-arm NOT_RUN; CPU-parameter offload fallback, NOT no-offload Resident or pureplacement or formal speedup')
 except BaseException as exc:
  report.update(error=repr(exc),traceback=traceback.format_exc())
 finally:signal.alarm(0)
 report['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 # Keep large tiny parity rows in their original file, not duplicate into summary.
 if 'tiny_state_parity' in report:
  report['tiny_state_parity']={k:v for k,v in report['tiny_state_parity'].items() if k!='rows'}
 base.write(out/'result.json',report)
 base.write(P/'manifests/capacity_and_timing_index_r1.json',dict(status=report['status'],head=HEAD,result=str(out/'result.json'),sha256=base.sha(out/'result.json'),single_C2_seconds=report.get('single_C2_seconds'),formal_speedup_eligibility=False))
 print(json.dumps({k:v for k,v in report.items() if k not in ('arms','GPU_primitive')}),flush=True)
 return 0 if report['status'].startswith('PASS_') else 1
if __name__=='__main__':raise SystemExit(main())
