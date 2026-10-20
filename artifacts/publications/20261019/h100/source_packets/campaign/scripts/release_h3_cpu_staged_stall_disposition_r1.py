from pathlib import Path
import json,subprocess,os,datetime
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930');O=R/'h100_resident_baselines_20261014_r1';F=R/'h3_h100_three_arm_matched_20261014_r4/03_correctness_candidate'
x=json.load(open(F/'summary.json'));assert x['status']=='FAIL_ARM'
assert not x['owned_alive_after_cleanup'] and not x['teardown_observation_errors'] and x['cgroup_oom_delta']=={'oom':0,'oom_kill':0}
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert (F/'manifests/owned_stall_early_stop_r1.json').exists()
assert json.load(open(O/'queue_r5/016_h3_BF16_verified_three_arm_matrix.failure.json'))['exit_code']!=0
for s in ('016x_h3_cuda_attention_library_gate','016y_h3_cuda_attention_residency_recovery'):
 assert json.load(open(O/f'queue_r5/{s}.ready.json'))['status']=='PASS_CPU_ADMISSION'
note={'status':'FAIL_STALLED_CPU_STAGED_CORRECTNESS_PRESERVED','run_id':'h3_h100_three_arm_matched_20261014_r4_03_correctness_candidate','source_head':'9bfd3eda959bd0c8415fe71ba64d3ad5f7e93e71','timing':'NOT_RUN','cgroup_oom_delta':x['cgroup_oom_delta'],'CUDA_OOM':'NOT_OBSERVED','GPU_cleanup':'PASS_ZERO_CONTEXTS','root_cause':'UNRESOLVED_WAIT_STALL; repeated fixed low-power GPU states/rank2 zero and all main threads futex, GDB ptrace unavailable','successful_FSDP_control_correctness_and_timing':'PRESERVED_NOT_INVALIDATED','FSDP_control_full5_seconds':638.522138278,'recovery':'only existing CUDA cache residency after real FA3 selective-checkpoint library gate','failed_cpu_staged_same_path_rerun':'NOT_ADMITTED_WITHOUT_NEW_EVIDENCE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (O/'reports/w5_cpu_staged_candidate_stall_r1.json').open('x') as f:json.dump(note,f,indent=2)
with (O/'research_log.jsonl').open('a') as f:f.write(json.dumps(dict(note,event='W5_CPU_STAGED_CANDIDATE_STALL_AND_EXISTING_GPU_CACHE_RECOVERY'))+'\n')
q=O/'queue_r5/016_h3_BF16_verified_three_arm_matrix.disposition.json'
p=q.with_suffix('.partial')
with p.open('x') as f:json.dump({'preserve_failed_artifacts':True,'action':'CONTINUE_OTHER_ADMITTED_STEPS','review_report':str(O/'reports/w5_cpu_staged_candidate_stall_r1.json'),'owned_cleanup':'PASS'},f,indent=2)
os.link(p,q);p.unlink()
print(json.dumps(note))
