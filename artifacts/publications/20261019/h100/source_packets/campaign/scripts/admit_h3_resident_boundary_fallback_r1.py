#!/usr/bin/env python3
"""Derive the actual Resident OOM and admit one named, existing-policy fallback."""
from pathlib import Path
import datetime,hashlib,json,re,subprocess
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
C=R/'research_artifacts/five_workload_h100_20260930'
O=C/'h100_resident_baselines_20261014_r1'
P=R/'envs/five-workload-h100-20260930/bin/python'
F=C/'h3_h100_three_arm_matched_20261014_r4/00_correctness_resident'
repo=R/'repos/H3-DMD-for-A100-h100-three-arm-20261014-r1'
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
assert head=='9bfd3eda959bd0c8415fe71ba64d3ad5f7e93e71'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
summary=json.loads((F/'summary.json').read_text())
assert summary['status']=='FAIL_ARM' and summary['launcher_exit_code']==1
assert summary['cgroup_oom_delta']=={'oom':0,'oom_kill':0}
assert summary['owned_alive_after_cleanup']==[] and summary['teardown_observation_errors']==[]
rows=[]
for rank in range(8):
 path=F/f'rank_receipts/rank_{rank:03d}.json'
 x=json.loads(path.read_text());assert x['rank']==rank
 assert x['error_type']=='OutOfMemoryError' and 'CUDA out of memory' in x['error']
 assert 'trainer_loop.py' in x['traceback'] and 'loss_fake / len(group)' in x['traceback']
 rows.append({'rank':rank,'receipt':str(path),'receipt_sha256':sha(path),
              'observed_optimizer_commits':x['observed_optimizer_commits'],
              'failure_api':'Fake backward checkpoint replay → FFN LoRA-B Linear',
              'transformer_block':'UNKNOWN_NOT_IN_FAILURE_TRACE',
              'allocation_error':x['error']})
assert all(x['observed_optimizer_commits']=={'student':1,'fake':1} for x in rows)
report={'run_id':'h3_h100_three_arm_matched_20261014_r4_00_correctness_resident',
 'status':'FAIL_CUDA_OOM_PRESERVED','source_head':head,
 'physical_ranks_failed':8,'logical_global_batch':16,'activation_policy':'none',
 'AdaLN_registered_parameters':0,'AdaLN_dropped_numel':13010457600,
 'last_completed_updates':{'student':1,'fake':1},'failure_stage':'second Fake optimizer update, checkpoint backward replay',
 'attempted_allocation_GiB':2.02,'allocated_at_failure_GiB_rounded':76.17,
 'driver_free_rank0_MiB_rounded':770.94,'node_cgroup_observed_peak_bytes':summary['cgroup_current_observed_peak'],
 'cgroup_oom_delta':summary['cgroup_oom_delta'],'owned_process_cleanup':'PASS',
 'timing':'NOT_RUN','full5_correctness':'FAIL_NOT_COMPLETED','per_rank':rows,
 'root_cause_boundary':'PROVEN CUDA capacity failure after full static Resident + Adam/grad/boundary/transient workload; exact additive object attribution UNRESOLVED',
 'rejected_paths':['blind segment8','generic threshold offload','dropping model/frames/rank/global batch'],
 'recovery_candidate':'existing checkpoint_boundary_cpu policy, distinct named capacity fallback; full-model GPU residency retained',
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
save(O/'reports/w5_resident_actual_capacity_failure_r4.json',report)
(O/'reports/w5_resident_actual_capacity_failure_r4.md').write_text('# W5 pure Resident capacity failure\n\n8/8 physical ranks CUDA OOM during the second Fake backward checkpoint replay, after Student1/Fake1 optimizer commits. FFN LoRA-B attempted 2.02 GiB; observed allocated 76.17 GiB and rank0 free 770.94 MiB.\n\n13,010,457,600 AdaLN parameters already removed. No cgroup OOM. Original run, traces, receipts and logs retained. Full5 correctness and timing did not complete. The specific block is UNKNOWN.\n\nExisting checkpoint_boundary_cpu is admitted as a separately named capacity fallback. This is not a pure no-offload Resident timing or isolated parameter-placement comparison. No segment8/threshold/shape/batch/precision changes.\n')
prep=json.loads((C/'h3_resident_boundary_fallback_CPU_prepare_20261014_r1/status.json').read_text())
assert prep['status']=='PASS_CPU_LIFECYCLE_PREPARATION_GPU_BOOTSTRAP_NOT_RUN'
for name in ('h3_h100_boundary_recording_entry_r1.py','run_h3_resident_boundary_fallback_r1.py','test_h3_boundary_fallback_observer_r1.py'):
 subprocess.run([str(P),'-c','import ast,sys;ast.parse(open(sys.argv[1]).read())',str(O/'scripts'/name)],check=True)
subprocess.run([str(P),str(O/'scripts/test_h3_boundary_fallback_observer_r1.py')],check=True)
note={'status':'GO_BOUNDED_EXISTING_POLICY_CAPACITY_FALLBACK','gate':'W5 Resident recovery after actual current-head CUDA OOM',
 'authorized_hypothesis':'native per-block checkpoint boundary CPU staging removes GPU saved boundaries without changing 31F/6B per sample',
 'frozen':{'head':head,'bootstrap_sha256':'8181204333a373679eae03a3af6fcd03026c9199a6100e470f13c866a041e86f','logical_batch':16,'GAS':2,'sigma':'Grid1000 approximate unchanged','checkpoint_segment':1},
 'matching_prior_evidence':str(F/'summary.json'),
 'scope':'ONLY H3_ACTIVATION_POLICY=checkpoint_boundary_cpu; original source implementation; receipt instrumentation post-timer',
 'rejected_paths':report['rejected_paths'],
 'new_falsifiable_question':'Does full Resident complete G1/F5 on8H100 after600 physical per-rank boundary CPU copies?',
 'expensive_attempt_budget':1,'promote_only':'8/8 census/replay/boundary/finite/no-OOM gates',
 'comparison_boundary':'memory policy differs from original none control; no pure-placement speedup claim',
 'CPU_admission':str(C/'h3_resident_boundary_fallback_CPU_prepare_20261014_r1/status.json'),
 'GPU_status':'NOT_RUN','source_sha256':{n:sha(O/'scripts'/n) for n in ('h3_h100_boundary_recording_entry_r1.py','run_h3_resident_boundary_fallback_r1.py')}}
save(O/'manifests/w5_resident_boundary_fallback_admission_r1.json',note)
script='''#!/usr/bin/env bash
set -Eeuo pipefail
R=/home/jiangzhikun/yongyan_liu/Helios-H100
O="$R/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1"
exec "$R/envs/five-workload-h100-20260930/bin/python" "$O/scripts/run_h3_resident_boundary_fallback_r1.py" --run-id h3_h100_resident_boundary_fallback_20261014_r1 --execute --bootstrap "$R/research_artifacts/five_workload_h100_20260930/h3_three_arm_common_bootstrap_build_20261014_r2/bootstrap/manifest.json" --bootstrap-sha256 8181204333a373679eae03a3af6fcd03026c9199a6100e470f13c866a041e86f
'''
q=O/'queue_r5/016z_h3_resident_boundary_fallback'
with q.with_suffix('.sh').open('x') as f:f.write(script)
subprocess.run(['bash','-n',str(q.with_suffix('.sh'))],check=True)
save(q.with_suffix('.ready.json'),{'status':'PASS_CPU_ADMISSION','script_sha256':sha(q.with_suffix('.sh')),'workload_math_changed':False,'admission':str(O/'manifests/w5_resident_boundary_fallback_admission_r1.json'),'source_head':head})
with (O/'research_log.jsonl').open('a') as f:f.write(json.dumps({'utc':report['utc'],'event':'W5_PURE_RESIDENT_CUDA_OOM_AND_NAMED_BOUNDARY_CAPACITY_RECOVERY_ADMITTED','failure_report':str(O/'reports/w5_resident_actual_capacity_failure_r4.json'),'admission':str(O/'manifests/w5_resident_boundary_fallback_admission_r1.json'),'old_run_preserved':True,'scope':'existing boundary policy only, math unchanged'})+'\n')
print(json.dumps({'status':'ADMITTED_NO_GPU_YET','failure_report':str(O/'reports/w5_resident_actual_capacity_failure_r4.json'),'queue':str(q)}))
