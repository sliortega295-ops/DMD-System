from pathlib import Path
import json,os,hashlib,subprocess,datetime,ast
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100');C=R/'research_artifacts/five_workload_h100_20260930';O=C/'h100_resident_baselines_20261014_r1'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
old=C/'h3_h100_resident_boundary_fallback_20261014_r1/00_correctness_resident'
log=(old/'logs/launcher.log').read_text();assert 'activation policy=none checkpoint_segment=1' in log
assert json.load(open(old/'manifests/environment.json'))['H3_ACTIVATION_POLICY']=='checkpoint_boundary_cpu'
summary=json.load(open(old/'summary.json'));assert not summary['owned_alive_after_cleanup'] and summary['cgroup_oom_delta']=={'oom':0,'oom_kill':0}
assert json.load(open(C/'h3_resident_boundary_launcher_fixed_CPU_prepare_20261014_r2/status.json'))['status']=='PASS_CPU_LIFECYCLE_PREPARATION_GPU_BOOTSTRAP_NOT_RUN'
assert 'PASS_CPU_NAMED_POLICY_RENDER_AND_REAL_FD_INHERITANCE' in (O/'logs/launch_delivery_FD_CPU_r2.log').read_text()
primitive=json.load(open(O/'artifacts/w6_attention_transport_GPU_canary_r2/receipt.json'));assert primitive['status']=='PASS_DETACHED_NOGRAD_NATIVE_FLASH_FIXED_GRADIENT_PARITY'
assert primitive['source_QKV_requires_grad'] is False and all(x['status']=='PASS' for x in primitive['QKV_gradient_metrics'])
oldtiny=json.load(open(C/'h100_w6_native_B16_GPU_tiny_gate_20261014_r1/result.json'))
assert oldtiny['GPU']=='NOT_RUN' and 'Bad file descriptor' in (O/'queue_r5/017r2_w6_actual_nograd_tiny_gate.launcher.log').read_text()
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def atomic(p,v):
 t=p.with_suffix('.partial');save(t,v);os.link(t,p);t.unlink()
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'MINIMAL_LAUNCHER_BUGS_FIXED_CPU_PASS_GPU_PENDING','W5':{'failed_run':str(old),'declared_policy':'checkpoint_boundary_cpu','actual_policy':'none','failure_class':'CONFIGURATION_DELIVERY_ERROR_REPRODUCED_KNOWN_RESIDENT_CUDA_OOM','true_offload_capacity':'NOT_RUN_IN_R1','fix':'owned rendered launcher replaces its forced none export with named boundary policy; observer asserts actual policy/segment before setup','core_training_head':'9bfd3eda959bd0c8415fe71ba64d3ad5f7e93e71','training_math_changed':False},'W6':{'failed_run':str(C/'h100_w6_native_B16_GPU_tiny_gate_20261014_r1'),'GPU':'NOT_RUN','fix':'only pass_fds=(7,8,9) to existing lease verifier subprocess; never bypass verifier','primitive':'completed detached no-grad output AND QKV gradients bitwise PASS, reused without rerun','training_worker_head':'3003a9903a96e0ccaafa10faec956e1846e2448e','training_math_changed':False},'CPU_tests':'actual OS defaultFD loss reproduced; explicit inheritance PASS; rendered boundaryCPU policy/source binding PASS','shared_environment_changes':False,'old_runs_overwritten':False}
save(O/'reports/launcher_delivery_and_FD_recovery_r2.json',report)
# The failed primitive+tiny queue never constructed a distributed model.
# Preserve it and supersede the real queue's dependency on failed tinyR1.
save(O/'queue_r5/018r2_w6_native_three_arm_matrix.failure.json',{'status':'NOT_RUN_DEPENDENCY_CPU_LAUNCHER_FD_FAILURE','replacement':'018r3_w6_native_three_arm_FD_fixed','GPU':'NOT_RUN'})
text=(O/'queue_r5/016z_h3_resident_boundary_fallback.sh').read_text().replace('run_h3_resident_boundary_fallback_r1.py','run_h3_resident_boundary_fallback_r2.py').replace('h3_h100_resident_boundary_fallback_20261014_r1','h3_h100_resident_boundary_fallback_20261014_r2')
text2='''#!/usr/bin/env bash
set -Eeuo pipefail
R=/home/jiangzhikun/yongyan_liu/Helios-H100
O="$R/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1"
"$R/envs/five-workload-h100-20260930/bin/python" - "$O" <<'PRIMITIVE'
import json,sys
from pathlib import Path
O=Path(sys.argv[1]);x=json.load(open(O/'artifacts/w6_attention_transport_GPU_canary_r2/receipt.json'))
assert x['status']=='PASS_DETACHED_NOGRAD_NATIVE_FLASH_FIXED_GRADIENT_PARITY' and x['source_QKV_requires_grad'] is False
assert all(v['status']=='PASS' for v in x['QKV_gradient_metrics'])
PRIMITIVE
exec "$R/envs/five-workload-h100-20260930/bin/python" "$O/scripts/run_w6_native_three_arm_r2.py" --run-id h100_w6_native_B16_GPU_tiny_gate_20261014_r2 --execute --tiny
'''
text3=(O/'queue_r5/018r2_w6_native_three_arm_matrix.sh').read_text().replace('h100_w6_native_B16_GPU_tiny_gate_20261014_r1','h100_w6_native_B16_GPU_tiny_gate_20261014_r2').replace('run_w6_native_three_arm_r1.py','run_w6_native_three_arm_r2.py').replace('h100_w6_native_three_arm_matched_20261014_r1','h100_w6_native_three_arm_matched_20261014_r2')
for name,s in (('016z2_h3_boundary_launcher_delivery_fixed',text),('017r3_w6_8rank_tiny_FD_fixed',text2),('018r3_w6_native_three_arm_FD_fixed',text3)):
 q=O/'queue_r5'/name
 with q.with_suffix('.sh').open('x') as f:f.write(s)
 subprocess.run(['bash','-n',str(q.with_suffix('.sh'))],check=True)
 atomic(q.with_suffix('.ready.json'),{'status':'PASS_CPU_ADMISSION','script_sha256':hashlib.sha256(s.encode()).hexdigest(),'workload_math_changed':False,'admission':str(O/'reports/launcher_delivery_and_FD_recovery_r2.json'),'W5_core_head':'9bfd3eda959bd0c8415fe71ba64d3ad5f7e93e71','W6_worker_head':'3003a9903a96e0ccaafa10faec956e1846e2448e'})
with (O/'research_log.jsonl').open('a') as f:f.write(json.dumps(dict(report,event='FIX_EFFECTIVE_BOUNDARY_LAUNCH_POLICY_AND_PRESERVE_FD_LEASE_VERIFICATION'))+'\n')
atomic(O/'queue_r5/017r2_w6_actual_nograd_tiny_gate.disposition.json',{'preserve_failed_artifacts':True,'action':'CONTINUE_OTHER_ADMITTED_STEPS','review_report':str(O/'reports/launcher_delivery_and_FD_recovery_r2.json'),'primitive_reused':True,'no_GPU_contexts':True})
print(json.dumps(report))
