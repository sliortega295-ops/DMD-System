from pathlib import Path
import json,subprocess,hashlib,importlib.util,sys,datetime,ast,os
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100');C=R/'research_artifacts/five_workload_h100_20260930';O=C/'h100_resident_baselines_20261014_r1';P=R/'envs/five-workload-h100-20260930/bin/python'
sys.path.insert(0,str(O/'scripts'))
import run_h3_h100_three_arm_r4 as original
g=original.load_guard();policy=g.read_hashed(g.PREPARED/'policy_freeze.json',g.POLICY_SHA);g.validate_policy(policy);g.verify_clean_sources()
prep=json.load(open(C/'h3_cuda_attention_cache_CPU_prepare_20261014_r1/status.json'))
assert prep['status']=='PASS_CPU_LIFECYCLE_PREPARATION_GPU_BOOTSTRAP_NOT_RUN'
for file in ('h3_cuda_attention_SAC_GPU_canary_r1.py','run_h3_cuda_attention_cache_r1.py','h3_h100_cuda_attention_recording_entry_r1.py'):
 ast.parse((O/'scripts'/file).read_text())
original_env=policy['candidate_environment'];assert original_env['H3_FA3_REPLAY_CACHE_STORAGE']=='cpu_staged' and original_env['H3_FA3_REPLAY_CACHE_TRIM_BEFORE_BACKWARD']=='1'
control=json.load(open(C/'h3_h100_three_arm_matched_20261014_r4/02_timing_control/summary.json'))
assert control['status']=='PASS_MATCHED_SEMANTIC_DIAGNOSTIC' and control['cgroup_oom_delta']=={'oom':0,'oom_kill':0}
# Runtime classes' existing CUDA mode and CUDA-specific census, not a new API.
env=g.make_environment(original_env,O/'artifacts/h3_three_arm_prepare_r4')
env['PYTHONPATH']=str(O/'scripts')+':'+env['PYTHONPATH'];env['CUDA_VISIBLE_DEVICES']=''
with (O/'logs/h3_existing_FA3_cuda_policy_CPU_tests_r1.log').open('x') as f:
 subprocess.run([str(P),'-m','pytest','-q',str(g.REPO/'tests/test_fa3_replay_cache.py')],env=env,cwd=g.REPO,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
GIB=1024**3
card={'id':'W5_H100_EXISTING_GPU_FA3_CACHE_RESIDENCY_RECOVERY_R1','status':'GO_BOUNDED_EXISTING_CUDA_POLICY_AFTER_REAL_LIBRARY_GATE',
 'gate':'H100CPU-staged candidate stalled; preserve failure, no repeat of same host queue path',
 'authorized_hypothesis':'GPU-resident out+LSE uses the same actual compactFA3 + backward + checkpoint replay but removes host cache queues/PCIe copies',
 'frozen':{'source':g.HEAD,'logical_batch':16,'GAS':2,'cycles':1,'sigma':'same approximate Grid1000','segment':1,'boundary_offload':False,'Student_Fake_updates':[1,5]},
 'source_policy_changes':{'H3_FA3_REPLAY_CACHE_STORAGE':['cpu_staged','cuda'],'H3_FA3_REPLAY_CACHE_TRIM_BEFORE_BACKWARD':['1','0']},
 'trim_explanation':'existing CPU-only allocator trim explicitly rejects CUDA storage; inapplicable trim disabled as part of one residency mechanism',
 'model_math_change':False,'new_optimization_code':'NONE; original existing CUDA cache mode',
 'reference_seconds':control['full5_seconds'],'target_seconds':control['full5_seconds']/1.5,
 'existing_FSDP_control_peak_GiB':control['peak_allocated_bytes']/GIB,
 'CPU_cache_max_reservoir_GiB':27489280000/GIB,
 'conservative_added_cache_peak_GiB_estimate':(control['peak_allocated_bytes']+27489280000)/GIB,
 'capacity_estimate_boundary':'upper-envelope additive estimate only; actual transient/lifetime overlap and allocator peak NOT_RUN',
 'movement_exposed_time':'UNKNOWN_NOT_PROFILED; never equate copy bytes with removable wall',
 'CPU_gate':'tests/test_fa3_replay_cache.py actual private CPU suite plus actual observer cases PASS',
 'cheap_GPU_gate':'three real FA3 shape cases, real non-reentrant selective checkpoint; output/QKV gradient bitwise equality before training',
 'budget':{'library_seconds':180,'real_correctness_seconds':1500,'real_timing_seconds':1500,'max_expensive_attempts':2},
 'stop':'any source/census/replay/finite/parity/cleanup/OOM failure; do not launch real model after failed library primitive',
 'rejected':['same stalled CPU-staged path repeat without new evidence','segment8','threshold offload','lower workload','global env/driver change'],
 'root_cause_of_CPU_stall':'UNRESOLVED, do not call it a capacity/OOM failure',
 'claim_boundary':'single-run approximate Grid H100commonstart diagnostics, strict crossarm numerical state NOT_RUN',
 'GPU':'NOT_RUN','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def ready_atomic(path,value):
 t=path.with_suffix('.partial');save(t,value);os.link(t,path);t.unlink()
save(O/'manifests/w5_cuda_cache_residency_recovery_admission_r1.json',card)
# Library script uses the identical source/overlay/FA3 asset environment.
script='''#!/usr/bin/env bash
set -Eeuo pipefail
R=/home/jiangzhikun/yongyan_liu/Helios-H100
O="$R/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1"
"$R/envs/five-workload-h100-20260930/bin/python" - "$O" <<'LIBRARY'
from pathlib import Path
import os,json,sys,subprocess
O=Path(sys.argv[1]);sys.path.insert(0,str(O/'scripts'))
import run_h3_h100_three_arm_r4 as original
G=original.load_guard();G.verify_clean_sources();policy=G.read_hashed(G.PREPARED/'policy_freeze.json',G.POLICY_SHA)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True);assert not apps.strip()
out=O/'artifacts/h3_FA3_cuda_attention_SAC_GPU_gate_r1'
assert not out.exists()
env=G.make_environment(policy['candidate_environment'],out,'0');env['H3_CUDA_ATTENTION_PRIMITIVE_OUTPUT']=str(out)
log=O/'logs/h3_FA3_cuda_attention_SAC_GPU_gate_r1.log'
def events():return {k:int(v) for k,v in map(str.split,Path('/sys/fs/cgroup/memory.events').read_text().splitlines())}
before=events()
with log.open('x') as f:rc=subprocess.run([str(G.PYTHON),str(O/'scripts/h3_cuda_attention_SAC_GPU_canary_r1.py')],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180).returncode
after=events();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
result={'launcher_rc':rc,'cgroup_oom_delta':{k:after[k]-before[k] for k in ('oom','oom_kill')},'GPU_contexts_after':apps,'model':'NOT_LOADED'}
with (O/'manifests/h3_FA3_cuda_attention_SAC_GPU_gate_health_r1.json').open('x') as f:json.dump(result,f,indent=2)
assert rc==0 and not apps.strip() and not any(result['cgroup_oom_delta'].values())
x=json.load(open(out/'receipt.json'));assert x['status']=='PASS_REAL_H100_FA3_CUDA_CACHE_OUTPUT_AND_GRADIENT_BITWISE'
print(json.dumps(x))
LIBRARY
'''
script2='''#!/usr/bin/env bash
set -Eeuo pipefail
R=/home/jiangzhikun/yongyan_liu/Helios-H100
O="$R/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1"
"$R/envs/five-workload-h100-20260930/bin/python" - "$O" <<'GATE'
import sys,json
from pathlib import Path
O=Path(sys.argv[1]);x=json.load(open(O/'artifacts/h3_FA3_cuda_attention_SAC_GPU_gate_r1/receipt.json'))
assert x['status']=='PASS_REAL_H100_FA3_CUDA_CACHE_OUTPUT_AND_GRADIENT_BITWISE'
assert len(x['cases'])==3 and all(v['QKV_gradients']=='BITWISE_EQUAL' and v['FA3_cache_CPU_copies']==0 for v in x['cases'])
GATE
exec "$R/envs/five-workload-h100-20260930/bin/python" "$O/scripts/run_h3_cuda_attention_cache_r1.py" --run-id h3_h100_cuda_attention_cache_20261014_r1 --execute --bootstrap "$R/research_artifacts/five_workload_h100_20260930/h3_three_arm_common_bootstrap_build_20261014_r2/bootstrap/manifest.json" --bootstrap-sha256 8181204333a373679eae03a3af6fcd03026c9199a6100e470f13c866a041e86f
'''
for name,text in (('016x_h3_cuda_attention_library_gate',script),('016y_h3_cuda_attention_residency_recovery',script2)):
 q=O/'queue_r5'/name
 with q.with_suffix('.sh').open('x') as f:f.write(text)
 subprocess.run(['bash','-n',str(q.with_suffix('.sh'))],check=True)
 ready_atomic(q.with_suffix('.ready.json'),{'status':'PASS_CPU_ADMISSION','script_sha256':hashlib.sha256(text.encode()).hexdigest(),'workload_math_changed':False,'source_head':G.HEAD if 'G' in globals() else g.HEAD,'admission':str(O/'manifests/w5_cuda_cache_residency_recovery_admission_r1.json')})
print(json.dumps({'status':'GPU_CACHE_RECOVERY_ADMITTED_CPU_ONLY','source':g.HEAD,'reference_seconds':control['full5_seconds'],'estimated_capacity_GiB':card['conservative_added_cache_peak_GiB_estimate'],'GPU':'NOT_RUN'}))
