from pathlib import Path
import json,hashlib,subprocess,os,datetime
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100');C=R/'research_artifacts/five_workload_h100_20260930';O=C/'h100_resident_baselines_20261014_r1'
x=json.load(open(C/'h3_named_boundary_registered_runner_CPU_prepare_20261014_r3/status.json'))
assert x['status']=='PASS_CPU_LIFECYCLE_PREPARATION_GPU_BOOTSTRAP_NOT_RUN' and x['source_head']=='d81ebb2014af9ea3a84011e3be198818e63cb1e7'
proof=json.load(open(O/'manifests/h3_named_boundary_contract_CPU_r3.json'));assert proof['status']=='PASS_ACTUAL_PRIVATE_CPU_TESTS_AND_UNCHANGED_MATH_SOURCE'
text='''#!/usr/bin/env bash
set -Eeuo pipefail
R=/home/jiangzhikun/yongyan_liu/Helios-H100
O="$R/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1"
exec "$R/envs/five-workload-h100-20260930/bin/python" "$O/scripts/run_h3_named_resident_boundary_r3.py" --run-id h3_h100_registered_resident_boundary_20261014_r3 --execute --bootstrap "$R/research_artifacts/five_workload_h100_20260930/h3_three_arm_common_bootstrap_build_20261014_r2/bootstrap/manifest.json" --bootstrap-sha256 8181204333a373679eae03a3af6fcd03026c9199a6100e470f13c866a041e86f
'''
q=O/'queue_r5/018s_w5_registered_resident_boundary_capacity'
with q.with_suffix('.sh').open('x') as f:f.write(text)
subprocess.run(['bash','-n',str(q.with_suffix('.sh'))],check=True)
meta={'status':'PASS_CPU_ADMISSION','script_sha256':hashlib.sha256(text.encode()).hexdigest(),'workload_math_changed':False,'source_head':x['source_head'],'new_contract':'h100-8-b16-resident-boundary-cpu','original_none_contract':'UNCHANGED_NO_OVERRIDE_ALLOWED','math_source_proof':str(O/'manifests/h3_named_boundary_contract_CPU_r3.json'),'initializer_bootstrap':'original immutable common artifact unchanged; source CPU initializer and replay functions unchanged','sample_update_contract':'8physical/16logical GAS2; G1/F5;600boundarycopies; same approximateGrid1000','comparison_boundary':'named memory fallback, not pure placement control; originalOOM and both launcher failures preserved','GPU':'NOT_RUN'}
t=q.with_suffix('.partial')
with t.open('x') as f:json.dump(meta,f,indent=2)
os.link(t,q.with_suffix('.ready.json'));t.unlink()
with (O/'research_log.jsonl').open('a') as f:f.write(json.dumps(dict(meta,event='ADMIT_DISTINCT_REGISTERED_H3_RESIDENT_BOUNDARY_CPU_PROFILE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
print(json.dumps(meta))
