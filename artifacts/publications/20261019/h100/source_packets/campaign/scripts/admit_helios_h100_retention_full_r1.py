from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess

R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
O=R/'research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1'
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def atomic(p,v):
    t=p.with_suffix('.partial');save(t,v);os.link(t,p);t.unlink()
cpu=json.load(open(O/'manifests/helios_retention_source_CPU_r1.json'));assert cpu['status']=='PASS_CPU_TESTS_AND_ACTUAL_GPU_API_RECEIPT_NO_MODEL'
proof=O/'artifacts/helios_hsdp_retention_api_GPU_r1/result.json'
assert hashlib.sha256(proof.read_bytes()).hexdigest()=='52f9b287011e681861731055b4b37ea7d65ab4cb39eb8ad0792025a7b0c23966'
report={'status':'GO_BOUNDED_SAME_REVISION_CONTROL_AND_RETENTION','head':cpu['head'],
        'hypothesis':'one-time unshard over five no-grad rollout batches reduces current end-to-end time at unchanged2x4 mesh/B1/global16',
        'current_best_seconds':98.21387330815196,'target_seconds':92.2479654972752,
        'measured_API_proof':'8rank tiny actual205→41 all-gather submissions, bitwise outputs and post-scope gradient/update,41 units restored',
        'real_gain_forecast':'UNKNOWN; hidden all-gather or allocator overhead can erase any benefit',
        'frozen':'same original trace/checkpoint, precision, loss, G1/C5/EMA5, all kernels and 2x4 placement; only retention boolean changes',
        'budget':'one validate and correctness per arm, then one unprofiled timing per PASS arm; no automatic candidate search',
        'strict_numeric':'NOT_RUN (not waived or claimed)','GPU':'NOT_RUN'}
p=O/'manifests/helios_retention_full_admission_r1.json';save(p,report)
text=f'''#!/usr/bin/env bash
set -Eeuo pipefail
exec {R}/envs/five-workload-h100-20260930/bin/python {O}/scripts/run_helios_h100_retention_matched_r1.py
'''
q=O/'queue_r5/022_helios_nograd_retention_matched'
with q.with_suffix('.sh').open('x') as f:f.write(text)
subprocess.run(['bash','-n',str(q.with_suffix('.sh'))],check=True)
atomic(q.with_suffix('.ready.json'),{'status':'PASS_CPU_ADMISSION','workload_math_changed':False,'script_sha256':hashlib.sha256(text.encode()).hexdigest(),'admission':str(p),'head':cpu['head']})
# Failed W6 Resident+boundary input staging is actual first-recompute capacity
# failure, not a known broken environment. Preserve it; other arms remain valid.
W=R/'research_artifacts/five_workload_h100_20260930/h100_w6_resident_boundary_matched_20261014_r1'
s=json.load(open(W/'resident_correctness/summary.json'));r=json.load(open(W/'resident_correctness/rank_receipts/rank_00_failure.json'))
assert s['cgroup_oom_delta']=={'oom':0,'oom_kill':0} and not s['GPU_contexts_after']
assert r['failure_type']=='CUDA_OOM' and r['failure_phase']=='critic_backward' and r['last_block']==49
failure={'status':'CONFIRMED_CUDA_CAPACITY_FAILURE_AFTER_REAL_BOUNDARY_STAGING','run':str(W),
         'first_Fake_grad_forward':'PASS_50_BOUNDARIES_ACTUALLY_COPIED','backward':'first block49 replay OOM,1.86GiB request',
         'memory_at_failure':r['memory_at_failure'],'boundary_at_failure':r['boundary_at_failure'],
         'optimizer_commits':r['commits_at_failure'],'host_OOM_delta':s['cgroup_oom_delta'],
         'cleanup':'PASS_NO_GPU_CONTEXTS','rejected_next':'optimizer offload cannot fix first backward before any Adam materialization; do not repeat same boundary-only run',
         'next_separate_capacity_proposal':'partial frozen AdaLN parameter CPU staging, original GPU BF16 projection, continuous sigma; NOT_RUN_NOT_YET_ADMITTED'}
save(O/'reports/w6_resident_boundary_actual_capacity_failure_r1.json',failure)
with (O/'research_log.jsonl').open('a') as f:
    for event in (dict(report,event='ADMIT_HELIOS_H100_RETENTION_SAME_REVISION_PAIR'),dict(failure,event='W6_RESIDENT_BOUNDARY_FIRST_RECOMPUTE_CUDA_OOM_PRESERVED')):
        f.write(json.dumps(dict(event,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
atomic(O/'queue_r5/021_w6_resident_boundary_real.disposition.json',{'preserve_failed_artifacts':True,'action':'CONTINUE_OTHER_ADMITTED_STEPS','review_report':str(O/'reports/w6_resident_boundary_actual_capacity_failure_r1.json')})
print(json.dumps(report))
