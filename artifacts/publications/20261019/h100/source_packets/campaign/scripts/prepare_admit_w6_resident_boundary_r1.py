from pathlib import Path
import ast
import datetime
import hashlib
import json
import os
import subprocess

ROOT=Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
CAMPAIGN=ROOT/'research_artifacts/five_workload_h100_20260930'
OWNER=CAMPAIGN/'h100_resident_baselines_20261014_r1'
REPO=ROOT/'repos/DMD-System-h100-w6-native-resident-boundary-20261014-r1'
OLDREPO=ROOT/'repos/DMD-System-h100-w6-native-logical-b16-20261014-r1'
HEAD='c0aeb5389c63d39ff86681ee452001b352f80c8e'
BUNDLE=OWNER/'artifacts/source_bundles/w6_resident_boundary_r1.bundle'
PYTHON=ROOT/'envs/five-workload-h100-20260930/bin/python'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def command(argv,**kw):return subprocess.check_output(list(map(str,argv)),text=True,stderr=subprocess.STDOUT,**kw)
def save(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def atomic(p,value):
    t=p.with_suffix('.partial');save(t,value);os.link(t,p);t.unlink()
assert sha(BUNDLE)=='846d4589b2f1108f3d45832c23a9d8e438e5a535f19e5e3e9d06ab737bdeaa9d'
assert not REPO.exists()
command(['git','clone','--no-hardlinks','--no-checkout',OLDREPO,REPO])
command(['git','-C',REPO,'fetch',BUNDLE,'HEAD:refs/heads/agent/h100-w6-native-resident-boundary-20261014'])
command(['git','-C',REPO,'checkout','agent/h100-w6-native-resident-boundary-20261014'])
assert command(['git','-C',REPO,'rev-parse','HEAD']).strip()==HEAD
assert not command(['git','-C',REPO,'status','--porcelain']).strip()
# No new model/RNG/cache payload. The immutable private reference is untouched.
for name in ('runtime/h3_diffsynth_logical_microbatches.py','runtime/h3_diffsynth_fsdp_attention_reuse.py'):
    assert sha(REPO/name)==sha(OLDREPO/name)
old=OWNER/'scripts/h3_w6_native_b16_environment_r1.sh'
env_script=OWNER/'scripts/h3_w6_resident_boundary_environment_r1.sh'
text=old.read_text();assert text.count(str(OLDREPO.name))==1
with env_script.open('x') as f:f.write(text.replace(OLDREPO.name,REPO.name))
command(['bash','-n',env_script])
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
code='import pytest,torch; r=pytest.main(["-q","tests/test_h3_diffsynth_checkpoint_boundary_cpu.py","tests/test_h3_diffsynth_logical_microbatches.py","tests/test_h3_diffsynth_fsdp_attention_reuse.py"]); assert not torch.cuda.is_initialized(); raise SystemExit(r)'
log=command(['bash',env_script,'-c',code],cwd=REPO,env=env,timeout=180)
assert '10 passed' in log
with (OWNER/'logs/w6_boundary_clean_checkout_CPU_r1.log').open('x') as f:f.write(log)
for name in ('run_w6_native_three_arm_r3.py','run_w6_resident_boundary_capacity_r1.py'):
    ast.parse((OWNER/'scripts'/name).read_text())
origin=CAMPAIGN/'h100_w6_native_three_arm_matched_20261014_r2/resident_correctness'
failure=json.loads((origin/'rank_receipts/rank_00_failure.json').read_text())
assert failure['failure_type']=='CUDA_OOM' and failure['last_block']==14
summary=json.loads((origin/'summary.json').read_text())
assert summary['cgroup_oom_delta']=={'oom':0,'oom_kill':0} and not summary['GPU_contexts_after']
proof=OWNER/'manifests/w6_resident_boundary_CPU_admission_r1.json'
record={'status':'GO_BOUNDED_CAPACITY_FALLBACK_CPU_GATES_PASS_GPU_NOT_RUN',
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'head':HEAD,
        'hypothesis':'native per-checkpoint pinned CPU input staging can make full Resident W6 fit without changing continuous sigma/math',
        'prior_evidence':str(origin),'failure_mechanism':'actual Fake grad-forward block14 capacity, not host OOM',
        'frozen_contract':'same native AR8 G1F1, FP32 LoRA storage, BF16 autocast, continuous sigma, independent16 logical rows, GAS2, identical initial LoRA and update AST',
        'AdaLN_grid_elimination':'REJECTED_FOR_W6_CONTINUOUS_SIGMA_CONTRACT',
        'CPU_tests':'10 PASS, no CUDA initialization',
        'CPU_staging_r1_failure':'preserved import-package staging error, GPU NOT_RUN; repaired isolated package staging passes',
        'new_flag':'--checkpoint-boundary-cpu, default OFF, Resident-only',
        'scope':'call-local checkpoint input hooks; no replay-intermediate, parameter, optimizer-state or no-grad offload',
        'copy_census':'200 grad main checkpoint calls per physical rank/cycle; actual D2H/H2D bytes/events reconcile',
        'budget':'one world8 tiny integration, then one correctness pair-of-cycles; if PASS one timing pair-of-cycles; 7200s existing per-arm hard limit',
        'strict_numerical_parity':'NOT_RUN; not waived or claimed',
        'source_bundle_sha256':sha(BUNDLE),'shared_environment_changes':False,
        'old_artifacts_overwritten':False,'gpu':'NOT_RUN'}
save(proof,record)
for label,runid,tiny in [('020_w6_resident_boundary_tiny','h100_w6_resident_boundary_tiny_20261014_r1',True),('021_w6_resident_boundary_real','h100_w6_resident_boundary_matched_20261014_r1',False)]:
    lines=['#!/usr/bin/env bash','set -Eeuo pipefail',f'O={OWNER}',f'P={PYTHON}']
    if not tiny:
        lines += ['"$P" - <<\'DEPENDENCY\'', 'import json', f'assert json.load(open("{CAMPAIGN}/h100_w6_resident_boundary_tiny_20261014_r1/result.json"))["status"]=="PASS_TINY_NATIVE_BOUNDARY_ONLY"', 'DEPENDENCY']
    lines += [f'exec "$P" "$O/scripts/run_w6_resident_boundary_capacity_r1.py" --run-id {runid}'+(' --tiny-only' if tiny else '')]
    script='\n'.join(lines)+'\n';prefix=OWNER/'queue_r5'/label
    with prefix.with_suffix('.sh').open('x') as f:f.write(script)
    command(['bash','-n',prefix.with_suffix('.sh')])
    atomic(prefix.with_suffix('.ready.json'),{'status':'PASS_CPU_ADMISSION','script_sha256':hashlib.sha256(script.encode()).hexdigest(),
                                             'workload_math_changed':False,'head':HEAD,'admission':str(proof),
                                             'runner_sha256':sha(OWNER/'scripts/run_w6_resident_boundary_capacity_r1.py')})
with (OWNER/'research_log.jsonl').open('a') as f:f.write(json.dumps(dict(record,event='ADMIT_SEPARATE_W6_NATIVE_RESIDENT_CPU_CHECKPOINT_BOUNDARY_AFTER_PROVEN_CAPACITY_FAILURE'))+'\n')
print(json.dumps(record))
