from pathlib import Path
import hashlib
import json
import os
import subprocess

R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
O=R/'research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1'
OLD=R/'repos/DMD-System-h100-helios-official-swap-20261014-r1'
REPO=R/'repos/DMD-System-h100-helios-nograd-retain-20261015-r1'
HEAD='a065830e550151c56934ec83771ea506708d0e14'
bundle=O/'artifacts/source_bundles/helios_h100_nograd_retention_r1.bundle'
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='e53462030ba58e5b979458487d98291aa31f0c813e74d9e9ea7e69aa4f42b578'
assert not REPO.exists()
def call(c,**kw):return subprocess.check_output(list(map(str,c)),text=True,stderr=subprocess.STDOUT,**kw)
call(['git','clone','--no-hardlinks','--no-checkout',OLD,REPO])
call(['git','-C',REPO,'fetch',bundle,'HEAD:refs/heads/agent/h100-helios-nograd-retain-20261014'])
call(['git','-C',REPO,'checkout','agent/h100-helios-nograd-retain-20261014'])
assert call(['git','-C',REPO,'rev-parse','HEAD']).strip()==HEAD
assert not call(['git','-C',REPO,'status','--porcelain']).strip()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(REPO/'src')+':'+str(REPO),USE_TF='0',USE_FLAX='0')
command=[R/'envs/five-workload-h100-20260930/bin/python','-m','pytest','-q',
         'tests/test_refactor_helios_h100_nograd_retention.py',
         'tests/test_refactor_helios_h100_evidence.py','tests/test_refactor_hardware_profile.py',
         'tests/test_refactor_helios_official_role_swap.py']
value=call(command,env=env,cwd=REPO,timeout=180)
with (O/'logs/helios_retention_clean_checkout_CPU_r1.log').open('x') as f:f.write(value)
code=f'''import importlib.util,pathlib,torch
p=pathlib.Path("{REPO}"); s=importlib.util.spec_from_file_location("summary",p/"scripts/distillschedule/summarize_helios_hardware_p0.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
f=m.freeze_policy(p,variant="no-gan",arm="h100-nograd-retain")
proof=m.validate_h100_retention_capability(pathlib.Path("{O}/artifacts/helios_hsdp_retention_api_GPU_r1/result.json"),f)
assert not torch.cuda.is_initialized()
print(proof)
'''
actual=call([R/'envs/five-workload-h100-20260930/bin/python','-c',code],env=env,cwd=REPO,timeout=30)
report={'status':'PASS_CPU_TESTS_AND_ACTUAL_GPU_API_RECEIPT_NO_MODEL','head':HEAD,
        'tests_output':value,'capability_validation':actual,'repo':str(REPO),
        'GPU_real_Helios':'NOT_RUN','shared_environment_changes':False,
        'GPU_capability_primitive_receipt_sha256':'52f9b287011e681861731055b4b37ea7d65ab4cb39eb8ad0792025a7b0c23966'}
with (O/'manifests/helios_retention_source_CPU_r1.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(value);print(actual);print('PASS_CPU_TESTS_ACTUAL_RECEIPT_NO_CUDA_INIT')
