from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess

R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100');C=R/'research_artifacts/five_workload_h100_20260930';O=C/'h100_resident_baselines_20261014_r1'
builder=C/'helios_gate_inputs_r1/scripts/build_helios_h100_asset_gate_r1.py'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
output=subprocess.check_output([str(R/'envs/five-workload-h100-20260930/bin/python'),str(builder),'--help'],text=True,stderr=subprocess.STDOUT,env=env)
assert '--repo-root' in output and '--shared-source' in output
runner=O/'scripts/run_helios_h100_retention_matched_r2.py';ast.parse(runner.read_text())
for token in ('HELIOS_RUNTIME_ROOT','DISTILLSCHEDULE_ASSET_ROOT','HF_HOME','PYTHONPATH','DISTILLSCHEDULE_MASTER_PORT'):assert token in runner.read_text()
old=O/'artifacts/helios_h100_retention_matched_20261015_r1'
assert "can't open file" in (old/'logs/02_fresh_live_asset_gate.log').read_text()
health=json.load(open(old/'health_02.json'));assert health['exit_code']==2 and health['cgroup_oom_delta']=={'oom':0,'oom_kill':0}
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def atomic(p,v):
    t=p.with_suffix('.partial');save(t,v);os.link(t,p);t.unlink()
report={'status':'CPU_LAUNCHER_PATH_AND_REQUIRED_ENV_FIXED_GPU_NOT_RUN_IN_FAILED_R1','failure':str(old),
        'bug':'new artifact runner guessed nonexistent repo asset-verifier path and omitted five validated launcher env bindings',
        'fix':'reuse exact already-validated build_helios_h100_asset_gate_r1.py and original launcher environment from official-swap runner',
        'actual_parser_help':'PASS_CURRENT_BUILDER_CLI_CPU_ONLY','training_source_changed':False,
        'head':'a065830e550151c56934ec83771ea506708d0e14','prior_GPU':'NOT_RUN'}
p=O/'reports/helios_retention_launcher_recovery_r2.json';save(p,report)
script=f'#!/usr/bin/env bash\nset -Eeuo pipefail\nexec {R}/envs/five-workload-h100-20260930/bin/python {runner}\n'
q=O/'queue_r5/022r2_helios_retention_actual_launcher'
with q.with_suffix('.sh').open('x') as f:f.write(script)
subprocess.run(['bash','-n',str(q.with_suffix('.sh'))],check=True)
atomic(q.with_suffix('.ready.json'),{'status':'PASS_CPU_ADMISSION','workload_math_changed':False,'script_sha256':hashlib.sha256(script.encode()).hexdigest(),'admission':str(p)})
atomic(O/'queue_r5/022_helios_nograd_retention_matched.disposition.json',{'preserve_failed_artifacts':True,'action':'CONTINUE_OTHER_ADMITTED_STEPS','review_report':str(p)})
print(json.dumps(report))
