from pathlib import Path
import json,subprocess,hashlib,ast
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100');C=R/'research_artifacts/five_workload_h100_20260930';O=C/'h100_resident_baselines_20261014_r1';P=R/'envs/five-workload-h100-20260930/bin/python'
script=O/'scripts/helios_hsdp_retention_api_canary_r1.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='56fdb5b4d1aca9517492031b27dfd784f53df74145011830c3a7c7ba8b5de41c'
ast.parse(script.read_text())
env=__import__('os').environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(R/'repos/DMD-System-h100-helios-official-swap-20261014-r1/src'),PYTHONDONTWRITEBYTECODE='1')
source='import torch;from torch.distributed.fsdp import FSDPModule;from distillschedule.runtime.fsdp2_nograd_persistence import generator_nograd_persistent_unshard_scope;assert torch.__version__.startswith("2.10.0");assert callable(FSDPModule.set_reshard_after_forward) and callable(FSDPModule.reshard);assert not torch.cuda.is_initialized();print("PASS_CPU_EXISTING_FSDP_PUBLIC_LIFECYCLE_API_NO_GPU")'
actual=subprocess.check_output([str(P),'-c',source],env=env,text=True,stderr=subprocess.STDOUT)
with (O/'logs/helios_hsdp_retention_API_CPU_gate_r1.log').open('x') as f:f.write(actual)
text='''#!/usr/bin/env bash
set -Eeuo pipefail
R=/home/jiangzhikun/yongyan_liu/Helios-H100
O="$R/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1"
P="$R/envs/five-workload-h100-20260930/bin/python"
B="$R/repos/DMD-System-h100-helios-official-swap-20261014-r1"
OUT="$O/artifacts/helios_hsdp_retention_api_GPU_r1"
[[ ! -e "$OUT" ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
"$P" "$B/scripts/distillschedule/verify_inherited_gpu_locks.py" --project-root "$R" --asset-fd 7
mkdir "$OUT"
cat /sys/fs/cgroup/memory.events > "$OUT/cgroup_before.txt"
export PYTHONPATH="$B/src:$B"
export PYTHONDONTWRITEBYTECODE=1
export H100_HSDP_RETENTION_API_OUTPUT="$OUT"
set +e
timeout --signal=TERM --kill-after=20s 180s "$P" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29371 "$O/scripts/helios_hsdp_retention_api_canary_r1.py" > "$OUT/launcher.log" 2>&1
RC=$?
set -e
cat /sys/fs/cgroup/memory.events > "$OUT/cgroup_after.txt"
nvidia-smi --query-compute-apps=pid --format=csv,noheader > "$OUT/gpu_contexts_after.txt"
"$P" - "$OUT" "$RC" <<'CHECK'
from pathlib import Path
import json,sys
p=Path(sys.argv[1]);rc=int(sys.argv[2])
def events(name):return {k:int(v) for k,v in map(str.split,(p/name).read_text().splitlines())}
a=events('cgroup_before.txt');b=events('cgroup_after.txt');delta={k:b[k]-a[k] for k in ('oom','oom_kill')}
rows=[json.loads(f.read_text()) for f in sorted(p.glob('rank_??.json'))]
passed=rc==0 and len(rows)==8 and sorted(x['rank'] for x in rows)==list(range(8)) and all(x['status']=='PASS_BITWISE_HSDP41_NOGRAD_AND_POST_SCOPE_GRAD_UPDATE' for x in rows) and not any(delta.values()) and not (p/'gpu_contexts_after.txt').read_text().strip()
result={'status':'PASS_PRIMITIVE_API_ONLY' if passed else 'FAIL_PRESERVED','rank_count':len(rows),'launcher_rc':rc,'cgroup_oom_delta':delta,'real_Helios_capacity':'NOT_RUN','real_Helios_timing':'NOT_RUN','per_rank':rows}
with (p/'result.json').open('x') as f:json.dump(result,f,indent=2)
assert passed,result['status']
CHECK
'''
q=O/'queue_r5/019_helios_hsdp_retention_API_probe'
with q.with_suffix('.sh').open('x') as f:f.write(text)
subprocess.run(['bash','-n',str(q.with_suffix('.sh'))],check=True)
ready={'status':'PASS_CPU_ADMISSION','workload_math_changed':False,'script_sha256':hashlib.sha256(text.encode()).hexdigest(),'probe_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'scope':'after H3 matrix only; one tiny-library API experiment; actual Helios workload/capacity/timing NOT_RUN; policy guard unchanged','admission':str(O/'manifests/w2_nograd_retention_opportunity_r1.json')}
with q.with_suffix('.ready.json').open('x') as f:json.dump(ready,f,indent=2)
print(actual);print(json.dumps(ready))
