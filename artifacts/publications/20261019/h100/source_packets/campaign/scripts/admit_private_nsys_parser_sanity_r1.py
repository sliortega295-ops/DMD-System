from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess

R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
O=R/'research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1'
T=R/'tools/nsight_systems_2024.6.2_20261014_r1/2024.6.2.225'
N=T/'target-linux-x64/nsys'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
assert sha(N)=='c7a1764113e90b450537ad10e051ca5e47849e50fee964d291212ca6c7277b0a'
assert sha(T/'host-linux-x64/QdstrmImporter')=='be5e3266faa2cc520fb226384c4b1597f0a477e7f6a936698766bb6a4532bb42'
version=subprocess.check_output([str(N),'--version'],text=True);assert '2024.6.2.225' in version
help=subprocess.check_output([str(N),'profile','--help'],text=True)
assert all(x in help for x in ('cudaProfilerApi','--capture-range-end','--trace-fork-before-exec','--cpuctxsw','--wait'))
proof=O/'manifests/private_nsys_tool_target_resolution_r2.json'
save(proof,{'status':'PASS_PRIVATE_TARGET_ELF_AND_OWN_LIBRARIES_CPU_ONLY','version':version,'nsys':str(N),
            'nsys_sha256':sha(N),'importer_sha256':sha(T/'host-linux-x64/QdstrmImporter'),
            'initial_check_failure':'R1 stream succeeded but dereferenced bin/nsys ran outside its target-library origin: libbpf.so.1 load failure; old files preserved',
            'fix':'invoke the original target-linux-x64 ELF; no new transfer, no shared LD_LIBRARY_PATH',
            'archive_stream_sha256':'NOT_PERSISTED_BY_FAILED_R1_POSTCHECK; individual exact binary identities revalidated',
            'system_nsys_changed':False,'training_environment_changed':False,'GPU_sanity':'NOT_RUN'})
script=f'''#!/usr/bin/env bash
set -Eeuo pipefail
R={R}
O={O}
P="$R/envs/five-workload-h100-20260930/bin/python"
N={N}
OUT="$O/artifacts/private_nsys_current_stack_sanity_r1"
[[ ! -e "$OUT" ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
"$P" "$R/repos/DMD-System-h100-helios-b16-20260930/scripts/distillschedule/verify_inherited_gpu_locks.py" --project-root "$R" --asset-fd 7
mkdir "$OUT"
export H100_NSYS_SANITY_OUTPUT="$OUT" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2
cat /sys/fs/cgroup/memory.events > "$OUT/cgroup_before.txt"
set +e
timeout --signal=TERM --kill-after=30s 300s "$N" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --wait=primary --trace-fork-before-exec=true --capture-range=cudaProfilerApi --capture-range-end=stop --output="$OUT/sanity" "$P" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29376 "$O/scripts/private_nsys_cuda128_sanity_worker_r1.py" > "$OUT/profiler.log" 2>&1
RC=$?
set -e
cat /sys/fs/cgroup/memory.events > "$OUT/cgroup_after.txt"
nvidia-smi --query-compute-apps=pid --format=csv,noheader > "$OUT/gpu_contexts_after.txt"
[[ "$RC" == 0 ]] || exit "$RC"
timeout 180s "$N" export --type=sqlite --output="$OUT/sanity.sqlite" "$OUT/sanity.nsys-rep" > "$OUT/export.log" 2>&1
"$P" "$O/scripts/validate_private_nsys_sanity_r1.py" "$OUT"
'''
prefix=O/'queue_r5/018t_private_nsys_parser_sanity'
with prefix.with_suffix('.sh').open('x') as f:f.write(script)
subprocess.run(['bash','-n',str(prefix.with_suffix('.sh'))],check=True)
t=prefix.with_suffix('.partial')
save(t,{'status':'PASS_CPU_ADMISSION','workload_math_changed':False,'script_sha256':hashlib.sha256(script.encode()).hexdigest(),
        'scope':'8 tiny independent GPUs, CUDA/NVTX/copy/NCCL parser compatibility only; no model; no formal performance','admission':str(proof)})
os.link(t,prefix.with_suffix('.ready.json'));t.unlink()
with (O/'research_log.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':'PRIVATE_NSYS_CPU_IDENTITY_RECOVERY_AND_BOUNDED_PARSER_CANARY_ADMITTED','manifest':str(proof),'GPU':'NOT_RUN'})+'\n')
print(json.dumps({'status':'PASS_CPU_TOOL_ADMISSION_ONLY','queue':str(prefix),'GPU':'NOT_RUN','version':version}))
