"""Read-only two-node end-of-task evidence; no process cleanup or GPU work."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

E = Path(__file__).resolve().parents[1]
CODE = r'''
import datetime,fcntl,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
def run(argv):return subprocess.check_output(argv,text=True,timeout=30).strip()
head=run(['git','-C',str(root),'rev-parse','HEAD'])
dirty=run(['git','-C',str(root),'status','--porcelain'])
assert head==expected and not dirty
devices=run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,driver_version','--format=csv,noheader,nounits'])
gpu_processes=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])
assert len(devices.splitlines())==8 and not gpu_processes
lock=pathlib.Path('/tmp/distillschedule_gpu_node.lock')
if lock.exists():
 with lock.open('r') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  lock_status='AVAILABLE'
  fcntl.flock(f,fcntl.LOCK_UN)
else:lock_status='ABSENT_NO_HELD_LOCK'
events=dict((k,int(v)) for k,v in (line.split() for line in pathlib.Path('/sys/fs/cgroup/memory.events').read_text().splitlines()))
assert events['oom']==0 and events['oom_kill']==0
processes=run(['ps','-eo','pid,comm,args']).splitlines()
owned_candidates=[p for p in processes if any(z in p for z in ['torch.distributed.run','probe_helios','nsys profile'])
 and not any(z in p for z in ['python3 -c','sh -c','bash -c','sshd:'])]
assert not owned_candidates,owned_candidates
print(json.dumps(dict(status='PASS_READBACK_NO_GPU_WORK',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 hostname=run(['hostname']),source_head=head,git_status=dirty,gpu_rows=devices.splitlines(),
 gpu_processes=gpu_processes,owned_training_profiler_processes=owned_candidates,lock_status=lock_status,
 cgroup_events=events,cgroup_current_bytes=int(pathlib.Path('/sys/fs/cgroup/memory.current').read_text()),
 memory_max=pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip(),
 mount=run(['df','-PT',str(root)]))))
'''


def main():
    d = json.loads((E / 'manifests/deployment-cc7e451bff73-r4.json').read_text())

    def inspect(item):
        index, node = item
        command = ['ssh', '-4', '-T', '-oBatchMode=yes', '-oConnectTimeout=15',
                   '-oControlMaster=no', '-oControlPath=none', '-p', str(node['ssh_port']),
                   node['host'], shlex.join(['python3', '-c', CODE, d['target'], d['source_head']])]
        result = subprocess.run(command, text=True, capture_output=True, timeout=90, check=True)
        return dict(node=index, **json.loads(result.stdout))

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(inspect, enumerate(d['assets']['gan']['nodes'])))
    output = dict(status='PASS_16_GPU_IDLE_NO_OWNED_WORKERS_LOCKS_AVAILABLE',
                  checked_utc=datetime.now(timezone.utc).isoformat(), nodes=rows,
                  scope='Point-in-time readback after final GAN native confirmation; no processes killed',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = E / 'manifests/final-cleanup-readback.json'
    with path.open('x') as f:
        json.dump(output, f, indent=2); f.write('\n')
    print(json.dumps(dict(status=output['status'], path=str(path)), indent=2))


if __name__ == '__main__':
    main()
