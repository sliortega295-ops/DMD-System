"""Read-only post-run audit; no process or environment mutation."""
import fcntl,json,os,socket,subprocess,sys,time
from pathlib import Path
plan=json.load(sys.stdin);n=plan['node'];run=plan['run_id'];root=Path(plan['output'])
ancestors=set();pid=os.getpid()
while pid>1 and pid not in ancestors:
 ancestors.add(pid)
 try:pid=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[1])
 except (FileNotFoundError,ValueError,IndexError):break
processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name) in ancestors:continue
 try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 except (OSError,PermissionError):continue
 if run in cmd and any(x in cmd for x in ('python','torchrun','bash','nsys')):processes.append({'pid':int(p.name),'command':cmd})
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,capture_output=True,check=True)
with open('/tmp/distillschedule_gpu_node.lock','a') as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock_free=True;fcntl.flock(lock,fcntl.LOCK_UN)
 except BlockingIOError:lock_free=False
listeners=[]
for name in ('tcp','tcp6'):
 for line in Path('/proc/net/'+name).read_text().splitlines()[1:]:
  fields=line.split()
  if fields[3]=='0A' and int(fields[1].split(':')[1],16)==plan['port']:listeners.append(line)
events=dict(line.split() for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines())
result=dict(schema_version='distillschedule.postrun_readback.v1',node=n,hostname=socket.gethostname(),time_ns=time.time_ns(),run_id=run,
 gpu_processes=gpu.stdout.splitlines(),owned_run_processes=processes,node_lock_free=lock_free,port=plan['port'],port_listeners=listeners,cgroup_events=events)
result['status']='PASS' if not gpu.stdout.strip() and not processes and lock_free and not listeners else 'FAIL_CLEANUP_READBACK'
dest=root/f'cleanup-readback.node{n}.json'
with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
