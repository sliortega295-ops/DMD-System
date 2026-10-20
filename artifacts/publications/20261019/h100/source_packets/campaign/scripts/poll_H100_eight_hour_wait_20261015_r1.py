"""Bounded read-only SSH poll for the explicitly requested eight-hour wait."""
from pathlib import Path
import datetime,json,subprocess,hashlib
O=Path('/home/lyy/Helios/research_artifacts/DMD-System/five_workload_h100_20260930/h100_resident_baselines_20261014_r1')
code=r"""
from pathlib import Path
import datetime,json,subprocess,os
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100');O=R/'research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1'
def cmd(args):
 p=subprocess.run(args,text=True,capture_output=True,timeout=25)
 return {'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
def read(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return {'status':'MISSING','path':str(p)}
h=read(O/'manifests/controller_handle_r6.json');pid=h.get('pid');process=Path(f'/proc/{pid}')
ticks=None
if process.exists():
 stat=(process/'stat').read_text();ticks=stat[stat.rfind(')')+2:].split()[19]
gpu=cmd(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'])
queue={p.name:read(p) for p in sorted((O/'queue_r6').glob('*.json')) if any(p.name.endswith(s) for s in ('.done.json','.failure.json','.disposition.json'))}
run=R/'research_artifacts/five_workload_h100_20260930/h100_w6_lora_unit_inplace_20261015_r1'
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':cmd(['hostname'])['stdout'],
 'R6':read(O/'campaign_r6_status.json'),'controller_handle':h,'actual_controller_start_ticks':ticks,
 'controller_identity_match':ticks is not None and str(h.get('start_ticks'))==ticks,
 'children':cmd(['ps','--ppid',str(pid),'-o','pid,ppid,pgid,stat,wchan:25,args']),
 'gpu':gpu,'GPU_processes':cmd(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']),
 'foreign_reservation':cmd(['pgrep','-af','^/share/project/wangzifeng/.*StreamingAR/runtime/scripts/gpu_lease.py']),
 'queue_receipts':queue,'new_run_directory_exists':run.exists(),
 'new_run_result':read(run/'result.json') if run.exists() else {'status':'NOT_RUN'},
 'cgroup_memory_events':Path('/sys/fs/cgroup/memory.events').read_text(),
 'disk':cmd(['df','-B1',str(R)])}
log=O/'logs/campaign_r6.log'
if log.exists():
 with log.open('rb') as f:
  f.seek(max(0,log.stat().st_size-5000));r['controller_log_tail']=f.read().decode(errors='replace')
print(json.dumps(r))
"""
args=['ssh','-T','-4','-S','none','-o','ControlMaster=no','-o','ControlPath=none','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','-p2222','inference.jiangzhikun.baai-worldsound_prod.cn-neimongol-helingeer.ws@ssh.platform-cuihu.jingneng-inner.ac.cn','python3 -']
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
try:
 p=subprocess.run(args,input=code,text=True,capture_output=True,timeout=70)
 if p.returncode:r={'utc':start,'status':'SSH_FAILED_LIVE_STATE_UNKNOWN','returncode':p.returncode,'stderr':p.stderr[-2000:]}
 else:r=json.loads(p.stdout);r['status']='PASS_READ_ONLY_SNAPSHOT'
except subprocess.TimeoutExpired:r={'utc':start,'status':'SSH_TIMEOUT_LIVE_STATE_UNKNOWN'}
r['new_GPU_launches_by_poll']=False;r['foreign_signals']=False
folder=O/'artifacts/eight_hour_wait_20261015T044405Z';folder.mkdir(exist_ok=True)
with (folder/'checks.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
print(json.dumps(r,indent=2))
