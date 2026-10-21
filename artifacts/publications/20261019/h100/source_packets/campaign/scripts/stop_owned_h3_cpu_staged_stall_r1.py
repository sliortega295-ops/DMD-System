from pathlib import Path
import json,os,signal,subprocess,datetime,time
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h3_h100_three_arm_matched_20261014_r4/03_correctness_candidate')
if (R/'summary.json').exists():raise SystemExit('arm has already exited; do not signal any PID')
handle=json.load(open(R/'manifests/launch.json'));pid=handle['pid'];p=Path(f'/proc/{pid}')
s=(p/'stat').read_text().split(') ',1)[1].split()
assert s[19]==handle['start_ticks'] and s[0]!='Z' and os.getpgid(pid)==pid
cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
assert 'torch.distributed.run' in cmd and 'h3_h100_recording_entry_r1' in cmd
assert int(s[1])==534759
workers=[]
for rank in range(8):
 x=json.load(open(R/f'rank_receipts/rank_{rank:03d}.start.json'))
 q=Path(f'/proc/{x["pid"]}')
 if not q.exists():continue
 t=(q/'stat').read_text().split(') ',1)[1].split();assert t[19]==x['start_ticks']
 workers.append({'rank':rank,'PID':x['pid'],'start_ticks':t[19],'wchan':(q/'wchan').read_text().strip(),'CPU_ticks':t[11:13]})
telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw','--format=csv,noheader'],text=True)
note={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'OWNED_EARLY_STOP_STALLED_DIAGNOSTIC_NOT_CUDA_OOM','owned_launcher':handle,'command':cmd,'workers_read_only':workers,'GPU_telemetry':telemetry,'last_progress':'train_start2026-10-14T20:53:38Z; fixed GPU memory/low-power100percent exceptrank2zero observed20:57,21:01,21:02,21:06','failure_type':'UNRESOLVED_WAIT_STALL_CPU_STAGED_CANDIDATE','root_cause':'UNRESOLVED; thread ownership/power are not operation attribution','GDB_attempt':'ptrace unavailable, no successful attach; no system/permission changes','timing':'NOT_ELIGIBLE_NOT_COMPLETED','budget_reason':'no measurable progress across repeated separated snapshots; do not wait7200seconds for a non-advancing candidate','recovery_hypothesis':'existing CUDA attention output/LSE cache removes host-stage queues; must first pass actual GPU library proof and then full8rank gates','original_source_and_workload_changed':False,'other_process_signals':False}
with (R/'manifests/owned_stall_early_stop_r1.json').open('x') as f:json.dump(note,f,indent=2)
os.killpg(pid,signal.SIGTERM)
print(json.dumps({'status':'SIGTERM_EXACT_OWNED_LAUNCHER_SESSION_ONLY','PID':pid,'start_ticks':s[19],'manifest':str(R/'manifests/owned_stall_early_stop_r1.json')}))
