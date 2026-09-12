"""Run beside remote raw receipts; download only this small derived summary."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
from statistics import median

from distillschedule.search.native import checked_summary,binding
from distillschedule.search.livewan_action_observation import validate_action_observations
from distillschedule.workloads.best_known.registry import write_new_json

p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();plan=json.loads(a.plan.read_text());out,summary=checked_summary(plan)
a.output.mkdir(exist_ok=False)
sources=[binding(a.plan),binding(out/'bound-summary.json')]
raws=[];native=[]
for rank in range(16):
 for folder,target in [('operation_calibration',raws),('receipts',native)]:
  path=out/folder/f'rank_{rank:03d}.json';target.append(json.loads(path.read_text()));sources.append(binding(path))
 for row in [native[-1]]:
  assert row['status']=='PASS' and row['final_versions']==dict(generator=2,critic=10,ema=0)
 assert native[-1]['rank']==rank
try:
 verified=validate_action_observations(raws,expected_head=plan['integration_head'])
except Exception as e:
 write_new_json(a.output/'failure.json',dict(status='FAIL_ACTION_OBSERVATION_ADMISSION',type=type(e).__name__,error=str(e),sources=sources,training_status=summary['status']))
 raise
resources=[]
for n in (0,1):
 rp=out/f'search-resources.node{n}.json';r=json.loads(rp.read_text());sources.append(binding(rp))
 assert r['status']=='PASS' and r['run_id']==plan['run_id']
 for event in ('oom','oom_kill'):
  counts=[s['cgroup_events'][event] for s in r['samples']];assert max(counts)==min(counts)
 # Native trajectory checks no GPU apps before exit, but only timing emits
 # teardown_status.txt. Use the existing source-bound exit + cgroup receipts
 # and a fresh, independent node/lock/port readback, not a fabricated file.
 assert plan['native_mode']=='trajectory'
 np=out/f'node-exit.{n}.json';node_exit=json.loads(np.read_text())
 assert node_exit['return_code']==0 and node_exit['integration_head']==plan['integration_head']
 cp=out.parent.parent/f'cleanup-readback.node{n}.json';cleanup=json.loads(cp.read_text())
 assert cleanup['status']=='PASS' and cleanup['run_id']==plan['run_id'] and cleanup['node']==n
 assert not cleanup['gpu_processes'] and not cleanup['owned_run_processes']
 assert cleanup['node_lock_free'] and not cleanup['port_listeners']
 before=out/f'launcher/node_{n}/cgroup_memory_events_before.txt'
 after=out/f'launcher/node_{n}/cgroup_memory_events_after.txt'
 read_events=lambda path: {k:int(v) for k,v in (l.split() for l in path.read_text().splitlines())}
 eb,ea=read_events(before),read_events(after)
 assert all(eb[k]==ea[k] for k in ('oom','oom_kill'))
 sources.extend(binding(p) for p in (np,cp,before,after))
 resources.append(dict(node=n,cgroup_oom_delta=0,cgroup_oom_kill_delta=0,
  sampled_cgroup_peak_bytes=max(s['cgroup_current_bytes'] for s in r['samples']),
  sampled_min_driver_free_bytes=min(g['free_bytes'] for s in r['samples'] for g in s['gpus']),cleanup='PASS'))
transport=[]
for raw in raws:
 groups=defaultdict(list)
 for row in raw['transport_observation']['rows']:groups[row['kind']].append(row)
 for kind,rows in sorted(groups.items()):
  transport.append(dict(rank=raw['rank'],kind=kind,count=len(rows),
    counted_bytes=sum(x['bytes'] for x in rows),duration_sum_ns=sum(x['duration_ns'] for x in rows),
    duration_median_ns=median(x['duration_ns'] for x in rows),
    clock=rows[0]['clock'],issuing_threads=sorted({x['thread_id'] for x in rows}),
    byte_scope='ACTUAL_LATENT_COPY' if kind in ('latent_d2h','latent_h2d') else 'REPEATED_PAYLOAD_METADATA_NOT_ADDITIONAL_COPY_BYTES'))
def write_csv(name,rows):
 with (a.output/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write_csv('helper-observations.csv',verified['helpers']);write_csv('transport-observations.csv',transport)
peaks=[r['peak_memory'] for r in native]
result=dict(**verified,run_id=plan['run_id'],native_summary_status=summary['status'],
 native_collection_mode='TRAJECTORY_WITH_ORIGINAL_CORRECTNESS_PROBES',
 final_versions=dict(generator=2,critic=10,ema=0),physical_critic_batch=2,global_batch=16,cycles=2,
 max_allocated_bytes=max(x['max_allocated_bytes'] for x in peaks),max_reserved_bytes=max(x['max_reserved_bytes'] for x in peaks),
 resources=resources,transport=transport,sources=sources,
 performance_timing='NOT_RUN',new_speedup=None,numerical_parity='NOT_REASSERTED_NO_NEW_BASELINE_MATRIX',
 claim_boundary='Native action identity and copy/wait accounting only. Contextual trajectory envelopes are not independent service costs.')
write_new_json(a.output/'summary.json',result)
print(json.dumps(result))
