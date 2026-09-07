"""Read completed H3 native action evidence remotely; no weights or CUDA."""
import argparse,ast,csv,hashlib,json,re,subprocess,sys
from collections import Counter
from pathlib import Path
from distillschedule.search.h3_native import summarize_native_two_cycle
from distillschedule.search.native import binding
from distillschedule.workloads.best_known.registry import write_new_json
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--head',required=True);a=p.parse_args()
assert 'torch' not in sys.modules
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==a.head
raws=[json.loads((a.root/'execution/ranks'/f'rank_{r:03d}.json').read_text()) for r in range(16)]
validated=summarize_native_two_cycle(raws,expected_head=a.head,candidate_id='joint-h3v2-d2-n128-w3')
assert validated['status']=='PASS' and not validated['formal_timing_eligible']
assert not list((a.root/'execution/action_failures').glob('*.json'))
sources=[binding(a.root/'execution/ranks'/f'rank_{r:03d}.json') for r in range(16)]
sources += [binding(a.root/n) for n in ('native-summary.json','bound-summary.json')]
resources=[]
def events(path):
 return {k:int(v) for k,v in (line.split() for line in path.read_text().splitlines())}
for node in (0,1):
 monitor=a.root/f'search-resources.node{node}.json';m=json.loads(monitor.read_text())
 assert m['status']=='PASS' and m['run_id']==a.root.name
 before=a.root/f'launcher/node_{node}/cgroup_before.txt';after=a.root/f'launcher/node_{node}/cgroup_after.txt'
 eb,ea=events(before),events(after);assert all(eb[k]==ea[k] for k in ('oom','oom_kill'))
 exit_path=a.root/f'node-exit.{node}.json';x=json.loads(exit_path.read_text());assert x['return_code']==0 and x['integration_head']==a.head
 teardown=a.root/f'launcher/node_{node}/teardown_status.txt'
 td=dict(line.split('=',1) for line in teardown.read_text().splitlines())
 assert td==dict(status='PASS',gpu_process_count='0',owner_run_id=a.root.name)
 cleanup=a.root.parent.parent/f'cleanup-readback.node{node}.json';c=json.loads(cleanup.read_text());assert c['status']=='PASS' and c['run_id']==a.root.name
 samples=m['samples'];assert all(max(s['cgroup_events'][k] for s in samples)==min(s['cgroup_events'][k] for s in samples) for k in ('oom','oom_kill'))
 resources.append(dict(node=node,oom_delta=0,oom_kill_delta=0,monitor_samples=len(samples),sampled_cgroup_peak_bytes=max(s['cgroup_current_bytes'] for s in samples),sampled_driver_free_min_bytes=min(g['free_bytes'] for s in samples for g in s['gpus']),cleanup='PASS',node_lock_free=c['node_lock_free'],gpu_processes=c['gpu_processes'],owned_processes=c['owned_run_processes'],listeners=c['port_listeners']))
 sources += [binding(x) for x in (monitor,before,after,exit_path,teardown,cleanup)]
rankrows=[];actions=[];draws=[];memory=[]
for r in raws:
 for c in r['cycles']:
  ac=c['action_execution'];state=c['memory_after']['hbm']
  rankrows.append(dict(rank=r['rank'],cycle=c['cycle'],status=c['status'],wall_seconds=c['native_wall_seconds'],student_updates=c['updates']['student'],fake_updates=c['updates']['fake'],peak_live_inputs=ac['dispatch']['peak_live_inputs'],action_count=len(ac['rows']),grid_hits=c['grid_replay_delta']['cache_hit_count'],grid_misses=c['grid_replay_delta']['cache_miss_count'],peak_allocated_bytes=state['peak_allocated_bytes'],peak_reserved_bytes=state['peak_reserved_bytes'],allocated_after=state['allocated_bytes'],driver_free_after=state['driver_free_bytes']))
  for boundary in ('memory_before','memory_after'):
   mm=c[boundary];memory.append(dict(rank=r['rank'],cycle=c['cycle'],boundary=boundary,**mm['hbm'],**mm['process'],cgroup_current_bytes=mm['cgroup']['memory.current']))
  for row in ac['rows']:
   actions.append(dict(rank=r['rank'],cycle=c['cycle'],action=row['action'],status=row['status'],sample_count=len(row['samples']),draw_sites=len(row['draws']),rng_unchanged=row['rng_before']==row['rng_after'],host_envelope_seconds=(row['host_stop_ns']-row['host_start_ns'])/1e9,cost_admission=False))
   for idx,draw in enumerate(row['draws']):
    for j,t in enumerate(draw['tensors']):
     draws.append(dict(rank=r['rank'],cycle=c['cycle'],action=row['action'],site=draw['site'],call=idx,tensor=j,sha256=t['sha256'],bytes=t['bytes'],shape=t['shape'],dtype=t['dtype'],device=t['device'],finite=t['finite']))
fa3={}
for path in [a.root/'h3/train.log',*(a.root/f'launcher/node_{n}/console.log' for n in (0,1))]:
 for line in path.read_text(errors='replace').splitlines():
  if '[h3-a100][fa3-replay-cache]' not in line or ' delta=' not in line:continue
  match=re.search(r'iter=(\d+) rank=(\d+)',line); assert match
  cycle,rank=int(match[1])+1,int(match[2]);delta=ast.literal_eval(line.split(' delta=',1)[1])
  key=(rank,cycle)
  if key in fa3:assert fa3[key]['delta']==delta
  else:fa3[key]=dict(rank=rank,cycle=cycle,delta=delta)
 source=binding(path)
 sources.append(source)
assert all(x['rng_unchanged'] for x in actions if x['action'].startswith('c'))
a.output.mkdir(exist_ok=False)
def csvout(name,rows):
 assert rows
 with (a.output/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
csvout('rank-cycles.csv',rankrows);csvout('action-observations.csv',actions);csvout('actual-draw-sha.csv',draws);csvout('memory-boundaries.csv',memory)
write_new_json(a.output/'native-summary.json',validated)
write_new_json(a.output/'fa3-cache-observations.json',list(fa3.values()))
summary=dict(status='PASS_NATIVE_ACTION_CORRECTNESS_NOT_TIMING',run_id=a.root.name,source_head=a.head,candidate_id='joint-h3v2-d2-n128-w3',rank_count=16,cycles=2,final_versions=validated['final_versions'],cycle_wall_seconds=validated['cycle_wall_seconds'],wall_scope=validated['timer_scope'],action_count=len(actions),actual_draw_tensor_hash_count=len(draws),per_rank_cycle_actions=11,peak_live_prepared_groups=3,action_order=raws[0]['cycles'][0]['action_execution']['dispatch']['executed_operations'],stochastic_consumer_rng_neutral='PASS_160_FAKE_UPDATES',observed_rng_scope=raws[0]['cycles'][0]['action_execution']['rng_scope'],application_forward_per_rank_cycle=dict(student=24,fake=6,teacher=1),backward_per_rank_cycle=dict(student=1,fake=5),grid_per_rank_cycle=raws[0]['cycles'][0]['grid_replay_delta'],samples_per_world_cycle=96,finite_losses=True,peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in rankrows),peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in rankrows),max_rank_rss_hwm_bytes=max(x['VmHWM'] for x in memory),resources=resources,fa3_log_observed_rank_cycles=sorted([list(k) for k in fa3]),boundary_offload='none',fa3_replay_storage='cpu_staged',strict_numerical_comparison='NOT_RUN_THIS_ACTION_UNIT_PREVIOUS_DRIFT_REMAINS_REVIEW',search_ready=False,cost_admission=False,native_performance='NOT_RUN',new_speedup=None,sources=sources)
write_new_json(a.output/'summary.json',summary)
write_new_json(a.output/'artifact_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.output.iterdir()) if p.is_file()})
print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2))
