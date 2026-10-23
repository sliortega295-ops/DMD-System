"""Read-only post-run analysis beside raw ranks; never loads Torch/weights."""
import argparse, ast, csv, json, re, subprocess, sys
from pathlib import Path
from distillschedule.search.h3_native_bridge import _checked_native, _unprofiled_launch_evidence
from distillschedule.search.runtime_identity import verify_measurement_runtime, observed_binding
from distillschedule.search.h3_action_costs import PRECISION
from distillschedule.search.native import binding
from distillschedule.workloads.best_known.registry import write_new_json, file_sha256

p=argparse.ArgumentParser()
p.add_argument('--plan',type=Path,required=True)
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert 'torch' not in sys.modules
plan=json.loads(a.plan.read_text()); runtime=json.loads(a.runtime.read_text())
head=plan['integration_head']; mode=plan['mode']
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==head
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
out,s,rows,refs=_checked_native(plan)
assert mode in ('calibration','diagnostic-timing')
assert not list((out/'execution/action_failures').glob('*.json'))
refs.extend(_unprofiled_launch_evidence(plan,out,s,mode=mode))
b=observed_binding(runtime['before'],source_head=head)
b.update(source_head=head,asset_fingerprint=plan['asset_fingerprint'],
         semantic_fingerprint=plan['workload_semantics_hash'],precision_fingerprint=PRECISION)
verify_measurement_runtime(dict(runtime_identity_evidence=runtime),b)
refs.extend([binding(a.plan),binding(a.runtime),binding(out/'bound-summary.json'),binding(out/'native-summary.json')])
resources=[]
for n in (0,1):
    rp=out/f'search-resources.node{n}.json'; r=json.loads(rp.read_text())
    assert r['status']=='PASS' and r['run_id']==plan['run_id'] and len(r['samples'])>=2
    for event in ('oom','oom_kill'):
        c=[x['cgroup_events'][event] for x in r['samples']]; assert max(c)==min(c)
    xp=out/f'node-exit.{n}.json'; x=json.loads(xp.read_text());assert x['return_code']==0 and x['integration_head']==head
    tp=out/'launcher'/f'node_{n}'/'teardown_status.txt'
    t=dict(x.split('=',1) for x in tp.read_text().splitlines())
    assert t==dict(status='PASS',gpu_process_count='0',owner_run_id=plan['run_id'])
    resources.append(dict(node=n,status='PASS',cgroup_oom_delta=0,cgroup_oom_kill_delta=0,
        sample_count=len(r['samples']),sampled_cgroup_peak_bytes=max(x['cgroup_current_bytes'] for x in r['samples']),
        sampled_driver_free_min_bytes=min(g['free_bytes'] for x in r['samples'] for g in x['gpus']),
        peak_scope='SAMPLED_MONITOR_NOT_KERNEL_MEMORY_PEAK',teardown='PASS'))
    refs.extend([binding(rp),binding(xp),binding(tp)])
cycles=[]; actions=[]; stages=[]; lifetimes=[]; hbm_observations=[]
for r in rows:
    for c in r['cycles']:
        h=c['memory_after']['hbm']; ac=c['action_execution']; hbm_observations.append(h)
        assert ac['tensor_value_reads']=='DISABLED_METADATA_ONLY_PRIOR_CORRECTNESS_REQUIRED'
        cycles.append(dict(rank=r['rank'],cycle=c['cycle'],wall_seconds=c['native_wall_seconds'],status=c['status'],
            student_updates=c['updates']['student'],fake_updates=c['updates']['fake'],
            memory_peak_scope='LAST_STAGE_SINCE_RESET' if mode=='calibration' else 'PROCESS_CUMULATIVE',
            peak_live_inputs=ac['dispatch']['peak_live_inputs'],grid_hits=c['grid_replay_delta']['cache_hit_count'],
            grid_misses=c['grid_replay_delta']['cache_miss_count'],**h))
        for x in ac['rows']:
            actions.append(dict(rank=r['rank'],cycle=c['cycle'],action=x['action'],status=x['status'],
                draw_sites=len(x['draws']),tensor_metadata_count=sum(len(d['tensors']) for d in x['draws']),
                rng_unchanged=x['rng_before']==x['rng_after'],sample_count=len(x['samples']),
                host_envelope_seconds=(x['host_stop_ns']-x['host_start_ns'])/1e9,
                timer_scope='HOST_ENVELOPE_NOT_FENCED_COMPONENT_COST'))
    if mode=='calibration':
        cal=r['stage_calibration']
        for x in cal['stages']:
            hbm_observations.append(x['memory_after'])
            stages.append(dict(rank=r['rank'],operation_id=x['operation_id'],duration_ns=x['duration_ns'],
                component_key=x['key'],workspace_bytes_per_rank=x['workspace_bytes_per_rank'],
                entry_retained_input_bytes=x['entry_retained_input_bytes'],produced_output_bytes=x.get('produced_output_bytes',0),
                peak_allocated_bytes=x['memory_after']['peak_allocated_bytes'],peak_reserved_bytes=x['memory_after']['peak_reserved_bytes'],
                driver_free_after_bytes=x['memory_after']['driver_free_bytes']))
        for x in cal['output_lifetimes']:
            lifetimes.append(dict(rank=r['rank'],producer=x['producer'],consumer=x['consumer'],bytes=x['bytes'],allocation_point=x['allocation_point']))
assert all(x['rng_unchanged'] for x in actions if x['action'].startswith('c'))
fa3={}
for p in [out/'h3/train.log',*(out/'launcher'/f'node_{n}'/'console.log' for n in (0,1))]:
    if not p.is_file(): continue
    for line in p.read_text(errors='replace').splitlines():
        if '[h3-a100][fa3-replay-cache]' not in line or ' delta=' not in line:continue
        m=re.search(r'iter=(\d+) rank=(\d+)',line);assert m
        key=(int(m[2]),int(m[1])+1); value=ast.literal_eval(line.split(' delta=',1)[1])
        if key in fa3: assert fa3[key]['delta']==value
        fa3[key]=dict(rank=key[0],cycle=key[1],delta=value)
    refs.append(binding(p))
a.output.mkdir(parents=True,exist_ok=False)
def table(name,rows):
    if not rows:return
    with (a.output/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
table('rank-cycles.csv',cycles);table('action-metadata.csv',actions)
table('calibrated-components.csv',stages);table('output-lifetimes.csv',lifetimes)
write_new_json(a.output/'native-summary.json',s)
write_new_json(a.output/'fa3-cache-observations.json',list(fa3.values()))
result=dict(status='PASS_ACTION_CALIBRATION' if mode=='calibration' else 'PASS_UNPROFILED_NATIVE_DIAGNOSTIC',
    source_head=head,run_id=plan['run_id'],candidate_id=plan['search_candidate']['id'],binding=b,
    mode=mode,profiled=False,rank_count=16,cycles=2,final_versions=s['final_versions'],
    cycle_wall_seconds=s['cycle_wall_seconds'],timer_scope=s['timer_scope'],
    same_process=True,initializations=1,cycle1_scope='COLD_WARMUP_DIAGNOSTIC',
    cycle2_native_measurement=mode=='diagnostic-timing',
    launch_proof='EXACT_SOURCE_BOUND_DIRECT_TORCHRUN_BOTH_NODES',
    action_count=len(actions),rng_neutral_consumers=160,finite_losses=True,
    application_forward_per_rank_cycle=dict(student=24,fake=6,teacher=1),
    backward_per_rank_cycle=dict(student=1,fake=5),grid_per_rank_cycle=rows[0]['cycles'][0]['grid_replay_delta'],
    boundary_offload='none',fa3_storage='cpu_staged',resources=resources,
    tensor_value_reads='DISABLED_METADATA_ONLY_PRIOR_CORRECTNESS_REQUIRED',
    numerical_status='REVIEW_NOT_STRICT_PASS',strict_numerical_parity='NOT_REESTABLISHED',
    formal_speedup=None,matched_native_baseline='NOT_RUN_IN_THIS_BOUNDED_CLOSURE',
    peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in hbm_observations),
    peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in hbm_observations),sources=refs)
write_new_json(a.output/'summary.json',result)
write_new_json(a.output/'artifact_sha256.json',{p.name:file_sha256(p) for p in sorted(a.output.iterdir()) if p.is_file()})
print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
