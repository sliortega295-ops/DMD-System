#!/usr/bin/env python3
"""Summarize a typed Route-A world16 run and compare it with the frozen diagnostic baseline."""
from __future__ import annotations
import argparse, collections, hashlib, json, statistics
from pathlib import Path

OLD_RESULT = Path('/share/project/shuchenweng/yongyan_liu/DistillSchedule/artifacts/manifests/task7_full5_routea_exact_b0_gs_fullstream_wave8_gate_result_001.json')
EXPECTED_STEPS = [10, 11, 12, 13, 14]

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def read_jsonl(path: Path):
    out=[]
    for line in path.read_text().splitlines():
        try: out.append(json.loads(line))
        except json.JSONDecodeError: pass
    return out

def parse_kv(path: Path):
    out={}
    for line in path.read_text(errors='replace').splitlines():
        if '=' in line:
            k,v=line.split('=',1); out[k]=v
    return out

def parse_memory_events(text: str):
    parts=text.replace(';',' ').split(); out={}
    for i in range(0,len(parts)-1,2):
        try: out[parts[i]]=int(parts[i+1])
        except ValueError: pass
    return out

def resource(path: Path):
    rows=[]
    for line in path.read_text().splitlines()[1:]:
        f=line.split(',')
        if len(f)>=3:
            try: rows.append((float(f[0]),int(f[1]),int(f[2])))
            except ValueError: pass
    return {'samples':len(rows),'span_seconds':rows[-1][0]-rows[0][0] if rows else None,
            'cgroup_peak_bytes':max((r[1] for r in rows),default=0),
            'host_used_peak_bytes':max((r[2] for r in rows),default=0)}

def rank_summary(path: Path):
    ev=read_jsonl(path); counts=collections.Counter(x.get('phase') for x in ev)
    starts={int(x['global_step']):float(x['time_unix']) for x in ev if x.get('phase')=='training_step_start'}
    ends={int(x['global_step']):float(x['time_unix']) for x in ev if x.get('phase')=='training_step_end'}
    validated=[x for x in ev if x.get('phase')=='operation_trace_input_validated']
    versions=[x for x in ev if x.get('phase')=='parameter_versions_initialized']
    activation=[x for x in ev if x.get('phase')=='activation_offload_final']
    streamers=[x for x in ev if x.get('phase')=='frozen_block_streamer_final']
    return {
      'rank':int(path.stem.split('_')[-1]), 'counts':dict(counts),
      'steps':sorted(ends), 'step_start_unix':starts, 'step_end_unix':ends,
      'trace_hashes':sorted(str(x.get('trace_hash')) for x in validated),
      'trace_authorities':sorted(str(x.get('source_authority')) for x in validated),
      'initial_versions':[[int(x.get('generator_version',-1)),int(x.get('critic_version',-1))] for x in versions],
      'training_complete':counts['training_complete_skip_final_save']==1,
      'activation_live_final':[int(x.get('stats',{}).get('live_cpu_bytes',-1)) for x in activation],
      'streamer_resident_final':[int(x.get('stats',{}).get('resident_bytes',-1)) for x in streamers],
      'streamer_materialized_bytes':sum(int(x.get('stats',{}).get('materialized_bytes',0)) for x in streamers),
      'streamer_materialize_count':sum(int(x.get('stats',{}).get('materialize_count',0)) for x in streamers),
      'cuda_peak_allocated':max((int(x.get('cuda',{}).get('max_allocated_bytes',0)) for x in ev),default=0),
      'cuda_peak_reserved':max((int(x.get('cuda',{}).get('max_reserved_bytes',0)) for x in ev),default=0),
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',required=True); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]); ap.add_argument('--output',type=Path)
    a=ap.parse_args(); run=a.root/'artifacts/runs'/a.run_id
    nodes=[]
    for n in (0,1):
        lr=run/'launcher'/f'node_{n}'; kv=parse_kv(lr/'launch_manifest.txt')
        before=parse_memory_events(kv.get('memory_events_before','')); after=parse_memory_events(kv.get('memory_events_after',''))
        nodes.append({'node_rank':n,'final_return_code':int(kv.get('final_return_code','-1')),'resource':resource(lr/'resource.csv'),
                      'oom_delta':after.get('oom',0)-before.get('oom',0),'oom_kill_delta':after.get('oom_kill',0)-before.get('oom_kill',0),
                      'manifest_sha256':sha(lr/'launch_manifest.txt'),'console_sha256':sha(lr/'console.log')})
    files=sorted((run/'phase_trace').glob('rank_*.jsonl'),key=lambda x:int(x.stem.split('_')[-1])); ranks=[rank_summary(x) for x in files]
    exact=[r['steps']==EXPECTED_STEPS and r['training_complete'] and r['counts'].get('generator_optimizer_step')==1 and r['counts'].get('critic_optimizer_step')==5 and r['counts'].get('ema_step')==5 and r['initial_versions']==[[2,10]] and r['activation_live_final']==[0] and r['streamer_resident_final']==[0,0] for r in ranks]
    starts={s:[] for s in EXPECTED_STEPS}; ends={s:[] for s in EXPECTED_STEPS}
    for r in ranks:
      for s in EXPECTED_STEPS:
        if s in r['step_start_unix']: starts[s].append(r['step_start_unix'][s])
        if s in r['step_end_unix']: ends[s].append(r['step_end_unix'][s])
    step_wall={str(s):max(ends[s])-min(starts[s]) for s in EXPECTED_STEPS if starts[s] and ends[s]}
    window=max(max(v) for v in ends.values() if v)-min(min(v) for v in starts.values() if v) if all(starts.values()) and all(ends.values()) else None
    old=json.loads(OLD_RESULT.read_text())['timing_summary']; old_window=float(old['global_training_window_seconds'])
    timing={'global_step_wall_span_seconds':step_wall,'global_training_window_seconds':window,'official_weighted_step_seconds':window/5 if window else None,
            'samples_per_second':80/window if window else None,'relative_delta_vs_old':window/old_window-1 if window else None,
            'within_preregistered_10pct_diagnostic_band':abs(window/old_window-1)<=0.10 if window else False,
            'old_global_training_window_seconds':old_window,'old_global_step_wall_span_seconds':old['global_step_wall_span_seconds']}
    success=len(ranks)==16 and all(exact) and all(n['final_return_code']==0 and n['oom_delta']==0 and n['oom_kill_delta']==0 for n in nodes) and (run/'checkpoint-15').is_dir()
    payload={'schema_version':'distillschedule.typed_routea_world16.revalidation.v1','run_id':a.run_id,'status':'PASS' if success else 'FAIL','evidence_label':'Measured',
             'claim_scope':'one-run diagnostic consistency validation of new typed DMD-System execution against the frozen Route-A world16 baseline; not a formal speedup',
             'contract':{'world_size':16,'resolution':[384,640],'latent':[1,16,9,48,80],'steps':EXPECTED_STEPS,'zero_stage':1,'optimizer_offload':'cpu','trace':'rank16_v3_sync_steps20'},
             'nodes':nodes,'rank_count':len(ranks),'rank_exact':exact,'timing':timing,'gpu':{'max_allocated_bytes':max((r['cuda_peak_allocated'] for r in ranks),default=0),'max_reserved_bytes':max((r['cuda_peak_reserved'] for r in ranks),default=0)},
             'lifecycle':{'max_materialized_bytes_per_rank':max((r['streamer_materialized_bytes'] for r in ranks),default=0),'max_materialize_count_per_rank':max((r['streamer_materialize_count'] for r in ranks),default=0)},
             'limitations':['Single revalidation launch; not a repeated formal timing distribution.','Diagnostic instrumentation and block memory snapshots match the old gate, so these are not low-overhead formal timings.','The typed executor still invokes the pinned validated Helios algorithm entrypoint through a compatibility adapter.','Host 64-GiB reserve is reported separately; functional success does not imply reserve admission.'],
             'raw_phase_sha256':{x.name:sha(x) for x in files}}
    out=a.output or a.root/'artifacts/manifests'/f'{a.run_id}_result.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); print(json.dumps({'output':str(out),'status':payload['status'],'rank_count':len(ranks),'window':window,'relative_delta':timing['relative_delta_vs_old']},sort_keys=True))
if __name__=='__main__': main()
