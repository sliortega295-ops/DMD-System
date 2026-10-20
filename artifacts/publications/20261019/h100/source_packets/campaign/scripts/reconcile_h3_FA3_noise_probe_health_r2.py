from pathlib import Path
import json,subprocess,hashlib,datetime,os
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930');O=R/'h100_resident_baselines_20261014_r1'
p=O/'artifacts/h3_FA3_cuda_attention_SAC_GPU_gate_r2/receipt.json';x=json.load(open(p))
assert x['status']=='PASS_REAL_FA3_CUDA_CACHE_EXISTING_MULTI_METRIC_GATE' and len(x['cases'])==3
rows=[dict(sequence=c['sequence'],group=t['group'],trial=t['trial'],**g) for c in x['cases'] for t in c['comparisons'] for g in t['gradients']]
assert len(rows)==54 and all(v['status']=='PASS' for v in rows)
assert all(t['output']=='BITWISE_EQUAL' for c in x['cases'] for t in c['comparisons'])
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True);assert not apps.strip()
after={k:int(v) for k,v in map(str.split,Path('/sys/fs/cgroup/memory.events').read_text().splitlines())}
assert after['oom']==after['oom_kill']==0
log=(O/'queue_r5/016x2_h3_fa3_gradient_noise_floor.launcher.log').read_text()
assert 'FileExistsError' in log and 'h3_FA3_cuda_attention_SAC_GPU_gate_health_r1.json' in log
assert 'PASS_REAL_FA3_CUDA_CACHE_EXISTING_MULTI_METRIC_GATE' in (O/'logs/h3_FA3_cuda_attention_SAC_GPU_gate_r2.log').read_text()
summaries={}
for group in ('uncached_self_repeat','cuda_cached_cross'):
 a=[v for v in rows if v['group']==group]
 summaries[group]={'gradient_checks':len(a),'bitwise_different_comparisons':sum(not v['bitwise_equal'] for v in a),'max_mismatched_elements':max(v['mismatched_elements'] for v in a),'max_abs':max(v['max_abs'] for v in a),'max_normalized_L2':max(v['normalized_l2'] for v in a),'max_mean_relative':max(v['mean_relative'] for v in a),'min_cosine':min(v['cosine'] for v in a)}
health={'status':'PASS_ACTUAL_GPU_PRIMITIVE_POSTPROCESSING_RECONCILED_NO_RERUN','GPU_run':'actual completed R2 only; not repeated to repair bookkeeping','receipt':str(p),'receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'primitive_return_evidence':'actual PASS JSON emitted after all54 gradient checks and CUDA synchronize; shell subsequently collided with immutable R1 health filename','wrapper_command_status':'FAIL_FILE_EXISTS_PRESERVED','prior_health_not_overwritten':True,'current_GPU_contexts':apps,'current_cgroup_events':after,'OOM_proof':'stage-spanning counters continuously0; exactR2 local before snapshot was not persisted by failed postprocessor','training_model':'NOT_LOADED','numerical_gate':'unchanged DEFAULT_P0_PARITY; no bitwise-gradient claim','noise_summary':summaries,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (O/'manifests/h3_FA3_cuda_attention_SAC_GPU_gate_health_reconciled_r2.json').open('x') as f:json.dump(health,f,indent=2)
with (O/'research_log.jsonl').open('a') as f:f.write(json.dumps(dict(health,event='GPU_CACHE_LIBRARY_NOISE_PROBE_AND_HEALTH_FILENAME_RECONCILIATION'))+'\n')
q=O/'queue_r5/016x2_h3_fa3_gradient_noise_floor.disposition.json';tmp=q.with_suffix('.partial')
with tmp.open('x') as f:json.dump({'preserve_failed_artifacts':True,'action':'CONTINUE_OTHER_ADMITTED_STEPS','reason':'actual GPU primitive PASS, health-file naming collision only; original failure preserved and independent derived health added','derived_health':str(O/'manifests/h3_FA3_cuda_attention_SAC_GPU_gate_health_reconciled_r2.json')},f,indent=2)
os.link(tmp,q);tmp.unlink()
print(json.dumps(health))
