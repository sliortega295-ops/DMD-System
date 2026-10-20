"""Derive a small W2 report from preserved, real run receipts; no GPU work."""
from pathlib import Path
import csv,datetime,hashlib,json
O=Path("/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1")
H=O/'artifacts/helios_h100_rollout_fence_probe_20261015_r1'
RUNS=O.parent
HEAD='8fc70e8ce9f6e0243604812120119c9f4c078e55'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def memory_values(value):
 allocated=[];reserved=[]
 if isinstance(value,dict):
  for k,v in value.items():
   if k=='max_allocated_gib' and isinstance(v,(float,int)):allocated.append(round(v*2**30))
   elif k=='max_allocated_bytes' and isinstance(v,int):allocated.append(v)
   elif k=='max_reserved_gib' and isinstance(v,(float,int)):reserved.append(round(v*2**30))
   elif k=='max_reserved_bytes' and isinstance(v,int):reserved.append(v)
   a,r=memory_values(v);allocated.extend(a);reserved.extend(r)
 elif isinstance(value,list):
  for v in value:
   a,r=memory_values(v);allocated.extend(a);reserved.extend(r)
 return allocated,reserved
def main():
 out=H/'reports/completed_fence_pair_r1';out.mkdir(parents=True,exist_ok=False)
 raw=H/'matched_pair_r1/result.json';x=json.load(open(raw))
 assert x['status']=='PASS_SAME_REVISION_SINGLE_DIAGNOSTIC_PAIR' and x['head']==HEAD
 bindings={str(raw):sha(raw)};arms={}
 for arm in ['control','candidate']:
  peaks=[];reserveds=[];rows=[];slots=[];elide=arm=='candidate'
  for mode in ['validate','correctness','timing']:
   rid=f'helios_h100_fence_{arm}_{mode}_20261015_r1';b=RUNS/rid;p=b/'summary.json';j=json.load(open(p))
   assert j['status']=='PASS' and j['expected_head']==HEAD
   assert j['physical_rank_count']==8 and j['logical_slots']==list(range(16))
   bindings[str(p)]=sha(p)
   if mode=='validate':continue
   assert j['logical_final_versions']==dict(generator=3,critic=15,ema=15)
   if mode=='correctness':assert j['execution_correctness_admitted']
   assert j['launcher_health']['cgroup_oom_delta']==dict(oom=0,oom_kill=0)
   assert (b/'logs/gpu_processes_after.txt').read_text().strip()==''
   for rank in range(8):
    p=b/'execution/ranks'/f'rank_{rank:02d}.json';r=json.load(open(p));bindings[str(p)]=sha(p)
    assert r['status']=='PASS' and r['updates']==dict(generator=1,critic=5,ema=5)
    assert r['world_size']==8 and r['model_swap_count']==r['transition_count']==r['parameter_streaming_count']==0
    expected=dict(critic_rollout_sync_count=0 if elide else 10,elide_explicit_rollout_sync=elide,
       generator_update_empty_cache_count=0 if elide else 2,generator_update_sync_count=0 if elide else 2,outer_full5_boundary_sync_preserved=True)
    assert r['rollout_fence_policy']==expected
    retention=r['generator_nograd_persistent_unshard'];assert retention['enabled'] and len(retention['microbatches'])==2
    for v in retention['microbatches']:
     p=v['persistence'];assert p['restored'] and p['unsharded_after_exit_count']==0 and p['fsdp_module_count']==41 and p['flush_allocator_cache']
    if mode=='timing':
     a,z=memory_values(r);peaks+=a;reserveds+=z
     slots.extend(r['hardware_profile']['logical_slots']);rows.append(dict(rank=rank,seconds=r['full5_seconds'],fences=r['rollout_fence_policy']))
   if mode=='timing':summary=j
  assert sorted(slots)==list(range(16)) and len(rows)==8
  arms[arm]=dict(seconds=summary['full5_seconds_max_rank'],rank_receipts=8,logical_slots=slots,rank_rows=rows,
     max_captured_allocated_bytes=max(peaks),max_captured_reserved_bytes=max(reserveds),
     memory_scope='maximum across preserved allocator snapshots; not continuously measured peak/minimum',
     updates=dict(generator=1,critic=5,ema=5),final_versions=dict(generator=3,critic=15,ema=15),
     CUDA_OOM=0,host_oom_delta=0,own_teardown='PASS_ORIGINAL_RUN_EMPTY',
     full_state_numeric_parity='NOT_RUN',formal_speedup_eligibility=False)
 control=arms['control']['seconds'];candidate=arms['candidate']['seconds']
 assert control==x['control_seconds'] and candidate==x['candidate_seconds']
 r=dict(status='PASS_ACTUAL_SAME_HEAD_SINGLE_NATIVE_DIAGNOSTIC_PAIR',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),head=HEAD,arms=arms,
    isolated_change='only existing internal rollout sync/empty_cache elision; retainedG/retention prepost allocator flush inbotharms',
    saved_seconds=control-candidate,ratio=control/candidate,replications=1,CV='NOT_MEASURED',profiled=False,
    frozen_official_swap_s=138.3719482459128,descriptive_cross_revision_ratio=138.3719482459128/candidate,
    strict_full_state_numeric_parity='NOT_RUN',formal_claimable=False,source_bindings=bindings)
 dump(out/'result.json',r)
 with (out/'arm_table.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['arm','native_Full5_seconds','physical_ranks','independent_logical_slots','max_captured_allocated_GiB','max_captured_reserved_GiB','numeric_full_state','formal_claimable']);writer.writeheader()
  for arm,a in arms.items():writer.writerow(dict(arm=arm,native_Full5_seconds=a['seconds'],physical_ranks=8,independent_logical_slots=16,max_captured_allocated_GiB=a['max_captured_allocated_bytes']/2**30,max_captured_reserved_GiB=a['max_captured_reserved_bytes']/2**30,numeric_full_state='NOT_RUN',formal_claimable=False))
 text=f"""# W2: retained G, then only internal rollout-fence elision

Exact source: {HEAD}. Native, one independent trial per arm; no profiler. Denominator: original world-max completed Full5 timer, G1/C5/EMA5.

| Arm | Full5 s | Rank/slot execution | Snapshot allocated GiB | Snapshot reserved GiB |
|---|---:|---|---:|---:|
| Retained-G control | {control:.9f} | 8 physical /16 independent slots PASS | {arms['control']['max_captured_allocated_bytes']/2**30:.3f} | {arms['control']['max_captured_reserved_bytes']/2**30:.3f} |
| Same retained-G + interior fence elision | {candidate:.9f} | 8 physical /16 independent slots PASS | {arms['candidate']['max_captured_allocated_bytes']/2**30:.3f} | {arms['candidate']['max_captured_reserved_bytes']/2**30:.3f} |

Single-pair observed saving {control-candidate:.6f}s; ratio {control/candidate:.6f}x. No median, CV or significance claim. Final G3/C15/EMA15; retained41units restore inboth logical B1 scopes; no remaining unsharded units; model swap/streaming/transition0. Candidate actually recorded0 interior rollout fences versus2G/10Critic-rollout syncs and2G empty_cache calls incontrol. Outer timer fences and retention pre/post allocator flush remain.

All actual validate/correctness/native receipts PASS; no new cgroup/CUDA OOM. Original own teardown had0 GPUcontexts. This does NOT assert the node is idle now; unrelated tasks appeared later.

Full-state cross-arm numerical parity NOT_RUN; formal speedup eligibility remains FALSE. The frozen official-role-swap reference138.371948s is from a different revision: its descriptive ratio {138.3719482459128/candidate:.6f}x is not a newly same-revision placement-only experiment. Memory is maximum across recorded snapshots; native reset policies prevent treating it as an independently continuous peak.
"""
 with (out/'report.md').open('x') as f:f.write(text)
 dump(out/'artifact_sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
 print(json.dumps(dict(path=str(out),status=r['status'],control=control,candidate=candidate,saving=control-candidate,ratio=control/candidate,formal_claimable=False)))
if __name__=='__main__':main()
