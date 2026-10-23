"""Read-only re-audit of the three preserved W5 H100 arms; no GPU work."""
from pathlib import Path
import csv,datetime,hashlib,json
O=Path("/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1")
HEAD='9bfd3eda959bd0c8415fe71ba64d3ad5f7e93e71'
SPEC=[
 ('resident_boundary_CPU','d81ebb2014af9ea3a84011e3be198818e63cb1e7','h3_h100_registered_resident_boundary_20261014_r3/01_timing_resident','checkpoint_boundary_cpu',600),
 ('FSDP_control',HEAD,'h3_h100_three_arm_matched_20261014_r4/02_timing_control','none',0),
 ('FSDP_GPU_attention_cache',HEAD,'h3_h100_cuda_attention_cache_20261014_r1/01_timing_candidate','none',0)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 out=O/'reports/completed_W5_three_arm_source_qualified_r2';out.mkdir(exist_ok=False)
 bindings={};arms={};samples={};rows=[]
 for label,head,relative,policy,copies in SPEC:
  base=O.parent/relative;path=base/'summary.json';j=json.load(open(path));bindings[str(path)]=sha(path)
  assert j['status']=='PASS_MATCHED_SEMANTIC_DIAGNOSTIC' and j['mode']=='timing' and j['profiler']=='NOT_RUN'
  assert j['physical_ranks']==j['rank_receipt_count']==8 and j['logical_slots']==16 and j['sample_identities']==96
  assert j['bootstrap_sha256']=='8181204333a373679eae03a3af6fcd03026c9199a6100e470f13c866a041e86f'
  assert j['cgroup_oom_delta']==dict(oom=0,oom_kill=0) and j['owned_alive_after_cleanup']==[]
  for rank in range(8):
   path=base/'rank_receipts'/f'rank_{rank:03d}.json';r=json.load(open(path));bindings[str(path)]=sha(path)
   assert r['status']=='PASS_REAL_WORKER_CENSUS_REPLAY_UPDATES_FINITE'
   assert set(r['local_state_finite_post_timer'])=={'fake_Adam','fake_parameters','student_Adam','student_parameters'}
   assert all(v['status']=='PASS_ACTUAL_LOCAL_SHARDS_FINITE' and not v['inside_timer'] for v in r['local_state_finite_post_timer'].values())
   c=r['actual_census']
   assert c['forward_counts']==dict(student=48,fake=12,teacher=2) and c['grad_forward_counts']==dict(student=2,fake=10,teacher=0)
   assert c['backward_counts']==dict(student=2,fake=10) and c['fixed_end_step_idx']==3 and c['logical_ranks']==[rank,rank+8]
   assert r['actual_optimizer_commits']==r['actual_scheduler_commits']==dict(student=1,fake=5)
   assert r['activation_policy']==policy and r['checkpoint_boundary_cpu_copies']==copies
   assert r['registered_adaln_parameters']==0 and r['dropped_adaln_parameter_numel']==13010457600
   assert r['Grid_replay_delta']==dict(cache_hit_count=1200,cache_miss_count=0,checkpoint_wrap_count=600,missing_key_count=0,scoped_execution_count=1200)
   digest=hashlib.sha256(json.dumps(c['samples'],sort_keys=True).encode()).hexdigest()
   if rank in samples:assert samples[rank]==digest
   else:samples[rank]=digest
   requested=None
   if copies:
    b=r['boundary_policy_actual_counters'];assert b['pin_memory'] and b['delta']['cpu_copy_count']==600
    requested=b['delta']['offloaded_logical_bytes']
   rows.append(dict(arm=label,rank=rank,logical_slots=f'{rank},{rank+8}',forward_student=48,forward_fake=12,forward_teacher=2,backward_student=2,backward_fake=10,checkpoint_CPU_copies=copies,requested_boundary_D2H_bytes=requested,source_sample_list_sha256=digest))
  arms[label]=dict(source_head=head,summary=str(base/'summary.json'),seconds=j['full5_seconds'],peak_allocated_bytes=j['peak_allocated_bytes'],peak_reserved_bytes=j['peak_reserved_bytes'],host_observed_cgroup_peak=j['cgroup_current_observed_peak'],host_scope=j['host_peak_basis'],driver_free_minimum=j['driver_free_minimum'],activation_policy=policy,physical_ranks=8,logical_slots=16,independent_samples=96,replications=1,strict_numerical_parity='NOT_RUN',formal_claimable=False)
 result=dict(status='PASS_REAUDIT_PRESERVED_THREE_ARMS_NOT_NEW_EXPERIMENT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workload='W5_Grid1000_APPROXIMATE_NOT_W6',common_start='H100_ONLY_NOT_A100_SLOT_STATE_REPLAY',bootstrap_sha256=j['bootstrap_sha256'],arms=arms,source_bindings=bindings,rank_rows=rows,sample_ids_exact_across_arms=True,pure_no_offload_resident='CUDA_OOM_FAKE2_BACKWARD_PRESERVED_NOT_RETRIED',
  ratio_FSDP_to_candidate=arms['FSDP_control']['seconds']/arms['FSDP_GPU_attention_cache']['seconds'],ratio_named_CPU_staged_Resident_to_candidate=arms['resident_boundary_CPU']['seconds']/arms['FSDP_GPU_attention_cache']['seconds'],same_revision_FS_candidate=True,all_three_same_revision=False,Resident_contract='named CPUboundarycapacityfallback_not_pureplacement',native_times_not_updated=True,GPU_new='NOT_RUN',profile_pending='054B_last_actual_capture_r3',copy_bytes='application_requested_not_Nsight_physicalPCIe',MFU='NOT_MEASURED')
 dump(out/'result.json',result)
 with (out/'per_rank_contract.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (out/'arm_table.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['arm','source','native_seconds','peak_allocated_GiB','peak_reserved_GiB','host_sampled_cgroup_peak_GiB','activation_policy','strict_numeric'])
  for label,a in arms.items():w.writerow([label,a['source_head'],a['seconds'],a['peak_allocated_bytes']/2**30,a['peak_reserved_bytes']/2**30,a['host_observed_cgroup_peak']/2**30,a['activation_policy'],'NOT_RUN'])
 text='# W5 H100: preserved three-arm re-audit, no new GPU experiment\n\nApproximate Grid1000, authorized common H100 bootstrap; not A100 slot/RNG replay, not W6 continuous AR8. Timer: one Student1+Fake5 outer cycle,96rank-qualified consumed samples,16independent logicalslots/8physicalGPUowners.\n\n| Arm | Source | Native s | Allocated GiB | Reserved GiB | Activation policy |\n|---|---|---:|---:|---:|---|\n'
 for label,a in arms.items():text+=f"| {label} | {a['source_head'][:7]} | {a['seconds']:.9f} | {a['peak_allocated_bytes']/2**30:.3f} | {a['peak_reserved_bytes']/2**30:.3f} | {a['activation_policy']} |\n"
 text+='\nAll24rankreceipts rechecked: perphysicalrank forward48/12/2, grad2/10/0, backwards2/10; Student1/Fake5 optimizer/schedulercommits, AdaLNparams0/dropped13,010,457,600,1200replayhits/0miss and600wraps. Rank-qualified source sample identities exactly match acrossarms. All finitechecks occur aftertimer; no newhostOOM; original ownedteardowns empty.\n\nNamed Resident requires600CPUboundarycopies/rank and is NOT a pure GPU no-offload baseline. Its d81source is a separately registered capacitycontract, not the9bfdsource of bothFSDParms. PureResident Fake2backwardOOM remains preserved. The CPUcopy bytecensus denotes requestedlogicalbytes, NOT measured PCIe D2H/H2D.\n\nFSDPcontrol / GPUattentioncache single-trial ratio '+str(result['ratio_FSDP_to_candidate'])+'x (same9bfdsource). Full-model cross-arm numericstate parity NOT_RUN; formal strict speedup eligibility FALSE. No median/CV/significanceclaim. Hostpeak is10second cgroup.current sampling, not perrankRSS; driverfree minimum notcontinuouslymeasured. Native times frozen.054B profile must pass real8ranklatephase andBWcensus gates before attribution; profilewall is not a performance denominator.\n\nInitial R1 postprocessor incorrectly treated profiler statusstring NOT_RUN asboolean; its CPU failure is preserved, no modelrerun/rawmutation.\n'
 (out/'report.md').write_text(text)
 dump(out/'artifact_sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
 print(json.dumps(dict(path=str(out),status=result['status'],seconds={k:a['seconds'] for k,a in arms.items()},ratio=result['ratio_FSDP_to_candidate'],newGPU=False)))
if __name__=='__main__':main()
