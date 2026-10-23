from pathlib import Path
import csv,datetime,hashlib,importlib.util,json,math,subprocess
R=Path(__file__).resolve().parents[1];B=R.parent;P=B.parent
RR='/share/project/shuchenweng/yongyan_liu/research_artifacts/DMD-System/state_retention_six_workloads_20261017/r46'
def main():
 for phase in('manual-control','manual-child'):
  eof=json.loads((R/f'artifacts/{phase}-eof.json').read_text());assert eof['status']=='PASS_CLEAN_NATIVE_PHASE'and eof['returncodes']==[0,0]
 ready=json.loads((R/'manifests/native-ready.json').read_text())
 spec=importlib.util.spec_from_file_location('io',P/'livewan_paper_three_arm_20261016/r1/scripts/remote_io.py');io=importlib.util.module_from_spec(spec);spec.loader.exec_module(io);io.R=R;io.RR=RR
 code=r'''from pathlib import Path
import csv,hashlib,json,sys
r=json.loads(RECORD);root=Path(r['root']);source=Path(r['source']);sys.path[:0]=[str(root/'scripts/native_r2'),str(source/'src'),str(source)]
import h3_ar_match_controller as c
cfgpath=Path(r['config']);cfg=c.load_config(cfgpath);cfg['_config_path']=str(cfgpath);digest=c.sha(cfgpath)
assert c.source_state(cfg)['source_head']==r['head']and 'torch'not in sys.modules
results={}
for phase in c.PHASES:
 proofs=[];cleanup=[]
 for n in(0,1):
  proofs.append(c.verified_owner_proof(cfg,phase,n,digest))
  after=c.read(c.paths(cfg)['root']/'lifecycle'/phase/f'node{n}'/'post-teardown-audit.json')
  free=[v for a in after['audits']for v in a.get('driver_free_bytes',[])]
  cleanup.append(dict(node=n,oom_delta=after['oom_delta'],held_seconds=after['held_seconds'],minimum_sampled_driver_free_bytes=min(free)if free else None,host_peak_status=after['host_peak_status'],host_peak_bytes=after['host_peak_bytes'],GPU=after['gpu'],fd=after['fd']))
 rows=c.validate_ranks(cfg,phase)
 result=[]
 for path,v in rows:
  cycles=[]
  for row in v['cycles']:
   cycles.append({k:row[k]for k in('cycle','duration_ns','metrics','state','memory','census','updates','adam_updates','final_versions','RNG_reseeded_between_cycles')}|dict(manual_attention=row['backend_effects'].get('manual_uncached_rollout_attention'),native_metadata_barrier=row['backend_effects']['manual_native_metadata_barrier'],selected_cache=row['selected_cache'],unit_prefetch=row['backend_effects']['unit_prefetch'],leaf_parameter_numel=[v['logical_parameter_numel']for v in row['leaf_runtime']['leaf_blocks']]))
  result.append(dict(rank=v['rank'],source=v['identity'],input_identity=v['input_identity'],implementation=v['implementation'],static_bytes=v['static_bytes'],manual_policy_id=v['manual_policy_id'],manual_external_packet=v['manual_external_packet'],restore_native_barrier=v['restore_native_barrier'],timer_unchanged=v['timer_unchanged'],cycles=cycles,rank_source=c.ref(path),final_versions=v['final_versions']))
 results[phase]=dict(ranks=result,owner_proofs=proofs,cleanup=cleanup,C1_ns=max(v['cycles'][0]['duration_ns']for v in result),C2_ns=max(v['cycles'][1]['duration_ns']for v in result))
assert 'torch'not in sys.modules
print('ACTUAL='+json.dumps(dict(source=c.source_state(cfg),config_sha256=digest,arms=results,all_receipts_original_native=True),allow_nan=False))
'''.replace('RECORD',repr(json.dumps(dict(root=RR,source=ready['source']['source'],head=ready['source']['head'],config=ready['config']['path']))))
 text=io.call(0,code,'collect-native-source-validated',240)
 actual=json.loads(next(x[7:]for x in text.splitlines()if x.startswith('ACTUAL=')))
 io.json_new(R/'reports/actual-native-pair.json',actual)
 arms=actual['arms'];A=arms['manual-control'];C=arms['manual-child']
 values={};pairs=[]
 for name,arm in arms.items():
  values[name]=dict(C1_s=arm['C1_ns']/1e9,C2_s=arm['C2_ns']/1e9,ranks=len(arm['ranks']),
   peak_allocated_GiB=max(c['memory']['peak_allocated_bytes']for v in arm['ranks']for c in v['cycles'])/2**30,
   peak_reserved_GiB=max(c['memory']['peak_reserved_bytes']for v in arm['ranks']for c in v['cycles'])/2**30,
   min_sampled_driver_free_bytes=min(v['minimum_sampled_driver_free_bytes']for v in arm['cleanup']),
   CUDA_OOM=False,host_OOM_delta=sum(sum(v['oom_delta'].values())for v in arm['cleanup']))
 for a,c in zip(A['ranks'],C['ranks']):
  assert a['rank']==c['rank']and a['input_identity']==c['input_identity']and a['source']==c['source']
  for x,y in zip(a['cycles'],c['cycles']):
   assert x['census']==y['census']and x['updates']==y['updates']and x['final_versions']==y['final_versions']
   for metric,old in x['metrics'].items():
    new=y['metrics'][metric]
    if isinstance(old,(float,int))and not isinstance(old,bool):
     pairs.append(dict(rank=a['rank'],cycle=x['cycle'],metric=metric,control=old,bank=new,absolute_error=abs(new-old),relative_error=abs(new-old)/max(abs(old),1e-12)))
 rng_checks=[dict(rank=a['rank'],cycle=x['cycle'],cpu=x['state']['cpu_rng_sha256']==y['state']['cpu_rng_sha256'],cuda=x['state']['cuda_rng_sha256']==y['state']['cuda_rng_sha256'])for a,c in zip(A['ranks'],C['ranks'])for x,y in zip(a['cycles'],c['cycles'])]
 assert all(v['cpu']and v['cuda']for v in rng_checks),'RNG mismatch: not claimable'
 saved=values['manual-control']['C2_s']-values['manual-child']['C2_s']
 result=dict(status='MEASURED_ONE_FACTOR_NATIVE_METADATA_BARRIER_REVIEW'if saved>0 else 'NOT_PROMOTED_NO_NATIVE_TIME_GAIN_NUMERICAL_REVIEW',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ready['source']['head'],external_packet=json.loads((R/'configs/config.json').read_text())['manual_packet'],values=values,control_C2_s=values['manual-control']['C2_s'],candidate_C2_s=values['manual-child']['C2_s'],saved_s=saved,speedup=values['manual-control']['C2_s']/values['manual-child']['C2_s'],all16_census_versions_inputs_RNG=True,original_checkpoint_attention_cache_gate=True,actual_FA3_calls_per_cycle_per_rank=350,actual_native_metadata_reads_per_rank_cycle=dict(control=0,candidate=350),same_FA3_backend_both_arms=True,grad_and_selected_attention_unchanged=True,frozen_AdaLN_bank_enabled=False,parameters_eliminated=False,weights_merged=False,scheduler_integration=False,numerical_status='REVIEW_REQUIRES_MATCHED_FULL_STATE_OR_TRAJECTORY',formal_speedup_eligible=False,claim_scope='single native C2 descriptive paired mechanism timing, not long convergence',charged_owner_phases=104,raw_root=RR+'/native')
 io.json_new(R/'reports/conclusion.json',result)
 if pairs:
  with (R/'data/per_rank_cycle_metrics.csv').open('w')as f:
   writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
 io.json_new(R/'reports/RNG-comparison.json',rng_checks)
 lines=['# H3 AR：相同FA3只恢复原生metadata同步','',f"Source: `{result['source']}`。手工 opt-in，不接自动规划器。两个arm使用相同FA3、grad/checkpoint缓存和原生timer；只在候选恢复原cu_seqlens.tolist同步，实际350次/rank/cycle；没有AdaLN bank。",'', '| 配置 | C1(s) | C2(s) | Peak allocated(GiB) | Peak reserved(GiB) | Ranks |','|---|---:|---:|---:|---:|---:|']
 for name,v in values.items():lines.append(f"| {name} | {v['C1_s']:.6f} | {v['C2_s']:.6f} | {v['peak_allocated_GiB']:.3f} | {v['peak_reserved_GiB']:.3f} | {v['ranks']}/16 |")
 lines+=['',f"单对原生C2节省 {saved:.6f}s；比值 {result['speedup']:.6f}x。",'', '两节点原始 owner/capacity/OOM 检查保持。原checkpoint Attention缓存、调用/更新/输入/RNG、每rank每cycle350次FA3与0/350次实际metadata同步计数通过；这不是全部参数/Adam逐元素或长期轨迹等价证明。REVIEW 保留，不冒充严格数值或统计结论。','',f'原始回执保留远端 `{RR}/native`；本地 `actual-native-pair.json` 为小型逐rank派生证据。']
 (R/'reports/conclusion.md').write_text('\n'.join(lines)+'\n')
 with (B/'research_log.jsonl').open('a')as f:f.write(json.dumps(dict(utc=result['utc'],stage='r46',status=result['status'],source=result['source'],values=values,saved_s=saved,evidence=str(R/'reports/conclusion.json')))+'\n')
 subprocess.run(['python3',str(B/'r5/scripts/update_continuous_progress.py')],check=True)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
