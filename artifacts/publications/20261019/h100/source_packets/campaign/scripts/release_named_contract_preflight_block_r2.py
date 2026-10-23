from pathlib import Path
import json,os,subprocess,datetime
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930');O=R/'h100_resident_baselines_20261014_r1'
run=R/'h3_h100_resident_boundary_fallback_20261014_r2';log=(run/'00_correctness_resident/logs/launcher.log').read_text()
assert 'H100 profile cannot override activation_policy=none' in log
assert 'full H3 loaded on CPU' not in log
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
x=json.load(open(run/'00_correctness_resident/summary.json'));assert not x['owned_alive_after_cleanup'] and x['cgroup_oom_delta']=={'oom':0,'oom_kill':0}
report={'status':'FAIL_CLOSED_OLD_FROZEN_NONE_PROFILE_NOT_A_CAPACITY_RESULT','failed_run':str(run),'model_load':'NOT_RUN','true_boundary_offload_capacity':'NOT_RUN','cause':'original h100-8-b16 explicitly forbids activation policy overrides; do not delete/bypass this guard','next':'new separately registered h100-8-b16-resident-boundary-cpu profile; original none control unchanged','new_scoped_commit':'d81ebb2014af9ea3a84011e3be198818e63cb1e7','W5_existing_timing':'preserved9bfd3ed nativeFSDP638.522138278 → GPUcache539.718383034; newfallback not a pure placement ratio','W6_next':'8rank tiny using the repaired FD-verifier subprocess and reuse completed detachedQKV primitive','cleanup':'zero GPUcontexts, no cgroupOOM','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (O/'reports/named_boundary_profile_preflight_block_r2.json').open('x') as f:json.dump(report,f,indent=2)
with (O/'research_log.jsonl').open('a') as f:f.write(json.dumps(dict(report,event='REGISTER_DISTINCT_CAPACITY_PROFILE_INSTEAD_OF_UNLOCKING_FROZEN_NONE'))+'\n')
p=O/'queue_r5/016z2_h3_boundary_launcher_delivery_fixed.disposition.json';t=p.with_suffix('.partial')
with t.open('x') as f:json.dump({'preserve_failed_artifacts':True,'action':'CONTINUE_OTHER_ADMITTED_STEPS','report':str(O/'reports/named_boundary_profile_preflight_block_r2.json')},f,indent=2)
os.link(t,p);t.unlink()
print(json.dumps(report))
