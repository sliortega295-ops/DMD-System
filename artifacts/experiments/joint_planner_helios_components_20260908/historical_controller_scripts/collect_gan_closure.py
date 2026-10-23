"""Read and hash remote receipts; transfer compact metadata only, never tensors."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

E = Path(__file__).resolve().parents[1]
REMOTE = r'''
import hashlib,json,sys
from collections import Counter
from pathlib import Path
x=json.load(sys.stdin); root=Path(x['remote_root']); head=x['head']
def read(path):return json.loads(Path(path).read_text())
def ref(path):
 p=Path(path);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
expected_order=['g','r_all','c0','e0','c1','c2','c3','c4']
runs=[]
for trial in x['trials']:
 m=trial['measurement']
 for reference in m['native_evidence']:
  assert ref(reference['path'])==reference,reference['path']
 run=Path(m['native_evidence'][0]['path']).parent
 rows=[read(run/'execution/ranks'/f'rank_{i:02d}.json') for i in range(16)]
 assert {r['rank'] for r in rows}==set(range(16))
 compact=[]
 for r in rows:
  assert r['status']=='PASS' and r['source_head']==head
  assert r['candidate_id']==m['candidate_id'] and not r['profiled']
  assert r['final_versions']==dict(generator=202,critic=1010,ema=802)
  assert r['final_physical_versions']==dict(generator=4,critic=20,ema=802)
  assert r['final']['generator_logical_version']==202 and r['final']['critic_logical_version']==1010
  assert r['final']['generator_physical_step']==4 and r['final']['critic_physical_step']==20
  assert len(r['cycles'])==2
  cycles=[]
  for c in r['cycles']:
   assert c['status']=='PASS' and c['selected_order']==expected_order
   assert c['dispatch']['executed_operations']==expected_order
   assert c['peak_live_rollouts']==5
   for key in ('census','dtype','finite','sample_rng','versions'):
    assert c['semantic_validation'][key]=='PASS'
   counts=Counter()
   for a in c['actions']:counts.update(a['counts'])
   for key,value in dict(generator_commit=1,generator_optimizer=1,critic_commit=5,
                        critic_optimizer=5,generator_backward=1,critic_backward=10,
                        critic_forward=12,teacher_forward=2).items():assert counts[key]==value
   cycles.append(dict(cycle=c['cycle'],wall_seconds=c['wall_seconds'],
    dispatch=c['dispatch']['executed_operations'],counts=dict(counts),
    peak_live_rollouts=c['peak_live_rollouts'],memory=c['memory'],
    semantic_validation=c['semantic_validation']))
  compact.append(dict(rank=r['rank'],status=r['status'],cycles=cycles,
   final_versions=r['final_versions'],final_physical_versions=r['final_physical_versions'],
   receipt=ref(run/'execution/ranks'/f"rank_{r['rank']:02d}.json")))
 teardown=[]
 for i in range(2):
  path=run/'launcher'/f'node_{i}'/'teardown_status.txt'
  content=path.read_text();assert 'PASS' in content and 'gpu_process_count=0' in content
  teardown.append(dict(node=i,receipt=ref(path),text=content))
 runs.append(dict(trial=trial['name'],run_id=run.name,candidate_id=m['candidate_id'],
  cycle_wall_seconds=m['cycle_wall_seconds'],numerical_status=m['numerical_status'],
  semantics_status=m['semantics_status'],cleanup_status=m['cleanup_status'],
  resource_status=m['resource_status'],profiled=m['profiled'],ranks=m['ranks'],
  native_evidence=m['native_evidence'],rank_receipts=compact,teardown=teardown))
plan=root/'model-search-cc7e451bff73/gan'
prediction=read(plan/'search.json')
rejections=[{k:r[k] for k in ('assignment','status','gpu_execution','memory',
 'predicted_cycle1_ns','predicted_cycle2_ns')} for r in prediction['rejections']
 if r['status']=='PREDICTED_CAPACITY_FAIL']
controller=read(plan/'controller-r2/identity.json')
assert controller['native_worker_source_head']==head
assert ref(controller['cpu_controller_module'])['sha256']==controller['cpu_controller_sha256']
native_dirs=sorted(p.name for p in root.glob('jointp_hv_20260908_cc7e451_gan-*') if p.is_dir())
assert native_dirs==sorted(r['run_id'] for r in runs),native_dirs
print(json.dumps(dict(status='PASS_GAN_NATIVE_DISPATCH_16_RANKS_TWO_CYCLES',
 source_head=head,expected_native_order=expected_order,trials=runs,
 capacity_rejections=rejections,cpu_controller_identity=controller,
 native_run_directories=native_dirs,rejected_incumbent_native_execution='NOT_RUN',
 raw_search=ref(plan/'search.json'),raw_validation=ref(plan/'validation-r2/result.json'))))
'''


def main():
    deployment = json.loads((E / 'manifests/deployment-cc7e451bff73-r4.json').read_text())
    folder = E / 'controller/cc7e451bff73/gan/model-search'
    result = json.loads((folder / 'validate-r2/result.json').read_text())
    assert result['status'] == 'COMPLETE_NATIVE_VALIDATION'
    assert result['incumbent'] is None and result['strict_winner'] is None
    trials = [dict(name=p.parent.name, measurement=json.loads(p.read_text()))
              for p in sorted((folder / 'validate-r2/trials').glob('*/measurement.json'))]
    assert len(trials) == 2
    node = deployment['assets']['gan']['nodes'][0]
    command = ['ssh', '-4', '-T', '-oBatchMode=yes', '-oConnectTimeout=15',
               '-oControlMaster=no', '-oControlPath=none', '-p', str(node['ssh_port']), node['host'],
               shlex.join(['python3', '-c', REMOTE])]
    response = subprocess.run(command, input=json.dumps(dict(trials=trials,
        remote_root=deployment['remote_root'], head=deployment['source_head'])),
        text=True, capture_output=True, timeout=180, check=True)
    compact = json.loads(response.stdout)
    plan = json.loads((folder / 'plan/result.json').read_text())
    compact.update(workload=plan['workload'], binding=plan['binding'],
        component_costs=plan['component_costs'], space_size=plan['space_size'],
        feasible=plan['feasible'], uncalibrated=plan['uncalibrated'],
        selected_by_model=plan['selected'], candidates=plan['candidates'],
        prediction_errors=result['prediction_errors'], strict_winner=None,
        incumbent=None, improvement_over_incumbent=None,
        cycle2_speedup_diagnostic=result['cycle2_speedup_diagnostic'],
        ratio_scope='SAME_BASELINE_REPEAT_NOT_AN_OPTIMIZATION_SPEEDUP',
        canonical_winner_updated=False,
        numerical_claim='REVIEW_SAMPLED_PARITY_ONLY_NOT_STRICT_EQUIVALENCE')
    target = E / 'reports/gan-model-native-closure-cc7e451.json'
    with target.open('x') as f:
        json.dump(compact, f, indent=2); f.write('\n')
    print(json.dumps(dict(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        status=compact['status'], trials=[dict(run_id=t['run_id'], seconds=t['cycle_wall_seconds'])
                                       for t in compact['trials']]), indent=2))


if __name__ == '__main__':
    main()
