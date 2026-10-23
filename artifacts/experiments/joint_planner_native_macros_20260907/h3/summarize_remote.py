"""Reduce hash-bound remote H3 receipts; preserve numerical failures and all raw data."""
import argparse,json,re,subprocess,shlex
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence-root',type=Path,required=True,help='Original R3 controller evidence with manifests/configs')
parser.add_argument('--output',type=Path,required=True,help='New local summary, never overwritten')
parser.add_argument('--remote-analysis-name',required=True,help='New directory name beneath the self-repeat run')
args=parser.parse_args()
assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]+',args.remote_analysis_name)
assert not args.output.exists()
E=args.evidence_root;d=json.load(open(E/'manifests/deploy.json'));assets=json.load(open(E/'configs/h3-assets.json'))
code=r'''
import hashlib,json,statistics
from pathlib import Path
from distillschedule.search.h3_native import read_native_receipts,compare_native_two_cycle,summarize_native_two_cycle
from distillschedule.workloads.best_known.registry import file_sha256,git_head,repository_root,write_new_json
root=Path('/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/joint_planner_20260907/native_macros_r3/h3')
head=git_head(repository_root());assert head=='b1706f073d09a5de03926c6e216273898af754f3'
ids=['jointp_h3_macro_20260907_r3-gate-joint-h3v1-d1-n128','jointp_h3_macro_20260907_r3-gate-joint-h3v1-d2-n128','jointp_h3_selfrepeat_20260907_r3-gate-joint-h3v1-d1-n128']
all_rows=[];runs=[]
for rid in ids:
 r=root/rid;rows,bindings=read_native_receipts(r);all_rows.append(rows)
 summary=summarize_native_two_cycle(rows,expected_head=head,candidate_id=rows[0]['candidate_id'])
 existing=json.load(open(r/'bound-summary.json'));assert all(existing[k]==v for k,v in summary.items())
 resource=[]
 for i in (0,1):
  p=r/f'search-resources.node{i}.json';v=json.load(p.open());assert v['status']=='PASS' and v['node_rank']==i
  counts={k:[s['cgroup_events'][k] for s in v['samples']] for k in ('oom','oom_kill')}
  assert all(min(vals)==max(vals) for vals in counts.values())
  resource.append({'node_rank':i,'path':str(p),'sha256':file_sha256(p),'sample_count':len(v['samples']),
   'host_peak_sampled_bytes':max(s['cgroup_current_bytes'] for s in v['samples']),
   'min_driver_free_sampled_bytes':min(g['free_bytes'] for s in v['samples'] for g in s['gpus']),
   'cgroup_oom_delta':{k:max(vals)-min(vals) for k,vals in counts.items()}})
 memories=[c[k]['hbm'] for z in rows for c in z['cycles'] for k in ('memory_before','memory_after')]
 runs.append({'run_id':rid,'candidate_id':summary['candidate_id'],'source_head':head,'mode':'correctness',
  'summary_path':str(r/'bound-summary.json'),'summary_sha256':file_sha256(r/'bound-summary.json'),
  'rank_count':len(rows),'cycle_wall_seconds':summary['cycle_wall_seconds'],'formal_timing_eligible':False,
  'final_versions':summary['final_versions'],'native_contract_status':summary['status'],'peak_allocated_bytes':max(m['peak_allocated_bytes'] for m in memories),
  'peak_reserved_bytes':max(m['peak_reserved_bytes'] for m in memories),'rank_receipts':bindings,
  'census_per_rank_per_cycle':{'student_forward':24,'fake_forward':6,'teacher_forward':1,'student_backward':1,'fake_backward':5,'sample_identities_world':96},
  'grid_per_rank_per_cycle':{'checkpoint_wraps':300,'scoped_executions':600,'hits':600,'misses':0},
  'resources':resource,'teardown':{str(i):(r/f'launcher/node_{i}/teardown_status.txt').read_text().strip() for i in (0,1)}})
self_comparison=compare_native_two_cycle(all_rows[0],all_rows[2],expected_head=head)
path=root/ids[2]/__import__('sys').argv[1];path.mkdir(exist_ok=False)
write_new_json(path/'self-comparison.json',self_comparison)
cross_path=root/ids[1]/'search-gate/comparison.json';cross=json.load(cross_path.open())
def dist(vals):
 v=sorted(vals);return {'min':min(v),'median':statistics.median(v),'mean':statistics.mean(v),'max':max(v)}
def reduce_comparison(x,raw_path,left,right):
 components={};per_rank=[]
 for kind in ('student_parameters','student_optimizer','fake_parameters','fake_optimizer'):
  m=[v['metrics'] for r in x['ranks'] for v in r['sampled_state'].get('components',{}).get(kind,[])]
  components[kind]={'tested_sampled_tensors':len(m),'failed_sampled_tensors':sum(v['status']=='FAIL' for v in m),
   **{key:dist([v[key] for v in m]) for key in ('normalized_l2','cosine','mean_relative','max_abs','reference_peak_abs')}}
 for r in x['ranks']:
  per_rank.append({'rank':r['rank'],'sample_status':r['sample_status'],'rng_status':r['rng_status'],'state_status':r['sampled_state']['status'],'loss_status':r['loss_status'],
   'state_failed_tensors':{kind:sum(v['metrics']['status']=='FAIL' for v in vals) for kind,vals in r['sampled_state'].get('components',{}).items()},
   'losses':[{'cycle':i+1,'role':role,**m} for i,c in enumerate(r['losses']) for role,m in c.items()]})
 return {**{k:v for k,v in x.items() if k!='ranks'},'path':str(raw_path),'sha256':file_sha256(raw_path),'components':components,'ranks':per_rank,
  'first_student_loss_equal_ranks':sum(a['cycles'][0]['losses']['student']==b['cycles'][0]['losses']['student'] for a,b in zip(left,right)),
  'first_fake_loss_equal_ranks':sum(a['cycles'][0]['losses']['fake'][0]==b['cycles'][0]['losses']['fake'][0] for a,b in zip(left,right)),
  'second_fake_loss_equal_ranks':sum(a['cycles'][0]['losses']['fake'][1]==b['cycles'][0]['losses']['fake'][1] for a,b in zip(left,right))}
out={'status':'BLOCKED_NUMERICAL_GATE','source_head':head,'workload':'minimax_h3_grid1000_best','runs':runs,
 'cross_comparison':reduce_comparison(cross,cross_path,all_rows[0],all_rows[1]),
 'self_comparison':reduce_comparison(self_comparison,path/'self-comparison.json',all_rows[0],all_rows[2]),
 'calibration':'NOT_RUN','model_search':'NOT_RUN','native_timing_validation':'NOT_RUN','timing_attempts':0,'correctness_attempts_current_head':3,
 'threshold_changes':False,'profiler':'NOT_RUN','strict_winner':None,'canonical_winner_updated':False}
write_new_json(path/'outcome-summary.json',out)
print(json.dumps(out))
'''
n=assets['nodes'][0]
cmd=['ssh','-4','-T','-oBatchMode=yes','-oConnectTimeout=15','-oControlMaster=no','-oControlPath=none','-p',str(n['ssh_port']),n['host'],shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+d['target']+'/src:'+d['target'],'LD_LIBRARY_PATH='+n['python_env']+'/lib',n['python_env']+'/bin/python','-c',code,args.remote_analysis_name])]
p=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
with args.output.with_suffix('.stderr.log').open('x') as f:f.write(p.stderr)
assert p.returncode==0,p.stderr
x=json.loads(p.stdout)
with args.output.open('x') as f:json.dump(x,f,indent=2)
for name in ('cross_comparison','self_comparison'):
 c=x[name];print(name,{k:c[k] for k in ('status','loss_status','sampled_state_status','sample_status','rng_status','first_student_loss_equal_ranks','first_fake_loss_equal_ranks','second_fake_loss_equal_ranks')})
 for k,v in c['components'].items():print(k,'failed',v['failed_sampled_tensors'],'/',v['tested_sampled_tensors'],'nL2',v['normalized_l2'],'cosine',v['cosine'])
for r in x['runs']:print(r['run_id'],r['cycle_wall_seconds'],'peak GiB',r['peak_allocated_bytes']/2**30, 'reserved',r['peak_reserved_bytes']/2**30,'resources',r['resources'])
print(args.output)
