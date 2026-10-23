"""One source-bound selected native confirmation; no extra candidate runs."""
import concurrent.futures, datetime, json, shlex, subprocess, sys
from pathlib import Path
from distillschedule.workloads.best_known.dispatch import run_resolved
from distillschedule.workloads.best_known.registry import write_new_json
E=Path(__file__).resolve().parents[1]
root=Path('/home/lyy/Helios/DMD-System/worktrees/joint-planner-20260907')
plan=json.loads((E/'reports/search/execution-plan-confirmation.json').read_text())
dep=json.loads((E/'manifests/deployment-r2.json').read_text())
H='bf59e39b8a6fe158ad15ebae07be7df71369f5fd'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==H
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
assert plan['integration_head']==H and plan['mode']=='diagnostic-timing'
assert plan['search_candidate']['id'].startswith('joint-h3v2-d2-n128-w')
assert 'model_plan' in plan
dep['assets']=plan['asset_config']
assert all(r['status']=='PASS' and r['cleanup']=='PASS_GPU_EMPTY' for r in dep['results'])
code=r'''import fcntl,json,pathlib,socket,subprocess,sys
x=json.load(sys.stdin);p=pathlib.Path(x['repo']); l=open('/tmp/distillschedule_gpu_node.lock','a');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
from distillschedule.workloads.best_known.dispatch import validate_local
validate_local(x['receipt'],root=p)
def cmd(v):return subprocess.check_output(v,text=True).strip()
assert cmd(['git','-C',str(p),'rev-parse','HEAD'])==x['head']
assert not cmd(['git','-C',str(p),'status','--porcelain','--untracked-files=all'])
assert not cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
assert not pathlib.Path(x['run_dir']).exists(), 'immutable run already exists'
s=socket.socket();s.bind(('',x['port']));s.close()
print(json.dumps(dict(status='GO_PRELAUNCH_RECHECK_NATIVE_LAUNCHER_REACQUIRES_LOCK',hostname=socket.gethostname(),gpu=cmd(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,driver_version','--format=csv,noheader']),cgroup=pathlib.Path('/sys/fs/cgroup/memory.events').read_text(),head=x['head'],run_dir=x['run_dir'])))
'''
def check(i):
 n=dep['assets']['nodes'][i]
 argv=['ssh','-4','-T','-oBatchMode=yes','-oConnectTimeout=15','-oControlMaster=no','-oControlPath=none','-p','2222',n['host'],shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+n['repo_root']+'/src:'+n['repo_root'],'LD_LIBRARY_PATH='+n['python_env']+'/lib',n['python_env']+'/bin/python','-c',code])]
 p=subprocess.run(argv,input=json.dumps(dict(receipt=plan,repo=n['repo_root'],head=H,run_dir=dep['assets']['run_root']+'/'+plan['run_id'],port=dep['assets']['master_port'])),text=True,capture_output=True,timeout=360)
 (E/'logs'/f'confirmation-preflight-node{i}.stdout').write_text(p.stdout);(E/'logs'/f'confirmation-preflight-node{i}.stderr').write_text(p.stderr)
 assert p.returncode==0,(i,p.stderr);return json.loads(p.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:checks=list(pool.map(check,range(2)))
write_new_json(E/'manifests/confirmation-go.json',dict(status='GO_MODEL_SELECTED_NATIVE_CONFIRMATION_2_CYCLES',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_id=plan['run_id'],source_head=H,checks=checks,previous_work='NOT_REPEATED',timing='CYCLE2_ONLY_NATIVE_NOT_PROFILE_NO_MATCHED_SPEEDUP_CLAIM',node_lock='Native launcher owns audit through teardown'))

# Record actual environments beside the raw remote action evidence before and after execution.
from distillschedule.search.execution import NativeTrialRunner
from distillschedule.search.runtime_identity import observed_binding, verify_measurement_runtime
from distillschedule.search.h3_action_costs import PRECISION
(E/'identity-rpc-confirmation').mkdir(exist_ok=False)
runner=NativeTrialRunner(workload=plan['profile_id'],assets=dep['assets'],identity=dict(source_head=H,asset_fingerprint=plan['asset_fingerprint']),output=E/'identity-rpc-confirmation',run_prefix=plan['run_id'],bootstrap_peer='joint-h3v2-d2-n128-w3',reference_id='joint-h3v2-d2-n128-w5')
before=[runner._rpc('runtime_identity',_node_rank=i,node_rank=i,receipt=plan) for i in (0,1)]
bind=observed_binding(before,source_head=H)
bind.update(source_head=H,asset_fingerprint=plan['asset_fingerprint'],semantic_fingerprint=plan['workload_semantics_hash'],precision_fingerprint=PRECISION)
write_new_json(E/'manifests/runtime-before-confirmation.json',dict(binding=bind,before=before))
print('START',plan['run_id'],flush=True)
try:
 rc=run_resolved(plan,coordinator_output=E/'coordinator-confirmation-r1')
except BaseException as e:
 write_new_json(E/'manifests/confirmation-controller-result.json',dict(status='FAIL',type=type(e).__name__,error=str(e),run_id=plan['run_id'],source_head=H))
 raise
write_new_json(E/'manifests/confirmation-controller-result.json',dict(status='PASS' if rc==0 else 'FAIL',returncode=rc,run_id=plan['run_id'],source_head=H))

after=[runner._rpc('runtime_identity',_node_rank=i,node_rank=i,receipt=plan) for i in (0,1)]
runtime=dict(before=before,after=after)
verify_measurement_runtime(dict(runtime_identity_evidence=runtime),bind)
write_new_json(E/'manifests/runtime-confirmation.json',runtime)
print('COMPLETE',plan['run_id'],rc,flush=True)
raise SystemExit(rc)
