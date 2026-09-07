"""Correct the CPU-only constructor assumption, under owned-node locks.

Pinned LightX2V scheduler calls torch.cuda.current_device() during __init__.
This creates a CUDA context without loading weights. Preserve the failed r1;
no training implementation or environment changes are made.
"""
import ast,concurrent.futures,copy,json,shlex,subprocess
from pathlib import Path
E=Path(__file__).resolve().parents[1];P='/share/project/shuchenweng/yongyan_liu';H='878aff490ce28c661ac1d927f08f8846992baf45'
T=P+'/DMD-System_deploy/joint-h3-actions-'+H[:12]+'-20260907'; OUT=P+'/DistillSchedule_runs/joint_planner_20260907/h3_actions_r1'
a=copy.deepcopy(json.loads((E.parent/'native_macro_closed_loop_r2/configs/h3-assets.json').read_text()))
for n in a['nodes']:n['repo_root']=T
a.update(master_port=31487,run_root=OUT+'/native',timeout_seconds=3000,node_local_cache_root='/tmp/lyy-experiments/DMD-System/joint_h3_actions_20260907_r1')
x=ast.parse((E/'scripts/deploy_prepare.py').read_text());code=next(n.value.value for n in x.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
code=code.replace("x=json.load(sys.stdin);r=Path(x['target']);o=Path(x['output']);n=x['node']", """x=json.load(sys.stdin);r=Path(x['target']);o=Path(x['output']);n=x['node']
import fcntl,time
lock=open('/tmp/distillschedule_gpu_node.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def gpu_apps():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not gpu_apps()
cgroup_before=Path('/sys/fs/cgroup/memory.events').read_text()
""")
code=code.replace('assert torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized()', 'assert torch.equal(before,torch.get_rng_state())\nassert torch.cuda.is_initialized(), "pinned scheduler constructor is expected to initialize its device"')
code=code.replace('PASS_NATIVE_CPU_LOADER_DRAW_HOOKS','PASS_NO_WEIGHT_CONSTRUCTOR_AND_CPU_DRAW_HOOKS')
code=code.replace("'cuda_context':False,'weights_loaded':False", "'cuda_context':torch.cuda.is_initialized(),'gpu_allocated_bytes':torch.cuda.memory_allocated(),'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'weights_loaded':False")
code=code.replace("write_new_json(o/f'execution-plan-node{n}.json',p)", """for attempt in range(15):
 if not gpu_apps():break
 time.sleep(1)
assert not gpu_apps(), 'no-weight constructor left GPU process'
cgroup_after=Path('/sys/fs/cgroup/memory.events').read_text()
before=dict(l.split() for l in cgroup_before.splitlines());after=dict(l.split() for l in cgroup_after.splitlines())
assert all(before[k]==after[k] for k in ('oom','oom_kill'))
write_new_json(o/f'execution-plan-node{n}.json',p)""")
code=code.replace("'import':json.loads(s.stdout),'plan':p", "'import':json.loads(s.stdout),'plan':p,'cleanup':'PASS_GPU_EMPTY','cgroup_before':cgroup_before,'cgroup_after':cgroup_after")
def ssh(n,command,**kw):
 return subprocess.run(['ssh','-4','-T','-oBatchMode=yes','-oConnectTimeout=15','-oControlMaster=no','-oControlPath=none','-p','2222',a['nodes'][n]['host'],command],**kw)
p=ssh(0,f'test ! -e {OUT}/cpu-preflight-r2 && mkdir {OUT}/cpu-preflight-r2',capture_output=True,timeout=30);p.check_returncode()
def prepare(n):
 node=a['nodes'][n];cmd=shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+T+'/src:'+T,'LD_LIBRARY_PATH='+node['python_env']+'/lib',node['python_env']+'/bin/python','-c',code])
 p=ssh(n,cmd,input=json.dumps(dict(target=T,output=OUT+'/cpu-preflight-r2',assets=a,head=H,node=n)),text=True,capture_output=True,timeout=360)
 (E/'logs'/f'locked-prepare-node{n}.stdout').write_text(p.stdout);(E/'logs'/f'locked-prepare-node{n}.stderr').write_text(p.stderr)
 assert p.returncode==0,(n,p.stderr[-6000:]);return json.loads(p.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(prepare,range(2)))
assert results[0]['plan']['profile_sha256']==results[1]['plan']['profile_sha256']
(E/'manifests/deployment.json').write_text(json.dumps(dict(source_head=H,target=T,remote_root=OUT,assets=a,results=results,prior_failure='CPU_ONLY_ASSERTION_WRONG_PINNED_SCHEDULER_GET_DEVICE_INITIALIZES_CUDA_NO_WEIGHTS',source_fix=False),indent=2)+'\n')
(E/'manifests/execution-plan.json').write_text(json.dumps(results[0]['plan'],indent=2)+'\n')
print(json.dumps({'status':'PASS_BOTH_NODES_LOCKED_NO_WEIGHT_GATE','head':H,'checks':[r['import'] for r in results],'cleanup':[r['cleanup'] for r in results],'training':'NOT_RUN'}))
