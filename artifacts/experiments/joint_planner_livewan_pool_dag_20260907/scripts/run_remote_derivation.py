"""Transfer only small analysis source. Reuse existing native receipts, no GPU."""
import base64,hashlib,json,shlex,subprocess
from pathlib import Path
E=Path(__file__).resolve().parents[1]
repo=Path('/home/lyy/Helios/DMD-System/worktrees/joint-planner-20260907')
host='job-c6cfcde5-db4b-4527-8eb8-7518319a0277-master-0.shuchenweng.baai-vision_vision.dx-calc1.job@ssh.platform-multi.baai.ac.cn'
remote='/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/joint_planner_20260907/livewan_dag_r1'
oldrepo='/share/project/shuchenweng/yongyan_liu/DMD-System_deploy/joint-livewan-observation-78e501bc1727-20260907'
oldplan='/share/project/shuchenweng/yongyan_liu/DistillSchedule_runs/joint_planner_20260907/livewan_components_r1/native-inputs/execution-plan.json'
files={p.name:p.read_bytes() for p in (repo/'src/distillschedule/search/livewan_pool_contract.py',E/'scripts/derive_pool_contract.py')}
sha={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
prep=r'''import base64,hashlib,json,pathlib,sys
x=json.load(sys.stdin);p=pathlib.Path(x['root']);p.mkdir(exist_ok=False);(p/'source').mkdir();(p/'logs').mkdir()
for name,data in x['files'].items():
 b=base64.b64decode(data);assert hashlib.sha256(b).hexdigest()==x['sha'][name]
 with (p/'source'/name).open('xb') as f:f.write(b)
(p/'manifest.json').write_text(json.dumps({k:v for k,v in x.items() if k!='files'},indent=2)+'\n')
print(json.dumps(dict(status='PASS_DEPLOY_CPU_ANALYSIS_ONLY',root=str(p),sha=x['sha'])))
'''
def ssh(command,input=None,timeout=180):
 return subprocess.run(['ssh','-4','-T','-oBatchMode=yes','-oConnectTimeout=15','-oControlMaster=no','-oControlPath=none','-p','2222',host,command],input=input,text=True,capture_output=True,timeout=timeout)
x=dict(root=remote,files={k:base64.b64encode(v).decode() for k,v in files.items()},sha=sha,source_mode='UNCOMMITTED_CPU_ANALYSIS_SHA_BOUND',native_source_head='78e501bc1727013d94e356f697051b4d11cba4be',local_base_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),input_plan=oldplan,new_gpu_runs=0)
p=ssh(shlex.join(['python3','-c',prep]),input=json.dumps(x))
(E/'logs/deploy.stdout').write_text(p.stdout);(E/'logs/deploy.stderr').write_text(p.stderr)
assert p.returncode==0,(p.stdout,p.stderr)
cmd=shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+oldrepo+'/src','LD_LIBRARY_PATH=/tmp/distillschedule-helios-official-torch210/lib','/tmp/distillschedule-helios-official-torch210/bin/python',remote+'/source/derive_pool_contract.py','--module',remote+'/source/livewan_pool_contract.py','--module-sha',sha['livewan_pool_contract.py'],'--plan',oldplan,'--output',remote+'/analysis-r1'])
(E/'manifests/remote-command.json').write_text(json.dumps(dict(command=cmd,host=host,source_hashes=sha,input_plan=oldplan,new_gpu_runs=0),indent=2)+'\n')
p=ssh(cmd)
(E/'logs/derive.stdout').write_text(p.stdout);(E/'logs/derive.stderr').write_text(p.stderr)
assert p.returncode==0,(p.stdout[-1000:],p.stderr[-3000:])
result=json.loads(p.stdout)
(E/'reports/summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
