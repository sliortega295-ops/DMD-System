import base64,hashlib,json,shlex,subprocess
from pathlib import Path
E=Path(__file__).resolve().parents[1];d=json.loads((E/'manifests/deployment.json').read_text());node=d['assets']['nodes'][0];run_id='jointp_h3_actions_20260907_r1-w3-correctness';root=d['assets']['run_root'].rsplit('/',1)[0]
src=(E/'scripts/analyze_action_canary.py').read_bytes();sha=hashlib.sha256(src).hexdigest();dest=root+'/analyze_action_canary_r1.py'
prep="import base64,hashlib,json,sys;from pathlib import Path;x=json.load(sys.stdin);b=base64.b64decode(x['b']);assert hashlib.sha256(b).hexdigest()==x['sha'];f=Path(x['p']).open('xb');f.write(b);f.close();print(x['sha'])"
def ssh(cmd,data=None):return subprocess.run(['ssh','-4','-T','-oBatchMode=yes','-oConnectTimeout=15','-oControlMaster=no','-oControlPath=none','-p','2222',node['host'],cmd],input=data,text=True,capture_output=True,timeout=120)
p=ssh(shlex.join(['python3','-c',prep]),json.dumps(dict(p=dest,b=base64.b64encode(src).decode(),sha=sha)));assert p.returncode==0,p.stderr
cmd='cd '+shlex.quote(node['repo_root'])+' && '+shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+node['repo_root']+'/src',node['python_env']+'/bin/python',dest,'--root',d['assets']['run_root']+'/'+run_id,'--output',root+'/analysis-action-r1','--head','878aff490ce28c661ac1d927f08f8846992baf45'])
with (E/'manifests/analysis-command-r1.json').open('x') as f:json.dump(dict(command=cmd,host=node['host'],analyzer_sha256=sha,new_gpu_run=False),f,indent=2)
p=ssh(cmd);(E/'logs/analysis-r1.stdout').write_text(p.stdout);(E/'logs/analysis-r1.stderr').write_text(p.stderr)
print(p.stdout);assert p.returncode==0,p.stderr
