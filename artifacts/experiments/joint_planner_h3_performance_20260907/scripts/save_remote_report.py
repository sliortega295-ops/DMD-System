"""Remote-first metadata analysis; copy only bounded small derived evidence."""
import argparse, base64, hashlib, json, shlex, subprocess
from pathlib import Path
E=Path(__file__).resolve().parents[1]
d=json.loads((E/'manifests/deployment-r2.json').read_text())
n=d['assets']['nodes'][0]; root=d['target']; out=d['remote_root']
p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibration','confirmation']);a=p.parse_args()
def ssh(cmd,data=None,timeout=900):
    r=subprocess.run(['ssh','-4','-T','-oBatchMode=yes','-oConnectTimeout=15','-oControlMaster=no','-oControlPath=none',
        '-p','2222',n['host'],cmd],input=data,text=True,capture_output=True,timeout=timeout)
    return r
def upload(local,dest):
    b=local.read_bytes();sha=hashlib.sha256(b).hexdigest()
    c="import json,base64,hashlib,sys;from pathlib import Path;x=json.load(sys.stdin);b=base64.b64decode(x['b']);assert hashlib.sha256(b).hexdigest()==x['sha'];p=Path(x['p']);p.parent.mkdir(parents=True,exist_ok=True);f=p.open('xb');f.write(b);f.close();print(x['sha'])"
    r=ssh(shlex.join(['python3','-c',c]),json.dumps(dict(p=dest,b=base64.b64encode(b).decode(),sha=sha)),timeout=90)
    assert r.returncode==0,r.stderr
    return dict(local=str(local),remote=dest,sha256=sha,bytes=len(b))
def execute(script,args,label):
    cmd='cd '+shlex.quote(root)+' && '+shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+root+'/src:'+root,
        'LD_LIBRARY_PATH='+n['python_env']+'/lib',n['python_env']+'/bin/python',script,*args])
    with (E/'manifests'/f'{label}-command.json').open('x') as f:json.dump(dict(host=n['host'],command=cmd),f,indent=2)
    r=ssh(cmd)
    (E/'logs'/f'{label}.stdout').write_text(r.stdout);(E/'logs'/f'{label}.stderr').write_text(r.stderr)
    print(r.stdout,flush=True);assert r.returncode==0,r.stderr[-9000:]
    return json.loads(r.stdout)
def download(remote,local):
    c="from pathlib import Path;import sys,hashlib,json,base64;p=Path(sys.argv[1]);assert p.stat().st_size<4*1024**2,'derived evidence too large';b=p.read_bytes();print(json.dumps(dict(data=base64.b64encode(b).decode(),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))))"
    r=ssh(shlex.join(['python3','-c',c,remote]),timeout=90);assert r.returncode==0,r.stderr
    x=json.loads(r.stdout);b=base64.b64decode(x['data']);assert hashlib.sha256(b).hexdigest()==x['sha256']
    local.parent.mkdir(parents=True,exist_ok=True)
    with local.open('xb') as f:f.write(b)
    return dict(remote=remote,local=str(local),sha256=x['sha256'],bytes=len(b))
records=[]
for name in ('performance-closure.json','performance-closure.md','serializer-fix.patch'):
    records.append(upload(E/'reports'/name,out+'/reports/'+name))
records.append(upload(E/'manifests/stop.json',out+'/reports/stop.json'))
with (E/'manifests/final-report-remote.json').open('x') as f:json.dump(records,f,indent=2)
print('PASS persistent remote report; no GPU launched')
