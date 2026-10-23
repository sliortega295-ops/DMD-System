"""Post-run readback only: no process killing, cache cleaning or environment edits."""
import argparse,concurrent.futures,json,shlex,subprocess
from pathlib import Path
E=Path(__file__).resolve().parents[1]
d=json.loads((E/'manifests/deployment-r2.json').read_text())
p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibration','confirmation']);a=p.parse_args()
plan=json.loads((E/('manifests/execution-plan-r2.json' if a.stage=='calibration' else 'reports/search/execution-plan-confirmation.json')).read_text())
code=(E.parent/'h3_actions_r1/scripts/cleanup_readback.py').read_text()
def check(i):
    n=d['assets']['nodes'][i]
    payload=dict(node=i,run_id=plan['run_id'],output=d['remote_root']+'/native/'+plan['run_id'],port=d['assets']['master_port'])
    r=subprocess.run(['ssh','-4','-T','-oBatchMode=yes','-oConnectTimeout=15','-oControlMaster=no','-oControlPath=none',
        '-p','2222',n['host'],shlex.join(['python3','-c',code])],input=json.dumps(payload),text=True,capture_output=True,timeout=60)
    with (E/'logs'/f'cleanup-{a.stage}-node{i}.stdout').open('x') as f:f.write(r.stdout)
    with (E/'logs'/f'cleanup-{a.stage}-node{i}.stderr').open('x') as f:f.write(r.stderr)
    assert r.returncode==0,r.stderr
    x=json.loads(r.stdout);assert x['status']=='PASS',x
    return x
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(check,(0,1)))
with (E/'manifests'/f'cleanup-{a.stage}.json').open('x') as f:json.dump(rows,f,indent=2)
print(json.dumps(rows,indent=2))
