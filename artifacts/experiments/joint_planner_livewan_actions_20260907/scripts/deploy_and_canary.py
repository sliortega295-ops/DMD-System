import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parents[1]
plan=json.loads((root/'manifests/deployment-plan.json').read_text())
hosts=[r['host'] for r in json.loads((root/'manifests/preflight.json').read_text())['preflight']]
def ssh(host,command,**kwargs):
    return subprocess.run(['ssh','-4','-T','-oBatchMode=yes','-oConnectTimeout=15','-oControlMaster=no','-oControlPath=none','-p','2222',host,command],**kwargs)
r=plan['remote_root'];deploy=plan['deployment'];head=plan['source_head']
bundle=Path(plan['bundle_path'])
with bundle.open('rb') as f:
    result=ssh(hosts[0],f'mkdir -p {r}/scripts {r}/manifests; test ! -e {r}/scripts/observer-source.bundle && cat > {r}/scripts/observer-source.bundle',stdin=f,capture_output=True,timeout=60)
    (root/'logs/source-upload.log').write_bytes(result.stdout+result.stderr)
    result.check_returncode()
cmd=f'''set -e
echo '{plan['bundle_sha256']}  {r}/scripts/observer-source.bundle' | sha256sum -c -
test ! -e {deploy}
git clone --shared --no-checkout /share/project/shuchenweng/yongyan_liu/DMD-System_deploy/joint-components-ceb2921985e5-20260907 {deploy}
git -C {deploy} fetch {r}/scripts/observer-source.bundle agent/dmd-joint-planner-20260907
git -C {deploy} checkout --detach {head}
test "$(git -C {deploy} rev-parse HEAD)" = {head}
test -z "$(git -C {deploy} status --short)"
mkdir {r}/transport-canary-r1
'''
result=ssh(hosts[0],cmd,capture_output=True,timeout=90)
(root/'logs/source-deployment.log').write_bytes(result.stdout+result.stderr)
result.check_returncode()
launcher=root/'scripts/run_transport_canary_node.sh'
with launcher.open('rb') as f:
    ssh(hosts[0],f'test ! -e {r}/scripts/run_transport_canary_node.sh && cat > {r}/scripts/run_transport_canary_node.sh',stdin=f,check=True,timeout=30)
record={'run_id':'jointp_livewan_actions_20260907_r1-transport-canary','source_head':head,'start_unix':time.time(),'training':'NOT_RUN','formal_timing':'NOT_RUN','node_results':[]}
def run(node):
    command=f'bash {r}/scripts/run_transport_canary_node.sh {node}'
    with (root/f'logs/transport-canary-node{node}.log').open('xb') as f:
        proc=ssh(hosts[node],command,stdout=f,stderr=subprocess.STDOUT,timeout=210)
    return {'node':node,'exit_code':proc.returncode,'command':command}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    record['node_results']=list(pool.map(run,range(2)))
record['stop_unix']=time.time()
record['status']='PASS' if all(r['exit_code']==0 for r in record['node_results']) else 'FAIL'
(root/'manifests/transport-canary-launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
