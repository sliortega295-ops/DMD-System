from pathlib import Path
import json,hashlib,ast,os,subprocess
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100');O=R/'research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1';p=O/'scripts/verify_w6_raw_model_payload_r1.py';ast.parse(p.read_text())
x=json.load(open(R/'research_artifacts/minimax_h3_full_assets_20260906/manifests/asset_sha256_receipt.json'))
assert x['status']=='PASS' and sum(v['path'].startswith('FL2VA/transformer/') for v in x['files'])==15
text='''#!/usr/bin/env bash
set -Eeuo pipefail
R=/home/jiangzhikun/yongyan_liu/Helios-H100
O="$R/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1"
exec env CUDA_VISIBLE_DEVICES='' "$R/envs/five-workload-h100-20260930/bin/python" "$O/scripts/verify_w6_raw_model_payload_r1.py"
'''
q=O/'queue_r5/017s_w6_raw_model_fresh_payload_sha'
with q.with_suffix('.sh').open('x') as f:f.write(text)
subprocess.run(['bash','-n',str(q.with_suffix('.sh'))],check=True)
ready={'status':'PASS_CPU_ADMISSION','script_sha256':hashlib.sha256(text.encode()).hexdigest(),'workload_math_changed':False,'scope':'fresh66GB hash on H100 only when GPUidle before real W6; same validated downloaded checkpoint; no local payload/staging'}
t=q.with_suffix('.partial')
with t.open('x') as f:json.dump(ready,f,indent=2)
os.link(t,q.with_suffix('.ready.json'));t.unlink()
print(json.dumps(ready))
