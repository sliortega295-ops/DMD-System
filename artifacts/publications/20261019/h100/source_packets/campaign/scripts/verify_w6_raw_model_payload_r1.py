from pathlib import Path
import json,hashlib,datetime,os,subprocess
R=Path('/home/jiangzhikun/yongyan_liu/Helios-H100');C=R/'research_artifacts/five_workload_h100_20260930';O=C/'h100_resident_baselines_20261014_r1'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'hash before real model construction, never during matched GPU timing'
source=R/'research_artifacts/minimax_h3_full_assets_20260906/manifests/asset_sha256_receipt.json'
x=json.load(open(source));assert x['status']=='PASS' and x['hash_all'] is True
assert x['repo_id']=='MiniMaxAI/MiniMax-H3' and x['revision']=='42ed227ee7df40d41602854ae760620d6eb651fe'
rows=[v for v in x['files'] if v['path'].startswith('FL2VA/transformer/')]
assert len(rows)==15 and sum(v['path'].endswith('.safetensors') for v in rows)==13
out=O/'artifacts/w6_raw_model_payload_revalidation_r1';out.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
actual=[]
with (out/'hash_rows.jsonl').open('x') as f:
 for v in rows:
  p=R/'assets/MiniMax-H3'/v['path'];assert p.stat().st_size==v['actual_size']
  digest=sha(p);assert digest==v['actual_sha256'],str(p)
  row={'path':str(p),'bytes':p.stat().st_size,'sha256':digest,'mtime_ns':p.stat().st_mtime_ns,'status':'PASS_FRESH_PAYLOAD_SHA_EQUALS_VALIDATED_DOWNLOAD'}
  actual.append(row);f.write(json.dumps(row)+'\n');f.flush()
result={'status':'PASS_RAW_NATIVE_MODEL_13_SHARD_FRESH_SHA','HF_repo_id':x['repo_id'],'HF_revision':x['revision'],'source_receipt':str(source),'source_receipt_sha256':sha(source),'model_root':str(R/'assets/MiniMax-H3/FL2VA/transformer'),'model_identity_sha256':hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest(),'checkpoint_payload_bytes':sum(v['bytes'] for v in actual if v['path'].endswith('.safetensors')),'files':actual,'GPU':'NOT_RUN','A100_raw_payload_replay':'NOT_CLAIMED_BY_THIS_H100_DOWNLOAD_REVALIDATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (out/'receipt.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='files'}))
