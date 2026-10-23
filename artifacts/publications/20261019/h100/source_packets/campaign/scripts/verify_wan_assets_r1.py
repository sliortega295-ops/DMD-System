#!/usr/bin/env python3
"""Read-only transferred-asset integrity gate; zero CUDA/model construction."""
import hashlib, json, os, sys, time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

root=Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
evidence=root/'research_artifacts/five_workload_h100_20260930'
audit=json.loads((evidence/'manifests/wan_assets_audit_r1.json').read_text())
assets=Path(audit['asset_destination_root_proposed'])
out=Path('/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/artifacts/wan_matched_r1/asset_integrity.json')
if out.exists(): raise FileExistsError(out)
rows=[]; errors=[]; started=time.time()
for group in audit['asset_groups']:
    for spec in group['files']:
        path=assets/spec['destination_relative_path']
        if spec['is_symlink']:
            valid=path.is_symlink() and os.readlink(path)==spec['link_target'] and path.exists()
            rows.append({'path':str(path),'symlink':True,'status':'PASS' if valid else 'FAIL'})
            if not valid: errors.append(str(path))
            continue
        expected=spec.get('live_file_sha256')
        if expected is None:
            suffixes=[(key,value) for key,value in audit['known_file_checksums'].items() if str(path).endswith(key)]
            if len(suffixes)==1: expected=suffixes[0][1]
            elif '/blobs/' in str(path) and len(path.name)==64: expected=path.name
        actual=sha(path)
        size=path.stat().st_size
        valid=size==spec['bytes'] and (expected is None or actual==expected)
        rows.append({'path':str(path),'bytes':size,'sha256':actual,'expected_sha256':expected,
                     'status':('PASS' if expected else 'HASH_RECORDED_SEMANTIC_PAYLOAD_CHECK_FOLLOWS') if valid else 'FAIL'})
        if not valid: errors.append(str(path))
# Verify immutable manifests and every tensor against its frozen descriptor SHA.
from distillschedule.runtime.wan13b_upper_bound import Wan13BBootstrap
from runtime.static_world16_trace import StaticWorld16Trace
from runtime.gst_pool_contract import NegativePromptAsset
identity=audit['semantic_identities']
bootstrap=Wan13BBootstrap.load(assets/'wan13b/systems_only_bootstrap_v3_head04e6e44',expected_hash=identity['bootstrap'],verify_payloads=True)
model=bootstrap.validate_model_assets(assets/'wan13b/Wan2.1-T2V-1.3B-Diffusers-0fad780a',verify_payloads=True)
trace=StaticWorld16Trace(assets/'m0/traces/syn_feature_trace_rank16_v4_strict_sync_steps20',variant='no-gan')
assert trace.bundle_hash==identity['trace_bundle'] and trace.binding_hash==identity['trace_binding']
for rank in range(16):
    loaded=trace._load(rank)
    print(f'verified trace rank={rank} tensors={len(loaded.tensors)}',flush=True)
negative=NegativePromptAsset.load(assets/'m1/negative_prompt_official_v1')
assert negative.manifest['manifest_hash']==identity['official_negative_prompt']
result={'run_id':'wan_resident_asset_integrity_20261014_r1','status':'FAIL' if errors else 'PASS',
        'rows':rows,'errors':errors,'model':model,'semantic_identities':identity,
        'all_16_rank_trace_tensor_descriptor_sha256':'PASS','gpu_work':'NOT_RUN',
        'wall_seconds':time.time()-started}
out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
sys.exit(bool(errors))
