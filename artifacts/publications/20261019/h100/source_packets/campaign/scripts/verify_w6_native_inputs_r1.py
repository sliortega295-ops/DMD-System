"""Private CPU-only exact cache relocation and native model-layout admission."""
from pathlib import Path
import hashlib
import json
import os
import time

import torch
from h3_dmd2 import DMD2Config, prepare_batch
from train_dmd2 import model_config, checkpoint_files

ROOT=Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
BASE=ROOT/'assets/five_workload_h100_20260930/h3_ar_frozen_r1/assets/native_training_inputs_r1'
OUT=ROOT/'research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/artifacts/h3_w6_preparation'
MODEL=ROOT/'assets/MiniMax-H3/FL2VA/transformer'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
started=time.monotonic()
cache=BASE/'assets/native_prev56_cache_r2'
source=cache/'qualified_manifest.json'
assert sha(source)=='c5d0218e63e772d8bad8b0e9e92099c1114497db3068f164db0b689c74447f48'
original=json.loads(source.read_text());assert original['status']=='PASS_CACHE_SOURCE_AND_SCHEMA' and len(original['rows'])==16
config_path=BASE/'configs/dmd2_ar8_g1f1_r1.json'
config=DMD2Config(**json.loads(config_path.read_text()))
assert config.num_inference_steps==8 and config.critic_updates==1 and config.lora_rank==64
result={**original,'source_manifest_sha256':sha(source),'path_relocation_only':True,'rows':[]}
shapes=[]
for slot,row in enumerate(original['rows']):
    path=cache/Path(row['path']).name
    assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    raw=torch.load(path,map_location='cpu',weights_only=True)
    positive,negative=prepare_batch(raw,config.teacher_cfg_scale)
    assert positive['ref_visual_anchor'] is not None and positive['ref_audio_anchor'] is not None
    for key,value in positive.items():
        if isinstance(value,torch.Tensor):assert torch.isfinite(value).all(),key
    result['rows'].append({**row,'path':str(path),'original_path':row['path'],'logical_rank':slot})
    shapes.append({'logical_rank':slot,'tensors':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in positive.items() if isinstance(v,torch.Tensor)}})
assert len({r['sha256'] for r in result['rows']})==16
assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
assert sha(MODEL/'model.safetensors.index.json')=='fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb'
files,directory=checkpoint_files(MODEL);architecture=model_config(directory)
assert len(files)==13 and architecture['num_layers']==50
manifest=OUT/'h100_qualified_cache_manifest_r1.json'
with manifest.open('x') as f:json.dump(result,f,indent=2)
receipt={'status':'PASS_CPU_NATIVE_16_REAL_ROWS_MODEL_LAYOUT','cache_manifest':str(manifest),
         'cache_manifest_sha256':sha(manifest),'original_cache_manifest_sha256':sha(source),
         'rows':16,'unique_logical_samples':16,'cache_bytes':sum(r['bytes'] for r in result['rows']),
         'model_root':str(MODEL),'model_index_sha256':sha(MODEL/'model.safetensors.index.json'),
         'model_files':len(files),'model_bytes':sum(f.stat().st_size for f in files),
         'full_model_payload_hash':'PREVIOUS_RAW_DOWNLOAD_MANIFEST; NOT_REHASHED_IN_THIS_CPU_ROW_GATE',
         'architecture':architecture,'config_path':str(config_path),'config_sha256':sha(config_path),
         'shapes':shapes,'GPU':'NOT_RUN','model_loaded':'NOT_RUN','CUDA_initialized':False,
         'elapsed_seconds':time.monotonic()-started}
with (OUT/'native_inputs_cpu_gate_r1.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps({k:v for k,v in receipt.items() if k not in ('shapes','architecture')}),flush=True)
