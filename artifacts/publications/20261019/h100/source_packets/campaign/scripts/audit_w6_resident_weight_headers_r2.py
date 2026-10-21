"""Header-only attribution, no weight payload load and no CUDA import."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import struct

ROOT=Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
RUN=ROOT/'research_artifacts/five_workload_h100_20260930'
OWNER=RUN/'h100_resident_baselines_20261014_r1'
MODEL=ROOT/'assets/MiniMax-H3/FL2VA/transformer'
INPUT=json.load(open(OWNER/'artifacts/h3_w6_preparation/native_inputs_cpu_gate_r1.json'))
config=json.load(open(INPUT['config_path']))
headers={};files=[]
index=MODEL/'diffusion_pytorch_model.safetensors.index.json'
if not index.exists():
    candidates=list(MODEL.glob('*.index.json'));assert len(candidates)==1;index=candidates[0]
for name in sorted(set(json.load(open(index))['weight_map'].values())):
    p=MODEL/name
    with p.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];assert size<32<<20
        header=f.read(size);record=json.loads(header)
    headers.update({k:v for k,v in record.items() if k!='__metadata__'})
    files.append({'path':str(p),'header_sha256':hashlib.sha256(header).hexdigest(),'payload_bytes_NOT_READ':p.stat().st_size-size-8})
dtype=Counter();total=0;adaln_numel=0;adaln_bytes=0
for name,v in headers.items():
    n=math.prod(v['shape']);b=v['data_offsets'][1]-v['data_offsets'][0]
    total+=b;dtype[v['dtype']]+=b
    if 'adaln_proj' in name:adaln_numel+=n;adaln_bytes+=b
targets=[x.strip() for x in config['lora_targets'].split(',')]
matches=[]
for name,v in headers.items():
    if not name.endswith('.weight') or len(v['shape'])!=2:continue
    module=name[:-len('.weight')]
    if any(module==target or module.endswith('.'+target) for target in targets):
        out_features,in_features=v['shape']
        matches.append({'module':module,'base_shape':v['shape'],
                        'per_role_LoRA_numel':config['lora_rank']*(out_features+in_features)})
assert matches
per_role=sum(x['per_role_LoRA_numel'] for x in matches)
failure=json.load(open(RUN/'h100_w6_native_three_arm_matched_20261014_r2/resident_correctness/rank_receipts/rank_00_failure.json'))
setup=next(x for x in failure['memory'] if x['phase']=='after_setup')
result={'status':'PASS_HEADER_ONLY_STORAGE_MODEL_WITH_EXPLICIT_LIMITS','raw_weight_payload_bytes':total,
        'raw_dtype_bytes':dict(dtype),'raw_tensor_count':len(headers),'AdaLN_numel':adaln_numel,
        'AdaLN_raw_bytes':adaln_bytes,'native_LoRA_rank':config['lora_rank'],
        'per_role_FP32_LoRA_storage_bytes_derived_from_exact_shapes':per_role*4,
        'two_role_FP32_LoRA_storage_bytes':per_role*8,
        'checkpoint_raw_base_plus_two_role_FP32_LoRA_bytes':total+per_role*8,
        'loaded_all_BF16_base_bytes_shape_derived':sum(math.prod(v['shape'])*2 for v in headers.values()),
        'loaded_BF16_base_plus_FP32_LoRAs_shape_derived':sum(math.prod(v['shape'])*2 for v in headers.values())+per_role*8,
        'loader_dtype_evidence':'train_dmd2.py:106-120, meta model explicitly converts all parameters to BF16 before to_empty/load_full_weights; raw F32 headers do not prove GPU F32 placement',
        'after_setup_actual_cuda':setup,'OOM_actual_cuda':failure['memory_at_failure'],
        'failure_phase':failure['failure_phase'],'block':failure['last_block'],
        'commits_at_failure':failure['commits_at_failure'],
        'optimizer_state_at_failure':'no optimizer commit yet; native Adam states remain lazily unmaterialized',
        'AdaLN_grid_elimination':'NOT_ADMISSIBLE_W6_CONTINUOUS_SIGMA; would change workload unlike W5',
        'dynamic_residual':'not additive exact activation/NCCL attribution; full tensor storage, allocator and live saved tensors not instrumented',
        'header_files':files,'LoRA_shape_matches':matches,'GPU':'NOT_RUN_BY_THIS_ANALYZER',
        'weight_payload_read':False}
out=OWNER/'reports/w6_resident_header_storage_attribution_r2.json'
with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('header_files','LoRA_shape_matches','after_setup_actual_cuda','OOM_actual_cuda')}))
