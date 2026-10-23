"""CPU-only reconstruction of already-completed tiny W6 state exports."""
from pathlib import Path
import hashlib
import json
import sys

import torch

ROOT=Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
R=ROOT/'research_artifacts/five_workload_h100_20260930'
O=R/'h100_resident_baselines_20261014_r1'
W=R/'h100_w6_native_B16_GPU_tiny_gate_20261014_r2'
sys.path.insert(0,str(ROOT/'repos/DMD-System-h100-w6-native-logical-b16-20261014-r1/src'))
from distillschedule.runtime.p0_parity import tensor_parity_metrics

torch.set_num_threads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(arm):
    states=[];sources=[];identities=[]
    for rank in range(8):
        path=W/f'tiny_{arm}_correctness/rank_receipts/rank_{rank:02d}_tiny_trainable_state.pt'
        receipt=json.load(open(path.with_name(f'rank_{rank:02d}.json')))
        assert receipt['status']=='PASS_TINY_NATIVE_INTEGRATION_ONLY'
        assert receipt['tiny_state_sha256']==sha(path)
        assert receipt['final_versions']=={'student':2,'fake':2,'ema':0}
        assert receipt['logical_slots']==[rank,rank+8]
        identities.append(receipt['initial_LoRA_sha256'])
        states.append(torch.load(path,map_location='cpu',weights_only=True))
        sources.append({'path':str(path),'sha256':receipt['tiny_state_sha256']})
    assert all(i==identities[0] for i in identities)
    return states,sources,identities[0]

resident,sources,initial=load('resident');reference=resident[0]
for state in resident[1:]:
    for name,value in reference['parameters'].items():assert torch.equal(value,state['parameters'][name])

def reconstruct(shards,expected):
    if expected.ndim==0:
        assert all(torch.equal(shards[0],s) for s in shards)
        return shards[0]
    tensor=torch.cat(shards,dim=0)
    assert tuple(tensor.shape)==tuple(expected.shape),(tuple(tensor.shape),tuple(expected.shape))
    return tensor

results=[];all_sources={'resident':sources}
for arm in ('fsdp','candidate'):
    values,sources,identity=load(arm);assert identity==initial
    all_sources[arm]=sources
    assert all(set(v['parameters'])==set(reference['parameters']) for v in values)
    for name,expected in reference['parameters'].items():
        observed=reconstruct([v['parameters'][name] for v in values],expected)
        results.append({'arm':arm,'kind':'parameter','name':name,**tensor_parity_metrics(expected,observed)})
    for role,states in reference['Adam'].items():
        assert all(len(v['Adam'][role])==len(states) for v in values)
        # Exact same worker's make_optimizers() preserves ordered parameter
        # iteration. Names were not exported, so do not invent a name mapping.
        for index,state in enumerate(states):
            for key,expected in state.items():
                assert all(set(v['Adam'][role][index])==set(state) for v in values)
                observed=reconstruct([v['Adam'][role][index][key] for v in values],expected)
                results.append({'arm':arm,'kind':'Adam','role':role,'ordered_state_index':index,'field':key,
                                **tensor_parity_metrics(expected,observed)})
assert not torch.cuda.is_initialized()
summary={}
for arm in ('fsdp','candidate'):
    rows=[r for r in results if r['arm']==arm]
    summary[arm]={'comparisons':len(rows),'failures':sum(r['status']!='PASS' for r in rows),
                  'worst_normalized_l2':max(r['normalized_l2'] for r in rows),
                  'worst_mean_relative':max(r['mean_relative'] for r in rows),
                  'lowest_cosine':min(r['cosine'] for r in rows),'max_absolute':max(r['max_abs'] for r in rows)}
report={'status':'PASS_TINY_NUMERICAL_STATE_ONLY' if all(s['failures']==0 for s in summary.values()) else 'REVIEW_TINY_NUMERICAL_STATE',
        'summary':summary,'rows':results,'artifact_sources':all_sources,
        'thresholds':'unchanged DEFAULT_P0_PARITY','H100_real_full_model_state_parity':'NOT_RUN',
        'rollout_output_parity':'NOT_EXPORTED_BY_THIS_TINY_WORKER',
        'model_capacity_or_speedup':'NOT_PROVEN_BY_TINY_TEST','GPU':'NOT_RUN_CPU_POSTPROCESS_ONLY',
        'Resident_replicas_parameter_bitwise_identity':'PASS'}
with (O/'reports/w6_tiny_trainable_state_parity_r1.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'summary':summary,'scope':'tiny state only, not real full-model numerical PASS'}))
