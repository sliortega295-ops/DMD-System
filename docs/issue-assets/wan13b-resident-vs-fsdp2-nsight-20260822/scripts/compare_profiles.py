#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def s(x): return x/1e9

def load(p): return json.loads(Path(p).read_text())

p=argparse.ArgumentParser();p.add_argument('--fsdp',required=True);p.add_argument('--resident',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
f=load(a.fsdp);r=load(a.resident); out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
assert f['status']==r['status']=='PASS'; assert f['aggregate']['rank_count']==r['aggregate']['rank_count']==16

def row(x):
 A=x['aggregate']; W=A['gpu_wall_ns']/16; M=A['metrics']; E=A['exclusive_ns']; C=A['nccl_collectives']; D=A['copy_directions']
 return {
  'wall_s':s(W),'compute_union_s':s(M['compute_union_ns']/16),'nccl_union_s':s(M['nccl_union_ns']/16),
  'nccl_hidden_s':s(M['nccl_hidden_by_compute_ns']/16),'nccl_not_overlapped_s':s(M['nccl_exposed_vs_compute_ns']/16),
  'copy_union_s':s(M['copy_union_ns']/16),'no_traced_s':s(E.get('no_traced_activity',0)/16),
  'all_gather_not_overlapped_s':s(C['all_gather']['exposed_vs_compute_ns']/16),
  'reduce_scatter_not_overlapped_s':s(C['reduce_scatter']['exposed_vs_compute_ns']/16),
  'all_reduce_not_overlapped_s':s(C['all_reduce']['exposed_vs_compute_ns']/16),
  'd2h_gib_per_rank':D['d2h']['bytes']/16/2**30,
  'h2d_gib_per_rank':D['h2d']['bytes']/16/2**30,
  'exclusive_s':{k:s(v/16) for k,v in E.items()},
 }
F=row(f);R=row(r)
comparison={'schema_version':'distillschedule.wan13b_matched_nsys_comparison.v1','status':'PASS_DIAGNOSTIC_ONLY','formal_speedup_eligible':False,'fsdp':F,'resident':R,'delta_fsdp_minus_resident':{k:F[k]-R[k] for k in F if isinstance(F[k],(int,float))},'definitions':{'not_overlapped':'same-GPU NCCL union minus intersection with non-NCCL kernels; not causal critical-path time','wall':'mean of 16 rank exact-range walls in profiled runs'}}
# stage rows
fs={x['label']:x for x in f['aggregate']['stages']}; rs={x['label']:x for x in r['aggregate']['stages']}; stages=[]
for label in fs:
 if label not in rs: continue
 def sr(x): return {'wall_s':s(x['gpu_wall_ns']/16),'compute_s':s(x['compute_ns']/16),'nccl_not_overlapped_s':s(x['nccl_exposed_ns']/16),'copy_not_overlapped_s':s(x['copy_exposed_ns']/16),'no_traced_s':s(x['no_traced_ns']/16)}
 stages.append({'stage':label,'fsdp':sr(fs[label]),'resident':sr(rs[label])})
comparison['stages']=stages
(out/'resident_vs_fsdp2_matched.json').write_text(json.dumps(comparison,indent=2,sort_keys=True)+'\n')

# plotting
fig,ax=plt.subplots(2,2,figsize=(16,10))
placements=['FSDP2','Resident']; values=[F,R]
keys=['compute','nccl','copy','compute+nccl','compute+copy','nccl+copy','compute+nccl+copy','no_traced_activity']
colors={'compute':'#2563eb','nccl':'#f97316','copy':'#8b5cf6','compute+nccl':'#0f766e','compute+copy':'#6d28d9','nccl+copy':'#c2410c','compute+nccl+copy':'#be123c','no_traced_activity':'#cbd5e1'}
bottom=np.zeros(2)
for k in keys:
 vals=[v['exclusive_s'].get(k,0) for v in values]
 ax[0,0].bar(placements,vals,bottom=bottom,label=k,color=colors[k]);bottom+=vals
ax[0,0].set_ylabel('profiled seconds / GPU');ax[0,0].set_title('Exclusive exact-range decomposition');ax[0,0].legend(fontsize=8,ncol=2)
metrics=['compute_union_s','nccl_union_s','nccl_not_overlapped_s','copy_union_s','no_traced_s']; ml=['compute union','NCCL union','NCCL not-overlapped','copy union','no traced']; x=np.arange(len(metrics)); w=.36
ax[0,1].bar(x-w/2,[F[k] for k in metrics],w,label='FSDP2');ax[0,1].bar(x+w/2,[R[k] for k in metrics],w,label='Resident');ax[0,1].set_xticks(x,ml,rotation=25,ha='right');ax[0,1].set_ylabel('seconds / GPU');ax[0,1].set_title('Union metrics (overlap-aware, not additive)');ax[0,1].legend()
labels=[x['stage'].replace('WAN13B/','') for x in stages]; x=np.arange(len(labels));
ax[1,0].bar(x-w/2,[x['fsdp']['wall_s'] for x in stages],w,label='FSDP2');ax[1,0].bar(x+w/2,[x['resident']['wall_s'] for x in stages],w,label='Resident');ax[1,0].set_xticks(x,labels,rotation=35,ha='right',fontsize=8);ax[1,0].set_ylabel('profiled seconds / GPU');ax[1,0].set_title('Matched stage wall');ax[1,0].legend()
collect=['all_gather_not_overlapped_s','reduce_scatter_not_overlapped_s','all_reduce_not_overlapped_s'];cl=['all-gather','reduce-scatter','all-reduce'];x=np.arange(3)
ax[1,1].bar(x-w/2,[F[k] for k in collect],w,label='FSDP2');ax[1,1].bar(x+w/2,[R[k] for k in collect],w,label='Resident');ax[1,1].set_xticks(x,cl);ax[1,1].set_ylabel('not-overlapped seconds / GPU');ax[1,1].set_title('NCCL by collective (not causal removable time)');ax[1,1].legend()
fig.suptitle('Wan1.3B matched Nsight diagnostic: world16 FSDP2 vs Resident',fontsize=15);fig.tight_layout();fig.savefig(out/'resident_vs_fsdp2_matched.png',dpi=180);plt.close(fig)
print(json.dumps({'FSDP2':F,'Resident':R,'delta_wall_s':F['wall_s']-R['wall_s']},indent=2))
