#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import torch
import torch.distributed as dist

from runtime.critic_pool_checkpoint import (
    LogicalCheckpointPackage,
    restore_zero3_optimizer_partition_from_logical,
)
from runtime.deepspeed_backend import (
    build_client_optimizer_offload_overrides,
    build_zero_optimization_config,
)

rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE']); local=int(os.environ['LOCAL_RANK'])
assert world == 2
torch.cuda.set_device(local)
dist.init_process_group('nccl')
root=Path(os.environ['RESTORE_TOY_ROOT'])
if rank == 0:
    root.mkdir(parents=True, exist_ok=False)
    vals={
        'master': torch.arange(8,dtype=torch.float32),
        'first_moment': torch.arange(8,dtype=torch.float32)+10,
        'second_moment': torch.arange(8,dtype=torch.float32)+20,
    }
    payloads={}
    for key,value in vals.items():
        path=root/f'{key}.f32'; value.numpy().tofile(path)
        payloads[key]={'path':path.name,'bytes':path.stat().st_size,'dtype':'torch.float32'}
    manifest={
      'schema_version':'distillschedule.logical_checkpoint.v3','global_step':10,
      'input_trace_hash':'t'*64,
      'components':{
       'generator':{'parameter_version':2,'optimizer_step':2,'scheduler_state':{
        'base_lrs':[2e-6],'last_epoch':2,'verbose':False,'_step_count':3,
        '_get_lr_called_within_step':False,'_last_lr':[2e-6],'lr_lambdas':[None]},
        'model_state':{'logical_samples':16},'logical_numel':8,'parameter_names':['a','b'],
        'records':{'a':{'offset_elements':0,'numel':5,'shape':[5]},'b':{'offset_elements':5,'numel':3,'shape':[3]}},
        'payloads':payloads},
       'critic':{'parameter_version':10,'scheduler_state':{'last_epoch':10}}},
      'ema':{'optimization_step':10}}
    manifest['manifest_hash']=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest))
dist.barrier()

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.a=torch.nn.Parameter(torch.zeros(5)); self.b=torch.nn.Parameter(torch.zeros(3))
    def forward(self,x): return x*self.a.sum()+self.b.sum()
model=Toy()
optimizer=torch.optim.AdamW(model.parameters(),lr=2e-6,betas=(0.0,0.999),eps=1e-8,weight_decay=1e-3)
scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
zero=build_zero_optimization_config(stage=3,reduce_bucket_mib=1,allgather_bucket_mib=1,overlap_comm=False,contiguous_gradients=False,stage3_param_persistence_threshold=0,optimizer_offload_device='cpu',optimizer_offload_pin_memory=False)
config={'train_micro_batch_size_per_gpu':1,'gradient_accumulation_steps':1,'train_batch_size':2,'fp16':{'enabled':False},'bf16':{'enabled':True},'communication_data_type':'fp32','gradient_clipping':1.0,'zero_allow_untested_optimizer':True,'zero_optimization':zero}
config.update(build_client_optimizer_offload_overrides('cpu'))
import deepspeed
engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,config=config,dist_init_required=False)
package=LogicalCheckpointPackage.load(root)
record=restore_zero3_optimizer_partition_from_logical(engine,package,component='generator',partition_rank=rank,dp_world_size=world)
expected={
  0:{'a':[0.,1.,2.],'b':[5.,6.]},
  1:{'a':[3.,4.,0.],'b':[7.,0.]},
}[rank]
valid_lengths={0:{'a':3,'b':2},1:{'a':2,'b':1}}[rank]
checks={}
for name,p in engine.module.named_parameters():
    master=engine.optimizer.get_local_fp32_param(p).cpu()
    m=engine.optimizer.get_local_fp32_param(p,'exp_avg').cpu()
    v=engine.optimizer.get_local_fp32_param(p,'exp_avg_sq').cpu()
    e=torch.tensor(expected[name])
    expected_m=torch.zeros_like(e); expected_v=torch.zeros_like(e)
    valid=valid_lengths[name]
    expected_m[:valid]=e[:valid]+10
    expected_v[:valid]=e[:valid]+20
    checks[name]={
      'master':master.tolist(),'exp_avg':m.tolist(),'exp_avg_sq':v.tolist(),'low':p.ds_tensor.float().cpu().tolist(),
      'exact':bool(torch.equal(master,e) and torch.equal(m,expected_m) and torch.equal(v,expected_v) and torch.equal(p.ds_tensor.float().cpu(),e))}
base=engine.optimizer.state
steps=[]
for state in base.values():
    if 'step' in state: steps.append(int(state['step'].item()) if isinstance(state['step'],torch.Tensor) else int(state['step']))
status='PASS' if all(v['exact'] for v in checks.values()) and steps and set(steps)=={2} and engine.global_steps==2 else 'FAIL'
out={'status':status,'rank':rank,'record':record,'checks':checks,'optimizer_steps':steps,'global_steps':engine.global_steps,'zero_stage':engine.zero_optimization_stage()}
(root/f'rank_{rank:02d}.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
dist.barrier(); engine.destroy(); dist.destroy_process_group()
if status!='PASS': raise SystemExit(out)
