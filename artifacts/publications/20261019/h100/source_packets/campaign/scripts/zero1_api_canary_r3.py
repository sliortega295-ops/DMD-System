"""World8, tiny no-asset optimizer API check. NOT a workload/capacity result."""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
import deepspeed
from distillschedule.runtime.helios_resident_zero1 import (
    resident_zero1_config, gpu_zero1_bf16_route_compat, zero1_master_partitions,
)
from distillschedule.runtime.hardware_profile import MicrobatchCommitEngine

rank=int(os.environ['RANK'])
device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
torch.cuda.set_device(device)
dist.init_process_group('nccl',device_id=device)
assert dist.get_world_size()==8
torch.manual_seed(0)
model=torch.nn.Linear(16,16).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=2e-6,betas=(0.,.999),eps=1e-8,weight_decay=1e-3)
original=deepspeed.runtime.engine.DeepSpeedEngine._configure_fp16_optimizer
with gpu_zero1_bf16_route_compat() as audit:
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,
        config=resident_zero1_config(),dist_init_required=False)
assert deepspeed.runtime.engine.DeepSpeedEngine._configure_fp16_optimizer is original
assert type(engine.optimizer).__name__=='BF16_Optimizer'
assert engine.optimizer.grad_acc_dtype==torch.float32
masters=zero1_master_partitions(engine)
assert len(masters)==1 and masters[0].device.type=='cuda' and masters[0].dtype==torch.float32
wrapped=MicrobatchCommitEngine(engine,microbatch_count=2)
for index in range(2):
    with wrapped.microbatch(index):
        loss=engine.module(torch.ones(1,16,device=device,dtype=torch.bfloat16)).float().square().mean()
        wrapped.backward(loss)
        wrapped.step()
assert engine.global_steps==1
state=engine.optimizer.optimizer.state[masters[0]]
assert all(state[k].device.type=='cuda' and state[k].dtype==torch.float32 for k in ('exp_avg','exp_avg_sq'))
torch.cuda.synchronize(device)
out=Path(os.environ['API_CANARY_OUTPUT']);out.mkdir(exist_ok=True,parents=True)
row=dict(status='PASS_API_ONLY',rank=rank,world_size=8,optimizer_class=type(engine.optimizer).__name__,
    master_numel=masters[0].numel(),full_parameter_numel=sum(p.numel() for p in model.parameters()),
    gradient_accumulation_dtype=str(engine.optimizer.grad_acc_dtype),route_audit=audit,
    actual_optimizer_commits=engine.global_steps,source_head=os.environ['API_CANARY_HEAD'],
    model_assets='NOT_LOADED',full5='NOT_RUN',capacity='NOT_RUN',timing='NOT_RUN')
with (out/f'rank_{rank:02d}.json').open('x') as f:json.dump(row,f,indent=2)
dist.barrier();dist.destroy_process_group()
