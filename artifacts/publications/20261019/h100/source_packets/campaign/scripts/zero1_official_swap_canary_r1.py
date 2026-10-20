"""Eight-rank real checkpoint/LoRA/Adam swap gate, no14B assets/timing."""
import json,os
from pathlib import Path
import torch
import torch.distributed as dist
import deepspeed
from torch.utils.checkpoint import checkpoint
from distillschedule.runtime.helios_resident_zero1 import resident_zero1_config,gpu_zero1_bf16_route_compat,zero1_master_partitions
from distillschedule.runtime.helios_official_role_swap import OfficialRoleSwap
from distillschedule.runtime.hardware_profile import MicrobatchCommitEngine
from distillschedule.runtime.helios_attention_import import import_registered_attention_backend
rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
torch.cuda.set_device(device);dist.init_process_group('nccl',device_id=device)
assert dist.get_world_size()==8
import_registered_attention_backend('sdpa')
class LoRABlock(torch.nn.Module):
 def __init__(self):
  super().__init__();self.base=torch.nn.Linear(32,32,bias=False);self.base.requires_grad_(False)
  self.lora_A=torch.nn.Linear(32,4,bias=False);self.lora_B=torch.nn.Linear(4,32,bias=False)
 def forward(self,x):return self.base(x)+self.lora_B(self.lora_A(x))
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.block=LoRABlock()
 def forward(self,x):return checkpoint(self.block,x,use_reentrant=False)
def engine(seed):
 torch.manual_seed(seed);m=Model().to(device)
 opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-6,betas=(0.,.999),eps=1e-8,weight_decay=1e-3)
 with gpu_zero1_bf16_route_compat() as route:
  e,_,_,_=deepspeed.initialize(model=m,optimizer=opt,config=resident_zero1_config(),dist_init_required=False)
 assert type(e.optimizer).__name__=='BF16_Optimizer' and route['routed_calls']==1
 return e
def train(e,swap=None):
 w=MicrobatchCommitEngine(e,microbatch_count=2)
 for commit in range(2):
  for mb in range(2):
   if swap:swap.activate('generator',device)
   x=torch.arange(32,device=device,dtype=torch.bfloat16).reshape(1,32)/32+(rank+mb*8)*.001
   with w.microbatch(mb):
    pred=e.module(x);grad=pred.detach().float()*.03125
    if swap:
     swap.activate('critic',device)
     # A read-only Score phase with a real model call between G and backward.
     with torch.no_grad():_score=swap.engines[next(k for k,v in swap.roles.items() if v=='critic')].module(pred.detach())
     swap.activate('generator',device)
    loss=(pred.float()*grad).sum();w.backward(loss);w.step()
 assert e.global_steps==2
 return [p.detach().clone() for p in zero1_master_partitions(e)]
reference=engine(17);candidate=engine(17);score=engine(19)
expected=train(reference);swap=OfficialRoleSwap();swap.register(candidate,role='generator');swap.register(score,role='critic');swap.move_to_cpu(score)
actual=train(candidate,swap)
for a,b in zip(expected,actual,strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)
for a,b in zip(zero1_master_partitions(reference),zero1_master_partitions(candidate),strict=True):
 for key in ('exp_avg','exp_avg_sq'):
  torch.testing.assert_close(reference.optimizer.optimizer.state[a][key],candidate.optimizer.optimizer.state[b][key],rtol=0,atol=0)
# Ensure the real optimizer's updated BF16 image is the model's compute image.
for params,flat in zip(candidate.optimizer.bf16_groups,candidate.optimizer.bf16_groups_flat,strict=True):
 offset=0
 for p in params:
  assert p.data_ptr()==flat.narrow(0,offset,p.numel()).data_ptr();offset+=p.numel()
swap.activate('critic',device);swap.activate('generator',device)
for a,b in zip(expected,zero1_master_partitions(candidate),strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)
torch.cuda.synchronize(device)
out=Path(os.environ['API_CANARY_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
row={'status':'PASS_REAL_GPU_SWAP_CHECKPOINT_ADAM_EQUIVALENCE','rank':rank,'world_size':8,'actual_generator_commits':candidate.global_steps,'model_assets':'NOT_LOADED','Full5':'NOT_RUN','parity':'BITWISE_MASTER_AND_ADAM','source_head':os.environ['API_CANARY_HEAD'],'swap':swap.receipt(),'max_allocated':torch.cuda.max_memory_allocated(device),'max_reserved':torch.cuda.max_memory_reserved(device)}
with (out/f'rank_{rank:02d}.json').open('x') as f:json.dump(row,f,indent=2)
dist.barrier();dist.destroy_process_group()
