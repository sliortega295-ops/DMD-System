#!/usr/bin/env python3
"""No-weight-world8 API probe of existing41-unit HSDP no-grad lifecycle.

Not a Helios model/capacity/timing run. No StaticFSDP2P0Policy guard is relaxed.
Only this child wraps a public collective to count actual calls.
"""
import json,os,hashlib,subprocess
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard,FSDPModule,MixedPrecisionPolicy
from distillschedule.runtime.fsdp2_nograd_persistence import generator_nograd_persistent_unshard_scope
ROOT=Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
REPO=ROOT/'repos/DMD-System-h100-helios-official-swap-20261014-r1'
OUT=Path(os.environ['H100_HSDP_RETENTION_API_OUTPUT'])
rank=int(os.environ['RANK']);device=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
assert world==8 and rank==device
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()=='624515e7c8d1511aa688e95e69cf94f1a2a273f7'
assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
OUT.mkdir(exist_ok=True)
result={'status':'FAIL','rank':rank,'kind':'DIAGNOSTIC_LIBRARY_API_NO_REAL_HELIOS_MODEL','source':'624515e7c8d1511aa688e95e69cf94f1a2a273f7','torch':torch.__version__,'real_Helios_capacity':'NOT_RUN','real_training_time':'NOT_RUN'}
torch.cuda.set_device(device);dist.init_process_group('nccl',device_id=torch.device('cuda',device))
class Tiny41(torch.nn.Module):
 def __init__(self):
  super().__init__();self.root_scale=torch.nn.Parameter(torch.ones(32,dtype=torch.bfloat16));self.blocks=torch.nn.ModuleList([torch.nn.Linear(32,32,bias=False,dtype=torch.bfloat16) for _ in range(40)])
 def forward(self,x):
  for block in self.blocks:x=x+block(x)*0.01
  return x*self.root_scale
original=dist.all_gather_into_tensor
counts={'calls':0}
def counted(*a,**kw):counts['calls']+=1;return original(*a,**kw)
try:
 torch.manual_seed(42);a=Tiny41();b=Tiny41();b.load_state_dict(a.state_dict())
 mesh=init_device_mesh('cuda',(2,4),mesh_dim_names=('replicate','shard'))
 mp=MixedPrecisionPolicy(param_dtype=None,reduce_dtype=torch.float32,cast_forward_inputs=False)
 for model in (a,b):
  for block in model.blocks:fully_shard(block,mesh=mesh,mp_policy=mp,reshard_after_forward=True)
  fully_shard(model,mesh=mesh,mp_policy=mp,reshard_after_forward=True)
  for m in model.modules():
   if isinstance(m,FSDPModule):m.set_force_sum_reduction_for_comms(True);m.set_gradient_divide_factor(8.)
 inputs=[torch.full((1,32),(rank+1+i)/64,device=device,dtype=torch.bfloat16) for i in range(5)]
 dist.all_gather_into_tensor=counted
 counts['calls']=0
 with torch.no_grad():reference=[a(x) for x in inputs]
 baseline_calls=counts['calls']
 # Explicit cold sharding is part of this library probe, not a workload override.
 for model in (a,b):
  for m in reversed(list(model.modules())):
   if isinstance(m,FSDPModule):m.reshard()
 counts['calls']=0
 with torch.no_grad(),generator_nograd_persistent_unshard_scope(b,enabled=True,flush_allocator_cache=True) as receipt:
  actual=[]
  for x in inputs:actual.append(b(x));receipt['rollout_batch_count']+=1
 retained_calls=counts['calls']
 assert all(torch.equal(x,y) for x,y in zip(reference,actual)),'bitwise no-grad outputs changed'
 assert receipt['restored'] and receipt['unsharded_after_exit_count']==0 and receipt['fully_unsharded_before_exit_count']==41
 assert retained_calls==41 and baseline_calls>=retained_calls*4,'actual all-gather calls did not demonstrate reuse'
 dist.all_gather_into_tensor=original
 # Re-enter normal grad training only after the scope restores all units.
 losses=[model(inputs[0]).float().square().mean() for model in (a,b)]
 for loss in losses:loss.backward()
 for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
  assert na==nb and pa.dtype==pb.dtype and pa.grad is not None and pb.grad is not None
  assert torch.equal(pa.grad.to_local(),pb.grad.to_local()),'post-scope gradient changed'
 for model in (a,b):
  opt=torch.optim.SGD(model.parameters(),lr=0.001);opt.step()
 for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
  assert torch.equal(pa.to_local(),pb.to_local()),'post-scope optimizer result changed'
 torch.cuda.synchronize()
 result.update(status='PASS_BITWISE_HSDP41_NOGRAD_AND_POST_SCOPE_GRAD_UPDATE',mesh={'replicate':2,'shard':4},baseline_actual_allgather_calls=baseline_calls,retained_actual_allgather_calls=retained_calls,persistence=receipt,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),source_math_changed=False)
except BaseException as e:
 import traceback
 result.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
finally:
 dist.all_gather_into_tensor=original
 with (OUT/f'rank_{rank:02d}.json').open('x') as f:json.dump(result,f,indent=2)
 if dist.is_initialized():dist.destroy_process_group()
