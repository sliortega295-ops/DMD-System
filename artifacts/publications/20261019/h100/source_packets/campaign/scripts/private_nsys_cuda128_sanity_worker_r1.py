"""Eight-device profiler parser sanity only: no DMD weights or timing claim."""
from pathlib import Path
from datetime import timedelta
import json
import os
import torch
import torch.distributed as dist

rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
assert world==8 and rank==int(os.environ['LOCAL_RANK'])
assert os.environ.get('DISTILLSCHEDULE_INHERITED_GPU_LOCKS')=='1'
device=torch.device('cuda',rank);torch.cuda.set_device(device)
dist.init_process_group('nccl',device_id=device,timeout=timedelta(seconds=120))
torch.manual_seed(51+rank)
a=torch.randn(512,512,device=device,dtype=torch.bfloat16)
for _ in range(3):a@a
torch.cuda.synchronize();dist.barrier()
if rank==0:assert torch.cuda.cudart().cudaProfilerStart()==0
dist.barrier()
torch.cuda.nvtx.range_push(f'NSYS_COMPATIBILITY_ONLY/rank_{rank}')
host=torch.randn(512,512,dtype=torch.bfloat16,pin_memory=True)
data=host.to(device,non_blocking=True)
for _ in range(8):result=data@a
event=torch.cuda.Event();event.record();event.synchronize()
dist.all_reduce(result)
returned=result.to('cpu')
assert returned.shape==(512,512) and torch.isfinite(returned).all()
torch.cuda.synchronize();torch.cuda.nvtx.range_pop();dist.barrier()
if rank==0:assert torch.cuda.cudart().cudaProfilerStop()==0
out=Path(os.environ['H100_NSYS_SANITY_OUTPUT'])
with (out/f'rank_{rank:02d}.json').open('x') as f:
    json.dump({'status':'PASS_NO_MODEL_CUDA_NVTX_EVENT_COPY_NCCL_CANARY','rank':rank,
               'pid':os.getpid(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,
               'real_DMD':'NOT_RUN','DMD_timing':'NOT_RUN'},f,indent=2)
dist.destroy_process_group()
