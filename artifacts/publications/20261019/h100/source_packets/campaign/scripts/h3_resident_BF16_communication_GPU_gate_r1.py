"""Real DP8 mean/storage primitive; no H3 backbone construction."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from h3_a100.trainer_loop import H3A100LoopMixin

rank=int(os.environ['RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
try:
    owner=SimpleNamespace(config={'training':{'parameter_placement':'resident'}})
    for dtype in (torch.bfloat16,torch.float32):
        p=torch.nn.Parameter(torch.ones(4,device='cuda',dtype=dtype))
        p.grad=torch.full_like(p,rank+5) # mean of logical samples r+1 and r+9
        H3A100LoopMixin._synchronize_resident_gradients(owner,[p])
        assert p.dtype==p.grad.dtype==dtype
        torch.testing.assert_close(p.grad,torch.full_like(p,8.5),atol=0,rtol=0)
    torch.cuda.synchronize()
    out=Path(os.environ['H3_RESIDENT_COMM_GATE_OUTPUT']);out.mkdir(exist_ok=True,parents=True)
    with (out/f'rank_{rank:02d}.json').open('x') as f:json.dump({'status':'PASS_REAL_DP8_FP32_COMMUNICATION_BF16_STORAGE','rank':rank,'global_logical_samples':16,'mean':8.5,'model':'NOT_LOADED','dtype_tested':['BF16','FP32']},f,indent=2)
finally:dist.destroy_process_group()
