from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import torch
import torch.distributed as dist

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
import deepspeed
deepspeed.init_distributed(dist_backend='nccl')
path = Path('/share/project/shuchenweng/yongyan_liu/DMD-System_rb16_f5_pipeline_20260809/scripts/distillschedule/probe_routeb_full5_candidate_b.py')
spec = importlib.util.spec_from_file_location('candidate_b_executor_group_probe', path)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)
control, *_ = module._create_groups(module.TOPOLOGY)
dist.barrier(group=control)
if rank == 0:
    print('CANDIDATE_B_GROUP_PROBE_PASS', flush=True)
dist.destroy_process_group()
