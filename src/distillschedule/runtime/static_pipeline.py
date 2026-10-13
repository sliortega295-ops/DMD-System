"""Checked PyTorch2.10 PP2/GPipe construction, not a native DMD task binding.

References: https://docs.pytorch.org/docs/2.10/distributed.pipelining.html
The model adapter must supply actual stage modules and static shape/dtype
examples. Never reinterpret TP/PP collaborators as independent samples.
"""
from __future__ import annotations
import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe


def build_static_gpipe(stage_module, *, stage_index, group, device, input_args,
                       output_args, loss_fn=None):
    if not dist.is_initialized() or dist.get_world_size(group)!=2:
        raise ValueError('explicit PP2 group required')
    if stage_index not in (0,1) or dist.get_rank(group)!=stage_index:
        raise ValueError('PP rank/stage mismatch')
    if not isinstance(stage_module,torch.nn.Module):
        raise ValueError('an actual native stage module is required')
    stage=PipelineStage(stage_module,stage_index,2,device,
        input_args=input_args,output_args=output_args,group=group)
    schedule=ScheduleGPipe(stage,n_microbatches=8,loss_fn=loss_fn,scale_grads=True)
    return stage,schedule
