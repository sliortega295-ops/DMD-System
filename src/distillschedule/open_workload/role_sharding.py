"""Conservative intra-Generator FSDP2 lowering for a role-pool pipeline.

Only Generator ranks participate. All world ranks establish the mesh in a
consistent order; pool collective order is prevalidated by PoolLayout.
No large-model GPU validity or performance is implied.
"""
from __future__ import annotations
from types import SimpleNamespace

import torch
import torch.distributed as dist
from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint


def admit(model, optimizer):
    """Use the existing generic FSDP2 parameter/Adam migration contract."""
    from .sharded import shard_roots
    from torch.distributed.tensor import DTensor
    require(isinstance(model,torch.nn.Module) and isinstance(optimizer,torch.optim.Optimizer),
            'FSDP2 requires an original model and optimizer',path='role.sharding')
    params=list(model.parameters())
    require(params and len({p.dtype for p in params})==1
            and all(p.ndim>0 and p.is_contiguous() and not isinstance(p,DTensor) for p in params),
            'FSDP2 role path requires homogeneous dense contiguous parameters',path='role.sharding')
    require(type(optimizer) in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD),
            'original optimizer unsupported by FSDP2 migration',path='role.sharding')
    for parameter,slots in optimizer.state.items():
        require(all(not isinstance(t,torch.Tensor) or t.ndim==0 or t.shape==parameter.shape
                    for t in slots.values()),'unsupported non-pointwise Adam/moment layout',path='role.sharding')
    return shard_roots


def install_generator(model,optimizer,device,generator_ranks):
    """ALL ranks must call the DeviceMesh constructor, including Score ranks."""
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import FSDPModule
    ranks=tuple(generator_ranks)
    require(dist.is_initialized() and len(ranks)>=2,'generator shards require a group',path='role.sharding')
    mesh=DeviceMesh(torch.device(device).type,torch.tensor(ranks,dtype=torch.int32))
    if dist.get_rank() not in ranks:
        return model,optimizer,mesh,{'role':'score_unsharded'}
    shard_roots=admit(model,optimizer)
    session=SimpleNamespace(state={'model':model,'optimizer':optimizer},device=torch.device(device))
    param_bytes,count,moments=shard_roots(session,('model',),mesh,reshard_after_forward=False)
    require(isinstance(model,FSDPModule),'FSDP2 generator module not installed',path='role.sharding')
    # Keep default FSDP2 mean reduction. Setting gradient_divide_factor(1)
    # uses PREMUL_SUM, unsupported on Gloo. The producer VJP boundary cancels
    # the native mean without ever reducing the same gradient twice.
    return model,optimizer,mesh,{'role':'generator_sharded','parameter_bytes':param_bytes,
        'parameters':count,'migrated_optimizer_tensors':moments,'world':len(ranks),
        'vjp_pre_backward_factor':len(ranks)}


def sharded_model_identity(model,group,group_size):
    """Source checksum of all local shards, identical for all G ranks."""
    from torch.distributed.tensor import DTensor
    from .torch_backend import signature
    local=fingerprint({name:signature(value.to_local() if isinstance(value,DTensor) else value)
                       for name,value in model.state_dict().items()})
    hashes=[None]*group_size
    dist.all_gather_object(hashes,local,group=group)
    return fingerprint(hashes)
