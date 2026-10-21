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
    require(params and all(p.ndim>0 and p.is_contiguous() and not isinstance(p,DTensor)
                           for p in params),
            'FSDP2 role path requires dense contiguous parameters',path='role.sharding')
    if len({p.dtype for p in params})>1:
        # Preserve BF16 frozen backbone and FP32 trainable adapters as separate
        # homogeneous parameter groups; never cast weights to force admission.
        owners=[(name,child) for name,child in model.named_modules() if name and
                list(child.parameters(recurse=False))]
        require(not list(model.parameters(recurse=False)) and owners and
                all(len({p.dtype for p in child.parameters(recurse=False)})==1
                    and len(list(child.parameters()))==len(list(child.parameters(recurse=False)))
                    for _,child in owners) and
                len({id(p) for _,child in owners for p in child.parameters(recurse=False)})==len(params),
                'mixed-dtype FSDP requires distinct homogeneous parameter submodules',path='role.sharding')
    require(type(optimizer) in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD),
            'original optimizer unsupported by FSDP2 migration',path='role.sharding')
    for parameter,slots in optimizer.state.items():
        require(all(not isinstance(t,torch.Tensor) or t.ndim==0 or t.shape==parameter.shape
                    for t in slots.values()),'unsupported non-pointwise Adam/moment layout',path='role.sharding')
    return shard_roots


def _install_trainable(model,optimizer,device,ranks,group,*,role,reshard_after_forward):
    """Reuse the role group collectively created before any model wrapping.

    Using DeviceMesh.from_group avoids creating a second communicator and
    prevents stale group names when trying several calibration candidates.
    """
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import FSDPModule
    ranks=tuple(ranks)
    require(dist.is_initialized() and len(ranks)>=2,'trainable role shards require a group',path='role.sharding')
    require(dist.get_rank() in ranks,'rank outside declared role shard group',path='role.sharding')
    require(group!=dist.GroupMember.NON_GROUP_MEMBER,
            'trainable role shard has no physical group',path='role.sharding')
    mesh=DeviceMesh.from_group(group,torch.device(device).type,
                               mesh=torch.tensor(ranks,dtype=torch.int32))
    shard_roots=admit(model,optimizer)
    dtypes={p.dtype for p in model.parameters()}
    if len(dtypes)==1:
        session=SimpleNamespace(state={'model':model,'optimizer':optimizer},device=torch.device(device))
        param_bytes,count,moments=shard_roots(session,('model',),mesh,reshard_after_forward=reshard_after_forward)
        dtype_groups=1
    else:
        # Existing native FSDP2 optimizer/Adam migration is reused for every
        # homogeneous leaf group. The outer root supplies ordering hooks.
        from torch.distributed.fsdp import fully_shard
        owners=[child for name,child in model.named_modules() if name and
                list(child.parameters(recurse=False))]
        keys=tuple(f'param_group_{i}' for i in range(len(owners)))
        state={'optimizer':optimizer, **dict(zip(keys,owners))}
        session=SimpleNamespace(state=state,device=torch.device(device))
        param_bytes,count,moments=shard_roots(session,keys,mesh,reshard_after_forward=reshard_after_forward)
        fully_shard(model,mesh=mesh,reshard_after_forward=reshard_after_forward)
        dtype_groups=len(owners)
    require(isinstance(model,FSDPModule),'FSDP2 role module not installed',path='role.sharding')
    # Keep default FSDP2 mean reduction. Setting gradient_divide_factor(1)
    # uses PREMUL_SUM, unsupported on Gloo. The producer VJP boundary cancels
    # the native mean without ever reducing the same gradient twice.
    return model,optimizer,mesh,{'role':role+'_sharded','parameter_bytes':param_bytes,
        'parameters':count,'migrated_optimizer_tensors':moments,'world':len(ranks),
        'vjp_pre_backward_factor':len(ranks),'dtype_groups':dtype_groups}


def install_generator(model,optimizer,device,generator_ranks,generator_group):
    return _install_trainable(model,optimizer,device,generator_ranks,generator_group,
                              role='generator',reshard_after_forward=False)


def install_trainable_score(model,optimizer,device,score_ranks,score_group):
    """Reuse actual FSDP2 optimizer migration, never create fresh Adam moments."""
    return _install_trainable(model,optimizer,device,score_ranks,score_group,
                              role='score_trainable',reshard_after_forward=True)


def sharded_model_identity(model,group,group_size):
    """Source checksum of all local shards, identical for all G ranks."""
    from torch.distributed.tensor import DTensor
    from .torch_backend import signature
    local=fingerprint({name:signature(value.to_local() if isinstance(value,DTensor) else value)
                       for name,value in model.state_dict().items()})
    hashes=[None]*group_size
    dist.all_gather_object(hashes,local,group=group)
    return fingerprint(hashes)


def admit_frozen_score(model):
    """Validate exact Score FSDP2 parameter owners before any collectives."""
    from torch.distributed.tensor import DTensor
    params=tuple(model.parameters())
    require(params and all(not p.requires_grad and p.ndim>0 and
            p.is_contiguous() and not isinstance(p,DTensor) for p in params),
            'frozen Score sharding requires immutable dense parameters',path='role.sharding')
    mixed=len({p.dtype for p in params})>1
    owners=[]
    if mixed:
        owners=[child for name,child in model.named_modules() if name and
                list(child.parameters(recurse=False))]
        require(not list(model.parameters(recurse=False)) and owners and
                all(len({p.dtype for p in child.parameters(recurse=False)})==1
                    and len(list(child.parameters()))==len(list(child.parameters(recurse=False)))
                    for child in owners) and
                len({id(p) for child in owners for p in child.parameters(recurse=False)})==len(params),
                'mixed dtype frozen Score requires distinct homogeneous leaf owners',path='role.sharding')
    return params,owners,mixed


def install_frozen_score(model,device,score_ranks,score_group):
    """Shard only a verified immutable Score backbone in its OWN process mesh.

    Score model parameters are never stepped and input-VJP is still computed on
    the receiving rank. This is not an optimizer-state migration, and does not
    silently freeze a trainable Score or create a fake optimizer.
    """
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import fully_shard,FSDPModule
    from torch.distributed.tensor import DTensor
    ranks=tuple(score_ranks)
    require(dist.is_initialized() and len(ranks)>1 and dist.get_rank() in ranks,
            'Score FSDP2 needs its allocated subgroup',path='role.sharding')
    require(score_group!=dist.GroupMember.NON_GROUP_MEMBER,
            'Score subgroup is unavailable',path='role.sharding')
    params,owners,mixed=admit_frozen_score(model)
    mesh=DeviceMesh.from_group(score_group,torch.device(device).type,
                               mesh=torch.tensor(ranks,dtype=torch.int32))
    param_bytes=sum(p.numel()*p.element_size() for p in params)
    for child in owners:
        fully_shard(child,mesh=mesh,reshard_after_forward=True)
    fully_shard(model,mesh=mesh,reshard_after_forward=True)
    require(isinstance(model,FSDPModule) and all(isinstance(p,DTensor) for p in model.parameters()),
            'Score FSDP2 sharding did not install',path='role.sharding')
    return model,mesh,{'role':'score_frozen_sharded','parameter_bytes':param_bytes,
                       'world':len(ranks),'immutable':True,'reshard_after_forward':True,
                       'dtype_groups':len(owners) if mixed else 1}
