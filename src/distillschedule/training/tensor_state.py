"""Safe local-shard checkpoint trees; no pickled DTensor/mesh/process groups.

Only replicated or dimension-sharded DTensors at a quiescent cycle boundary
are admitted. Topology/layout restoration uses the *existing live* mesh.
"""
from __future__ import annotations

_MARKER = "distillschedule.dtensor_local.v1"


def layout(tensor):
    from torch.distributed.tensor import Replicate, Shard
    placements = []
    for item in tensor.placements:
        if isinstance(item, Replicate):
            placements.append({"kind": "replicate"})
        elif isinstance(item, Shard):
            placements.append({"kind": "shard", "dim": item.dim})
        else:
            raise ValueError("checkpoint requires completed non-partial DTensor state")
    mesh = tensor.device_mesh
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype),
            "local_shape": list(tensor.to_local().shape), "placements": placements,
            "mesh": mesh.mesh.tolist(), "mesh_dim_names": list(mesh.mesh_dim_names) if mesh.mesh_dim_names else None,
            "device_type": mesh.device_type}


def pack(value):
    import torch
    from torch.distributed.tensor import DTensor
    if isinstance(value, DTensor):
        return {"_tensor_kind": _MARKER, "layout": layout(value),
                "local": value.to_local().detach().to(device="cpu", copy=True)}
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        if value.get("_tensor_kind") == _MARKER:
            raise ValueError("reserved checkpoint tensor marker in user state")
        return {key: pack(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(pack(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError(f"checkpoint state needs an explicit serializer: {type(value).__name__}")


def decode(value, meshes):
    import torch
    from torch.distributed.tensor import DTensor, Replicate, Shard
    if isinstance(value, dict) and value.get("_tensor_kind") == _MARKER:
        if set(value) != {"_tensor_kind", "layout", "local"}:
            raise ValueError("invalid checkpoint local tensor record")
        info, local = value["layout"], value["local"]
        if (not isinstance(local, torch.Tensor) or list(local.shape) != info["local_shape"]
                or str(local.dtype) != info["dtype"]):
            raise ValueError("checkpoint local shard shape/dtype changed")
        mesh = next((m for m in meshes if m.mesh.tolist() == info["mesh"]
                     and (list(m.mesh_dim_names) if m.mesh_dim_names else None) == info["mesh_dim_names"]
                     and m.device_type == info["device_type"]), None)
        if mesh is None:
            raise ValueError("checkpoint DTensor topology/layout has no live owner")
        placements = []
        for row in info["placements"]:
            if row == {"kind": "replicate"}:
                placements.append(Replicate())
            elif set(row) == {"kind", "dim"} and row["kind"] == "shard" and type(row["dim"]) is int:
                placements.append(Shard(row["dim"]))
            else:
                raise ValueError("checkpoint tensor placement changed")
        if len(placements) != mesh.ndim:
            raise ValueError("checkpoint tensor mesh dimension changed")
        device = torch.device("cuda", torch.cuda.current_device()) if mesh.device_type == "cuda" else torch.device("cpu")
        return DTensor.from_local(local.to(device), device_mesh=mesh, placements=placements,
                                  shape=torch.Size(info["shape"]), stride=tuple(info["stride"]), run_check=False)
    if isinstance(value, dict):
        return {key: decode(item, meshes) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(decode(item, meshes) for item in value)
    if isinstance(value, torch.Tensor) or value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError("unsafe checkpoint tree")


def live_meshes(state):
    from distillschedule.open_workload.torch_backend import tensors
    from torch.distributed.tensor import DTensor
    meshes = []
    for tensor in tensors(state):
        if isinstance(tensor, DTensor) and all(tensor.device_mesh is not mesh for mesh in meshes):
            meshes.append(tensor.device_mesh)
    return meshes
