"""Safe local-shard checkpoint trees; no pickled DTensor/mesh/process groups.

Only replicated or dimension-sharded DTensors at a quiescent cycle boundary
are admitted. Topology/layout restoration uses the *existing live* mesh.
"""
from __future__ import annotations

_MARKER = "distillschedule.dtensor_local.v1"


class ParameterShadows(dict):
    """An explicitly bound parameter map, opted in before session construction.

    The algorithm still owns every update and decay. This envelope binds the
    checkpoint to the same root and complete tensor layout, unlike an ordinary
    value dictionary. Dense/default EMA serialization is deliberately unchanged.
    """
    SCHEMA = "distillschedule.parameter_shadows.v1"

    def __init__(self, owner, parameters):
        if not isinstance(owner, str) or not owner:
            raise ValueError('parameter shadows require an explicit model root')
        super().__init__(parameters)
        self.owner = owner

    def runtime_tensors(self):
        yield from self.values()

    def state_dict(self):
        return dict(schema_version=self.SCHEMA, owner=self.owner, parameters=dict(sorted(self.items())))

    def validate_state_dict(self, value):
        from .checkpoint import _validate_layout
        if (not isinstance(value, dict) or set(value) != {'schema_version', 'owner', 'parameters'}
                or value['schema_version'] != self.SCHEMA or value['owner'] != self.owner):
            raise ValueError('checkpoint parameter shadow owner/schema changed')
        _validate_layout(dict(self), value['parameters'], 'parameter shadows ' + self.owner)
        _validate_shadow_values(value['parameters'])

    def validate_checkpoint_boundary(self):
        self.validate_state_dict(self.state_dict())

    def load_state_dict(self, value):
        import torch
        from distillschedule.open_workload.torch_backend import physical_tensor
        self.validate_state_dict(value)
        with torch.no_grad():
            for name, live in self.items():
                physical_tensor(live).copy_(physical_tensor(value['parameters'][name]))


def _validate_shadow_values(values):
    """Local checks only, including on a rank with an empty physical shard."""
    import torch
    from torch.distributed.tensor import DTensor
    from distillschedule.open_workload.torch_backend import physical_tensor
    for value in values.values():
        if (type(value) not in (torch.Tensor, DTensor) or value.is_meta
                or value.layout != torch.strided or value.requires_grad or value.grad_fn is not None
                or not bool(torch.isfinite(physical_tensor(value)).all())):
            raise ValueError('parameter shadows require finite detached strided tensors')


def validate_parameter_shadows(state, bindings, roots, *, feeds=()):
    """Exact named-parameter ownership, without invoking tensor collectives."""
    from collections.abc import Mapping
    if (not isinstance(bindings, Mapping) or not all(isinstance(k, str) and k
            and isinstance(v, str) and v for k, v in bindings.items())
            or not set(bindings) <= set(state) or not set(bindings.values()) <= set(roots)):
        raise ValueError('parameter shadow bindings require state keys and registered model roots')
    if not bindings:
        return
    import torch
    from .checkpoint import _validate_layout
    from distillschedule.open_workload.torch_backend import tensors, storage_key, unwrap_module
    # No shadow may alias a parameter, optimizer slot, feed, another logical
    # state, or another shadow. Empty local shards do not own any storage.
    occupied = {storage_key(t) for t in tensors(({k:v for k,v in state.items() if k not in bindings}, feeds))
                if storage_key(t)[2]}
    for key, root in bindings.items():
        value = state[key]
        if type(value) is not ParameterShadows:
            raise ValueError('parameter shadows require an explicit ParameterShadows owner before session construction')
        if value.owner != root:
            raise ValueError('parameter shadow owner differs from its binding')
        model = unwrap_module(state[root])
        if not isinstance(model, torch.nn.Module):
            raise ValueError('parameter shadow owner must be a model root')
        parameters = dict(model.named_parameters())
        named = list(model.named_parameters(remove_duplicate=False))
        stores = [storage_key(p) for _, p in named if storage_key(p)[2]]
        if len(named) != len(parameters) or len(stores) != len(set(stores)):
            raise ValueError('parameter shadow owner cannot contain parameter aliases')
        if not parameters or value.keys() != parameters.keys():
            raise ValueError('parameter shadows must cover every named parameter, including frozen parameters')
        _validate_shadow_values(value)
        for name, parameter in parameters.items():
            shadow = value[name]
            _validate_layout(parameter, shadow, f'parameter shadow {key}.{name}')
            if shadow.device != parameter.device or shadow.stride() != parameter.stride():
                raise ValueError('parameter shadow device/stride differs from owner')
            storage = storage_key(shadow)
            if storage[2]:
                if storage in occupied:
                    raise ValueError('parameter shadow storage aliases another owned tensor')
                occupied.add(storage)


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
