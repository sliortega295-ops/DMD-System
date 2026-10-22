"""Complete-cycle checkpoints for framework-owned training state.

Native distributed adapters keep their own shard writers. Program sessions use
this same-topology atomic writer, with one safe local shard per actual rank.
"""
from __future__ import annotations
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil
import uuid
from distillschedule.workloads.best_known.registry import file_sha256 as _sha
from . import distributed
from .tensor_state import pack, decode, layout, live_meshes

SCHEMA = "distillschedule.training_checkpoint.v2"


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        from distillschedule.open_workload.torch_backend import full_tensor
        value = full_tensor(value)
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError(f"checkpoint state needs an explicit serializer: {type(value).__name__}")


def _class(value):
    return type(value).__module__ + "." + type(value).__qualname__


def _parameter_names(state):
    import torch
    from distillschedule.open_workload.torch_backend import unwrap_module
    return {id(p): f"{key}.{name}" for key, value in state.items()
            if isinstance(value, torch.nn.Module)
            for name, p in unwrap_module(value).named_parameters()}


def pack_state(state):
    import torch
    from distillschedule.open_workload.torch_backend import unwrap_module
    parameter_names = _parameter_names(state)
    packed = {}
    for key, value in state.items():
        if isinstance(value, torch.nn.Module):
            value = unwrap_module(value)
            packed[key] = {"kind": "module", "class": _class(value),
                           "value": pack(value.state_dict()),
                           "training": {name: child.training for name, child in value.named_modules()},
                           "requires_grad": {name: p.requires_grad for name, p in value.named_parameters()}}
        elif isinstance(value, torch.optim.Optimizer):
            packed[key] = {"kind": "optimizer", "class": _class(value), "value": pack(value.state_dict()),
                           "parameters": [[parameter_names[id(p)] for p in group["params"]]
                                          for group in value.param_groups]}
        elif callable(getattr(value, "state_dict", None)) and callable(getattr(value, "load_state_dict", None)):
            packed[key] = {"kind": "stateful", "class": _class(value), "value": pack(value.state_dict())}
        else:
            packed[key] = {"kind": "value", "value": pack(value)}
    return packed


def _validate_layout(current, restored, name):
    """Validate module extra-state with the same explicit safe-tree contract."""
    import torch
    from torch.distributed.tensor import DTensor
    if isinstance(current, torch.Tensor):
        if not isinstance(restored, torch.Tensor) or current.shape != restored.shape or current.dtype != restored.dtype:
            raise ValueError(f"checkpoint module shape/dtype changed: {name}")
        if isinstance(current, DTensor) != isinstance(restored, DTensor) or (
                isinstance(current, DTensor) and layout(current) != layout(restored)):
            raise ValueError(f"checkpoint tensor layout changed: {name}")
    elif isinstance(current, dict):
        if not isinstance(restored, dict) or current.keys() != restored.keys():
            raise ValueError(f"checkpoint module keys changed: {name}")
        for key, value in current.items():
            _validate_layout(value, restored[key], f"{name}.{key}")
    elif isinstance(current, (tuple, list)):
        if type(restored) is not type(current) or len(current) != len(restored):
            raise ValueError(f"checkpoint module sequence layout changed: {name}")
        for index, (a, b) in enumerate(zip(current, restored)):
            _validate_layout(a, b, f"{name}.{index}")
    elif type(current) not in (type(None), str, bool, int, float) or type(restored) is not type(current):
        raise ValueError(f"checkpoint module extra-state type changed: {name}")


def _validate_optimizer_layout(optimizer, restored, name):
    """Check known pointwise slots, not native factored/flattened state.

    The native loader owns other optimizer schemas. Exact types avoid imposing
    a built-in schema on custom subclasses. Validate the decoded source slots:
    load_state_dict may otherwise silently cast a malformed moment's dtype.
    """
    import torch
    fields = {
        torch.optim.Adam: ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"),
        torch.optim.AdamW: ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"),
        torch.optim.SparseAdam: ("exp_avg", "exp_avg_sq"),
        torch.optim.SGD: ("momentum_buffer",),
        torch.optim.Adadelta: ("square_avg", "acc_delta"),
        torch.optim.Adagrad: ("sum",),
        torch.optim.Adamax: ("exp_avg", "exp_inf"),
        torch.optim.ASGD: ("ax",),
        torch.optim.NAdam: ("exp_avg", "exp_avg_sq"),
        torch.optim.RAdam: ("exp_avg", "exp_avg_sq"),
        torch.optim.RMSprop: ("square_avg", "momentum_buffer", "grad_avg"),
        torch.optim.Rprop: ("prev", "step_size"),
        torch.optim.Adafactor: ("variance",),
    }.get(type(optimizer), ())
    for group, saved_group in zip(optimizer.param_groups, restored["param_groups"]):
        for parameter, parameter_id in zip(group["params"], saved_group["params"]):
            values = restored["state"].get(parameter_id, {})
            for field in fields:
                if field in values:
                    _validate_layout(parameter, values[field], f"{name}.{field}")


def _stage_state(packed, state, *, progress=None):
    import torch
    from copy import copy
    from distillschedule.open_workload.torch_backend import unwrap_module
    if set(packed) != set(state):
        raise ValueError("checkpoint state names changed")
    meshes = live_meshes(state)
    restored = {key: decode(row["value"], meshes) for key, row in packed.items()}
    parameter_names = _parameter_names(state)
    optimizers = {}
    # Validate/load staged optimizer objects before mutating any live tensor.
    for key, row in packed.items():
        value = unwrap_module(state[key]) if isinstance(state[key], torch.nn.Module) else state[key]
        expected_kind = ("module" if isinstance(value, torch.nn.Module) else "optimizer"
                         if isinstance(value, torch.optim.Optimizer) else "stateful"
                         if callable(getattr(value, "state_dict", None)) and callable(getattr(value, "load_state_dict", None))
                         else "value")
        if row.get("kind") != expected_kind:
            raise ValueError(f"checkpoint state kind changed: {key}")
        if row["kind"] != "value" and row["class"] != _class(value):
            raise ValueError(f"checkpoint state class changed: {key}")
        if row["kind"] == "module":
            current = value.state_dict()
            _validate_layout(current, restored[key], key)
            if row["requires_grad"] != {name: p.requires_grad for name, p in value.named_parameters()}:
                raise ValueError(f"checkpoint trainables changed: {key}")
            if (set(row["training"]) != {name for name, _ in value.named_modules()}
                    or any(type(item) is not bool for item in row["training"].values())):
                raise ValueError(f"checkpoint module training modes changed: {key}")
        elif row["kind"] == "optimizer":
            expected = [[parameter_names[id(p)] for p in group["params"]] for group in value.param_groups]
            if row.get("parameters") != expected:
                raise ValueError(f"checkpoint optimizer parameter ownership changed: {key}")
            staged = copy(value)
            staged.param_groups = [dict(group, params=list(group["params"])) for group in value.param_groups]
            staged.state = {}
            staged.load_state_dict(restored[key])
            _validate_optimizer_layout(staged, restored[key], key)
            optimizers[key] = staged
        elif row["kind"] == "value" and isinstance(value, torch.Tensor):
            _validate_layout(value, restored[key], key)
        elif row["kind"] == "stateful" and callable(getattr(value, "validate_state_dict", None)):
            value.validate_state_dict(restored[key])
        if row["kind"] == "stateful" and callable(getattr(value, "validate_checkpoint_progress", None)):
            value.validate_checkpoint_progress(restored[key], progress)
    return restored, optimizers


def unpack_state(packed, state, *, progress=None):
    import torch
    from distillschedule.open_workload.torch_backend import unwrap_module
    # No rank begins mutation while any rank rejects a shard/layout/optimizer.
    restored, optimizers = distributed.checked(lambda: _stage_state(packed, state, progress=progress), "checkpoint state admission")
    for key, row in packed.items():
        value = unwrap_module(state[key]) if isinstance(state[key], torch.nn.Module) else state[key]
        if row["kind"] == "module":
            value.load_state_dict(restored[key], strict=True)
            for name, child in value.named_modules():
                child.training = row["training"][name]
        elif row["kind"] == "optimizer":
            value.state = optimizers[key].state
            value.param_groups = optimizers[key].param_groups
        elif row["kind"] == "stateful":
            value.load_state_dict(restored[key])
        elif isinstance(value, torch.Tensor):
            with torch.no_grad():
                value.copy_(restored[key].to(value.device))
        else:
            state[key] = deepcopy(restored[key])


def rng_state():
    import torch
    result = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        # The execution ABI owns one device per rank. Querying all visible
        # devices can initialize unrelated CUDA contexts and consume their HBM.
        device = torch.cuda.current_device()
        result["cuda"] = {"device": device, "state": torch.cuda.get_rng_state(device)}
    try:
        import numpy as np
        name, values, position, gauss, cached = np.random.get_state()
        result["numpy"] = (name, values.tolist(), position, gauss, cached)
    except ImportError:
        pass
    return result


def validate_rng(value):
    import torch
    random.Random().setstate(value["python"])
    torch.Generator(device="cpu").set_state(value["torch"])
    if "cuda" in value:
        if (not torch.cuda.is_initialized() or value["cuda"]["device"] != torch.cuda.current_device()
                or not isinstance(value["cuda"]["state"], torch.Tensor)
                or value["cuda"]["state"].dtype != torch.uint8
                or value["cuda"]["state"].shape != torch.cuda.get_rng_state(torch.cuda.current_device()).shape):
            raise ValueError("checkpoint CUDA topology changed")
    if "numpy" in value:
        import numpy as np
        name, values, position, gauss, cached = value["numpy"]
        np.random.RandomState().set_state((name, np.asarray(values, dtype=np.uint32), position, gauss, cached))


def restore_rng(value):
    import torch
    validate_rng(value)
    random.setstate(value["python"])
    torch.set_rng_state(value["torch"])
    if "cuda" in value:
        torch.cuda.set_rng_state(value["cuda"]["state"], device=value["cuda"]["device"])
    if "numpy" in value:
        import numpy as np
        name, values, position, gauss, cached = value["numpy"]
        np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, gauss, cached))


def save_checkpoint(path, state, *, identity, cycle, cursor, versions, policy=None):
    import torch
    if type(cycle) is not int or cycle < 0 or type(cursor) is not int or cursor < 0:
        raise ValueError("checkpoint must describe a complete cycle and data cursor")
    def boundary():
        if int(os.environ.get("WORLD_SIZE", "1")) != distributed.rank_world()[1]:
            raise ValueError("checkpoint requires the actual initialized rank group")
        if any(p.grad is not None for value in state.values() if isinstance(value, torch.nn.Module)
               for p in value.parameters()):
            raise ValueError("checkpoint requires a zero-gradient complete-cycle boundary")
        for value in state.values():
            validate = getattr(value, "validate_checkpoint_boundary", None)
            if callable(validate):
                validate()
            progress_hook = getattr(value, "validate_checkpoint_progress", None)
            if callable(progress_hook):
                progress_hook(value.state_dict(), {"cycle":cycle, "cursor":cursor, "versions":dict(versions)})
    distributed.checked(boundary, "checkpoint boundary")
    rank, world = distributed.rank_world()
    allocated = distributed.topology()
    common = {"identity": identity, "cycle": cycle, "cursor": cursor,
              "versions": dict(versions), "policy": policy, "topology": allocated}
    if any(row != common for row in distributed.gather(common)):
        raise ValueError("ranks disagree on checkpoint identity or completed progress")
    path = Path(path).resolve()
    token = distributed.gather(uuid.uuid4().hex if rank == 0 else None)[0]
    staging = path.with_name("." + path.name + ".tmp-" + token)
    def create():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        staging.mkdir()
    distributed.primary(create, "checkpoint staging")
    try:
        payload = distributed.checked(lambda: {"state": pack_state(state), "rng": rng_state(), "cycle": cycle,
                   "cursor": cursor, "versions": dict(versions), "policy": policy}, "checkpoint packing")
        filename = "state.pt" if world == 1 else f"rank-{rank:05d}.pt"
        def write_shard():
            if not staging.is_dir():
                raise ValueError("checkpoint output must be shared by all ranks")
            with (staging / filename).open("xb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            return {filename: _sha(staging / filename)}
        shard = distributed.checked(write_shard, "checkpoint shard writing")
        files = {key: value for row in distributed.gather(shard) for key, value in row.items()}
        def publish():
            if len(files) != world or any(not (staging / name).is_file() or _sha(staging / name) != sha
                                          for name, sha in files.items()):
                raise ValueError("checkpoint shards incomplete on shared storage")
            manifest = {"schema_version": SCHEMA, "kind": "RESUMABLE_COMPLETE_CYCLE",
                        "identity": identity, "cycle": cycle, "cursor": cursor,
                        "versions": dict(versions), "policy": policy,
                        "topology": allocated, "torch": torch.__version__, "files": files}
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            (staging / "_SUCCESS").write_text(_sha(staging / "manifest.json") + "\n")
            os.rename(staging, path)
        distributed.primary(publish, "checkpoint publication")
    except BaseException:
        if world == 1:
            shutil.rmtree(staging, ignore_errors=True)
        # Distributed incomplete staging remains explicitly unpublished for audit.
        raise
    return path


def read_manifest(path, *, identity):
    """Validate restart metadata before constructing a physical model layout.

    Tensor hashes are checked again before any state restore. Metadata does not
    load a model-sized snapshot simply to read its saved policy and cursor.
    All ranks participate when a distributed group is initialized.
    """
    import torch
    allocated = distributed.topology()
    path = Path(path).resolve()
    if not (path / "_SUCCESS").is_file() or not (path / "manifest.json").is_file():
        raise ValueError("checkpoint is not complete")
    if (path / "_SUCCESS").read_text().strip() != _sha(path / "manifest.json"):
        raise ValueError("checkpoint manifest hash changed")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA or manifest.get("identity") != identity:
        raise ValueError("checkpoint identity changed")
    if manifest["topology"] != allocated:
        raise ValueError("checkpoint topology changed")
    if manifest.get("torch") != torch.__version__:
        raise ValueError("checkpoint runtime version changed")
    _, world = distributed.rank_world()
    expected = {"state.pt"} if world == 1 else {f"rank-{r:05d}.pt" for r in range(world)}
    if set(manifest.get("files", {})) != expected or any(not (path / name).is_file() for name in expected):
        raise ValueError("checkpoint shard missing or hash changed")
    return manifest


def read_checkpoint(path, *, identity):
    import torch
    path = Path(path).resolve()
    manifest = read_manifest(path, identity=identity)
    if any(_sha(path / name) != sha for name, sha in manifest["files"].items()):
        raise ValueError("checkpoint shard missing or hash changed")
    rank, world = distributed.rank_world()
    payload = torch.load(path / ("state.pt" if world == 1 else f"rank-{rank:05d}.pt"),
                         map_location="cpu", weights_only=True)
    if any(payload[key] != manifest[key] for key in ("cycle", "cursor", "versions", "policy")):
        raise ValueError("checkpoint progress disagrees with manifest")
    return payload


def restore_checkpoint(path, state, *, identity):
    payload = distributed.checked(lambda: read_checkpoint(path, identity=identity), "checkpoint files admission")
    distributed.checked(lambda: validate_rng(payload["rng"]), "checkpoint RNG admission")
    unpack_state(payload["state"], state, progress={key:payload[key] for key in ("cycle", "cursor", "versions")})
    restore_rng(payload["rng"])
    return payload
