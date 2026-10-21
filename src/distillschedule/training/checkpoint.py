"""Complete-cycle checkpoints for framework-owned training state.

Native distributed adapters keep their own shard writers. This writer owns a
single-process session and never mislabels a final weight export as resumable.
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

SCHEMA = "distillschedule.training_checkpoint.v1"


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
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


def pack_state(state):
    import torch
    packed = {}
    for key, value in state.items():
        if isinstance(value, torch.nn.Module):
            packed[key] = {"kind": "module", "class": _class(value),
                           "value": cpu_tree(value.state_dict()),
                           "training": {name: child.training for name, child in value.named_modules()},
                           "requires_grad": {name: p.requires_grad for name, p in value.named_parameters()}}
        elif isinstance(value, torch.optim.Optimizer):
            packed[key] = {"kind": "optimizer", "class": _class(value), "value": cpu_tree(value.state_dict())}
        elif callable(getattr(value, "state_dict", None)) and callable(getattr(value, "load_state_dict", None)):
            packed[key] = {"kind": "stateful", "class": _class(value), "value": cpu_tree(value.state_dict())}
        else:
            packed[key] = {"kind": "value", "value": cpu_tree(value)}
    return packed


def _validate_layout(current, restored, name):
    """Validate module extra-state with the same explicit safe-tree contract."""
    import torch
    if isinstance(current, torch.Tensor):
        if not isinstance(restored, torch.Tensor) or current.shape != restored.shape or current.dtype != restored.dtype:
            raise ValueError(f"checkpoint module shape/dtype changed: {name}")
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


def unpack_state(packed, state):
    import torch
    if set(packed) != set(state):
        raise ValueError("checkpoint state names changed")
    # Validate before mutating any live model or optimizer.
    for key, row in packed.items():
        value = state[key]
        if row["kind"] != "value" and row["class"] != _class(value):
            raise ValueError(f"checkpoint state class changed: {key}")
        if row["kind"] == "module":
            current = value.state_dict()
            _validate_layout(current, row["value"], key)
            if row["requires_grad"] != {name: p.requires_grad for name, p in value.named_parameters()}:
                raise ValueError(f"checkpoint trainables changed: {key}")
    for key, row in packed.items():
        value = state[key]
        if row["kind"] == "module":
            value.load_state_dict(row["value"], strict=True)
            for name, child in value.named_modules():
                child.training = row["training"][name]
        elif row["kind"] in ("optimizer", "stateful"):
            value.load_state_dict(row["value"])
        elif isinstance(value, torch.Tensor):
            restored = row["value"]
            if not isinstance(restored, torch.Tensor) or restored.shape != value.shape or restored.dtype != value.dtype:
                raise ValueError(f"checkpoint tensor shape/dtype changed: {key}")
            with torch.no_grad():
                value.copy_(restored.to(value.device))
        else:
            state[key] = deepcopy(row["value"])


def rng_state():
    import torch
    result = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        result["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        name, values, position, gauss, cached = np.random.get_state()
        result["numpy"] = (name, values.tolist(), position, gauss, cached)
    except ImportError:
        pass
    return result


def restore_rng(value):
    import torch
    random.setstate(value["python"])
    torch.set_rng_state(value["torch"])
    if "cuda" in value:
        if not torch.cuda.is_initialized() or len(value["cuda"]) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA topology changed")
        torch.cuda.set_rng_state_all(value["cuda"])
    if "numpy" in value:
        import numpy as np
        name, values, position, gauss, cached = value["numpy"]
        np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, gauss, cached))


def save_checkpoint(path, state, *, identity, cycle, cursor, versions, policy=None):
    import torch
    if type(cycle) is not int or cycle < 0 or type(cursor) is not int or cursor < 0:
        raise ValueError("checkpoint must describe a complete cycle and data cursor")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("distributed checkpoints must use the native adapter shard writer")
    if any(p.grad is not None for value in state.values() if isinstance(value, torch.nn.Module)
           for p in value.parameters()):
        raise ValueError("checkpoint requires a zero-gradient complete-cycle boundary")
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    staging = path.with_name("." + path.name + ".tmp-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        payload = {"state": pack_state(state), "rng": rng_state(), "cycle": cycle,
                   "cursor": cursor, "versions": dict(versions), "policy": policy}
        torch.save(payload, staging / "state.pt")
        manifest = {"schema_version": SCHEMA, "kind": "RESUMABLE_COMPLETE_CYCLE",
                    "identity": identity, "cycle": cycle, "cursor": cursor,
                    "topology": {"world_size": 1}, "torch": torch.__version__,
                    "files": {"state.pt": _sha(staging / "state.pt")}}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (staging / "_SUCCESS").write_text(_sha(staging / "manifest.json") + "\n")
        os.rename(staging, path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return path


def read_checkpoint(path, *, identity):
    import torch
    path = Path(path).resolve()
    if not (path / "_SUCCESS").is_file() or not (path / "manifest.json").is_file():
        raise ValueError("checkpoint is not complete")
    if (path / "_SUCCESS").read_text().strip() != _sha(path / "manifest.json"):
        raise ValueError("checkpoint manifest hash changed")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA or manifest.get("identity") != identity:
        raise ValueError("checkpoint identity changed")
    if manifest["topology"] != {"world_size": int(os.environ.get("WORLD_SIZE", "1"))}:
        raise ValueError("checkpoint topology changed")
    if manifest.get("torch") != torch.__version__:
        raise ValueError("checkpoint runtime version changed")
    if manifest.get("files") != {"state.pt": _sha(path / "state.pt")}:
        raise ValueError("checkpoint shard missing or hash changed")
    payload = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    if payload["cycle"] != manifest["cycle"] or payload["cursor"] != manifest["cursor"]:
        raise ValueError("checkpoint progress disagrees with manifest")
    return payload


def restore_checkpoint(path, state, *, identity):
    payload = read_checkpoint(path, identity=identity)
    unpack_state(payload["state"], state)
    restore_rng(payload["rng"])
    return payload
