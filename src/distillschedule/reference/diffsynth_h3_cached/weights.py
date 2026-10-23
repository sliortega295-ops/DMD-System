"""Selected original full-weight helpers; no model payload is included."""
import inspect
import json
from pathlib import Path
from contextlib import ExitStack
import torch
from safetensors import safe_open
from .models.minimax_h3_dit import MiniMaxH3DiT
from .cache import gathered_parameter


def checkpoint_files(path):
    """Accept a safetensors file, shard index, transformer directory, or model root."""
    path = Path(path).expanduser().resolve()
    if path.is_dir() and (path / "transformer").is_dir():
        path = path / "transformer"
    if path.is_dir():
        index = path / "model.safetensors.index.json"
        if index.is_file():
            path = index
        else:
            files = sorted(path.glob("model*.safetensors"))
            if not files:
                raise FileNotFoundError(f"No model*.safetensors in {path}")
            return files, path
    if path.name.endswith(".safetensors.index.json"):
        mapping = json.loads(path.read_text())["weight_map"]
        files = [path.parent / name for name in sorted(set(mapping.values()))]
    elif path.suffix == ".safetensors":
        files = [path]
    else:
        raise ValueError("Expected a full DiT safetensors file, shard index, or checkpoint directory")
    if not files or any(not file.is_file() for file in files):
        raise FileNotFoundError(f"Missing checkpoint shards: {path}")
    return files, path.parent


def model_config(directory):
    path = directory / "config.json"
    source = json.loads(path.read_text()) if path.is_file() else {}
    # The native H3 defaults match FL2VA; preserve architecture for future saves.
    config = {}
    for name, parameter in inspect.signature(MiniMaxH3DiT.__init__).parameters.items():
        if name not in ("self", "kwargs"):
            config[name] = source.get(name, parameter.default)
    return config


def normalize_key(name):
    for prefix in ("module.pipe.dit.", "pipe.dit.", "module.dit.", "dit."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def load_full_weights(model, files):
    """Validate all keys/shapes before loading; gather one ZeRO-3 parameter at a time.

    Safetensors stay memory-mapped on CPU. Only rank 0 reads/copies sharded weights;
    GatheredParameters broadcasts them when it repartitions the parameter.
    """
    expected = model.state_dict(keep_vars=True)
    with ExitStack() as stack:
        tensors = {}
        for path in files:
            handle = stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            for key in handle.keys():
                name = normalize_key(key)
                if name in tensors:
                    raise ValueError(f"Duplicate checkpoint key: {name}")
                tensors[name] = (handle, key)
        missing, unexpected = set(expected) - set(tensors), set(tensors) - set(expected)
        if missing or unexpected:
            raise ValueError(f"Full DiT checkpoint required (LoRA is not supported). "
                             f"Missing keys: {sorted(missing)[:10]}; unexpected keys: {sorted(unexpected)[:10]}")
        for name, target in expected.items():
            handle, key = tensors[name]
            shape = getattr(target, "ds_shape", target.shape)
            if tuple(handle.get_slice(key).get_shape()) != tuple(shape):
                raise ValueError(f"Checkpoint shape mismatch for {name}: expected {tuple(shape)}")
        for name, target in expected.items():
            handle, key = tensors[name]
            with gathered_parameter(target), torch.no_grad():
                if not hasattr(target, "ds_id") or torch.distributed.get_rank() == 0:
                    target.copy_(handle.get_tensor(key).to(device=target.device, dtype=target.dtype))
