"""Lazy Torch serialization; no pickle fallback, implicit reshape or dtype cast."""
from __future__ import annotations

from pathlib import Path
import pickle

from .manifest import require


def tensor_tree(value, *, cpu=False):
    import torch

    require(isinstance(value, dict) and bool(value), "features must be a nonempty tensor dict")

    def visit(item, depth):
        require(depth <= 64, "tensor tree is cyclic or too deeply nested")
        if isinstance(item, torch.Tensor):
            require(item.layout == torch.strided and not item.is_quantized and item.device.type != "meta",
                    "features require real dense, non-quantized tensors")
            tensor = item.detach().cpu() if cpu else item
            require(tensor.device.type == "cpu", "prepared tensors must load on CPU")
            return tensor, {"kind": "tensor", "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        if isinstance(item, dict):
            require(bool(item) and all(isinstance(k, str) and k.strip() for k in item),
                    "tensor dicts need nonempty string keys")
            children = {name: visit(child, depth + 1) for name, child in item.items()}
            return {name: child[0] for name, child in children.items()}, {
                "kind": "dict", "items": {name: child[1] for name, child in children.items()}}
        if isinstance(item, (list, tuple)):
            require(bool(item), "tensor sequences must be nonempty")
            children = [visit(child, depth + 1) for child in item]
            values = [child[0] for child in children]
            return (tuple(values) if isinstance(item, tuple) else values), {
                "kind": "tuple" if isinstance(item, tuple) else "list", "items": [child[1] for child in children]}
        raise ValueError("feature leaves must be tensors, not arbitrary Python objects")

    return visit(value, 0)


def load_tensors(path):
    import torch

    try:
        # Never retry with weights_only=False, including on older Torch versions.
        value = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError, ValueError, OSError, EOFError) as exc:
        raise ValueError(f"cannot safely load tensor file: {path}") from exc
    return tensor_tree(value)
