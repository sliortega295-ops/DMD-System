"""Small, topology-independent numeric sketches for trajectory state.

The strict one-cycle parity path already owns full-tensor comparisons.  A
50-cycle trajectory run instead needs a bounded artifact that can be written at
several anchors without gathering every LoRA and Adam tensor in full.  This
module samples deterministic *global* flat coordinates.  Resident replicas and
FSDP2 DTensor shards therefore produce the same coordinate/value schema.

This is diagnostic evidence, not a substitute for the strict parity gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from typing import Any


SCHEMA_VERSION = "distillschedule.trajectory_numeric_state_sketch.v1"


def canonical_state_name(name: str) -> str:
    """Remove wrapper-only prefixes while preserving model parameter identity."""

    value = str(name)
    prefixes = ("module.", "_orig_mod.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if value.startswith(prefix):
                value = value[len(prefix) :]
                changed = True
    if not value:
        raise ValueError("state tensor name cannot be empty")
    return value


def deterministic_flat_indices(name: str, numel: int, *, count: int = 16) -> tuple[int, ...]:
    """Choose stable global coordinates without consuming workload RNG."""

    if numel <= 0:
        return ()
    if count <= 0:
        raise ValueError("sample count must be positive")
    if numel <= count:
        return tuple(range(numel))
    digest = hashlib.sha256(canonical_state_name(name).encode()).digest()
    start = int.from_bytes(digest[:8], "big") % numel
    stride = int.from_bytes(digest[8:16], "big") % numel or 1
    while math.gcd(stride, numel) != 1:
        stride = (stride + 1) % numel or 1
    chosen = {0, numel - 1}
    cursor = start
    while len(chosen) < count:
        chosen.add(cursor)
        cursor = (cursor + stride) % numel
    return tuple(sorted(chosen))


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = [0] * len(shape)
    for index in range(len(shape) - 1, -1, -1):
        result[index] = stride
        stride *= shape[index]
    return tuple(result)


def _coordinate(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = _contiguous_strides(shape)
    value = int(flat_index)
    result: list[int] = []
    for dimension, stride in zip(shape, strides, strict=True):
        result.append(value // stride)
        value %= stride
    return tuple(result)


def _inventory_indices(length: int, limit: int) -> tuple[int, ...]:
    if limit <= 0:
        raise ValueError("max_tensors must be positive")
    if length <= limit:
        return tuple(range(length))
    if limit == 1:
        return (0,)
    return tuple(
        sorted({round(index * (length - 1) / (limit - 1)) for index in range(limit)})
    )


def _layout(
    *,
    torch: Any,
    logical_tensor: Any,
    local_tensor: Any,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], str]:
    global_shape = tuple(int(value) for value in logical_tensor.shape)
    if hasattr(logical_tensor, "placements") and hasattr(logical_tensor, "device_mesh"):
        try:
            from torch.distributed.tensor._utils import (  # type: ignore[attr-defined]
                compute_local_shape_and_global_offset,
            )
        except ImportError as error:  # pragma: no cover - pinned runtime owns this path
            raise RuntimeError(
                "pinned Torch lacks DTensor local-shape/global-offset utility"
            ) from error
        local_shape, offset = compute_local_shape_and_global_offset(
            global_shape,
            logical_tensor.device_mesh,
            logical_tensor.placements,
        )
        local_shape = tuple(int(value) for value in local_shape)
        offset = tuple(int(value) for value in offset)
        if tuple(int(value) for value in local_tensor.shape) != local_shape:
            raise AssertionError(
                "DTensor local layout changed: "
                f"observed={tuple(local_tensor.shape)} expected={local_shape}"
            )
        return global_shape, local_shape, offset, "DTENSOR_GLOBAL_COORDINATES"
    local_shape = tuple(int(value) for value in local_tensor.shape)
    if local_shape != global_shape:
        raise AssertionError("plain state tensor is not a full replica")
    return global_shape, local_shape, (0,) * len(global_shape), "FULL_REPLICA"


def _local_samples(
    *,
    torch: Any,
    name: str,
    value: Any,
    layout_source: Any | None,
    count: int,
    rank: int,
    primary_rank: int,
) -> dict[str, Any]:
    logical = value if layout_source is None else layout_source
    local = value.detach().to_local() if hasattr(value, "to_local") else value.detach()
    global_shape, local_shape, offset, representation = _layout(
        torch=torch,
        logical_tensor=logical,
        local_tensor=local,
    )
    numel = math.prod(global_shape)
    indices = deterministic_flat_indices(name, numel, count=count)
    owned_global: list[int] = []
    owned_local: list[int] = []
    local_strides = _contiguous_strides(local_shape)
    for flat_index in indices:
        coordinate = _coordinate(flat_index, global_shape)
        if all(
            start <= current < start + extent
            for current, start, extent in zip(
                coordinate, offset, local_shape, strict=True
            )
        ):
            local_coordinate = tuple(
                current - start
                for current, start in zip(coordinate, offset, strict=True)
            )
            local_index = sum(
                current * stride
                for current, stride in zip(
                    local_coordinate, local_strides, strict=True
                )
            )
            owned_global.append(flat_index)
            owned_local.append(local_index)
    if representation == "FULL_REPLICA" and rank != primary_rank:
        owned_global.clear()
        owned_local.clear()
    if owned_local:
        flat = local.contiguous().view(-1)
        selected = flat.index_select(
            0,
            torch.tensor(owned_local, dtype=torch.int64, device=flat.device),
        )
        numbers = selected.detach().float().cpu().tolist()
        if not all(math.isfinite(float(number)) for number in numbers):
            raise ValueError(f"numeric state sketch found a non-finite value in {name}")
    else:
        numbers = []
    return {
        "name": canonical_state_name(name),
        "shape": list(global_shape),
        "dtype": str(local.dtype),
        "numel": numel,
        "requested_indices": list(indices),
        "values": {str(index): float(number) for index, number in zip(owned_global, numbers, strict=True)},
        "representation": representation,
        "contributor_rank": rank,
    }


def collect_numeric_state_sketch(
    *,
    torch: Any,
    named_values: Iterable[tuple[str, Any, Any | None]],
    samples_per_tensor: int = 16,
    max_tensors: int = 96,
    primary_rank: int = 0,
) -> dict[str, Any]:
    """Collect a canonical numeric sketch on every rank.

    Each input tuple is ``(logical name, stored value, layout source)``.  The
    layout source is normally the corresponding FSDP2 DTensor parameter for a
    local Adam/EMA tensor; it is ``None`` when ``value`` itself describes the
    global layout.
    """

    dist = torch.distributed
    if not dist.is_initialized():
        raise RuntimeError("numeric trajectory state sketch requires distributed")
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    if not 0 <= primary_rank < world_size:
        raise ValueError("numeric sketch primary rank is outside the process group")
    local_items = sorted(
        (
            (canonical_state_name(name), value, layout_source)
            for name, value, layout_source in named_values
        ),
        key=lambda row: row[0],
    )
    inventory_names = [name for name, _value, _layout in local_items]
    if len(inventory_names) != len(set(inventory_names)):
        raise ValueError("numeric state sketch tensor names must be unique")
    local_rows = [
        _local_samples(
            torch=torch,
            name=name,
            value=value,
            layout_source=layout_source,
            count=samples_per_tensor,
            rank=rank,
            primary_rank=primary_rank,
        )
        for item_index in _inventory_indices(len(local_items), max_tensors)
        for name, value, layout_source in (local_items[item_index],)
    ]
    local_envelope = {"inventory_names": inventory_names, "rows": local_rows}
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_envelope)
    nonempty_inventories = {
        tuple(str(name) for name in contributor["inventory_names"])
        for contributor in gathered
        if contributor is not None and contributor["inventory_names"]
    }
    if len(nonempty_inventories) > 1:
        raise AssertionError("state sketch tensor inventory differs across owner ranks")
    global_inventory = (
        next(iter(nonempty_inventories)) if nonempty_inventories else ()
    )
    by_name: dict[str, dict[str, Any]] = {}
    for contributor in gathered:
        if contributor is None:
            raise AssertionError("numeric state sketch gather returned an empty rank")
        for row in contributor["rows"]:
            name = str(row["name"])
            current = by_name.setdefault(
                name,
                {
                    key: row[key]
                    for key in (
                        "name",
                        "shape",
                        "dtype",
                        "numel",
                        "requested_indices",
                        "representation",
                    )
                }
                | {"values": {}, "contributor_ranks": []},
            )
            for key in ("shape", "dtype", "numel", "requested_indices", "representation"):
                if current[key] != row[key]:
                    raise AssertionError(f"state sketch metadata differs for {name}: {key}")
            if row["values"]:
                current["contributor_ranks"].append(int(row["contributor_rank"]))
            for index, number in row["values"].items():
                if index in current["values"] and current["values"][index] != number:
                    raise AssertionError(
                        f"multiple shards disagree at {name}[{index}]"
                    )
                current["values"][index] = float(number)
    coordinates: list[dict[str, Any]] = []
    for name in sorted(by_name):
        row = by_name[name]
        missing = [
            index
            for index in row["requested_indices"]
            if str(index) not in row["values"]
        ]
        if missing:
            raise AssertionError(f"state sketch missed {name} coordinates {missing}")
        for index in row["requested_indices"]:
            coordinates.append(
                {
                    "tensor": name,
                    "flat_index": int(index),
                    "value_fp32": float(row["values"][str(index)]),
                }
            )
        row["values"] = [
            float(row["values"][str(index)]) for index in row["requested_indices"]
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "coordinate_schema": "canonical_tensor_name/global_flat_index/value_fp32",
        "samples_per_tensor": int(samples_per_tensor),
        "max_tensors": int(max_tensors),
        "primary_rank_for_replicas": int(primary_rank),
        "inventory_tensor_count": len(global_inventory),
        "inventory_sha256": hashlib.sha256(
            json.dumps(global_inventory, separators=(",", ":")).encode()
        ).hexdigest(),
        "tensor_count": len(by_name),
        "coordinate_count": len(coordinates),
        "tensors": [by_name[name] for name in sorted(by_name)],
        "coordinates": coordinates,
        "comparison": "align (tensor, flat_index), then compute cosine and normalized_l2",
    }
