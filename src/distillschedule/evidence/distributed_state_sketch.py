"""Topology-independent, bounded training-state trajectory sketches.

The state compared by the 50-cycle trajectory experiment can be represented in
three different ways at runtime:

* Resident parameters are replicated plain tensors on all ranks.
* world16 FSDP2 parameters are one-dimensional sharded ``DTensor`` objects.
* node-HSDP parameters use a replicated dimension and a sharded dimension.

The helpers below select deterministic *global* flat coordinates. Every rank
emits the coordinates it owns. Replicated ranks intentionally emit duplicate
coordinates: the offline merge verifies that replicas agree instead of silently
choosing rank zero and potentially hiding replica drift.

This is diagnostic evidence. Sampling happens only at registered trajectory
anchors and never participates in training math.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .trajectory_coarse import (
    COARSE_50C_THRESHOLDS,
    TrajectoryStateComparison,
    TrajectoryThresholds,
)


SCHEMA_VERSION = "distillschedule.distributed_state_sketch.v2"
RANK_ANCHOR_SCHEMA_VERSION = "distillschedule.distributed_state_anchor.v1"
COMPARISON_SCHEMA_VERSION = "distillschedule.distributed_state_comparison.v1"

_STATE_KEY = re.compile(
    r"^(?P<component>.+)_(?P<kind>parameter|adam|ema|gradient)$"
)


class _LocalLayoutView:
    """Attach a parameter's global DTensor layout to a plain local state tensor."""

    def __init__(self, parameter: Any, local: Any) -> None:
        self.shape = parameter.shape
        self.placements = parameter.placements
        self.device_mesh = parameter.device_mesh
        self.dtype = local.dtype
        self._local = local

    def detach(self) -> _LocalLayoutView:
        return self

    def to_local(self) -> Any:
        return self._local.detach()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(value))).hexdigest()


def _indices(length: int, count: int) -> tuple[int, ...]:
    if length <= 0 or count <= 0:
        return ()
    count = min(length, count)
    if count == 1:
        return (0,)
    return tuple(
        sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})
    )


def _unravel(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coordinate: list[int] = []
    for dimension in reversed(shape):
        coordinate.append(index % dimension)
        index //= dimension
    return tuple(reversed(coordinate))


def _is_dtensor(value: Any) -> bool:
    return (
        callable(getattr(value, "to_local", None))
        and hasattr(value, "placements")
        and hasattr(value, "device_mesh")
    )


def _local_shape_offset(value: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    try:
        from torch.distributed.tensor._utils import (
            compute_local_shape_and_global_offset,
        )
    except ImportError as error:  # pragma: no cover - pinned Torch has this API
        raise RuntimeError(
            "the installed PyTorch cannot expose DTensor global shard offsets"
        ) from error

    local_shape, global_offset = compute_local_shape_and_global_offset(
        tuple(int(item) for item in value.shape),
        value.device_mesh,
        value.placements,
    )
    return (
        tuple(int(item) for item in local_shape),
        tuple(int(item) for item in global_offset),
    )


def _validate_placements(value: Any) -> None:
    coordinate = value.device_mesh.get_coordinate()
    if coordinate is None:
        raise RuntimeError("current rank is outside the DTensor DeviceMesh")
    placements = tuple(value.placements)
    if len(coordinate) != len(placements):
        raise RuntimeError("DTensor placement count differs from DeviceMesh dimensions")
    for placement in placements:
        if placement.is_partial():
            raise RuntimeError("trajectory state sketches do not accept Partial DTensors")
        if not (placement.is_replicate() or placement.is_shard()):
            raise RuntimeError(
                f"unsupported trajectory DTensor placement {type(placement).__name__}"
            )


def _distributed_rank_and_world() -> tuple[int, int] | None:
    """Return the live process-group identity without initializing one."""

    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - torch is required by the runners
        return None
    if not dist.is_available() or not dist.is_initialized():
        return None
    return int(dist.get_rank()), int(dist.get_world_size())


def _expected_replica_count(value: Any) -> int:
    """Return how many ranks must contribute each owned global coordinate."""

    if not _is_dtensor(value):
        identity = _distributed_rank_and_world()
        return identity[1] if identity is not None else 1
    _validate_placements(value)
    mesh_shape = tuple(int(item) for item in value.device_mesh.shape)
    placements = tuple(value.placements)
    if len(mesh_shape) != len(placements):
        raise RuntimeError("DTensor mesh shape differs from placement dimensions")
    replicas = 1
    for dimension, placement in zip(mesh_shape, placements, strict=True):
        if placement.is_replicate():
            replicas *= dimension
    return replicas


def _global_contract(name: str, value: Any) -> dict[str, Any]:
    detached = value.detach()
    if _is_dtensor(detached):
        _validate_placements(detached)
        local = detached.to_local().detach()
        global_shape = tuple(int(item) for item in detached.shape)
    else:
        local = detached
        global_shape = tuple(int(item) for item in detached.shape)
    return {
        "name": str(name),
        "shape": list(global_shape),
        "numel": int(math.prod(global_shape)) if global_shape else 1,
        "dtype": str(local.dtype),
    }


def sketch_named_tensors(
    named_tensors: Iterable[tuple[str, Any]],
    *,
    global_rank: int,
    samples_per_tensor: int = 8,
    max_tensors: int = 96,
) -> dict[str, Any]:
    """Return globally addressed sampled values owned by ``global_rank``.

    Selection occurs from the complete sorted tensor inventory before shard
    ownership is considered, so Resident, FSDP2, and HSDP choose the same names
    and global flat coordinates. The full inventory digest also catches a
    missing gradient/optimizer tensor even when that tensor was not sampled.
    """

    import torch

    if isinstance(global_rank, bool) or int(global_rank) < 0:
        raise ValueError("global_rank must be a non-negative integer")
    if isinstance(samples_per_tensor, bool) or int(samples_per_tensor) <= 0:
        raise ValueError("samples_per_tensor must be positive")
    if isinstance(max_tensors, bool) or int(max_tensors) <= 0:
        raise ValueError("max_tensors must be positive")
    distributed_identity = _distributed_rank_and_world()
    if distributed_identity is not None and int(global_rank) != distributed_identity[0]:
        raise ValueError(
            "state sketch global_rank does not match the initialized process group: "
            f"{global_rank} != {distributed_identity[0]}"
        )

    items = sorted(
        ((str(name), value) for name, value in named_tensors if value is not None),
        key=lambda item: item[0],
    )
    if not items:
        raise ValueError("state sketch tensor inventory cannot be empty")
    names = [name for name, _value in items]
    if len(names) != len(set(names)):
        raise ValueError("state sketch tensor names must be unique")
    inventory = [_global_contract(name, value) for name, value in items]
    selected_items = [items[index] for index in _indices(len(items), max_tensors)]

    values: dict[str, float] = {}
    tensor_contracts: dict[str, dict[str, Any]] = {}
    expected_replica_counts: dict[str, int] = {}
    for name, raw in selected_items:
        value = raw.detach()
        if _is_dtensor(value):
            _validate_placements(value)
            local = value.to_local().detach()
            global_shape = tuple(int(item) for item in value.shape)
            local_shape, offset = _local_shape_offset(value)
            if tuple(int(item) for item in local.shape) != local_shape:
                raise RuntimeError(
                    f"{name}: local tensor shape {tuple(local.shape)} does not match "
                    f"DTensor layout {local_shape}"
                )
        else:
            local = value
            global_shape = tuple(int(item) for item in value.shape)
            local_shape = global_shape
            offset = (0,) * len(global_shape)
        global_numel = int(math.prod(global_shape)) if global_shape else 1
        sample_indices = _indices(global_numel, samples_per_tensor)
        tensor_contracts[name] = {
            "shape": list(global_shape),
            "numel": global_numel,
            "dtype": str(local.dtype),
            "sample_flat_indices": list(sample_indices),
        }
        expected_replica_counts[name] = _expected_replica_count(value)
        if global_numel == 0:
            continue

        local_values: list[tuple[str, Any]] = []
        for flat_index in sample_indices:
            global_coordinate = _unravel(flat_index, global_shape) if global_shape else ()
            if not all(
                offset[axis]
                <= global_coordinate[axis]
                < offset[axis] + local_shape[axis]
                for axis in range(len(global_shape))
            ):
                continue
            local_coordinate = tuple(
                global_coordinate[axis] - offset[axis]
                for axis in range(len(global_shape))
            )
            local_values.append((f"{name}[{flat_index}]", local[local_coordinate]))
        if local_values:
            selected = torch.stack(
                [number.detach().float() for _key, number in local_values]
            ).cpu()
            if not bool(torch.isfinite(selected).all()):
                raise ValueError(f"{name}: sampled state contains non-finite values")
            for (key, _value), number in zip(
                local_values, selected.tolist(), strict=True
            ):
                values[key] = float(number)

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "global_rank": int(global_rank),
        "samples_per_tensor": int(samples_per_tensor),
        "max_tensors": int(max_tensors),
        "inventory_tensor_count": len(inventory),
        "inventory_sha256": hashlib.sha256(_canonical_json(inventory)).hexdigest(),
        "selected_tensor_names": [name for name, _value in selected_items],
        "tensor_contracts": tensor_contracts,
        "expected_replica_counts": expected_replica_counts,
        "values": values,
    }
    body["sha256"] = _sha256_mapping(body)
    return body


def module_parameter_sketch(module: Any, *, global_rank: int) -> dict[str, Any]:
    return sketch_named_tensors(
        (
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ),
        global_rank=global_rank,
    )


def module_gradient_sketch(module: Any, *, global_rank: int) -> dict[str, Any]:
    return sketch_named_tensors(
        (
            (
                name,
                _state_with_parameter_layout(parameter, parameter.grad),
            )
            for name, parameter in module.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        ),
        global_rank=global_rank,
    )


def _state_with_parameter_layout(parameter: Any, value: Any) -> Any:
    """Return state with the parameter's global layout when state is local-only."""

    if _is_dtensor(value) or not _is_dtensor(parameter):
        return value
    local = value.detach()
    expected_shape, _offset = _local_shape_offset(parameter)
    if tuple(int(item) for item in local.shape) != expected_shape:
        raise RuntimeError(
            "plain optimizer/EMA state does not match its parameter's local shard: "
            f"state={tuple(local.shape)} expected={expected_shape}"
        )
    return _LocalLayoutView(parameter, local)


def optimizer_sketch(optimizer: Any, module: Any, *, global_rank: int) -> dict[str, Any]:
    name_by_id = {id(parameter): name for name, parameter in module.named_parameters()}
    tensors: list[tuple[str, Any]] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = name_by_id.get(id(parameter))
            if name is None:
                raise RuntimeError("optimizer parameter is absent from named_parameters")
            state = optimizer.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq", "step"):
                value = state.get(key)
                if value is not None:
                    # Adam's scalar step is replicated optimizer metadata even
                    # when its parameter/moments are sharded.  Do not attach a
                    # parameter shard layout to that scalar.
                    state_value = (
                        value
                        if key == "step"
                        else _state_with_parameter_layout(parameter, value)
                    )
                    tensors.append(
                        (
                            f"{name}/{key}",
                            state_value,
                        )
                    )
    return sketch_named_tensors(tensors, global_rank=global_rank)


def ema_sketch(ema: Any, module: Any, *, global_rank: int) -> dict[str, Any]:
    parameters = list(getattr(ema, "_parameters"))
    shadows = list(getattr(ema, "_shadows"))
    names = list(getattr(ema, "_names", ()))
    if len(parameters) != len(shadows):
        raise RuntimeError("EMA parameter/shadow inventory differs")
    if names and len(names) != len(parameters):
        raise RuntimeError("EMA name/parameter inventory differs")

    name_by_id = {
        id(parameter): name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    tensors: list[tuple[str, Any]] = []
    observed_names: list[str] = []
    for index, (parameter, shadow) in enumerate(
        zip(parameters, shadows, strict=True)
    ):
        name = name_by_id.get(id(parameter))
        if name is None:
            raise RuntimeError("EMA parameter is absent from trainable named_parameters")
        if names and str(names[index]) != name:
            raise RuntimeError("EMA stored name differs from module parameter name")
        observed_names.append(name)
        tensors.append((name, _state_with_parameter_layout(parameter, shadow)))
    if set(observed_names) != set(name_by_id.values()):
        raise RuntimeError("EMA does not cover the complete trainable inventory")
    return sketch_named_tensors(tensors, global_rank=global_rank)


def _state_identity(key: str) -> tuple[str, str]:
    match = _STATE_KEY.fullmatch(str(key))
    if match is None:
        raise ValueError(
            f"state key {key!r} must end in _parameter, _adam, _ema, or _gradient"
        )
    state_kind = match.group("kind")
    if state_kind == "adam":
        state_kind = "optimizer"
    return match.group("component"), state_kind


def write_rank_state_anchor(
    path: Path | str,
    *,
    run_id: str,
    workload: str,
    variant: str,
    cycle: int,
    rank: int,
    world_size: int,
    states: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Write one rank's complete state-anchor file without overwriting evidence.

    ``states`` accepts runner-friendly keys such as ``generator_parameter``,
    ``critic_adam``, ``generator_ema``, ``generator_gradient``, and
    ``critic_0_gradient``. Returned values can be merged directly into
    :class:`TrajectoryCycleReceipt.state_fingerprints`.
    """

    if not run_id or not workload or not variant:
        raise ValueError("run_id, workload, and variant must be non-empty")
    if isinstance(cycle, bool) or int(cycle) <= 0:
        raise ValueError("cycle must be positive")
    if isinstance(world_size, bool) or int(world_size) <= 0:
        raise ValueError("world_size must be positive")
    if isinstance(rank, bool) or not 0 <= int(rank) < int(world_size):
        raise ValueError("rank must be in [0, world_size)")
    if not states:
        raise ValueError("state anchor must contain at least one state")

    normalized: dict[str, Any] = {}
    for raw_key, sketch in sorted(states.items()):
        key = str(raw_key)
        component, state_kind = _state_identity(key)
        if not isinstance(sketch, Mapping):
            raise TypeError(f"state sketch {key!r} must be a mapping")
        value = dict(sketch)
        supplied_sha = value.pop("sha256", None)
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"state sketch {key!r} has an unsupported schema")
        if int(value.get("global_rank", -1)) != int(rank):
            raise ValueError(f"state sketch {key!r} belongs to a different rank")
        calculated_sha = _sha256_mapping(value)
        if supplied_sha != calculated_sha:
            raise ValueError(f"state sketch {key!r} SHA256 does not verify")
        value["sha256"] = supplied_sha
        normalized[key] = {
            "component": component,
            "state_kind": state_kind,
            "sketch": value,
        }

    body: dict[str, Any] = {
        "schema_version": RANK_ANCHOR_SCHEMA_VERSION,
        "run_id": str(run_id),
        "workload": str(workload),
        "variant": str(variant),
        "cycle": int(cycle),
        "rank": int(rank),
        "world_size": int(world_size),
        "states": normalized,
    }
    body["sha256"] = _sha256_mapping(body)
    target = Path(path)
    artifact_sha256 = write_state_sketch(target, body)
    return {
        "state_sketch_path": str(target),
        "state_sketch_sha256": artifact_sha256,
        "state_content_sha256": str(body["sha256"]),
    }


def write_state_sketch(path: Path | str, value: Mapping[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(value), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Linking is the portable POSIX no-replace publication primitive. It
        # prevents two writers that raced past an exists() check from silently
        # replacing one another's evidence.
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite state sketch: {target}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class _Anchor:
    path: Path
    run_id: str
    workload: str
    variant: str
    cycle: int
    rank: int
    world_size: int
    states: Mapping[str, Mapping[str, Any]]
    sha256: str


def _load_anchor(path: Path) -> _Anchor:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: anchor must be a JSON object")
    body = dict(value)
    supplied_sha = body.pop("sha256", None)
    if body.get("schema_version") != RANK_ANCHOR_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported state-anchor schema")
    if supplied_sha != _sha256_mapping(body):
        raise ValueError(f"{path}: state-anchor SHA256 does not verify")
    states = body.get("states")
    if not isinstance(states, Mapping) or not states:
        raise ValueError(f"{path}: states must be a non-empty mapping")
    rank = int(body.get("rank", -1))
    world_size = int(body.get("world_size", 0))
    cycle = int(body.get("cycle", 0))
    if world_size <= 0 or not 0 <= rank < world_size or cycle <= 0:
        raise ValueError(f"{path}: invalid rank/world_size/cycle identity")
    for key, record in states.items():
        component, state_kind = _state_identity(str(key))
        if not isinstance(record, Mapping):
            raise TypeError(f"{path}: state {key!r} must be an object")
        if record.get("component") != component or record.get("state_kind") != state_kind:
            raise ValueError(f"{path}: state {key!r} identity does not match its key")
        sketch = record.get("sketch")
        if not isinstance(sketch, Mapping):
            raise TypeError(f"{path}: state {key!r} has no sketch object")
        sketch_body = dict(sketch)
        sketch_sha = sketch_body.pop("sha256", None)
        if sketch_body.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: state {key!r} has an unsupported sketch schema")
        if int(sketch_body.get("global_rank", -1)) != rank:
            raise ValueError(f"{path}: state {key!r} sketch rank differs from anchor")
        if sketch_sha != _sha256_mapping(sketch_body):
            raise ValueError(f"{path}: state {key!r} sketch SHA256 does not verify")
        contracts = sketch_body.get("tensor_contracts")
        replica_counts = sketch_body.get("expected_replica_counts")
        selected = sketch_body.get("selected_tensor_names")
        values = sketch_body.get("values")
        if (
            not isinstance(contracts, Mapping)
            or not isinstance(replica_counts, Mapping)
            or not isinstance(selected, Sequence)
            or isinstance(selected, (str, bytes))
            or not isinstance(values, Mapping)
            or set(map(str, selected)) != set(map(str, contracts))
            or set(map(str, selected)) != set(map(str, replica_counts))
            or any(int(count) <= 0 for count in replica_counts.values())
        ):
            raise ValueError(f"{path}: state {key!r} sketch structure is invalid")
    return _Anchor(
        path=path,
        run_id=str(body.get("run_id", "")),
        workload=str(body.get("workload", "")),
        variant=str(body.get("variant", "")),
        cycle=cycle,
        rank=rank,
        world_size=world_size,
        states=states,
        sha256=str(supplied_sha),
    )


def validate_rank_state_anchor(path: Path | str) -> dict[str, Any]:
    """Fully validate one published rank anchor and return its identity."""

    anchor = _load_anchor(Path(path))
    return {
        "run_id": anchor.run_id,
        "workload": anchor.workload,
        "variant": anchor.variant,
        "cycle": anchor.cycle,
        "rank": anchor.rank,
        "world_size": anchor.world_size,
        "state_keys": sorted(anchor.states),
        "state_content_sha256": anchor.sha256,
    }


def _state_directory(root: Path) -> Path:
    candidates = (root / "trajectory" / "state", root / "state", root)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("rank_*_cycle_*.json")):
            return candidate
    raise FileNotFoundError(f"no rank_*_cycle_*.json state anchors below {root}")


def _load_run(root: Path) -> dict[tuple[int, int], _Anchor]:
    directory = _state_directory(root)
    anchors: dict[tuple[int, int], _Anchor] = {}
    for path in sorted(directory.glob("rank_*_cycle_*.json")):
        anchor = _load_anchor(path)
        address = (anchor.cycle, anchor.rank)
        if address in anchors:
            raise ValueError(
                f"duplicate state anchor cycle={anchor.cycle} rank={anchor.rank}: "
                f"{anchors[address].path} and {path}"
            )
        anchors[address] = anchor
    return anchors


def _merge_state(
    anchors: Sequence[_Anchor], state_key: str
) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, Any]]:
    values: dict[str, float] = {}
    sources: dict[str, list[int]] = defaultdict(list)
    contracts: dict[str, dict[str, Any]] = {}
    inventory_sha256: str | None = None
    inventory_tensor_count: int | None = None
    sampling: tuple[int, int] | None = None
    selected_tensor_names: tuple[str, ...] | None = None
    expected_contracts: dict[str, dict[str, Any]] | None = None
    expected_replica_counts: dict[str, int] | None = None

    for anchor in anchors:
        record = anchor.states[state_key]
        sketch = record["sketch"]
        current_inventory = str(sketch["inventory_sha256"])
        current_count = int(sketch["inventory_tensor_count"])
        current_sampling = (
            int(sketch["samples_per_tensor"]),
            int(sketch["max_tensors"]),
        )
        current_selected = tuple(str(name) for name in sketch["selected_tensor_names"])
        current_contracts = {
            str(name): dict(contract)
            for name, contract in sketch["tensor_contracts"].items()
        }
        raw_replica_counts = sketch.get("expected_replica_counts")
        if not isinstance(raw_replica_counts, Mapping):
            raise ValueError(
                f"cycle={anchor.cycle} {state_key}: replica-count contract is missing"
            )
        current_replica_counts = {
            str(name): int(count) for name, count in raw_replica_counts.items()
        }
        if set(current_selected) != set(current_contracts):
            raise ValueError(
                f"cycle={anchor.cycle} {state_key}: selected-name/contract inventory differs"
            )
        if set(current_selected) != set(current_replica_counts) or any(
            count <= 0 for count in current_replica_counts.values()
        ):
            raise ValueError(
                f"cycle={anchor.cycle} {state_key}: replica-count inventory differs"
            )
        if inventory_sha256 is None:
            inventory_sha256 = current_inventory
            inventory_tensor_count = current_count
            sampling = current_sampling
            selected_tensor_names = current_selected
            expected_contracts = current_contracts
            expected_replica_counts = current_replica_counts
        elif (
            current_inventory != inventory_sha256
            or current_count != inventory_tensor_count
            or current_sampling != sampling
            or current_selected != selected_tensor_names
            or current_contracts != expected_contracts
            or current_replica_counts != expected_replica_counts
        ):
            raise ValueError(
                f"cycle={anchor.cycle} {state_key}: rank inventories/sampling differ"
            )

        for name, raw_contract in current_contracts.items():
            contract = dict(raw_contract)
            prior = contracts.setdefault(str(name), contract)
            if prior != contract:
                raise ValueError(
                    f"cycle={anchor.cycle} {state_key}/{name}: tensor contract differs across ranks"
                )
        for coordinate, raw_number in sketch["values"].items():
            number = float(raw_number)
            if not math.isfinite(number):
                raise ValueError(
                    f"cycle={anchor.cycle} {state_key}/{coordinate}: non-finite value"
                )
            if coordinate in values and values[coordinate] != number:
                raise ValueError(
                    f"cycle={anchor.cycle} {state_key}/{coordinate}: replicated ranks disagree "
                    f"({values[coordinate]!r} != {number!r})"
                )
            values[str(coordinate)] = number
            sources[str(coordinate)].append(anchor.rank)

    if not values:
        raise ValueError(f"cycle={anchors[0].cycle} {state_key}: merged sketch is empty")
    expected_coordinates = {
        f"{name}[{flat_index}]"
        for name, contract in contracts.items()
        for flat_index in contract["sample_flat_indices"]
    }
    missing = sorted(expected_coordinates - set(values))
    extra = sorted(set(values) - expected_coordinates)
    if missing or extra:
        raise ValueError(
            f"cycle={anchors[0].cycle} {state_key}: coordinate coverage differs from contract "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    assert expected_replica_counts is not None
    wrong_replication = {
        coordinate: {
            "observed": len(sources[coordinate]),
            "expected": expected_replica_counts[coordinate.rsplit("[", 1)[0]],
            "ranks": sources[coordinate],
        }
        for coordinate in sorted(expected_coordinates)
        if len(sources[coordinate])
        != expected_replica_counts[coordinate.rsplit("[", 1)[0]]
    }
    if wrong_replication:
        examples = list(wrong_replication.items())[:10]
        raise ValueError(
            f"cycle={anchors[0].cycle} {state_key}: replica contribution count differs "
            f"from placement contract: {examples}"
        )
    replica_counts = [len(sources[coordinate]) for coordinate in expected_coordinates]
    return values, contracts, {
        "inventory_sha256": inventory_sha256,
        "inventory_tensor_count": inventory_tensor_count,
        "samples_per_tensor": sampling[0] if sampling else None,
        "max_tensors": sampling[1] if sampling else None,
        "coordinate_count": len(values),
        "replicated_coordinate_count": sum(len(ranks) > 1 for ranks in sources.values()),
        "min_replica_count": min(replica_counts),
        "max_replica_count": max(map(len, sources.values())),
        "expected_replica_counts": expected_replica_counts,
        "replica_agreement": "EXACT",
    }


def _vector_metrics(
    reference: Mapping[str, float],
    candidate: Mapping[str, float],
    contracts: Mapping[str, Mapping[str, Any]],
    *,
    near_zero: float,
) -> dict[str, Any]:
    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate))
        extra = sorted(set(candidate) - set(reference))
        raise ValueError(
            f"state coordinate mismatch missing={missing[:10]} extra={extra[:10]}"
        )
    keys = sorted(reference)
    ref2 = sum(reference[key] * reference[key] for key in keys)
    candidate2 = sum(candidate[key] * candidate[key] for key in keys)
    diff2 = sum((candidate[key] - reference[key]) ** 2 for key in keys)
    dot = sum(reference[key] * candidate[key] for key in keys)
    ref_norm = math.sqrt(ref2)
    candidate_norm = math.sqrt(candidate2)
    diff_norm = math.sqrt(diff2)
    if ref_norm <= near_zero and candidate_norm <= near_zero:
        cosine = 1.0
        normalized_l2 = 0.0
        norm_ratio = 1.0
        zero_policy = "BOTH_ZERO_IDENTICAL"
    elif ref_norm <= near_zero or candidate_norm <= near_zero:
        cosine = 0.0
        normalized_l2 = diff_norm / max(ref_norm, candidate_norm, near_zero)
        norm_ratio = 0.0 if candidate_norm <= near_zero else 1e30
        zero_policy = "ONE_VECTOR_ZERO"
    else:
        cosine = max(-1.0, min(1.0, dot / (ref_norm * candidate_norm)))
        normalized_l2 = diff_norm / ref_norm
        norm_ratio = candidate_norm / ref_norm
        zero_policy = "STANDARD"

    named: dict[str, dict[str, Any]] = {}
    for name, contract in sorted(contracts.items()):
        tensor_keys = [f"{name}[{index}]" for index in contract["sample_flat_indices"]]
        tensor_reference = {key: reference[key] for key in tensor_keys}
        tensor_candidate = {key: candidate[key] for key in tensor_keys}
        metric = _vector_metrics(
            tensor_reference,
            tensor_candidate,
            {},
            near_zero=near_zero,
        )
        named[name] = {
            key: metric[key]
            for key in ("cosine", "normalized_l2", "norm_ratio", "zero_policy")
        }
    min_named_cosine = min(
        (float(value["cosine"]) for value in named.values()), default=1.0
    )
    return {
        "cosine": cosine,
        "normalized_l2": normalized_l2,
        "norm_ratio": norm_ratio,
        "min_named_tensor_cosine": min_named_cosine,
        "zero_policy": zero_policy,
        "reference_norm": ref_norm,
        "candidate_norm": candidate_norm,
        "difference_norm": diff_norm,
        "coordinate_count": len(keys),
        "named_tensors": named,
    }


def compare_state_sketch_roots(
    reference_root: Path | str,
    candidate_root: Path | str,
    *,
    required_cycles: Sequence[int] = (1, 10, 25, 50),
    thresholds: TrajectoryThresholds = COARSE_50C_THRESHOLDS,
) -> dict[str, Any]:
    """Merge rank anchors and compute topology-independent state metrics."""

    reference = _load_run(Path(reference_root))
    candidate = _load_run(Path(candidate_root))
    cycles = tuple(sorted({int(value) for value in required_cycles}))
    if not cycles or cycles[0] <= 0:
        raise ValueError("required_cycles must contain positive cycle numbers")

    reference_first = next(iter(reference.values()))
    candidate_first = next(iter(candidate.values()))
    if reference_first.workload != candidate_first.workload:
        raise ValueError("reference/candidate workload differs")
    world_size = reference_first.world_size
    if candidate_first.world_size != world_size:
        raise ValueError("reference/candidate world size differs")
    extra_cycles_by_run: dict[str, list[int]] = {}
    for label, anchors in (("reference", reference), ("candidate", candidate)):
        identities = {
            (anchor.run_id, anchor.workload, anchor.variant, anchor.world_size)
            for anchor in anchors.values()
        }
        if len(identities) != 1:
            raise ValueError(f"{label} state-anchor run identity is inconsistent")
        expected_addresses = {
            (cycle, rank) for cycle in cycles for rank in range(world_size)
        }
        observed_addresses = set(anchors)
        missing = sorted(expected_addresses - observed_addresses)
        if missing:
            raise ValueError(f"{label} state anchors are incomplete: {missing[:20]}")
        extra_cycles_by_run[label] = sorted(
            {cycle for cycle, _rank in observed_addresses if cycle not in cycles}
        )

    comparisons: list[dict[str, Any]] = []
    state_comparisons: list[TrajectoryStateComparison] = []
    temporal_contracts: dict[str, dict[str, Any]] = {
        "reference": {},
        "candidate": {},
    }
    temporal_state_keys: dict[str, set[str] | None] = {
        "reference": None,
        "candidate": None,
    }
    for cycle in cycles:
        reference_anchors = [reference[(cycle, rank)] for rank in range(world_size)]
        candidate_anchors = [candidate[(cycle, rank)] for rank in range(world_size)]
        reference_keys = set(reference_anchors[0].states)
        candidate_keys = set(candidate_anchors[0].states)
        for label, keys in (
            ("reference", reference_keys),
            ("candidate", candidate_keys),
        ):
            if temporal_state_keys[label] is None:
                temporal_state_keys[label] = set(keys)
            elif keys != temporal_state_keys[label]:
                raise ValueError(
                    f"{label} cycle={cycle}: state-key inventory changed across anchors"
                )
        for anchors, label, expected_keys in (
            (reference_anchors, "reference", reference_keys),
            (candidate_anchors, "candidate", candidate_keys),
        ):
            for anchor in anchors:
                if set(anchor.states) != expected_keys:
                    raise ValueError(
                        f"{label} cycle={cycle}: state-key inventory differs on rank {anchor.rank}"
                    )
        if reference_keys != candidate_keys:
            raise ValueError(
                f"cycle={cycle}: reference/candidate state keys differ: "
                f"reference={sorted(reference_keys)} candidate={sorted(candidate_keys)}"
            )

        for state_key in sorted(reference_keys):
            reference_values, reference_contracts, reference_merge = _merge_state(
                reference_anchors, state_key
            )
            candidate_values, candidate_contracts, candidate_merge = _merge_state(
                candidate_anchors, state_key
            )
            for label, contracts, merged in (
                ("reference", reference_contracts, reference_merge),
                ("candidate", candidate_contracts, candidate_merge),
            ):
                temporal_contract = {
                    "inventory_sha256": merged["inventory_sha256"],
                    "inventory_tensor_count": merged["inventory_tensor_count"],
                    "samples_per_tensor": merged["samples_per_tensor"],
                    "max_tensors": merged["max_tensors"],
                    "expected_replica_counts": merged[
                        "expected_replica_counts"
                    ],
                    "contracts": contracts,
                }
                prior = temporal_contracts[label].setdefault(
                    state_key, temporal_contract
                )
                if prior != temporal_contract:
                    raise ValueError(
                        f"{label} cycle={cycle} {state_key}: state inventory/sampling "
                        "changed across anchors"
                    )
            if reference_contracts != candidate_contracts:
                raise ValueError(
                    f"cycle={cycle} {state_key}: reference/candidate tensor contracts differ"
                )
            if (
                reference_merge["inventory_sha256"]
                != candidate_merge["inventory_sha256"]
                or reference_merge["inventory_tensor_count"]
                != candidate_merge["inventory_tensor_count"]
            ):
                raise ValueError(
                    f"cycle={cycle} {state_key}: reference/candidate full inventory differs"
                )
            metrics = _vector_metrics(
                reference_values,
                candidate_values,
                reference_contracts,
                near_zero=float(thresholds.near_zero),
            )
            component, state_kind = _state_identity(state_key)
            comparison = TrajectoryStateComparison(
                cycle=cycle,
                component=component,
                state_kind=state_kind,
                cosine=float(metrics["cosine"]),
                normalized_l2=float(metrics["normalized_l2"]),
                norm_ratio=float(metrics["norm_ratio"]),
                min_named_tensor_cosine=float(metrics["min_named_tensor_cosine"]),
                finite=True,
                reference_digest=hashlib.sha256(
                    _canonical_json(reference_values)
                ).hexdigest(),
                candidate_digest=hashlib.sha256(
                    _canonical_json(candidate_values)
                ).hexdigest(),
            )
            evaluated = comparison.evaluate(thresholds)
            state_comparisons.append(comparison)
            comparisons.append(
                {
                    "state_key": state_key,
                    **evaluated,
                    "metrics": metrics,
                    "reference_merge": reference_merge,
                    "candidate_merge": candidate_merge,
                }
            )

    review_count = sum(row["status"] != "PASS" for row in comparisons)
    result = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "status": "PASS" if review_count == 0 else "REVIEW",
        "evaluation_policy": "POST_RUN_ONLY_NO_NUMERICAL_EARLY_STOP",
        "required_cycles": list(cycles),
        "extra_cycles_ignored": extra_cycles_by_run,
        "world_size": world_size,
        "workload": reference_first.workload,
        "reference": {
            "root": str(Path(reference_root)),
            "run_id": reference_first.run_id,
            "variant": reference_first.variant,
            "anchor_count": len(reference),
        },
        "candidate": {
            "root": str(Path(candidate_root)),
            "run_id": candidate_first.run_id,
            "variant": candidate_first.variant,
            "anchor_count": len(candidate),
        },
        "thresholds": dataclasses.asdict(thresholds),
        "comparison_count": len(comparisons),
        "review_count": review_count,
        "comparisons": comparisons,
        "state_comparisons": [dataclasses.asdict(value) for value in state_comparisons],
    }
    result["sha256"] = _sha256_mapping(result)
    return result


__all__ = [
    "COMPARISON_SCHEMA_VERSION",
    "RANK_ANCHOR_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "compare_state_sketch_roots",
    "ema_sketch",
    "module_gradient_sketch",
    "module_parameter_sketch",
    "optimizer_sketch",
    "sketch_named_tensors",
    "validate_rank_state_anchor",
    "write_rank_state_anchor",
    "write_state_sketch",
]
