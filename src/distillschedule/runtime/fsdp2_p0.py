"""P0 optimizations for the static Helios FSDP2 Full5 execution.

The module keeps the PR #13 baseline intact and provides three opt-in changes:

* batch the five post-Generator-update no-grad rollouts by logical step;
* pack the Real-Score conditional/unconditional requests into one B2 forward;
* choose world16 FSDP or a two-node ``replicate=2, shard=8`` HSDP mesh.

All helpers are fail-closed: operations with different physical DiT schedules,
non-stackable trace tensors, or unsupported execution backends are rejected
before model work.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


WORLD_SIZE = 16
NODE_COUNT = 2
GPUS_PER_NODE = 8
SUPPORTED_ROLLOUT_BATCH_SIZES = (1, 2, 5)
SUPPORTED_MESH_POLICIES = ("world16-fsdp", "node-hsdp")
SUPPORTED_ROLLOUT_BACKENDS = ("native-fsdp2-batch",)
ROLLOUT_TENSOR_ROLES = (
    "text_embedding",
    "continuous_history",
    "history_short",
    "history_mid",
    "history_long",
    "indices_hidden_states",
    "indices_history_short",
    "indices_history_mid",
    "indices_history_long",
    "entry_noise",
)


@dataclasses.dataclass(frozen=True)
class StaticFSDP2P0Policy:
    """Frozen execution policy for one P0 experiment.

    Defaults deliberately reproduce the validated PR #13 execution shape.  Any
    optimized candidate must opt in to batching and/or Score packing.
    """

    mesh_policy: str = "world16-fsdp"
    rollout_batch_size: int = 1
    pack_real_score_pair: bool = False
    rollout_backend: str = "native-fsdp2-batch"

    def __post_init__(self) -> None:
        if self.mesh_policy not in SUPPORTED_MESH_POLICIES:
            raise ValueError(f"unsupported mesh policy: {self.mesh_policy}")
        if self.rollout_batch_size not in SUPPORTED_ROLLOUT_BATCH_SIZES:
            raise ValueError(
                "rollout_batch_size must be one of "
                f"{SUPPORTED_ROLLOUT_BATCH_SIZES}, got {self.rollout_batch_size}"
            )
        if self.rollout_backend not in SUPPORTED_ROLLOUT_BACKENDS:
            raise ValueError(
                f"rollout backend {self.rollout_backend!r} is not integrated; "
                "external serving engines require an explicit latent/trace/LoRA-sync adapter"
            )

    @property
    def id(self) -> str:
        return (
            f"{self.mesh_policy}-rollout-b{self.rollout_batch_size}-"
            f"realpair-{int(self.pack_real_score_pair)}"
        )


def mesh_shape_for_policy(policy: str, *, world_size: int = WORLD_SIZE) -> tuple[int, ...]:
    if world_size != WORLD_SIZE:
        raise ValueError(f"P0 Helios experiments require world16, got {world_size}")
    if policy == "world16-fsdp":
        return (WORLD_SIZE,)
    if policy == "node-hsdp":
        return (NODE_COUNT, GPUS_PER_NODE)
    raise ValueError(f"unsupported mesh policy: {policy}")


def create_device_mesh(policy: str, *, world_size: int = WORLD_SIZE) -> Any:
    """Create the exact FSDP/HSDP mesh used by the P0 worker."""

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    if not dist.is_initialized() or dist.get_world_size() != world_size:
        raise RuntimeError("the world16 process group must exist before mesh creation")
    shape = mesh_shape_for_policy(policy, world_size=world_size)
    if policy == "world16-fsdp":
        return init_device_mesh("cuda", shape, mesh_dim_names=("fsdp",))
    # Row-major global ranks make each 8-rank shard group node-local:
    # node0=[0..7], node1=[8..15]. Replica groups connect equal local ranks.
    return init_device_mesh(
        "cuda", shape, mesh_dim_names=("replicate", "shard")
    )


def apply_blockwise_fsdp2_policy(
    module: Any,
    *,
    mesh_policy: str,
    world_size: int = WORLD_SIZE,
    expected_block_count: int = 40,
    param_dtype: Any | None = None,
    reduce_dtype: Any | None = None,
) -> dict[str, Any]:
    """Shard every expected block bottom-up under FSDP or node-local HSDP.

    The default remains the fail-closed 40-block Helios-14B contract.  A
    caller must opt in explicitly to another audited architecture, such as the
    30-block Wan2.1-T2V-1.3B control used by the FSDP upper-bound experiment.
    """

    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    if not dist.is_initialized() or dist.get_world_size() != world_size:
        raise RuntimeError("world16 must be initialized before FSDP2")
    blocks = list(getattr(module, "blocks", ()))
    if expected_block_count <= 0:
        raise ValueError("expected_block_count must be positive")
    if len(blocks) != expected_block_count:
        raise AssertionError(
            f"Transformer block count changed: {len(blocks)} != {expected_block_count}"
        )
    mesh = create_device_mesh(mesh_policy, world_size=world_size)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype if param_dtype is not None else torch.bfloat16,
        reduce_dtype=reduce_dtype if reduce_dtype is not None else torch.float32,
    )
    for block in blocks:
        fully_shard(
            block,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=mp_policy,
        )
    fully_shard(
        module,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=mp_policy,
    )
    torch.cuda.synchronize()
    parameters = list(module.parameters())
    if not parameters or not all(hasattr(value, "device_mesh") for value in parameters):
        raise AssertionError("FSDP2 did not convert every parameter to DTensor")
    return {
        "backend": (
            "FSDP2_COMPOSABLE_FULL_SHARD"
            if mesh_policy == "world16-fsdp"
            else "FSDP2_COMPOSABLE_HYBRID_SHARD"
        ),
        "mesh_policy": mesh_policy,
        "mesh_shape": list(mesh_shape_for_policy(mesh_policy, world_size=world_size)),
        "world_size": world_size,
        "block_count": len(blocks),
        "wrap_policy": f"{expected_block_count}_BLOCKS_BOTTOM_UP_THEN_ROOT",
        "reshard_after_forward": True,
        "param_compute_dtype": str(mp_policy.param_dtype),
        "reduce_dtype": str(mp_policy.reduce_dtype),
        "cpu_parameter_offload": False,
        "parameter_streaming": False,
        "transition": False,
    }


def chunked(values: Sequence[Any], size: int) -> tuple[tuple[Any, ...], ...]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return tuple(tuple(values[index : index + size]) for index in range(0, len(values), size))


def build_p0_execution_order(
    logical_steps: Sequence[int],
    *,
    rollout_batch_size: int,
    ema_steps: Iterable[int] = (),
) -> tuple[str, ...]:
    """Return the legal phase order used by the optimized worker."""

    steps = tuple(int(value) for value in logical_steps)
    if len(steps) != 5 or len(set(steps)) != 5:
        raise ValueError("P0 Full5 requires five unique logical steps")
    ema = {int(value) for value in ema_steps}
    result = ["generator_update"]
    for group in chunked(steps, rollout_batch_size):
        result.append("generator_rollout_batch:" + ",".join(map(str, group)))
    for step in steps:
        result.append(f"critic_update:{step}")
        if step in ema:
            result.append(f"ema_update:{step}")
    return tuple(result)


def estimate_rollout_collective_groups(
    *, physical_calls_per_rollout: int, logical_rollouts: int, batch_size: int
) -> dict[str, int | float]:
    if min(physical_calls_per_rollout, logical_rollouts, batch_size) <= 0:
        raise ValueError("rollout collective estimate requires positive values")
    baseline = physical_calls_per_rollout * logical_rollouts
    batches = (logical_rollouts + batch_size - 1) // batch_size
    optimized = physical_calls_per_rollout * batches
    return {
        "logical_rollouts": logical_rollouts,
        "rollout_batches": batches,
        "baseline_forward_groups": baseline,
        "optimized_forward_groups": optimized,
        "group_reduction": baseline - optimized,
        "ideal_group_reduction_fraction": 1.0 - optimized / baseline,
    }


def validate_exact_rollout_call_inventory(
    row: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Validate the shared exact-rollout receipt schema.

    Generator-update and B1 no-grad rollout producers use the same physical
    and logical DiT census.  Keep that equality explicit in the receipt rather
    than treating an absent logical count as if it matched the physical one.
    """

    required = (
        "call_count",
        "stage_call_count",
        "logical_call_count",
        "logical_stage_call_count",
        "dummy_call_count",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{label} is missing call-inventory fields: {missing}")

    def count(key: str) -> int:
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label}.{key} must be an integer")
        if value < 0:
            raise ValueError(f"{label}.{key} must be non-negative")
        return int(value)

    def stage_counts(key: str) -> dict[int, int]:
        raw = row[key]
        if not isinstance(raw, Mapping) or not raw:
            raise TypeError(f"{label}.{key} must be a non-empty mapping")
        result: dict[int, int] = {}
        for raw_stage, raw_count in raw.items():
            if isinstance(raw_count, bool) or not isinstance(raw_count, int):
                raise TypeError(f"{label}.{key}[{raw_stage!r}] must be an integer")
            stage = int(raw_stage)
            if stage in result:
                raise ValueError(f"{label}.{key} has duplicate stage {stage}")
            if raw_count < 0:
                raise ValueError(
                    f"{label}.{key}[{raw_stage!r}] must be non-negative"
                )
            result[stage] = int(raw_count)
        return result

    physical = count("call_count")
    logical = count("logical_call_count")
    dummy = count("dummy_call_count")
    physical_stages = stage_counts("stage_call_count")
    logical_stages = stage_counts("logical_stage_call_count")
    if sum(physical_stages.values()) != physical:
        raise AssertionError(f"{label} physical stage-call census does not add up")
    if sum(logical_stages.values()) != logical:
        raise AssertionError(f"{label} logical stage-call census does not add up")
    if dummy != 0 or physical != logical or physical_stages != logical_stages:
        raise AssertionError(
            f"{label} changed exact physical/logical DiT census: "
            f"physical={physical}/{physical_stages}, "
            f"logical={logical}/{logical_stages}, dummy={dummy}"
        )
    return {
        "call_count": physical,
        "stage_call_count": physical_stages,
        "logical_call_count": logical,
        "logical_stage_call_count": logical_stages,
        "dummy_call_count": dummy,
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _stack_tensors(values: Sequence[Any]) -> Any:
    import torch

    if not values:
        raise ValueError("cannot stack an empty tensor list")
    if not all(torch.is_tensor(value) for value in values):
        raise TypeError("trace batching only supports tensors")
    reference = values[0]
    if any(value.dtype != reference.dtype for value in values):
        raise ValueError("trace tensor dtypes differ across rollout steps")
    if any(tuple(value.shape) != tuple(reference.shape) for value in values):
        raise ValueError("trace tensor shapes differ across rollout steps")
    if reference.ndim == 0:
        return torch.stack(tuple(values), dim=0)
    if int(reference.shape[0]) == 1:
        return torch.cat(tuple(values), dim=0)
    return torch.stack(tuple(values), dim=0)


def _share_identical_tensors(values: Sequence[Any], *, label: str) -> Any:
    import torch

    if not values or not all(torch.is_tensor(value) for value in values):
        raise TypeError(f"{label} requires a non-empty tensor sequence")
    reference = values[0]
    if any(
        value.dtype != reference.dtype
        or tuple(value.shape) != tuple(reference.shape)
        or not torch.equal(value, reference)
        for value in values[1:]
    ):
        raise ValueError(f"shared rollout metadata {label!r} differs across steps")
    return reference


def _is_shared_index_key(key: str) -> bool:
    leaf = key.rsplit("/", 1)[-1]
    return leaf.startswith("indices_")


def _normalized_rollout_control(control: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(control)
    # ``global_step`` is logical-step metadata and therefore differs across the
    # five source traces. Every other physical control field must match. Real
    # B2/B5 correctness runs compare each batched row against that row's B1
    # source-step execution, so any runtime use of this metadata is caught by
    # the numerical parity gate rather than assumed away.
    value.pop("global_step", None)
    return value


def validate_rollout_entries(entries: Sequence[Any]) -> dict[str, Any]:
    """Prove that a set of rank-local critic rollouts can share one model call batch."""

    if not entries:
        raise ValueError("rollout batching requires at least one entry")
    signatures: list[dict[str, Any]] = []
    tensor_inventories: list[dict[str, tuple[tuple[int, ...], str]]] = []
    global_steps: list[int] = []
    for loaded in entries:
        operation = loaded.operation_trace.operation(loaded.pool_entry.rollout_operation)
        raw_control = dict(operation["control"])
        # The official rank16 trace stores logical global_step on the sibling
        # ``step/<step>/branch`` operation, not on the no-grad rollout row.
        # Accept either representation, but fail closed if the two disagree.
        rollout_step = raw_control.get("global_step")
        if not isinstance(rollout_step, int):
            logical_step = int(loaded.pool_entry.logical_step)
            branch = loaded.operation_trace.operation(
                f"step/{logical_step}/branch"
            )
            branch_step = branch.get("control", {}).get("global_step")
            if not isinstance(branch_step, int):
                raise ValueError("critic rollout lacks integer global_step metadata")
            rollout_step = branch_step
        logical_step = int(loaded.pool_entry.logical_step)
        if int(rollout_step) != logical_step:
            raise ValueError(
                "critic rollout global_step disagrees with logical-step binding: "
                f"{rollout_step} != {logical_step}"
            )
        global_steps.append(int(rollout_step))
        control = _normalized_rollout_control(raw_control)
        schedule = control.get("pyramid_stage_call_sequence")
        if not isinstance(schedule, list) or not schedule:
            raise ValueError("critic rollout lacks a frozen pyramid schedule")
        signatures.append(control)
        missing_roles = set(ROLLOUT_TENSOR_ROLES) - set(loaded.tensors)
        if missing_roles:
            raise ValueError(
                f"critic rollout misses required tensors: {sorted(missing_roles)}"
            )
        tensor_inventories.append(
            {
                name: (
                    tuple(int(dim) for dim in loaded.tensors[name].shape),
                    str(loaded.tensors[name].dtype),
                )
                for name in ROLLOUT_TENSOR_ROLES
            }
        )
    if any(value != signatures[0] for value in signatures[1:]):
        raise ValueError("critic rollout controls differ across logical steps")
    if any(value != tensor_inventories[0] for value in tensor_inventories[1:]):
        raise ValueError("critic rollout tensor inventories differ across logical steps")
    schedule = signatures[0]["pyramid_stage_call_sequence"]
    return {
        "status": "PASS",
        "batch_size": len(entries),
        "logical_steps": [int(value.pool_entry.logical_step) for value in entries],
        "global_steps": global_steps,
        "global_step_policy": "metadata_only_subject_to_b1_vs_batched_integration_parity",
        "call_count": int(signatures[0]["call_count"]),
        "stage_call_count": {
            int(item["stage"]): int(item["call_count"]) for item in schedule
        },
    }


class BatchedOperationTrace:
    """Map one synthetic rollout prefix onto multiple immutable step traces."""

    def __init__(self, entries: Sequence[Any], *, synthetic_prefix: str) -> None:
        if not synthetic_prefix or synthetic_prefix.endswith("/"):
            raise ValueError("synthetic_prefix must be a non-empty normalized path")
        self.entries = tuple(entries)
        self.synthetic_prefix = synthetic_prefix
        self.validation = validate_rollout_entries(self.entries)
        self.root = Path("<batched-operation-trace>")
        self.manifest = dict(self.entries[0].operation_trace.manifest)
        self.tensors: dict[str, Any] = {}
        self.trace_hash = hashlib.sha256(
            _canonical(
                {
                    "synthetic_prefix": synthetic_prefix,
                    "trace_hashes": [value.operation_trace.trace_hash for value in self.entries],
                    "steps": [int(value.pool_entry.logical_step) for value in self.entries],
                }
            )
        ).hexdigest()

    def _map(self, entry: Any, key: str) -> str:
        if key.startswith(self.synthetic_prefix):
            suffix = key[len(self.synthetic_prefix) :]
            return f"step/{int(entry.pool_entry.logical_step)}/critic{suffix}"
        return key

    def operation(self, key: str) -> dict[str, Any]:
        operations = [
            entry.operation_trace.operation(self._map(entry, key)) for entry in self.entries
        ]
        normalized = [
            {
                **value,
                "key": key,
                "control": _normalized_rollout_control(value.get("control", {})),
            }
            for value in operations
        ]
        if any(value != normalized[0] for value in normalized[1:]):
            raise ValueError(f"operation {key!r} is not batch-equivalent across steps")
        result = dict(operations[0])
        result["key"] = key
        result["control"] = dict(operations[0].get("control", {}))
        return result

    def tensor(self, key: str, *, clone: bool = True) -> Any:
        cache_key = f"{key}|clone={int(clone)}"
        if cache_key not in self.tensors:
            values = [
                entry.operation_trace.tensor(self._map(entry, key), clone=clone)
                for entry in self.entries
            ]
            self.tensors[cache_key] = (
                _share_identical_tensors(values, label=key)
                if _is_shared_index_key(key)
                else _stack_tensors(values)
            )
        value = self.tensors[cache_key]
        return value.clone() if clone else value


@dataclasses.dataclass(frozen=True)
class BatchedLoadedPoolEntry:
    pool_entry: Any
    operation_trace: BatchedOperationTrace
    tensors: dict[str, Any]
    members: tuple[Any, ...]


def make_batched_loaded_entry(
    entries: Sequence[Any], *, synthetic_prefix: str = "batched/critic"
) -> BatchedLoadedPoolEntry:
    entries = tuple(entries)
    validation = validate_rollout_entries(entries)
    roles = ROLLOUT_TENSOR_ROLES
    for value in entries:
        missing_roles = set(roles) - set(value.tensors)
        if missing_roles:
            raise ValueError(
                f"critic rollout misses required tensors: {sorted(missing_roles)}"
            )
    tensors = {
        role: (
            _share_identical_tensors(
                [entry.tensors[role] for entry in entries], label=role
            )
            if role.startswith("indices_")
            else _stack_tensors([entry.tensors[role] for entry in entries])
        )
        for role in roles
    }
    trace = BatchedOperationTrace(entries, synthetic_prefix=synthetic_prefix)
    identity = hashlib.sha256(
        _canonical(
            {
                "members": [value.pool_entry.identity_sha256 for value in entries],
                "validation": validation,
            }
        )
    ).hexdigest()
    pool_entry = SimpleNamespace(
        trace_key=(
            f"static-fsdp2-p0/batched/"
            + "-".join(str(value.pool_entry.logical_step) for value in entries)
        ),
        identity_sha256=identity,
        rollout_operation=f"{synthetic_prefix}/rollout",
        logical_step=int(entries[0].pool_entry.logical_step),
        logical_steps=tuple(int(value.pool_entry.logical_step) for value in entries),
        synthetic_trace_prefix=synthetic_prefix,
    )
    return BatchedLoadedPoolEntry(pool_entry, trace, tensors, entries)


def _repeat_batch_tensor(
    value: Any, copies: int, *, source_batch: int, allow_shared: bool = False
) -> Any:
    import torch

    if copies <= 0:
        raise ValueError("copies must be positive")
    if not torch.is_tensor(value):
        raise TypeError("Score packing only accepts tensors")
    if value.ndim == 0:
        return value.repeat(copies)
    if int(value.shape[0]) == source_batch:
        return torch.cat([value] * copies, dim=0)
    if allow_shared:
        return value
    raise ValueError(
        "Score packing could not identify a leading batch dimension for "
        f"shape={tuple(value.shape)}, source_batch={source_batch}"
    )


def pack_score_conditions(
    batch: Mapping[str, Any], conditions: Sequence[Any]
) -> tuple[dict[str, Any], Any]:
    """Pack same-input Score requests that differ only in text condition."""

    import torch

    if len(conditions) < 2:
        raise ValueError("condition packing requires at least two requests")
    hidden = batch.get("hidden_states")
    if not torch.is_tensor(hidden) or hidden.ndim == 0:
        raise ValueError("Score packing requires a batched hidden_states tensor")
    source_batch = int(hidden.shape[0])
    if source_batch != 1:
        raise ValueError(f"Real-Score pair packing expects source B1, got B{source_batch}")
    packed: dict[str, Any] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            continue
        if key in {"generated_x0", "dmd_noise", "sigmas", "timesteps"}:
            # These fields are not consumed by score_forward(). Keep them out of
            # the packed model-call dictionary to avoid misleading memory use.
            continue
        if key == "encoder_hidden_states":
            continue
        packed[key] = _repeat_batch_tensor(
            value,
            len(conditions),
            source_batch=source_batch,
            allow_shared=key.startswith("indices_"),
        )
    packed_condition = torch.cat(tuple(conditions), dim=0)
    if int(packed_condition.shape[0]) != len(conditions):
        raise ValueError("packed conditions did not preserve one row per request")
    return packed, packed_condition


def split_packed_prediction(prediction: Any, count: int) -> tuple[Any, ...]:
    if prediction.ndim == 0 or int(prediction.shape[0]) != count:
        raise ValueError(
            f"packed Score output has shape={tuple(prediction.shape)}, expected B{count}"
        )
    return tuple(prediction[index : index + 1] for index in range(count))


def score_forward_real_pair(
    model: Any,
    batch: Mapping[str, Any],
    *,
    conditional: Any,
    unconditional: Any,
) -> tuple[Any, Any]:
    """Evaluate Real-Score cond/uncond in one adapter-disabled B2 forward."""

    from runtime.gst_pool_helios import score_forward

    packed_batch, packed_condition = pack_score_conditions(
        batch, (conditional, unconditional)
    )
    prediction = score_forward(
        model,
        packed_batch,
        condition=packed_condition,
        adapter_enabled=False,
    )
    conditional_prediction, unconditional_prediction = split_packed_prediction(
        prediction, 2
    )
    return conditional_prediction, unconditional_prediction


__all__ = [
    "BatchedLoadedPoolEntry",
    "BatchedOperationTrace",
    "GPUS_PER_NODE",
    "NODE_COUNT",
    "ROLLOUT_TENSOR_ROLES",
    "SUPPORTED_MESH_POLICIES",
    "SUPPORTED_ROLLOUT_BACKENDS",
    "StaticFSDP2P0Policy",
    "WORLD_SIZE",
    "apply_blockwise_fsdp2_policy",
    "build_p0_execution_order",
    "chunked",
    "create_device_mesh",
    "estimate_rollout_collective_groups",
    "make_batched_loaded_entry",
    "mesh_shape_for_policy",
    "pack_score_conditions",
    "score_forward_real_pair",
    "split_packed_prediction",
    "validate_exact_rollout_call_inventory",
    "validate_rollout_entries",
]
