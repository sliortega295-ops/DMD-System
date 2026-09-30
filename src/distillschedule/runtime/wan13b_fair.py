"""Fair Wan1.3B placement policies and a faster full-resident candidate.

This module is layered on the audited PR #21 workload. It does not change the
B1/global-B16 trace, model calls, G1/C5/EMA5 update inventory, loss, AdamW
hyperparameters, checkpointing, or no-offload contract.

The fair comparison fixes the previous precision confound by keeping frozen
base parameters BF16 and LoRA trainables FP32 in both Resident and FSDP2.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import socket
import time
from pathlib import Path
from typing import Any

from .wan13b_upper_bound import (
    FreshTrainableEMA,
    GLOBAL_BATCH,
    ReplicatedTrainableAllReduceEngine,
    WAN13B_BLOCK_COUNT,
    WORLD_SIZE,
    build_fresh_engine,
    materialize_adam_state,
    model_trainable_metrics,
)


REFERENCE_RESIDENT = "resident"
CLEAN_RESIDENT = "resident-clean"
ASYNC_128_RESIDENT = "resident-async128"
ASYNC_128_TREE_RESIDENT = "resident-async128-tree"
ASYNC_128_TREE_CHUNK2048_RESIDENT = "resident-async128-tree-chunk2048"
ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT = (
    "resident-async128-tree-chunk2048-compile"
)
ASYNC_256_RESIDENT = "resident-async256"
LOCAL_FAST_RESIDENT = "resident-local-fast"
FLAT_256_RESIDENT = "resident-flat256"
FLAT_512_RESIDENT = "resident-flat512"
LORA_ZERO2_RESIDENT = "resident-lora-zero2"
ROLLOUT_FUSED_RESIDENT = "resident-rollout-fused"
DDP_RESIDENT = "resident-ddp"
FAST_RESIDENT = "resident-fast"
FSDP_PLACEMENT = "world16-fsdp"
RESIDENT_PLACEMENTS = (
    REFERENCE_RESIDENT,
    CLEAN_RESIDENT,
    ASYNC_128_RESIDENT,
    ASYNC_128_TREE_RESIDENT,
    ASYNC_128_TREE_CHUNK2048_RESIDENT,
    ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
    ASYNC_256_RESIDENT,
    LOCAL_FAST_RESIDENT,
    FLAT_256_RESIDENT,
    FLAT_512_RESIDENT,
    LORA_ZERO2_RESIDENT,
    ROLLOUT_FUSED_RESIDENT,
    DDP_RESIDENT,
    FAST_RESIDENT,
)
FAIR_PLACEMENTS = (*RESIDENT_PLACEMENTS, FSDP_PLACEMENT)

DEFAULT_MODULATION_CHUNK_TOKENS = 512
MODULATION_CHUNK_TOKENS_BY_PLACEMENT = {
    ASYNC_128_TREE_RESIDENT: DEFAULT_MODULATION_CHUNK_TOKENS,
    ASYNC_128_TREE_CHUNK2048_RESIDENT: 2048,
    ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT: 2048,
}


def _torch() -> Any:
    import torch

    return torch


def _named_trainables(module: Any) -> dict[str, Any]:
    return {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def configure_modulation_chunk_tokens(module: Any, *, placement: str) -> dict[str, Any]:
    """Pin the split-modulation tile size and emit a fail-closed receipt."""

    requested = MODULATION_CHUNK_TOKENS_BY_PLACEMENT.get(placement)
    blocks = list(getattr(module, "blocks", ()))
    if len(blocks) != WAN13B_BLOCK_COUNT:
        raise AssertionError(
            f"Wan modulation chunk audit requires {WAN13B_BLOCK_COUNT} blocks, "
            f"got {len(blocks)}"
        )
    if requested is not None:
        module._distillschedule_modulation_chunk_tokens = int(requested)
        for block in blocks:
            block._distillschedule_modulation_chunk_tokens = int(requested)
    observed = [
        int(
            getattr(
                block,
                "_distillschedule_modulation_chunk_tokens",
                DEFAULT_MODULATION_CHUNK_TOKENS,
            )
        )
        for block in blocks
    ]
    if requested is not None and observed != [int(requested)] * WAN13B_BLOCK_COUNT:
        raise AssertionError("modulation chunk token setting was not applied uniformly")
    return {
        "requested_tokens": int(requested) if requested is not None else None,
        "effective_tokens": sorted(set(observed)),
        "block_count": len(blocks),
        "all_blocks_match": len(set(observed)) == 1,
        "compute_semantics": "SAME_SPLIT_MODULATION_TOKENWISE_MATH",
    }


def install_generator_nograd_block_compile(
    module: Any,
    *,
    placement: str,
    role: str,
) -> dict[str, Any]:
    """Compile only Generator no-grad Transformer-block calls.

    This is intentionally narrower than compiling the model or training loop:
    gradient-bearing Generator work, all Critic work, checkpoint replay,
    communication, optimizer, and EMA stay on the audited eager path.  The
    dispatch predicate is solely ``torch.is_grad_enabled()`` and therefore
    does not alter the call inventory or tensor inputs.
    """

    requested = placement == ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT
    blocks = list(getattr(module, "blocks", ()))
    if len(blocks) != WAN13B_BLOCK_COUNT:
        raise AssertionError(
            f"Wan compile audit requires {WAN13B_BLOCK_COUNT} blocks, got {len(blocks)}"
        )
    receipt: dict[str, Any] = {
        "requested": requested,
        "enabled": False,
        "role": str(role),
        "scope": "GENERATOR_NOGRAD_TRANSFORMER_BLOCKS_ONLY",
        "block_count": len(blocks),
        "backend": "inductor" if requested and role == "generator" else None,
        "mode": "default" if requested and role == "generator" else None,
        "dynamic": False if requested and role == "generator" else None,
        "fullgraph": False if requested and role == "generator" else None,
        "compiled_dispatch_count": 0,
        "eager_dispatch_count": 0,
        "first_dispatch_wall_seconds_by_block": [None] * len(blocks),
        "semantic_contract": (
            "NO_MODEL_CALL_OR_TENSOR_MATH_CHANGE; GRAD_PATH_REMAINS_EAGER"
        ),
    }
    module._distillschedule_nograd_compile_runtime = receipt
    if not requested or role != "generator":
        receipt["disabled_reason"] = (
            "PLACEMENT_NOT_REQUESTED" if not requested else "GENERATOR_ONLY"
        )
        return receipt

    torch = _torch()
    if not callable(getattr(torch, "compile", None)):
        raise RuntimeError("requested torch.compile arm but torch.compile is unavailable")
    cache_root = Path(
        os.environ.get(
            "DISTILLSCHEDULE_WAN13B_COMPILE_CACHE_ROOT",
            "/tmp/lyy-experiments/DMD-System/wan13b_compile_cache",
        )
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cache_dir = cache_root / socket.gethostname() / f"local_rank_{local_rank:02d}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    receipt["cache_dir"] = str(cache_dir)

    for index, block in enumerate(blocks):
        original_forward = block.forward
        compiled_forward = torch.compile(
            original_forward,
            backend="inductor",
            mode="default",
            fullgraph=False,
            dynamic=False,
        )

        @functools.wraps(original_forward)
        def dispatch(
            *args: Any,
            _compiled: Any = compiled_forward,
            _original: Any = original_forward,
            _index: int = index,
            **kwargs: Any,
        ) -> Any:
            if torch.is_grad_enabled():
                receipt["eager_dispatch_count"] += 1
                return _original(*args, **kwargs)
            first = receipt["first_dispatch_wall_seconds_by_block"][_index] is None
            started = time.perf_counter() if first else 0.0
            output = _compiled(*args, **kwargs)
            receipt["compiled_dispatch_count"] += 1
            if first:
                receipt["first_dispatch_wall_seconds_by_block"][_index] = (
                    time.perf_counter() - started
                )
            return output

        block.forward = dispatch

    receipt["enabled"] = True
    receipt["all_blocks_wrapped"] = True
    return receipt


@dataclasses.dataclass
class ResidentCleanModelState:
    """Make repeated root mode/adapter requests idempotent without changing state."""

    training: bool = True
    adapter_enabled: bool = True
    train_requests: int = 0
    train_transitions: int = 0
    adapter_enable_requests: int = 0
    adapter_enable_transitions: int = 0
    adapter_disable_requests: int = 0
    adapter_disable_transitions: int = 0

    def receipt(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def install_resident_clean_model_state(module: Any) -> ResidentCleanModelState:
    """Install an instance-local state machine after establishing train/adapter-on."""

    if hasattr(module, "_distillschedule_resident_clean_state"):
        raise RuntimeError("resident-clean model state was installed twice")
    if not hasattr(module, "enable_adapters") or not hasattr(module, "disable_adapters"):
        raise TypeError("resident-clean requires explicit PEFT adapter controls")

    # Establish the same initial state as the reference arm before wrapping.
    module.train()
    module.enable_adapters()
    state = ResidentCleanModelState(training=True, adapter_enabled=True)
    original_train = module.train
    original_enable = module.enable_adapters
    original_disable = module.disable_adapters

    def train(mode: bool = True) -> Any:
        requested = bool(mode)
        state.train_requests += 1
        if state.training == requested:
            return module
        result = original_train(requested)
        state.training = requested
        state.train_transitions += 1
        return result

    def enable_adapters() -> Any:
        state.adapter_enable_requests += 1
        if state.adapter_enabled:
            return None
        result = original_enable()
        state.adapter_enabled = True
        state.adapter_enable_transitions += 1
        return result

    def disable_adapters() -> Any:
        state.adapter_disable_requests += 1
        if not state.adapter_enabled:
            return None
        result = original_disable()
        state.adapter_enabled = False
        state.adapter_disable_transitions += 1
        return result

    module.train = train
    module.enable_adapters = enable_adapters
    module.disable_adapters = disable_adapters
    module._distillschedule_resident_clean_state = state
    return state


def apply_wan13b_fair_fsdp2_policy(module: Any) -> dict[str, Any]:
    """FULL_SHARD while preserving original BF16-base/FP32-LoRA dtypes."""

    torch = _torch()
    dist = torch.distributed
    if not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
        raise RuntimeError("fair Wan FSDP2 policy requires world16")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

    blocks = list(getattr(module, "blocks", ()))
    if len(blocks) != WAN13B_BLOCK_COUNT:
        raise AssertionError(
            f"Wan block count changed: {len(blocks)} != {WAN13B_BLOCK_COUNT}"
        )
    mesh = init_device_mesh("cuda", (WORLD_SIZE,), mesh_dim_names=("fsdp",))
    policy = MixedPrecisionPolicy(
        param_dtype=None,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=False,
    )
    for block in blocks:
        fully_shard(
            block,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=policy,
        )
    fully_shard(
        module,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=policy,
    )
    fsdp_group_count = 0
    for child in module.modules():
        if isinstance(child, FSDPModule):
            child.set_force_sum_reduction_for_comms(True)
            child.set_gradient_divide_factor(float(WORLD_SIZE))
            fsdp_group_count += 1
    torch.cuda.synchronize()

    parameters = list(module.parameters())
    if not parameters or not all(hasattr(value, "device_mesh") for value in parameters):
        raise AssertionError("fair FSDP2 did not convert all parameters to DTensor")
    trainable_dtypes = sorted(
        {str(parameter.dtype) for parameter in parameters if parameter.requires_grad}
    )
    frozen_dtypes = sorted(
        {str(parameter.dtype) for parameter in parameters if not parameter.requires_grad}
    )
    if trainable_dtypes != ["torch.float32"]:
        raise AssertionError(f"FSDP2 LoRA trainables are not FP32: {trainable_dtypes}")
    if "torch.bfloat16" not in frozen_dtypes:
        raise AssertionError(f"FSDP2 frozen base does not retain BF16: {frozen_dtypes}")
    return {
        "backend": "FSDP2_COMPOSABLE_FULL_SHARD",
        "world_size": WORLD_SIZE,
        "block_count": len(blocks),
        "wrap_policy": "30_BLOCKS_BOTTOM_UP_THEN_ROOT",
        "reshard_after_forward": True,
        "param_compute_dtype_policy": "PRESERVE_ORIGINAL_PER_PARAMETER",
        "frozen_parameter_dtypes": frozen_dtypes,
        "trainable_parameter_dtypes": trainable_dtypes,
        "reduce_dtype": "torch.float32",
        "gradient_reduction": "SUM_THEN_DIVIDE_WORLD_SIZE",
        "force_sum_reduction_for_comms": True,
        "fsdp_group_count": fsdp_group_count,
        "cpu_parameter_offload": False,
        "parameter_streaming": False,
        "transition": False,
    }


@dataclasses.dataclass
class _DDPCommState:
    process_group: Any
    world_size: int
    bucket_bytes: list[int] = dataclasses.field(default_factory=list)
    bucket_indices: list[int] = dataclasses.field(default_factory=list)


def ddp_sum_then_divide_hook(state, bucket):
    """Issue async SUM all-reduce and divide once after the collective."""

    # PyTorch 2.10 validates comm-hook annotations by exact runtime identity.
    # With ``from __future__ import annotations``, even a nominal GradBucket
    # annotation is a string and is rejected. Leaving the hook unannotated is
    # the compatible form and does not alter its collective math.

    dist = _torch().distributed
    buffer = bucket.buffer()
    state.bucket_bytes.append(int(buffer.numel() * buffer.element_size()))
    state.bucket_indices.append(int(bucket.index()))
    return (
        dist.all_reduce(
            buffer,
            op=dist.ReduceOp.SUM,
            group=state.process_group,
            async_op=True,
        )
        .get_future()
        .then(lambda future: future.value()[0].mul_(1.0 / state.world_size))
    )


class ReferenceResidentEngine(ReplicatedTrainableAllReduceEngine):
    """PR #21 reference engine with an explicit auditable backend receipt."""

    def _synchronize_trainable_gradients(self) -> dict[str, Any]:
        receipt = super()._synchronize_trainable_gradients()
        # The parent appends the same dictionary object in correctness mode, so
        # adding this field after the call also updates the stored receipt.
        receipt["backend"] = "MANUAL_ALL_REDUCE_COALESCED"
        receipt["overlap_with_backward"] = False
        return receipt


class LocalFastResidentEngine(ReferenceResidentEngine):
    """Reference manual all-reduce with foreach clipping and fused local AdamW."""

    def step(self) -> None:
        torch = _torch()
        receipt = self._synchronize_trainable_gradients()
        trainables = [
            parameter for parameter in self.module.parameters() if parameter.requires_grad
        ]
        clip_norm = torch.nn.utils.clip_grad_norm_(
            trainables,
            self._gradient_clip,
            foreach=True,
        )
        if self._audit_gradients:
            receipt["gradient_clip_input_norm"] = float(clip_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_steps += 1
        self.global_samples += GLOBAL_BATCH
        self._step_calls += 1


class PersistentFlatResidentEngine(LocalFastResidentEngine):
    """Post-backward resident sync using stable flat gradient buffers."""

    def __init__(self, *args: Any, flat_bucket_bytes: int, **kwargs: Any) -> None:
        if flat_bucket_bytes <= 0:
            raise ValueError("flat_bucket_bytes must be positive")
        super().__init__(*args, bucket_bytes=flat_bucket_bytes, **kwargs)
        torch = _torch()
        self._flat_bucket_bytes = int(flat_bucket_bytes)
        named = [
            (name, parameter)
            for name, parameter in self.module.named_parameters()
            if parameter.requires_grad
        ]
        if not named:
            raise RuntimeError("persistent flat reducer has no trainables")
        grouped = self._buckets(
            [parameter for _name, parameter in named],
            self._flat_bucket_bytes,
        )
        self._flat_buckets: list[Any] = []
        self._flat_views: list[tuple[str, Any, Any]] = []
        name_by_id = {id(parameter): name for name, parameter in named}
        for parameters in grouped:
            dtypes = {parameter.dtype for parameter in parameters}
            devices = {parameter.device for parameter in parameters}
            if len(dtypes) != 1 or len(devices) != 1:
                raise AssertionError("persistent flat bucket mixed dtype/device")
            flat = torch.zeros(
                sum(int(parameter.numel()) for parameter in parameters),
                dtype=parameters[0].dtype,
                device=parameters[0].device,
            )
            self._flat_buckets.append(flat)
            offset = 0
            for parameter in parameters:
                count = int(parameter.numel())
                view = flat.narrow(0, offset, count).view_as(parameter)
                self._flat_views.append((name_by_id[id(parameter)], parameter, view))
                offset += count
        self._install_zero_grad_views()

    def _install_zero_grad_views(self) -> None:
        for flat in self._flat_buckets:
            flat.zero_()
        for _name, parameter, view in self._flat_views:
            parameter.grad = view

    def _synchronize_trainable_gradients(self) -> dict[str, Any]:
        torch = _torch()
        dist = torch.distributed
        if not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
            raise RuntimeError("persistent flat reducer requires world16")
        for name, parameter, view in self._flat_views:
            gradient = parameter.grad
            if gradient is None or gradient.data_ptr() != view.data_ptr():
                raise AssertionError(f"persistent gradient view changed: {name}")
        names = tuple(name for name, _parameter, _view in self._flat_views)
        if self._audit_gradients:
            inventories: list[Any] = [None] * WORLD_SIZE
            dist.all_gather_object(inventories, names)
            if any(tuple(value) != names for value in inventories):
                raise AssertionError("persistent gradient inventory differs by rank")
        for flat in self._flat_buckets:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.mul_(1.0 / WORLD_SIZE)
        receipt = {
            "backend": "PERSISTENT_FLAT_ALL_REDUCE",
            "world_size": WORLD_SIZE,
            "reduction": "SUM_THEN_DIVIDE_WORLD_SIZE",
            "overlap_with_backward": False,
            "trainable_gradient_tensors": len(self._flat_views),
            "gradient_bytes": sum(
                int(flat.numel() * flat.element_size()) for flat in self._flat_buckets
            ),
            "coalesced_bucket_count": len(self._flat_buckets),
            "bucket_bytes_limit": self._flat_bucket_bytes,
            "bucket_bytes_observed": [
                int(flat.numel() * flat.element_size()) for flat in self._flat_buckets
            ],
            "persistent": True,
            "scale_kernels_per_update": len(self._flat_buckets),
        }
        if self._audit_gradients:
            self.gradient_sync_receipts.append(receipt)
        return receipt

    def step(self) -> None:
        torch = _torch()
        receipt = self._synchronize_trainable_gradients()
        trainables = [parameter for _name, parameter, _view in self._flat_views]
        clip_norm = torch.nn.utils.clip_grad_norm_(
            trainables,
            self._gradient_clip,
            foreach=True,
        )
        if self._audit_gradients:
            receipt["gradient_clip_input_norm"] = float(clip_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.global_steps += 1
        self.global_samples += GLOBAL_BATCH
        self._step_calls += 1
        self._install_zero_grad_views()

    def zero_grad(self) -> None:
        self._install_zero_grad_views()


@dataclasses.dataclass
class _GradReadyBucket:
    index: int
    flat: Any
    names: tuple[str, ...]
    parameters: tuple[Any, ...]
    ready_parameter_ids: set[int] = dataclasses.field(default_factory=set)
    work: Any | None = None


class GradReadyAsyncResidentEngine(ReferenceResidentEngine):
    """Overlap deterministic FP32 LoRA all-reduce buckets with backward.

    This does not wrap the model in DDP and does not change the reference
    AdamW, gradient clipping, or EMA implementations. Persistent flat gradient
    views remove per-step coalescing allocations. Buckets use reverse parameter
    order (the expected transformer backward-ready order) and are issued by a
    fixed index on a dedicated CUDA stream. Missing or duplicate hooks fail
    before the optimizer can commit.
    """

    def __init__(self, *args: Any, async_bucket_bytes: int, **kwargs: Any) -> None:
        if async_bucket_bytes <= 0:
            raise ValueError("async_bucket_bytes must be positive")
        super().__init__(*args, bucket_bytes=async_bucket_bytes, **kwargs)
        torch = _torch()
        named = [
            (name, parameter)
            for name, parameter in self.module.named_parameters()
            if parameter.requires_grad
        ]
        if not named:
            raise RuntimeError("grad-ready reducer has no trainables")
        if {parameter.dtype for _name, parameter in named} != {torch.float32}:
            raise AssertionError("grad-ready reducer requires FP32 LoRA trainables")

        self._async_bucket_bytes = int(async_bucket_bytes)
        grouped = self._buckets(
            [parameter for _name, parameter in reversed(named)],
            self._async_bucket_bytes,
        )
        name_by_id = {id(parameter): name for name, parameter in named}
        self._buckets_by_index: list[_GradReadyBucket] = []
        self._parameter_bucket: dict[int, _GradReadyBucket] = {}
        self._hook_handles: list[Any] = []
        self._hook_counts: dict[int, int] = {}
        for index, parameters in enumerate(grouped):
            devices = {parameter.device for parameter in parameters}
            dtypes = {parameter.dtype for parameter in parameters}
            if len(devices) != 1 or dtypes != {torch.float32}:
                raise AssertionError("grad-ready bucket mixed dtype/device")
            flat = torch.zeros(
                sum(int(parameter.numel()) for parameter in parameters),
                dtype=torch.float32,
                device=parameters[0].device,
            )
            names: list[str] = []
            offset = 0
            for parameter in parameters:
                count = int(parameter.numel())
                parameter.grad = flat.narrow(0, offset, count).view_as(parameter)
                names.append(name_by_id[id(parameter)])
                offset += count
            bucket = _GradReadyBucket(
                index=index,
                flat=flat,
                names=tuple(names),
                parameters=tuple(parameters),
            )
            self._buckets_by_index.append(bucket)
            for parameter in parameters:
                identity = id(parameter)
                self._parameter_bucket[identity] = bucket
                self._hook_counts[identity] = 0
                self._hook_handles.append(
                    parameter.register_post_accumulate_grad_hook(
                        self._make_grad_ready_hook(identity)
                    )
                )

        self._comm_stream = torch.cuda.Stream(device=named[0][1].device)
        self._active_backward = False
        self._next_bucket_to_issue = 0
        self._issue_order: list[int] = []
        self._issue_times_ns: list[int] = []

    def _make_grad_ready_hook(self, parameter_id: int):
        def hook(parameter: Any) -> None:
            del parameter
            if not self._active_backward:
                raise RuntimeError("grad-ready hook fired outside engine.backward")
            count = self._hook_counts[parameter_id] + 1
            self._hook_counts[parameter_id] = count
            if count != 1:
                raise RuntimeError("grad-ready hook fired more than once in one backward")
            self._parameter_bucket[parameter_id].ready_parameter_ids.add(parameter_id)
            self._drain_ready_buckets()

        return hook

    def _drain_ready_buckets(self) -> None:
        import time

        # A logical B16 update on eight physical ranks contains two B1
        # backwards. The first contribution stays in the persistent views;
        # only the final backward may issue the combined gradient buckets.
        if not self._gradient_accumulation_boundary:
            return
        torch = _torch()
        dist = torch.distributed
        while self._next_bucket_to_issue < len(self._buckets_by_index):
            bucket = self._buckets_by_index[self._next_bucket_to_issue]
            if len(bucket.ready_parameter_ids) != len(bucket.parameters):
                break
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(bucket.flat.device))
            with torch.cuda.stream(self._comm_stream):
                self._comm_stream.wait_event(event)
                bucket.work = dist.all_reduce(
                    bucket.flat,
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                )
            self._issue_order.append(bucket.index)
            self._issue_times_ns.append(time.perf_counter_ns())
            self._next_bucket_to_issue += 1

    def _reset_backward_state(self) -> None:
        self._next_bucket_to_issue = 0
        self._issue_order.clear()
        self._issue_times_ns.clear()
        for parameter_id in self._hook_counts:
            self._hook_counts[parameter_id] = 0
        for bucket in self._buckets_by_index:
            bucket.ready_parameter_ids.clear()
            bucket.work = None

    def backward(self, loss: Any) -> None:
        self._reset_backward_state()
        self._active_backward = True
        try:
            loss.backward()
        finally:
            self._active_backward = False
        expected = (
            list(range(len(self._buckets_by_index)))
            if self._gradient_accumulation_boundary else []
        )
        if self._issue_order != expected:
            missing = [
                name
                for bucket in self._buckets_by_index
                for name, parameter in zip(bucket.names, bucket.parameters, strict=True)
                if self._hook_counts[id(parameter)] != 1
            ]
            raise RuntimeError(
                f"grad-ready collective order incomplete: {self._issue_order}; "
                f"missing={missing[:8]}"
            )

    def _finish_gradient_sync(self) -> dict[str, Any]:
        torch = _torch()
        for bucket in self._buckets_by_index:
            if bucket.work is None:
                raise RuntimeError(f"grad-ready bucket {bucket.index} was not issued")
            bucket.work.wait()
        torch.cuda.current_stream().wait_stream(self._comm_stream)
        for bucket in self._buckets_by_index:
            bucket.flat.mul_(1.0 / WORLD_SIZE)
        receipt = {
            "backend": "GRAD_READY_ASYNC_FLAT_ALL_REDUCE",
            "world_size": WORLD_SIZE,
            "reduction": "SUM_THEN_DIVIDE_WORLD_SIZE",
            "overlap_with_backward": True,
            "trainable_gradient_tensors": len(self._parameter_bucket),
            "gradient_bytes": sum(
                int(bucket.flat.numel() * bucket.flat.element_size())
                for bucket in self._buckets_by_index
            ),
            "coalesced_bucket_count": len(self._buckets_by_index),
            "bucket_bytes_limit": self._async_bucket_bytes,
            "bucket_bytes_observed": [
                int(bucket.flat.numel() * bucket.flat.element_size())
                for bucket in self._buckets_by_index
            ],
            "bucket_issue_order": list(self._issue_order),
            "hook_count_min": min(self._hook_counts.values()),
            "hook_count_max": max(self._hook_counts.values()),
            "persistent": True,
            "dedicated_comm_stream": True,
        }
        if self._audit_gradients:
            self.gradient_sync_receipts.append(receipt)
        return receipt

    def step(self) -> None:
        torch = _torch()
        receipt = self._finish_gradient_sync()
        trainables = [
            parameter
            for _name, parameter in self.module.named_parameters()
            if parameter.requires_grad
        ]
        clip_norm = torch.nn.utils.clip_grad_norm_(trainables, self._gradient_clip)
        if getattr(self, "_trajectory_capture_grad_norm", False):
            self.last_gradient_norm = float(clip_norm.detach().float().cpu())
        if self._audit_gradients:
            receipt["gradient_clip_input_norm"] = float(clip_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.global_steps += 1
        self.global_samples += GLOBAL_BATCH
        self._step_calls += 1
        self.zero_grad()

    def zero_grad(self) -> None:
        for bucket in self._buckets_by_index:
            bucket.flat.zero_()
            offset = 0
            for parameter in bucket.parameters:
                count = int(parameter.numel())
                parameter.grad = bucket.flat.narrow(0, offset, count).view_as(parameter)
                offset += count


class LoRAZero2ResidentEngine(ReplicatedTrainableAllReduceEngine):
    """Replicated model with world16-sharded LoRA gradients and Adam state.

    The full frozen backbone and full FP32 LoRA parameters stay resident for
    every forward/backward.  Only the post-backward trainable gradient and the
    Adam moments are sharded.  Each commit performs one reduce-scatter, a local
    AdamW update, and one all-gather of the updated LoRA values.
    """

    def __init__(
        self,
        module: Any,
        *,
        physical_step: int,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        audit_gradients: bool = False,
        gradient_clip: float = 1.0,
    ) -> None:
        torch = _torch()
        dist = torch.distributed
        if not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
            raise RuntimeError("LoRA-only ZeRO-2 requires world16")
        self.module = module
        self.global_steps = int(physical_step)
        self.global_samples = int(physical_step) * GLOBAL_BATCH
        self.dp_world_size = WORLD_SIZE
        self._gradient_clip = float(gradient_clip)
        self._audit_gradients = bool(audit_gradients)
        self._step_calls = 0
        self._gradient_accumulation_boundary = True
        self.gradient_sync_receipts: list[dict[str, Any]] = []
        self._named_trainables = list(_named_trainables(module).items())
        if not self._named_trainables:
            raise RuntimeError("LoRA-only ZeRO-2 has no trainables")
        if {parameter.dtype for _name, parameter in self._named_trainables} != {
            torch.float32
        }:
            raise AssertionError("LoRA-only ZeRO-2 requires FP32 LoRA parameters")
        self._offsets: list[tuple[str, Any, int, int]] = []
        total = 0
        for name, parameter in self._named_trainables:
            count = int(parameter.numel())
            self._offsets.append((name, parameter, total, total + count))
            total += count
        padded = ((total + WORLD_SIZE - 1) // WORLD_SIZE) * WORLD_SIZE
        self._logical_numel = total
        self._padded_numel = padded
        self._local_numel = padded // WORLD_SIZE
        self._rank = dist.get_rank()
        device = self._named_trainables[0][1].device
        self._full_gradient = torch.zeros(padded, dtype=torch.float32, device=device)
        self._local_gradient = torch.zeros(
            self._local_numel, dtype=torch.float32, device=device
        )
        self._full_parameter = torch.zeros(padded, dtype=torch.float32, device=device)
        for _name, parameter, start, stop in self._offsets:
            self._full_parameter[start:stop].copy_(parameter.detach().reshape(-1))
        local_start = self._rank * self._local_numel
        local_stop = local_start + self._local_numel
        self._local_parameter = torch.nn.Parameter(
            self._full_parameter[local_start:local_stop].clone(), requires_grad=True
        )
        self.optimizer = torch.optim.AdamW(
            [self._local_parameter],
            lr=float(lr),
            betas=tuple(float(value) for value in betas),
            eps=float(eps),
            weight_decay=float(weight_decay),
            fused=True,
        )
        state = self.optimizer.state[self._local_parameter]
        state["step"] = torch.tensor(
            float(physical_step), dtype=torch.float32, device=device
        )
        state["exp_avg"] = torch.zeros_like(self._local_parameter)
        state["exp_avg_sq"] = torch.zeros_like(self._local_parameter)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda _: 1.0
        )
        self.lr_scheduler.last_epoch = int(physical_step)
        self.lr_scheduler._step_count = int(physical_step) + 1
        self.lr_scheduler._last_lr = [
            float(group["lr"]) for group in self.optimizer.param_groups
        ]
        module._distillschedule_lora_zero2_engine = self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def backward(self, loss: Any) -> None:
        loss.backward()

    def _pack_gradients(self) -> tuple[str, ...]:
        names: list[str] = []
        self._full_gradient.zero_()
        for name, parameter, start, stop in self._offsets:
            if parameter.grad is None:
                raise AssertionError(f"LoRA-only ZeRO-2 missing gradient: {name}")
            self._full_gradient[start:stop].copy_(parameter.grad.detach().reshape(-1))
            names.append(name)
        return tuple(names)

    def _synchronize_trainable_gradients(self) -> dict[str, Any]:
        torch = _torch()
        dist = torch.distributed
        names = self._pack_gradients()
        if self._audit_gradients:
            inventories: list[Any] = [None] * WORLD_SIZE
            dist.all_gather_object(inventories, names)
            if any(tuple(value) != names for value in inventories):
                raise AssertionError("LoRA-only ZeRO-2 gradient inventory differs")
        dist.reduce_scatter_tensor(
            self._local_gradient,
            self._full_gradient,
            op=dist.ReduceOp.SUM,
        )
        self._local_gradient.mul_(1.0 / WORLD_SIZE)
        for _name, parameter, _start, _stop in self._offsets:
            parameter.grad = None
        receipt = {
            "backend": "LORA_ZERO2_REDUCE_SCATTER_ALL_GATHER",
            "world_size": WORLD_SIZE,
            "reduction": "SUM_THEN_DIVIDE_WORLD_SIZE",
            "overlap_with_backward": False,
            "trainable_gradient_tensors": len(names),
            "gradient_bytes": self._logical_numel * 4,
            "local_gradient_bytes": self._local_numel * 4,
            "reduce_scatter_count": 1,
            "parameter_all_gather_count": 1,
            "optimizer_state_sharded": True,
        }
        if self._audit_gradients:
            self.gradient_sync_receipts.append(receipt)
        return receipt

    def _clip_local_gradient(self) -> Any:
        torch = _torch()
        dist = torch.distributed
        total_sq = self._local_gradient.double().square().sum()
        dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)
        total_norm = total_sq.sqrt().float()
        coefficient = torch.clamp(
            torch.tensor(
                self._gradient_clip,
                dtype=torch.float32,
                device=self._local_gradient.device,
            )
            / (total_norm + 1.0e-6),
            max=1.0,
        )
        self._local_gradient.mul_(coefficient)
        return total_norm

    def _publish_updated_parameters(self) -> None:
        torch = _torch()
        dist = torch.distributed
        dist.all_gather_into_tensor(self._full_parameter, self._local_parameter.detach())
        with torch.no_grad():
            for _name, parameter, start, stop in self._offsets:
                parameter.copy_(self._full_parameter[start:stop].view_as(parameter))

    def step(self) -> None:
        receipt = self._synchronize_trainable_gradients()
        clip_norm = self._clip_local_gradient()
        self._local_parameter.grad = self._local_gradient
        if self._audit_gradients:
            receipt["gradient_clip_input_norm"] = float(clip_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._publish_updated_parameters()
        self.global_steps += 1
        self.global_samples += GLOBAL_BATCH
        self._step_calls += 1

    def zero_grad(self) -> None:
        for _name, parameter, _start, _stop in self._offsets:
            parameter.grad = None
        self.optimizer.zero_grad(set_to_none=True)

    def set_gradient_accumulation_boundary(self, *, is_boundary: bool) -> None:
        self._gradient_accumulation_boundary = bool(is_boundary)

    @staticmethod
    def is_gradient_accumulation_boundary() -> bool:
        return True

    @staticmethod
    def gradient_accumulation_steps() -> int:
        return 1

    @staticmethod
    def train_micro_batch_size_per_gpu() -> int:
        return 1

    @staticmethod
    def train_batch_size() -> int:
        return GLOBAL_BATCH

    @property
    def step_calls(self) -> int:
        return self._step_calls

    def full_optimizer_state_evidence(self) -> dict[str, dict[str, Any]]:
        torch = _torch()
        dist = torch.distributed
        state = self.optimizer.state[self._local_parameter]
        result: dict[str, dict[str, Any]] = {}
        for key in ("exp_avg", "exp_avg_sq"):
            full = torch.empty_like(self._full_parameter)
            dist.all_gather_into_tensor(full, state[key])
            for name, parameter, start, stop in self._offsets:
                result.setdefault(name, {})[key] = full[start:stop].view_as(parameter)
        return result

    @property
    def local_parameter_shard(self) -> Any:
        return self._local_parameter.detach()


class RolloutFusedLoRAZero2Engine(LoRAZero2ResidentEngine):
    """LoRA-only ZeRO-2 plus a BF16-base fused Generator rollout shadow."""

    def __init__(self, module: Any, **kwargs: Any) -> None:
        shadow = getattr(module, "_distillschedule_rollout_shadow", None)
        if shadow is None:
            raise RuntimeError("rollout-fused arm is missing its shadow model")
        self._rollout_shadow = shadow
        self._shadow_ready = False
        self._shadow_fused = False
        self.shadow_refresh_count = 0
        self.shadow_routed_call_count = 0
        super().__init__(module, **kwargs)
        original_forward = module.forward

        def routed_forward(*args: Any, **kwargs: Any) -> Any:
            torch = _torch()
            if self._shadow_ready and not torch.is_grad_enabled():
                self.shadow_routed_call_count += 1
                return self._rollout_shadow(*args, **kwargs)
            return original_forward(*args, **kwargs)

        module.forward = routed_forward

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def _refresh_rollout_shadow(self) -> None:
        torch = _torch()
        if self._shadow_fused:
            self._rollout_shadow.unfuse_lora()
            self._shadow_fused = False
        source = _named_trainables(self.module)
        shadow_parameters = dict(self._rollout_shadow.named_parameters())
        if any(name not in shadow_parameters for name in source):
            raise AssertionError("rollout shadow LoRA inventory changed")
        with torch.no_grad():
            for name, parameter in source.items():
                shadow_parameters[name].copy_(parameter)
        self._rollout_shadow.fuse_lora(
            lora_scale=1.0,
            safe_fusing=False,
            adapter_names=None,
        )
        self._shadow_fused = True
        self._shadow_ready = True
        self.shadow_refresh_count += 1

    def step(self) -> None:
        super().step()
        self._refresh_rollout_shadow()


class ShardedTrainableEMA:
    """EMA shadow over the local LoRA-only ZeRO shard."""

    def __init__(
        self,
        module: Any,
        *,
        optimization_step: int = 10,
        decay: float = 0.999,
        update_after_step: int = 0,
    ) -> None:
        engine = getattr(module, "_distillschedule_lora_zero2_engine", None)
        if not isinstance(engine, LoRAZero2ResidentEngine):
            raise RuntimeError("sharded EMA requires LoRA-only ZeRO-2 engine")
        self._engine = engine
        self.optimization_step = int(optimization_step)
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.min_decay = 0.0
        self.cur_decay_value: float | None = None
        self.last_transfer_bytes = 0
        self._shadow = engine.local_parameter_shard.float().clone()

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = (1 + step) / (10 + step)
        return max(self.min_decay, min(value, self.decay))

    def step(self) -> float:
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        self.last_transfer_bytes = 0
        self._shadow.lerp_(self._engine.local_parameter_shard.float(), 1.0 - decay)
        return decay

    def checksum(self) -> dict[str, Any]:
        torch = _torch()
        dist = torch.distributed
        values = torch.stack(
            (self._shadow.double().sum(), self._shadow.double().square().sum())
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return {
            "tracked_tensor_count": 1,
            "local_numel": int(self._shadow.numel()),
            "global_numel": int(self._engine._logical_numel),
            "global_sum": float(values[0].cpu()),
            "global_norm": float(values[1].sqrt().cpu()),
            "optimization_step": self.optimization_step,
            "decay": self.cur_decay_value,
            "communication_per_update": False,
            "device": "cuda",
            "implementation": "sharded_foreach_equivalent",
            "state_representation": "WORLD16_LOCAL_FLAT_SHARD_FP32",
        }


class ResidentDDPEngine:
    """Full GPU residency with DDP used only for trainable gradients."""

    def __init__(
        self,
        module: Any,
        optimizer: Any,
        scheduler: Any,
        *,
        physical_step: int,
        gradient_clip: float = 1.0,
        bucket_cap_mb: int = 128,
        audit_gradients: bool = False,
    ) -> None:
        if bucket_cap_mb <= 0:
            raise ValueError("DDP bucket_cap_mb must be positive")
        torch = _torch()
        dist = torch.distributed
        if not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
            raise RuntimeError("resident DDP requires world16")
        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.global_steps = int(physical_step)
        self.global_samples = int(physical_step) * GLOBAL_BATCH
        self.dp_world_size = WORLD_SIZE
        self._gradient_clip = float(gradient_clip)
        self._audit_gradients = bool(audit_gradients)
        self._step_calls = 0
        self._gradient_accumulation_boundary = True
        self.gradient_sync_receipts: list[dict[str, Any]] = []
        self._comm_state = _DDPCommState(dist.group.WORLD, WORLD_SIZE)
        self._ddp = torch.nn.parallel.DistributedDataParallel(
            module,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            broadcast_buffers=False,
            init_sync=True,
            bucket_cap_mb=int(bucket_cap_mb),
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        self._ddp.register_comm_hook(self._comm_state, ddp_sum_then_divide_hook)
        self._bucket_cap_mb = int(bucket_cap_mb)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._ddp(*args, **kwargs)

    def backward(self, loss: Any) -> None:
        loss.backward()

    def step(self) -> None:
        torch = _torch()
        trainables = [
            parameter for parameter in self.module.parameters() if parameter.requires_grad
        ]
        if not trainables:
            raise RuntimeError("resident DDP has no trainable parameters")
        clip_norm = torch.nn.utils.clip_grad_norm_(trainables, self._gradient_clip)
        receipt = {
            "backend": "DDP_GRADIENT_BUCKET_VIEW",
            "world_size": WORLD_SIZE,
            "reduction": "SUM_THEN_DIVIDE_WORLD_SIZE",
            "overlap_with_backward": True,
            "bucket_cap_mb": self._bucket_cap_mb,
            "bucket_count_observed": len(self._comm_state.bucket_bytes),
            "bucket_bytes_observed": list(self._comm_state.bucket_bytes),
            "bucket_indices_observed": list(self._comm_state.bucket_indices),
        }
        if self._audit_gradients:
            receipt["gradient_clip_input_norm"] = float(clip_norm)
            self.gradient_sync_receipts.append(receipt)
        self._comm_state.bucket_bytes.clear()
        self._comm_state.bucket_indices.clear()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_steps += 1
        self.global_samples += GLOBAL_BATCH
        self._step_calls += 1

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def set_gradient_accumulation_boundary(self, *, is_boundary: bool) -> None:
        self._gradient_accumulation_boundary = bool(is_boundary)

    @staticmethod
    def is_gradient_accumulation_boundary() -> bool:
        return True

    @staticmethod
    def gradient_accumulation_steps() -> int:
        return 1

    @staticmethod
    def train_micro_batch_size_per_gpu() -> int:
        return 1

    @staticmethod
    def train_batch_size() -> int:
        return GLOBAL_BATCH

    @property
    def step_calls(self) -> int:
        return self._step_calls


def _materialize_resident_adam_state(
    *,
    optimizer: Any,
    scheduler: Any,
    physical_step: int,
    fused: bool,
) -> dict[str, Any]:
    """Materialize an identical logical Adam state for default/fused kernels."""

    torch = _torch()
    state = materialize_adam_state(
        optimizer=optimizer,
        scheduler=scheduler,
        physical_step=physical_step,
    )
    if fused:
        # PyTorch fused Adam requires the scalar step on the parameter device.
        # Its numeric value and the two FP32 moment tensors remain identical.
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                optimizer.state[parameter]["step"] = torch.tensor(
                    float(physical_step),
                    dtype=torch.float32,
                    device=parameter.device,
                )
    return {
        **state,
        "step_tensor_device": "parameter_device" if fused else "cpu",
    }


def build_fair_engine(
    *,
    module: Any,
    placement: str,
    physical_step: int,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    audit_gradients: bool = False,
    resident_ddp_bucket_mb: int = 128,
) -> tuple[Any, dict[str, Any]]:
    """Build one engine without changing optimizer hyperparameters or state."""

    if placement not in FAIR_PLACEMENTS:
        raise ValueError(f"unsupported fair placement: {placement}")
    if placement == FSDP_PLACEMENT:
        engine, metadata = build_fresh_engine(
            module=module,
            placement=placement,
            physical_step=physical_step,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            audit_gradients=audit_gradients,
        )
        metadata = dict(metadata)
        metadata.update(
            {
                "optimizer_state_representation": "LOCAL_DTENSOR_SHARD_FP32",
                "gradient_sync": "FSDP2_SUM_THEN_DIVIDE_REDUCE_SCATTER",
                "optimizer_implementation": "PYTORCH_DEFAULT",
            }
        )
        return engine, metadata

    torch = _torch()
    trainables = list(_named_trainables(module).values())
    if not trainables:
        raise ValueError("Wan1.3B resident engine has no trainables")
    if placement in {LORA_ZERO2_RESIDENT, ROLLOUT_FUSED_RESIDENT}:
        engine_type = (
            RolloutFusedLoRAZero2Engine
            if placement == ROLLOUT_FUSED_RESIDENT
            and hasattr(module, "_distillschedule_rollout_shadow")
            else LoRAZero2ResidentEngine
        )
        engine = engine_type(
            module,
            physical_step=physical_step,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            audit_gradients=audit_gradients,
        )
        return engine, {
            "optimizer_step": int(physical_step),
            "scheduler_step": int(physical_step),
            "state_materialized_before_timing": True,
            "trainable_tensor_count": len(trainables),
            "local_moment_numel_each": int(engine._local_numel),
            "step_tensor_device": "parameter_device",
            "placement": placement,
            "gradient_sync": "WORLD16_LORA_ZERO2_REDUCE_SCATTER_ALL_GATHER",
            "optimizer": "AdamW",
            "optimizer_implementation": "PYTORCH_FUSED_LOCAL_SHARD",
            "gradient_clip_implementation": "GLOBAL_L2_FROM_LOCAL_SHARDS",
            "persistent_flat_bucket_mb": None,
            "optimizer_state_representation": "WORLD16_LOCAL_FLAT_SHARD_FP32",
            "optimizer_state_device": "cuda",
            "optimizer_state_offload": False,
            "resident_ddp_bucket_mb": None,
        }
    flat_bucket_mb = {
        ASYNC_128_RESIDENT: 128,
        ASYNC_128_TREE_RESIDENT: 128,
        ASYNC_128_TREE_CHUNK2048_RESIDENT: 128,
        ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT: 128,
        ASYNC_256_RESIDENT: 256,
        FLAT_256_RESIDENT: 256,
        FLAT_512_RESIDENT: 512,
    }.get(placement)
    fused = placement in {
        LOCAL_FAST_RESIDENT,
        FLAT_256_RESIDENT,
        FLAT_512_RESIDENT,
        FAST_RESIDENT,
    }
    optimizer_kwargs: dict[str, Any] = {}
    if fused:
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        trainables,
        lr=float(lr),
        betas=tuple(float(value) for value in betas),
        eps=float(eps),
        weight_decay=float(weight_decay),
        **optimizer_kwargs,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    state = _materialize_resident_adam_state(
        optimizer=optimizer,
        scheduler=scheduler,
        physical_step=physical_step,
        fused=fused,
    )
    if placement in {
        REFERENCE_RESIDENT,
        CLEAN_RESIDENT,
        ASYNC_128_RESIDENT,
        ASYNC_128_TREE_RESIDENT,
        ASYNC_128_TREE_CHUNK2048_RESIDENT,
        ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
        ASYNC_256_RESIDENT,
        LOCAL_FAST_RESIDENT,
        FLAT_256_RESIDENT,
        FLAT_512_RESIDENT,
    }:
        if placement in {
            ASYNC_128_RESIDENT,
            ASYNC_128_TREE_RESIDENT,
            ASYNC_128_TREE_CHUNK2048_RESIDENT,
            ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
            ASYNC_256_RESIDENT,
        }:
            engine = GradReadyAsyncResidentEngine(
                module,
                optimizer,
                scheduler,
                physical_step=physical_step,
                async_bucket_bytes=flat_bucket_mb * 1024 * 1024,
                audit_gradients=audit_gradients,
            )
            gradient_sync = "WORLD16_GRAD_READY_ASYNC_SUM_THEN_DIVIDE_ALL_REDUCE"
        elif flat_bucket_mb is not None:
            engine: Any = PersistentFlatResidentEngine(
                module,
                optimizer,
                scheduler,
                physical_step=physical_step,
                flat_bucket_bytes=flat_bucket_mb * 1024 * 1024,
                audit_gradients=audit_gradients,
            )
            gradient_sync = "WORLD16_PERSISTENT_FLAT_SUM_THEN_DIVIDE_ALL_REDUCE"
        else:
            engine_type = (
                LocalFastResidentEngine
                if placement == LOCAL_FAST_RESIDENT
                else ReferenceResidentEngine
            )
            engine = engine_type(
                module,
                optimizer,
                scheduler,
                physical_step=physical_step,
                audit_gradients=audit_gradients,
            )
            gradient_sync = "WORLD16_MANUAL_SUM_THEN_DIVIDE_ALL_REDUCE"
    else:
        engine = ResidentDDPEngine(
            module,
            optimizer,
            scheduler,
            physical_step=physical_step,
            bucket_cap_mb=resident_ddp_bucket_mb,
            audit_gradients=audit_gradients,
        )
        gradient_sync = "WORLD16_DDP_SUM_THEN_DIVIDE_ALL_REDUCE_OVERLAPPED"
    return engine, {
        **state,
        "placement": placement,
        "gradient_sync": gradient_sync,
        "optimizer": "AdamW",
        "optimizer_implementation": (
            "PYTORCH_FUSED" if fused else "PYTORCH_DEFAULT"
        ),
        "gradient_clip_implementation": (
            "PYTORCH_FOREACH_TRUE"
            if placement
            in {LOCAL_FAST_RESIDENT, FLAT_256_RESIDENT, FLAT_512_RESIDENT}
            else "PYTORCH_DEFAULT"
        ),
        "persistent_flat_bucket_mb": (
            flat_bucket_mb
            if placement
            not in {
                ASYNC_128_RESIDENT,
                ASYNC_128_TREE_RESIDENT,
                ASYNC_128_TREE_CHUNK2048_RESIDENT,
                ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
                ASYNC_256_RESIDENT,
            }
            else None
        ),
        "async_gradient_bucket_mb": (
            flat_bucket_mb
            if placement
            in {
                ASYNC_128_RESIDENT,
                ASYNC_128_TREE_RESIDENT,
                ASYNC_128_TREE_CHUNK2048_RESIDENT,
                ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
                ASYNC_256_RESIDENT,
            }
            else None
        ),
        "optimizer_state_representation": "FULL_REPLICA_FP32",
        "optimizer_state_device": "cuda",
        "optimizer_state_offload": False,
        "resident_ddp_bucket_mb": (
            int(resident_ddp_bucket_mb)
            if placement in {DDP_RESIDENT, FAST_RESIDENT}
            else None
        ),
    }


class FairTrainableEMA(FreshTrainableEMA):
    """Reference EMA or a foreach-lerp Resident optimization candidate."""

    def __init__(self, *args: Any, implementation: str = "reference", **kwargs: Any):
        if implementation not in {"reference", "foreach"}:
            raise ValueError("EMA implementation must be reference or foreach")
        self.implementation = implementation
        super().__init__(*args, **kwargs)

    def step(self) -> float:
        if self.implementation == "reference":
            return super().step()
        torch = _torch()
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        self.last_transfer_bytes = 0
        sources = [
            (
                parameter.detach().to_local()
                if hasattr(parameter, "to_local")
                else parameter.detach()
            ).float()
            for parameter in self._parameters
        ]
        if decay == 0.0:
            for shadow, source in zip(self._shadows, sources, strict=True):
                shadow.copy_(source)
        else:
            torch._foreach_lerp_(self._shadows, sources, 1.0 - float(decay))
        return decay

    def checksum(self) -> dict[str, Any]:
        value = super().checksum()
        value["implementation"] = self.implementation
        value["state_representation"] = "FULL_REPLICA_FP32"
        return value


class LoRAComputeDtypeAudit:
    """Record real LoRA input/base/A/B dtypes in blocks 0, 15, and 29."""

    def __init__(
        self,
        module: Any,
        *,
        block_indices: tuple[int, ...] = (0, 15, 29),
    ) -> None:
        self.records: list[dict[str, Any]] = []
        self._handles: list[Any] = []
        self._seen: set[tuple[str, bool]] = set()
        targets: dict[int, tuple[str, Any]] = {}
        for name, child in module.named_modules():
            for block_index in block_indices:
                if block_index in targets:
                    continue
                if (
                    f"blocks.{block_index}." in name
                    and hasattr(child, "lora_A")
                    and hasattr(child, "lora_B")
                    and hasattr(child, "base_layer")
                ):
                    targets[block_index] = (name, child)
        if set(targets) != set(block_indices):
            raise RuntimeError(
                "could not locate audited LoRA layers: "
                f"found={sorted(targets)}, expected={list(block_indices)}"
            )

        def make_hook(name: str):
            def hook(child: Any, inputs: tuple[Any, ...]) -> None:
                torch = _torch()
                key = (name, bool(torch.is_grad_enabled()))
                if key in self._seen:
                    return
                self._seen.add(key)
                tensor = next(
                    (value for value in inputs if torch.is_tensor(value)),
                    None,
                )
                adapter = next(iter(child.active_adapters))
                self.records.append(
                    {
                        "module": name,
                        "grad_enabled": bool(torch.is_grad_enabled()),
                        "input_dtype": str(tensor.dtype) if tensor is not None else None,
                        "base_weight_dtype": str(child.get_base_layer().weight.dtype),
                        "lora_a_weight_dtype": str(child.lora_A[adapter].weight.dtype),
                        "lora_b_weight_dtype": str(child.lora_B[adapter].weight.dtype),
                    }
                )

            return hook

        for _block, (name, child) in sorted(targets.items()):
            self._handles.append(child.register_forward_pre_hook(make_hook(name)))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def fair_model_trainable_metrics(module: Any, *, placement: str) -> dict[str, Any]:
    mapped = "resident" if placement in RESIDENT_PLACEMENTS else placement
    return model_trainable_metrics(module, placement=mapped)


__all__ = [
    "CLEAN_RESIDENT",
    "ASYNC_128_RESIDENT",
    "ASYNC_128_TREE_RESIDENT",
    "ASYNC_128_TREE_CHUNK2048_RESIDENT",
    "ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT",
    "ASYNC_256_RESIDENT",
    "DDP_RESIDENT",
    "FLAT_256_RESIDENT",
    "FLAT_512_RESIDENT",
    "FAIR_PLACEMENTS",
    "FAST_RESIDENT",
    "FSDP_PLACEMENT",
    "FairTrainableEMA",
    "LoRAComputeDtypeAudit",
    "LOCAL_FAST_RESIDENT",
    "LORA_ZERO2_RESIDENT",
    "LocalFastResidentEngine",
    "LoRAZero2ResidentEngine",
    "ROLLOUT_FUSED_RESIDENT",
    "REFERENCE_RESIDENT",
    "PersistentFlatResidentEngine",
    "ReferenceResidentEngine",
    "RESIDENT_PLACEMENTS",
    "ResidentDDPEngine",
    "RolloutFusedLoRAZero2Engine",
    "ShardedTrainableEMA",
    "apply_wan13b_fair_fsdp2_policy",
    "build_fair_engine",
    "configure_modulation_chunk_tokens",
    "install_generator_nograd_block_compile",
    "ddp_sum_then_divide_hook",
    "fair_model_trainable_metrics",
]
