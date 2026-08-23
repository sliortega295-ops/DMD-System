"""Fair Wan1.3B placement policies and a faster full-resident candidate.

This module is layered on the audited PR #21 workload. It does not change the
B1/global-B16 trace, model calls, G1/C5/EMA5 update inventory, loss, AdamW
hyperparameters, checkpointing, or no-offload contract.

The fair comparison fixes the previous precision confound by keeping frozen
base parameters BF16 and LoRA trainables FP32 in both Resident and FSDP2.
"""
from __future__ import annotations

import dataclasses
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
DDP_RESIDENT = "resident-ddp"
FAST_RESIDENT = "resident-fast"
FSDP_PLACEMENT = "world16-fsdp"
RESIDENT_PLACEMENTS = (
    REFERENCE_RESIDENT,
    CLEAN_RESIDENT,
    DDP_RESIDENT,
    FAST_RESIDENT,
)
FAIR_PLACEMENTS = (*RESIDENT_PLACEMENTS, FSDP_PLACEMENT)


def _torch() -> Any:
    import torch

    return torch


def _named_trainables(module: Any) -> dict[str, Any]:
    return {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


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
    fused = placement == FAST_RESIDENT
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
    if placement in {REFERENCE_RESIDENT, CLEAN_RESIDENT}:
        engine: Any = ReferenceResidentEngine(
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
    "DDP_RESIDENT",
    "FAIR_PLACEMENTS",
    "FAST_RESIDENT",
    "FSDP_PLACEMENT",
    "FairTrainableEMA",
    "LoRAComputeDtypeAudit",
    "REFERENCE_RESIDENT",
    "ReferenceResidentEngine",
    "RESIDENT_PLACEMENTS",
    "ResidentDDPEngine",
    "apply_wan13b_fair_fsdp2_policy",
    "build_fair_engine",
    "ddp_sum_then_divide_hook",
    "fair_model_trainable_metrics",
]
