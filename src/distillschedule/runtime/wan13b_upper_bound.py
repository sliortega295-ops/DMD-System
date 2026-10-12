"""Real Wan2.1-1.3B resident-vs-FSDP2 upper-bound experiment primitives.

This module is deliberately separate from the frozen Helios-14B baseline.  It
keeps the world16/B16 No-GAN Full5 update contract, but replaces both physical
backbones with the real 30-block Wan2.1-T2V-1.3B model.  The only candidate
variable is parameter placement:

* ``resident`` keeps both frozen backbones and their LoRA state on every GPU
  and mean-all-reduces only trainable gradients;
* ``world16-fsdp`` shards all 30 blocks plus the root with composable FSDP2.

No parameter, activation, residual, optimizer, or EMA state is offloaded.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


WORLD_SIZE = 16
WAN13B_BLOCK_COUNT = 30
GLOBAL_BATCH = 16
LORA_RANK = 256
WAN13B_PYRAMID_PATCH_SEED = 20260819
PLACEMENTS = ("resident", "world16-fsdp")


def _torch() -> Any:
    import torch

    return torch


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _named_trainables(module: Any) -> dict[str, Any]:
    return {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


@dataclasses.dataclass(frozen=True)
class Wan13BBootstrap:
    root: Path
    manifest: Mapping[str, Any]
    manifest_hash: str

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        expected_hash: str | None = None,
        verify_payloads: bool = True,
        expected_lora_rank: int = LORA_RANK,
    ) -> "Wan13BBootstrap":
        root_path = Path(root).resolve()
        path = root_path / "manifest.json"
        value = json.loads(path.read_text())
        if value.get("schema_version") != "distillschedule.wan13b_bootstrap.v1":
            raise ValueError("unsupported Wan1.3B bootstrap schema")
        recorded = str(value.get("manifest_hash", ""))
        unsigned = dict(value)
        unsigned.pop("manifest_hash", None)
        actual = hashlib.sha256(_canonical(unsigned)).hexdigest()
        if recorded != actual:
            raise ValueError("Wan1.3B bootstrap manifest hash changed")
        if expected_hash is not None and actual != expected_hash:
            raise ValueError("Wan1.3B bootstrap identity differs from launcher")
        if int(value.get("block_count", -1)) != WAN13B_BLOCK_COUNT:
            raise ValueError("bootstrap is not the audited 30-block Wan1.3B model")
        if type(expected_lora_rank) is not int or not 1 <= expected_lora_rank <= 256:
            raise ValueError("unsupported explicit bootstrap LoRA rank")
        if int(value.get("lora_rank", -1)) != expected_lora_rank:
            raise ValueError("bootstrap LoRA rank changed")
        for component in ("generator", "critic"):
            spec = value["components"][component]
            payload = root_path / str(spec["trainable_payload"])
            if not payload.is_file():
                raise FileNotFoundError(payload)
            if verify_payloads and file_sha256(payload) != spec["sha256"]:
                raise ValueError(f"{component} trainable payload hash changed")
        return cls(root_path, value, actual)

    def component(self, name: str) -> Mapping[str, Any]:
        if name not in {"generator", "critic"}:
            raise KeyError(name)
        return self.manifest["components"][name]

    def restore_trainables(self, module: Any, *, component: str) -> dict[str, Any]:
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError("Wan1.3B bootstrap restore requires safetensors") from error
        spec = self.component(component)
        state = load_file(
            str(self.root / str(spec["trainable_payload"])), device="cpu"
        )
        trainables = _named_trainables(module)
        if tuple(trainables) != tuple(spec["parameter_names"]):
            raise ValueError(f"{component} trainable inventory changed")
        if set(state) != set(trainables):
            raise ValueError(f"{component} bootstrap tensors changed")
        for name, parameter in trainables.items():
            value = state[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"{component}/{name} shape changed")
            if str(value.dtype) != "torch.float32":
                raise ValueError(f"{component}/{name} bootstrap must be FP32")
            parameter.data.copy_(value)
        return {
            "component": component,
            "payload_sha256": str(spec["sha256"]),
            "trainable_tensor_count": len(trainables),
            "trainable_numel": sum(int(value.numel()) for value in trainables.values()),
            "optimizer_step": int(spec["optimizer_step"]),
            "scheduler_step": int(spec["scheduler_step"]),
        }


def materialize_adam_state(
    *, optimizer: Any, scheduler: Any, physical_step: int
) -> dict[str, Any]:
    """Create zero Adam moments before timing for Tensor or DTensor params."""

    torch = _torch()
    optimizer.state.clear()
    tensors = 0
    local_numel = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            state["step"] = torch.tensor(
                float(physical_step), dtype=torch.float32, device="cpu"
            )
            state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
            local = (
                parameter.to_local()
                if hasattr(parameter, "to_local")
                else parameter
            )
            tensors += 1
            local_numel += int(local.numel())
    scheduler.last_epoch = int(physical_step)
    scheduler._step_count = int(physical_step) + 1
    scheduler._last_lr = [float(group["lr"]) for group in optimizer.param_groups]
    return {
        "optimizer_step": int(physical_step),
        "scheduler_step": int(scheduler.last_epoch),
        "state_materialized_before_timing": True,
        "trainable_tensor_count": tensors,
        "local_moment_numel_each": local_numel,
    }


class ReplicatedTrainableAllReduceEngine:
    """DeepSpeed-shaped engine for a fully resident frozen backbone.

    Only present trainable gradients are synchronized.  The all-reduce is a
    world16 SUM followed by division by 16, exactly preserving global-B16 mean
    gradient semantics without wrapping the frozen base in DDP.
    """

    def __init__(
        self,
        module: Any,
        optimizer: Any,
        scheduler: Any,
        *,
        physical_step: int,
        gradient_clip: float = 1.0,
        bucket_bytes: int = 128 * 1024 * 1024,
        audit_gradients: bool = False,
    ) -> None:
        if bucket_bytes <= 0:
            raise ValueError("bucket_bytes must be positive")
        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.global_steps = int(physical_step)
        self.global_samples = int(physical_step) * GLOBAL_BATCH
        self.dp_world_size = WORLD_SIZE
        self._gradient_clip = float(gradient_clip)
        self._bucket_bytes = int(bucket_bytes)
        self._audit_gradients = bool(audit_gradients)
        self._step_calls = 0
        self._gradient_accumulation_boundary = True
        self.gradient_sync_receipts: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def backward(self, loss: Any) -> None:
        loss.backward()

    @staticmethod
    def _buckets(values: Iterable[Any], limit: int) -> list[list[Any]]:
        result: list[list[Any]] = []
        current: list[Any] = []
        current_bytes = 0
        for value in values:
            size = int(value.numel() * value.element_size())
            if current and current_bytes + size > limit:
                result.append(current)
                current = []
                current_bytes = 0
            current.append(value)
            current_bytes += size
        if current:
            result.append(current)
        return result

    def _synchronize_trainable_gradients(self) -> dict[str, Any]:
        torch = _torch()
        dist = torch.distributed
        if not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
            raise RuntimeError("resident gradient synchronization requires world16")
        named = [
            (name, parameter.grad)
            for name, parameter in self.module.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not named:
            raise RuntimeError("resident engine has no active trainable gradients")
        names = tuple(name for name, _ in named)
        if self._audit_gradients:
            inventories: list[Any] = [None] * WORLD_SIZE
            dist.all_gather_object(inventories, names)
            if any(tuple(value) != names for value in inventories):
                raise AssertionError("trainable gradient inventory differs by rank")
        gradients = [value for _, value in named]
        buckets = self._buckets(gradients, self._bucket_bytes)
        for bucket in buckets:
            # PyTorch/NCCL coalesces the list without a persistent flattened
            # model-sized buffer.  This is the only collective in the resident
            # control and is intentionally limited to trainable LoRA gradients.
            dist.all_reduce_coalesced(bucket, op=dist.ReduceOp.SUM)
            for gradient in bucket:
                gradient.mul_(1.0 / WORLD_SIZE)
        receipt = {
            "world_size": WORLD_SIZE,
            "reduction": "SUM_THEN_DIVIDE_WORLD_SIZE",
            "trainable_gradient_tensors": len(gradients),
            "gradient_bytes": sum(
                int(value.numel() * value.element_size()) for value in gradients
            ),
            "coalesced_bucket_count": len(buckets),
            "bucket_bytes_limit": self._bucket_bytes,
        }
        if self._audit_gradients:
            self.gradient_sync_receipts.append(receipt)
        return receipt

    def step(self) -> None:
        torch = _torch()
        receipt = self._synchronize_trainable_gradients()
        trainables = [
            parameter for parameter in self.module.parameters() if parameter.requires_grad
        ]
        clip_norm = torch.nn.utils.clip_grad_norm_(trainables, self._gradient_clip)
        if self._audit_gradients:
            receipt["gradient_clip_input_norm"] = float(clip_norm)
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


def build_fresh_engine(
    *,
    module: Any,
    placement: str,
    physical_step: int,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    audit_gradients: bool = False,
) -> tuple[Any, dict[str, Any]]:
    if placement not in PLACEMENTS:
        raise ValueError(f"unsupported Wan1.3B placement: {placement}")
    torch = _torch()
    trainables = list(_named_trainables(module).values())
    if not trainables:
        raise ValueError("Wan1.3B engine has no trainables")
    optimizer = torch.optim.AdamW(
        trainables,
        lr=float(lr),
        betas=tuple(float(value) for value in betas),
        eps=float(eps),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    state = materialize_adam_state(
        optimizer=optimizer, scheduler=scheduler, physical_step=physical_step
    )
    if placement == "resident":
        engine = ReplicatedTrainableAllReduceEngine(
            module,
            optimizer,
            scheduler,
            physical_step=physical_step,
            audit_gradients=audit_gradients,
        )
    else:
        from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter

        engine = FSDP2EngineAdapter(
            module,
            optimizer,
            scheduler,
            physical_step=physical_step,
            global_batch=GLOBAL_BATCH,
            gradient_clip=1.0,
        )
    return engine, {
        **state,
        "placement": placement,
        "gradient_sync": (
            "WORLD16_TRAINABLE_MEAN_ALL_REDUCE"
            if placement == "resident"
            else "FSDP2_REDUCE_SCATTER"
        ),
        "optimizer": "AdamW",
        "optimizer_state_device": "cuda",
        "optimizer_state_offload": False,
    }


class FreshTrainableEMA:
    """GPU-resident EMA initialized from the immutable Generator LoRA state."""

    def __init__(
        self,
        module: Any,
        *,
        optimization_step: int = 10,
        decay: float = 0.999,
        update_after_step: int = 0,
    ) -> None:
        torch = _torch()
        self.optimization_step = int(optimization_step)
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.min_decay = 0.0
        self.cur_decay_value: float | None = None
        self.last_transfer_bytes = 0
        self._parameters: list[Any] = []
        self._shadows: list[Any] = []
        for parameter in _named_trainables(module).values():
            local = (
                parameter.detach().to_local()
                if hasattr(parameter, "to_local")
                else parameter.detach()
            )
            self._parameters.append(parameter)
            self._shadows.append(local.float().clone())
        if not self._parameters:
            raise ValueError("EMA has no trainable Generator parameters")
        if any(value.device.type != "cuda" for value in self._shadows):
            raise AssertionError("Wan1.3B upper-bound EMA must stay on GPU")

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = (1 + step) / (10 + step)
        return max(self.min_decay, min(value, self.decay))

    def step(self) -> float:
        from distillschedule.runtime.fsdp2_static import ema_update_reference

        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        self.last_transfer_bytes = 0
        for shadow, parameter in zip(
            self._shadows, self._parameters, strict=True
        ):
            local = (
                parameter.detach().to_local()
                if hasattr(parameter, "to_local")
                else parameter.detach()
            )
            ema_update_reference(shadow, local.float(), decay)
        return decay

    def checksum(self) -> dict[str, Any]:
        torch = _torch()
        total_sum = torch.zeros((), dtype=torch.float64, device="cuda")
        total_sq = torch.zeros((), dtype=torch.float64, device="cuda")
        local_numel = 0
        for value in self._shadows:
            double = value.double()
            total_sum += double.sum()
            total_sq += double.square().sum()
            local_numel += int(value.numel())
        return {
            "tracked_tensor_count": len(self._shadows),
            "local_numel": local_numel,
            "local_sum": float(total_sum.cpu()),
            "local_norm": float(total_sq.sqrt().cpu()),
            "optimization_step": self.optimization_step,
            "decay": self.cur_decay_value,
            "communication_per_update": False,
            "device": "cuda",
        }


def model_trainable_metrics(module: Any, *, placement: str) -> dict[str, Any]:
    """Return topology-aware, globally comparable trainable-state moments."""

    if placement not in PLACEMENTS:
        raise ValueError(placement)
    torch = _torch()
    dist = torch.distributed
    device = torch.device("cuda", torch.cuda.current_device())
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    # FSDP contributes each local shard once.  Resident replicas contribute
    # rank0 only, preventing a spurious factor-of-16 in the comparison.
    contribute = placement == "world16-fsdp" or dist.get_rank() == 0
    local_numel = 0
    if contribute:
        for parameter in _named_trainables(module).values():
            local = parameter.detach().to_local() if hasattr(parameter, "to_local") else parameter.detach()
            value = local.double()
            sums[0] += value.sum()
            sums[1] += value.square().sum()
            sums[2] += value.abs().sum()
            sums[3] = torch.maximum(sums[3], value.abs().max())
            local_numel += int(value.numel())
    count = torch.tensor(local_numel, dtype=torch.int64, device=device)
    dist.all_reduce(sums[:3], op=dist.ReduceOp.SUM)
    dist.all_reduce(sums[3], op=dist.ReduceOp.MAX)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return {
        "global_numel": int(count.cpu()),
        "sum": float(sums[0].cpu()),
        "l2": float(sums[1].sqrt().cpu()),
        "l1": float(sums[2].cpu()),
        "max_abs": float(sums[3].cpu()),
    }


__all__ = [
    "FreshTrainableEMA",
    "GLOBAL_BATCH",
    "LORA_RANK",
    "PLACEMENTS",
    "ReplicatedTrainableAllReduceEngine",
    "WAN13B_BLOCK_COUNT",
    "WAN13B_PYRAMID_PATCH_SEED",
    "WORLD_SIZE",
    "Wan13BBootstrap",
    "build_fresh_engine",
    "file_sha256",
    "materialize_adam_state",
    "model_trainable_metrics",
]
