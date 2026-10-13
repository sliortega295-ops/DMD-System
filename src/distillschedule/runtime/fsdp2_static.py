"""Static world16 FSDP2 baseline primitives.

These helpers intentionally implement a baseline, not a scheduling policy: both
Helios 14B backbones remain sharded over the same world16 for the whole Full5.
There is no model swap, role transition, CPU parameter streaming, or parameter
offload in the timed window.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclasses.dataclass(frozen=True)
class StaticFSDP2Full5Contract:
    variant: str
    logical_steps: tuple[int, ...]
    global_batch: int
    generator_updates: int
    critic_updates: int
    ema_update_steps: tuple[int, ...]
    initial_versions: Mapping[str, int]
    final_versions: Mapping[str, int]
    gan_enabled: bool

    def __post_init__(self) -> None:
        if self.variant not in {"no-gan", "gan"}:
            raise ValueError(f"unsupported static FSDP2 variant: {self.variant}")
        if len(self.logical_steps) != 5 or self.global_batch != 16:
            raise ValueError("static FSDP2 baseline is one world16/B16 Full5")
        if self.generator_updates != 1 or self.critic_updates != 5:
            raise ValueError("Full5 update inventory must remain G1/C5")


NO_GAN_CONTRACT = StaticFSDP2Full5Contract(
    variant="no-gan",
    logical_steps=tuple(range(10, 15)),
    global_batch=16,
    generator_updates=1,
    critic_updates=5,
    ema_update_steps=tuple(range(10, 15)),
    initial_versions={"generator": 2, "critic": 10, "ema": 10},
    final_versions={"generator": 3, "critic": 15, "ema": 15},
    gan_enabled=False,
)

GAN_CONTRACT = StaticFSDP2Full5Contract(
    variant="gan",
    logical_steps=tuple(range(1000, 1005)),
    global_batch=16,
    generator_updates=1,
    critic_updates=5,
    ema_update_steps=(1000,),
    initial_versions={"generator": 200, "critic": 1000, "ema": 800},
    final_versions={"generator": 201, "critic": 1005, "ema": 801},
    gan_enabled=True,
)


def contract_for_variant(variant: str) -> StaticFSDP2Full5Contract:
    if variant == "no-gan":
        return NO_GAN_CONTRACT
    if variant == "gan":
        return GAN_CONTRACT
    raise ValueError(f"unknown static FSDP2 variant: {variant}")


def _torch() -> Any:
    import torch

    return torch


def fsdp2_metadata_tolerant_checkpoint(
    module: Any,
    *args: Any,
    operation_source: bool = False,
    **kwargs: Any,
) -> Any:
    """Run Helios' non-reentrant checkpoint under the FSDP2 compatibility policy.

    PyTorch's default non-reentrant checkpoint validator compares saved and
    recomputed tensor metadata.  FSDP2's mixed-precision parameter materializer
    can legitimately expose BF16 tensors in the original forward and FP32
    tensors while replaying the same Helios block.  The validator then raises
    before autograd can finish, even though the checkpoint function, RNG
    handling, and model operation are unchanged.

    This adapter changes only that metadata *check*: it keeps non-reentrant
    checkpointing, preserves RNG state, and does not cast inputs, disable
    gradients, or alter the wrapped module.  The real-model correctness gate
    remains mandatory after installing this FSDP2-only policy.
    """

    from torch.utils.checkpoint import checkpoint

    if "use_reentrant" in kwargs or "determinism_check" in kwargs:
        raise TypeError(
            "FSDP2 checkpoint policy owns use_reentrant and determinism_check"
        )
    checkpoint_kwargs = dict(kwargs)
    if operation_source:
        from distillschedule.profiling.fsdp2_operation_source import checkpoint_context_fn

        checkpoint_kwargs["context_fn"] = checkpoint_context_fn
    return checkpoint(
        module.__call__,
        *args,
        use_reentrant=False,
        determinism_check="none",
        preserve_rng_state=True,
        **checkpoint_kwargs,
    )


def configure_fsdp2_checkpointing(
    model: Any,
    *,
    operation_source: bool = False,
) -> None:
    """Install the FSDP2-only checkpoint adapter on an enabled Helios model."""

    if not bool(getattr(model, "gradient_checkpointing", False)):
        raise RuntimeError(
            "FSDP2 checkpoint policy requires Helios gradient checkpointing"
        )
    if operation_source:
        model._gradient_checkpointing_func = functools.partial(
            fsdp2_metadata_tolerant_checkpoint,
            operation_source=True,
        )
    else:
        model._gradient_checkpointing_func = fsdp2_metadata_tolerant_checkpoint


def _named_trainables(module: Any) -> dict[str, Any]:
    return {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def apply_blockwise_fsdp2(
    module: Any,
    *,
    world_size: int,
    param_dtype: Any | None = None,
    reduce_dtype: Any | None = None,
) -> dict[str, Any]:
    """Shard every Helios Transformer block bottom-up, then shard the root.

    ``fully_shard`` is in-place.  Per-block application is deliberate: it keeps
    the transient all-gather/reduce-scatter unit at one Transformer block and
    gives FSDP2 an opportunity to overlap communication with neighboring block
    compute instead of treating the whole 14B backbone as one root-only unit.
    """

    torch = _torch()
    if world_size != 16:
        raise ValueError("static FSDP2 baseline requires world16")
    if not torch.distributed.is_initialized():
        raise RuntimeError("default process group must exist before FSDP2")
    if torch.distributed.get_world_size() != world_size:
        raise AssertionError("FSDP2 world size differs from baseline contract")
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError as error:
        raise RuntimeError(
            "this baseline requires the composable FSDP2 fully_shard API"
        ) from error

    blocks = list(getattr(module, "blocks", ()))
    if len(blocks) != 40:
        raise AssertionError(f"Helios block count changed: {len(blocks)} != 40")
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
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
    sharded = list(module.parameters())
    if not sharded or not all(hasattr(parameter, "device_mesh") for parameter in sharded):
        raise AssertionError("FSDP2 did not convert the model parameters to DTensor")
    return {
        "backend": "FSDP2_COMPOSABLE_FULL_SHARD",
        "world_size": world_size,
        "block_count": len(blocks),
        "wrap_policy": "40_BLOCKS_BOTTOM_UP_THEN_ROOT",
        "reshard_after_forward": True,
        "param_compute_dtype": str(mp_policy.param_dtype),
        "reduce_dtype": str(mp_policy.reduce_dtype),
        "cpu_parameter_offload": False,
        "parameter_streaming": False,
        "transition": False,
    }


def _logical_tensor(
    package: Any,
    component: str,
    payload_name: str,
    parameter_name: str,
) -> Any:
    torch = _torch()
    spec = package.component(component)
    payload = spec["payloads"][payload_name]
    if payload.get("dtype") != "torch.float32":
        raise ValueError(f"logical {component}/{payload_name} must be FP32")
    path = package.root / str(payload["path"])
    expected_bytes = int(payload["bytes"])
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"logical {component}/{payload_name} byte size changed")
    record = spec["records"][parameter_name]
    total_numel = expected_bytes // 4
    raw = torch.from_file(str(path), shared=False, size=total_numel, dtype=torch.float32)
    offset = int(record["offset_elements"])
    numel = int(record["numel"])
    shape = tuple(int(value) for value in record["shape"])
    return raw.narrow(0, offset, numel).view(shape)


def restore_fsdp2_optimizer_from_logical(
    *,
    module: Any,
    optimizer: Any,
    scheduler: Any,
    package: Any,
    component: str,
) -> dict[str, Any]:
    """Restore Adam moments directly into each trainable FSDP2 DTensor shard."""

    torch = _torch()
    try:
        from torch.distributed.tensor import distribute_tensor
    except ImportError as error:
        raise RuntimeError("FSDP2 optimizer restore requires DTensor") from error

    spec = package.component(component)
    expected_names = [str(name) for name in spec["parameter_names"]]
    trainables = _named_trainables(module)
    if list(trainables) != expected_names:
        raise ValueError(
            f"{component} FSDP2 trainable order differs from logical checkpoint"
        )
    optimizer.state.clear()
    restored_local_numel = 0
    for name in expected_names:
        parameter = trainables[name]
        if not hasattr(parameter, "device_mesh") or not hasattr(parameter, "placements"):
            raise TypeError(f"{component}/{name} is not an FSDP2 DTensor parameter")
        state = optimizer.state[parameter]
        state["step"] = torch.tensor(
            float(spec["optimizer_step"]), dtype=torch.float32, device="cpu"
        )
        for target_key, payload_name in (
            ("exp_avg", "first_moment"),
            ("exp_avg_sq", "second_moment"),
        ):
            full_cpu = _logical_tensor(package, component, payload_name, name)
            full_cuda = full_cpu.to(
                device=torch.cuda.current_device(), dtype=torch.float32, non_blocking=False
            )
            value = distribute_tensor(
                full_cuda,
                device_mesh=parameter.device_mesh,
                placements=parameter.placements,
            )
            if tuple(value.shape) != tuple(parameter.shape):
                raise AssertionError(f"{component}/{name} optimizer global shape changed")
            state[target_key] = value
            restored_local_numel += int(value.to_local().numel())
            del full_cpu, full_cuda, value
    scheduler.load_state_dict(dict(spec["scheduler_state"]))
    if int(scheduler.state_dict()["last_epoch"]) != int(spec["scheduler_state"]["last_epoch"]):
        raise AssertionError("FSDP2 scheduler restore changed the logical step")
    return {
        "component": component,
        "optimizer_step": int(spec["optimizer_step"]),
        "scheduler_step": int(spec["scheduler_state"]["last_epoch"]),
        "trainable_tensor_count": len(trainables),
        "restored_local_moment_numel": restored_local_numel,
        "state_representation": "DTENSOR_MATCHING_PARAMETER_MESH_AND_PLACEMENTS",
    }


class FSDP2EngineAdapter:
    """Small DeepSpeed-shaped adapter for audited Helios helper functions."""

    def __init__(
        self,
        module: Any,
        optimizer: Any,
        scheduler: Any,
        *,
        physical_step: int,
        global_batch: int = 16,
        gradient_clip: float = 1.0,
        dp_world_size: int = 16,
    ) -> None:
        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.global_steps = int(physical_step)
        self.global_samples = int(physical_step) * int(global_batch)
        if isinstance(dp_world_size, bool) or dp_world_size not in (8, 16):
            raise ValueError("explicit FSDP2 DP world size must be 8 or 16")
        self.dp_world_size = int(dp_world_size)
        self._global_batch = int(global_batch)
        self._gradient_clip = float(gradient_clip)
        self._step_calls = 0
        self._gradient_accumulation_boundary = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def backward(self, loss: Any) -> None:
        loss.backward()

    def step(self) -> None:
        torch = _torch()
        trainables = [
            parameter for parameter in self.module.parameters() if parameter.requires_grad
        ]
        if not trainables:
            raise RuntimeError("FSDP2 engine has no trainable parameters")
        clip_norm = torch.nn.utils.clip_grad_norm_(trainables, self._gradient_clip)
        if getattr(self, "_trajectory_capture_grad_norm", False):
            norm = (
                clip_norm.full_tensor()
                if hasattr(clip_norm, "full_tensor")
                else clip_norm
            )
            self.last_gradient_norm = float(norm.detach().float().cpu())
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_steps += 1
        self.global_samples += self._global_batch
        self._step_calls += 1

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def set_gradient_accumulation_boundary(self, *, is_boundary: bool) -> None:
        """Expose the DeepSpeed-shaped boundary hook used by official GAN loss.

        FSDP2 static Full5 is GAS=1, so the hook only records the boundary;
        ``_critic_loss`` still calls ``step()`` exactly once on its final
        micro-wave.  Keeping this API lets the official loss path run without
        changing its two-backward/one-optimizer-commit semantics.
        """

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

    def train_batch_size(self) -> int:
        return self._global_batch

    @property
    def step_calls(self) -> int:
        return self._step_calls



def build_fsdp2_engine(
    *,
    module: Any,
    package: Any,
    component: str,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> tuple[FSDP2EngineAdapter, dict[str, Any]]:
    torch = _torch()
    trainables = list(_named_trainables(module).values())
    if not trainables:
        raise ValueError(f"FSDP2 {component} has no trainables")
    optimizer = torch.optim.AdamW(
        trainables,
        lr=float(lr),
        betas=tuple(float(value) for value in betas),
        eps=float(eps),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    restore = restore_fsdp2_optimizer_from_logical(
        module=module,
        optimizer=optimizer,
        scheduler=scheduler,
        package=package,
        component=component,
    )
    engine = FSDP2EngineAdapter(
        module,
        optimizer,
        scheduler,
        physical_step=int(package.component(component)["optimizer_step"]),
    )
    return engine, restore


class FSDP2ShardEMA:
    """EMA over fixed local FSDP2 shards; no per-step communication.

    ``device='cpu'`` is the frozen baseline.  ``device='cuda'`` is a
    diagnostic-only variant that keeps each local shadow on its owning GPU and
    therefore removes the per-update local-shard D2H copy.  Both paths use the
    same FP32/decay recurrence and never introduce a collective.
    """

    def __init__(
        self,
        *,
        module: Any,
        checkpoint_root: Path,
        device: str = "cpu",
        nvtx_enabled: bool = False,
    ) -> None:
        torch = _torch()
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"unsupported FSDP2 EMA device: {device}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda EMA requested but CUDA is unavailable")
        self.device = device
        self.nvtx_enabled = bool(nvtx_enabled)
        self.last_transfer_bytes = 0
        metadata = json.loads((checkpoint_root / "sparse_ema_manifest.json").read_text())
        if metadata.get("schema_version") != "distillschedule.sparse_ema.v1":
            raise ValueError("unsupported sparse EMA checkpoint")
        trainables = _named_trainables(module)
        tracked = [str(name) for name in metadata["tracked_names"]]
        normalized = [
            name[7:] if name.startswith("module.") and name[7:] in trainables else name
            for name in tracked
        ]
        if set(normalized) != set(trainables):
            raise ValueError("FSDP2 EMA trainable inventory differs from checkpoint")
        loaded: dict[str, Any] = {}
        for shard in metadata["shards"]:
            path = checkpoint_root / str(shard["file"])
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(path, map_location="cpu")
            loaded.update(state)
        try:
            from torch.distributed.tensor import distribute_tensor
        except ImportError as error:
            raise RuntimeError("FSDP2 EMA requires DTensor") from error
        self._parameters: list[Any] = []
        self._names: list[str] = []
        self._shadows: list[Any] = []
        for checkpoint_name, name in zip(tracked, normalized, strict=True):
            parameter = trainables[name]
            full = loaded[checkpoint_name]
            if tuple(full.shape) != tuple(parameter.shape):
                raise ValueError(f"EMA global shape changed for {name}")
            distributed = distribute_tensor(
                full.to(torch.cuda.current_device()),
                device_mesh=parameter.device_mesh,
                placements=parameter.placements,
            )
            local = distributed.to_local().detach().to(
                device=(torch.cuda.current_device() if device == "cuda" else "cpu"),
                dtype=full.dtype,
                copy=True,
            ).contiguous()
            self._parameters.append(parameter)
            self._names.append(name)
            self._shadows.append(local)
            del distributed
        self.decay = float(metadata["decay"])
        self.min_decay = 0.0
        self.update_after_step = int(metadata["update_after_step"])
        self.optimization_step = int(metadata["optimization_step"])
        self.cur_decay_value = metadata.get("cur_decay_value")

    @classmethod
    def load_from_logical(
        cls,
        module: Any,
        logical_root: Path,
        *,
        device: str = "cpu",
        nvtx_enabled: bool = False,
    ) -> "FSDP2ShardEMA":
        candidates = (logical_root / "ema", logical_root / "model_ema")
        for candidate in candidates:
            if (candidate / "sparse_ema_manifest.json").is_file():
                return cls(
                    module=module,
                    checkpoint_root=candidate,
                    device=device,
                    nvtx_enabled=nvtx_enabled,
                )
        raise FileNotFoundError("logical package contains no sparse EMA checkpoint")

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
        torch = _torch()
        for shadow, parameter in zip(self._shadows, self._parameters, strict=True):
            source_local = parameter.detach().to_local()
            if self.device == "cpu" and self.nvtx_enabled:
                torch.cuda.nvtx.range_push("EMA_CPU_D2H")
            try:
                source = source_local.to(device=shadow.device, dtype=shadow.dtype)
            finally:
                if self.device == "cpu" and self.nvtx_enabled:
                    torch.cuda.nvtx.range_pop()
            self.last_transfer_bytes += int(source.numel() * source.element_size())
            if source.shape != shadow.shape:
                raise ValueError("FSDP2 EMA local shard shape changed")
            if self.nvtx_enabled:
                torch.cuda.nvtx.range_push(
                    "EMA_GPU_UPDATE" if self.device == "cuda" else "EMA_CPU_UPDATE"
                )
            try:
                ema_update_reference(shadow, source, decay)
            finally:
                if self.nvtx_enabled:
                    torch.cuda.nvtx.range_pop()
        return decay

    def checksum(self) -> dict[str, Any]:
        total_sum = 0.0
        total_sq = 0.0
        local_numel = 0
        for tensor in self._shadows:
            value = tensor.double()
            total_sum += float(value.sum())
            total_sq += float(value.square().sum())
            local_numel += int(value.numel())
        return {
            "tracked_tensor_count": len(self._shadows),
            "local_numel": local_numel,
            "local_sum": total_sum,
            "local_norm": math.sqrt(total_sq),
            "optimization_step": self.optimization_step,
            "decay": self.cur_decay_value,
            "communication_per_update": False,
            "device": self.device,
        }


def ema_update_reference(shadow: Any, source: Any, decay: float) -> Any:
    """Apply the EMA recurrence in-place; shared by CPU/GPU and parity tests."""

    one_minus_decay = 1.0 - float(decay)
    if float(decay) == 0.0:
        shadow.copy_(source)
    else:
        shadow.sub_(one_minus_decay * (shadow - source))
    return shadow


__all__ = [
    "FSDP2EngineAdapter",
    "FSDP2ShardEMA",
    "ema_update_reference",
    "GAN_CONTRACT",
    "NO_GAN_CONTRACT",
    "StaticFSDP2Full5Contract",
    "apply_blockwise_fsdp2",
    "build_fsdp2_engine",
    "contract_for_variant",
    "restore_fsdp2_optimizer_from_logical",
]
