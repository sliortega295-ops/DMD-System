"""Typed, phase-scoped control of DeepSpeed ZeRO-1 low-precision state.

This module deliberately exposes one narrow mechanism.  It does not offload
optimizer moments, gradients, frozen parameters, or arbitrary engine state.
The intended Helios use is to release the inactive Generator ZeRO-1 replicated
BF16 flat buffer while Score/Critic owns the GPU, then restore it before the
next Generator read (EMA included).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import torch


@dataclass(frozen=True)
class LPFlatTensorSpec:
    group_index: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    logical_bytes: int
    storage_bytes: int
    storage_offset: int


@dataclass(frozen=True)
class LPStateInventory:
    logical_component: str
    generation: int
    state: str
    group_count: int
    parameter_count: int
    parameter_numel: int
    logical_bytes: int
    unique_storage_bytes: int
    devices: tuple[str, ...]
    tensors: tuple[LPFlatTensorSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tensors"] = [asdict(spec) for spec in self.tensors]
        return value


def _tensor_logical_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _unique_storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    seen: set[tuple[str, int, int]] = set()
    total = 0
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
        if key in seen:
            continue
        seen.add(key)
        total += int(storage.nbytes())
    return total


class DeepSpeedZero1LPStateController:
    """Offload/reload only ``bit16_groups_flat`` under a strict ZeRO-1 gate."""

    def __init__(
        self,
        engine: Any,
        *,
        logical_component: str,
        event_sink: Callable[..., None] | None = None,
    ) -> None:
        self.engine = engine
        self.logical_component = str(logical_component)
        self.event_sink = event_sink
        self.optimizer = getattr(engine, "optimizer", None)
        if self.optimizer is None:
            raise TypeError("DeepSpeed engine has no optimizer")
        self._lp_state_type, self._cpu_device_type = self._load_state_types()
        self._validate_backend_contract()
        self.generation = 0
        self._offloaded = self._lp_state_type in self.optimizer.offloaded_states
        if self._offloaded:
            raise RuntimeError("Generator LP state is already offloaded at registration")
        inventory = self.inventory(state="ACTIVE")
        if inventory.group_count <= 0 or inventory.logical_bytes <= 0:
            raise RuntimeError("Generator ZeRO-1 LP flat-state inventory is empty")
        self._active_device_types = tuple(
            sorted({torch.device(device).type for device in inventory.devices})
        )
        if len(self._active_device_types) != 1:
            raise RuntimeError(
                f"Generator LP state spans multiple active device types: {self._active_device_types}"
            )
        self._record("generator_lp_state_registered", inventory=inventory.to_dict())

    @staticmethod
    def _load_state_types():
        try:
            from deepspeed.runtime.zero.offload_config import (
                OffloadDeviceEnum,
                OffloadStateTypeEnum,
            )
        except ImportError as error:  # pragma: no cover - live runtime dependency
            raise RuntimeError("DeepSpeed state-offload API is unavailable") from error
        return OffloadStateTypeEnum.lp_params, OffloadDeviceEnum.cpu

    def _validate_backend_contract(self) -> None:
        stage = getattr(self.engine, "zero_optimization_stage", None)
        if not callable(stage) or int(stage()) != 1:
            observed = stage() if callable(stage) else None
            raise RuntimeError(f"Generator LP phase offload requires ZeRO-1, got {observed}")
        if not bool(getattr(self.engine, "zero_offload_optimizer", lambda: False)()):
            raise RuntimeError("Generator LP phase offload requires CPU optimizer offload")
        if not bool(getattr(self.optimizer, "cpu_offload", False)):
            raise RuntimeError("ZeRO optimizer does not report cpu_offload=True")
        if bool(getattr(self.optimizer, "partition_gradients", True)):
            raise RuntimeError("Generator LP phase offload rejects ZeRO-2 partitioned gradients")
        required = (
            "bit16_groups_flat",
            "bit16_groups",
            "offloaded_states",
            "offload_states",
            "reload_states",
        )
        missing = [name for name in required if not hasattr(self.optimizer, name)]
        if missing:
            raise RuntimeError(f"DeepSpeed ZeRO-1 LP-state API is incomplete: {missing}")

    @property
    def is_offloaded(self) -> bool:
        marker = self._lp_state_type in self.optimizer.offloaded_states
        if marker != self._offloaded:
            raise RuntimeError("Generator LP controller state diverged from DeepSpeed marker")
        return marker

    def _record(self, event: str, **fields: Any) -> None:
        if self.event_sink is not None:
            self.event_sink(event, logical_component=self.logical_component, **fields)

    def _flat_tensors(self) -> tuple[torch.Tensor, ...]:
        tensors = tuple(self.optimizer.bit16_groups_flat)
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("bit16_groups_flat contains a non-Tensor entry")
        return tensors

    def _parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(param for group in self.optimizer.bit16_groups for param in group)

    def inventory(self, *, state: str | None = None) -> LPStateInventory:
        flats = self._flat_tensors()
        parameters = self._parameters()
        specs = tuple(
            LPFlatTensorSpec(
                group_index=index,
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
                logical_bytes=_tensor_logical_bytes(tensor),
                storage_bytes=int(tensor.untyped_storage().nbytes()),
                storage_offset=int(tensor.storage_offset()),
            )
            for index, tensor in enumerate(flats)
        )
        return LPStateInventory(
            logical_component=self.logical_component,
            generation=self.generation,
            state=state or ("OFFLOADED" if self._offloaded else "ACTIVE"),
            group_count=len(flats),
            parameter_count=len(parameters),
            parameter_numel=sum(int(param.numel()) for param in parameters),
            logical_bytes=sum(spec.logical_bytes for spec in specs),
            unique_storage_bytes=_unique_storage_bytes(flats),
            devices=tuple(spec.device for spec in specs),
            tensors=specs,
        )

    def _assert_no_live_gradients(self) -> None:
        offenders: list[str] = []
        module = getattr(self.engine, "module", self.engine)
        for index, parameter in enumerate(module.parameters()):
            if parameter.grad is not None:
                offenders.append(f"parameter[{index}].grad")
            if getattr(parameter, "grad_accum", None) is not None:
                offenders.append(f"parameter[{index}].grad_accum")
            if len(offenders) >= 8:
                break
        if offenders:
            raise RuntimeError(
                "Generator LP state cannot be offloaded with live gradients: "
                + ", ".join(offenders)
            )

    def _assert_parameter_views(self, expected_device_type: str) -> None:
        for group_index, (flat, parameters) in enumerate(
            zip(self._flat_tensors(), self.optimizer.bit16_groups)
        ):
            if flat.device.type != expected_device_type:
                raise RuntimeError(
                    f"LP flat group {group_index} is on {flat.device}, expected {expected_device_type}"
                )
            flat_storage_ptr = int(flat.untyped_storage().data_ptr())
            for parameter_index, parameter in enumerate(parameters):
                if parameter.device.type != expected_device_type:
                    raise RuntimeError(
                        f"LP parameter {group_index}:{parameter_index} is on "
                        f"{parameter.device}, expected {expected_device_type}"
                    )
                if parameter.numel() and int(parameter.untyped_storage().data_ptr()) != flat_storage_ptr:
                    raise RuntimeError(
                        f"LP parameter {group_index}:{parameter_index} is not a view of its flat group"
                    )

    def offload(self) -> LPStateInventory:
        if self.is_offloaded:
            raise RuntimeError("Generator LP state is already offloaded")
        self._assert_no_live_gradients()
        before = self.inventory(state="ACTIVE")
        active_types = {torch.device(device).type for device in before.devices}
        if active_types != set(self._active_device_types):
            raise RuntimeError(
                f"Generator LP active device changed before offload: {sorted(active_types)}"
            )
        self.optimizer.offload_states(
            include={self._lp_state_type},
            device=self._cpu_device_type,
            pin_memory=False,
            non_blocking=False,
        )
        self._offloaded = True
        self.generation += 1
        if not self.is_offloaded:
            raise RuntimeError("DeepSpeed did not mark Generator LP state offloaded")
        self._assert_parameter_views("cpu")
        after = self.inventory(state="OFFLOADED")
        if after.logical_bytes != before.logical_bytes:
            raise RuntimeError("Generator LP logical bytes changed during offload")
        self._record(
            "generator_lp_state_offloaded",
            generation=self.generation,
            bytes=after.logical_bytes,
            before=before.to_dict(),
            after=after.to_dict(),
        )
        return after

    def reload(self) -> LPStateInventory:
        if not self.is_offloaded:
            raise RuntimeError("Generator LP state is not offloaded")
        before = self.inventory(state="OFFLOADED")
        self.optimizer.reload_states(non_blocking=False)
        self._offloaded = False
        if self.is_offloaded:
            raise RuntimeError("DeepSpeed did not clear Generator LP offload marker")
        expected_type = self._active_device_types[0]
        self._assert_parameter_views(expected_type)
        after = self.inventory(state="ACTIVE")
        if after.logical_bytes != before.logical_bytes:
            raise RuntimeError("Generator LP logical bytes changed during reload")
        self._record(
            "generator_lp_state_reloaded",
            generation=self.generation,
            bytes=after.logical_bytes,
            before=before.to_dict(),
            after=after.to_dict(),
        )
        return after

    def assert_balanced(self) -> None:
        if self.is_offloaded:
            raise RuntimeError("Generator LP state remains offloaded at lifecycle boundary")
