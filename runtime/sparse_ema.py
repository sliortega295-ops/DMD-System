"""Sparse CPU EMA used by the A100 Full5 substrates.

Only trainable Generator tensors are stored.  Frozen backbone state is
reconstructed from the immutable base checkpoint.  This is the already
validated Helios-A100 substrate primitive, not a Route-B optimization.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch


def _backing_tensor(parameter: torch.nn.Parameter) -> torch.Tensor:
    tensor = getattr(parameter, "ds_tensor", None)
    return tensor if tensor is not None else parameter.data


class SparseShardEMA:
    format_version = 1

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        decay: float,
        update_after_step: int,
        base_model: str,
        initialize_shadows: bool = True,
    ) -> None:
        items = list(named_parameters)
        self.parameter_count = len(items)
        self.names: list[str] = []
        self.indices: list[int] = []
        self.shadows: list[torch.Tensor] = []
        self.decay = float(decay)
        self.min_decay = 0.0
        self.update_after_step = int(update_after_step)
        self.optimization_step = 0
        self.cur_decay_value: float | None = None
        self.base_model = str(base_model)
        for index, (name, parameter) in enumerate(items):
            if not parameter.requires_grad:
                continue
            source = _backing_tensor(parameter).detach()
            self.names.append(name)
            self.indices.append(index)
            self.shadows.append(
                source.to(device="cpu", copy=True).contiguous()
                if initialize_shadows
                else torch.empty(0, dtype=source.dtype)
            )
        if not self.names:
            raise ValueError("sparse EMA found no trainable parameters")

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = (1 + step) / (10 + step)
        return max(self.min_decay, min(value, self.decay))

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]) -> float:
        values = list(parameters)
        if len(values) != self.parameter_count:
            raise ValueError(
                f"parameter count changed: initialized with {self.parameter_count}, "
                f"got {len(values)}"
            )
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1.0 - decay
        for shadow, index in zip(self.shadows, self.indices, strict=True):
            source = _backing_tensor(values[index]).detach()
            if source.shape != shadow.shape:
                raise ValueError(
                    f"EMA shard shape changed at index {index}: "
                    f"{source.shape} != {shadow.shape}"
                )
            source_cpu = source.to(device="cpu", dtype=shadow.dtype)
            # Helios is still in the bootstrap region at steps 10--14.  The
            # mathematical decay-zero branch is exact assignment; evaluating
            # the generic subtraction form in BF16 introduces a false drift.
            if decay == 0.0:
                shadow.copy_(source_cpu)
            else:
                shadow.sub_(one_minus_decay * (shadow - source_cpu))
        return decay

    def checksum(self) -> dict[str, Any]:
        total_sum = 0.0
        total_sq = 0.0
        numel = 0
        for tensor in self.shadows:
            value = tensor.double()
            total_sum += value.sum().item()
            total_sq += value.square().sum().item()
            numel += tensor.numel()
        return {
            "num_tensors": len(self.shadows),
            "numel": numel,
            "sum": total_sum,
            "norm": math.sqrt(total_sq),
            "optimization_step": self.optimization_step,
            "decay": self.cur_decay_value,
        }

    def compare_to_parameters(
        self, parameters: Iterable[torch.nn.Parameter]
    ) -> dict[str, Any]:
        values = list(parameters)
        if len(values) != self.parameter_count:
            raise ValueError("EMA parameter count changed")
        mismatched_tensors = 0
        mismatched_numel = 0
        for shadow, index in zip(self.shadows, self.indices, strict=True):
            source = _backing_tensor(values[index]).detach().to(
                device="cpu", dtype=shadow.dtype
            )
            if not torch.equal(shadow, source):
                mismatched_tensors += 1
                mismatched_numel += source.numel()
        return {
            "matches_exact": mismatched_tensors == 0,
            "mismatched_tensors": mismatched_tensors,
            "mismatched_numel": mismatched_numel,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "base_model": self.base_model,
            "frozen_state_policy": "reconstruct_from_base_model",
            "parameter_count": self.parameter_count,
            "tracked_names": self.names,
            "tracked_indices": self.indices,
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "optimization_step": self.optimization_step,
        }

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def save_pretrained(
        self,
        output_dir: str | os.PathLike[str],
        *,
        max_shard_bytes: int = 1024**3,
    ) -> None:
        if max_shard_bytes <= 0:
            raise ValueError("max_shard_bytes must be positive")
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=False)
        shard_specs: list[dict[str, Any]] = []
        current: dict[str, torch.Tensor] = {}
        current_bytes = 0

        def flush() -> None:
            nonlocal current, current_bytes
            if not current:
                return
            filename = f"sparse_ema-{len(shard_specs):05d}.pt"
            destination = root / filename
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(current, temporary)
            os.replace(temporary, destination)
            shard_specs.append(
                {
                    "file": filename,
                    "sha256": self._file_sha256(destination),
                    "storage_bytes": current_bytes,
                    "names": list(current),
                }
            )
            current = {}
            current_bytes = 0

        for name, shadow in zip(self.names, self.shadows, strict=True):
            if shadow.device.type != "cpu" or not shadow.is_contiguous():
                raise ValueError(f"EMA shadow {name!r} is not contiguous CPU storage")
            storage_bytes = shadow.numel() * shadow.element_size()
            if current and current_bytes + storage_bytes > max_shard_bytes:
                flush()
            current[name] = shadow
            current_bytes += storage_bytes
        flush()
        metadata = self.metadata()
        metadata.update(
            {
                "schema_version": "distillschedule.sparse_ema.v1",
                "cur_decay_value": self.cur_decay_value,
                "shards": shard_specs,
            }
        )
        (root / "sparse_ema_manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    @classmethod
    def load_pretrained(
        cls,
        module: torch.nn.Module,
        input_dir: str | os.PathLike[str],
    ) -> "SparseShardEMA":
        root = Path(input_dir)
        metadata = json.loads((root / "sparse_ema_manifest.json").read_text())
        if metadata.get("schema_version") != "distillschedule.sparse_ema.v1":
            raise ValueError("unsupported sparse EMA checkpoint")
        ema = cls(
            module.named_parameters(),
            decay=float(metadata["decay"]),
            update_after_step=int(metadata["update_after_step"]),
            base_model=str(metadata["base_model"]),
            initialize_shadows=False,
        )
        checkpoint_names = [str(name) for name in metadata["tracked_names"]]
        checkpoint_indices = [int(index) for index in metadata["tracked_indices"]]
        if ema.indices != checkpoint_indices:
            raise ValueError("sparse EMA tracked-parameter identity changed")
        if ema.names != checkpoint_names:
            # The frozen M0 EMA package was serialized from a DeepSpeedEngine,
            # whose wrapper contributes exactly one ``module.`` prefix.  The
            # Route-B executor intentionally owns the underlying model and
            # therefore passes ``engine.module`` here.  Accept only this exact,
            # lossless wrapper-name projection; arbitrary renames still fail
            # closed and tensor order/shape/dtype are checked below.
            wrapped_names = [f"module.{name}" for name in ema.names]
            if wrapped_names != checkpoint_names:
                raise ValueError("sparse EMA tracked-parameter identity changed")
            ema.names = checkpoint_names
        if ema.parameter_count != int(metadata["parameter_count"]):
            raise ValueError("sparse EMA model parameter count changed")
        loaded: dict[str, torch.Tensor] = {}
        for shard in metadata["shards"]:
            path = root / shard["file"]
            if cls._file_sha256(path) != shard["sha256"]:
                raise ValueError(f"sparse EMA shard checksum mismatch: {path}")
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(path, map_location="cpu")
            if list(state) != shard["names"]:
                raise ValueError(f"sparse EMA shard name order changed: {path}")
            overlap = loaded.keys() & state.keys()
            if overlap:
                raise ValueError("duplicate sparse EMA tensors")
            loaded.update(state)
        if set(loaded) != set(ema.names):
            raise ValueError("sparse EMA tensor set changed")
        parameters = list(module.parameters())
        ema.shadows = []
        for name, parameter_index in zip(ema.names, ema.indices, strict=True):
            tensor = loaded[name]
            source = _backing_tensor(parameters[parameter_index])
            if tensor.shape != source.shape or tensor.dtype != source.dtype:
                raise ValueError(f"sparse EMA tensor metadata changed for {name}")
            ema.shadows.append(tensor.contiguous())
        ema.optimization_step = int(metadata["optimization_step"])
        ema.cur_decay_value = metadata.get("cur_decay_value")
        return ema
