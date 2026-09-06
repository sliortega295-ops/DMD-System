"""EMA state representations used by DistillSchedule experiments.

The upstream Helios EMA owns another complete model.  ``TrainableStateEMA`` keeps
only parameters whose source can change and treats frozen state as an immutable
physical-state reference.  It intentionally understands DeepSpeed ZeRO-3's local
``ds_tensor`` without importing DeepSpeed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import torch


def _local_tensor(parameter: torch.nn.Parameter) -> torch.Tensor:
    partition = getattr(parameter, "ds_tensor", None)
    return partition if partition is not None else parameter.detach()


@dataclass
class _EMAEntry:
    name: str
    parameter: torch.nn.Parameter
    shadow: torch.Tensor
    logical_numel: int


class TrainableStateEMA:
    """EMA over mutable parameter shards only.

    Frozen parameters are semantically shared with the immutable base checkpoint;
    callers must include any mutable non-LoRA state by setting ``requires_grad``.
    The decay schedule exactly follows Helios ``EMAModel_Zero3.get_decay``.
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        device: torch.device | str | None = None,
    ) -> None:
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value: float | None = None
        self._entries: list[_EMAEntry] = []
        self.frozen_logical_numel = 0

        for name, parameter in named_parameters:
            logical_numel = int(getattr(parameter, "ds_numel", parameter.numel()))
            if not parameter.requires_grad:
                self.frozen_logical_numel += logical_numel
                continue
            source = _local_tensor(parameter)
            shadow_device = source.device if device is None else torch.device(device)
            self._entries.append(
                _EMAEntry(
                    name=name,
                    parameter=parameter,
                    shadow=source.detach().to(device=shadow_device, dtype=torch.float32).clone(),
                    logical_numel=logical_numel,
                )
            )

    @property
    def trainable_logical_numel(self) -> int:
        return sum(entry.logical_numel for entry in self._entries)

    @property
    def local_state_bytes(self) -> int:
        return sum(entry.shadow.numel() * entry.shadow.element_size() for entry in self._entries)

    @property
    def logical_state_bytes(self) -> int:
        # Shadows are deliberately FP32, matching upstream EMA storage.
        return self.trainable_logical_numel * torch.tensor([], dtype=torch.float32).element_size()

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def named_state_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        for entry in self._entries:
            yield entry.name, entry.shadow

    def state_tensors(self) -> Iterator[torch.Tensor]:
        for _, tensor in self.named_state_tensors():
            yield tensor

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        if self.use_ema_warmup:
            value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            value = (1 + step) / (10 + step)
        return max(self.min_decay, min(value, self.decay))

    @torch.no_grad()
    def step(self) -> float:
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay
        for entry in self._entries:
            source = _local_tensor(entry.parameter)
            if source.shape != entry.shadow.shape:
                raise RuntimeError(
                    f"EMA shard shape changed for {entry.name}: "
                    f"{tuple(source.shape)} != {tuple(entry.shadow.shape)}"
                )
            source = source.to(device=entry.shadow.device, dtype=entry.shadow.dtype)
            entry.shadow.sub_(one_minus_decay * (entry.shadow - source))
        return decay
