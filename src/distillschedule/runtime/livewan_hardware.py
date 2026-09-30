"""Explicit physical placement of the immutable LiveWan logical batch16.

This module owns routing and loss normalization, not model math or trace
generation. The original A100 placement is the default. Four H100 role lanes
reuse the existing four-wave/two-B2-accumulation execution structure, but do
not use the historical B32 trace or its source-reuse contract.
"""
from __future__ import annotations

from dataclasses import dataclass

from distillschedule.runtime.hardware_profile import HardwareProfile, hardware_profile


@dataclass(frozen=True)
class LiveWanHardware:
    profile: HardwareProfile

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def physical_world_size(self) -> int:
        return self.profile.physical_world_size

    @property
    def is_h100(self) -> bool:
        return self.name == "h100-8-b16"

    @property
    def generator_pool_size(self) -> int:
        return self.physical_world_size // 2

    @property
    def node_count(self) -> int:
        return self.physical_world_size // 8

    @property
    def generator_ranks(self) -> tuple[int, ...]:
        return tuple(range(self.generator_pool_size))

    @property
    def score_ranks(self) -> tuple[int, ...]:
        return tuple(range(self.generator_pool_size, self.physical_world_size))

    @property
    def pipeline_waves(self) -> int:
        return self.profile.global_batch // self.generator_pool_size

    @property
    def critic_batch_size(self) -> int:
        return 2

    @property
    def critic_accumulations(self) -> int:
        return self.pipeline_waves // self.critic_batch_size

    @property
    def generator_loss_scale(self) -> float:
        return 1.0 / self.pipeline_waves

    @property
    def critic_loss_scale(self) -> float:
        return 1.0 / self.critic_accumulations

    def baseline_slots(self, rank: int) -> tuple[int, ...]:
        return self.profile.logical_slots(rank)

    def is_generator(self, rank: int) -> bool:
        self.profile.logical_slots(rank)  # Validate the physical rank.
        return rank < self.generator_pool_size

    def lane(self, rank: int) -> int:
        return rank if self.is_generator(rank) else rank - self.generator_pool_size

    def score_peer(self, rank: int) -> int:
        return rank + self.generator_pool_size if self.is_generator(rank) else rank - self.generator_pool_size

    def pipeline_slots(self, lane: int) -> tuple[int, ...]:
        if not 0 <= lane < self.generator_pool_size:
            raise ValueError(f"invalid LiveWan Generator lane {lane}")
        return tuple(lane + wave * self.generator_pool_size for wave in range(self.pipeline_waves))

    def receipt_rank(self, rank: int) -> int:
        """Coordinate used only by legacy G-vs-Score scope validators."""
        return self.lane(rank) if self.is_generator(rank) else self.lane(rank) + 8

    def scope_multiplicity(self) -> int:
        """Observed samples per lane divided by the legacy two samples."""
        return self.pipeline_waves // 2

    def receipt_errors(self, receipt: dict[str, object]) -> list[str]:
        """Never admit legacy A100 evidence for an explicit H100 run."""
        if not self.is_h100:
            return []
        required = {"hardware_profile": self.to_dict(),
                    "physical_world_size": self.physical_world_size,
                    "logical_world_size": 16}
        return [f"hardware receipt changed {key}" for key, value in required.items()
                if receipt.get(key) != value]

    def manifest_errors(self, manifest: dict[str, str]) -> list[str]:
        if not self.is_h100:
            return []
        required = {"hardware_profile": self.name,
                    "physical_world_size": str(self.physical_world_size),
                    "logical_world_size": "16"}
        return [f"hardware manifest changed {key}" for key, value in required.items()
                if manifest.get(key) != value]

    def validate_variant(self, *, mode: str, variant: str) -> None:
        if self.is_h100 and variant != (
            "b1" if mode == "baseline" else "c34-continuous-b16-residual-layout"
        ):
            raise ValueError("H100 LiveWan only admits baseline B1 or canonical C34 B16")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "physical_world_size": self.physical_world_size,
            "logical_world_size": 16,
            "global_batch": 16,
            "generator_pool_size": self.generator_pool_size,
            "score_pool_size": self.generator_pool_size,
            "pipeline_waves": self.pipeline_waves,
            "critic_physical_batch_size": self.critic_batch_size,
            "critic_accumulations_per_commit": self.critic_accumulations,
        }


def livewan_hardware(name: str | None = None) -> LiveWanHardware:
    return LiveWanHardware(hardware_profile(name))
