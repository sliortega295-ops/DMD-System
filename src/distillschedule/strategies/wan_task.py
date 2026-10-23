"""Parameterized contract for the ORIGINAL Wan/Helios-pyramid no-GAN protocol.

Not an Euler/Diffusers substitute. The first backend intentionally retains B1,
world16, three pyramid stages, and the original loss/precision/history rules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from typing import Any, Mapping

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint

SCHEMA = "distillschedule.original_wan_task.v1"
TRACE_KEY = "distillschedule_original_wan_task"


@dataclass(frozen=True)
class OriginalWanTask:
    schema_version: str = SCHEMA
    model_family: str = "wan21-t2v-1.3b-helios-pyramid"
    video_height: int = 384
    video_width: int = 640
    video_frames: int = 33
    fake_updates: int = 5
    first_step: int = 10
    lora_rank: int = 256
    denoising_step_list: tuple[int, ...] = (1000, 750, 500, 250)
    stage2_inference_steps: tuple[int, ...] = (2, 2, 2)
    ema_every: int = 1
    world_size: int = 16
    physical_batch: int = 1
    # An explicit systems-benchmark scope, never an implicit fallback from
    # upstream rank-local amplification. Default capture still fails closed.
    stage_schedule_policy: str = "source_normal_only"

    def __post_init__(self):
        path = "original_wan.task"
        require(self.schema_version == SCHEMA and self.model_family == "wan21-t2v-1.3b-helios-pyramid",
                "unsupported original model/protocol; no automatic protocol substitution", path=path)
        for name in ("video_height", "video_width", "video_frames", "fake_updates", "lora_rank", "world_size", "physical_batch"):
            require(type(getattr(self, name)) is int and getattr(self, name) > 0,
                    f"{name} must be a positive integer", path=path)
        require(self.world_size == 16 and self.physical_batch == 1,
                "existing native engine supports world16/B1 only", path=path)
        require(self.stage_schedule_policy in ("source_normal_only", "benchmark_v4_strict_global"),
                "unsupported stage schedule policy", path=path)
        require(self.video_height % 64 == self.video_width % 64 == 0
                and (self.video_frames - 1) % 4 == 0,
                "shape must fit VAE and all three pyramid patch lattices", path=path)
        require(1 <= self.fake_updates <= 64 and 1 <= self.lora_rank <= 256,
                "update count or LoRA rank outside declared range", path=path)
        require(type(self.first_step) is int and self.first_step >= 0
                and self.first_step % self.fake_updates == 0,
                "first step must be an original Generator-update boundary", path=path)
        require(type(self.ema_every) is int and 0 <= self.ema_every <= 64,
                "EMA cadence must be zero (disabled) or a positive interval", path=path)
        # In the active three-stage path run_generator replaces this list with
        # scheduler.timesteps. Do not advertise an inactive setting as adaptable.
        require(self.denoising_step_list == (1000, 750, 500, 250),
                "active pyramid steps are stage2_inference_steps; ordinary list is not a search input", path=path)
        require(isinstance(self.stage2_inference_steps, tuple) and len(self.stage2_inference_steps) == 3
                and all(type(n) is int and 1 <= n <= 128 for n in self.stage2_inference_steps),
                "three explicit positive pyramid horizons required", path=path)

    @property
    def source_steps(self):
        return tuple(range(self.first_step, self.first_step + self.fake_updates))

    @property
    def ema_after(self):
        return tuple(i for i, step in enumerate(self.source_steps)
                     if self.ema_every and (step + 1) % self.ema_every == 0)

    @property
    def latent_shape(self):
        return (1, 16, (self.video_frames - 1) // 4 + 1, self.video_height // 8, self.video_width // 8)

    @property
    def stage_spatial_shapes(self):
        h, w = self.latent_shape[-2:]
        return ((h // 4, w // 4), (h // 2, w // 2), (h, w))

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    def to_dict(self):
        value = asdict(self)
        # Keep all existing task identities and default serialized plans stable.
        if self.stage_schedule_policy == "source_normal_only":
            value.pop("stage_schedule_policy")
        return json.loads(json.dumps(value, allow_nan=False))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        require(isinstance(value, Mapping) and set(value) <= {f.name for f in fields(cls)},
                "unknown task fields are not ignored", path="original_wan.task")
        result = dict(value)
        for key in ("denoising_step_list", "stage2_inference_steps"):
            if key in result:
                require(isinstance(result[key], (list, tuple)), "expected timestep sequence", path=key)
                result[key] = tuple(result[key])
        return cls(**result)

    def trace_contract(self):
        return {"schema_version": "distillschedule.wan_prepared_trace_contract.v1",
                "protocol": "wan_native_nogan_post_g_independent_rollouts",
                "source_steps": self.source_steps, "generator_source_step": self.first_step,
                "ema_after": self.ema_after, "physical_batch": 1, "world_size": 16,
                "input_binding": "validated_operation_qualified_trace"}


def rollout_geometry(operation_trace):
    """Legacy helpers retain their exact defaults; new tasks are explicitly bound."""
    raw = operation_trace.manifest.get("contract", {}).get(TRACE_KEY)
    task = OriginalWanTask() if raw is None else OriginalWanTask.from_dict(raw)
    return task
