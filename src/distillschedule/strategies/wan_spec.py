"""Executable strategy domains for the original Wan/Helios-pyramid backend.

Resolution and update frequency are task inputs, never optimization knobs.
This version retains world16/B1, rank-256 LoRA and the original model builder.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from distillschedule.core.contracts import require
from distillschedule.search.wan_implementation import WanImplementation
from distillschedule.workloads.best_known.registry import fingerprint


@dataclass(frozen=True)
class PyramidRolloutSpec:
    latent_shape: tuple[int, ...] = (1, 16, 9, 48, 80)
    stage_steps: tuple[int, ...] = (2, 2, 2)
    denoising_timesteps: tuple[int, ...] = (1000, 750, 500, 250)

    def __post_init__(self):
        s = self.latent_shape
        require(len(s) == 5 and all(type(x) is int and x > 0 for x in s)
                and s[:2] == (1,16), "pyramid backend requires physical B1/C16", path="wan.spec")
        require(s[-2] % 8 == s[-1] % 8 == 0,
                "spatial latent dimensions must support three patch-aligned pyramid levels", path="wan.spec")
        require(len(self.stage_steps) == 3 and all(type(x) is int and 1 <= x <= 128 for x in self.stage_steps),
                "three positive bounded stage budgets required", path="wan.spec")
        ts = self.denoising_timesteps
        require(len(ts) >= 1 and all(type(x) is int and 1 <= x <= 1000 for x in ts)
                and all(a > b for a,b in zip(ts,ts[1:])), "invalid denoising timestep schedule", path="wan.spec")

    @property
    def spatial_shapes(self):
        h,w = self.latent_shape[-2:]
        return [[h//4,w//4], [h//2,w//2], [h,w]]


@dataclass(frozen=True)
class WanTask:
    task_id: str
    resolution: tuple[int, int] = (384,640)
    video_frames: int = 33
    source_steps: tuple[int, ...] = (10,11,12,13,14)
    ema_after: tuple[int, ...] = (0,1,2,3,4)
    stage_steps: tuple[int, ...] = (2,2,2)
    denoising_timesteps: tuple[int, ...] = (1000,750,500,250)
    world_size: int = 16
    lora_rank: int = 256
    schema_version: str = "distillschedule.parameterized_wan_pyramid.v1"

    def __post_init__(self):
        require(self.schema_version == "distillschedule.parameterized_wan_pyramid.v1", "wrong native protocol", path="wan.task")
        require(isinstance(self.task_id,str) and self.task_id and all(c.isalnum() or c in '-_.' for c in self.task_id),
                "explicit portable task ID required", path="wan.task")
        require(type(self.world_size) is int and self.world_size == 16 and type(self.lora_rank) is int and self.lora_rank == 256,
                "existing native engines currently require world16 and LoRA rank256", path="wan.task")
        require(len(self.resolution) == 2 and all(type(x) is int and x > 0 and x % 64 == 0 for x in self.resolution),
                "resolution must lie on the three-stage VAE/patch lattice", path="wan.task")
        require(type(self.video_frames) is int and self.video_frames > 0 and (self.video_frames-1)%4 == 0,
                "frames must lie on the Wan temporal VAE lattice", path="wan.task")
        n = len(self.source_steps)
        require(1 <= n <= 64 and all(type(x) is int and x >= 0 for x in self.source_steps)
                and tuple(self.source_steps) == tuple(range(self.source_steps[0], self.source_steps[0]+n)),
                "require 1..64 consecutive trace-qualified source steps", path="wan.task")
        require(tuple(sorted(set(self.ema_after))) == self.ema_after
                and all(type(i) is int and 0 <= i < n for i in self.ema_after), "invalid EMA boundaries", path="wan.task")
        self.rollout_spec

    @property
    def rollout_spec(self):
        h,w=self.resolution
        return PyramidRolloutSpec((1,16,(self.video_frames-1)//4+1,h//8,w//8),
                                  self.stage_steps,self.denoising_timesteps)

    @property
    def digest(self): return fingerprint(self.to_dict())

    def to_dict(self):
        import json
        return json.loads(json.dumps(asdict(self),allow_nan=False))

    @classmethod
    def from_dict(cls, value: Mapping[str,Any]):
        require(set(value) <= {f.name for f in fields(cls)}, "unknown or ignored task fields", path="wan.task")
        data=dict(value)
        for key in ('resolution','source_steps','ema_after','stage_steps','denoising_timesteps'):
            if key in data:
                require(isinstance(data[key],(list,tuple)), f"{key} must be a sequence", path="wan.task")
                data[key]=tuple(data[key])
        return cls(**data)


@dataclass(frozen=True)
class WanStrategy:
    placement: str = "fsdp"
    chunk_tokens: int = 512
    lora_epilogue: bool = False
    rollout_window: int = 1

    def __post_init__(self):
        self.implementation
        require(type(self.rollout_window) is int and 1 <= self.rollout_window <= 64,
                "invalid native retained window", path="wan.strategy")

    @property
    def implementation(self):
        # Reuse the real audited mechanism mapping, but do not encode the
        # new update schedule in the old w1..w5 candidate identity.
        return WanImplementation(self.placement,self.chunk_tokens,self.lora_epilogue)

    @property
    def id(self):
        return f"wan-pyramid-{self.placement}-c{self.chunk_tokens}-e{int(self.lora_epilogue)}-w{self.rollout_window}"

    def to_dict(self): return asdict(self)

    def environment(self):
        impl=self.implementation
        return {**impl.environment(), "DISTILLSCHEDULE_SEARCH_CANDIDATE_ID":impl.id,
                "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE":str(int(self.lora_epilogue)),
                "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE":"0",
                "DISTILLSCHEDULE_WAN13B_EMA_IMPLEMENTATION":"reference",
                **({"NCCL_ALGO":"Tree"} if self.placement == "resident" else {})}


def strategies_for(task: WanTask) -> tuple[WanStrategy,...]:
    domains=[("fsdp",512,False)] + [("resident",c,e) for c in (512,1024,2048) for e in (False,True)]
    return tuple(WanStrategy(p,c,e,w) for p,c,e in domains for w in range(1,len(task.source_steps)+1))
