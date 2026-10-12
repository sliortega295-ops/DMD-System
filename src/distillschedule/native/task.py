"""An explicit new flow-DMD protocol, not an override of the canonical Full5 runs."""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import math
from typing import Any, Mapping

from distillschedule.workloads.best_known.registry import fingerprint

SCHEMA = "distillschedule.native_flow_task.v1"
PROTOCOL = "euler_last_step_detached_dmd_post_g_independent_fake_v1"


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class NativeTask:
    schema_version: str = SCHEMA
    protocol: str = PROTOCOL
    backend: str = "torch-reference"
    video_height: int = 32
    video_width: int = 32
    video_frames: int = 5
    compression: tuple[int, int, int] = (4, 8, 8)
    channels: int = 4
    text_tokens: int = 4
    text_dim: int = 8
    microbatch: int = 1
    accumulation: int = 1
    fake_updates: int = 3
    denoising_steps: int = 4
    sigma_shift: float = 1.0
    score_sigma_min: float = 0.02
    score_sigma_max: float = 0.98
    normalization_floor: float = 1e-6
    teacher_guidance: float = 0.0
    lora_rank: int = 2
    lora_alpha: float = 2.0
    target_modules: tuple[str, ...] = ("projection",)
    learning_rate: float = 1e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    ema_after: tuple[int, ...] = ()
    ema_decay: float = 0.99
    seed: int = 17
    dtype: str = "float32"
    device: str = "cpu"
    max_retained_bytes: int = 1073741824
    gpu_reserve_bytes: int = 2147483648
    fixture: bool = True
    model_config: Mapping[str, Any] | None = None
    checkpoints: Mapping[str, str] | None = None
    conditions_path: str | None = None

    def __post_init__(self) -> None:
        ensure(self.schema_version == SCHEMA and self.protocol == PROTOCOL,
               "unsupported native protocol; DiffSynth/history/GAN require an explicit adapter")
        ensure(self.backend in {"torch-reference", "diffusers-wan"}, "unknown native model adapter")
        for name in ("video_height", "video_width", "video_frames", "channels", "text_tokens", "text_dim",
                     "microbatch", "accumulation", "fake_updates", "denoising_steps", "lora_rank", "max_retained_bytes"):
            ensure(type(getattr(self, name)) is int and getattr(self, name) > 0, f"{name} must be a positive integer")
        ensure(self.fake_updates <= 64 and self.denoising_steps <= 128, "update/step bound exceeded")
        ensure(len(self.compression) == 3 and all(type(x) is int and x > 0 for x in self.compression), "invalid compression")
        t, h, w = self.compression
        ensure((self.video_frames - 1) % t == 0 and self.video_height % h == 0 and self.video_width % w == 0,
               "input is outside the declared VAE compression lattice; no silent rounding")
        ensure(self.dtype in {"float32", "bfloat16"}, "only FP32/BF16 are admitted; no silent quantization")
        ensure(self.device == "cpu" or self.device == "cuda" or
               (self.device.startswith("cuda:") and self.device[5:].isdigit()), "invalid device")
        ensure(type(self.fixture) is bool, "fixture must be boolean")
        ensure(type(self.seed) is int and 0 <= self.seed < 2**63, "invalid seed")
        ensure(type(self.gpu_reserve_bytes) is int and self.gpu_reserve_bytes >= 0, "invalid GPU reserve")
        ensure(tuple(sorted(set(self.ema_after))) == tuple(self.ema_after) and
               all(type(i) is int and 0 <= i < self.fake_updates for i in self.ema_after), "invalid EMA positions")
        ensure(self.target_modules and len(set(self.target_modules)) == len(self.target_modules)
               and all(isinstance(x, str) and x for x in self.target_modules), "explicit unique LoRA targets required")
        for name in ("sigma_shift", "normalization_floor", "lora_alpha", "learning_rate", "adam_epsilon", "max_grad_norm"):
            value = getattr(self, name)
            ensure(type(value) in (int, float) and math.isfinite(value) and value > 0, f"invalid {name}")
        for name in ("score_sigma_min", "score_sigma_max", "teacher_guidance", "weight_decay", "ema_decay"):
            ensure(type(getattr(self, name)) in (int, float) and math.isfinite(getattr(self, name)), f"invalid {name}")
        ensure(0 < self.score_sigma_min < self.score_sigma_max < 1, "score sigma interval must be within (0,1)")
        ensure(0 <= self.ema_decay < 1 and self.teacher_guidance >= 0 and self.weight_decay >= 0, "invalid optimizer/guidance")
        ensure(len(self.adam_betas) == 2 and all(type(b) in (int, float) and math.isfinite(b) and 0 <= b < 1
                                               for b in self.adam_betas), "invalid Adam betas")
        if not self.fixture:
            ensure(self.backend == "diffusers-wan" and self.checkpoints is not None and
                   set(self.checkpoints) == {"generator", "critic", "teacher"} and
                   all(isinstance(p, str) and p for p in self.checkpoints.values()),
                   "production needs explicit local Generator/Critic/Teacher checkpoints")
            ensure(bool(self.conditions_path), "production needs pre-encoded conditions; no random-data fallback")
            ensure(self.model_config is None, "production model architecture comes from its checkpoint, not ignored overrides")
        else:
            ensure(self.checkpoints is None and self.conditions_path is None, "fixtures cannot impersonate pretrained runs")
        if self.backend == "diffusers-wan":
            ensure(tuple(self.compression) == (4, 8, 8), "this adapter is Wan2.1 T2V compression only")

    @property
    def batch(self) -> int:
        return self.microbatch * self.accumulation

    @property
    def latent_shape(self) -> tuple[int, ...]:
        t, h, w = self.compression
        return (self.microbatch, self.channels, (self.video_frames - 1) // t + 1,
                self.video_height // h, self.video_width // w)

    @property
    def rollout_bytes(self) -> int:
        return math.prod(self.latent_shape) * self.accumulation * (4 if self.dtype == "float32" else 2)

    @property
    def digest(self) -> str:
        return fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        # JSON-normalize tuple/list fields for round-trip identity.
        import json
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NativeTask":
        ensure(set(value) <= {f.name for f in fields(cls)}, "unknown task field (no ignored GAN/history/precision flags)")
        value = dict(value)
        for key in ("compression", "target_modules", "adam_betas", "ema_after"):
            if key in value:
                ensure(isinstance(value[key], (list, tuple)), f"{key} must be a sequence")
                value[key] = tuple(value[key])
        return cls(**value)
