"""Stable enums shared by workload specifications and runtime backends."""

from __future__ import annotations

from enum import Enum
from typing import TypeVar

from .contracts import ContractViolation


_E = TypeVar("_E", bound="StringEnum")


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    @classmethod
    def parse(cls: type[_E], value: object, *, path: str) -> _E:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ContractViolation(
                f"expected one of [{choices}], got {value!r}", path=path
            ) from exc


class ModelRole(StringEnum):
    GENERATOR = "generator"
    SCORE = "score"
    TEACHER = "teacher"
    CRITIC = "critic"
    EMA = "ema"
    REWARD = "reward"
    VAE = "vae"


class Precision(StringEnum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"


class TrainingBackendKind(StringEnum):
    NATIVE = "native"
    DEEPSPEED = "deepspeed"
    FSDP = "fsdp"
    HSDP = "hsdp"


class RolloutEngineKind(StringEnum):
    NATIVE = "native"
    VLLM = "vllm"
    SGLANG = "sglang"
    REMOTE = "remote"


class HostStatePolicy(StringEnum):
    PRIVATE = "private"
    SHARED_IMMUTABLE = "shared_immutable"


class InstrumentationMode(StringEnum):
    NONE = "none"
    TIMING = "timing"
    MEMORY = "memory"


class OperationKind(StringEnum):
    TENSOR_TRANSFER = "tensor_transfer"
    CALL = "call"
    GENERATOR_ROLLOUT = "generator_rollout"
    SCORE_FAKE = "score_fake"
    SCORE_REAL_CONDITIONAL = "score_real_conditional"
    SCORE_REAL_UNCONDITIONAL = "score_real_unconditional"
    DMD_GRADIENT = "dmd_gradient"
    VAE_DECODE = "vae_decode"
    GAN_LOSS = "gan_loss"
    REWARD_LOSS = "reward_loss"
    GENERATOR_BACKWARD = "generator_backward"
    GENERATOR_UPDATE = "generator_update"
    CRITIC_ROLLOUT = "critic_rollout"
    CRITIC_FORWARD = "critic_forward"
    CRITIC_BACKWARD = "critic_backward"
    CRITIC_UPDATE = "critic_update"
    EMA_UPDATE = "ema_update"
    VERSION_SYNC = "version_sync"
    BARRIER = "barrier"


class GradientMode(StringEnum):
    NO_GRAD = "no_grad"
    GRAD = "grad"
    EXTERNAL_GRAD = "external_grad"


class StepKind(StringEnum):
    FULL = "full"
    CRITIC_ONLY = "critic_only"
    GENERATOR_ONLY = "generator_only"
    IDLE = "idle"
