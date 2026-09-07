"""Explicit execution precision for new plans; no inherited-env assumptions.

This module does not change Torch globals. A lowerer emits the environment
before Torch import, and the worker must verify the effective policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import require


@dataclass(frozen=True)
class ExecutionPrecision:
    base_dtype: str
    trainable_dtype: str
    reduction_dtype: str
    matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    attention_backend: str

    def __post_init__(self) -> None:
        for dtype in (self.base_dtype, self.trainable_dtype, self.reduction_dtype):
            require(dtype in {"float32", "bfloat16", "float16"}, "unsupported execution dtype")
        require(type(self.matmul_allow_tf32) is bool and type(self.cudnn_allow_tf32) is bool,
                "TF32 policies must be explicit booleans")
        require(bool(self.attention_backend), "attention backend identity required")

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))

    def launch_environment(self) -> dict[str, str]:
        # This is the exact PyTorch override implicated by the recorded Wan
        # regression. Do not invent a driver override to "enable" TF32.
        return {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1" if self.matmul_allow_tf32 else "0"}

    def assert_runtime(self, torch: Any, environment: Mapping[str, str]) -> dict[str, Any]:
        for key, value in self.launch_environment().items():
            require(environment.get(key) == value, f"effective precision environment differs: {key}")
        require(environment.get("NVIDIA_TF32_OVERRIDE") in {None, ""},
                "unregistered driver-level NVIDIA_TF32_OVERRIDE")
        observed = {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                    "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                    "float32_matmul_precision": torch.get_float32_matmul_precision()}
        require(observed["matmul_allow_tf32"] == self.matmul_allow_tf32,
                "effective matmul TF32 differs from plan")
        require(observed["cudnn_allow_tf32"] == self.cudnn_allow_tf32,
                "effective cuDNN TF32 differs from plan")
        return {"status": "PASS", "precision_fingerprint": self.digest, "observed": observed,
                "parameter_dtype_gate": "REQUIRES_NATIVE_LAYER_RECEIPTS"}
