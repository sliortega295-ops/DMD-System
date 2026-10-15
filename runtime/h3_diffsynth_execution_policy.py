"""Explicit process-local reproducibility policy, before any CUDA construction."""
from __future__ import annotations

import os


def validate_deterministic_timing_gate(enabled, mode, capture, reference=None,
                                       warmup_only_reference=None):
    if reference is not None and warmup_only_reference is not None:
        raise ValueError("ONE_WARMUP_REFERENCE_SCOPE_REQUIRED")
    if warmup_only_reference is not None and not capture:
        raise ValueError("WARMUP_ONLY_REFERENCE_REQUIRES_CAPTURE")
    if enabled and mode == "timing" and not (
            capture and (reference is not None or warmup_only_reference is not None)):
        raise ValueError("DETERMINISTIC_TIMING_REQUIRES_REAL_WARMUP_REFERENCE_GATE")


def configure_deterministic_runtime(enabled=False):
    # Default execution must neither import Torch nor change global state.
    if not enabled:
        return {"requested": False, "configured": False}
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG_MUST_PRECEDE_PROCESS_START")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("DETERMINISTIC_POLICY_MUST_PRECEDE_CUDA_INITIALIZATION")
    torch.use_deterministic_algorithms(True)
    return {
        "requested": True,
        "configured": True,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "CUDA_initialized_before_worker": torch.cuda.is_initialized(),
        "CUDA_matmul_allow_TF32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch": torch.__version__,
        "CUDA_runtime": torch.version.cuda,
    }
