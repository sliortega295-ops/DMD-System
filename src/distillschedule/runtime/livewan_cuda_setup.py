"""Shared, pre-model CUDA library initialization; no training operations.

cuBLAS creates a handle lazily per host thread. A later first backward can
fail in cublasCreate even with few live tensors if the CUDA caching allocator
has reserved almost all driver memory. Initialize the main and CUDA autograd
thread handles before model construction, identically for both experiment arms.
"""
from __future__ import annotations

import threading
import time
from typing import Any


def materialize_blas_handles(torch: Any, device: Any) -> dict[str, Any]:
    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "policy": "main_and_autograd_blas_handles_before_model_v1",
        "status": "RUNNING", "timing_scope": "SETUP_ONLY",
        "model_calls": 0, "optimizer_updates": 0, "random_draws": 0,
        "main_thread": threading.get_native_id(),
    }
    torch.cuda.current_blas_handle()

    class InitializeAutogradHandle(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any) -> Any:
            return x.clone()

        @staticmethod
        def backward(ctx: Any, grad: Any) -> Any:
            torch.cuda.current_blas_handle()
            receipt["autograd_thread"] = threading.get_native_id()
            receipt["autograd_handle_created"] = True
            return grad

    # No model/optimizer state or RNG is accessed. The tiny graph also loads
    # scalar CUDA modules before allocator reservation pressure can build up.
    with torch.enable_grad():
        x = torch.ones((), device=device, requires_grad=True)
        InitializeAutogradHandle.apply(x).backward()
    torch.cuda.synchronize(device)
    if x.grad is None or x.grad.item() != 1.0:
        raise RuntimeError("CUDA library initialization identity backward failed")
    del x
    if not receipt.get("autograd_handle_created"):
        raise RuntimeError("CUDA autograd-thread BLAS initialization missing")
    receipt.update(status="PASS", wall_seconds=time.perf_counter() - started)
    return receipt
