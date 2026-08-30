#!/usr/bin/env python3
"""Locked-A100 C27 frozen cross-attention LayerNorm canary."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable

import torch

from distillschedule.runtime.helios_frozen_cross_norm_kernel import (
    SUPPORTED_WIDTH,
    fused_frozen_affine_layernorm,
    runtime_identity,
)
from distillschedule.runtime.p0_parity import tensor_parity_metrics


EXPECTED_TRANSFORMER_SHA256 = (
    "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886"
)
EPSILON = 1.0e-6
TOKENS = 2048


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measure(function: Callable[[], Any], *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop) / repeats)


def _one_batch(
    transformer_module: Any,
    norm: torch.nn.Module,
    batch: int,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda:0").manual_seed(20260830 + batch)
    hidden = torch.randn(
        (batch, TOKENS, SUPPORTED_WIDTH),
        device="cuda:0",
        dtype=torch.bfloat16,
        generator=generator,
    )
    grad_output = torch.randn(
        hidden.shape,
        device=hidden.device,
        dtype=hidden.dtype,
        generator=generator,
    )

    def source() -> tuple[torch.Tensor, torch.Tensor]:
        value = hidden.detach().requires_grad_(True)
        output = transformer_module._ChunkedFrozenNorm.apply(value, norm, TOKENS)
        (gradient,) = torch.autograd.grad(output, value, grad_output)
        return output.detach(), gradient.detach()

    def candidate() -> tuple[torch.Tensor, torch.Tensor]:
        value = hidden.detach().requires_grad_(True)
        output = fused_frozen_affine_layernorm(
            value,
            norm.weight,
            norm.bias,
            eps=EPSILON,
        )
        (gradient,) = torch.autograd.grad(output, value, grad_output)
        return output.detach(), gradient.detach()

    reference_output, reference_gradient = source()
    candidate_output, candidate_gradient = candidate()
    torch.cuda.synchronize()
    parity = {
        "output": tensor_parity_metrics(reference_output, candidate_output),
        "hidden_gradient": tensor_parity_metrics(
            reference_gradient, candidate_gradient
        ),
        "output_bitwise": bool(torch.equal(reference_output, candidate_output)),
        "hidden_gradient_bitwise": bool(
            torch.equal(reference_gradient, candidate_gradient)
        ),
    }
    source_ms = _measure(source, warmup=warmup, repeats=repeats)
    candidate_ms = _measure(candidate, warmup=warmup, repeats=repeats)
    return {
        "batch": batch,
        "tokens": TOKENS,
        "width": SUPPORTED_WIDTH,
        "source_forward_plus_backward_ms": source_ms,
        "candidate_forward_plus_backward_ms": candidate_ms,
        "speedup": source_ms / candidate_ms,
        "parity": parity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C27 canary requires exactly one visible locked CUDA GPU")

    transformer_module = importlib.import_module("helios.modules.transformer_helios")
    transformer_source = Path(transformer_module.__file__).resolve()
    transformer_sha256 = _sha256(transformer_source)
    if transformer_sha256 != EXPECTED_TRANSFORMER_SHA256:
        raise RuntimeError(
            f"Helios transformer identity changed: {transformer_sha256}"
        )

    normalization = importlib.import_module("diffusers.models.normalization")
    norm = normalization.FP32LayerNorm(
        SUPPORTED_WIDTH,
        EPSILON,
        elementwise_affine=True,
    ).to(device="cuda:0", dtype=torch.float32)
    triton_norm = importlib.import_module(
        "helios.modules.helios_kernels.triton_norm"
    )
    norm.forward = (
        lambda value: triton_norm.flash_layernorm(norm, value)
    )  # production replace_all_norms_with_flash_norms behavior
    with torch.no_grad():
        generator = torch.Generator(device="cuda:0").manual_seed(20260830027)
        norm.weight.copy_(
            torch.randn(
                norm.weight.shape,
                device=norm.weight.device,
                dtype=norm.weight.dtype,
                generator=generator,
            )
            * 0.2
            + 1.0
        )
        norm.bias.copy_(
            torch.randn(
                norm.bias.shape,
                device=norm.bias.device,
                dtype=norm.bias.dtype,
                generator=generator,
            )
            * 0.1
        )
    norm.requires_grad_(False)

    rows = [
        _one_batch(
            transformer_module,
            norm,
            batch,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for batch in (1, 3)
    ]
    correctness_pass = all(
        row["parity"]["output"]["status"] == "PASS"
        and row["parity"]["hidden_gradient"]["status"] == "PASS"
        for row in rows
    )
    latency_pass = all(float(row["speedup"]) >= 1.15 for row in rows)
    status = "PASS" if correctness_pass and latency_pass else "FAIL"
    result = {
        "schema_version": "distillschedule.helios_frozen_cross_norm_canary.v1",
        "status": status,
        "promotion_gate": {
            "actual_source_output_and_hidden_gradient_p0_parity": True,
            "minimum_forward_plus_backward_speedup_each_batch": 1.15,
        },
        "correctness_pass": correctness_pass,
        "latency_pass": latency_pass,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "transformer_source": str(transformer_source),
        "transformer_sha256": transformer_sha256,
        "source_function": "_ChunkedFrozenNorm",
        "norm_class": f"{type(norm).__module__}.{type(norm).__name__}",
        "norm_weight_dtype": str(norm.weight.dtype),
        "norm_bias_dtype": str(norm.bias.dtype),
        "kernel": runtime_identity(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
