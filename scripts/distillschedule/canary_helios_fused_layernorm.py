#!/usr/bin/env python3
"""Locked-A100 C14 forward-only numerical and latency canary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import torch

from distillschedule.runtime.helios_fused_layernorm_kernel import (
    fused_bf16_layernorm_modulation,
)
from distillschedule.runtime.helios_modulated_norm_kernel import (
    fused_modulated_norm_materialization,
)
from distillschedule.runtime.p0_parity import compare_tensors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _measure(function: Callable[[], None], *, warmup: int, repeats: int) -> float:
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


def _one_shape(batch: int, tokens: int, *, warmup: int, repeats: int) -> dict[str, Any]:
    device = torch.device("cuda:0")
    width = 5120
    epsilon = 1.0e-6
    generator = torch.Generator(device=device).manual_seed(20260830 + batch)
    hidden = torch.randn(
        (batch, tokens, width),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    table = torch.randn(
        (1, 6, width),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    scale_component_base = torch.randn(
        (batch, 1, width),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    shift_component_base = torch.randn(
        (batch, 1, width),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    scale_component = scale_component_base.expand_as(hidden)
    shift_component = shift_component_base.expand_as(hidden)
    reference = torch.empty_like(hidden)
    candidate = torch.empty_like(hidden)

    def source() -> None:
        normalized = torch.nn.functional.layer_norm(
            hidden.float(), (width,), None, None, epsilon
        )
        fused_modulated_norm_materialization(
            normalized,
            table,
            4,
            3,
            scale_component,
            shift_component,
            reference,
        )

    def fused() -> None:
        fused_bf16_layernorm_modulation(
            hidden,
            table,
            4,
            3,
            scale_component,
            shift_component,
            candidate,
            eps=epsilon,
        )

    source()
    fused()
    torch.cuda.synchronize()
    parity = compare_tensors(reference, candidate)
    source_ms = _measure(source, warmup=warmup, repeats=repeats)
    candidate_ms = _measure(fused, warmup=warmup, repeats=repeats)
    return {
        "batch": batch,
        "tokens": tokens,
        "width": width,
        "source_ms": source_ms,
        "candidate_ms": candidate_ms,
        "speedup": source_ms / candidate_ms,
        "parity": parity,
        "component_stride": list(scale_component.stride()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("C14 canary requires one CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("C14 canary must see exactly one locked GPU")
    rows = [
        _one_shape(batch, 2048, warmup=args.warmup, repeats=args.repeats)
        for batch in (1, 3)
    ]
    status = (
        "PASS"
        if all(row["parity"]["status"] == "PASS" for row in rows)
        and all(float(row["speedup"]) >= 1.15 for row in rows)
        else "FAIL"
    )
    result = {
        "schema_version": "helios-c14-fused-layernorm-canary-v1",
        "status": status,
        "promotion_gate": {
            "p0_output_parity_all_shapes": True,
            "minimum_speedup_all_shapes": 1.15,
        },
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0),
        "kernel_source": str(
            Path(fused_bf16_layernorm_modulation.__code__.co_filename).resolve()
        ),
        "kernel_source_sha256": _sha256(
            Path(fused_bf16_layernorm_modulation.__code__.co_filename).resolve()
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
