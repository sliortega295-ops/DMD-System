#!/usr/bin/env python3
"""Locked-A100 C15 modulated-LayerNorm backward parity/latency canary."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable

import torch

from distillschedule.runtime.helios_fused_layernorm_backward_kernel import (
    fused_bf16_layernorm_modulation_hidden_backward,
    fused_reduced_component_gradient,
    fused_tensor4d_component_gradient,
)
from distillschedule.runtime.helios_fused_layernorm_kernel import (
    fused_bf16_layernorm_modulation,
)
from distillschedule.runtime.p0_parity import tensor_parity_metrics


EXPECTED_TRANSFORMER_SHA256 = (
    "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886"
)
WIDTH = 5120
SCALE_INDEX = 4
SHIFT_INDEX = 3
EPSILON = 1.0e-6
CHUNK_TOKENS = 2048


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _expand_component(value: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    if value.shape == hidden.shape:
        return value
    if (
        value.ndim == 3
        and value.shape[0] == hidden.shape[0]
        and value.shape[1] == 1
        and value.shape[2] == hidden.shape[2]
    ):
        return value.expand_as(hidden)
    raise RuntimeError(f"component shape changed: {value.shape} != {hidden.shape}")


def _forward(
    module: Any,
    hidden: torch.Tensor,
    table: torch.Tensor,
    descriptor: tuple[str, tuple[int, ...]],
    values: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    temb = module._rebuild_temb(descriptor, values)
    output = torch.empty_like(hidden)
    for start in range(0, int(hidden.shape[1]), CHUNK_TOKENS):
        end = min(int(hidden.shape[1]), start + CHUNK_TOKENS)
        hidden_chunk = hidden[:, start:end]
        scale_component = _expand_component(
            (
                temb.materialize_component(SCALE_INDEX, start, end)
                if descriptor[0] == "phase_local"
                else (
                    temb[:, start:end, SCALE_INDEX, :]
                    if temb.ndim == 4
                    else temb[:, SCALE_INDEX, :].unsqueeze(1)
                )
            ),
            hidden_chunk,
        )
        shift_component = _expand_component(
            (
                temb.materialize_component(SHIFT_INDEX, start, end)
                if descriptor[0] == "phase_local"
                else (
                    temb[:, start:end, SHIFT_INDEX, :]
                    if temb.ndim == 4
                    else temb[:, SHIFT_INDEX, :].unsqueeze(1)
                )
            ),
            hidden_chunk,
        )
        fused_bf16_layernorm_modulation(
            hidden_chunk,
            table,
            SCALE_INDEX,
            SHIFT_INDEX,
            scale_component,
            shift_component,
            output[:, start:end],
            eps=EPSILON,
        )
    return output


def _source_backward(
    module: Any,
    hidden: torch.Tensor,
    grad_output: torch.Tensor,
    table: torch.Tensor,
    descriptor: tuple[str, tuple[int, ...]],
    values: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    kind, lengths = descriptor
    temb = module._rebuild_temb(descriptor, values)
    hidden_gradient = torch.empty_like(hidden)
    value_gradients: list[torch.Tensor | None] = [None for _ in values]
    normalized_shape = [WIDTH]
    for start in range(0, int(hidden.shape[1]), CHUNK_TOKENS):
        end = min(int(hidden.shape[1]), start + CHUNK_TOKENS)
        hidden_float = hidden[:, start:end].float()
        normalized, mean, rstd = torch.native_layer_norm(
            hidden_float, normalized_shape, None, None, EPSILON
        )
        scale = module._materialize_scale_shift_component(
            table, temb, SCALE_INDEX, start, end
        )
        grad_float = grad_output[:, start:end].float()
        grad_normalized = grad_float * (1.0 + scale)
        hidden_chunk, _weight_gradient, _bias_gradient = (
            torch.ops.aten.native_layer_norm_backward.default(
                grad_normalized,
                hidden_float,
                normalized_shape,
                mean,
                rstd,
                None,
                None,
                [True, False, False],
            )
        )
        hidden_gradient[:, start:end].copy_(hidden_chunk.to(hidden.dtype))

        if kind == "phase_local":
            segment_start = 0
            for index, (value, length) in enumerate(zip(values, lengths, strict=True)):
                segment_end = segment_start + int(length)
                overlap_start = max(start, segment_start)
                overlap_end = min(end, segment_end)
                if overlap_start < overlap_end:
                    local = slice(overlap_start - start, overlap_end - start)
                    chunk_gradient = torch.zeros_like(value)
                    chunk_gradient[:, SCALE_INDEX, :].copy_(
                        (grad_float[:, local] * normalized[:, local])
                        .sum(dim=1)
                        .to(value.dtype)
                    )
                    chunk_gradient[:, SHIFT_INDEX, :].copy_(
                        grad_float[:, local].sum(dim=1).to(value.dtype)
                    )
                    value_gradients[index] = (
                        chunk_gradient
                        if value_gradients[index] is None
                        else value_gradients[index] + chunk_gradient
                    )
                segment_start = segment_end
        else:
            value = values[0]
            chunk_gradient = torch.zeros_like(value)
            if value.ndim == 4:
                chunk_gradient[:, start:end, SCALE_INDEX, :].copy_(
                    (grad_float * normalized).to(value.dtype)
                )
                chunk_gradient[:, start:end, SHIFT_INDEX, :].copy_(
                    grad_float.to(value.dtype)
                )
            elif value.ndim == 3:
                chunk_gradient[:, SCALE_INDEX, :].copy_(
                    (grad_float * normalized).sum(dim=1).to(value.dtype)
                )
                chunk_gradient[:, SHIFT_INDEX, :].copy_(
                    grad_float.sum(dim=1).to(value.dtype)
                )
            else:
                raise RuntimeError(f"unsupported tensor layout: {value.shape}")
            value_gradients[0] = (
                chunk_gradient
                if value_gradients[0] is None
                else value_gradients[0] + chunk_gradient
            )
    if any(value is None for value in value_gradients):
        raise RuntimeError("source canary failed to produce every component gradient")
    return hidden_gradient, tuple(value for value in value_gradients if value is not None)


def _candidate_backward(
    hidden: torch.Tensor,
    grad_output: torch.Tensor,
    table: torch.Tensor,
    descriptor: tuple[str, tuple[int, ...]],
    values: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    kind, lengths = descriptor
    hidden_gradient = torch.empty_like(hidden)
    mean = torch.empty(hidden.shape[:2], device=hidden.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    value_gradients: list[torch.Tensor | None] = [None for _ in values]
    for start in range(0, int(hidden.shape[1]), CHUNK_TOKENS):
        end = min(int(hidden.shape[1]), start + CHUNK_TOKENS)
        if kind == "phase_local":
            segment_start = 0
            for index, (value, length) in enumerate(zip(values, lengths, strict=True)):
                segment_end = segment_start + int(length)
                overlap_start = max(start, segment_start)
                overlap_end = min(end, segment_end)
                if overlap_start < overlap_end:
                    hidden_slice = hidden[:, overlap_start:overlap_end]
                    grad_slice = grad_output[:, overlap_start:overlap_end]
                    component = value[:, SCALE_INDEX, :].unsqueeze(1).expand_as(
                        hidden_slice
                    )
                    mean_slice = mean[:, overlap_start:overlap_end]
                    rstd_slice = rstd[:, overlap_start:overlap_end]
                    fused_bf16_layernorm_modulation_hidden_backward(
                        hidden_slice,
                        grad_slice,
                        table,
                        SCALE_INDEX,
                        component,
                        hidden_gradient[:, overlap_start:overlap_end],
                        mean_slice,
                        rstd_slice,
                        eps=EPSILON,
                    )
                    chunk_gradient = torch.zeros_like(value)
                    fused_reduced_component_gradient(
                        hidden_slice,
                        grad_slice,
                        mean_slice,
                        rstd_slice,
                        chunk_gradient,
                        SCALE_INDEX,
                        SHIFT_INDEX,
                    )
                    value_gradients[index] = (
                        chunk_gradient
                        if value_gradients[index] is None
                        else value_gradients[index] + chunk_gradient
                    )
                segment_start = segment_end
        else:
            value = values[0]
            hidden_slice = hidden[:, start:end]
            grad_slice = grad_output[:, start:end]
            component = (
                value[:, start:end, SCALE_INDEX, :]
                if value.ndim == 4
                else value[:, SCALE_INDEX, :].unsqueeze(1).expand_as(hidden_slice)
            )
            mean_slice = mean[:, start:end]
            rstd_slice = rstd[:, start:end]
            fused_bf16_layernorm_modulation_hidden_backward(
                hidden_slice,
                grad_slice,
                table,
                SCALE_INDEX,
                component,
                hidden_gradient[:, start:end],
                mean_slice,
                rstd_slice,
                eps=EPSILON,
            )
            chunk_gradient = torch.zeros_like(value)
            if value.ndim == 4:
                fused_tensor4d_component_gradient(
                    hidden_slice,
                    grad_slice,
                    mean_slice,
                    rstd_slice,
                    chunk_gradient[:, start:end],
                    SCALE_INDEX,
                    SHIFT_INDEX,
                )
            else:
                fused_reduced_component_gradient(
                    hidden_slice,
                    grad_slice,
                    mean_slice,
                    rstd_slice,
                    chunk_gradient,
                    SCALE_INDEX,
                    SHIFT_INDEX,
                )
            value_gradients[0] = (
                chunk_gradient
                if value_gradients[0] is None
                else value_gradients[0] + chunk_gradient
            )
    if any(value is None for value in value_gradients):
        raise RuntimeError("candidate canary failed to produce every component gradient")
    return hidden_gradient, tuple(value for value in value_gradients if value is not None)


def _inputs(
    batch: int,
    tokens: int,
    layout: str,
    *,
    generator: torch.Generator,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[str, tuple[int, ...]],
    tuple[torch.Tensor, ...],
]:
    device = torch.device("cuda:0")
    hidden = torch.randn(
        (batch, tokens, WIDTH),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    grad_output = torch.randn_like(hidden, generator=generator)
    table = torch.randn(
        (1, 6, WIDTH),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    if layout == "phase_local":
        first = max(1, tokens // 4)
        second = max(1, tokens // 3)
        lengths = (first, second, tokens - first - second)
        values = tuple(
            torch.randn(
                (batch, 6, WIDTH),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            for _ in lengths
        )
        descriptor = ("phase_local", lengths)
    elif layout == "tensor3d":
        values = (
            torch.randn(
                (batch, 6, WIDTH),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            ),
        )
        descriptor = ("tensor", ())
    elif layout == "tensor4d":
        values = (
            torch.randn(
                (batch, tokens, 6, WIDTH),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            ),
        )
        descriptor = ("tensor", ())
    else:
        raise ValueError(f"unsupported canary layout: {layout}")
    return hidden, grad_output, table, descriptor, values


def _one_case(
    module: Any,
    batch: int,
    tokens: int,
    layout: str,
    *,
    warmup: int,
    repeats: int,
    benchmark: bool,
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda:0").manual_seed(
        20260830 + batch * 10000 + tokens
    )
    hidden, grad_output, table, descriptor, values = _inputs(
        batch, tokens, layout, generator=generator
    )

    def source() -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        return (
            _forward(module, hidden, table, descriptor, values),
            *_source_backward(module, hidden, grad_output, table, descriptor, values),
        )

    def candidate() -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        return (
            _forward(module, hidden, table, descriptor, values),
            *_candidate_backward(hidden, grad_output, table, descriptor, values),
        )

    with torch.no_grad():
        reference_output, reference_hidden_gradient, reference_value_gradients = source()
        candidate_output, candidate_hidden_gradient, candidate_value_gradients = candidate()
        torch.cuda.synchronize()
        parity = {
            "output": tensor_parity_metrics(reference_output, candidate_output),
            "hidden_gradient": tensor_parity_metrics(
                reference_hidden_gradient, candidate_hidden_gradient
            ),
            "component_gradients": [
                tensor_parity_metrics(reference, actual)
                for reference, actual in zip(
                    reference_value_gradients,
                    candidate_value_gradients,
                    strict=True,
                )
            ],
        }
        row: dict[str, Any] = {
            "batch": batch,
            "tokens": tokens,
            "width": WIDTH,
            "layout": layout,
            "parity": parity,
        }
        if benchmark:
            source_ms = _measure(source, warmup=warmup, repeats=repeats)
            candidate_ms = _measure(candidate, warmup=warmup, repeats=repeats)
            row.update(
                {
                    "source_forward_plus_backward_ms": source_ms,
                    "candidate_forward_plus_backward_ms": candidate_ms,
                    "speedup": source_ms / candidate_ms,
                }
            )
    del hidden, grad_output, table, values
    torch.cuda.empty_cache()
    return row


def _row_passed(row: dict[str, Any]) -> bool:
    parity = row["parity"]
    return (
        parity["output"]["status"] == "PASS"
        and parity["hidden_gradient"]["status"] == "PASS"
        and all(value["status"] == "PASS" for value in parity["component_gradients"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C15 canary requires exactly one visible locked CUDA GPU")
    module = importlib.import_module("helios.modules.transformer_helios")
    transformer_source = Path(module.__file__).resolve()
    transformer_sha256 = _sha256(transformer_source)
    if transformer_sha256 != EXPECTED_TRANSFORMER_SHA256:
        raise RuntimeError(
            f"Helios transformer identity changed: {transformer_sha256}"
        )

    rows = [
        _one_case(
            module,
            batch,
            2048,
            "phase_local",
            warmup=args.warmup,
            repeats=args.repeats,
            benchmark=True,
        )
        for batch in (1, 3)
    ]
    rows.extend(
        _one_case(
            module,
            batch,
            tokens,
            layout,
            warmup=0,
            repeats=0,
            benchmark=False,
        )
        for layout, tokens in (("tensor3d", 257), ("tensor4d", 33))
        for batch in (1, 3)
    )
    rows.append(
        _one_case(
            module,
            1,
            4097,
            "phase_local",
            warmup=0,
            repeats=0,
            benchmark=False,
        )
    )
    correctness_pass = all(_row_passed(row) for row in rows)
    measured = [row for row in rows if "speedup" in row]
    latency_pass = all(float(row["speedup"]) >= 1.15 for row in measured)
    status = "PASS" if correctness_pass and latency_pass else "FAIL"
    result = {
        "schema_version": "helios-c15-fused-layernorm-backward-canary-v1",
        "status": status,
        "promotion_gate": {
            "all_registered_layouts_p0_gradient_parity": True,
            "minimum_phase_local_forward_plus_backward_speedup": 1.15,
        },
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
        "transformer_source": str(transformer_source),
        "transformer_sha256": transformer_sha256,
        "kernel_source": str(
            Path(
                fused_bf16_layernorm_modulation_hidden_backward.__code__.co_filename
            ).resolve()
        ),
        "kernel_source_sha256": _sha256(
            Path(
                fused_bf16_layernorm_modulation_hidden_backward.__code__.co_filename
            ).resolve()
        ),
        "correctness_pass": correctness_pass,
        "latency_pass": latency_pass,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
