#!/usr/bin/env python3
"""Bounded CUDA parity canary for the Helios token-chunk candidate.

This deliberately exercises only the source operations whose physical launch
grouping changes.  Attention, MLP, trace, and optimizer math are outside the
candidate and therefore outside this canary.  A world16 correctness run remains
mandatory before any timing claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import torch

from distillschedule.runtime.helios_modulated_norm_fusion import (
    install_exact_modulated_norm_epilogue,
)
from distillschedule.runtime.helios_residual_fusion import (
    generator_nograd_residual_fusion_scope,
    install_generator_nograd_residual_fusion,
)
from distillschedule.runtime.p0_parity import require_tensor_parity


EXPECTED_TRANSFORMER_SHA256 = (
    "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare(reference: torch.Tensor, candidate: torch.Tensor, label: str) -> dict[str, Any]:
    metrics = require_tensor_parity(reference, candidate, label=label)
    metrics["bitwise_equal"] = bool(torch.equal(reference, candidate))
    return metrics


def _phase_local(module: Any, values: tuple[torch.Tensor, torch.Tensor], length: int) -> Any:
    first = length // 3
    return module.PhaseLocalTimestepProjection(
        [(values[0], first), (values[1], length - first)]
    )


def _run_residual(
    module: Any,
    *,
    hidden_base: torch.Tensor,
    residual_base: torch.Tensor,
    table: torch.Tensor,
    temb_base: tuple[torch.Tensor, torch.Tensor],
    upstream: torch.Tensor,
    chunk_tokens: int,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    hidden = hidden_base.detach().clone().requires_grad_(True)
    residual = residual_base.detach().clone().requires_grad_(True)
    temb_values = tuple(value.detach().clone().requires_grad_(True) for value in temb_base)
    temb = _phase_local(module, temb_values, int(hidden.shape[1]))
    output = module._apply_chunked_gated_residual(
        hidden,
        residual,
        table,
        temb,
        2,
        int(chunk_tokens),
        "generator",
        "cuda_out_of_place",
    )
    output.backward(upstream)
    return {
        "output": output.detach(),
        "hidden_grad": hidden.grad.detach(),
        "residual_grad": residual.grad.detach(),
        "temb_grads": [value.grad.detach() for value in temb_values],
    }


def _run_modulated_norm(
    module: Any,
    *,
    hidden_base: torch.Tensor,
    table: torch.Tensor,
    temb_base: tuple[torch.Tensor, torch.Tensor],
    upstream: torch.Tensor,
    chunk_tokens: int,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    hidden = hidden_base.detach().clone().requires_grad_(True)
    temb_values = tuple(value.detach().clone().requires_grad_(True) for value in temb_base)
    temb = _phase_local(module, temb_values, int(hidden.shape[1]))
    norm = module.FP32LayerNorm(
        int(hidden.shape[-1]), eps=1e-6, elementwise_affine=False
    ).to(hidden.device)
    output = module._apply_chunked_modulated_norm(
        norm,
        hidden,
        table,
        temb,
        1,
        0,
        int(chunk_tokens),
    )
    output.backward(upstream)
    return {
        "output": output.detach(),
        "hidden_grad": hidden.grad.detach(),
        "temb_grads": [value.grad.detach() for value in temb_values],
    }


def _parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--sequence-length", type=int, default=4097)
    value.add_argument("--reference-chunk", type=int, default=512)
    value.add_argument("--candidate-chunk", type=int, default=2048)
    return value


def main() -> int:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    module = importlib.import_module("helios.modules.transformer_helios")
    transformer_path = Path(module.__file__).resolve()
    transformer_sha256 = _sha256(transformer_path)
    if transformer_sha256 != EXPECTED_TRANSFORMER_SHA256:
        raise RuntimeError(
            f"Helios transformer identity changed: {transformer_sha256}"
        )

    residual_receipt = install_generator_nograd_residual_fusion(
        enabled=True,
        exact_grad_enabled=True,
        role="generator",
        transformer_module=module,
    )
    norm_receipt = install_exact_modulated_norm_epilogue(
        enabled=True,
        role="generator",
        fuse_materialization=True,
        transformer_module=module,
    )

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(20260829)
    results: dict[str, Any] = {}
    for batch_size in (1, 3):
        shape = (batch_size, int(args.sequence_length), 5120)
        hidden = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        residual = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        upstream = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        table = torch.randn(
            (1, 6, 5120), device=device, dtype=torch.bfloat16, generator=generator
        )
        temb = (
            torch.randn(
                (batch_size, 6, 5120),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            ),
            torch.randn(
                (batch_size, 6, 5120),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            ),
        )

        reference_residual = _run_residual(
            module,
            hidden_base=hidden,
            residual_base=residual,
            table=table,
            temb_base=temb,
            upstream=upstream,
            chunk_tokens=args.reference_chunk,
        )
        candidate_residual = _run_residual(
            module,
            hidden_base=hidden,
            residual_base=residual,
            table=table,
            temb_base=temb,
            upstream=upstream,
            chunk_tokens=args.candidate_chunk,
        )
        reference_norm = _run_modulated_norm(
            module,
            hidden_base=hidden,
            table=table,
            temb_base=temb,
            upstream=upstream,
            chunk_tokens=args.reference_chunk,
        )
        candidate_norm = _run_modulated_norm(
            module,
            hidden_base=hidden,
            table=table,
            temb_base=temb,
            upstream=upstream,
            chunk_tokens=args.candidate_chunk,
        )

        with torch.no_grad(), generator_nograd_residual_fusion_scope():
            no_grad_reference = module._apply_chunked_gated_residual(
                hidden,
                residual,
                table,
                _phase_local(module, temb, int(hidden.shape[1])),
                2,
                int(args.reference_chunk),
                "generator",
                "cuda_out_of_place",
            )
            no_grad_candidate = module._apply_chunked_gated_residual(
                hidden,
                residual,
                table,
                _phase_local(module, temb, int(hidden.shape[1])),
                2,
                int(args.candidate_chunk),
                "generator",
                "cuda_out_of_place",
            )

        label = f"B{batch_size}"
        results[label] = {
            "residual_output": _compare(
                reference_residual["output"], candidate_residual["output"], f"{label}/residual_output"
            ),
            "residual_hidden_grad": _compare(
                reference_residual["hidden_grad"], candidate_residual["hidden_grad"], f"{label}/residual_hidden_grad"
            ),
            "residual_residual_grad": _compare(
                reference_residual["residual_grad"], candidate_residual["residual_grad"], f"{label}/residual_residual_grad"
            ),
            "residual_temb_grad_0": _compare(
                reference_residual["temb_grads"][0], candidate_residual["temb_grads"][0], f"{label}/residual_temb_grad_0"
            ),
            "residual_temb_grad_1": _compare(
                reference_residual["temb_grads"][1], candidate_residual["temb_grads"][1], f"{label}/residual_temb_grad_1"
            ),
            "modnorm_output": _compare(
                reference_norm["output"], candidate_norm["output"], f"{label}/modnorm_output"
            ),
            "modnorm_hidden_grad": _compare(
                reference_norm["hidden_grad"], candidate_norm["hidden_grad"], f"{label}/modnorm_hidden_grad"
            ),
            "modnorm_temb_grad_0": _compare(
                reference_norm["temb_grads"][0], candidate_norm["temb_grads"][0], f"{label}/modnorm_temb_grad_0"
            ),
            "modnorm_temb_grad_1": _compare(
                reference_norm["temb_grads"][1], candidate_norm["temb_grads"][1], f"{label}/modnorm_temb_grad_1"
            ),
            "residual_nograd_output": _compare(
                no_grad_reference, no_grad_candidate, f"{label}/residual_nograd_output"
            ),
        }
        del hidden, residual, upstream, table, temb
        del reference_residual, candidate_residual, reference_norm, candidate_norm
        del no_grad_reference, no_grad_candidate
        torch.cuda.empty_cache()

    torch.cuda.synchronize(device)
    report = {
        "status": "PASS",
        "transformer_source": str(transformer_path),
        "transformer_sha256": transformer_sha256,
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(device),
        "sequence_length": int(args.sequence_length),
        "reference_chunk_tokens": int(args.reference_chunk),
        "candidate_chunk_tokens": int(args.candidate_chunk),
        "residual_receipt": residual_receipt,
        "modulated_norm_receipt": norm_receipt,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
