#!/usr/bin/env python3
"""Locked-A100 B1/B3 parity and latency canary for C26 paired RoPE."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Callable

import torch

from distillschedule.runtime.helios_paired_rope_kernel import (
    exact_paired_rope,
    runtime_identity,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_once(
    function: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ],
    query: torch.Tensor,
    key: torch.Tensor,
    freqs: torch.Tensor,
    query_upstream: torch.Tensor,
    key_upstream: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_input = query.detach().requires_grad_(True)
    key_input = key.detach().requires_grad_(True)
    query_output, key_output = function(query_input, key_input, freqs)
    torch.autograd.backward(
        (query_output, key_output), (query_upstream, key_upstream)
    )
    if query_input.grad is None or key_input.grad is None:
        raise RuntimeError("paired RoPE canary observed a missing input gradient")
    return (
        query_output.detach(),
        key_output.detach(),
        query_input.grad.detach(),
        key_input.grad.detach(),
    )


def _measure_ms(
    function: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ],
    query: torch.Tensor,
    key: torch.Tensor,
    freqs: torch.Tensor,
    query_upstream: torch.Tensor,
    key_upstream: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        _run_once(
            function, query, key, freqs, query_upstream, key_upstream
        )
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        _run_once(
            function, query, key, freqs, query_upstream, key_upstream
        )
        stop.record()
        stop.synchronize()
        values.append(float(start.elapsed_time(stop)))
    return float(statistics.median(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    from helios.modules.helios_kernels.triton_rope import (
        apply_rotary_emb_transposed_flash,
    )

    def reference(
        query: torch.Tensor, key: torch.Tensor, freqs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            apply_rotary_emb_transposed_flash(query, freqs),
            apply_rotary_emb_transposed_flash(key, freqs),
        )

    torch.manual_seed(20260830)
    torch.cuda.manual_seed_all(20260830)
    device = torch.device("cuda:0")
    rows: list[dict[str, object]] = []
    # Exact live C25 receipt shapes include B1 H2400+C8640 and packed B3
    # H2400+C2160.  RoPE sees the combined sequence length.
    for batch, sequence in ((1, 11040), (3, 4560)):
        shape = (batch, sequence, 40, 128)
        query = torch.randn(shape, device=device, dtype=torch.bfloat16)
        key = torch.randn(shape, device=device, dtype=torch.bfloat16)
        freqs = torch.randn(
            (batch, sequence, 256), device=device, dtype=torch.float32
        )
        query_upstream = torch.randn_like(query)
        key_upstream = torch.randn_like(key)
        expected = _run_once(
            reference, query, key, freqs, query_upstream, key_upstream
        )
        actual = _run_once(
            exact_paired_rope,
            query,
            key,
            freqs,
            query_upstream,
            key_upstream,
        )
        equality = [torch.equal(left, right) for left, right in zip(expected, actual)]
        source_ms = _measure_ms(
            reference,
            query,
            key,
            freqs,
            query_upstream,
            key_upstream,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        candidate_ms = _measure_ms(
            exact_paired_rope,
            query,
            key,
            freqs,
            query_upstream,
            key_upstream,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        rows.append(
            {
                "batch": batch,
                "sequence": sequence,
                "query_output_bitwise": equality[0],
                "key_output_bitwise": equality[1],
                "query_gradient_bitwise": equality[2],
                "key_gradient_bitwise": equality[3],
                "source_forward_backward_ms": source_ms,
                "candidate_forward_backward_ms": candidate_ms,
                "speedup": source_ms / candidate_ms,
            }
        )
        del query, key, freqs, query_upstream, key_upstream, expected, actual
        torch.cuda.empty_cache()

    kernel_path = Path(__file__).resolve().parents[2] / (
        "src/distillschedule/runtime/helios_paired_rope_kernel.py"
    )
    passed = all(
        all(
            bool(row[key])
            for key in (
                "query_output_bitwise",
                "key_output_bitwise",
                "query_gradient_bitwise",
                "key_gradient_bitwise",
            )
        )
        and float(row["speedup"]) >= 1.15
        for row in rows
    )
    result = {
        "schema_version": "distillschedule.helios_paired_rope_canary.v1",
        "status": "PASS" if passed else "FAIL",
        "promotion_threshold": 1.15,
        "rows": rows,
        "runtime_identity": runtime_identity(),
        "kernel_source_sha256": _sha256(kernel_path),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
