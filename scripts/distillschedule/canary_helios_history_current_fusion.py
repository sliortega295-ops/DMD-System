#!/usr/bin/env python3
"""Locked-A100 C25 exact history/current add-and-cat fusion canary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import torch

from distillschedule.runtime.helios_history_current_kernel import (
    exact_history_current_add,
)
from distillschedule.runtime.p0_parity import tensor_parity_metrics


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


def _execute(
    *,
    hidden_base: torch.Tensor,
    attention_base: torch.Tensor,
    upstream: torch.Tensor,
    history_tokens: int,
    candidate: bool,
) -> dict[str, torch.Tensor]:
    hidden = hidden_base.detach().clone().requires_grad_(True)
    attention = attention_base.detach().clone().requires_grad_(True)
    if candidate:
        output = exact_history_current_add(hidden, attention, history_tokens)
    else:
        history = hidden[:, :history_tokens, :]
        current = hidden[:, history_tokens:, :]
        output = torch.cat([history, current + attention], dim=1)
    output.backward(upstream)
    return {
        "output": output.detach(),
        "hidden_grad": hidden.grad.detach(),
        "attention_grad": attention.grad.detach(),
    }


def _one_shape(
    batch: int,
    *,
    history_tokens: int,
    current_tokens: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    width = 5120
    total_tokens = int(history_tokens + current_tokens)
    generator = torch.Generator(device=device).manual_seed(20260830 + batch)
    hidden = torch.randn(
        (batch, total_tokens, width),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    attention = torch.randn(
        (batch, current_tokens, width),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    upstream = torch.randn(
        hidden.shape,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    reference = _execute(
        hidden_base=hidden,
        attention_base=attention,
        upstream=upstream,
        history_tokens=history_tokens,
        candidate=False,
    )
    candidate = _execute(
        hidden_base=hidden,
        attention_base=attention,
        upstream=upstream,
        history_tokens=history_tokens,
        candidate=True,
    )

    def source() -> None:
        _execute(
            hidden_base=hidden,
            attention_base=attention,
            upstream=upstream,
            history_tokens=history_tokens,
            candidate=False,
        )

    def fused() -> None:
        _execute(
            hidden_base=hidden,
            attention_base=attention,
            upstream=upstream,
            history_tokens=history_tokens,
            candidate=True,
        )

    source_ms = _measure(source, warmup=warmup, repeats=repeats)
    candidate_ms = _measure(fused, warmup=warmup, repeats=repeats)
    parity = {
        name: {
            **tensor_parity_metrics(reference[name], candidate[name]),
            "bitwise_equal": bool(torch.equal(reference[name], candidate[name])),
        }
        for name in reference
    }
    return {
        "batch": batch,
        "history_tokens": int(history_tokens),
        "current_tokens": int(current_tokens),
        "total_tokens": total_tokens,
        "width": width,
        "source_forward_backward_ms": source_ms,
        "candidate_forward_backward_ms": candidate_ms,
        "speedup": source_ms / candidate_ms,
        "parity": parity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-tokens", type=int, default=512)
    parser.add_argument("--current-tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("C25 canary requires one CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("C25 canary must see exactly one locked GPU")
    rows = [
        _one_shape(
            batch,
            history_tokens=args.history_tokens,
            current_tokens=args.current_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for batch in (1, 3)
    ]
    parity_pass = all(
        metric["status"] == "PASS"
        for row in rows
        for metric in row["parity"].values()
    )
    status = (
        "PASS"
        if parity_pass and all(float(row["speedup"]) >= 1.15 for row in rows)
        else "FAIL"
    )
    kernel_path = Path(exact_history_current_add.__code__.co_filename).resolve()
    result = {
        "schema_version": "helios-c25-history-current-fusion-canary-v1",
        "status": status,
        "promotion_gate": {
            "p0_output_and_gradient_parity_all_shapes": True,
            "minimum_forward_backward_speedup_all_shapes": 1.15,
        },
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0),
        "kernel_source": str(kernel_path),
        "kernel_source_sha256": _sha256(kernel_path),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
