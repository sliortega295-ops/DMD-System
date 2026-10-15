#!/usr/bin/env python3
"""Primitive-only width1536 test against the actual pinned source Function.

No model construction, no distributed run, and no end-to-end speed claim.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
from pathlib import Path

import torch

from distillschedule.runtime.p0_parity import tensor_parity_metrics
from distillschedule.runtime.wan13b_modulated_norm_fusion import (
    install_exact_modulated_norm_epilogue,
    warmup_exact_modulated_norm_epilogue,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("materialization", "full"), default="full")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("canary output already exists")
    progress = args.output.with_suffix(".rows.jsonl")
    if progress.exists():
        raise FileExistsError("canary progress already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Match the production builder's offline attention binding before importing
    # the Transformer. Importing it bare may attempt an unrelated Hub download.
    from runtime.critic_pool_helios import configure_offline_helios

    configure_offline_helios()
    from helios.modules import transformer_helios as native

    source_forward = native._TembAwareChunkedNorm.forward
    source_backward = native._TembAwareChunkedNorm.backward

    class ReferenceFunction(torch.autograd.Function):
        forward = staticmethod(source_forward)
        backward = staticmethod(source_backward)

    requested_full = args.variant == "full"
    receipt = install_exact_modulated_norm_epilogue(
        enabled=True,
        role="generator",
        transformer_module=native,
        fuse_materialization=True,
        fuse_layernorm=requested_full,
        fuse_backward=requested_full,
        fuse_backward_kernel=requested_full,
    )
    warmup_exact_modulated_norm_epilogue(receipt, device="cuda:0", chunk_tokens=2048)
    candidate_function = native._TembAwareChunkedNorm
    rows = []
    torch.cuda.reset_peak_memory_stats()

    def apply(function, hidden, norm, table, descriptor, inputs):
        return function.apply(hidden, norm, table, 1, 0, 2048, descriptor, *inputs)

    def bench(function, hidden, norm, table, descriptor, inputs, upstream, backward):
        values = []
        for _ in range(2):
            output = apply(function, hidden, norm, table, descriptor, inputs)
            if backward:
                torch.autograd.grad(output, (hidden, *inputs), upstream)
        torch.cuda.synchronize()
        for _ in range(7):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            output = apply(function, hidden, norm, table, descriptor, inputs)
            if backward:
                torch.autograd.grad(output, (hidden, *inputs), upstream)
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
        return {"median_ms": statistics.median(values), "samples_ms": values}

    for tokens in (540, 2160, 8640):
        for layout in ("tensor3d", "phase_local", "tensor4d"):
            torch.manual_seed(3100 + tokens)
            hidden = torch.randn((1, tokens, 1536), device="cuda", dtype=torch.bfloat16).requires_grad_()
            table = torch.randn((1, 6, 1536), device="cuda", dtype=torch.bfloat16)
            upstream = torch.randn_like(hidden)
            norm = native.FP32LayerNorm(1536, eps=1e-6, elementwise_affine=False).cuda()
            if layout == "tensor3d":
                descriptor = ("tensor", ())
                inputs = (torch.randn((1, 6, 1536), device="cuda", dtype=torch.bfloat16).requires_grad_(),)
            elif layout == "tensor4d":
                descriptor = ("tensor", ())
                inputs = (torch.randn((1, tokens, 6, 1536), device="cuda", dtype=torch.bfloat16).requires_grad_(),)
            else:
                lengths = (tokens // 3, tokens - tokens // 3)
                descriptor = ("phase_local", lengths)
                inputs = tuple(torch.randn((1, 6, 1536), device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in lengths)
            reference = apply(ReferenceFunction, hidden, norm, table, descriptor, inputs)
            ref_grad = torch.autograd.grad(reference, (hidden, *inputs), upstream)
            candidate = apply(candidate_function, hidden, norm, table, descriptor, inputs)
            cand_grad = torch.autograd.grad(candidate, (hidden, *inputs), upstream)
            parity = {"output": tensor_parity_metrics(reference, candidate), "hidden_gradient": tensor_parity_metrics(ref_grad[0], cand_grad[0])}
            for index, (expected, actual) in enumerate(zip(ref_grad[1:], cand_grad[1:], strict=True)):
                parity[f"timestep_gradient_{index}"] = tensor_parity_metrics(expected, actual)
            passed = all(value["status"] == "PASS" for value in parity.values())
            row = {"tokens": tokens, "layout": layout, "batch": 1, "width": 1536, "parity": parity, "status": "PASS" if passed else "FAIL", "timing": {}}
            if passed:
                for name, backward in (("forward", False), ("forward_backward", True)):
                    a = bench(ReferenceFunction, hidden, norm, table, descriptor, inputs, upstream, backward)
                    b = bench(candidate_function, hidden, norm, table, descriptor, inputs, upstream, backward)
                    row["timing"][name] = {"reference": a, "candidate": b, "primitive_speedup": a["median_ms"] / b["median_ms"]}
            rows.append(row)
            with progress.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps({"tokens": tokens, "layout": layout, "status": row["status"], "timing": {k: v["primitive_speedup"] for k, v in row["timing"].items()} }), flush=True)
            del hidden, table, upstream, norm, inputs, reference, candidate, ref_grad, cand_grad
            gc.collect()
    data = {"schema_version": "wan13b.modnorm_primitive.v1", "variant": args.variant, "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL", "diagnostic_only": True, "world16_status": "NOT_RUN", "end_to_end_timing": "NOT_RUN", "shape_basis": "B1 width1536; 9x6x10/9x12x20/9x24x40 patch token counts; full history-bearing enclosing workload NOT_RUN", "source_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "runtime_receipt": receipt, "max_allocated_bytes": torch.cuda.max_memory_allocated(), "max_reserved_bytes": torch.cuda.max_memory_reserved(), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    return 0 if data["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
