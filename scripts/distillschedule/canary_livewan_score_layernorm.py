#!/usr/bin/env python3
"""Standalone source-native frozen BF16 Score LayerNorm parity/performance gate."""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import traceback


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--seed", type=int, default=20260930)
    args = p.parse_args()
    out = dict(status="NOT_RUN", candidate="C36_SCORE_FROZEN_NON_AFFINE_LAYERNORM",
        parent="C35", world16="NOT_RUN", production_installed=False,
        kind="SYNTHETIC_REAL_SHAPE_SOURCE_NATIVE_CANARY_NOT_FULL5",
        numerical_gate="UNCHANGED_DEFAULT_P0_PARITY", cases=[], benchmarks=[],
        operator_profiles=[], command=sys.argv, seed=args.seed)
    if args.plan_only:
        print(json.dumps(out)); return 0
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.output)
    save()
    try:
        import torch
        from torch.utils.checkpoint import checkpoint
        from wan.modules.model import WanLayerNorm
        from distillschedule.runtime.livewan_score_layernorm import build, SOURCE_SHA256
        from distillschedule.runtime.p0_parity import require_tensor_parity

        source_sha = hashlib.sha256(inspect.getsource(WanLayerNorm).encode()).hexdigest()
        assert source_sha == SOURCE_SHA256, source_sha
        assert torch.cuda.device_count() == 1
        torch.cuda.set_device(0)
        torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
        op, receipt = build(torch)
        native_forward = WanLayerNorm.forward
        out.update(torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0), source_sha256=source_sha,
            source_file=inspect.getfile(WanLayerNorm), kernel_receipt=receipt,
            candidate_sha256=hashlib.sha256(Path(inspect.getfile(build)).read_bytes()).hexdigest(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))

        def incoming(shape, layout):
            b, s, d = shape
            return (torch.randn((b, d, s), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
                    if layout == "channel" else torch.randn(shape, device="cuda", dtype=torch.bfloat16))

        def measured(fn):
            for _ in range(4): fn()
            samples = []
            for _ in range(9):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); fn(); b.record(); b.synchronize()
                samples.append(a.elapsed_time(b))
            return dict(median_ms=statistics.median(samples), samples_ms=samples,
                scope="CUDA_EVENTS_CANARY_ONLY_NOT_NATIVE_C2")

        torch.cuda.reset_peak_memory_stats()
        specs = [(b, s, d, amplitude, offset, layout)
            for d in (1536, 5120) for b in (1, 2)
            for s in (64, 19320) for layout in ("row", "channel")
            for amplitude, offset in ((1.0, 0.0),)]
        specs += [(1, 64, d, a, o, "row") for d in (1536, 5120)
                  for a, o in ((0.001, 0.0), (0.125, 2.0), (0.0, 1.0))]
        for b, s, d, amplitude, offset, layout in specs:
            shape = (b, s, d)
            x = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * amplitude + offset).requires_grad_(True)
            dy = incoming(shape, layout)
            norm = WanLayerNorm(d).to(device="cuda", dtype=torch.bfloat16)
            assert norm.weight is None and norm.bias is None and norm.eps == 1e-6
            version = x._version
            rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                a, c = norm(x), op(x, norm.eps)
            forward = require_tensor_parity(a, c, label=f"C36 forward {shape}/{amplitude}/{offset}")
            da, = torch.autograd.grad(a, x, dy)
            dc, = torch.autograd.grad(c, x, dy)
            gradient = require_tensor_parity(da, dc, label=f"C36 dInput {shape}/{layout}")
            assert x._version == version
            assert torch.equal(rng[0], torch.get_rng_state()) and torch.equal(rng[1], torch.cuda.get_rng_state())
            row = dict(shape=shape, amplitude=amplitude, offset=offset, gradient_stride=dy.stride(),
                       output=forward, dInput=gradient, rng="PASS", input_version="PASS")
            if s == 64 and amplitude == 1.0 and layout == "row":
                counts = [0, 0]
                def nfn(x):
                    counts[0] += 1; return norm(x)
                def cfn(x):
                    counts[1] += 1; return op(x, norm.eps)
                a, c = checkpoint(nfn, x, use_reentrant=False), checkpoint(cfn, x, use_reentrant=False)
                row["checkpoint_output"] = require_tensor_parity(a, c, label=f"C36 checkpoint output {shape}")
                da, = torch.autograd.grad(a, x, dy)
                dc, = torch.autograd.grad(c, x, dy)
                row["checkpoint_dInput"] = require_tensor_parity(da, dc, label=f"C36 checkpoint dInput {shape}")
                assert counts == [2, 2], counts
                row["checkpoint_execution_counts"] = counts
            out["cases"].append(row); save()

            if s == 19320 and layout == "row":
                with torch.no_grad():
                    result = dict(shape=shape, direction="forward",
                        native=measured(lambda: norm(x)), candidate=measured(lambda: op(x, norm.eps)))
                result["speedup"] = result["native"]["median_ms"] / result["candidate"]["median_ms"]
                out["benchmarks"].append(result); save()
                # Include the real forward/backward chain, not a forward-only promotion.
                def chain(enabled):
                    y = op(x, norm.eps) if enabled else norm(x)
                    return torch.autograd.grad(y, x, dy)[0]
                result = dict(shape=shape, direction="forward_backward",
                    native=measured(lambda: chain(False)), candidate=measured(lambda: chain(True)))
                result["speedup"] = result["native"]["median_ms"] / result["candidate"]["median_ms"]
                out["benchmarks"].append(result); save()

                if b == 2:
                    for enabled in (False, True):
                        arm = "candidate" if enabled else "native"
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                            with torch.profiler.record_function("C36_SCORE_LAYERNORM/" + arm):
                                chain(enabled)
                        path = args.output.parent / f"{arm}_layernorm_d{d}_operator_trace.json"
                        prof.export_chrome_trace(str(path))
                        events = []
                        for event in prof.events():
                            if event.name not in ("aten::to", "aten::_to_copy", "aten::copy_", "aten::native_layer_norm", "aten::native_layer_norm_backward"):
                                continue
                            parents = []; parent = event.cpu_parent
                            while parent is not None:
                                parents.append(parent.name); parent = parent.cpu_parent
                            events.append(dict(operator=event.name, parents=parents, shapes=event.input_shapes,
                                thread=event.thread, cpu_us=event.cpu_time_total, device_us=event.device_time_total))
                        out["operator_profiles"].append(dict(arm=arm, shape=shape, path=str(path), events=events)); save()
            del x, dy, norm, a, c, da, dc
            gc.collect()
        assert WanLayerNorm.forward is native_forward
        out.update(status="PASS_CANARY_ONLY", production_promotion="NOT_RUN",
            native_patch_unchanged=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), driver_free_bytes=torch.cuda.mem_get_info()[0])
        save(); print(json.dumps({k: out[k] for k in ("status", "benchmarks", "peak_allocated_bytes", "peak_reserved_bytes")}))
        return 0
    except Exception:
        out.update(status="FAIL_CANARY", error=traceback.format_exc()); save(); raise


if __name__ == "__main__":
    raise SystemExit(main())
