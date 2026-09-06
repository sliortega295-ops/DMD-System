#!/usr/bin/env python3
"""One-owned-GPU exact FA2 SAC canary. Synthetic QKV, observed B2 shapes.

No training models or weights are loaded. Microbenchmarks are not Full5 timing.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--repetitions", type=int, default=10)
    a = p.parse_args()
    shapes = [(2, 512, 512), (2, 19320, 19320), (2, 19320, 512)]
    out = {"status": "NOT_RUN", "kind": "SYNTHETIC_REAL_SHAPE_PINNED_FA2_CANARY",
           "production_installed": False, "world16": "NOT_RUN", "cases": [],
           "planned_batch_query_key": shapes, "seed": a.seed, "command": sys.argv,
           "thresholds": {"forward": "BITWISE_EQUAL", "gradient_assert_close": "torch BF16 defaults",
                          "normalized_l2_max": 1e-3, "cosine_min": .999999}}
    if a.plan_only:
        print(json.dumps(out, indent=2))
        return 0
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = a.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
        tmp.replace(a.output)

    save()
    try:
        import torch
        from torch.utils.checkpoint import checkpoint
        from distillschedule.runtime.livewan_attention_backend import _load_flash_attn2
        from distillschedule.runtime.livewan_attention_checkpoint import (
            AttentionCheckpointContexts, discover_flash_forward)
        from distillschedule.runtime.livewan_cuda_setup import materialize_blas_handles

        assert torch.cuda.device_count() == 1, "requires one explicitly owned visible GPU"
        torch.cuda.set_device(0)
        torch.manual_seed(a.seed)
        flash_attn2, kernel_root = _load_flash_attn2()
        op, identity = discover_flash_forward(flash_attn2)
        identity["kernel_root"] = str(kernel_root)
        out.update(torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0), source=identity,
                   cuda_library_setup=materialize_blas_handles(torch, torch.device("cuda:0")))

        def attention(q, k, v):
            # The live worker defaults; do not enable deterministic backward.
            return flash_attn2.flash_attn_func(q, k, v, dropout_p=0.0, causal=False)

        def compare(x, y):
            assert x.shape == y.shape and x.dtype == y.dtype == torch.bfloat16
            # Diagnostic-only CPU-visible metrics, outside all measurements.
            xf, yf = x.float().flatten(), y.float().flatten()
            nx, ny = xf.norm(), yf.norm()
            r = {"finite": bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
                 "nL2": float((xf-yf).norm()/nx.clamp_min(1e-20)),
                 "cosine": float(torch.dot(xf, yf)/(nx*ny).clamp_min(1e-20)),
                 "max_abs": float((xf-yf).abs().max())}
            try:
                torch.testing.assert_close(y, x)
                r["assert_close"] = "PASS"
            except AssertionError as e:
                r.update(assert_close="FAIL", error=str(e)[:1200])
            r["status"] = "PASS" if r["finite"] and r["assert_close"] == "PASS" and r["nL2"] <= 1e-3 and r["cosine"] >= .999999 else "FAIL"
            return r

        for b, nq, nk in shapes:
            gc.collect()
            torch.cuda.empty_cache()  # canary setup only, never a training path
            q = torch.randn((b,nq,12,128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn((b,nk,12,128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            dy = torch.randn_like(q)
            policy = AttentionCheckpointContexts(torch, op, identity)
            row = {"shape_q": list(q.shape), "shape_kv": list(k.shape), "status": "RUNNING"}
            out["cases"].append(row)
            save()
            rng_cpu, rng_gpu = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            y0 = checkpoint(attention, q, k, v, use_reentrant=False)
            g0 = torch.autograd.grad(y0, (q,k,v), dy)
            after_ref = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
            torch.set_rng_state(rng_cpu)
            torch.cuda.set_rng_state(rng_gpu)
            y1 = checkpoint(attention, q, k, v, use_reentrant=False, context_fn=policy)
            g1 = torch.autograd.grad(y1, (q,k,v), dy)
            row["forward_equal"] = bool(torch.equal(y0, y1))
            row["rng_equal"] = bool(torch.equal(after_ref[0],torch.get_rng_state()) and
                                    torch.equal(after_ref[1],torch.cuda.get_rng_state()))
            row["gradients"] = {key: compare(x,y) for key,x,y in zip(("q","k","v"),g0,g1)}
            row["policy"] = policy.receipt()
            assert row["policy"]["original_fa_calls"] == row["policy"]["replay_fa_cache_reads"] == 1
            if not (row["forward_equal"] and row["rng_equal"] and all(x["status"] == "PASS" for x in row["gradients"].values())):
                row["status"] = out["status"] = "FAIL_LOCAL_PARITY"
                save()
                return 2
            del y0, y1, g0, g1

            def execute(use_sac):
                kw = {"context_fn": AttentionCheckpointContexts(torch,op,identity)} if use_sac else {}
                y = checkpoint(attention, q, k, v, use_reentrant=False, **kw)
                return torch.autograd.grad(y, (q,k,v), dy)

            for use_sac in (False, True):
                arm = "candidate" if use_sac else "reference"
                for _ in range(3):
                    execute(use_sac)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                samples = []
                for _ in range(a.repetitions):
                    start, end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    execute(use_sac)
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                row[arm] = {"checkpoint_forward_backward_ms": samples,
                            "median_ms": statistics.median(samples),
                            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
                            "driver_free_bytes": torch.cuda.mem_get_info()[0]}
                # Short separate profiler proves actual CUDA forward elimination.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    execute(use_sac)
                    torch.cuda.synchronize()
                hist = {}
                for event in prof.events():
                    if event.device_type == torch.autograd.DeviceType.CUDA:
                        hist[event.name] = hist.get(event.name, 0) + 1
                row[arm]["cuda_kernel_counts"] = hist
                save()
            row["status"] = "PASS_LOCAL_PARITY"
            print(json.dumps(row), flush=True)
            del q,k,v,dy,policy
            save()
        out["status"] = "PASS_LOCAL_PARITY_ONLY"
        out["end_to_end_speedup"] = "NOT_RUN"
        save()
        return 0
    except Exception:
        out["status"] = "FAIL_CANARY_RUNTIME"
        out["error"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
