#!/usr/bin/env python3
"""Standalone, one-GPU C25 canary; never constructs a training model.

Inputs are seeded synthetic tensors at observed Score shapes. Affine weights
come from the actual checkpoints. The reference class is imported from the
source-pinned Wan installation. This is not an end-to-end parity receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--critic-checkpoint", type=Path, required=True)
    p.add_argument("--teacher-checkpoint", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--expected-reference-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--repetitions", type=int, default=20)
    p.add_argument("--plan-only", action="store_true")
    return p.parse_args()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    args = parse_args()
    shapes = [(b, n, d) for d in (1536, 5120) for b in (1, 2)
              for n in (512, 19320)]
    cases = [(s, 1.0) for s in shapes]
    cases += [((1, 512, d), scale) for d in (1536, 5120) for scale in (1e-3, 100.)]
    out = {"status": "NOT_RUN", "kind": "SYNTHETIC_REAL_SHAPE_REAL_AFFINE_CANARY",
           "production_installed": False, "world16": "NOT_RUN", "seed": args.seed,
           "cases": [], "thresholds": {"normalized_l2_max": 1e-3,
           "cosine_min": 0.999999, "assert_close": "torch BF16 defaults"},
           "planned_shapes_scales": cases, "command": sys.argv,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if args.plan_only:
        print(json.dumps(out, indent=2))
        return 0
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, out)
    try:
        import torch
        from safetensors import safe_open
        from torch.utils.checkpoint import checkpoint
        from wan.modules.model import WanRMSNorm
        from distillschedule.runtime.livewan_score_rmsnorm import build_frozen_score_rmsnorm

        actual_source = Path(inspect.getfile(WanRMSNorm)).resolve()
        expected_file = (args.source_root / "wan/modules/model.py").resolve()
        assert actual_source == expected_file, (actual_source, expected_file)
        class_source = inspect.getsource(WanRMSNorm)
        source_sha = hashlib.sha256(class_source.encode()).hexdigest()
        assert source_sha == args.expected_reference_sha256, (source_sha, args.expected_reference_sha256)
        assert torch.cuda.device_count() == 1, "canary requires exactly one visible owned GPU"
        torch.cuda.set_device(0)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        op, kernel_receipt = build_frozen_score_rmsnorm(torch)
        out.update({"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(0), "reference_file": str(actual_source),
                    "reference_class_sha256": source_sha,
                    "candidate_file_sha256": hashlib.sha256(Path(inspect.getfile(build_frozen_score_rmsnorm)).read_bytes()).hexdigest(),
                    "kernel_receipt": kernel_receipt, "weights": {}})

        weights = {}
        key = "blocks.0.self_attn.norm_q.weight"
        for dim, root in [(1536, args.critic_checkpoint), (5120, args.teacher_checkpoint)]:
            index = root / "diffusion_pytorch_model.safetensors.index.json"
            filename = (json.loads(index.read_text())["weight_map"][key]
                        if index.exists() else "diffusion_pytorch_model.safetensors")
            with safe_open(str(root / filename), framework="pt", device="cpu") as f:
                weight = f.get_tensor(key)
            assert weight.shape == (dim,), (key, weight.shape, dim)
            out["weights"][str(dim)] = {"root": str(root), "file": filename, "key": key,
                "original_dtype": str(weight.dtype), "shape": list(weight.shape),
                "tensor_sha256": hashlib.sha256(weight.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            weights[dim] = weight.to(device="cuda", dtype=torch.bfloat16).requires_grad_(False)

        def compare(a, b):
            result = {"shape": list(a.shape), "dtype": str(a.dtype), "device": str(a.device)}
            assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
            result["finite"] = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            af, bf = a.detach().float().flatten(), b.detach().float().flatten()
            delta = af - bf
            an, bn = torch.linalg.vector_norm(af), torch.linalg.vector_norm(bf)
            result.update({"normalized_l2": float(torch.linalg.vector_norm(delta) / an.clamp_min(1e-20)),
                           "cosine": float(torch.dot(af, bf) / (an * bn).clamp_min(1e-20)),
                           "max_abs": float(delta.abs().max()),
                           "equal_fraction": float((a == b).float().mean())})
            try:
                torch.testing.assert_close(b, a)
                result["assert_close"] = "PASS"
            except AssertionError as e:
                result["assert_close"] = "FAIL"
                result["assert_close_error"] = str(e)[:1500]
            result["status"] = "PASS" if (result["finite"] and result["assert_close"] == "PASS"
                and result["normalized_l2"] <= 1e-3 and result["cosine"] >= 0.999999) else "FAIL"
            return result

        def measure(fn):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            timings = []
            for _ in range(args.repetitions):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
            return {"samples_ms": timings, "median_ms": statistics.median(timings)}

        torch.cuda.reset_peak_memory_stats()
        for shape, scale in cases:
            row = {"shape": shape, "input_scale": scale, "status": "RUNNING"}
            out["cases"].append(row)
            d = shape[-1]
            ref = WanRMSNorm(d, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
            ref.weight = torch.nn.Parameter(weights[d], requires_grad=False)
            x = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * scale).requires_grad_(True)
            dy = torch.randn_like(x)
            y_ref, y_new = ref(x), op(x, ref.weight, ref.eps)
            dx_ref, = torch.autograd.grad(y_ref, x, dy, retain_graph=True)
            dx_new, = torch.autograd.grad(y_new, x, dy, retain_graph=True)
            row["forward"] = compare(y_ref, y_new)
            row["input_gradient"] = compare(dx_ref, dx_new)
            # Non-reentrant replay; grad() must work, including saved-tensor recomputation.
            z_ref = checkpoint(ref, x, use_reentrant=False)
            z_new = checkpoint(lambda v: op(v, ref.weight, ref.eps), x, use_reentrant=False)
            dg_ref, = torch.autograd.grad(z_ref, x, dy)
            dg_new, = torch.autograd.grad(z_new, x, dy)
            row["checkpoint_gradient"] = compare(dg_ref, dg_new)
            row["status"] = "PASS" if all(row[k]["status"] == "PASS" for k in
                ("forward", "input_gradient", "checkpoint_gradient")) else "FAIL"
            if row["status"] != "PASS":
                out["status"] = "FAIL_LOCAL_PARITY"
                atomic_json(args.output, out)
                print(json.dumps(row), flush=True)
                return 2
            if scale == 1.0:
                with torch.no_grad():
                    row["reference_forward"] = measure(lambda: ref(x))
                    row["candidate_forward"] = measure(lambda: op(x, ref.weight, ref.eps))
                row["reference_backward"] = measure(lambda: torch.autograd.grad(y_ref, x, dy, retain_graph=True))
                row["candidate_backward"] = measure(lambda: torch.autograd.grad(y_new, x, dy, retain_graph=True))
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            row["driver_free_bytes"] = torch.cuda.mem_get_info()[0]
            atomic_json(args.output, out)
            print(json.dumps(row), flush=True)
            del x, dy, y_ref, y_new, dx_ref, dx_new, z_ref, z_new, dg_ref, dg_new, ref
        out["status"] = "PASS_LOCAL_PARITY_ONLY"
        out["end_to_end_speedup"] = "NOT_RUN"
        atomic_json(args.output, out)
        return 0
    except Exception:
        out["status"] = "FAIL_CANARY_RUNTIME"
        out["error"] = traceback.format_exc()
        atomic_json(args.output, out)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
