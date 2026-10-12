#!/usr/bin/env python3
"""One owned GPU, real rank0 B1 trace: merged shadow capacity/parity probe.

This does not execute a training update and is NOT a Full5 speed measurement.
The launcher must hold the existing node lock across audit and teardown.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import timedelta
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--bootstrap-hash", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--trace-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {"status": "FAIL", "probe_kind": "ONE_GPU_REAL_B1_COMPONENT_NOT_FULL5",
              "training_updates": "NOT_RUN", "profiled": False,
              "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    started = time.perf_counter()
    process_group = False
    try:
        import torch
        from distillschedule.runtime.p0_parity import tensor_parity_metrics
        from distillschedule.runtime.wan13b_upper_bound import Wan13BBootstrap
        from distillschedule.runtime.wan13b_rollout_shadow import ENV, PLACEMENT
        from runtime.static_world16_trace import StaticWorld16Trace
        from runtime.static_fsdp2_p0_helios import run_generator_trace_rollout_batch

        spec = importlib.util.spec_from_file_location("shadow_probe_fair", ROOT / "scripts/distillschedule/probe_wan13b_fair_world16.py")
        fair = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = fair
        spec.loader.exec_module(fair)
        if os.environ.get(ENV) != "1":
            raise ValueError("probe requires explicit shadow opt-in")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        rendezvous = Path("/tmp/lyy-experiments/DMD-System/wan13b_target2x_20261012_r1")
        rendezvous.mkdir(parents=True, exist_ok=True)
        torch.distributed.init_process_group(
            "nccl", init_method=f"file://{rendezvous / ('shadow_probe_' + uuid.uuid4().hex)}",
            rank=0, world_size=1, timeout=timedelta(seconds=60),
        )
        process_group = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        bootstrap = Wan13BBootstrap.load(args.bootstrap, expected_hash=args.bootstrap_hash)
        trace = StaticWorld16Trace(args.trace, variant="no-gan")
        if trace.bundle_hash != args.trace_hash:
            raise ValueError("trace hash mismatch")
        result.update(bootstrap_hash=bootstrap.manifest_hash, trace_hash=trace.bundle_hash,
                      environment={"torch": torch.__version__, "cuda": torch.version.cuda,
                                   "tf32_override": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
                                   "gpu": torch.cuda.get_device_name(device)})
        build_args = argparse.Namespace(placement=PLACEMENT, model_root=args.model_root,
                                      timing_mode=True, profile_mode=False,
                                      trajectory_mode=False, validate_only=False)
        model, build = fair._build_model_fair(role="generator", args=build_args,
                                            bootstrap=bootstrap, device=device)
        shadow = model._distillschedule_merged_rollout_shadow

        def fingerprint():
            digest = hashlib.sha256()
            for name, p in model.named_parameters():
                digest.update(name.encode())
                digest.update(str(p.dtype).encode())
                digest.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
            return digest.hexdigest()

        source_before = fingerprint()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        minimum_driver_free = torch.cuda.mem_get_info(device)[0]
        row = trace.load_critic_entry(rank=0, step=10, device=device)
        cache = {}
        rng_cpu = torch.get_rng_state().clone()
        rng_cuda = torch.cuda.get_rng_state(device).clone()
        with torch.no_grad():
            # Both arms warm exactly this immutable real trace row once.
            reference_warm, ref_record = run_generator_trace_rollout_batch(
                model=model, loaded_entries=(row,), device=device,
                clean_runtime=True, runtime_cache=cache)
            minimum_driver_free = min(minimum_driver_free, torch.cuda.mem_get_info(device)[0])
            torch.cuda.synchronize(device)
            merge_start = time.perf_counter()
            # G2 is the unchanged bootstrap, not a fabricated G3 training commit.
            shadow.begin_rollout(version=2, logical_steps=(10,))
            torch.cuda.synchronize(device)
            merge_wall = time.perf_counter() - merge_start
            try:
                candidate_warm, candidate_record = run_generator_trace_rollout_batch(
                    model=shadow, loaded_entries=(row,), device=device,
                    clean_runtime=True, runtime_cache=cache)
            finally:
                shadow.end_rollout()
            torch.cuda.synchronize(device)
            minimum_driver_free = min(minimum_driver_free, torch.cuda.mem_get_info(device)[0])
            component_times = {}
            outputs = {}
            records = {}
            for name, execution in (("reference", model), ("merged_shadow", shadow)):
                torch.cuda.synchronize(device)
                begin = time.perf_counter()
                if execution is shadow:
                    shadow.begin_rollout(version=2, logical_steps=(10,))
                try:
                    outputs[name], records[name] = run_generator_trace_rollout_batch(
                        model=execution, loaded_entries=(row,), device=device,
                        clean_runtime=True, runtime_cache=cache)
                finally:
                    if execution is shadow:
                        shadow.end_rollout()
                torch.cuda.synchronize(device)
                component_times[name] = time.perf_counter() - begin
                minimum_driver_free = min(minimum_driver_free, torch.cuda.mem_get_info(device)[0])
        parity = tensor_parity_metrics(outputs["reference"][0], outputs["merged_shadow"][0])
        source_after = fingerprint()
        if not torch.equal(rng_cpu, torch.get_rng_state()) or not torch.equal(rng_cuda, torch.cuda.get_rng_state(device)):
            raise AssertionError("component probe changed RNG state")
        if source_after != source_before:
            raise AssertionError("merged refresh mutated training master")
        for name, output in outputs.items():
            if list(output[0].shape) != [1, 16, 9, 48, 80] or output[0].dtype != torch.bfloat16:
                raise AssertionError(f"{name}: output wire contract changed")
        result.update(
            source_parameter_sha256_before=source_before,
            source_parameter_sha256_after=source_after,
            source_unchanged=True, rng_unchanged=True,
            bootstrap_generator_version=2, committed_G3="NOT_RUN_COMPONENT_PROBE",
            row={"rank": 0, "step": 10, "identity": row.pool_entry.identity_sha256},
            reference_record=ref_record, candidate_record=candidate_record,
            warm_component_times_seconds=component_times,
            initial_merge_wall_seconds=merge_wall,
            parity=parity, shadow_receipt=shadow.receipt, build=build,
            memory={"peak_allocated_gib": torch.cuda.max_memory_allocated(device)/2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/2**30,
                    "minimum_driver_free_observed_gib": minimum_driver_free/2**30},
            minimum_driver_free_scope="SAMPLED_AT_COMPONENT_BOUNDARIES_NOT_CONTINUOUS",
        )
        result["status"] = "PASS" if parity["status"] == "PASS" and minimum_driver_free >= 2*2**30 else "FAIL"
    except BaseException as error:
        result.update(error=str(error), traceback=traceback.format_exc())
    finally:
        if process_group:
            torch.distributed.destroy_process_group()
    result["probe_wall_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result.get(k) for k in ("status", "parity", "memory", "warm_component_times_seconds", "initial_merge_wall_seconds", "error")}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
