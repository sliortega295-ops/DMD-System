#!/usr/bin/env python3
"""Opt-in B1 weak scaling, reusing the admitted Wan two-cycle workload body.

The canonical worker/defaults remain world16. This entry requires a separate
rank8/rank16 trace admission, and never borrows a world16 numerical PASS for
world8. G1/C5/EMA5, BF16 backbone/FP32 LoRA, optimizer math, and all winner
mechanisms are unchanged. The only search dimension here is GPU count.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch
import torch.distributed as dist

from distillschedule.runtime.wan13b_execution import WanDistributedExecution
from distillschedule.runtime.wan13b_fair import build_fair_engine


def shared_worker():
    path = ROOT / "scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py"
    spec = importlib.util.spec_from_file_location("wan_weakscale_shared_native", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--world-size", type=int, choices=(8, 16), required=True)
    p.add_argument("--placement", choices=("world16-fsdp", "resident-async128-tree-chunk2048"), required=True)
    p.add_argument("--bootstrap", type=Path, required=True)
    p.add_argument("--expected-bootstrap-hash", required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--trace-bundle", type=Path, required=True)
    p.add_argument("--expected-trace-hash", required=True)
    p.add_argument("--negative-prompt", type=Path, required=True)
    p.add_argument("--timing-seed", type=int, default=20260903)
    p.add_argument("--trace-admission", type=Path, required=True)
    p.add_argument("--expected-trace-admission-hash", required=True)
    p.add_argument("--validate-only", action="store_true")
    return p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, indent=2) + "\n")
    os.replace(tmp, path)


def trace_contract(native, trace, execution):
    return [dict(operation=operation, inventory=native.expected_trace_rollout_call_inventory(
        trace.rank_control_validation, operation=operation,
        expected_rank_count=execution.world_size,
    )) for operation in native._rollout_operations()]


def build_engines(native, args, bootstrap, device):
    """Same existing fair model, Adam materialization and EMA factories."""
    modules = {}
    builds = {}
    engines = {}
    for role, lr in (("generator", 2e-6), ("critic", 4e-7)):
        module, model_build = native.fair._build_model_fair(
            role=role, args=args, bootstrap=bootstrap, device=device,
        )
        dist.barrier()  # setup only; identical original construction boundary
        modules[role] = module
        engine, engine_build = build_fair_engine(
            module=module, placement=args.placement,
            physical_step=int(bootstrap.component(role)["optimizer_step"]),
            lr=lr, betas=(0.0, 0.999), eps=1e-8, weight_decay=1e-3,
            audit_gradients=False, execution=args.distributed_execution,
        )
        engines[role] = engine
        builds[role] = dict(model_build, engine=engine_build)
        if engine.dp_world_size != args.world_size or engine.train_batch_size() != args.world_size:
            raise AssertionError("engine gradient divisor/global sample accounting mismatch")
    ema = native.fair._ema_factory(
        modules["generator"],
        optimization_step=int(bootstrap.manifest["ema"]["optimization_step"]),
        decay=float(bootstrap.manifest["ema"]["decay"]),
        update_after_step=int(bootstrap.manifest["ema"]["update_after_step"]),
    )
    builds["static_residency"] = dict(
        placement=args.placement, both_backbones_present_for_full5=True,
        parameter_offload=False, activation_offload=False, optimizer_offload=False,
        frozen_parameter_streaming=False, role_transition=False, model_swap=False,
        residual_storage="cuda_out_of_place", ema_device="cuda",
    )
    builds["memory_after_optimizer_and_ema"] = native.base._memory(device)
    return engines["generator"], engines["critic"], ema, builds


def main():
    args = parser().parse_args()
    validate_only = args.validate_only
    args.distributed_execution = WanDistributedExecution(
        world_size=args.world_size, experiment="weak_scaling",
    )
    execution = args.distributed_execution
    if subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() != args.expected_head:
        raise ValueError("exact source HEAD changed")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip():
        raise ValueError("weak-scaling source must be clean")
    if args.timing_seed != 20260903:
        raise ValueError("this weak-scaling experiment freezes the existing timing seed")
    if sha(args.trace_admission) != args.expected_trace_admission_hash:
        raise ValueError("trace admission identity changed")
    admission = json.loads(args.trace_admission.read_text())
    if (admission["world_size"] != execution.world_size
            or admission["global_batch"] != execution.global_batch
            or admission["bundle_hash"] != args.expected_trace_hash
            or admission["retained_source_ranks"] != list(range(execution.world_size))
            or admission["tensor_payloads_changed"] is not False):
        raise ValueError("rank-qualified weak-scaling trace admission mismatch")
    if os.environ.get("DISTILLSCHEDULE_WAN13B_GENERATOR_ROLLOUT_MERGED_SHADOW", "0") != "0":
        raise ValueError("systemically biased merged-LoRA is excluded")
    if os.environ.get("DISTILLSCHEDULE_STATIC_FSDP2_NSYS_EXACT_RANGE", "0") not in ("", "0"):
        raise ValueError("weak-scaling timing must be unprofiled")
    from distillschedule.runtime.wan13b_upper_bound import Wan13BBootstrap
    from runtime.static_world16_trace import StaticWorld16Trace
    bootstrap = Wan13BBootstrap.load(args.bootstrap, expected_hash=args.expected_bootstrap_hash,
                                     verify_payloads=validate_only)
    if Path(bootstrap.manifest["model_root"]).resolve() != args.model_root.resolve():
        raise ValueError("bootstrap model identity mismatch")
    trace = StaticWorld16Trace(args.trace_bundle, variant="no-gan", execution=execution)
    if trace.bundle_hash != args.expected_trace_hash:
        raise ValueError("weak-scaling trace bundle changed")
    native = shared_worker()
    native._configure_shared_mode_args(args)
    arm = native._validate_arm_environment(args.placement)
    if arm["generator_rollout_merged_shadow"]:
        raise ValueError("merged weights may not participate in this experiment")
    rollout_contract = trace_contract(native, trace, execution)
    source_files = {
        str(Path(__file__).relative_to(ROOT)): sha(__file__),
        "shared_native_worker": sha(ROOT / "scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py"),
        "typed_execution": sha(ROOT / "src/distillschedule/runtime/wan13b_execution.py"),
    }
    row = dict(
        schema="distillschedule.wan13b_weakscale_native_c2.v1",
        status="PASS_CPU_WEAKSCALE_ADMISSION_GPU_NOT_RUN" if validate_only else "RUNNING",
        run_id=args.run_id, source_head=args.expected_head, hostname=socket.gethostname(),
        execution=execution.receipt(), placement=args.placement,
        reported_placement=(f"world{execution.world_size}-fsdp" if args.placement == native.FSDP_BASELINE else args.placement),
        arm_contract=arm, source_files=source_files,
        bootstrap_hash=bootstrap.manifest_hash,
        model_identity_sha256=bootstrap.manifest["model_identity_sha256"],
        trace_bundle_hash=trace.bundle_hash, trace_binding_hash=trace.binding_hash,
        trace_admission_sha256=sha(args.trace_admission), trace_admission=admission,
        rank_control_validation=trace.rank_control_validation,
        trace_rollout_contract=rollout_contract, seed=args.timing_seed,
        numerical_status="DEFERRED_BY_USER_NOT_A_BORROWED_WORLD16_PASS",
        profiler="NOT_RUN", merged_lora_count=0, defaults_changed=False,
        timer_contract="ONE_INIT_C1_WARM_C2_CLAIMABLE_HOST_FENCED_EXTRACTION_AFTER_TIMERS",
    )
    if validate_only:
        assert not torch.cuda.is_initialized()
        row.update(torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                   cuda_initialized=False)
        print(json.dumps(row))
        return 0
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    execution.validate_live_world(dist.get_world_size())
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    runtime_seed = args.timing_seed + rank
    random.seed(runtime_seed); torch.manual_seed(runtime_seed); torch.cuda.manual_seed_all(runtime_seed)
    row.update(rank=rank, local_rank=local_rank, rank_runtime_seed=runtime_seed)
    output = args.output_dir / "ranks" / f"rank_{rank:03d}.json"
    try:
        negative = native.NegativePromptAsset.load(args.negative_prompt).embedding.to(device=device, dtype=torch.bfloat16)
        construction_start = time.perf_counter()
        generator, critic, ema, build = build_engines(native, args, bootstrap, device)
        torch.cuda.synchronize(device)
        row["construction_seconds"] = time.perf_counter() - construction_start
        row["build"] = build
        row["runtime_contract"] = native._validate_build_contract(build, placement=args.placement, ema=ema)
        for role in ("generator", "critic"):
            if build[role]["placement"]["world_size"] != execution.world_size:
                raise AssertionError("actual parameter placement world-size mismatch")
        if (generator.global_steps, critic.global_steps, ema.optimization_step) != (2, 10, 10):
            raise AssertionError("initial G/C/EMA versions changed")
        clean = native._clean_runtime(args.placement)
        cache = {} if clean else None
        policy = native.StaticFSDP2P0Policy(mesh_policy="world16-fsdp", rollout_batch_size=1,
                                           pack_real_score_pair=False, rollout_backend="native-fsdp2-batch")
        # The historical policy string labels the B1 rollout helper, not the
        # actual mesh. The typed execution above binds actual shard/reducer size.
        diagnostics = native.StaticProfileDiagnostics(enabled=False, rank=rank,
                            output_path=None, collect_cpu_diagnostics=False)
        deferred = []
        rows = []
        raws = []
        slices = []
        torch.cuda.reset_peak_memory_stats(device)
        for cycle in (1, 2):
            dist.barrier(); torch.cuda.synchronize(device)
            check_start = len(deferred)
            start_ns = time.time_ns(); started = time.perf_counter()
            raw = native._run_workload_cycle(
                rank=rank, device=device, generator_engine=generator, critic_engine=critic,
                ema=ema, trace=trace, negative=negative, diagnostics=diagnostics, policy=policy,
                clean_runtime=clean, runtime_cache=cache, deferred_finite_checks=deferred,
            )
            torch.cuda.synchronize(device); dist.barrier()
            elapsed = time.perf_counter() - started; stop_ns = time.time_ns()
            slices.append((check_start, len(deferred))); raws.append(raw)
            rows.append(dict(cycle=cycle, wall_seconds=elapsed, start_unix_ns=start_ns,
                             stop_unix_ns=stop_ns, role="WARMUP" if cycle == 1 else "STEADY_STATE",
                             versions=dict(generator=int(generator.global_steps), critic=int(critic.global_steps),
                                           ema=int(ema.optimization_step))))
        # All host-visible finite checks, census and JSON are outside both timers.
        finite = [bool(torch.stack(deferred[a:b]).all().detach().cpu()) for a, b in slices]
        if finite != [True, True]:
            raise FloatingPointError(f"weak-scaling finite gate: {finite}")
        expected_versions = [dict(generator=3, critic=15, ema=15), dict(generator=4, critic=20, ema=20)]
        if [r["versions"] for r in rows] != expected_versions:
            raise AssertionError("weak-scaling optimizer/EMA count changed")
        census = [native._cycle_census(raw, cycle=i, trace_rollout_contract=rollout_contract)
                  for i, raw in enumerate(raws, 1)]
        row["fused_epilogue_receipt"] = native._validate_fused_receipt(build, placement=args.placement)
        from distillschedule.runtime.wan13b_lora_grad_epilogue import validate_grad_epilogue_receipts
        from distillschedule.runtime.wan13b_norm_policy import validate as validate_norm
        from distillschedule.runtime.wan13b_rollout_cuda_graph import validate as validate_rollout_graph
        from distillschedule.runtime.wan13b_rollout_shadow import validate as validate_shadow
        from distillschedule.runtime.wan13b_cast_graph_composition import validate_build as validate_cast
        from distillschedule.runtime.wan13b_critic_grad_graph import validate_build as validate_critic_graph
        from distillschedule.runtime.wan13b_score_cuda_graph import validate_build as validate_score_graph
        row.update(
            grad_epilogue_receipt=validate_grad_epilogue_receipts(build, requested=arm["grad_lora_fused_epilogue"]),
            modulated_norm_fusion_receipt=validate_norm(build, enabled=arm["modulated_norm_fusion"]),
            generator_rollout_cuda_graph_receipt=validate_rollout_graph(build, enabled=arm["generator_rollout_cuda_graph"], cycles=2),
            generator_rollout_merged_shadow_receipt=validate_shadow(build, enabled=False, cycles=2),
            cast_graph_composition_receipt=validate_cast(build, enabled=arm["generator_nograd_lora_cast_cache"]),
            critic_grad_cuda_graph_receipt=validate_critic_graph(build, cycles=2, reference_checks=False),
            score_cuda_graph_receipt=validate_score_graph(build, cycles=2, reference_checks=False),
            status="PASS_NATIVE_WEAKSCALE_CONTRACTS", cycles=rows, cycle_census=census,
            finite_status=finite, final_versions=expected_versions[-1],
            memory=native.base._memory(device),
            engine_gradient_divisors={"generator": generator.dp_world_size, "critic": critic.dp_world_size},
            engine_global_samples={"generator": generator.global_samples, "critic": critic.global_samples},
            evidence_extraction="AFTER_BOTH_TIMED_WINDOWS",
        )
        atomic(output, row); dist.barrier(); return 0
    except BaseException as error:
        row.update(status="CUDA_OOM" if isinstance(error, torch.cuda.OutOfMemoryError) else "FAIL",
                   error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        try: row["memory_failure"] = native.base._memory(device)
        except BaseException: pass
        atomic(output, row)
        raise
    finally:
        if dist.is_initialized():
            try: dist.destroy_process_group()
            except BaseException: pass


if __name__ == "__main__":
    raise SystemExit(main())
