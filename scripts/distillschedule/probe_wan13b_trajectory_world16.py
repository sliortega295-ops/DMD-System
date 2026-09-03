#!/usr/bin/env python3
"""Run an append-only, continuous Wan1.3B coarse trajectory diagnostic.

This is deliberately a separate worker from the historical one-cycle
correctness/timing path.  It constructs Generator, Critic, optimizers, and EMA
once, then executes ``G1 + C5 + EMA5`` for every requested cycle without a
state reset.  The immutable five-step trace cohort is replayed intentionally;
the receipt says so and gives every replay a cycle-qualified identity.

The run is diagnostic-only.  Per-cycle scalar extraction and four state-anchor
reductions are not a performance timing path, and normal numerical drift never
stops the run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_FAIR_PATH = ROOT / "scripts/distillschedule/probe_wan13b_fair_world16.py"
_SPEC = importlib.util.spec_from_file_location("wan13b_trajectory_fair_worker", _FAIR_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot import {_FAIR_PATH}")
fair = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fair
_SPEC.loader.exec_module(fair)
base = fair.base
p0 = base.p0

from distillschedule.evidence.trajectory_coarse import (  # noqa: E402
    AppendOnlyTrajectoryWriter,
    TrajectoryCycleReceipt,
)
from distillschedule.evidence.trajectory_state_sketch import (  # noqa: E402
    canonical_state_name,
    collect_numeric_state_sketch,
)
from distillschedule.profiling.static_profile_diagnostics import (  # noqa: E402
    StaticProfileDiagnostics,
)
from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy  # noqa: E402
from runtime.gst_pool_contract import NegativePromptAsset  # noqa: E402
from runtime.static_world16_trace import StaticWorld16Trace  # noqa: E402


WORLD_SIZE = 16
SOURCE_STEPS = tuple(range(10, 15))
ANCHOR_CYCLES = frozenset({1, 10, 25, 50})
REGISTERED_CYCLE_COUNTS = frozenset({1, 5, 50})
RESIDENT_WINNER = "resident-async128-tree-chunk2048"
FSDP_BASELINE = "world16-fsdp"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-id", required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--placement", choices=fair.FAIR_PLACEMENTS, required=True)
    value.add_argument("--bootstrap", type=Path, required=True)
    value.add_argument("--expected-bootstrap-hash", required=True)
    value.add_argument("--model-root", type=Path, required=True)
    value.add_argument("--trace-bundle", type=Path, required=True)
    value.add_argument("--expected-trace-hash", required=True)
    value.add_argument("--negative-prompt", type=Path, required=True)
    value.add_argument("--trajectory-cycles", type=int, required=True)
    value.add_argument("--trajectory-seed", type=int, required=True)
    value.add_argument("--trajectory-arm-id", required=True)
    value.add_argument("--validate-only", action="store_true")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def _local_tensor(value: Any) -> torch.Tensor:
    return value.detach().to_local() if hasattr(value, "to_local") else value.detach()


def _aggregate_state_metrics(
    values: Iterable[Any], *, sharded: bool, device: torch.device
) -> dict[str, Any]:
    """Small exact moment receipt without materializing full FSDP tensors."""

    tensors = [value for value in values if value is not None]
    if not tensors:
        return {"status": "EMPTY", "tensor_count": 0, "numel": 0}
    moments = torch.zeros(4, dtype=torch.float64, device=device)
    count = torch.zeros(1, dtype=torch.int64, device=device)
    for index, raw in enumerate(tensors):
        value = _local_tensor(raw).double()
        moments[0] += value.sum()
        moments[1] += value.square().sum()
        moments[2] = torch.maximum(moments[2], value.abs().max())
        moments[3] += value.sum() * float((index % 97) + 1)
        count += int(value.numel())
    if sharded:
        dist.all_reduce(moments[:2], op=dist.ReduceOp.SUM)
        dist.all_reduce(moments[2:3], op=dist.ReduceOp.MAX)
        dist.all_reduce(moments[3:], op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    cpu = moments.cpu().tolist()
    return {
        "status": "PASS",
        "tensor_count": len(tensors),
        "numel": int(count.cpu()),
        "sum": cpu[0],
        "l2": math.sqrt(max(cpu[1], 0.0)),
        "max_abs": cpu[2],
        "weighted_sum": cpu[3],
        "representation": "GLOBAL_FROM_LOCAL_SHARDS" if sharded else "FULL_REPLICA",
    }


def _named_trainable_values(engine: Any) -> list[tuple[str, Any, Any | None]]:
    return [
        (canonical_state_name(name), parameter, None)
        for name, parameter in engine.module.named_parameters()
        if parameter.requires_grad
    ]


def _optimizer_named_values(engine: Any) -> list[tuple[str, Any, Any | None]]:
    names = {
        id(parameter): canonical_state_name(name)
        for name, parameter in engine.module.named_parameters()
        if parameter.requires_grad
    }
    result: list[tuple[str, Any, Any | None]] = []
    for group in engine.optimizer.param_groups:
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None:
                raise AssertionError("optimizer parameter is absent from named trainables")
            state = engine.optimizer.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key not in state:
                    continue
                value = state[key]
                layout_source = None if hasattr(value, "placements") else parameter
                result.append((f"{name}/{key}", value, layout_source))
    return result


def _ema_named_values(
    ema: Any, *, generator_engine: Any
) -> list[tuple[str, Any, Any | None]]:
    shadows = list(getattr(ema, "_shadows", ()))
    parameters = list(getattr(ema, "_parameters", ()))
    if shadows and len(shadows) == len(parameters):
        names = {
            id(parameter): canonical_state_name(name)
            for name, parameter in generator_engine.module.named_parameters()
            if parameter.requires_grad
        }
        explicit_names = list(getattr(ema, "_names", ()))
        result: list[tuple[str, Any, Any | None]] = []
        for index, (shadow, parameter) in enumerate(
            zip(shadows, parameters, strict=True)
        ):
            name = (
                canonical_state_name(explicit_names[index])
                if explicit_names
                else names.get(id(parameter))
            )
            if name is None:
                raise AssertionError("EMA parameter is absent from Generator trainables")
            result.append((f"{name}/ema", shadow, parameter))
        return result
    if hasattr(ema, "_shadow"):
        raise RuntimeError(
            "flat-sharded EMA numeric coordinates are not implemented for this "
            "trajectory arm; use resident winner or world16-fsdp"
        )
    raise AssertionError("EMA exposes no numeric state")


def _raw_values(rows: Iterable[tuple[str, Any, Any | None]]) -> list[Any]:
    return [value for _name, value, _layout in rows]


def _state_anchor(
    *,
    cycle: int,
    rank: int,
    placement: str,
    output_dir: Path,
    generator_engine: Any,
    critic_engine: Any,
    ema: Any,
    device: torch.device,
) -> tuple[dict[str, str], dict[str, Any]]:
    sharded = placement == fair.FSDP_PLACEMENT
    named = {
        "generator_trainable": _named_trainable_values(generator_engine),
        "critic_trainable": _named_trainable_values(critic_engine),
        "generator_adam": _optimizer_named_values(generator_engine),
        "critic_adam": _optimizer_named_values(critic_engine),
        "generator_ema": _ema_named_values(ema, generator_engine=generator_engine),
    }
    values: dict[str, Any] = {}
    for component, rows in named.items():
        values[component] = {
            "global_moments": _aggregate_state_metrics(
                _raw_values(rows), sharded=sharded, device=device
            ),
            "numeric_sketch": collect_numeric_state_sketch(
                torch=torch,
                named_values=rows,
                samples_per_tensor=16,
            ),
        }
    anchor = {
        "schema_version": "distillschedule.wan13b_trajectory_state_anchor.v2",
        "cycle": cycle,
        "writer_rank": 0,
        "placement": placement,
        "components": values,
        "comparison_scope": (
            "align each component numeric_sketch by (tensor, flat_index); "
            "the resulting FP32 vectors support cosine and normalized-L2"
        ),
    }
    anchor_hash = _canonical_hash(anchor)
    anchor["anchor_sha256"] = anchor_hash
    path = output_dir / "trajectory" / "anchors" / f"cycle_{cycle:02d}_global.json"
    if rank == 0:
        _atomic_json(path, anchor)
    dist.barrier()
    fingerprints = {
        key: _canonical_hash(metric) for key, metric in values.items()
    }
    fingerprints["anchor"] = anchor_hash
    return fingerprints, {"path": str(path), "sha256": anchor_hash}


def _clean_runtime(placement: str) -> bool:
    return placement in {
        "resident-clean",
        "resident-async128",
        "resident-async128-tree",
        "resident-async128-tree-chunk2048",
        "resident-async128-tree-chunk2048-compile",
        "resident-async256",
        "resident-local-fast",
        "resident-flat256",
        "resident-flat512",
        "resident-lora-zero2",
        "resident-rollout-fused",
    }


def main() -> int:
    args = parser().parse_args()
    # The shared construction helper only branches on these legacy execution
    # flags.  Trajectory is its own diagnostic mode and uses neither path.
    args.timing_mode = False
    args.profile_mode = False
    args.trajectory_mode = True
    if args.trajectory_cycles not in REGISTERED_CYCLE_COUNTS:
        raise ValueError("trajectory cycles must be one of 1, 5, or 50")
    if args.trajectory_seed < 0:
        raise ValueError("trajectory seed must be non-negative")
    if args.placement not in {RESIDENT_WINNER, FSDP_BASELINE}:
        raise ValueError("trajectory placement is not a registered comparison arm")
    fused_epilogue = (
        os.environ.get(
            "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE", "0"
        )
        == "1"
    )
    cast_cache = (
        os.environ.get(
            "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE", "0"
        )
        == "1"
    )
    if args.placement == RESIDENT_WINNER and not fused_epilogue:
        raise ValueError("Resident trajectory is not the registered fused-epilogue winner")
    if args.placement == RESIDENT_WINNER and cast_cache:
        raise ValueError("Resident winner cannot enable the rejected cast cache")
    if args.placement == FSDP_BASELINE and (fused_epilogue or cast_cache):
        raise ValueError("world16-FSDP baseline cannot enable Resident-only LoRA arms")
    if not args.run_id.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("unsafe run id")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != WORLD_SIZE:
        raise AssertionError(f"Wan1.3B trajectory requires world16, got {world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    runtime_seed = args.trajectory_seed + rank
    random.seed(runtime_seed)
    torch.manual_seed(runtime_seed)
    torch.cuda.manual_seed_all(runtime_seed)
    output = args.output_dir / "ranks" / f"rank_{rank:02d}.json"
    result: dict[str, Any] = {
        "schema_version": "distillschedule.wan13b_trajectory_rank.v1",
        "status": "RUNNING",
        "run_id": args.run_id,
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "world_size": world,
        "placement": args.placement,
        "trajectory_cycles": args.trajectory_cycles,
        "trajectory_seed": args.trajectory_seed,
        "rank_runtime_seed": runtime_seed,
        "trajectory_arm_id": args.trajectory_arm_id,
        "diagnostic_only": True,
        "formal_timing_eligible": False,
        "arm_contract": (
            "RESIDENT_ASYNC128_TREE_CHUNK2048_FUSED_EPILOGUE"
            if args.placement == RESIDENT_WINNER
            else "WORLD16_FSDP_BASELINE"
        ),
    }
    try:
        contract = base.contract_for_variant("no-gan")
        bootstrap = base.Wan13BBootstrap.load(
            args.bootstrap,
            expected_hash=args.expected_bootstrap_hash,
            verify_payloads=rank == 0,
        )
        trace = StaticWorld16Trace(args.trace_bundle, variant="no-gan")
        if trace.bundle_hash != args.expected_trace_hash:
            raise ValueError("trace bundle identity changed")
        if Path(bootstrap.manifest["model_root"]).resolve() != args.model_root.resolve():
            raise ValueError("bootstrap/model-root path identity changed")
        result["identities"] = {
            "bootstrap_hash": bootstrap.manifest_hash,
            "trace_bundle_hash": trace.bundle_hash,
            "trace_binding_hash": trace.binding_hash,
            "trace_replay_policy": "FIXED_FIVE_STEP_COHORT_REPLAY",
        }
        result["contract"] = {
            "source_logical_steps": list(SOURCE_STEPS),
            "cycles": args.trajectory_cycles,
            "global_batch": 16,
            "updates_per_cycle": {"generator": 1, "critic": 5, "ema": 5},
            "expected_final_versions": {
                "generator": int(contract.initial_versions["generator"])
                + args.trajectory_cycles,
                "critic": int(contract.initial_versions["critic"])
                + 5 * args.trajectory_cycles,
                "ema": int(contract.initial_versions["ema"])
                + 5 * args.trajectory_cycles,
            },
        }
        if args.validate_only:
            result.update(status="PASS", validate_only=True, gpu_workload="NOT_RUN")
            _atomic_json(output, result)
            dist.barrier()
            return 0

        negative = NegativePromptAsset.load(args.negative_prompt).embedding.to(
            device=device, dtype=torch.bfloat16
        )
        generator_engine, critic_engine, ema, build = base._build_engines(
            args=args, bootstrap=bootstrap, device=device
        )
        generator_engine._trajectory_capture_grad_norm = True
        critic_engine._trajectory_capture_grad_norm = True
        clean_runtime = _clean_runtime(args.placement)
        runtime_cache: dict[str, Any] | None = {} if clean_runtime else None
        policy = StaticFSDP2P0Policy(
            mesh_policy="world16-fsdp",
            rollout_batch_size=1,
            pack_real_score_pair=False,
            rollout_backend="native-fsdp2-batch",
        )
        diagnostics = StaticProfileDiagnostics(
            enabled=False,
            rank=rank,
            output_path=None,
            collect_cpu_diagnostics=False,
        )
        writer = AppendOnlyTrajectoryWriter(
            args.output_dir / "trajectory" / f"rank_{rank:02d}.jsonl",
            run_id=args.run_id,
            rank=rank,
        )
        cycle_summaries: list[dict[str, Any]] = []
        initial_g = int(generator_engine.global_steps)
        initial_c = int(critic_engine.global_steps)
        initial_ema = int(ema.optimization_step)
        expected_initial = dict(contract.initial_versions)
        if (initial_g, initial_c, initial_ema) != (
            expected_initial["generator"],
            expected_initial["critic"],
            expected_initial["ema"],
        ):
            raise AssertionError("trajectory bootstrap versions changed")
        result["build"] = build
        result["initial_memory"] = base._memory(device)

        dist.barrier()
        started = time.perf_counter()
        for cycle in range(1, args.trajectory_cycles + 1):
            cycle_started = time.perf_counter()
            deferred_finite_checks: list[torch.Tensor] = []
            critic_entries: dict[int, Any] = {}
            generator_record = p0._run_nogan_generator_update_p0(
                rank=rank,
                device=device,
                generator_engine=generator_engine,
                critic_engine=critic_engine,
                trace=trace,
                negative=negative,
                collect_diagnostics=False,
                collect_scalars=True,
                diagnostics=diagnostics,
                policy=policy,
                score_pack_parity_check=False,
                clean_runtime=clean_runtime,
                runtime_cache=runtime_cache,
                deferred_finite_checks=deferred_finite_checks,
            )
            generator_grad_norm = _finite_number(
                getattr(generator_engine, "last_gradient_norm", None)
            )
            rollout_outputs, rollout_records = p0._precompute_critic_rollouts(
                rank=rank,
                steps=SOURCE_STEPS,
                device=device,
                generator_engine=generator_engine,
                trace=trace,
                policy=policy,
                collect_diagnostics=False,
                diagnostics=diagnostics,
                rollout_parity_check=False,
                clean_runtime=clean_runtime,
                runtime_cache=runtime_cache,
                entry_outputs=critic_entries,
            )
            critic_records: list[dict[str, Any]] = []
            critic_grad_norms: list[float | None] = []
            for source_step in SOURCE_STEPS:
                critic_records.append(
                    p0._run_nogan_critic_precomputed(
                        rank=rank,
                        step=source_step,
                        x0=rollout_outputs[source_step],
                        device=device,
                        critic_engine=critic_engine,
                        trace=trace,
                        collect_diagnostics=False,
                        collect_scalars=True,
                        diagnostics=diagnostics,
                        entry=critic_entries[source_step],
                        runtime_cache=runtime_cache,
                        deferred_finite_checks=deferred_finite_checks,
                    )
                )
                critic_grad_norms.append(
                    _finite_number(getattr(critic_engine, "last_gradient_norm", None))
                )
                ema.step()
                del rollout_outputs[source_step]

            finite_gate = bool(torch.stack(deferred_finite_checks).all().detach().cpu())
            generator_loss = _finite_number(generator_record["surrogate_loss"])
            critic_losses = [_finite_number(row["loss"]) for row in critic_records]
            finite = bool(
                finite_gate
                and generator_loss is not None
                and all(value is not None for value in critic_losses)
            )
            expected_g = initial_g + cycle
            expected_c = initial_c + cycle * 5
            expected_ema = initial_ema + cycle * 5
            if (
                int(generator_engine.global_steps),
                int(critic_engine.global_steps),
                int(ema.optimization_step),
            ) != (expected_g, expected_c, expected_ema):
                raise AssertionError("continuous G1/C5/EMA5 version inventory changed")
            if len(rollout_records) != 5 or any(
                int(row.get("batch_size", 1)) != 1 for row in rollout_records
            ):
                raise AssertionError("continuous trajectory changed physical rollout B1")
            if int(generator_record["physical_score_forwards"]) != 3:
                raise AssertionError("continuous trajectory changed no-GAN Score forward inventory")
            rollout_inventory = [generator_record["rollout"], *rollout_records]
            if any(
                int(row["call_count"]) != int(row["logical_call_count"])
                or int(row.get("dummy_call_count", 0)) != 0
                for row in rollout_inventory
            ):
                raise AssertionError("continuous trajectory changed physical/logical DiT census")

            state_fingerprints: dict[str, str] = {}
            anchor_artifact = None
            if cycle in ANCHOR_CYCLES or cycle == args.trajectory_cycles:
                state_fingerprints, anchor_artifact = _state_anchor(
                    cycle=cycle,
                    rank=rank,
                    placement=args.placement,
                    output_dir=args.output_dir,
                    generator_engine=generator_engine,
                    critic_engine=critic_engine,
                    ema=ema,
                    device=device,
                )
            generator_rollout_calls = int(generator_record["rollout"]["call_count"])
            critic_rollout_calls = sum(
                int(row["call_count"]) for row in rollout_records
            )
            cycle_receipt = TrajectoryCycleReceipt(
                run_id=args.run_id,
                workload="wan1.3b_nogan_world16",
                variant=args.trajectory_arm_id,
                rank=rank,
                world_size=world,
                seed=args.trajectory_seed,
                cycle=cycle,
                expected_total_cycles=args.trajectory_cycles,
                logical_step=10 + (cycle - 1) * 5,
                matched_contract={
                    "trace_bundle_hash": trace.bundle_hash,
                    "trace_binding_hash": trace.binding_hash,
                    "trace_replay_policy": "FIXED_FIVE_STEP_COHORT_REPLAY",
                    "global_batch": "16",
                    "updates": "G1_C5_EMA5",
                    "score_packing": "off",
                    "rollout_batch": "B1",
                },
                implementation={
                    "placement": args.placement,
                    "arm_id": args.trajectory_arm_id,
                    "arm_contract": result["arm_contract"],
                    "worker": "probe_wan13b_trajectory_world16.py",
                },
                sample_identities=(
                    f"cycle={cycle}/generator/source_step=10/rank={rank}",
                    *(f"cycle={cycle}/critic/source_step={step}/rank={rank}" for step in SOURCE_STEPS),
                ),
                rng_identities={
                    "runtime_seed": str(runtime_seed),
                    "generator": f"{trace.binding_hash}/rank={rank}/generator/step=10",
                    **{
                        f"critic_step_{step}": (
                            f"{trace.binding_hash}/rank={rank}/critic/step={step}"
                        )
                        for step in SOURCE_STEPS
                    },
                },
                versions={"generator": expected_g, "critic": expected_c, "ema": expected_ema},
                cycle_updates={"generator": 1, "critic": 5, "ema": 5},
                cumulative_updates={"generator": cycle, "critic": 5 * cycle, "ema": 5 * cycle},
                call_census={
                    "generator_update_rollout_dit": generator_rollout_calls,
                    "generator_critic_rollout_dit": critic_rollout_calls,
                    "generator_score_forwards": int(generator_record["physical_score_forwards"]),
                    "critic_score_forwards": 5,
                    "physical_rollout_batches": 5,
                },
                losses={
                    "generator": generator_loss if finite else None,
                    "critic_mean": _mean(critic_losses) if finite else None,
                    **{
                        f"critic_{index}": value if finite else None
                        for index, value in enumerate(critic_losses)
                    },
                },
                gradient_norms={
                    "generator": generator_grad_norm if finite else None,
                    "critic_mean": _mean(critic_grad_norms) if finite else None,
                    **{
                        f"critic_{index}": value if finite else None
                        for index, value in enumerate(critic_grad_norms)
                    },
                },
                tensor_contracts={
                    "latent": {"shape": [1, 16, 9, 48, 80], "dtype": "torch.bfloat16"}
                },
                state_fingerprints=state_fingerprints,
                finite=finite,
                runtime_status="PASS" if finite else "NONFINITE",
            )
            writer.append(cycle_receipt)
            cycle_summaries.append(
                {
                    "cycle": cycle,
                    "wall_seconds_diagnostic": time.perf_counter() - cycle_started,
                    "versions": {"generator": expected_g, "critic": expected_c, "ema": expected_ema},
                    "losses": dict(cycle_receipt.losses),
                    "gradient_norms": dict(cycle_receipt.gradient_norms),
                    "finite": finite,
                    "state_anchor": anchor_artifact,
                    "memory": base._memory(device),
                }
            )
            if not finite:
                raise FloatingPointError(f"non-finite trajectory value at cycle {cycle}")

        torch.cuda.synchronize(device)
        dist.barrier()
        result.update(
            status="PASS",
            completed_cycles=args.trajectory_cycles,
            updates={
                "generator": args.trajectory_cycles,
                "critic": 5 * args.trajectory_cycles,
                "ema": 5 * args.trajectory_cycles,
            },
            final_versions={
                "generator": int(generator_engine.global_steps),
                "critic": int(critic_engine.global_steps),
                "ema": int(ema.optimization_step),
            },
            diagnostic_wall_seconds=time.perf_counter() - started,
            cycle_summaries=cycle_summaries,
            final_memory=base._memory(device),
            trajectory_jsonl=str(writer.path),
        )
        _atomic_json(output, result)
        dist.barrier()
        return 0
    except BaseException as error:
        result.update(
            status=(
                "OOM"
                if isinstance(error, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(error).lower()
                else "FAIL"
            ),
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        _atomic_json(output, result)
        raise
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
