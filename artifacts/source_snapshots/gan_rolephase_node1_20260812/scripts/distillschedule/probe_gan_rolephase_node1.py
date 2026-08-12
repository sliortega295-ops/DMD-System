#!/usr/bin/env python3
"""Real-14B node-1 half of the GAN 8G/8Score -> 4G/12C prototype.

The executable is deliberately restricted to global ranks 8..15.  Those
ranks keep one physical Score/Critic backbone resident across both phases:

* Phase A: eight read-only Score gradient-service lanes, two B1 waves each;
* transition: bind the same resident module to the Phase-B DP12 ZeRO-1 engine;
* Phase B: eight of the twelve B1 Critic lanes (logical slots 4..11), one
  physical ``_critic_loss`` call and one optimizer commit per logical step.

It is not a standalone world8 launcher.  Actual model execution requires the
future node-0 companion so that all sixteen ranks create the same groups and
the flex ranks 4..7 join the DP12 Critic collective.  ``--contract-only`` is
the low-cost node-1 validation path and never initializes CUDA/distributed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.activation_offload import install_activation_offload_hooks
from runtime.critic_pool import ChecksumKind, StudentDPGroupAdapter
from runtime.critic_pool_helios import build_helios_pool_model
from runtime.gan_rolephase_node1 import (
    assert_phase_b_zero1_contract,
    assert_topology_parity,
    bind_node1_rank,
    phase_b_zero1_config,
)
from scripts.distillschedule.probe_gan_pipeline_world16 import (
    _Accelerator,
    _critic_kwargs,
    _load_logical_trainables,
    _load_official_config,
    _restore_logical_zero1_optimizer,
    _score_official_kwargs,
    _source,
    _trace,
)

from distillschedule.runtime.gan_pipeline_helios import (
    compute_official_score_boundary_grad,
)
from distillschedule.runtime.gan_pipeline_transport import (
    GanGradientP2PTransport,
    GanGradientReplyHeader,
)
from distillschedule.runtime.gan_rolephase import GanRolePhaseTopology
from distillschedule.runtime.gan_rolephase_critic import CriticReductionContract
from distillschedule.runtime.gan_rolephase_groups import (
    create_rolephase_execution_groups,
)
from distillschedule.runtime.gan_rolephase_transport import (
    RolePhaseCriticP2PTransport,
    make_phase_b_route,
)
from runtime.critic_pool_transport import tensor_identity


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _memory(device: torch.device) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "driver_free_bytes": int(free),
        "driver_total_bytes": int(total),
    }


def _require_real_paths(args: argparse.Namespace) -> None:
    required = (
        "config",
        "logical_package",
        "trace_bundle",
        "score_root",
        "negative_prompt",
        "output_dir",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError("real node1 execution requires: " + ", ".join(missing))
    inputs = required[:-1]
    absent = [name for name in inputs if not Path(getattr(args, name)).exists()]
    if absent:
        raise FileNotFoundError("missing node1 execution assets: " + ", ".join(absent))


def _assert_resident_score(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    parameters = list(model.parameters())
    if not parameters or any(parameter.device != device for parameter in parameters):
        raise AssertionError("fixed node1 Score/Critic backbone is not fully resident")
    unique: dict[tuple[int, int], tuple[int, bool]] = {}
    for parameter in parameters:
        storage = parameter.untyped_storage()
        key = (int(storage.data_ptr()), int(storage.nbytes()))
        unique.setdefault(key, (int(storage.nbytes()), bool(parameter.requires_grad)))
    return {
        "physical_model_identity": id(model),
        "parameter_tensor_count": len(parameters),
        "unique_parameter_storage_bytes": sum(item[0] for item in unique.values()),
        "trainable_parameter_count": sum(parameter.requires_grad for parameter in parameters),
        "parameter_policy": "FULL_RESIDENT_FIXED_SCORE_ROLE",
        "repeated_frozen_block_streaming": False,
    }


def _assert_no_parameter_gradients(model: torch.nn.Module, *, phase: str) -> None:
    unexpected = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
    if unexpected:
        raise AssertionError(
            f"{phase} created Score parameter gradients: {unexpected[:3]}"
        )


def _initialize_phase_b_engine(
    *,
    model: torch.nn.Module,
    logical_package: Path,
    logical_spec: dict[str, Any],
    global_rank: int,
    phase_b_group: Any,
    singleton_group: Any,
) -> tuple[Any, dict[str, Any]]:
    import deepspeed

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise AssertionError("Phase-B Critic trainable inventory is empty")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=4e-7,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    adapter = StudentDPGroupAdapter(
        data_parallel_group=phase_b_group,
        data_parallel_ranks=tuple(range(4, 16)),
        global_rank=global_rank,
        singleton_model_parallel_group=singleton_group,
    )
    module_identity = id(model)
    started = time.perf_counter()
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=phase_b_zero1_config(),
        mpu=adapter,
        dist_init_required=False,
    )
    initialize_seconds = time.perf_counter() - started
    if engine.module is not model or id(engine.module) != module_identity:
        raise AssertionError("node1 transition replaced the resident physical Score model")
    resolved = assert_phase_b_zero1_contract(
        engine,
        expected_dp_group=phase_b_group,
    )
    restore = _restore_logical_zero1_optimizer(
        engine,
        logical_package,
        logical_spec,
        role_rank=global_rank - 4,
        role_world_size=12,
    )
    engine.lr_scheduler.load_state_dict(logical_spec["scheduler_state"])
    engine.global_steps = int(logical_spec["optimizer_step"])
    return engine, {
        "transition": "RETAIN_RESIDENT_SCORE_REBIND_DP8_READ_ONLY_TO_DP12_ZERO1",
        "physical_model_identity": module_identity,
        "deepspeed_initialize_seconds": initialize_seconds,
        "resolved": resolved,
        "logical_optimizer_restore": restore,
        "parameter_redistribution": "NONE_ON_FIXED_NODE1_SCORE_RANKS",
        "optimizer_repartition": "LOGICAL_FP32_STATE_TO_DP12_CPU_PARTITION",
    }


def _phase_a_score_service(
    *,
    args: argparse.Namespace,
    cfg: Any,
    model: torch.nn.Module,
    device: torch.device,
    binding: Any,
    pair_group: Any,
    negative: torch.Tensor,
) -> list[dict[str, Any]]:
    transport = GanGradientP2PTransport(group=pair_group, device=device)
    records: list[dict[str, Any]] = []
    for wave, logical_slot in enumerate(binding.phase_a_logical_slots):
        trace = _trace(args.trace_bundle, logical_slot)
        source = _source(trace, device)
        header, generated = transport.receive_request(
            source_global_rank=binding.phase_a_lane
        )
        expected = (
            args.run_nonce,
            1000,
            wave,
            binding.phase_a_lane,
            binding.global_rank,
            200,
            1000,
            bytes.fromhex(trace.trace_hash),
        )
        observed = (
            header.run_nonce,
            header.logical_step,
            header.micro_wave,
            header.generator_rank,
            header.score_rank,
            header.generator_version,
            header.critic_version,
            header.trace_sha256,
        )
        if observed != expected:
            raise AssertionError(f"Phase-A request identity drift: {observed[:-1]}")
        boundary = compute_official_score_boundary_grad(
            score_model=model,
            generated_latent=generated,
            gan_g_weight=float(cfg.training_config.gan_g_weight),
            official_kwargs=_score_official_kwargs(
                cfg,
                trace,
                source,
                1000,
                negative,
                device,
            ),
        )
        _assert_no_parameter_gradients(model, phase="Phase-A Score service")
        digest = tensor_identity(
            boundary.gradient,
            ChecksumKind.GPU_FINGERPRINT_V1,
        )
        reply = GanGradientReplyHeader(
            args.run_nonce,
            1000,
            wave,
            binding.phase_a_lane,
            binding.global_rank,
            200,
            1000,
            1000,
            bytes.fromhex(trace.trace_hash),
            digest,
        )
        transport.wait_and_reap(
            transport.send_reply(
                reply,
                boundary.gradient,
                destination_global_rank=binding.phase_a_lane,
            )
        )
        records.append(
            {
                "phase": "A_SCORE_GRADIENT_SERVICE",
                "logical_step": 1000,
                "wave": wave,
                "logical_slot": logical_slot,
                "trace_hash": trace.trace_hash,
                "total_loss": boundary.total_loss,
                "dmd_loss": boundary.dmd_loss,
                "gan_g_loss": boundary.gan_g_loss_raw,
                "optimizer_committed": False,
            }
        )
    transport.flush()
    transport.assert_flushed()
    return records


def _phase_b_critic(
    *,
    args: argparse.Namespace,
    cfg: Any,
    engine: Any,
    device: torch.device,
    binding: Any,
    world_group: Any,
) -> list[dict[str, Any]]:
    from helios.utils.utils_helios_post import _critic_loss

    reduction = CriticReductionContract()
    if reduction.local_loss_scale("DP_MEAN") != 1.0:
        raise AssertionError("DP12/B1 mean reduction unexpectedly needs loss scaling")
    trace = _trace(args.trace_bundle, binding.phase_b_logical_slot)
    source = _source(trace, device)
    transport = RolePhaseCriticP2PTransport(
        group=world_group,
        device=device,
        global_rank=binding.global_rank,
    )

    def route(step: int):
        critic_before = 1000 + (step - 1000)
        return make_phase_b_route(
            logical_step=step,
            logical_slot=binding.phase_b_logical_slot,
            generator_rank=binding.phase_b_generator_rank,
            critic_rank=binding.phase_b_critic_rank,
            generator_version=201,
            critic_before=critic_before,
            trace_sha256=bytes.fromhex(trace.trace_hash),
        )

    pending = transport.post_receive(route=route(1000))
    records: list[dict[str, Any]] = []
    expected_global_step = int(engine.global_steps)
    for step in range(1000, 1005):
        header, generated_x0 = transport.finish_receive(pending)
        if header.run_nonce != args.run_nonce:
            raise AssertionError("Phase-B run nonce drift")
        if tuple(generated_x0.shape) != (1, 16, 9, 48, 80):
            raise AssertionError("Phase-B payload lost official B1 latent shape")
        # Q2: post the next receive before consuming the current batch.  This
        # does not relax the C_t -> C_{t+1} optimizer dependency.
        if step < 1004:
            pending = transport.post_receive(route=route(step + 1))

        kwargs = _critic_kwargs(
            cfg,
            trace,
            source,
            step,
            generated_x0,
            engine,
            0,
        )
        # Wire micro_wave=0 identifies the single Phase-B transport wave.  The
        # official Helios helper uses ``None`` to mean a complete one-call
        # logical Critic batch and therefore commits exactly once; passing 0
        # would intentionally defer the commit for the old two-wave GAS2 path.
        kwargs["pipeline_micro_wave"] = None
        loss, logs = _critic_loss(fake_score_model=engine.module, **kwargs)
        expected_global_step += 1
        if not bool(logs.get("critic_optimizer_committed")):
            raise AssertionError("Phase-B B1 call did not commit the Critic optimizer")
        if int(engine.global_steps) != expected_global_step:
            raise AssertionError(
                f"Critic version drift after step {step}: "
                f"{engine.global_steps} != {expected_global_step}"
            )
        records.append(
            {
                "phase": "B_CRITIC",
                "logical_step": step,
                "logical_slot": binding.phase_b_logical_slot,
                "source_generator_rank": binding.phase_b_generator_rank,
                "trace_hash": trace.trace_hash,
                "loss": float(loss.detach()),
                "optimizer_committed": True,
                "critic_engine_global_steps": int(engine.global_steps),
                "critic_loss_calls": 1,
                "expected_score_forwards": 2,
                "transport_micro_wave": 0,
                "helios_pipeline_micro_wave": None,
            }
        )
    transport.flush()
    transport.assert_flushed()
    return records


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--contract-only", action="store_true")
    value.add_argument("--global-rank", type=int)
    value.add_argument("--local-rank", type=int)
    value.add_argument("--config", type=Path)
    value.add_argument("--logical-package", type=Path)
    value.add_argument("--trace-bundle", type=Path)
    value.add_argument("--score-root", type=Path)
    value.add_argument("--negative-prompt", type=Path)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--run-nonce", type=int, default=812001)
    value.add_argument("--activation-min-bytes", type=int, default=128 * 1024 * 1024)
    return value


def _contract_only(args: argparse.Namespace) -> int:
    if args.global_rank is None or args.local_rank is None:
        raise ValueError("--contract-only requires --global-rank and --local-rank")
    binding = bind_node1_rank(
        global_rank=args.global_rank,
        local_rank=args.local_rank,
    )
    topology = GanRolePhaseTopology()
    assert_topology_parity(binding, topology)
    result = {
        "schema_version": "distillschedule.gan_rolephase_node1_contract.v1",
        "status": "PASS",
        "claim_status": "CONTRACT_ONLY_NO_GPU_NO_MODEL",
        "binding": binding.as_dict(),
        "phase_b_deepspeed": phase_b_zero1_config(),
        "not_run": [
            "14B model construction",
            "Phase-A gradient service",
            "DP12 DeepSpeed initialization",
            "Phase-B Critic updates",
            "reverse handoff",
            "world16 correctness",
            "formal timing",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.contract_only:
        return _contract_only(args)
    _require_real_paths(args)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    binding = bind_node1_rank(
        global_rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )
    topology = GanRolePhaseTopology()
    assert_topology_parity(binding, topology)
    output = args.output_dir / f"rank_{rank:03d}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable result: {output}")

    result: dict[str, Any] = {
        "schema_version": "distillschedule.gan_rolephase_node1_real14b.v1",
        "status": "RUNNING",
        "claim_status": "NODE1_HALF_REQUIRES_NODE0_COMPANION",
        "rank": rank,
        "local_rank": local_rank,
        "binding": binding.as_dict(),
        "updates": {"generator": 0, "critic": 0, "ema": 0},
    }
    activation = None
    try:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl")
        groups = create_rolephase_execution_groups(topology)
        pair_group = groups.phase_a_pair_for_rank(rank, topology)
        singleton = groups.singleton_for_rank(rank)
        result["group_contract"] = {
            "phase_a_score_ranks": list(topology.phase_a_score_ranks),
            "phase_b_critic_ranks": list(topology.phase_b_critic_ranks),
            "phase_a_pair": [binding.phase_a_lane, rank],
            "singleton": [rank],
            "created_before_model_construction": True,
        }
        cfg = _load_official_config(args.config)
        model, model_metadata = build_helios_pool_model(
            role="critic",
            model_root=args.score_root,
            subfolder="",
            enable_gan=True,
        )
        logical_spec = _load_logical_trainables(
            model,
            args.logical_package,
            "critic",
        )
        model.to(device)
        result["model"] = model_metadata
        result["resident_before_phase_a"] = _assert_resident_score(model, device)
        negative = torch.load(
            args.negative_prompt,
            map_location="cpu",
            weights_only=True,
        ).to(device=device, dtype=torch.bfloat16)
        dist.barrier(group=groups.core.world)
        result["phase_a"] = _phase_a_score_service(
            args=args,
            cfg=cfg,
            model=model,
            device=device,
            binding=binding,
            pair_group=pair_group,
            negative=negative,
        )
        _assert_no_parameter_gradients(model, phase="Phase-A flush")
        dist.barrier(group=groups.core.world)

        engine, transition = _initialize_phase_b_engine(
            model=model,
            logical_package=args.logical_package,
            logical_spec=logical_spec,
            global_rank=rank,
            phase_b_group=groups.core.phase_b_critic,
            singleton_group=singleton,
        )
        result["handoff_to_b"] = transition
        result["resident_after_handoff"] = _assert_resident_score(
            engine.module,
            device,
        )
        activation = install_activation_offload_hooks(
            engine.module,
            logical_component="gan_rolephase_b_critic_train",
            event_path=(
                args.output_dir
                / "activation_offload"
                / f"rank_{rank:03d}_phase_b.jsonl"
            ),
            pin_memory=True,
            min_offload_bytes=int(args.activation_min_bytes),
            detailed_events=False,
            dynamic_excluded_storage_providers=(),
        )
        result["phase_b"] = _phase_b_critic(
            args=args,
            cfg=cfg,
            engine=engine,
            device=device,
            binding=binding,
            world_group=groups.core.world,
        )
        result["updates"]["critic"] = len(result["phase_b"])
        if result["updates"] != {"generator": 0, "critic": 5, "ema": 0}:
            raise AssertionError("node1 update contract drift")
        activation_stats = dict(activation.offloader.stats)
        activation.remove()
        activation = None
        if int(activation_stats.get("live_cpu_bytes", 0)) != 0:
            raise AssertionError("Phase-B activation handles leaked")
        dist.barrier(group=groups.core.world)
        result.update(
            status="PASS",
            activation_offload=activation_stats,
            memory=_memory(device),
            reverse_handoff="BOUNDARY_REACHED_IMPLEMENTATION_NOT_RUN",
            not_claimed=[
                "standalone node1 execution",
                "world16 end-to-end correctness",
                "reverse handoff",
                "formal timing",
            ],
        )
    except BaseException as error:
        result.update(
            status="FAIL",
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        _atomic_json(output, result)
        raise
    finally:
        if activation is not None:
            activation.remove()
        if dist.is_initialized():
            dist.destroy_process_group()
    _atomic_json(output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
