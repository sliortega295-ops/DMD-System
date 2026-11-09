#!/usr/bin/env python3
"""Optimized entrypoint for the real GAN Role-Phase world16 executor.

The audited model math and existing executor remain authoritative. This
entrypoint preconstructs all forward Phase-B targets before formal timing,
emits transition subphase receipts, records producer/Critic service spans and
supports an opt-in reverse-target capacity canary.

The forward targets remain complete persistent-pinned CPU modules.  To keep
that semantic under a finite host cgroup, cold target construction is admitted
one local rank at a time and the allocator is trimmed after each completed
pageable->pinned replacement.  No batch, topology, ZeRO, loss, trace, EMA or
steady-state offload policy changes here.
"""
from __future__ import annotations

import functools
import gc
import os
from pathlib import Path
from typing import Any, Callable

from scripts.distillschedule import probe_gan_rolephase_world16 as legacy
from distillschedule.runtime.gan_rolephase_optimization import (
    JsonlStageRecorder,
    PersistentTargetPolicy,
    TransitionSubphase,
    assert_no_new_cgroup_oom,
    assert_preconstruction_admitted,
    build_preconstruction_waves,
    read_cgroup_memory_snapshot,
)

_STATES: dict[int, dict[str, Any]] = {}
_ORIGINALS: dict[str, Callable[..., Any]] = {}


def _global_rank() -> int:
    node_rank = int(os.environ.get("DISTILLSCHEDULE_NODE_RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    return node_rank * 8 + local_rank


def _state(rank: int | None = None) -> dict[str, Any] | None:
    return _STATES.get(_global_rank() if rank is None else int(rank))


def _recorder(rank: int | None = None) -> JsonlStageRecorder | None:
    value = _state(rank)
    return None if value is None else value["recorder"]


def _timed(name: str | TransitionSubphase, function: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        recorder = _recorder()
        if recorder is None:
            return function(*args, **kwargs)
        with recorder.stage(name):
            return function(*args, **kwargs)

    return wrapped


def _host_admission_policy() -> tuple[float, int]:
    max_used_fraction = float(
        os.environ.get(
            "DISTILLSCHEDULE_ROLEPHASE_HOST_ADMISSION_MAX_USED_FRACTION",
            "0.90",
        )
    )
    min_headroom_gib = float(
        os.environ.get(
            "DISTILLSCHEDULE_ROLEPHASE_HOST_ADMISSION_MIN_HEADROOM_GIB",
            "64",
        )
    )
    if min_headroom_gib < 0:
        raise ValueError("Role-Phase host-admission headroom must be non-negative")
    return max_used_fraction, int(min_headroom_gib * 2**30)


def _prepare_persistent_target(
    *,
    args: Any,
    global_rank: int,
    logical: Any,
    role: str,
    direction: str,
    timing_scope: str,
    wave_index: int,
    recorder: JsonlStageRecorder,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    max_used_fraction, min_headroom_bytes = _host_admission_policy()
    before = read_cgroup_memory_snapshot()
    with recorder.stage(
        TransitionSubphase.HOST_ADMISSION,
        direction=direction,
        role=role,
        preconstruction_wave=wave_index,
        max_concurrent_targets=1,
        timing_scope=timing_scope,
        cgroup_before=before.as_dict(),
    ):
        admission = assert_preconstruction_admitted(
            before,
            max_used_fraction=max_used_fraction,
            min_headroom_bytes=min_headroom_bytes,
        )

    with recorder.stage(
        TransitionSubphase.TARGET_PREPARE,
        direction=direction,
        role=role,
        preconstruction_wave=wave_index,
        max_concurrent_targets=1,
        timing_scope=timing_scope,
    ):
        # PR6's validated cold-target path explicitly disables accidental
        # Transformers/DeepSpeed ZeRO-3 init inheritance for the future CPU
        # module.  The target becomes a normal full CPU module until the legal
        # phase transition materializes it.
        with legacy.suspend_transformers_zero3_init():
            model, metadata, restore = legacy._build_cpu_model(
                args=args, role=role, logical=logical
            )
        host_preparation = legacy.pr6.pin_module_cpu_registered_state(model)
        host_preparation["timing_scope"] = timing_scope

        # pin_module_cpu_registered_state preserves the final fully-pinned
        # representation, but pageable source allocations may remain in the
        # host allocator cache.  Reclaim them before admitting the next 14B
        # target so cold-preparation transients cannot accumulate across ranks.
        gc.collect()
        allocator_trimmed = legacy.pr6.trim_host_allocator()
        after = read_cgroup_memory_snapshot()
        assert_no_new_cgroup_oom(before, after)
        host_preparation.update(
            {
                "admission_policy": "SERIAL_LOCAL_RANK_CGROUP_V2_FAIL_CLOSED",
                "preconstruction_wave": wave_index,
                "max_concurrent_targets": 1,
                "cgroup_before": before.as_dict(),
                "cgroup_after": after.as_dict(),
                "admission": admission,
                "post_target_allocator_trimmed": allocator_trimmed,
            }
        )
    return model, metadata, restore, host_preparation


def _optimized_preconstruct_targets(
    *, args: Any, global_rank: int, logical: Any, control_group: Any
) -> dict[str, Any]:
    reverse_canary = os.environ.get(
        "DISTILLSCHEDULE_ROLEPHASE_PRECONSTRUCT_REVERSE", "0"
    ) == "1"
    policy = PersistentTargetPolicy(
        reverse_target_policy=(
            "preconstruct_before_timing_capacity_canary"
            if reverse_canary
            else "lazy_after_critic_release"
        )
    )
    recorder = JsonlStageRecorder(
        args.output_dir / "optimization" / f"rank_{int(global_rank):02d}_transition.jsonl",
        rank=global_rank,
    )
    value: dict[str, Any] = {
        "forward_model": None,
        "forward_metadata": None,
        "forward_restore": None,
        "forward_pin": None,
        "reverse_model": None,
        "reverse_metadata": None,
        "reverse_restore": None,
        "reverse_pin": None,
        "reverse_policy": policy.reverse_target_policy,
        "forward_policy": (
            "PERSISTENT_PRECONSTRUCTED_PINNED_BEFORE_FORMAL_TIMER"
            if global_rank in (*legacy.PHASE_B_G_RANKS, *legacy.FLEX_RANKS)
            else "REUSE_PHASE_A_SCORE"
        ),
        "host_admission_policy": (
            "SERIAL_LOCAL_RANK_CGROUP_V2_FAIL_CLOSED"
            if global_rank < legacy.PHASE_A_WORLD_SIZE
            else "NO_FORWARD_TARGET_ON_NODE1"
        ),
    }
    if global_rank in legacy.PHASE_B_G_RANKS:
        role = "generator"
    elif global_rank in legacy.FLEX_RANKS:
        role = "critic"
    else:
        role = None

    local_rank = int(global_rank) % legacy.PHASE_A_WORLD_SIZE
    # All eight node0 ranks own a future Phase-B target.  They are deliberately
    # prepared in singleton waves.  This is excluded cold preparation, so the
    # longer startup does not contaminate Full5 timing, while the final target
    # inventory remains exactly the same as PR10 before this fix.
    forward_waves = build_preconstruction_waves(
        range(legacy.PHASE_A_WORLD_SIZE),
        local_world_size=legacy.PHASE_A_WORLD_SIZE,
        max_concurrent=1,
    )
    value["forward_preconstruction_waves"] = [list(wave) for wave in forward_waves]
    for wave_index, wave in enumerate(forward_waves):
        legacy.pr6._control_barrier(control_group)
        if role is not None and local_rank in wave:
            model, metadata, restore, host_preparation = _prepare_persistent_target(
                args=args,
                global_rank=global_rank,
                logical=logical,
                role=role,
                direction="to_phase_b",
                timing_scope="outside_formal_timer",
                wave_index=wave_index,
                recorder=recorder,
            )
            value.update(
                forward_model=model,
                forward_metadata=metadata,
                forward_restore=restore,
                forward_pin=host_preparation,
            )
        legacy.pr6._control_barrier(control_group)

    if reverse_canary:
        reverse_local_ranks = tuple(
            int(rank) % legacy.PHASE_A_WORLD_SIZE for rank in legacy.FLEX_RANKS
        )
        reverse_waves = build_preconstruction_waves(
            reverse_local_ranks,
            local_world_size=legacy.PHASE_A_WORLD_SIZE,
            max_concurrent=1,
        )
        value["reverse_preconstruction_waves"] = [
            list(wave) for wave in reverse_waves
        ]
        for wave_index, wave in enumerate(reverse_waves):
            legacy.pr6._control_barrier(control_group)
            if global_rank in legacy.FLEX_RANKS and local_rank in wave:
                model, metadata, restore, host_preparation = _prepare_persistent_target(
                    args=args,
                    global_rank=global_rank,
                    logical=logical,
                    role="generator",
                    direction="to_phase_a",
                    timing_scope="outside_formal_timer_capacity_canary",
                    wave_index=wave_index,
                    recorder=recorder,
                )
                value.update(
                    reverse_model=model,
                    reverse_metadata=metadata,
                    reverse_restore=restore,
                    reverse_pin=host_preparation,
                )
            legacy.pr6._control_barrier(control_group)

    legacy.pr6._control_barrier(control_group)
    _STATES[int(global_rank)] = {
        "args": args,
        "logical": logical,
        "targets": value,
        "policy": policy,
        "recorder": recorder,
    }
    return value


def _require_preconstructed_forward_target(
    *, args: Any, global_rank: int, logical: Any, targets: dict[str, Any]
) -> dict[str, Any]:
    del args, logical
    if global_rank in (*legacy.PHASE_B_G_RANKS, *legacy.FLEX_RANKS):
        if targets.get("forward_model") is None or targets.get("forward_pin") is None:
            raise AssertionError("optimized Role-Phase lost a preconstructed forward target")
    return {
        "policy": targets["forward_policy"],
        "constructed_in_handoff": False,
        "target_prepared_before_formal_timer": True,
        "host_admission_policy": targets.get("host_admission_policy"),
    }


def _optimized_reverse_to_phase_a(
    *,
    args: Any,
    global_rank: int,
    local_rank: int,
    device: Any,
    logical: Any,
    captured_g201: dict[str, Any] | None,
    generator_model: Any | None,
    critic_engine: Any | None,
    reverse_model: Any | None,
    control_group: Any,
    exchange: Path,
    local_rendezvous: dict[str, str],
) -> tuple[dict[str, Any], Any | None, Any | None]:
    recorder = _recorder(global_rank)
    if recorder is None:
        raise RuntimeError("optimized reverse handoff has no transition recorder")
    record: dict[str, Any] = {}
    with recorder.stage(TransitionSubphase.REVERSE_TOTAL, direction="to_phase_a"):
        with recorder.stage(TransitionSubphase.BARRIER_PEER_WAIT, name="reverse_entry"):
            legacy.pr6._control_barrier(control_group)
        if global_rank in legacy.FLEX_RANKS:
            if critic_engine is None:
                raise AssertionError("flex rank lacks obsolete Phase-B Critic")
            with recorder.stage(
                TransitionSubphase.SOURCE_RELEASE,
                direction="to_phase_a",
                role="critic",
            ):
                record["flex_critic_release"] = legacy._release_obsolete_zero1_engine(
                    critic_engine, device=device
                )
                critic_engine = None
            if reverse_model is None:
                with recorder.stage(
                    TransitionSubphase.TARGET_PREPARE,
                    direction="to_phase_a",
                    role="generator",
                    timing_scope="included_reverse_handoff",
                ):
                    reverse_model, reverse_metadata, reverse_restore = legacy._build_cpu_model(
                        args=args, role="generator", logical=logical
                    )
                    reverse_host_preparation = legacy.pr6.pin_module_cpu_registered_state(
                        reverse_model
                    )
                    reverse_host_preparation["timing_scope"] = "included_reverse_handoff"
                    record["reverse_target_construction"] = {
                        "policy": "LAZY_AFTER_PHASE_B_CRITIC_RELEASE",
                        "metadata": reverse_metadata,
                        "logical_restore": reverse_restore,
                        "host_preparation": reverse_host_preparation,
                    }
            else:
                record["reverse_target_construction"] = {
                    "policy": "PRECONSTRUCTED_BEFORE_FORMAL_TIMER_CAPACITY_CANARY",
                    "constructed_in_handoff": False,
                }
        with recorder.stage(
            TransitionSubphase.BARRIER_PEER_WAIT,
            name="reverse_source_released",
        ):
            legacy.pr6._control_barrier(control_group)
        with recorder.stage(
            TransitionSubphase.PROCESS_GROUP_LIFECYCLE,
            direction="phase_b_world16_to_phase_a_world8",
        ):
            legacy.dist.destroy_process_group()
            if legacy.dist.is_initialized():
                raise AssertionError("global Phase-B world survived reverse handoff")
            legacy._restore_local_phase_a_rendezvous(
                local_rendezvous,
                master_port=args.phase_a_reverse_master_port,
            )
            legacy.pr6._initialize_phase_a_distributed()
            import deepspeed

            deepspeed.init_distributed(dist_backend="nccl", dist_init_required=False)
            if int(deepspeed.comm.get_world_size()) != legacy.PHASE_A_WORLD_SIZE:
                raise AssertionError("DeepSpeed did not rebind to reverse Phase-A world8")
            local_group = legacy.pr6._create_phase_a_groups(node_rank=args.node_rank)
        reverse_engine = None
        model = None
        if args.node_rank == 0:
            if global_rank in legacy.PHASE_B_G_RANKS:
                model = generator_model
                if model is None:
                    raise AssertionError("fixed Generator rank lost G201 model")
                materialization = {"policy": "REUSE_RESIDENT_G201"}
            else:
                model = reverse_model
                if model is None:
                    raise AssertionError("flex rank lacks reverse Generator target")
                with recorder.stage(
                    TransitionSubphase.TARGET_MATERIALIZE,
                    direction="to_phase_a",
                    role="generator",
                ):
                    materialization = legacy.pr6.materialize_module_from_persistent_pinned_cpu(
                        model, device=device, cleanup_policy="post_full5"
                    )
            with recorder.stage(
                TransitionSubphase.BROADCAST_RESHARD,
                direction="to_phase_a",
                role="generator",
            ):
                redistribution = legacy.pr6._broadcast_generator_g3(
                    model=model,
                    logical=logical,
                    rank=global_rank,
                    group=local_group,
                    timing_mode=args.timing_mode,
                    bucket_mib=args.generator_broadcast_bucket_mib,
                )
                redistribution["parameter_version"] = 201
            with recorder.stage(
                TransitionSubphase.ENGINE_INITIALIZE,
                direction="to_phase_a",
                role="generator_zero3",
            ):
                reverse_engine, initialization = legacy._initialize_phase_a_generator(
                    args=args, model=model, logical=logical, local_rank=local_rank
                )
            if captured_g201 is None:
                raise AssertionError("rank lacks captured post-update ZeRO-3 optimizer state")
            with recorder.stage(
                TransitionSubphase.OPTIMIZER_RESTORE,
                direction="to_phase_a",
                role="generator_zero3",
            ):
                restore = legacy._restore_captured_zero3(reverse_engine, captured_g201)
            if (
                int(reverse_engine.global_steps),
                legacy.pr6._optimizer_step(reverse_engine.optimizer),
                legacy.pr6._scheduler_step(reverse_engine),
            ) != (3, 3, 3):
                raise AssertionError("reverse Phase-A Generator is not G201-ready")
            record.update(
                role="phase_a_generator",
                materialization=materialization,
                redistribution=redistribution,
                zero3_initialization=initialization,
                captured_optimizer_restore=restore,
            )
        else:
            if critic_engine is None or int(critic_engine.global_steps) != legacy.TARGET_CRITIC_OPTIMIZER_STEP:
                raise AssertionError("fixed Score rank did not retain C1005")
            model = critic_engine.module
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise AssertionError("C1005 Score retained gradients at reverse boundary")
            record.update(
                role="phase_a_score_gradient_service",
                parameter_version=1005,
                physical_model_reused=True,
                repeated_backbone_materialization=False,
            )
        with recorder.stage(
            TransitionSubphase.BARRIER_PEER_WAIT,
            name="reverse_phase_a_ready",
        ):
            legacy.pr6._phase_a_file_barrier(
                exchange=exchange,
                name="reverse_phase_a_ready",
                node_rank=args.node_rank,
                control_group=local_group,
                timeout_seconds=args.timeout_seconds,
            )
        return record, reverse_engine, model


def _install_patches() -> None:
    if _ORIGINALS:
        return
    _ORIGINALS.update(
        preconstruct=legacy._preconstruct_targets,
        construct_forward=legacy._construct_forward_target_after_phase_a,
        reverse=legacy._reverse_to_phase_a,
    )
    legacy._preconstruct_targets = _optimized_preconstruct_targets
    legacy._construct_forward_target_after_phase_a = _require_preconstructed_forward_target
    legacy._reverse_to_phase_a = _optimized_reverse_to_phase_a
    patches = (
        (legacy, "_capture_zero3_optimizer", TransitionSubphase.SOURCE_EXPORT),
        (legacy.pr6, "_materialize_zero3_g3_into_target", TransitionSubphase.SOURCE_EXPORT),
        (legacy.pr6, "_destroy_obsolete_zero3_engine", TransitionSubphase.SOURCE_RELEASE),
        (legacy.pr6, "_initialize_phase_b_distributed", TransitionSubphase.PROCESS_GROUP_LIFECYCLE),
        (legacy, "create_phase_b_execution_groups", TransitionSubphase.PROCESS_GROUP_LIFECYCLE),
        (legacy.pr6, "materialize_module_from_persistent_pinned_cpu", TransitionSubphase.TARGET_MATERIALIZE),
        (legacy, "_initialize_phase_b_critic", TransitionSubphase.ENGINE_INITIALIZE),
        (legacy, "_run_phase_b_generator", TransitionSubphase.PHASE_B_GENERATOR),
        (legacy, "_run_phase_b_critic", TransitionSubphase.PHASE_B_CRITIC),
    )
    for owner, name, stage in patches:
        original = getattr(owner, name)
        _ORIGINALS[f"{id(owner)}:{name}"] = original
        setattr(owner, name, _timed(stage, original))


def main() -> int:
    os.environ["DISTILLSCHEDULE_ROLEPHASE_OPTIMIZED_V2"] = "1"
    _install_patches()
    try:
        return int(legacy.main())
    finally:
        for value in _STATES.values():
            recorder: JsonlStageRecorder = value["recorder"]
            recorder.write_summary()


if __name__ == "__main__":
    raise SystemExit(main())
