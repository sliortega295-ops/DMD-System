#!/usr/bin/env python3
"""Fair Wan1.3B Resident/FSDP2 worker with an optimized Resident ladder.

This worker reuses the audited PR #21 workload implementation and changes only
the placement substrate:

* FSDP2 preserves original parameter dtypes (BF16 base, FP32 LoRA) instead of
  blanket-casting all unsharded parameters to BF16;
* every FSDP2 group uses SUM followed by division by world16;
* ``resident-ddp`` and ``resident-fast`` keep full model residency and use DDP
  only for overlapped LoRA-gradient synchronization;
* ``resident-fast`` additionally uses a foreach EMA update.

Model calls, B1/global-B16 trace, G1/C5/EMA5, loss, optimizer hyperparameters,
checkpointing, and all no-offload/no-streaming constraints are inherited from
the pinned worker.
"""
from __future__ import annotations

import gc
import copy
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_BASE_PATH = ROOT / "scripts/distillschedule/probe_wan13b_upper_bound_world16.py"
_SPEC = importlib.util.spec_from_file_location("wan13b_pr21_worker", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot import {_BASE_PATH}")
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)

from distillschedule.runtime.wan13b_fair import (
    ASYNC_128_RESIDENT,
    ASYNC_128_TREE_RESIDENT,
    ASYNC_128_TREE_CHUNK2048_RESIDENT,
    ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
    ASYNC_256_RESIDENT,
    CLEAN_RESIDENT,
    FAIR_PLACEMENTS,
    FSDP_PLACEMENT,
    FLAT_256_RESIDENT,
    FLAT_512_RESIDENT,
    LORA_ZERO2_RESIDENT,
    ROLLOUT_FUSED_RESIDENT,
    LOCAL_FAST_RESIDENT,
    FairTrainableEMA,
    LoRAComputeDtypeAudit,
    ShardedTrainableEMA,
    RESIDENT_PLACEMENTS,
    apply_wan13b_fair_fsdp2_policy,
    build_fair_engine,
    configure_modulation_chunk_tokens,
    fair_model_trainable_metrics,
    install_resident_clean_model_state,
    install_generator_nograd_block_compile,
)
from distillschedule.runtime.wan13b_lora_cast import (
    install_generator_nograd_lora_cast_cache,
    install_generator_nograd_lora_fused_epilogue,
    install_lora_operator_attribution,
)
from distillschedule.runtime.wan13b_lora_grad_epilogue import (
    grad_epilogue_requested,
    install_grad_lora_fused_epilogue,
)


def _build_model_fair(
    *,
    role: str,
    args: Any,
    bootstrap: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if grad_epilogue_requested() and args.placement != ASYNC_128_TREE_CHUNK2048_RESIDENT:
        raise ValueError("grad epilogue canary is registered only for the Resident winner")
    seed = int(bootstrap.component(role)["seed"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with base.suspend_transformers_zero3_init():
            model, metadata = base.build_helios_pool_model(
                role=role,
                model_root=args.model_root,
                subfolder="transformer",
                precast_model_root=None,
                lora_rank=256,
                enable_gan=False,
                chunked_residual_storage="cuda_out_of_place",
                missing_pyramid_patch_seed=base.WAN13B_PYRAMID_PATCH_SEED,
            )
    if int(metadata["block_count"]) != base.WAN13B_BLOCK_COUNT:
        raise AssertionError("Wan1.3B fair worker requires exactly 30 blocks")
    restore = bootstrap.restore_trainables(model, component=role)
    modulation_chunk = configure_modulation_chunk_tokens(
        model, placement=args.placement
    )
    metadata = dict(metadata)
    metadata["modulation_chunk_runtime"] = modulation_chunk

    # Use the same non-reentrant checkpoint wrapper on every arm. Disabling the
    # metadata check changes no tensor math but removes a wrapper-only variable.
    base.configure_fsdp2_checkpointing(model)
    model.to(device)
    torch.cuda.synchronize(device)

    if args.placement == ROLLOUT_FUSED_RESIDENT and role == "generator":
        shadow = copy.deepcopy(model)
        shadow.requires_grad_(False)
        shadow.train(model.training)
        object.__setattr__(model, "_distillschedule_rollout_shadow", shadow)

    if args.placement == FSDP_PLACEMENT:
        placement = apply_wan13b_fair_fsdp2_policy(model)
    elif args.placement in RESIDENT_PLACEMENTS:
        parameters = list(model.parameters())
        if any(hasattr(value, "device_mesh") for value in parameters):
            raise AssertionError("resident fair arm unexpectedly contains DTensors")
        if any(value.device != device for value in parameters):
            raise AssertionError("resident fair arm has non-resident parameters")
        trainable_dtypes = sorted(
            {str(value.dtype) for value in parameters if value.requires_grad}
        )
        frozen_dtypes = sorted(
            {str(value.dtype) for value in parameters if not value.requires_grad}
        )
        if trainable_dtypes != ["torch.float32"]:
            raise AssertionError(
                f"resident LoRA trainables must remain FP32, got {trainable_dtypes}"
            )
        if "torch.bfloat16" not in frozen_dtypes:
            raise AssertionError(
                f"resident frozen base must retain BF16, got {frozen_dtypes}"
            )
        placement = {
            "backend": (
                "FULL_REPLICA_GPU_RESIDENT_LORA_ZERO2"
                if args.placement in {LORA_ZERO2_RESIDENT, ROLLOUT_FUSED_RESIDENT}
                else (
                    "FULL_REPLICA_GPU_RESIDENT_DDP_GRAD_SYNC"
                    if args.placement
                    not in {
                        "resident",
                        CLEAN_RESIDENT,
                        ASYNC_128_RESIDENT,
                        ASYNC_128_TREE_RESIDENT,
                        ASYNC_128_TREE_CHUNK2048_RESIDENT,
                        ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
                        ASYNC_256_RESIDENT,
                        LOCAL_FAST_RESIDENT,
                        FLAT_256_RESIDENT,
                        FLAT_512_RESIDENT,
                        ROLLOUT_FUSED_RESIDENT,
                    }
                    else "FULL_REPLICA_GPU_RESIDENT_MANUAL_GRAD_SYNC"
                )
            ),
            "world_size": base.WORLD_SIZE,
            "block_count": base.WAN13B_BLOCK_COUNT,
            "wrap_policy": (
                "LORA_ONLY_ZERO2"
                if args.placement in {LORA_ZERO2_RESIDENT, ROLLOUT_FUSED_RESIDENT}
                else (
                    "DDP_TRAINABLE_REDUCER_ONLY"
                    if args.placement
                    not in {
                        "resident",
                        CLEAN_RESIDENT,
                        ASYNC_128_RESIDENT,
                        ASYNC_128_TREE_RESIDENT,
                        ASYNC_128_TREE_CHUNK2048_RESIDENT,
                        ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
                        ASYNC_256_RESIDENT,
                        LOCAL_FAST_RESIDENT,
                        FLAT_256_RESIDENT,
                        FLAT_512_RESIDENT,
                        ROLLOUT_FUSED_RESIDENT,
                    }
                    else "NONE"
                )
            ),
            "reshard_after_forward": False,
            "param_compute_dtype_policy": "PRESERVE_ORIGINAL_PER_PARAMETER",
            "frozen_parameter_dtypes": frozen_dtypes,
            "trainable_parameter_dtypes": trainable_dtypes,
            "reduce_dtype": "torch.float32",
            "gradient_reduction": "SUM_THEN_DIVIDE_WORLD_SIZE",
            "cpu_parameter_offload": False,
            "parameter_streaming": False,
            "transition": False,
        }
    else:
        raise ValueError(args.placement)

    clean_model_state = None
    if args.placement in {
        CLEAN_RESIDENT,
        ASYNC_128_RESIDENT,
        ASYNC_128_TREE_RESIDENT,
        ASYNC_128_TREE_CHUNK2048_RESIDENT,
        ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
        ASYNC_256_RESIDENT,
        LOCAL_FAST_RESIDENT,
        FLAT_256_RESIDENT,
        FLAT_512_RESIDENT,
        LORA_ZERO2_RESIDENT,
        ROLLOUT_FUSED_RESIDENT,
    }:
        clean_model_state = install_resident_clean_model_state(model)

    compile_runtime = install_generator_nograd_block_compile(
        model,
        placement=args.placement,
        role=role,
    )
    metadata["nograd_compile_runtime"] = compile_runtime
    lora_cast_cache = install_generator_nograd_lora_cast_cache(
        model,
        role=role,
    )
    metadata["generator_nograd_lora_cast_cache"] = lora_cast_cache
    lora_fused_epilogue = install_generator_nograd_lora_fused_epilogue(
        model,
        role=role,
    )
    metadata["generator_nograd_lora_fused_epilogue"] = lora_fused_epilogue
    metadata["grad_lora_fused_epilogue"] = install_grad_lora_fused_epilogue(model, role=role)
    lora_operator_attribution = install_lora_operator_attribution(
        model,
        role=role,
        profile_mode=bool(args.profile_mode),
        global_rank=int(torch.distributed.get_rank()),
    )
    metadata["lora_operator_attribution"] = lora_operator_attribution

    audit_records: list[dict[str, Any]] | str = "OMITTED_IN_LOW_OVERHEAD_MODE"
    if not (
        args.timing_mode
        or args.profile_mode
        or getattr(args, "trajectory_mode", False)
    ):
        audit = LoRAComputeDtypeAudit(model)
        audit_records = audit.records
        # Keep the hooks alive through the run without serializing the object.
        model._distillschedule_lora_dtype_audit = audit

    gc.collect()
    host_trimmed = base.trim_host_allocator()
    torch.cuda.empty_cache()
    return model, {
        "metadata": metadata,
        "bootstrap_restore": restore,
        "placement": placement,
        "lora_compute_dtype_contract": {
            "frozen_base": "torch.bfloat16",
            "trainable_lora": "torch.float32",
            "records": audit_records,
        },
        "host_allocator_trimmed": host_trimmed,
        "resident_clean_model_state_initial": (
            clean_model_state.receipt() if clean_model_state is not None else None
        ),
        "memory_after_placement": base._memory(device),
    }


def _ema_factory(
    module: Any,
    *,
    optimization_step: int = 10,
    decay: float = 0.999,
    update_after_step: int = 0,
) -> FairTrainableEMA | ShardedTrainableEMA:
    if hasattr(module, "_distillschedule_lora_zero2_engine"):
        return ShardedTrainableEMA(
            module,
            optimization_step=optimization_step,
            decay=decay,
            update_after_step=update_after_step,
        )
    implementation = os.environ.get(
        "DISTILLSCHEDULE_WAN13B_EMA_IMPLEMENTATION", "reference"
    )
    return FairTrainableEMA(
        module,
        optimization_step=optimization_step,
        decay=decay,
        update_after_step=update_after_step,
        implementation=implementation,
    )


base.PLACEMENTS = FAIR_PLACEMENTS
base._build_model = _build_model_fair
base.build_fresh_engine = build_fair_engine
base.model_trainable_metrics = fair_model_trainable_metrics
base.FreshTrainableEMA = _ema_factory


if __name__ == "__main__":
    raise SystemExit(base.main())
