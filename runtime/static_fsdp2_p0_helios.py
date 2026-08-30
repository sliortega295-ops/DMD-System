"""Helios model operations for the static-FSDP2 P0 schedule."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import torch

from distillschedule.runtime.helios_cross_kv_cache import (
    generator_nograd_cross_kv_cache_scope,
)
from distillschedule.runtime.helios_residual_fusion import (
    generator_nograd_residual_fusion_scope,
)
from distillschedule.runtime.fsdp2_p0 import (
    estimate_rollout_collective_groups,
    make_batched_loaded_entry,
)
from runtime.operation_trace import tensor_sha256
from runtime.rollout_trace import ForceNoGradCalls, LightweightNestedCallAudit, NestedCallTracer


def _memory_snapshot(device: torch.device) -> dict[str, float]:
    free, total = torch.cuda.mem_get_info(device)
    gib = float(2**30)
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "max_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "max_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
        "driver_free_gib": free / gib,
        "driver_total_gib": total / gib,
    }


@torch.no_grad()
def run_generator_trace_rollout_batch(
    *,
    model: torch.nn.Module,
    loaded_entries: Sequence[Any],
    device: torch.device,
    detailed_profile: bool = False,
    record_output_sha256: bool = False,
    collect_diagnostics: bool = False,
    diagnostics_scope: Any | None = None,
    use_cross_kv_cache: bool = False,
    synchronize_after_rollout: bool = True,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Run multiple step-qualified critic rollouts in one physical B>1 trace.

    Every member must have the same frozen pyramid-stage schedule. Random inputs
    and histories remain independently replayed by the batched operation-trace
    adapter; only the physical model calls are shared.
    """

    from helios.scheduler.scheduling_helios import HeliosScheduler
    from helios.utils.utils_helios_post import run_generator

    loaded = make_batched_loaded_entry(tuple(loaded_entries))
    tensors = loaded.tensors
    validation = loaded.operation_trace.validation
    expected_call_count = int(validation["call_count"])
    expected_stage_calls = {
        int(stage): int(count)
        for stage, count in validation["stage_call_count"].items()
    }
    gt_all_data = (
        tensors["entry_noise"],
        tensors["indices_hidden_states"],
        tensors["indices_history_short"],
        tensors["indices_history_mid"],
        tensors["indices_history_long"],
        tensors["history_short"],
        tensors["history_mid"],
        tensors["history_long"],
        tensors["continuous_history"],
    )
    scheduler = HeliosScheduler(
        shift=1.0,
        stages=3,
        stage_range=[0, 1 / 3, 2 / 3, 1],
        gamma=1 / 3,
    )
    sigmas = torch.linspace(
        1.0, 1.0 / 1000, 1000, device=device, dtype=torch.float64
    )
    timesteps = sigmas * 1000
    args = SimpleNamespace(
        training_config=SimpleNamespace(
            base_seq_len=256,
            max_seq_len=4096,
            base_shift=0.5,
            max_shift=1.15,
            saturation_ratio_min=0.3,
            saturation_ratio_max=1.7,
            saturation_ratio_clean_prob=0.1,
        )
    )
    tracer: Any
    if detailed_profile:
        tracer = NestedCallTracer(
            model,
            base_height=48,
            base_width=80,
            num_stages=3,
            synchronize=lambda: torch.cuda.synchronize(device),
            before_call=lambda: torch.cuda.reset_peak_memory_stats(device),
            memory_snapshot=lambda: _memory_snapshot(device),
            retain_output_gradients=False,
        )
    else:
        tracer = LightweightNestedCallAudit(
            model,
            base_height=48,
            base_width=80,
            num_stages=3,
        )
    physical_call = ForceNoGradCalls(tracer)
    model.enable_adapters()
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    with generator_nograd_cross_kv_cache_scope(
        model,
        enabled=use_cross_kv_cache,
        expected_model_calls=expected_call_count,
        source_encoder_hidden_states=tensors["text_embedding"],
    ) as cross_kv_cache, generator_nograd_residual_fusion_scope():
        (
            generated_x0,
            gradient_mask,
            denoised_from,
            denoised_to,
            consistency_loss,
        ) = run_generator(
            args=args,
            accelerator=SimpleNamespace(device=device, is_main_process=False),
            transformer=physical_call,
            scheduler=scheduler,
            noise=tensors["entry_noise"],
            prompt_embeds=tensors["text_embedding"],
            dmd_is_low_vram_mode=False,
            is_keep_x0=True,
            history_sizes=[16, 2, 1],
            is_enable_stage2=True,
            stage2_num_stages=3,
            stage2_num_inference_steps_list=[2, 2, 2],
            denoising_step_list=[1000, 750, 500, 250],
            last_step_only=False,
            last_section_grad_only=False,
            return_sim_step=False,
            sigmas=sigmas,
            timesteps=timesteps,
            timestep_shift=5.0,
            use_dynamic_shifting=True,
            time_shift_type="linear",
            num_critic_input_frames=9,
            num_rollout_sections=1,
            is_skip_first_section=False,
            is_amplify_first_chunk=True,
            is_corrupt_history_latents=True,
            is_add_saturation=True,
            is_use_gt_history=True,
            gt_all_data=gt_all_data,
            is_dmd_vae_decode=False,
            is_multi_pyramid_stage_backward_simulated=False,
            init_pyramid_stage_flag=2,
            is_consistency_align=False,
            use_kv_cache=False,
            operation_trace=loaded.operation_trace,
            trace_prefix=loaded.pool_entry.synthetic_trace_prefix,
        )
    if synchronize_after_rollout:
        torch.cuda.synchronize(device)
    batch_size = len(loaded.members)
    if list(generated_x0.shape) != [batch_size, 16, 9, 48, 80]:
        raise AssertionError(
            "batched Generator returned a malformed x0: "
            f"{list(generated_x0.shape)} != {[batch_size, 16, 9, 48, 80]}"
        )
    if generated_x0.dtype != torch.bfloat16:
        raise AssertionError("batched Generator wire contract requires BF16 x0")
    if generated_x0.requires_grad or generated_x0.grad_fn is not None:
        raise AssertionError("batched critic rollout retained Generator autograd")
    if len(tracer.calls) != expected_call_count:
        raise AssertionError(
            f"batched physical call count changed: {len(tracer.calls)} != {expected_call_count}"
        )
    stage_calls = {
        stage: sum(int(event["stage"]) == stage for event in tracer.calls)
        for stage in range(3)
    }
    if stage_calls != expected_stage_calls:
        raise AssertionError(
            f"batched stage calls changed: {stage_calls} != {expected_stage_calls}"
        )
    if any(
        event["grad_enabled"] or event.get("output_requires_grad", False)
        for event in tracer.calls
    ):
        raise AssertionError("a batched Generator call escaped no-grad")

    outputs = tuple(
        generated_x0.narrow(0, index, 1).detach() for index in range(batch_size)
    )
    details: dict[str, Any] = {
        "trace_key": loaded.pool_entry.trace_key,
        "trace_identity_sha256": loaded.pool_entry.identity_sha256,
        "member_trace_identities": [
            value.pool_entry.identity_sha256 for value in loaded.members
        ],
        "logical_steps": [int(value.pool_entry.logical_step) for value in loaded.members],
        "batch_size": batch_size,
        "call_count": len(tracer.calls),
        "stage_call_count": stage_calls,
        "calls": tracer.calls,
        "output_shape": list(generated_x0.shape),
        "output_dtype": str(generated_x0.dtype),
        "output_requires_grad": False,
        "synchronize_after_rollout": bool(synchronize_after_rollout),
        "gradient_mask_is_none": gradient_mask is None,
        "materialization_estimate": estimate_rollout_collective_groups(
            physical_calls_per_rollout=expected_call_count,
            logical_rollouts=batch_size,
            batch_size=batch_size,
        ),
        "memory": _memory_snapshot(device),
        "cross_kv_cache": cross_kv_cache,
    }
    if collect_diagnostics:
        details.update(
            denoised_timestep_from=float(torch.as_tensor(denoised_from).flatten()[0]),
            denoised_timestep_to=float(torch.as_tensor(denoised_to).flatten()[0]),
            consistency_loss=float(torch.as_tensor(consistency_loss)),
        )
    else:
        details.update(
            denoised_timestep_from="omitted_in_clean_profile",
            denoised_timestep_to="omitted_in_clean_profile",
            consistency_loss="omitted_in_clean_profile",
        )
    if record_output_sha256 and collect_diagnostics:
        hashes = []
        for index, output in enumerate(outputs):
            if diagnostics_scope is None:
                hashes.append(tensor_sha256(output))
            else:
                step = int(loaded.members[index].pool_entry.logical_step)
                with diagnostics_scope(f"DIAG_OUTPUT_SHA_D2H/generator_x0_step{step}"):
                    hashes.append(tensor_sha256(output))
        details["output_sha256"] = hashes
    return outputs, details


__all__ = ["run_generator_trace_rollout_batch"]
