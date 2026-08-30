"""Real-14B Helios operations for the isolated Step15-A G/S/T target graph.

The functions here deliberately do not own placement, transport or search
policy.  They consume immutable rank-qualified trace entries and implement the
same Generator rollout and detached DMD boundary for every candidate.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from .dmd_boundary import add_noise, dmd_input_gradient, shifted_sigma_schedule
from .gst_pool_trace import GSTTraceEntry
from .rollout_trace import LightweightNestedCallAudit, NestedCallTracer


def _module(value: Any) -> torch.nn.Module:
    module = getattr(value, "module", value)
    if not isinstance(module, torch.nn.Module):
        raise TypeError("expected a PyTorch module or DeepSpeed engine")
    return module


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


def run_generator_update_trace_rollout(
    *,
    engine: Any,
    entry: GSTTraceEntry,
    device: torch.device,
    detailed_profile: bool,
    collect_diagnostics: bool = True,
    empty_cache_before_rollout: bool = True,
    synchronize_after_rollout: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Replay one grad-selected Generator-update sample without RNG sampling."""

    from helios.scheduler.scheduling_helios import HeliosScheduler
    from helios.utils.utils_helios_post import run_generator

    tensors = entry.tensors
    operation = entry.operation_trace.operation(entry.rollout_operation)
    control = operation["control"]
    expected_call_count = int(control["call_count"])
    expected_stage_calls = {
        int(item["stage"]): int(item["call_count"])
        for item in control["pyramid_stage_call_sequence"]
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
    if detailed_profile:
        tracer: Any = NestedCallTracer(
            engine,
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
            engine, base_height=48, base_width=80, num_stages=3
        )
    model = _module(engine)
    model.enable_adapters()
    model.train()
    if empty_cache_before_rollout:
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    (
        generated_x0,
        gradient_mask,
        denoised_from,
        denoised_to,
        consistency_loss,
    ) = run_generator(
        args=args,
        accelerator=SimpleNamespace(device=device, is_main_process=False),
        transformer=tracer,
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
        operation_trace=entry.operation_trace,
        trace_prefix=f"step/{entry.logical_step}/generator",
    )
    if synchronize_after_rollout:
        torch.cuda.synchronize(device)
    if list(generated_x0.shape) != [1, 16, 9, 48, 80]:
        raise AssertionError("Generator returned a reduced or malformed x0")
    if generated_x0.dtype != torch.bfloat16:
        raise AssertionError("Generator-update output must be BF16")
    if not generated_x0.requires_grad or generated_x0.grad_fn is None:
        raise AssertionError("Generator-update output lost its autograd graph")
    expected_physical_call_count = expected_call_count
    expected_physical_stage_calls = expected_stage_calls
    if len(tracer.calls) != expected_physical_call_count:
        raise AssertionError(
            "physical trace call count changed: "
            f"{len(tracer.calls)} != {expected_physical_call_count}"
        )
    stage_calls = {
        stage: sum(int(event["stage"]) == stage for event in tracer.calls)
        for stage in range(3)
    }
    if stage_calls != expected_physical_stage_calls:
        raise AssertionError(
            "physical trace stage calls changed: "
            f"{stage_calls} != {expected_physical_stage_calls}"
        )
    selected = [
        int(event["call_index"])
        for event in tracer.calls
        if event.get("grad_enabled") and event.get("output_requires_grad")
    ]
    selected_stages = {
        int(event["stage"])
        for event in tracer.calls
        if event.get("grad_enabled") and event.get("output_requires_grad")
    }
    if len(selected) != 3 or selected_stages != {0, 1, 2}:
        raise AssertionError(
            "Generator-update selected-call contract changed: "
            f"{selected}/{selected_stages}"
        )
    return generated_x0, gradient_mask, {
        **entry.identity,
        "trace_identity_sha256": entry.identity_sha256,
        "operation_trace_hash": entry.operation_trace.trace_hash,
        "call_count": len(tracer.calls),
        "stage_call_count": stage_calls,
        "logical_call_count": expected_call_count,
        "logical_stage_call_count": expected_stage_calls,
        "dummy_call_count": 0,
        "selected_call_indices": selected,
        "calls": tracer.calls,
        "output_shape": list(generated_x0.shape),
        "output_dtype": str(generated_x0.dtype),
        "output_requires_grad": True,
        "gradient_mask_is_none": gradient_mask is None,
        "empty_cache_before_rollout": bool(empty_cache_before_rollout),
        "synchronize_after_rollout": bool(synchronize_after_rollout),
        "denoised_timestep_from": (
            float(torch.as_tensor(denoised_from).detach().flatten()[0].cpu())
            if collect_diagnostics
            else "omitted_in_clean_profile"
        ),
        "denoised_timestep_to": (
            float(torch.as_tensor(denoised_to).detach().flatten()[0].cpu())
            if collect_diagnostics
            else "omitted_in_clean_profile"
        ),
        "consistency_loss": (
            float(torch.as_tensor(consistency_loss).detach().cpu())
            if collect_diagnostics
            else "omitted_in_clean_profile"
        ),
        "memory": _memory_snapshot(device),
    }


def prepare_dmd_score_batch(
    *, generated_x0: torch.Tensor, entry: GSTTraceEntry, device: torch.device
) -> dict[str, torch.Tensor]:
    """Use traced Step15 DMD noise/timestep rather than consuming RNG."""

    from helios.utils.utils_base import apply_schedule_shift

    x0 = generated_x0.detach().to(device=device, dtype=torch.bfloat16)
    sigmas = torch.linspace(
        1.0, 1.0 / 1000, 1000, device=device, dtype=torch.float64
    )
    sigmas = apply_schedule_shift(
        sigmas,
        x0,
        base_seq_len=256,
        max_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
        time_shift_type="linear",
    )
    timesteps = sigmas * 1000
    timestep = entry.tensors["dmd_timestep"].to(device=device)
    noise = entry.tensors["dmd_noise"].to(
        device=device, dtype=torch.bfloat16
    )
    noisy = add_noise(x0, noise, timestep, sigmas, timesteps).detach()
    return {
        "generated_x0": x0,
        "hidden_states": noisy,
        "timestep": timestep,
        "sigmas": sigmas,
        "timesteps": timesteps,
        "encoder_hidden_states": entry.tensors["text_embedding"],
        "indices_hidden_states": entry.tensors["indices_hidden_states"],
        "indices_latents_history_short": entry.tensors["indices_history_short"],
        "indices_latents_history_mid": entry.tensors["indices_history_mid"],
        "indices_latents_history_long": entry.tensors["indices_history_long"],
        "latents_history_short": entry.tensors["history_short"],
        "latents_history_mid": entry.tensors["history_mid"],
        "latents_history_long": entry.tensors["history_long"],
        "dmd_noise": noise,
    }


@torch.no_grad()
def score_forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    condition: torch.Tensor,
    adapter_enabled: bool,
) -> torch.Tensor:
    """One resident Student or Teacher Score call with explicit adapter mode."""

    has_adapter = bool(getattr(model, "_distillschedule_has_adapter", True))
    if adapter_enabled:
        if not has_adapter:
            raise AssertionError("Fake Score call was routed to an adapter-free model")
        model.enable_adapters()
    elif has_adapter and hasattr(model, "disable_adapters"):
        model.disable_adapters()
    model.train()
    prediction = model(
        hidden_states=batch["hidden_states"],
        timestep=batch["timestep"],
        encoder_hidden_states=condition,
        indices_hidden_states=batch["indices_hidden_states"],
        indices_latents_history_short=batch["indices_latents_history_short"],
        indices_latents_history_mid=batch["indices_latents_history_mid"],
        indices_latents_history_long=batch["indices_latents_history_long"],
        latents_history_short=batch["latents_history_short"],
        latents_history_mid=batch["latents_history_mid"],
        latents_history_long=batch["latents_history_long"],
        return_dict=False,
    )[0]
    if prediction.requires_grad or prediction.grad_fn is not None:
        raise AssertionError("resident Score inference unexpectedly retained autograd")
    if not torch.isfinite(prediction).all():
        raise AssertionError("resident Score inference returned non-finite values")
    return prediction


@torch.no_grad()
def construct_dmd_gradient(
    *,
    batch: dict[str, torch.Tensor],
    fake_flow_cond: torch.Tensor,
    real_flow_cond: torch.Tensor,
    real_flow_uncond: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    gradient, auxiliary = dmd_input_gradient(
        estimated_clean=batch["generated_x0"],
        noisy_sample=batch["hidden_states"],
        timestep=batch["timestep"],
        sigmas=batch["sigmas"],
        timesteps=batch["timesteps"],
        fake_flow_cond=fake_flow_cond,
        real_flow_cond=real_flow_cond,
        real_flow_uncond=real_flow_uncond,
        fake_guidance_scale=0.0,
        real_guidance_scale=3.0,
        normalization=True,
    )
    if gradient.requires_grad or gradient.grad_fn is not None:
        raise AssertionError("detached DMD boundary retained Score autograd")
    if not torch.isfinite(gradient).all():
        raise AssertionError("DMD gradient contains non-finite values")
    return gradient, auxiliary
