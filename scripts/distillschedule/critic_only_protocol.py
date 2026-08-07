"""Protected Critic-only exchange for the real Helios frequency branch."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import torch

from runtime.dmd_boundary import (
    add_noise,
    critic_denoising_loss,
    sample_uniform_timestep,
    shifted_sigma_schedule,
)


def run_critic_only_score_protocol(
    *,
    args: Any,
    engine: Any,
    model: torch.nn.Module,
    result: dict[str, Any],
    device: torch.device,
    rank: int,
    world_size: int,
    source_git_commit: str,
    started: float,
    record: Callable[[str], None],
    barrier: Callable[[], None],
    wait_for_file: Callable[[Path, float], float],
    atomic_torch_save: Callable[[Path, dict[str, Any]], None],
    tensor_sha256: Callable[[torch.Tensor], str],
    file_sha256: Callable[[Path], str],
    trainable_checksum: Callable[[torch.nn.Module, torch.device], dict[str, Any]],
    gradient_stats: Callable[[torch.nn.Module], dict[str, Any]],
    gib: Callable[[int], float],
) -> None:
    """Train shared Score once without running the sparse Generator/DMD branch."""
    exchange_dir = args.integrated_exchange_dir
    exchange_dir.mkdir(parents=True, exist_ok=True)
    critic_input_path = exchange_dir / f"critic_input_rank_{rank:02d}.pt"
    critic_ack_path = exchange_dir / f"critic_ack_rank_{rank:02d}.pt"
    if critic_ack_path.exists():
        raise FileExistsError(
            f"stale Critic ack {critic_ack_path}; use a fresh exchange directory"
        )

    result["frequency_step"] = {
        "kind": "critic_only",
        "generator_update": False,
        "dmd_score": False,
        "critic_update": True,
        "ema_update": False,
        "source_world_size": world_size,
        "score_world_size": world_size,
    }
    result["limitations"].append(
        "Critic-only frequency probe keeps one source sample per rank; it is not a fixed-global-batch placement comparison"
    )
    record("critic_only_score_ready")
    critic_wait_s = wait_for_file(critic_input_path, args.dmd_timeout_s)
    bundle = torch.load(critic_input_path, map_location="cpu", weights_only=False)
    for field, expected in (
        ("schema_version", "distillschedule.real_critic_exchange.v1"),
        ("run_id", args.integrated_run_id),
        ("rank", rank),
        ("world_size", world_size),
        ("source_git_commit", source_git_commit),
        ("step_kind", "critic_only"),
    ):
        if bundle.get(field) != expected:
            raise ValueError(
                f"Critic-only identity mismatch for {field}: "
                f"{bundle.get(field)!r} != {expected!r}"
            )

    generated_x0_cpu = bundle["generated_x0"]
    generated_x0_sha = tensor_sha256(generated_x0_cpu)
    if generated_x0_sha != bundle.get("generated_x0_sha256"):
        raise ValueError("Critic-only x0 digest mismatch")
    generated_x0 = generated_x0_cpu.to(device=device, dtype=torch.bfloat16).detach()
    prompt = bundle["prompt_embeds"].to(device=device, dtype=torch.bfloat16)
    history = {
        name: value.to(device=device)
        for name, value in bundle["history_kwargs"].items()
    }
    sigmas, schedule_timesteps = shifted_sigma_schedule(
        num_train_timesteps=1000,
        timestep_shift=5.0,
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 20_000 + rank)
    timestep = sample_uniform_timestep(
        generated_x0.shape[0],
        num_train_timesteps=1000,
        timestep_shift=5.0,
        device=device,
        generator=generator,
    )
    noise = torch.randn(
        generated_x0.shape,
        device=device,
        dtype=generated_x0.dtype,
        generator=generator,
    )
    noisy_fake = add_noise(
        generated_x0,
        noise,
        timestep,
        sigmas,
        schedule_timesteps,
    ).detach()

    model.enable_adapters()
    model.train()
    barrier()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    forward_started = time.perf_counter()
    prediction = engine(
        hidden_states=noisy_fake,
        timestep=timestep,
        encoder_hidden_states=prompt,
        return_dict=False,
        **history,
    )[0]
    loss = critic_denoising_loss(prediction, noise, generated_x0)
    torch.cuda.synchronize(device)
    prediction_cpu = prediction.detach().cpu()
    noise_cpu = noise.detach().cpu()
    result["integrated_critic_forward"] = {
        "duration_s": time.perf_counter() - forward_started,
        "loss": float(loss.detach().cpu()),
        "loss_kind": "source_default_critic_denoising_mse",
        "producer_kind": "source_nested_generator_rollout",
        "generated_x0_sha256": generated_x0_sha,
        "fake_flow_prediction_sha256": tensor_sha256(prediction_cpu),
        "critic_noise_sha256": tensor_sha256(noise_cpu),
        "timestep": [int(value) for value in timestep.cpu().tolist()],
        "wait_for_input_s": critic_wait_s,
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
        "generator_checksum_after": bundle["generator_checksum_after"],
        "dmd_executed": False,
    }
    record("integrated_critic_forward_complete")

    torch.cuda.reset_peak_memory_stats(device)
    backward_started = time.perf_counter()
    engine.backward(loss)
    torch.cuda.synchronize(device)
    result["integrated_critic_backward"] = {
        "duration_s": time.perf_counter() - backward_started,
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
        "visible_gradients": gradient_stats(model),
    }
    record("integrated_critic_backward_complete")

    torch.cuda.reset_peak_memory_stats(device)
    step_started = time.perf_counter()
    engine.step()
    torch.cuda.synchronize(device)
    result["integrated_critic_optimizer_step"] = {
        "duration_s": time.perf_counter() - step_started,
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
    }
    checksum_after = trainable_checksum(model, device)
    checksum_before = result["trainable_checksum_before"]
    result["trainable_checksum_after"] = checksum_after
    result["trainable_checksum_delta"] = {
        "sum": checksum_after["sum"] - checksum_before["sum"],
        "l2_norm": checksum_after["l2_norm"] - checksum_before["l2_norm"],
    }
    if result["trainable_checksum_delta"]["sum"] == 0.0:
        raise AssertionError("Critic-only step did not advance Score trainables")

    ack = {
        "schema_version": "distillschedule.real_critic_ack.v1",
        "run_id": args.integrated_run_id,
        "rank": rank,
        "source_git_commit": source_git_commit,
        "step_kind": "critic_only",
        "source_generated_x0_sha256": generated_x0_sha,
        "generator_checksum_after": bundle["generator_checksum_after"],
        "score_checksum_before": checksum_before,
        "score_checksum_after_dmd": checksum_before,
        "score_checksum_after_critic": checksum_after,
        "dmd_executed": False,
        "critic_loss": result["integrated_critic_forward"]["loss"],
        "critic_timestep": result["integrated_critic_forward"]["timestep"],
        "critic_noise": noise_cpu,
        "critic_noise_sha256": result["integrated_critic_forward"][
            "critic_noise_sha256"
        ],
        "fake_flow_prediction": prediction_cpu,
        "fake_flow_prediction_sha256": result["integrated_critic_forward"][
            "fake_flow_prediction_sha256"
        ],
    }
    atomic_torch_save(critic_ack_path, ack)
    result["integrated_critic_ack"] = {
        "path": str(critic_ack_path),
        "file_bytes": critic_ack_path.stat().st_size,
        "file_sha256": file_sha256(critic_ack_path),
    }
    record("integrated_critic_optimizer_step_complete")
    result["status"] = "success"
    result["total_duration_s"] = time.perf_counter() - started
    record("complete")
    barrier()


def run_critic_only_generator_boundary(
    *,
    args: Any,
    model: torch.nn.Module,
    result: dict[str, Any],
    generated_x0: torch.Tensor,
    prompt_embeds: torch.Tensor,
    score_history_kwargs: dict[str, torch.Tensor],
    rank: int,
    device: torch.device,
    started: float,
    component_started: float,
    record: Callable[[str], None],
    barrier: Callable[[], None],
    wait_for_file: Callable[[Path, float], float],
    atomic_torch_save: Callable[[Path, dict[str, Any]], None],
    tensor_sha256: Callable[[torch.Tensor], str],
    file_sha256: Callable[[Path], str],
    trainable_checksum: Callable[[torch.nn.Module, torch.device], dict[str, Any]],
) -> None:
    """Publish one detached Critic input and validate the returned version ack."""
    exchange_dir = args.integrated_exchange_dir
    exchange_dir.mkdir(parents=True, exist_ok=True)
    critic_input_path = exchange_dir / f"critic_input_rank_{rank:02d}.pt"
    critic_ack_path = exchange_dir / f"critic_ack_rank_{rank:02d}.pt"
    if critic_ack_path.exists():
        raise FileExistsError(
            f"stale Critic ack {critic_ack_path}; use a fresh exchange directory"
        )
    generated_x0_cpu = generated_x0.detach().cpu()
    x0_sha = tensor_sha256(generated_x0_cpu)
    checksum_before = result["trainable_checksum_before"]
    checksum_current = trainable_checksum(model, device)
    if checksum_current != checksum_before:
        raise AssertionError("Critic-only Generator changed before Score execution")
    result["trainable_checksum_after"] = checksum_current
    result["trainable_checksum_delta"] = {"sum": 0.0, "l2_norm": 0.0}
    result["optimizer_step"] = "Not Run: critic_only frequency branch"
    result["ema_update"] = "Not Run: critic_only frequency branch after EMA start"
    result["frequency_step"] = {
        "kind": "critic_only",
        "generator_update": False,
        "dmd_score": False,
        "critic_update": True,
        "ema_update": False,
        "source_world_size": torch.distributed.get_world_size(),
    }
    bundle = {
        "schema_version": "distillschedule.real_critic_exchange.v1",
        "run_id": args.integrated_run_id,
        "source_git_commit": result["source_git_commit"],
        "rank": rank,
        "world_size": torch.distributed.get_world_size(),
        "step_kind": "critic_only",
        "producer_kind": "source_nested_generator_rollout",
        "generated_x0": generated_x0_cpu,
        "generated_x0_sha256": x0_sha,
        "prompt_embeds": prompt_embeds.detach().cpu(),
        "history_kwargs": score_history_kwargs,
        "generator_checksum_before": checksum_before,
        "generator_checksum_after": checksum_current,
    }
    atomic_torch_save(critic_input_path, bundle)
    result["critic_only_exchange"] = {
        "input_path": str(critic_input_path),
        "input_file_bytes": critic_input_path.stat().st_size,
        "input_file_sha256": file_sha256(critic_input_path),
        "generated_x0_sha256": x0_sha,
    }
    record("critic_only_input_published")
    torch.cuda.empty_cache()
    record("critic_only_generator_cache_released")

    wait_started = time.perf_counter()
    wait_s = wait_for_file(critic_ack_path, args.dmd_timeout_s)
    ack = torch.load(critic_ack_path, map_location="cpu", weights_only=False)
    for field, expected in (
        ("schema_version", "distillschedule.real_critic_ack.v1"),
        ("run_id", args.integrated_run_id),
        ("rank", rank),
        ("source_git_commit", result["source_git_commit"]),
        ("step_kind", "critic_only"),
        ("source_generated_x0_sha256", x0_sha),
        ("generator_checksum_after", checksum_current),
        ("dmd_executed", False),
    ):
        if ack.get(field) != expected:
            raise ValueError(
                f"Critic-only ack mismatch for {field}: {ack.get(field)!r} != {expected!r}"
            )
    noise_cpu = ack["critic_noise"]
    prediction_cpu = ack["fake_flow_prediction"]
    if tensor_sha256(noise_cpu) != ack["critic_noise_sha256"]:
        raise ValueError("Critic-only noise digest mismatch")
    if tensor_sha256(prediction_cpu) != ack["fake_flow_prediction_sha256"]:
        raise ValueError("Critic-only prediction digest mismatch")
    expected_loss = float(
        critic_denoising_loss(prediction_cpu, noise_cpu, generated_x0_cpu).item()
    )
    loss_error = abs(ack["critic_loss"] - expected_loss)
    if loss_error > 1e-6:
        raise AssertionError(f"Critic-only loss identity mismatch: {loss_error}")
    if ack["score_checksum_before"] != ack["score_checksum_after_dmd"]:
        raise AssertionError("Critic-only step reported a nonexistent DMD state change")
    if ack["score_checksum_after_dmd"] == ack["score_checksum_after_critic"]:
        raise AssertionError("Critic-only step did not advance Score")
    barrier()
    checksum_after_score = trainable_checksum(model, device)
    if checksum_after_score != checksum_before:
        raise AssertionError("Score Critic update changed Generator")
    result["integrated_critic_ack"] = {
        "input_path": str(critic_input_path),
        "ack_path": str(critic_ack_path),
        "ack_file_sha256": file_sha256(critic_ack_path),
        "wait_s": wait_s,
        "exchange_to_ack_s": time.perf_counter() - wait_started,
        "critic_loss": ack["critic_loss"],
        "expected_critic_loss": expected_loss,
        "critic_loss_absolute_error": loss_error,
        "critic_timestep": ack["critic_timestep"],
        "critic_noise_sha256": ack["critic_noise_sha256"],
        "fake_flow_prediction_sha256": ack["fake_flow_prediction_sha256"],
        "score_checksum_before": ack["score_checksum_before"],
        "score_checksum_after_dmd": ack["score_checksum_after_dmd"],
        "score_checksum_after_critic": ack["score_checksum_after_critic"],
        "generator_checksum_after_critic": checksum_after_score,
        "dmd_executed": False,
    }
    result["critic_only_component_chain_s"] = time.perf_counter() - component_started
    record("critic_only_ack_received")
    result["status"] = "success"
    result["total_duration_s"] = time.perf_counter() - started
    record("complete")
    barrier()
