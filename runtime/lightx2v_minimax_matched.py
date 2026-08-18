"""Matched LightX2V MiniMax-H3 native vs DistillSchedule benchmark runtime.

The two modes execute the same model/data/loss contract and the same application-
level DiT inventory.  The DistillSchedule mode changes only the order of the five
Fake-stage Student no-grad rollouts: it precomputes them as one Student phase,
then commits the five Fake updates in the original order.
"""
from __future__ import annotations

import hashlib
import json
import contextlib
import os
import time
from pathlib import Path
from typing import Any

EXPECTED_FORWARD_COUNTS = {"student": 24, "fake": 6, "teacher": 1}
EXPECTED_GRAD_FORWARD_COUNTS = {"student": 1, "fake": 5, "teacher": 0}
EXPECTED_BACKWARD_COUNTS = {"student": 1, "fake": 5}
EXPECTED_STAGE_SEQUENCE = ["student"] + ["fake"] * 5
EXPECTED_END_STEP_IDX = 3
EXPECTED_FAKE_UPDATES = 5


def expected_application_inventory() -> dict[str, Any]:
    return {
        "forward_counts": dict(EXPECTED_FORWARD_COUNTS),
        "grad_forward_counts": dict(EXPECTED_GRAD_FORWARD_COUNTS),
        "backward_counts": dict(EXPECTED_BACKWARD_COUNTS),
        "total_forwards": sum(EXPECTED_FORWARD_COUNTS.values()),
        "total_backward_graphs": sum(EXPECTED_BACKWARD_COUNTS.values()),
        "stage_sequence": list(EXPECTED_STAGE_SEQUENCE),
        "fixed_end_step_idx": EXPECTED_END_STEP_IDX,
    }


def validate_matched_receipt(receipt: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    inventory = receipt.get("application_inventory", {})
    expected = expected_application_inventory()
    for key in (
        "forward_counts",
        "grad_forward_counts",
        "backward_counts",
        "total_forwards",
        "total_backward_graphs",
        "stage_sequence",
        "fixed_end_step_idx",
    ):
        if inventory.get(key) != expected[key]:
            errors.append(f"{key}: {inventory.get(key)!r} != {expected[key]!r}")
    if receipt.get("status") != "PASS":
        errors.append(f"status={receipt.get('status')!r}")
    return errors


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sample_identity(sample: Any) -> dict[str, Any]:
    meta = sample.get("meta", {}) if isinstance(sample, dict) else {}
    if not isinstance(meta, dict):
        return {"meta": repr(meta)}
    keys = (
        "id",
        "row_index",
        "condition_path",
        "prompt_path",
        "lmdb_path",
        "video_latent_path",
        "audio_latent_path",
    )
    return {key: str(meta[key]) for key in keys if key in meta}


def _local_tensor(value: Any) -> Any:
    return value.to_local() if hasattr(value, "to_local") else value


def _hash_tensor_sequence(torch: Any, values: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in values:
        digest.update(name.encode())
        if torch.is_tensor(value):
            local = _local_tensor(value).detach().contiguous()
            digest.update(str(local.dtype).encode())
            digest.update(str(tuple(local.shape)).encode())
            digest.update(local.view(torch.uint8).cpu().numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def _hash_params(torch: Any, params: list[Any]) -> str:
    return _hash_tensor_sequence(
        torch,
        [(f"param/{index}", parameter) for index, parameter in enumerate(params)],
    )


def _hash_optimizer(torch: Any, optimizer: Any, params: list[Any]) -> str:
    rows: list[tuple[str, Any]] = []
    for index, parameter in enumerate(params):
        state = optimizer.state.get(parameter, {})
        for key in sorted(state):
            rows.append((f"state/{index}/{key}", state[key]))
    return _hash_tensor_sequence(torch, rows)


def create_minimax_matched_trainer(mode: str, *, receipt_dir: Path, metadata: dict[str, Any]) -> Any:
    """Create an instrumented MiniMax-H3 trainer for one matched benchmark mode."""
    if mode not in {"native", "distillschedule"}:
        raise ValueError(f"unsupported MiniMax matched mode: {mode}")

    import torch
    import torch.distributed as dist
    from loguru import logger
    from lightx2v_train.runtime.distributed import (
        barrier,
        get_rank,
        get_world_size,
        is_main_process,
        reduce_mean,
    )
    from lightx2v_train.trainers.dmd.math import weighted_mse_pair
    from lightx2v_train.trainers.dmd.minimax_h3_trainer import MiniMaxH3T2AVDmdTrainer

    class MatchedMiniMaxTrainer(MiniMaxH3T2AVDmdTrainer):
        benchmark_mode = mode

        def __init__(self, config: dict[str, Any]):
            super().__init__(config)
            self._benchmark_receipt_dir = Path(receipt_dir)
            self._benchmark_metadata = dict(metadata)
            self._benchmark_forward_counts = {key: 0 for key in EXPECTED_FORWARD_COUNTS}
            self._benchmark_grad_forward_counts = {key: 0 for key in EXPECTED_GRAD_FORWARD_COUNTS}
            self._benchmark_backward_counts = {key: 0 for key in EXPECTED_BACKWARD_COUNTS}
            self._benchmark_stage_sequence: list[str] = []
            self._benchmark_samples: list[dict[str, Any]] = []
            self._benchmark_started: float | None = None
            self._benchmark_local_seconds: float | None = None
            self._benchmark_global_seconds: float | None = None
            self._benchmark_rng_neutral_fake_updates: list[bool] = []
            self._benchmark_force_end_step = False
            self._benchmark_phase_order: list[str] = []
            self._memory_attribution = None
            self._memory_call_index = 0
            self._student_activation_offload = os.environ.get(
                "LIGHTX2V_MINIMAX_ACTIVATION_OFFLOAD", "0"
            ) == "1"
            raw_segment = os.environ.get("LIGHTX2V_MINIMAX_CHECKPOINT_SEGMENT", "0")
            self._checkpoint_segment = int(raw_segment)
            if self._checkpoint_segment not in (0, 2, 4, 5, 8):
                raise RuntimeError(
                    "LIGHTX2V_MINIMAX_CHECKPOINT_SEGMENT must be 0 (native) or one of 2,4,5,8"
                )

        def setup(self, *args: Any, **kwargs: Any):
            result = super().setup(*args, **kwargs)
            if self._checkpoint_segment:
                if self._student_activation_offload:
                    raise RuntimeError(
                        "segmented checkpoint canary requires activation offload disabled"
                    )
                from runtime.lightx2v_segmented_checkpoint import install_segmented_checkpoint

                installed: dict[str, Any] = {}
                # Both trainable backbones use the same checkpoint contract.  The
                # teacher is no-grad and is intentionally left untouched.
                for role, model in (("student", self.model), ("fake", self.fake_model)):
                    installed[role] = install_segmented_checkpoint(model, self._checkpoint_segment)
                self._segmented_checkpoint_receipt = installed
            else:
                self._segmented_checkpoint_receipt = {
                    "enabled": False,
                    "segment": 0,
                    "block_count": 50,
                    "fsdp_granularity": "one_fully_shard_per_transformer_block",
                    "activation_offload": False,
                }
            return result

        def _memory_attribution_begin(self) -> None:
            if self._memory_attribution is not None:
                return
            from runtime.lightx2v_memory_attribution import MemoryAttribution

            self._memory_attribution = MemoryAttribution(
                torch,
                rank=get_rank(),
                role_objects={
                    "student": self.model,
                    "fake": self.fake_model,
                    "teacher": self.teacher_model,
                },
            )

        def _memory_optimizers(self) -> dict[str, Any]:
            return {
                "student": getattr(self, "optimizer", None),
                "fake": getattr(self, "fake_optimizer", None),
            }

        def close_memory_attribution(self) -> None:
            if self._memory_attribution is not None:
                self._memory_attribution.close()

        def _iter_train_samples(self):
            parent = super()._iter_train_samples()
            for sample in parent:
                if len(self._benchmark_samples) < 6:
                    self._benchmark_samples.append(_sample_identity(sample))
                yield sample

        def _sample_synced_int(self, low: int, high: int) -> int:
            if (
                self._benchmark_force_end_step
                and int(low) == 0
                and int(high) == int(self.num_inference_steps)
            ):
                return EXPECTED_END_STEP_IDX
            return super()._sample_synced_int(low, high)

        def forward_loss(self, *args: Any, **kwargs: Any):
            previous = self._benchmark_force_end_step
            self._benchmark_force_end_step = True
            try:
                if self._memory_attribution is None:
                    return super().forward_loss(*args, **kwargs)
                with self._memory_attribution.saved_tensors():
                    return super().forward_loss(*args, **kwargs)
            finally:
                self._benchmark_force_end_step = previous

        def _predict_velocity(self, model: Any, *args: Any, **kwargs: Any):
            if model is self.model:
                role = "student"
            elif model is self.fake_model:
                role = "fake"
            elif model is self.teacher_model:
                role = "teacher"
            else:
                raise AssertionError("MiniMax matched benchmark observed an unknown DiT role")
            self._benchmark_forward_counts[role] += 1
            if torch.is_grad_enabled():
                self._benchmark_grad_forward_counts[role] += 1
            self._memory_call_index += 1
            call_index = self._memory_call_index
            marker = f"MINIMAX_MEMORY_ATTRIBUTION/{role}/forward/{call_index}"
            if self._memory_attribution is not None and self._memory_attribution.enabled:
                self._memory_attribution.record(
                    "forward_pre", phase="model_forward", role=role, call_index=call_index,
                    optimizers=self._memory_optimizers(),
                    extra={"grad_enabled": bool(torch.is_grad_enabled())},
                )
                torch.cuda.nvtx.range_push(marker)
            try:
                result = super()._predict_velocity(model, *args, **kwargs)
            except torch.OutOfMemoryError as exc:
                if self._memory_attribution is not None:
                    self._memory_attribution.exception(
                        exc, phase="model_forward", role=role, call_index=call_index,
                        optimizers=self._memory_optimizers(),
                    )
                raise
            finally:
                if self._memory_attribution is not None and self._memory_attribution.enabled:
                    torch.cuda.nvtx.range_pop()
            if self._memory_attribution is not None and self._memory_attribution.enabled:
                self._memory_attribution.record(
                    "forward_post", phase="model_forward", role=role, call_index=call_index,
                    optimizers=self._memory_optimizers(),
                )
            return result

        def _benchmark_begin(self) -> None:
            if self._benchmark_started is not None:
                return
            barrier()
            torch.cuda.synchronize(self.model.device)
            torch.cuda.reset_peak_memory_stats(self.model.device)
            self._memory_attribution_begin()
            if self._memory_attribution is not None:
                self._memory_attribution.record(
                    "cycle_begin", phase="cycle", optimizers=self._memory_optimizers(),
                )
            torch.cuda.nvtx.range_push(f"MINIMAX_MATCHED_CYCLE/{self.benchmark_mode}")
            self._benchmark_started = time.perf_counter()

        def _benchmark_note_stage_end(self, stage: str) -> None:
            if stage not in EXPECTED_BACKWARD_COUNTS:
                raise ValueError(stage)
            self._benchmark_backward_counts[stage] += 1
            self._benchmark_stage_sequence.append(stage)
            if self._memory_attribution is not None:
                self._memory_attribution.record(
                    "stage_end", phase=stage, role=stage,
                    optimizers=self._memory_optimizers(),
                    extra={"backward_count": self._benchmark_backward_counts[stage]},
                )
            if stage == "fake" and self._benchmark_backward_counts["fake"] == EXPECTED_FAKE_UPDATES:
                torch.cuda.synchronize(self.model.device)
                torch.cuda.nvtx.range_pop()
                barrier()
                if self._benchmark_started is None:
                    raise RuntimeError("MiniMax matched timer never started")
                local_seconds = time.perf_counter() - self._benchmark_started
                value = torch.tensor(local_seconds, device=self.model.device, dtype=torch.float64)
                dist.all_reduce(value, op=dist.ReduceOp.MAX)
                self._benchmark_local_seconds = local_seconds
                self._benchmark_global_seconds = float(value.item())

        def _train_one_stage(self, samples: Any, stage: str, grad_accum_iters: int):
            if int(grad_accum_iters) != 1:
                raise RuntimeError("MiniMax matched benchmark requires gradient_accumulation_iters=1")
            self._benchmark_begin()
            # This is an opt-in capacity canary only.  The context must cover
            # both the Student forward and the backward performed inside the
            # upstream stage implementation; wrapping only forward_loss would
            # release the saved tensors before backward.  No model, loss,
            # trace, optimizer or distributed semantics are changed.
            if self._student_activation_offload and stage == "student":
                context = torch.autograd.graph.save_on_cpu(pin_memory=True)
            else:
                context = contextlib.nullcontext()
            with context:
                result = super()._train_one_stage(samples, stage, grad_accum_iters)
            self._benchmark_note_stage_end(stage)
            return result

        def _assert_contract(self) -> None:
            if int(self.max_train_iters) != 1:
                raise RuntimeError("MiniMax matched benchmark requires exactly one outer iteration")
            if int(self.gradient_accumulation_iters) != 1:
                raise RuntimeError("MiniMax matched benchmark requires gradient accumulation 1")
            if int(self.fake_update_ratio) != EXPECTED_FAKE_UPDATES:
                raise RuntimeError("MiniMax matched benchmark requires fake_update_ratio=5")
            if int(self.num_inference_steps) != 4:
                raise RuntimeError("MiniMax matched benchmark requires four rollout evaluations")
            if self.infer_every_iters:
                raise RuntimeError("MiniMax matched benchmark forbids inference in the timed run")
            if self.save_every_iters:
                raise RuntimeError("MiniMax matched benchmark forbids checkpointing in the timed run")
            if self.diversity_trick.enabled:
                raise RuntimeError("MiniMax matched benchmark forbids diversity loss")
            if self.real_data_fake_trick.enabled_for("main"):
                raise RuntimeError("MiniMax matched benchmark forbids real-data fake auxiliary updates")

        def _candidate_fake_update(self, plan: dict[str, Any]) -> float:
            self.fake_optimizer.zero_grad(set_to_none=True)
            self._set_fake_gradient_sync(True)
            condition = plan["conditions"][0]
            latent_shape = plan["latent_shape"]
            generated = plan["generated"]
            sigmas = plan["sigmas"]
            noises = plan["noises"]
            cpu_rng_before = torch.random.get_rng_state().clone()
            cuda_rng_before = torch.cuda.get_rng_state(self.model.device).clone()
            renoised = self._add_noise(generated, noises, sigmas)
            self.fake_model.transformer.train()
            velocity_fake = self._predict_velocity(
                self.fake_model,
                renoised,
                sigmas,
                condition,
                latent_shape,
            )
            target = (
                generated[0].float() - noises[0],
                generated[1].float() - noises[1],
            )
            loss = weighted_mse_pair(
                velocity_fake,
                target,
                self.video_loss_weight,
                self.audio_loss_weight,
            )
            loss.backward()
            self._sync_sequence_parallel_grads(self.fake_trainable_params)
            torch.nn.utils.clip_grad_norm_(self.fake_trainable_params, self.max_grad_norm)
            self.fake_optimizer.step()
            self.fake_lr_scheduler.step()
            self.fake_optimizer.zero_grad(set_to_none=True)
            rng_neutral = bool(
                torch.equal(cpu_rng_before, torch.random.get_rng_state())
                and torch.equal(cuda_rng_before, torch.cuda.get_rng_state(self.model.device))
            )
            self._benchmark_rng_neutral_fake_updates.append(rng_neutral)
            if not rng_neutral:
                raise RuntimeError(
                    "Fake compute changed global RNG state; phase clustering cannot claim exact input replay"
                )
            self._benchmark_note_stage_end("fake")
            return float(loss.detach().item())

        def _train_distillschedule(self) -> None:
            resume_ckpt_path, current_iter = self._resolve_resume()
            if resume_ckpt_path is not None or current_iter != 0:
                raise RuntimeError("MiniMax matched candidate requires a fresh one-cycle run")
            self.setup(resume_ckpt_path=None)
            self._assert_contract()
            if is_main_process():
                Path(self.output_train_dir).mkdir(parents=True, exist_ok=True)
            barrier()
            samples = self._iter_train_samples()

            self._benchmark_phase_order.append("student_update")
            student_result = self._train_one_stage(samples, stage="student", grad_accum_iters=1)

            plans: list[dict[str, Any]] = []
            for fake_index in range(EXPECTED_FAKE_UPDATES):
                self._benchmark_phase_order.append(f"student_nograd_rollout_{fake_index}")
                sample = next(samples)
                conditions = self._encode_conditions(sample)
                latent_shape = self._latent_shape(sample)
                initial_noise = self.sample_initial_latents(latent_shape)
                generated = self.run_back_simulation(
                    conditions[0],
                    latent_shape,
                    EXPECTED_END_STEP_IDX,
                    grad_enabled=False,
                    xt=initial_noise,
                )
                sigmas = self._sample_renoise_sigmas()
                noises = (
                    torch.randn_like(generated[0], dtype=torch.float32),
                    torch.randn_like(generated[1], dtype=torch.float32),
                )
                plans.append(
                    {
                        "conditions": conditions,
                        "latent_shape": latent_shape,
                        "generated": generated,
                        "sigmas": sigmas,
                        "noises": noises,
                    }
                )
                del initial_noise

            running_fake = 0.0
            for fake_index, plan in enumerate(plans):
                self._benchmark_phase_order.append(f"fake_update_{fake_index}")
                running_fake += self._candidate_fake_update(plan) / EXPECTED_FAKE_UPDATES
                del plan

            current_iter += 1
            display_fake = reduce_mean(running_fake)
            display_dmd = reduce_mean(student_result["dmd"])
            logger.info(
                "[matched:{}] iter={}/1 dmd={:.6f} fake={:.6f}",
                self.benchmark_mode,
                current_iter,
                display_dmd,
                display_fake,
            )

        def train(self):
            if self.benchmark_mode == "native":
                self._assert_contract()
                self._benchmark_phase_order = [
                    "student_update",
                    "fake_rollout+update_0",
                    "fake_rollout+update_1",
                    "fake_rollout+update_2",
                    "fake_rollout+update_3",
                    "fake_rollout+update_4",
                ]
                return super().train()
            return self._train_distillschedule()

        def benchmark_receipt(self) -> dict[str, Any]:
            if self._benchmark_global_seconds is None or self._benchmark_local_seconds is None:
                raise RuntimeError("MiniMax matched cycle did not complete its timing boundary")
            application_inventory = {
                "forward_counts": dict(self._benchmark_forward_counts),
                "grad_forward_counts": dict(self._benchmark_grad_forward_counts),
                "backward_counts": dict(self._benchmark_backward_counts),
                "total_forwards": sum(self._benchmark_forward_counts.values()),
                "total_backward_graphs": sum(self._benchmark_backward_counts.values()),
                "stage_sequence": list(self._benchmark_stage_sequence),
                "fixed_end_step_idx": EXPECTED_END_STEP_IDX,
            }
            value = {
                "schema_version": "distillschedule.lightx2v_minimax_matched.v1",
                "status": "PASS",
                "mode": self.benchmark_mode,
                "rank": get_rank(),
                "world_size": get_world_size(),
                "local_cycle_seconds": self._benchmark_local_seconds,
                "global_cycle_seconds": self._benchmark_global_seconds,
                "application_inventory": application_inventory,
                "sample_sequence": list(self._benchmark_samples),
                "phase_order": list(self._benchmark_phase_order),
                "rng_neutral_fake_updates": list(self._benchmark_rng_neutral_fake_updates),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(self.model.device)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(self.model.device)),
                "state_hashes": {
                    "student_params": _hash_params(torch, list(self.trainable_params)),
                    "fake_params": _hash_params(torch, list(self.fake_trainable_params)),
                    "student_optimizer": _hash_optimizer(torch, self.optimizer, list(self.trainable_params)),
                    "fake_optimizer": _hash_optimizer(torch, self.fake_optimizer, list(self.fake_trainable_params)),
                },
                "checkpoint_policy": getattr(self, "_segmented_checkpoint_receipt", {
                    "enabled": False,
                    "segment": 0,
                    "block_count": 50,
                    "fsdp_granularity": "one_fully_shard_per_transformer_block",
                    "activation_offload": False,
                }),
                **self._benchmark_metadata,
            }
            errors = validate_matched_receipt(value)
            if errors:
                value["status"] = "FAIL"
                value["contract_errors"] = errors
            return value

        def write_benchmark_receipt(self) -> Path:
            receipt = self.benchmark_receipt()
            target = self._benchmark_receipt_dir / f"rank_{get_rank():03d}.json"
            if target.exists():
                raise FileExistsError(target)
            _atomic_json(target, receipt)
            if receipt["status"] != "PASS":
                raise RuntimeError(f"MiniMax matched contract failed: {receipt.get('contract_errors')}")
            return target

    return MatchedMiniMaxTrainer
