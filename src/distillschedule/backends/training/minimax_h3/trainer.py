"""A100-oriented shared-backbone DMD trainer registration and setup."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v_train.runtime.distributed import get_rank
from lightx2v_train.runtime.parallel import apply_parallel
from lightx2v_train.trainers.dmd.minimax_h3_trainer import MiniMaxH3T2AVDmdTrainer
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .activation_offload import maybe_saved_tensor_offload
from .checkpoint_boundary_offload import (
    POLICY_NAME as CHECKPOINT_BOUNDARY_CPU,
    install_checkpoint_boundary_cpu_offload,
)
from .checkpointing import H3A100CheckpointMixin
from .fused_block_pointwise import install_fused_block_pointwise
from .fused_qk_rmsnorm_rotary import install_fused_qk_rmsnorm_rotary
from .fused_rotary import install_fused_rotary
from .fused_swiglu import install_fused_swiglu
from .fa3_nograd_splits import install_fa3_nograd_splits
from .lora_scale1_elision import install_lora_scale1_elision
from .triton_lora_epilogue import warmup_lora_epilogue
from .matched_contract import FIXED_END_STEP_IDX, MatchedCycleCensus
from .model import FAKE_ADAPTER, STUDENT_ADAPTER, MiniMaxH3A100Model
from .trainer_loop import H3A100LoopMixin
from .trainer_runtime import H3A100RuntimeMixin


@TRAINER_REGISTER("minimax_h3_t2av_dmd_a100")
class MiniMaxH3A100DmdTrainer(
    H3A100CheckpointMixin,
    H3A100LoopMixin,
    H3A100RuntimeMixin,
    MiniMaxH3T2AVDmdTrainer,
):
    """One physical H3 backbone, two LoRAs, cached AdaLN, reordered critic."""

    trainer_name = "minimax_h3_t2av_dmd_a100"
    allowed_model_names = {"minimax_h3_t2av_a100"}

    def __init__(self, config):
        super().__init__(config)
        if self.student_train_type != "lora" or self.fake_train_type != "lora":
            raise ValueError("H3 A100 shared-backbone training requires student/fake train_type=lora")
        if self.ida_trick.enabled:
            raise ValueError("H3 A100 shared-backbone trainer does not yet support SenseFlow IDA")

        a100 = self.training_config.get("a100", {})
        cache = a100.get("adaln_cache", {})
        reorder = a100.get("critic_rollout_reorder", {})
        matched = a100.get("matched_compute", {})
        checkpointing = a100.get("activation_checkpointing", {})
        activation_policy = a100.get("activation_policy", {})
        pointwise_fusion = a100.get("block_pointwise_fusion", {})
        rotary_fusion = a100.get("rotary_fusion", {})
        swiglu_fusion = a100.get("swiglu_fusion", {})
        fa3_splits = a100.get("fa3_nograd_splits", {})
        lora_scale1 = a100.get("lora_scale1_elision", {})

        self.adaln_cache_enabled = bool(cache.get("enabled", True))
        self.adaln_dynamic_keys = int(cache.get("max_dynamic_keys", 2))
        self.reorder_critic_rollouts = bool(reorder.get("enabled", True))
        self.critic_rollout_group_size = int(reorder.get("group_size", self.fake_update_ratio))
        if self.critic_rollout_group_size != self.fake_update_ratio:
            raise ValueError(
                "critic_rollout_reorder.group_size must equal "
                f"fake_update_ratio={self.fake_update_ratio} to preserve optimizer-step semantics"
            )

        self.matched_compute_enabled = bool(matched.get("enabled", True))
        self.matched_fixed_end_step_idx = int(matched.get("fixed_end_step_idx", FIXED_END_STEP_IDX))
        self.matched_expected_world_size = int(matched.get("expected_world_size", 16))
        self.matched_require_unique_samples = bool(matched.get("require_unique_samples", True))
        self.matched_cycle_census = MatchedCycleCensus(
            enabled=self.matched_compute_enabled,
            fixed_end_step_idx=self.matched_fixed_end_step_idx,
            expected_world_size=self.matched_expected_world_size,
            require_unique_samples=self.matched_require_unique_samples,
        )

        self.activation_checkpoint_segment_size = int(
            os.environ.get(
                "H3_ACTIVATION_CHECKPOINT_SEGMENT",
                checkpointing.get("segment_size", 1),
            )
        )
        if self.activation_checkpoint_segment_size < 1:
            raise ValueError(
                "H3_ACTIVATION_CHECKPOINT_SEGMENT must be >= 1, got "
                f"{self.activation_checkpoint_segment_size}"
            )

        # Production world16 now mirrors the DMD-System capacity path that
        # actually passed: native per-block checkpointing plus CPU staging of
        # every checkpoint boundary input.  The old thresholded saved-tensor
        # offload remains available only as an explicit diagnostic fallback.
        self.activation_policy = str(
            os.environ.get(
                "H3_ACTIVATION_POLICY",
                activation_policy.get("name", "none"),
            )
        )
        if self.activation_policy not in {"none", CHECKPOINT_BOUNDARY_CPU}:
            raise ValueError(
                "H3_ACTIVATION_POLICY must be 'none' or "
                f"'{CHECKPOINT_BOUNDARY_CPU}', got {self.activation_policy!r}"
            )
        self.boundary_offload_pin_memory = os.environ.get(
            "H3_BOUNDARY_OFFLOAD_PIN_MEMORY",
            str(int(bool(activation_policy.get("pin_memory", True)))),
        ).lower() in {"1", "true", "yes", "on"}
        self.boundary_offload_events = os.environ.get(
            "H3_BOUNDARY_OFFLOAD_EVENTS",
            str(int(bool(activation_policy.get("detailed_events", False)))),
        ).lower() in {"1", "true", "yes", "on"}
        self.boundary_offload_registration = None
        fusion_value = os.environ.get(
            "H3_FUSED_BLOCK_POINTWISE",
            pointwise_fusion.get("enabled", False),
        )
        self.fused_block_pointwise_enabled = (
            fusion_value.lower() in {"1", "true", "yes", "on"}
            if isinstance(fusion_value, str)
            else bool(fusion_value)
        )
        pointwise_grad_value = os.environ.get(
            "H3_FUSED_BLOCK_POINTWISE_GRAD",
            pointwise_fusion.get("grad_enabled", False),
        )
        self.fused_block_pointwise_grad_enabled = (
            pointwise_grad_value.lower() in {"1", "true", "yes", "on"}
            if isinstance(pointwise_grad_value, str)
            else bool(pointwise_grad_value)
        )
        if self.fused_block_pointwise_grad_enabled and not self.fused_block_pointwise_enabled:
            raise ValueError(
                "H3_FUSED_BLOCK_POINTWISE_GRAD requires H3_FUSED_BLOCK_POINTWISE=1"
            )
        rmsnorm_modulate_value = os.environ.get(
            "H3_FUSED_RMSNORM_MODULATE",
            pointwise_fusion.get("rmsnorm_modulate", False),
        )
        self.fused_rmsnorm_modulate_enabled = (
            rmsnorm_modulate_value.lower() in {"1", "true", "yes", "on"}
            if isinstance(rmsnorm_modulate_value, str)
            else bool(rmsnorm_modulate_value)
        )
        if self.fused_rmsnorm_modulate_enabled and not self.fused_block_pointwise_enabled:
            raise ValueError(
                "H3_FUSED_RMSNORM_MODULATE requires H3_FUSED_BLOCK_POINTWISE=1"
            )
        self.fused_block_pointwise_registration = None
        rotary_value = os.environ.get(
            "H3_FUSED_ROTARY",
            rotary_fusion.get("enabled", False),
        )
        self.fused_rotary_enabled = (
            rotary_value.lower() in {"1", "true", "yes", "on"}
            if isinstance(rotary_value, str)
            else bool(rotary_value)
        )
        rotary_grad_value = os.environ.get(
            "H3_FUSED_ROTARY_GRAD",
            rotary_fusion.get("grad_enabled", False),
        )
        self.fused_rotary_grad_enabled = (
            rotary_grad_value.lower() in {"1", "true", "yes", "on"}
            if isinstance(rotary_grad_value, str)
            else bool(rotary_grad_value)
        )
        if self.fused_rotary_grad_enabled and not self.fused_rotary_enabled:
            raise ValueError("H3_FUSED_ROTARY_GRAD requires H3_FUSED_ROTARY=1")
        self.fused_rotary_registration = None
        fused_qk_value = os.environ.get(
            "H3_FUSED_QK_RMSNORM_ROTARY",
            rotary_fusion.get("qk_rmsnorm", False),
        )
        self.fused_qk_rmsnorm_rotary_enabled = (
            fused_qk_value.lower() in {"1", "true", "yes", "on"}
            if isinstance(fused_qk_value, str)
            else bool(fused_qk_value)
        )
        if self.fused_qk_rmsnorm_rotary_enabled and not self.fused_rotary_enabled:
            raise ValueError(
                "H3_FUSED_QK_RMSNORM_ROTARY requires H3_FUSED_ROTARY=1"
            )
        fused_qk_grad_value = os.environ.get(
            "H3_FUSED_QK_RMSNORM_ROTARY_GRAD",
            rotary_fusion.get("qk_rmsnorm_grad", False),
        )
        self.fused_qk_rmsnorm_rotary_grad_enabled = (
            fused_qk_grad_value.lower() in {"1", "true", "yes", "on"}
            if isinstance(fused_qk_grad_value, str)
            else bool(fused_qk_grad_value)
        )
        if (
            self.fused_qk_rmsnorm_rotary_grad_enabled
            and not self.fused_qk_rmsnorm_rotary_enabled
        ):
            raise ValueError(
                "H3_FUSED_QK_RMSNORM_ROTARY_GRAD requires "
                "H3_FUSED_QK_RMSNORM_ROTARY=1"
            )
        if self.fused_qk_rmsnorm_rotary_grad_enabled and not self.fused_rotary_grad_enabled:
            raise ValueError(
                "H3_FUSED_QK_RMSNORM_ROTARY_GRAD requires H3_FUSED_ROTARY_GRAD=1"
            )
        self.fused_qk_rmsnorm_rotary_registration = None
        swiglu_value = os.environ.get(
            "H3_FUSED_SWIGLU",
            swiglu_fusion.get("enabled", False),
        )
        self.fused_swiglu_enabled = (
            swiglu_value.lower() in {"1", "true", "yes", "on"}
            if isinstance(swiglu_value, str)
            else bool(swiglu_value)
        )
        swiglu_grad_value = os.environ.get(
            "H3_FUSED_SWIGLU_GRAD",
            swiglu_fusion.get("grad_enabled", False),
        )
        self.fused_swiglu_grad_enabled = (
            swiglu_grad_value.lower() in {"1", "true", "yes", "on"}
            if isinstance(swiglu_grad_value, str)
            else bool(swiglu_grad_value)
        )
        if self.fused_swiglu_grad_enabled and not self.fused_swiglu_enabled:
            raise ValueError("H3_FUSED_SWIGLU_GRAD requires H3_FUSED_SWIGLU=1")
        self.fused_swiglu_registration = None
        self.fa3_nograd_num_splits = int(
            os.environ.get(
                "H3_FA3_NOGRAD_NUM_SPLITS",
                fa3_splits.get("num_splits", 1),
            )
        )
        if self.fa3_nograd_num_splits not in {1, 2}:
            raise ValueError(
                "H3_FA3_NOGRAD_NUM_SPLITS must be 1 or the bounded candidate 2, "
                f"got {self.fa3_nograd_num_splits}"
            )
        self.fa3_grad_num_splits = int(
            os.environ.get(
                "H3_FA3_GRAD_NUM_SPLITS",
                fa3_splits.get("grad_num_splits", 1),
            )
        )
        if self.fa3_grad_num_splits not in {1, 2}:
            raise ValueError(
                "H3_FA3_GRAD_NUM_SPLITS must be 1 or the bounded candidate 2, "
                f"got {self.fa3_grad_num_splits}"
            )
        self.fa3_nograd_split_registration = None
        lora_scale1_value = os.environ.get(
            "H3_LORA_SCALE1_ELISION",
            lora_scale1.get("enabled", False),
        )
        self.lora_scale1_elision_enabled = (
            lora_scale1_value.lower() in {"1", "true", "yes", "on"}
            if isinstance(lora_scale1_value, str)
            else bool(lora_scale1_value)
        )
        lora_epilogue_value = os.environ.get(
            "H3_LORA_NOGRAD_EPILOGUE",
            lora_scale1.get("fused_nograd_epilogue", False),
        )
        self.lora_nograd_epilogue_enabled = (
            lora_epilogue_value.lower() in {"1", "true", "yes", "on"}
            if isinstance(lora_epilogue_value, str)
            else bool(lora_epilogue_value)
        )
        if self.lora_nograd_epilogue_enabled and not self.lora_scale1_elision_enabled:
            raise ValueError("H3_LORA_NOGRAD_EPILOGUE requires H3_LORA_SCALE1_ELISION=1")
        lora_grad_epilogue_value = os.environ.get(
            "H3_LORA_GRAD_EPILOGUE",
            lora_scale1.get("fused_grad_epilogue", False),
        )
        self.lora_grad_epilogue_enabled = (
            lora_grad_epilogue_value.lower() in {"1", "true", "yes", "on"}
            if isinstance(lora_grad_epilogue_value, str)
            else bool(lora_grad_epilogue_value)
        )
        if self.lora_grad_epilogue_enabled and not self.lora_nograd_epilogue_enabled:
            raise ValueError("H3_LORA_GRAD_EPILOGUE requires H3_LORA_NOGRAD_EPILOGUE=1")
        self.lora_scale1_elision_registration = None

        self.activation_offload_enabled = os.environ.get(
            "H3_ACTIVATION_OFFLOAD", "0"
        ).lower() in {"1", "true", "yes", "on"}
        self.activation_offload_min_bytes = int(
            os.environ.get("H3_ACTIVATION_OFFLOAD_MIN_BYTES", 128 * 1024 * 1024)
        )
        self.activation_offload_pin_memory = os.environ.get(
            "H3_ACTIVATION_OFFLOAD_PIN_MEMORY", "1"
        ).lower() in {"1", "true", "yes", "on"}
        if self.activation_offload_min_bytes < 0:
            raise ValueError("H3_ACTIVATION_OFFLOAD_MIN_BYTES must be >= 0")
        if self.activation_policy == CHECKPOINT_BOUNDARY_CPU:
            if self.activation_checkpoint_segment_size != 1:
                raise ValueError(
                    "checkpoint_boundary_cpu requires native per-block checkpointing: "
                    "H3_ACTIVATION_CHECKPOINT_SEGMENT must be 1"
                )
            if self.activation_offload_enabled:
                raise ValueError(
                    "checkpoint_boundary_cpu is mutually exclusive with the legacy "
                    "H3_ACTIVATION_OFFLOAD threshold policy"
                )
            if not self.gradient_checkpointing:
                raise ValueError("checkpoint_boundary_cpu requires gradient_checkpointing=true")

        self._validate_matched_static_contract()
        self._score_serial = 0

    @property
    def shared_model(self) -> MiniMaxH3A100Model:
        if not isinstance(self.model, MiniMaxH3A100Model):
            raise TypeError(f"Expected MiniMaxH3A100Model, got {type(self.model)!r}")
        return self.model

    def _validate_matched_static_contract(self) -> None:
        if not self.matched_compute_enabled:
            return
        if self.matched_fixed_end_step_idx != FIXED_END_STEP_IDX:
            raise ValueError(
                f"matched_compute.fixed_end_step_idx must be {FIXED_END_STEP_IDX}, "
                f"got {self.matched_fixed_end_step_idx}"
            )
        if int(self.num_inference_steps) != 4:
            raise ValueError("matched MiniMax benchmark requires num_inference_steps=4")
        if int(self.fake_update_ratio) != 5:
            raise ValueError("matched MiniMax benchmark requires fake_update_ratio=5")
        if int(self.gradient_accumulation_iters) != 1:
            raise ValueError("matched MiniMax benchmark requires gradient_accumulation_iters=1")
        if self.infer_every_iters:
            raise ValueError("matched MiniMax benchmark forbids in-run inference")

    def setup(self, resume_ckpt_path=None):
        model = self.shared_model
        model.prepare_shared_backbone(
            student_lora=self.student_lora_config,
            fake_lora=self.fake_lora_config,
            cache_enabled=self.adaln_cache_enabled,
            max_dynamic_cache_keys=self.adaln_dynamic_keys,
        )
        model.configure_activation_checkpoint_segments(self.activation_checkpoint_segment_size)
        self.fused_block_pointwise_registration = install_fused_block_pointwise(
            model.denoiser_module(),
            enabled=self.fused_block_pointwise_enabled,
            grad_enabled=self.fused_block_pointwise_grad_enabled,
            rmsnorm_modulate_enabled=self.fused_rmsnorm_modulate_enabled,
        )
        self.fused_rotary_registration = install_fused_rotary(
            enabled=self.fused_rotary_enabled,
            grad_enabled=self.fused_rotary_grad_enabled,
        )
        # Install after the base rotary patch. Grad/checkpoint-replay calls keep
        # the already validated rotary path; only main-block no-grad attention
        # replaces Q/K RMSNorm plus rotary with the bounded fused kernel.
        self.fused_qk_rmsnorm_rotary_registration = install_fused_qk_rmsnorm_rotary(
            model.denoiser_module(),
            enabled=self.fused_qk_rmsnorm_rotary_enabled,
            grad_enabled=self.fused_qk_rmsnorm_rotary_grad_enabled,
            base_rotary_registration=self.fused_rotary_registration,
        )
        self.fused_swiglu_registration = install_fused_swiglu(
            model.denoiser_module(),
            enabled=self.fused_swiglu_enabled,
            grad_enabled=self.fused_swiglu_grad_enabled,
        )
        self.fa3_nograd_split_registration = install_fa3_nograd_splits(
            num_splits=self.fa3_nograd_num_splits,
            grad_num_splits=self.fa3_grad_num_splits,
        )
        self.lora_scale1_elision_registration = install_lora_scale1_elision(
            model.denoiser_module(),
            enabled=self.lora_scale1_elision_enabled,
            epilogue_enabled=self.lora_nograd_epilogue_enabled,
            grad_epilogue_enabled=self.lora_grad_epilogue_enabled,
        )
        self._validate_reorder_contract()
        if resume_ckpt_path is not None:
            self._load_role_weights_before_fsdp(resume_ckpt_path)

        apply_parallel(model, self.config)
        self._release_full_cpu_checkpoint_storage()
        if self.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        # Compile/load the single Triton specialization before the cycle timer.
        # One local rank compiles per node, then every rank launches the tiny
        # warmup once so no first-use work contaminates the formal cycle.
        if self.lora_nograd_epilogue_enabled:
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                warmup_lora_epilogue(torch.cuda.current_device())
            if dist.is_initialized():
                dist.barrier()
            warmup_lora_epilogue(torch.cuda.current_device())
            if dist.is_initialized():
                dist.barrier()

        if self.fused_rmsnorm_modulate_enabled:
            from .triton_fused_pointwise import warmup_fused_rmsnorm_modulate

            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                warmup_fused_rmsnorm_modulate(torch.cuda.current_device())
            if dist.is_initialized():
                dist.barrier()
            warmup_fused_rmsnorm_modulate(torch.cuda.current_device())
            if dist.is_initialized():
                dist.barrier()

        if self.fused_qk_rmsnorm_rotary_enabled:
            from .triton_fused_rotary import (
                warmup_fused_qk_rmsnorm_rotary,
                warmup_fused_qk_rmsnorm_rotary_backward,
            )

            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                warmup_fused_qk_rmsnorm_rotary(torch.cuda.current_device())
                if self.fused_qk_rmsnorm_rotary_grad_enabled:
                    warmup_fused_qk_rmsnorm_rotary_backward(torch.cuda.current_device())
            if dist.is_initialized():
                dist.barrier()
            warmup_fused_qk_rmsnorm_rotary(torch.cuda.current_device())
            if self.fused_qk_rmsnorm_rotary_grad_enabled:
                warmup_fused_qk_rmsnorm_rotary_backward(torch.cuda.current_device())
            if dist.is_initialized():
                dist.barrier()

        if self.activation_policy == CHECKPOINT_BOUNDARY_CPU:
            event_path = None
            if self.boundary_offload_events:
                event_path = (
                    Path(self.output_train_dir)
                    / "checkpoint_boundary_offload"
                    / f"rank_{get_rank():03d}.jsonl"
                )
            self.boundary_offload_registration = install_checkpoint_boundary_cpu_offload(
                model.denoiser_module(),
                # The shared backbone uses a ContextVar so this returns the
                # logical role that owns the current grad-enabled transformer
                # call without changing adapter or autograd behavior.
                role_getter=lambda: model._role.get(),
                event_path=event_path,
                pin_memory=self.boundary_offload_pin_memory,
            )

        self.trainable_params = model.role_parameters(STUDENT_ADAPTER)
        self.fake_trainable_params = model.role_parameters(FAKE_ADAPTER)
        self._validate_shared_parameter_contract()
        self.optimizer = self._build_optimizer(self.trainable_params)
        self.fake_optimizer = self._build_optimizer(
            self.fake_trainable_params,
            {
                "learning_rate": self.fake_optimizer_learning_rate,
                "adam_beta1": self.fake_optimizer_adam_beta1,
                "adam_beta2": self.fake_optimizer_adam_beta2,
                "weight_decay": self.fake_optimizer_weight_decay,
                "adam_epsilon": self.fake_optimizer_adam_epsilon,
            },
        )
        self.lr_scheduler = self._build_lr_scheduler(self.optimizer)
        self.fake_lr_scheduler = self._build_lr_scheduler(
            self.fake_optimizer,
            num_warmup_steps=0,
            num_training_steps=max(1, self.max_train_iters * self.fake_update_ratio),
        )

        self.fake_model = model
        self.teacher_model = model
        if resume_ckpt_path is not None:
            self._load_rank_state_after_fsdp(resume_ckpt_path)

        self._log_cuda_memory("after_setup")
        self._install_residency_block_hooks()
        logger.info(
            "[h3-a100] setup physical_backbones=1 logical_roles=3 "
            "student_params={} fake_params={} fsdp={} critic_reorder={} matched_compute={}",
            sum(p.numel() for p in self.trainable_params),
            sum(p.numel() for p in self.fake_trainable_params),
            model.is_fsdp2_wrapped(),
            self.reorder_critic_rollouts,
            self.matched_compute_enabled,
        )
        logger.info(
            "[h3-a100] activation policy={} checkpoint_segment={} boundary_pin={} "
            "boundary_events={} legacy_threshold_offload={} fused_block_pointwise={} "
            "fused_block_pointwise_grad={} fused_rmsnorm_modulate={} "
            "fused_qk_rmsnorm_rotary={} fused_qk_rmsnorm_rotary_grad={} "
            "fused_swiglu={} fused_swiglu_grad={} "
            "fa3_nograd_num_splits={} fa3_grad_num_splits={} lora_scale1_elision={} "
            "lora_nograd_epilogue={} lora_grad_epilogue={}",
            self.activation_policy,
            self.activation_checkpoint_segment_size,
            self.boundary_offload_pin_memory,
            self.boundary_offload_events,
            self.activation_offload_enabled,
            self.fused_block_pointwise_enabled,
            self.fused_block_pointwise_grad_enabled,
            self.fused_rmsnorm_modulate_enabled,
            self.fused_qk_rmsnorm_rotary_enabled,
            self.fused_qk_rmsnorm_rotary_grad_enabled,
            self.fused_swiglu_enabled,
            self.fused_swiglu_grad_enabled,
            self.fa3_nograd_num_splits,
            self.fa3_grad_num_splits,
            self.lora_scale1_elision_enabled,
            self.lora_nograd_epilogue_enabled,
            self.lora_grad_epilogue_enabled,
        )

    def _install_residency_block_hooks(self) -> None:
        """Install opt-in block markers for a bounded OOM attribution run."""
        if os.environ.get("H3_MEMORY_ATTRIBUTION_BLOCKS", "0").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        handles = []
        blocks = list(getattr(self.shared_model.denoiser_module(), "transformer_blocks", ()))
        for index, block in enumerate(blocks):
            def pre_hook(_module, _inputs, block_index=index):
                self._log_residency(f"block_{block_index}_pre")

            def post_hook(_module, _inputs, _output, block_index=index):
                self._log_residency(f"block_{block_index}_post")

            handles.append(block.register_forward_pre_hook(pre_hook))
            handles.append(block.register_forward_hook(post_hook, always_call=True))
        self._residency_block_hooks = handles
        logger.info("[h3-a100][residency] installed block hooks count={}", len(blocks))

    def _activation_offload_scope(self, logical_component: str):
        return maybe_saved_tensor_offload(
            self.shared_model.denoiser_module(),
            enabled=self.activation_offload_enabled,
            logical_component=logical_component,
            min_offload_bytes=self.activation_offload_min_bytes,
            pin_memory=self.activation_offload_pin_memory,
        )

    def _boundary_offload_receipt(self):
        if self.boundary_offload_registration is None:
            return None
        return self.boundary_offload_registration.receipt()

    def _validate_reorder_contract(self) -> None:
        if not self.reorder_critic_rollouts:
            return
        stochastic_dropout = [
            (name, module.p)
            for name, module in self.shared_model.denoiser_module().named_modules()
            if isinstance(module, torch.nn.Dropout) and float(module.p) != 0.0
        ]
        if stochastic_dropout:
            preview = ", ".join(f"{name}:p={prob}" for name, prob in stochastic_dropout[:6])
            raise RuntimeError(
                "Critic rollout reordering requires deterministic H3 forwards; "
                f"found nonzero dropout modules: {preview}"
            )

    def _validate_shared_parameter_contract(self) -> None:
        student_ids = {id(parameter) for parameter in self.trainable_params}
        fake_ids = {id(parameter) for parameter in self.fake_trainable_params}
        overlap = student_ids & fake_ids
        if overlap:
            raise RuntimeError(f"Student/Fake optimizers share {len(overlap)} parameter objects")
        non_lora = [
            name
            for name, parameter in self.shared_model.denoiser_module().named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ]
        if non_lora:
            raise RuntimeError(
                f"Shared H3 base has trainable non-LoRA parameters: {', '.join(non_lora[:6])}"
            )
        if not all(parameter.requires_grad for parameter in self.trainable_params):
            raise RuntimeError("Student adapter contains frozen parameters after FSDP wrapping")
        if not all(parameter.requires_grad for parameter in self.fake_trainable_params):
            raise RuntimeError("Fake adapter contains frozen parameters after FSDP wrapping")
