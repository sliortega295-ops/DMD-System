"""Same-topology checkpointing for one shared model and two optimizers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from loguru import logger

from lightx2v_train.runtime.checkpoint import find_latest_checkpoint, prune_checkpoints
from lightx2v_train.runtime.distributed import barrier, get_rank, get_world_size, is_main_process

from .model import FAKE_ADAPTER, STUDENT_ADAPTER


class H3A100CheckpointMixin:
    """LoRA-only model state plus rank-local optimizer/RNG state."""

    def _resolve_resume_a100(self):
        if not self.auto_resume:
            return None, 0
        return find_latest_checkpoint(self.output_train_dir)

    def _load_role_weights_before_fsdp(self, checkpoint_dir):
        student_dir = os.path.join(checkpoint_dir, "student")
        fake_dir = os.path.join(checkpoint_dir, "fake")
        if not os.path.isdir(student_dir) or not os.path.isdir(fake_dir):
            raise RuntimeError(f"Checkpoint is missing student/fake LoRA dirs: {checkpoint_dir}")
        self.shared_model.load_role_lora(student_dir, STUDENT_ADAPTER)
        self.shared_model.load_role_lora(fake_dir, FAKE_ADAPTER)

    def _rank_state_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, f"rank-{get_rank():04d}.pt")

    def _load_rank_state_after_fsdp(self, checkpoint_dir):
        state_path = self._rank_state_path(checkpoint_dir)
        if not os.path.isfile(state_path):
            raise RuntimeError(f"Rank-local checkpoint state not found: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if int(state["world_size"]) != get_world_size():
            raise RuntimeError(
                f"Checkpoint world_size={state['world_size']} does not match current {get_world_size()}"
            )
        self.optimizer.load_state_dict(state["student_optimizer"])
        self.fake_optimizer.load_state_dict(state["fake_optimizer"])
        self.lr_scheduler.load_state_dict(state["student_scheduler"])
        self.fake_lr_scheduler.load_state_dict(state["fake_scheduler"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        logger.info("[h3-a100] resumed rank-local state from {}", state_path)

    def save_checkpoint_a100(self, iteration):
        if is_main_process():
            prune_checkpoints(self.output_train_dir, self.save_total_limit)
        checkpoint_dir = os.path.join(self.output_train_dir, f"checkpoint-{iteration:09d}")
        student_dir = os.path.join(checkpoint_dir, "student")
        fake_dir = os.path.join(checkpoint_dir, "fake")
        if is_main_process():
            os.makedirs(student_dir, exist_ok=True)
            os.makedirs(fake_dir, exist_ok=True)
        barrier()

        self.shared_model.save_all_role_loras(student_dir, fake_dir)
        barrier()

        rank_state = {
            "iteration": int(iteration),
            "world_size": get_world_size(),
            "student_optimizer": self.optimizer.state_dict(),
            "fake_optimizer": self.fake_optimizer.state_dict(),
            "student_scheduler": self.lr_scheduler.state_dict(),
            "fake_scheduler": self.fake_lr_scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }
        torch.save(rank_state, self._rank_state_path(checkpoint_dir))
        barrier()
        if is_main_process():
            torch.save(
                {
                    "iteration": int(iteration),
                    "world_size": get_world_size(),
                    "format": "h3-a100-shared-backbone-v1",
                },
                os.path.join(checkpoint_dir, "trainer_state.pt"),
            )
            with open(os.path.join(checkpoint_dir, "_SUCCESS"), "w", encoding="utf-8") as handle:
                handle.write("ok\n")
        barrier()
        logger.info("[h3-a100] checkpoint saved iteration={} path={}", iteration, checkpoint_dir)
