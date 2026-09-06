"""Fail-closed boundary for an isolated Generator update experiment.

The gate is deliberately small and reusable.  It does not implement training;
it validates the resolved residency/activation contract, opens the output
boundary, and writes one immutable rank result.  The official training loop is
the only owner of the actual Generator -> Score -> Generator execution graph.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _required_nonnegative_int(name: str) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise ValueError(f"{name} is required")
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class GeneratorUpdateGateOptions:
    enabled: bool = False
    logical_step: int | None = None
    result_dir: Path | None = None
    expected_world_size: int | None = None
    expected_global_batch: int | None = None
    expected_activation_min_bytes: int | None = None
    expected_swap_wave_size: int | None = None
    expected_construction_wave_size: int | None = None
    nvtx_label: str = "distillschedule_generator_update_gate"

    @classmethod
    def from_env(cls) -> "GeneratorUpdateGateOptions":
        enabled = _enabled("DISTILLSCHEDULE_STOP_AFTER_GENERATOR_UPDATE")
        gate_variables = (
            "DISTILLSCHEDULE_STOP_AFTER_GENERATOR_UPDATE_STEP",
            "DISTILLSCHEDULE_GENERATOR_UPDATE_RESULT_DIR",
            "DISTILLSCHEDULE_GENERATOR_UPDATE_EXPECTED_WORLD_SIZE",
            "DISTILLSCHEDULE_GENERATOR_UPDATE_EXPECTED_GLOBAL_BATCH",
            "DISTILLSCHEDULE_GENERATOR_UPDATE_EXPECTED_ACTIVATION_MIN_BYTES",
            "DISTILLSCHEDULE_GENERATOR_UPDATE_EXPECTED_SWAP_WAVE_SIZE",
            "DISTILLSCHEDULE_GENERATOR_UPDATE_EXPECTED_CONSTRUCTION_WAVE_SIZE",
            "DISTILLSCHEDULE_GENERATOR_UPDATE_NVTX_LABEL",
        )
        if not enabled:
            unexpected = [name for name in gate_variables if os.environ.get(name, "").strip()]
            if unexpected:
                raise ValueError(
                    "Generator update gate variables require "
                    "DISTILLSCHEDULE_STOP_AFTER_GENERATOR_UPDATE=1: "
                    + ", ".join(unexpected)
                )
            return cls()

        result_raw = os.environ.get(
            "DISTILLSCHEDULE_GENERATOR_UPDATE_RESULT_DIR", ""
        ).strip()
        if not result_raw:
            raise ValueError("DISTILLSCHEDULE_GENERATOR_UPDATE_RESULT_DIR is required")
        label = os.environ.get(
            "DISTILLSCHEDULE_GENERATOR_UPDATE_NVTX_LABEL",
            "distillschedule_generator_update_gate",
        ).strip()
        if not label:
            raise ValueError("Generator update NVTX label must not be empty")
        return cls(
            enabled=True,
            logical_step=_required_nonnegative_int(
                "DISTILLSCHEDULE_STOP_AFTER_GENERATOR_UPDATE_STEP"
            ),
            result_dir=Path(result_raw),
            expected_world_size=_required_nonnegative_int(
                "DISTILLSCHEDULE_GENERATOR_UPDATE_EXPECTED_WORLD_SIZE"
            ),
            expected_global_batch=_required_nonnegative_int(
                "DISTILLSCHEDULE_GENERATOR_UPDATE_EXPECTED_GLOBAL_BATCH"
            ),
            expected_activation_min_bytes=_required_nonnegative_int(
                "DISTILLSCHEDULE_GENERATOR_UPDATE_EXPECTED_ACTIVATION_MIN_BYTES"
            ),
            expected_swap_wave_size=_required_nonnegative_int(
                "DISTILLSCHEDULE_GENERATOR_UPDATE_EXPECTED_SWAP_WAVE_SIZE"
            ),
            expected_construction_wave_size=_required_nonnegative_int(
                "DISTILLSCHEDULE_GENERATOR_UPDATE_EXPECTED_CONSTRUCTION_WAVE_SIZE"
            ),
            nvtx_label=label,
        )

    def validate_baseline(self, baseline_options: Any) -> None:
        if not self.enabled:
            return
        assert self.logical_step is not None
        assert self.result_dir is not None
        if not self.result_dir.is_absolute():
            raise ValueError("Generator update result directory must be absolute")
        if not baseline_options.enabled:
            raise ValueError("Generator update gate requires the official baseline")
        if not baseline_options.semantic_assertions:
            raise ValueError("Generator update gate requires semantic assertions")
        if baseline_options.logical_start_step != self.logical_step:
            raise ValueError(
                "Generator update gate step must equal the synthetic logical start step"
            )
        has_b3_streaming = baseline_options.frozen_block_streaming_window != 0
        if baseline_options.hybrid_residency:
            if not has_b3_streaming:
                raise ValueError(
                    "hybrid Generator update gate requires partial block streaming"
                )
            if set(baseline_options.immutable_model_backing_components) != {
                "generator",
                "score",
            }:
                raise ValueError(
                    "hybrid Generator update gate requires Generator and Score "
                    "phase-resident backing"
                )
            if baseline_options.frozen_block_streaming_components != ("generator",):
                raise ValueError(
                    "hybrid Generator update gate streams Generator blocks only"
                )
            if baseline_options.frozen_block_streaming_block_count <= 0:
                raise ValueError(
                    "hybrid Generator update gate requires a positive tail-block count"
                )
            if baseline_options.frozen_block_host_policy != "HOST_SHARED_IMMUTABLE":
                raise ValueError(
                    "hybrid Generator update gate requires HOST_SHARED_IMMUTABLE"
                )
            if baseline_options.frozen_block_streaming_window != 1:
                raise ValueError(
                    "hybrid Generator update gate requires streaming window 1"
                )
        elif has_b3_streaming:
            if baseline_options.immutable_model_backing_components:
                raise ValueError(
                    "B3 generator update gate forbids whole-model immutable backing"
                )
            if baseline_options.frozen_block_host_policy != "HOST_SHARED_IMMUTABLE":
                raise ValueError(
                    "B3 generator update gate requires HOST_SHARED_IMMUTABLE"
                )
            if baseline_options.frozen_block_streaming_window != 1:
                raise ValueError(
                    "B3 generator update gate requires streaming window 1"
                )
        elif set(baseline_options.immutable_model_backing_components) != {
            "generator",
            "score",
        }:
            raise ValueError(
                "Generator update gate requires shared immutable backing for Generator and Score"
            )
        if baseline_options.activation_offload_components != ("generator_selected",):
            raise ValueError(
                "Generator update gate requires Generator-only saved-activation offload"
            )
        if (
            baseline_options.activation_offload_min_bytes
            != self.expected_activation_min_bytes
        ):
            raise ValueError("Generator activation threshold drift")
        if baseline_options.swap_wave_size != self.expected_swap_wave_size:
            raise ValueError("Generator update swap-wave drift")
        if (
            baseline_options.construction_wave_size
            != self.expected_construction_wave_size
        ):
            raise ValueError("Generator update construction-wave drift")

    def validate_runtime(
        self,
        *,
        global_step: int,
        world_size: int,
        global_batch: int,
        gradient_checkpointing: bool,
        is_train_dmd: bool,
        dmd_is_low_vram_mode: bool,
        is_use_gan: bool,
    ) -> None:
        if not self.enabled:
            return
        expected = {
            "global_step": self.logical_step,
            "world_size": self.expected_world_size,
            "global_batch": self.expected_global_batch,
        }
        observed = {
            "global_step": int(global_step),
            "world_size": int(world_size),
            "global_batch": int(global_batch),
        }
        if observed != expected:
            raise AssertionError(
                f"Generator update runtime contract drift: expected={expected}, observed={observed}"
            )
        if not gradient_checkpointing:
            raise AssertionError(
                "Generator shared-backing release/reload requires gradient checkpointing"
            )
        if not is_train_dmd or not dmd_is_low_vram_mode:
            raise AssertionError(
                "Generator update gate requires DMD whole-model low-VRAM switching"
            )
        if is_use_gan:
            raise AssertionError("Generator update gate is preregistered as no-GAN")

    def write_rank_result(self, rank: int, payload: dict[str, Any]) -> Path:
        if not self.enabled or self.result_dir is None:
            raise RuntimeError("Generator update gate is not enabled")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        destination = self.result_dir / f"rank_{int(rank):03d}.json"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite rank result: {destination}")
        temporary = destination.with_suffix(
            destination.suffix + f".{os.getpid()}.tmp"
        )
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
        return destination
