"""Fail-closed runtime checks for the Grid-1000 experiment branch."""

from __future__ import annotations

from typing import Any

from loguru import logger


def install_grid_contract_patch() -> None:
    from .trainer import MiniMaxH3A100DmdTrainer

    if getattr(MiniMaxH3A100DmdTrainer, "_grid_contract_patch_installed", False):
        return

    original_begin = MiniMaxH3A100DmdTrainer._begin_boundary_offload_cycle
    original_validate = MiniMaxH3A100DmdTrainer._validate_boundary_offload_cycle

    def begin_cycle(self) -> None:
        original_begin(self)
        registration = getattr(self, "grid_replay_registration", None)
        self._grid_replay_cycle_start = None if registration is None else dict(registration.stats)

    def validate_cycle(self, current_iter: int) -> None:
        original_validate(self, current_iter)
        registration = getattr(self, "grid_replay_registration", None)
        if registration is None:
            raise RuntimeError("Grid-1000 checkpoint replay registration is missing")
        start = self._grid_replay_cycle_start
        if start is None:
            raise RuntimeError("Grid-1000 replay cycle baseline is missing")

        def delta(key: str) -> int:
            return int(registration.stats[key]) - int(start.get(key, 0))

        expected = {
            "checkpoint_wrap_count": 300,
            "scoped_execution_count": 600,
            "cache_hit_count": 600,
            "cache_miss_count": 0,
            "missing_key_count": 0,
        }
        observed = {key: delta(key) for key in expected}
        errors = [
            f"{key}={observed[key]} expected={value}"
            for key, value in expected.items()
            if observed[key] != value
        ]
        remaining_adaln = [
            name
            for name, _ in self.shared_model.denoiser_module().named_parameters()
            if ".adaln_proj." in name
        ]
        if remaining_adaln:
            errors.append(f"registered_adaln_params={len(remaining_adaln)} expected=0")
        controller = self.shared_model.adaln_cache()
        if int(getattr(controller, "dropped_parameter_numel", 0)) < 12_000_000_000:
            errors.append(
                "dropped_adaln_parameter_numel="
                f"{getattr(controller, 'dropped_parameter_numel', 0)} expected_about_13B"
            )
        if errors:
            raise RuntimeError(
                "Grid-1000 runtime contract failed at outer iteration "
                f"{current_iter}: {'; '.join(errors)}"
            )
        logger.info(
            "[h3-a100][grid1000-contract] iter={} replay={} dropped_adaln_params={} cache_mib={:.2f}",
            current_iter,
            observed,
            controller.dropped_parameter_numel,
            controller.stats().bytes / 1024**2,
        )

    MiniMaxH3A100DmdTrainer._begin_boundary_offload_cycle = begin_cycle
    MiniMaxH3A100DmdTrainer._validate_boundary_offload_cycle = validate_cycle
    MiniMaxH3A100DmdTrainer._grid_contract_patch_installed = True


install_grid_contract_patch()
