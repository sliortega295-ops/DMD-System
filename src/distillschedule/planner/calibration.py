"""Bounded initial calibration matrix for the profile-guided Full5 planner."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from .profile_db import ProfileDB
from .schema import PrimitiveKey


def _key(
    workload: str,
    operation: str,
    role: str,
    backend: str,
    mesh: int,
    *,
    stage: int | None = None,
    grad: str = "no_grad",
    residency: str = "resident",
    activation: str = "none",
    offload_blocks: Iterable[int] = (),
    local_batch: int = 1,
) -> PrimitiveKey:
    return PrimitiveKey(
        workload=workload,
        operation=operation,
        role=role,
        backend=backend,
        mesh_size=mesh,
        local_batch=local_batch,
        stage=stage,
        grad_mode=grad,
        residency=residency,
        activation_policy=activation,
        offload_blocks=tuple(offload_blocks),
    )


def initial_calibration_matrix() -> tuple[PrimitiveKey, ...]:
    """Return a deliberately small matrix that spans the legal plan families.

    It is not the Cartesian product of every knob. The matrix contains only
    measurements needed to identify the dominant cost terms: pyramid compute,
    Score/Critic functions, sharding communication, block streaming, optimizer
    commits, EMA, and activation policy. Search may request additional points
    after this bootstrap via :func:`profile_requests`.
    """

    keys: list[PrimitiveKey] = []
    for workload in ("no-gan", "gan"):
        for mesh, backend, residency, activation in (
            (16, "zero1", "cpu_block_stream", "saved_tensor_cpu"),
            (8, "zero3", "sharded_resident", "checkpoint"),
            (16, "fsdp2", "sharded_resident", "checkpoint"),
            (4, "replicated", "resident", "none"),
            (12, "zero1", "resident", "saved_tensor_cpu"),
        ):
            for role in ("generator", "score"):
                for stage in (0, 1, 2):
                    keys.append(
                        _key(
                            workload,
                            "dit_forward",
                            role,
                            backend,
                            mesh,
                            stage=stage,
                            grad="no_grad",
                            residency=residency,
                            activation=activation,
                        )
                    )
            for stage in (0, 1, 2):
                keys.append(
                    _key(
                        workload,
                        "dit_selected_forward_backward",
                        "generator",
                        backend,
                        mesh,
                        stage=stage,
                        grad="selected_grad",
                        residency=residency,
                        activation=activation,
                    )
                )

        for backend, mesh, residency in (
            ("zero1", 16, "cpu_block_stream"),
            ("zero3", 8, "sharded_resident"),
            ("fsdp2", 16, "sharded_resident"),
            ("zero1", 12, "resident"),
        ):
            for operation in (
                "score_fake_cond",
                "score_real_cond",
                "score_real_uncond",
                "critic_denoise_forward_backward",
            ):
                keys.append(
                    _key(
                        workload,
                        operation,
                        "score",
                        backend,
                        mesh,
                        grad=(
                            "train"
                            if operation == "critic_denoise_forward_backward"
                            else "no_grad"
                        ),
                        residency=residency,
                        activation=(
                            "saved_tensor_cpu"
                            if operation == "critic_denoise_forward_backward"
                            else "none"
                        ),
                    )
                )
            if workload == "gan":
                for operation, grad in (
                    ("gan_g_score_boundary", "latent_grad"),
                    ("gan_d_r1_forward_backward", "train"),
                ):
                    keys.append(
                        _key(
                            workload,
                            operation,
                            "score",
                            backend,
                            mesh,
                            grad=grad,
                            residency=residency,
                            activation=(
                                "saved_tensor_cpu" if grad == "train" else "none"
                            ),
                        )
                    )

        for backend, mesh in (
            ("zero1", 16),
            ("zero3", 8),
            ("fsdp2", 16),
            ("zero1", 12),
        ):
            keys.extend(
                (
                    _key(
                        workload,
                        "generator_optimizer_step",
                        "generator",
                        backend,
                        mesh,
                        grad="optimizer",
                        residency="sharded_resident",
                    ),
                    _key(
                        workload,
                        "critic_optimizer_step",
                        "score",
                        backend,
                        mesh,
                        grad="optimizer",
                        residency="sharded_resident",
                    ),
                    _key(
                        workload,
                        "ema_step",
                        "generator",
                        backend,
                        mesh,
                        grad="ema",
                        residency="sharded_resident",
                    ),
                )
            )

        for count in (0, 1, 4, 10, 20, 40):
            indices = tuple(range(max(0, 40 - count), 40)) if count else ()
            keys.append(
                _key(
                    workload,
                    "dit_forward",
                    "score",
                    "zero1",
                    16,
                    stage=2,
                    grad="no_grad",
                    residency="cpu_block_stream",
                    activation="none",
                    offload_blocks=indices,
                )
            )

    deduped: dict[str, PrimitiveKey] = {key.id: key for key in keys}
    return tuple(deduped[key] for key in sorted(deduped))


def calibration_coverage(db: ProfileDB) -> Mapping[str, Any]:
    matrix = initial_calibration_matrix()
    coverage = db.coverage(matrix, allow_interpolation=False)
    return {
        "required_primitive_count": coverage.required,
        "exact_primitive_count": coverage.exact,
        "missing_primitive_count": len(coverage.missing),
        "missing": [
            {**key.__dict__, "offload_blocks": list(key.offload_blocks), "id": key.id}
            for key in coverage.missing
        ],
        "ready_for_unseen_plan_search": coverage.claimable,
        "non_primitive_required_records": [
            "model inventory: parameter/buffer/block/trainable/optimizer/EMA bytes",
            "hardware calibration: pinned/pageable H2D, D2H, intra/inter-node collective bandwidth and latency",
            "activation saved-tensor census and liveness by operation/stage",
            "transition decomposition for each legal layout edge",
            "node cgroup stable and construction-transient memory receipts",
            "at least one full-plan validation anchor per backend family",
        ],
    }


__all__ = ["calibration_coverage", "initial_calibration_matrix"]
