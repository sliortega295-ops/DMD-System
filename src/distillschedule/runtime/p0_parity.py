"""Strict numerical parity gates for P0 physical-execution rewrites.

The P0 optimizations are allowed to change execution order and physical batching,
but not training semantics. A mean-relative metric alone can hide sparse large
errors, so correctness requires several complementary checks.
"""
from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class TensorParityThresholds:
    mean_relative: float = 5e-3
    normalized_l2: float = 5e-3
    min_cosine: float = 0.9999
    max_abs_atol: float = 5e-2
    max_abs_rtol: float = 1e-2

    def __post_init__(self) -> None:
        if min(
            self.mean_relative,
            self.normalized_l2,
            self.max_abs_atol,
            self.max_abs_rtol,
        ) < 0:
            raise ValueError("parity tolerances must be non-negative")
        if not -1.0 <= self.min_cosine <= 1.0:
            raise ValueError("min_cosine must be in [-1, 1]")


DEFAULT_P0_PARITY = TensorParityThresholds()


def tensor_parity_metrics(
    reference: Any,
    candidate: Any,
    *,
    thresholds: TensorParityThresholds = DEFAULT_P0_PARITY,
) -> dict[str, Any]:
    """Compare two tensors with scale-aware, non-additive correctness metrics."""

    import torch

    if not torch.is_tensor(reference) or not torch.is_tensor(candidate):
        raise TypeError("P0 parity requires tensors")
    same_shape = tuple(reference.shape) == tuple(candidate.shape)
    same_dtype = reference.dtype == candidate.dtype
    if not same_shape:
        return {
            "status": "FAIL",
            "finite": False,
            "same_shape": False,
            "same_dtype": same_dtype,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "reference_dtype": str(reference.dtype),
            "candidate_dtype": str(candidate.dtype),
            "reason": "shape_mismatch",
        }

    ref = reference.detach().float()
    cand = candidate.detach().float()
    finite = bool(torch.isfinite(ref).all() and torch.isfinite(cand).all())
    if finite:
        difference = (ref - cand).abs()
        mean_abs = difference.mean()
        ref_mean_abs = ref.abs().mean().clamp_min(1e-6)
        mean_relative = mean_abs / ref_mean_abs

        raw_ref_l2 = torch.linalg.vector_norm(ref.flatten())
        raw_candidate_l2 = torch.linalg.vector_norm(cand.flatten())
        normalized_l2 = (
            torch.linalg.vector_norm((ref - cand).flatten())
            / raw_ref_l2.clamp_min(1e-12)
        )
        if float(raw_ref_l2.cpu()) <= 1e-12 and float(raw_candidate_l2.cpu()) <= 1e-12:
            # Cosine is mathematically undefined for two zero vectors, but for
            # a parity gate they are exactly identical and should not fail.
            cosine = torch.tensor(1.0, device=ref.device)
            cosine_policy = "both_zero_defined_as_one"
        elif float(raw_ref_l2.cpu()) <= 1e-12 or float(raw_candidate_l2.cpu()) <= 1e-12:
            cosine = torch.tensor(0.0, device=ref.device)
            cosine_policy = "single_zero_defined_as_zero"
        else:
            cosine = torch.nn.functional.cosine_similarity(
                ref.flatten().unsqueeze(0),
                cand.flatten().unsqueeze(0),
                dim=1,
                eps=1e-12,
            )[0]
            cosine_policy = "standard"

        max_abs = difference.max()
        reference_peak = ref.abs().max()
        max_abs_limit = thresholds.max_abs_atol + thresholds.max_abs_rtol * reference_peak
        values = {
            "max_abs": float(max_abs.cpu()),
            "max_abs_limit": float(max_abs_limit.cpu()),
            "mean_abs": float(mean_abs.cpu()),
            "mean_relative": float(mean_relative.cpu()),
            "normalized_l2": float(normalized_l2.cpu()),
            "cosine": float(cosine.cpu()),
            "cosine_policy": cosine_policy,
            "reference_peak_abs": float(reference_peak.cpu()),
        }
        passed = (
            values["mean_relative"] <= thresholds.mean_relative
            and values["normalized_l2"] <= thresholds.normalized_l2
            and values["cosine"] >= thresholds.min_cosine
            and values["max_abs"] <= values["max_abs_limit"]
        )
    else:
        values = {
            "max_abs": float("inf"),
            "max_abs_limit": float("nan"),
            "mean_abs": float("inf"),
            "mean_relative": float("inf"),
            "normalized_l2": float("inf"),
            "cosine": float("nan"),
            "cosine_policy": "non_finite",
            "reference_peak_abs": float("nan"),
        }
        passed = False

    return {
        "status": "PASS" if passed and same_dtype else "FAIL",
        "finite": finite,
        "same_shape": same_shape,
        "same_dtype": same_dtype,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        **values,
        "thresholds": dataclasses.asdict(thresholds),
    }


def require_tensor_parity(
    reference: Any,
    candidate: Any,
    *,
    label: str,
    thresholds: TensorParityThresholds = DEFAULT_P0_PARITY,
) -> dict[str, Any]:
    metrics = tensor_parity_metrics(reference, candidate, thresholds=thresholds)
    if metrics.get("status") != "PASS":
        raise AssertionError(f"P0 parity failed for {label}: {metrics}")
    return metrics


def require_recorded_parity(metrics: dict[str, Any], *, label: str) -> None:
    if metrics.get("status") != "PASS":
        raise AssertionError(f"P0 recorded parity failed for {label}: {metrics}")


__all__ = [
    "DEFAULT_P0_PARITY",
    "TensorParityThresholds",
    "require_recorded_parity",
    "require_tensor_parity",
    "tensor_parity_metrics",
]
