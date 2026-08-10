"""Exact semantic model-call inventory for GAN baseline/pipeline fairness.

Random-tensor parity is necessary but not sufficient: a scheduler could still
silently omit a fixed Score/DiT call.  This contract binds the complete logical
model work per sample for the frozen Helios GAN configuration at steps1000..1004.

Pinned upstream semantics used here:
* fake_guidance_scale=0 -> one Fake-Score conditional forward, no fake-uncond;
* real_guidance_scale=3 -> one Real-Score conditional + one unconditional;
* low-VRAM GAN-G -> one GAN-mode Score forward;
* Critic -> one denoising Score forward;
* low-VRAM GAN-D/R1 -> one GAN-mode Score forward whose batch packs
  fake + real + R1 (three items); R2 is disabled by r2_weight=0.
Variable Generator rollout DiT calls remain trace-controlled and are cross-
checked against the stochastic fairness receipt.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .gan_fairness import (
    FULL5_STEPS,
    LOGICAL_SAMPLE_SLOTS,
    normalize_gan_fairness_receipt,
)

SCHEMA_VERSION = "distillschedule.gan_compute_receipt.v1"

GENERATOR_FIXED_CALLS = {
    "dmd_fake_cond_score_forwards": 1,
    "dmd_fake_uncond_score_forwards": 0,
    "dmd_real_cond_score_forwards": 1,
    "dmd_real_uncond_score_forwards": 1,
    "gan_g_score_forwards": 1,
    "gan_g_score_batch_items": 1,
}

CRITIC_FIXED_CALLS = {
    "critic_denoise_score_forwards": 1,
    "critic_denoise_score_batch_items": 1,
    "gan_d_score_forwards": 1,
    "gan_d_score_batch_items": 3,
}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{path} must be non-negative")
    return result


def _normalize_calls(
    raw: Any,
    *,
    fixed: Mapping[str, int],
    path: str,
) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must be a mapping")
    expected = {"rollout_generator_dit_forwards", *fixed.keys()}
    observed = set(str(key) for key in raw)
    if observed != expected:
        raise ValueError(
            f"{path} call inventory drift: "
            f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    result = {
        key: _nonnegative_int(raw[key], f"{path}.{key}")
        for key in sorted(expected)
    }
    for key, value in fixed.items():
        if result[key] != value:
            raise ValueError(
                f"{path}.{key}={result[key]}, expected official value {value}"
            )
    return result


def normalize_gan_compute_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != SCHEMA_VERSION or value.get("status") != "PASS":
        raise ValueError("GAN compute receipt schema/status mismatch")
    family = value.get("family")
    if family not in {"baseline", "pipeline"}:
        raise ValueError("GAN compute receipt family must be baseline or pipeline")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise ValueError("GAN compute receipt entries must be a list")
    expected_pairs = {
        (step, slot) for step in FULL5_STEPS for slot in LOGICAL_SAMPLE_SLOTS
    }
    observed: set[tuple[int, int]] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(entries):
        path = f"entries[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must be a mapping")
        step = _nonnegative_int(raw.get("logical_step"), f"{path}.logical_step")
        slot = _nonnegative_int(
            raw.get("logical_sample_slot"), f"{path}.logical_sample_slot"
        )
        pair = (step, slot)
        if pair not in expected_pairs or pair in observed:
            raise ValueError(f"{path} invalid/duplicate step-slot {pair}")
        observed.add(pair)
        generator_raw = raw.get("generator")
        if step == 1000:
            generator = _normalize_calls(
                generator_raw,
                fixed=GENERATOR_FIXED_CALLS,
                path=f"{path}.generator",
            )
        else:
            if generator_raw not in (None, {}):
                raise ValueError(
                    f"{path}.generator must be absent on Critic-only step {step}"
                )
            generator = None
        critic = _normalize_calls(
            raw.get("critic"),
            fixed=CRITIC_FIXED_CALLS,
            path=f"{path}.critic",
        )
        normalized.append(
            {
                "logical_step": step,
                "logical_sample_slot": slot,
                "generator": generator,
                "critic": critic,
            }
        )
    if observed != expected_pairs:
        raise ValueError(
            f"GAN compute receipt misses logical work: {sorted(expected_pairs-observed)}"
        )
    normalized.sort(key=lambda item: (item["logical_step"], item["logical_sample_slot"]))
    canonical = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "logical_steps": list(FULL5_STEPS),
        "logical_sample_slots": list(LOGICAL_SAMPLE_SLOTS),
        "entries": normalized,
    }
    canonical["compute_contract_digest"] = _digest(canonical)
    return canonical


def validate_compute_against_fairness(
    compute: Mapping[str, Any], fairness: Mapping[str, Any]
) -> dict[str, Any]:
    c = normalize_gan_compute_receipt(compute)
    f = normalize_gan_fairness_receipt(fairness)
    c_entries = {
        (item["logical_step"], item["logical_sample_slot"]): item
        for item in c["entries"]
    }
    for item in f["entries"]:
        pair = (item["logical_step"], item["logical_sample_slot"])
        comp = c_entries[pair]
        if item["generator"] is not None:
            observed = comp["generator"]["rollout_generator_dit_forwards"]
            expected = item["generator"]["dit"]["total_dit_forwards"]
            if observed != expected:
                raise ValueError(
                    f"Generator rollout DiT count mismatch at {pair}: {observed}!={expected}"
                )
        observed = comp["critic"]["rollout_generator_dit_forwards"]
        expected = item["critic"]["dit"]["total_dit_forwards"]
        if observed != expected:
            raise ValueError(
                f"Critic rollout DiT count mismatch at {pair}: {observed}!={expected}"
            )
    return {
        "status": "PASS",
        "compute_contract_digest": c["compute_contract_digest"],
        "fairness_contract_digest": f["fairness_contract_digest"],
    }


def compare_gan_compute_receipts(
    baseline: Mapping[str, Any], pipeline: Mapping[str, Any]
) -> dict[str, Any]:
    a = normalize_gan_compute_receipt(baseline)
    b = normalize_gan_compute_receipt(pipeline)
    if a != b:
        raise ValueError("GAN baseline/pipeline model-call inventory mismatch")
    return {
        "schema_version": "distillschedule.gan_compute_pair.v1",
        "status": "PASS",
        "compute_contract_digest": a["compute_contract_digest"],
        "fixed_generator_score_forwards_per_sample": 4,
        "fixed_critic_score_forwards_per_sample": 2,
        "gan_d_batched_items_per_sample": 3,
    }


__all__ = [
    "CRITIC_FIXED_CALLS",
    "GENERATOR_FIXED_CALLS",
    "SCHEMA_VERSION",
    "compare_gan_compute_receipts",
    "normalize_gan_compute_receipt",
    "validate_compute_against_fairness",
]
