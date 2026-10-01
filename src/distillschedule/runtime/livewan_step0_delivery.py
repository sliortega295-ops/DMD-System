"""C39 default-off step0 wave1 delivery contract (no training math changes)."""
from __future__ import annotations

from typing import Any, Mapping

VARIANT = "c39-continuous-b16-step0-delivery"
CONTRACT = {
    "enabled": True,
    "name": "step0_wave1_pinned_early_delivery_v1",
    "scope": "continuous_b16_split_dmd_step0_wave1_only",
    "latent_transport": "existing_pinned_cpu_gloo",
    "teacher0_return": "async_cuda_nccl_retained_until_commit",
    "enqueue_position": "after_rollout1_before_fake0",
    "global_batch": 16,
    "payload_dtype": "float32",
    "payload_shape": [1, 16, 21, 46, 80],
    "depth1_lookahead_unchanged": True,
    "math_rng_and_calls_preserved": True,
}


def enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    metadata = plan.get("metadata", {})
    contract = metadata.get("step0_wave1_early_delivery")
    if contract is None:
        return False
    if dict(contract) != CONTRACT:
        raise RuntimeError("C39 step0 delivery contract changed")
    if (mode != "pipeline"
        or not metadata.get("continuous_pipeline", {}).get("enabled")
        or not metadata.get("generator_runahead_transport", {}).get("enabled")
        or metadata.get("split_dmd_shadow", {}).get("enabled") is not True
        or metadata.get("full_context_lens_elision") is not True):
        raise RuntimeError("C39 requires continuous B16 split-DMD/local-context transport")
    return True


def runtime_receipt_errors(receipt: Mapping[str, Any], *, rank: int, cycles: int) -> list[str]:
    """Require actual per-cycle early sends/receives, not configuration alone."""
    expected = {
        "enabled": True,
        "name": CONTRACT["name"],
        "role": "generator" if rank < 8 else "score",
        "early_latent_sends": cycles if rank < 8 else 0,
        "early_latent_receives": 0 if rank < 8 else cycles,
        "async_teacher0_sends": 0 if rank < 8 else cycles,
        "teacher0_send_waits": 0 if rank < 8 else cycles,
    }
    if dict(receipt) != expected:
        return [f"rank{rank}: C39 step0 actual-operation receipt mismatch"]
    return []


def transport_receipt_errors(receipt: Mapping[str, Any], *, rank: int, cycles: int) -> list[str]:
    from .livewan_host_runahead import runahead_tag

    keys = {(c * 5 + s, w) for c in range(cycles) for s in (1, 2, 3, 4) for w in (0, 1)}
    keys |= {(c * 5, 0) for c in range(1, cycles)}
    keys |= {(c * 5, 1) for c in range(cycles)}
    count = len(keys)
    is_g = rank < 8
    expected = {
        "execution_status": "PASS",
        "scope": "continuous_critic_steps_and_next_wave0_plus_step0_wave1",
        "step0_transport": "wave0_cuda_nccl_wave1_pinned_gloo_teacher_cuda_nccl",
        "step0_wave1_early_delivery": True,
        "cycles": cycles,
        "buffer_count": count,
        "pinned_buffer_count": count,
        "send_event_count": count if is_g else 0,
        "receive_event_count": 0 if is_g else count,
        "d2h_bytes": count * 4_945_920 if is_g else 0,
        "h2d_bytes": 0 if is_g else count * 4_945_920,
        "tags": [runahead_tag(s, w, allow_step0_wave1=True) for s, w in sorted(keys)],
    }
    return ([f"rank{rank}: C39 host transport inventory mismatch"]
            if any(receipt.get(k) != v for k, v in expected.items()) else [])
