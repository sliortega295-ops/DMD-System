"""Small, testable DeepSpeed configuration helpers for Runtime backends."""
from __future__ import annotations

from typing import Any, Literal


ParamOffloadDevice = Literal["none", "cpu"]
OptimizerOffloadDevice = Literal["none", "cpu"]


def build_client_optimizer_offload_overrides(
    optimizer_offload_device: OptimizerOffloadDevice,
) -> dict[str, Any]:
    """Keep a caller-provided optimizer when ZeRO optimizer offload is enabled.

    DeepSpeed otherwise rejects a client ``torch.optim`` instance and asks the
    caller to switch to DeepSpeedCPUAdam.  The real-model probes deliberately
    keep the same AdamW implementation across the offload ablation, so the
    required top-level opt-out is explicit and testable rather than hidden in
    the launcher.
    """
    if optimizer_offload_device not in ("none", "cpu"):
        raise ValueError(f"unsupported optimizer offload device: {optimizer_offload_device}")
    if optimizer_offload_device == "cpu":
        return {"zero_force_ds_cpu_optimizer": False}
    return {}


def build_zero_optimization_config(
    *,
    stage: int,
    reduce_bucket_mib: int,
    allgather_bucket_mib: int,
    overlap_comm: bool,
    contiguous_gradients: bool,
    stage3_param_persistence_threshold: int,
    param_offload_device: ParamOffloadDevice = "none",
    param_offload_pin_memory: bool = False,
    optimizer_offload_device: OptimizerOffloadDevice = "none",
    optimizer_offload_pin_memory: bool = False,
) -> dict[str, Any]:
    """Build the probe's ZeRO policy with explicit admission/offload state."""
    if stage not in (1, 2, 3):
        raise ValueError(f"unsupported ZeRO stage: {stage}")
    if reduce_bucket_mib <= 0 or allgather_bucket_mib <= 0:
        raise ValueError("ZeRO communication buckets must be positive")
    if stage3_param_persistence_threshold < 0:
        raise ValueError("stage3_param_persistence_threshold must be non-negative")
    if param_offload_device not in ("none", "cpu"):
        raise ValueError(f"unsupported parameter offload device: {param_offload_device}")
    if param_offload_device != "none" and stage != 3:
        raise ValueError("parameter offload requires ZeRO-3")
    if param_offload_pin_memory and param_offload_device != "cpu":
        raise ValueError("pinned parameter offload requires the CPU device")
    if optimizer_offload_device not in ("none", "cpu"):
        raise ValueError(f"unsupported optimizer offload device: {optimizer_offload_device}")
    if optimizer_offload_pin_memory and optimizer_offload_device != "cpu":
        raise ValueError("pinned optimizer offload requires the CPU device")

    # DeepSpeed bucket sizes are element counts. Probe communication is FP32.
    bucket_elements = reduce_bucket_mib * 2**20 // 4
    allgather_elements = allgather_bucket_mib * 2**20 // 4
    zero: dict[str, Any] = {
        "stage": stage,
        "overlap_comm": overlap_comm,
        "contiguous_gradients": contiguous_gradients,
        "reduce_bucket_size": bucket_elements,
        "allgather_bucket_size": allgather_elements,
    }
    if stage == 3:
        zero.update(
            {
                "stage3_prefetch_bucket_size": allgather_elements,
                "stage3_param_persistence_threshold": stage3_param_persistence_threshold,
                "stage3_max_live_parameters": 256 * 2**20,
                "stage3_max_reuse_distance": 256 * 2**20,
                "stage3_gather_16bit_weights_on_model_save": False,
            }
        )
        if param_offload_device == "cpu":
            zero["offload_param"] = {
                "device": "cpu",
                "pin_memory": param_offload_pin_memory,
            }
    if optimizer_offload_device == "cpu":
        zero["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": optimizer_offload_pin_memory,
        }
    return zero
