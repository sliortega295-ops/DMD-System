#!/usr/bin/env python3
"""Static Full5 worker with vLLM-Omni latent Generator rollouts.

The differentiable Generator update, Score calls, Critic commits, and EMA remain
in the hardened FSDP2 training worker. Only the five post-update no-grad
Critic-input rollouts are delegated to a dedicated vLLM-Omni adapter.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_HARDENED = ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py"
_SPEC = importlib.util.spec_from_file_location(
    "distillschedule_p0_vllm_hardened", _HARDENED
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load hardened P0 worker from {_HARDENED}")
hardened = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = hardened
_SPEC.loader.exec_module(hardened)
impl = hardened.impl

import distillschedule.runtime.fsdp2_p0 as p0_policy
from distillschedule.runtime.vllm_omni_rollout import (
    NATIVE_ROLLOUT_BACKEND,
    VLLM_OMNI_BACKEND,
)
from runtime.vllm_omni_helios_rollout import (
    VLLMOmniAdapterEnvironment,
    VLLMOmniRolloutClient,
    get_last_vllm_omni_receipt,
    precompute_vllm_omni_critic_rollouts,
)


# Enable the external backend only for this explicit entry point. The ordinary
# hardened worker remains fail-closed and native by default.
p0_policy.SUPPORTED_ROLLOUT_BACKENDS = (
    NATIVE_ROLLOUT_BACKEND,
    VLLM_OMNI_BACKEND,
)

_original_policy_id = p0_policy.StaticFSDP2P0Policy.id.fget
if _original_policy_id is None:
    raise RuntimeError("StaticFSDP2P0Policy.id lost its property getter")


def _policy_id(self: Any) -> str:
    native_id = str(_original_policy_id(self))
    if self.rollout_backend == NATIVE_ROLLOUT_BACKEND:
        return native_id
    return f"{native_id}-backend-{self.rollout_backend}"


p0_policy.StaticFSDP2P0Policy.id = property(_policy_id)

_original_parser = impl._parser


def _parser() -> Any:
    parser = _original_parser()
    for action in parser._actions:
        if action.dest == "rollout_backend":
            action.choices = p0_policy.SUPPORTED_ROLLOUT_BACKENDS
            action.default = VLLM_OMNI_BACKEND
            break
    else:
        raise RuntimeError("P0 parser lost --rollout-backend")
    return parser


impl._parser = _parser
_native_hardened_precompute = impl._precompute_critic_rollouts


def _variant_from_argv() -> str:
    try:
        index = sys.argv.index("--variant")
        return str(sys.argv[index + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("vLLM-Omni worker requires explicit --variant") from error


# A service may report a versioned adapter id while still loading stale bytes.
# Require an export checksum from the training state and the checksum observed
# after vLLM-Omni hot-load, then prove every training rank saw the same identity.
_original_client_sync = VLLMOmniRolloutClient.sync_generator


def _checksum_verified_sync(
    self: VLLMOmniRolloutClient,
    *,
    generator_engine: Any,
    variant: str,
    generator_version: int,
) -> dict[str, Any]:
    import torch.distributed as dist

    receipt = _original_client_sync(
        self,
        generator_engine=generator_engine,
        variant=variant,
        generator_version=generator_version,
    )
    source_checksum = str(receipt.get("source_trainable_sha256", ""))
    loaded_checksum = str(receipt.get("loaded_trainable_sha256", ""))
    if not source_checksum or not loaded_checksum:
        raise ValueError(
            "vLLM-Omni Generator sync must report source_trainable_sha256 "
            "and loaded_trainable_sha256"
        )
    if source_checksum != loaded_checksum:
        raise ValueError(
            "vLLM-Omni loaded stale/different Generator LoRA: "
            f"{source_checksum} != {loaded_checksum}"
        )
    identity = (
        int(receipt["generator_version"]),
        str(receipt["adapter_identity"]),
        source_checksum,
        loaded_checksum,
    )
    if dist.is_initialized():
        gathered: list[Any] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, identity)
        if any(item != identity for item in gathered):
            raise ValueError(
                f"rank-local vLLM-Omni LoRA checksums differ: {gathered}"
            )
        dist.barrier()
    receipt["checksum_match"] = True
    return receipt


VLLMOmniRolloutClient.sync_generator = _checksum_verified_sync


def _control_trace(trace: Any, rank: int) -> Any:
    inner = getattr(trace, "_inner", trace)
    controls = getattr(inner, "_control", None)
    if not isinstance(controls, dict) or int(rank) not in controls:
        raise TypeError("static vLLM-Omni worker cannot access control-only trace")
    return controls[int(rank)]


def _expected_rollout_schedule(trace: Any, *, rank: int, step: int) -> tuple[int, dict[int, int]]:
    operation = _control_trace(trace, rank).operation(
        f"step/{int(step)}/critic/rollout"
    )
    control = operation["control"]
    schedule = {
        int(item["stage"]): int(item["call_count"])
        for item in control["pyramid_stage_call_sequence"]
    }
    return int(control["call_count"]), schedule


def _validate_external_trace_receipts(
    records: list[dict[str, Any]], kwargs: dict[str, Any]
) -> None:
    """Bind each service receipt to the immutable rank-local trace schedule."""

    from distillschedule.runtime.fsdp2_p0 import chunked

    required = ("trace", "rank", "steps", "policy")
    missing = [name for name in required if name not in kwargs]
    if missing:
        raise TypeError(f"external trace receipt validation misses: {missing}")
    rank = int(kwargs["rank"])
    trace = kwargs["trace"]
    groups = chunked(
        tuple(int(step) for step in kwargs["steps"]),
        int(kwargs["policy"].rollout_batch_size),
    )
    if len(records) != len(groups):
        raise ValueError(
            f"vLLM-Omni receipt groups={len(records)}, expected={len(groups)}"
        )
    for record, group in zip(records, groups, strict=True):
        actual_steps = [int(step) for step in record.get("logical_steps", [])]
        if actual_steps != list(group):
            raise ValueError(
                f"vLLM-Omni receipt steps={actual_steps}, expected={list(group)}"
            )
        expectations = [
            _expected_rollout_schedule(trace, rank=rank, step=step)
            for step in group
        ]
        if any(value != expectations[0] for value in expectations[1:]):
            raise ValueError(
                f"trace schedules differ inside external batch {list(group)}"
            )
        expected_calls, expected_stages = expectations[0]
        actual_stages = {
            int(stage): int(count)
            for stage, count in dict(record.get("stage_call_count", {})).items()
        }
        if int(record.get("call_count", -1)) != expected_calls:
            raise ValueError(
                f"vLLM-Omni call_count={record.get('call_count')}, "
                f"expected={expected_calls} for steps={list(group)}"
            )
        if actual_stages != expected_stages:
            raise ValueError(
                f"vLLM-Omni stage_call_count={actual_stages}, "
                f"expected={expected_stages} for steps={list(group)}"
            )
        record["trace_schedule_match"] = True


def _stage_timing_critic_inputs(kwargs: dict[str, Any]) -> None:
    """Preserve the hardened one-H2D Critic-entry contract without B1 parity."""

    if bool(kwargs.get("rollout_parity_check")):
        return
    required = ("trace", "rank", "steps", "device", "diagnostics")
    missing = [name for name in required if name not in kwargs]
    if missing:
        raise TypeError(f"external rollout staging misses arguments: {missing}")
    trace = kwargs["trace"]
    rank = int(kwargs["rank"])
    device = kwargs["device"]
    diagnostics = kwargs["diagnostics"]
    for step in kwargs["steps"]:
        with diagnostics.scope(
            f"INPUT_H2D/critic_entry_external_stage/step{step}"
        ):
            entry = trace.load_critic_entry(
                rank=rank,
                step=int(step),
                device=device,
            )
        diagnostics.add_tensor_tree(
            f"critic_entry_external_stage/step{step}",
            entry,
            direction="H2D",
        )


def _dispatch_precompute(*args: Any, **kwargs: Any):
    policy = kwargs.get("policy")
    if policy is None:
        raise TypeError("P0 rollout dispatch did not receive policy")
    if policy.rollout_backend == NATIVE_ROLLOUT_BACKEND:
        return _native_hardened_precompute(*args, **kwargs)
    if policy.rollout_backend != VLLM_OMNI_BACKEND:
        raise ValueError(f"unsupported rollout backend: {policy.rollout_backend}")
    outputs, records = precompute_vllm_omni_critic_rollouts(
        *args,
        **kwargs,
        variant=_variant_from_argv(),
    )
    _validate_external_trace_receipts(records, kwargs)
    # In correctness mode the external precompute already stages each entry for
    # the native B1 reference. Timing skips that reference, so stage once here
    # and let the matching Critic update consume the hardened one-use cache.
    _stage_timing_critic_inputs(kwargs)
    hardened._enforce_rollout_records(records)
    return outputs, records


impl._precompute_critic_rollouts = _dispatch_precompute
_original_atomic_json = impl.base._atomic_json


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    policy = value.get("p0_policy", {})
    configured = policy.get("rollout_backend") == VLLM_OMNI_BACKEND
    if configured:
        receipt = get_last_vllm_omni_receipt()
        value["external_rollout_backend_configured"] = True
        value["external_rollout_engine_used"] = receipt is not None
        value["rollout_collective_estimate"] = {
            "status": "NOT_APPLICABLE_TO_EXTERNAL_VLLM_OMNI_ENGINE",
            "reason": (
                "native FSDP materialization estimate does not describe the "
                "dedicated vLLM-Omni rollout pool"
            ),
            "remote_service_profile_required": True,
        }
        value["external_profile_boundary"] = {
            "training_node_nsys": "client wait plus returned-latent transfer",
            "rollout_gpu_nsys": "must be captured on the vLLM-Omni service",
        }
        if receipt is not None:
            value["vllm_omni_rollout"] = receipt
        if (
            value.get("status") == "PASS"
            and not value.get("validate_only")
            and receipt is None
        ):
            raise AssertionError("PASS external rollout run lacks vLLM-Omni receipt")
    _original_atomic_json(path, value)


impl.base._atomic_json = _atomic_json


def main() -> int:
    os.environ["DISTILLSCHEDULE_VARIANT"] = _variant_from_argv()
    VLLMOmniAdapterEnvironment.from_environment()
    if "--rollout-backend" not in sys.argv:
        sys.argv.extend(("--rollout-backend", VLLM_OMNI_BACKEND))
    return int(hardened.main())


if __name__ == "__main__":
    raise SystemExit(main())
