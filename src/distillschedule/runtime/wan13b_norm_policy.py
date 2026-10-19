"""Default-off Wan width1536 normalization fusion experiment binding."""
from __future__ import annotations

import os
from typing import Any


ENV = "DISTILLSCHEDULE_WAN13B_MODULATED_NORM_FUSION"


def requested() -> bool:
    value = os.environ.get(ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{ENV} must be 0 or 1")
    return value == "1"


def install(*, role: str, placement: str, device: Any, fsdp_compatible: bool = False) -> dict[str, Any]:
    enabled = requested()
    if not enabled:
        return {"requested": False, "enabled": False, "disabled_reason": "NOT_REQUESTED"}
    if fsdp_compatible and placement != 'world16-fsdp':
        raise ValueError('compatible B normalization requires world16 FSDP')
    if placement != "resident-async128-tree-chunk2048" and not fsdp_compatible:
        raise ValueError("width1536 fusion is registered only for the Resident control")
    from helios.modules import transformer_helios as transformer
    from .wan13b_modulated_norm_fusion import (
        install_exact_modulated_norm_epilogue,
        warmup_exact_modulated_norm_epilogue,
    )

    if role == "generator":
        receipt = install_exact_modulated_norm_epilogue(
            enabled=True, role=role, transformer_module=transformer,
            fuse_materialization=True, fuse_backward=True,
            fuse_layernorm=True, fuse_backward_kernel=True,
        )
        receipt["registered_width"] = 1536
        receipt["native_chunk_tokens"] = 512 if fsdp_compatible else 2048
        receipt["scope"] = "PROCESS_SHARED_GENERATOR_AND_CRITIC_WIDTH1536_B1"
        receipt['fsdp_compatible_b'] = fsdp_compatible
        warmup_exact_modulated_norm_epilogue(receipt, device=device, chunk_tokens=receipt['native_chunk_tokens'])
    elif role == "critic":
        receipt = getattr(transformer, "_distillschedule_modulated_norm_receipt", None)
        if not isinstance(receipt, dict) or not receipt.get("enabled"):
            raise RuntimeError("Critic fusion has no source-qualified Generator installation")
        if (receipt.get('native_chunk_tokens') != (512 if fsdp_compatible else 2048)
                or receipt.get('fsdp_compatible_b', False) is not fsdp_compatible):
            raise RuntimeError('Critic normalization source/placement scope differs from Generator')
    else:
        raise ValueError(f"unregistered Wan role {role!r}")
    return receipt


def validate(build: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    receipts = {
        role: build.get(role, {}).get("metadata", {}).get("modulated_norm_fusion", {})
        for role in ("generator", "critic")
    }
    for role, receipt in receipts.items():
        if receipt.get("enabled") is not enabled or receipt.get("requested") is not enabled:
            raise AssertionError(f"{role}: requested norm policy differs from actual runtime")
        if enabled:
            if receipt.get("registered_width") != 1536:
                raise AssertionError(f"{role}: fusion width mismatch")
            for name in ("fused_call_count", "layernorm_fused_call_count", "backward_kernel_call_count"):
                if int(receipt.get(name, 0)) <= 0:
                    raise AssertionError(f"{role}: actual fusion counter missing: {name}")
            if receipt.get("reference_fallback_count") != 0:
                raise AssertionError(f"{role}: unexpected source fallback")
            if receipt.get("backward_kernel_reference_fallback_count", 0) != 0:
                raise AssertionError(f"{role}: unexpected backward fallback")
    return {"enabled": enabled, "counter_scope": "PROCESS_SHARED; DO_NOT_ADD_DUPLICATE_ROLE_RECEIPTS", "roles": receipts}
