"""Explicit R3 factor definitions; canonical/R2 arms retain their identities.

A changes device groups, sharding and legal scheduling. B changes retention,
reuse and existing operator execution, not those groups or update ordering.
Only these separately named candidates may use the new adapters.
"""
from __future__ import annotations

from typing import Any, Mapping

KEY = "system_factorial_scope_r3"
A_ARM = "only_A_g8_fsdp8_v1"
B_ARM = "only_B_full_fsdp16_v1"
ARMS = (A_ARM, B_ARM)


def is_arm(plan: Mapping[str, Any], arm: str) -> bool:
    return plan.get("metadata", {}).get("system_factorial_r2", {}).get("arm") == arm


def generator_fsdp8(plan: Mapping[str, Any], *, mode: str) -> bool:
    if not is_arm(plan, A_ARM):
        return False
    from .livewan_system_factorial_r2 import validate
    validate(plan, mode=mode)
    return True


def complete_b(plan: Mapping[str, Any], *, mode: str) -> bool:
    if not is_arm(plan, B_ARM):
        return False
    from .livewan_system_factorial_r2 import validate
    validate(plan, mode=mode)
    return True


def lower(plan: dict[str, Any], *, arm: str, full: Mapping[str, Any]) -> None:
    """Called only by exact source-bound R2 lowering for the two R3 IDs."""
    from .livewan_system_factorial_r2 import B_METADATA
    if arm == A_ARM:
        backend = plan["training_backends"]["generator"]
        backend.update(kind="fsdp")
        backend["options"].update(reshard_after_forward=True,
                                  optimizer_impl="foreach_adamw", resident=False,
                                  mesh_ranks=list(range(8)),
                                  gradient_sync="fsdp2_reduce_scatter_sum_divide_8")
        plan["metadata"][KEY] = dict(
            revision=1, A=True, B=False,
            placement="G8_FSDP8_C8_REPLICATED_TEACHER8_FSDP8",
            gradient_mean="G_REDUCE_SCATTER_SUM_DIVIDE_8_LOSS_SAMPLE_WEIGHT_HALF",
            generator_reshard_after_forward=True, optimizer_device="cuda",
            generator_collective_padding="max-noop-per-wave", reuse_mechanisms=[],
            reason="RESELECT_FEASIBLE_A_WITH_B_OFF_NOT_FIXED_RESIDENT_PLUS_OFFLOAD")
    elif arm == B_ARM:
        mechanism = {k: full["metadata"][k] for k in sorted(B_METADATA)
                     if k not in ("checkpoint_composition", "capacity_recovery")}
        plan["metadata"][KEY] = dict(
            revision=1, A=False, B=True,
            placement="ORIGINAL_WORLD16_FSDP2_ALL_ROLES",
            mechanism_contracts=mechanism,
            retention="ONE_NOGRAD_ROLLOUT_SAME_FSDP_UNITS_RESHARD_BEFORE_COMMIT",
            text_kv="LAZY_INSIDE_GATHERED_CAUSAL_BLOCK_NO_PARAMETER_VIEWS",
            checkpoint="CAPTURE_ROLE_ORIGINAL_RNG_CRITIC_FA3_SAC_ONLY",
            complete_B=True, known_wrong_merged_lora=False)
        plan["training_backends"]["generator"]["options"]["optimizer_impl"] = "fused_adamw"
    else:
        raise ValueError("not an R3 arm")


def mechanism_errors(row: Mapping[str, Any], *, arm: str, rank: int) -> list[str]:
    """Observe actual execution; requested configuration is never a hit count."""
    errors = []
    proof = row.get(KEY, {})
    if arm == A_ARM:
        if rank < 8:
            build = row.get("build", {}).get("student", {})
            if (build.get("r3_fsdp_group_size") != 8
                    or build.get("r3_fsdp_unit_count") != 35
                    or build.get("r3_registered_dtensor_parameters") is not True
                    or build.get("r3_optimizer_parameters_bound") is not True):
                errors.append("R3 G FSDP8 placement/optimizer binding not observed")
            comms = [x for x in row.get("gradient_sync", []) if x.get("role") == "generator"]
            if len(comms) != 4 or any(x.get("policy") != "fsdp2_reduce_scatter_sum_divide_8"
                                     for x in comms):
                errors.append("R3 G must reduce-scatter once per backward, no second manual all-reduce")
            padding = row.get("student_collective_padding", {})
            if len(padding.get("records", [])) != 40:
                errors.append("R3 FSDP8 per-wave padding records missing")
        if proof.get("complete_B") is not False:
            errors.append("R3 A must report B disabled")
        return errors
    if arm != B_ARM:
        return errors
    helper = proof.get("runtime", {})
    dispatch = helper.get("role_dispatch", {})
    if (helper.get("placement") != "UNCHANGED_ORIGINAL_FSDP16_UNITS_AND_GROUPS"
            or helper.get("generator_commit_version") != 4
            or helper.get("observed_generator_commits") != 4
            or dispatch.get("scope_failures") != 0 or dispatch.get("current_role") is not None):
        errors.append("R3 complete-B role/placement/commit contract failed")
    checkpoint = helper.get("critic_attention_checkpoint", {})
    inventory = row.get("physical_call_inventory", {})
    g = int(inventory.get("student/grad/plain", 0)) * 30
    # Some source versions call the application label 'checkpoint'; both are
    # genuine inventory labels, not values inferred from the requested plan.
    if not g:
        g = sum(int(v) for k, v in inventory.items() if k.startswith("student/grad/")) * 30
    c = int(inventory.get("critic/grad/checkpoint", 0)) * 30
    if (checkpoint.get("role_checkpoint_calls") != dict(student=g, critic=c, teacher=0, original=0)
            or checkpoint.get("checkpoint_calls") != g+c
            or checkpoint.get("function_entries") != 2*(g+c)
            or checkpoint.get("replay_entries") != g+c
            or checkpoint.get("sac_checkpoint_calls") != c
            or checkpoint.get("production_installed") is not True):
        errors.append("R3 captured-role original/replay census mismatch")
    sac = checkpoint.get("fa3_sac", {})
    if (sac.get("replay_fa_cache_reads") != c or sac.get("contract_failures") != 0):
        errors.append("R3 actual Critic FA3 SAC reuse census failed")
    retention = helper.get("generator_rollout_cast_cache", {})
    if (retention.get("scope_entries") != 20 or retention.get("scope_exits") != 20
            or retention.get("unit_count") != 35 or retention.get("active") is not False
            or any(retention.get(k) != 700 for k in
                   ("retain_policy_calls", "reshard_calls", "restore_policy_calls"))
            or retention.get("contract_failures") != 0
            or retention.get("parameter_version_failures") != 0):
        errors.append("R3 rollout retention/reshard/cast lifetime failed")
    kv = helper.get("generator_cross_attention_kv_cache", {})
    if (kv.get("scope_entries") != 20 or kv.get("scope_exits") != 20
            or kv.get("cache_build_context_count") != 20
            or kv.get("cache_build_block_count") != 600
            or kv.get("cache_hit_count", 0) <= 600
            or kv.get("active") is not False or kv.get("retained_cache_bytes") != 0
            or kv.get("contract_failure_count") != 0):
        errors.append("R3 lazy text K/V execution/invalidation failed")
    kernels = helper.get("kernels", {})
    expected = ("critic_lora_kernel", "score_modulation_kernel", "score_rmsnorm_kernel",
                "score_rope_kernel", "score_attention_backend", "score_residual_gate",
                "score_layernorm", "generator_attention_backend", "generator_rms_epilogue",
                "generator_modulation", "generator_gate", "generator_residual_layout",
                "generator_readonly_splitkv")
    for key in expected:
        if kernels.get(key, {}).get("execution_status") != "PASS":
            errors.append("R3 complete-B mechanism not observed: " + key)
        for name, value in kernels.get(key, {}).items():
            if (name.endswith(("failure_count", "failures")) and isinstance(value, int)
                    and value != 0):
                errors.append("R3 kernel contract failure: " + key + "/" + name)
    if (proof.get("observed_generator_optimizer_fused") is not True
            or proof.get("observed_generator_optimizer_dtensor_parameters") is not True):
        errors.append("R3 original FSDP16 fused Generator AdamW not observed")
    return errors
