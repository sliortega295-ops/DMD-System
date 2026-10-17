"""Exact opt-in C51 full bundle; not independent factor/ablation admission.

Historical timings are deliberately absent. Fresh source-qualified native
calibration/correctness remains owned by the existing common workflow.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

from distillschedule.workloads.best_known.registry import fingerprint, repository_root, file_sha256

VARIANT = "c51-continuous-b16-generator-fused-rms"
SOURCE_HEAD = "aa30527db5d048e2ab23222eef587cb7cfe909f6"
PLAN = "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c51_generator_fused_rms.json"
TEMPLATE_SHA256 = "a50e4e3a969a513dae0399878f7e4f25e23085ee119e1175cad83192c0c6f7bb"


def source_template(*, root=None):
    from distillschedule.core.config import _load_with_includes
    value = _load_with_includes((root or repository_root()) / PLAN, stack=(), sources=[])
    if fingerprint(value) != TEMPLATE_SHA256:
        raise RuntimeError("C51 measured full-bundle source template changed")
    return value


def validate_plan(plan, *, mode):
    """Fail before model construction on shape/math/RNG/backend/placement drift."""
    from distillschedule.runtime.livewan_generator_fused_rms import CONTRACT, enabled
    rms = plan.get("metadata", {}).get("generator_rms_epilogue")
    if ((not isinstance(rms, dict) or rms.get("name") != CONTRACT["name"])
            and plan.get("id") != "livewan_wan13b_14b_sf_dmd_c51_generator_fused_rms"):
        return False
    if not enabled(plan, mode=mode):
        raise RuntimeError("C51 full bundle is not enabled")
    from distillschedule.core.plan import ExecutionPlan
    expected = ExecutionPlan.from_mapping(source_template()).to_dict()
    # Only the existing diagnostic adapters may alter cycle representation.
    current = plan["metadata"].get("continuous_pipeline", {})
    if current != expected["metadata"]["continuous_pipeline"]:
        from distillschedule.runtime.livewan_crossscale_worker import (
            _apply_trajectory_contract, _apply_two_cycle_timing_contract,
            TRAJECTORY_CONTINUOUS_NAME, TWO_CYCLE_TIMING_CONTINUOUS_NAME,
        )
        if current.get("name") == TRAJECTORY_CONTINUOUS_NAME and current.get("cycles") == 2:
            _apply_trajectory_contract(expected, cycles=2, mode=mode)
        elif current.get("name") == TWO_CYCLE_TIMING_CONTINUOUS_NAME:
            _apply_two_cycle_timing_contract(expected, mode=mode)
        else:
            raise RuntimeError("C51 requires the original source or two continuous cycles")
    actual = deepcopy(plan)
    # Paths are separately bound by the common assets/source/trace receipts.
    for value in (actual, expected):
        value.pop("id", None)
        value.pop("evidence", None)
        for role in ("generator", "score", "teacher"):
            value["workload"][role]["checkpoint"] = "{bound_checkpoint}"
        for key in ("legacy",):
            value.pop(key, None)
    if actual != expected:
        raise RuntimeError("C51 exact full-bundle shape/math/RNG/backend/placement contract changed")
    return True


def receipt_contract():
    return dict(schema_version="distillschedule.livewan_c51_full_bundle.v1",
                bundle="c51_full_v1", source_origin=SOURCE_HEAD,
                source_template_sha256=TEMPLATE_SHA256, independent_factors_admitted=False)


def effective_plan_bindings(run_root, *, kind="trajectory"):
    """Read the two actual native metadata-producer outputs; never reconstruct."""
    from distillschedule.runtime.livewan_crossscale_worker import (
        _effective_plan_sha256, TRAJECTORY_CONTINUOUS_NAME, TWO_CYCLE_TIMING_CONTINUOUS_NAME,
    )
    names = {"trajectory": TRAJECTORY_CONTINUOUS_NAME, "two-cycle-timing": TWO_CYCLE_TIMING_CONTINUOUS_NAME}
    if kind not in names:
        raise RuntimeError("C51 effective plan has an unsupported consumer")
    rows = []
    for node in (0, 1):
        path = Path(run_root) / f"effective_plan.node{node}.json"
        value = json.loads(path.read_text())
        recorded = value.pop("effective_plan_sha256", "")
        if recorded != _effective_plan_sha256(value) or not validate_plan(value, mode="pipeline"):
            raise RuntimeError("C51 actual effective plan source/shape/hash changed")
        continuous = value["metadata"]["continuous_pipeline"]
        if continuous.get("name") != names[kind] or continuous.get("cycles") != 2:
            raise RuntimeError("C51 actual effective plan is not the lowered two-cycle consumer plan")
        rows.append(dict(node_rank=node, path=str(path.resolve()), sha256=file_sha256(path),
                         effective_plan_sha256=recorded))
    if rows[0]["effective_plan_sha256"] != rows[1]["effective_plan_sha256"]:
        raise RuntimeError("C51 two-node actual effective plans differ")
    return rows


def validate_summary_effective_plans(summary):
    rows = effective_plan_bindings(summary["run_root"])
    if summary.get("c51_effective_plans") != rows:
        raise RuntimeError("C51 trajectory summary lost actual effective-plan bindings")
    return rows


def validate_timing_gate_binding(value):
    """C51-only v2 extension; canonical gate schemas/policies stay unchanged."""
    arm = value.get("arms", {}).get("pipeline", {})
    if arm.get("pipeline_variant") != VARIANT:
        raise RuntimeError("C51 timing gate physical bundle differs")
    path = Path(arm["trajectory_summary"])
    if file_sha256(path) != arm.get("trajectory_summary_sha256"):
        raise RuntimeError("C51 timing gate trajectory summary hash changed")
    summary = json.loads(path.read_text())
    if (summary.get("status") != "PASS" or summary.get("pipeline_variant") != VARIANT
            or summary.get("expected_head") != value.get("expected_head")
            or summary.get("arm_id") != arm.get("arm_id") or summary.get("seed") != value.get("seed")):
        raise RuntimeError("C51 timing gate trajectory source/arm/seed differs")
    rows = validate_summary_effective_plans(summary)
    if arm.get("artifact_binding", {}).get("c51_effective_plans") != rows:
        raise RuntimeError("C51 timing gate effective-plan binding changed")
    comparison = value["trajectory_comparison"]
    if file_sha256(Path(comparison["path"])) != comparison.get("sha256"):
        raise RuntimeError("C51 timing gate comparison hash changed")
    return rows


def runtime_receipt_errors(row, *, rank, cycles):
    """Actual per-rank operator counters, never config-only success."""
    from distillschedule.runtime.livewan_score_gate import runtime_receipt_errors as score_gate
    from distillschedule.runtime.livewan_score_layernorm import runtime_receipt_errors as layernorm
    from distillschedule.runtime.livewan_score_fa3 import runtime_receipt_errors as sac, backend_receipt_errors
    from distillschedule.runtime.livewan_generator_splitkv import runtime_receipt_errors as splitkv
    from distillschedule.runtime.livewan_generator_rms_epilogue import runtime_receipt_errors as rms
    from distillschedule.runtime.livewan_rollout_cast_cache import runtime_receipt_errors as casts
    from distillschedule.runtime.livewan_generator_modulation import runtime_receipt_errors as modulation
    from distillschedule.runtime.livewan_generator_gate import runtime_receipt_errors as gate
    from distillschedule.runtime.livewan_score_rmsnorm import runtime_receipt_errors as score_rms
    inventory = row.get("physical_call_inventory", row.get("call_inventory", {}))
    errors = []
    if "call_inventory" in row and "physical_call_inventory" in row and row["call_inventory"] != inventory:
        errors.append("C51 actual physical call inventories disagree")
    from distillschedule.search.livewan_request import LiveWanRequest
    request = LiveWanRequest()
    shapes = row.get("actual_latent_shapes", [])
    if (row.get("workload_request") != request.to_dict() or not shapes
            or any(not isinstance(s, list) or len(s) != 5 or s[0] not in (1, 2)
                   or tuple(s[1:]) != request.latent_shape[1:] for s in shapes)
            or not re.fullmatch("[0-9a-f]{64}", str(row.get("installed_plan_sha256", "")))):
        errors.append("C51 actual latent geometry/request/installed-plan observation missing or changed")
    marker = row.get("planner_c51_bundle", {})
    if any(marker.get(k) != v for k, v in receipt_contract().items()) or marker.get("execution_status") != "PASS":
        errors.append("C51 actual full-bundle version/source marker missing or changed")
    errors += score_gate(row.get("score_residual_gate", {}), rank=rank, inventory=inventory)
    errors += layernorm(row.get("score_layernorm", {}), rank=rank, inventory=inventory)
    errors += backend_receipt_errors(row.get("score_attention_backend", {}), rank=rank)
    errors += sac(row.get("critic_attention_checkpoint", {}), rank=rank, expected_calls=cycles * 5 * 30)
    attention = row.get("generator_attention_backend", {})
    errors += splitkv(attention.get("readonly_splitkv", {}), rank=rank,
                      total_calls=attention.get("invocation_count") if rank < 8 else None)
    errors += rms(row.get("generator_rms_epilogue", {}), rank=rank, cycles=cycles)
    errors += casts(row.get("generator_rollout_cast_cache", {}), rank=rank, cycles=cycles)
    errors += modulation(row.get("generator_modulation", {}), rank=rank, cycles=cycles)
    errors += gate(row.get("generator_gate", {}), rank=rank, cycles=cycles, output_layout="row_major")
    errors += score_rms(row.get("score_rmsnorm_kernel", {}), rank=rank)
    return errors
