"""Source-pinned world8 Wan bridge into the *common* model planner.

The two operations below are coupled native cycles, not invented independent
G/C costs. The unchanged native launcher owns their internal schedule. This
module performs no process launch, CUDA import, SSH, or candidate ranking.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import shlex
from typing import Any, Mapping

from distillschedule.core.enums import GradientMode, OperationKind
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation
from distillschedule.runtime.p0_parity import DEFAULT_P0_PARITY
from distillschedule.workloads.best_known.registry import fingerprint, file_sha256, write_new_json
from .allocation import ExecutionAllocation
from .contracts import finite_number, require
from .cost_model import ComponentContract, CostModel, OperationCost
from .model_search import validate_saved_prediction
from .plan_ir import BindingRule, PlanningProblem, validate_binding

ADAPTER = "hardware_wan_coupled_v1"
WORKLOAD = "wan13b_nogan_resident_best"
FAMILY = "hardware_native.wan_cycle"
NATIVE_HEAD = "7a668bcb12ad37d831dced3aca1be4ec7be0a2fe"
LAUNCHER = "scripts/distillschedule/launch_wan13b_hardware_two_cycle_node.sh"
WORKER = "scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py"
# An allowlisted dependency, not an arbitrary user head plus self-reported hash.
NATIVE_FILES = {
    "launcher_sha256": "044788b6d478c4797656c65d2d85572648d1ceec9bbc551780cb8a199eb70fd0",
    "worker_sha256": "4608568e7d305d0ca0c9bc1674e0d3a80e0d54c9c16cb0df6c79575d240668d9",
}
PLACEMENTS = ("world16-fsdp", "resident", "resident-async128-tree-chunk2048")
PRECISION = {"frozen_base": "torch.bfloat16", "trainable_lora": "torch.float32",
             "gradient_reduction": "SUM_THEN_DIVIDE_WORLD_SIZE",
             "loss_scale_per_logical_b1": 0.5, "ema_implementation": "reference"}
SEMANTICS = {
    "bootstrap_hash": "6a713476d9839cf1ea37689d63cb1cc2b405b9f9915ca5314ccdadb67b3cd60d",
    "model_identity_sha256": "d52cf9e2d04451c018ac50c4ee1043c4ceba4014e51549bfda2ea5a123c65f54",
    "trace_bundle_hash": "654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3",
    "trace_binding_hash": "bddcf32b379ea3d0cb36237b0c21f103ea59b1635473dd8080b7182cf3c1eded",
    "negative_prompt_manifest_hash": "ca0de6928f83ebd450418d33f8990696763d3228069ec4ac0efb2444694cf78c",
    "trace_rollout_contract_sha256": "d4964217e77fed2434feffbfc0c16b80a6bb9fb7b6adbdd9d9172a6ee0f2f6f8",
    "trace_replay_policy": "FIXED_FIVE_STEP_COHORT_REPLAY", "seed": 20260903,
    "global_batch": 16, "updates": "G1_C5_EMA5", "rollout_batch_size": 1, "score_packing": False,
}
MEMORY = "MAX_RANK_POST_SETUP_RESERVED_PEAK_AS_WHOLE_CYCLE_ENVELOPE"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_STATE_COMPONENTS = frozenset(("generator_trainable", "critic_trainable", "generator_adam", "critic_adam", "generator_ema"))
_CYCLE_CONTRACT = {"cycles_in_same_process": 2, "cycle_1_role": "WARMUP_DIAGNOSTIC",
    "cycle_2_role": "STEADY_STATE_CLAIMABLE", "claimable_cycle": 2,
    "trace_replay_policy": "FIXED_FIVE_STEP_COHORT_REPLAY",
    "updates_per_cycle": {"generator": 1, "critic": 5, "ema": 5}}


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _absolute(value, label):
    require(isinstance(value, str) and Path(value).is_absolute()
            and ".." not in Path(value).parts, f"{label}: absolute normalized path required")
    return Path(value)


def _allocation(value):
    a = value if isinstance(value, ExecutionAllocation) else ExecutionAllocation.from_dict(value)
    require(a.world_size == 8 and a.node_count == 1 and a.logical_global_batch == 16
            and all(d.architecture == "sm90" for d in a.devices),
            "only one eight-H100 allocation with sixteen independent B1 samples is admitted")
    require({"generator", "critic"} <= set(a.role_slots)
            and set(a.role_slots) <= {"generator", "critic", "teacher"}, "unexpected Wan roles")
    require(all(rows == tuple((r, r + 8) for r in range(8)) for rows in a.role_slots.values()),
            "native Wan requires two independent slots (rank, rank+8), not physical B2")
    return a


def validate_deployment(value: Mapping[str, Any]) -> dict[str, Any]:
    """Pure-data admission; the dispatcher additionally verifies live source/host."""
    required = {"schema_version", "binding", "native_source", "allocation", "environment", "assets",
                "semantics", "precision", "evidence_scope", "site", "transport"}
    require(isinstance(value, Mapping) and required <= set(value)
            and set(value) <= required | {"workload"}, "unknown/missing hardware deployment fields")
    d = _json(dict(value))
    require(d["schema_version"] == "distillschedule.hardware_wan_deployment.v1", "deployment schema mismatch")
    require(d.get("workload", WORKLOAD) == WORKLOAD, "W2-W6/A100 are NOT_ADMITTED by this adapter")
    require(d["evidence_scope"] in ("MEASURED", "CPU_FIXTURE_ONLY"), "unknown hardware evidence scope")
    native = d["native_source"]
    require(set(native) == {"head", "repo_root", *NATIVE_FILES}, "unknown/missing native source identity")
    require(native["head"] == NATIVE_HEAD and all(native[k] == v for k, v in NATIVE_FILES.items()),
            "native source is not the registered clean7a Wan runtime")
    _absolute(native["repo_root"], "native repo")
    a = _allocation(d["allocation"])
    validate_binding(d["binding"])
    for key in ("environment", "assets"):
        require(isinstance(d[key], dict) and bool(d[key]), f"missing {key} contract")
    require(not any(k.startswith(("NCCL_", "DISTILLSCHEDULE_")) for k in d["environment"]),
            "candidate launcher knobs are not environment identity")
    if d["evidence_scope"] == "MEASURED":
        require(isinstance(d["environment"].get("driver_version"), str)
                and re.fullmatch(r"\d+(?:\.\d+)+", d["environment"]["driver_version"]),
                "MEASURED environment must bind its observed NVIDIA driver_version")
    require(d["semantics"] == SEMANTICS and d["precision"] == PRECISION,
            "native matched workload/precision changed; new requests require a new adapter")
    for key in ("bootstrap_hash", "trace_bundle_hash", "model_identity_sha256", "negative_prompt_manifest_hash"):
        require(d["assets"].get(key) == SEMANTICS[key], f"native asset identity changed: {key}")
    for key, material in (("hardware_fingerprint", a.to_dict()), ("environment_fingerprint", d["environment"]),
                          ("asset_fingerprint", d["assets"]), ("semantic_fingerprint", d["semantics"]),
                          ("precision_fingerprint", d["precision"])):
        require(d["binding"][key] == fingerprint(material), f"deployment identity mismatch: {key}")
    require(set(d["site"]) == {"project_root", "python", "helios_root", "asset_root", "run_root"},
            "site fields changed")
    for k, p in d["site"].items():
        _absolute(p, f"site.{k}")
    require(Path(d["site"]["run_root"]) == Path(d["site"]["project_root"]) /
            "research_artifacts/five_workload_h100_20260930", "native launcher run root cannot be overridden")
    transport = d["transport"]
    require(set(transport) == {"host", "ssh_port", "address_family"}, "transport fields changed")
    require(isinstance(transport["host"], str) and bool(transport["host"])
            and type(transport["ssh_port"]) is int and 0 < transport["ssh_port"] < 65536
            and transport["address_family"] in ("auto", "ipv4", "ipv6", 4, 6), "invalid controller transport")
    return d


def _arm(placement):
    require(placement in PLACEMENTS, "unregistered native placement")
    return {"arm_contract": dict(zip(PLACEMENTS, ("WORLD16_FSDP_BASELINE", "RESIDENT_REPLICATED_REFERENCE",
        "RESIDENT_ASYNC128_TREE_CHUNK2048_FUSED_EPILOGUE")))[placement],
        "generator_nograd_lora_cast_cache": False,
        "generator_nograd_lora_fused_epilogue": placement == PLACEMENTS[2]}


def _runtime(placement):
    return {"helios_transformer_sha256": "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886",
        "ema_implementation": "reference", "nccl_algo": "Tree" if placement == PLACEMENTS[2] else "AUTO",
        "residual_storage": "cuda_out_of_place", "parameter_offload": False, "activation_offload": False,
        "optimizer_offload": False, "parameter_streaming": False, "model_swap": False}


def _fused(row, placement):
    receipt = row.get("fused_epilogue_receipt", {})
    g, c = receipt.get("generator", {}), receipt.get("critic", {})
    if placement == PLACEMENTS[2]:
        require(g.get("requested") is True and g.get("enabled") is True
                and g.get("model_call_count") == 112 and g.get("fused_invocation_count") == 34160
                and g.get("reference_fallback_count") == 448
                and g.get("reference_fallback_reasons") == {"BASE_DTYPE_torch.float32": 448}
                and c.get("enabled") is False and c.get("disabled_reason") == "GENERATOR_ONLY",
                "selected fused kernel was not executed with the native call census")
    else:
        require(g.get("requested") is False and g.get("enabled") is False and c.get("enabled") is False
                and g.get("fused_invocation_count") == 0, "reference placement enabled a candidate kernel")


def _census(row, payload_sets):
    trace = row.get("trace_rollout_contract")
    require(isinstance(trace, list) and len(trace) == 6
            and fingerprint(trace) == SEMANTICS["trace_rollout_contract_sha256"], "rollout call inventory changed")
    census = row.get("cycle_census", [])
    require(len(census) == 2, "missing two-cycle sample census")
    previous = None
    for index, outer in enumerate(census):
        require(all(outer.get(k) == v for k, v in {"generator_updates": 1, "critic_updates": 5,
            "ema_updates": 5, "physical_rollout_batches": 10, "rollout_batch_size": 1,
            "logical_slots": row["logical_slots"]}.items()), "physical update or B1 census changed")
        samples = outer.get("logical_sample_census", [])
        require(len(samples) == 2, "native physical worker did not consume two logical B1 slots")
        payloads = []
        for sample in samples:
            require(all(sample.get(k) == v for k, v in {"generator_updates": 1, "critic_updates": 5,
                "ema_updates": 5, "generator_score_forwards": 3, "critic_score_forwards": 5,
                "physical_rollout_batches": 5, "rollout_batch_size": 1}.items()), "logical B1 call census changed")
            ids = sample.get("sample_payload_identities", [])
            require(len(ids) == 6 and all(isinstance(x, str) and _SHA.fullmatch(x) for x in ids)
                    and ids == [sample.get("generator_trace_payload_identity"),
                                *sample.get("critic_trace_payload_identities", [])], "missing independent trace payloads")
            require(sample.get("rollout_operation_keys") == [x["operation"] for x in trace]
                    and sample.get("rollout_call_inventory") == [x["inventory"] for x in trace],
                    "executed rollout inventory differs from the trace contract")
            require(sample.get("sample_occurrence_identities") ==
                    [f"cycle={index+1}/slot={i}/payload={p}" for i, p in enumerate(ids)],
                    "sample occurrence is not bound to the native cycle")
            payloads.extend(ids)
        require(previous is None or previous == payloads, "C2 did not replay the fixed five-step cohort")
        previous = payloads
        payload_sets[index].extend(payloads)


def validate_native_summary(summary: Mapping[str, Any], *, expected_native_head: str,
                            placement: str, allocation: ExecutionAllocation | Mapping[str, Any]) -> dict[str, Any]:
    """Validate native data, including raw per-rank execution rather than PASS alone.

    Historical schema inspection may pass its historical head explicitly. Cost
    admission always calls this with the registered native head, never that head.
    Correctness summaries are accepted for launch admission, never as costs.
    """
    _allocation(allocation)
    arm = _arm(placement)
    require(isinstance(summary, Mapping), "native summary must be an object")
    run_id, mode = summary.get("run_id"), summary.get("mode")
    require(isinstance(run_id, str) and _RUN_ID.fullmatch(run_id), "invalid native run id")
    require(mode in ("timing", "correctness"), "only native correctness/timing receipts are admitted")
    expected = {"schema_version": "distillschedule.wan13b_trajectory_summary.v1", "status": "PASS",
        "errors": [], "cycles": 2, "rank_count": 8, "logical_world_size": 16,
        "expected_head": expected_native_head, "placement": placement, "hardware_profile": "h100-8-b16",
        "final_versions": {"generator": 4, "critic": 20, "ema": 20}, "matched_contract": SEMANTICS,
        "workload_fingerprint": fingerprint(SEMANTICS), "diagnostic_only": mode != "timing",
        "formal_timing_eligible": mode == "timing"}
    require(fingerprint({k: summary.get(k) for k in expected}) == fingerprint(expected),
            "native summary identity/semantic/status mismatch")
    impl = {**arm, "hardware_profile": "h100-8-b16", "microbatch_count": 2, "trajectory_arm_id": f"h100-{placement}"}
    require(summary.get("implementation_contract") == impl
            and summary.get("implementation_fingerprint") == fingerprint(impl), "native implementation fingerprint mismatch")
    health = summary.get("launcher_health", {})
    require(health.get("status") == "PASS" and health.get("exit_code") == 0
            and health.get("expected_head") == expected_native_head
            and health.get("cgroup_oom_delta") == {"oom": 0, "oom_kill": 0}
            and health.get("gpu_processes_after") == [], "native launcher OOM/exit/cleanup admission failed")
    for key in ("oom", "oom_kill"):
        before, after = health.get("cgroup_before", {}).get(key), health.get("cgroup_after", {}).get(key)
        require(type(before) is int and type(after) is int and after == before and before >= 0,
                "OOM delta is inconsistent with raw cgroup counters")
    rows = summary.get("ranks", [])
    require(isinstance(rows, list) and len(rows) == 8
            and all(type(r.get("rank")) is int for r in rows)
            and sorted(r["rank"] for r in rows) == list(range(8)), "missing/duplicate physical rank receipts")
    require(all(type(s) is int for s in summary.get("logical_slots", []))
            and sorted(summary.get("logical_slots", [])) == list(range(16)), "summary logical-slot coverage changed")
    rows = sorted(rows, key=lambda r: r["rank"])
    payload_sets, times, reserved, allocated = [[], []], [[], []], [], []
    correctness, comparison = rows[0].get("correctness_binding"), rows[0].get("trajectory_comparison_binding")
    hostnames = set()
    for row in rows:
        rank = row["rank"]
        required = {"schema_version": "distillschedule.wan13b_two_cycle_timing_rank.v1", "status": "PASS",
            "mode": mode, "run_id": run_id, "placement": placement, "expected_head": expected_native_head,
            "world_size": 8, "local_rank": rank, "hardware_profile": "h100-8-b16", "logical_world_size": 16,
            "logical_slots": [rank, rank+8], "microbatch_count": 2, "profiled": False,
            "profiler_status": "NOT_RUN", "finite_status": [True, True], "cycle_contract": _CYCLE_CONTRACT,
            "initial_versions": {"generator": 2, "critic": 10, "ema": 10},
            "cycle_versions": [{"generator": 3, "critic": 15, "ema": 15}, {"generator": 4, "critic": 20, "ema": 20}],
            "final_versions": expected["final_versions"], "updates": {"generator": 2, "critic": 10, "ema": 10},
            "matched_contract": SEMANTICS, "workload_fingerprint": fingerprint(SEMANTICS),
            "arm_contract": arm, "implementation_fingerprint": fingerprint(arm), "runtime_contract": _runtime(placement),
            "trace_rollout_contract_sha256": SEMANTICS["trace_rollout_contract_sha256"],
            "formal_timing_eligible": mode == "timing", "timing_seed": SEMANTICS["seed"],
            "rank_runtime_seed": SEMANTICS["seed"] + rank,
            "timing_arm_id": f"h100-{placement}", "correctness_binding": correctness,
            "trajectory_comparison_binding": comparison}
        require(fingerprint({k: row.get(k) for k in required}) == fingerprint(required),
                f"native rank {rank} contract/version/finite mismatch")
        require(row.get("configuration_fingerprint") == fingerprint({"workload_fingerprint": fingerprint(SEMANTICS),
            "implementation_fingerprint": fingerprint(arm)}), "native rank configuration fingerprint mismatch")
        require(isinstance(row.get("hostname"), str) and row["hostname"], "native hostname missing")
        hostnames.add(row["hostname"])
        require(row.get("model_relocation", {}).get("content_validation") ==
                ("SHA256_PASS" if rank == 0 else "SIZE_PASS_HASH_VERIFIED_BY_RANK0")
                and row["model_relocation"].get("model_identity_sha256") == SEMANTICS["model_identity_sha256"],
                "native model content identity unverified")
        _fused(row, placement)
        _census(row, payload_sets)
        cycles = row.get("cycles", [])
        require(len(cycles) == 2, "two uninterrupted native cycles required")
        last_stop = 0
        for i, cycle in enumerate(cycles):
            seconds = finite_number(cycle.get("local_elapsed_seconds"), "native cycle duration", positive=True)
            start, stop = cycle.get("start_unix_ns"), cycle.get("stop_unix_ns")
            require(cycle.get("cycle") == i+1 and cycle.get("role") == _CYCLE_CONTRACT[f"cycle_{i+1}_role"]
                    and type(start) is int and type(stop) is int and last_stop < start < stop,
                    "cycle clocks/order changed")
            # UNIX marks bracket the same perf-counter interval; tolerate timer-call overhead.
            require(abs((stop-start)/1e9-seconds) <= max(0.1, seconds*0.01), "native duration disagrees with clock envelope")
            last_stop = stop
            times[i].append(seconds)
        memory = row.get("memory", {})
        for k in ("max_reserved_bytes", "max_allocated_bytes", "driver_total_bytes"):
            require(type(memory.get(k)) is int and memory[k] > 0, f"native memory evidence missing: {k}")
        require(memory["max_allocated_bytes"] <= memory["max_reserved_bytes"] <= memory["driver_total_bytes"],
                "inconsistent native memory counters")
        reserved.append(memory["max_reserved_bytes"])
        allocated.append(memory["max_allocated_bytes"])
    require(len(hostnames) == 1, "world8 ranks came from different hosts")
    require(all(len(p) == 96 and len(set(p)) == 96 for p in payload_sets), "logical samples were duplicated across physical slots")
    measured = [max(values) for values in times]
    require(summary.get("cycle_seconds_max_rank") == measured, "summary timing differs from raw rank maxima")
    if mode == "timing":
        require(isinstance(correctness, dict) and correctness.get("status") == "PASS"
                and correctness.get("expected_head") == expected_native_head
                and correctness.get("matched_contract") == SEMANTICS
                and correctness.get("implementation_contract") == impl, "native timing lacks same-source correctness binding")
        require(isinstance(comparison, dict) and comparison.get("contract_status") == "PASS"
                and comparison.get("complete") is True
                and comparison.get("status") in ("HARDWARE_TWO_CYCLE_PASS", "HARDWARE_TWO_CYCLE_REVIEW")
                and comparison.get("matched_correctness_run_id") == correctness.get("run_id"),
                "native timing lacks a complete diagnostic trajectory comparison")
    else:
        require(correctness == comparison == {"status": "NOT_RUN"}, "correctness receipt has timing-only bindings")
    return {"run_id": run_id, "mode": mode, "placement": placement, "native_head": expected_native_head,
        "cycle_seconds": measured, "cycle_ns": [round(x*1e9) for x in measured],
        "peak_reserved_bytes": max(reserved), "max_allocated_bytes": max(allocated),
        "matched_contract": _json(SEMANTICS), "workload_fingerprint": fingerprint(SEMANTICS),
        "implementation_fingerprint": fingerprint(impl), "numerical_status": "REVIEW",
        "strict_full_state_parity": "NOT_RUN", "correctness_binding": correctness,
        "trajectory_comparison_binding": comparison}


def _read_ref(ref, *, text=False):
    require(isinstance(ref, Mapping) and set(ref) == {"path", "sha256"}, "expected content-hashed raw file reference")
    path = _absolute(ref["path"], "raw reference")
    require(isinstance(ref["sha256"], str) and _SHA.fullmatch(ref["sha256"]), "invalid raw file digest")
    require(path.is_file(), f"raw evidence missing: {path}")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == ref["sha256"], f"raw evidence hash changed: {path}")
    return data.decode() if text else json.loads(data)


def _validate_evidence_scope(value, scope):
    """Legacy native receipts have no scope; explicit fixture marks never upgrade."""
    if scope != "MEASURED":
        return
    if isinstance(value, Mapping):
        require(all(value.get(k) != "CPU_FIXTURE_ONLY" for k in ("evidence_scope", "fixture_scope")),
                "CPU fixture evidence cannot authorize MEASURED native timing")
        for child in value.values():
            _validate_evidence_scope(child, scope)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_evidence_scope(child, scope)


def validate_native_comparison(value, *, evidence_scope=None):
    """Pure-data verification of the pinned comparator, without importing Torch.

    Finite numerical threshold failures remain REVIEW, never strict parity.
    Recorded latent dtype is preserved; only agreement between sides is required.
    """
    require(isinstance(value, Mapping), "native comparison must be an object")
    _validate_evidence_scope(value, evidence_scope)
    expected = {"schema_version": "distillschedule.trajectory_coarse_comparison.v1",
        "comparison_kind": "semantic", "expected_cycles": 2, "expected_world_size": 8,
        "hardware_profile": "h100-8-b16", "expected_head": NATIVE_HEAD,
        "contract": {"status": "PASS", "mismatches": []}, "complete": True,
        "diagnostic_timing_eligible": True, "strict_full_state_parity": "NOT_RUN",
        "strict_claimable_for_paired_timing": False}
    require(fingerprint({k: value.get(k) for k in expected}) == fingerprint(expected),
            "native comparator did not admit diagnostic timing with unchanged strict parity boundary")
    metrics = value.get("numeric_metrics")
    require(isinstance(metrics, list) and len(metrics) == 165, "native comparison needs 160 rollout and five sampled-state metrics")
    expected_keys = {("rollout", c, slot, step) for c in (1, 2) for slot in range(16) for step in range(10, 15)}
    expected_keys |= {("sampled_state", name) for name in _STATE_COMPONENTS}
    keys, statuses = set(), []
    thresholds = asdict(DEFAULT_P0_PARITY)
    for metric in metrics:
        require(isinstance(metric, Mapping), "invalid native comparison metric")
        kind = metric.get("kind")
        if kind == "rollout":
            fields = tuple(metric.get(k) for k in ("cycle", "logical_slot", "step"))
            require(all(type(x) is int for x in fields), "rollout metric keys must be integers")
            key = (kind, *fields)
        else:
            key = (kind, metric.get("component"))
        require(key in expected_keys and key not in keys, "missing/duplicate or unknown native comparison metric")
        keys.add(key)
        require(all(metric.get(k) is True for k in ("finite", "same_shape", "same_dtype")),
                "native comparison finite/shape/dtype gate failed")
        shape, dtype = metric.get("reference_shape"), metric.get("reference_dtype")
        require(isinstance(shape, list) and all(type(x) is int and x > 0 for x in shape)
                and shape == metric.get("candidate_shape") and isinstance(dtype, str) and dtype.startswith("torch.")
                and dtype == metric.get("candidate_dtype"), "native metric shape/dtype evidence contradicts flags")
        require(metric.get("thresholds") == thresholds, "native DEFAULT_P0_PARITY thresholds changed")
        numbers = {k: finite_number(metric.get(k), f"metric.{k}") for k in
                   ("max_abs", "max_abs_limit", "mean_abs", "mean_relative", "normalized_l2", "reference_peak_abs")}
        cosine = metric.get("cosine")
        require(type(cosine) in (int, float) and math.isfinite(cosine) and -1.000001 <= cosine <= 1.000001,
                "native metric cosine is not finite/bounded")
        policy = metric.get("cosine_policy")
        require(policy in ("standard", "both_zero_defined_as_one", "single_zero_defined_as_zero")
                and (policy != "both_zero_defined_as_one" or cosine == 1)
                and (policy != "single_zero_defined_as_zero" or cosine == 0), "native cosine policy changed")
        limit = thresholds["max_abs_atol"] + thresholds["max_abs_rtol"] * numbers["reference_peak_abs"]
        # The native implementation computes this expression in FP32.
        require(math.isclose(numbers["max_abs_limit"], limit, rel_tol=1e-6, abs_tol=1e-8),
                "native max-absolute threshold does not match DEFAULT_P0_PARITY")
        passed = (numbers["mean_relative"] <= thresholds["mean_relative"]
                  and numbers["normalized_l2"] <= thresholds["normalized_l2"]
                  and cosine >= thresholds["min_cosine"] and numbers["max_abs"] <= numbers["max_abs_limit"])
        status = "PASS" if passed else "FAIL"
        require(metric.get("status") == status, "native metric status contradicts its finite numerical values")
        statuses.append(status)
    require(keys == expected_keys, "native comparison metric inventory is incomplete")
    passed = all(status == "PASS" for status in statuses)
    require(value.get("status") == ("HARDWARE_TWO_CYCLE_PASS" if passed else "HARDWARE_TWO_CYCLE_REVIEW")
            and value.get("numerical_status") == ("PASS_OBSERVED_ROLLOUTS_AND_SAMPLED_STATE" if passed else "REVIEW"),
            "native comparison status/numerical_status contradicts metric results")
    return value


def _correctness_proof(d, reference):
    summary = _read_ref(reference)
    _validate_evidence_scope(summary, d["evidence_scope"])
    normalized = validate_native_summary(summary, expected_native_head=NATIVE_HEAD,
        placement=summary.get("placement"), allocation=d["allocation"])
    require(normalized["mode"] == "correctness", "native comparison requires correctness inputs")
    root = Path(d["site"]["run_root"])/normalized["run_id"]
    require(Path(reference["path"]) == root/"summary.json", "correctness summary outside its native run root")
    seen, anchor_refs = set(), []
    for row in summary["ranks"]:
        artifacts = row.get("rollout_evidence")
        require(isinstance(artifacts, list) and len(artifacts) == 4, "correctness rank needs four rollout files")
        for artifact in artifacts:
            require(isinstance(artifact, dict) and {"cycle", "logical_slot", "path", "sha256"} <= set(artifact),
                    "missing correctness rollout artifact identity")
            c, slot = artifact["cycle"], artifact["logical_slot"]
            require(type(c) is int and c in (1, 2) and type(slot) is int and slot in row["logical_slots"]
                    and (c, slot) not in seen, "correctness rollout inventory changed/duplicated")
            seen.add((c, slot))
            path = _absolute(artifact["path"], "correctness rollout")
            require(path == root/f"execution/rollouts/cycle{c}_slot{slot:02d}.pt"
                    and path.is_file() and isinstance(artifact["sha256"], str) and _SHA.fullmatch(artifact["sha256"]),
                    "correctness rollout artifact missing or relocated")
            # Bounded-memory streaming; never unpickle or copy tensor artifacts.
            require(file_sha256(path) == artifact["sha256"], "correctness rollout artifact hash changed")
        for role in ("generator", "critic"):
            records = row.get("build", {}).get(role, {}).get("lora_compute_dtype_contract", {}).get("records")
            require(isinstance(records, list) and bool(records) and all(
                r.get("base_weight_dtype") == "torch.bfloat16" and r.get("lora_a_weight_dtype") == "torch.float32"
                and r.get("lora_b_weight_dtype") == "torch.float32" for r in records),
                "correctness runtime LoRA/base dtype evidence missing or changed")
        anchor_refs.append(row.get("state_anchors"))
    require(seen == {(c, s) for c in (1, 2) for s in range(16)}, "correctness rollout inventory incomplete")
    require(all(isinstance(ref, dict) and ref == anchor_refs[0] for ref in anchor_refs),
            "correctness ranks disagree on the global state anchor")
    spec = anchor_refs[0]
    require(set(spec) == {"path", "sha256"}, "global state anchor reference changed")
    path = _absolute(spec["path"], "global state anchor")
    require(path == root/"execution/trajectory/anchors/cycle_02_global.json" and path.is_file(),
            "correctness global state anchor missing or relocated")
    anchor = json.loads(path.read_text())
    _validate_evidence_scope(anchor, d["evidence_scope"])
    require(isinstance(anchor, dict), "invalid global state anchor")
    digest = anchor.get("anchor_sha256")
    require(digest == spec["sha256"] == fingerprint({k: v for k, v in anchor.items() if k != "anchor_sha256"}),
            "correctness global state anchor canonical digest changed")
    require(anchor.get("schema_version") == "distillschedule.wan13b_trajectory_state_anchor.v2"
            and type(anchor.get("cycle")) is int and anchor["cycle"] == 2
            and type(anchor.get("writer_rank")) is int and anchor["writer_rank"] == 0
            and anchor.get("placement") == summary["placement"]
            and isinstance(anchor.get("components"), dict) and set(anchor["components"]) == _STATE_COMPONENTS,
            "global state anchor component/source inventory changed")
    for component in anchor["components"].values():
        sketch = component.get("numeric_sketch", {})
        require(isinstance(sketch.get("inventory_sha256"), str) and _SHA.fullmatch(sketch["inventory_sha256"]),
                "global state sketch inventory hash missing")
        require(all(type(sketch.get(k)) is int and sketch[k] > 0 for k in
                    ("inventory_tensor_count", "tensor_count", "coordinate_count")), "global state sketch count missing")
        coords = sketch.get("coordinates", [])
        require(isinstance(coords, list) and len(coords) == sketch["coordinate_count"], "global state coordinate count changed")
        coordinate_keys = set()
        for coordinate in coords:
            require(isinstance(coordinate.get("tensor"), str) and coordinate["tensor"]
                    and type(coordinate.get("flat_index")) is int and coordinate["flat_index"] >= 0,
                    "invalid global state coordinate")
            key = (coordinate["tensor"], coordinate["flat_index"])
            require(key not in coordinate_keys, "duplicate global state coordinate")
            coordinate_keys.add(key)
            number = coordinate.get("value_fp32")
            require(type(number) in (int, float) and math.isfinite(number), "non-finite global state coordinate")
    return summary, anchor


def validate_native_correctness(deployment, reference):
    """Reopen a correctness summary and every artifact required by the native gate."""
    return _correctness_proof(validate_deployment(deployment), reference)[0]


def validate_native_comparison_inputs(deployment, comparison):
    """Revalidate both hash-bound inputs, not just the selected timing arm."""
    d = validate_deployment(deployment)
    validate_native_comparison(comparison, evidence_scope=d["evidence_scope"])
    summaries, anchors = {}, {}
    for side in ("reference", "candidate"):
        inventory = comparison.get(side, {})
        run_ids, variants = inventory.get("run_ids"), inventory.get("variants")
        require(isinstance(run_ids, list) and len(run_ids) == 1 and isinstance(run_ids[0], str)
                and _RUN_ID.fullmatch(run_ids[0]) and isinstance(variants, list) and len(variants) == 1,
                "comparison side must identify one native correctness run")
        reference = {"path": str(Path(d["site"]["run_root"])/run_ids[0]/"summary.json"),
                     "sha256": comparison.get("summary_sha256", {}).get(side)}
        summary, anchor = _correctness_proof(d, reference)
        require(variants == [f"h100-{summary['placement']}"] and summary["run_id"] == run_ids[0],
                "comparison input arm/run differs from its correctness summary")
        summaries[side], anchors[side] = summary, anchor
    require(summaries["reference"]["placement"] != summaries["candidate"]["placement"],
            "comparison must bind two distinct native arms")
    a, b = summaries["reference"], summaries["candidate"]
    require({r["rank"]: r["cycle_census"] for r in a["ranks"]} ==
            {r["rank"]: r["cycle_census"] for r in b["ranks"]}, "comparison input trace/DiT census differs")
    for name in _STATE_COMPONENTS:
        first, second = (anchors[side]["components"][name]["numeric_sketch"] for side in ("reference", "candidate"))
        require(all(first[k] == second[k] for k in ("inventory_sha256", "inventory_tensor_count", "tensor_count", "coordinate_count"))
                and {(r["tensor"], r["flat_index"]) for r in first["coordinates"]} ==
                    {(r["tensor"], r["flat_index"]) for r in second["coordinates"]}, "comparison global state coordinates differ")
    return summaries


def validate_native_timing_prerequisites(deployment, correctness, comparison, *, placement):
    d = validate_deployment(deployment)
    correct, compared = _read_ref(correctness), _read_ref(comparison)
    _validate_evidence_scope(correct, d["evidence_scope"])
    require(correct.get("placement") == placement and correct.get("mode") == "correctness"
            and correct.get("status") == "PASS", "correctness arm/mode changed")
    summaries = validate_native_comparison_inputs(d, compared)
    matching = [side for side, summary in summaries.items() if summary["placement"] == placement]
    require(len(matching) == 1 and summaries[matching[0]] == correct
            and compared["summary_sha256"][matching[0]] == correctness["sha256"]
            and Path(correctness["path"]) == Path(d["site"]["run_root"])/correct["run_id"]/"summary.json",
            "comparison is for another native arm or correctness summary")
    return compared


def _binding_ref(value):
    require(isinstance(value, dict) and {"path", "sha256"} <= set(value), "missing native gate evidence reference")
    return {k: value[k] for k in ("path", "sha256")}


def _worker_argv(d, run_root, run_id, placement, correctness, comparison):
    a = Path(d["site"]["asset_root"])
    return [d["site"]["python"], "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
        str(Path(d["native_source"]["repo_root"])/WORKER), "--run-id", run_id,
        "--output-dir", str(run_root/"execution"), "--mode", "timing", "--placement", placement,
        "--expected-head", NATIVE_HEAD, "--timing-seed", "20260903", "--timing-arm-id", f"h100-{placement}",
        "--model-root", str(a/"wan13b/Wan2.1-T2V-1.3B-Diffusers-0fad780a"),
        "--bootstrap", str(a/"wan13b/systems_only_bootstrap_v3_head04e6e44"),
        "--expected-bootstrap-hash", SEMANTICS["bootstrap_hash"],
        "--trace-bundle", str(a/"m0/traces/syn_feature_trace_rank16_v4_strict_sync_steps20"),
        "--expected-trace-hash", SEMANTICS["trace_bundle_hash"], "--negative-prompt", str(a/"m1/negative_prompt_official_v1"),
        "--correctness-receipt", correctness["path"], "--expected-correctness-hash", correctness["sha256"],
        "--trajectory-comparison-receipt", comparison["path"], "--expected-trajectory-comparison-hash", comparison["sha256"]]


def _environment(d, placement):
    native, site = d["native_source"]["repo_root"], d["site"]
    env = {"DISTILLSCHEDULE_PYTHON": site["python"], "DISTILLSCHEDULE_HELIOS_ROOT": site["helios_root"],
        "DISTILLSCHEDULE_HARDWARE_PROFILE": "h100-8-b16", "PYTHONPATH": f"{native}/src:{native}:{site['helios_root']}",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "OMP_NUM_THREADS": "1",
        "HF_HOME": str(Path(site["asset_root"])/"hf_home_official_torch210"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "DISTILLSCHEDULE_WAN13B_EMA_IMPLEMENTATION": "reference",
        "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE": "0",
        "DISTILLSCHEDULE_WAN13B_LORA_OPERATOR_ATTRIBUTION": "0",
        "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE": "1" if placement == PLACEMENTS[2] else "0"}
    if placement == PLACEMENTS[2]:
        env["NCCL_ALGO"] = "Tree"
    return env


def _validate_observation(d, observation):
    require(isinstance(observation, dict), "native observation must be an object")
    required = {"placement", "run_root", "summary", "rank_receipts", "launcher_health", "launcher_argv",
                "launcher_environment", "launch_binding"}
    require(required <= set(observation) and set(observation) <= required | {"correctness", "comparison"},
            "unknown/missing native observation fields")
    placement = observation["placement"]
    root = _absolute(observation["run_root"], "run root")
    s = _read_ref(observation["summary"])
    normalized = validate_native_summary(s, expected_native_head=NATIVE_HEAD, placement=placement, allocation=d["allocation"])
    require(normalized["mode"] == "timing", "correctness diagnostic timing cannot calibrate costs")
    run_id = normalized["run_id"]
    require(root == Path(d["site"]["run_root"])/run_id, "observation is outside the selected native run root")
    fixed_paths = {"summary": root/"summary.json", "launcher_health": root/"run_status.json",
                   "launcher_argv": root/"manifests/command.sh", "launcher_environment": root/"manifests/environment.txt"}
    for key, path in fixed_paths.items():
        require(Path(observation[key]["path"]) == path, f"native raw path mismatch: {key}")
    require(Path(observation["launch_binding"]["path"]).is_relative_to(root/"manifests"), "launch binding is outside its native run")
    require(_read_ref(observation["launcher_health"]) == s["launcher_health"], "summary launcher health differs from raw health")
    refs = observation["rank_receipts"]
    require(isinstance(refs, list) and len(refs) == 8 and len({r["path"] for r in refs}) == 8,
            "eight distinct raw rank files required")
    raw_ranks = []
    for ref in refs:
        require(Path(ref["path"]).parent == root/"execution/ranks", "rank receipt outside its native run")
        raw_ranks.append(_read_ref(ref))
    require(sorted(raw_ranks, key=lambda r: r["rank"]) == sorted(s["ranks"], key=lambda r: r["rank"]),
            "summary ranks differ from actual raw rank files")
    correct_ref, comparison_ref = map(_binding_ref, (normalized["correctness_binding"], normalized["trajectory_comparison_binding"]))
    correct, compared = _read_ref(correct_ref), _read_ref(comparison_ref)
    correct_normal = validate_native_summary(correct, expected_native_head=NATIVE_HEAD, placement=placement, allocation=d["allocation"])
    require(correct_normal["mode"] == "correctness"
            and correct_normal["run_id"] == normalized["correctness_binding"]["run_id"], "wrong correctness trajectory")
    validate_native_comparison(compared)
    require(compared["status"] == normalized["trajectory_comparison_binding"]["status"],
            "raw comparison status differs from executed timing binding")
    side = normalized["trajectory_comparison_binding"].get("matched_side")
    require(side in ("reference", "candidate")
            and compared.get(side, {}).get("run_ids") == [correct_normal["run_id"]]
            and f"h100-{placement}" in compared.get(side, {}).get("variants", [])
            and f"h100-{placement}" not in compared.get("candidate" if side == "reference" else "reference", {}).get("variants", [])
            and compared.get("summary_sha256", {}).get(side) == correct_ref["sha256"],
            "comparison is for another native arm or correctness summary")
    for key, ref in (("correctness", correct_ref), ("comparison", comparison_ref)):
        require(key not in observation or observation[key] == ref, f"observation {key} differs from executed binding")
    argv = shlex.split(_read_ref(observation["launcher_argv"], text=True))
    require(argv == _worker_argv(d, root, run_id, placement, correct_ref, comparison_ref),
            "actual native worker argv differs from selected lowering")
    env = {}
    for line in _read_ref(observation["launcher_environment"], text=True).splitlines():
        require("=" in line, "invalid native environment manifest")
        k, v = line.split("=", 1)
        require(k not in env, "duplicate native environment key")
        env[k] = v
    expected_env = _environment(d, placement)
    require(env == expected_env, "actual native environment differs from selected lowering")
    launch = _read_ref(observation["launch_binding"])
    binding = {"schema_version": "distillschedule.hardware_wan_launch.v1", "deployment_sha256": fingerprint(d),
        "integration_head": d["binding"]["source_head"], "native_head": NATIVE_HEAD,
        "integration_clean": True, "native_clean": True, "allocation_sha256": fingerprint(d["allocation"]),
        "placement": placement, "run_id": run_id, "argv_sha256": observation["launcher_argv"]["sha256"],
        "environment_sha256": observation["launcher_environment"]["sha256"],
        "native_launcher_sha256": NATIVE_FILES["launcher_sha256"], "native_worker_sha256": NATIVE_FILES["worker_sha256"],
        "observed_devices": d["allocation"]["devices"]}
    require(all(launch.get(k) == v for k, v in binding.items()), "fresh launch source/allocation/argv/environment binding mismatch")
    started, completed = launch.get("started_unix_ns"), launch.get("completed_unix_ns")
    require(type(started) is int and type(completed) is int and 0 < started < completed, "invalid launch interval")
    require(all(started <= c["start_unix_ns"] < c["stop_unix_ns"] <= completed for r in raw_ranks for c in r["cycles"]),
            "historical native timing is outside the bound fresh launch interval")
    if d["evidence_scope"] == "MEASURED":
        require({"observed_hostname", "observed_environment", "native_worker_argv", "native_worker_environment"} <= set(launch),
                "MEASURED launch is missing actual host/environment/worker evidence")
    if "observed_hostname" in launch:
        require(all(r["hostname"] == launch["observed_hostname"] for r in raw_ranks), "observed launch hostname differs from ranks")
    for k, expected_value in (("observed_environment", d["environment"]), ("native_worker_argv", argv), ("native_worker_environment", env)):
        require(k not in launch or launch[k] == expected_value, f"launch {k} changed")
    _validate_evidence_scope(s, d["evidence_scope"])
    validate_native_timing_prerequisites(d, correct_ref, comparison_ref, placement=placement)
    return normalized


def validate_native_observation(deployment, observation):
    """Same raw-proof gate for calibration and the selected plan's actual run."""
    return _validate_observation(validate_deployment(deployment), observation)


def component_contract(deployment, placement: str, cycle: int) -> ComponentContract:
    d = validate_deployment(deployment)
    require(type(cycle) is int and cycle in (1, 2), "two-cycle component index required")
    return ComponentContract(FAMILY,
        work={"workload": WORKLOAD, "matched_contract": d["semantics"], "coupled_updates": "G1_C5_EMA5",
              "physical_microbatch": 1, "logical_slots": 16}, precision=d["precision"],
        placement={"choice": placement, "allocation_sha256": fingerprint(d["allocation"]), "world_size": 8},
        implementation={"adapter": ADAPTER, "native_source": d["native_source"], "arm": _arm(placement), "runtime": _runtime(placement)},
        context={"cycle": cycle, "entry": "LOADED_BOOTSTRAP" if cycle == 1 else "CONTINUOUS_AFTER_C1",
                 "resources": "ALL_EIGHT_GPUS_EXCLUSIVE", "memory_definition": MEMORY,
                 "overlap": "NATIVE_INTERNAL_ONLY_NO_DISCOUNT", "deployment_sha256": fingerprint(d)})


def _problem(d):
    a = _allocation(d["allocation"])
    ranks = tuple(range(8))
    ops = tuple(Operation(f"cycle{c}.native_cycle", (c-1)*5, "native_runtime", OperationKind.CRITIC_UPDATE,
        GradientMode.GRAD, required_versions={"completed_cycles": c-1}, produced_versions={"completed_cycles": c},
        metadata={"cycle": c, "scope": "NATIVE_COUPLED_CYCLE", "parameter_version_proof": "NATIVE_RANK_RECEIPTS"}) for c in (1, 2))
    graph = ExecutionGraph(ops, (Dependency(ops[0].operation_id, ops[1].operation_id, "same_native_process"),),
        {"adapter": ADAPTER, "workload": WORKLOAD, "initial_versions": {"completed_cycles": 0},
         "final_versions": {"completed_cycles": 2}, "cycle_ends": {str(c): ops[c-1].operation_id for c in (1, 2)},
         "memory_definition": MEMORY, "construction_peak": "NOT_MODELED", "independent_overlap_discounts": False})
    rules = tuple(BindingRule(ops[c-1].operation_id, {"placement": p}, component_contract(d, p, c).digest,
        ranks, tuple(f"gpu:{r}" for r in ranks), operation_contract_sha256=component_contract(d, p, c).digest)
        for p in PLACEMENTS for c in (1, 2))
    return PlanningProblem(WORKLOAD, d["binding"], graph, {"placement": PLACEMENTS}, rules, (), (), a.memory_limits,
        tuple({"placement": p} for p in PLACEMENTS), evidence_scope=d["evidence_scope"],
        lowering={"adapter": ADAPTER, "deployment_sha256": fingerprint(d), "native_head": NATIVE_HEAD,
                  "unsupported_workloads": ["W2", "W3", "W4", "W5", "W6"], "arbitrary_reorder_pipeline_kernel_search": False})


def _derive(d, placement, cycle, observations):
    require(isinstance(observations, list) and observations, "empty native calibration observations")
    require(all(o.get("placement") == placement for o in observations), "mixed native calibration placements")
    normalized = [_validate_observation(d, o) for o in observations]
    require(len({n["run_id"] for n in normalized}) == len(normalized), "duplicate calibration runs")
    return [n["cycle_ns"][cycle-1] for n in normalized], max(n["peak_reserved_bytes"] for n in normalized)


def validate_component_receipt(raw, cache=None) -> None:
    """CostModel provenance hook: re-open the real raw evidence, not its wrapper.

    ``cache`` is accepted for the common interface but deliberately not used:
    every validation observes current bytes, including mutations after a call.
    """
    require(raw.get("schema_version") == "distillschedule.operation_calibration.v1", "native cost receipt schema mismatch")
    d = validate_deployment(raw.get("deployment", {}))
    placement, cycle = raw.get("placement"), raw.get("cycle")
    contract = component_contract(d, placement, cycle)
    require(raw.get("component_contract") == contract.to_dict() and raw.get("key") == contract.digest
            and raw.get("operation_contract_sha256") == contract.digest, "native component contract changed")
    require(raw.get("binding") == d["binding"] and raw.get("evidence_scope") == d["evidence_scope"]
            and raw.get("status") == "PASS" and raw.get("profiled") is False and raw.get("kind") == "native_operation",
            "native component provenance changed")
    samples, memory = _derive(d, placement, cycle, raw.get("observations"))
    require(raw.get("samples_ns") == samples and raw.get("workspace_bytes_per_rank") == memory,
            "native component samples/memory differ from actual raw receipts")
    require(raw.get("claimable_speedup") is False and raw.get("strict_full_state_parity") == "NOT_RUN"
            and raw.get("numerical_status") == "REVIEW",
            "native diagnostic timing was promoted to strict equivalence")


def assemble_problem_and_costs(deployment, observations, *, output_dir: Path):
    """Create common IR and calibrated envelopes; all ranking remains model_search."""
    d = validate_deployment(deployment)
    require(isinstance(observations, list) and observations, "native observations required")
    require(len({o.get("run_root") for o in observations}) == len(observations), "duplicate native run observations")
    for o in observations:
        require(o.get("placement") in PLACEMENTS, "observation is for an unregistered placement")
    records = []
    output_dir = Path(output_dir).resolve()
    for placement in PLACEMENTS:
        rows = [o for o in observations if o["placement"] == placement]
        if not rows:
            continue  # The common planner reports UNCAlIBRATED; no fallback cost.
        for cycle in (1, 2):
            contract = component_contract(d, placement, cycle)
            samples, memory = _derive(d, placement, cycle, rows)
            raw = {"schema_version": "distillschedule.operation_calibration.v1", "binding": d["binding"],
                "key": contract.digest, "samples_ns": samples, "workspace_bytes_per_rank": memory,
                "status": "PASS", "profiled": False, "kind": "native_operation", "operation_contract_sha256": contract.digest,
                "component_contract": contract.to_dict(), "deployment": d, "placement": placement, "cycle": cycle,
                "observations": rows, "evidence_scope": d["evidence_scope"], "claimable_speedup": False,
                "strict_full_state_parity": "NOT_RUN", "numerical_status": "REVIEW"}
            path = output_dir/f"{contract.digest}.json"
            write_new_json(path, raw)
            evidence = {k: raw[k] for k in ("status", "profiled", "kind", "operation_contract_sha256", "component_contract")}
            evidence.update(source_receipt=str(path), source_receipt_sha256=file_sha256(path))
            records.append(OperationCost(contract.digest, tuple(samples), memory, evidence))
    return _problem(d), CostModel(d["binding"], tuple(records), d["evidence_scope"])


def _calibration_sources(raw, d):
    """Reuse the public flat-pool proof, then recover native launch observations."""
    if "component_sources" in raw:
        from .component_calibration import validate_pooled_receipt
        validate_pooled_receipt(raw)
        sources = [_read_ref(ref) for ref in raw["component_sources"]]
    else:
        sources = [raw]
    seen_runs = set()
    for source in sources:
        require("component_sources" not in source, "native component sources must be flat")
        validate_component_receipt(source)
        require(source["deployment"] == d, "pooled native launch belongs to a different deployment")
        for observation in source["observations"]:
            require(observation["run_root"] not in seen_runs, "pooled native costs double-count a calibration run")
            seen_runs.add(observation["run_root"])
    return sources


def _validate_model(problem, costs, d):
    require(problem.to_dict() == _problem(d).to_dict(), "saved native graph/domains/lowering changed")
    costs.bind(problem.binding, problem_scope=problem.evidence_scope)
    require(costs.scope == d["evidence_scope"], "cost evidence scope changed")
    allowed = {component_contract(d, p, c).digest for p in PLACEMENTS for c in (1, 2)}
    for record in costs.records:
        require(record.key in allowed, "cost is not an admitted native component")
        raw = _read_ref({"path": record.evidence.get("source_receipt"), "sha256": record.evidence.get("source_receipt_sha256")})
        _calibration_sources(raw, d)
        require(raw["samples_ns"] == list(record.samples_ns)
                and raw["workspace_bytes_per_rank"] == record.workspace_bytes_per_rank
                and raw["component_contract"] == record.evidence.get("component_contract"), "saved cost differs from source receipt")
    costs.verify_receipts()


def _selected_observations(costs, d, placement):
    contract = component_contract(d, placement, 2)
    record = costs.get_component(contract)
    raw = _read_ref({"path": record.evidence["source_receipt"], "sha256": record.evidence["source_receipt_sha256"]})
    return [observation for source in _calibration_sources(raw, d) for observation in source["observations"]]


def lower_selection(result, problem, costs, deployment, *, run_id: str):
    d = validate_deployment(deployment)
    _validate_model(problem, costs, d)
    actual = validate_saved_prediction(result, problem, costs)
    require(actual["status"] == "PREDICTED_FEASIBLE", "no feasible calibrated native selection")
    placement = actual["assignment"]["placement"]
    observations = _selected_observations(costs, d, placement)
    require(isinstance(run_id, str) and _RUN_ID.fullmatch(run_id)
            and all(Path(o["run_root"]).name != run_id for o in observations), "selected execution needs a fresh run id")
    plan = {"schema_version": "distillschedule.hardware_wan_plan.v1", "adapter": ADAPTER,
        "deployment": d, "deployment_sha256": fingerprint(d), "run_id": run_id, "selected_placement": placement,
        "allocation": d["allocation"], "model_plan": {"problem": problem.to_dict(), "cost_model": costs.to_dict(), "search_result": result},
        "selected_observations": observations, "claimable_speedup": False, "strict_full_state_parity": "NOT_RUN",
        "numerical_status": "REVIEW", "evidence_scope": d["evidence_scope"], "executable": d["evidence_scope"] == "MEASURED"}
    plan["plan_sha256"] = fingerprint(plan)
    return _json(plan)


def validate_lowered_plan(plan):
    require(isinstance(plan, Mapping) and plan.get("schema_version") == "distillschedule.hardware_wan_plan.v1"
            and plan.get("adapter") == ADAPTER, "unknown native saved plan")
    require(plan.get("plan_sha256") == fingerprint({k: v for k, v in plan.items() if k != "plan_sha256"}),
            "native saved plan hash changed")
    model = plan["model_plan"]
    expected = lower_selection(model["search_result"], PlanningProblem.from_dict(model["problem"]),
        CostModel.from_dict(model["cost_model"]), plan["deployment"], run_id=plan["run_id"])
    require(dict(plan) == expected, "saved placement/allocation/observations differ from the validated model selection")
    return expected
