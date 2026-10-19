"""Four-cycle collector. Native and CPU-mock evidence never share eligibility."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
from typing import Any, Mapping

from . import livewan_system_factorial_r2 as r2
from . import livewan_factorial_r3 as r3
from .livewan_crossscale_worker import FormalTrace


def _parent_helpers():
    path = Path(__file__).resolve().parents[3] / "scripts/distillschedule/summarize_livewan_two_cycle_timing_world16.py"
    spec = importlib.util.spec_from_file_location("_livewan_r2_parent_collector_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mechanism_errors(row: dict[str, Any], *, arm: str, rank: int) -> list[str]:
    """Reuse original receipt checkers; no count is filled from a plan flag."""
    errors: list[str] = []
    if arm in r3.ARMS:
        errors += r3.mechanism_errors(row, arm=arm, rank=rank)
        if arm == r3.B_ARM:
            return errors
    keys = ("critic_lora_kernel", "score_modulation_kernel", "score_rmsnorm_kernel",
            "score_rope_kernel", "score_attention_backend", "score_residual_gate", "score_layernorm",
            "generator_cross_attention_kv_cache", "generator_attention_backend",
            "generator_rollout_cast_cache", "generator_rms_epilogue", "generator_modulation",
            "generator_gate", "critic_attention_checkpoint")
    if arm == r2.COMPATIBLE_ARM:
        errors += r2.compatible_mechanism_errors(row)
        errors.extend(f"unrequested A-dependent B mechanism executed: {key}" for key in keys
                      if key not in r2.COMPATIBLE_OPERATORS and row.get(key, {}).get("enabled") is True)
        return errors
    if arm != "full":
        errors.extend(f"unrequested B mechanism executed: {key}" for key in keys
                      if row.get(key, {}).get("enabled") is True)
        if arm in r2.A_ARMS and rank >= 8:
            audit = row.get("system_factorial_recompute", {})
            calls = int(row.get("physical_call_inventory", {}).get("critic/grad/checkpoint", -1)) * 30
            if (audit.get("checkpoint_calls") != calls or audit.get("function_entries") != 2 * calls
                    or audit.get("replay_entries") != calls or audit.get("cache_reads") != 0
                    or audit.get("saved_tensor_policy") != "ORIGINAL" or audit.get("rng_policy") != "ORIGINAL"):
                errors.append("native Critic checkpoint recompute census changed")
        return errors
    from .livewan_score_fa3 import backend_receipt_errors, runtime_receipt_errors as sac_errors
    from .livewan_score_rmsnorm import runtime_receipt_errors as score_rms_errors
    from .livewan_score_gate import runtime_receipt_errors as score_gate_errors
    from .livewan_score_layernorm import runtime_receipt_errors as score_ln_errors
    from .livewan_rollout_cast_cache import runtime_receipt_errors as cast_errors
    from .livewan_generator_rms_epilogue import runtime_receipt_errors as rms_errors
    from .livewan_generator_modulation import runtime_receipt_errors as mod_errors
    from .livewan_generator_gate import runtime_receipt_errors as gate_errors
    from .livewan_generator_splitkv import runtime_receipt_errors as split_errors
    inventory = row.get("physical_call_inventory", {})
    errors += backend_receipt_errors(row.get("score_attention_backend", {}), rank=rank)
    errors += score_rms_errors(row.get("score_rmsnorm_kernel", {}), rank=rank)
    errors += score_gate_errors(row.get("score_residual_gate", {}), rank=rank, inventory=inventory)
    errors += score_ln_errors(row.get("score_layernorm", {}), rank=rank, inventory=inventory)
    errors += sac_errors(row.get("critic_attention_checkpoint", {}), rank=rank,
                         expected_calls=int(inventory.get("critic/grad/checkpoint", 0)) * 30)
    errors += cast_errors(row.get("generator_rollout_cast_cache", {}), rank=rank, cycles=4)
    errors += rms_errors(row.get("generator_rms_epilogue", {}), rank=rank, cycles=4, fuse_cast_square=False)
    errors += mod_errors(row.get("generator_modulation", {}), rank=rank, cycles=4)
    errors += gate_errors(row.get("generator_gate", {}), rank=rank, cycles=4, output_layout="row_major")
    attention = row.get("generator_attention_backend", {})
    errors += split_errors(attention.get("readonly_splitkv", {}), rank=rank,
                           total_calls=int(attention.get("invocation_count", 0)))
    return errors


def capacity_optimizer_errors(row: Mapping[str, Any], *, arm: str, rank: int) -> list[str]:
    build = row.get("build", {})
    observed = build.get("capacity_offload", {})
    if arm != r2.CAPACITY_ARM:
        return ["unrequested capacity offload executed"] if observed else []
    errors = []
    if any(observed.get(k) != v for k, v in r2.CAPACITY_POLICY.items()):
        errors.append("capacity offload policy receipt changed")
    if rank >= 8:
        if observed.get("execution_status") != "NOT_APPLICABLE_SCORE_RANK" or build.get("critic", {}).get("cpu_optimizer"):
            errors.append("capacity offload must not affect Critic/Teacher")
        return errors
    opt = build.get("student", {}).get("cpu_optimizer", {})
    numel = r2.CAPACITY_POLICY["generator_numel"]
    expected = dict(name="replicated_cpu_foreach_adamw", state_device="cpu", steps=4,
        moment_bytes=8 * numel, cpu_parameter_mirror_bytes=4 * numel,
        cpu_gradient_buffer_bytes=4 * numel, pinned_parameter_mirror_bytes=4 * numel,
        pinned_gradient_buffer_bytes=4 * numel, cpu_consumption_fences=4,
        transfers_and_cpu_update_inside_timer=True, gradient_mean_and_clip_unchanged=True,
        original_parameter_dtype_preserved=True)
    if observed.get("execution_status") != "PASS" or not str(opt.get("execution_parameter_device", "")).startswith("cuda"):
        errors.append("Generator CPU optimizer execution not observed")
    for k, v in expected.items():
        if opt.get(k) != v:errors.append("capacity optimizer receipt mismatch: " + k)
    for left, right in (("gradient_d2h_bytes", "updated_parameter_h2d_bytes"),
                        ("gradient_d2h_events", "updated_parameter_h2d_events")):
        if type(opt.get(left)) is not int or opt[left] <= 0 or opt.get(right) != opt[left]:
            errors.append("capacity optimizer transfer census changed: " + left)
    return errors


def collect(run_root: str | Path, *, arm: str, expected_head: str, trace: FormalTrace,
            allow_synthetic: bool = False) -> dict[str, Any]:
    r2.require(arm in r2.ARMS, "only_B BLOCKED or unknown arm")
    r2.require(trace.cycles == 4 and trace.global_batch == 16, "wrong trace geometry/cycles")
    root = Path(run_root)
    helper = _parent_helpers()
    mode = r2.mode_for_arm(arm)
    errors: list[str] = []
    plans = []
    for node in (0, 1):
        plan, sha = helper._effective_plan(root / f"effective_plan.node{node}.json")
        binding = plan.get("metadata", {}).get(r2.KEY, {})
        r2.require(binding.get("schema_version") == r2.SCHEMA and binding.get("arm") == arm
                   and binding.get("source_head") == expected_head and binding.get("seed") == r2.SEED,
                   "node arm/source/seed binding changed")
        r2.require(binding.get("cycles") == 4 and binding.get("measured_cycles") == [2, 3, 4],
                   "node cycle contract changed")
        plans.append((plan, sha))
    r2.require(plans[0] == plans[1], "node effective plans differ")
    plan, plan_sha = plans[0]
    expected_plan = r2.make_plan(Path(__file__).resolve().parents[3], arm=arm,
                                source_head=expected_head)
    r2.require(plan == expected_plan, "effective plan differs from the complete frozen arm")
    r2.require(Path(plan["evidence"]["output_dir"]).resolve() == root.resolve()
               and Path(plan["evidence"]["operation_trace"]).resolve() == trace.path.resolve(),
               "producer/collector output or trace path mismatch")
    rows = []
    for rank in range(16):
        path = root / "receipts" / f"rank_{rank:03d}.json"
        r2.require(path.is_file(), f"missing rank{rank}")
        row = json.loads(path.read_text())
        timing = row.get("system_factorial_timing", {})
        r2.require(row.get("run_id") == plan["evidence"]["run_id"], "rank run identity changed")
        r2.require(row.get("rank") == rank and row.get("status") == "PASS"
                   and row.get("run_kind") == r2.RUN_KIND and row.get("mode") == mode,
                   f"rank{rank} identity/status changed")
        r2.require(row.get("head") == expected_head and timing.get("source_head") == expected_head
                   and timing.get("arm") == arm and timing.get("seed") == r2.SEED,
                   f"rank{rank} source/arm/seed changed")
        r2.require(row.get("effective_plan_sha256") == timing.get("effective_plan_sha256") == plan_sha,
                   f"rank{rank} effective plan hash changed")
        r2.require(row.get("trace_sha256") == trace.sha256 and row.get("profiled") is False
                   and timing.get("profiled") is False, f"rank{rank} trace/profile changed")
        r2.require(timing.get("schema_version") == r2.SCHEMA
                   and timing.get("extraction") == "AFTER_ALL_FOUR_CUDA_COMPLETION_ENDPOINTS"
                   and timing.get("midpoint_synchronize") is False and timing.get("midpoint_barrier") is False,
                   f"rank{rank} timer boundary changed")
        synthetic = timing.get("evidence_kind") == "SYNTHETIC_TEST_ONLY"
        r2.require(timing.get("evidence_kind") == "NATIVE" or (allow_synthetic and synthetic),
                   "CPU mock evidence is not native evidence")
        r2.require(row.get("final_versions") == dict(generator=4, critic=20, ema=0), "final version mismatch")
        r2.require(row.get("logical_generator_updates") == 4 and row.get("logical_critic_updates") == 20,
                   "logical update count changed")
        expected_commits = (4, 20) if mode == "baseline" else (4, 0) if rank < 8 else (0, 20)
        r2.require((row.get("physical_generator_commits"), row.get("physical_critic_commits")) == expected_commits,
                   "physical commit count changed")
        finite = row.get("system_factorial_c1_finite", {})
        r2.require(finite.get("finite") is True and finite.get("count") ==
                   (6 if mode == "baseline" else 1 if rank < 8 else 5), "C1 observed finite gate failed")
        for role in (("generator", "critic") if mode == "baseline" else ("generator",) if rank < 8 else ("critic",)):
            metrics = row.get(f"{role}_state_metrics", {})
            r2.require(metrics and all(math.isfinite(float(x)) for x in metrics.values()), "nonfinite final state")
        for key in ("oom", "oom_kill"):
            r2.require(key in row.get("cgroup_oom_before", {}) and key in row.get("cgroup_oom_after", {})
                       and row["cgroup_oom_before"][key] == row["cgroup_oom_after"][key], "cgroup OOM changed/unknown")
        cycles = timing.get("cycles", [])
        r2.require(len(cycles) == 4 and [x["cycle"] for x in cycles] == [1, 2, 3, 4], "missing cycle")
        previous = 0.0
        for index, cycle in enumerate(cycles, 1):
            offset = float(cycle["completion_offset_seconds"])
            local = float(cycle["local_endpoint_delta_seconds"])
            expected = (dict(generator=index, critic=5 * index, ema=0) if mode == "baseline" else
                        dict(generator=index, critic=0, ema=0) if rank < 8 else dict(generator=0, critic=5 * index, ema=0))
            r2.require(math.isfinite(offset) and offset > previous and abs(offset - previous - local) < 1e-6,
                       "nonfinite/reordered endpoint")
            r2.require(cycle.get("local_versions") == expected, "per-cycle role version mismatch")
            previous = offset
        errors += [f"rank{rank}: {e}" for e in mechanism_errors(row, arm=arm, rank=rank)]
        errors += [f"rank{rank}: {e}" for e in capacity_optimizer_errors(row, arm=arm, rank=rank)]
        if mode == "pipeline":
            continuous = row.get("continuous_pipeline", {})
            hot = row.get("critic_shadow_hot_sync", {})
            transport = row.get("generator_runahead_transport", {})
            r2.require(continuous.get("name") == r2.CONTINUOUS_NAME and continuous.get("cycles") == 4
                       and continuous.get("lookahead_depth") == 1 and continuous.get("execution_status") == "PASS",
                       "complete A continuous pipeline missing")
            r2.require(hot.get("execution_status") == "PASS"
                       and [x.get("version") for x in hot.get("events", [])] == [5, 10, 15], "hot-sync missing")
            r2.require(transport.get("execution_status") == "PASS" and transport.get("cycles") == 4
                       and transport.get("cross_cycle_lookahead") is True, "host runahead missing")
        rows.append(row)
    identity = {(x.get("bootstrap_sha256"), x.get("source_receipt_sha256")) for x in rows}
    r2.require(len(identity) == 1 and all(identity_value for identity_value in next(iter(identity))), "asset identity mismatch")
    # Reuse the unchanged parent trace-derived logical/physical census formulas.
    for cycle in range(4):
        logical, physical = helper._expected_inventory(SimpleNamespace(
            rows=[x for x in trace.rows if x.cycle_index == cycle], cycles=1), mode=mode)
        deltas = [x["system_factorial_timing"]["cycles"][cycle]["observed_call_deltas"] for x in rows]
        for key, expected in (("logical_call_inventory", logical), ("physical_call_inventory", physical)):
            r2.require(helper._core(helper._aggregate(deltas, key)) == expected, f"cycle{cycle + 1} observed census changed")
    critical = rows if mode == "baseline" else rows[8:]
    endpoints = [max(x["system_factorial_timing"]["cycles"][i]["completion_offset_seconds"]
                     for x in critical) for i in range(4)]
    walls = [v - (endpoints[i - 1] if i else 0.0) for i, v in enumerate(endpoints)]
    evidence_kinds = {x["system_factorial_timing"]["evidence_kind"] for x in rows}
    r2.require(len(evidence_kinds) == 1, "mixed native and CPU evidence")
    synthetic = evidence_kinds != {"NATIVE"}
    for row in rows:
        memory = row.get("peak_memory", {})
        r2.require(all(type(memory.get(k)) is int and memory[k] >= 0 for k in
                       ("max_allocated_bytes", "max_reserved_bytes", "driver_free_bytes")),
                   "missing observed memory receipt")
    if not synthetic:
        for node in (0, 1):
            folder = root / "launcher" / f"node_{node}"
            manifest = helper._manifest(folder / "launch_manifest.txt")
            r2.require(manifest.get("head") == expected_head and manifest.get("run_kind") == r2.RUN_KIND
                       and manifest.get("system_factorial_arm") == arm, "launcher source/arm mismatch")
            state = helper._status(folder / "teardown_status.txt", owner_run_id=manifest["run_id"])
            r2.require(state.get("status") == "PASS" and state.get("gpu_process_count") == "0", "owned cleanup incomplete")
            before = helper._memory_events(folder / "cgroup_memory_events_before.txt")
            after = helper._memory_events(folder / "cgroup_memory_events_after.txt")
            r2.require(all(k in before and k in after and before[k] == after[k] for k in ("oom", "oom_kill")),
                       "node cgroup OOM changed/unknown")
    return dict(schema_version=r2.SCHEMA, status=("FAIL" if errors else "PASS_CPU_MOCK_ONLY" if synthetic else "PASS"),
                arm=arm, source_head=expected_head, effective_plan_sha256=plan_sha,
                rank_count=16, cycles_in_same_process=4, independent_initializations=1,
                evidence_kind="SYNTHETIC_TEST_ONLY" if synthetic else "NATIVE",
                cycle_walls_seconds=None if synthetic else walls,
                cycle_2_seconds=None if synthetic else walls[1],
                C2_C4_mean_seconds=None if synthetic else statistics.mean(walls[1:]),
                cpu_mock_endpoint_arithmetic=walls if synthetic else None,
                observed_peak_allocated_bytes=None if synthetic else max(x["peak_memory"]["max_allocated_bytes"] for x in rows),
                observed_peak_reserved_bytes=None if synthetic else max(x["peak_memory"]["max_reserved_bytes"] for x in rows),
                final_boundary_driver_free_min_bytes=None if synthetic else min(x["peak_memory"]["driver_free_bytes"] for x in rows),
                driver_free_sampling_scope="FINAL_BOUNDARY_ONLY_NOT_CYCLE_MINIMUM",
                numerical_status="NOT_ESTABLISHED" if synthetic else "REVIEW_NOT_STRICT_EQUIVALENCE",
                formal_timing_eligible=False, native_descriptive_timing_eligible=not synthetic and not errors,
                C1_semantic_contract="CPU_MOCK_ONLY" if synthetic else "PASS_DEFERRED_OBSERVED_CHECKS",
                errors=errors, profiler=False)
