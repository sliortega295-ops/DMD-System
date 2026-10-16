"""Bounded A1008/A10016 metadata -> common planner -> native worker lowering.

No launcher, CUDA import, hardware inference, H100 admission, or timing claims.
Correctness-mode native C2 timers remain diagnostic even with MEASURED costs.
Run validation beside the immutable native receipts, normally on their host.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType

from distillschedule.core.enums import GradientMode, OperationKind
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, write_new_json
from .allocation import ExecutionAllocation
from .contracts import finite_number, require
from .cost_model import ComponentContract, CostModel, OperationCost
from .hardware_wan import SEMANTICS, WORKER, _arm, _runtime, _CYCLE_CONTRACT
from .model_search import search_plans, validate_saved_prediction
from .plan_ir import BindingRule, PlanningProblem, validate_binding

ADAPTER = "a100_allocation_coupled_v1"
NATIVE_HEAD = "8148e84de048820df32f0a12b5657f1d9f530323"
NATIVE_TREE = "83fbc856430dc7b7ab34268ef69bf4754296e482"
NATIVE_FILES = {
    WORKER: "5435559802afbb662c0f98fa80896dff145d7f1c1a6cf48cb459404559a0dcb8",
    "src/distillschedule/runtime/hardware_profile.py": "75ad34d39aa009fb36ebbeafa3ae08a80d93eb13afa6897accc88b8da2d3a3b1",
    "src/distillschedule/runtime/wan13b_allocation.py": "9313b11c33de5d768d6e7e1591fe08c6949e7b951488575db0e5cbdb559e301e",
}
PLACEMENTS = ("world16-fsdp", "resident")
MEMORY = "MAX_RANK_POST_SETUP_RESERVED_PEAK_AS_WHOLE_CYCLE_ENVELOPE"
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_EXTRA_ENV = {"LD_LIBRARY_PATH", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_IB_DISABLE"}


def _same(value, expected, label):
    require(fingerprint({k: value.get(k) for k in expected}) == fingerprint(expected), label)


def _ref(ref):
    require(set(ref) == {"path", "sha256"} and Path(ref["path"]).is_absolute(), "absolute hashed reference required")
    require(file_sha256(Path(ref["path"])) == ref["sha256"], "immutable receipt hash changed")
    return json.loads(Path(ref["path"]).read_text())


def _scope(value, scope):
    if isinstance(value, dict):
        require(scope == "CPU_FIXTURE_ONLY" or all(value.get(k) != "CPU_FIXTURE_ONLY" for k in ("fixture_scope", "evidence_scope")),
                "CPU fixture evidence cannot be promoted to MEASURED")
        for item in value.values(): _scope(item, scope)
    elif isinstance(value, list):
        for item in value: _scope(item, scope)


def _allocation(d):
    a = ExecutionAllocation.from_dict(d["allocation"])
    require(d["profile"] in {"a100-8-b16", "a100-16-b16"} and d["profile"] == f"a100-{a.world_size}-b16",
            "only explicit A1008/A10016 profiles are supported")
    require(a.world_size in (8, 16) and a.node_count == a.world_size // 8 and a.logical_global_batch == 16
            and all(x.architecture == "sm80" and x.node == x.rank // 8 and x.local_rank == x.rank % 8 for x in a.devices),
            "A100 allocation requires ordered nodes of eight sm80 devices and full logical16")
    require(set(a.role_slots) == {"generator", "critic"} and all(rows == tuple(tuple(range(r, 16, a.world_size))
            for r in range(a.world_size)) for rows in a.role_slots.values()), "independent B1 slot mapping changed")
    return a


def make_deployment(*, profile, allocation, native_root, environment, assets, site, evidence_scope="MEASURED"):
    """Assemble only supplied, observed metadata; never infer a GPU allocation."""
    d = {"schema_version": "distillschedule.a100_allocation_deployment.v1", "profile": profile, "allocation": allocation,
        "native_source": {"repo_root": native_root, "head": NATIVE_HEAD, "tree": NATIVE_TREE, "files": NATIVE_FILES},
        "environment": environment, "assets": assets, "site": site, "semantics": SEMANTICS,
        "precision": {"frozen_base": "torch.bfloat16", "trainable_lora": "torch.float32", "ema_implementation": "reference",
                      "gradient_reduction": "SUM_THEN_DIVIDE_PHYSICAL_WORLD_SIZE", "loss_scale_per_logical_b1": len(allocation["devices"]) / 16},
        "evidence_scope": evidence_scope}
    d["binding"] = {"source_head": NATIVE_HEAD, **{f"{key}_fingerprint": fingerprint(d[field]) for key, field in
        (("hardware", "allocation"), ("environment", "environment"), ("asset", "assets"), ("semantic", "semantics"), ("precision", "precision"))}}
    return validate_deployment(d)


def validate_deployment(value):
    d = json.loads(json.dumps(value, allow_nan=False))
    require(set(d) == {"schema_version", "profile", "allocation", "native_source", "environment", "assets", "site", "semantics", "precision", "evidence_scope", "binding"}, "deployment fields changed")
    require(d["schema_version"] == "distillschedule.a100_allocation_deployment.v1" and d["evidence_scope"] in ("MEASURED", "CPU_FIXTURE_ONLY"), "new A100 deployment schema/scope required")
    a = _allocation(d)
    require(isinstance(d["native_source"], dict) and d["native_source"] == {"repo_root": d["native_source"].get("repo_root"), "head": NATIVE_HEAD, "tree": NATIVE_TREE, "files": NATIVE_FILES}, "unregistered native source")
    require(d["semantics"] == SEMANTICS and d["precision"] == {"frozen_base": "torch.bfloat16", "trainable_lora": "torch.float32", "ema_implementation": "reference",
        "gradient_reduction": "SUM_THEN_DIVIDE_PHYSICAL_WORLD_SIZE", "loss_scale_per_logical_b1": a.world_size / 16}, "frozen workload/precision changed")
    require(isinstance(d["environment"], dict) and all(isinstance(d["environment"].get(k), str) and d["environment"][k] for k in ("python", "torch"))
            and re.fullmatch(r"\d+(?:\.\d+)+", d["environment"].get("driver_version", "")), "observed Python/Torch/driver identity required")
    require(all(d["assets"].get(k) == SEMANTICS[k] for k in ("bootstrap_hash", "trace_bundle_hash", "model_identity_sha256", "negative_prompt_manifest_hash")), "old immutable asset identities required")
    site = d["site"]
    require(set(site) - {"launch_environment"} == {"python", "helios_root", "asset_root", "run_root", "master_addr", "master_port", "node_hosts"}, "site fields changed")
    require(len(site["node_hosts"]) == a.node_count and len(set(site["node_hosts"])) == a.node_count and all(isinstance(x, str) and x for x in site["node_hosts"]), "explicit node hostnames required")
    require(isinstance(site["master_addr"], str) and site["master_addr"] and type(site["master_port"]) is int and 0 < site["master_port"] < 65536, "invalid rendezvous metadata")
    require(set(site.get("launch_environment", {})) <= _EXTRA_ENV and all(isinstance(v, str) for v in site.get("launch_environment", {}).values()), "unregistered extra environment knob")
    require(all(isinstance(p, str) and Path(p).is_absolute() and ".." not in Path(p).parts for p in
        [d["native_source"]["repo_root"], *(site[k] for k in ("python", "helios_root", "asset_root", "run_root"))]), "absolute normalized native/site paths required")
    validate_binding(d["binding"])
    require(d["binding"]["source_head"] == NATIVE_HEAD, "stale source binding")
    for key, field in (("hardware", "allocation"), ("environment", "environment"), ("asset", "assets"), ("semantic", "semantics"), ("precision", "precision")):
        require(d["binding"][f"{key}_fingerprint"] == fingerprint(d[field]), f"deployment {key} identity changed")
    _scope(d, d["evidence_scope"])
    return d


def _native_helper(d):
    """Load only the two content-pinned, stdlib-only native receipt helpers."""
    root = Path(d["native_source"]["repo_root"])
    require(all(file_sha256(root / path) == sha for path, sha in NATIVE_FILES.items()), "native helper/worker source bytes changed")
    package = "_distillschedule_a100_allocation_native"
    if package not in sys.modules:
        module = ModuleType(package)
        module.__path__ = [str(root / "src/distillschedule/runtime")]
        sys.modules[package] = module
        for name in ("hardware_profile", "wan13b_allocation"):
            spec = importlib.util.spec_from_file_location(f"{package}.{name}", root / f"src/distillschedule/runtime/{name}.py")
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = loaded
            spec.loader.exec_module(loaded)
    return sys.modules[f"{package}.wan13b_allocation"]


def _expected_receipt(d, rank):
    helper = _native_helper(d)
    slots = tuple(range(rank, 16, len(d["allocation"]["devices"])))
    raw = [{"logical_samples": [{"logical_slot": slot, "generator": {"source_rank": slot, "optimizer_committed": i == len(slots) - 1,
        "physical_generator_step_after": 2 + cycle + int(i == len(slots) - 1)}, "critic": [{"source_rank": slot,
        "logical_step": 10 + step, "physical_critic_step_after": 10 + 5 * cycle + step + int(i == len(slots) - 1)} for step in range(5)]}
        for i, slot in enumerate(slots)]} for cycle in range(2)]
    return helper.build_rank_allocation_receipt(profile_name=d["profile"], physical_rank=rank, observed_world_size=len(d["allocation"]["devices"]),
        raw_cycles=raw, initial_versions={"generator": 2, "critic": 10, "ema": 10}, cycle_versions=[{"generator": 3, "critic": 15, "ema": 15}, {"generator": 4, "critic": 20, "ema": 20}],
        observed_step_calls={"generator": 2, "critic": 10}, observed_ema_updates=10)


def command_lowering(deployment, *, run_id, placement):
    """Return data only: exact original-worker commands per explicitly bound node."""
    d = validate_deployment(deployment)
    require(placement in PLACEMENTS and isinstance(run_id, str) and _RUN.fullmatch(run_id), "invalid placement/run id")
    a, site, native = _allocation(d), d["site"], d["native_source"]["repo_root"]
    assets, root = Path(site["asset_root"]), Path(site["run_root"]) / run_id
    worker = [str(Path(native) / WORKER), "--run-id", run_id, "--output-dir", str(root / "execution"), "--mode", "correctness",
        "--placement", placement, "--expected-head", NATIVE_HEAD, "--timing-seed", str(SEMANTICS["seed"]), "--timing-arm-id", f"{d['profile']}-{placement}",
        "--model-root", str(assets / "wan13b/Wan2.1-T2V-1.3B-Diffusers-0fad780a"), "--bootstrap", str(assets / "wan13b/systems_only_bootstrap_v3_head04e6e44"),
        "--expected-bootstrap-hash", SEMANTICS["bootstrap_hash"], "--trace-bundle", str(assets / "m0/traces/syn_feature_trace_rank16_v4_strict_sync_steps20"),
        "--expected-trace-hash", SEMANTICS["trace_bundle_hash"], "--negative-prompt", str(assets / "m1/negative_prompt_official_v1")]
    env = {"PYTHONPATH": f"{native}/src:{native}:{site['helios_root']}", "OMP_NUM_THREADS": "1", "HF_HOME": str(assets / "hf_home_official_torch210"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "DISTILLSCHEDULE_HARDWARE_PROFILE": d["profile"],
        "DISTILLSCHEDULE_WAN13B_EMA_IMPLEMENTATION": "reference", "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE": "0",
        "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE": "0", "DISTILLSCHEDULE_WAN13B_LORA_OPERATOR_ATTRIBUTION": "0",
        "DISTILLSCHEDULE_STATIC_FSDP2_NSYS_EXACT_RANGE": "0", **site.get("launch_environment", {})}
    commands = []
    for node in range(a.node_count):
        distributed = ["--standalone", "--nproc-per-node=8"] if a.node_count == 1 else ["--nnodes=2", "--nproc-per-node=8", f"--node-rank={node}", f"--master-addr={site['master_addr']}", f"--master-port={site['master_port']}"]
        commands.append({"node_rank": node, "hostname": site["node_hosts"][node], "cwd": native,
            "argv": [site["python"], "-m", "torch.distributed.run", *distributed, *worker],
            "environment": {**env, "CUDA_VISIBLE_DEVICES": ",".join(x.uuid for x in a.devices if x.node == node)},
            "clear_environment_prefixes": ["DISTILLSCHEDULE_", "NCCL_"]})
    return commands


def validate_native_observation(deployment, observation):
    d = validate_deployment(deployment)
    require(set(observation) == {"placement", "summary", "rank_receipts", "launcher_health", "allocation_manifest"}, "observation fields changed")
    s, launch, health = (_ref(observation[k]) for k in ("summary", "allocation_manifest", "launcher_health"))
    a, placement, run_id = _allocation(d), observation["placement"], s.get("run_id")
    commands = command_lowering(d, run_id=run_id, placement=placement)
    root = Path(d["site"]["run_root"]) / run_id
    for key, path in (("summary", root / "summary.json"), ("allocation_manifest", root / "allocation-manifest.json"), ("launcher_health", root / "run_status.json")):
        require(Path(observation[key]["path"]) == path, "observation outside selected run root")
    _same(launch, {"schema_version": "distillschedule.a100_allocation_launch.v1", "evidence_scope": d["evidence_scope"],
        "deployment_sha256": fingerprint(d), "profile": d["profile"], "source_head": NATIVE_HEAD, "source_tree": NATIVE_TREE, "source_clean": True,
        "native_files": NATIVE_FILES, "allocation": d["allocation"], "observed_environment": d["environment"], "asset_identities": d["assets"],
        "run_id": run_id, "placement": placement, "mode": "correctness"}, "observed launch identity changed")
    nodes = launch.get("nodes", [])
    require(len(nodes) == a.node_count, "missing node lifecycle receipt")
    for actual, expected in zip(nodes, commands, strict=True):
        _same(actual, {**expected, "exit_code": 0, "owned_cleanup": True, "gpu_processes_after": []}, "actual worker argv/environment/cleanup differs from lowering")
        for key in ("oom", "oom_kill"):
            before, after = actual.get("cgroup_before", {}).get(key), actual.get("cgroup_after", {}).get(key)
            require(type(before) is int and before >= 0 and type(after) is int and before == after, "node cgroup OOM counter changed/missing")
    _same(health, {"status": "PASS", "exit_code": 0, "expected_head": NATIVE_HEAD, "cgroup_oom_delta": {"oom": 0, "oom_kill": 0}, "gpu_processes_after": []}, "native lifecycle failed")
    require(all(type(health.get("cgroup_before", {}).get(k)) is int and health["cgroup_before"][k] >= 0 and health["cgroup_before"][k] == health.get("cgroup_after", {}).get(k)
                for k in ("oom", "oom_kill")), "raw launcher cgroup counters disagree")
    _same(s, {"schema_version": "distillschedule.wan13b_trajectory_summary.v1", "status": "PASS", "errors": [], "mode": "correctness",
        "expected_head": NATIVE_HEAD, "placement": placement, "rank_count": a.world_size, "hardware_profile": d["profile"], "logical_world_size": 16,
        "cycles": 2, "logical_slots": [slot for r in range(a.world_size) for slot in range(r, 16, a.world_size)],
        "final_versions": {"generator": 4, "critic": 20, "ema": 20}, "matched_contract": SEMANTICS, "workload_fingerprint": fingerprint(SEMANTICS),
        "diagnostic_only": True, "formal_timing_eligible": False, "launcher_health": health}, "native summary/source/diagnostic contract changed")
    helper = _native_helper(d)
    helper._logical_census(s)  # Exact original slot/commit/census validator; no mode relabeling.
    refs = observation["rank_receipts"]
    require(len(refs) == a.world_size and len({r["path"] for r in refs}) == a.world_size, "missing/duplicate raw ranks")
    rows = [_ref(ref) for ref in refs]
    require(sorted(rows, key=lambda r: r["rank"]) == sorted(s["ranks"], key=lambda r: r["rank"]), "summary differs from raw ranks")
    impl = {**_arm(placement), "hardware_profile": d["profile"], "microbatch_count": 16 // a.world_size, "trajectory_arm_id": f"{d['profile']}-{placement}"}
    _same(s, {"implementation_contract": impl, "implementation_fingerprint": fingerprint(impl)}, "implementation identity changed")
    started, completed = launch.get("started_unix_ns"), launch.get("completed_unix_ns")
    require(type(started) is int and type(completed) is int and 0 < started < completed, "launch interval missing")
    payloads, times, memory = [[], []], [[], []], []
    for row, ref in zip(rows, refs, strict=True):
        rank = row["rank"]
        device, slots = a.devices[rank], list(range(rank, 16, a.world_size))
        require(Path(ref["path"]) == root / f"execution/ranks/rank_{rank:02d}.json", "raw rank outside its native run")
        expected = {"schema_version": "distillschedule.wan13b_two_cycle_timing_rank.v1", "status": "PASS", "run_id": run_id, "expected_head": NATIVE_HEAD,
            "mode": "correctness", "placement": placement, "world_size": a.world_size, "local_rank": device.local_rank, "hostname": d["site"]["node_hosts"][device.node],
            "hardware_profile": d["profile"], "logical_world_size": 16, "logical_slots": slots, "microbatch_count": 16 // a.world_size,
            "profiled": False, "profiler_status": "NOT_RUN", "finite_status": [True, True], "cycle_contract": _CYCLE_CONTRACT,
            "initial_versions": {"generator": 2, "critic": 10, "ema": 10}, "cycle_versions": [{"generator": 3, "critic": 15, "ema": 15}, {"generator": 4, "critic": 20, "ema": 20}],
            "final_versions": s["final_versions"], "updates": {"generator": 2, "critic": 10, "ema": 10}, "allocation_receipt": _expected_receipt(d, rank),
            "matched_contract": SEMANTICS, "workload_fingerprint": fingerprint(SEMANTICS), "arm_contract": _arm(placement), "implementation_fingerprint": fingerprint(_arm(placement)),
            "runtime_contract": _runtime(placement), "trace_rollout_contract_sha256": SEMANTICS["trace_rollout_contract_sha256"],
            "configuration_fingerprint": fingerprint({"workload_fingerprint": fingerprint(SEMANTICS), "implementation_fingerprint": fingerprint(_arm(placement))}),
            "timing_seed": SEMANTICS["seed"], "rank_runtime_seed": SEMANTICS["seed"] + rank,
            "timing_arm_id": f"{d['profile']}-{placement}", "formal_timing_eligible": False, "evidence_extraction": "AFTER_BOTH_TIMED_WINDOWS"}
        _same(row, expected, f"rank{rank} source/allocation/version/implementation changed")
        profiler = row.get("profiler_environment", {})
        require(profiler.get("exact_range") in (None, "", "0") and profiler.get("lora_operator_attribution") == "0", "profiler was enabled")
        trace = row.get("trace_rollout_contract", [])
        require(len(trace) == 6 and fingerprint(trace) == SEMANTICS["trace_rollout_contract_sha256"], "immutable trace inventory changed")
        previous, last_stop = None, started
        for index, (cycle, census) in enumerate(zip(row["cycles"], row["cycle_census"], strict=True)):
            seconds = finite_number(cycle.get("local_elapsed_seconds"), "native cycle duration", positive=True)
            begin, end = cycle.get("start_unix_ns"), cycle.get("stop_unix_ns")
            require(type(begin) is int and type(end) is int and last_stop < begin < end <= completed and abs((end - begin) / 1e9 - seconds) <= max(.1, .01 * seconds), "native cycle clock interval changed")
            _same(cycle, {"cycle": index + 1, "role": _CYCLE_CONTRACT[f"cycle_{index+1}_role"]}, "native cycle boundary changed")
            last_stop = end
            times[index].append(round(seconds * 1e9))
            _same(census, {"generator_updates": 1, "critic_updates": 5, "ema_updates": 5, "physical_rollout_batches": 5 * len(slots), "rollout_batch_size": 1}, "cycle commit/B1 census changed")
            leaves, consumed = census.get("logical_sample_census", [census]), []
            for leaf in leaves:
                _same(leaf, {"generator_updates": 1, "critic_updates": 5, "ema_updates": 5}, "logical B1 update census changed")
                ids = leaf["sample_payload_identities"]
                require(len(ids) == 6 and all(isinstance(x, str) and _SHA.fullmatch(x) for x in ids) and ids == [leaf.get("generator_trace_payload_identity"), *leaf.get("critic_trace_payload_identities", [])], "trace payload identities changed")
                _same(leaf, {"rollout_operation_keys": [x["operation"] for x in trace], "rollout_call_inventory": [x["inventory"] for x in trace],
                    "sample_occurrence_identities": [f"cycle={index+1}/slot={i}/payload={p}" for i, p in enumerate(ids)]}, "executed DiT/trace consumption changed")
                consumed.extend(ids)
            require(previous is None or previous == consumed, "C2 cohort replay changed")
            previous = consumed
            payloads[index].extend(consumed)
        m = row.get("memory", {})
        require(all(type(m.get(k)) is int and m[k] > 0 for k in ("max_allocated_bytes", "max_reserved_bytes", "driver_total_bytes")) and m["max_allocated_bytes"] <= m["max_reserved_bytes"] <= m["driver_total_bytes"] == device.capacity_bytes, "native memory/device capacity changed")
        memory.append(m["max_reserved_bytes"])
    require(all(len(p) == len(set(p)) == 96 for p in payloads), "duplicated or missing logical B1 sample content")
    cycle_ns = [max(t) for t in times]
    require([round(x * 1e9) for x in s["cycle_seconds_max_rank"]] == cycle_ns, "summary times differ from actual rank maxima")
    for raw in (s, launch, health): _scope(raw, d["evidence_scope"])
    return {"run_id": run_id, "placement": placement, "cycle_ns": cycle_ns, "peak_reserved_bytes": max(memory)}


def component_contract(deployment, placement, cycle):
    d = validate_deployment(deployment)
    require(placement in PLACEMENTS and type(cycle) is int and cycle in (1, 2), "unregistered native component")
    return ComponentContract("a100_allocation.wan_cycle", {"matched_contract": SEMANTICS, "updates": "G1_C5_EMA5", "logical_batch": 16, "physical_microbatch": 1},
        d["precision"], {"choice": placement, "allocation_sha256": fingerprint(d["allocation"])},
        {"adapter": ADAPTER, "native_source": d["native_source"], "arm": _arm(placement)},
        {"cycle": cycle, "mode": "correctness", "entry": "LOADED_BOOTSTRAP" if cycle == 1 else "CONTINUOUS_AFTER_C1", "memory_definition": MEMORY,
         "overlap": "NATIVE_INTERNAL_ONLY_NO_DISCOUNT", "deployment_sha256": fingerprint(d), "formal_timing_eligible": False})


def _problem(d):
    a = _allocation(d)
    ops = tuple(Operation(f"cycle{c}.native", c - 1, "native_runtime", OperationKind.CRITIC_UPDATE, GradientMode.GRAD,
        required_versions={"cycles": c - 1}, produced_versions={"cycles": c}, metadata={"cycle": c}) for c in (1, 2))
    graph = ExecutionGraph(ops, (Dependency(ops[0].operation_id, ops[1].operation_id, "same_process"),),
        {"initial_versions": {"cycles": 0}, "cycle_ends": {str(c): ops[c-1].operation_id for c in (1, 2)}, "construction_peak": "NOT_MODELED"})
    ranks = tuple(range(a.world_size))
    rules = tuple(BindingRule(ops[c-1].operation_id, {"placement": p}, component_contract(d, p, c).digest, ranks,
        tuple(f"gpu:{r}" for r in ranks), operation_contract_sha256=component_contract(d, p, c).digest) for p in PLACEMENTS for c in (1, 2))
    return PlanningProblem(ADAPTER, d["binding"], graph, {"placement": PLACEMENTS}, rules, (), (), a.memory_limits,
        tuple({"placement": p} for p in PLACEMENTS), evidence_scope=d["evidence_scope"], lowering={"adapter": ADAPTER, "deployment_sha256": fingerprint(d), "formal_timing_eligible": False})


def _cost_receipt(d, placement, cycle, observations):
    normalized = [validate_native_observation(d, o) for o in observations]
    require(normalized and all(o["placement"] == placement for o in normalized) and len({o["run_id"] for o in normalized}) == len(normalized), "mixed/duplicate calibration runs")
    contract = component_contract(d, placement, cycle)
    return {"schema_version": "distillschedule.operation_calibration.v1", "binding": d["binding"], "key": contract.digest,
        "samples_ns": [o["cycle_ns"][cycle-1] for o in normalized], "workspace_bytes_per_rank": max(o["peak_reserved_bytes"] for o in normalized),
        "status": "PASS", "profiled": False, "kind": "native_operation", "operation_contract_sha256": contract.digest, "component_contract": contract.to_dict(),
        "deployment": d, "placement": placement, "cycle": cycle, "observations": observations, "evidence_scope": d["evidence_scope"], "formal_timing_eligible": False, "claimable_speedup": False}


def assemble_problem_and_costs(deployment, observations, *, output_dir):
    d = validate_deployment(deployment)
    require(observations and all(o.get("placement") in PLACEMENTS for o in observations) and len({o["summary"]["path"] for o in observations}) == len(observations), "empty/duplicate/unsupported observations")
    records = []
    for placement in PLACEMENTS:
        rows = [o for o in observations if o["placement"] == placement]
        if not rows: continue  # No invented fallback cost for an absent native arm.
        for cycle in (1, 2):
            raw = _cost_receipt(d, placement, cycle, rows)
            path = Path(output_dir).resolve() / f"{raw['key']}.json"
            write_new_json(path, raw)
            evidence = {k: raw[k] for k in ("status", "profiled", "kind", "operation_contract_sha256", "component_contract")}
            records.append(OperationCost(raw["key"], tuple(raw["samples_ns"]), raw["workspace_bytes_per_rank"], {**evidence, "source_receipt": str(path), "source_receipt_sha256": file_sha256(path)}))
    return _problem(d), CostModel(d["binding"], tuple(records), d["evidence_scope"])


def lower_selection(result, problem, costs, deployment, *, run_id):
    d = validate_deployment(deployment)
    require(problem.to_dict() == _problem(d).to_dict() and costs.scope == d["evidence_scope"], "saved graph/domains/scope changed")
    for record in costs.records:
        raw = _ref({"path": record.evidence["source_receipt"], "sha256": record.evidence["source_receipt_sha256"]})
        expected = _cost_receipt(d, raw["placement"], raw["cycle"], raw["observations"])
        require(raw == expected and record.key == raw["key"] and list(record.samples_ns) == raw["samples_ns"] and record.workspace_bytes_per_rank == raw["workspace_bytes_per_rank"]
                and record.evidence["component_contract"] == raw["component_contract"], "cost differs from immutable native evidence")
        require(all(run_id != validate_native_observation(d, o)["run_id"] for o in raw["observations"]), "fresh selected run id required")
    costs.verify_receipts()
    predicted = validate_saved_prediction(result, problem, costs)
    require(predicted["status"] == "PREDICTED_FEASIBLE", "no calibrated feasible selection")
    placement = predicted["assignment"]["placement"]
    plan = {"schema_version": "distillschedule.a100_allocation_plan.v1", "adapter": ADAPTER, "deployment": d, "run_id": run_id, "selected_placement": placement,
        "allocation": d["allocation"], "model_plan": {"problem": problem.to_dict(), "cost_model": costs.to_dict(), "search_result": result},
        "commands": command_lowering(d, run_id=run_id, placement=placement), "evidence_scope": d["evidence_scope"], "formal_timing_eligible": False,
        "claimable_speedup": False, "numerical_parity": "NOT_RUN", "gpu_execution": "NOT_RUN", "execution_permitted": d["evidence_scope"] == "MEASURED", "requires_external_owned_launcher": True}
    return json.loads(json.dumps({**plan, "plan_sha256": fingerprint(plan)}, allow_nan=False))


def validate_lowered_plan(plan):
    require(plan.get("schema_version") == "distillschedule.a100_allocation_plan.v1" and plan.get("plan_sha256") == fingerprint({k: v for k, v in plan.items() if k != "plan_sha256"}), "saved plan hash/schema changed")
    model = plan["model_plan"]
    expected = lower_selection(model["search_result"], PlanningProblem.from_dict(model["problem"]), CostModel.from_dict(model["cost_model"]), plan["deployment"], run_id=plan["run_id"])
    require(plan == expected, "saved native allocation/command/selection changed")
    return expected


def search_and_lower(deployment, observations, *, output_dir, run_id, **search_options):
    problem, costs = assemble_problem_and_costs(deployment, observations, output_dir=output_dir)
    result = search_plans(problem, costs, algorithm="hierarchical", **search_options)
    return lower_selection(result, problem, costs, deployment, run_id=run_id)
