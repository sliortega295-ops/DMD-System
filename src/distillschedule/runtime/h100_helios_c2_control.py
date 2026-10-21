"""CPU native C2 preparation and raw evidence admission; never launches workers.

This module is copied into an exact 8fc-derived checkout. The old Full5 auditor
stays unchanged: only its policy, asset, retention and trace validators are used.
The frozen runtime's historical launcher_auditor_gate marker is not admission;
only this separately sealed raw-eight-rank audit can grant diagnostic admission.
"""
from __future__ import annotations

import __future__
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace

from . import h100_helios_two_cycle as runtime

SCHEMA = "distillschedule.helios_h100_c2_summary.v1"
PLAN_SCHEMA = "distillschedule.helios_h100_c2_plan.v1"
LAUNCH_SCHEMA = "distillschedule.helios_h100_c2_launch.v1"
SOURCE_SCHEMA = "distillschedule.helios_h100_c2_source.v1"
CONTROL = "src/distillschedule/runtime/h100_helios_c2_control.py"
ADAPTER = "src/distillschedule/runtime/h100_helios_two_cycle.py"
ENTRY = "scripts/distillschedule/summarize_helios_h100_c2.py"
LAUNCHER = "scripts/distillschedule/launch_helios_h100_c2_node.sh"
OLD_LAUNCHER = "scripts/distillschedule/launch_helios_hardware_p0_node.sh"
OLD_LAUNCHER_SHA = "4d14a1ff685161dac6909e96df1b709ab3ca33b29fbccc013114e33acc430211"
WORKER = "scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py"
PREFIX = "DISTILLSCHEDULE_HELIOS_H100_C2_"
TIMER_SCOPE = "NATIVE_CYCLE_WITH_REQUIRED_RECEIPT_COUNTERS_OUTSIDE_SETUP_AND_AUDITS"
ENTRY_TEXT = '''#!/usr/bin/env python3
"""Explicit C2 control entry; the original Full5 auditor is deliberately intact."""
from distillschedule.runtime.h100_helios_c2_control import main
if __name__ == "__main__":
    raise SystemExit(main())
'''
INITIAL = dict(runtime.INITIAL)
FINAL = {"generator": 4, "critic": 20, "ema": 20}
CONTRACT = {"continuous_cycles": 2, "physical_world_size": 8, "logical_global_batch": 16,
    "physical_microbatch": 1, "loss_scale_per_logical_b1": .5, "source_steps_per_cycle": list(runtime.STEPS),
    "initial_versions": INITIAL, "final_versions": FINAL, "updates": {"generator": 2, "critic": 10, "ema": 10}}
require = runtime._require
digest = runtime._digest


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "JSON object required")
    return value


def reference(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def read_ref(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}
        and isinstance(ref["path"], str) and Path(ref["path"]).is_absolute()
        and ".." not in Path(ref["path"]).parts and re.fullmatch(r"[0-9a-f]{64}", str(ref["sha256"])), "exact absolute reference required")
    require(sha(ref["path"]) == ref["sha256"], "referenced bytes changed")
    return read(ref["path"])


def write_new(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def launcher_text(original):
    """Exact, reviewable derivative; preserve all native locking/health/teardown."""
    require(hashlib.sha256(original.encode()).hexdigest() == OLD_LAUNCHER_SHA, "original native launcher changed")
    edits = {
        'case "$VARIANT" in no-gan|gan)': 'case "$VARIANT" in no-gan)',
        'case "$MODE" in validate|correctness|timing)': 'case "$MODE" in correctness|timing)',
        'case "$ARM" in baseline|best|resident|official-swap|h100-nograd-retain|h100-nograd-retain-nosync)':
            'case "$ARM" in best|h100-nograd-retain|h100-nograd-retain-nosync)',
        'export LD_LIBRARY_PATH="$PYENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"': 'export LD_LIBRARY_PATH="$PYENV/lib"',
        'SUMMARY="$REPO/scripts/distillschedule/summarize_helios_hardware_p0.py"':
            '[[ "${DISTILLSCHEDULE_HELIOS_H100_C2_OPT_IN:-0}" == 1 ]] || { echo "C2 requires explicit opt-in" >&2; exit 2; }\n'
            ': "${DISTILLSCHEDULE_HELIOS_H100_C2_PLAN:?set immutable compiled plan}"\n'
            ': "${DISTILLSCHEDULE_HELIOS_H100_C2_PLAN_SHA256:?set compiled plan file SHA256}"\n'
            'SUMMARY="$REPO/scripts/distillschedule/summarize_helios_h100_c2.py"',
    }
    for old, new in edits.items():
        require(original.count(old) == 1, "native launcher seam changed")
        original = original.replace(old, new)
    return original


def inspect_source(root):
    """Exact allowed diff and helper bodies; no Torch import or hardware query."""
    root = Path(root).resolve()
    require(git(root, "rev-parse", "--show-toplevel").strip() == str(root), "not a checkout root")
    require(not git(root, "status", "--porcelain", "--untracked-files=all").strip(), "derived source is dirty")
    head = git(root, "rev-parse", "HEAD").strip()
    require(head != runtime.PARENT_HEAD and git(root, "rev-parse", "HEAD^").strip() == runtime.PARENT_HEAD,
            "C2 source must be one scoped commit directly from exact 8fc")
    allowed = {runtime.WORKER, ADAPTER, CONTROL, ENTRY, LAUNCHER}
    require(set(git(root, "diff", "--name-only", runtime.PARENT_HEAD, head).splitlines()) == allowed,
            "unregistered source files changed outside C2 hook/control/launcher")
    original = git(root, "show", f"{runtime.PARENT_HEAD}:{OLD_LAUNCHER}")
    expected = {ADAPTER: Path(runtime.__file__).read_text(), CONTROL: Path(__file__).read_text(),
                ENTRY: ENTRY_TEXT, LAUNCHER: launcher_text(original)}
    for relative, text in expected.items():
        require((root / relative).read_text() == text, f"C2 source not the registered implementation: {relative}")
    path = root / runtime.WORKER
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in runtime.HELPERS]
    namespace = {"__file__": str(path)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    source = runtime.inspect_helper_source(SimpleNamespace(**namespace))
    require(source["hook_present"] is True, "exact C2 hook missing")
    diff = git(root, "diff", "--binary", "--full-index", runtime.PARENT_HEAD, head)
    return {"schema_version": SOURCE_SCHEMA, "starting_head": runtime.PARENT_HEAD, "derived_head": head,
        "files": {p: sha(root / p) for p in sorted(allowed)}, "source_contract": source,
        "full_diff": diff, "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "execution": "NOT_RUN", "scope": "LOCAL_SOURCE_ONLY_NO_GPU"}


def native_module(root):
    return runtime._load_native(SimpleNamespace(__file__=str(Path(root) / runtime.WORKER)))


def check_plan(plan, root):
    require(plan.get("schema_version") == PLAN_SCHEMA and plan.get("plan_sha256") == digest({k: v for k, v in plan.items() if k != "plan_sha256"}), "invalid compiled C2 plan")
    require(plan.get("explicit_opt_in") is True and plan.get("mode") in ("correctness", "timing")
            and plan.get("arm") in runtime.POLICIES and plan.get("contract") == CONTRACT
            and plan.get("execution") == "NOT_RUN" and plan.get("claimable_speedup") is False
            and plan.get("numerical_status") == "REVIEW" and plan.get("strict_full_state_parity") == "NOT_RUN", "C2 plan scope/contract changed")
    require(plan.get("purpose") in ("EXPLICIT_CALIBRATION", "PLANNER_SELECTED"), "C2 launch purpose missing")
    if plan["purpose"] == "PLANNER_SELECTED":
        model = plan.get("model_plan", {})
        require(set(model) == {"problem", "cost_model", "search_result"}
            and model["problem"].get("binding") == plan["deployment"]["binding"]
            and model["cost_model"].get("binding") == plan["deployment"]["binding"]
            and model["search_result"].get("predicted_best", {}).get("assignment") == {"arm": plan["arm"]}, "selected model/source binding missing")
    else:
        require(plan.get("model_plan") is None, "calibration cannot claim a planner selection")
    d = plan["deployment"]
    require(plan["evidence_scope"] == d["evidence_scope"] and plan["deployment_sha256"] == digest(d), "deployment scope/hash changed")
    require(d["native_source"]["repo_root"] == str(Path(root).resolve()), "receipt-host source path mismatch")
    proof = inspect_source(root)
    require(read_ref(d["source_proof"]) == proof and d["native_source"]["head"] == proof["derived_head"], "derived source proof changed")
    require(plan["run_root"] == str(Path(d["site"]["project_root"]) / "research_artifacts/five_workload_h100_20260930" / plan["run_id"])
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", plan["run_id"]), "unsafe native run binding")
    return d, proof


def _render(plan, native, gate, retention):
    d, arm, mode = plan["deployment"], plan["arm"], plan["mode"]
    root, site = Path(d["native_source"]["repo_root"]), d["site"]
    frozen = native.freeze_policy(root, variant="no-gan", arm=arm)
    groups = gate["groups"]
    assets = {name: group["actual_root"] for name, group in groups.items()}
    worker = [str(root / WORKER), "--variant", "no-gan", "--run-id", plan["run_id"], "--output-dir", str(Path(plan["run_root"]) / "execution"),
        "--logical-package", assets["logical_checkpoint"], "--expected-logical-hash", frozen["logical_hash"],
        "--trace-bundle", assets["trace"], "--expected-trace-hash", frozen["trace_hash"],
        "--generator-root", groups["generator_precast"]["provenance_root"], "--precast-generator-root", assets["generator_precast"],
        "--score-root", groups["score_precast"]["provenance_root"], "--precast-score-root", assets["score_precast"],
        "--negative-prompt", assets["negative_prompt"], "--ema-device", "cuda", *native.policy_arguments(frozen["policy"])]
    worker += ["--timing-mode", "--no-profile-diagnostics"] if mode == "timing" else ["--profile-diagnostics"]
    environment = dict(frozen["environment"], CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", HF_HOME=site["hf_home"],
        PYTHONPATH=f"{root / 'src'}:{root}:{site['helios_root']}", LOCAL_KERNELS=f"kernels-community/flash-attn2={assets['flash_attn_kernel']}")
    environment[runtime.MODE_ENV] = mode
    if retention:
        environment["DISTILLSCHEDULE_HELIOS_H100_RETENTION_API_RECEIPT"] = retention["path"]
    unset = [k for k in frozen["unset_environment"] if k != "LOCAL_KERNELS"] + ["DISTILLSCHEDULE_NSYS_PROFILE", "DISTILLSCHEDULE_NSYS_OPTIONS"]
    return {"frozen": frozen, "assets": assets, "worker_argv": worker, "environment": environment,
        "unset_environment": unset, "retention_capability": retention,
        "precast_manifest_sha256s": {role: groups[name]["precast_manifest_sha256"] for role, name in (("generator", "generator_precast"), ("critic", "score_precast"))},
        "source_provenance": {name: groups[name]["source_path"] for name in ("generator_precast", "score_precast")}}


def _assets(plan, native):
    d, root = plan["deployment"], Path(plan["deployment"]["native_source"]["repo_root"])
    frozen = native.freeze_policy(root, variant="no-gan", arm=plan["arm"])
    gate = native.verify_asset_gate(Path(d["asset_gate"]["path"]), d["asset_gate"]["sha256"], expected_head=d["native_source"]["head"],
        asset_root=Path(d["site"]["asset_root"]), frozen=frozen)
    require(sha(Path(d["site"]["helios_root"]) / "helios/modules/transformer_helios.py") == native.HELIOS_SHA, "Helios source changed")
    native._confined(Path(d["site"]["asset_root"]), Path(d["site"]["hf_home"]))
    require(Path(d["site"]["hf_home"]).is_dir(), "offline HF cache missing")
    retention = native.validate_h100_retention_capability(Path(d["retention_api"]["path"]), frozen) if plan["arm"] != "best" else None
    controls = native.trace_controls(Path(gate["groups"]["trace"]["actual_root"]), "no-gan", frozen["trace_hash"])
    checkpoint = read(Path(gate["groups"]["logical_checkpoint"]["actual_root"]) / "manifest.json")
    unsigned = dict(checkpoint)
    require(unsigned.pop("manifest_hash", None) == frozen["logical_hash"] and digest(unsigned) == frozen["logical_hash"], "checkpoint bytes/hash changed")
    require({k: checkpoint["components"][k]["optimizer_step"] for k in ("generator", "critic")} == {"generator": 2, "critic": 10}
        and checkpoint["ema"]["optimization_step"] == 10, "actual initial G/C/EMA checkpoint versions changed")
    return gate, retention, controls


def correctness_gate(plan, root, *, at_ns):
    ref = plan.get("correctness")
    if plan["mode"] == "correctness":
        require(ref is None, "correctness cannot depend on another run")
        return None
    saved = read_ref(ref)
    require(saved.get("schema_version") == SCHEMA and saved.get("mode") == "correctness"
        and saved.get("arm") == plan["arm"] and saved.get("evidence_scope") == plan["evidence_scope"], "fresh same-arm native C2 correctness required; old Full5 is not C2")
    observed = audit_run(root, Path(saved["run_root"]), evidence_scope=plan["evidence_scope"])
    require(saved == observed and observed["mode"] == "correctness" and observed["deployment_sha256"] == digest(plan["deployment"])
        and observed["run_id"] != plan["run_id"] and 0 <= at_ns - observed["completed_unix_ns"] <= 86400 * 10**9,
        "correctness raw evidence/source/assets/allocation changed or expired")
    return dict(ref, admission="PASS_C2_PROTOCOL_ONLY_NUMERICAL_REVIEW")


def prepare_launch(args):
    """Asset/source reads and command rendering only; outer shell owns execution."""
    require(os.environ.get(PREFIX + "OPT_IN") == "1", "explicit C2 opt-in required")
    plan_ref = {"path": os.environ.get(PREFIX + "PLAN"), "sha256": os.environ.get(PREFIX + "PLAN_SHA256")}
    plan = read_ref(plan_ref)
    root = Path(__file__).resolve().parents[3]
    d, source = check_plan(plan, root)
    require(plan["evidence_scope"] == "MEASURED", "CPU fixture cannot execute a native launch")
    expected = {"run_root": Path(plan["run_root"]), "run_id": plan["run_id"], "variant": "no-gan", "arm": plan["arm"], "mode": plan["mode"],
        "expected_head": d["native_source"]["head"], "asset_root": Path(d["site"]["asset_root"]), "asset_gate": Path(d["asset_gate"]["path"]),
        "asset_gate_sha256": d["asset_gate"]["sha256"], "helios_root": Path(d["site"]["helios_root"]), "hf_home": Path(d["site"]["hf_home"])}
    require(all(getattr(args, k) == v for k, v in expected.items()), "native launch arguments differ from compiled plan")
    if plan["mode"] == "timing":
        require({"path": str(args.correctness_summary), "sha256": args.correctness_sha256} == plan["correctness"], "timing gate ref changed")
    native = native_module(root)
    gate, retention, controls = _assets(plan, native)
    require(set(controls) == set(range(16)), "sixteen native source controls missing")
    rendered = _render(plan, native, gate, retention)
    now = time.time_ns()
    admission = correctness_gate(plan, root, at_ns=now)
    manifest = dict(rendered, schema_version=LAUNCH_SCHEMA, run_id=plan["run_id"], arm=plan["arm"], mode=plan["mode"], variant="no-gan",
        expected_head=d["native_source"]["head"], plan=plan, source_proof=source, asset_gate=d["asset_gate"], asset_fingerprint=gate["fingerprint"],
        correctness_gate=admission, started_unix_ns=now, evidence_scope="MEASURED", gpu_claim="GPU_NOT_RUN_UNTIL_EXPLICIT_LAUNCH")
    directory = Path(plan["run_root"]) / "manifests"
    write_new(directory / "launch.json", manifest)
    for name, rows in (("worker_argv", rendered["worker_argv"]), ("environment", [f"{k}={v}" for k, v in sorted(rendered["environment"].items())]),
                       ("unset_environment", rendered["unset_environment"])):
        with (directory / (name + ".nul")).open("xb") as handle:
            handle.write(b"".join(s.encode() + b"\0" for s in rows))
    require(inspect_source(root) == source, "source changed during prepare")
    return manifest


def _no_fixture(value):
    if isinstance(value, str):
        require("CPU_FIXTURE" not in value and "CPU_ONLY" not in value and "CPU_PROTOCOL" not in value, "CPU fixture cannot become measured evidence")
    elif isinstance(value, dict):
        for v in value.values(): _no_fixture(v)
    elif isinstance(value, list):
        for v in value: _no_fixture(v)


def audit_rank_rows(rows, *, plan, source, native, frozen, controls, manifest, started_ns, completed_ns, evidence_scope="MEASURED"):
    """Pure raw-row validator. Fixture acceptance is never an execution API."""
    require(evidence_scope in ("MEASURED", "CPU_FIXTURE_ONLY"), "invalid evidence scope")
    require(len(rows) == 8 and all(type(r["rank"]) is int for r in rows) and sorted(r["rank"] for r in rows) == list(range(8)), "exact eight distinct raw physical ranks required")
    require(len({r["hostname"] for r in rows}) == 1 and rows[0]["hostname"], "one actual host required")
    require(type(started_ns) is int and type(completed_ns) is int and 0 < started_ns < completed_ns, "launch interval missing")
    slots_seen, times = [], [[], []]
    for row in rows:
        if evidence_scope == "MEASURED": _no_fixture(row)
        else: require(row.get("evidence_scope") in ("CPU_PROTOCOL_FIXTURE_ONLY", "CPU_FIXTURE_ONLY"), "fixture rows must stay marked")
        rank, slots = row["rank"], [row["rank"], row["rank"] + 8]
        require(row["schema_version"] == runtime.SCHEMA and row["status"] == "PASS" and row["run_id"] == plan["run_id"]
            and row["variant"] == "no-gan" and row["mode"] == plan["mode"] and row["arm"] == plan["arm"], "rank/schema/run/mode/arm changed; old Full5 cannot admit C2")
        require(row["world_size"] == 8 and row["local_rank"] == rank and row["hardware_profile"] == {
            "name": "h100-8-b16", "physical_world_size": 8, "logical_global_batch": 16, "logical_b1_microbatches_per_rank": 2,
            "logical_slots": slots, "loss_scale_per_logical_b1": .5, "rank_mapping": "slot = physical_rank + microbatch * physical_world_size"}, "hardware/logical mapping changed")
        slots_seen += slots
        require(row["source_contract"] == source["source_contract"] and row["p0_policy"] == frozen["policy"]
            and row["native_full5_contract"] == frozen["contract"] and row["contract"] == CONTRACT, "source/math/precision policy changed")
        require(row["profile_mode"] is False and row["timing_mode"] is (plan["mode"] == "timing")
            and row.get("validate_only", False) is False and row["ema_device"] == "cuda"
            and row["profile_diagnostics"] is (plan["mode"] == "correctness")
            and row["operation_source"]["enabled"] is False and row["diagnostic_single_critic"] is False, "profiled/validation/CPU EMA cannot calibrate")
        require(row["same_process"] is True and row["initializations"] == 1 and row["measured_cycle"] == 2
            and row["source_cohort_policy"] == "fixed_source_cohort_replay" and row["source_steps"] == list(runtime.STEPS)
            and row["timer_scope"] == TIMER_SCOPE and row["claimable_speedup"] is False
            and row["numerical_status"] == "REVIEW_NOT_FULL_STATE_PARITY", "continuous timing/numerical scope changed")
        require("full5_seconds" not in row and row["initial_versions"] == row["initial_physical_versions"] == INITIAL
            and row["final_versions"] == row["final_physical_versions"] == FINAL
            and row["updates"] == CONTRACT["updates"] and row["version_mapping"] == {"logical_minus_physical": {k: 0 for k in INITIAL}}, "actual physical/logical versions changed")
        identities = row["identities"]
        binding = digest({"schema": "distillschedule.static_world16_trace.v1", "variant": "no-gan", "bundle_hash": frozen["trace_hash"], "logical_steps": list(runtime.STEPS), "rank_slots": list(range(16))})
        require(identities["logical_manifest_hash"] == frozen["logical_hash"] and identities["trace_bundle_hash"] == frozen["trace_hash"]
            and identities["trace_binding_hash"] == binding and identities["cross_step_rollout_batchability"] == "PASS"
            and identities["rank_control_validation"].get("status") == "PASS" and identities["rank_control_validation"].get("rank_count") == 16
            and identities["rank_control_validation"].get("cross_rank_call_count_parity") is True, "immutable source trace/checkpoint evidence changed")
        backend = row["attention_backend"]
        require(backend["status"] == "PASS" and backend["requested"] == backend["runtime"] == "local-fa2"
            and backend["extension_sha256"] == native.FLASH_SHA and backend["flash_attn_func_available"] is True
            and "flash_attn2" in backend["flash_attn_func_module"], "actual local-FA2 evidence required; not FA3")
        _audit_build(row, manifest, frozen)
        # The C2 hook returns before the legacy Full5-only synthetic zero fields.
        # Do not require or invent them: exact source + executed FSDP build binds
        # the no-offload/no-role-swap mechanism. Reject contradictory extras.
        require(all(row[k] == 0 for k in ("transition_count", "model_swap_count", "parameter_streaming_count") if k in row)
            and row.get("external_rollout_engine_used", False) is False and "official_role_swap" not in row,
            "native no-offload/no-external-engine contract changed")
        require({k: row["initial"][field] for k, field in (("generator", "generator_physical_step"), ("critic", "critic_physical_step"), ("ema", "ema_step"))} == INITIAL,
            "actual initialized checkpoint counters changed")
        cycles = row["cycles"]
        require(len(cycles) == 2 and [c["cycle"] for c in cycles] == [1, 2], "two actual continuous cycles required")
        previous, stop, objects = INITIAL, started_ns, None
        for i, cycle in enumerate(cycles, 1):
            require(cycle["status"] == "PASS_NATIVE_PROTOCOL" and cycle["measured"] is (i == 2) and cycle["logical_slots"] == slots
                and cycle["initial_versions"] == cycle["initial_physical_versions"] == previous
                and cycle["final_versions"] == cycle["final_physical_versions"]
                and cycle["source_identity"] == identities, "cycle identity/continuation changed")
            require(type(cycle["start_unix_ns"]) is int and type(cycle["stop_unix_ns"]) is int
                and stop <= cycle["start_unix_ns"] < cycle["stop_unix_ns"] <= completed_ns, "cycle timer outside fresh launch or not continuous")
            require(cycle["outer_boundary_fences"] == {"pre_barrier": 1, "pre_cuda_sync": 1, "post_cuda_sync": 1}, "actual outer fences missing")
            obj = cycle["object_identity"]
            require(set(obj) == {"generator", "critic", "ema"} and all(isinstance(obj[k], list) and len(obj[k]) == 4
                and all(type(x) is int and x > 0 for x in obj[k]) for k in ("generator", "critic"))
                and type(obj["ema"]) is int and obj["ema"] > 0 and obj["generator"][0] != obj["critic"][0]
                and obj["generator"][1] != obj["critic"][1] and (objects is None or objects == obj), "engine/module/optimizer/scheduler/EMA rebuilt or aliased")
            require(all(type(v) is int for v in cycle["census"].values()) and all(type(v) is int for v in cycle["rollout_fences"].values())
                and all(type(v) is int for v in cycle["outer_boundary_fences"].values()), "actual census/fence counts must be integers, not booleans")
            runtime._audit_cycle(cycle, rank=rank, slots=slots, native=native, frozen=frozen, controls=controls)
            require(row["cycle_wall_seconds"][str(i)] == cycle["wall_seconds"], "timer summary differs from raw cycle")
            # Wall clock only bounds perf_counter plausibility; never substitutes a fabricated cost.
            elapsed = (cycle["stop_unix_ns"] - cycle["start_unix_ns"]) / 1e9
            require(abs(elapsed - cycle["wall_seconds"]) <= max(.05, .02 * elapsed), "actual timer clocks disagree")
            if plan["mode"] == "correctness":
                checksum = cycle["ema_checksum_outside_timer"]
                require(checksum["device"] == "cuda" and checksum["optimization_step"] == cycle["final_versions"]["ema"]
                    and checksum["communication_per_update"] is False and checksum["tracked_tensor_count"] > 0
                    and all(type(checksum[k]) in (float, int) and math.isfinite(checksum[k]) for k in ("local_sum", "local_norm")), "actual outside-timer EMA checksum missing")
                errors = []
                needed = bool(frozen["policy"]["generator_nograd_cross_kv_cache"] or frozen["policy"]["generator_nograd_static_context_cache"])
                for batch in cycle["rollout_batches"]: native._audit_parity(batch, required=needed, label="C2", errors=errors)
                require(not errors, "; ".join(errors))
            previous, stop, objects = cycle["final_versions"], cycle["stop_unix_ns"], obj
            times[i-1].append(round(cycle["wall_seconds"] * 1e9))
        require(previous == FINAL and row["final"] == {"generator_physical_step": 4, "critic_physical_step": 20,
            "generator_logical_version": 4, "critic_logical_version": 20, "ema_step": 20}, "final native state changed")
    require(sorted(slots_seen) == list(range(16)), "sixteen distinct source slots required")
    return {"cycle_ns": [max(values) for values in times], "logical_slots": sorted(slots_seen), "physical_rank_count": 8,
            "initial_versions": INITIAL, "final_versions": FINAL, "evidence_scope": evidence_scope}


def _audit_build(row, manifest, frozen):
    build = row["build"]
    expected = {"both_backbones_sharded_for_full5": True, "parameter_offload": False, "frozen_parameter_streaming": False,
                "role_transition": False, "process_group_rebuild_in_full5": False}
    require(all(build["static_residency"].get(k) is v for k, v in expected.items()), "actual static residency changed")
    for role, asset, subfolder in (("generator", "generator_precast", "transformer_ode"), ("critic", "score_precast", "transformer")):
        metadata, fsdp = build[role]["metadata"], build[role]["fsdp2"]
        require(metadata["loading_mode"] == "mixed_bf16_precast_direct" and metadata["block_count"] == 40
            and metadata["trainable_dtype_inventory"] == ["torch.float32"] and metadata["chunked_residual_storage"] == "cuda_out_of_place"
            and metadata["precast_manifest_sha256"] == manifest["precast_manifest_sha256s"][role]
            and metadata["load_root"] == manifest["assets"][asset] and metadata["subfolder"] == subfolder
            and metadata["backbone_root"] == str(Path(manifest["source_provenance"][asset]).parent), "actual model/dtype/asset identity changed")
        fields = {"mesh_policy": frozen["policy"]["mesh_policy"], "mesh_shape": frozen["mesh_shape"], "world_size": 8,
            "hardware_profile": "h100-8-b16", "logical_global_batch": 16, "logical_b1_microbatches_per_rank": 2, "block_count": 40,
            "reshard_after_forward": True, "param_compute_dtype": "torch.bfloat16", "reduce_dtype": "torch.float32", "cpu_parameter_offload": False,
            "parameter_streaming": False, "transition": False}
        require(all(type(fsdp.get(k)) is type(v) and fsdp[k] == v for k, v in fields.items()), "actual HSDP/layout/precision changed")


def audit_run(root, run_root, *, evidence_scope="MEASURED"):
    """Reopen hashes, assets, launcher health and all eight raw C2 receipts."""
    root, run_root = Path(root).resolve(), Path(run_root).resolve()
    manifest_path, health_path = run_root / "manifests/launch.json", run_root / "run_status.json"
    manifest, health = read(manifest_path), read(health_path)
    plan = manifest["plan"]
    d, source = check_plan(plan, root)
    require(evidence_scope in ("MEASURED", "CPU_FIXTURE_ONLY") and plan["evidence_scope"] == manifest["evidence_scope"] == evidence_scope, "fixture cannot pass native admission")
    require(str(run_root) == plan["run_root"] and manifest["schema_version"] == LAUNCH_SCHEMA and manifest["source_proof"] == source
        and all(manifest[k] == plan[k] for k in ("run_id", "arm", "mode")) and manifest["variant"] == "no-gan"
        and manifest["expected_head"] == d["native_source"]["head"], "launch/source/selection binding changed")
    native = native_module(root)
    gate, retention, controls = _assets(plan, native)
    expected = _render(plan, native, gate, retention)
    require(all(manifest[k] == v for k, v in expected.items()) and manifest["asset_gate"] == d["asset_gate"]
        and manifest["asset_fingerprint"] == gate["fingerprint"], "actual worker lowering or full-hash assets changed")
    require(manifest["correctness_gate"] == correctness_gate(plan, root, at_ns=manifest["started_unix_ns"]), "fresh C2 correctness binding changed")
    require(health["schema_version"] == native.HEALTH_SCHEMA and health["status"] == "PASS" and health["exit_code"] == 0
        and health["expected_head"] == health["observed_head_after"] == d["native_source"]["head"] and health["clean_source_after"] is True
        and health["cgroup_oom_delta"] == {"oom": 0, "oom_kill": 0} and health["gpu_processes_before"] == health["gpu_processes_after"] == []
        and health["launch_manifest_sha256"] == sha(manifest_path), "raw launcher/source/OOM/ownership health failed")
    inventory, devices = health["gpu_inventory"], d["allocation"]["devices"]
    require(len(inventory) == len(devices) == 8 and len({g["uuid"] for g in inventory}) == 8, "actual eight GPU inventory missing")
    for actual, device in zip(inventory, devices, strict=True):
        require("H100" in actual["name"] and (actual["index"], actual["uuid"], actual["memory_mib"] * 2**20, actual["driver"]) ==
            (device["rank"], device["uuid"], device["capacity_bytes"], d["runtime_versions"]["driver"]), "actual hardware differs from deployment")
    sealed = health["control_files"]
    require(len({f["path"] for f in sealed}) == len(sealed), "duplicate sealed control file")
    by_path = {f["path"]: f["sha256"] for f in sealed}
    required = {"manifests/launch.json", "manifests/worker_argv.nul", "manifests/environment.nul", "manifests/unset_environment.nul",
                "manifests/environment.txt", "manifests/environment_versions.json", "manifests/command.sh", "logs/source_status_after.txt",
                "logs/cgroup_before.txt", "logs/cgroup_after.txt", "logs/gpu_processes_before.txt", "logs/gpu_processes_after.txt"}
    require(required <= set(by_path), "raw control evidence omitted from health seal")
    for relative, expected_sha in by_path.items():
        path = native._confined(run_root, run_root / relative)
        require(sha(path) == expected_sha, "sealed raw control bytes changed")
    for name, values in (("worker_argv", expected["worker_argv"]), ("environment", [f"{k}={v}" for k,v in sorted(expected["environment"].items())]), ("unset_environment", expected["unset_environment"])):
        require((run_root / ("manifests/" + name + ".nul")).read_bytes() == b"".join(v.encode() + b"\0" for v in values), "actual NUL-rendered launch input differs from frozen policy")
    def cgroup(label):
        rows = [line.split() for line in (run_root / f"logs/cgroup_{label}.txt").read_text().splitlines()]
        require(all(len(row) == 2 for row in rows) and len({row[0] for row in rows}) == len(rows), "malformed cgroup counters")
        return {k: int(v) for k, v in rows}
    before, after = cgroup("before"), cgroup("after")
    require(all(k in before and k in after and after[k] - before[k] == 0 for k in ("oom", "oom_kill"))
        and health["cgroup_before"] == before and health["cgroup_after"] == after, "raw host/cgroup OOM counters changed")
    require(all(not (run_root / f"logs/{name}.txt").read_text().strip() for name in ("source_status_after", "gpu_processes_before", "gpu_processes_after")), "actual source/ownership logs are not clean")
    require(read(run_root / "logs/gpu_inventory.json")["gpus"] == inventory, "actual GPU inventory differs from health receipt")
    command = shlex.split((run_root / "manifests/command.sh").read_text())
    prefix = [d["site"]["python"], "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=8", "--node-rank=0", "--master-addr=127.0.0.1", f"--master-port={d['master_port']}"]
    require(command == prefix + expected["worker_argv"], "executed argv differs from selected C2 worker")
    environment = {}
    for line in (run_root / "manifests/environment.txt").read_text().splitlines():
        key, sep, value = line.partition("=")
        require(sep and key not in environment, "duplicate/malformed actual environment")
        environment[key] = value
    # The derived launcher pins only its selected Python library directory;
    # unlike the old launcher it cannot inherit an unbound loader search path.
    tracked = {k: v for k, v in expected["environment"].items() if k.startswith(("DISTILLSCHEDULE_", "NCCL_", "HELIOS_", "PYTORCH_"))
        or k in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "PYTHONPATH", "LOCAL_KERNELS")}
    require(environment.get("LD_LIBRARY_PATH") == str(Path(d["site"]["python"]).parent.parent / "lib")
        and {k: v for k, v in environment.items() if k != "LD_LIBRARY_PATH"} == tracked, "executed frozen environment changed")
    versions = read(run_root / "manifests/environment_versions.json")
    require(versions["executable"] == d["site"]["python"] and versions["python"].split()[0] == d["runtime_versions"]["python"]
        and versions["packages"] == d["runtime_versions"]["packages"], "actual native Python/package versions changed")
    paths = sorted((run_root / "execution/ranks").glob("rank_*.json"))
    require(len(paths) == 8 and len(health["rank_receipts"]) == 8
        and {r["path"]: r["sha256"] for r in health["rank_receipts"]} == {str(p.relative_to(run_root)): sha(p) for p in paths}, "raw eight-rank seal changed")
    rows = [read(p) for p in paths]
    observed = audit_rank_rows(rows, plan=plan, source=source, native=native, frozen=expected["frozen"], controls=controls,
        manifest=manifest, started_ns=manifest["started_unix_ns"], completed_ns=health["completed_unix_ns"], evidence_scope=evidence_scope)
    require(inspect_source(root) == source, "source changed during raw audit")
    return dict(observed, schema_version=SCHEMA, status="PASS_C2_PROTOCOL_ONLY", run_id=plan["run_id"], run_root=str(run_root),
        mode=plan["mode"], arm=plan["arm"], expected_head=d["native_source"]["head"], deployment_sha256=digest(d),
        plan_sha256=plan["plan_sha256"], source_proof_sha256=d["source_proof"]["sha256"],
        completed_unix_ns=health["completed_unix_ns"], raw_ranks=[reference(p) for p in paths], launcher_health=reference(health_path),
        launch_manifest=reference(manifest_path), claimable_speedup=False, numerical_status="REVIEW", strict_full_state_parity="NOT_RUN",
        timer_scope=TIMER_SCOPE, measured_cycle=2, memory_definition="WHOLE_GPU_CAPACITY_EXCLUSIVE_RESERVATION_NOT_MEASURED_PEAK")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-launch", action="store_true")
    action.add_argument("--seal-run", action="store_true")
    for key in ("run-root", "asset-root", "asset-gate", "helios-root", "hf-home", "correctness-summary", "output"):
        parser.add_argument("--" + key, type=Path, required=key == "run-root")
    for key in ("expected-head", "run-id", "variant", "mode", "arm", "asset-gate-sha256", "correctness-sha256"):
        parser.add_argument("--" + key, required=key == "expected-head")
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    require(inspect_source(root)["derived_head"] == args.expected_head, "clean exact derived HEAD required")
    if args.prepare_launch:
        prepare_launch(args)
    elif args.seal_run:
        require(args.exit_code is not None, "actual child exit code required")
        native_module(root).seal_run(args)
    else:
        require(args.output is not None, "summary output required")
        value = audit_run(root, args.run_root)
        write_new(args.output, value)
        print(json.dumps(value, sort_keys=True, allow_nan=False))
    require("torch" not in sys.modules, "CPU control unexpectedly imported Torch")
    return 0
