"""Reuse the workload-owned summarizers; bind their results to this integration."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
from typing import Any
from .registry import load_profile, write_new_json, fingerprint, git_head, repository_root, _require


def bind_livewan_gate(gate: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    """Attach canonical identity after the native matched-gate builder passes.

    This does not convert a trajectory summary or numeric REVIEW into strict
    parity. Native worker validation still checks every gate field and asset.
    """
    p = load_profile("livewan_crossscale_pipeline_best")
    _require(gate.get("schema_version") == "distillschedule.livewan_two_cycle_timing_gate.v2"
             and gate.get("status") == "PASS" and gate.get("claimable_for_two_cycle_timing") is True,
             "canonical binding requires the native matched two-cycle gate")
    _require(receipt.get("integration_head") == gate.get("expected_head") == git_head(repository_root()),
             "canonical gate source HEAD differs")
    _require(receipt.get("profile_id") == p["id"] and receipt.get("profile_sha256") == fingerprint(p),
             "canonical gate profile differs")
    _require(receipt.get("mode") == "correctness" and receipt.get("native_mode") == "trajectory",
             "binding must come from canonical correctness, not timing")
    _require(receipt.get("asset_fingerprint") == fingerprint(receipt["asset_config"]["assets"]),
             "canonical gate asset fingerprint differs")
    arm = gate.get("arms", {}).get("pipeline", {})
    _require(arm.get("run_id") == receipt.get("run_id")
             and arm.get("arm_id") == p["runtime"]["environment"]["LIVEWAN_TWO_CYCLE_ARM_ID"]
             and arm.get("pipeline_variant") == p["contract"]["variant"]
             and gate.get("seed") == p["contract"]["seed"], "canonical pipeline arm binding differs")
    result = dict(gate)
    result["integration_binding"] = {k: receipt[k] for k in
                                    ("integration_head", "profile_sha256", "asset_fingerprint")}
    result["integration_binding"]["profile_id"] = p["id"]
    return result


def summary_command(receipt: dict[str, Any]) -> list[str] | None:
    n = receipt["nodes"][0]; root = Path(n["repo_root"])
    p = receipt["fully_resolved_profile"]; env = n["environment"]
    run = str(Path(receipt["asset_config"]["run_root"]) / receipt["run_id"])
    prefix = [n["python"]]
    tail = ["--run-root", run, "--output", str(Path(run)/"native-summary.json")]
    if p["id"].startswith("wan13b"):
        native_mode = receipt.get("native_mode")
        if native_mode in {"trajectory", "two-cycle-timing", "two-cycle-calibration"}:
            stem = "trajectory" if native_mode == "trajectory" else "two_cycle_timing"
            command = prefix + [str(root / f"scripts/distillschedule/summarize_wan13b_{stem}_world16.py"),
                "--placement", p["topology"]["placement"],
                "--expected-head", receipt["integration_head"], *tail]
            if native_mode == "trajectory":
                command += ["--cycles", "2"]
            elif native_mode == "two-cycle-calibration":
                command += ["--calibrate-stages"]
            return command
        return prefix + [str(root/"scripts/distillschedule/summarize_wan13b_fair_world16.py"),
                         "--placement", p["topology"]["placement"], "--mode", receipt["mode"], *tail]
    if p["id"].startswith("helios"):
        command = prefix + [str(root/"scripts/distillschedule/summarize_static_fsdp2_p0_world16.py"),
            "--variant", p["contract"]["variant"], "--mode", receipt["mode"],
            "--mesh-policy", p["topology"]["placement"], "--rollout-batch-size", "1",
            "--pack-real-score-pair", "off", *tail]
        for flag in (
            "attention-backend", "modulation-chunk-tokens", "selective-flash-checkpoint-stride",
            "generator-nograd-cross-kv-cache", "generator-nograd-static-context-cache",
            "generator-nograd-persistent-unshard", "generator-nograd-persistent-unshard-flush-allocator-cache",
            "generator-residual-storage", "critic-residual-storage", "generator-nograd-lora-fused-epilogue",
            "all-role-lora-scale1-add", "optimizer-dispatch-warmup", "generator-nograd-residual-fusion",
            "exact-grad-residual-fusion", "exact-modulated-norm-epilogue", "exact-modulated-norm-materialization",
            "exact-modulated-norm-backward", "exact-modulated-norm-layernorm", "exact-modulated-norm-fused-backward",
            "exact-history-current-fusion", "elide-explicit-rollout-sync", "elide-critic-scalar-logs",
            "elide-official-generator-free-memory", "cuda-allocator-expandable-segments",
        ):
            suffix = flag.replace("-", "_").upper()
            keys = ["DISTILLSCHEDULE_"+suffix, "DISTILLSCHEDULE_HELIOS_"+suffix]
            selected = [k for k in keys if k in env]
            _require(len(selected)==1, f"ambiguous native summarizer flag: {flag}")
            command += ["--"+flag, env[selected[0]]]
        return command
    if p["id"].startswith("livewan"):
        if receipt.get("native_mode") == "trajectory":
            return prefix + [str(root/"scripts/distillschedule/summarize_livewan_trajectory_world16.py"),
                "--cycles", str(p["contract"]["continuous_cycles"]),
                "--expected-head", receipt["integration_head"], *tail]
        if receipt.get("native_mode") == "two-cycle-timing":
            return prefix + [str(root/"scripts/distillschedule/summarize_livewan_two_cycle_timing_world16.py"),
                "--execution-mode", "baseline" if p["contract"]["variant"] == "b1" else "pipeline", "--pipeline-variant", p["contract"]["variant"],
                "--trace", env["LIVEWAN_OPERATION_TRACE"],
                "--expected-head", receipt["integration_head"], *tail]
        return prefix + [str(root/"scripts/distillschedule/summarize_livewan_continuous_world16.py"),
            "--execution-mode", "pipeline", "--run-kind", receipt["mode"],
            "--trace", env["LIVEWAN_OPERATION_TRACE"], "--plan", str(Path(run)/"resolved_plan.node0.json"),
            "--expected-head", receipt["integration_head"], *tail]
    return None


def collect(receipt: dict[str, Any]) -> int:
    from .dispatch import clean_environment
    node = receipt["nodes"][0]
    root = Path(node["repo_root"])
    if "search_candidate" in receipt:
        from distillschedule.search.execution import validated_profile
        p = validated_profile(receipt, root=root)
    else:
        p = load_profile(receipt["profile_id"], root=root)
    out = Path(receipt["asset_config"]["run_root"]) / receipt["run_id"]
    native = out / "native-summary.json"
    _require(not native.exists(), "refusing to overwrite native evidence")
    command = summary_command(receipt)
    if command:
        subprocess.run(command, cwd=root, env=clean_environment(node), check=True)
        value = json.loads(native.read_text())
        _require(value.get("status") == "PASS", "native evidence gate failed")
    else:
        from distillschedule.backends.training.minimax_h3.matched_contract import validate_global_snapshots
        rows = [json.loads(f.read_text()) for f in sorted((out/"execution/ranks").glob("rank_*.json"))]
        _require(len(rows)==16 and {r["rank"] for r in rows}==set(range(16)), "H3 missing/duplicate ranks")
        for r in rows:
            _require(r["status"]=="PASS_NATIVE_RUNTIME_CONTRACTS" and r["integration_head"]==receipt["integration_head"], "H3 source/runtime gate failed")
            _require(r["boundary_policy"]=="none" and r["fa3_replay_storage"]=="cpu_staged", "H3 storage contract changed")
        errors = validate_global_snapshots([r["census"] for r in rows], expected_world_size=16, require_unique_samples=True)
        _require(not errors, f"H3 census failed: {errors}")
        value = {"status":"PASS", "rank_count":16, "mode":receipt["mode"],
            "semantics":"approximate_grid1000", "claimable_timing":False,
            "numerical_comparison":"NOT_RUN", "native_contract_receipts":[str(f) for f in sorted((out/"execution/ranks").glob("rank_*.json"))]}
        write_new_json(native,value)
    # Add metadata, do not replace/waive any native gate or invent a new speedup.
    value["integration_binding"] = {k:receipt[k] for k in ("integration_head","profile_sha256","asset_fingerprint")}
    value["integration_binding"].update(profile_id=p["id"],historical_time_reproduced=False)
    if p["id"] == "livewan_crossscale_pipeline_best":
        value["integration_binding"].update(
            numeric_policy="REJECT_EXTREME_ONLY_RECORD_ORDINARY_REVIEW",
            strict_cross_implementation_parity="NOT_ESTABLISHED",
            speedup_requires_matched_current_head_baseline=True)
    write_new_json(out/"bound-summary.json", value)
    print(out/"bound-summary.json")
    return 0
