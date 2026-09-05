"""Reuse the workload-owned summarizers; bind their results to this integration."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
from typing import Any
from .registry import load_profile, write_new_json, _require


def summary_command(receipt: dict[str, Any]) -> list[str] | None:
    n = receipt["nodes"][0]; root = Path(n["repo_root"])
    p = receipt["fully_resolved_profile"]; env = n["environment"]
    run = str(Path(receipt["asset_config"]["run_root"]) / receipt["run_id"])
    prefix = [n["python"]]
    tail = ["--run-root", run, "--output", str(Path(run)/"native-summary.json")]
    if p["id"].startswith("wan13b"):
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
        return prefix + [str(root/"scripts/distillschedule/summarize_livewan_continuous_world16.py"),
            "--execution-mode", "pipeline", "--run-kind", receipt["mode"],
            "--trace", env["LIVEWAN_OPERATION_TRACE"], "--plan", str(Path(run)/"resolved_plan.node0.json"),
            "--expected-head", receipt["integration_head"], *tail]
    return None


def collect(receipt: dict[str, Any]) -> int:
    from .dispatch import clean_environment
    node = receipt["nodes"][0]
    root = Path(node["repo_root"])
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
    write_new_json(out/"bound-summary.json", value)
    print(out/"bound-summary.json")
    return 0
