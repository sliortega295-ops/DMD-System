"""Thin, process-owned dispatch; never implements training math or runs on import."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

from .registry import (file_sha256, fingerprint, git_head, load_profile, source_layout,
                       verify_assets, write_new_json, _require)


def clean_environment(node: dict[str, Any]) -> dict[str, str]:
    # Do not inherit shell-resident performance flags, Python paths or profiler
    # variables. Site-specific library paths belong to the selected environment.
    result = {k: os.environ[k] for k in ("HOME", "USER", "LOGNAME", "LANG", "TMPDIR") if k in os.environ}
    result["PATH"] = str(Path(node["python"]).parent) + ":/usr/local/bin:/usr/bin:/bin"
    result.update(node["environment"])
    return result


def _correctness_binding(receipt: dict[str, Any]) -> None:
    if receipt["mode"] not in {"timing", "diagnostic-timing", "calibration"}:
        return
    binding = receipt["asset_config"]["correctness"]
    path = Path(binding["path"])
    _require(path.is_file() and file_sha256(path) == binding["sha256"], "correctness receipt hash mismatch")
    # The common binding belongs to this integration profile/HEAD. The native
    # correctness report remains nested input to the original worker's gate.
    data = json.loads(path.read_text())
    expected = {"integration_head": receipt["integration_head"],
                "profile_sha256": receipt["profile_sha256"], "asset_fingerprint": receipt["asset_fingerprint"]}
    for key, value in expected.items():
        _require(data.get("integration_binding", {}).get(key) == value,
                 f"correctness report lacks current integration binding: {key}")


def validate_local(receipt: dict[str, Any], *, root: Path) -> dict[str, Any]:
    if "search_candidate" in receipt:
        from distillschedule.search.execution import validated_profile
        profile = validated_profile(receipt, root=root)
    else:
        profile = load_profile(receipt["profile_id"], root=root)
    _require(fingerprint(profile) == receipt["profile_sha256"], "resolved profile changed")
    _require(git_head(root) == receipt["integration_head"], "integration HEAD changed")
    # Legacy workers reject tracked dirt too; include untracked Python/configs
    # so an uncommitted module cannot shadow the validated checkout.
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"], text=True)
    _require(not dirty.strip(), "run requires a clean committed integration checkout")
    for node in receipt["nodes"]:
        for key, expected in profile["runtime"]["environment"].items():
            _require(node["environment"].get(key) == expected, f"canonical environment override: {key}")
        _require(node["command"][1] == str(Path(node["repo_root"])/profile["runtime"]["launcher"]), "noncanonical launcher")
    layout = source_layout(root, profile, verify_upstreams=True)
    verified = verify_assets(profile, receipt["asset_config"])
    _correctness_binding(receipt)
    if receipt["profile_id"] == "livewan_crossscale_pipeline_best":
        a = receipt["asset_config"]["assets"]
        source = json.loads(Path(a["source_receipt"]["path"]).read_text())
        boot = json.loads(Path(a["bootstrap"]["path"]).read_text())
        _require(boot["source_receipt"]["sha256"] == source["source_receipt_sha256"],
                 "LiveWan bootstrap belongs to another source receipt; prepare portable assets first")
        _require(Path(boot["source_receipt"]["path"]).resolve() == Path(a["source_receipt"]["path"]).resolve(),
                 "LiveWan bootstrap source-receipt path is stale")
        for row in source["files"].values():
            p = Path(row["path"])
            _require(p.is_file() and file_sha256(p) == row["sha256"], "LiveWan prepared source bytes changed")
    return {"status": "PASS", "source_layout": layout, "assets": verified,
            "scope": "CPU_SOURCE_AND_ASSET_ONLY", "gpu": "NOT_RUN_ON_INTEGRATION_BRANCH"}


def ssh_command(node: dict[str, Any]) -> list[str]:
    """Per-allocation SSH transport; never edits the user's SSH configuration."""
    command = ["ssh", "-oBatchMode=yes", "-oConnectTimeout=15", "-oControlMaster=no", "-oControlPath=none"]
    if node.get("address_family") in {"ipv4", "ipv6"}:
        command.append("-4" if node["address_family"] == "ipv4" else "-6")
    if "ssh_port" in node:
        command += ["-p", str(node["ssh_port"])]
    return command + [node["host"]]


def execute_node(payload: dict[str, Any]) -> int:
    receipt, index = payload["receipt"], int(payload["node_rank"])
    _require(index in (0, 1), "invalid node rank")
    node = receipt["nodes"][index]
    root = Path(node["repo_root"])
    checked = validate_local(receipt, root=root)
    out = Path(receipt["asset_config"]["run_root"]) / receipt["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    write_new_json(out / f"effective-contract.node{index}.json", {**receipt, "local_validation": checked})
    if "resolved_model_config" in node:
        write_new_json(Path(node["environment"]["H3_CONFIG"]), node["resolved_model_config"])
    # Everything above is CPU-only. The explicit `run` command alone gets here.
    env = clean_environment(node)
    wrapper = root / "scripts/distillschedule/remote_stage_wrapper.sh"
    pid = out / f"owned-node{index}.pid"
    command = ["bash", str(wrapper), str(pid), *node["command"]]
    if "search_candidate" in receipt:
        from distillschedule.search.resources import ResourceObserver
        with ResourceObserver(out / f"search-resources.node{index}.json", receipt["run_id"], index):
            code = subprocess.call(command, cwd=root, env=env)
    else:
        code = subprocess.call(command, cwd=root, env=env)
    write_new_json(out / f"node-exit.{index}.json", {
        "status": "PROCESS_EXIT_ONLY_NOT_CORRECTNESS", "return_code": code,
        "integration_head": receipt["integration_head"], "profile_id": receipt["profile_id"],
        "native_evidence_root": str(out),
    })
    return int(code)


def run_resolved(receipt: dict[str, Any], *, dry_run: bool = False,
                 coordinator_output: Path | None = None) -> int:
    _require(len(receipt.get("nodes", [])) == 2, "run requires an assets config")
    if dry_run:
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    remote_out = Path(receipt["asset_config"]["run_root"]) / receipt["run_id"]
    out = coordinator_output if coordinator_output is not None else remote_out
    out.mkdir(parents=True, exist_ok=False)
    write_new_json(out / "coordinator-contract.json", receipt)
    jobs: list[tuple[Any, Any, dict[str, Any]]] = []
    try:
        for i, node in enumerate(receipt["nodes"]):
            worker = str(Path(node["repo_root"]) / "scripts/distillschedule/run_best_known_workload.py")
            remote = shlex.join([node["python"], worker, "_node"])
            log = (out / f"ssh-node{i}.log").open("x")
            proc = subprocess.Popen([*ssh_command(node), remote],
                                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
            jobs.append((proc, log, node))
            assert proc.stdin is not None
            proc.stdin.write(json.dumps({"receipt": receipt, "node_rank": i}))
            proc.stdin.close()
        deadline = time.monotonic() + receipt["asset_config"].get("timeout_seconds", 18000)
        while any(p.poll() is None for p, _, _ in jobs):
            failed = [p.returncode for p, _, _ in jobs if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"peer exited {failed}; cancelling only owned process trees")
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded workload timeout; cancelling only owned process trees")
            time.sleep(0.25)
        codes = [p.returncode for p, _, _ in jobs]
        _require(codes == [0, 0], f"node launcher exit codes: {codes}")
        node = receipt["nodes"][0]
        worker = str(Path(node["repo_root"]) / "scripts/distillschedule/run_best_known_workload.py")
        subprocess.run([*ssh_command(node),
                        shlex.join([node["python"], worker, "_collect"])],
                       input=json.dumps(receipt), text=True, check=True, timeout=300)
        # Do NOT manufacture a correctness PASS or speed number from exit status.
        write_new_json(out / "dispatch-result.json", {
            "status": "DISPATCH_AND_NATIVE_SUMMARY_COMPLETE", "node_exit_codes": codes,
            "integration_head": receipt["integration_head"], "formal_speedup_eligible": False,
        })
        print(f"Dispatch completed. Native/bound summaries written in {out}; no historical speedup is reasserted.")
        return 0
    except BaseException:
        for i, (proc, _, node) in enumerate(jobs):
            cancel = Path(node["repo_root"]) / "scripts/distillschedule/cancel_remote_stage.sh"
            try:
                subprocess.run([*ssh_command(node),
                                shlex.join(["bash", str(cancel), str(remote_out / f"owned-node{i}.pid")])], timeout=60, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if proc.poll() is None:
                proc.terminate()
        raise
    finally:
        for proc, log, _ in jobs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
