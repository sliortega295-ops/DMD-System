"""Thin, process-owned dispatch; never implements training math or runs on import."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any
import zlib

from .registry import (file_sha256, fingerprint, git_head, load_profile, source_layout, repository_root,
                       verify_assets, write_new_json, _require, validate_resolved_asset_binding,
                       verify_python_path_overlays)


def clean_environment(node: dict[str, Any]) -> dict[str, str]:
    # Do not inherit shell-resident performance flags, Python paths or profiler
    # variables. Site-specific library paths belong to the selected environment.
    result = {k: os.environ[k] for k in ("HOME", "USER", "LOGNAME", "LANG", "TMPDIR") if k in os.environ}
    result["PATH"] = str(Path(node["python"]).parent) + ":/usr/local/bin:/usr/bin:/bin"
    result.update(node["environment"])
    return result


def _correctness_binding(receipt: dict[str, Any]) -> None:
    if receipt["mode"] not in {"timing", "diagnostic-timing", "calibration", "native"}:
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
    if receipt.get("search_candidate", {}).get("id", "").startswith("joint-h3"):
        from distillschedule.search.h3_implementation import H3Implementation
        if H3Implementation.parse(receipt["search_candidate"]["id"]).rollout_window is not None:
            from distillschedule.search.h3_action_admission import validate_action_admission
            validate_action_admission(data, receipt, repository_root())
    if receipt.get("search_candidate", {}).get("id", "").startswith("joint-helios1-"):
        from distillschedule.search.helios_native import validate_admission
        validate_admission(data, receipt, repository_root())


def validate_local(receipt: dict[str, Any], *, root: Path, node_rank: int | None = None) -> dict[str, Any]:
    if "search_candidate" in receipt:
        from distillschedule.search.execution import validated_profile
        profile = validated_profile(receipt, root=root)
    else:
        profile = load_profile(receipt["profile_id"], root=root)
    _require(fingerprint(profile) == receipt["profile_sha256"], "resolved profile changed")
    validate_resolved_asset_binding(receipt, profile)
    _require(git_head(root) == receipt["integration_head"], "integration HEAD changed")
    # Legacy workers reject tracked dirt too; include untracked Python/configs
    # so an uncommitted module cannot shadow the validated checkout.
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"], text=True)
    _require(not dirty.strip(), "run requires a clean committed integration checkout")
    for node in receipt["nodes"]:
        for key, expected in profile["runtime"]["environment"].items():
            if key == "DISTILLSCHEDULE_HELIOS_PLANNER_MODE" and "search_candidate" in receipt:
                _require(expected == "off", "Helios planner must default off")
                expected = "native" if receipt["mode"] == "diagnostic-timing" else receipt["mode"]
            if (receipt["mode"] == "calibration" and receipt["profile_id"] == "minimax_h3_grid1000_best"
                    and "search_candidate" in receipt and key == "H3_PLANNER_CALIBRATION"):
                _require(expected == "0", "H3 calibration must have a disabled default")
                expected = "1"
            if (receipt["mode"] in {"calibration", "action-observation"}
                    and receipt["profile_id"] == "livewan_crossscale_pipeline_best"
                    and "search_candidate" in receipt
                    and key == "DISTILLSCHEDULE_LIVEWAN_OPERATION_CALIBRATION"):
                _require(expected == "0", "calibration must be opt-in over a disabled default")
                expected = "1"
            _require(node["environment"].get(key) == expected, f"canonical environment override: {key}")
        _require(node["command"][1] == str(Path(node["repo_root"])/profile["runtime"]["launcher"]), "noncanonical launcher")
    layout = source_layout(root, profile, verify_upstreams=True)
    verified = verify_assets(profile, receipt["asset_config"])
    dependencies = []
    if any(n.get("python_path_overlays") for n in receipt["asset_config"]["nodes"]):
        _require(type(node_rank) is int and node_rank in (0, 1), "dependency overlays require explicit local node_rank")
        dependencies = verify_python_path_overlays(receipt["asset_config"]["nodes"][node_rank])
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
    return {"status": "PASS", "source_layout": layout, "assets": verified, "python_path_overlays": dependencies,
            "scope": "CPU_SOURCE_AND_ASSET_ONLY", "gpu": "NOT_RUN_ON_INTEGRATION_BRANCH"}


def ssh_command(node: dict[str, Any]) -> list[str]:
    """Per-allocation SSH transport; never edits the user's SSH configuration."""
    command = ["ssh", "-oBatchMode=yes", "-oConnectTimeout=15", "-oControlMaster=no", "-oControlPath=none"]
    if node.get("address_family") in {"ipv4", "ipv6"}:
        command.append("-4" if node["address_family"] == "ipv4" else "-6")
    if "ssh_port" in node:
        command += ["-p", str(node["ssh_port"])]
    return command + [node["host"]]


def _worker_request(node: dict[str, Any], action: str, payload: dict[str, Any], *,
                    transport: str) -> tuple[list[str], str | None]:
    """Transport only the existing native node/collector entry points."""
    _require(action in {"_node", "_collect"}, "unsupported native worker action")
    _require(transport in {"stdin", "argv-json-tty-v1"}, "unsupported native dispatch transport")
    worker = str(Path(node["repo_root"]) / "scripts/distillschedule/run_best_known_workload.py")
    stdin_payload = json.dumps(payload)
    if transport == "stdin":
        return [*ssh_command(node), shlex.join([node["python"], worker, action])], stdin_payload
    # Match the metadata RPC escape hatch for gateways that never forward EOF.
    # Keep training stdout streamed; frame only successful worker completion.
    encoded = base64.b64encode(zlib.compress(stdin_payload.encode())).decode()
    _require(len(encoded) <= 100_000, "native dispatch payload exceeds argv transport bound")
    frame = "\nDMD_NATIVE_DISPATCH_JSON=" + base64.b64encode(
        json.dumps({"action": action, "returncode": 0}).encode()).decode()
    boot = ("import base64,io,runpy,sys,zlib\n"
            f"sys.argv=[{worker!r},{action!r}]\n"
            f"sys.stdin=io.StringIO(zlib.decompress(base64.b64decode({encoded!r})).decode())\n"
            "try:\n"
            f" runpy.run_path({worker!r},run_name='__main__')\n"
            "except SystemExit as error:\n"
            " if error.code not in (None,0): raise\n"
            f"print({frame!r},flush=True)\n")
    environment = ["env", "PYTHONDONTWRITEBYTECODE=1",
                   "LD_LIBRARY_PATH=" + str(Path(node["python"]).parent.parent / "lib"), node["python"]]
    ssh = ssh_command(node)
    return [*ssh[:-1], "-n", "-tt", ssh[-1],
            shlex.join([*environment, "-c", boot]) + "; exit"], None


def _check_worker_frame(path: Path, action: str) -> None:
    # Stream logs rather than loading a full native training log into memory.
    prefix = "DMD_NATIVE_DISPATCH_JSON="
    with path.open(encoding="utf-8") as log:
        frames = [line.strip()[len(prefix):] for line in log if line.startswith(prefix)]
    _require(len(frames) == 1, "native dispatch must return exactly one framed result")
    value = json.loads(base64.b64decode(frames[0], validate=True))
    _require(isinstance(value, dict) and value == {"action": action, "returncode": 0}
             and type(value.get("returncode")) is int, "native dispatch worker completion differs")


def _cancel_command(node: dict[str, Any], pid: Path, *, transport: str) -> list[str]:
    """Cancel only the PID record owned by the existing native stage wrapper."""
    _require(transport in {"stdin", "argv-json-tty-v1"}, "unsupported native dispatch transport")
    cancel = Path(node["repo_root"]) / "scripts/distillschedule/cancel_remote_stage.sh"
    remote = shlex.join(["bash", str(cancel), str(pid)])
    ssh = ssh_command(node)
    if transport == "argv-json-tty-v1":
        return [*ssh[:-1], "-n", "-tt", ssh[-1], remote + "; exit"]
    return [*ssh, remote]


def execute_node(payload: dict[str, Any]) -> int:
    receipt, index = payload["receipt"], int(payload["node_rank"])
    _require(index in (0, 1), "invalid node rank")
    node = receipt["nodes"][index]
    root = Path(node["repo_root"])
    checked = validate_local(receipt, root=root, node_rank=index)
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
    # The saved asset config is authoritative: older resolved plans do not
    # copy rpc_transport into their expanded node records. Preflight both
    # requests before launching either peer or creating coordinator evidence.
    specs = receipt["asset_config"]["nodes"]
    _require(len(specs) == 2, "run requires two transport node specs")
    transports = [node.get("rpc_transport", "stdin") for node in specs]
    requests = [_worker_request(node, "_node", {"receipt": receipt, "node_rank": i},
                                transport=transports[i])
                for i, node in enumerate(receipt["nodes"])]
    collect_command, collect_input = _worker_request(receipt["nodes"][0], "_collect", receipt,
                                                    transport=transports[0])
    remote_out = Path(receipt["asset_config"]["run_root"]) / receipt["run_id"]
    out = coordinator_output if coordinator_output is not None else remote_out
    out.mkdir(parents=True, exist_ok=False)
    write_new_json(out / "coordinator-contract.json", receipt)
    jobs: list[tuple[Any, Any, dict[str, Any]]] = []
    validated_frames: set[int] = set()
    try:
        for i, node in enumerate(receipt["nodes"]):
            command, stdin_payload = requests[i]
            log = (out / f"ssh-node{i}.log").open("x")
            try:
                proc = subprocess.Popen(command, stdin=(subprocess.PIPE if stdin_payload is not None
                                                        else subprocess.DEVNULL),
                                        stdout=log, stderr=subprocess.STDOUT, text=True)
            except BaseException:
                log.close()
                raise
            jobs.append((proc, log, node))
            if stdin_payload is not None:
                assert proc.stdin is not None
                proc.stdin.write(stdin_payload)
                proc.stdin.close()
        deadline = time.monotonic() + receipt["asset_config"].get("timeout_seconds", 18000)
        while any(p.poll() is None for p, _, _ in jobs):
            failed = [p.returncode for p, _, _ in jobs if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"peer exited {failed}; cancelling only owned process trees")
            # A gateway's zero exit is not worker completion; fail before
            # waiting on a rank whose peer never completed the native worker.
            for i, (proc, log, _) in enumerate(jobs):
                if transports[i] == "argv-json-tty-v1" and i not in validated_frames and proc.poll() == 0:
                    log.flush()
                    _check_worker_frame(out / f"ssh-node{i}.log", "_node")
                    validated_frames.add(i)
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded workload timeout; cancelling only owned process trees")
            time.sleep(0.25)
        codes = [p.returncode for p, _, _ in jobs]
        _require(codes == [0, 0], f"node launcher exit codes: {codes}")
        # Also cover peers that both exit before the polling loop observes them.
        for i, (_, log, _) in enumerate(jobs):
            if transports[i] == "argv-json-tty-v1" and i not in validated_frames:
                log.flush()
                _check_worker_frame(out / f"ssh-node{i}.log", "_node")
        if collect_input is None:
            path = out / "ssh-collect.log"
            with path.open("x") as log:
                subprocess.run(collect_command, stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, text=True, check=True, timeout=300)
            _check_worker_frame(path, "_collect")
        else:
            subprocess.run(collect_command, input=collect_input, text=True, check=True, timeout=300)
        # Do NOT manufacture a correctness PASS or speed number from exit status.
        write_new_json(out / "dispatch-result.json", {
            "status": "DISPATCH_AND_NATIVE_SUMMARY_COMPLETE", "node_exit_codes": codes,
            "integration_head": receipt["integration_head"], "formal_speedup_eligible": False,
        })
        print(f"Dispatch completed. Native/bound summaries written in {out}; no historical speedup is reasserted.")
        return 0
    except BaseException:
        for i, (proc, _, node) in enumerate(jobs):
            try:
                options = {"stdin": subprocess.DEVNULL} if transports[i] == "argv-json-tty-v1" else {}
                subprocess.run(_cancel_command(node, remote_out / f"owned-node{i}.pid", transport=transports[i]),
                               timeout=60, check=False, **options)
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
