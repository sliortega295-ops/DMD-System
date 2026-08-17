#!/usr/bin/env python3
"""Launch one pinned LightX2V MiniMax-H3 DMD cycle on the BAAI 2x8 A100 cluster."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _hosts(value: str) -> tuple[str, str]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(items) != 2 or items[0] == items[1]:
        raise argparse.ArgumentTypeError("--hosts requires two distinct SSH aliases")
    return items  # type: ignore[return-value]


def _ssh(host: str, command: str) -> list[str]:
    return ["ssh", "-4", "-o", "BatchMode=yes", "-o", "UpdateHostKeys=no", host, "bash", "-lc", shlex.quote(command)]


def _resolve_master(host: str) -> str:
    values = subprocess.check_output(_ssh(host, "hostname -I"), text=True, timeout=30).split()
    ipv4 = [value for value in values if ":" not in value]
    if not ipv4:
        raise RuntimeError("node0 has no routable IPv4 address")
    return ipv4[0]


@dataclass
class Worker:
    node: int
    host: str
    command: list[str]
    log: Path
    process: subprocess.Popen[bytes] | None = None
    handle: Any = None

    def start(self) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log.open("wb")
        self.process = subprocess.Popen(self.command, stdout=self.handle, stderr=subprocess.STDOUT, start_new_session=True)

    def poll(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def terminate_transport(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--hosts", type=_hosts, default=("baai-helios-node0", "baai-helios-node1"))
    p.add_argument("--run-id", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--prompt-cache", type=Path, required=True)
    p.add_argument("--python", dest="python_bin", type=Path, required=True)
    p.add_argument("--attention-backend", default="_flash_3_hub")
    p.add_argument("--seed", type=int, default=20260817)
    p.add_argument("--end-step-idx", type=int, choices=range(4), default=3)
    p.add_argument("--master-addr")
    p.add_argument("--master-port", type=int, default=29970)
    p.add_argument(
        "--remote-root",
        type=Path,
        default=Path("/share/project/shuchenweng/yongyan_liu/DMD-System_fsdp2_p0_20260815"),
    )
    p.add_argument("--benchmark-source-root", type=Path)
    p.add_argument(
        "--local-evidence-root",
        type=Path,
        default=Path("/home/lyy/Helios/research_artifacts/DMD-System/lightx2v_minimax"),
    )
    p.add_argument("--timeout-seconds", type=int, default=30000)
    p.add_argument("--dry-run", action="store_true")
    return p


def _source_root(args: argparse.Namespace) -> Path:
    return args.benchmark_source_root or args.remote_root / "artifacts/source/lightx2v-minimax-d034-benchmark"


def _pid_file(args: argparse.Namespace, node: int) -> Path:
    return args.remote_root / "artifacts/runs" / args.run_id / "lightx2v_minimax" / "controller" / f"node_{node}.pid"


def _prepare(args: argparse.Namespace) -> None:
    script = args.remote_root / "scripts/distillschedule/prepare_lightx2v_minimax.sh"
    exports = {
        "DMD_SYSTEM_ROOT": str(args.remote_root),
        "LIGHTX2V_BENCHMARK_SOURCE_ROOT": str(_source_root(args)),
        "LIGHTX2V_PYTHON": str(args.python_bin),
    }
    command = "set -Eeuo pipefail; export " + " ".join(f"{k}={shlex.quote(v)}" for k, v in exports.items())
    command += "; " + shlex.quote(str(script))
    subprocess.run(_ssh(args.hosts[0], command), check=True, timeout=900)


def _command(args: argparse.Namespace, node: int, master: str) -> list[str]:
    launcher = args.remote_root / "scripts/distillschedule/launch_lightx2v_minimax_world16_node.sh"
    wrapper = args.remote_root / "scripts/distillschedule/remote_stage_wrapper.sh"
    exports = {
        "DMD_SYSTEM_ROOT": str(args.remote_root),
        "LIGHTX2V_BENCHMARK_SOURCE_ROOT": str(_source_root(args)),
        "LIGHTX2V_PYTHON": str(args.python_bin),
        "LIGHTX2V_MINIMAX_MODEL_PATH": str(args.model_path),
        "LIGHTX2V_MINIMAX_PROMPT_CACHE": str(args.prompt_cache),
        "LIGHTX2V_MINIMAX_ATTENTION_BACKEND": args.attention_backend,
        "LIGHTX2V_BENCHMARK_SEED": str(args.seed),
        "LIGHTX2V_BENCHMARK_END_STEP_IDX": str(args.end_step_idx),
    }
    launch = " ".join(
        shlex.quote(str(value))
        for value in ("bash", launcher, node, args.run_id, args.expected_head, master, args.master_port)
    )
    remote = "set -Eeuo pipefail; export " + " ".join(f"{k}={shlex.quote(v)}" for k, v in exports.items())
    remote += "; exec " + shlex.quote(str(wrapper)) + " " + shlex.quote(str(_pid_file(args, node))) + " " + launch
    return _ssh(args.hosts[node], remote)


def _cancel(args: argparse.Namespace, node: int) -> None:
    script = args.remote_root / "scripts/distillschedule/cancel_remote_stage.sh"
    subprocess.run(
        _ssh(args.hosts[node], f"{shlex.quote(str(script))} {shlex.quote(str(_pid_file(args, node)))}"),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=90,
    )


def _summarize(args: argparse.Namespace, local: Path) -> dict[str, Any]:
    run_root = args.remote_root / "artifacts/runs" / args.run_id / "lightx2v_minimax"
    summary = run_root / "summary.json"
    summarizer = args.remote_root / "scripts/distillschedule/summarize_lightx2v_minimax_world16.py"
    contract = args.remote_root / "configs/distillschedule/baselines/lightx2v_minimax_h3_world16_dmd_contract.json"
    command = (
        f"python3 {shlex.quote(str(summarizer))} "
        f"--run-root {shlex.quote(str(run_root))} "
        f"--contract {shlex.quote(str(contract))} "
        f"--expected-head {shlex.quote(args.expected_head)} "
        f"--output {shlex.quote(str(summary))}"
    )
    completed = subprocess.run(_ssh(args.hosts[0], command), check=False, timeout=300)
    raw = subprocess.check_output(_ssh(args.hosts[0], f"cat {shlex.quote(str(summary))}"), timeout=60)
    local.mkdir(parents=True, exist_ok=True)
    (local / "summary.json").write_bytes(raw)
    value = json.loads(raw)
    if completed.returncode != 0 or value.get("status") != "PASS":
        raise RuntimeError(f"LightX2V MiniMax summary failed: {value.get('errors')}")
    return value


def main() -> int:
    args = parser().parse_args()
    if not args.run_id.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("unsafe run id")
    if len(args.expected_head) != 40:
        raise ValueError("--expected-head must be a full Git SHA")
    local = args.local_evidence_root / args.run_id
    if local.exists():
        raise FileExistsError(local)
    master = args.master_addr or _resolve_master(args.hosts[0])
    commands = [_command(args, node, master) for node in (0, 1)]
    if args.dry_run:
        print(json.dumps({"master": master, "source_root": str(_source_root(args)), "commands": commands}, indent=2))
        return 0

    _prepare(args)
    workers = [Worker(node, args.hosts[node], commands[node], local / f"node_{node}.ssh.log") for node in (0, 1)]
    local.mkdir(parents=True)
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + args.timeout_seconds
        failure = False
        while time.monotonic() < deadline:
            codes = [worker.poll() for worker in workers]
            if all(code is not None for code in codes):
                break
            if any(code not in (None, 0) for code in codes):
                failure = True
                break
            time.sleep(2)
        else:
            failure = True
        codes = [worker.poll() for worker in workers]
        if failure or codes != [0, 0]:
            for worker in workers:
                _cancel(args, worker.node)
                worker.terminate_transport()
            raise RuntimeError(f"LightX2V MiniMax remote run failed or timed out: returncodes={codes}")
        value = _summarize(args, local)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
