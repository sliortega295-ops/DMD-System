#!/usr/bin/env python3
"""Run one bounded world16 MiniMax FSDP-first construction gate."""
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


def _hosts(value: str) -> tuple[str, str]:
    items = tuple(x.strip() for x in value.split(",") if x.strip())
    if len(items) != 2 or items[0] == items[1]:
        raise argparse.ArgumentTypeError("--hosts requires two distinct SSH aliases")
    return items  # type: ignore[return-value]


def _ssh(host: str, command: str) -> list[str]:
    return ["ssh", "-4", "-o", "BatchMode=yes", "-o", "UpdateHostKeys=no", host, "bash", "-lc", shlex.quote(command)]


def _master(host: str) -> str:
    values = subprocess.check_output(_ssh(host, "hostname -I"), text=True, timeout=30).split()
    ipv4 = [x for x in values if ":" not in x]
    if not ipv4:
        raise RuntimeError("node0 has no IPv4 address")
    return ipv4[0]


@dataclass
class Worker:
    node: int
    command: list[str]
    log: Path
    proc: subprocess.Popen[bytes] | None = None
    handle: object | None = None

    def start(self) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log.open("wb")
        self.proc = subprocess.Popen(self.command, stdout=self.handle, stderr=subprocess.STDOUT, start_new_session=True)

    def poll(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def terminate(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()  # type: ignore[union-attr]
            self.handle = None


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--gate", choices=("A", "B"), required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--prompt-cache", type=Path, required=True)
    p.add_argument("--python", dest="python_bin", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--remote-root", type=Path, required=True)
    p.add_argument("--hosts", type=_hosts, default=("baai-helios-node0", "baai-helios-node1"))
    p.add_argument("--master-port", type=int, default=29981)
    p.add_argument("--local-evidence-root", type=Path, default=Path("/home/lyy/Helios/research_artifacts/DMD-System/lightx2v_minimax_fsdp_first_init_20260817"))
    p.add_argument("--timeout-seconds", type=int, default=36000)
    p.add_argument("--dry-run", action="store_true")
    return p


def _remote_run_root(args: argparse.Namespace) -> Path:
    return args.remote_root / "artifacts/runs" / args.run_id / "fsdp_first" / args.gate


def _command(args: argparse.Namespace, node: int, master: str) -> list[str]:
    launcher = args.remote_root / "scripts/distillschedule/launch_lightx2v_minimax_fsdp_first_node.sh"
    wrapper = args.remote_root / "scripts/distillschedule/remote_stage_wrapper.sh"
    pid = _remote_run_root(args) / "controller" / f"node_{node}.pid"
    exports = {
        "DMD_SYSTEM_ROOT": str(args.remote_root),
        "LIGHTX2V_PYTHON": str(args.python_bin),
        "LIGHTX2V_MINIMAX_MODEL_PATH": str(args.model_path),
        "LIGHTX2V_MINIMAX_PROMPT_CACHE": str(args.prompt_cache),
        "LIGHTX2V_BENCHMARK_SOURCE_ROOT": str(args.source_root),
        "LIGHTX2V_MINIMAX_ATTENTION_BACKEND": "_flash_3_hub",
    }
    launch = " ".join(shlex.quote(str(x)) for x in ("bash", launcher, node, args.run_id, args.gate, args.expected_head, master, args.master_port))
    remote = "set -Eeuo pipefail; export " + " ".join(f"{k}={shlex.quote(v)}" for k, v in exports.items())
    remote += "; exec " + shlex.quote(str(wrapper)) + " " + shlex.quote(str(pid)) + " " + launch
    return _ssh(args.hosts[node], remote)


def _cancel(args: argparse.Namespace, node: int) -> None:
    script = args.remote_root / "scripts/distillschedule/cancel_remote_stage.sh"
    pid = _remote_run_root(args) / "controller" / f"node_{node}.pid"
    subprocess.run(_ssh(args.hosts[node], f"{shlex.quote(str(script))} {shlex.quote(str(pid))}"), check=False, timeout=90)


def _collect(args: argparse.Namespace, local: Path) -> None:
    remote_root = _remote_run_root(args)
    local.mkdir(parents=True, exist_ok=True)
    archive = local / "remote_gate_artifacts.tar"
    command = f"tar -cf - -C {shlex.quote(str(remote_root))} ."
    with archive.open("wb") as handle:
        completed = subprocess.run(_ssh(args.hosts[0], command), stdout=handle, check=False, timeout=120)
    if completed.returncode != 0:
        raise RuntimeError("failed to collect remote gate artifacts")
    subprocess.run(["tar", "-xf", str(archive), "-C", str(local)], check=True)
    archive.unlink()


def main() -> int:
    args = _parser().parse_args()
    if len(args.expected_head) != 40:
        raise ValueError("--expected-head must be a full SHA")
    local = args.local_evidence_root / args.run_id / f"gate_{args.gate}"
    if local.exists():
        raise FileExistsError(local)
    master = _master(args.hosts[0])
    commands = [_command(args, node, master) for node in (0, 1)]
    if args.dry_run:
        print(json.dumps({"master": master, "commands": commands}, indent=2))
        return 0
    workers = [Worker(node, commands[node], local / f"node_{node}.ssh.log") for node in (0, 1)]
    local.mkdir(parents=True)
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + args.timeout_seconds
        failed = False
        while time.monotonic() < deadline:
            codes = [worker.poll() for worker in workers]
            if all(code is not None for code in codes):
                break
            if any(code not in (None, 0) for code in codes):
                failed = True
                break
            time.sleep(2)
        else:
            failed = True
        codes = [worker.poll() for worker in workers]
        if failed or codes != [0, 0]:
            for node, worker in enumerate(workers):
                _cancel(args, node)
                worker.terminate()
            raise RuntimeError(f"FSDP-first Gate {args.gate} failed: returncodes={codes}")
        _collect(args, local)
        print(json.dumps({"run_id": args.run_id, "gate": args.gate, "status": "PASS", "evidence": str(local)}, indent=2))
        return 0
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
