#!/usr/bin/env python3
"""Centrally launch/cancel/summarize one static FSDP2 world16 run."""
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
    return [
        "ssh", "-4", "-o", "BatchMode=yes", "-o", "UpdateHostKeys=no",
        host, "bash", "-lc", shlex.quote(command),
    ]


def _resolve_master(host: str) -> str:
    text = subprocess.check_output(_ssh(host, "hostname -I"), text=True, timeout=30)
    values = [item for item in text.split() if ":" not in item]
    if not values:
        raise RuntimeError("node0 has no routable IPv4 address")
    return values[0]


@dataclass
class Worker:
    rank: int
    host: str
    command: list[str]
    log: Path
    process: subprocess.Popen[bytes] | None = None
    handle: Any = None

    def start(self) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log.open("wb")
        self.process = subprocess.Popen(
            self.command,
            stdout=self.handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def poll(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def terminate_transport(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--hosts", type=_hosts, default=("baai-helios-node0", "baai-helios-node1"))
    value.add_argument("--run-id", required=True)
    value.add_argument("--variant", choices=("no-gan", "gan"), required=True)
    value.add_argument("--mode", choices=("validate", "correctness", "timing", "profile"), required=True)
    value.add_argument("--expected-head", required=True)
    value.add_argument("--remote-root", type=Path, default=Path("/share/project/shuchenweng/yongyan_liu/DMD-System_fsdp2_static_20260814"))
    value.add_argument("--master-addr")
    value.add_argument("--master-port", type=int, default=29940)
    value.add_argument("--correctness-summary", type=Path)
    value.add_argument("--correctness-sha256")
    value.add_argument("--local-evidence-root", type=Path, default=Path("/home/lyy/Helios/research_artifacts/DMD-System/fsdp2_static"))
    value.add_argument("--timeout-seconds", type=int, default=30000)
    value.add_argument("--dry-run", action="store_true")
    return value


def _pid_file(args: argparse.Namespace, node: int) -> Path:
    return args.remote_root / "artifacts" / "runs" / args.run_id / "controller" / f"node_{node}.pid"


def _command(args: argparse.Namespace, node: int, master: str) -> list[str]:
    launcher = args.remote_root / "scripts/distillschedule/launch_static_fsdp2_full5_world16_node.sh"
    wrapper = args.remote_root / "scripts/distillschedule/remote_stage_wrapper.sh"
    exports = [f"DMD_SYSTEM_ROOT={shlex.quote(str(args.remote_root))}"]
    if args.mode == "timing":
        if args.correctness_summary is None or not args.correctness_sha256:
            raise ValueError("timing requires pinned correctness summary and SHA256")
        exports += [
            f"DISTILLSCHEDULE_FSDP2_CORRECTNESS_SUMMARY={shlex.quote(str(args.correctness_summary))}",
            f"DISTILLSCHEDULE_FSDP2_CORRECTNESS_SHA256={shlex.quote(args.correctness_sha256)}",
        ]
    launch = " ".join(
        shlex.quote(str(item))
        for item in (
            "bash", str(launcher), str(node), args.run_id, args.variant, args.mode,
            args.expected_head, master, str(args.master_port),
        )
    )
    remote = (
        "set -Eeuo pipefail; export " + " ".join(exports) + "; exec "
        + shlex.quote(str(wrapper)) + " " + shlex.quote(str(_pid_file(args, node)))
        + " " + launch
    )
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
    root = args.remote_root / "artifacts" / "runs" / args.run_id
    output = root / "summary.json"
    script = args.remote_root / "scripts/distillschedule/summarize_static_fsdp2_full5_world16.py"
    pythonpath = f"{args.remote_root}/src:{args.remote_root}"
    command = (
        f"export PYTHONPATH={shlex.quote(pythonpath)}; "
        f"/tmp/distillschedule-helios-official-torch210/bin/python {shlex.quote(str(script))} "
        f"--run-root {shlex.quote(str(root))} --variant {shlex.quote(args.variant)} "
        f"--output {shlex.quote(str(output))}"
    )
    completed = subprocess.run(_ssh(args.hosts[0], command), check=False, timeout=300)
    raw = subprocess.check_output(_ssh(args.hosts[0], f"cat {shlex.quote(str(output))}"), timeout=60)
    local.mkdir(parents=True, exist_ok=True)
    (local / "summary.json").write_bytes(raw)
    value = json.loads(raw)
    if completed.returncode != 0 or value.get("status") != "PASS":
        raise RuntimeError(f"static FSDP2 summary failed: {value.get('errors')}")
    return value


def main() -> int:
    args = parser().parse_args()
    if not args.run_id.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("unsafe run id")
    local = args.local_evidence_root / args.run_id
    if local.exists():
        raise FileExistsError(local)
    master = args.master_addr or _resolve_master(args.hosts[0])
    commands = [_command(args, node, master) for node in (0, 1)]
    if args.dry_run:
        print(json.dumps({"master_addr": master, "commands": commands}, indent=2))
        return 0
    workers = [Worker(node, args.hosts[node], commands[node], local / f"node_{node}.ssh.log") for node in (0, 1)]
    local.mkdir(parents=True)
    for worker in workers:
        worker.start()
    deadline = time.monotonic() + args.timeout_seconds
    failure: Worker | None = None
    while time.monotonic() < deadline:
        codes = [worker.poll() for worker in workers]
        if all(code is not None for code in codes):
            break
        for worker, code in zip(workers, codes, strict=True):
            if code not in (None, 0):
                failure = worker
                break
        if failure is not None:
            break
        time.sleep(2)
    else:
        failure = workers[0]
    if failure is not None or [worker.poll() for worker in workers] != [0, 0]:
        for worker in workers:
            _cancel(args, worker.rank)
            worker.terminate_transport()
        raise RuntimeError("static FSDP2 remote run failed or timed out")
    value = _summarize(args, local)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
