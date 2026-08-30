#!/usr/bin/env python3
"""Central two-node controller for one immutable LiveWan cross-scale run."""

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


def hosts(value: str) -> tuple[str, str]:
    rows = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(rows) != 2 or rows[0] == rows[1]:
        raise argparse.ArgumentTypeError("--hosts requires two distinct SSH aliases")
    return rows  # type: ignore[return-value]


def ssh(host: str, command: str, *, port: int) -> list[str]:
    return [
        "ssh", "-4", "-p", str(port),
        "-o", "ControlMaster=no", "-o", "ControlPath=none",
        "-o", "BatchMode=yes", "-o", "UpdateHostKeys=no", host,
        "bash", "-lc", shlex.quote(command),
    ]


def resolve_master(host: str, *, port: int) -> str:
    values = subprocess.check_output(
        ssh(host, "hostname -I", port=port), text=True, timeout=30
    ).split()
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
        self.process = subprocess.Popen(
            self.command, stdout=self.handle, stderr=subprocess.STDOUT,
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

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--hosts", type=hosts, required=True)
    value.add_argument("--ssh-port", type=int, default=2222)
    value.add_argument("--run-id", required=True)
    value.add_argument("--run-kind", choices=("validate", "correctness", "timing"), required=True)
    value.add_argument("--execution-mode", choices=("baseline", "pipeline"), required=True)
    value.add_argument("--expected-head", required=True)
    value.add_argument("--remote-root", required=True, type=Path)
    value.add_argument("--student-checkpoint", required=True, type=Path)
    value.add_argument("--critic-checkpoint", required=True, type=Path)
    value.add_argument("--teacher-checkpoint", required=True, type=Path)
    value.add_argument("--trace", required=True, type=Path)
    value.add_argument("--bootstrap", required=True, type=Path)
    value.add_argument("--source-root", required=True, type=Path)
    value.add_argument("--source-receipt", required=True, type=Path)
    value.add_argument("--pyenv", type=Path, default=Path("/tmp/distillschedule-helios-official-torch210"))
    value.add_argument("--correctness-receipt", type=Path)
    value.add_argument("--correctness-sha256")
    value.add_argument("--master-addr")
    value.add_argument("--master-port", type=int, default=29930)
    value.add_argument("--timeout-seconds", type=int, default=21600)
    value.add_argument(
        "--local-evidence-root", type=Path,
        default=Path("/home/lyy/Helios/research_artifacts/DMD-System/livewan_crossscale_20260830"),
    )
    value.add_argument("--dry-run", action="store_true")
    return value


def pid_file(args: argparse.Namespace, node: int) -> Path:
    return args.remote_root / "artifacts/runs" / args.run_id / "controller" / f"node_{node}.pid"


def command(args: argparse.Namespace, node: int, master: str) -> list[str]:
    run_root = args.remote_root / "artifacts/runs" / args.run_id
    launcher = args.remote_root / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh"
    wrapper = args.remote_root / "scripts/distillschedule/remote_stage_wrapper.sh"
    exports = {
        "DMD_SYSTEM_ROOT": args.remote_root,
        "DISTILLSCHEDULE_PYENV": args.pyenv,
        "DISTILLSCHEDULE_OUTPUT_DIR": run_root,
        "LIVEWAN_STUDENT_CHECKPOINT": args.student_checkpoint,
        "LIVEWAN_CRITIC_CHECKPOINT": args.critic_checkpoint,
        "LIVEWAN_TEACHER_CHECKPOINT": args.teacher_checkpoint,
        "LIVEWAN_OPERATION_TRACE": args.trace,
        "LIVEWAN_SYSTEMS_BOOTSTRAP": args.bootstrap,
        "LIVEWAN_WAN21_SOURCE": args.source_root,
        "LIVEWAN_SOURCE_RECEIPT": args.source_receipt,
    }
    if args.run_kind == "timing":
        if args.correctness_receipt is None or not args.correctness_sha256:
            raise ValueError("timing requires paired correctness receipt and SHA256")
        exports["LIVEWAN_PAIR_CORRECTNESS_RECEIPT"] = args.correctness_receipt
        exports["LIVEWAN_PAIR_CORRECTNESS_SHA256"] = args.correctness_sha256
    export_text = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in exports.items())
    launch = " ".join(
        shlex.quote(str(value))
        for value in (
            "bash", launcher, node, args.run_id, args.run_kind,
            args.execution_mode, args.expected_head, master, args.master_port,
        )
    )
    remote = (
        "set -Eeuo pipefail; export " + export_text + "; exec "
        + shlex.quote(str(wrapper)) + " " + shlex.quote(str(pid_file(args, node)))
        + " " + launch
    )
    return ssh(args.hosts[node], remote, port=args.ssh_port)


def cancel(args: argparse.Namespace, node: int) -> None:
    script = args.remote_root / "scripts/distillschedule/cancel_remote_stage.sh"
    try:
        subprocess.run(
            ssh(
                args.hosts[node],
                f"{shlex.quote(str(script))} {shlex.quote(str(pid_file(args, node)))}",
                port=args.ssh_port,
            ),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=60,
        )
    except subprocess.TimeoutExpired:
        pass


def summarize(args: argparse.Namespace, local: Path) -> dict[str, Any]:
    run_root = args.remote_root / "artifacts/runs" / args.run_id
    summary = run_root / "summary.json"
    script = args.remote_root / "scripts/distillschedule/summarize_livewan_crossscale_world16.py"
    remote = (
        f"export PYTHONPATH={shlex.quote(str(args.remote_root / 'src'))}:{shlex.quote(str(args.remote_root))}; "
        f"{shlex.quote(str(args.pyenv / 'bin/python'))} {shlex.quote(str(script))} "
        f"--run-root {shlex.quote(str(run_root))} "
        f"--execution-mode {shlex.quote(args.execution_mode)} "
        f"--run-kind {shlex.quote(args.run_kind)} "
        f"--trace {shlex.quote(str(args.trace))} "
        f"--expected-head {shlex.quote(args.expected_head)} "
        f"--output {shlex.quote(str(summary))}"
    )
    completed = subprocess.run(
        ssh(args.hosts[0], remote, port=args.ssh_port), check=False, timeout=900
    )
    raw = subprocess.check_output(
        ssh(
            args.hosts[0], f"cat {shlex.quote(str(summary))}", port=args.ssh_port
        ),
        timeout=60,
    )
    local.mkdir(parents=True, exist_ok=True)
    (local / "summary.json").write_bytes(raw)
    value = json.loads(raw)
    if completed.returncode != 0 or value.get("status") != "PASS":
        raise RuntimeError(f"LiveWan summary failed: {value.get('errors')}")
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
    master = args.master_addr or resolve_master(args.hosts[0], port=args.ssh_port)
    commands = [command(args, node, master) for node in (0, 1)]
    if args.dry_run:
        print(json.dumps({"master": master, "commands": commands}, indent=2))
        return 0
    workers = [
        Worker(node, args.hosts[node], commands[node], local / f"node_{node}.ssh.log")
        for node in (0, 1)
    ]
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
            for worker in workers:
                cancel(args, worker.node)
                worker.terminate_transport()
            raise RuntimeError(f"LiveWan run failed or timed out: {codes}")
        result = summarize(args, local)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
