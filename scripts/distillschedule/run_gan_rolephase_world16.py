#!/usr/bin/env python3
"""Centrally launch/cancel/summarize one real GAN role-phase world16 run."""
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
from typing import Any, Sequence


def parse_hosts(value: str) -> tuple[str, str]:
    hosts = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(hosts) != 2 or hosts[0] == hosts[1]:
        raise argparse.ArgumentTypeError("--hosts requires two distinct SSH aliases")
    return hosts  # type: ignore[return-value]


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def ssh_command(host: str, command: str) -> list[str]:
    return ["ssh", host, "bash", "-lc", shlex.quote(command)]


@dataclass
class Worker:
    node_rank: int
    host: str
    command: list[str]
    log_path: Path
    process: subprocess.Popen[bytes] | None = None
    handle: Any = None
    return_code: int | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log_path.open("wb")
        self.process = subprocess.Popen(
            self.command,
            stdout=self.handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def poll(self) -> int | None:
        if self.process is None:
            return self.return_code
        value = self.process.poll()
        if value is not None:
            self.return_code = int(value)
            if self.handle is not None:
                self.handle.flush()
                self.handle.close()
                self.handle = None
        return self.return_code

    def terminate_transport(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--hosts",
        type=parse_hosts,
        default=("baai-helios-node0", "baai-helios-node1"),
    )
    value.add_argument("--run-id", required=True)
    value.add_argument("--mode", choices=("validate", "correctness", "timing"), required=True)
    value.add_argument(
        "--remote-root",
        type=Path,
        default=Path("/share/project/shuchenweng/yongyan_liu/DMD-System_gan_rolephase_world16_exec_20260812"),
    )
    value.add_argument(
        "--helios-root",
        type=Path,
        default=Path("/share/project/shuchenweng/yongyan_liu/DistillSchedule_gan_rolephase_exec_20260812"),
    )
    value.add_argument("--expected-head", required=True)
    value.add_argument("--source-identity-hash", required=True)
    value.add_argument("--master-addr")
    value.add_argument("--master-port", type=int, default=29880)
    value.add_argument(
        "--phase-b-streamed-blocks", type=int, choices=(0, 4, 10), default=0
    )
    value.add_argument("--correctness-summary", type=Path)
    value.add_argument("--correctness-sha256")
    value.add_argument("--timeout-seconds", type=int, default=30000)
    value.add_argument(
        "--local-evidence-root",
        type=Path,
        default=Path("/home/lyy/Helios/research_artifacts/DistillSchedule/gan_rolephase_world16"),
    )
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--summarize-only", action="store_true")
    return value


def resolve_master(host: str) -> str:
    text = subprocess.check_output(
        ["ssh", host, "hostname -I"], text=True, timeout=30
    ).strip()
    addresses = [item for item in text.split() if ":" not in item]
    if not addresses:
        raise RuntimeError("node0 has no routable IPv4 address")
    return addresses[0]


def pid_file(args: argparse.Namespace, rank: int) -> Path:
    return args.remote_root / "artifacts" / "runs" / args.run_id / "controller" / f"node_{rank}.pid"


def worker_command(args: argparse.Namespace, rank: int, master_addr: str) -> list[str]:
    launcher = args.remote_root / "scripts" / "distillschedule" / "launch_gan_rolephase_world16_node.sh"
    wrapper = args.remote_root / "scripts" / "distillschedule" / "remote_stage_wrapper.sh"
    command = [
        str(launcher), str(rank), args.run_id, args.mode, args.expected_head,
        master_addr, str(args.master_port), args.source_identity_hash,
    ]
    exports = [
        f"DMD_SYSTEM_ROOT={shlex.quote(str(args.remote_root))}",
        f"HELIOS_RUNTIME_ROOT={shlex.quote(str(args.helios_root))}",
        f"DISTILLSCHEDULE_PHASE_B_STREAMED_BLOCKS={args.phase_b_streamed_blocks}",
    ]
    if args.mode == "timing":
        if args.correctness_summary is None or not args.correctness_sha256:
            raise ValueError("timing requires --correctness-summary and --correctness-sha256")
        exports += [
            f"DISTILLSCHEDULE_CORRECTNESS_SUMMARY={shlex.quote(str(args.correctness_summary))}",
            f"DISTILLSCHEDULE_CORRECTNESS_SHA256={shlex.quote(args.correctness_sha256)}",
        ]
    remote = (
        "set -Eeuo pipefail; export " + " ".join(exports) + "; exec "
        + shlex.quote(str(wrapper)) + " " + shlex.quote(str(pid_file(args, rank)))
        + " " + shell_join(command)
    )
    return ssh_command(args.hosts[rank], remote)


def cancel_remote(args: argparse.Namespace, rank: int) -> None:
    cancel = args.remote_root / "scripts" / "distillschedule" / "cancel_remote_stage.sh"
    command = f"{shlex.quote(str(cancel))} {shlex.quote(str(pid_file(args, rank)))}"
    subprocess.run(
        ssh_command(args.hosts[rank], command),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=90,
    )


def summarize(args: argparse.Namespace, local_dir: Path) -> dict[str, Any]:
    run_root = args.remote_root / "artifacts" / "runs" / args.run_id
    remote_output = run_root / "summary.json"
    summarizer = args.remote_root / "scripts" / "distillschedule" / "summarize_gan_rolephase_world16.py"
    command = shell_join([
        "/tmp/distillschedule-helios-official-torch210/bin/python",
        str(summarizer), "--run-root", str(run_root), "--output", str(remote_output),
    ])
    completed = subprocess.run(
        ssh_command(args.hosts[0], command), check=False, timeout=300
    )
    raw = subprocess.check_output(
        ["ssh", args.hosts[0], "cat", str(remote_output)], timeout=60
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "summary.json").write_bytes(raw)
    value = json.loads(raw)
    if completed.returncode != 0 or value.get("status") != "PASS":
        raise RuntimeError(f"summary failed: {value.get('errors')}")
    return value


def main() -> int:
    args = parser().parse_args()
    if not args.run_id.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("run id contains unsafe characters")
    local_dir = args.local_evidence_root / args.run_id
    if local_dir.exists() and not args.summarize_only:
        raise FileExistsError(local_dir)
    master_addr = args.master_addr or resolve_master(args.hosts[0])
    commands = [worker_command(args, rank, master_addr) for rank in (0, 1)]
    if args.dry_run:
        print(json.dumps({"master_addr": master_addr, "commands": commands}, indent=2))
        return 0
    if args.summarize_only:
        print(json.dumps(summarize(args, local_dir), indent=2, sort_keys=True))
        return 0

    workers = [
        Worker(rank, args.hosts[rank], commands[rank], local_dir / f"node_{rank}.ssh.log")
        for rank in (0, 1)
    ]
    local_dir.mkdir(parents=True)
    (local_dir / "controller.json").write_text(json.dumps({
        "schema_version": "distillschedule.gan_rolephase_controller.v1",
        "run_id": args.run_id,
        "mode": args.mode,
        "hosts": list(args.hosts),
        "master_addr": master_addr,
        "master_port": args.master_port,
        "expected_head": args.expected_head,
        "source_identity_hash": args.source_identity_hash,
        "phase_b_streamed_blocks": args.phase_b_streamed_blocks,
        "commands": commands,
    }, indent=2, sort_keys=True) + "\n")
    for worker in workers:
        worker.start()
    deadline = time.monotonic() + args.timeout_seconds
    failed: Worker | None = None
    while time.monotonic() < deadline:
        codes = [worker.poll() for worker in workers]
        if all(code is not None for code in codes):
            break
        for worker, code in zip(workers, codes, strict=True):
            if code not in (None, 0):
                failed = worker
                break
        if failed is not None:
            break
        time.sleep(2)
    else:
        failed = workers[0]

    if failed is not None:
        for worker in workers:
            cancel_remote(args, worker.node_rank)
            worker.terminate_transport()
        raise RuntimeError(f"node{failed.node_rank} failed or controller timed out")
    codes = [worker.poll() for worker in workers]
    if codes != [0, 0]:
        for worker in workers:
            cancel_remote(args, worker.node_rank)
        raise RuntimeError(f"remote return codes: {codes}")
    value = summarize(args, local_dir)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
