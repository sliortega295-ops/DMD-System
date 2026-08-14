#!/usr/bin/env python3
"""One-command central coordinator for the conservative Helios GAN Route-A baseline."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "distillschedule.routea_gan_world16.controller.v1"
PLAN = "configs/distillschedule/plans/helios_routea_gan_full5_world16_baseline.json"
LAUNCHER = "scripts/distillschedule/launch_routea_gan_world16_node.sh"
SUMMARIZER = "scripts/distillschedule/summarize_routea_gan_world16.py"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def parse_hosts(value: str) -> tuple[str, str]:
    hosts = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(hosts) != 2 or hosts[0] == hosts[1]:
        raise argparse.ArgumentTypeError("--hosts must contain two distinct hosts")
    return hosts  # type: ignore[return-value]


def shell_join(items: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in items)


def remote(ssh: str, host: str, command: str) -> list[str]:
    return [ssh, host, "bash", "-lc", shlex.quote(command)]


def clean_tree(root: Path) -> bool:
    return not subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True).strip()


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


@dataclass
class Worker:
    rank: int
    host: str
    command: list[str]
    log: Path
    process: subprocess.Popen[bytes] | None = None
    handle: Any = None
    return_code: int | None = None

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
        if self.process is None:
            return self.return_code
        rc = self.process.poll()
        if rc is not None:
            self.return_code = int(rc)
            if self.handle is not None:
                self.handle.flush(); self.handle.close(); self.handle = None
        return self.return_code

    def terminate_transport(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--hosts", required=True, type=parse_hosts)
    p.add_argument("--run-id", required=True)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--remote-root", type=Path)
    p.add_argument("--site-env", default="configs/distillschedule/server/baai_gan_2x8_a100.env.example")
    p.add_argument("--master-addr")
    p.add_argument("--master-port", type=int, default=29777)
    p.add_argument("--expected-head")
    p.add_argument("--ssh", default="ssh")
    p.add_argument("--timeout-seconds", type=int, default=20000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    return p


def pid_file(remote_root: Path, run_id: str, rank: int) -> Path:
    return remote_root / "artifacts/runs" / run_id / "controller" / f"remote_node_{rank}.pid"


def worker_command(args: argparse.Namespace, *, root: Path, remote_root: Path, head: str, plan_sha: str, rank: int, host: str) -> list[str]:
    env = Path(args.site_env)
    if not env.is_absolute():
        env = remote_root / env
    wrapper = remote_root / "scripts/distillschedule/remote_stage_wrapper.sh"
    launcher = remote_root / LAUNCHER
    command = [
        str(launcher), str(rank), args.run_id, head, plan_sha,
        args.master_addr or args.hosts[0], str(args.master_port),
    ]
    text = (
        f"set -Eeuo pipefail; source {shlex.quote(str(env))}; "
        f"export DMD_SYSTEM_ROOT={shlex.quote(str(remote_root))}; "
        f"exec {shlex.quote(str(wrapper))} {shlex.quote(str(pid_file(remote_root,args.run_id,rank)))} {shell_join(command)}"
    )
    return remote(args.ssh, host, text)


def cancel_peer(args: argparse.Namespace, remote_root: Path, rank: int, host: str) -> None:
    env = Path(args.site_env)
    if not env.is_absolute(): env = remote_root / env
    cancel = remote_root / "scripts/distillschedule/cancel_remote_stage.sh"
    text = (
        f"set -Eeuo pipefail; source {shlex.quote(str(env))}; "
        f"{shlex.quote(str(cancel))} {shlex.quote(str(pid_file(remote_root,args.run_id,rank)))}"
    )
    subprocess.run(remote(args.ssh, host, text), check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)


def run_summary(args: argparse.Namespace, *, root: Path, remote_root: Path, output: Path) -> int:
    if remote_root == root:
        return subprocess.run(
            [sys.executable, str(root / SUMMARIZER), "--run-id", args.run_id, "--root", str(root), "--output", str(output)],
            check=False,
        ).returncode
    env = Path(args.site_env)
    if not env.is_absolute(): env = remote_root / env
    remote_output = remote_root / "artifacts/manifests" / f"{args.run_id}_result.json"
    command = (
        f"set -Eeuo pipefail; source {shlex.quote(str(env))}; "
        'python_bin="${DISTILLSCHEDULE_PYTHON:-python3}"; '
        f'"$python_bin" {shlex.quote(str(remote_root / SUMMARIZER))} '
        f"--run-id {shlex.quote(args.run_id)} --root {shlex.quote(str(remote_root))} --output {shlex.quote(str(remote_output))}"
    )
    rc = subprocess.run(remote(args.ssh, args.hosts[0], command), check=False).returncode
    fetched = subprocess.run(
        remote(args.ssh, args.hosts[0], f"cat {shlex.quote(str(remote_output))}"),
        check=False, capture_output=True, text=True, timeout=90,
    )
    if fetched.returncode != 0:
        return rc if rc != 0 else int(fetched.returncode)
    value = json.loads(fetched.stdout)
    if not isinstance(value, Mapping): return 2
    atomic_json(output, value)
    return rc


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.repo_root.expanduser().resolve()
    remote_root = (args.remote_root or root).expanduser()
    plan = root / PLAN
    if not plan.is_file():
        print(json.dumps({"status":"FAIL","error":f"missing plan {plan}"})); return 2
    if not args.dry_run and not clean_tree(root):
        print(json.dumps({"status":"FAIL","error":"controller repository is dirty"})); return 2
    head = args.expected_head or git_head(root)
    plan_sha = sha256_file(plan)
    run_dir = root / "artifacts/runs" / args.run_id
    ctl = run_dir / "controller"
    manifest = ctl / "gan_controller_manifest.json"
    result = root / "artifacts/manifests" / f"{args.run_id}_result.json"
    identity = {
        "schema_version": SCHEMA,
        "run_id": args.run_id,
        "hosts": list(args.hosts),
        "head": head,
        "plan": PLAN,
        "plan_sha256": plan_sha,
        "remote_root": str(remote_root),
        "master_addr": args.master_addr or args.hosts[0],
        "master_port": args.master_port,
        "official_window": [1000,1001,1002,1003,1004],
        "baseline": "full_offload_no_overlap",
    }
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get("identity") != identity:
            print(json.dumps({"status":"FAIL","error":"immutable run id identity mismatch"})); return 2
    else:
        ctl.mkdir(parents=True, exist_ok=False)
        atomic_json(manifest, {"identity":identity,"created_at":dt.datetime.now(dt.timezone.utc).isoformat(),"state":"PREREGISTERED"})

    commands = [worker_command(args,root=root,remote_root=remote_root,head=head,plan_sha=plan_sha,rank=i,host=h) for i,h in enumerate(args.hosts)]
    if args.dry_run:
        print(json.dumps({"status":"DRY_RUN","identity":identity,"commands":commands},indent=2,sort_keys=True)); return 0
    if args.summarize_only:
        return run_summary(args,root=root,remote_root=remote_root,output=result)

    workers = [Worker(i,h,commands[i],ctl/f"ssh_node_{i}.log") for i,h in enumerate(args.hosts)]
    for worker in workers: worker.start()
    deadline = time.monotonic() + args.timeout_seconds
    try:
        while True:
            states = [worker.poll() for worker in workers]
            failures = [i for i,rc in enumerate(states) if rc is not None and rc != 0]
            if failures:
                failed = failures[0]
                peer = 1 - failed
                cancel_peer(args,remote_root,peer,args.hosts[peer])
                workers[peer].terminate_transport()
                for worker in workers:
                    if worker.process is not None: worker.process.wait(timeout=120)
                print(json.dumps({"status":"FAIL","failed_node":failed,"return_codes":[w.poll() for w in workers]})); return int(states[failed] or 2)
            if all(rc == 0 for rc in states): break
            if time.monotonic() >= deadline:
                for i,worker in enumerate(workers):
                    cancel_peer(args,remote_root,i,args.hosts[i]); worker.terminate_transport()
                print(json.dumps({"status":"FAIL","error":"controller timeout"})); return 124
            time.sleep(2)
    except BaseException:
        for i,worker in enumerate(workers):
            cancel_peer(args,remote_root,i,args.hosts[i]); worker.terminate_transport()
        raise

    rc = run_summary(args,root=root,remote_root=remote_root,output=result)
    if rc != 0:
        print(json.dumps({"status":"FAIL","error":"strict GAN summarizer rejected run","result":str(result)})); return rc
    payload = json.loads(result.read_text())
    print(json.dumps({"status":payload.get("status"),"run_id":args.run_id,"result":str(result)},sort_keys=True))
    return 0 if payload.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
