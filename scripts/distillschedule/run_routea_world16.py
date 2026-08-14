#!/usr/bin/env python3
"""Centrally launch, cancel, finalize and summarize one Route-A world16 run.

The coordinator never stores SSH credentials.  It uses the caller's existing SSH
configuration and starts one fail-closed node worker on each requested host.
"""

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
from typing import Any, Mapping, Sequence

SCHEMA = "distillschedule.routea_world16.controller.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def clean_tree(root: Path) -> bool:
    return not subprocess.check_output(
        ["git", "-C", str(root), "status", "--short"], text=True
    ).strip()


def parse_hosts(value: str) -> tuple[str, str]:
    hosts = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(hosts) != 2 or hosts[0] == hosts[1]:
        raise argparse.ArgumentTypeError("--hosts must contain two distinct hosts")
    return hosts  # type: ignore[return-value]


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def remote_bash(ssh: str, host: str, command: str) -> list[str]:
    return [ssh, host, "bash", "-lc", shlex.quote(command)]


@dataclass
class Worker:
    node_rank: int
    host: str
    command: list[str]
    log_path: Path
    process: subprocess.Popen[bytes] | None = None
    log_handle: Any = None
    return_code: int | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("wb")
        self.process = subprocess.Popen(
            self.command,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def poll(self) -> int | None:
        if self.process is None:
            return self.return_code
        value = self.process.poll()
        if value is not None:
            self.return_code = int(value)
            if self.log_handle is not None:
                self.log_handle.flush()
                self.log_handle.close()
                self.log_handle = None
        return self.return_code

    def terminate_local_transport(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--hosts", required=True, type=parse_hosts)
    result.add_argument("--run-id", required=True)
    result.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--remote-root", type=Path)
    result.add_argument(
        "--site-env",
        default="configs/distillschedule/server/baai_2x8_a100.env.example",
    )
    result.add_argument(
        "--plan",
        default="configs/distillschedule/plans/helios_routea_full5_world16_typed.json",
    )
    result.add_argument("--master-addr")
    result.add_argument("--master-port", type=int, default=29677)
    result.add_argument("--expected-head")
    result.add_argument("--ssh", default="ssh")
    result.add_argument("--timeout-seconds", type=int, default=16200)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--summarize-only", action="store_true")
    result.add_argument("--output", type=Path)
    return result


def _manifest_identity(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    remote_root = (args.remote_root or root).expanduser()
    plan = root / args.plan
    if not plan.is_file():
        raise RuntimeError(f"plan does not exist: {plan}")
    expected_head = args.expected_head or git_head(root)
    plan_payload = load_json(plan)
    evidence = plan_payload.get("evidence", {})
    evidence_metadata = (
        evidence.get("metadata", {}) if isinstance(evidence, dict) else {}
    )
    experiment_id = (
        evidence_metadata.get("experiment_id")
        if isinstance(evidence_metadata, dict)
        else None
    )
    required_run_prefix = {
        "RA16-F5-PR4": "ra16_f5_pr4_",
        "RA16-F5-PR4-EC": "ra16_f5_pr4_ec_",
        "RA16-F5-PR8": "ra16_f5_pr8_",
        "RA16-F5-PR12": "ra16_f5_pr12_",
        "RA16-F5-GLP-OFFLOAD": "ra16_f5_glp_offload_",
        "RA16-F5-GLP-SCORE4": "ra16_f5_glp_score4_",
    }.get(experiment_id)
    if required_run_prefix is not None and not args.run_id.startswith(
        required_run_prefix
    ):
        raise RuntimeError(
            f"{experiment_id} run IDs must start with {required_run_prefix!r}"
        )
    return {
        "schema_version": SCHEMA,
        "run_id": args.run_id,
        "hosts": list(args.hosts),
        "master_addr": args.master_addr or args.hosts[0],
        "master_port": args.master_port,
        "repo_root": str(root),
        "remote_root": str(remote_root),
        "site_env": args.site_env,
        "plan": args.plan,
        "experiment_id": experiment_id,
        "plan_sha256": sha256_file(plan),
        "expected_head": expected_head,
        "worker_script": "scripts/distillschedule/launch_routea_world16_typed_node.sh",
        "worker_script_sha256": sha256_file(
            root / "scripts/distillschedule/launch_routea_world16_typed_node.sh"
        ),
        "summarizer_sha256": sha256_file(
            root / "scripts/distillschedule/summarize_routea_world16_typed.py"
        ),
    }


def _existing_result(output: Path) -> dict[str, Any] | None:
    if not output.is_file():
        return None
    try:
        return load_json(output)
    except Exception:
        return None


def _pid_file(remote_root: Path, run_id: str, node_rank: int) -> str:
    return str(
        remote_root
        / "artifacts"
        / "runs"
        / run_id
        / "controller"
        / f"remote_node_{node_rank}.pid"
    )


def _worker_command(
    *,
    args: argparse.Namespace,
    identity: Mapping[str, Any],
    node_rank: int,
    host: str,
) -> list[str]:
    remote_root = Path(str(identity["remote_root"]))
    site_env = Path(args.site_env)
    if not site_env.is_absolute():
        site_env = remote_root / site_env
    wrapper = remote_root / "scripts/distillschedule/remote_stage_wrapper.sh"
    launcher = remote_root / "scripts/distillschedule/launch_routea_world16_typed_node.sh"
    pid_file = _pid_file(remote_root, args.run_id, node_rank)
    command = [
        str(launcher),
        str(node_rank),
        args.run_id,
        str(identity["expected_head"]),
        str(identity["plan_sha256"]),
        str(identity["master_addr"]),
        str(identity["master_port"]),
        str(identity["plan"]),
    ]
    remote = (
        f"set -Eeuo pipefail; source {shlex.quote(str(site_env))}; "
        f"export DMD_SYSTEM_ROOT={shlex.quote(str(remote_root))}; "
        f"exec {shlex.quote(str(wrapper))} {shlex.quote(pid_file)} {shell_join(command)}"
    )
    return remote_bash(args.ssh, host, remote)


def _cancel_remote(
    *, args: argparse.Namespace, identity: Mapping[str, Any], node_rank: int, host: str
) -> None:
    remote_root = Path(str(identity["remote_root"]))
    site_env = Path(args.site_env)
    if not site_env.is_absolute():
        site_env = remote_root / site_env
    cancel = remote_root / "scripts/distillschedule/cancel_remote_stage.sh"
    pid_file = _pid_file(remote_root, args.run_id, node_rank)
    remote = (
        f"set -Eeuo pipefail; source {shlex.quote(str(site_env))}; "
        f"{shlex.quote(str(cancel))} {shlex.quote(pid_file)}"
    )
    subprocess.run(
        remote_bash(args.ssh, host, remote),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=90,
    )


def _remote_site_env(args: argparse.Namespace, remote_root: Path) -> Path:
    site_env = Path(args.site_env)
    if not site_env.is_absolute():
        site_env = remote_root / site_env
    return site_env


def _remote_summary_command(
    *,
    args: argparse.Namespace,
    identity: Mapping[str, Any],
) -> tuple[list[str], Path]:
    remote_root = Path(str(identity["remote_root"]))
    site_env = _remote_site_env(args, remote_root)
    remote_output = (
        remote_root / "artifacts" / "manifests" / f"{args.run_id}_result.json"
    )
    summarizer = remote_root / "scripts/distillschedule/summarize_routea_world16_typed.py"
    remote = (
        f"set -Eeuo pipefail; source {shlex.quote(str(site_env))}; "
        f"export DMD_SYSTEM_ROOT={shlex.quote(str(remote_root))}; "
        'python_bin="${DISTILLSCHEDULE_PYTHON:-python3}"; '
        f'"$python_bin" {shlex.quote(str(summarizer))} '
        f"--run-id {shlex.quote(args.run_id)} "
        f"--root {shlex.quote(str(remote_root))} "
        f"--output {shlex.quote(str(remote_output))}"
    )
    return remote_bash(args.ssh, args.hosts[0], remote), remote_output


def _node_manifests_complete(
    *,
    args: argparse.Namespace,
    identity: Mapping[str, Any],
    root: Path,
) -> bool:
    remote_root = Path(str(identity["remote_root"]))
    if remote_root == root:
        run_dir = root / "artifacts" / "runs" / args.run_id
        return all(
            (
                run_dir
                / "launcher"
                / f"node_{rank}"
                / "launch_manifest.txt"
            ).is_file()
            for rank in (0, 1)
        )
    paths = [
        remote_root
        / "artifacts"
        / "runs"
        / args.run_id
        / "launcher"
        / f"node_{rank}"
        / "launch_manifest.txt"
        for rank in (0, 1)
    ]
    command = "set -Eeuo pipefail; " + " && ".join(
        f"test -f {shlex.quote(str(path))}" for path in paths
    )
    return (
        subprocess.run(
            remote_bash(args.ssh, args.hosts[0], command),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=90,
        ).returncode
        == 0
    )


def _run_summary(
    *,
    args: argparse.Namespace,
    identity: Mapping[str, Any],
    root: Path,
    output: Path,
) -> int:
    remote_root = Path(str(identity["remote_root"]))
    if remote_root != root:
        command, remote_output = _remote_summary_command(
            args=args,
            identity=identity,
        )
        summary = subprocess.run(command, check=False)
        if summary.returncode != 0:
            return int(summary.returncode)
        fetch = subprocess.run(
            remote_bash(
                args.ssh,
                args.hosts[0],
                f"cat {shlex.quote(str(remote_output))}",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if fetch.returncode != 0:
            print(fetch.stderr, file=sys.stderr, end="")
            return int(fetch.returncode)
        try:
            payload = json.loads(fetch.stdout)
            if not isinstance(payload, Mapping):
                raise ValueError("remote result is not a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"invalid remote summary result: {exc}", file=sys.stderr)
            return 2
        atomic_json(output, payload)
        return 0
    command = [
        sys.executable,
        str(root / "scripts/distillschedule/summarize_routea_world16_typed.py"),
        "--run-id",
        run_id,
        "--root",
        str(root),
        "--output",
        str(output),
    ]
    return subprocess.run(command, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.repo_root.expanduser().resolve()
    if not root.is_dir():
        print(json.dumps({"status": "FAIL", "error": "repo root does not exist"}))
        return 2
    if not clean_tree(root) and not args.dry_run:
        print(json.dumps({"status": "FAIL", "error": "controller repository is dirty"}))
        return 2
    try:
        identity = _manifest_identity(args, root)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        return 2

    run_dir = root / "artifacts/runs" / args.run_id
    controller_dir = run_dir / "controller"
    manifest_path = controller_dir / "controller_manifest.json"
    output = args.output or root / "artifacts/manifests" / f"{args.run_id}_result.json"
    existing = _existing_result(output)

    if manifest_path.exists():
        previous = load_json(manifest_path)
        previous_identity = previous.get("identity")
        if previous_identity != identity:
            print(
                json.dumps(
                    {
                        "status": "FAIL",
                        "error": "immutable run ID exists with different identity",
                        "manifest": str(manifest_path),
                    },
                    sort_keys=True,
                )
            )
            return 2
        if existing and existing.get("status") == "PASS" and args.resume:
            print(
                json.dumps(
                    {
                        "status": "REUSED_ACCEPTED",
                        "run_id": args.run_id,
                        "result": str(output),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if not (args.resume or args.summarize_only or args.dry_run):
            print(
                json.dumps(
                    {
                        "status": "FAIL",
                        "error": "immutable run ID already exists; use --resume only for evidence closure or choose a new run ID",
                    },
                    sort_keys=True,
                )
            )
            return 2
    else:
        controller_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": SCHEMA,
            "identity": identity,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "state": "PREREGISTERED",
        }
        atomic_json(manifest_path, manifest)

    if args.dry_run:
        commands = [
            _worker_command(
                args=args,
                identity=identity,
                node_rank=rank,
                host=host,
            )
            for rank, host in enumerate(args.hosts)
        ]
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "run_id": args.run_id,
                    "manifest": str(manifest_path),
                    "commands": commands,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.summarize_only or args.resume:
        if _node_manifests_complete(
            args=args,
            identity=identity,
            root=root,
        ):
            summary_rc = _run_summary(
                args=args,
                identity=identity,
                root=root,
                output=output,
            )
            if summary_rc == 0 and _existing_result(output):
                print(json.dumps({"status": "SUMMARIZED", "result": str(output)}))
                return 0
            if args.summarize_only:
                print(
                    json.dumps(
                        {
                            "status": "FAIL",
                            "error": "existing evidence did not pass strict summarization",
                            "result": str(output),
                        },
                        sort_keys=True,
                    )
                )
                return summary_rc or 2
        if args.summarize_only:
            print(json.dumps({"status": "FAIL", "error": "node manifests are incomplete"}))
            return 2
        if manifest_path.exists() and existing and existing.get("status") != "PASS":
            print(
                json.dumps(
                    {
                        "status": "FAIL",
                        "error": "failed/partial immutable run cannot be relaunched under the same run ID",
                    },
                    sort_keys=True,
                )
            )
            return 2

    workers = [
        Worker(
            node_rank=rank,
            host=host,
            command=_worker_command(
                args=args, identity=identity, node_rank=rank, host=host
            ),
            log_path=controller_dir / f"ssh_node_{rank}.log",
        )
        for rank, host in enumerate(args.hosts)
    ]
    for worker in workers:
        worker.start()

    deadline = time.monotonic() + args.timeout_seconds
    cancelled = False
    try:
        while True:
            return_codes = [worker.poll() for worker in workers]
            failed = [
                worker
                for worker in workers
                if worker.return_code is not None and worker.return_code != 0
            ]
            if failed and not cancelled:
                cancelled = True
                for worker in workers:
                    if worker.return_code is None:
                        _cancel_remote(
                            args=args,
                            identity=identity,
                            node_rank=worker.node_rank,
                            host=worker.host,
                        )
                        worker.terminate_local_transport()
            if all(value is not None for value in return_codes):
                break
            if time.monotonic() >= deadline:
                cancelled = True
                for worker in workers:
                    if worker.return_code is None:
                        _cancel_remote(
                            args=args,
                            identity=identity,
                            node_rank=worker.node_rank,
                            host=worker.host,
                        )
                        worker.terminate_local_transport()
                break
            time.sleep(1)
    except KeyboardInterrupt:
        cancelled = True
        for worker in workers:
            _cancel_remote(
                args=args,
                identity=identity,
                node_rank=worker.node_rank,
                host=worker.host,
            )
            worker.terminate_local_transport()
    finally:
        for worker in workers:
            if worker.process is not None:
                try:
                    worker.process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    worker.terminate_local_transport()
                worker.poll()

    return_codes = {str(worker.node_rank): worker.return_code for worker in workers}
    state = {
        "schema_version": SCHEMA,
        "identity": identity,
        "created_at": load_json(manifest_path).get("created_at"),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "state": "WORKERS_FINISHED",
        "worker_return_codes": return_codes,
        "peer_cancelled": cancelled,
    }
    atomic_json(manifest_path, state)
    if any(value != 0 for value in return_codes.values()):
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "run_id": args.run_id,
                    "worker_return_codes": return_codes,
                    "downstream": "NOT_RUN",
                },
                sort_keys=True,
            )
        )
        return 1

    summary_rc = _run_summary(
        args=args,
        identity=identity,
        root=root,
        output=output,
    )
    result = _existing_result(output)
    status = result.get("status") if result else "MISSING_RESULT"
    print(
        json.dumps(
            {
                "status": status,
                "run_id": args.run_id,
                "result": str(output),
                "summary_return_code": summary_rc,
            },
            sort_keys=True,
        )
    )
    return 0 if summary_rc == 0 and status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
