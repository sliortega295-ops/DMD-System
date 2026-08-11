#!/usr/bin/env python3
"""Centrally coordinate one immutable two-node N2 capacity candidate."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from runtime.gst_pool_contract import GeneratorOffloadCandidate


def _parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--run-id", required=True)
    result.add_argument("--candidate-manifest", type=Path, required=True)
    result.add_argument("--result-path", type=Path, required=True)
    result.add_argument("--generator-host", default="baai-helios-node0")
    result.add_argument("--score-host", default="baai-helios-node1")
    result.add_argument(
        "--remote-repo",
        default="/share/project/shuchenweng/yongyan_liu/DMD-System_pr3_0cd40e4",
    )
    result.add_argument("--timeout-seconds", type=float, default=10800.0)
    return result


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, check=False, **kwargs)


def _ssh(host: str, command: str, **kwargs) -> subprocess.CompletedProcess[str]:
    return _run(["ssh", "-o", "BatchMode=yes", host, command], **kwargs)


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _cancel(host: str, remote_repo: str, pid_file: str) -> None:
    _ssh(
        host,
        f"{shlex.quote(remote_repo)}/scripts/distillschedule/cancel_remote_stage.sh {shlex.quote(pid_file)}",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    args = _parser().parse_args()
    candidate = GeneratorOffloadCandidate.load(args.candidate_manifest)
    if candidate.run_id != args.run_id:
        raise ValueError("controller run ID differs from candidate manifest")
    local_repo = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(
        ["git", "-C", str(local_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(local_repo), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("local candidate checkout must be clean")

    remote_run = f"{args.remote_repo}/artifacts/gst_pool_candidates/{args.run_id}"
    for host in (args.generator_host, args.score_host):
        audit = _ssh(
            host,
            "set -e; "
            f"test \"$(git -C {shlex.quote(args.remote_repo)} rev-parse HEAD)\" = {shlex.quote(head)}; "
            f"test -z \"$(git -C {shlex.quote(args.remote_repo)} status --porcelain --untracked-files=no)\"; "
            "test -z \"$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')\"",
            capture_output=True,
        )
        if audit.returncode != 0:
            raise RuntimeError(
                f"{host} preflight failed: {audit.stdout}\n{audit.stderr}"
            )
    create = _ssh(
        args.generator_host,
        f"set -e; test ! -e {shlex.quote(remote_run)}; "
        f"mkdir -p {shlex.quote(remote_run)}/launcher",
        capture_output=True,
    )
    if create.returncode != 0:
        raise RuntimeError(f"remote immutable run root exists: {remote_run}")
    remote_candidate = f"{remote_run}/candidate.json"
    copied = _run(
        [
            "scp",
            "-q",
            str(args.candidate_manifest),
            f"{args.generator_host}:{remote_candidate}",
        ],
        capture_output=True,
    )
    if copied.returncode != 0:
        raise RuntimeError(f"candidate upload failed: {copied.stderr}")

    launcher = f"{args.remote_repo}/scripts/distillschedule/launch_gst_pool_candidate_node.sh"
    commands = {
        "generator": (
            args.generator_host,
            " ".join(
                shlex.quote(value)
                for value in (
                    launcher,
                    "generator",
                    args.run_id,
                    head,
                    candidate.contract_digest,
                    remote_candidate,
                    remote_run,
                )
            ),
        ),
        "score": (
            args.score_host,
            " ".join(
                shlex.quote(value)
                for value in (
                    launcher,
                    "score",
                    args.run_id,
                    head,
                    candidate.contract_digest,
                    remote_candidate,
                    remote_run,
                )
            ),
        ),
    }
    processes: dict[str, subprocess.Popen[str]] = {}
    for role, (host, command) in commands.items():
        processes[role] = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", host, command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    deadline = time.monotonic() + args.timeout_seconds
    return_codes: dict[str, int] = {}
    outputs: dict[str, str] = {}
    while len(return_codes) < len(processes):
        for role, process in processes.items():
            if role in return_codes:
                continue
            code = process.poll()
            if code is not None:
                return_codes[role] = int(code)
                outputs[role] = process.stdout.read() if process.stdout else ""
                if code != 0:
                    other = "score" if role == "generator" else "generator"
                    if other not in return_codes:
                        other_host = commands[other][0]
                        _cancel(
                            other_host,
                            args.remote_repo,
                            f"{remote_run}/launcher/{other}.pid",
                        )
        if time.monotonic() >= deadline:
            for role, process in processes.items():
                if role not in return_codes:
                    _cancel(
                        commands[role][0],
                        args.remote_repo,
                        f"{remote_run}/launcher/{role}.pid",
                    )
            raise TimeoutError(f"candidate exceeded {args.timeout_seconds}s")
        time.sleep(1.0)
    for role, output in outputs.items():
        local_log = args.result_path.parent / f"{args.run_id}_{role}_controller.log"
        local_log.parent.mkdir(parents=True, exist_ok=True)
        local_log.write_text(output)

    remote_summary = f"{remote_run}/result.json"
    summary_command = " ".join(
        shlex.quote(value)
        for value in (
            f"/tmp/distillschedule-helios-official-torch210/bin/python",
            f"{args.remote_repo}/scripts/distillschedule/summarize_gst_pool_candidate.py",
            "--run-root",
            remote_run,
            "--candidate-manifest",
            remote_candidate,
            "--contract-digest",
            candidate.contract_digest,
            "--generator-return-code",
            str(return_codes["generator"]),
            "--score-return-code",
            str(return_codes["score"]),
            "--output",
            remote_summary,
        )
    )
    summarized = _ssh(
        args.generator_host,
        f"cd {shlex.quote(args.remote_repo)} && "
        f"PYTHONPATH={shlex.quote(args.remote_repo)}:{shlex.quote(args.remote_repo + '/src')} "
        + summary_command,
        capture_output=True,
    )
    fetched = _ssh(
        args.generator_host,
        f"cat {shlex.quote(remote_summary)}",
        capture_output=True,
    )
    if fetched.returncode != 0:
        failure = {
            "schema_version": "distillschedule.g_offload_candidate_result.v1",
            "status": "FAIL",
            "run_id": args.run_id,
            "contract_digest": candidate.contract_digest,
            "candidate_manifest_sha256": candidate.manifest_sha256,
            "execution_graph": "target_no_gan_gst_pool",
            "student_residency": "resident",
            "teacher_residency": "resident",
            "failure": "remote summarizer did not produce a result",
            "summarizer_return_code": summarized.returncode,
            "summarizer_output": summarized.stdout + summarized.stderr,
            "remote_run_root": remote_run,
        }
        _atomic(args.result_path, failure)
        return 1
    result = json.loads(fetched.stdout)
    _atomic(args.result_path, result)
    return 0 if result.get("status") == "PASS" else (
        42 if result.get("status") == "OOM" else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
