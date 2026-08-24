#!/usr/bin/env python3
"""Run one bounded Helios14B world16 NCU target on node0/device0."""

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


HOSTS = ("baai-a100-new-node0", "baai-a100-new-node1")


def ssh(host: str, command: str) -> list[str]:
    return [
        "ssh", "-4", "-o", "BatchMode=yes", "-o", "UpdateHostKeys=no",
        host, "bash", "-lc", shlex.quote(command),
    ]


def q(value: object) -> str:
    return shlex.quote(str(value))


@dataclass
class Worker:
    node: int
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

    def terminate(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--target-json", type=Path, required=True)
    ap.add_argument("--target-id", required=True)
    ap.add_argument("--expected-head", required=True)
    ap.add_argument(
        "--remote-root", type=Path,
        default=Path("/share/project/shuchenweng/yongyan_liu/DMD-System_helios14b_opt_20260824"),
    )
    ap.add_argument(
        "--helios-runtime-root", type=Path,
        default=Path("/share/project/shuchenweng/yongyan_liu/DistillSchedule_fsdp2_gpu_residual_20260816"),
    )
    ap.add_argument("--correctness-summary", type=Path, required=True)
    ap.add_argument("--correctness-sha256", required=True)
    ap.add_argument("--master-port", type=int, default=29860)
    ap.add_argument("--local-output", type=Path, required=True)
    ap.add_argument("--timeout-seconds", type=int, default=7200)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def target(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.target_json.read_text())
    matches = [row for row in payload["targets"] if row["target_id"] == args.target_id]
    if len(matches) != 1:
        raise ValueError(f"expected one target {args.target_id}, found {len(matches)}")
    return matches[0]


def resolve_master() -> str:
    values = subprocess.check_output(ssh(HOSTS[0], "hostname -I"), text=True).split()
    return next(value for value in values if ":" not in value)


def pid_file(args: argparse.Namespace, node: int) -> Path:
    return args.remote_root / "artifacts/runs" / args.run_id / "controller" / f"node_{node}.pid"


def command(args: argparse.Namespace, row: dict[str, Any], node: int, master: str) -> list[str]:
    launcher = args.remote_root / "scripts/distillschedule/launch_static_fsdp2_p0_ncu_world16_node.sh"
    wrapper = args.remote_root / "scripts/distillschedule/remote_stage_wrapper.sh"
    exports = {
        "DMD_SYSTEM_ROOT": args.remote_root,
        "HELIOS_RUNTIME_ROOT": args.helios_runtime_root,
        "DISTILLSCHEDULE_EXPECTED_HELIOS_TRANSFORMER_SHA256": "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886",
        "DISTILLSCHEDULE_GENERATOR_RESIDUAL_STORAGE": "cuda_out_of_place",
        "DISTILLSCHEDULE_CRITIC_RESIDUAL_STORAGE": "cuda_out_of_place",
        "DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE": "off",
        "DISTILLSCHEDULE_HELIOS_ALL_ROLE_LORA_SCALE1_ADD": "on",
        "DISTILLSCHEDULE_OPTIMIZER_DISPATCH_WARMUP": "on",
        "DISTILLSCHEDULE_EMA_DEVICE": "cuda",
        "DISTILLSCHEDULE_PROFILE_DIAGNOSTICS": "off",
        "DISTILLSCHEDULE_NCU_BIN": "/opt/nvidia/nsight-compute/2023.2.1/ncu",
        "DISTILLSCHEDULE_NCU_TARGET_ID": row["target_id"],
        "DISTILLSCHEDULE_NCU_NVTX_RANGE": row["phase"],
        "DISTILLSCHEDULE_NCU_KERNEL_NAME": row["ncu_kernel_filter"],
        "DISTILLSCHEDULE_NCU_LAUNCH_SKIP": row["launch_skip"],
        "DISTILLSCHEDULE_NCU_CORRECTNESS_SUMMARY": args.correctness_summary,
        "DISTILLSCHEDULE_NCU_CORRECTNESS_SHA256": args.correctness_sha256,
    }
    launch = " ".join(
        q(value)
        for value in (
            "bash", launcher, node, args.run_id, "no-gan", "profile",
            args.expected_head, master, args.master_port, "node-hsdp", 1, "off",
        )
    )
    remote = (
        "set -Eeuo pipefail; export "
        + " ".join(f"{key}={q(value)}" for key, value in exports.items())
        + "; exec " + q(wrapper) + " " + q(pid_file(args, node)) + " " + launch
    )
    return ssh(HOSTS[node], remote)


def cancel(args: argparse.Namespace, node: int) -> None:
    script = args.remote_root / "scripts/distillschedule/cancel_remote_stage.sh"
    subprocess.run(
        ssh(HOSTS[node], f"{q(script)} {q(pid_file(args, node))}"),
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90,
    )


def scp_from(host: str, remote: Path, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", "-4", "-q", f"{host}:{remote}", str(local)], check=True)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.remote_root / "artifacts/runs" / args.run_id
    summary = run_root / "summary.json"
    script = args.remote_root / "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py"
    command_value = (
        f"export PYTHONPATH={q(args.remote_root / 'src')}:{q(args.remote_root)}; "
        f"/tmp/distillschedule-helios-official-torch210/bin/python {q(script)} "
        f"--run-root {q(run_root)} --variant no-gan --mode profile "
        "--mesh-policy node-hsdp --rollout-batch-size 1 --pack-real-score-pair off "
        "--generator-residual-storage cuda_out_of_place "
        "--critic-residual-storage cuda_out_of_place "
        "--generator-nograd-lora-fused-epilogue off --all-role-lora-scale1-add on "
        f"--optimizer-dispatch-warmup on --output {q(summary)}"
    )
    subprocess.run(ssh(HOSTS[0], command_value), check=True, timeout=300)
    scp_from(HOSTS[0], summary, args.local_output / "summary.json")
    value = json.loads((args.local_output / "summary.json").read_text())
    expected_policy = {
        "mesh_policy": "node-hsdp", "rollout_batch_size": 1,
        "pack_real_score_pair": False, "generator_residual_storage": "cuda_out_of_place",
        "critic_residual_storage": "cuda_out_of_place", "all_role_lora_scale1_add": True,
        "optimizer_dispatch_warmup": True,
    }
    if value.get("status") != "PASS" or value.get("rank_count") != 16:
        raise RuntimeError(f"world16 semantic summary failed: {value.get('errors')}")
    for key, expected in expected_policy.items():
        if value.get("policy", {}).get(key) != expected:
            raise RuntimeError(f"world16 policy drift: {key}")
    return value


def main() -> int:
    args = parser().parse_args()
    if len(args.expected_head) != 40 or len(args.correctness_sha256) != 64:
        raise ValueError("full source/correctness hashes are required")
    if args.local_output.exists():
        raise FileExistsError(args.local_output)
    row = target(args)
    master = resolve_master()
    commands = [command(args, row, node, master) for node in (0, 1)]
    manifest = {
        "schema_version": "distillschedule.helios14b_ncu_launch.v1",
        "status": "NOT_RUN" if args.dry_run else "RUNNING",
        "run_id": args.run_id,
        "target": row,
        "expected_head": args.expected_head,
        "correctness_summary": str(args.correctness_summary),
        "correctness_sha256": args.correctness_sha256,
        "master_addr": master,
        "master_port": args.master_port,
        "commands": commands,
        "claim_boundary": "DIAGNOSTIC_ONLY",
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    args.local_output.mkdir(parents=True)
    (args.local_output / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    workers = [
        Worker(node, commands[node], args.local_output / "logs" / f"node_{node}.ssh.log")
        for node in (0, 1)
    ]
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
                worker.terminate()
            raise RuntimeError(f"world16 NCU run failed: {codes}")

        summarize(args)
        run_root = args.remote_root / "artifacts/runs" / args.run_id
        ncu = run_root / "ncu/node_0"
        for remote, local in (
            (ncu / f"{args.target_id}.ncu-rep", args.local_output / "reports" / f"{args.target_id}.ncu-rep"),
            (ncu / "ncu_details.csv", args.local_output / "raw" / "ncu_details.csv"),
            (ncu / "ncu_raw.csv", args.local_output / "raw" / "ncu_raw.csv"),
            (ncu / "ncu_manifest.txt", args.local_output / "ncu_manifest.txt"),
            (run_root / "launcher/node_0/console.log", args.local_output / "logs" / "node0_console.log"),
            (run_root / "launcher/node_1/console.log", args.local_output / "logs" / "node1_console.log"),
        ):
            scp_from(HOSTS[0], remote, local)
        manifest["status"] = "PASS"
        (args.local_output / "launch_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return 0
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
