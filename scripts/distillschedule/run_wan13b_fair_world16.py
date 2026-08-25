#!/usr/bin/env python3
"""Centrally launch, cancel, and summarize one fair Wan1.3B world16 run."""
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

from distillschedule.runtime.wan13b_fair import FAIR_PLACEMENTS


def _hosts(value: str) -> tuple[str, str]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(items) != 2 or items[0] == items[1]:
        raise argparse.ArgumentTypeError("--hosts requires two distinct SSH aliases")
    return items  # type: ignore[return-value]


def _ssh(host: str, command: str) -> list[str]:
    return [
        "ssh",
        "-4",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "BatchMode=yes",
        "-o",
        "UpdateHostKeys=no",
        host,
        "bash",
        "-lc",
        shlex.quote(command),
    ]


def _resolve_master(host: str) -> str:
    values = subprocess.check_output(
        _ssh(host, "hostname -I"), text=True, timeout=30
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

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--hosts",
        type=_hosts,
        default=("baai-helios-node0", "baai-helios-node1"),
    )
    value.add_argument("--run-id", required=True)
    value.add_argument(
        "--mode",
        choices=("validate", "correctness", "timing", "diagnostic-timing", "profile"),
        required=True,
    )
    value.add_argument("--placement", choices=FAIR_PLACEMENTS, required=True)
    value.add_argument("--expected-head", required=True)
    value.add_argument("--model-root", type=Path, required=True)
    value.add_argument("--bootstrap", type=Path, required=True)
    value.add_argument("--bootstrap-hash", required=True)
    value.add_argument("--master-addr")
    value.add_argument("--master-port", type=int, default=29860)
    value.add_argument(
        "--remote-root",
        type=Path,
        default=Path(
            "/share/project/shuchenweng/yongyan_liu/"
            "DMD-System_wan13b_upper_bound_20260820"
        ),
    )
    value.add_argument(
        "--helios-runtime-root",
        type=Path,
        default=Path(
            "/share/project/shuchenweng/yongyan_liu/"
            "DistillSchedule_fsdp2_gpu_residual_20260816"
        ),
    )
    value.add_argument(
        "--expected-helios-transformer-sha256",
        default="5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886",
    )
    value.add_argument("--correctness-receipt", type=Path)
    value.add_argument("--correctness-sha256")
    value.add_argument(
        "--nsys-bin",
        type=Path,
        default=Path("/home/lyy/Helios/tools/nsight_systems/2024.6.2.225/bin/nsys"),
    )
    value.add_argument(
        "--local-evidence-root",
        type=Path,
        default=Path(
            "/home/lyy/Helios/research_artifacts/DMD-System/"
            "wan13b_fair_resident_fsdp2_20260822"
        ),
    )
    value.add_argument("--lora-operator-attribution", action="store_true")
    value.add_argument("--lora-attribution-target-rank", type=int, default=0)
    value.add_argument("--lora-attribution-target-nograd-calls", type=int, default=3)
    value.add_argument("--generator-nograd-lora-cast-cache", action="store_true")
    value.add_argument("--generator-nograd-lora-fused-epilogue", action="store_true")
    value.add_argument("--timeout-seconds", type=int, default=18000)
    value.add_argument("--dry-run", action="store_true")
    return value


def _pid_file(args: argparse.Namespace, node: int) -> Path:
    return (
        args.remote_root
        / "artifacts/runs"
        / args.run_id
        / "controller"
        / f"node_{node}.pid"
    )


def _command(args: argparse.Namespace, node: int, master: str) -> list[str]:
    launcher = (
        args.remote_root
        / "scripts/distillschedule/launch_wan13b_fair_world16_node.sh"
    )
    wrapper = args.remote_root / "scripts/distillschedule/remote_stage_wrapper.sh"
    exports = [
        f"DMD_SYSTEM_ROOT={shlex.quote(str(args.remote_root))}",
        f"HELIOS_RUNTIME_ROOT={shlex.quote(str(args.helios_runtime_root))}",
        f"DISTILLSCHEDULE_WAN13B_MODEL_ROOT={shlex.quote(str(args.model_root))}",
        f"DISTILLSCHEDULE_WAN13B_BOOTSTRAP={shlex.quote(str(args.bootstrap))}",
        f"DISTILLSCHEDULE_WAN13B_BOOTSTRAP_HASH={shlex.quote(args.bootstrap_hash)}",
        "DISTILLSCHEDULE_EXPECTED_HELIOS_TRANSFORMER_SHA256="
        + shlex.quote(args.expected_helios_transformer_sha256),
    ]
    if args.mode in {"timing", "diagnostic-timing", "profile"}:
        if args.correctness_receipt is None or not args.correctness_sha256:
            raise ValueError("low-overhead execution requires a pinned correctness receipt")
        exports.extend(
            (
                "DISTILLSCHEDULE_WAN13B_FAIR_CORRECTNESS_RECEIPT="
                + shlex.quote(str(args.correctness_receipt)),
                "DISTILLSCHEDULE_WAN13B_FAIR_CORRECTNESS_SHA256="
                + shlex.quote(args.correctness_sha256),
            )
        )
    if args.mode == "profile":
        exports.append("DISTILLSCHEDULE_NSYS_BIN=" + shlex.quote(str(args.nsys_bin)))
    if args.lora_operator_attribution:
        if args.mode != "profile":
            raise ValueError("LoRA operator attribution requires --mode profile")
        if not 0 <= args.lora_attribution_target_rank < 16:
            raise ValueError("LoRA attribution target rank must be in [0, 15]")
        if args.lora_attribution_target_nograd_calls <= 0:
            raise ValueError("LoRA attribution target call count must be positive")
        exports.extend(
            (
                "DISTILLSCHEDULE_WAN13B_LORA_OPERATOR_ATTRIBUTION=1",
                "DISTILLSCHEDULE_WAN13B_LORA_ATTRIBUTION_TARGET_RANK="
                + str(args.lora_attribution_target_rank),
                "DISTILLSCHEDULE_WAN13B_LORA_ATTRIBUTION_TARGET_NOGRAD_CALLS="
                + str(args.lora_attribution_target_nograd_calls),
            )
        )
    if args.generator_nograd_lora_cast_cache:
        exports.append("DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE=1")
    if args.generator_nograd_lora_fused_epilogue:
        exports.append(
            "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE=1"
        )
    launch = " ".join(
        shlex.quote(str(value))
        for value in (
            "bash",
            launcher,
            node,
            args.run_id,
            args.mode,
            args.expected_head,
            master,
            args.master_port,
            args.placement,
        )
    )
    remote = (
        "set -Eeuo pipefail; export "
        + " ".join(exports)
        + "; exec "
        + shlex.quote(str(wrapper))
        + " "
        + shlex.quote(str(_pid_file(args, node)))
        + " "
        + launch
    )
    return _ssh(args.hosts[node], remote)


def _cancel(args: argparse.Namespace, node: int) -> None:
    script = args.remote_root / "scripts/distillschedule/cancel_remote_stage.sh"
    try:
        subprocess.run(
            _ssh(
                args.hosts[node],
                f"{shlex.quote(str(script))} "
                f"{shlex.quote(str(_pid_file(args, node)))}",
            ),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pass


def _summarize(args: argparse.Namespace, local: Path) -> dict[str, Any]:
    run_root = args.remote_root / "artifacts/runs" / args.run_id
    summary = run_root / "summary.json"
    summarizer = (
        args.remote_root
        / "scripts/distillschedule/summarize_wan13b_fair_world16.py"
    )
    command = (
        f"export PYTHONPATH={shlex.quote(str(args.remote_root / 'src'))}:"
        f"{shlex.quote(str(args.remote_root))}; "
        "/tmp/distillschedule-helios-official-torch210/bin/python "
        f"{shlex.quote(str(summarizer))} "
        f"--run-root {shlex.quote(str(run_root))} "
        f"--placement {shlex.quote(args.placement)} "
        f"--mode {shlex.quote(args.mode)} "
        f"--output {shlex.quote(str(summary))}"
    )
    completed = subprocess.run(_ssh(args.hosts[0], command), check=False, timeout=300)
    raw = subprocess.check_output(
        _ssh(args.hosts[0], f"cat {shlex.quote(str(summary))}"), timeout=60
    )
    local.mkdir(parents=True, exist_ok=True)
    (local / "summary.json").write_bytes(raw)
    value = json.loads(raw)
    if completed.returncode != 0 or value.get("status") != "PASS":
        raise RuntimeError(f"fair Wan1.3B summary failed: {value.get('errors')}")
    return value


def main() -> int:
    args = parser().parse_args()
    if not args.run_id.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("unsafe run id")
    if len(args.expected_head) != 40:
        raise ValueError("--expected-head must be a full Git SHA")
    if len(args.bootstrap_hash) != 64:
        raise ValueError("--bootstrap-hash must be SHA256")
    local = args.local_evidence_root / args.run_id
    if local.exists():
        raise FileExistsError(local)
    master = args.master_addr or _resolve_master(args.hosts[0])
    commands = [_command(args, node, master) for node in (0, 1)]
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
            raise RuntimeError(f"fair Wan1.3B run failed or timed out: {codes}")
        value = _summarize(args, local)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
