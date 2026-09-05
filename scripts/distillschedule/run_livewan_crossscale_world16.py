#!/usr/bin/env python3
"""Central two-node controller for one immutable LiveWan cross-scale run."""

from __future__ import annotations

import argparse
import base64
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
        # This cluster's multi-tenant SSH gateway currently closes non-PTY
        # controller channels even though the same job endpoint is healthy.
        # A strong PTY is transport-only; the remote launcher still writes all
        # evidence to immutable files and owns the atomic GPU lock.
        "ssh", "-CAXY", "-tt", "-4", "-p", str(port),
        "-o", "ControlMaster=no", "-o", "ControlPath=none",
        "-o", "BatchMode=yes", "-o", "UpdateHostKeys=no", host,
        "bash", "-lc", shlex.quote(command),
    ]


_PAYLOAD_BEGIN = b"__DISTILLSCHEDULE_PAYLOAD_BEGIN__"
_PAYLOAD_END = b"__DISTILLSCHEDULE_PAYLOAD_END__"


def decode_remote_payload(raw: bytes) -> bytes:
    """Extract one file payload from a noisy mandatory-PTY SSH channel."""

    try:
        start = raw.index(_PAYLOAD_BEGIN) + len(_PAYLOAD_BEGIN)
        end = raw.index(_PAYLOAD_END, start)
    except ValueError as error:
        raise RuntimeError("remote payload markers are missing") from error
    return base64.b64decode(raw[start:end].strip(), validate=True)


def read_remote_file(host: str, path: Path, *, port: int) -> bytes:
    # The platform banner and connection epilogue are emitted on every PTY.
    # Mark and base64 the actual file so those transport bytes never corrupt
    # JSON receipts copied into the local evidence directory.
    command = (
        f"printf '{_PAYLOAD_BEGIN.decode()}\\n'; "
        f"base64 -w0 {shlex.quote(str(path))}; "
        f"printf '\\n{_PAYLOAD_END.decode()}\\n'"
    )
    raw = subprocess.check_output(ssh(host, command, port=port), timeout=60)
    return decode_remote_payload(raw)


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
    value.add_argument(
        "--run-kind",
        choices=(
            "validate", "capacity", "correctness", "timing", "profile",
            "trajectory", "two-cycle-timing",
        ),
        required=True,
    )
    value.add_argument("--execution-mode", choices=("baseline", "pipeline"), required=True)
    value.add_argument(
        "--pipeline-variant",
        choices=(
            "b1", "b1-runahead", "b1-runahead-critic-b2",
            "b1-runahead-critic-b2-local-context",
            "b1-runahead-critic-b2-local-context-async32",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-context-prefetch",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1-modulation",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-fused-cast-add",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow-fused-adam",
            "c15-score-rope",
            "c16-score-flash-attn2",
            "c18-generator-cross-kv-cache",
            "c19-host-runahead",
            "c20-generator-flash-attn2",
            "c21-continuous-b16",
            "c21-c1-continuous-b16",
            "c22-critic-checkpoint-rng",
            "c23-continuous-b32",
            "c24-continuous-b16-score-modulation",
            "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue",
            "c24-continuous-b16-score-modulation-b1-attribution",
            "phase-expanded-critic16",
            "phase-expanded-critic16-fused-lora-elision",
            "score-b2",
            "6g10score", "6g10score-node-local",
            "6g10score-node-local-fused",
        ),
        default="b1",
        help="physical pipeline policy; logical workload remains frozen",
    )
    value.add_argument("--expected-head", required=True)
    value.add_argument("--remote-root", required=True, type=Path)
    value.add_argument(
        "--remote-output-root", required=True, type=Path,
        help="persistent run root outside the clean source checkout",
    )
    value.add_argument("--student-checkpoint", required=True, type=Path)
    value.add_argument("--critic-checkpoint", required=True, type=Path)
    value.add_argument("--teacher-checkpoint", required=True, type=Path)
    value.add_argument("--trace", required=True, type=Path)
    value.add_argument("--bootstrap", required=True, type=Path)
    value.add_argument("--source-root", required=True, type=Path)
    value.add_argument("--source-receipt", required=True, type=Path)
    value.add_argument("--pyenv", type=Path, default=Path("/tmp/distillschedule-helios-official-torch210"))
    value.add_argument("--python-overlay", required=True, type=Path)
    value.add_argument(
        "--flash-attn-kernel",
        type=Path,
        help=(
            "optional node-local copy of the pinned FlashAttention-2 build; "
            "the launcher and runtime still verify the frozen artifact hashes"
        ),
    )
    value.add_argument("--correctness-receipt", type=Path)
    value.add_argument("--correctness-sha256")
    value.add_argument("--nsys-bin", type=Path)
    value.add_argument("--ncu-bin", type=Path)
    value.add_argument("--ncu-node-rank", type=int, choices=(0, 1), default=1)
    value.add_argument("--ncu-local-rank", type=int, choices=range(8), default=0)
    value.add_argument("--ncu-kernel-regex", default="flash_fwd_kernel")
    value.add_argument("--ncu-launch-skip-before-match", type=int, default=0)
    value.add_argument("--ncu-launch-count", type=int, default=1)
    value.add_argument("--ncu-set", default="full")
    value.add_argument("--ncu-output-name", default="bounded_kernel")
    value.add_argument("--trajectory-cycles", type=int, default=0)
    value.add_argument("--trajectory-seed", type=int, default=20260903)
    value.add_argument("--trajectory-arm-id")
    value.add_argument("--two-cycle-seed", type=int)
    value.add_argument("--two-cycle-arm-id")
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
    return args.remote_output_root / args.run_id / "controller" / f"node_{node}.pid"


def command(args: argparse.Namespace, node: int, master: str) -> list[str]:
    run_root = args.remote_output_root / args.run_id
    launcher = args.remote_root / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh"
    wrapper = args.remote_root / "scripts/distillschedule/remote_stage_wrapper.sh"
    exports = {
        "DMD_SYSTEM_ROOT": args.remote_root,
        "DISTILLSCHEDULE_PYENV": args.pyenv,
        "LIVEWAN_PYTHON_OVERLAY": args.python_overlay,
        "DISTILLSCHEDULE_OUTPUT_DIR": run_root,
        "LIVEWAN_STUDENT_CHECKPOINT": args.student_checkpoint,
        "LIVEWAN_CRITIC_CHECKPOINT": args.critic_checkpoint,
        "LIVEWAN_TEACHER_CHECKPOINT": args.teacher_checkpoint,
        "LIVEWAN_OPERATION_TRACE": args.trace,
        "LIVEWAN_SYSTEMS_BOOTSTRAP": args.bootstrap,
        "LIVEWAN_WAN21_SOURCE": args.source_root,
        "LIVEWAN_SOURCE_RECEIPT": args.source_receipt,
        "LIVEWAN_PIPELINE_VARIANT": args.pipeline_variant,
        "LIVEWAN_EXPECTED_HEAD": args.expected_head,
    }
    if args.flash_attn_kernel is not None:
        exports["DISTILLSCHEDULE_FLASH_ATTN_KERNEL"] = args.flash_attn_kernel
    if args.run_kind in {"timing", "profile", "two-cycle-timing"}:
        if args.correctness_receipt is None or not args.correctness_sha256:
            raise ValueError(f"{args.run_kind} requires a correctness/diagnostic receipt and SHA256")
        exports["LIVEWAN_PAIR_CORRECTNESS_RECEIPT"] = args.correctness_receipt
        exports["LIVEWAN_PAIR_CORRECTNESS_SHA256"] = args.correctness_sha256
    if args.run_kind == "profile":
        if args.nsys_bin is None:
            raise ValueError("profile requires --nsys-bin")
        exports["DISTILLSCHEDULE_NSYS_BIN"] = args.nsys_bin
    if args.run_kind == "trajectory":
        if args.execution_mode == "baseline":
            if args.pipeline_variant != "b1":
                raise ValueError("trajectory baseline requires sequential B1")
        elif args.pipeline_variant not in {
            "c24-continuous-b16-score-modulation",
            "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue",
            "c24-continuous-b16-score-modulation-b1-attribution",
        }:
            raise ValueError("trajectory pipeline requires exact C24 B2/B1 attribution")
        if args.trajectory_cycles not in {1, 2, 5, 50}:
            raise ValueError("trajectory cycles must be one of 1, 2, 5, or 50")
        if not args.trajectory_arm_id:
            raise ValueError("trajectory mode requires --trajectory-arm-id")
        exports.update(
            LIVEWAN_TRAJECTORY_CYCLES=args.trajectory_cycles,
            LIVEWAN_TRAJECTORY_SEED=args.trajectory_seed,
            LIVEWAN_TRAJECTORY_ARM_ID=args.trajectory_arm_id,
        )
    if args.run_kind == "two-cycle-timing":
        if args.execution_mode == "baseline":
            if args.pipeline_variant != "b1":
                raise ValueError("two-cycle baseline requires sequential B1")
        elif args.pipeline_variant not in {"c24-continuous-b16-score-modulation", "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue"}:
            raise ValueError("two-cycle pipeline requires production C24 physical B2")
        if args.two_cycle_seed is None or args.two_cycle_seed < 0:
            raise ValueError("two-cycle timing requires --two-cycle-seed")
        if not args.two_cycle_arm_id:
            raise ValueError("two-cycle timing requires --two-cycle-arm-id")
        exports.update(
            LIVEWAN_TWO_CYCLE_SEED=args.two_cycle_seed,
            LIVEWAN_TWO_CYCLE_ARM_ID=args.two_cycle_arm_id,
        )
    if args.ncu_bin is not None:
        if args.run_kind != "correctness":
            raise ValueError("bounded NCU capture is allowed only on a correctness run")
        if args.ncu_launch_skip_before_match < 0 or args.ncu_launch_count != 1:
            raise ValueError("bounded NCU capture requires non-negative skip and launch-count=1")
        if not args.ncu_output_name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("unsafe --ncu-output-name")
        exports.update({
            "DISTILLSCHEDULE_NCU_BIN": args.ncu_bin,
            "DISTILLSCHEDULE_NCU_NODE_RANK": args.ncu_node_rank,
            "DISTILLSCHEDULE_NCU_LOCAL_RANK": args.ncu_local_rank,
            "DISTILLSCHEDULE_NCU_KERNEL_REGEX": args.ncu_kernel_regex,
            "DISTILLSCHEDULE_NCU_LAUNCH_SKIP_BEFORE_MATCH": (
                args.ncu_launch_skip_before_match
            ),
            "DISTILLSCHEDULE_NCU_LAUNCH_COUNT": args.ncu_launch_count,
            "DISTILLSCHEDULE_NCU_SET": args.ncu_set,
            "DISTILLSCHEDULE_NCU_OUTPUT_NAME": args.ncu_output_name,
        })
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
    run_root = args.remote_output_root / args.run_id
    summary = run_root / "summary.json"
    summary_name = (
        "summarize_livewan_two_cycle_timing_world16.py"
        if args.run_kind == "two-cycle-timing"
        else (
            "summarize_livewan_trajectory_world16.py"
            if args.run_kind == "trajectory"
            else (
                "summarize_livewan_continuous_world16.py"
                if args.pipeline_variant in {
                    "c21-continuous-b16", "c21-c1-continuous-b16",
                    "c22-critic-checkpoint-rng",
                    "c23-continuous-b32",
                    "c24-continuous-b16-score-modulation",
                    "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue",
                    "c24-continuous-b16-score-modulation-b1-attribution",
                }
                else "summarize_livewan_crossscale_world16.py"
            )
        )
    )
    script = args.remote_root / "scripts/distillschedule" / summary_name
    if args.run_kind == "trajectory":
        summary_arguments = (
            f"--run-root {shlex.quote(str(run_root))} "
            f"--cycles {args.trajectory_cycles} "
            f"--expected-head {shlex.quote(args.expected_head)} "
            f"--output {shlex.quote(str(summary))}"
        )
    elif args.run_kind == "two-cycle-timing":
        summary_arguments = (
            f"--run-root {shlex.quote(str(run_root))} "
            f"--execution-mode {shlex.quote(args.execution_mode)} "
            f"--pipeline-variant {shlex.quote(args.pipeline_variant)} "
            f"--trace {shlex.quote(str(args.trace))} "
            f"--expected-head {shlex.quote(args.expected_head)} "
            f"--output {shlex.quote(str(summary))}"
        )
    else:
        summary_arguments = (
            f"--run-root {shlex.quote(str(run_root))} "
            f"--execution-mode {shlex.quote(args.execution_mode)} "
            f"--run-kind {shlex.quote(args.run_kind)} "
            f"--trace {shlex.quote(str(args.trace))} "
            f"--plan {shlex.quote(str(run_root / 'resolved_plan.node0.json'))} "
            f"--expected-head {shlex.quote(args.expected_head)} "
            f"--output {shlex.quote(str(summary))}"
        )
    remote = (
        f"export PYTHONPATH={shlex.quote(str(args.remote_root / 'src'))}:{shlex.quote(str(args.remote_root))}; "
        f"{shlex.quote(str(args.pyenv / 'bin/python'))} {shlex.quote(str(script))} "
        f"{summary_arguments}"
    )
    completed = subprocess.run(
        ssh(args.hosts[0], remote, port=args.ssh_port), check=False, timeout=900
    )
    raw = read_remote_file(args.hosts[0], summary, port=args.ssh_port)
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
    if args.execution_mode == "baseline" and args.pipeline_variant != "b1":
        raise ValueError("sequential baseline cannot select a pipeline variant")
    if (
        args.pipeline_variant
        == "c24-continuous-b16-score-modulation-b1-attribution"
        and args.run_kind != "trajectory"
    ):
        raise ValueError("C24 Critic-B1 attribution is trajectory-only")
    if args.run_kind == "two-cycle-timing":
        if args.execution_mode == "baseline" and args.pipeline_variant != "b1":
            raise ValueError("two-cycle baseline requires sequential B1")
        if (
            args.execution_mode == "pipeline"
            and args.pipeline_variant not in {"c24-continuous-b16-score-modulation", "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue"}
        ):
            raise ValueError("two-cycle pipeline requires production C24 physical B2")
    if args.run_kind == "trajectory":
        if args.execution_mode == "baseline" and args.pipeline_variant != "b1":
            raise ValueError("trajectory baseline requires sequential B1")
        if (
            args.execution_mode == "pipeline"
            and args.pipeline_variant not in {
                "c24-continuous-b16-score-modulation",
                "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue",
                "c24-continuous-b16-score-modulation-b1-attribution",
            }
        ):
            raise ValueError("trajectory pipeline requires exact C24 B2/B1 attribution")
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
