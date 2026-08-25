#!/usr/bin/env python3
"""Centrally launch/cancel/summarize one two-node static-FSDP2 P0 run."""
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
        "ssh",
        "-4",
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
    value.add_argument("--variant", choices=("no-gan", "gan"), required=True)
    value.add_argument(
        "--mode", choices=("validate", "correctness", "timing", "profile"), required=True
    )
    value.add_argument("--expected-head", required=True)
    value.add_argument("--mesh-policy", choices=("world16-fsdp", "node-hsdp"), required=True)
    value.add_argument("--rollout-batch-size", type=int, choices=(1, 2, 5), required=True)
    value.add_argument("--pack-real-score-pair", choices=("on", "off"), default="on")
    value.add_argument(
        "--generator-residual-storage",
        choices=("cpu_inplace", "cuda_out_of_place"),
        default="cuda_out_of_place",
    )
    value.add_argument(
        "--critic-residual-storage",
        choices=("cpu_inplace", "cuda_out_of_place"),
        default="cuda_out_of_place",
    )
    value.add_argument(
        "--generator-nograd-lora-fused-epilogue",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--all-role-lora-scale1-add",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--optimizer-dispatch-warmup",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--generator-nograd-residual-fusion",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--exact-grad-residual-fusion",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--exact-modulated-norm-epilogue",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--exact-modulated-norm-materialization",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--remote-root",
        type=Path,
        default=Path(
            "/share/project/shuchenweng/yongyan_liu/"
            "DMD-System_fsdp2_p0_20260815"
        ),
    )
    value.add_argument(
        "--helios-runtime-root",
        type=Path,
        default=Path(
            "/share/project/shuchenweng/yongyan_liu/"
            "DistillSchedule_gan_rolephase_exec_20260812"
        ),
    )
    value.add_argument(
        "--expected-helios-transformer-sha256",
        default="776a26c3b2921d599821f0767910830471ff94a955f5013089f9f689ac20fee7",
    )
    value.add_argument("--master-addr")
    value.add_argument("--master-port", type=int, default=29950)
    value.add_argument(
        "--nsys-bin",
        type=Path,
        help=(
            "Exact remote Nsight Systems binary used for profile mode.  The path "
            "must exist on both hosts; timing/correctness modes ignore it."
        ),
    )
    value.add_argument("--correctness-summary", type=Path)
    value.add_argument("--correctness-sha256")
    value.add_argument(
        "--allow-numerical-parity-failure",
        action="store_true",
        help=(
            "Run an explicitly diagnostic timing pass after a failed numerical "
            "parity gate; never marks timing claimable."
        ),
    )
    value.add_argument("--ema-device", choices=("cpu", "cuda"), default="cuda")
    value.add_argument("--profile-diagnostics", choices=("on", "off"), default="off")
    value.add_argument("--critic-only-step", type=int, choices=(10,))
    value.add_argument("--operation-source", action="store_true")
    value.add_argument("--operation-source-rank", type=int, default=0)
    value.add_argument("--operation-source-target-bytes", type=int, default=113049600)
    value.add_argument("--operation-source-max-events", type=int, default=256)
    value.add_argument(
        "--local-evidence-root",
        type=Path,
        default=Path(
            "/home/lyy/Helios/research_artifacts/DMD-System/fsdp2_p0"
        ),
    )
    value.add_argument("--timeout-seconds", type=int, default=30000)
    value.add_argument("--dry-run", action="store_true")
    return value


def _pid_file(args: argparse.Namespace, node: int) -> Path:
    return (
        args.remote_root
        / "artifacts"
        / "runs"
        / args.run_id
        / "controller"
        / f"node_{node}.pid"
    )


def _command(args: argparse.Namespace, node: int, master: str) -> list[str]:
    launcher = (
        args.remote_root
        / "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh"
    )
    wrapper = args.remote_root / "scripts/distillschedule/remote_stage_wrapper.sh"
    exports = [
        f"DMD_SYSTEM_ROOT={shlex.quote(str(args.remote_root))}",
        f"HELIOS_RUNTIME_ROOT={shlex.quote(str(args.helios_runtime_root))}",
        "DISTILLSCHEDULE_EXPECTED_HELIOS_TRANSFORMER_SHA256="
        + shlex.quote(args.expected_helios_transformer_sha256),
        "DISTILLSCHEDULE_GENERATOR_RESIDUAL_STORAGE="
        + shlex.quote(args.generator_residual_storage),
        "DISTILLSCHEDULE_CRITIC_RESIDUAL_STORAGE="
        + shlex.quote(args.critic_residual_storage),
        "DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE="
        + shlex.quote(args.generator_nograd_lora_fused_epilogue),
        "DISTILLSCHEDULE_HELIOS_ALL_ROLE_LORA_SCALE1_ADD="
        + shlex.quote(args.all_role_lora_scale1_add),
        "DISTILLSCHEDULE_OPTIMIZER_DISPATCH_WARMUP="
        + shlex.quote(args.optimizer_dispatch_warmup),
        "DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_RESIDUAL_FUSION="
        + shlex.quote(args.generator_nograd_residual_fusion),
        "DISTILLSCHEDULE_HELIOS_EXACT_GRAD_RESIDUAL_FUSION="
        + shlex.quote(args.exact_grad_residual_fusion),
        "DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_EPILOGUE="
        + shlex.quote(args.exact_modulated_norm_epilogue),
        "DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_MATERIALIZATION="
        + shlex.quote(args.exact_modulated_norm_materialization),
        f"DISTILLSCHEDULE_EMA_DEVICE={shlex.quote(args.ema_device)}",
        "DISTILLSCHEDULE_PROFILE_DIAGNOSTICS="
        + shlex.quote(args.profile_diagnostics),
    ]
    if args.nsys_bin is not None:
        exports.append("NSYS_BIN=" + shlex.quote(str(args.nsys_bin)))
    if args.critic_only_step is not None:
        exports.append(f"DISTILLSCHEDULE_CRITIC_ONLY_STEP={args.critic_only_step}")
    if args.operation_source:
        exports.extend(
            (
                "DISTILLSCHEDULE_OPERATION_SOURCE=on",
                f"DISTILLSCHEDULE_OPERATION_SOURCE_RANK={args.operation_source_rank}",
                "DISTILLSCHEDULE_OPERATION_SOURCE_TARGET_BYTES="
                f"{args.operation_source_target_bytes}",
                "DISTILLSCHEDULE_OPERATION_SOURCE_MAX_EVENTS="
                f"{args.operation_source_max_events}",
            )
        )
    if args.mode == "timing" and not args.allow_numerical_parity_failure:
        if args.correctness_summary is None or not args.correctness_sha256:
            raise ValueError("timing requires a pinned correctness summary")
        exports.extend(
            (
                "DISTILLSCHEDULE_FSDP2_CORRECTNESS_SUMMARY="
                + shlex.quote(str(args.correctness_summary)),
                "DISTILLSCHEDULE_FSDP2_CORRECTNESS_SHA256="
                + shlex.quote(args.correctness_sha256),
            )
        )
    if args.allow_numerical_parity_failure:
        if args.mode != "timing":
            raise ValueError("--allow-numerical-parity-failure is timing-only")
        exports.append("DISTILLSCHEDULE_DIAGNOSTIC_TIMING=1")
    launch = " ".join(
        shlex.quote(str(value))
        for value in (
            "bash",
            launcher,
            node,
            args.run_id,
            args.variant,
            args.mode,
            args.expected_head,
            master,
            args.master_port,
            args.mesh_policy,
            args.rollout_batch_size,
            args.pack_real_score_pair,
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
    subprocess.run(
        _ssh(
            args.hosts[node],
            f"{shlex.quote(str(script))} {shlex.quote(str(_pid_file(args, node)))}",
        ),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=90,
    )


def _summarize(args: argparse.Namespace, local: Path) -> dict[str, Any]:
    run_root = args.remote_root / "artifacts/runs" / args.run_id
    summary = run_root / "summary.json"
    summarizer = (
        args.remote_root
        / "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py"
    )
    pythonpath = f"{args.remote_root}/src:{args.remote_root}"
    pyenv = Path("/tmp/distillschedule-helios-official-torch210")
    command = (
        f"export PYTHONPATH={shlex.quote(pythonpath)}; "
        + "export LD_LIBRARY_PATH="
        + shlex.quote(str(pyenv / "lib"))
        + "${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}; "
        + shlex.quote(str(pyenv / "bin/python"))
        + " "
        + f"{shlex.quote(str(summarizer))} "
        + f"--run-root {shlex.quote(str(run_root))} "
        + f"--variant {shlex.quote(args.variant)} --mode {shlex.quote(args.mode)} "
        + f"--mesh-policy {shlex.quote(args.mesh_policy)} "
        + f"--rollout-batch-size {args.rollout_batch_size} "
        + f"--pack-real-score-pair {shlex.quote(args.pack_real_score_pair)} "
        + "--generator-residual-storage "
        + f"{shlex.quote(args.generator_residual_storage)} "
        + "--critic-residual-storage "
        + f"{shlex.quote(args.critic_residual_storage)} "
        + "--generator-nograd-lora-fused-epilogue "
        + f"{shlex.quote(args.generator_nograd_lora_fused_epilogue)} "
        + "--all-role-lora-scale1-add "
        + f"{shlex.quote(args.all_role_lora_scale1_add)} "
        + "--optimizer-dispatch-warmup "
        + f"{shlex.quote(args.optimizer_dispatch_warmup)} "
        + "--generator-nograd-residual-fusion "
        + f"{shlex.quote(args.generator_nograd_residual_fusion)} "
        + "--exact-grad-residual-fusion "
        + f"{shlex.quote(args.exact_grad_residual_fusion)} "
        + "--exact-modulated-norm-epilogue "
        + f"{shlex.quote(args.exact_modulated_norm_epilogue)} "
        + "--exact-modulated-norm-materialization "
        + f"{shlex.quote(args.exact_modulated_norm_materialization)} "
        + ("--diagnostic-timing " if args.allow_numerical_parity_failure else "")
        + (
            f"--critic-only-step {args.critic_only_step} "
            if args.critic_only_step is not None
            else ""
        )
        + f"--output {shlex.quote(str(summary))}"
    )
    completed = subprocess.run(
        _ssh(args.hosts[0], command), check=False, timeout=300
    )
    raw = subprocess.check_output(
        _ssh(args.hosts[0], f"cat {shlex.quote(str(summary))}"), timeout=60
    )
    local.mkdir(parents=True, exist_ok=True)
    (local / "summary.json").write_bytes(raw)
    value = json.loads(raw)
    if completed.returncode != 0 or value.get("status") != "PASS":
        raise RuntimeError(f"P0 summary failed: {value.get('errors')}")
    return value


def main() -> int:
    args = parser().parse_args()
    if not args.run_id.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("unsafe run id")
    if len(args.expected_head) != 40:
        raise ValueError("--expected-head must be a full Git SHA")
    if len(args.expected_helios_transformer_sha256) != 64:
        raise ValueError("--expected-helios-transformer-sha256 must be SHA256")
    if (
        args.generator_nograd_lora_fused_epilogue == "on"
        and args.all_role_lora_scale1_add == "on"
    ):
        raise ValueError("LoRA epilogue candidates are mutually exclusive")
    if args.optimizer_dispatch_warmup == "on" and args.mesh_policy != "node-hsdp":
        raise ValueError("optimizer dispatch warmup is restricted to node-HSDP")
    if args.generator_nograd_residual_fusion == "on" and (
        args.rollout_batch_size != 1
        or args.generator_residual_storage != "cuda_out_of_place"
    ):
        raise ValueError(
            "Generator no-grad residual fusion requires B1 and cuda_out_of_place"
        )
    if args.exact_grad_residual_fusion == "on" and (
        args.rollout_batch_size != 1
        or args.generator_residual_storage != "cuda_out_of_place"
        or args.critic_residual_storage != "cuda_out_of_place"
    ):
        raise ValueError(
            "Exact grad residual fusion requires B1 and cuda_out_of_place "
            "for both Generator and Critic"
        )
    if args.exact_modulated_norm_epilogue == "on" and args.rollout_batch_size != 1:
        raise ValueError("Exact modulated-norm epilogue requires B1")
    if (
        args.exact_modulated_norm_materialization == "on"
        and args.exact_modulated_norm_epilogue != "on"
    ):
        raise ValueError(
            "Exact modulation materialization requires the exact epilogue"
        )
    if args.critic_only_step is not None and (
        args.variant != "no-gan" or args.mode != "profile"
    ):
        raise ValueError("--critic-only-step requires No-GAN profile mode")
    if args.operation_source and args.critic_only_step is None:
        raise ValueError("--operation-source requires --critic-only-step")
    if not 0 <= args.operation_source_rank < 16:
        raise ValueError("--operation-source-rank must be in [0, 15]")
    if args.operation_source_target_bytes <= 0 or args.operation_source_max_events <= 0:
        raise ValueError("operation-source byte/event limits must be positive")
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
            raise RuntimeError(f"P0 remote run failed or timed out: returncodes={codes}")
        value = _summarize(args, local)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
