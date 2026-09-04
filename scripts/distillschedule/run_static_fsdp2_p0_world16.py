#!/usr/bin/env python3
"""Centrally launch/cancel/summarize one two-node static-FSDP2 P0 run."""
from __future__ import annotations

import argparse
import ipaddress
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


def _ssh(
    host: str, command: str, *, port: int = 22, allocate_tty: bool = False
) -> list[str]:
    result = [
        "ssh",
        *(["-tt"] if allocate_tty else []),
        "-4",
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "UpdateHostKeys=no",
        host,
        "bash",
        "-lc",
        shlex.quote(command),
    ]
    return result


def _resolve_master(host: str, *, port: int, allocate_tty: bool) -> str:
    values = subprocess.check_output(
        _ssh(host, "hostname -I", port=port, allocate_tty=allocate_tty),
        text=True,
        timeout=30,
    ).split()
    ipv4: list[str] = []
    for value in values:
        try:
            ipv4.append(str(ipaddress.IPv4Address(value)))
        except ipaddress.AddressValueError:
            continue
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
    value.add_argument("--ssh-port", type=int, default=22)
    value.add_argument(
        "--ssh-tty",
        action="store_true",
        help="Allocate the PTY required by managed platform SSH endpoints.",
    )
    value.add_argument("--run-id", required=True)
    value.add_argument("--variant", choices=("no-gan", "gan"), required=True)
    value.add_argument(
        "--mode",
        choices=("validate", "correctness", "timing", "profile", "trajectory"),
        required=True,
    )
    value.add_argument("--expected-head", required=True)
    value.add_argument("--mesh-policy", choices=("world16-fsdp", "node-hsdp"), required=True)
    value.add_argument("--rollout-batch-size", type=int, choices=(1, 2, 5), required=True)
    value.add_argument("--pack-real-score-pair", choices=("on", "off"), default="on")
    value.add_argument(
        "--attention-backend",
        choices=("sdpa", "local-fa2"),
        default="sdpa",
    )
    value.add_argument(
        "--modulation-chunk-tokens",
        type=int,
        choices=(512, 1024, 2048),
        default=512,
    )
    value.add_argument(
        "--selective-flash-checkpoint-stride",
        type=int,
        choices=(0, 2, 4, 10),
        default=0,
    )
    value.add_argument(
        "--generator-nograd-cross-kv-cache",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--generator-nograd-static-context-cache",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--generator-nograd-persistent-unshard",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--generator-nograd-persistent-unshard-flush-allocator-cache",
        choices=("on", "off"),
        default="on",
    )
    value.add_argument(
        "--cuda-allocator-expandable-segments",
        choices=("on", "off"),
        default="off",
    )
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
        "--exact-modulated-norm-backward",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--exact-modulated-norm-layernorm",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--exact-modulated-norm-fused-backward",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--exact-history-current-fusion",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--elide-explicit-rollout-sync",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--elide-critic-scalar-logs",
        choices=("on", "off"),
        default="off",
    )
    value.add_argument(
        "--elide-official-generator-free-memory",
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
    value.add_argument("--trajectory-cycles", type=int, default=1)
    value.add_argument("--trajectory-seed", type=int, default=20260903)
    value.add_argument("--trajectory-arm-id", default="single-full5")
    value.add_argument(
        "--local-evidence-root",
        type=Path,
        default=Path(
            "/home/lyy/Helios/research_artifacts/DMD-System/fsdp2_p0"
        ),
    )
    value.add_argument(
        "--summary-transport",
        choices=("full", "compact"),
        default="full",
        help=(
            "Use compact to validate the persistent remote summary over SSH "
            "without copying the large rank-expanded JSON to the controller host."
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
        "DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_BACKWARD="
        + shlex.quote(args.exact_modulated_norm_backward),
        "DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_LAYERNORM="
        + shlex.quote(args.exact_modulated_norm_layernorm),
        "DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_FUSED_BACKWARD="
        + shlex.quote(args.exact_modulated_norm_fused_backward),
        "DISTILLSCHEDULE_HELIOS_EXACT_HISTORY_CURRENT_FUSION="
        + shlex.quote(args.exact_history_current_fusion),
        "DISTILLSCHEDULE_ELIDE_EXPLICIT_ROLLOUT_SYNC="
        + shlex.quote(args.elide_explicit_rollout_sync),
        "DISTILLSCHEDULE_ELIDE_CRITIC_SCALAR_LOGS="
        + shlex.quote(args.elide_critic_scalar_logs),
        "DISTILLSCHEDULE_ELIDE_OFFICIAL_GENERATOR_FREE_MEMORY="
        + shlex.quote(args.elide_official_generator_free_memory),
        "DISTILLSCHEDULE_HELIOS_ATTENTION_BACKEND="
        + shlex.quote(args.attention_backend),
        "DISTILLSCHEDULE_HELIOS_MODULATION_CHUNK_TOKENS="
        + shlex.quote(str(args.modulation_chunk_tokens)),
        "DISTILLSCHEDULE_HELIOS_SELECTIVE_FLASH_CHECKPOINT_STRIDE="
        + shlex.quote(str(args.selective_flash_checkpoint_stride)),
        "DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_CROSS_KV_CACHE="
        + shlex.quote(args.generator_nograd_cross_kv_cache),
        "DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_STATIC_CONTEXT_CACHE="
        + shlex.quote(args.generator_nograd_static_context_cache),
        "DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_PERSISTENT_UNSHARD="
        + shlex.quote(args.generator_nograd_persistent_unshard),
        "DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_PERSISTENT_UNSHARD_FLUSH_ALLOCATOR_CACHE="
        + shlex.quote(
            args.generator_nograd_persistent_unshard_flush_allocator_cache
        ),
        "DISTILLSCHEDULE_CUDA_ALLOCATOR_EXPANDABLE_SEGMENTS="
        + shlex.quote(args.cuda_allocator_expandable_segments),
        f"DISTILLSCHEDULE_EMA_DEVICE={shlex.quote(args.ema_device)}",
        "DISTILLSCHEDULE_PROFILE_DIAGNOSTICS="
        + shlex.quote(args.profile_diagnostics),
        f"DISTILLSCHEDULE_TRAJECTORY_CYCLES={args.trajectory_cycles}",
        f"DISTILLSCHEDULE_TRAJECTORY_SEED={args.trajectory_seed}",
        "DISTILLSCHEDULE_TRAJECTORY_ARM_ID="
        + shlex.quote(args.trajectory_arm_id),
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
    return _ssh(
        args.hosts[node],
        remote,
        port=args.ssh_port,
        allocate_tty=args.ssh_tty,
    )


def _cancel(args: argparse.Namespace, node: int) -> None:
    script = args.remote_root / "scripts/distillschedule/cancel_remote_stage.sh"
    subprocess.run(
        _ssh(
            args.hosts[node],
            f"{shlex.quote(str(script))} {shlex.quote(str(_pid_file(args, node)))}",
            port=args.ssh_port,
            allocate_tty=args.ssh_tty,
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
        + f"--attention-backend {shlex.quote(args.attention_backend)} "
        + f"--modulation-chunk-tokens {args.modulation_chunk_tokens} "
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
        + "--exact-modulated-norm-backward "
        + f"{shlex.quote(args.exact_modulated_norm_backward)} "
        + "--exact-modulated-norm-layernorm "
        + f"{shlex.quote(args.exact_modulated_norm_layernorm)} "
        + "--exact-modulated-norm-fused-backward "
        + f"{shlex.quote(args.exact_modulated_norm_fused_backward)} "
        + "--exact-history-current-fusion "
        + f"{shlex.quote(args.exact_history_current_fusion)} "
        + "--elide-explicit-rollout-sync "
        + f"{shlex.quote(args.elide_explicit_rollout_sync)} "
        + "--elide-critic-scalar-logs "
        + f"{shlex.quote(args.elide_critic_scalar_logs)} "
        + "--elide-official-generator-free-memory "
        + f"{shlex.quote(args.elide_official_generator_free_memory)} "
        + "--selective-flash-checkpoint-stride "
        + f"{args.selective_flash_checkpoint_stride} "
        + "--generator-nograd-cross-kv-cache "
        + f"{shlex.quote(args.generator_nograd_cross_kv_cache)} "
        + "--generator-nograd-static-context-cache "
        + f"{shlex.quote(args.generator_nograd_static_context_cache)} "
        + "--generator-nograd-persistent-unshard "
        + f"{shlex.quote(args.generator_nograd_persistent_unshard)} "
        + "--generator-nograd-persistent-unshard-flush-allocator-cache "
        + f"{shlex.quote(args.generator_nograd_persistent_unshard_flush_allocator_cache)} "
        + "--cuda-allocator-expandable-segments "
        + f"{shlex.quote(args.cuda_allocator_expandable_segments)} "
        + ("--diagnostic-timing " if args.allow_numerical_parity_failure else "")
        + (
            f"--critic-only-step {args.critic_only_step} "
            if args.critic_only_step is not None
            else ""
        )
        + f"--output {shlex.quote(str(summary))}"
    )
    # The summarizer persists the authoritative rank-expanded JSON remotely.
    # Compact transport must not stream that large JSON through SSH merely
    # because the summarizer also writes it to stdout; retain stderr so a
    # fail-closed diagnostic is still visible to the controller.
    if args.summary_transport == "compact":
        command += " >/dev/null"
    completed = subprocess.run(
        _ssh(
            args.hosts[0],
            command,
            port=args.ssh_port,
            allocate_tty=args.ssh_tty,
        ),
        check=False,
        timeout=300,
    )
    local.mkdir(parents=True, exist_ok=True)
    if args.summary_transport == "compact":
        compact_code = (
            "import hashlib,json,sys;"
            "p=sys.argv[1];raw=open(p,'rb').read();d=json.loads(raw);"
            "keys=('status','errors','claimable_correctness','claimable_timing',"
            "'diagnostic_timing','global_full5_seconds','rank_count','mode','policy',"
            "'student_version','critic_version','ema_version');"
            "o={k:d.get(k) for k in keys};"
            "o['remote_summary_path']=p;"
            "o['remote_summary_sha256']=hashlib.sha256(raw).hexdigest();"
            "print(json.dumps(o,sort_keys=True))"
        )
        raw = subprocess.check_output(
            _ssh(
                args.hosts[0],
                shlex.quote(str(pyenv / "bin/python"))
                + " -c "
                + shlex.quote(compact_code)
                + " "
                + shlex.quote(str(summary)),
                port=args.ssh_port,
                allocate_tty=args.ssh_tty,
            ),
            timeout=60,
        )
        (local / "summary_pointer.json").write_bytes(raw)
    else:
        raw = subprocess.check_output(
            _ssh(
                args.hosts[0],
                f"cat {shlex.quote(str(summary))}",
                port=args.ssh_port,
                allocate_tty=args.ssh_tty,
            ),
            timeout=60,
        )
        (local / "summary.json").write_bytes(raw)
    value = json.loads(raw)
    if completed.returncode != 0 or value.get("status") != "PASS":
        raise RuntimeError(f"P0 summary failed: {value.get('errors')}")
    return value


def _summarize_trajectory(args: argparse.Namespace, local: Path) -> dict[str, Any]:
    """Validate the persistent rank JSONL and transport only a compact summary."""

    run_root = args.remote_root / "artifacts/runs" / args.run_id
    # The launcher passes ``RUN_ROOT/execution`` to the worker as --output-dir,
    # so its rank JSONL and state anchors live below that exact directory.
    # Keep the controller summary beside the launcher evidence, but validate the
    # worker-owned execution tree rather than a nonexistent sibling trajectory.
    execution_root = run_root / "execution"
    summary = run_root / "trajectory_summary.json"
    summarizer = args.remote_root / "scripts/distillschedule/summarize_trajectory_run.py"
    pythonpath = f"{args.remote_root}/src:{args.remote_root}"
    pyenv = Path("/tmp/distillschedule-helios-official-torch210")
    initial = {"no-gan": (2, 10, 10), "gan": (200, 1000, 800)}[args.variant]
    final = (
        initial[0] + args.trajectory_cycles,
        initial[1] + 5 * args.trajectory_cycles,
        initial[2]
        + (5 * args.trajectory_cycles if args.variant == "no-gan" else args.trajectory_cycles),
    )
    command = (
        f"export PYTHONPATH={shlex.quote(pythonpath)}; "
        + shlex.quote(str(pyenv / "bin/python"))
        + " "
        + shlex.quote(str(summarizer))
        + f" --run-root {shlex.quote(str(execution_root))}"
        + f" --run-id {shlex.quote(args.run_id)}"
        + " --workload helios14b_static_fsdp2"
        + f" --variant {shlex.quote(args.variant)}"
        + f" --cycles {args.trajectory_cycles} --world-size 16"
        + f" --expected-final-generator {final[0]}"
        + f" --expected-final-critic {final[1]}"
        + f" --expected-final-ema {final[2]}"
        + f" --output {shlex.quote(str(summary))}"
    )
    completed = subprocess.run(
        _ssh(
            args.hosts[0],
            command,
            port=args.ssh_port,
            allocate_tty=args.ssh_tty,
        ),
        check=False,
        capture_output=True,
        timeout=300,
    )
    local.mkdir(parents=True, exist_ok=True)
    raw = subprocess.check_output(
        _ssh(
            args.hosts[0],
            f"cat {shlex.quote(str(summary))}",
            port=args.ssh_port,
            allocate_tty=args.ssh_tty,
        ),
        timeout=60,
    )
    # Managed PTYs may append a CRLF/connection-close line.  Parse the object
    # after trimming terminal whitespace rather than treating terminal text as evidence.
    decoded = raw.decode()
    if args.ssh_tty:
        decoded = decoded[decoded.index("{") : decoded.rindex("}") + 1]
    value = json.loads(decoded)
    (local / "trajectory_summary.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    (local / "summarizer.stdout.log").write_bytes(completed.stdout)
    (local / "summarizer.stderr.log").write_bytes(completed.stderr)
    if completed.returncode != 0 or value.get("status") != "PASS":
        raise RuntimeError(f"trajectory summary failed: {value.get('errors')}")
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
    if (
        args.exact_modulated_norm_backward == "on"
        and args.exact_modulated_norm_epilogue != "on"
    ):
        raise ValueError(
            "Exact modulated-norm backward requires the exact epilogue"
        )
    if args.exact_modulated_norm_layernorm == "on" and not (
        args.exact_modulated_norm_epilogue == "on"
        and args.exact_modulated_norm_materialization == "on"
        and args.exact_modulated_norm_backward == "on"
    ):
        raise ValueError(
            "Fused modulated LayerNorm requires epilogue, materialization, "
            "and explicit backward"
        )
    if args.exact_modulated_norm_fused_backward == "on" and not (
        args.exact_modulated_norm_epilogue == "on"
        and args.exact_modulated_norm_materialization == "on"
        and args.exact_modulated_norm_backward == "on"
        and args.exact_modulated_norm_layernorm == "on"
    ):
        raise ValueError(
            "Fused modulated LayerNorm backward requires the complete C14 substrate"
        )
    if (
        args.selective_flash_checkpoint_stride
        and args.attention_backend != "local-fa2"
    ):
        raise ValueError(
            "selective Flash checkpoint save requires the pinned local-fa2 backend"
        )
    if (
        args.generator_nograd_cross_kv_cache == "on"
        and args.rollout_batch_size != 1
    ):
        raise ValueError("Generator no-grad cross-attention KV reuse requires B1")
    if (
        args.generator_nograd_static_context_cache == "on"
        and args.rollout_batch_size != 1
    ):
        raise ValueError("Generator no-grad static-context reuse requires B1")
    if args.generator_nograd_persistent_unshard == "on" and (
        args.variant != "gan"
        or args.mesh_policy != "world16-fsdp"
        or args.rollout_batch_size != 1
    ):
        raise ValueError(
            "Generator no-grad persistent unshard requires GAN world16 FSDP2 B1"
        )
    if (
        args.generator_nograd_persistent_unshard_flush_allocator_cache == "off"
        and args.generator_nograd_persistent_unshard != "on"
    ):
        raise ValueError(
            "disabling the persistent-unshard allocator flush requires persistent unshard"
        )
    if args.cuda_allocator_expandable_segments == "on" and not (
        args.generator_nograd_persistent_unshard == "on"
        and args.generator_nograd_persistent_unshard_flush_allocator_cache == "off"
    ):
        raise ValueError(
            "expandable CUDA segments require persistent unshard with allocator flush off"
        )
    if args.critic_only_step is not None and (
        args.variant != "no-gan" or args.mode != "profile"
    ):
        raise ValueError("--critic-only-step requires No-GAN profile mode")
    if args.operation_source and args.critic_only_step is None:
        raise ValueError("--operation-source requires --critic-only-step")
    if args.operation_source and args.selective_flash_checkpoint_stride:
        raise ValueError(
            "operation-source diagnostics cannot be combined with selective Flash save"
        )
    if not 0 <= args.operation_source_rank < 16:
        raise ValueError("--operation-source-rank must be in [0, 15]")
    if args.operation_source_target_bytes <= 0 or args.operation_source_max_events <= 0:
        raise ValueError("operation-source byte/event limits must be positive")
    if not 1 <= args.ssh_port <= 65535:
        raise ValueError("--ssh-port must be in [1, 65535]")
    if args.trajectory_cycles <= 0:
        raise ValueError("--trajectory-cycles must be positive")
    if args.mode == "trajectory":
        if not args.trajectory_arm_id.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise ValueError("unsafe trajectory arm id")
    elif args.trajectory_cycles != 1:
        raise ValueError("non-trajectory modes require --trajectory-cycles 1")
    local = args.local_evidence_root / args.run_id
    if local.exists():
        raise FileExistsError(local)
    master = args.master_addr or _resolve_master(
        args.hosts[0], port=args.ssh_port, allocate_tty=args.ssh_tty
    )
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
        value = (
            _summarize_trajectory(args, local)
            if args.mode == "trajectory"
            else _summarize(args, local)
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
