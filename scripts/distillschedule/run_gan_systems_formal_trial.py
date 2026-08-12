#!/usr/bin/env python3
"""Run one real systems-only GAN Full5 formal trial.

This is a deliberately thin experiment adapter.  ``baseline`` delegates to
the validated PR4 two-node controller.  ``pipeline`` launches the already
validated PR5 8G+8Score executor.  Both paths then build the same actual-
consumption/compute receipts, validate the exact unprofiled timing range, and
copy only the small formal evidence bundle back beside ``result.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_HOSTS = ("baai-helios-node0", "baai-helios-node1")
DEFAULT_MASTER_ADDR = "172.24.129.137"
DEFAULT_BASELINE_LOCAL_ROOT = Path("/home/lyy/DMD-System_gan_baseline_pr4")
DEFAULT_BASELINE_REMOTE_ROOT = Path(
    "/share/project/shuchenweng/yongyan_liu/DMD-System_gan_pr4_exec_20260810"
)
DEFAULT_PIPELINE_REMOTE_ROOT = Path(
    "/share/project/shuchenweng/yongyan_liu/DMD-System_gan_pipeline_exec_20260811"
)
DEFAULT_PIPELINE_RUNTIME_ROOT = Path(
    "/share/project/shuchenweng/yongyan_liu/DistillSchedule_gan_pipeline_exec_20260811"
)
DEFAULT_SITE_ENV = Path(
    "/share/project/shuchenweng/yongyan_liu/gan_systems_formal_20260811.env"
)
DEFAULT_TRACE = Path(
    "/share/project/shuchenweng/DistillSchedule_assets/gan/systems_only_gan_trace_rank16_v1"
)
DEFAULT_TARGET = Path(
    "/share/project/shuchenweng/DistillSchedule_assets/gan/systems_only_gan_pipeline_world8_v3"
)
DEFAULT_GENERATOR = Path(
    "/share/project/shuchenweng/DistillSchedule_assets/precast/helios-distilled-transformer-ode-mixed-bf16-v1"
)
DEFAULT_SCORE = Path(
    "/share/project/shuchenweng/DistillSchedule_assets/precast/helios-score-transformer-mixed-bf16-gan-systems-v2"
)
DEFAULT_NEGATIVE = Path(
    "/share/project/shuchenweng/DistillSchedule_assets/m1/negative_prompt_official_v1/negative_prompt_embedding.pt"
)
DEFAULT_PIPELINE_MMAP = Path(
    "/share/project/shuchenweng/DistillSchedule_assets/gan/pipeline_fixed_policy_mmap_v2_postds_bf16"
)
DEFAULT_CONFIG = DEFAULT_BASELINE_REMOTE_ROOT / Path(
    "artifacts/runs/gan_seq_systems_functional_20260811_008/"
    "resolved_helios_config.node0.yaml"
)
PYTHON_REMOTE = Path("/tmp/distillschedule-helios-official-torch210/bin/python")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _prepare_result_path(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"immutable result already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _workload_digest(path: Path) -> str:
    value = _load(path)
    digest = value.get("workload_contract_digest") or value.get("shared_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"workload contract lacks a SHA256 digest: {path}")
    return digest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_optimized_policy_manifest(
    path: Path,
    *,
    workload_digest: str,
    generator_streamed_count: int,
    score_streamed_count: int,
    generator_critic_phase_resident: bool,
    score_critic_activation_offload_min_mib: int,
) -> str:
    """Validate the immutable real-14B capacity gate before formal timing.

    The worker receives only this manifest's digest.  This local gate binds the
    digest to the exact admitted policy and to the immutable capacity result;
    it prevents a non-default policy from becoming formal merely by omitting
    ``--capacity-search``.
    """

    path = path.expanduser().resolve()
    value = _load(path)
    if value.get("schema_version") != "distillschedule.gan_validated_policy.v1":
        raise RuntimeError("unsupported validated-policy manifest schema")
    if value.get("status") != "PASS":
        raise RuntimeError("validated-policy manifest is not PASS")
    if value.get("workload_contract_digest") != workload_digest:
        raise RuntimeError("validated-policy workload digest mismatch")
    expected = {
        "generator_streamed_block_count": int(generator_streamed_count),
        "score_streamed_block_count": int(score_streamed_count),
        "generator_critic_phase_resident": bool(
            generator_critic_phase_resident
        ),
        "score_critic_activation_offload_min_mib": int(
            score_critic_activation_offload_min_mib
        ),
        "streaming_window": 1,
        "host_policy": "HOST_SHARED_IMMUTABLE",
        "prefetch": False,
    }
    if value.get("validated_policy") != expected:
        raise RuntimeError(
            "validated-policy settings do not match requested formal policy"
        )
    capacity_result = Path(str(value.get("capacity_result", ""))).expanduser()
    if not capacity_result.is_absolute():
        capacity_result = (path.parent / capacity_result).resolve()
    if not capacity_result.is_file():
        raise RuntimeError("validated-policy capacity result is missing")
    expected_result_sha = value.get("capacity_result_sha256")
    if not isinstance(expected_result_sha, str) or _sha256(
        capacity_result
    ) != expected_result_sha:
        raise RuntimeError("validated-policy capacity result hash mismatch")
    result = _load(capacity_result)
    if result.get("status") != "PASS" or not result.get("capacity_search"):
        raise RuntimeError("validated-policy evidence is not a PASS capacity gate")
    if result.get("workload_contract_digest") != workload_digest:
        raise RuntimeError("capacity-result workload digest mismatch")
    fixed = result.get("fixed_memory_policy", {})
    if not isinstance(fixed, dict):
        raise RuntimeError("capacity result lacks fixed memory policy")
    observed = {
        "generator_streamed_block_count": len(
            fixed.get("generator_streamed_blocks", [])
        ),
        "score_streamed_block_count": len(fixed.get("score_streamed_blocks", [])),
        "generator_critic_phase_resident": fixed.get(
            "generator_critic_phase_resident"
        ),
        "score_critic_activation_offload_min_mib": int(
            fixed.get("score_critic_activation_offload_min_bytes", -1)
        )
        // (1024 * 1024),
        "streaming_window": fixed.get("streaming_window"),
        "host_policy": fixed.get("host_policy"),
        "prefetch": fixed.get("prefetch"),
    }
    if observed != expected:
        raise RuntimeError("capacity-result policy does not match manifest")
    return _sha256(path)


def _role_streaming_policy(args: argparse.Namespace) -> tuple[int, int, bool]:
    legacy = int(getattr(args, "streamed_block_count", 40))
    generator = getattr(args, "generator_streamed_block_count", None)
    score = getattr(args, "score_streamed_block_count", None)
    if (generator is None) != (score is None):
        raise RuntimeError(
            "Generator and Score streamed-block counts must be specified together"
        )
    if generator is None:
        generator = score = legacy
    generator = int(generator)
    score = int(score)
    if not 0 <= generator <= 40 or not 0 <= score <= 40:
        raise RuntimeError("role streamed-block counts must be in [0,40]")
    phase_resident = bool(
        getattr(args, "generator_critic_phase_resident", False)
    )
    return generator, score, phase_resident


def _tail_indices(count: int) -> tuple[int, ...]:
    count = int(count)
    if not 0 <= count <= 40:
        raise ValueError("streamed-block count must be in [0,40]")
    return tuple(range(40 - count, 40)) if count else ()


def _run(command: Sequence[str], *, log: Path, timeout: int = 24000) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as handle:
        completed = subprocess.run(
            list(command),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {shlex.join(command)}; log={log}"
        )


def _ssh(host: str, script: str, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", host, "bash", "-lc", shlex.quote(script)],
        check=False,
        capture_output=capture,
        text=True,
        timeout=300,
    )


def _ssh_checked(host: str, script: str) -> str:
    completed = _ssh(host, script, capture=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"remote command failed on {host} rc={completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout


def _rsync_from(host: str, source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["rsync", "-a", "--checksum", f"{host}:{source}", str(destination)],
        log=destination.parent / f"rsync_{destination.name}.log",
        timeout=1800,
    )


def _memory_events(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    values: dict[str, int] = {}
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            values[fields[0]] = int(fields[1])
        except ValueError:
            continue
    return values


def _classify_pipeline_failure_evidence(root: Path) -> dict[str, Any]:
    """Classify only observed capacity signatures; never infer OOM from rc/NCCL."""

    text_parts: list[str] = []
    for path in sorted((root / "launcher").glob("*.log")):
        text_parts.append(path.read_text(errors="replace"))
    for path in sorted((root / "results").glob("rank_*.json")):
        text_parts.append(path.read_text(errors="replace"))
    text = "\n".join(text_parts)
    cuda_patterns = {
        "torch_out_of_memory": r"(?:torch\.)?OutOfMemoryError",
        "cuda_out_of_memory": r"CUDA out of memory",
    }
    cuda_matches = {
        name: bool(re.search(pattern, text, flags=re.IGNORECASE))
        for name, pattern in cuda_patterns.items()
    }
    cgroup: list[dict[str, Any]] = []
    for node_rank in (0, 1):
        before = _memory_events(
            root / "launcher" / f"node_{node_rank}.memory.events.before"
        )
        after = _memory_events(
            root / "launcher" / f"node_{node_rank}.memory.events.after"
        )
        cgroup.append(
            {
                "node_rank": node_rank,
                "before": before,
                "after": after,
                "oom_delta": after.get("oom", 0) - before.get("oom", 0),
                "oom_kill_delta": after.get("oom_kill", 0)
                - before.get("oom_kill", 0),
            }
        )
    cuda_oom = any(cuda_matches.values())
    cgroup_oom = any(
        row["oom_delta"] > 0 or row["oom_kill_delta"] > 0 for row in cgroup
    )
    return {
        "schema_version": "distillschedule.gan_pipeline_failure_classification.v1",
        "cuda_signatures": cuda_matches,
        "cgroup": cgroup,
        "actual_capacity_failure": bool(cuda_oom or cgroup_oom),
        "classification_rule": (
            "capacity only if an explicit CUDA OutOfMemory signature or positive "
            "cgroup oom/oom_kill delta is observed; return code, NCCL, timeout, and "
            "missing artifacts are not capacity evidence"
        ),
    }


def _collect_pipeline_failure_evidence(
    args: argparse.Namespace, *, run_root: Path, destination: Path
) -> dict[str, Any]:
    """Best-effort fetch of a failed immutable candidate's small evidence set."""

    destination.mkdir(parents=True, exist_ok=False)
    fetched: list[str] = []
    errors: dict[str, str] = {}
    for relative in ("launcher", "results", "block_streaming", "activation_offload"):
        try:
            _rsync_from(
                args.hosts[0],
                f"{run_root}/{relative}/",
                destination / relative,
            )
            fetched.append(relative)
        except BaseException as exc:
            errors[relative] = repr(exc)
    classification = _classify_pipeline_failure_evidence(destination)
    report = {
        "schema_version": "distillschedule.gan_pipeline_failure_evidence.v1",
        "status": "COLLECTED" if fetched else "UNAVAILABLE",
        "remote_run_root": str(run_root),
        "fetched": fetched,
        "collection_errors": errors,
        "classification": classification,
    }
    _atomic_json(destination / "failure_evidence.json", report)
    return report


def _assert_remote_idle(hosts: Sequence[str]) -> None:
    audit = r'''set -euo pipefail
test -x /usr/bin/nvidia-smi || command -v nvidia-smi >/dev/null
procs="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')"
test -z "$procs" || { echo "gpu_processes_present=$procs" >&2; exit 81; }
flock -n /tmp/distillschedule_gpu_node.lock -c true || { echo gpu_lock_busy >&2; exit 73; }
'''
    for host in hosts:
        _ssh_checked(host, audit)


def _baseline_launch(args: argparse.Namespace, run_log: Path) -> Path:
    controller = args.baseline_local_root / "scripts/distillschedule/run_routea_gan_world16.py"
    if not controller.is_file():
        raise RuntimeError(f"missing PR4 controller: {controller}")
    command = [
        sys.executable,
        str(controller),
        "--hosts",
        ",".join(args.hosts),
        "--run-id",
        args.run_id,
        "--repo-root",
        str(args.baseline_local_root),
        "--remote-root",
        str(args.baseline_remote_root),
        "--site-env",
        str(args.site_env),
        "--master-addr",
        args.master_addr,
        "--master-port",
        str(args.master_port),
        "--expected-head",
        args.baseline_expected_head,
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    _run(command, log=run_log, timeout=args.timeout_seconds + 600)
    return args.baseline_remote_root / "artifacts" / "runs" / args.run_id


def _pipeline_worker_script(args: argparse.Namespace, node_rank: int, run_root: Path) -> str:
    rank_log = run_root / "launcher" / f"node_{node_rank}.log"
    state = run_root / "launcher" / f"node_{node_rank}.state"
    memory_before = run_root / "launcher" / f"node_{node_rank}.memory.events.before"
    memory_after = run_root / "launcher" / f"node_{node_rank}.memory.events.after"
    result_root = run_root / "results"
    timing_root = run_root / "timing_receipts"
    checkpoint = args.target_package / "checkpoint-1000"
    logical = args.target_package / "logical_package"
    command = [
        str(PYTHON_REMOTE),
        "-m",
        "torch.distributed.run",
        "--nnodes=2",
        "--nproc-per-node=8",
        f"--node-rank={node_rank}",
        f"--master-addr={args.master_addr}",
        f"--master-port={args.master_port}",
        str(args.pipeline_runtime_root / "scripts/distillschedule/probe_gan_pipeline_world16.py"),
        "--config",
        str(args.config),
        "--checkpoint",
        str(checkpoint),
        "--logical-package",
        str(logical),
        "--trace-bundle",
        str(args.trace_bundle),
        "--generator-root",
        str(args.generator_root),
        "--score-root",
        str(args.score_root),
        "--negative-prompt",
        str(args.negative_prompt),
        "--mmap-cache",
        str(args.pipeline_mmap),
        "--output-dir",
        str(result_root),
        "--run-nonce",
        str(args.run_nonce),
    ]
    profile_nsys = bool(getattr(args, "profile_nsys", False))
    detailed_events = bool(getattr(args, "capacity_search", False)) or profile_nsys
    if detailed_events:
        command.append("--detailed-events")
    if profile_nsys:
        command = [
            str(args.pipeline_remote_root / "scripts/distillschedule/gan_launch_shim.sh"),
            *command,
        ]
    command_text = shlex.join(command)
    exact_timing = "0" if profile_nsys else "1"
    nsys_profile = "1" if profile_nsys else "0"
    generator_streamed_count, score_streamed_count, phase_resident = (
        _role_streaming_policy(args)
    )
    score_critic_activation_min_mib = int(
        getattr(args, "score_critic_activation_offload_min_mib", 0)
    )
    if score_critic_activation_min_mib < 0:
        raise ValueError("Score Critic activation-offload threshold must be non-negative")
    score_critic_activation_min_bytes = score_critic_activation_min_mib * 1024 * 1024
    generator_streamed_indices = ",".join(
        str(index) for index in _tail_indices(generator_streamed_count)
    )
    score_streamed_indices = ",".join(
        str(index) for index in _tail_indices(score_streamed_count)
    )
    capacity_search = "1" if bool(getattr(args, "capacity_search", False)) else "0"
    validated_policy_digest = str(
        getattr(args, "validated_policy_digest", "") or ""
    )
    return f'''set -Eeuo pipefail
exec 9>/tmp/distillschedule_gpu_node.lock
flock -n 9 || {{ echo gpu_lock_busy >&2; exit 73; }}
procs="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')"
test -z "$procs" || {{ echo "gpu_processes_present=$procs" >&2; exit 81; }}
mkdir -p {shlex.quote(str(run_root / 'launcher'))} {shlex.quote(str(result_root))} {shlex.quote(str(timing_root))} {shlex.quote(str(run_root / 'trace_access'))}
cat /sys/fs/cgroup/memory.events > {shlex.quote(str(memory_before))} 2>/dev/null || true
before_events="$(cat {shlex.quote(str(memory_before))} 2>/dev/null || true)"
{{ echo start=$(date -Is); printf 'memory_events_before=%q\n' "$before_events"; }} > {shlex.quote(str(state))}
export PYTHONPATH={shlex.quote(str(args.pipeline_remote_root / 'src'))}:{shlex.quote(str(args.pipeline_runtime_root))}:{shlex.quote(str(args.pipeline_remote_root))}
export DMD_SYSTEM_ROOT={shlex.quote(str(args.pipeline_remote_root))}
export DISTILLSCHEDULE_OUTPUT_DIR={shlex.quote(str(run_root))}
export DISTILLSCHEDULE_NODE_RANK={node_rank}
export DISTILLSCHEDULE_PYTHON={shlex.quote(str(PYTHON_REMOTE))}
export HF_HOME=/share/project/shuchenweng/DistillSchedule_assets/hf_home_official_torch210
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LOCAL_KERNELS=kernels-community/flash-attn2=/share/project/shuchenweng/DistillSchedule_assets/hf_home_official_torch210/hub/models--kernels-community--flash-attn2/snapshots/6a810a7a010ba2aad6613f1f0f7b226fb8bdee3c/build/torch210-cxx11-cu128-x86_64-linux
export MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 NCCL_DEBUG=WARN NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9
export DISTILLSCHEDULE_OPERATION_TRACE_ACCESS_DIR={shlex.quote(str(run_root / 'trace_access'))}
export DISTILLSCHEDULE_STABLE_BLOCK_NOISE=1 DISTILLSCHEDULE_NSYS_ANNOTATE=1
export DISTILLSCHEDULE_GAN_NSYS_PROFILE={nsys_profile} DISTILLSCHEDULE_GAN_EXACT_TIMING={exact_timing}
export DISTILLSCHEDULE_GAN_CAPACITY_SEARCH={capacity_search}
export DISTILLSCHEDULE_GAN_VALIDATED_POLICY_DIGEST={shlex.quote(validated_policy_digest)}
unset DISTILLSCHEDULE_GAN_STREAMED_BLOCK_COUNT DISTILLSCHEDULE_GAN_STREAMED_BLOCK_INDICES
export DISTILLSCHEDULE_GAN_GENERATOR_STREAMED_BLOCK_COUNT={generator_streamed_count}
export DISTILLSCHEDULE_GAN_GENERATOR_STREAMED_BLOCK_INDICES={shlex.quote(generator_streamed_indices)}
export DISTILLSCHEDULE_GAN_SCORE_STREAMED_BLOCK_COUNT={score_streamed_count}
export DISTILLSCHEDULE_GAN_SCORE_STREAMED_BLOCK_INDICES={shlex.quote(score_streamed_indices)}
export DISTILLSCHEDULE_GAN_GENERATOR_CRITIC_PHASE_RESIDENT={1 if phase_resident else 0}
export DISTILLSCHEDULE_GAN_SCORE_CRITIC_ACTIVATION_OFFLOAD_MIN_BYTES={score_critic_activation_min_bytes}
export DISTILLSCHEDULE_GAN_TIMING_RECEIPT_DIR={shlex.quote(str(timing_root))}
set +e
timeout --signal=TERM --kill-after=180s {args.timeout_seconds}s {command_text} > {shlex.quote(str(rank_log))} 2>&1
rc=$?
set -e
cat /sys/fs/cgroup/memory.events > {shlex.quote(str(memory_after))} 2>/dev/null || true
after_events="$(cat {shlex.quote(str(memory_after))} 2>/dev/null || true)"
gpu_after="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')"
{{ echo stop=$(date -Is); echo rc=$rc; printf 'memory_events_after=%q\n' "$after_events"; printf 'gpu_processes_after=%q\n' "$gpu_after"; }} >> {shlex.quote(str(state))}
test -z "$gpu_after" || rc=88
exit "$rc"
'''


def _cancel_pipeline(args: argparse.Namespace, node_rank: int, run_root: Path) -> None:
    pid_file = run_root / "launcher" / f"remote_node_{node_rank}.pid"
    cancel = args.pipeline_remote_root / "scripts/distillschedule/cancel_remote_stage.sh"
    _ssh(args.hosts[node_rank], f"{shlex.quote(str(cancel))} {shlex.quote(str(pid_file))}")


def _pipeline_launch(args: argparse.Namespace, run_log: Path) -> Path:
    run_root = args.pipeline_remote_root / "artifacts" / "runs" / args.run_id
    _ssh_checked(
        args.hosts[0],
        f"test ! -e {shlex.quote(str(run_root))} && mkdir -p {shlex.quote(str(run_root / 'launcher'))}",
    )
    processes: list[subprocess.Popen[bytes]] = []
    logs = []
    try:
        for node_rank, host in enumerate(args.hosts):
            node_log = run_log.with_name(f"{run_log.stem}_node{node_rank}.log")
            handle = node_log.open("wb")
            logs.append(handle)
            wrapper = args.pipeline_remote_root / "scripts/distillschedule/remote_stage_wrapper.sh"
            pid_file = run_root / "launcher" / f"remote_node_{node_rank}.pid"
            script = _pipeline_worker_script(args, node_rank, run_root)
            remote = (
                f"{shlex.quote(str(wrapper))} {shlex.quote(str(pid_file))} "
                f"bash -lc {shlex.quote(script)}"
            )
            processes.append(
                subprocess.Popen(
                    ["ssh", host, "bash", "-lc", shlex.quote(remote)],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        deadline = time.monotonic() + args.timeout_seconds + 600
        while True:
            codes = [process.poll() for process in processes]
            failed = [index for index, code in enumerate(codes) if code not in (None, 0)]
            if failed:
                for index in range(2):
                    _cancel_pipeline(args, index, run_root)
                for process in processes:
                    try:
                        process.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        process.kill()
                raise RuntimeError(f"pipeline node failure return_codes={codes}")
            if all(code == 0 for code in codes):
                break
            if time.monotonic() >= deadline:
                for index in range(2):
                    _cancel_pipeline(args, index, run_root)
                raise RuntimeError("pipeline controller timeout")
            time.sleep(2)
    finally:
        for handle in logs:
            handle.close()
    return run_root


def _remote_finalize(args: argparse.Namespace, family: str, run_root: Path) -> dict[str, Any]:
    receipt_root = run_root / "execution_receipts"
    dmd_root = args.baseline_remote_root if family == "baseline" else args.pipeline_remote_root
    runtime_root = (
        Path("/share/project/shuchenweng/yongyan_liu/DistillSchedule_gan_runtime_20260810")
        if family == "baseline"
        else args.pipeline_runtime_root
    )
    run_evidence_root = run_root if family == "baseline" else run_root
    profile_nsys = bool(getattr(args, "profile_nsys", False))
    generator_streamed_count, score_streamed_count, phase_resident = (
        _role_streaming_policy(args)
    )
    score_critic_activation_min_bytes = int(
        getattr(args, "score_critic_activation_offload_min_mib", 0)
    ) * 1024 * 1024
    role_receipt_args = ""
    if family == "pipeline":
        role_receipt_args = (
            f" --expected-generator-streamed-block-count {generator_streamed_count}"
            f" --expected-score-streamed-block-count {score_streamed_count}"
            + (" --generator-critic-phase-resident" if phase_resident else "")
        )
    timing_validation = "" if profile_nsys else f'''\
{shlex.quote(str(PYTHON_REMOTE))} {shlex.quote(str(dmd_root / 'scripts/distillschedule/validate_gan_timing_receipts.py'))} \\
  --receipt-dir {shlex.quote(str(run_root / 'timing_receipts'))} --world-size 16 \\
  --output {shlex.quote(str(receipt_root / 'timing_validation.json'))}
'''
    profile_validation = "" if not profile_nsys else f'''\
test -s {shlex.quote(str(run_root / 'nsight/node_0/gan_full5_node0.nsys-rep'))}
test -s {shlex.quote(str(run_root / 'nsight/node_1/gan_full5_node1.nsys-rep'))}
test -s {shlex.quote(str(run_root / 'nsight/node_0/profile_manifest.txt'))}
test -s {shlex.quote(str(run_root / 'nsight/node_1/profile_manifest.txt'))}
{shlex.quote(str(PYTHON_REMOTE))} {shlex.quote(str(dmd_root / 'scripts/distillschedule/validate_gan_nsys_receipts.py'))} \\
  --receipt-dir {shlex.quote(str(run_root / 'nsight/range_receipts'))} --world-size 16 \\
  --local-world-size 8 --output {shlex.quote(str(run_root / 'nsight/range_receipts_validation.json'))}
'''
    build = f'''set -Eeuo pipefail
export PYTHONPATH={shlex.quote(str(dmd_root / 'src'))}:{shlex.quote(str(dmd_root))}:{shlex.quote(str(runtime_root))}
test ! -e {shlex.quote(str(receipt_root))} || {{ echo receipt_root_exists >&2; exit 82; }}
{shlex.quote(str(PYTHON_REMOTE))} {shlex.quote(str(dmd_root / 'scripts/distillschedule/build_gan_execution_receipts.py'))} \
  --family {family} --run-root {shlex.quote(str(run_evidence_root))} \
  --trace-bundle {shlex.quote(str(args.trace_bundle))} \
  --output-dir {shlex.quote(str(receipt_root))} --expected-streamed-block-count {int(getattr(args, 'streamed_block_count', 40))}{role_receipt_args}
{timing_validation}{profile_validation}
'''
    _ssh_checked(args.hosts[0], build)
    summary_script = f'''set -Eeuo pipefail
{shlex.quote(str(PYTHON_REMOTE))} - <<'PY'
import json
from pathlib import Path
root=Path({str(run_root)!r})
result_root=root/({'results'!r} if {family!r}=='pipeline' else {'phase_trace'!r})
out={{'family':{family!r},'run_root':str(root)}}
if {family!r}=='pipeline':
  files=sorted(result_root.glob('rank_*.json'))
  values=[json.loads(p.read_text()) for p in files]
  out['rank_count']=len(values)
  out['rank_statuses']={{str(v.get('rank')):v.get('status') for v in values}}
  out['updates']={{str(v.get('rank')):v.get('updates') for v in values}}
  out['max_allocated_bytes']=max(int(v.get('max_allocated_bytes',0)) for v in values) if values else 0
  out['max_reserved_bytes']=max(int(v.get('max_reserved_bytes',0)) for v in values) if values else 0
  out['streamer_stats_by_role']={{}}
  for role in ('generator','score'):
    rows=[v for v in values if v.get('role')==role]
    out['streamer_stats_by_role'][role]={{
      key:[row.get('streamer_stats',{{}}).get(key) for row in rows]
      for key in (
        'streamed_block_count','materialize_count','materialized_bytes',
        'h2d_wall_seconds','peak_resident_bytes','phase_resident_enter_count',
        'phase_resident_exit_count','resident_bytes'
      )
    }}
  out['ema_optimization_steps']={{str(v.get('rank')):v.get('ema_optimization_step') for v in values}}
  expected_resolved={{'dp_world_size':8,'train_micro_batch_size_per_gpu':1,
                     'gradient_accumulation_steps':2,'train_batch_size':16}}
  expected=[]
  for rank in range(16):
    expected.append({{'generator':1,'critic':0,'ema':1}} if rank<8 else {{'generator':0,'critic':5,'ema':0}})
  lifecycle_ok=all(
      int(v.get('streamer_stats',{{}}).get('resident_bytes',-1))==0
      and int(v.get('streamer_stats',{{}}).get('materialize_count',-1))
          ==int(v.get('streamer_stats',{{}}).get('release_count',-2))
      for v in values)
  activation_threshold={score_critic_activation_min_bytes}
  activation_ok=True
  for v in values:
    rank=int(v.get('rank',-1)); stats=v.get('activation_offload_stats')
    if rank >= 8 and activation_threshold:
      activation_ok &= (
          isinstance(stats,dict)
          and int(stats.get('scope_count',-1))==20
          and int(stats.get('cpu_copy_count',0))>0
          and int(stats.get('live_cpu_bytes',-1))==0)
  out['activation_ok']=activation_ok
  batch_ok=all(v.get('initialization',{{}}).get('resolved')==expected_resolved for v in values)
  ema_ok=all((v.get('ema_optimization_step')==801 if int(v.get('rank',-1))<8
              else v.get('ema_optimization_step') is None) for v in values)
  records_ok=True
  for v in values:
    rank=int(v.get('rank',-1)); records=v.get('records',[])
    if rank<8:
      records_ok &= (sum(r.get('phase')=='generator' for r in records)==2
                     and sum(r.get('phase')=='critic_rollout' for r in records)==10)
    else:
      commits=[bool(r.get('committed')) for r in records if r.get('phase')=='critic_train']
      records_ok &= (sum(r.get('phase')=='score_gradient' for r in records)==2
                     and commits==[False,True]*5)
  def events(path):
    out={{}}
    if path.is_file():
      for line in path.read_text().splitlines():
        fields=line.split()
        if len(fields)==2: out[fields[0]]=int(fields[1])
    return out
  oom=[]
  for node in range(2):
    before=events(root/'launcher'/f'node_{{node}}.memory.events.before')
    after=events(root/'launcher'/f'node_{{node}}.memory.events.after')
    oom.append({{'node':node,'oom_delta':after.get('oom',0)-before.get('oom',0),
                'oom_kill_delta':after.get('oom_kill',0)-before.get('oom_kill',0)}})
  out['oom']=oom
  out['lifecycle_ok']=lifecycle_ok; out['batch_ok']=batch_ok
  out['ema_ok']=ema_ok; out['records_ok']=records_ok
  ok=(len(values)==16 and all(v.get('status')=='PASS' for v in values)
      and [v.get('updates') for v in values]==expected and lifecycle_ok
      and activation_ok and batch_ok and ema_ok and records_ok
      and all(x['oom_delta']==0 and x['oom_kill_delta']==0 for x in oom))
  out['status']='PASS' if ok else 'FAIL'
else:
  manifest=root.parent.parent/'manifests'/{(args.run_id + '_result.json')!r}
  value=json.loads(manifest.read_text())
  out['baseline_result']=str(manifest)
  out['baseline_status']=value.get('status')
  out['nodes']=value.get('nodes')
  out['status']='PASS' if value.get('status')=='PASS' else 'FAIL'
(root/'execution_receipts'/'formal_trial_summary.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\\n')
print(json.dumps(out,sort_keys=True))
PY
'''
    value = json.loads(_ssh_checked(args.hosts[0], summary_script).strip().splitlines()[-1])
    if value.get("status") != "PASS":
        raise RuntimeError(f"{family} strict result summary failed: {value}")
    if profile_nsys:
        manifests: list[dict[str, Any]] = []
        for node_rank in (0, 1):
            manifest_path = run_root / "nsight" / f"node_{node_rank}" / "profile_manifest.txt"
            raw = _ssh_checked(
                args.hosts[0],
                f"cat {shlex.quote(str(manifest_path))}",
            )
            fields = dict(
                line.split("=", 1) for line in raw.splitlines() if "=" in line
            )
            if fields.get("status") != "PASS":
                raise RuntimeError(f"node {node_rank} Nsight manifest failed: {fields}")
            if fields.get("diagnostic_only") != "true" or fields.get(
                "formal_speedup_eligible"
            ) != "false":
                raise RuntimeError(f"node {node_rank} Nsight claim boundary drift")
            manifests.append({"node_rank": node_rank, **fields})
        value["nsight_node_manifests"] = manifests
    return value


def _fetch_and_write_result(
    args: argparse.Namespace,
    *,
    family: str,
    run_root: Path,
    workload_digest: str,
    remote_summary: Mapping[str, Any],
) -> None:
    generator_streamed_count, score_streamed_count, phase_resident = (
        _role_streaming_policy(args)
    )
    evidence = args.result_path.parent / "evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    _rsync_from(
        args.hosts[0],
        f"{run_root}/execution_receipts/",
        evidence / "execution_receipts",
    )
    profile_nsys = bool(getattr(args, "profile_nsys", False))
    if profile_nsys:
        profile_evidence = evidence / "nsight"
        profile_evidence.mkdir(parents=True, exist_ok=False)
        for relative in (
            "range_receipts",
            "range_receipts_validation.json",
            "node_0/profile_manifest.txt",
            "node_0/nsys_stats_validation.log",
            "node_1/profile_manifest.txt",
            "node_1/nsys_stats_validation.log",
        ):
            _rsync_from(
                args.hosts[0],
                f"{run_root}/nsight/{relative}",
                profile_evidence / relative,
            )
    else:
        _rsync_from(
            args.hosts[0],
            f"{run_root}/timing_receipts/",
            evidence / "timing_receipts",
        )
    receipt_root = evidence / "execution_receipts"
    timing_root = evidence / "timing_receipts"
    result = {
        "schema_version": "distillschedule.gan_systems_formal_trial.v1",
        "status": "PASS",
        "family": family,
        "run_id": args.run_id,
        "profiled": profile_nsys,
        "diagnostic_only": profile_nsys or bool(getattr(args, "capacity_search", False)),
        "formal_speedup_eligible": not profile_nsys and not bool(getattr(args, "capacity_search", False)),
        "capacity_search": bool(getattr(args, "capacity_search", False)),
        "detailed_events": bool(getattr(args, "capacity_search", False))
        or profile_nsys,
        "validated_policy_manifest": (
            None
            if getattr(args, "validated_policy_manifest", None) is None
            else str(Path(args.validated_policy_manifest).expanduser().resolve())
        ),
        "validated_policy_digest": getattr(args, "validated_policy_digest", None),
        "logical_steps": [1000, 1001, 1002, 1003, 1004],
        "workload_contract_digest": workload_digest,
        "gan_fairness_receipt": str(receipt_root / "gan_fairness_receipt.json"),
        "gan_compute_receipt": str(receipt_root / "gan_compute_receipt.json"),
        "gan_timing_receipt_dir": None if profile_nsys else str(timing_root),
        "nsight_evidence_dir": str(evidence / "nsight") if profile_nsys else None,
        "remote_run_root": str(run_root),
        "remote_summary": dict(remote_summary),
        "fixed_memory_policy": {
            "generator_streamed_blocks": list(
                _tail_indices(generator_streamed_count)
            ),
            "score_streamed_blocks": list(_tail_indices(score_streamed_count)),
            "generator_critic_phase_resident": phase_resident,
            "score_critic_activation_offload_min_bytes": int(
                getattr(args, "score_critic_activation_offload_min_mib", 0)
            )
            * 1024
            * 1024,
            "streaming_window": 1,
            "host_policy": "HOST_SHARED_IMMUTABLE",
            "prefetch": False,
            "generator_selected_activation_offload_min_bytes": 33554432,
        },
    }
    _atomic_json(args.result_path, result)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=("baseline", "pipeline"), required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--result-path", type=Path, required=True)
    p.add_argument("--workload-contract", type=Path, required=True)
    p.add_argument("--hosts", nargs=2, default=list(DEFAULT_HOSTS))
    p.add_argument("--master-addr", default=DEFAULT_MASTER_ADDR)
    p.add_argument("--master-port", type=int)
    p.add_argument("--timeout-seconds", type=int, default=20000)
    p.add_argument("--baseline-local-root", type=Path, default=DEFAULT_BASELINE_LOCAL_ROOT)
    p.add_argument("--baseline-remote-root", type=Path, default=DEFAULT_BASELINE_REMOTE_ROOT)
    p.add_argument("--baseline-expected-head", default="99aa0a31cd1fd5aebcb1d72bd721ece5fd09b683")
    p.add_argument("--pipeline-remote-root", type=Path, default=DEFAULT_PIPELINE_REMOTE_ROOT)
    p.add_argument("--pipeline-runtime-root", type=Path, default=DEFAULT_PIPELINE_RUNTIME_ROOT)
    p.add_argument("--site-env", type=Path, default=DEFAULT_SITE_ENV)
    p.add_argument("--trace-bundle", type=Path, default=DEFAULT_TRACE)
    p.add_argument("--target-package", type=Path, default=DEFAULT_TARGET)
    p.add_argument("--generator-root", type=Path, default=DEFAULT_GENERATOR)
    p.add_argument("--score-root", type=Path, default=DEFAULT_SCORE)
    p.add_argument("--negative-prompt", type=Path, default=DEFAULT_NEGATIVE)
    p.add_argument("--pipeline-mmap", type=Path, default=DEFAULT_PIPELINE_MMAP)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--profile-nsys",
        action="store_true",
        help="diagnostic exact-range Nsight run; never formal-speedup eligible",
    )
    p.add_argument(
        "--capacity-search",
        action="store_true",
        help="capacity/correctness candidate; never formal-speedup eligible",
    )
    p.add_argument(
        "--validated-policy-manifest",
        type=Path,
        help=(
            "immutable PASS capacity-gate manifest required to promote a "
            "non-default lifecycle policy into unprofiled formal timing"
        ),
    )
    p.add_argument("--streamed-block-count", type=int, default=40)
    p.add_argument("--generator-streamed-block-count", type=int)
    p.add_argument("--score-streamed-block-count", type=int)
    p.add_argument("--generator-critic-phase-resident", action="store_true")
    p.add_argument(
        "--score-critic-activation-offload-min-mib",
        type=int,
        default=0,
        help="capacity-gated Score Critic saved-activation offload threshold",
    )
    return p


def main() -> int:
    args = parser().parse_args()
    args.result_path = args.result_path.expanduser().resolve()
    args.workload_contract = args.workload_contract.expanduser().resolve()
    try:
        _prepare_result_path(args.result_path)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id) is None:
        raise SystemExit("run id contains unsupported characters")
    args.master_port = args.master_port or (
        30000 + int(hashlib.sha256(args.run_id.encode()).hexdigest()[:6], 16) % 2000
    )
    args.run_nonce = int(hashlib.sha256(args.run_id.encode()).hexdigest()[:12], 16)
    workload_digest = _workload_digest(args.workload_contract)
    if args.profile_nsys and args.family != "pipeline":
        raise SystemExit("--profile-nsys is implemented only for the pipeline worker")
    if args.capacity_search and args.family != "pipeline":
        raise SystemExit("--capacity-search is implemented only for the pipeline worker")
    if args.profile_nsys and args.capacity_search:
        raise SystemExit("Nsight and capacity search must remain separate runs")
    try:
        generator_count, score_count, phase_resident = _role_streaming_policy(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.family != "pipeline" and (
        generator_count != args.streamed_block_count
        or score_count != args.streamed_block_count
        or phase_resident
    ):
        raise SystemExit("role-aware streaming is implemented only for pipeline")
    optimized_policy = (
        generator_count != 40
        or score_count != 40
        or phase_resident
        or args.score_critic_activation_offload_min_mib != 0
    )
    if args.capacity_search and args.validated_policy_manifest is not None:
        raise SystemExit(
            "capacity search and validated formal policy are mutually exclusive"
        )
    args.validated_policy_digest = None
    if args.validated_policy_manifest is not None:
        if args.family != "pipeline" or not optimized_policy:
            raise SystemExit(
                "validated policy manifest is only valid for optimized pipeline"
            )
        try:
            args.validated_policy_digest = _validate_optimized_policy_manifest(
                args.validated_policy_manifest,
                workload_digest=workload_digest,
                generator_streamed_count=generator_count,
                score_streamed_count=score_count,
                generator_critic_phase_resident=phase_resident,
                score_critic_activation_offload_min_mib=(
                    args.score_critic_activation_offload_min_mib
                ),
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    if optimized_policy and not (
        args.capacity_search or args.validated_policy_digest
    ):
        raise SystemExit(
            "optimized role/phase residency requires --capacity-search or a "
            "hash-bound validated policy manifest"
        )
    run_log = args.result_path.parent / f"{args.family}_controller.log"
    pipeline_launch_attempted = False
    try:
        _assert_remote_idle(args.hosts)
        if args.family == "baseline":
            run_root = _baseline_launch(args, run_log)
        else:
            pipeline_launch_attempted = True
            run_root = _pipeline_launch(args, run_log)
        summary = _remote_finalize(args, args.family, run_root)
        _assert_remote_idle(args.hosts)
        _fetch_and_write_result(
            args,
            family=args.family,
            run_root=run_root,
            workload_digest=workload_digest,
            remote_summary=summary,
        )
        print(json.dumps({"status": "PASS", "result": str(args.result_path)}, sort_keys=True))
        return 0
    except BaseException as exc:
        failure_evidence = None
        failure_evidence_error = None
        remote_run_root = None
        if args.family == "pipeline" and pipeline_launch_attempted:
            remote_run_root = (
                args.pipeline_remote_root / "artifacts" / "runs" / args.run_id
            )
            try:
                failure_evidence = _collect_pipeline_failure_evidence(
                    args,
                    run_root=remote_run_root,
                    destination=args.result_path.parent / "failure_evidence",
                )
            except BaseException as evidence_exc:
                failure_evidence_error = repr(evidence_exc)
        failure = {
            "schema_version": "distillschedule.gan_systems_formal_trial.v1",
            "status": "FAIL",
            "claimable": False,
            "family": args.family,
            "run_id": args.run_id,
            "error": repr(exc),
            "remote_run_root": (
                None if remote_run_root is None else str(remote_run_root)
            ),
            "failure_evidence": failure_evidence,
            "failure_evidence_error": failure_evidence_error,
        }
        _atomic_json(args.result_path, failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
