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
        "--detailed-events",
    ]
    command_text = shlex.join(command)
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
export HF_HOME=/share/project/shuchenweng/DistillSchedule_assets/hf_home_official_torch210
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LOCAL_KERNELS=kernels-community/flash-attn2=/share/project/shuchenweng/DistillSchedule_assets/hf_home_official_torch210/hub/models--kernels-community--flash-attn2/snapshots/6a810a7a010ba2aad6613f1f0f7b226fb8bdee3c/build/torch210-cxx11-cu128-x86_64-linux
export MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 NCCL_DEBUG=WARN NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ENABLE_MONITORING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9
export DISTILLSCHEDULE_OPERATION_TRACE_ACCESS_DIR={shlex.quote(str(run_root / 'trace_access'))}
export DISTILLSCHEDULE_STABLE_BLOCK_NOISE=1 DISTILLSCHEDULE_NSYS_ANNOTATE=1
export DISTILLSCHEDULE_GAN_NSYS_PROFILE=0 DISTILLSCHEDULE_GAN_EXACT_TIMING=1
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
    build = f'''set -Eeuo pipefail
export PYTHONPATH={shlex.quote(str(dmd_root / 'src'))}:{shlex.quote(str(dmd_root))}:{shlex.quote(str(runtime_root))}
test ! -e {shlex.quote(str(receipt_root))} || {{ echo receipt_root_exists >&2; exit 82; }}
{shlex.quote(str(PYTHON_REMOTE))} {shlex.quote(str(dmd_root / 'scripts/distillschedule/build_gan_execution_receipts.py'))} \
  --family {family} --run-root {shlex.quote(str(run_evidence_root))} \
  --trace-bundle {shlex.quote(str(args.trace_bundle))} \
  --output-dir {shlex.quote(str(receipt_root))} --expected-streamed-block-count 40
{shlex.quote(str(PYTHON_REMOTE))} {shlex.quote(str(dmd_root / 'scripts/distillschedule/validate_gan_timing_receipts.py'))} \
  --receipt-dir {shlex.quote(str(run_root / 'timing_receipts'))} --world-size 16 \
  --output {shlex.quote(str(receipt_root / 'timing_validation.json'))}
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
      and batch_ok and ema_ok and records_ok
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
    return value


def _fetch_and_write_result(
    args: argparse.Namespace,
    *,
    family: str,
    run_root: Path,
    workload_digest: str,
    remote_summary: Mapping[str, Any],
) -> None:
    evidence = args.result_path.parent / "evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    _rsync_from(
        args.hosts[0],
        f"{run_root}/execution_receipts/",
        evidence / "execution_receipts",
    )
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
        "profiled": False,
        "diagnostic_only": False,
        "logical_steps": [1000, 1001, 1002, 1003, 1004],
        "workload_contract_digest": workload_digest,
        "gan_fairness_receipt": str(receipt_root / "gan_fairness_receipt.json"),
        "gan_compute_receipt": str(receipt_root / "gan_compute_receipt.json"),
        "gan_timing_receipt_dir": str(timing_root),
        "remote_run_root": str(run_root),
        "remote_summary": dict(remote_summary),
        "fixed_memory_policy": {
            "generator_streamed_blocks": list(range(40)),
            "score_streamed_blocks": list(range(40)),
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
    run_log = args.result_path.parent / f"{args.family}_controller.log"
    try:
        _assert_remote_idle(args.hosts)
        run_root = (
            _baseline_launch(args, run_log)
            if args.family == "baseline"
            else _pipeline_launch(args, run_log)
        )
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
        failure = {
            "schema_version": "distillschedule.gan_systems_formal_trial.v1",
            "status": "FAIL",
            "claimable": False,
            "family": args.family,
            "run_id": args.run_id,
            "error": repr(exc),
        }
        _atomic_json(args.result_path, failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
