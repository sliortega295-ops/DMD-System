#!/usr/bin/env python3
"""Bounded actual-capacity search for the persistent 8G+8Score GAN pipeline.

Each candidate runs the same real 14B Full5 executor and changes only the
nested tail of immutable frozen blocks streamed on both role pools.  A bound
may move only on a strict correctness PASS or an actual CUDA/cgroup OOM.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "distillschedule.gan_pipeline_offload_search.v1"
OOM_RE = re.compile(
    r"CUDA out of memory|torch\.OutOfMemoryError|CUDACachingAllocator.*out of memory",
    re.IGNORECASE,
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tail_blocks(count: int) -> list[int]:
    if not 0 <= count <= 40:
        raise ValueError("streamed block count must be in [0,40]")
    return list(range(40 - count, 40)) if count else []


def _memory_events(text: str | None) -> dict[str, int]:
    result: dict[str, int] = {}
    if text is None:
        return result
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2:
            try:
                result[fields[0]] = int(fields[1])
            except ValueError:
                pass
    return result


def _remote_text(host: str, path: Path) -> str | None:
    completed = subprocess.run(
        ["ssh", host, "bash", "-lc", shlex.quote(f"cat {shlex.quote(str(path))}")],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout if completed.returncode == 0 else None


def actual_oom(host: str, remote_root: Path, run_id: str) -> bool:
    launcher = remote_root / "artifacts" / "runs" / run_id / "launcher"
    for node_rank in (0, 1):
        log = _remote_text(host, launcher / f"node_{node_rank}.log")
        if log is not None and OOM_RE.search(log):
            return True
        before = _memory_events(
            _remote_text(host, launcher / f"node_{node_rank}.memory.events.before")
        )
        after = _memory_events(
            _remote_text(host, launcher / f"node_{node_rank}.memory.events.after")
        )
        if after.get("oom", 0) > before.get("oom", 0):
            return True
        if after.get("oom_kill", 0) > before.get("oom_kill", 0):
            return True
    return False


def _candidate_command(
    args: argparse.Namespace,
    *,
    run_id: str,
    streamed_count: int,
    result_path: Path,
    port: int,
) -> list[str]:
    return [
        sys.executable,
        str(args.repo_root / "scripts/distillschedule/run_gan_systems_formal_trial.py"),
        "--family",
        "pipeline",
        "--capacity-search",
        "--streamed-block-count",
        str(streamed_count),
        "--run-id",
        run_id,
        "--result-path",
        str(result_path),
        "--workload-contract",
        str(args.workload_contract),
        "--master-port",
        str(port),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]


def _run_candidate(
    args: argparse.Namespace,
    *,
    root: Path,
    streamed_count: int,
    label: str,
    ordinal: int,
) -> dict[str, Any]:
    run_id = f"{args.search_id}-{label}-k{streamed_count:02d}"
    candidate_root = root / "candidates" / run_id
    result_path = candidate_root / "result.json"
    record_path = candidate_root / "capacity_record.json"
    identity = {
        "run_id": run_id,
        "streamed_blocks_per_backbone": streamed_count,
        "streamed_block_indices": tail_blocks(streamed_count),
        "streaming_components": ["generator", "score"],
        "search_policy": "symmetric_nested_tail",
        "official_gan_window": [1000, 1001, 1002, 1003, 1004],
    }
    if record_path.exists():
        previous = _load(record_path)
        if previous.get("identity") != identity:
            raise RuntimeError(f"immutable candidate collision: {run_id}")
        if args.resume and previous.get("classification") in {
            "PASS",
            "CAPACITY_FAIL",
        }:
            return previous
        raise RuntimeError(f"candidate already exists without reusable result: {run_id}")

    command = _candidate_command(
        args,
        run_id=run_id,
        streamed_count=streamed_count,
        result_path=result_path,
        port=args.master_port + ordinal,
    )
    candidate_root.mkdir(parents=True, exist_ok=False)
    log_path = candidate_root / "controller.log"
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            command,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    result = _load(result_path) if result_path.is_file() else {}
    oom = actual_oom(args.hosts[0], args.pipeline_remote_root, run_id)
    if completed.returncode == 0 and result.get("status") == "PASS":
        fixed = result.get("fixed_memory_policy", {})
        expected = tail_blocks(streamed_count)
        if fixed.get("generator_streamed_blocks") != expected or fixed.get(
            "score_streamed_blocks"
        ) != expected:
            classification = "ABORT"
            error = "result did not prove the requested effective streaming set"
        elif result.get("capacity_search") is not True or result.get(
            "formal_speedup_eligible"
        ) is not False:
            classification = "ABORT"
            error = "capacity candidate crossed the formal-claim boundary"
        else:
            classification = "PASS"
            error = None
    elif oom:
        classification = "CAPACITY_FAIL"
        error = None
    else:
        classification = "ABORT"
        error = "candidate failed outside the actual capacity contract"
    record: dict[str, Any] = {
        "schema_version": SCHEMA,
        "identity": identity,
        "classification": classification,
        "controller_return_code": completed.returncode,
        "actual_oom_observed": oom,
        "result_path": str(result_path) if result_path.is_file() else None,
        "result_sha256": _sha256(result_path) if result_path.is_file() else None,
        "controller_log": str(log_path),
    }
    if error is not None:
        record["error"] = error
    _atomic_json(record_path, record)
    return record


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--search-id", required=True)
    result.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    result.add_argument("--workload-contract", type=Path, required=True)
    result.add_argument(
        "--pipeline-remote-root",
        type=Path,
        default=Path(
            "/share/project/shuchenweng/yongyan_liu/"
            "DMD-System_gan_pipeline_exec_20260811"
        ),
    )
    result.add_argument(
        "--hosts", nargs=2, default=["baai-helios-node0", "baai-helios-node1"]
    )
    result.add_argument("--master-port", type=int, default=32200)
    result.add_argument("--timeout-seconds", type=int, default=20000)
    result.add_argument("--resume", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.repo_root = args.repo_root.expanduser().resolve()
    args.workload_contract = args.workload_contract.expanduser().resolve()
    if not args.workload_contract.is_file():
        raise SystemExit(f"missing workload contract: {args.workload_contract}")
    if not 1 <= args.master_port <= 65490:
        raise SystemExit("invalid master-port range")
    root = args.repo_root / "artifacts" / "g_offload_search" / f"gan-{args.search_id}"
    manifest = root / "search_manifest.json"
    identity = {
        "schema_version": SCHEMA,
        "search_id": args.search_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_head": subprocess.check_output(
            ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "workload_contract": str(args.workload_contract),
        "workload_contract_sha256": _sha256(args.workload_contract),
        "pipeline_remote_root": str(args.pipeline_remote_root),
        "hosts": list(args.hosts),
        "search_policy": "fresh_k40_then_k0_then_binary_tail_then_fresh_kstar",
        "formal_timing_claim": False,
    }
    comparable = dict(identity)
    comparable.pop("created_at")
    if manifest.exists():
        previous = _load(manifest)
        old = dict(previous)
        old.pop("created_at", None)
        if old != comparable:
            raise SystemExit("immutable search-id collision")
        if not args.resume:
            raise SystemExit("search exists; use --resume or a new ID")
    else:
        _atomic_json(manifest, identity)

    observations: dict[int, dict[str, Any]] = {}
    ordinal = 0
    try:
        def run(count: int, label: str) -> dict[str, Any]:
            nonlocal ordinal
            ordinal += 1
            return _run_candidate(
                args,
                root=root,
                streamed_count=count,
                label=label,
                ordinal=ordinal,
            )

        observations[40] = run(40, "anchor")
        if observations[40]["classification"] != "PASS":
            raise RuntimeError("fresh all40 pipeline anchor did not PASS")
        observations[0] = run(0, "probe")
        if observations[0]["classification"] == "ABORT":
            raise RuntimeError("k=0 failed outside the capacity contract")
        if observations[0]["classification"] == "PASS":
            minimum = 0
            failed_boundary: int | None = None
        else:
            low, high = 0, 40
            while high - low > 1:
                middle = (low + high) // 2
                observations[middle] = run(middle, "probe")
                outcome = observations[middle]["classification"]
                if outcome == "PASS":
                    high = middle
                elif outcome == "CAPACITY_FAIL":
                    low = middle
                else:
                    raise RuntimeError(
                        f"k={middle} failed outside the capacity contract"
                    )
            minimum, failed_boundary = high, low
        verification = run(minimum, "verify")
        if verification["classification"] != "PASS":
            raise RuntimeError(f"fresh k*={minimum} verification failed")
        output = {
            "schema_version": SCHEMA,
            "status": "PASS",
            "minimum_passing_blocks_per_backbone": minimum,
            "minimum_passing_block_indices": tail_blocks(minimum),
            "capacity_fail_boundary_blocks_per_backbone": failed_boundary,
            "observations": {
                str(key): value for key, value in sorted(observations.items())
            },
            "independent_revalidation": verification,
            "formal_timing_claim": False,
        }
        _atomic_json(root / "result.json", output)
        print(json.dumps(output, sort_keys=True))
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        output = {
            "schema_version": SCHEMA,
            "status": "ABORT",
            "error": str(exc),
            "observations": {
                str(key): value for key, value in sorted(observations.items())
            },
            "formal_timing_claim": False,
            "downstream": "NOT_RUN",
        }
        _atomic_json(root / "result.json", output)
        print(json.dumps(output, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
