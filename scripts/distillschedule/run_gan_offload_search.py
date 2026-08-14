#!/usr/bin/env python3
"""Find the minimum symmetric frozen-block streaming set for official GAN Full5.

The search changes exactly one memory variable relative to the conservative GAN
Route-A baseline: the number/set of frozen Transformer blocks streamed for both
Generator and Score.  GAN math, step1000..1004 scheduling, ZeRO-1, batch, trace,
checkpoint, optimizer order, EMA order, activation policy, prefetch and overlap
remain owned by the existing official-semantics baseline.

A fresh all-40 run establishes the same-contract passing anchor.  The driver then
checks k=0 and, when that is an actual capacity failure, performs a bounded binary
search on nested tail sets.  Only a strict PASS or actual OOM can move a capacity
bound; contract/infrastructure failures abort the search.  The selected minimum is
re-run once under a fresh immutable run ID before it is reported.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "distillschedule.gan_offload_search.v1"
OOM_RE = re.compile(
    r"CUDA out of memory|torch\.OutOfMemoryError|CUDACachingAllocator.*out of memory",
    re.IGNORECASE,
)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_hosts(value: str) -> tuple[str, str]:
    hosts = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(hosts) != 2 or hosts[0] == hosts[1]:
        raise argparse.ArgumentTypeError("--hosts must contain two distinct hosts")
    return hosts  # type: ignore[return-value]


def tail_blocks(block_count: int, streamed_count: int) -> list[int]:
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    if not 0 <= streamed_count <= block_count:
        raise ValueError("streamed_count is outside [0, block_count]")
    if streamed_count == 0:
        return []
    return list(range(block_count - streamed_count, block_count))


def candidate_environment(block_count: int, streamed_count: int) -> dict[str, str]:
    indices = tail_blocks(block_count, streamed_count)
    return {
        "DISTILLSCHEDULE_GAN_CAPACITY_SEARCH": "1",
        "DISTILLSCHEDULE_GAN_STREAMING_WINDOW": "0" if streamed_count == 0 else "1",
        "DISTILLSCHEDULE_GAN_STREAMING_COMPONENTS": "generator,score",
        "DISTILLSCHEDULE_GAN_STREAMED_BLOCK_COUNT": str(streamed_count),
        "DISTILLSCHEDULE_GAN_STREAMED_BLOCK_INDICES": ",".join(str(i) for i in indices),
    }


def parse_memory_events(text: str) -> dict[str, int]:
    tokens = text.replace(";", " ").split()
    result: dict[str, int] = {}
    for index in range(0, len(tokens) - 1, 2):
        try:
            result[tokens[index]] = int(tokens[index + 1])
        except ValueError:
            continue
    return result


def strict_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(errors="replace").splitlines():
        if not raw.strip():
            continue
        if "=" not in raw:
            raise RuntimeError(f"non key=value launcher manifest: {path}")
        key, value = raw.split("=", 1)
        if not key or key in result:
            raise RuntimeError(f"invalid/duplicate launcher manifest key: {path}")
        result[key] = value
    return result


def actual_oom_observed(root: Path, run_id: str) -> bool:
    run = root / "artifacts/runs" / run_id
    for node_rank in (0, 1):
        launcher = run / "launcher" / f"node_{node_rank}"
        console = launcher / "console.log"
        if console.is_file() and OOM_RE.search(console.read_text(errors="replace")):
            return True
        manifest = launcher / "launch_manifest.txt"
        if not manifest.is_file():
            continue
        try:
            values = strict_kv(manifest)
        except RuntimeError:
            continue
        before = parse_memory_events(values.get("memory_events_before", ""))
        after = parse_memory_events(values.get("memory_events_after", ""))
        if after.get("oom", 0) > before.get("oom", 0):
            return True
        if after.get("oom_kill", 0) > before.get("oom_kill", 0):
            return True
    return False


def verify_effective_streaming(
    root: Path, run_id: str, *, block_count: int, streamed_count: int
) -> dict[str, Any]:
    expected = candidate_environment(block_count, streamed_count)
    observed: dict[str, Any] = {}
    for node_rank in (0, 1):
        path = (
            root
            / "artifacts/runs"
            / run_id
            / "launcher"
            / f"node_{node_rank}"
            / "compiled_preflight.json"
        )
        if not path.is_file():
            raise RuntimeError(f"missing compiled preflight for node {node_rank}: {path}")
        payload = load_json(path)
        invocation = payload.get("invocation")
        if not isinstance(invocation, Mapping):
            raise RuntimeError(f"node {node_rank}: compiled invocation missing")
        environment = invocation.get("environment")
        if not isinstance(environment, Mapping):
            raise RuntimeError(f"node {node_rank}: compiled invocation environment missing")
        effective = {
            "DISTILLSCHEDULE_GAN_CAPACITY_SEARCH": str(
                environment.get("DISTILLSCHEDULE_GAN_CAPACITY_SEARCH", "")
            ),
            "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_WINDOW": str(
                environment.get("DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_WINDOW", "")
            ),
            "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_COMPONENTS": str(
                environment.get("DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_COMPONENTS", "")
            ),
            "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_BLOCK_COUNT": str(
                environment.get("DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_BLOCK_COUNT", "")
            ),
            "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_INDICES": str(
                environment.get("DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_INDICES", "")
            ),
        }
        wanted = {
            "DISTILLSCHEDULE_GAN_CAPACITY_SEARCH": expected[
                "DISTILLSCHEDULE_GAN_CAPACITY_SEARCH"
            ],
            "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_WINDOW": expected[
                "DISTILLSCHEDULE_GAN_STREAMING_WINDOW"
            ],
            "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_COMPONENTS": expected[
                "DISTILLSCHEDULE_GAN_STREAMING_COMPONENTS"
            ],
            "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_BLOCK_COUNT": expected[
                "DISTILLSCHEDULE_GAN_STREAMED_BLOCK_COUNT"
            ],
            "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_INDICES": expected[
                "DISTILLSCHEDULE_GAN_STREAMED_BLOCK_INDICES"
            ],
        }
        if effective != wanted:
            raise RuntimeError(
                f"node {node_rank}: effective streaming override mismatch; "
                f"observed={effective}, expected={wanted}"
            )
        observed[str(node_rank)] = {
            "compiled_preflight": str(path),
            "compiled_preflight_sha256": sha256_file(path),
            "effective_streaming": effective,
        }
    return observed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--hosts", required=True, type=parse_hosts)
    result.add_argument("--search-id", required=True)
    result.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    result.add_argument(
        "--site-env",
        default="configs/distillschedule/server/baai_gan_2x8_a100.env.example",
    )
    result.add_argument("--master-addr", required=True)
    result.add_argument("--master-port", type=int, default=29777)
    result.add_argument("--ssh", default="ssh")
    result.add_argument("--timeout-seconds", type=int, default=20000)
    result.add_argument("--block-count", type=int, default=40)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def write_candidate_site_env(
    *,
    search_root: Path,
    base_site_env: Path,
    block_count: int,
    streamed_count: int,
    label: str,
) -> Path:
    env = candidate_environment(block_count, streamed_count)
    path = search_root / "candidate_env" / f"{label}.env"
    content = [
        "# Generated by run_gan_offload_search.py; immutable capacity candidate.",
        f"source {shlex.quote(str(base_site_env))}",
    ]
    content.extend(f"export {key}={shlex.quote(value)}" for key, value in env.items())
    text = "\n".join(content) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"immutable candidate env collision: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return path


def controller_command(
    args: argparse.Namespace,
    *,
    root: Path,
    run_id: str,
    candidate_site_env: Path,
    summarize_only: bool = False,
    dry_run: bool = False,
) -> list[str]:
    relative_env = candidate_site_env.relative_to(root)
    command = [
        sys.executable,
        str(root / "scripts/distillschedule/run_routea_gan_world16.py"),
        "--hosts",
        ",".join(args.hosts),
        "--run-id",
        run_id,
        "--repo-root",
        str(root),
        "--site-env",
        str(relative_env),
        "--master-addr",
        args.master_addr,
        "--master-port",
        str(args.master_port),
        "--ssh",
        args.ssh,
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if summarize_only:
        command.append("--summarize-only")
    if dry_run:
        command.append("--dry-run")
    return command


def monotonic_violation(observations: Mapping[int, Mapping[str, Any]]) -> bool:
    tested = sorted(observations)
    for smaller in tested:
        for larger in tested:
            if smaller >= larger:
                continue
            small = observations[smaller].get("classification")
            large = observations[larger].get("classification")
            if small == "PASS" and large == "CAPACITY_FAIL":
                return True
    return False


def run_candidate(
    args: argparse.Namespace,
    *,
    root: Path,
    search_root: Path,
    base_site_env: Path,
    streamed_count: int,
    label: str,
) -> dict[str, Any]:
    run_id = f"{args.search_id}-{label}-k{streamed_count:02d}"
    record_path = search_root / "candidates" / f"{run_id}.json"
    env_path = write_candidate_site_env(
        search_root=search_root,
        base_site_env=base_site_env,
        block_count=args.block_count,
        streamed_count=streamed_count,
        label=run_id,
    )
    identity = {
        "run_id": run_id,
        "streamed_blocks_per_backbone": streamed_count,
        "streamed_block_indices": tail_blocks(args.block_count, streamed_count),
        "streaming_components": ["generator", "score"],
        "candidate_env": str(env_path),
        "candidate_env_sha256": sha256_file(env_path),
        "search_policy": "symmetric_nested_tail",
        "official_gan_window": [1000, 1001, 1002, 1003, 1004],
    }
    if record_path.exists():
        previous = load_json(record_path)
        if previous.get("identity") != identity:
            raise RuntimeError(f"immutable candidate record collision: {record_path}")
        if args.resume and previous.get("classification") in {"PASS", "CAPACITY_FAIL"}:
            return previous
        if not args.dry_run:
            raise RuntimeError(
                f"candidate already exists without reusable classification: {run_id}"
            )

    command = controller_command(
        args,
        root=root,
        run_id=run_id,
        candidate_site_env=env_path,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        record = {
            "schema_version": SCHEMA,
            "identity": identity,
            "classification": "DRY_RUN",
            "command": command,
        }
        atomic_json(record_path, record)
        return record

    log_path = search_root / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            command, check=False, stdout=handle, stderr=subprocess.STDOUT
        )

    result_path = root / "artifacts/manifests" / f"{run_id}_result.json"
    if completed.returncode != 0:
        summarize = controller_command(
            args,
            root=root,
            run_id=run_id,
            candidate_site_env=env_path,
            summarize_only=True,
        )
        with (search_root / "logs" / f"{run_id}.summarize.log").open("wb") as handle:
            subprocess.run(summarize, check=False, stdout=handle, stderr=subprocess.STDOUT)

    result = load_json(result_path) if result_path.is_file() else {}
    status = result.get("status")
    oom = status == "OOM" or actual_oom_observed(root, run_id)
    effective: dict[str, Any] | None = None
    try:
        effective = verify_effective_streaming(
            root,
            run_id,
            block_count=args.block_count,
            streamed_count=streamed_count,
        )
    except RuntimeError as exc:
        if not oom:
            record = {
                "schema_version": SCHEMA,
                "identity": identity,
                "classification": "ABORT",
                "controller_return_code": completed.returncode,
                "result_status": status,
                "error": str(exc),
                "result_path": str(result_path) if result_path.is_file() else None,
            }
            atomic_json(record_path, record)
            return record

    if status == "PASS" and completed.returncode == 0:
        classification = "PASS"
    elif oom:
        classification = "CAPACITY_FAIL"
    else:
        classification = "ABORT"

    record = {
        "schema_version": SCHEMA,
        "identity": identity,
        "classification": classification,
        "controller_return_code": completed.returncode,
        "result_status": status,
        "actual_oom_observed": oom,
        "effective_streaming": effective,
        "result_path": str(result_path) if result_path.is_file() else None,
        "result_sha256": sha256_file(result_path) if result_path.is_file() else None,
        "controller_log": str(log_path),
    }
    if classification == "ABORT":
        record["error"] = (
            "candidate failed outside the capacity contract; do not use it as a search bound"
        )
    atomic_json(record_path, record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.repo_root.expanduser().resolve()
    if args.block_count <= 0:
        print(json.dumps({"status": "FAIL", "error": "block_count must be positive"}))
        return 2
    if not 1 <= args.master_port <= 65535:
        print(json.dumps({"status": "FAIL", "error": "invalid master port"}))
        return 2
    site_env = Path(args.site_env).expanduser()
    if not site_env.is_absolute():
        site_env = (root / site_env).resolve()
    if not site_env.is_file():
        print(json.dumps({"status": "FAIL", "error": f"missing site env {site_env}"}))
        return 2

    search_root = root / "artifacts/g_offload_search" / f"gan-{args.search_id}"
    manifest_path = search_root / "search_manifest.json"
    identity = {
        "schema_version": SCHEMA,
        "search_id": args.search_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_root": str(root),
        "hosts": list(args.hosts),
        "master_addr": args.master_addr,
        "master_port": args.master_port,
        "base_site_env": str(site_env),
        "base_site_env_sha256": sha256_file(site_env),
        "block_count": args.block_count,
        "search_policy": "fresh_k40_anchor_then_k0_bound_then_binary_tail_search",
        "variable": "same frozen block tail streamed on Generator and Score",
        "formal_timing_claim": False,
    }
    immutable = dict(identity)
    immutable.pop("created_at", None)
    if manifest_path.exists():
        previous = load_json(manifest_path)
        old = dict(previous)
        old.pop("created_at", None)
        if old != immutable:
            print(json.dumps({"status": "FAIL", "error": "immutable search ID collision"}))
            return 2
        if not args.resume and not args.dry_run:
            print(json.dumps({"status": "FAIL", "error": "search exists; use --resume or a new ID"}))
            return 2
    else:
        atomic_json(manifest_path, identity)

    if args.dry_run:
        anchor = run_candidate(
            args,
            root=root,
            search_root=search_root,
            base_site_env=site_env,
            streamed_count=args.block_count,
            label="anchor",
        )
        zero = run_candidate(
            args,
            root=root,
            search_root=search_root,
            base_site_env=site_env,
            streamed_count=0,
            label="probe",
        )
        output = {
            "schema_version": SCHEMA,
            "status": "DRY_RUN",
            "adaptive_after": [args.block_count, 0],
            "anchor": anchor,
            "zero": zero,
        }
        atomic_json(search_root / "result.json", output)
        print(json.dumps({"status": "DRY_RUN", "result": str(search_root / 'result.json')}, sort_keys=True))
        return 0

    observations: dict[int, dict[str, Any]] = {}
    try:
        anchor = run_candidate(
            args,
            root=root,
            search_root=search_root,
            base_site_env=site_env,
            streamed_count=args.block_count,
            label="anchor",
        )
        observations[args.block_count] = anchor
        if anchor.get("classification") != "PASS":
            status = "ANCHOR_OOM" if anchor.get("classification") == "CAPACITY_FAIL" else "ABORT"
            output = {
                "schema_version": SCHEMA,
                "status": status,
                "observations": {str(k): v for k, v in observations.items()},
                "minimum_passing_blocks_per_backbone": None,
                "formal_timing_claim": False,
            }
            atomic_json(search_root / "result.json", output)
            print(json.dumps({"status": status, "result": str(search_root / 'result.json')}, sort_keys=True))
            return 1

        zero = run_candidate(
            args,
            root=root,
            search_root=search_root,
            base_site_env=site_env,
            streamed_count=0,
            label="probe",
        )
        observations[0] = zero
        if zero.get("classification") == "ABORT":
            raise RuntimeError("k=0 failed outside the capacity contract")

        if zero.get("classification") == "PASS":
            minimum = 0
            failed_boundary: int | None = None
        else:
            low = 0
            high = args.block_count
            while high - low > 1:
                mid = (low + high) // 2
                outcome = run_candidate(
                    args,
                    root=root,
                    search_root=search_root,
                    base_site_env=site_env,
                    streamed_count=mid,
                    label="probe",
                )
                observations[mid] = outcome
                if outcome.get("classification") == "PASS":
                    high = mid
                elif outcome.get("classification") == "CAPACITY_FAIL":
                    low = mid
                else:
                    raise RuntimeError(
                        f"k={mid} failed outside the capacity contract"
                    )
                if monotonic_violation(observations):
                    raise RuntimeError("non-monotonic PASS/OOM ordering observed")
            minimum = high
            failed_boundary = low

        verification = run_candidate(
            args,
            root=root,
            search_root=search_root,
            base_site_env=site_env,
            streamed_count=minimum,
            label="verify",
        )
        if verification.get("classification") != "PASS":
            raise RuntimeError(
                f"selected k={minimum} did not PASS independent revalidation"
            )

        output = {
            "schema_version": SCHEMA,
            "status": "PASS",
            "search_id": args.search_id,
            "search_policy": "symmetric_nested_tail",
            "block_count_per_backbone": args.block_count,
            "minimum_passing_blocks_per_backbone": minimum,
            "minimum_total_streamed_blocks_generator_plus_score": minimum * 2,
            "minimum_passing_block_indices": tail_blocks(args.block_count, minimum),
            "capacity_fail_boundary_blocks_per_backbone": failed_boundary,
            "observations": {
                str(key): value for key, value in sorted(observations.items())
            },
            "independent_revalidation": verification,
            "baseline_label": (
                "official GAN Full5 + measured-minimum symmetric G/Score frozen-block "
                "streaming under the Route-A world16 execution graph"
            ),
            "formal_timing_claim": False,
            "limitations": [
                "k counts streamed blocks per backbone; Generator and Score use the same nested tail set.",
                "Search runs are capacity/correctness experiments and are excluded from formal timing.",
                "An asymmetric G-vs-Score search is a separate experiment and is not inferred from this result.",
            ],
        }
        atomic_json(search_root / "result.json", output)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "minimum_passing_blocks_per_backbone": minimum,
                    "result": str(search_root / "result.json"),
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
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
        atomic_json(search_root / "result.json", output)
        print(json.dumps({"status": "ABORT", "error": str(exc), "result": str(search_root / 'result.json')}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
