#!/usr/bin/env python3
"""Run the authoritative No-GAN world16 experiment DAG.

The suite reuses accepted/bounded historical evidence and schedules only current
gaps.  Every GPU stage has an immutable stage-run ID, exact contract digest,
peer-failure cancellation and fail-closed downstream NOT_RUN propagation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

SCHEMA = "distillschedule.no_gan_suite_run.v1"
ACCEPTED = {"PASS", "REUSED_ACCEPTED", "REUSED_BOUNDED", "REUSED_BOUNDED_DIAGNOSTIC"}
ENV_PATTERN = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError("suite loading requires OmegaConf") from exc
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(value, Mapping):
        raise RuntimeError("suite root must be a mapping")
    return dict(value)


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            return default or ""
        return ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, Mapping):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def parse_hosts(value: str) -> tuple[str, str]:
    hosts = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(hosts) != 2 or hosts[0] == hosts[1]:
        raise argparse.ArgumentTypeError("--hosts must be two distinct comma-separated hosts")
    return hosts  # type: ignore[return-value]


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def git_clean(root: Path) -> bool:
    return not subprocess.check_output(
        ["git", "-C", str(root), "status", "--short"], text=True
    ).strip()


def stage_map(suite: Mapping[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    raw = suite.get("stages")
    if not isinstance(raw, list):
        raise RuntimeError("suite.stages must be a list")
    order: list[str] = []
    mapping: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise RuntimeError("stage entries must be mappings")
        stage_id = str(item.get("id") or "")
        if not stage_id or stage_id in mapping:
            raise RuntimeError(f"invalid/duplicate stage id {stage_id!r}")
        order.append(stage_id)
        mapping[stage_id] = dict(item)
    for stage_id, stage in mapping.items():
        dependencies = stage.get("depends_on", [])
        if not isinstance(dependencies, list):
            raise RuntimeError(f"{stage_id}.depends_on must be a list")
        for dependency in dependencies:
            if dependency not in mapping:
                raise RuntimeError(f"{stage_id}: unknown dependency {dependency}")
            if order.index(dependency) >= order.index(stage_id):
                raise RuntimeError(f"{stage_id}: dependency order is not acyclic")
    return order, mapping


def select_stages(
    order: Sequence[str],
    *,
    only: Sequence[str],
    from_stage: str | None,
    stop_after: str | None,
) -> set[str]:
    selected = set(order)
    if from_stage:
        if from_stage not in order:
            raise RuntimeError(f"unknown --from-stage {from_stage}")
        selected = set(order[order.index(from_stage):])
    if stop_after:
        if stop_after not in order:
            raise RuntimeError(f"unknown --stop-after {stop_after}")
        selected &= set(order[: order.index(stop_after) + 1])
    if only:
        unknown = set(only) - set(order)
        if unknown:
            raise RuntimeError(f"unknown --only stages: {sorted(unknown)}")
        selected &= set(only)
    return selected


def format_command(value: Any, context: Mapping[str, str]) -> list[str]:
    if isinstance(value, str):
        return shlex.split(value.format(**context))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item.format(**context) for item in value]
    raise RuntimeError("stage command must be a string or list of strings")


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def remote_bash(ssh: str, host: str, command: str) -> list[str]:
    return [ssh, host, "bash", "-lc", shlex.quote(command)]


@dataclass
class RemoteProcess:
    host: str
    rank: int
    process: subprocess.Popen[bytes]
    log: Any
    pid_file: str
    return_code: int | None = None


class RemotePair:
    def __init__(
        self,
        *,
        ssh: str,
        hosts: tuple[str, str],
        root: Path,
        site_env: str,
        suite_run_id: str,
        stage_id: str,
        stage_run_id: str,
        command: Sequence[str],
        log_dir: Path,
        dry_run: bool,
    ) -> None:
        self.ssh = ssh
        self.hosts = hosts
        self.root = root
        self.site_env = site_env
        self.suite_run_id = suite_run_id
        self.stage_id = stage_id
        self.stage_run_id = stage_run_id
        self.command = list(command)
        self.log_dir = log_dir
        self.dry_run = dry_run
        self.processes: list[RemoteProcess] = []

    def _pid_file(self, rank: int) -> str:
        return str(
            self.root
            / "artifacts/suites"
            / self.suite_run_id
            / "remote"
            / self.stage_id
            / self.stage_run_id
            / f"node_{rank}.pid"
        )

    def _remote_command(self, rank: int) -> list[str]:
        site_env = Path(self.site_env)
        if not site_env.is_absolute():
            site_env = self.root / site_env
        wrapper = self.root / "scripts/distillschedule/remote_stage_wrapper.sh"
        pid_file = self._pid_file(rank)
        env = [
            "env",
            f"DISTILLSCHEDULE_SUITE_RUN_ID={self.suite_run_id}",
            f"DISTILLSCHEDULE_STAGE_ID={self.stage_id}",
            f"DISTILLSCHEDULE_STAGE_RUN_ID={self.stage_run_id}",
            f"DISTILLSCHEDULE_NODE_RANK={rank}",
        ]
        remote = (
            f"set -Eeuo pipefail; source {shlex.quote(str(site_env))}; "
            f"export DMD_SYSTEM_ROOT={shlex.quote(str(self.root))}; "
            f"exec {shlex.quote(str(wrapper))} {shlex.quote(pid_file)} "
            f"{shell_join([*env, *self.command])}"
        )
        return remote_bash(self.ssh, self.hosts[rank], remote)

    def commands(self) -> list[list[str]]:
        return [self._remote_command(rank) for rank in (0, 1)]

    def cancel(self, rank: int) -> None:
        site_env = Path(self.site_env)
        if not site_env.is_absolute():
            site_env = self.root / site_env
        cancel = self.root / "scripts/distillschedule/cancel_remote_stage.sh"
        remote = (
            f"set -Eeuo pipefail; source {shlex.quote(str(site_env))}; "
            f"{shlex.quote(str(cancel))} {shlex.quote(self._pid_file(rank))}"
        )
        subprocess.run(
            remote_bash(self.ssh, self.hosts[rank], remote),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=90,
        )

    def run(self, timeout_seconds: int) -> dict[str, Any]:
        if self.dry_run:
            return {"status": "DRY_RUN", "commands": self.commands()}
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for rank, command in enumerate(self.commands()):
            handle = (self.log_dir / f"ssh_node_{rank}.log").open("wb")
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.processes.append(
                RemoteProcess(
                    host=self.hosts[rank],
                    rank=rank,
                    process=process,
                    log=handle,
                    pid_file=self._pid_file(rank),
                )
            )
        deadline = time.monotonic() + timeout_seconds
        cancelled = False
        while True:
            for item in self.processes:
                if item.return_code is None:
                    value = item.process.poll()
                    if value is not None:
                        item.return_code = int(value)
                        item.log.flush()
                        item.log.close()
            failed = [
                item
                for item in self.processes
                if item.return_code is not None and item.return_code != 0
            ]
            if failed and not cancelled:
                cancelled = True
                for item in self.processes:
                    if item.return_code is None:
                        self.cancel(item.rank)
                        try:
                            os.killpg(item.process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
            if all(item.return_code is not None for item in self.processes):
                break
            if time.monotonic() >= deadline:
                cancelled = True
                for item in self.processes:
                    if item.return_code is None:
                        self.cancel(item.rank)
                        try:
                            os.killpg(item.process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                break
            time.sleep(1)
        for item in self.processes:
            try:
                item.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(item.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                item.process.wait()
            if item.return_code is None:
                item.return_code = item.process.returncode
            if not item.log.closed:
                item.log.close()
        return_codes = {str(item.rank): item.return_code for item in self.processes}
        return {
            "status": "PASS" if all(value == 0 for value in return_codes.values()) else "FAIL",
            "return_codes": return_codes,
            "peer_cancelled": cancelled,
            "pid_files": {str(item.rank): item.pid_file for item in self.processes},
        }

    def cancel_all(self) -> None:
        for item in self.processes:
            if item.return_code is None:
                self.cancel(item.rank)
                try:
                    os.killpg(item.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


_ACTIVE_PAIR: RemotePair | None = None


def _signal_handler(signum: int, _frame: Any) -> None:
    if _ACTIVE_PAIR is not None:
        _ACTIVE_PAIR.cancel_all()
    raise KeyboardInterrupt(f"signal {signum}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--suite", required=True, type=Path)
    result.add_argument("--hosts", required=True, type=parse_hosts)
    result.add_argument("--run-id", required=True)
    result.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--site-env")
    result.add_argument("--ssh", default="ssh")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--only", action="append", default=[])
    result.add_argument("--from-stage")
    result.add_argument("--stop-after")
    result.add_argument("--force-rerun-stage", action="append", default=[])
    result.add_argument("--include-optional", action="store_true")
    return result


def reuse_evidence(stage: Mapping[str, Any]) -> dict[str, Any]:
    disposition = str(stage.get("disposition"))
    path_text = str(stage.get("manifest") or "")
    required = bool(stage.get("required", disposition == "REUSED_ACCEPTED"))
    if not path_text:
        if required:
            return {"status": "FAIL", "reason": "required reuse manifest is not configured"}
        return {
            "status": disposition,
            "evidence_available": False,
            "reason": "bounded diagnostic manifest is not configured",
        }
    path = Path(path_text).expanduser()
    if not path.is_file():
        if required:
            return {"status": "FAIL", "reason": f"required reuse manifest missing: {path}"}
        return {
            "status": disposition,
            "evidence_available": False,
            "manifest": str(path),
        }
    expected_sha = str(stage.get("sha256") or "")
    actual_sha = sha256_file(path)
    if expected_sha and expected_sha != actual_sha:
        return {"status": "FAIL", "reason": f"reuse manifest SHA mismatch: {path}"}
    payload = load_json(path)
    accepted_statuses = stage.get("accepted_manifest_statuses", ["PASS"])
    if payload.get("status") not in accepted_statuses:
        return {
            "status": "FAIL",
            "reason": f"reuse manifest status={payload.get('status')!r} not accepted",
            "manifest": str(path),
        }
    return {
        "status": disposition,
        "evidence_available": True,
        "manifest": str(path),
        "manifest_sha256": actual_sha,
        "manifest_status": payload.get("status"),
    }


def result_markdown(run_id: str, order: Sequence[str], results: Mapping[str, Any]) -> str:
    lines = [f"# No-GAN suite `{run_id}`", "", "| Stage | Status | Stage run |", "|---|---|---|"]
    for stage_id in order:
        result = results.get(stage_id, {})
        lines.append(
            f"| `{stage_id}` | `{result.get('status', 'NOT_RUN')}` | "
            f"`{result.get('stage_run_id', '-')}` |"
        )
    lines.extend(
        [
            "",
            "A downstream stage marked `NOT_RUN` was not executed. Failed or OOM attempts are preserved under their immutable stage-run IDs and are never overwritten.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    global _ACTIVE_PAIR
    args = parser().parse_args(argv)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _signal_handler)
    root = args.repo_root.expanduser().resolve()
    suite_path = args.suite.expanduser().resolve()
    if not git_clean(root) and not args.dry_run:
        print(json.dumps({"status": "FAIL", "error": "repository is dirty"}))
        return 2
    try:
        suite = expand_env(load_yaml(suite_path))
        order, stages = stage_map(suite)
        selected = select_stages(
            order,
            only=args.only,
            from_stage=args.from_stage,
            stop_after=args.stop_after,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        return 2
    suite_root = root / "artifacts/suites" / args.run_id
    prereg_path = suite_root / "suite_preregistration.json"
    site_env = args.site_env or str(
        (suite.get("remote") or {}).get(
            "site_env", "configs/distillschedule/server/baai_2x8_a100.env.example"
        )
    )
    identity = {
        "schema_version": SCHEMA,
        "suite_id": suite.get("suite_id"),
        "suite_path": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "repo_head": git_head(root),
        "hosts": list(args.hosts),
        "run_id": args.run_id,
        "selected_stages": sorted(selected),
        "site_env": site_env,
    }
    if prereg_path.exists():
        previous = load_json(prereg_path)
        if previous.get("identity") != identity:
            print(json.dumps({"status": "FAIL", "error": "immutable suite ID collision"}))
            return 2
        if not args.resume and not args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "FAIL",
                        "error": "suite ID exists; use --resume or choose a new immutable ID",
                    }
                )
            )
            return 2
    else:
        suite_root.mkdir(parents=True, exist_ok=False)
        atomic_json(
            prereg_path,
            {
                "schema_version": SCHEMA,
                "identity": identity,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "stage_contract_digests": {
                    stage_id: digest(
                        {
                            "suite_sha": identity["suite_sha256"],
                            "repo_head": identity["repo_head"],
                            "hosts": identity["hosts"],
                            "stage": stages[stage_id],
                        }
                    )
                    for stage_id in order
                },
            },
        )
    prereg = load_json(prereg_path)
    contract_digests = prereg["stage_contract_digests"]
    results: dict[str, Any] = {}
    blocked = False

    for stage_id in order:
        stage = stages[stage_id]
        disposition = str(stage.get("disposition", "RUN_REQUIRED"))
        stage_contract = contract_digests[stage_id]
        stage_dir = suite_root / "stages" / stage_id
        latest_path = stage_dir / "latest.json"
        dependencies = list(stage.get("depends_on", []))

        if stage_id not in selected:
            previous = load_json(latest_path) if latest_path.is_file() else None
            results[stage_id] = previous or {
                "status": "NOT_RUN",
                "reason": "stage excluded by selection",
            }
            continue
        if disposition == "OPTIONAL" and not args.include_optional:
            results[stage_id] = {"status": "OPTIONAL", "reason": "not requested"}
            continue
        unmet = [
            dependency
            for dependency in dependencies
            if results.get(dependency, {}).get("status") not in ACCEPTED
        ]
        if blocked or unmet:
            result = {
                "status": "NOT_RUN",
                "reason": "upstream gate failed or was not accepted",
                "unmet_dependencies": unmet,
            }
            results[stage_id] = result
            atomic_json(latest_path, result)
            continue

        if disposition.startswith("REUSED_"):
            result = reuse_evidence(stage)
            result.update(
                {
                    "stage_id": stage_id,
                    "stage_contract_digest": stage_contract,
                    "stage_run_id": f"{args.run_id}-{stage_id}-reuse",
                }
            )
            results[stage_id] = result
            atomic_json(latest_path, result)
            if result["status"] == "FAIL":
                blocked = True
            continue

        previous = load_json(latest_path) if latest_path.is_file() else None
        if previous and args.resume and previous.get("status") in ACCEPTED:
            if previous.get("stage_contract_digest") == stage_contract:
                reused = dict(previous)
                reused["resumed"] = True
                results[stage_id] = reused
                continue
        force = stage_id in args.force_rerun_stage
        if previous and previous.get("status") not in ACCEPTED and not force:
            result = {
                "status": "FAIL",
                "reason": "failed/partial immutable stage is not rerun; use --force-rerun-stage to create a new stage-run ID",
                "previous": previous,
            }
            results[stage_id] = result
            blocked = True
            continue
        suffix = ""
        if force:
            suffix = "-rerun-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
        stage_run_id = f"{args.run_id}-{stage_id}{suffix}"
        attempt_dir = stage_dir / "attempts" / stage_run_id
        if attempt_dir.exists():
            print(json.dumps({"status": "FAIL", "error": f"attempt exists: {stage_run_id}"}))
            return 2
        attempt_dir.mkdir(parents=True)
        context = {
            "repo_root": str(root),
            "hosts_csv": ",".join(args.hosts),
            "host0": args.hosts[0],
            "host1": args.hosts[1],
            "suite_run_id": args.run_id,
            "stage_id": stage_id,
            "stage_run_id": stage_run_id,
            "stage_dir": str(attempt_dir),
            "site_env": site_env,
        }
        executor = str(stage.get("executor", "local"))
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        execution: dict[str, Any]
        return_code = 0
        try:
            if executor == "dual_host":
                command = format_command(stage.get("command"), context)
                pair = RemotePair(
                    ssh=args.ssh,
                    hosts=args.hosts,
                    root=root,
                    site_env=site_env,
                    suite_run_id=args.run_id,
                    stage_id=stage_id,
                    stage_run_id=stage_run_id,
                    command=command,
                    log_dir=attempt_dir,
                    dry_run=args.dry_run,
                )
                _ACTIVE_PAIR = pair
                execution = pair.run(int(stage.get("timeout_seconds", 1800)))
                _ACTIVE_PAIR = None
                return_code = 0 if execution["status"] in {"PASS", "DRY_RUN"} else 1
            elif executor == "external_env":
                variable = str(stage.get("command_env") or "")
                template = os.environ.get(variable, "")
                if not template:
                    execution = {
                        "status": "NOT_RUN",
                        "reason": f"required external command env {variable} is not set",
                    }
                    return_code = 3
                else:
                    command = format_command(template, context)
                    if args.dry_run:
                        execution = {"status": "DRY_RUN", "command": command}
                    else:
                        completed = subprocess.run(
                            command,
                            check=False,
                            stdout=(attempt_dir / "console.log").open("wb"),
                            stderr=subprocess.STDOUT,
                        )
                        return_code = completed.returncode
                        execution = {
                            "status": "PASS" if return_code == 0 else "FAIL",
                            "return_code": return_code,
                            "command": command,
                        }
            else:
                command_value = stage.get("command")
                if executor == "g_offload":
                    candidate = os.environ.get(
                        str(stage.get("candidate_command_env")), ""
                    )
                    continuous = os.environ.get(
                        str(stage.get("continuous_command_env")), ""
                    )
                    if not candidate:
                        execution = {
                            "status": "NOT_RUN",
                            "reason": "target G/S/T pool candidate command is not configured",
                            "historical_bounds_used": False,
                        }
                        return_code = 3
                        command_value = None
                    else:
                        context = dict(context)
                        context["candidate_command"] = candidate
                        context["continuous_command"] = continuous
                if command_value is not None:
                    command = format_command(command_value, context)
                    if args.dry_run:
                        execution = {"status": "DRY_RUN", "command": command}
                    else:
                        with (attempt_dir / "console.log").open("wb") as handle:
                            completed = subprocess.run(
                                command,
                                check=False,
                                stdout=handle,
                                stderr=subprocess.STDOUT,
                            )
                        return_code = completed.returncode
                        execution = {
                            "status": "PASS" if return_code == 0 else "FAIL",
                            "return_code": return_code,
                            "command": command,
                        }
        except KeyboardInterrupt:
            if _ACTIVE_PAIR is not None:
                _ACTIVE_PAIR.cancel_all()
            _ACTIVE_PAIR = None
            execution = {"status": "FAIL", "reason": "interrupted; peer cancellation requested"}
            return_code = 130
        except Exception as exc:
            if _ACTIVE_PAIR is not None:
                _ACTIVE_PAIR.cancel_all()
            _ACTIVE_PAIR = None
            execution = {"status": "FAIL", "reason": str(exc)}
            return_code = 1

        status = execution.get("status")
        if status == "DRY_RUN":
            final_status = "PASS"
        elif status == "NOT_RUN":
            final_status = "NOT_RUN"
        else:
            final_status = "PASS" if return_code == 0 and status == "PASS" else "FAIL"
        result = {
            "schema_version": SCHEMA,
            "stage_id": stage_id,
            "stage_run_id": stage_run_id,
            "stage_contract_digest": stage_contract,
            "status": final_status,
            "dry_run": args.dry_run,
            "started_at": started,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "execution": execution,
        }
        atomic_json(attempt_dir / "result.json", result)
        atomic_json(latest_path, result)
        results[stage_id] = result
        if final_status not in ACCEPTED:
            blocked = True

    overall = "PASS" if all(
        result.get("status") in ACCEPTED or result.get("status") == "OPTIONAL"
        for stage_id, result in results.items()
        if stage_id in selected
    ) else "PARTIAL"
    summary = {
        "schema_version": SCHEMA,
        "suite_id": suite.get("suite_id"),
        "run_id": args.run_id,
        "status": overall,
        "repo_head": identity["repo_head"],
        "suite_sha256": identity["suite_sha256"],
        "hosts": list(args.hosts),
        "stages": results,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "claim_boundary": (
            "No formal Route-A/Route-B speedup is claimable unless the paired formal timing stage is PASS with matched repeated trials."
        ),
    }
    atomic_json(suite_root / "summary.json", summary)
    (suite_root / "summary.md").write_text(result_markdown(args.run_id, order, results))
    print(
        json.dumps(
            {
                "status": overall,
                "summary": str(suite_root / "summary.json"),
                "markdown": str(suite_root / "summary.md"),
            },
            sort_keys=True,
        )
    )
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
