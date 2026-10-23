"""Receipt-host orchestration for the pinned H100 Wan launcher.

The controller owns transport and GPU allocation. This module never opens SSH,
imports Torch, modifies the native source, or chooses a winning recipe itself.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import time
from typing import Any, Mapping

from distillschedule.workloads.best_known.registry import (
    _object, file_sha256, fingerprint, git_head, repository_root, write_new_json,
)
from .allocation import ExecutionAllocation
from .contracts import require
from .cost_model import CostModel
from .model_search import search_plans, validate_saved_prediction
from .plan_ir import PlanningProblem

PLACEMENTS = ("world16-fsdp", "resident", "resident-async128-tree-chunk2048")
DEFAULT_PLACEMENTS = (PLACEMENTS[0], PLACEMENTS[2])
LAUNCHER = "scripts/distillschedule/launch_wan13b_hardware_two_cycle_node.sh"
WORKER = "scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py"
COMPARATOR = "scripts/distillschedule/compare_wan13b_hardware_two_cycle.py"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_LABEL = {PLACEMENTS[0]: "fsdp", PLACEMENTS[1]: "resident", PLACEMENTS[2]: "async"}


def _adapter():
    # Kept lazy so --help remains a cheap, stdlib-only command.
    from . import hardware_wan
    return hardware_wan


def _ref(path: Path) -> dict[str, str]:
    require(path.is_file(), f"native evidence missing: {path}")
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _read_ref(value: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256"},
            "expected a content-hashed native evidence reference")
    require(isinstance(value["path"], str) and isinstance(value["sha256"], str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", value["sha256"])),
            "invalid native evidence reference path or digest")
    path = Path(value["path"])
    require(path.is_absolute() and path.is_file(), "native evidence reference missing")
    require(file_sha256(path) == value["sha256"], "native evidence reference hash changed")
    return _object(path)


def _source_state(root: Path, expected_head: str) -> dict[str, Any]:
    require(git_head(root) == expected_head, "source checkout HEAD changed")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True, timeout=30,
    ).strip()
    require(not dirty, f"execution requires a clean source checkout: {root}")
    return {"head": expected_head, "clean": True}


def validate_deployment(value: Mapping[str, Any], *, local_files: bool = False) -> dict[str, Any]:
    deployment = _adapter().validate_deployment(value)
    if local_files:
        native = Path(deployment["native_source"]["repo_root"])
        _source_state(repository_root(), deployment["binding"]["source_head"])
        _source_state(native, deployment["native_source"]["head"])
        for name, key in ((LAUNCHER, "launcher_sha256"), (WORKER, "worker_sha256")):
            require((native / name).is_file()
                    and file_sha256(native / name) == deployment["native_source"][key],
                    f"pinned native source bytes changed: {name}")
        require((native / COMPARATOR).is_file(), "pinned native comparator missing")
        site = deployment["site"]
        require(Path(site["python"]).is_file() and os.access(site["python"], os.X_OK),
                "site Python executable missing")
        for key in ("project_root", "helios_root", "asset_root"):
            require(Path(site[key]).is_dir(), f"site directory missing: {key}")
        assets = Path(site["asset_root"])
        for relative in ("wan13b/Wan2.1-T2V-1.3B-Diffusers-0fad780a",
                         "wan13b/systems_only_bootstrap_v3_head04e6e44",
                         "m0/traces/syn_feature_trace_rank16_v4_strict_sync_steps20",
                         "m1/negative_prompt_official_v1"):
            require((assets / relative).exists(), f"native asset path missing: {relative}")
    return deployment


def inspect_deployment(deployment: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_deployment(deployment, local_files=True)
    return {"status": "PASS_CPU_SOURCE_AND_MANIFEST", "deployment_sha256": fingerprint(value),
            "binding": value["binding"], "native_source": value["native_source"],
            "allocation": value["allocation"], "evidence_scope": value["evidence_scope"],
            "gpu_execution": "NOT_RUN", "claimable_speedup": False}


def launch_environment(deployment: Mapping[str, Any], *, inherited_locks: bool = False) -> dict[str, str]:
    # No caller-provided DISTILLSCHEDULE/NCCL/PYTHONPATH knob reaches the native
    # worker. Its unchanged launcher establishes the measured per-arm settings.
    env = {key: os.environ[key] for key in
           ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR") if key in os.environ}
    env.update(DISTILLSCHEDULE_PYTHON=deployment["site"]["python"],
               DISTILLSCHEDULE_HELIOS_ROOT=deployment["site"]["helios_root"],
               PYTHONDONTWRITEBYTECODE="1")
    # The isolated environment may own libstdc++/OpenMP. Do not inherit the
    # controller's libraries; use the declared path or this Python prefix.
    prefix_lib = Path(deployment["site"]["python"]).parent.parent / "lib"
    library_path = deployment["environment"].get("ld_library_path",
        str(prefix_lib) if prefix_lib.is_dir() else "")
    require(isinstance(library_path, str) and all(Path(part).is_absolute()
            for part in library_path.split(":") if part), "LD_LIBRARY_PATH must be explicitly absolute")
    require(not library_path or all(library_path.split(":")), "LD_LIBRARY_PATH cannot include caller working directories")
    if library_path:
        env["LD_LIBRARY_PATH"] = library_path
    if inherited_locks:
        for fd in (7, 8, 9):
            os.fstat(fd)
        env["DISTILLSCHEDULE_INHERITED_GPU_LOCKS"] = "1"
    return env


_ENV_PROBE = """
import hashlib, importlib.metadata, json, os, platform, subprocess, sys
from pathlib import Path
expected = json.loads(sys.argv[1])
result = {}
extension = None
if any(key in expected for key in ('torch_extension_origin', 'torch_extension_sha256')):
    directory = importlib.metadata.distribution('torch').locate_file('torch')
    extensions = sorted(Path(directory).glob('_C*.so'))
    if len(extensions) != 1:
        raise ValueError('expected one installed torch._C extension')
    extension = extensions[0].resolve()
for key, value in expected.items():
    if key == 'python':
        result[key] = platform.python_version()
    elif key in ('python_executable_sha256', 'executable_sha256'):
        result[key] = hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    elif key == 'executable':
        result[key] = str(Path(sys.executable).resolve())
    elif key == 'ld_library_path':
        result[key] = os.environ.get('LD_LIBRARY_PATH', '')
    elif key == 'driver_version':
        observed = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            text=True, timeout=30)
        versions = {line.strip() for line in observed.splitlines() if line.strip()}
        if len(versions) != 1:
            raise ValueError('expected one observed NVIDIA driver version')
        result[key] = versions.pop()
    elif key == 'torch_extension_origin':
        result[key] = str(extension)
    elif key == 'torch_extension_sha256':
        result[key] = hashlib.sha256(extension.read_bytes()).hexdigest()
    elif key == 'packages':
        result[key] = {name: importlib.metadata.version(name) for name in value}
    else:
        result[key] = importlib.metadata.version(key)
print(json.dumps(result, sort_keys=True))
"""


def observe_host(deployment: Mapping[str, Any]) -> dict[str, Any]:
    """Driver and installed-package metadata; no model or CUDA context."""
    expected = deployment["environment"]
    require(isinstance(expected.get("python"), str) and expected["python"],
            "runtime environment must declare its Python version")
    if "packages" in expected:
        require(isinstance(expected["packages"], Mapping) and expected["packages"],
                "runtime environment packages must be a nonempty version map")
    env = launch_environment(deployment)
    require(expected.get("ld_library_path") == env.get("LD_LIBRARY_PATH", ""),
            "deployment environment must bind the actual ld_library_path (including an empty path)")
    observed = json.loads(subprocess.check_output(
        [deployment["site"]["python"], "-c", _ENV_PROBE, json.dumps(expected)],
        env=env, text=True, timeout=60,
    ))
    require(observed == expected, "observed installed environment differs from deployment")
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
        env=env, text=True, timeout=30,
    )
    devices = []
    for row in csv.reader(io.StringIO(raw)):
        require(len(row) == 4 and "H100" in row[2], "expected an eight-H100 host")
        rank, uuid, _, mib = (part.strip() for part in row)
        devices.append({"rank": int(rank), "node": 0, "local_rank": int(rank),
                        "uuid": uuid, "architecture": "sm90", "capacity_bytes": int(mib) * 2**20})
    devices.sort(key=lambda device: device["rank"])
    require(devices == deployment["allocation"]["devices"],
            "fresh GPU inventory differs from selected allocation")
    return {"hostname": socket.gethostname(), "devices": devices, "environment": observed}


def _native_manifests(deployment: Mapping[str, Any], run_root: Path,
                      argv: list[str]) -> tuple[list[str], dict[str, str]]:
    """Read what the pinned shell actually launched, rather than echo a plan."""
    site, native = deployment["site"], Path(deployment["native_source"]["repo_root"])
    run_id, mode, placement = argv[3:6]
    assets = Path(site["asset_root"])
    expected = [site["python"], "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
        str(native / WORKER), "--run-id", run_id, "--output-dir", str(run_root / "execution"),
        "--mode", mode, "--placement", placement, "--expected-head", deployment["native_source"]["head"],
        "--timing-seed", "20260903", "--timing-arm-id", f"h100-{placement}",
        "--model-root", str(assets / "wan13b/Wan2.1-T2V-1.3B-Diffusers-0fad780a"),
        "--bootstrap", str(assets / "wan13b/systems_only_bootstrap_v3_head04e6e44"),
        "--expected-bootstrap-hash", "6a713476d9839cf1ea37689d63cb1cc2b405b9f9915ca5314ccdadb67b3cd60d",
        "--trace-bundle", str(assets / "m0/traces/syn_feature_trace_rank16_v4_strict_sync_steps20"),
        "--expected-trace-hash", "654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3",
        "--negative-prompt", str(assets / "m1/negative_prompt_official_v1"), *argv[8:]]
    actual = shlex.split((run_root / "manifests/command.sh").read_text())
    require(actual == expected, "recorded native worker argv differs from selected launch")
    environment = {}
    for line in (run_root / "manifests/environment.txt").read_text().splitlines():
        key, separator, val = line.partition("=")
        require(bool(separator) and key not in environment, "malformed native environment receipt")
        environment[key] = val
    expected_env = {"DISTILLSCHEDULE_PYTHON": site["python"],
        "DISTILLSCHEDULE_HELIOS_ROOT": site["helios_root"],
        "DISTILLSCHEDULE_HARDWARE_PROFILE": "h100-8-b16",
        "PYTHONPATH": f"{native / 'src'}:{native}:{site['helios_root']}",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "OMP_NUM_THREADS": "1",
        "HF_HOME": str(assets / "hf_home_official_torch210"), "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "DISTILLSCHEDULE_WAN13B_EMA_IMPLEMENTATION": "reference",
        "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE": "0",
        "DISTILLSCHEDULE_WAN13B_LORA_OPERATOR_ATTRIBUTION": "0",
        "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE":
            "1" if placement == PLACEMENTS[2] else "0"}
    if placement == PLACEMENTS[2]:
        expected_env["NCCL_ALGO"] = "Tree"
    require(environment == expected_env, "recorded native worker environment differs from pinned launcher")
    return actual, environment


def _proc_identity(pid: int) -> tuple[int, int, int] | None:
    """(birth ticks, process group, session); a reused PID is a different owner."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19]), int(fields[2]), int(fields[3])
    except (OSError, ValueError, IndexError):
        return None


def _proc_is_live(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except (OSError, IndexError):
        return False


def _owned_groups(pid: int, birth: int) -> dict[int, dict[int, tuple[int, int, int]]]:
    # GNU timeout may create its own process group. The private session created
    # by Popen still identifies those descendants after they are reparented.
    groups = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        child = int(path.name)
        identity = _proc_identity(child)
        if identity and identity[0] >= birth and identity[2] == pid and _proc_is_live(child):
            groups.setdefault(identity[1], {})[child] = identity
    return groups


def _signal_owned(groups: Mapping[int, Mapping[int, tuple[int, int, int]]], sig: int) -> None:
    for pgid, members in groups.items():
        # A group remains ours when its original leader has exited. Verify a
        # recorded live member's birth, group AND private session before signal.
        if any(identity[1] == pgid and _proc_identity(member) == identity and _proc_is_live(member)
               for member, identity in members.items()):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass


def _cleanup_session(pid: int, birth: int) -> bool:
    groups = _owned_groups(pid, birth)
    if not groups:
        return False
    _signal_owned(groups, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while groups and time.monotonic() < deadline:
        time.sleep(0.05)
        groups = _owned_groups(pid, birth)
    if groups:
        _signal_owned(groups, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while groups and time.monotonic() < deadline:
            time.sleep(0.05)
            groups = _owned_groups(pid, birth)
    require(not groups, "owned native session still has live descendants after bounded teardown")
    return True


def _run_owned(command: list[str], *, env: dict[str, str], cwd: Path,
               log_path: Path, timeout_seconds: float, pass_fds: tuple[int, ...] = ()) -> None:
    require(math.isfinite(timeout_seconds) and 0 < timeout_seconds <= 7200,
            "native execution deadline must be positive and <=7200 seconds")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x") as log:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=pass_fds)
        identity = _proc_identity(proc.pid)
        require(identity is not None and identity[1:] == (proc.pid, proc.pid),
                "could not establish task-owned process birth/group/session")
        succeeded = False
        try:
            code = proc.wait(timeout=timeout_seconds)
            require(code == 0, f"native command exited {code}; evidence retained in {log_path}")
            succeeded = True
        finally:
            cleaned = _cleanup_session(proc.pid, identity[0])
            # Reap only the child created here, never a foreign GPU PID. The
            # session scan also finds children after a failed parent was reaped.
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("owned native child survived bounded teardown") from error
            require(not (succeeded and cleaned), "native command exited successfully but left live owned descendants")


def _check_run_id(run_id: str) -> None:
    require(isinstance(run_id, str) and bool(_RUN_ID.fullmatch(run_id)), "unsafe native run id")


def _native_argv(deployment: Mapping[str, Any], *, run_id: str, mode: str, placement: str,
                 correctness: Mapping[str, Any] | None, comparison: Mapping[str, Any] | None) -> list[str]:
    _check_run_id(run_id)
    require(mode in ("correctness", "timing") and placement in PLACEMENTS,
            "unsupported native mode or placement")
    source, site = deployment["native_source"], deployment["site"]
    argv = ["bash", str(Path(source["repo_root"]) / LAUNCHER), site["project_root"],
            run_id, mode, placement, source["head"], site["asset_root"]]
    if mode == "timing":
        require(correctness is not None and comparison is not None,
                "fresh timing requires native correctness and trajectory comparison")
        _adapter().validate_native_timing_prerequisites(deployment, correctness, comparison, placement=placement)
        argv += ["--correctness-receipt", correctness["path"],
                 "--expected-correctness-hash", correctness["sha256"],
                 "--trajectory-comparison-receipt", comparison["path"],
                 "--expected-trajectory-comparison-hash", comparison["sha256"]]
    else:
        require(correctness is None and comparison is None, "correctness cannot reuse timing bindings")
    return argv


def run_native(deployment: Mapping[str, Any], *, run_id: str, mode: str, placement: str,
               output_dir: Path, timeout_seconds: float = 1800, inherited_locks: bool = False,
               correctness: Mapping[str, Any] | None = None,
               comparison: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Launch once, then bind only the evidence produced during this execution."""
    value = validate_deployment(deployment)
    require(value["evidence_scope"] == "MEASURED", "CPU fixtures cannot execute native GPU workloads")
    _check_run_id(run_id)
    run_root = Path(value["site"]["run_root"]) / run_id
    require(not run_root.exists(), "native run directory already exists; historical runs cannot be rebound")
    validate_deployment(value, local_files=True)
    before = observe_host(value)
    argv = _native_argv(value, run_id=run_id, mode=mode, placement=placement,
                        correctness=correctness, comparison=comparison)
    env = launch_environment(value, inherited_locks=inherited_locks)
    started = time.time_ns()
    _run_owned(argv, env=env, cwd=Path(value["native_source"]["repo_root"]),
               log_path=Path(output_dir) / f"{run_id}.log", timeout_seconds=timeout_seconds,
               pass_fds=(7, 8, 9) if inherited_locks else ())
    completed = time.time_ns()
    validate_deployment(value, local_files=True)
    after = observe_host(value)
    require(before == after, "host, devices, or runtime changed during native execution")
    summary = _object(run_root / "summary.json")
    require(summary.get("evidence_scope") != "CPU_FIXTURE_ONLY",
            "CPU fixture outputs cannot become fresh native execution evidence")
    require(summary.get("run_id") == run_id and summary.get("mode") == mode,
            "native summary belongs to another run or mode")
    _adapter().validate_native_summary(summary, expected_native_head=value["native_source"]["head"],
        placement=placement, allocation=ExecutionAllocation.from_dict(value["allocation"]))
    ranks = sorted((run_root / "execution/ranks").glob("rank_*.json"))
    require(len(ranks) == 8, "native execution requires all eight fresh rank receipts")
    raw_ranks = [_object(path) for path in ranks]
    require(all(type(row.get("rank")) is int for row in raw_ranks)
            and sorted(row["rank"] for row in raw_ranks) == list(range(8)),
            "native raw physical rank coverage changed")
    require(sorted(raw_ranks, key=lambda row: row["rank"]) ==
            sorted(summary.get("ranks", []), key=lambda row: row["rank"]),
            "native summary ranks differ from actual raw rank files")
    for row in raw_ranks:
        require(row.get("hostname") == before["hostname"], "native rank belongs to another host")
        require(row.get("run_id") == run_id, "native rank run identity changed")
        for cycle in row.get("cycles", []):
            start, stop = cycle.get("start_unix_ns"), cycle.get("stop_unix_ns")
            require(type(start) is int and type(stop) is int
                    and started <= start < stop <= completed,
                    "rank timer is outside this fresh native launch")
    command_ref = _ref(run_root / "manifests/command.sh")
    environment_ref = _ref(run_root / "manifests/environment.txt")
    worker_argv, worker_environment = _native_manifests(value, run_root, argv)
    binding = {"schema_version": "distillschedule.hardware_wan_launch.v1",
        "deployment_sha256": fingerprint(value), "integration_head": value["binding"]["source_head"],
        "native_head": value["native_source"]["head"], "integration_clean": True, "native_clean": True,
        "allocation_sha256": fingerprint(value["allocation"]), "placement": placement, "run_id": run_id,
        "argv_sha256": command_ref["sha256"], "environment_sha256": environment_ref["sha256"],
        "native_launcher_sha256": value["native_source"]["launcher_sha256"],
        "native_worker_sha256": value["native_source"]["worker_sha256"],
        "observed_devices": before["devices"], "started_unix_ns": started, "completed_unix_ns": completed,
        "observed_hostname": before["hostname"], "observed_environment": before["environment"],
        "environment_observation_scope": "DECLARED_METADATA_AND_OPTIONAL_SELECTED_BINARIES_NOT_FULL_ENVIRONMENT_HASH",
        "native_worker_argv": worker_argv, "native_worker_environment": worker_environment,
        "native_argv": argv, "native_environment": {k: env[k] for k in env if k.startswith("DISTILLSCHEDULE_")
                                                   or k in ("PYTHONDONTWRITEBYTECODE", "LD_LIBRARY_PATH")}}
    binding_path = run_root / "manifests/hardware_wan_launch.json"
    write_new_json(binding_path, binding)
    observation = {"placement": placement, "run_root": str(run_root),
        "summary": _ref(run_root / "summary.json"), "rank_receipts": [_ref(p) for p in ranks],
        "launcher_health": _ref(run_root / "run_status.json"), "launcher_argv": command_ref,
        "launcher_environment": environment_ref, "launch_binding": _ref(binding_path)}
    if mode == "timing":
        observation.update(correctness=dict(correctness), comparison=dict(comparison))
        _adapter().validate_native_observation(value, observation)
    return observation


def _validate_comparison(value: Mapping[str, Any]) -> None:
    from .hardware_wan import validate_native_comparison
    validate_native_comparison(value)


def compare_native(deployment: Mapping[str, Any], reference: Mapping[str, Any],
                   candidate: Mapping[str, Any], *, output: Path, timeout_seconds: float = 600) -> dict[str, str]:
    value = validate_deployment(deployment, local_files=True)
    require(value["evidence_scope"] == "MEASURED", "CPU fixtures cannot invoke native comparison")
    for summary in (reference, candidate):
        _adapter().validate_native_correctness(value, summary)
    require(not output.exists(), "comparison receipt already exists")
    native = Path(value["native_source"]["repo_root"])
    env = launch_environment(value)
    env["PYTHONPATH"] = f"{native / 'src'}:{native}"
    command = [value["site"]["python"], str(native / COMPARATOR),
               "--reference-summary", reference["path"], "--candidate-summary", candidate["path"],
               "--output", str(output)]
    _run_owned(command, env=env, cwd=native, log_path=output.with_suffix(".log"),
               timeout_seconds=timeout_seconds)
    _adapter().validate_native_comparison_inputs(value, _object(output))
    return _ref(output)


def calibrate(deployment: Mapping[str, Any], *, output_dir: Path, run_prefix: str,
              placements: tuple[str, ...] = DEFAULT_PLACEMENTS, reference: str = PLACEMENTS[0],
              timeout_seconds: float = 1800, inherited_locks: bool = False,
              native_run=None, native_compare=None) -> dict[str, Any]:
    value = validate_deployment(deployment)
    require(value["evidence_scope"] == "MEASURED", "CPU fixtures cannot execute calibration")
    require(2 <= len(placements) <= 3 and len(set(placements)) == len(placements)
            and set(placements) <= set(PLACEMENTS) and reference in placements,
            "calibration needs two or three distinct native arms including its reference")
    _check_run_id(run_prefix)
    run_ids = {(placement, mode): f"{run_prefix}-{_LABEL[placement]}-{mode}"
               for placement in placements for mode in ("correctness", "timing")}
    for run_id in run_ids.values():
        _check_run_id(run_id)
        require(not (Path(value["site"]["run_root"]) / run_id).exists(),
                "calibration requires fresh native run directories")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_new_json(output_dir / "deployment.json", value)
    native_run, native_compare = native_run or run_native, native_compare or compare_native
    correctness = {}
    for placement in placements:
        correctness[placement] = native_run(value, run_id=run_ids[placement, "correctness"],
            mode="correctness", placement=placement, output_dir=output_dir / "logs",
            timeout_seconds=timeout_seconds, inherited_locks=inherited_locks)
    comparisons = {}
    for placement in placements:
        if placement != reference:
            path = output_dir / "comparisons" / f"{_LABEL[reference]}--{_LABEL[placement]}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            comparisons[placement] = native_compare(value, correctness[reference]["summary"],
                correctness[placement]["summary"], output=path)
            _validate_comparison(_read_ref(comparisons[placement]))
    comparisons[reference] = next(iter(comparisons.values()))
    observations = []
    for placement in placements:
        observations.append(native_run(value, run_id=run_ids[placement, "timing"], mode="timing",
            placement=placement, output_dir=output_dir / "logs", timeout_seconds=timeout_seconds,
            inherited_locks=inherited_locks, correctness=correctness[placement]["summary"],
            comparison=comparisons[placement]))
    review = any(_read_ref(ref)["status"] == "HARDWARE_TWO_CYCLE_REVIEW" for ref in comparisons.values())
    result = {"schema_version": "distillschedule.hardware_wan_calibration_session.v1",
        "deployment_sha256": fingerprint(value), "evidence_scope": value["evidence_scope"],
        "run_state": "REVIEW" if review else "PASS", "reference": reference,
        "correctness_observations": list(correctness.values()), "comparisons": comparisons,
        "observations": observations, "strict_full_state_parity": "NOT_RUN", "claimable_speedup": False}
    write_new_json(output_dir / "calibration.json", result)
    return result


def search(deployment: Mapping[str, Any], observations: list[dict[str, Any]], *, output_dir: Path,
           run_id: str, max_proposals: int = 10000, timeout_seconds: float = 600,
           beam_width: int = 0) -> dict[str, Any]:
    value = validate_deployment(deployment)
    _check_run_id(run_id)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    problem, costs = _adapter().assemble_problem_and_costs(value, observations,
        output_dir=output_dir / "components")
    costs.verify_receipts()
    result = search_plans(problem, costs, algorithm="hierarchical", max_proposals=max_proposals,
                          timeout_seconds=timeout_seconds, beam_width=beam_width)
    write_new_json(output_dir / "problem.json", problem.to_dict())
    write_new_json(output_dir / "cost_model.json", costs.to_dict())
    write_new_json(output_dir / "search_result.json", result)
    require(result.get("predicted_best"), "no calibrated feasible hardware plan")
    plan = _adapter().lower_selection(result, problem, costs, value, run_id=run_id)
    write_new_json(output_dir / "plan.json", plan)
    return {"status": result["status"], "evidence_scope": value["evidence_scope"],
            "plan": str(output_dir / "plan.json"), "plan_sha256": plan["plan_sha256"],
            "selected_placement": plan["selected_placement"],
            "predicted_cycle2_ns": result["predicted_best"]["predicted_cycle2_ns"],
            "gpu_execution": "NOT_RUN", "claimable_speedup": False}


def run_selected(plan: Mapping[str, Any], *, output_dir: Path,
                 deployment: Mapping[str, Any] | None = None, timeout_seconds: float = 1800,
                 inherited_locks: bool = False, native_run=None) -> dict[str, Any]:
    selected = _adapter().validate_lowered_plan(plan)
    value = validate_deployment(selected["deployment"])
    require(value["evidence_scope"] == "MEASURED", "CPU fixtures cannot execute selected GPU plans")
    require(deployment is None or validate_deployment(deployment) == value,
            "selected deployment cannot be overridden at execution")
    model = selected["model_plan"]
    problem = PlanningProblem.from_dict(model["problem"])
    costs = CostModel.from_dict(model["cost_model"])
    costs.bind(problem.binding, problem_scope=problem.evidence_scope)
    costs.verify_receipts()
    prediction = validate_saved_prediction(model["search_result"], problem, costs)
    require(problem.evidence_scope == "MEASURED" and costs.scope == "MEASURED",
            "fixture calibration cannot execute a native GPU plan")
    placement = selected["selected_placement"]
    observations = selected["selected_observations"]
    require(observations and all(row["placement"] == placement for row in observations),
            "selected calibration observations changed placement")
    raw = _read_ref(observations[0]["summary"])
    normalized = _adapter().validate_native_summary(raw,
        expected_native_head=value["native_source"]["head"], placement=placement,
        allocation=ExecutionAllocation.from_dict(value["allocation"]))
    correctness, comparison = normalized["correctness_binding"], normalized["trajectory_comparison_binding"]
    # Native rank bindings include descriptive metadata; launcher args use only
    # the content-hashed path pair. The native worker rechecks their contents.
    correctness = {key: correctness[key] for key in ("path", "sha256")}
    comparison = {key: comparison[key] for key in ("path", "sha256")}
    comparison_raw = _adapter().validate_native_timing_prerequisites(value, correctness, comparison, placement=placement)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_new_json(output_dir / "selected_plan.json", dict(selected))
    observation = (native_run or run_native)(value, run_id=selected["run_id"], mode="timing",
        placement=placement, output_dir=output_dir / "logs", timeout_seconds=timeout_seconds,
        inherited_locks=inherited_locks, correctness=correctness, comparison=comparison)
    require(observation.get("placement") == placement, "native execution changed selected placement")
    actual_raw = _read_ref(observation["summary"])
    require(actual_raw.get("evidence_scope") != "CPU_FIXTURE_ONLY",
            "CPU fixture outputs cannot close a measured native execution")
    # Actual execution has the same proof obligation as calibration: re-open
    # all eight ranks, launcher health, argv/env, gates and fresh launch binding.
    actual = _adapter().validate_native_observation(value, observation)
    require(actual["run_id"] == selected["run_id"], "native execution changed selected run identity")
    predicted_ns, measured_ns = prediction["predicted_cycle2_ns"], actual["cycle_ns"][1]
    require(predicted_ns > 0 and measured_ns > 0, "invalid cycle2 prediction/measurement")
    launch = _read_ref(observation["launch_binding"])
    require(launch["placement"] == placement and launch["run_id"] == selected["run_id"]
            and launch["allocation_sha256"] == fingerprint(value["allocation"]),
            "actual launch differs from selected execution plan")
    result = {"schema_version": "distillschedule.hardware_wan_closed_loop.v1",
        "status": "PASS_DIAGNOSTIC_CLOSED_LOOP", "run_state": "REVIEW" if
            comparison_raw["status"] == "HARDWARE_TWO_CYCLE_REVIEW" else "PASS",
        "evidence_scope": "MEASURED", "plan_sha256": selected["plan_sha256"],
        "deployment_sha256": fingerprint(value), "selected_placement": placement,
        "executed_placement": actual["placement"], "allocation": value["allocation"],
        "physical_ranks": 8, "logical_slots": 16, "claimable_cycle": 2,
        "predicted_cycle2_ns": predicted_ns, "measured_cycle2_ns": measured_ns,
        "prediction_relative_error": (measured_ns - predicted_ns) / predicted_ns,
        "prediction_error_denominator": "PREDICTED_CYCLE2_NS",
        "prediction_absolute_error_ns": abs(measured_ns - predicted_ns),
        "observation": observation, "native_argv": launch.get("native_argv"),
        "native_environment": launch.get("native_environment"),
        "native_worker_argv": launch.get("native_worker_argv"),
        "native_worker_environment": launch.get("native_worker_environment"),
        "numerical_status": actual["numerical_status"], "strict_full_state_parity": "NOT_RUN",
        "claimable_speedup": False}
    write_new_json(output_dir / "closed_loop.json", result)
    return result
