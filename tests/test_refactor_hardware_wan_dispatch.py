"""CPU fixtures only: every native effect is replaced; no GPU evidence here."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import select
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from distillschedule.core.enums import GradientMode, OperationKind
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation
from distillschedule.search import hardware_wan_cli as cli
from distillschedule.search import hardware_wan_dispatch as dispatch
from distillschedule.search import hardware_wan
from distillschedule.search.allocation import ExecutionAllocation, PhysicalDevice
from distillschedule.search.cost_model import CostModel, OperationCost
from distillschedule.search.plan_ir import BindingRule, PlanningProblem

FIXTURE = "CPU_FIXTURE_ONLY"


def deployment(tmp_path, *, scope=FIXTURE):
    """MEASURED is used only with a monkeypatched, visibly fixture native host."""
    devices = tuple(PhysicalDevice(r, 0, r, f"GPU-CPU-FIXTURE-{r}", "sm90", 80 << 30) for r in range(8))
    slots = tuple((r, r + 8) for r in range(8))
    allocation = ExecutionAllocation(devices, 16, {role: slots for role in ("generator", "critic", "teacher")})
    environment = {"python": "3.10.20", "packages": {"torch": "2.10.0+cu128"}, "ld_library_path": "", "driver_version": "535.183.06"}
    assets = {key: hardware_wan.SEMANTICS[key] for key in
              ("bootstrap_hash", "trace_bundle_hash", "model_identity_sha256", "negative_prompt_manifest_hash")}
    return {"schema_version": "distillschedule.hardware_wan_deployment.v1", "evidence_scope": scope,
        "binding": {"source_head": "a" * 40, "hardware_fingerprint": allocation.digest,
                    "environment_fingerprint": dispatch.fingerprint(environment),
                    **{f"{k}_fingerprint": dispatch.fingerprint(v) for k, v in (
                        ("asset", assets), ("semantic", hardware_wan.SEMANTICS), ("precision", hardware_wan.PRECISION))}},
        "native_source": {"repo_root": str(tmp_path / "native"), "head": hardware_wan.NATIVE_HEAD,
                          **hardware_wan.NATIVE_FILES},
        "allocation": allocation.to_dict(), "environment": environment,
        "assets": assets, "semantics": copy.deepcopy(hardware_wan.SEMANTICS),
        "precision": copy.deepcopy(hardware_wan.PRECISION),
        "transport": {"host": "CPU-FIXTURE-UNROUTABLE", "ssh_port": 22, "address_family": "auto"},
        "site": {"project_root": str(tmp_path), "python": str(tmp_path / "python"),
                 "helios_root": str(tmp_path / "helios"), "asset_root": str(tmp_path / "assets"),
                 "run_root": str(tmp_path / "research_artifacts/five_workload_h100_20260930")}}


@pytest.fixture
def fixture_adapter(monkeypatch):
    def summary(value, **kwargs):
        assert value["fixture_scope"] == FIXTURE
        assert value["placement"] == kwargs["placement"]
        assert kwargs["allocation"].world_size == 8
        return {**value, "cycle_ns": [10, 20], "numerical_status": "REVIEW"}
    adapter = SimpleNamespace(validate_deployment=lambda v: copy.deepcopy(v), validate_native_summary=summary,
                              validate_lowered_plan=lambda plan: plan)
    def fixture_prerequisites(deployment, correctness, comparison, *, placement):
        # Isolate host plumbing only; this explicit double cannot prove native evidence.
        value = dispatch._read_ref(comparison)
        assert value.get("fixture_scope") == FIXTURE or value.get("evidence_scope") == FIXTURE
        return value
    adapter.validate_native_timing_prerequisites = fixture_prerequisites
    monkeypatch.setattr(dispatch, "_adapter", lambda: adapter)
    return adapter


def comparison():
    from test_refactor_hardware_wan import fixture_comparison
    return {**fixture_comparison(), "fixture_scope": FIXTURE}


def json_ref(path, value):
    dispatch.write_new_json(path, value)
    return dispatch._ref(path)


def test_import_and_help_do_not_import_torch():
    root = Path(__file__).resolve().parents[1]
    code = ("import sys; from distillschedule.search.hardware_wan_cli import main; "
            "assert 'torch' not in sys.modules; "
            "\ntry: main(['--help'])\nexcept SystemExit as e: assert e.code == 0\n"
            "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**dispatch.os.environ, "PYTHONPATH": str(root / "src")}, capture_output=True)


def test_native_environment_excludes_caller_recipe_knobs(tmp_path, monkeypatch):
    value = deployment(tmp_path)
    monkeypatch.setenv("DISTILLSCHEDULE_HARDWARE_PROFILE", "wrong-batch")
    monkeypatch.setenv("DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE", "1")
    monkeypatch.setenv("NCCL_ALGO", "Ring")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/imports")
    env = dispatch.launch_environment(value)
    assert env["DISTILLSCHEDULE_PYTHON"] == value["site"]["python"]
    assert env["DISTILLSCHEDULE_HELIOS_ROOT"] == value["site"]["helios_root"]
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert all(key not in env for key in ("NCCL_ALGO", "PYTHONPATH", "DISTILLSCHEDULE_HARDWARE_PROFILE",
               "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE",
               "DISTILLSCHEDULE_INHERITED_GPU_LOCKS"))
    seen = []
    monkeypatch.setattr(dispatch.os, "fstat", lambda fd: seen.append(fd))
    assert dispatch.launch_environment(value, inherited_locks=True)["DISTILLSCHEDULE_INHERITED_GPU_LOCKS"] == "1"
    assert seen == [7, 8, 9]


def test_fixture_execution_is_rejected_before_native_callback(tmp_path, fixture_adapter):
    value = deployment(tmp_path)
    called = []
    with pytest.raises(ValueError, match="fixtures"):
        dispatch.calibrate(value, output_dir=tmp_path / "cal", run_prefix="fixture", native_run=lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="fixtures"):
        dispatch.run_native(value, run_id="fixture", mode="timing", placement=dispatch.PLACEMENTS[0], output_dir=tmp_path)
    with pytest.raises(ValueError, match="fixtures"):
        dispatch.run_selected({"deployment": value}, output_dir=tmp_path / "run",
                              native_run=lambda *a, **k: called.append(1))
    assert not called and not (tmp_path / "cal").exists()


def test_calibration_runs_all_correctness_then_native_comparisons_then_timing(tmp_path, fixture_adapter):
    value = deployment(tmp_path, scope="MEASURED")
    calls = []
    def native_run(_, **kw):
        calls.append((kw["mode"], kw["placement"]))
        raw = {"fixture_scope": FIXTURE, "placement": kw["placement"], "mode": kw["mode"], "run_id": kw["run_id"]}
        ref = json_ref(tmp_path / "fixture_receipts" / f'{kw["run_id"]}.json', raw)
        if kw["mode"] == "timing":
            assert dispatch._read_ref(kw["correctness"])["placement"] == kw["placement"]
            assert dispatch._read_ref(kw["comparison"])["status"] == "HARDWARE_TWO_CYCLE_REVIEW"
        return {"fixture_scope": FIXTURE, "placement": kw["placement"], "summary": ref}
    def compare(_, reference, candidate, *, output):
        calls.append(("compare", dispatch._read_ref(candidate)["placement"]))
        assert dispatch._read_ref(reference)["placement"] == dispatch.PLACEMENTS[0]
        return json_ref(output, comparison())
    result = dispatch.calibrate(value, output_dir=tmp_path / "cal", run_prefix="fixture",
        placements=dispatch.PLACEMENTS, native_run=native_run, native_compare=compare)
    assert calls == [("correctness", p) for p in dispatch.PLACEMENTS] + [
        ("compare", p) for p in dispatch.PLACEMENTS[1:]] + [("timing", p) for p in dispatch.PLACEMENTS]
    assert result["run_state"] == "REVIEW"
    assert result["strict_full_state_parity"] == "NOT_RUN" and result["claimable_speedup"] is False
    assert json.loads((tmp_path / "cal/calibration.json").read_text()) == result


def test_old_native_run_cannot_receive_a_fresh_launch_binding(tmp_path, fixture_adapter):
    value = deployment(tmp_path, scope="MEASURED")
    old = Path(value["site"]["run_root"]) / "old"
    old.mkdir(parents=True)
    with pytest.raises(ValueError, match="historical"):
        dispatch.run_native(value, run_id="old", mode="correctness", placement=dispatch.PLACEMENTS[0], output_dir=tmp_path)
    assert not (old / "manifests/hardware_wan_launch.json").exists()
    old_cal = Path(value["site"]["run_root"]) / "fixture-fsdp-correctness"
    old_cal.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        dispatch.calibrate(value, output_dir=tmp_path / "cal", run_prefix="fixture")
    assert not (tmp_path / "cal").exists()


@pytest.mark.parametrize("mutate", ["placement", "old_timer", "wrong_host", "fixture_output", "raw_rank"])
def test_new_launch_rejects_changed_placement_old_timers_and_foreign_host(tmp_path, fixture_adapter, monkeypatch, mutate):
    value = deployment(tmp_path, scope="MEASURED")
    monkeypatch.setattr(dispatch, "validate_deployment", lambda v, **_: v)
    host = {"hostname": "CPU-FIXTURE-HOST", "devices": value["allocation"]["devices"], "environment": value["environment"]}
    monkeypatch.setattr(dispatch, "observe_host", lambda v: host)
    def run(argv, **kwargs):
        root = Path(value["site"]["run_root"]) / argv[3]
        rows = []
        for rank in range(8):
            now = time.time_ns()
            row = {"fixture_scope": FIXTURE, "run_id": argv[3], "rank": rank,
                "hostname": "FOREIGN" if mutate == "wrong_host" else host["hostname"],
                "cycles": [{"start_unix_ns": 1 if mutate == "old_timer" else now - 1, "stop_unix_ns": now}] * 2}
            rows.append(row)
            json_ref(root / "execution/ranks" / f"rank_{rank:02d}.json", row)
        json_ref(root / "summary.json", {"fixture_scope": FIXTURE, "placement":
            dispatch.PLACEMENTS[2] if mutate == "placement" else argv[5], "mode": argv[4], "run_id": argv[3], "ranks": rows,
            **({"evidence_scope": FIXTURE} if mutate == "fixture_output" else {})})
        if mutate == "raw_rank":
            path = root / "execution/ranks/rank_00.json"
            path.write_text(json.dumps({**rows[0], "fixture_tamper": "RAW_ONLY"}))
    monkeypatch.setattr(dispatch, "_run_owned", run)
    error = AssertionError if mutate == "placement" else ValueError
    with pytest.raises(error):
        dispatch.run_native(value, run_id="fixture", mode="correctness", placement=dispatch.PLACEMENTS[0], output_dir=tmp_path)
    assert not (Path(value["site"]["run_root"]) / "fixture/manifests/hardware_wan_launch.json").exists()


def test_owned_signal_requires_unchanged_birth_and_group(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dispatch.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    monkeypatch.setattr(dispatch, "_proc_identity", lambda pid: (200, 42, 42))
    monkeypatch.setattr(dispatch, "_proc_is_live", lambda pid: True)
    dispatch._signal_owned({42: {43: (100, 42, 42)}}, 15)
    assert calls == []
    dispatch._signal_owned({42: {43: (200, 42, 42)}}, 15)
    assert calls == [(42, 15)]
    dispatch._signal_owned({99: {43: (200, 42, 42)}}, 15)
    assert calls == [(42, 15)]


@pytest.mark.parametrize("exit_code", [0, 7])
def test_exited_parent_does_not_leave_a_live_owned_child(tmp_path, exit_code):
    # Real, bounded CPU subprocess regression: the leader is reaped before its
    # child exits. No shell/GPU command and no foreign process is involved.
    log = tmp_path / "cpu-fixture.log"
    program = ("import subprocess,sys; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "birth=Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]; "
        f"print(p.pid,birth,flush=True); sys.exit({exit_code})")
    try:
        with pytest.raises(ValueError, match="descendants|exited 7"):
            dispatch._run_owned([sys.executable, "-c", program], env=dict(dispatch.os.environ),
                cwd=tmp_path, log_path=log, timeout_seconds=10)
        child, birth = map(int, log.read_text().split())
        assert not dispatch._proc_is_live(child)
    finally:
        if log.exists() and len(log.read_text().split()) == 2:
            child, birth = map(int, log.read_text().split())
            # Emergency fixture teardown only if the bounded regression failed.
            identity = dispatch._proc_identity(child)
            if identity and identity[0] == birth and dispatch._proc_is_live(child):
                dispatch.os.kill(child, 9)


def _fixture_child_ready(read_fd, session, *, timeout_seconds):
    readable, _, _ = select.select([read_fd], [], [], timeout_seconds)
    assert readable, "CPU cleanup fixture did not report child readiness before startup deadline"
    fields = dispatch.os.read(read_fd, 128).split()
    assert len(fields) == 2, "CPU cleanup fixture did not report a child PID and birth"
    child, birth = map(int, fields)
    assert dispatch._proc_identity(child) == (birth, session, session)
    assert dispatch._proc_is_live(child)
    return child, birth


@pytest.mark.parametrize("startup_delay", [0, 0.4], ids=["nominal", "delayed-startup"])
def test_timeout_cleans_owned_session_without_signalling_another_session(tmp_path, monkeypatch, startup_delay):
    log = tmp_path / "cpu-timeout-fixture.log"
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    foreign_identity = dispatch._proc_identity(foreign.pid)
    read_fd, write_fd = dispatch.os.pipe()
    ready_children = []
    real_wait = subprocess.Popen.wait

    def wait_after_child_ready(proc, timeout=None):
        if timeout == 0.2:
            # Startup is separately bounded; the real 200 ms execution wait and
            # _run_owned's original finally/cleanup/reap path remain unchanged.
            ready_children.append(_fixture_child_ready(read_fd, proc.pid, timeout_seconds=10))
        return real_wait(proc, timeout=timeout)

    # Deliberately exceed the execution deadline before the child is spawned.
    delay = f"time.sleep({startup_delay}); " if startup_delay else ""
    program = ("import os,subprocess,sys,time; from pathlib import Path; " + delay +
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "birth=Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]; "
        "print(p.pid,birth,flush=True); "
        f"os.write({write_fd}, f'{{p.pid}} {{birth}}\\n'.encode()); time.sleep(30)")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(subprocess.Popen, "wait", wait_after_child_ready)
            with pytest.raises(subprocess.TimeoutExpired) as error:
                dispatch._run_owned([sys.executable, "-c", program], env=dict(dispatch.os.environ),
                    cwd=tmp_path, log_path=log, timeout_seconds=0.2, pass_fds=(write_fd,))
        assert error.value.timeout == 0.2
        child, birth = map(int, log.read_text().split())
        assert ready_children == [(child, birth)]
        assert not dispatch._proc_is_live(child)
        assert foreign.poll() is None and dispatch._proc_identity(foreign.pid) == foreign_identity
    finally:
        dispatch.os.close(read_fd)
        dispatch.os.close(write_fd)
        if dispatch._proc_identity(foreign.pid) == foreign_identity:
            foreign.terminate()
        foreign.wait(timeout=5)
        if log.exists() and len(log.read_text().split()) == 2:
            child, birth = map(int, log.read_text().split())
            identity = dispatch._proc_identity(child)
            if identity and identity[0] == birth and dispatch._proc_is_live(child):
                dispatch.os.kill(child, 9)


def test_fixture_readiness_deadline_still_cleans_owned_session(tmp_path, monkeypatch):
    read_fd, write_fd = dispatch.os.pipe()
    owned = []
    real_wait = subprocess.Popen.wait

    def wait_without_readiness(proc, timeout=None):
        if timeout == 0.2:
            identity = dispatch._proc_identity(proc.pid)
            assert identity is not None and identity[1:] == (proc.pid, proc.pid)
            owned.append((proc.pid, identity))
            _fixture_child_ready(read_fd, proc.pid, timeout_seconds=0.05)
        return real_wait(proc, timeout=timeout)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(subprocess.Popen, "wait", wait_without_readiness)
            with pytest.raises(AssertionError, match="child readiness before startup deadline"):
                dispatch._run_owned([sys.executable, "-c", "import time; time.sleep(30)"],
                    env=dict(dispatch.os.environ), cwd=tmp_path, log_path=tmp_path / "cpu-not-ready.log",
                    timeout_seconds=0.2, pass_fds=(write_fd,))
        assert len(owned) == 1
        pid, identity = owned[0]
        assert not dispatch._proc_is_live(pid)
        assert dispatch._owned_groups(pid, identity[0]) == {}
    finally:
        dispatch.os.close(read_fd)
        dispatch.os.close(write_fd)
        for pid, identity in owned:
            if dispatch._proc_identity(pid) == identity and dispatch._proc_is_live(pid):
                dispatch.os.kill(pid, 9)


def _planning_fixture(value, seconds):
    arms = dispatch.DEFAULT_PLACEMENTS
    operations = tuple(Operation(f"cycle{n}", n, "generator", OperationKind.GENERATOR_ROLLOUT,
        GradientMode.NO_GRAD, metadata={"cycle": n}) for n in (1, 2))
    graph = ExecutionGraph(operations, (Dependency("cycle1", "cycle2"),),
        {"initial_versions": {}, "cycle_ends": {"1": "cycle1", "2": "cycle2"}})
    rules = tuple(BindingRule(op.operation_id, {"placement": p}, p, (0,), ("gpu:0",), ())
                  for op in operations for p in arms)
    problem = PlanningProblem("wan13b_nogan_resident_best", value["binding"], graph,
        {"placement": arms}, rules, (), (), {0: 100}, tuple({"placement": p} for p in arms),
        evidence_scope=FIXTURE)
    costs = CostModel(value["binding"], tuple(OperationCost(p, (t,), 10, {"kind": FIXTURE})
                      for p, t in zip(arms, seconds)), scope=FIXTURE)
    return problem, costs


@pytest.mark.parametrize("seconds,winner", [((10, 20), 0), ((20, 10), 1)])
def test_search_uses_shared_planner_and_can_select_either_arm(tmp_path, fixture_adapter, seconds, winner):
    value = deployment(tmp_path)
    fixture_adapter.assemble_problem_and_costs = lambda *args, **kw: _planning_fixture(value, seconds)
    def lower(result, problem, costs, deployment, *, run_id):
        assert result["algorithm"] == "hierarchical"
        return {"fixture_scope": FIXTURE, "selected_placement": result["predicted_best"]["assignment"]["placement"],
                "plan_sha256": dispatch.fingerprint(result), "run_id": run_id}
    fixture_adapter.lower_selection = lower
    result = dispatch.search(value, [], output_dir=tmp_path / "search", run_id="fixture-selected")
    assert result["selected_placement"] == dispatch.DEFAULT_PLACEMENTS[winner]
    assert result["evidence_scope"] == FIXTURE and result["claimable_speedup"] is False
    assert result["predicted_cycle2_ns"] == min(seconds)


def test_native_timing_refuses_changed_receipt_and_wrong_correctness_arm(tmp_path):
    value = deployment(tmp_path)
    ref = json_ref(tmp_path / "correctness.json", {"placement": dispatch.PLACEMENTS[0],
        "mode": "correctness", "status": "PASS", "fixture_scope": FIXTURE})
    comp = json_ref(tmp_path / "comparison.json", comparison())
    with pytest.raises(ValueError, match="arm"):
        dispatch._native_argv(value, run_id="fixture", mode="timing", placement=dispatch.PLACEMENTS[2],
                              correctness=ref, comparison=comp)
    Path(ref["path"]).write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        dispatch._native_argv(value, run_id="fixture", mode="timing", placement=dispatch.PLACEMENTS[0],
                              correctness=ref, comparison=comp)


def test_real_adapter_rejects_fixture_execution_before_touching_native_host(tmp_path):
    value = deployment(tmp_path)
    assert dispatch.validate_deployment(value) == value
    with pytest.raises(ValueError, match="fixtures"):
        dispatch.run_native(value, run_id="fixture", mode="correctness", placement=dispatch.PLACEMENTS[0], output_dir=tmp_path)
    assert not Path(value["site"]["run_root"]).exists()


@pytest.mark.parametrize("placement", dispatch.PLACEMENTS)
def test_dispatcher_and_real_adapter_agree_on_native_raw_argv_environment(tmp_path, placement):
    from test_refactor_hardware_wan import fixture_observation
    value = deployment(tmp_path)
    run_root = Path(value["site"]["run_root"]) / "fixture-timing"
    manifests = run_root / "manifests"
    observation = fixture_observation(value, placement, run_root.name)
    correct, comp = observation["correctness"], observation["comparison"]
    outer = dispatch._native_argv(value, run_id=run_root.name, mode="timing", placement=placement,
                                   correctness=correct, comparison=comp)
    worker = hardware_wan._worker_argv(value, run_root, run_root.name, placement, correct, comp)
    environment = hardware_wan._environment(value, placement)
    (manifests / "command.sh").write_text(shlex.join(worker) + "\n")
    (manifests / "environment.txt").write_text("\n".join(f"{k}={v}" for k, v in sorted(environment.items())) + "\n")
    assert dispatch._native_manifests(value, run_root, outer) == (worker, environment)
    (manifests / "environment.txt").write_text("NCCL_ALGO=UNREGISTERED\n")
    with pytest.raises(ValueError, match="environment"):
        dispatch._native_manifests(value, run_root, outer)


def test_cli_preserves_session_identity_and_returns_errors(tmp_path, fixture_adapter, monkeypatch, capsys):
    value = deployment(tmp_path)
    dep = tmp_path / "deployment.json"
    dispatch.write_new_json(dep, value)
    session = tmp_path / "session.json"
    dispatch.write_new_json(session, {"deployment_sha256": "wrong", "evidence_scope": FIXTURE, "observations": []})
    assert cli.main(["search", "--deployment", str(dep), "--observations", str(session),
                     "--output", str(tmp_path / "out"), "--run-id", "fixture"]) == 2
    assert "another deployment" in capsys.readouterr().err
    monkeypatch.setattr(dispatch, "inspect_deployment", lambda value: {"status": "PASS_CPU_FIXTURE_ONLY", "gpu_execution": "NOT_RUN"})
    assert cli.main(["inspect", "--deployment", str(dep)]) == 0
    assert json.loads(capsys.readouterr().out)["gpu_execution"] == "NOT_RUN"


def test_real_native_host_probe_uses_only_metadata_and_matches_uuid(tmp_path, fixture_adapter, monkeypatch):
    value = deployment(tmp_path)
    commands = []
    def output(command, **kwargs):
        commands.append(command)
        if command[0] == value["site"]["python"]:
            assert "import torch" not in command[2]
            return json.dumps(value["environment"])
        return "\n".join(f"{r}, GPU-CPU-FIXTURE-{r}, NVIDIA H100 80GB HBM3, 81920" for r in range(8))
    monkeypatch.setattr(dispatch.subprocess, "check_output", output)
    observed = dispatch.observe_host(value)
    assert observed["devices"] == value["allocation"]["devices"]
    assert len(commands) == 2
    value["allocation"]["devices"][0]["uuid"] = "GPU-FOREIGN"
    with pytest.raises(ValueError, match="inventory"):
        dispatch.observe_host(value)


def test_metadata_probe_hashes_declared_torch_extension_without_importing_it(tmp_path):
    # A fake installed distribution is sufficient for metadata lookup. Importing
    # its Python module would fail, so this cannot accidentally initialize CUDA.
    root = tmp_path / "fixture_site_packages"
    package = root / "torch"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('CPU fixture forbids importing Torch')\n")
    extension = package / "_C.fixture.so"
    extension.write_bytes(b"CPU_FIXTURE_ONLY_NOT_A_SHARED_LIBRARY")
    metadata = root / "torch-0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 0.0\n")
    expected = {"python": __import__("platform").python_version(), "packages": {"torch": "0.0"},
        "ld_library_path": "", "torch_extension_origin": str(extension.resolve()),
        "torch_extension_sha256": dispatch.file_sha256(extension),
        "executable_sha256": dispatch.file_sha256(Path(sys.executable))}
    result = subprocess.check_output([sys.executable, "-c", dispatch._ENV_PROBE, json.dumps(expected)],
        text=True, env={**dispatch.os.environ, "PYTHONPATH": str(root), "LD_LIBRARY_PATH": ""})
    assert json.loads(result) == expected


def test_host_probe_requires_explicit_binding_of_effective_library_path(tmp_path, fixture_adapter, monkeypatch):
    value = deployment(tmp_path)
    value["environment"].pop("ld_library_path")
    with pytest.raises(ValueError, match="ld_library_path"):
        dispatch.observe_host(value)


def _mock_selected_fixture(tmp_path, monkeypatch, fixture_adapter):
    """Isolate selected-host plumbing; model/native execution are CPU doubles.

    Production fixture admission is tested independently above. These synthetic
    objects cannot pass the real saved-plan validator or execute a launcher.
    The actual observation proof below is the unmodified production validator.
    """
    value = deployment(tmp_path, scope="MEASURED")
    placement = dispatch.PLACEMENTS[0]
    correct = json_ref(tmp_path / "CPU_FIXTURE_correctness.json", {"fixture_scope": FIXTURE,
        "status": "PASS", "mode": "correctness", "placement": placement})
    comp = json_ref(tmp_path / "CPU_FIXTURE_comparison.json", comparison())
    calibrated = json_ref(tmp_path / "CPU_FIXTURE_calibration.json", {"fixture_scope": FIXTURE,
        "placement": placement, "correctness_binding": correct, "trajectory_comparison_binding": comp})
    plan = {"fixture_scope": FIXTURE, "deployment": value, "selected_placement": placement,
        "selected_observations": [{"placement": placement, "summary": calibrated}],
        "run_id": "CPU_FIXTURE_actual", "plan_sha256": "0" * 64,
        "model_plan": {"problem": {"fixture_scope": FIXTURE}, "cost_model": {"fixture_scope": FIXTURE}, "search_result": {}}}
    monkeypatch.setattr(dispatch, "PlanningProblem", SimpleNamespace(from_dict=lambda _: SimpleNamespace(
        binding=value["binding"], evidence_scope="MEASURED")))
    monkeypatch.setattr(dispatch, "CostModel", SimpleNamespace(from_dict=lambda _: SimpleNamespace(
        scope="MEASURED", bind=lambda *a, **k: None, verify_receipts=lambda: None)))
    monkeypatch.setattr(dispatch, "validate_saved_prediction", lambda *a: {"predicted_cycle2_ns": 20})
    fixture_adapter.validate_native_observation = hardware_wan.validate_native_observation
    return value, plan


@pytest.mark.parametrize("removed", ["rank_receipts", "launcher_health", "launcher_argv",
                                      "launcher_environment", "launch_binding"])
def test_selected_actual_cannot_drop_raw_proof(tmp_path, monkeypatch, fixture_adapter, removed):
    from test_refactor_hardware_wan import fixture_observation
    value, plan = _mock_selected_fixture(tmp_path, monkeypatch, fixture_adapter)
    def native(_, **kwargs):
        observation = fixture_observation(value, kwargs["placement"], kwargs["run_id"])
        path = Path(observation["summary"]["path"])
        raw = dispatch._object(path)
        raw.pop("evidence_scope")
        raw["fixture_scope"] = FIXTURE  # explicit CPU test double, never GPU data
        path.write_text(json.dumps(raw))
        observation["summary"] = dispatch._ref(path)
        observation.pop(removed)
        return observation
    out = tmp_path / "CPU_FIXTURE_selected"
    with pytest.raises(ValueError, match="unknown/missing native observation"):
        dispatch.run_selected(plan, output_dir=out, native_run=native)
    assert not (out / "closed_loop.json").exists()


@pytest.mark.parametrize("tamper", ["raw_rank", "rank_hash", "launch_source", "launch_env", "raw_env"])
def test_selected_actual_rechecks_raw_rank_and_launch_identity(tmp_path, monkeypatch, fixture_adapter, tamper):
    from test_refactor_hardware_wan import fixture_observation
    value, plan = _mock_selected_fixture(tmp_path, monkeypatch, fixture_adapter)
    def native(_, **kwargs):
        observation = fixture_observation(value, kwargs["placement"], kwargs["run_id"])
        path = Path(observation["summary"]["path"])
        raw = dispatch._object(path)
        raw.pop("evidence_scope")
        raw["fixture_scope"] = FIXTURE
        path.write_text(json.dumps(raw))
        observation["summary"] = dispatch._ref(path)
        if tamper in ("raw_rank", "rank_hash"):
            path = Path(observation["rank_receipts"][0]["path"])
            row = dispatch._object(path)
            row["fixture_tamper"] = "RAW_ONLY"
            path.write_text(json.dumps(row))
            if tamper == "raw_rank":
                observation["rank_receipts"][0] = dispatch._ref(path)
        elif tamper in ("launch_source", "launch_env"):
            path = Path(observation["launch_binding"]["path"])
            launch = dispatch._object(path)
            if tamper == "launch_source":
                launch["integration_head"] = "f" * 40
            else:
                launch["observed_environment"]["packages"]["torch"] = "CPU_FIXTURE_CHANGED"
            path.write_text(json.dumps(launch))
            observation["launch_binding"] = dispatch._ref(path)
        else:
            path = Path(observation["launcher_environment"]["path"])
            path.write_text(path.read_text() + "NCCL_ALGO=CPU_FIXTURE_CHANGED\n")
            observation["launcher_environment"] = dispatch._ref(path)
        return observation
    out = tmp_path / "CPU_FIXTURE_selected"
    message = {"raw_rank": "summary ranks differ", "rank_hash": "raw evidence hash changed",
        "launch_source": "launch.*binding mismatch", "launch_env": "observed_environment changed",
        "raw_env": "native environment differs"}[tamper]
    with pytest.raises(ValueError, match=message):
        dispatch.run_selected(plan, output_dir=out, native_run=native)
    assert not (out / "closed_loop.json").exists()


@pytest.mark.parametrize("target", ["correctness", "comparison", "nested_correctness"])
def test_measured_launcher_rejects_fixture_prerequisites_before_execution(tmp_path, target):
    value = deployment(tmp_path, scope="MEASURED")
    correct = {"status": "PASS", "mode": "correctness", "placement": dispatch.PLACEMENTS[0]}
    comp = comparison()
    if target == "correctness": correct["evidence_scope"] = FIXTURE
    elif target == "nested_correctness": correct["ranks"] = [{"fixture_scope": FIXTURE}]
    # Even an unmarked timing request cannot upgrade its explicitly synthetic gates.
    cref = json_ref(tmp_path/"CPU_FIXTURE_correctness.json", correct)
    pref = json_ref(tmp_path/"CPU_FIXTURE_comparison.json", comp)
    with pytest.raises(ValueError, match="CPU fixture"):
        dispatch._native_argv(value, run_id="must-not-launch", mode="timing", placement=dispatch.PLACEMENTS[0],
                              correctness=cref, comparison=pref)


def test_compare_native_rejects_fixture_input_before_subprocess(tmp_path, monkeypatch):
    from test_refactor_hardware_wan import fixture_correctness
    value = deployment(tmp_path, scope="MEASURED")
    _, ref = fixture_correctness(value, dispatch.PLACEMENTS[0], "CPU_FIXTURE_reference")
    _, cand = fixture_correctness(value, dispatch.PLACEMENTS[2], "CPU_FIXTURE_candidate")
    calls = []
    monkeypatch.setattr(dispatch, "validate_deployment", lambda v, **kw: hardware_wan.validate_deployment(v))
    monkeypatch.setattr(dispatch, "_run_owned", lambda *a, **kw: calls.append(kw))
    with pytest.raises(ValueError, match="CPU fixture"):
        dispatch.compare_native(value, ref, cand, output=tmp_path/"must-not-exist.json")
    assert not calls and not (tmp_path/"must-not-exist.json").exists()


def test_dispatch_and_adapter_share_numeric_comparison_gate():
    comp = comparison()
    comp["numeric_metrics"] = []
    with pytest.raises(ValueError, match="160 rollout"):
        dispatch._validate_comparison(comp)


@pytest.mark.parametrize("versions", ["535.183.06\n", "535.183.06\n535.183.06\n"])
def test_driver_probe_is_cpu_metadata_not_torch_import(tmp_path, versions):
    binary = tmp_path/"nvidia-smi"
    binary.write_text("#!/bin/sh\nprintf '%s' " + shlex.quote(versions) + "\n")
    binary.chmod(0o755)
    code = dispatch._ENV_PROBE + "\nassert 'torch' not in sys.modules\n"
    result = subprocess.run([sys.executable, "-c", code, json.dumps({"driver_version": "535.183.06"})],
        env={**dispatch.os.environ, "PATH": str(tmp_path)+":/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"driver_version": "535.183.06"}


@pytest.mark.parametrize("versions", ["", "535.183.06\n550.54.15\n"])
def test_driver_probe_refuses_missing_or_mixed_versions(tmp_path, versions):
    binary = tmp_path/"nvidia-smi"
    binary.write_text("#!/bin/sh\nprintf '%s' " + shlex.quote(versions) + "\n")
    binary.chmod(0o755)
    result = subprocess.run([sys.executable, "-c", dispatch._ENV_PROBE, json.dumps({"driver_version": "535.183.06"})],
        env={**dispatch.os.environ, "PATH": str(tmp_path)+":/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True, text=True)
    assert result.returncode != 0 and "one observed NVIDIA driver version" in result.stderr
