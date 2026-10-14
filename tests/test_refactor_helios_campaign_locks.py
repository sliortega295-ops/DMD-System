"""Real CPU flock probes, not GPU or training-correctness evidence."""
import fcntl
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/distillschedule/verify_inherited_gpu_locks.py"
SPEC = importlib.util.spec_from_file_location("campaign_lock_check", HELPER)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def held(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a")
    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return f


def test_verification_preserves_parent_lease(tmp_path, monkeypatch):
    # Use an isolated node path for this test; the production constant is fixed.
    node_path = tmp_path / "node.lock"
    project_path = tmp_path / "locks/helios_world8_gpu.lock"
    with held(project_path) as project, held(node_path) as node:
        original = module.Path
        monkeypatch.setattr(module, "Path", lambda p: node_path if p == "/tmp/distillschedule_gpu_node.lock" else original(p))
        module.verify_locks(tmp_path, project.fileno(), node.fileno())
        with project_path.open("a") as independent:
            with pytest.raises(BlockingIOError):
                fcntl.flock(independent.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The verifier never unlocked either lease.
        with node_path.open("a") as independent:
            with pytest.raises(BlockingIOError):
                fcntl.flock(independent.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_actual_two_children_preserve_lease_between_arms(tmp_path):
    project_path = tmp_path / "locks/helios_world8_gpu.lock"
    node_path = tmp_path / "isolated-node.lock"
    source = """
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location('check', sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
original = m.Path
m.Path = lambda p: pathlib.Path(sys.argv[3]) if p == '/tmp/distillschedule_gpu_node.lock' else original(p)
m.verify_locks(pathlib.Path(sys.argv[2]), int(sys.argv[4]), int(sys.argv[5]))
"""
    with held(project_path) as project, held(node_path) as node:
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, "-c", source, str(HELPER), str(tmp_path), str(node_path),
                 str(project.fileno()), str(node.fileno())],
                pass_fds=(project.fileno(), node.fileno()), capture_output=True, text=True,
            )
            assert result.returncode == 0, result.stderr
            for path in (project_path, node_path):
                with path.open("a") as contender:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_missing_descriptor_fails_closed(tmp_path):
    with pytest.raises(OSError):
        module.verify_locks(tmp_path, 1000000, 1000001)


def test_duplicate_descriptor_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        module.verify_locks(tmp_path, 9, 9)


def test_wrong_project_path_fails_closed(tmp_path):
    with held(tmp_path / "wrong.lock") as project:
        (tmp_path / "locks").mkdir()
        (tmp_path / "locks/helios_world8_gpu.lock").touch()
        with pytest.raises(ValueError, match="does not refer"):
            module.verify_locks(tmp_path, project.fileno(), 1000000)


def test_foreign_file_description_cannot_replace_owner(tmp_path):
    path = tmp_path / "locks/helios_world8_gpu.lock"
    with held(path), path.open("a") as wrong_description:
        with pytest.raises(BlockingIOError):
            module.verify_locks(tmp_path, wrong_description.fileno(), 1000000)


def test_pipe_descriptor_fails_closed(tmp_path):
    r, w = os.pipe()
    try:
        with pytest.raises(ValueError, match="regular file"):
            module.verify_locks(tmp_path, r, w)
    finally:
        os.close(r)
        os.close(w)


def test_actual_subprocess_rejects_noninherited_descriptors(tmp_path):
    result = subprocess.run(
        [sys.executable, str(HELPER), "--project-root", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 75
    assert "verification failed" in result.stderr


def test_launcher_default_still_acquires_locks_and_inheritance_is_launcher_only():
    text = (ROOT / "scripts/distillschedule/launch_helios_hardware_p0_node.sh").read_text()
    assert 'DISTILLSCHEDULE_INHERITED_GPU_LOCKS:-0' in text
    assert 'exec 9>"$PROJECT_ROOT/locks/helios_world8_gpu.lock"' in text
    assert 'exec 8>/tmp/distillschedule_gpu_node.lock' in text
    assert "verify_inherited_gpu_locks.py" in text
    assert text.index("verify_inherited_gpu_locks.py") < text.index("nvidia-smi --query-compute-apps")
    # Existing filtering removes this launcher-only option before model workers.
    assert "DISTILLSCHEDULE_*|NCCL_*|HELIOS_*" in text
