"""Execute the actual launchers' CPU-only lock sections in isolated paths.

These are filesystem lifecycle tests, not model/GPU correctness evidence.
"""
import fcntl
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/distillschedule"


def _section(launcher):
    text = (SCRIPTS / launcher).read_text()
    start = text.index('case "${DISTILLSCHEDULE_INHERITED_GPU_LOCKS:-0}" in')
    if launcher.startswith("launch_wan13b"):
        end = text.index('test ! -e "$RUN"', start)
    else:
        end = text.index('[[ -z "$(nvidia-smi', start)
    return text[start:end]


def _sandbox(tmp_path):
    project = tmp_path / "project"
    repo = tmp_path / "repo"
    (project / "locks").mkdir(parents=True)
    helper = repo / "scripts/distillschedule/verify_inherited_gpu_locks.py"
    helper.parent.mkdir(parents=True)
    node = tmp_path / "node.lock"
    # Only test resource paths are substituted; execute the real verifier.
    helper.write_text((SCRIPTS / helper.name).read_text().replace(
        "/tmp/distillschedule_gpu_node.lock", str(node)))
    paths = [project / "locks/helios_asset_transfer.lock", node,
             project / "locks/helios_world8_gpu.lock"]
    return project, repo, paths


def _acquire(paths):
    owned = []
    for path in paths:
        path.touch()
        fd = os.open(path, os.O_RDWR)
        high = fcntl.fcntl(fd, fcntl.F_DUPFD, 30)
        os.close(fd)
        fcntl.flock(high, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owned.append(high)
    return owned


def _command(launcher, project, repo, paths, flag="1", inherited=()):
    imports = "\n".join(f"exec {slot}>&{fd}" for slot, fd in zip((7, 8, 9), inherited))
    variables = (f'ROOT="{project}"; REPO="{repo}"; PYTHON="{sys.executable}"; '
                 f'DISTILLSCHEDULE_CAMPAIGN_ROOT="{project}"; '
                 f'LOCK_FILE="{paths[1]}"; HARDWARE_PROFILE=h100-8-b16; '
                 f'DISTILLSCHEDULE_INHERITED_GPU_LOCKS="{flag}"\n')
    if "livewan" in launcher:
        variables += f'ROOT="{repo}"\n'
    snippet = _section(launcher).replace("/tmp/distillschedule_gpu_node.lock", str(paths[1]))
    return "set -Eeuo pipefail\n" + imports + "\n" + variables + snippet


@pytest.mark.parametrize("launcher", [
    "launch_wan13b_hardware_two_cycle_node.sh",
    "launch_livewan_crossscale_world16_node.sh",
])
def test_actual_launcher_sections_preserve_all_parent_leases(tmp_path, launcher):
    project, repo, paths = _sandbox(tmp_path)
    owned = _acquire(paths)
    try:
        command = _command(launcher, project, repo, paths, inherited=owned)
        result = subprocess.run(["bash", "-c", command], pass_fds=owned,
                                text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        for path in paths:
            result = subprocess.run(["flock", "-n", "-E", "75", str(path), "true"])
            assert result.returncode == 75
    finally:
        for fd in owned:
            os.close(fd)


@pytest.mark.parametrize("launcher", [
    "launch_wan13b_hardware_two_cycle_node.sh",
    "launch_livewan_crossscale_world16_node.sh",
])
def test_actual_launcher_rejects_noninherited_or_invalid_flags(tmp_path, launcher):
    project, repo, paths = _sandbox(tmp_path)
    for path in paths:
        path.touch()
    for flag, expected in (("1", 75), ("2", 2)):
        command = _command(launcher, project, repo, paths, flag=flag)
        result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
        assert result.returncode == expected, result.stderr


def test_actual_livewan_source_unlock_does_not_unlock_parent_node(tmp_path):
    project, repo, paths = _sandbox(tmp_path)
    owned = _acquire(paths)
    try:
        text = (SCRIPTS / "launch_livewan_crossscale_world16_node.sh").read_text()
        start = text.index('if [[ "$LIVEWAN_SOURCE_PREPARE_FD" == 11 ]]; then')
        end = text.index('GENERATED_SOURCE_RECEIPT=', start)
        prepare = text[start:end]
        command = _command("launch_livewan_crossscale_world16_node.sh", project, repo,
                           paths, inherited=owned)
        command += f'SOURCE_ROOT="{tmp_path / "source"}"\n' + prepare
        command += 'flock -u "$LIVEWAN_SOURCE_PREPARE_FD"\n'
        command += 'test "$LIVEWAN_SOURCE_PREPARE_FD" == 11\n'
        result = subprocess.run(["bash", "-c", command], pass_fds=owned,
                                text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        for path in paths:
            assert subprocess.run(["flock", "-n", "-E", "75", str(path), "true"]).returncode == 75
        # The source preparation lock is independently and normally released.
        assert subprocess.run(["flock", "-n", str(tmp_path / "source.prepare.lock"), "true"]).returncode == 0
    finally:
        for fd in owned:
            os.close(fd)


@pytest.mark.parametrize("launcher", [
    "launch_wan13b_hardware_two_cycle_node.sh",
    "launch_livewan_crossscale_world16_node.sh",
])
def test_default_path_still_takes_its_original_lock(tmp_path, launcher):
    project, repo, paths = _sandbox(tmp_path)
    command = _command(launcher, project, repo, paths, flag="0")
    if "livewan" in launcher:
        command += 'test "$LIVEWAN_SOURCE_PREPARE_FD" == 8\n'
    command += 'test -z "${DISTILLSCHEDULE_INHERITED_GPU_LOCKS+x}"\n'
    result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_no_training_files_or_gpu_calls_in_lock_sections():
    for launcher in ("launch_wan13b_hardware_two_cycle_node.sh",
                     "launch_livewan_crossscale_world16_node.sh"):
        section = _section(launcher)
        assert "torch" not in section and "nvidia-smi" not in section
        assert "LOCK_UN" not in section and "flock -u" not in section
        assert "verify_inherited_gpu_locks.py" in section
