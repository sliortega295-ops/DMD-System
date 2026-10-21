#!/usr/bin/env python3
"""Build/install in a fresh venv and verify imports outside the source tree.

No training/GPU dependencies are installed. Build dependencies must already be
available; package installation uses --no-index and the locally built wheel.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    with tempfile.TemporaryDirectory(prefix="dmd-wheel-check-") as name:
        work = Path(name)
        wheels = work / "wheels"
        wheels.mkdir()
        subprocess.run([sys.executable, "-m", "pip", "wheel", str(root), "--no-deps",
                        "--no-build-isolation", "--wheel-dir", str(wheels)],
                       check=True, cwd=work, env=env)
        wheel_files = list(wheels.glob("distillschedule-*.whl"))
        if len(wheel_files) != 1:
            raise RuntimeError(f"expected exactly one wheel, found {wheel_files}")
        target = work / "venv"
        venv.EnvBuilder(with_pip=True).create(target)
        python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps",
                        "--force-reinstall", str(wheel_files[0])], check=True, cwd=work, env=env)
        probe = r'''
import json, sys, sysconfig
from pathlib import Path
import distillschedule, runtime, runtime.host_admission
from distillschedule.workloads.catalog import load_catalog
site = Path(sysconfig.get_paths()['purelib']).resolve()
for package in (distillschedule, runtime):
    assert Path(package.__file__).resolve().is_relative_to(site), package.__file__
assert [r['id'] for r in load_catalog()['workloads']] == ['W1','W2','W3','W4','W5','W6']
assert 'torch' not in sys.modules, 'discovery unexpectedly imported Torch'
print(json.dumps({'status':'PASS_ISOLATED_WHEEL','workloads':6,'gpu_execution':'NOT_RUN'}))
'''
        subprocess.run([str(python), "-I", "-c", probe], check=True, cwd=work, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
