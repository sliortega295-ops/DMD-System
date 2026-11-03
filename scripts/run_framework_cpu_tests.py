#!/usr/bin/env python3
"""Run the fast core checks without loading model-specific dependencies."""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

CORE_TESTS = (
    "test_core_ci.py",
    "test_data_pipeline.py",
    "test_open_workload_program.py",
    "test_refactor_config.py",
    "test_refactor_protocol.py",
    "test_training_checkpoint_policy.py",
    "test_training_cli.py",
    "test_training_config.py",
    "test_training_end_to_end.py",
    "test_training_model_catalog.py",
    "test_training_packaging.py",
    "test_training_plugins.py",
    "test_training_resume_compatibility.py",
)


def test_files(root):
    root = Path(root).resolve()
    paths = [Path("tests") / name for name in CORE_TESTS]
    missing = [str(path) for path in paths if not (root / path).is_file()]
    if missing:
        raise ValueError(f"missing core tests: {missing}")
    return [str(path) for path in paths]


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "pytest", "-q", *test_files(root),
               *(sys.argv[1:] if argv is None else argv)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.run(command, cwd=root, check=False, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
