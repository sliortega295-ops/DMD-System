#!/usr/bin/env python3
"""Run the common CPU lane with explicit, tested dependency-lane selection.

Pytest --ignore only prunes recursive discovery; shell-expanded explicit file
arguments can still run. Filter the actual argv before starting pytest instead.
Provider-specific full-class suites have their own zero-skip acceptance jobs.
"""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

EXCLUDED = frozenset({
    "test_training_livewan_reference_inputs.py",
    "test_training_livewan_public_cpu.py",
    "test_training_livewan_inference_cpu.py",
    # Actual Diffusers 0.40 Grid output and installed DeepSpeed serialization
    # are mandatory in their own dependency-separated H3 jobs.
    "test_training_h3_grid_outputs.py",
    "test_training_h3_grid_native_outputs.py",
    "test_training_h3_native_consolidation.py",
    # This actual original transformer import needs the full 0.37 model
    # closure, not the 0.34 source-helper environment used here.
    "test_training_wan_generator_import_upstream_cpu.py",
})
PATTERNS = ("test_data_*.py", "test_training_*.py", "test_open_workload_*.py",
            "test_framework_*.py", "test_runtime_*.py", "test_cpu_receipt_gate.py",
            "test_evidence_livewan_c51_phases.py",
            "test_registry_atomic_json.py",
            "test_refactor_h3_attention_capture_entrypoint.py")


def test_files(root):
    root = Path(root).resolve()
    paths = {path for pattern in PATTERNS for path in (root / "tests").glob(pattern)
             if path.is_file() and path.name not in EXCLUDED}
    if not paths:
        raise ValueError("no common CPU tests found")
    return [str(path.relative_to(root)) for path in sorted(paths)]


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "pytest", "-q", *test_files(root),
               *(sys.argv[1:] if argv is None else argv)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return subprocess.run(command, cwd=root, check=False, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
