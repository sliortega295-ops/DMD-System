"""Explicit deterministic policy for exact CUDA oracle/restart assertions.

This is a test policy, not a change to training defaults or performance claims.
The same policy must be active in every fresh distillctl subprocess.
"""
import json
import os
import subprocess
import sys

import pytest
import torch


@pytest.fixture
def cuda_bitwise_policy(monkeypatch):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def run_cli(*args):
    # Exercise the actual CLI in a new process, with no patched model/runtime.
    entry = ('import torch, runpy; torch.use_deterministic_algorithms(True); '
             'runpy.run_module("distillschedule", run_name="__main__", alter_sys=True)')
    result = subprocess.run([sys.executable, '-c', entry, *map(str, args)],
                            capture_output=True, text=True,
                            env={**os.environ, 'OMP_NUM_THREADS': '1',
                                 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)
