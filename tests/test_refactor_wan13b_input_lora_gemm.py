"""CPU admission tests; real primitive and model parity are separate GPU gates."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command():
    return [sys.executable, str(ROOT / "scripts/distillschedule/run_wan13b_fair_world16.py"),
            "--run-id", "input-gemm-plan", "--mode", "trajectory", "--placement",
            "resident-async128-tree-chunk2048", "--expected-head", "a" * 40,
            "--model-root", "/model", "--bootstrap", "/bootstrap", "--bootstrap-hash",
            "b" * 64, "--master-addr", "127.0.0.1", "--trajectory-cycles", "2",
            "--trajectory-arm-id", "input-gemm", "--generator-nograd-lora-fused-epilogue",
            "--tf32-cublas-override", "1", "--dry-run"]


def run(args):
    return subprocess.run(args, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
                          text=True, capture_output=True)


def test_default_off_and_explicit_export():
    for enabled in (False, True):
        args = command() + ["--generator-rollout-cuda-graph"]
        if enabled:
            args += ["--fused-input-lora-a"]
        result = run(args)
        assert result.returncode == 0, result.stderr
        for row in json.loads(result.stdout)["commands"]:
            assert f"DISTILLSCHEDULE_WAN13B_FUSED_INPUT_LORA_A={int(enabled)}" in " ".join(row)


def test_fails_closed_without_graph_or_matched_tf32():
    args = command() + ["--fused-input-lora-a"]
    assert "requires" in run(args).stderr
    args += ["--generator-rollout-cuda-graph"]
    args[args.index("--tf32-cublas-override") + 1] = "0"
    assert "requires" in run(args).stderr
