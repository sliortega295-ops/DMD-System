#!/usr/bin/env python3
"""Run native and DistillSchedule MiniMax sequentially on the same 16 GPUs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-id", required=True)
    value.add_argument("--expected-head", required=True)
    value.add_argument("--model-path", type=Path, required=True)
    value.add_argument("--prompt-cache", type=Path, required=True)
    value.add_argument("--python", dest="python_bin", type=Path, required=True)
    value.add_argument("--hosts", default="baai-helios-node0,baai-helios-node1")
    value.add_argument("--attention-backend", default="_flash_3_hub")
    value.add_argument("--seed", type=int, default=20260817)
    value.add_argument("--master-port", type=int, default=29980)
    value.add_argument("--remote-root", type=Path, default=Path("/share/project/shuchenweng/yongyan_liu/DMD-System_fsdp2_p0_20260815"))
    value.add_argument("--benchmark-source-root", type=Path)
    value.add_argument("--evidence-root", type=Path, default=Path("/home/lyy/Helios/research_artifacts/DMD-System/lightx2v_minimax_matched"))
    value.add_argument("--order", choices=("native-first", "distillschedule-first"), default="native-first")
    value.add_argument("--dry-run", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    root = Path(__file__).resolve().parents[2]
    controller = root / "scripts/distillschedule/run_lightx2v_minimax_matched_world16.py"
    comparator = root / "scripts/distillschedule/compare_lightx2v_minimax_matched_pair.py"
    order = ["native", "distillschedule"] if args.order == "native-first" else ["distillschedule", "native"]
    commands: list[list[str]] = []
    for offset, mode in enumerate(order):
        command = [
            sys.executable,
            str(controller),
            "--mode", mode,
            "--hosts", args.hosts,
            "--run-id", args.run_id,
            "--expected-head", args.expected_head,
            "--model-path", str(args.model_path),
            "--prompt-cache", str(args.prompt_cache),
            "--python", str(args.python_bin),
            "--attention-backend", args.attention_backend,
            "--seed", str(args.seed),
            "--master-port", str(args.master_port + offset),
            "--remote-root", str(args.remote_root),
            "--benchmark-source-root", str(args.benchmark_source_root or (args.remote_root / "artifacts/source/lightx2v-minimax-d034-benchmark")),
            "--local-evidence-root", str(args.evidence_root),
        ]
        if args.dry_run:
            command.append("--dry-run")
        commands.append(command)

    if args.dry_run:
        print(json.dumps({"order": order, "commands": commands}, indent=2))
        return 0

    for command in commands:
        subprocess.run(command, check=True)

    native = args.evidence_root / args.run_id / "native" / "summary.json"
    candidate = args.evidence_root / args.run_id / "distillschedule" / "summary.json"
    output = args.evidence_root / args.run_id / "pair.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(comparator),
            "--native", str(native),
            "--distillschedule", str(candidate),
            "--output", str(output),
        ],
        check=False,
    )
    if output.is_file():
        print(output.read_text())
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
