#!/usr/bin/env python3
"""One owned-GPU construction-only check, never a world16 timing result.

Caller must hold the node lock through audit, invocation and process teardown.
Exact Student state, Critic LoRA and optimizers are kept alive before the frozen
14B Teacher placement is attempted. No forward or optimizer update is executed.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--bootstrap", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("LIVEWAN_CAPACITY_LOCK_HELD") != "1":
        raise RuntimeError("caller must hold the node GPU lock")
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.repo_root / "src"))
    from distillschedule.runtime.livewan_crossscale_worker import (
        Bootstrap, FormalTrace, _wan_model, _load_student_state, _load_checkpoint,
        materialize_adam_state, configure_python_sources, validate_source_receipt,
        memory_snapshot, host_memory_snapshot,
    )
    raw = json.loads(args.bootstrap.read_text())
    trace = FormalTrace.load(raw["trace"]["path"], expected_cycles=2)
    bootstrap = Bootstrap.load(args.bootstrap, trace=trace)
    receipt = validate_source_receipt(bootstrap, repo_root=args.repo_root)
    configure_python_sources(repo_root=args.repo_root, source_receipt=receipt)
    import torch
    from safetensors.torch import load_file
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wanstreamer.lora import inject_lora
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    events = Path("/sys/fs/cgroup/memory.events")
    result = {"run_id": args.output.parent.name, "scope": "SINGLE_GPU_CONSTRUCTION_ONLY",
        "hostname": socket.gethostname(), "code_head": subprocess.check_output(
            ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True).strip(),
        "trace_sha256": trace.sha256, "bootstrap_sha256": bootstrap.sha256,
        "source_receipt_sha256": receipt["source_receipt_sha256"],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "world16_training": "NOT_RUN", "timing": "NOT_RUN", "snapshots": [],
        "cgroup_before": events.read_text()}
    start = time.monotonic()
    role = "before_construction"
    def snap(name):
        row = {"stage": name, "elapsed_seconds": time.monotonic()-start,
               **memory_snapshot(torch, device), **host_memory_snapshot()}
        result["snapshots"].append(row)
        print(json.dumps(row), flush=True)
    student = critic = teacher = opt_g = opt_c = None
    try:
        snap(role)
        role = "student_gpu_with_full_adam"
        torch.manual_seed(0)
        student = _wan_model(WanModel, size="1.3B", config=WAN_CONFIGS["t2v-1.3B"])
        if raw.get("student_state"):
            result["student_state"] = _load_student_state(torch, student, bootstrap)
        else:
            student.load_state_dict(_load_checkpoint(load_file, Path(raw["student_base"]["root"])))
        student.to(device).train()
        opt_g = torch.optim.AdamW(student.parameters(), lr=2e-6, betas=(0., .99),
                                 weight_decay=0., eps=1e-8, foreach=True)
        materialize_adam_state(torch, opt_g, step=int(raw["optimizer"]["generator_step"]))
        gc.collect(); snap(role)
        role = "critic_gpu_with_lora_and_adam"
        torch.manual_seed(0)
        critic = _wan_model(WanModel, size="1.3B", config=WAN_CONFIGS["t2v-1.3B"]).to(torch.bfloat16)
        critic.load_state_dict(_load_checkpoint(load_file, Path(raw["critic_base"]["root"])))
        critic.eval().requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(raw["critic_lora"]["seed"]))
            params, modules, numel = inject_lora(critic, rank=32)
        result["critic_lora"] = {"rank": 32, "modules": modules, "numel": numel}
        critic.to(device)
        opt_c = torch.optim.AdamW(params, lr=4e-5, betas=(0., .99),
                                 weight_decay=0., eps=1e-8, foreach=True)
        materialize_adam_state(torch, opt_c, step=int(raw["optimizer"]["critic_step"]))
        gc.collect(); snap(role)
        role = "teacher_cpu_load"
        torch.manual_seed(0)
        teacher = _wan_model(WanModel, size="14B", config=WAN_CONFIGS["t2v-1.3B"]).to(torch.bfloat16)
        teacher.load_state_dict(_load_checkpoint(load_file, Path(raw["teacher_base"]["root"])))
        teacher.eval().requires_grad_(False)
        gc.collect(); snap(role)
        role = "teacher_full_cuda_materialization"
        teacher.to(device)
        torch.cuda.synchronize(device); snap(role)
        result["status"] = "CONSTRUCTION_PASS_NO_TRAINING"
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="FAIL_CUDA_OOM", failure_stage=role,
                      error=str(exc), traceback=traceback.format_exc())
        snap("oom_" + role)
    except Exception as exc:
        result.update(status="FAIL_RUNTIME_ERROR", failure_stage=role,
                      error=str(exc), traceback=traceback.format_exc())
    finally:
        result["construction_wall_seconds"] = time.monotonic()-start
        result["cgroup_after"] = events.read_text()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k:v for k,v in result.items() if k not in ("traceback", "snapshots")}), flush=True)
    return 0 if result["status"] in ("FAIL_CUDA_OOM", "CONSTRUCTION_PASS_NO_TRAINING") else 1


if __name__ == "__main__":
    raise SystemExit(main())
