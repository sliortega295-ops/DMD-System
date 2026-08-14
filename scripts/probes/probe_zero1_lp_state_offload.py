#!/usr/bin/env python3
"""One-GPU correctness gate for the typed ZeRO-1 LP-state controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import deepspeed
import torch
from torch import nn

from runtime.deepspeed_phase_state import DeepSpeedZero1LPStateController


class TinyBF16Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256, bias=False),
            nn.GELU(),
            nn.Linear(256, 128, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    deepspeed.init_distributed(dist_backend="nccl")
    torch.manual_seed(7319)
    torch.cuda.manual_seed_all(7319)
    model = TinyBF16Model()
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "train_batch_size": 1,
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8},
        },
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {"device": "cpu", "pin_memory": False},
            "offload_param": {"device": "none"},
            "overlap_comm": False,
            "contiguous_gradients": False,
            "reduce_bucket_size": 16777216,
            "allgather_bucket_size": 16777216,
        },
        "steps_per_print": 1000000,
        "wall_clock_breakdown": False,
    }
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=config,
    )
    device = engine.device
    value = torch.randn(4, 128, device=device, dtype=torch.bfloat16)

    # Initialize the CPU Adam state and verify the post-step gradient boundary.
    loss = engine(value).float().square().mean()
    engine.backward(loss)
    engine.step()
    if any(parameter.grad is not None for parameter in engine.module.parameters()):
        raise RuntimeError("DeepSpeed step left a live parameter gradient")

    events = []
    controller = DeepSpeedZero1LPStateController(
        engine,
        logical_component="generator",
        event_sink=lambda event, **fields: events.append({"event": event, **fields}),
    )
    with torch.no_grad():
        output_before = engine(value).detach().clone()
    flats_before = [tensor.detach().cpu().clone() for tensor in engine.optimizer.bit16_groups_flat]
    hashes_before = [tensor_sha256(tensor) for tensor in flats_before]
    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated(device))

    offloaded = controller.offload()
    torch.cuda.synchronize()
    allocated_offloaded = int(torch.cuda.memory_allocated(device))
    if {tensor.device.type for tensor in engine.optimizer.bit16_groups_flat} != {"cpu"}:
        raise RuntimeError("LP flat state did not move to CPU")

    reloaded = controller.reload()
    torch.cuda.synchronize()
    allocated_reloaded = int(torch.cuda.memory_allocated(device))
    if {tensor.device.type for tensor in engine.optimizer.bit16_groups_flat} != {"cuda"}:
        raise RuntimeError("LP flat state did not return to CUDA")
    hashes_after = [tensor_sha256(tensor) for tensor in engine.optimizer.bit16_groups_flat]
    if hashes_after != hashes_before:
        raise RuntimeError("LP flat-state SHA changed across offload/reload")
    with torch.no_grad():
        output_after = engine(value).detach()
    torch.testing.assert_close(output_after, output_before, rtol=0, atol=0)

    # A subsequent optimizer update proves that the rebuilt parameter/optimizer
    # mappings are usable rather than merely readable.
    second_loss = engine(value).float().square().mean()
    engine.backward(second_loss)
    engine.step()
    controller.assert_balanced()

    payload = {
        "schema_version": "distillschedule.zero1_lp_state_cuda_gate.v1",
        "status": "PASS",
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "device": str(device),
        "zero_stage": int(engine.zero_optimization_stage()),
        "optimizer_cpu_offload": bool(engine.zero_offload_optimizer()),
        "lp_state": {
            "logical_bytes": offloaded.logical_bytes,
            "logical_mib": offloaded.logical_bytes / 2**20,
            "group_count": offloaded.group_count,
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "reload_logical_bytes": reloaded.logical_bytes,
        },
        "cuda_allocated_bytes": {
            "before": allocated_before,
            "offloaded": allocated_offloaded,
            "reloaded": allocated_reloaded,
            "released": allocated_before - allocated_offloaded,
        },
        "losses": {"first": float(loss.item()), "second": float(second_loss.item())},
        "events": events,
        "tests": {
            "no_live_grad_after_step": True,
            "lp_flat_cpu_after_offload": True,
            "lp_flat_cuda_after_reload": True,
            "flat_sha_exact": True,
            "forward_exact": True,
            "second_optimizer_step": True,
            "balanced_exit": True,
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
