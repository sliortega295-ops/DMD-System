#!/usr/bin/env python3
"""No-workload NCCL/FSDP/async-reducer admission canary for logical B16.

This tiny executable checks actual gradients, not a timing or model-capacity
result. Run with torchrun on the explicitly selected physical profile.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from distillschedule.runtime.hardware_profile import hardware_profile, MicrobatchCommitEngine
from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
from distillschedule.runtime.wan13b_fair import GradReadyAsyncResidentEngine


class MixedDtypeProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.ones(8, 8, dtype=torch.bfloat16) / 32,
                                       requires_grad=False)
        self.lora = torch.nn.Parameter(torch.ones(8, 8, dtype=torch.float32) / 64)
        self.seen_dtypes = []

    def forward(self, x):
        self.seen_dtypes.append((str(self.base.dtype), str(self.lora.dtype)))
        return torch.nn.functional.linear(x.bfloat16(), self.base).float() + torch.nn.functional.linear(x.float(), self.lora)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    profile = hardware_profile()
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == profile.physical_world_size
    device = torch.device("cuda", local)
    probe = torch.tensor(float(rank), device=device)
    dist.all_reduce(probe)
    assert probe.item() == sum(range(profile.physical_world_size))
    slots = profile.logical_slots(rank)
    # The CPU reference consumes all original logical B1 rows as one B16 mean.
    reference = MixedDtypeProbe()
    reference_loss = sum(reference(torch.ones(1, 8) * (i + 1)).square().mean()
                         for i in range(16)) / 16
    reference_loss.backward()
    expected = reference.lora.grad.clone()
    reference_optimizer = torch.optim.SGD([reference.lora], lr=0.001)
    reference_optimizer.step()
    rows = []
    for placement in ("fsdp", "resident-async"):
        model = MixedDtypeProbe().to(device)
        if placement == "fsdp":
            fully_shard(model, mesh=init_device_mesh("cuda", (profile.physical_world_size,)),
                        reshard_after_forward=True,
                        mp_policy=MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32,
                                                       output_dtype=None, cast_forward_inputs=False))
            model.set_force_sum_reduction_for_comms(True)
            model.set_gradient_divide_factor(float(profile.physical_world_size))
        optimizer = torch.optim.SGD([model.lora], lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        if placement == "fsdp":
            raw = FSDP2EngineAdapter(model, optimizer, scheduler, physical_step=2,
                                    gradient_clip=1e8)
        else:
            raw = GradReadyAsyncResidentEngine(model, optimizer, scheduler,
                                               physical_step=2, gradient_clip=1e8,
                                               async_bucket_bytes=128)
        engine = MicrobatchCommitEngine(raw, microbatch_count=profile.microbatch_count)
        for index, slot in enumerate(slots):
            with engine.microbatch(index):
                loss = engine(torch.ones(1, 8, device=device) * (slot + 1)).square().mean()
                engine.backward(loss)
                if index == len(slots) - 1:
                    if placement == "fsdp":
                        observed = model.lora.grad.full_tensor().detach().cpu()
                        torch.testing.assert_close(observed, expected, atol=1e-5, rtol=1e-6)
                engine.step()
        updated = model.lora.full_tensor() if placement == "fsdp" else model.lora
        torch.testing.assert_close(updated.detach().cpu(), reference.lora.detach(), atol=1e-6, rtol=1e-6)
        assert raw.global_steps == 3 and raw.step_calls == 1
        assert set(model.seen_dtypes) == {("torch.bfloat16", "torch.float32")}
        rows.append({"placement": placement, "status": "PASS", "commits": raw.step_calls,
                     "logical_slots": slots, "effective_global_batch": 16,
                     "compute_dtypes": model.seen_dtypes})
        del engine, raw, optimizer, scheduler, model, updated
        torch.cuda.empty_cache()
        dist.barrier()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"rank_{rank:02d}.json").write_text(json.dumps({
        "status": "PASS", "hardware_profile": profile.name, "rank": rank,
        "physical_world_size": dist.get_world_size(), "nccl": "PASS", "rows": rows,
        "workload_capacity": "NOT_RUN", "workload_timing": "NOT_RUN"}, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
