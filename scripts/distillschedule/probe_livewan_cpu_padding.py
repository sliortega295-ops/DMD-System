#!/usr/bin/env python3
"""Low-cost mechanism probes, not real-model correctness or performance."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("cpu-adam", "fsdp-padding"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("LIVEWAN_CAPACITY_LOCK_HELD") != "1":
        raise RuntimeError("mechanism probe requires owned node lock")
    import torch
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.manual_seed(73)
    result = {"scope": "LOW_COST_TOY_MECHANISM_ONLY", "mode": args.mode,
              "real_model_parity": "NOT_RUN", "formal_timing": "NOT_RUN"}
    if args.mode == "cpu-adam":
        from distillschedule.runtime.livewan_cpu_optimizer import CPUAdamW
        params = [torch.nn.Parameter(torch.randn(31, 37, device=device)),
                  torch.nn.Parameter(torch.randn(17, device=device))]
        controls = [torch.nn.Parameter(p.detach().clone()) for p in params]
        kw = dict(lr=2e-6, betas=(0., .99), eps=1e-8, weight_decay=0., foreach=True)
        candidate = CPUAdamW(torch, params, **kw)
        candidate.materialize_state()
        control = torch.optim.AdamW(controls, **kw)
        maximum = {"parameters": 0., "exp_avg": 0., "exp_avg_sq": 0.}
        for step in range(6):
            for a, b in zip(params, controls):
                gradient = torch.randn_like(a)
                a.grad = gradient.clone(); b.grad = gradient.clone()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            torch.nn.utils.clip_grad_norm_(controls, 1.)
            candidate.step(); control.step()
            torch.cuda.synchronize(device)
            for a, b in zip(params, controls):
                maximum["parameters"] = max(maximum["parameters"], (a-b).abs().max().item())
                for key in ("exp_avg", "exp_avg_sq"):
                    maximum[key] = max(maximum[key], (candidate.state[a][key]-
                        control.state[b][key].cpu()).abs().max().item())
            candidate.zero_grad(); control.zero_grad()
        if any(v > 1e-6 for v in maximum.values()):
            raise RuntimeError(f"CPU Adam formula smoke failed: {maximum}")
        result.update(status="PASS", maximum_absolute_errors=maximum,
                      optimizer=candidate.receipt())
        rank = 0
    else:
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        from distillschedule.runtime.livewan_collective_padding import StudentRolloutPadding
        from distillschedule.runtime.livewan_crossscale_worker import CallInventory
        dist.init_process_group("nccl", device_id=device)
        rank = dist.get_rank()
        if dist.get_world_size() != 2:
            raise RuntimeError("bounded padding mechanism canary must be world2")
        model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.SiLU(),
                                    torch.nn.Linear(16, 16)).to(device)
        reference = copy.deepcopy(model)
        mesh = init_device_mesh("cuda", (2,))
        for layer in model:
            if list(layer.parameters()):
                fully_shard(layer, mesh=mesh, reshard_after_forward=True)
                layer.set_force_sum_reduction_for_comms(True)
                layer.set_gradient_divide_factor(2.)
        z = torch.ones(2, 16, device=device) * (rank+1)
        blocks = 4 if rank == 0 else 6
        row = SimpleNamespace(logical_step=0, new_blocks=blocks,
                             operation_key=f"step=0/rank={rank}")
        padding = StudentRolloutPadding(torch, [row, SimpleNamespace(logical_step=0, new_blocks=6)])
        def forward(m, x, *a, **kw): return m(x)
        bc = SimpleNamespace(block_forward=forward)
        streamer = SimpleNamespace(schedule=[1., .75, .5, .25], noisy_context=False,
            model=model, n_frames=21, n_world=21-blocks*3,
            cache=SimpleNamespace(num_tokens=21, evictions=0), ctx=None,
            ctx_lens=None, rope=None, dtype=torch.float32, time_scale=1000.)
        recs = [[{"z": z, "block_start": i*3}]] * blocks
        inventory = CallInventory()
        with inventory.install(torch=torch, blockcausal=bc, roles={id(model): "student"}):
            with torch.no_grad():
                for _ in range(1+5*blocks): output = bc.block_forward(model, z)
                expected = reference(z)
            assert torch.equal(output, expected)
            rng_before = torch.cuda.get_rng_state(device).clone()
            padding.pad(blockcausal=bc, streamer=streamer, row=row, recs=recs)
            assert torch.equal(rng_before, torch.cuda.get_rng_state(device))
            for _ in range(3):
                loss = bc.block_forward(model, z, grad_checkpoint=True).square().mean()
                loss.backward()
        assert inventory.counts["student/nograd/plain"] == 1+5*blocks
        assert inventory.counts["student/grad/checkpoint"] == 3
        assert inventory.padding_counts.get("student/nograd/plain", 0) == 5*(6-blocks)
        result.update(status="PASS", rank=rank, application_inventory=inventory.counts,
            padding_inventory=inventory.padding_counts, padding=padding.receipt(),
            forward_bitwise_equal=True, rng_unchanged=True, gradient_finite=True)
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad.to_local()).all()
        dist.barrier()
        dist.destroy_process_group()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.mode}.rank{rank}.json"
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
