#!/usr/bin/env python3
"""CPU-only eight-rank test of real role-pool anchors and DTensor sketches."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor
from distillschedule.evidence.trajectory_state_sketch import collect_numeric_state_sketch, compare_numeric_state_sketches
from distillschedule.runtime.livewan_crossscale_worker import _bind_trajectory_optimizer_parameter_names, _trajectory_state_anchor


def main():
    output = Path(os.environ['CANARY_OUTPUT'])
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    assert dist.get_world_size() == 8
    assert not torch.cuda.is_initialized()
    roles = ((0, 1, 2, 3), (4, 5, 6, 7))
    def model_and_optimizer(offset):
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad(): model.weight.copy_(torch.arange(4).reshape(2, 2) * .1 + offset)
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        binding = _bind_trajectory_optimizer_parameter_names(named_parameters=list(model.named_parameters()), optimizer=opt)
        model.weight.square().sum().backward()
        opt.step()
        return model, opt, binding
    student, gopt, gnames = model_and_optimizer(1.) if rank in roles[0] else (None, None, {})
    critic, copt, cnames = model_and_optimizer(2.) if rank in roles[1] else (None, None, {})
    bundle = SimpleNamespace(metadata={'generator_ranks': roles[0], 'critic_ranks': roles[1]},
        student=student, critic=critic, critic_lora_params=list(critic.parameters()) if critic else [],
        generator_optimizer=gopt, critic_optimizer=copt,
        generator_optimizer_parameter_names=gnames, critic_optimizer_parameter_names=cnames)
    digest, artifact = _trajectory_state_anchor(torch=torch, rank=rank, cycle=1, output_dir=output, bundle=bundle)
    anchor = json.loads(Path(artifact['path']).read_text())
    for name, value in anchor['components'].items():
        owners = roles[0] if name.startswith('generator') else roles[1]
        assert value['owner_ranks'] == list(owners)
        assert value['tensor_count'] > 0
        assert all(row['contributor_ranks'] == list(owners) for row in value['tensors'])
    reference_tensor = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    reference = collect_numeric_state_sketch(torch=torch, named_values=[('weight', reference_tensor, None)])
    for owners in roles:
        mesh = DeviceMesh('cpu', list(owners))
        tensor = distribute_tensor(reference_tensor, mesh, [Shard(0)]) if rank in owners else None
        candidate = collect_numeric_state_sketch(torch=torch, named_values=[('weight', tensor, None)] if tensor is not None else [],
            primary_rank=min(owners), owner_ranks=owners)
        comparison = compare_numeric_state_sketches(reference, candidate, cycle=1, component='sharded_role')
        assert comparison.normalized_l2 == 0.0
    negative = 'NOT_RUN'
    try:
        collect_numeric_state_sketch(torch=torch,
            named_values=[('weight', reference_tensor, None)] if rank in (0, 1, 2) else [],
            primary_rank=0, owner_ranks=roles[0])
    except AssertionError as error:
        assert 'inventory differs' in str(error)
        negative = 'PASS_REJECTED_MISSING_DECLARED_OWNER'
    assert negative != 'NOT_RUN'
    assert not torch.cuda.is_initialized()
    (output / f'rank_{rank:03d}.json').write_text(json.dumps({
        'status': 'PASS', 'rank': rank, 'world_size': 8, 'backend': 'gloo',
        'actual_anchor': artifact, 'four_components': list(digest),
        'dtensor_uneven_shards': 'PASS_BOTH_FOUR_RANK_ROLE_POOLS',
        'missing_owner_negative': negative, 'cuda_context_created': False,
        'claim_boundary': 'CPU evidence collector integration only; no model capacity/timing/parity'}, indent=2)+'\n')
    dist.barrier()
    dist.destroy_process_group()
    print(f'CPU_OWNER_CANARY rank={rank} PASS', flush=True)

if __name__ == '__main__': main()
