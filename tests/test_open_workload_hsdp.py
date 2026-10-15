"""Four real Gloo processes; two logical CPU nodes, NOT GPU performance.

The workload has no strategy, topology, planner imports or known model name.
Topology is a launcher/resource fact. All Adam states are checked against an
independent concatenated-global-batch execution, including warmed moments.
"""
from datetime import timedelta
import json
import os
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.sharded import HSDP_MODES
from distillschedule.open_workload.torch_backend import Session, candidates, compare
from test_open_workload_multirole_replica import build_workload

pytestmark = pytest.mark.skipif(tuple(int(x) for x in torch.__version__.split('.')[:2]) < (2,10),
                               reason='actual CPU HSDP2 requires Torch >=2.10')


def setup(rank, rendezvous):
    torch.set_num_threads(1)
    os.environ.update(GROUP_RANK=str(rank//2), LOCAL_RANK=str(rank%2), LOCAL_WORLD_SIZE='2')
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank,
                            world_size=4, timeout=timedelta(seconds=75))


def worker(rank, rendezvous, output, frequency, width):
    setup(rank, rendezvous)
    try:
        from distillschedule.open_workload.execution import ExecutionBackend
        from torch.distributed.tensor import DTensor, Replicate, Shard
        w = build_workload(frequency, width, 3, rank=rank)
        backend = ExecutionBackend(w)
        inventory = backend.inventory(7)
        options = [o for o in inventory['candidates'] if o['reduction'] in HSDP_MODES]
        assert len(options) == 2
        assert inventory['topology']['logical_nodes_on_one_host']
        warm = backend.session(inventory['candidates'][0]); warm.run()
        w.state = warm.state
        global_w = build_workload(frequency, width, 3)
        global_w.feeds = {'input': torch.cat([global_w.feeds['input'] + i*.2 for i in range(4)])}
        global_w.state = warm.state
        ref = Session(global_w, candidates(global_w, limit=1)['candidates'][0])
        expected = ref.result_state(ref.run())['state']
        effects = []
        for option in options:
            s = backend.session(option)
            state = s.result_state(s.run())['state']
            gate = compare(expected, state, rtol=4e-5, atol=4e-7)
            assert gate['status'] == 'PASS', gate
            assert s.effects['optimizer_moments_migrated'] == 4
            assert s.effects['mesh_shape'] == [2, 2]
            for key in ('role_a','role_b'):
                for p in s.state[key].parameters():
                    assert isinstance(p, DTensor)
                    assert p.placements == (Replicate(), Shard(0))
                    assert p.to_local().numel()*2 == p.numel()
            effects.append(s.effects)
        Path(output, f'rank-{rank}.json').write_text(json.dumps(effects))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('frequency,width', [(1,4), (3,8)])
def test_real_hsdp_preserves_global_objective_and_initialized_adam(tmp_path, frequency, width):
    mp.spawn(worker, args=(str(tmp_path/'rdzv'), str(tmp_path), frequency, width), nprocs=4, join=True)
    assert all(len(json.loads((tmp_path/f'rank-{r}.json').read_text())) == 2 for r in range(4))


def planner_worker(rank, rendezvous, output):
    setup(rank, rendezvous)
    try:
        from distillschedule.open_workload.execution import ExecutionBackend
        from distillschedule.open_workload.planning import calibrate, plan, run
        from distillschedule.workloads.best_known.registry import fingerprint
        w = build_workload(1,4,3,rank=rank)
        backend = ExecutionBackend(w); root = Path(output)
        manifest = calibrate(w, root/'cal', samples=1, max_candidates=7, max_program_runs=8)
        assert set(HSDP_MODES) <= {v['implementation']['reduction'] for v in manifest['variants']}
        selected = backend.select_plan(lambda: plan(root/'cal/calibration.json', memory_limit_bytes=10**7))
        backend.publish(root/'plan.json', selected)
        result,_,_ = run(w, root/'plan.json', root/'run')
        assert result['status'] == 'PASS_SELECTED_PROGRAM'
        # Replay a genuinely measured HSDP candidate without claiming that the
        # CPU timing-selected winner necessarily used HSDP.
        selected['selected'] = next(c for c in selected['search']['ranked']
                                    if c['implementation']['reduction'] == 'hsdp2_reshard')
        selected.pop('plan_sha256'); selected['plan_sha256'] = fingerprint(selected)
        backend.publish(root/'hsdp-plan.json',selected)
        result,_,_ = run(w,root/'hsdp-plan.json',root/'hsdp-run')
        assert result['status'] == 'PASS_SELECTED_PROGRAM'
        assert all(row['effects']['mesh_shape'] == [2,2] for row in result['rank_executions'])
    finally:
        dist.destroy_process_group()


def test_hsdp_is_measured_planned_and_replayed_through_public_backend(tmp_path):
    mp.spawn(planner_worker, args=(str(tmp_path/'rdzv'),str(tmp_path/'auto')), nprocs=4, join=True)
    result = json.loads((tmp_path/'auto/hsdp-run/execution.json').read_text())
    assert result['selected_implementation']['node_groups'] == [[0,1],[2,3]]
