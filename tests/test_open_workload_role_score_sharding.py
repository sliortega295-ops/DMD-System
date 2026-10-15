"""Frozen Score FSDP2 is a real role-pool capability, not a mock backend.

Only CPU/Gloo execution and independently computed global update parity are
claimed here; A100/H100 NCCL / production video capacity are NOT_RUN.
"""
from datetime import timedelta
from pathlib import Path
import json

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillschedule.open_workload.role_program import (
    AllocatedWorld, PoolLayout, RoleTask, enumerate_layouts,
)
from distillschedule.open_workload.role_transport import RolePoolExecutor, RoleFunctions
from test_open_workload_role_transport import (
    make_model, make_opt, sample, generate, guidance, score_train, independent_reference,
)


def make_frozen_score(role, device):
    model = make_model(role, device)
    if role == 'score':
        model.requires_grad_(False)
    return model


def _worker(rank, rendezvous, output, generator_fsdp):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank,
                            world_size=4, timeout=timedelta(seconds=70))
    runner = None
    try:
        task = RoleTask(4, (2, 4), score_updates=0)
        layout = PoolLayout((0,2), (1,3), 2, 'resident',
                            'fsdp2_keep_forward' if generator_fsdp else 'replicated',
                            'fsdp2_frozen')
        hardware = AllocatedWorld((2**26,)*4, (0,0,1,1))
        bindings = RoleFunctions(make_frozen_score,make_opt,sample,generate,guidance,
                                 score_parameters_frozen=True)
        runner = RolePoolExecutor(task, layout, hardware, bindings, device='cpu')
        receipt, module, optimizer = runner.run()
        from torch.distributed.tensor import DTensor
        if runner.role == 'score':
            assert optimizer is None
            assert all(not p.requires_grad for p in module.parameters())
            assert all(isinstance(p,DTensor) for p in module.parameters())
            assert receipt['ranks'][rank]['sharding']['role']=='score_frozen_sharded'
        elif generator_fsdp:
            assert any(isinstance(p, DTensor) for p in module.parameters())
        actual = {key:(value.full_tensor() if isinstance(value, DTensor) else value).detach().cpu()
                  for key,value in module.state_dict().items()}
        reference = independent_reference(task)[runner.role]
        assert all(torch.allclose(v,reference[key],rtol=4e-5,atol=4e-6)
                   for key,v in actual.items())
        assert receipt['status']=='PASS_ROLE_POOL_EXECUTION'
        assert receipt['claimable_speedup'] is False
        Path(output,f'{rank}.json').write_text(json.dumps({'ok':True}))
    finally:
        if runner is not None: runner.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('generator_fsdp',[False,True])
def test_frozen_score_fsdp2_preserves_g_score_input_vjp(tmp_path,generator_fsdp):
    mp.spawn(_worker,args=(str(tmp_path/'rdzv'),str(tmp_path),generator_fsdp),nprocs=4,join=True)
    assert all(json.loads((tmp_path/f'{rank}.json').read_text())['ok'] for rank in range(4))


def test_score_pool_capability_matches_frozen_fact_and_balanced_collectives():
    task=RoleTask(4,(2,4),score_updates=0)
    hardware=AllocatedWorld((2**26,)*4,(0,0,1,1))
    variants=enumerate_layouts(task,hardware,generator_bytes=3000,score_bytes=3000,
                               max_candidates=8,allow_frozen_score_fsdp=True)['candidates']
    assert any(v.score_parallelism=='fsdp2_frozen' for v in variants)
    assert all(v.validate(task,hardware) is v for v in variants)
    assert all(v.score_parallelism=='replicated' for v in enumerate_layouts(
        task,hardware,generator_bytes=3000,score_bytes=3000,max_candidates=8)['candidates'])
    with pytest.raises(ValueError, match='balanced paired calls'):
        PoolLayout((0,), (1,2,3), 1, 'resident', 'replicated','fsdp2_frozen').validate(task,hardware)
    with pytest.raises(ValueError,match='frozen Score'):
        RoleFunctions(make_model,make_opt,sample,generate,guidance,score_train,
                      score_parameters_frozen=True).validate(RoleTask(4,(2,4),score_updates=1))


def _planned_worker(rank,rendezvous,output):
    from distillschedule.open_workload.role_planning import calibrate,search,run_selected
    from test_open_workload_role_continuous import cycle_data,reference
    from distillschedule.workloads.best_known.registry import write_new_json
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,
                            world_size=4,timeout=timedelta(seconds=75))
    try:
        task=RoleTask(4,(2,4),score_updates=0)
        hardware=AllocatedWorld((2**26,)*4,(0,0,1,1))
        bindings=RoleFunctions(make_frozen_score,make_opt,sample,generate,guidance,
                               data_identity='e'*64,sample_at=cycle_data,
                               score_parameters_frozen=True)
        root=Path(output)
        calibration=calibrate(task,hardware,bindings,root/'cal',generator_bytes=3000,
                              score_bytes=3000,max_candidates=2,samples=1,
                              max_native_runs=3)
        assert any(c['layout']['score_parallelism']=='fsdp2_frozen'
                   for c in calibration['candidates'])
        if rank==0:
            planned=search(root/'cal/calibration.json')
            assert any(x['implementation']['score_parallelism']=='fsdp2_frozen'
                       for x in planned['search']['ranked'])
            write_new_json(root/'plan.json',planned)
        dist.barrier()
        receipt,model,optimizer=run_selected(task,hardware,bindings,root/'plan.json',
                                             root/'train',cycles=2)
        assert receipt['final_versions']['generator']==2
        reference_model,_=reference(task,2)['generator' if rank in receipt['layout']['generator'] else 'score']
        from torch.distributed.tensor import DTensor
        actual={k:(v.full_tensor() if isinstance(v,DTensor) else v).detach().cpu()
                for k,v in model.state_dict().items()}
        expected=reference_model.state_dict()
        assert all(torch.allclose(v,expected[k],rtol=3e-5,atol=3e-6)
                   for k,v in actual.items())
        if rank in receipt['layout']['score']:
            assert optimizer is None
    finally:
        dist.destroy_process_group()


def test_calibration_and_selected_multicycle_for_frozen_score_fsdp2(tmp_path):
    mp.spawn(_planned_worker,args=(str(tmp_path/'rdzv'),str(tmp_path)),nprocs=4,join=True)


class MixedFrozenScore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=torch.nn.Linear(4,2).to(torch.bfloat16)
        self.norm=torch.nn.LayerNorm(2)
        self.requires_grad_(False)

    def forward(self, latent):
        return self.norm(self.backbone(latent.to(torch.bfloat16)).float())


def mixed_frozen_factory(role,device):
    if role == 'generator': return make_model(role,device)
    with torch.random.fork_rng():
        torch.manual_seed(97)
        score=MixedFrozenScore()
    return score.to(device)


def _mixed_worker(rank,rendezvous,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,
                            world_size=4,timeout=timedelta(seconds=70))
    runner=None
    try:
        task=RoleTask(4,(2,4),score_updates=0)
        hw=AllocatedWorld((2**26,)*4,(0,0,1,1))
        layout=PoolLayout((0,2),(1,3),2,'resident','replicated','fsdp2_frozen')
        fn=RoleFunctions(mixed_frozen_factory,make_opt,sample,generate,guidance,
                         score_parameters_frozen=True)
        runner=RolePoolExecutor(task,layout,hw,fn,device='cpu')
        receipt,model,opt=runner.run()
        if runner.role=='score':
            assert opt is None
            assert set(p.dtype for p in model.parameters())=={torch.bfloat16,torch.float32}
            assert receipt['ranks'][rank]['sharding']['dtype_groups']==2
        ref_g=mixed_frozen_factory('generator','cpu');ref_s=mixed_frozen_factory('score','cpu')
        ref_opt=make_opt('generator',ref_g)
        for i in range(task.microbatches):
            result=generate(ref_g,sample(i,'generator','cpu'))
            remote=result.detach().clone().requires_grad_(True)
            vjp=torch.autograd.grad(guidance(ref_s,remote,sample(i,'score','cpu')),remote)[0]
            result.backward(vjp/task.microbatches)
        ref_opt.step()
        expected=(ref_g if runner.role=='generator' else ref_s).state_dict()
        from torch.distributed.tensor import DTensor
        actual={key:(v.full_tensor() if isinstance(v,DTensor) else v).detach().cpu()
                for key,v in model.state_dict().items()}
        assert all(torch.allclose(v,expected[k].detach().cpu(),rtol=8e-4,atol=8e-5)
                   for k,v in actual.items())
        Path(output,f'mixed-{rank}.json').write_text(json.dumps({'ok':True}))
    finally:
        if runner is not None:runner.close()
        dist.destroy_process_group()


def test_bf16_frozen_score_and_fp32_norm_keep_precision_in_score_pool(tmp_path):
    mp.spawn(_mixed_worker,args=(str(tmp_path/'rdzv'),str(tmp_path)),nprocs=4,join=True)
    assert all(json.loads((tmp_path/f'mixed-{rank}.json').read_text())['ok'] for rank in range(4))


class UnsupportedFrozenScore(torch.nn.Module):
    """Mixed direct-root and leaf parameters: unsafe for current FSDP owners."""
    def __init__(self):
        super().__init__()
        self.backbone=torch.nn.Linear(4,2).to(torch.bfloat16)
        self.root_scale=torch.nn.Parameter(torch.ones(2,dtype=torch.float32),requires_grad=False)
        self.requires_grad_(False)

    def forward(self,latent):
        return self.backbone(latent.to(torch.bfloat16)).float()*self.root_scale


def unsupported_frozen_factory(role,device):
    if role=='generator':return make_model(role,device)
    with torch.random.fork_rng():
        torch.manual_seed(97)
        model=UnsupportedFrozenScore()
    return model.to(device)


def _rejection_worker(rank,rdzv,out):
    from distillschedule.open_workload.role_planning import calibrate,search
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,world_size=4,
                            timeout=timedelta(seconds=70))
    try:
        task=RoleTask(4,(2,4),score_updates=0)
        hw=AllocatedWorld((2**26,)*4,(0,0,1,1))
        f=RoleFunctions(unsupported_frozen_factory,make_opt,sample,generate,guidance,
                        data_identity='f'*64,score_parameters_frozen=True)
        record=calibrate(task,hw,f,Path(out)/'cal',generator_bytes=3000,score_bytes=3000,
                         max_candidates=2,max_native_runs=3)
        assert record['budget']['calibration_runs']==2
        assert len(record['candidates'])==1
        assert record['candidates'][0]['layout']['score_parallelism']=='replicated'
        rejected=[x for x in record['candidate_inventory']['rejected']
                  if x['reason']=='NATIVE_CAPABILITY_REJECTED']
        assert len(rejected)==1 and rejected[0]['layout']['score_parallelism']=='fsdp2_frozen'
        assert 'mixed dtype frozen Score requires' in rejected[0]['detail']
        if rank==0:
            chosen=search(Path(out)/'cal/calibration.json')['selected']
            assert chosen['implementation']['score_parallelism']=='replicated'
    finally:
        dist.destroy_process_group()


def test_incompatible_score_fsdp_is_pruned_without_distributed_hang(tmp_path):
    mp.spawn(_rejection_worker,args=(str(tmp_path/'rdzv'),str(tmp_path)),nprocs=4,join=True)
