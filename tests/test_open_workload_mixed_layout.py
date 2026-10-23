"""Real different layouts for different roots of an unseen callable program."""
from datetime import timedelta
from pathlib import Path
import json
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.mixed_layout import enumerate_assignments, assignment_count
from distillschedule.open_workload.torch_backend import Session, candidates, compare
from test_open_workload_multirole_replica import build_workload


def test_mixed_domains_are_bounded_complete_and_have_no_model_names():
    domains = {'unseen_a':('post_backward','ddp_overlap','fsdp2_reshard'),
               'unseen_b':('post_backward','ddp_overlap')}
    rows=list(enumerate_assignments(domains))
    assert len(rows)==assignment_count(domains)==4
    assert len({tuple(r.items()) for r in rows})==4
    assert not list(enumerate_assignments({'single':('post_backward','ddp_overlap')}))


def test_existing_constraints_only_apply_to_wrapped_roots():
    from distillschedule.open_workload.replica_contract import execution_graph
    w=build_workload(2,4,3)
    all_graph=execution_graph(w.program,w.replica_semantics,'ddp_overlap')
    a_graph=execution_graph(w.program,w.replica_semantics,'ddp_overlap',{'role_a'})
    assert len(a_graph.dependencies)<=len(all_graph.dependencies)
    assert a_graph.metadata['initial_versions']==all_graph.metadata['initial_versions']


def worker(rank,rendezvous,output,frequency):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=70))
    try:
        from distillschedule.open_workload.execution import ExecutionBackend
        from torch.distributed.tensor import DTensor
        from torch.nn.parallel import DistributedDataParallel
        w=build_workload(frequency,6,4,rank=rank);backend=ExecutionBackend(w)
        warm=backend.session(backend.inventory(1)['candidates'][0]);warm.run();w.state=warm.state
        global_w=build_workload(frequency,6,4,global_batch=True);global_w.state=warm.state
        ref=Session(global_w,candidates(global_w,limit=1)['candidates'][0]);expected=ref.result_state(ref.run())['state']
        inventory=backend.inventory(24)
        selected=[]
        for option in inventory['candidates']:
            if option['reduction']!='per_module': continue
            mapping=option['module_layouts']
            if ('fsdp2_reshard' in mapping.values() or 'fsdp2_keep_forward' in mapping.values()):
                selected.append(option)
        assert any(set(x['module_layouts'].values())=={'post_backward','fsdp2_reshard'} for x in selected)
        assert any(set(x['module_layouts'].values())=={'ddp_overlap','fsdp2_reshard'} for x in selected)
        for option in selected:
            s=backend.session(option);actual=s.result_state(s.run())['state']
            gate=compare(expected,actual,rtol=4e-5,atol=4e-7)
            assert gate['status']=='PASS',gate
            assert s.effects['optimizer_moments_migrated']==2*sum(
                m.startswith('fsdp2_') for m in option['module_layouts'].values())
            for key,mode in option['module_layouts'].items():
                module=s.state[key]
                assert isinstance(module,DistributedDataParallel)==(mode=='ddp_overlap')
                assert all(isinstance(p,DTensor)==mode.startswith('fsdp2_') for p in module.parameters())
        Path(output,f'{rank}.json').write_text(json.dumps({'variants':len(selected)}))
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('frequency',[1,3])
def test_native_mixed_layouts_match_global_batch_and_preserve_both_adams(tmp_path,frequency):
    mp.spawn(worker,args=(str(tmp_path/'rdzv'),str(tmp_path),frequency),nprocs=2,join=True)
    assert json.loads((tmp_path/'0.json').read_text())['variants']>=8


def planner_worker(rank,rendezvous,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=70))
    try:
        from distillschedule.open_workload.execution import ExecutionBackend
        from distillschedule.open_workload.planning import calibrate,plan,run
        from distillschedule.workloads.best_known.registry import fingerprint
        w=build_workload(1,4,3,rank=rank);backend=ExecutionBackend(w);root=Path(output)
        report=calibrate(w,root/'cal',samples=1,max_candidates=9,max_program_runs=10)
        assert any(v['implementation']['reduction']=='per_module' for v in report['variants'])
        result=backend.select_plan(lambda:plan(root/'cal/calibration.json',memory_limit_bytes=10**7))
        backend.publish(root/'plan.json',result)
        assert run(w,root/'plan.json',root/'run')[0]['status']=='PASS_SELECTED_PROGRAM'
        # Verify lowering of a measured nonuniform plan without faking an automatic win.
        result['selected']=next(r for r in result['search']['ranked']
                                if r['implementation']['reduction']=='per_module'
                                and any(m.startswith('fsdp2_') for m in r['implementation']['module_layouts'].values()))
        result.pop('plan_sha256');result['plan_sha256']=fingerprint(result)
        backend.publish(root/'mixed.json',result)
        executed,_,_=run(w,root/'mixed.json',root/'mixed-run')
        assert executed['status']=='PASS_SELECTED_PROGRAM'
        assert all(len(set(r['effects']['module_layouts'].values()))>1 for r in executed['rank_executions'])
    finally:dist.destroy_process_group()


def test_measured_per_module_assignment_reaches_common_planner_and_runtime(tmp_path):
    mp.spawn(planner_worker,args=(str(tmp_path/'rdzv'),str(tmp_path/'work')),nprocs=2,join=True)
