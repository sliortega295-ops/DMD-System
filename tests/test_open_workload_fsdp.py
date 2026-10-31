"""Actual CPU FSDP2 on an unknown model, including already-initialized Adam."""
from datetime import timedelta
from dataclasses import replace
from pathlib import Path
import json
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.sharded import ShardedLifecycle,MODES
from distillschedule.open_workload.torch_backend import Session,candidates,compare
from test_open_workload_multirole_replica import build_workload

pytestmark=pytest.mark.skipif(tuple(int(x) for x in torch.__version__.split('.')[:2])<(2,10),
                             reason='CPU FSDP2 validation uses Torch >=2.10; not a CUDA pass')


def worker(rank,rendezvous,output,frequency):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        from distillschedule.open_workload.replicated import make_session,inventory
        w=build_workload(frequency,4,3,rank=rank)
        # First train once using averaged replicated gradients, then keep Adam
        # moments for the next round. FSDP must redistribute these exact moments.
        warm=make_session(w,inventory(w,limit=1)['candidates'][0]);warm.run()
        w.state=warm.state
        global_w=build_workload(frequency,4,3,global_batch=True)
        global_w.state=warm.state
        ref=Session(global_w,candidates(global_w,limit=1)['candidates'][0])
        expected=ref.result_state(ref.run())['state']
        options=candidates(w,limit=16)['candidates']
        chosen=[options[0],next(o for o in options if not o['share_immutable'] and len(o['checkpoint_bindings'])==3)]
        for mode in MODES:
            for local in chosen:
                impl={'backend':'torch-callable-replicated-v1','local':local,'reduction':mode,'bucket_cap_mb':0}
                s=make_session(w,impl)
                result=s.run();state=s.result_state(result)
                checked=compare(expected,state['state'],rtol=4e-5,atol=4e-7)
                assert checked['status']=='PASS',checked
                from torch.distributed.tensor import DTensor
                assert all(isinstance(p,DTensor) for key in ('role_a','role_b') for p in s.state[key].parameters())
                assert s.effects['optimizer_moments_migrated']==4
                assert s.effects['fsdp_sharded_parameters']==6
                assert all(p.to_local().numel()<p.numel() for key in ('role_a','role_b') for p in s.state[key].parameters())
        Path(output,f'rank{rank}.json').write_text(json.dumps({'variants':4,'gpu_execution':'NOT_RUN'}))
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('frequency',[1,3])
def test_real_fsdp2_preserves_multirole_adam_and_input_gradients(tmp_path,frequency):
    mp.spawn(worker,args=(str(tmp_path/'rdzv'),str(tmp_path),frequency),nprocs=2,join=True)
    assert json.loads((tmp_path/'rank0.json').read_text())['variants']==4


def auto_worker(rank,rendezvous,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        from distillschedule.open_workload.execution import ExecutionBackend
        from distillschedule.open_workload.planning import calibrate,plan,run
        from distillschedule.workloads.best_known.registry import fingerprint
        w=build_workload(1,4,3,rank=rank);backend=ExecutionBackend(w);root=Path(output)
        manifest=calibrate(w,root/'cal',samples=1,max_candidates=5,max_program_runs=6)
        assert {v['implementation']['reduction'] for v in manifest['variants']}=={
            'post_backward','ddp_overlap',*MODES}
        selected=backend.select_plan(lambda:plan(root/'cal/calibration.json',memory_limit_bytes=10**7))
        backend.publish(root/'plan.json',selected)
        result,_,_=run(w,root/'plan.json',root/'run')
        assert result['status']=='PASS_SELECTED_PROGRAM'
        # Independent execution of a measured sharded candidate verifies lowering
        # even when actual observed CPU costs prefer replication on this input.
        candidate=next(x for x in selected['search']['ranked'] if x['implementation']['reduction']=='fsdp2_reshard')
        selected['selected']=candidate
        selected.pop('plan_sha256');selected['plan_sha256']=fingerprint(selected)
        backend.publish(root/'shard-plan.json',selected)
        result,_,_=run(w,root/'shard-plan.json',root/'shard-run')
        assert result['status']=='PASS_SELECTED_PROGRAM'
        assert all(r['effects']['fsdp_sharded_parameters']>0 for r in result['rank_executions'])
    finally:dist.destroy_process_group()


def test_shared_auto_search_includes_actual_fsdp2_layout(tmp_path):
    mp.spawn(auto_worker,args=(str(tmp_path/'rdzv'),str(tmp_path/'auto')),nprocs=2,join=True)
    manifest=json.loads((tmp_path/'auto/cal/calibration.json').read_text())
    assert len(manifest['variants'])==5


class MethodFlowBlock(torch.nn.Module):
    """Unregistered model math; rollout deliberately calls denoise twice."""
    def __init__(self, width):
        super().__init__()
        self.base = torch.nn.Linear(width, width).requires_grad_(False)
        self.adapter = torch.nn.Linear(width, width, bias=False)

    def denoise(self, value):
        from torch.distributed.tensor import DTensor
        assert all(not isinstance(p, DTensor) for p in self.parameters()), 'method was not unsharded'
        return torch.sin(self.base(value) + self.adapter(value))

    def rollout(self, value):
        first = self.denoise(value)
        second = self.denoise(first)
        # This use after the inner call catches prematurely nested post-hooks.
        return second + .1 * self.adapter(first)

    def forward(self, value):
        return self.rollout(value)


def method_workload(rank=0, *, global_batch=False, method='rollout'):
    from test_open_workload_execution import build
    from distillschedule.open_workload.replica_contract import ReplicaSemantics, module_method_binding
    w = build(n=2)
    with torch.random.fork_rng():
        model = MethodFlowBlock(4)
    model.load_state_dict(w.state['learner'].state_dict())
    w.state['learner'] = model
    w.state['adam'] = torch.optim.Adam(model.adapter.parameters(), lr=.001)
    # Initialize real moments identically on every rank, before native sharding.
    model(w.feeds['x']).square().mean().backward()
    w.state['adam'].step(); w.state['adam'].zero_grad(set_to_none=True)
    if not global_batch: w.feeds = {'x': w.feeds['x'][rank*4:(rank+1)*4].clone()}
    w.bindings = {**w.bindings, 'forward': module_method_binding('learner', method),
                  'second': module_method_binding('learner', method)}
    w.program = replace(w.program, calls=tuple(
        replace(c, binding='second') if c.id == 'p1' else c for c in w.program.calls),
        outputs=('p0', 'p1', 'l0', 'l1'))
    w.replica_semantics = ReplicaSemantics({'forward':'learner', 'second':'learner'}, ('b0', 'b1'),
        8 if global_batch else 4, forward_interface='registered_methods',
        forward_methods={'forward':method, 'second':method})
    return w.validate()


def observe_method_calls(session, *, sharded=None):
    """Observe actual completion boundaries without substituting any native hook."""
    from distillschedule.open_workload.torch_backend import full_tensor
    from torch.distributed.tensor import DTensor
    gradients = []
    lifecycle = session.lifecycle
    if lifecycle is None:
        class Observer:
            def before_run(self, *args): pass
            def before_call(self, *args): pass
            def after_call(self, *args): pass
            def finish(self, *args): pass
        lifecycle = session.lifecycle = Observer()
    native_after = lifecycle.after_call
    def after(current, call):
        native_after(current, call)
        model = current.state['learner']
        if call.id in ('p0', 'p1') and sharded is not None:
            assert all(isinstance(p, DTensor) == sharded for p in model.parameters())
        if call.id in ('b0', 'b1'):
            if sharded is not None:
                assert all(isinstance(p, DTensor) for p in model.parameters())
            gradients.append({name: full_tensor(p.grad).detach().clone() for name, p in model.named_parameters()
                              if p.requires_grad})
    lifecycle.after_call = after
    return gradients


def method_cycles(session, start, stop, *, expected=None):
    from distillschedule.open_workload.execution import ExecutionBackend
    from distillschedule.training.checkpoint import pack_state
    result = []
    initial = session.workload
    feed = initial.feeds['x'].clone()
    for cycle in range(start, stop):
        if cycle != start or session._ran:
            current = replace(initial, state=session.state, feeds={'x': feed + cycle*.05},
                program=replace(initial.program, initial_versions=session.receipt['dispatcher']['final_versions']))
            session.bind_cycle(current, graph=ExecutionBackend(current).graph(session.lifecycle.option)
                               if hasattr(session.lifecycle, 'option') else current.program.compile())
        elif cycle:
            session.feeds = {'x': feed + cycle*.05}
        outputs = session.run()
        actual = session.result_state(outputs)
        if expected is not None:
            gate = compare(expected[cycle]['state'], actual['state'], rtol=4e-5, atol=4e-7)
            assert gate['status'] == 'PASS', gate
            for name in ('p0', 'p1'):
                ref = expected[cycle]['outputs'][name][dist.get_rank()*4:(dist.get_rank()+1)*4]
                torch.testing.assert_close(actual['outputs'][name], ref, rtol=4e-5, atol=4e-7)
            for name in ('l0', 'l1'):
                loss = actual['outputs'][name].clone()
                dist.all_reduce(loss); loss.div_(dist.get_world_size())
                torch.testing.assert_close(loss, expected[cycle]['outputs'][name], rtol=4e-5, atol=4e-7)
        result.append({'actual': actual, 'packed': pack_state(session.state),
                       'versions': session.receipt['dispatcher']['final_versions']})
    return result


def registered_method_worker(rank, rendezvous, output, mode, method, stage, checkpointed=False):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE='2',
                      GROUP_RANK='0')
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        from distillschedule.open_workload.execution import ExecutionBackend
        from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint, pack_state, rng_state
        from distillschedule.open_workload.replica_contract import module_method_binding
        from test_training_end_to_end import equal
        from torch.distributed.tensor import DTensor
        root = Path(output)
        local = candidates(method_workload(rank, method=method), limit=1)['candidates'][0]
        if checkpointed:
            local = {**local, 'checkpoint_bindings': ['forward', 'second']}
        implementation = dict(backend='torch-callable-replicated-v1', local=local,
                              reduction=mode, bucket_cap_mb=0)
        identity = dict(fixture='registered-method-cpu2', implementation=implementation, method=method)
        if stage == 'write':
            global_w = method_workload(global_batch=True, method=method)
            reference = Session(global_w, local, copy_state=False, persistent=True)
            ref_grads = observe_method_calls(reference)
            expected = [row['actual'] for row in method_cycles(reference, 0, 3)]
            for length in (3, 1):
                w = method_workload(rank, method=method); backend = ExecutionBackend(w)
                assert backend.graph(implementation).metadata['execution_constraint'] == 'fsdp_forward_backward_context_v1'
                session = backend.session(implementation, copy_state=False, persistent=True)
                observed = observe_method_calls(session, sharded=mode.endswith('_reshard'))
                model, optimizer = session.state['learner'], session.state['adam']
                assert session.state['learner'] is w.state['learner']
                assert optimizer.param_groups[0]['params'][0] is model.adapter.weight
                assert isinstance(model.adapter.weight, DTensor)
                assert model.adapter.weight.to_local().numel() < model.adapter.weight.numel()
                assert session.effects['optimizer_moments_migrated'] == 2
                assert session.effects['fsdp_forward_methods'] == {'learner':method}
                if method == 'rollout': assert model.denoise.__func__ is MethodFlowBlock.denoise
                assert getattr(model, method).__wrapped__.__func__ is getattr(MethodFlowBlock, method)
                rows = method_cycles(session, 0, length, expected=expected)
                if checkpointed:
                    assert session.effects['checkpoint_calls'] == 2 * length
                    assert session.effects['recompute_calls'] >= 2 * length
                gate = compare(ref_grads[:2*length], observed, rtol=4e-5, atol=4e-7)
                assert gate['status'] == 'PASS', gate
                assert session.effects['lifecycle_installations'] == 1
                assert optimizer.param_groups[0]['params'][0] is model.adapter.weight
                assert all(p.grad is None for p in session.state['guide'].parameters())
                if length == 3:
                    torch.save(rows, root / f'whole-{rank}.pt')
                else:
                    save_checkpoint(root / 'checkpoint', session.state, identity=identity, cycle=1, cursor=1,
                        versions=session.receipt['dispatcher']['final_versions'])
                    # Persistent rebinds cannot sneak in an unhooked entry or
                    # regress to arbitrary private methods after preparation.
                    before = pack_state(session.state)
                    for interface in ('registered_methods', 'private_methods'):
                        changed = replace(w, state=session.state,
                            program=replace(w.program, initial_versions=session.receipt['dispatcher']['final_versions']),
                            bindings={**w.bindings, 'forward':module_method_binding('learner','forward'),
                                      'second':module_method_binding('learner','forward')},
                            replica_semantics=replace(w.replica_semantics, forward_interface=interface,
                                forward_methods={'forward':'forward','second':'forward'}
                                if interface == 'registered_methods' else {}))
                        session.bind_cycle(changed)
                        with pytest.raises(ValueError, match='installed FSDP method contract changed'):
                            session.run()
                        assert equal(before, pack_state(session.state))
            # Neither an arbitrary private wrapper nor DDP can exploit the new capability.
            bad = method_workload(rank, method=method)
            with pytest.raises(ValueError, match='forward interface'):
                ExecutionBackend(bad).session({**implementation, 'reduction':'ddp_overlap', 'bucket_cap_mb':1})
        else:
            w = method_workload(rank, method=method)
            session = ExecutionBackend(w).session(implementation, copy_state=False, persistent=True)
            payload = restore_checkpoint(root / 'checkpoint', session.state, identity=identity)
            assert payload['cycle'] == payload['cursor'] == 1
            assert equal(payload['rng'], rng_state())
            restored = replace(w, state=session.state,
                program=replace(w.program, initial_versions=payload['versions']))
            session.bind_cycle(restored, graph=ExecutionBackend(restored).graph(implementation))
            saved = torch.load(root / f'whole-{rank}.pt', weights_only=True)
            assert equal(saved[0]['packed'], pack_state(session.state))
            rows = method_cycles(session, 1, 3)
            for expected, actual in zip(saved[1:], rows):
                assert equal(expected, actual)
            assert session.effects['lifecycle_installations'] == 1
            Path(root, f'restored-{rank}.json').write_text(json.dumps({'cycles':3, 'fresh_process':True,
                'gpu_execution':'NOT_RUN', 'forward_methods':session.effects['fsdp_forward_methods']}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('method', ['rollout', 'denoise'])
def test_registered_fsdp_method_updates_and_fresh_process_checkpoint(tmp_path, mode, method):
    for stage in ('write', 'restore'):
        mp.spawn(registered_method_worker,
                 args=(str(tmp_path / ('rdzv-'+stage)), str(tmp_path), mode, method, stage), nprocs=2, join=True)
    for rank in range(2):
        assert json.loads((tmp_path / f'restored-{rank}.json').read_text())['fresh_process']


@pytest.mark.parametrize('mode', MODES)
def test_registered_fsdp_method_recomputation_and_fresh_process_checkpoint(tmp_path, mode):
    for stage in ('write', 'restore'):
        mp.spawn(registered_method_worker,
                 args=(str(tmp_path / ('rdzv-'+stage)), str(tmp_path), mode, 'rollout', stage, True),
                 nprocs=2, join=True)
    for rank in range(2):
        assert json.loads((tmp_path / f'restored-{rank}.json').read_text())['fresh_process']


def asymmetric_method_worker(rank, rendezvous, output):
    from distillschedule.open_workload.execution import ExecutionBackend
    from distillschedule.open_workload.replica_contract import module_method_binding
    from distillschedule.training.checkpoint import pack_state
    from test_training_end_to_end import equal
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=10))
    try:
        w = method_workload(rank)
        local = candidates(w, limit=1)['candidates'][0]
        option = dict(backend='torch-callable-replicated-v1', local=local,
                      reduction='fsdp2_reshard', bucket_cap_mb=0)
        session = ExecutionBackend(w).session(option, copy_state=False, persistent=True)
        current = replace(w, state=session.state)
        if rank == 0:
            current = replace(current,
                bindings={**w.bindings, 'forward': module_method_binding('learner', 'forward'),
                          'second': module_method_binding('learner', 'forward')},
                replica_semantics=replace(w.replica_semantics,
                    forward_methods={'forward': 'forward', 'second': 'forward'}))
        session.bind_cycle(current)
        before = pack_state(session.state)
        with pytest.raises(ValueError, match='FSDP cycle admission.*installed FSDP method contract changed'):
            session.run()
        assert equal(before, pack_state(session.state))
        assert session.profile == {} and session.lifecycle.backward_count == 0
        # All peers must remain usable after collective rejection, not timeout.
        dist.barrier()
        Path(output, f'rejected-{rank}.json').write_text(json.dumps({'rejected_before_execution': True}))
    finally:
        dist.destroy_process_group()


def test_registered_fsdp_method_rebind_failure_is_collective(tmp_path):
    mp.spawn(asymmetric_method_worker, args=(str(tmp_path / 'rdzv'), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        assert json.loads((tmp_path / f'rejected-{rank}.json').read_text())['rejected_before_execution']
