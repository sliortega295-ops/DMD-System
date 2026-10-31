"""Actual upstream tiny SD1.5 math with real CPU FSDP2 shard/state seams.

CPU admission below is an explicit test-only monkeypatch, not a supported
public CPU FSDP recipe or CUDA qualification. Production settings/device/source
checks remain exercised by test_training_dmd2_fsdp_reference.py.
"""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_training_dmd2_cpu import source_root


def _native_worker(rank, rendezvous, root, source):
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import construct
    from distillschedule.training.dmd2_owner import DMD2Owner
    from distillschedule.training.dmd2_collective import DMD2CollectiveOwner
    from distillschedule.training.dmd2_fsdp import DMD2FSDP2Owner, shard_tree
    from distillschedule.training.checkpoint import pack_state, unpack_state
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    repo = Path(__file__).resolve().parents[1]
    builder, _ = load_callable(str(repo / 'recipes/dmd2_sd15/create_cpu_fixture.py') + ':build_fixture')
    torch.set_num_threads(1)
    files = builder(Path(root) / f'assets-{rank}', source, cycles=3, ratio=2)
    config = load_training_config(files['config'])
    # Use the real media/prepared-input/factory path before the CPU shard test.
    made = [construct(config) for _ in range(3)]
    feeds = made[0][1].cycle_sample(0)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=180))
    patch = pytest.MonkeyPatch()
    try:
        # Only topology admission and original world-scaled scheduler control are
        # adapted to the CPU test; loss/update body and checkpoint are unmodified.
        def cpu_group(owner):
            assert owner.device.type == 'cpu' and dist.get_backend() == 'gloo' and dist.get_world_size() == 2
        patch.setattr(DMD2Owner, '_validate_process_ownership', cpu_group)
        patch.setattr(DMD2Owner, '_clip_grad_norm', DMD2FSDP2Owner._clip_grad_norm)
        patch.setattr(DMD2Owner, '_validate_checkpoint_tensors', DMD2FSDP2Owner._validate_checkpoint_tensors)
        patch.setattr(DMD2Owner, '_scheduler_clock', DMD2CollectiveOwner._scheduler_clock)
        patch.setattr(DMD2Owner, '_step_scheduler', DMD2CollectiveOwner._step_scheduler)
        mesh = init_device_mesh('cpu', (2,), mesh_dim_names=('fsdp',))
        owners = []
        for components, *_ in made:
            original = components.state['native_cycle']
            model = original.model
            for name, module in [('generator', model.feedforward_model), ('guidance', model.guidance_model)]:
                shard_tree(module, mesh, role='ACTUAL_TINY_UPSTREAM_CPU_' + name)
            owner = DMD2Owner.__new__(DMD2Owner)
            owner._collective = {'world_size': 2}
            DMD2Owner.__init__(owner, model, original.args, original.source, original.uncond_embedding,
                              identity=original.identity, seed=17)
            owners.append(owner)
        first, resumed, full = owners
        # Producer is the real native owner + safe local-shard checkpoint writer;
        # consumer is the real validator/loader, not a hand-built success file.
        first.run(feeds, cycle=0)
        saved = pack_state({'native_cycle': first})
        checkpoint = Path(root) / f'native-rank-{rank}.pt'
        torch.save(saved, checkpoint)
        loaded = torch.load(checkpoint, weights_only=True)
        unpack_state(loaded, {'native_cycle': resumed}, progress=dict(cycle=1, cursor=1,
                     versions={'native_cycle': 1}))
        full.run(feeds, cycle=0)
        for owner in (resumed, full):
            owner.run(feeds, cycle=1)
        a, b = pack_state({'native_cycle': resumed}), pack_state({'native_cycle': full})
        from test_training_end_to_end import equal
        assert equal(a, b), 'native parameters, Adam, schedulers, RNG and consumption must resume exactly'
        assert resumed.native_step == resumed.guidance_updates == 4 and resumed.generator_updates == 2
        assert all(s.last_epoch == 8 for s in resumed.schedulers.values())
        assert all(isinstance(p, DTensor) for p in resumed.model.feedforward_model.parameters())
        assert all(isinstance(optimizer.state[p]['exp_avg'], DTensor)
                   for optimizer in resumed.optimizers.values() for p in optimizer.state)
        # Rejection occurs before any live tensor mutation.
        before = pack_state({'native_cycle': resumed})
        bad = deepcopy(before)
        bad['native_cycle']['value']['identity']['upstream']['revision'] = 'wrong-source'
        with pytest.raises(ValueError, match='identity'):
            unpack_state(bad, {'native_cycle': resumed}, progress=dict(cycle=2, cursor=2,
                         versions={'native_cycle': 2}))
        assert equal(before, pack_state({'native_cycle': resumed}))
        (Path(root) / f'native-{rank}.json').write_text(json.dumps(dict(
            native_step=resumed.native_step, generator_updates=resumed.generator_updates,
            guidance_updates=resumed.guidance_updates, cuda_initialized=torch.cuda.is_initialized(),
            evidence='ACTUAL_TINY_SD15_CPU_FSDP_STATE_SEAMS_NOT_PUBLIC_ADMISSION')))
    finally:
        patch.undo()
        dist.destroy_process_group()


def test_actual_native_sd15_local_shard_continuation(tmp_path, source_root):
    mp.spawn(_native_worker, args=(str(tmp_path / 'rdzv'), str(tmp_path), str(source_root)), nprocs=2, join=True)
    assert all(json.loads((tmp_path / f'native-{rank}.json').read_text())['cuda_initialized'] is False
               for rank in range(2))


def test_actual_native_classifier_uses_hooks_with_identical_loss_gradient_rng(source_root):
    from distillschedule.training.dmd2_fsdp import ClassifierCallScope
    from test_training_official_dmd2_mechanisms import _guidance
    torch.set_num_threads(1)
    torch.manual_seed(803)
    first = _guidance(source_root)
    second = deepcopy(first)
    sample = torch.randn(1, 4, 8, 8)
    text = torch.randn(1, 3, 8)
    rng = torch.get_rng_state().clone()
    hook_calls = []
    hook = second.fake_unet.register_forward_pre_hook(lambda *args: hook_calls.append('fake'))
    expected = first.compute_cls_logits(sample, text, None).square().mean()
    expected.backward()
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    with ClassifierCallScope(second):
        actual = second.compute_cls_logits(sample, text, None).square().mean()
        actual.backward()
    assert hook_calls == ['fake']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for (_, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    hook.remove()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('failure', (False, True))
def test_actual_factory_managed_scope_closes_real_native_hooks(tmp_path, source_root, monkeypatch, failure):
    from distillschedule.training.dmd2_owner import DMD2Owner
    from distillschedule.training.dmd2_fsdp import DMD2FSDP2Owner, ClassifierCallScope
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import construct
    repo = Path(__file__).resolve().parents[1]
    builder, _ = load_callable(str(repo / 'recipes/dmd2_sd15/create_cpu_fixture.py') + ':build_fixture')
    files = builder(tmp_path / 'assets', source_root, cycles=1, ratio=2)
    # Exercise the actual factory's callback registration without simulating a
    # CUDA group. The production close method and real native source hooks are
    # used, not a dummy ModelComponents or a manually invoked owner.close().
    monkeypatch.setattr(DMD2Owner, 'close', DMD2FSDP2Owner.close, raising=False)
    components, *_ = construct(load_training_config(files['config']))
    owner = components.state['native_cycle']
    scope = ClassifierCallScope(owner.model.guidance_model)
    owner.model._dmd2_classifier_call_scope = scope
    handle = owner.model.guidance_model.fake_unet.register_forward_pre_hook(lambda *args: None)
    owner._forward_handles = [handle]
    assert components.close.__self__ is owner and components.close.__func__ is DMD2FSDP2Owner.close
    def use():
        with components.managed():
            assert not scope.closed
            if failure:
                raise RuntimeError('injected user path failure')
    if failure:
        with pytest.raises(RuntimeError, match='user path'):
            use()
    else:
        use()
    assert scope.closed and owner._forward_handles == []
    assert handle.id not in owner.model.guidance_model.fake_unet._forward_pre_hooks
    assert 'compute_cls_logits' not in vars(owner.model.guidance_model)
    assert not torch.cuda.is_initialized()


def _factory_wiring_worker(rank, rendezvous, root, source):
    """Exercise actual factory routing, with CPU topology/precision only."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    from distillschedule.training import sd, dmd2_fsdp
    from distillschedule.training.dmd2_owner import DMD2Owner
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.config import load_training_config
    torch.set_num_threads(1)
    repo = Path(__file__).resolve().parents[1]
    builder, _ = load_callable(str(repo / 'recipes/dmd2_sd15/create_cpu_fixture.py') + ':build_fixture')
    files = builder(Path(root) / f'wiring-assets-{rank}', source, cycles=1, ratio=2)
    parameters = dict(load_training_config(files['config']).model['parameters'])
    args = sd._arguments(parameters, sdxl=False)
    parameters['native_role_backend'] = dmd2_fsdp.BACKEND
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=180))
    try:
        mesh = init_device_mesh('cpu', (2,), mesh_dim_names=('fsdp',))
        native_place, native_shard = dmd2_fsdp.place_roles, dmd2_fsdp.shard_tree
        native_scope_init = dmd2_fsdp.ClassifierCallScope.__init__
        native_owner_init = dmd2_fsdp.DMD2FSDP2Owner.__init__
        completed = []
        for failure in ('none', 'user', 'shard', 'owner'):
            scopes, events = [], []
            def cpu_owner(owner):
                assert owner.device == torch.device('cpu') and dist.get_world_size() == 2
                assert owner._collective['backend'] == dmd2_fsdp.BACKEND
                parameters = [p for module in (owner.model.feedforward_model, owner.model.guidance_model)
                              for p in module.parameters()]
                if owner._prepared:
                    assert all(isinstance(p, DTensor) and p.device_mesh == mesh for p in parameters)
                else:
                    assert all(not isinstance(p, DTensor) for p in parameters), (
                        'construction must not shard before collective admission')
            def observed_scope(scope, guidance):
                native_scope_init(scope, guidance)
                scopes.append(scope)
            def observed_shard(module, current_mesh, *, role, synchronize=True):
                events.append('shard-' + role)
                if failure == 'shard' and role == 'guidance':
                    raise RuntimeError('injected shard construction failure')
                return native_shard(module, current_mesh, role=role, synchronize=synchronize)
            def observed_owner(owner, *values, **settings):
                events.append('owner')
                native_owner_init(owner, *values, **settings)
                assert all(not isinstance(p, DTensor) for optimizer in owner.optimizers.values()
                           for group in optimizer.param_groups for p in group['params'])
                if failure == 'owner':
                    raise RuntimeError('injected owner construction failure')
            # These are explicit test-only CPU/dtype adaptations. Production
            # world16/CUDA admission and native mixed precision are unchanged.
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(dmd2_fsdp.DMD2FSDP2Owner, '_validate_process_ownership', cpu_owner)
                patch.setattr(dmd2_fsdp.DMD2FSDP2Owner, '_validate_precision', DMD2Owner._validate_precision)
                patch.setattr(dmd2_fsdp.ClassifierCallScope, '__init__', observed_scope)
                patch.setattr(dmd2_fsdp.DMD2FSDP2Owner, '__init__', observed_owner)
                patch.setattr(dmd2_fsdp, 'shard_tree', observed_shard)
                patch.setattr(dmd2_fsdp, 'place_roles', lambda model, api: native_place(model, api, mesh=mesh))
                if failure == 'owner':
                    with pytest.raises(RuntimeError, match='injected owner'):
                        sd._build_placed(parameters, deepcopy(args), torch.device('cpu'), sdxl=False)
                    assert events == ['owner'] and not scopes
                else:
                    components = sd._build_placed(parameters, deepcopy(args), torch.device('cpu'), sdxl=False)
                    owner = components.state['native_cycle']
                    assert type(owner) is dmd2_fsdp.DMD2FSDP2Owner
                    assert components.close.__self__ is owner
                    assert events == ['owner'] and not scopes, 'factory must be collective-free'
                    optimizers = dict(owner.optimizers)
                    schedulers = dict(owner.schedulers)
                    before_groups = {name: [dict(group) for group in optimizer.param_groups]
                                     for name, optimizer in owner.optimizers.items()}
                    # The CPU fixture is No-GAN. Supply only the unused hook
                    # observation boundary; no classifier math is substituted.
                    owner.model.guidance_model.cls_pred_branch = torch.nn.Identity()
                    if failure == 'shard':
                        with pytest.raises(RuntimeError, match='injected shard'):
                            with components.managed():
                                owner.prepare_collective_execution()
                    else:
                        def execute():
                            with components.managed():
                                owner.prepare_collective_execution()
                                assert events == ['owner', 'shard-generator', 'shard-guidance']
                                assert all(isinstance(p, DTensor) for optimizer in owner.optimizers.values()
                                           for group in optimizer.param_groups for p in group['params'])
                                named = dict(owner._model_named('parameters'))
                                for role, optimizer in owner.optimizers.items():
                                    assert optimizer is optimizers[role] and not optimizer.state
                                    assert owner.schedulers[role] is schedulers[role]
                                    assert owner.schedulers[role].optimizer is optimizer
                                    for index, group in enumerate(optimizer.param_groups):
                                        assert all(group[key] == value for key, value in before_groups[role][index].items()
                                                   if key != 'params')
                                        assert all(parameter is named[name] for parameter, name in
                                                   zip(group['params'], owner.ownership[role][index], strict=True))
                                assert not scopes[0].closed
                                if failure == 'user':
                                    raise RuntimeError('user execution failure')
                        if failure == 'user':
                            with pytest.raises(RuntimeError, match='user execution failure'):
                                execute()
                        else:
                            execute()
                    assert len(scopes) == 1 and scopes[0].closed
                    assert 'compute_cls_logits' not in vars(scopes[0].guidance)
                completed.append(failure)
        # A rank-local model-load failure must meet the outer error exchange,
        # never another rank's first sharding/broadcast collective.
        from distillschedule.training.distributed import checked
        native_class = sd.original(source).unified.SDUniModel
        original_init = native_class.__init__
        def asymmetric_init(model, *values, **settings):
            if rank == 1:
                raise RuntimeError('rank1 asset construction failure')
            original_init(model, *values, **settings)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(native_class, '__init__', asymmetric_init)
            patch.setattr(dmd2_fsdp.DMD2FSDP2Owner, '_validate_process_ownership', cpu_owner)
            patch.setattr(dmd2_fsdp.DMD2FSDP2Owner, '_validate_precision', DMD2Owner._validate_precision)
            patch.setattr(dmd2_fsdp, 'place_roles', lambda *a, **k: pytest.fail('premature sharding collective'))
            with pytest.raises(ValueError, match='rank1 asset construction failure'):
                checked(lambda: sd._build_placed(parameters, deepcopy(args), torch.device('cpu'), sdxl=False),
                        'asymmetric test construction')
            dist.barrier()
        (Path(root) / f'wiring-{rank}.json').write_text(json.dumps(dict(
            completed=completed, cuda_initialized=torch.cuda.is_initialized(),
            evidence='ACTUAL_FACTORY_ROUTING_WITH_CPU_ONLY_TOPOLOGY_PRECISION_SEAMS')))
    finally:
        dist.destroy_process_group()


def test_actual_fsdp_factory_admits_before_sharding_and_cleans_failures(tmp_path, source_root):
    mp.spawn(_factory_wiring_worker, args=(str(tmp_path / 'wiring.rdzv'), str(tmp_path), str(source_root)),
             nprocs=2, join=True)
    rows = [json.loads((tmp_path / f'wiring-{rank}.json').read_text()) for rank in range(2)]
    assert all(row['completed'] == ['none', 'user', 'shard', 'owner']
               and row['cuda_initialized'] is False for row in rows)
