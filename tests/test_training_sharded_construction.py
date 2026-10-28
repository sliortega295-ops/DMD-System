"""Actual native CPU/Gloo construction handoff; synthetic weights, not GPU proof.

FLUX.1 is only a local compatibility probe. The same tests target actual FLUX.2
when that class is installed; no missing class is replaced by another family.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillschedule.training import dmd, flow_image
from distillschedule.training.adapters import ModelComponents
from distillschedule.training.prediction import PredictionSpec
from test_training_flow_sharded_weight_loading import _case, _arguments, _spec, _SafeOpen
from test_open_workload_parameter_shadows import _init, implementation
from test_training_dmd_flow import OtherFlow, policy
from test_training_end_to_end import equal


pytestmark = pytest.mark.skipif(__import__('os').environ.get('DMD_MODERN_FLOW_CPU') != '1',
    reason='explicit actual native CPU opt-in required')
CONSTRUCTION = 'cpu-fsdp2-frozen-teacher-v1'
_FAILED_CONSTRUCTION_ROOTS = []


class NativeFlow(dmd.FlowDMDMethods, OtherFlow):
    """Fixture ABI around the real Diffusers class, not a replacement network."""
    def __init__(self, native, family):
        torch.nn.Module.__init__(self)
        self.native, self.family = native, family
        self.generation_sigmas = (1., .75, .5, .25, 0.)
        self.register_buffer('construction_constant', torch.tensor(.125), persistent=False)

    def denoise(self, sample, sigma, feeds, *, conditional):
        args = _arguments(self.family)
        args.update(hidden_states=sample['payload'] +
                    feeds['condition' if conditional else 'null_condition'].mean(), timestep=sigma)
        return {'payload': self.native(**args)[0]}


def _assets(root, family):
    cls, config = _case(family)
    config['out_channels'] = config['in_channels']
    with torch.random.fork_rng():
        torch.manual_seed(71)
        native = cls(**config)
        native.save_pretrained(root / 'native', safe_serialization=True, max_shard_size='1KB')
        # Different real checkpoint weights make the first Student objective
        # nonzero. Identical Teacher/Fake would give a legitimate zero update.
        with torch.no_grad():
            native.x_embedder.weight[0].add_(.1)
        native.save_pretrained(root / 'teacher', safe_serialization=True, max_shard_size='1KB')
    return dict(student=_spec(root, root / 'native'), teacher=_spec(root, root / 'teacher'))


def _build_model(parameters, *, construction=None):
    from distillschedule.training.low_rank import trainable_scope
    from distillschedule.training.single_device import construction_rng
    cls, _ = _case(parameters['family'])
    scope = dict(mode='lora', rank=1, alpha=1., targets=['x_embedder'])
    roles = {}
    with construction_rng(torch.device('cpu'), parameters.get('seed', 0)):
        for name in ('student', 'fake', 'teacher'):
            if name == 'teacher' and construction is not None:
                native, _ = construction.load_teacher(cls, parameters['assets'], parameters['spec']['teacher'])
            else:
                def load():
                    spec = parameters['spec']['teacher' if name == 'teacher' else 'student']
                    native, _ = flow_image.load_native(cls, Path(parameters['assets']) / spec['path'], 100_000)
                    if name == 'teacher':
                        native.requires_grad_(False).eval()
                    else:
                        trainable_scope(native, scope)
                    return native
                native = construction.local(load, 'fixture dense role') if construction else load()
            roles[name] = (construction.wrap_teacher(NativeFlow, family=parameters['family'])
                if name == 'teacher' and construction is not None else NativeFlow(native, parameters['family']))
            if name == 'teacher':
                roles[name].eval()
                if construction:
                    if parameters.get('fail_after_teacher'):
                        _FAILED_CONSTRUCTION_ROOTS.append(roles[name])
                        def fail_before_optimizer():
                            if dist.get_rank() == 0:
                                raise OSError('injected local failure after Teacher binding, before publication')
                        construction.local(fail_before_optimizer, 'fixture construction publication')
        state = dmd.training_state(**roles, student_lr=.001, fake_lr=.002,
            scheduler_gamma=.97, ema_decay=.9)
    return ModelComponents(state=state, immutable_modules=('teacher',),
        prediction=PredictionSpec('flow'),
        capabilities=('dmd_flow_v1', 'dmd_flow_private_replicated_b1_v1'),
        sharded_construction=construction).validate() if construction else ModelComponents(
            state=state, immutable_modules=('teacher',), prediction=PredictionSpec('flow'),
            capabilities=('dmd_flow_v1', 'dmd_flow_private_replicated_b1_v1')).validate()


def build_model(parameters):
    return _build_model(parameters)


def _admit(**kwargs):
    return flow_image.admit_frozen_teacher_construction(**kwargs, factory=build_model, builder=_build_model)


build_model.admit_construction = _admit


def _observe_teacher_reads(patch, root):
    import safetensors
    import safetensors.torch
    from torch.distributed.tensor import DTensor
    safe_open, load_file = safetensors.safe_open, safetensors.torch.load_file
    reads = []
    def open_local(path, *args, **kwargs):
        handle = safe_open(path, *args, **kwargs)
        return _SafeOpen(handle, reads) if Path(path).parent == root / 'teacher' else handle
    def load_dense(path, *args, **kwargs):
        assert Path(path).parent != root / 'teacher', 'Teacher read a full checkpoint payload'
        return load_file(path, *args, **kwargs)
    def no_full(*args, **kwargs):
        raise AssertionError('construction or rejected handoff performed a full-tensor gather')
    patch.setattr(safetensors, 'safe_open', open_local)
    patch.setattr(safetensors.torch, 'load_file', load_dense)
    patch.setattr(DTensor, 'full_tensor', no_full)
    return reads


def _config(root, family, spec, *, enabled=True):
    from types import SimpleNamespace
    parameters = dict(assets=str(root), family=family, spec=spec, seed=137)
    if enabled:
        parameters['construction'] = CONSTRUCTION
    return SimpleNamespace(model=dict(factory=__name__ + ':build_model', parameters=parameters),
        trainer=dict(device='cpu', seed=137, distributed={'backend': 'gloo'}),
        execution=dict(mode='manual', implementation=implementation()))


def _feeds(family):
    return dict(latent=torch.zeros_like(_arguments(family)['hidden_states']),
        condition=torch.tensor([.1]), null_condition=torch.tensor([-.1]))


def _settings():
    value = policy()
    value.update(fake_updates=5, teacher_cfg=1., fake_cfg=1.)
    return value


def _legacy_worker(rank, rendezvous, directory, family, spec):
    _init(rank, rendezvous)
    try:
        from distillschedule.open_workload.replicated import make_session
        from distillschedule.open_workload.replica_contract import module_method_binding
        from distillschedule.open_workload.sharded import rejection
        from distillschedule.training.tensor_state import ParameterShadows
        from torch.distributed.device_mesh import init_device_mesh
        root = Path(directory)
        model = _build_model(_config(root, family, spec, enabled=False).model['parameters'])
        teacher, _ = flow_image._load_native_frozen_shards(_case(family)[0], root, spec['teacher'],
                                                         init_device_mesh('cpu', (2,)))
        model.state['teacher'] = NativeFlow(teacher, family).eval()
        model.state['student_ema'] = ParameterShadows('student', model.state['student_ema'])
        w = dmd.build_algorithm(model, _feeds(family), _settings(), 0, 53)
        w.replica_semantics = replace(w.replica_semantics, forward_interface='registered_methods',
            forward_methods={key: 'dmd_forward' for key in
                             ('student_forward', 'fake_forward', 'teacher_forward')},
            frozen_module_bindings={'teacher_forward': 'teacher'},
            parameter_shadows={'student_ema': 'student'}, shard_boundaries={'teacher': ('native',)})
        w.validate()
        with pytest.raises(ValueError, match='live construction certificate'):
            _ = w.identity
        assert rejection(w) == 'DENSE_CONTIGUOUS_PARAMETER_FACTORY_REQUIRED'
        with pytest.raises(ValueError, match='DENSE_CONTIGUOUS_PARAMETER_FACTORY_REQUIRED'):
            make_session(w, implementation(), copy_state=False, persistent=True)
        (root / f'legacy-{rank}.json').write_text(json.dumps(
            dict(actual_class=_case(family)[0].__name__, rejection=rejection(w),
                 cuda_initialized=torch.cuda.is_initialized())))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('family', ['flux1', 'flux2'])
def test_caller_pre_sharded_actual_native_root_still_rejected(tmp_path, family):
    spec = _assets(tmp_path, family)
    mp.spawn(_legacy_worker, args=(str(tmp_path / 'legacy-rdzv'), str(tmp_path), family, spec),
             nprocs=2, join=True)


def _handoff_worker(rank, rendezvous, directory, family, spec, stage):
    _init(rank, rendezvous)
    try:
        from distillschedule.training.policy import CyclePolicy
        from distillschedule.open_workload.torch_backend import full_tensor
        from distillschedule.training.session import _construct_model
        from distillschedule.training.checkpoint import pack_state, rng_state, save_checkpoint, restore_checkpoint, read_manifest
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor
        root = Path(directory)
        config = _config(root, family, spec)
        before = rng_state(device='cpu')
        # Default dense construction and opt-in Student/Fake LoRA draws match;
        # construction forks RNG, so neither advances the caller's train stream.
        dense = _build_model(_config(root, family, spec, enabled=False).model['parameters'])
        with pytest.MonkeyPatch.context() as patch:
            reads = _observe_teacher_reads(patch, root)
            model, _ = _construct_model(config)
        assert reads and sum(row['elements'] for row in reads) < sum(
            p.numel() for p in model.state['teacher'].parameters())
        coverage = [None, None]
        dist.all_gather_object(coverage, sum(row['elements'] for row in reads))
        assert sum(coverage) == sum(p.numel() for p in model.state['teacher'].parameters())
        assert equal(before, rng_state(device='cpu'))
        for role in ('student', 'fake'):
            assert equal(pack_state({role: dense.state[role]}), pack_state({role: model.state[role]}))
        with torch.no_grad():
            expected = dense.state['teacher'].dmd_forward(
                {'noise': {'sample': {'payload': _feeds(family)['latent']}, 'sigma': torch.tensor([.4])},
                 'feeds': _feeds(family)}, flow_role='teacher', phase='student', cycle=0,
                fake_index=0, flow_policy=_settings())
        state = model.state
        native_parameters = dict(state['teacher'].native.named_parameters())
        w = dmd.build_algorithm(model, _feeds(family), _settings(), 0, 53)
        runner = CyclePolicy(config.execution)
        identity = dict(fixture='native-construction-v1', family=family, spec=spec)
        old = {role: {name: p.detach().clone() for name, p in state[role].named_parameters() if p.requires_grad}
               for role in ('student', 'fake')}
        if stage == 'restore':
            manifest = read_manifest(root / 'checkpoint', identity=identity)
            runner.restore(manifest['policy'])
            session = runner.prepare_resume(w)
            payload = restore_checkpoint(root / 'checkpoint', session.state, identity=identity)
            model.state.update({key: session.state[key] for key in model.state})
            assert payload['cycle'] == 1 and equal(payload['rng'], rng_state(device='cpu'))
            w = dmd.build_algorithm(model, _feeds(family), _settings(), 1, 54)
            w.program = replace(w.program, initial_versions=payload['versions'])
        session, output = runner.run(w, root / ('policy-' + stage))
        model.state.update({key: session.state[key] for key in model.state})
        assert session.state['teacher'] is state['teacher']
        assert isinstance(state['teacher'], FSDPModule) and isinstance(state['teacher'].native, FSDPModule)
        assert all(state['teacher'].native.get_parameter(name) is parameter
                   for name, parameter in native_parameters.items())
        for role in ('student', 'fake'):
            assert {id(p) for group in state[role + '_optimizer'].param_groups for p in group['params']} == {
                id(p) for p in state[role].parameters() if p.requires_grad}
        # The real registered outer method closes native child AND outer root.
        with torch.no_grad():
            actual = state['teacher'].dmd_forward(
                {'noise': {'sample': {'payload': _feeds(family)['latent']}, 'sigma': torch.tensor([.4])},
                 'feeds': _feeds(family)}, flow_role='teacher', phase='student', cycle=0,
                fake_index=0, flow_policy=_settings())
        torch.testing.assert_close(actual['payload'], expected['payload'], rtol=0, atol=0)
        assert all(isinstance(p, DTensor) for p in state['teacher'].parameters())
        assert all(torch.isfinite(value).all() for value in output.values())
        for role in ('student', 'fake'):
            assert any(not torch.equal(old[role][name], full_tensor(p)) for name, p in state[role].named_parameters()
                       if p.requires_grad)
        assert state['dmd_counters'].tolist() == ([1, 1, 5] if stage == 'write' else [2, 2, 10])
        assert all(p.grad is None and not p.requires_grad for p in state['teacher'].parameters())
        if stage == 'write':
            save_checkpoint(root / 'checkpoint', state, identity=identity, cycle=1, cursor=2,
                versions=session.receipt['dispatcher']['final_versions'], policy=runner.bound)
            w = dmd.build_algorithm(model, _feeds(family), _settings(), 1, 54)
            w.program = replace(w.program, initial_versions=session.receipt['dispatcher']['final_versions'])
            session, expected_output = runner.run(w, root / 'policy-second')
            torch.save(dict(state=pack_state(state), rng=rng_state(device='cpu'), output=expected_output),
                       root / f'expected-{rank}.pt')
        else:
            expected_row = torch.load(root / f'expected-{rank}.pt', weights_only=True)
            assert equal(expected_row, dict(state=pack_state(state), rng=rng_state(device='cpu'), output=output))
        (root / f'proof-{stage}-{rank}.json').write_text(json.dumps(dict(
            family=family, actual_class=_case(family)[0].__name__, local_reads=reads,
            full_teacher_payload=False, constructor_full_tensor_gather=False,
            exact_forward=True, exact_student_fake_initialization=True, caller_rng_preserved=True,
            actual_student_fake_adam_updates=True, full_parameter_ema=True,
            lifecycle_installations=session.effects['lifecycle_installations'],
            restored_exact_state_and_rng=stage == 'restore', cuda_initialized=torch.cuda.is_initialized())))
        model.close()
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('family', ['flux1', 'flux2'])
def test_public_provider_actual_native_teacher_adopt_update_and_new_process_restore(tmp_path, family):
    spec = _assets(tmp_path, family)
    for stage in ('write', 'restore'):
        mp.spawn(_handoff_worker, args=(str(tmp_path / ('rdzv-' + stage)), str(tmp_path),
                 family, spec, stage), nprocs=2, join=True)


def _refusal_worker(rank, rendezvous, directory, family, spec, only_kinds=None):
    from types import MethodType
    from distillschedule.training.session import _construct_model
    from distillschedule.training.checkpoint import rng_state
    from distillschedule.open_workload.replicated import make_session
    from distillschedule.open_workload.sharded import _CONSTRUCTION_HANDOFFS
    from distillschedule.training.policy import CyclePolicy
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    _init(rank, rendezvous)
    root, rejected = Path(directory), []
    try:
        for kind in only_kinds or ('expired', 'forwarded', 'parameter', 'method', 'mesh', 'forged', 'trainable', 'impostor_class', 'buffer',
                     'training_mode',
                     'clone', 'wrong_implementation', 'asymmetric_implementation', 'cross_rank_owner', 'forged_selection',
                     'native_forward', 'child_forward'):
            model, _ = _construct_model(_config(root, family, spec))
            other = None
            try:
                w = dmd.build_algorithm(model, _feeds(family), _settings(), 0, 53)
                handoff = model.sharded_construction.handoff
                option = implementation()
                if kind == 'forwarded':
                    assert next(iter(handoff.native._forward_pre_hooks.values())).__self__ is handoff
                    with pytest.raises(ValueError, match='cannot forward'):
                        handoff.native(**_arguments(family))
                elif kind == 'cross_rank_owner':
                    other, _ = _construct_model(_config(root, family, spec))
                    if rank:
                        w = dmd.build_algorithm(other, _feeds(family), _settings(), 0, 53)
                elif kind == 'clone':
                    pass
                elif kind == 'forged_selection':
                    option['reduction'] = handoff.option['reduction'] = 'fsdp2_keep_forward'
                elif kind == 'wrong_implementation' or (kind == 'asymmetric_implementation' and rank == 0):
                    option['reduction'] = 'fsdp2_keep_forward'
                elif rank == 0:
                    if kind == 'expired':
                        model.close()
                    elif kind == 'parameter':
                        name, parameter = next(iter(handoff.native.named_parameters()))
                        parent, _, leaf = name.rpartition('.')
                        handoff.native.get_submodule(parent).register_parameter(leaf,
                            torch.nn.Parameter(parameter.detach().clone(), requires_grad=False))
                    elif kind == 'method':
                        handoff.root.dmd_forward = MethodType(NativeFlow.denoise, handoff.root)
                    elif kind == 'mesh':
                        # A different real CPU mesh object cannot replace the
                        # construction mesh even if it names the same ranks.
                        handoff.mesh = init_device_mesh('cpu', (2,))
                    elif kind == 'forged':
                        del _CONSTRUCTION_HANDOFFS[handoff.root]
                        handoff.root._sharded_construction = handoff
                    elif kind == 'trainable':
                        next(handoff.native.parameters()).requires_grad_(True)
                    elif kind == 'impostor_class':
                        handoff.native.__class__ = type('FSDPImpostor', (FSDPModule, handoff.native_class), {})
                    elif kind == 'buffer':
                        handoff.root.construction_constant.add_(1)
                    elif kind == 'training_mode':
                        handoff.root.train()
                    elif kind in ('native_forward', 'child_forward'):
                        module = handoff.native if kind == 'native_forward' else handoff.native.x_embedder
                        def foreign_forward(self, *args, **kwargs):
                            raise AssertionError('rebound issued forward executed')
                        module.forward = MethodType(foreign_forward, module)
                if kind == 'mesh' and rank != 0:
                    init_device_mesh('cpu', (2,))
                before = rng_state(device='cpu')
                with pytest.MonkeyPatch.context() as patch:
                    def no_full(*args, **kwargs):
                        raise AssertionError('rejected owner reached a tensor gather')
                    patch.setattr(DTensor, 'full_tensor', no_full)
                    with pytest.raises(ValueError):
                        if kind == 'clone':
                            make_session(w, option, copy_state=True, persistent=True)
                        else:
                            CyclePolicy(dict(mode='manual', implementation=option)).run(w, root / 'must-not-exist')
                assert equal(before, rng_state(device='cpu'))
                assert model.state['dmd_counters'].tolist() == [0, 0, 0]
                assert all(not model.state[name + '_optimizer'].state for name in ('student', 'fake'))
                rejected.append(kind)
                dist.barrier()
            finally:
                model.close()
                if other is not None:
                    other.close()
        if only_kinds is not None:
            (root / f'forward-refused-{rank}.json').write_text(json.dumps(dict(
                rejected=rejected, before_update=True, before_full_tensor_gather=True,
                cuda_initialized=torch.cuda.is_initialized())))
            return
        # One rank cannot skip the provider while another starts construction.
        request = _config(root, family, spec, enabled=rank == 0)
        with pytest.raises(ValueError, match='different construction providers'):
            _construct_model(request)
        rejected.append('asymmetric_provider')
        # All unsupported recipe choices fail before the native builder, and
        # never initialize a CUDA context to discover they are unsupported.
        for kind in ('auto', 'cuda', 'hsdp', 'foreign_recipe'):
            request = _config(root, family, spec)
            if kind == 'auto':
                request.execution = dict(mode='auto', memory_limit_bytes=10**8)
            elif kind == 'cuda':
                request.trainer['device'] = request.model['parameters']['device'] = 'cuda:0'
            elif kind == 'hsdp':
                request.execution['implementation']['reduction'] = 'hsdp2_reshard'
            else:
                request.model['parameters']['construction'] = 'caller-owned-pre-sharded'
            before = rng_state(device='cpu')
            with pytest.raises(ValueError, match='requires explicit CPU/Gloo'):
                _construct_model(request)
            assert equal(before, rng_state(device='cpu'))
            rejected.append(kind)
        (root / f'refused-{rank}.json').write_text(json.dumps(dict(
            rejected=rejected, before_update=True, before_full_tensor_gather=True,
            cuda_initialized=torch.cuda.is_initialized())))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('family', ['flux1', 'flux2'])
def test_real_frozen_handoff_refusals_are_collective_before_gather_or_update(tmp_path, family):
    spec = _assets(tmp_path, family)
    mp.spawn(_refusal_worker, args=(str(tmp_path / 'refusal-rdzv'), str(tmp_path), family, spec),
             nprocs=2, join=True)


@pytest.mark.parametrize('family', ['flux1', 'flux2'])
@pytest.mark.parametrize('kind', ['native_forward', 'child_forward'])
def test_issued_native_and_child_forward_rebinding_rejected_before_signatures(tmp_path, family, kind):
    spec = _assets(tmp_path, family)
    mp.spawn(_refusal_worker, args=(str(tmp_path / 'forward-rdzv'), str(tmp_path), family, spec,
             (kind,)), nprocs=2, join=True)


def _mesh_integrity_worker(rank, rendezvous, directory, family, spec):
    from distillschedule.open_workload import sharded
    from distillschedule.training.session import _construct_model
    from distillschedule.training.checkpoint import rng_state
    _init(rank, rendezvous)
    try:
        root = Path(directory)
        mesh = sharded.frozen_construction_mesh()
        original = mesh.mesh.clone()
        before = rng_state(device='cpu')
        if rank == 0:
            mesh.mesh[0] = mesh.mesh[1]  # CPU rank metadata only, not a new group.
        try:
            with pytest.MonkeyPatch.context() as patch:
                def forbidden_cohort(*args, **kwargs):
                    raise AssertionError('invalid cached mesh reached subsequent handoff/cohort creation')
                patch.setattr(sharded, '_FrozenRootHandoff', forbidden_cohort)
                with pytest.raises(ValueError, match='construction mesh changed'):
                    _construct_model(_config(root, family, spec))
            assert equal(before, rng_state(device='cpu'))
        finally:
            mesh.mesh.copy_(original)
        dist.barrier()
        fresh, _ = _construct_model(_config(root, family, spec))
        fresh.close()
        (root / f'mesh-refused-{rank}.json').write_text(json.dumps(dict(
            asymmetric_cached_mesh_collectively_rejected=True, before_handoff_cohort=True,
            caller_rng_preserved=True, fresh_construction=True,
            cuda_initialized=torch.cuda.is_initialized())))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('family', ['flux1', 'flux2'])
def test_asymmetric_cached_construction_mesh_rejected_before_new_cohort(tmp_path, family):
    spec = _assets(tmp_path, family)
    mp.spawn(_mesh_integrity_worker, args=(str(tmp_path / 'mesh-rdzv'), str(tmp_path), family, spec),
             nprocs=2, join=True)


def test_actual_flux2_public_factory_train_and_new_process_resume(tmp_path):
    """Target gate: actual model/media/data factory, no fixture-role replacement."""
    _case('flux2')  # Missing actual class is explicitly deferred, never substituted.
    assert torch.__version__.split('+')[0] == '2.10.0', 'public FLUX.2 factory requires its existing pinned target lane'
    from distillschedule.training.plugins import load_callable
    recipe = Path(__file__).resolve().parents[1] / 'recipes/flow_image/create_cpu_fixture.py'
    builder, _ = load_callable(str(recipe) + ':build_fixture')
    assets = builder(tmp_path / 'native', 'flux2-klein-base', cycles=2, samples=4)
    raw = json.loads(assets['config'].read_text())
    raw['model']['parameters']['construction'] = CONSTRUCTION
    raw['data']['sharding'] = 'rank_strided'
    raw['trainer']['distributed'] = dict(backend='gloo', timeout_seconds=90)
    raw['execution'] = dict(mode='manual', implementation=implementation())
    path = tmp_path / 'public.json'
    path.write_text(json.dumps(raw))
    from native_flow_replicated import fresh_cli
    for name, cycles, resume in (('whole', 2, None), ('partial', 1, None),
                                ('resumed', 2, tmp_path / 'partial/checkpoints/cycle-000001')):
        args = ['train', path, '--cycles', cycles, '--output', tmp_path / name]
        if resume is not None:
            args += ['--resume', resume]
        fresh_cli(path, tmp_path / (name + '.log'), 2, *args)
    # The producer keeps identity in the common manifest, not each rank payload.
    # This parent is outside the two-rank group, so verify the published files
    # offline instead of asking the live-topology restart loader to impersonate it.
    import hashlib
    checkpoints = [tmp_path / name / 'checkpoints/cycle-000002' for name in ('whole', 'resumed')]
    manifests = []
    for checkpoint in checkpoints:
        manifest_bytes = (checkpoint / 'manifest.json').read_bytes()
        assert (checkpoint / '_SUCCESS').read_text().strip() == hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        assert set(manifest['files']) == {f'rank-{rank:05d}.pt' for rank in range(2)}
        for name, digest in manifest['files'].items():
            assert hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() == digest
        manifests.append(manifest)
    assert equal({key: value for key, value in manifests[0].items() if key != 'files'},
                 {key: value for key, value in manifests[1].items() if key != 'files'})
    for rank in range(2):
        whole, resumed = [torch.load(tmp_path / name / 'checkpoints/cycle-000002' /
            f'rank-{rank:05d}.pt', weights_only=True) for name in ('whole', 'resumed')]
        assert equal(whole, resumed), 'complete rank payload after new-process restore'
        summary = json.loads((tmp_path / 'resumed/summary.json').read_text())
        assert manifests[1]['identity'] == summary['identity']
        assert summary['status'] == 'COMPLETE' and summary['finite'] and summary['world_size'] == 2
        assert summary['identity']['model_metadata']['construction']['schema_version'] == CONSTRUCTION
        assert summary['policy']['implementation'] == implementation()
        records = (tmp_path / f'resumed/ranks/rank-{rank:05d}/metrics.jsonl').read_text().splitlines()
        effects = json.loads(records[0])['execution']['effects']
        assert effects['construction_owned_frozen_roots'] == ['teacher']
        assert effects['lifecycle_installations'] == 1
        (tmp_path / f'public-proof-{rank}.json').write_text(json.dumps(dict(
            actual_family='flux2-klein-base', public_cli_train=True, fresh_process=True,
            state_rng_identity_policy_exact=True)))


def _disposal_worker(rank, rendezvous, directory, family, spec):
    from distillschedule.training.session import _construct_model
    from distillschedule.training.checkpoint import rng_state
    from distillschedule.open_workload.sharded import _CONSTRUCTION_HANDOFFS
    from torch.distributed.tensor import DTensor
    _init(rank, rendezvous)
    try:
        root = Path(directory)
        config = _config(root, family, spec)
        config.model['parameters']['fail_after_teacher'] = True
        before = rng_state(device='cpu')
        with pytest.raises(ValueError, match='fixture construction publication'):
            _construct_model(config)
        assert equal(before, rng_state(device='cpu'))
        discarded = _FAILED_CONSTRUCTION_ROOTS[-1]
        assert discarded not in _CONSTRUCTION_HANDOFFS
        assert all(isinstance(p, DTensor) and not p.requires_grad and p.grad is None
                   for p in discarded.parameters())
        with pytest.raises(ValueError, match='cannot forward'):
            discarded.native(**_arguments(family))
        # A new controlled owner on the SAME live group remains usable.
        fresh, _ = _construct_model(_config(root, family, spec))
        assert fresh.state['teacher'] is not discarded
        fresh.close()
        (root / f'disposed-{rank}.json').write_text(json.dumps(dict(
            failure_collective=True, unpublished_root_expired=True, rng_unchanged=True,
            fresh_reconstruction=True, cuda_initialized=torch.cuda.is_initialized())))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('family', ['flux1', 'flux2'])
def test_failed_new_construction_expires_unpublished_root_without_rebinding_live_state(tmp_path, family):
    spec = _assets(tmp_path, family)
    mp.spawn(_disposal_worker, args=(str(tmp_path / 'disposal-rdzv'), str(tmp_path), family, spec),
             nprocs=2, join=True)


def test_default_dense_identity_retains_exact_legacy_model_code():
    from distillschedule.open_workload.torch_backend import code_signature
    from test_training_flow_model_methods import _owned_components, _feeds as dense_feeds
    model = _owned_components()
    w = dmd.build_algorithm(model, dense_feeds(), policy(), 0, 53)
    expected = {type(module).__module__ + '.' + type(module).__qualname__: code_signature(type(module))
        for value in w.state.values() if isinstance(value, torch.nn.Module) for module in value.modules()}
    assert w.identity['model_code'] == expected
    assert model.sharded_construction is None and w.replica_semantics is None


@pytest.mark.parametrize('fault', ['cuda', 'cuda_model', 'auto', 'hsdp', 'no_group', 'recipe', 'order', 'other_family'])
def test_public_catalog_refuses_unsupported_construction_before_assets_or_cuda(fault):
    from distillschedule.training.catalog import validate_compatibility
    model = dict(factory='distillschedule.training.flux:build_model',
                 parameters=dict(assets='/MUST-NOT-READ', construction=CONSTRUCTION))
    trainer = dict(device='cpu', distributed={'backend': 'gloo'})
    execution = dict(mode='manual', implementation=implementation())
    if fault == 'cuda':
        trainer['device'] = 'cuda'
    elif fault == 'cuda_model':
        model['parameters']['device'] = 'cuda'
    elif fault == 'auto':
        execution = dict(mode='auto', memory_limit_bytes=10**8)
    elif fault == 'hsdp':
        execution['implementation']['reduction'] = 'hsdp2_reshard'
    elif fault == 'no_group':
        del trainer['distributed']
    elif fault == 'recipe':
        model['parameters']['construction'] = 'foreign'
    elif fault == 'order':
        execution['order'] = []
    else:
        model['factory'] = 'distillschedule.training.qwen_image:build_model'
    with pytest.raises(ValueError, match='unsupported construction'):
        validate_compatibility(model, dict(factory='distillschedule.training.dmd:build_algorithm'), trainer, execution)
    assert not torch.cuda.is_initialized()
