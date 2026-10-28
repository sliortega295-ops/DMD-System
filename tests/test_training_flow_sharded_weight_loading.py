"""SYNTHETIC_TEST_ONLY: real tiny Diffusers native frozen FSDP construction.

FLUX.1 is an explicitly separate local compatibility probe, NOT FLUX.2 proof.
FLUX.2 uses the existing tiny native configuration from test_training_flow_images_cpu.
No environment creation, pretrained assets, GPU or factory admission is involved.
"""
from datetime import timedelta
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('DMD_MODERN_FLOW_CPU') != '1',
    reason='opt in to focused native CPU tests with DMD_MODERN_FLOW_CPU=1')


def _case(family):
    import diffusers
    if family == 'flux2':
        cls = getattr(diffusers, 'Flux2Transformer2DModel', None)
        if cls is None:
            pytest.skip('actual Flux2 class requires the existing modern-flow target lane')
        config = dict(patch_size=1, in_channels=16, out_channels=1,
            num_layers=1, num_single_layers=1, num_attention_heads=2, attention_head_dim=16,
            joint_attention_dim=48, timestep_guidance_channels=16, mlp_ratio=2,
            axes_dims_rope=[4, 4, 4, 4], guidance_embeds=False)
    else:
        cls = diffusers.FluxTransformer2DModel
        config = dict(patch_size=1, in_channels=4, out_channels=1, num_layers=1,
            num_single_layers=1, num_attention_heads=2, attention_head_dim=8,
            joint_attention_dim=16, pooled_projection_dim=16, axes_dims_rope=[2, 2, 4])
    return cls, config


def _arguments(family):
    import torch
    if family == 'flux2':
        return dict(hidden_states=torch.linspace(-.5, .5, 64).reshape(1, 4, 16),
            encoder_hidden_states=torch.linspace(-.2, .7, 144).reshape(1, 3, 48),
            timestep=torch.tensor([.4]), img_ids=torch.tensor([[0, 0, 0, 0], [0, 0, 1, 0],
                [0, 1, 0, 0], [0, 1, 1, 0]]), txt_ids=torch.zeros(3, 4), return_dict=False)
    return dict(hidden_states=torch.linspace(-.5, .5, 16).reshape(1, 4, 4),
        encoder_hidden_states=torch.linspace(-.2, .7, 48).reshape(1, 3, 16),
        pooled_projections=torch.linspace(-.1, .4, 16).reshape(1, 16),
        timestep=torch.tensor([.4]), img_ids=torch.tensor([[0, 0, 0], [0, 0, 1],
            [0, 1, 0], [0, 1, 1]]), txt_ids=torch.zeros(3, 3), return_dict=False)


def _spec(base, root):
    from distillschedule.workloads.best_known.registry import file_sha256
    return dict(path=str(root.relative_to(base)), files={str(p.relative_to(root)): file_sha256(p)
        for p in root.rglob('*') if p.is_file()}, max_parameters=100_000)


def _checkpoint(base, family, dtype='float32'):
    import torch
    cls, config = _case(family)
    with torch.random.fork_rng():
        torch.manual_seed(41)
        model = cls(**config).eval().to(getattr(torch, dtype))
        root = base / 'native'
        model.save_pretrained(root, safe_serialization=True, max_shard_size='1KB')
    spec = _spec(base, root)
    if dtype == 'bfloat16':
        spec['dtype_policy'] = 'bf16_or_fp32_to_fp32'
    return spec


def _init(rank, root):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE='2', GROUP_RANK='0')
    dist.init_process_group('gloo', init_method='file://' + str(root / 'rdzv'),
        rank=rank, world_size=2, timeout=timedelta(seconds=35))


def _assert_rng(before):
    import torch
    from distillschedule.training.checkpoint import rng_state
    after = rng_state(device='cpu')
    assert torch.equal(before['torch'], after['torch'])
    assert before['python'] == after['python'] and before.get('numpy') == after.get('numpy')


class _Slice:
    def __init__(self, value, key, reads, fail):
        self.value, self.key, self.reads, self.fail = value, key, reads, fail

    def get_shape(self):
        return self.value.get_shape()

    def get_dtype(self):
        return self.value.get_dtype()

    def __getitem__(self, slices):
        if self.fail:
            raise OSError('rank-local slice IO failure')
        assert isinstance(slices, tuple) and all(isinstance(s, slice) for s in slices)
        value = self.value[slices]
        self.reads.append(dict(key=self.key, shape=self.get_shape(), local_shape=list(value.shape),
            offsets=[s.start for s in slices], elements=value.numel()))
        return value


class _SafeOpen:
    def __init__(self, handle, reads, fail=False):
        self.handle, self.reads, self.fail = handle, reads, fail

    def __enter__(self):
        self.handle.__enter__()
        return self

    def __exit__(self, *args):
        return self.handle.__exit__(*args)

    def keys(self):
        return self.handle.keys()

    def get_tensor(self, *args, **kwargs):
        raise AssertionError('sharded loading must not request a full tensor')

    def get_slice(self, key):
        return _Slice(self.handle.get_slice(key), key, self.reads, self.fail)


def _observe(monkeypatch, *, fail=False):
    import safetensors
    import safetensors.torch
    import torch
    from torch.distributed.tensor import DTensor
    original, empty = safetensors.safe_open, torch.nn.Module.to_empty
    reads, owners = [], []

    def forbidden(*args, **kwargs):
        raise AssertionError('no full checkpoint/tensor loading or dense constructor fallback')

    def to_empty(model, *args, **kwargs):
        value = empty(model, *args, **kwargs)
        owners.append((model, {name: (id(p), p.to_local().untyped_storage().data_ptr())
            for name, p in model.named_parameters(remove_duplicate=False)}))
        return value

    reset = torch.nn.Linear.reset_parameters

    def meta_reset(module):
        assert module.weight.is_meta, 'native construction allocated dense concrete parameters'
        return reset(module)

    monkeypatch.setattr(safetensors, 'safe_open', lambda *a, **kw: _SafeOpen(original(*a, **kw), reads, fail))
    monkeypatch.setattr(safetensors.torch, 'load_file', forbidden)
    monkeypatch.setattr(DTensor, 'full_tensor', forbidden)
    monkeypatch.setattr(torch.nn.Module, 'to_empty', to_empty)
    monkeypatch.setattr(torch.nn.Linear, 'reset_parameters', meta_reset)
    return reads, owners


def _happy_worker(rank, root, family, spec):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    from distillschedule.training.checkpoint import rng_state
    from distillschedule.training.flow_image import _load_native_frozen_shards, load_native
    root = Path(root)
    _init(rank, root)
    try:
        cls, _ = _case(family)
        dense, raw = load_native(cls, root / spec['path'], spec['max_parameters'],
            dtype_policy=spec.get('dtype_policy', 'strict_fp32'))
        with torch.no_grad():
            expected = dense(**_arguments(family))[0]
        mesh = init_device_mesh('cpu', (2,))
        before = rng_state(device='cpu')
        with pytest.MonkeyPatch.context() as patch:
            reads, owners = _observe(patch)
            model, actual_raw = _load_native_frozen_shards(cls, root, spec, mesh)
            assert actual_raw == raw and len(owners) == 1 and owners[0][0] is model
            _assert_rng(before)
            assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
            empty, uneven = False, False
            for name, p in model.named_parameters(remove_duplicate=False):
                assert isinstance(p, DTensor) and not p.is_meta
                assert (id(p), p.to_local().untyped_storage().data_ptr()) == owners[0][1][name]
                shape, offsets = compute_local_shape_and_global_offset(p.shape, mesh, p.placements)
                slices = tuple(slice(o, o + n) for o, n in zip(offsets, shape))
                assert torch.equal(p.to_local(), dense.state_dict()[name][slices])
                empty |= p.to_local().numel() == 0
                uneven |= p.shape[0] % 2 != 0
            assert reads and uneven
            coverage = [None, None]
            dist.all_gather_object(coverage, dict(elements=sum(r['elements'] for r in reads), empty=empty))
            assert any(row['empty'] for row in coverage)
            assert sum(row['elements'] for row in coverage) == sum(v.numel() for v in dense.state_dict().values())
            with torch.no_grad():
                actual = model(**_arguments(family))[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            (root / f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank, family=family,
                actual_class=cls.__module__ + '.' + cls.__name__, reads=reads, coverage=coverage,
                exact_forward=True, parameter_and_storage_identity=True, rng_unchanged=True,
                cuda_initialized=torch.cuda.is_initialized()), indent=2))
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
@pytest.mark.parametrize('family', ['flux1', 'flux2'])
def test_actual_native_frozen_shard_loader_cpu2(tmp_path, family, dtype):
    import torch.multiprocessing as mp
    spec = _checkpoint(tmp_path, family, dtype)
    mp.spawn(_happy_worker, args=(str(tmp_path), family, spec), nprocs=2, join=True)


@pytest.mark.parametrize('family', ['flux1', 'flux2'])
def test_default_native_load_constructor_rng_and_failure_unchanged(tmp_path, family):
    import torch
    from safetensors.torch import load_file, save_file
    from distillschedule.training.flow_image import architecture, load_native, shell
    cls, _ = _case(family)
    spec = _checkpoint(tmp_path, family)
    root = tmp_path / spec['path']
    raw = json.loads((root / 'config.json').read_text())
    torch.manual_seed(209)
    shell(cls, raw).float()
    with torch.device('cpu'):
        cls(**architecture(cls, raw)).float()
    expected = torch.get_rng_state().clone()
    torch.manual_seed(209)
    loaded, _ = load_native(cls, root, 100_000)
    assert torch.equal(torch.get_rng_state(), expected)
    assert not loaded.training
    final = sorted(root.glob('*.safetensors'))[-1]
    payload = load_file(final)
    payload[next(iter(payload))].fill_(float('nan'))
    save_file(payload, final)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='nonfinite'):
        load_native(cls, root, 100_000)
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def _failure_worker(rank, root, family, jobs):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
    from torch.distributed.tensor import Shard
    import torch.distributed.fsdp as fsdp
    from distillschedule.training import flow_image
    from distillschedule.training.checkpoint import rng_state
    from distillschedule.training.flow_image import _load_native_frozen_shards, _unpublished_native_shard_forward
    root = Path(root)
    _init(rank, root)
    try:
        mesh = init_device_mesh('cpu', (2,))
        cls, _ = _case(family)
        rows = []
        for kind, choices in jobs.items():
            spec = choices[rank]
            before = rng_state(device='cpu')
            selected_mesh = DeviceMesh('cpu', mesh.mesh.reshape(1, 2), _init_backend=False) \
                if kind == 'asymmetric-mesh' and rank == 1 else mesh
            with pytest.MonkeyPatch.context() as patch:
                reads, owners = _observe(patch, fail=kind == 'asymmetric-io' and rank == 1)
                transform = None
                if kind == 'asymmetric-effective-config' and rank == 1:
                    def transform(config):
                        import random
                        import numpy as np
                        random.random(); np.random.random(); torch.rand(1)
                        config['axes_dims_rope'] = [2, 2, 4, 8] if family == 'flux2' else [4, 2, 2]
                        return config
                if kind == 'asymmetric-placement' and rank == 1:
                    original = fsdp.fully_shard
                    patch.setattr(fsdp, 'fully_shard', lambda model, **kw: original(model, **kw,
                        shard_placement_fn=lambda p: Shard(1) if p.ndim > 1 else Shard(0)))
                if kind == 'asymmetric-rehash' and rank == 1:
                    original_hash = flow_image.file_sha256
                    patch.setattr(flow_image, 'file_sha256', lambda path: '0' * 64 if owners else original_hash(path))
                with pytest.raises(ValueError, match='rank failures|ranks.*differ') as failure:
                    _load_native_frozen_shards(cls, root, spec, selected_mesh, config_transform=transform)
                if kind == 'asymmetric-placement':
                    assert 'unsupported actual native shard mesh/placement' in str(failure.value)
                _assert_rng(before)
                for owner, _ in owners:
                    assert owner._native_shard_load_status == 'poisoned'
                    assert next(iter(owner._forward_pre_hooks.values())) is _unpublished_native_shard_forward
                    with pytest.raises(ValueError, match='not published'):
                        owner(**_arguments(family))
                rows.append(dict(kind=kind, error=str(failure.value), reads=len(reads), poisoned=len(owners)))
            # All ranks reach this after each failure; no rank is left in FSDP collectives.
            dist.barrier()
        model, _ = _load_native_frozen_shards(cls, root, jobs['asymmetric-io'][rank], mesh)
        with torch.no_grad():
            assert bool(torch.isfinite(model(**_arguments(family))[0]).all())
        (root / f'failures-rank-{rank}.json').write_text(json.dumps(rows, indent=2))
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('family', ['flux1', 'flux2'])
def test_native_shard_errors_are_collective_and_poison_unpublished_target(tmp_path, family):
    from copy import deepcopy
    import shutil
    import torch
    import torch.multiprocessing as mp
    from safetensors.torch import load_file, save_file
    good = _checkpoint(tmp_path, family)
    jobs = {}
    for kind in ('wronghash', 'shape', 'dtype', 'missing', 'nonfinite', 'asymmetric-io',
                 'asymmetric-mesh', 'asymmetric-placement', 'asymmetric-rehash', 'asymmetric-effective-config'):
        if kind.startswith('asymmetric-'):
            jobs[kind] = [good, good]
            continue
        target = tmp_path / kind
        shutil.copytree(tmp_path / 'native', target)
        path = next(target.glob('*.safetensors'))
        payload = load_file(path)
        key = next(iter(payload))
        if kind == 'wronghash':
            bad = deepcopy(good)
            bad['files'][path.name] = '0' * 64
            jobs[kind] = [good, bad]
            continue
        if kind == 'shape':
            payload[key] = torch.zeros(1)
        elif kind == 'dtype':
            payload[key] = payload[key].half()
        elif kind == 'missing':
            del payload[key]
        else:
            # Put NaN only in rank 1's actual slice of one genuine native weight.
            key = next(k for k, value in payload.items() if value.shape[0] >= 2)
            payload[key][-1].fill_(float('nan'))
        save_file(payload, path)
        bad = _spec(tmp_path, target)
        jobs[kind] = [bad, bad] if kind == 'nonfinite' else [good, bad]
    mp.spawn(_failure_worker, args=(str(tmp_path), family, jobs), nprocs=2, join=True)


def _slot_class():
    """Supplemental real nn.Parameter aliases/constants, not the native-model proof."""
    import torch
    from torch import nn

    class NativeSlots(nn.Module):
        def __init__(self, constant_mode='cpu'):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(3, 2))
            nn.init.normal_(self.weight)
            self.alias = self.weight
            self.small = nn.Parameter(torch.ones(1))
            kwargs = {'device': 'cpu'} if constant_mode == 'cpu' else {}
            self.register_buffer('frequency', torch.arange(3, **kwargs).float(), persistent=False)
            self.phase = torch.tensor([.25], **kwargs)

        def forward(self, value):
            return value @ self.weight.t() + self.frequency + self.phase + self.small

    return NativeSlots


def _slots_worker(rank, root):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from distillschedule.training.flow_image import _load_native_frozen_shards
    root = Path(root)
    _init(rank, root)
    try:
        mesh = init_device_mesh('cpu', (2,))
        cls = _slot_class()
        for kind in ('omitted', 'both-equal', 'contradictory', 'meta-constant'):
            path = root / kind
            spec = _spec(root, path)
            with pytest.MonkeyPatch.context() as patch:
                _, owners = _observe(patch)
                if kind in ('contradictory', 'meta-constant'):
                    with pytest.raises(ValueError, match='contradictory|constant'):
                        _load_native_frozen_shards(cls, root, spec, mesh)
                else:
                    model, _ = _load_native_frozen_shards(cls, root, spec, mesh)
                    assert model.weight is model.alias
                    assert torch.equal(model.frequency, torch.arange(3).float())
                    assert torch.equal(model.phase, torch.tensor([.25]))
                    assert owners[0][1]['weight'] == owners[0][1]['alias']
                    with torch.no_grad():
                        actual = model(torch.ones(1, 2))
                    torch.testing.assert_close(actual, torch.tensor([[2.25, 7.25, 12.25]]), rtol=0, atol=0)
            dist.barrier()
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_native_alias_slots_and_nonpersistent_constants_cpu2(tmp_path):
    import torch
    import torch.multiprocessing as mp
    from safetensors.torch import save_file
    for kind in ('omitted', 'both-equal', 'contradictory', 'meta-constant'):
        path = tmp_path / kind
        path.mkdir()
        config = dict(_class_name='NativeSlots', constant_mode='meta' if kind == 'meta-constant' else 'cpu')
        (path / 'config.json').write_text(json.dumps(config))
        value = torch.arange(6).float().reshape(3, 2)
        save_file(dict(weight=value, small=torch.ones(1)), path / 'a.safetensors')
        index = dict(weight_map={'weight': 'a.safetensors', 'small': 'a.safetensors'})
        if kind != 'omitted':
            save_file(dict(alias=value + (1 if kind == 'contradictory' else 0)), path / 'b.safetensors')
            index['weight_map']['alias'] = 'b.safetensors'
        (path / 'weights.safetensors.index.json').write_text(json.dumps(index))
    mp.spawn(_slots_worker, args=(str(tmp_path),), nprocs=2, join=True)


def _t5_worker(rank, root, specs):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import T5EncoderModel
    from distillschedule.training.flow_image import _load_native_frozen_shards, load_native
    root = Path(root)
    _init(rank, root)
    try:
        mesh = init_device_mesh('cpu', (2,))
        arguments = dict(input_ids=torch.tensor([[3, 4, 5, 0]]), attention_mask=torch.tensor([[1, 1, 1, 0]]))
        receipts = []
        for kind, spec in specs.items():
            if kind != 'contradictory':
                dense, _ = load_native(T5EncoderModel, root / spec['path'], 100_000,
                    dtype_policy=spec.get('dtype_policy', 'strict_fp32'))
                with torch.no_grad():
                    expected = dense(**arguments).last_hidden_state
            before = torch.get_rng_state().clone()
            with pytest.MonkeyPatch.context() as patch:
                reads, _ = _observe(patch)
                if kind == 'contradictory':
                    with pytest.raises(ValueError, match='contradictory'):
                        _load_native_frozen_shards(T5EncoderModel, root, spec, mesh)
                else:
                    model, _ = _load_native_frozen_shards(T5EncoderModel, root, spec, mesh)
                    assert model.shared.weight is model.encoder.embed_tokens.weight
                    with torch.no_grad():
                        actual = model(**arguments).last_hidden_state
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert torch.equal(before, torch.get_rng_state())
                receipts.append(dict(kind=kind, reads=len(reads), real_native_tied_slot=True))
            dist.barrier()
        (root / f't5-rank-{rank}.json').write_text(json.dumps(receipts, indent=2))
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_actual_t5_native_alias_slots_cpu2(tmp_path, dtype):
    # The shared loader intentionally uses native Transformers5 renaming ops.
    # Do not inject a fake TF5 conversion stack into the local TF4 environment.
    pytest.importorskip('transformers.conversion_mapping')
    import shutil
    import torch
    import torch.multiprocessing as mp
    from transformers import T5Config, T5EncoderModel
    from safetensors.torch import load_file, save_file
    model = T5EncoderModel(T5Config(vocab_size=33, d_model=16, d_ff=32, d_kv=8,
        num_layers=1, num_heads=2, dropout_rate=0.)).eval().to(getattr(torch, dtype))
    model.save_pretrained(tmp_path / 'omitted', safe_serialization=True, max_shard_size='1KB')
    specs = {}
    for kind in ('omitted', 'both-equal', 'contradictory'):
        path = tmp_path / kind
        if kind != 'omitted':
            shutil.copytree(tmp_path / 'omitted', path)
            index_path = next(path.glob('*.safetensors.index.json'))
            index = json.loads(index_path.read_text())
            other = next(p for p in sorted(path.glob('*.safetensors')) if p.name != index['weight_map']['shared.weight'])
            payload = load_file(other)
            value = model.shared.weight.detach().clone()
            if kind == 'contradictory':
                value[-1].add_(1)  # Only rank 1 owns this contradictory alias piece.
            payload['encoder.embed_tokens.weight'] = value
            save_file(payload, other)
            index['weight_map']['encoder.embed_tokens.weight'] = other.name
            index_path.write_text(json.dumps(index))
        specs[kind] = _spec(tmp_path, path)
        if dtype == 'bfloat16':
            specs[kind]['dtype_policy'] = 'bf16_or_fp32_to_fp32'
    mp.spawn(_t5_worker, args=(str(tmp_path), specs), nprocs=2, join=True)


def _class_constant_native(kind, rank):
    """Constructor-state counterexamples using real nn.Module/FSDP, not mocks."""
    import torch
    from torch import nn
    descriptor_calls = []

    class GetBomb:
        def __get__(self, instance, owner):
            descriptor_calls.append('descriptor')
            raise AssertionError('arbitrary descriptor was evaluated')

    class ConstantBase:
        pass

    class NativeClassState(ConstantBase, nn.Module):
        inactive_descriptor = GetBomb()

        @property
        def inactive_property(self):
            descriptor_calls.append('property')
            raise AssertionError('arbitrary property was evaluated')

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 2))

        def forward(self, value):
            return value @ self.weight + self.phase

    value = torch.ones(1, device='cpu' if kind.endswith('cpu') else 'meta')
    owner = ConstantBase if kind.startswith('inherited') or kind == 'asymmetric-inherited' else NativeClassState
    if kind == 'descriptors-unread' or (kind == 'asymmetric-inherited' and rank == 0):
        value = 0.
    elif kind == 'nested-container':
        value = {'nested': [([], (value,))]}
    elif kind == 'set-container':
        value = {value}
    elif kind == 'frozenset-container':
        value = frozenset([value])
    elif kind == 'cyclic-container':
        value = {'tensor': value}
        value['cycle'] = value
    owner.phase = value
    return NativeClassState, descriptor_calls


def _class_constant_worker(rank, root):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from distillschedule.training.checkpoint import rng_state
    from distillschedule.training.flow_image import _load_native_frozen_shards, _unpublished_native_shard_forward
    root = Path(root)
    _init(rank, root)
    try:
        mesh = init_device_mesh('cpu', (2,))
        spec = _spec(root, root / 'native')
        receipts = []
        for kind in ('descriptors-unread', 'meta', 'cpu', 'inherited-meta', 'inherited-cpu',
                     'nested-container', 'set-container', 'frozenset-container', 'cyclic-container',
                     'asymmetric-inherited'):
            cls, descriptor_calls = _class_constant_native(kind, rank)
            before = rng_state(device='cpu')
            with pytest.MonkeyPatch.context() as patch:
                reads, owners = _observe(patch)
                if kind == 'descriptors-unread':
                    model, _ = _load_native_frozen_shards(cls, root, spec, mesh)
                    with torch.no_grad():
                        actual = model(torch.ones(1, 2))
                    torch.testing.assert_close(actual, torch.full((1, 2), 2.), rtol=0, atol=0)
                else:
                    with pytest.raises(ValueError, match='class-owned native tensor') as failure:
                        model, _ = _load_native_frozen_shards(cls, root, spec, mesh)
                        (root / f'unexpected-ready-rank-{rank}.json').write_text(json.dumps(dict(
                            case=kind, status=model._native_shard_load_status,
                            phase_is_meta=model.phase.is_meta if isinstance(model.phase, torch.Tensor) else None,
                            descriptor_calls=descriptor_calls, cuda_initialized=torch.cuda.is_initialized()), indent=2))
                    assert not reads, 'class-owned constants must fail before any checkpoint payload read'
                    for model, _ in owners:
                        assert model._native_shard_load_status == 'poisoned'
                        assert next(iter(model._forward_pre_hooks.values())) is _unpublished_native_shard_forward
                        with pytest.raises(ValueError, match='not published'):
                            model(torch.ones(1, 2))
                assert not descriptor_calls
                _assert_rng(before)
                receipts.append(dict(case=kind, descriptor_calls=descriptor_calls, reads=len(reads),
                    result='ready' if kind == 'descriptors-unread' else 'collective-rejection',
                    allocated_roots=len(owners)))
            dist.barrier()
        (root / f'class-constant-rank-{rank}.json').write_text(json.dumps(receipts, indent=2))
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_class_and_inherited_tensor_constants_rejected_without_descriptors_cpu2(tmp_path):
    import torch
    import torch.multiprocessing as mp
    from safetensors.torch import save_file
    root = tmp_path / 'native'
    root.mkdir()
    (root / 'config.json').write_text(json.dumps(dict(_class_name='NativeClassState')))
    save_file(dict(weight=torch.ones(2, 2)), root / 'weights.safetensors')
    mp.spawn(_class_constant_worker, args=(str(tmp_path),), nprocs=2, join=True)
