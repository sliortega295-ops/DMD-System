"""Real native two-block LTX CPU/Gloo acceptance through the public lifecycle.

Reuses the independent eager rank-mean oracle and public fresh-process CLI.
No model substitutes, provider monkeypatches, new trainer or relaxed tolerance.
"""
from copy import deepcopy
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
from unittest.mock import patch
import weakref

import torch

from distillschedule.training import distributed
from distillschedule.training.checkpoint import cpu_tree, pack_state, rng_state
from distillschedule.training.config import load_training_config
from distillschedule.training.session import construct, build_cycle
from distillschedule.training.ltx_sharding import CONSTRUCTION, EXECUTION, ROLES
from flow_replica_oracle import sequential_b1_oracle, observe_gradients
from native_flow_replicated import (
    deterministic, _write, _close, _assert_native, _frozen, _student_mean_norm,
    fresh_cli, _single_cli, ORACLE_CYCLES, ORACLE_RTOL, ORACLE_ATOL)
from test_open_workload_parameter_shadows import _snapshot
from test_training_end_to_end import equal


def _boundaries():
    return {role: ['native.transformer_blocks.0', 'native.transformer_blocks.1'] for role in ROLES}


def snapshot(state):
    return _snapshot(state if 'student_ema' in state else {**state, 'student_ema': {}})


def owner_refusals(model, workload, config, data):
    """Reject foreign owners before the first native group is installed."""
    import pytest
    provider = model.sharded_construction
    from distillschedule.training import ltx_sharding
    receipt = ltx_sharding._provider_source()
    assert model.metadata['construction']['provider_source'] == receipt
    assert workload.identity_extra['native_ltx_block_construction']['provider_source'] == receipt
    with patch.object(ltx_sharding, '_provider_source', lambda: {**receipt, 'sha256': '0' * 64}):
        with pytest.raises(ValueError, match='provider source identity'):
            provider.bind_workload(model, workload)
    precision = config.model['parameters'].get('compute_precision', 'fp32')
    other = 'bf16-mixed' if precision == 'fp32' else 'fp32'
    for role in ROLES:
        with patch.object(model.state[role], 'compute_precision', other):
            with pytest.raises(ValueError, match='compute precision'):
                provider.bind_workload(model, workload)
    with patch.object(provider, 'compute_precision', other):
        with pytest.raises(ValueError, match='compute precision'):
            provider.bind_workload(model, workload)
    for values in (provider.parameters, model.metadata, model.metadata['construction']):
        with patch.dict(values, compute_precision=other):
            with pytest.raises(ValueError, match='compute precision'):
                provider.bind_workload(model, workload)
    with patch.dict(model.state, flow_compute_precision=other):
        with pytest.raises(ValueError, match='compute precision'):
            provider.bind_workload(model, workload)
    wrong_request = replace(config, model=dict(config.model,
        parameters=dict(config.model['parameters'], compute_precision=other)))
    with pytest.raises(ValueError, match='compute precision'):
        model.admit_training(state=model.state, request=wrong_request,
            prepared_identity=data.identity, prepared_manifest=data.manifest)
    root = model.state['student']
    with pytest.raises(ValueError, match='provider/factory/parameters'):
        provider.construct(provider.factory, provider.parameters)
    with pytest.raises(ValueError, match='single-use'):
        provider.attach(model)
    with pytest.raises(ValueError, match='provider/root'):
        provider.bind_workload(model, replace(workload, state={**workload.state, 'student': model.state['fake']}))
    source, native = root.source, root.native
    try:
        root.source = object()
        with pytest.raises(ValueError, match='source/native/child'):
            provider.bind_workload(model, workload)
    finally:
        root.source = source
    try:
        root.native = model.state['fake'].native
        with pytest.raises(ValueError, match='source/native/child'):
            provider.bind_workload(model, workload)
    finally:
        root.native = native
    child = native.transformer_blocks[0]
    try:
        native.transformer_blocks[0] = native.transformer_blocks[1]
        with pytest.raises(ValueError, match='source/native/child'):
            provider.bind_workload(model, workload)
    finally:
        native.transformer_blocks[0] = child
    name, parameter = next(iter(root.named_parameters()))
    parent, leaf = name.rsplit('.', 1)
    parent = root.get_submodule(parent)
    try:
        setattr(parent, leaf, torch.nn.Parameter(parameter.detach().clone(), requires_grad=parameter.requires_grad))
        with pytest.raises(ValueError, match='Parameter identity'):
            provider.bind_workload(model, workload)
    finally:
        setattr(parent, leaf, parameter)


def assert_sharded(model, session):
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor
    from distillschedule.training.tensor_state import ParameterShadows, layout
    effects = session.effects
    assert effects['fsdp_shard_boundaries'] == _boundaries()
    assert effects['fsdp_forward_methods'] == {role: 'dmd_forward' for role in ROLES}
    assert effects.get('parameter_shadows', {}) == ({'student_ema': 'student'} if 'student_ema' in model.state else {})
    assert effects['frozen_only_models'] == {'sharded': ['teacher'], 'resident': ['media']}
    assert effects['lifecycle_installations'] == 1
    assert effects['reduction'] == 'fsdp2_reshard' and effects['reshard_after_forward']
    assert effects['shared_parameter_aliases'] == 0
    assert not isinstance(model.state['media'], FSDPModule)
    assert all(p.dtype == torch.float32 for p in model.state['media'].parameters())
    for role in ROLES:
        root = model.state[role]
        assert isinstance(root, FSDPModule) and not isinstance(root.native, FSDPModule)
        assert len(root.native.transformer_blocks) == 2
        assert all(isinstance(block, FSDPModule) for block in root.native.transformer_blocks)
        assert sum(isinstance(child, FSDPModule) for child in root.modules()) == 3
        assert all(isinstance(p, DTensor) and p.grad is None and p.dtype == torch.float32
                   for p in root.parameters())
        if role == 'teacher':
            assert not any(p.requires_grad for p in root.parameters())
        else:
            optimizer = model.state[role + '_optimizer']
            assert {id(p) for p in root.parameters() if p.requires_grad} == {
                id(p) for group in optimizer.param_groups for p in group['params']}
            assert optimizer.state and all(value.dtype == torch.float32 for slot in optimizer.state.values()
                for value in slot.values() if isinstance(value, torch.Tensor))
    shadows = model.state.get('student_ema')
    if shadows is None:
        return dict(native_groups_per_role=3, actual_child_paths=_boundaries(), ema='DISABLED_UNCHANGED',
                    frozen_teacher_gradients=False, media_resident=True)
    assert type(shadows) is ParameterShadows and shadows.owner == 'student'
    assert shadows is session.state['student_ema'] is session.workload.state['student_ema']
    for name, parameter in model.state['student'].named_parameters():
        assert (isinstance(shadows[name], DTensor) and layout(shadows[name]) == layout(parameter)
                and shadows[name].dtype == torch.float32)
    full_bytes = sum(value.numel() * value.element_size() for value in shadows.values())
    local_bytes = sum(value.to_local().numel() * value.element_size() for value in shadows.values())
    assert local_bytes < full_bytes
    coverage = torch.tensor(local_bytes, dtype=torch.int64)
    torch.distributed.all_reduce(coverage)
    assert int(coverage) == full_bytes
    return dict(native_groups_per_role=3, actual_child_paths=_boundaries(), ema_local_bytes=local_bytes,
                ema_full_bytes=full_bytes, frozen_teacher_gradients=False, media_resident=True)


def oracle(config_path, output):
    from distillschedule.open_workload.execution import ExecutionBackend
    import torch.distributed.fsdp as fsdp
    config = load_training_config(config_path)
    root = Path(output)
    with distributed.context(config):
        rank, world = distributed.rank_world()
        assert world == 2
        precision = config.model['parameters'].get('compute_precision', 'fp32')
        import pytest
        from distillschedule.training.session import _construct_model
        before_rng = rng_state(device='cpu')
        mismatched = replace(config, model=dict(config.model, parameters=dict(config.model['parameters'],
            compute_precision='fp32' if rank == 0 else 'bf16-mixed')))
        with pytest.raises(ValueError, match='ranks selected different native block compute precision'):
            _construct_model(mismatched)
        assert equal(before_rng, rng_state(device='cpu'))
        distributed.primary(lambda: root.mkdir(parents=True, exist_ok=False), 'LTX block oracle output')
        parameters = dict(config.model['parameters']); parameters.pop('construction')
        dense_config = replace(config, model=dict(config.model, parameters=parameters), execution=dict(mode='reference'))
        dense, data, algorithm, _ = construct(dense_config)
        _assert_native(dense)
        from torch.utils._pytree import tree_leaves
        flattened_before = []
        def observe_flattening(records):
            def hook(module, args, value):
                records.append(dict(tensor_leaves=sum(isinstance(leaf, torch.Tensor) for leaf in tree_leaves(value)),
                    native_tensor_fields=isinstance(value[0].x, torch.Tensor) and isinstance(value[1].x, torch.Tensor)))
            return hook
        before_hook = dense.state['student'].native.transformer_blocks[0].register_forward_hook(observe_flattening(flattened_before))
        initial, frozen, initial_rng = snapshot(dense.state), _frozen(dense.state), rng_state(device='cpu')
        expected = []
        for cycle in range(ORACLE_CYCLES):
            workload = build_cycle(dense_config, dense, data, algorithm, cycle=cycle, cursor=cycle * world + rank)
            feeds = [data.sample(cycle * world + owner) for owner in range(world)]
            trace = sequential_b1_oracle(dense.state, feeds, config.algorithm['parameters'], workload.seed)
            assert any(not equal(row['local'][0], row['local'][1]) for row in trace['gradients'])
            assert equal(frozen, _frozen(dense.state)) and equal(initial_rng, rng_state(device='cpu'))
            expected.append(dict(state=snapshot(dense.state), trace=trace, seed=workload.seed))
        before_hook.remove()
        assert flattened_before and all(row['tensor_leaves'] == 0 and row['native_tensor_fields'] for row in flattened_before)
        norms = [_student_mean_norm(row['trace']['gradients']) for row in expected]
        assert any(value > 0 for value in norms[1:])
        del dense, data, algorithm, workload
        gc.collect()
        model, data, algorithm, _ = construct(config)
        _assert_native(model)
        assert equal(initial, snapshot(model.state)) and equal(initial_rng, rng_state(device='cpu'))
        assert model.metadata['construction']['schema_version'] == CONSTRUCTION
        session, versions, records = None, None, []
        installed, registered = [], []
        flattened_after = []
        native_compute = {role: dict(calls=0, bf16_outputs=0, autocast_calls=0) for role in ROLES}
        def native_linear(role):
            def hook(module, args, value):
                row = native_compute[role]
                row['calls'] += 1
                row['bf16_outputs'] += int(value.dtype == torch.bfloat16)
                row['autocast_calls'] += int(torch.is_autocast_enabled('cpu'))
                assert value.dtype == (torch.bfloat16 if precision == 'bf16-mixed' else torch.float32)
            return hook
        compute_handles = [module.register_forward_hook(native_linear(role)) for role in ROLES
            for module in model.state[role].native.modules() if type(module) is torch.nn.Linear]
        after_hook = model.state['student'].native.transformer_blocks[0].register_forward_hook(observe_flattening(flattened_after))
        fully_shard, register = fsdp.fully_shard, fsdp.register_fsdp_forward_method
        def shard(module, *args, **kwargs):
            installed.append(id(module))
            return fully_shard(module, *args, **kwargs)
        def method(module, name):
            registered.append((id(module), name))
            return register(module, name)
        try:
            for cycle, target in enumerate(expected):
                workload = build_cycle(config, model, data, algorithm, cycle=cycle,
                    cursor=cycle * world + rank, versions=versions)
                backend = ExecutionBackend(workload)
                if session is None:
                    owner_refusals(model, workload, config, data)
                    with patch.object(fsdp, 'fully_shard', shard), patch.object(fsdp, 'register_fsdp_forward_method', method):
                        session = backend.session(deepcopy(EXECUTION['manual_implementation']), copy_state=False, persistent=True)
                    gradients, handles = observe_gradients(session)
                    assert len(installed) == len(set(installed)) == 9
                    assert sorted(registered) == sorted((id(model.state[role]), 'dmd_forward') for role in ROLES)
                else:
                    session.bind_cycle(workload, graph=backend.graph(EXECUTION['manual_implementation']))
                before_rng, offset = rng_state(device='cpu'), len(gradients)
                outputs = session.run()
                versions = session.receipt['dispatcher']['final_versions']
                observed = gradients[offset:]
                assert all(value.dtype == torch.float32 for value in tree_leaves(observed)
                           if isinstance(value, torch.Tensor))
                _close(target['state'], snapshot(model.state))
                _close(target['trace']['losses'][rank], cpu_tree(outputs))
                _close([{key: value for key, value in row.items() if key != 'local'}
                        for row in target['trace']['gradients']], observed)
                assert any(not equal(row['mean'], row['clipped']) for row in observed)
                assert equal(frozen, _frozen(model.state)) and equal(before_rng, rng_state(device='cpu'))
                assert model.state['dmd_counters'].tolist() == [cycle + 1, cycle + 1, 5 * (cycle + 1)]
                assert versions['teacher'] == 0
                if 'student_ema' in model.state:
                    assert versions['student_ema'] == cycle + 1
                ownership = assert_sharded(model, session)
                torch.save(dict(expected=target, gradients=observed, outputs=cpu_tree(outputs)), root/f'gradient-{rank}-{cycle}.pt')
                records.append(dict(cycle=cycle, sample_id=data.sample_id(cycle * world + rank), seed=workload.seed,
                    student_mean_gradient_l2=norms[cycle], effects=deepcopy(session.effects), ownership=ownership))
            assert flattened_after and all(row['tensor_leaves'] > 0 and row['native_tensor_fields'] for row in flattened_after)
            proof = dict(rank=rank, world_size=world, construction=CONSTRUCTION,
                compute_precision=precision, native_linear_compute=native_compute,
                fp32_owned_state=True, precision_drift_rejected=True,
                evidence_kind=model.metadata['evidence_kind'], average_before_clip_adam=True,
                oracle_cycles=ORACLE_CYCLES, oracle_rtol=ORACLE_RTOL, oracle_atol=ORACLE_ATOL,
                dense_initial_state_and_rng_exact=True, native_method_registrations=len(registered),
                native_group_installations=len(installed), later_student_gradient_nonzero=True, records=records,
                native_pytree_before=flattened_before[0], native_pytree_after=flattened_after[0],
                cuda_initialized=torch.cuda.is_initialized(), quality='NOT_RUN', full22b='NOT_RUN', performance='NOT_RUN')
            _write(root/f'rank-{rank:05d}.json', proof)
            rows = distributed.gather(proof)
            distributed.primary(lambda: _write(root/'proof.json', dict(ranks=rows)), 'LTX block oracle proof')
        finally:
            if session is not None:
                for handle in handles:
                    handle.remove()
            after_hook.remove()
            for handle in compute_handles:
                handle.remove()
            if model.close is not None:
                model.close()
        import pytest
        provider = model.sharded_construction
        with pytest.raises(ValueError, match='lifetime/provider/root'):
            provider.bind_workload(model, workload)
        model_reference = weakref.ref(model)
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            del model
            assert model_reference() is None, 'closed provider retained component owners until cyclic GC'
        finally:
            if gc_was_enabled:
                gc.enable()
        default, _, _, _ = construct(dense_config)
        assert default.sharded_construction is None and default.close is None
        assert 'construction' not in default.metadata and equal(initial, snapshot(default.state))
        assert ('student_ema' not in default.state or type(default.state['student_ema']) is dict)


def output_probe(config_path, checkpoint, output):
    """Fresh-process restore and native AV generation, without a train update."""
    from distillschedule.training.session import _load_for_output, _generate_values
    config = load_training_config(config_path)
    root = Path(output)
    with distributed.context(config):
        rank, _ = distributed.rank_world()
        model, data, payload, _, session = _load_for_output(config, checkpoint)
        try:
            for module in model.state.values():
                if isinstance(module, torch.nn.Module):
                    module.eval()
            before = pack_state(model.state)
            feeds = data.sample(rank)
            value = _generate_values(config, model, feeds, {}, output_kind='audio_video')
            assert equal(before, pack_state(model.state))
            ownership = assert_sharded(model, session)
            torch.save(dict(student=cpu_tree(model.state['student'].state_dict()), values=cpu_tree(value)),
                       root/f'generation-{rank}.pt')
            assert value['decoded']['video'].shape == (3, 8, 8, 3)
            assert value['decoded']['audio']['samples'].shape == (9840, 2)
            _write(root/f'generation-{rank}.json', dict(current_student=True, native_audio_video=True,
                restored_cycle=payload['cycle'], healthy_shards_after_generation=True, ownership=ownership,
                cuda_initialized=torch.cuda.is_initialized()))
        finally:
            model.close()


def lifecycle(config_path, output):
    from distillschedule.data import PreparedDataset
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    config = load_training_config(config_path)
    for name, cycles, resume in [('partial', 1, None), ('resumed', 3, root/'partial/checkpoints/cycle-000001'), ('whole', 3, None)]:
        args = ['train', config_path, '--cycles', cycles, '--output', root/name]
        if resume:
            args += ['--resume', resume]
        fresh_cli(config_path, root/(name+'.log'), 2, *args)
    checkpoint = root/'resumed/checkpoints/cycle-000003'
    data = PreparedDataset(config.data['prepared'])
    for rank in range(2):
        resumed = torch.load(checkpoint/f'rank-{rank:05d}.pt', weights_only=True)
        whole = torch.load(root/'whole/checkpoints/cycle-000003'/f'rank-{rank:05d}.pt', weights_only=True)
        assert equal(resumed, whole), (rank, 'all owned state/RNG/data/order/checkpoint fields must be exact')
        assert resumed['cycle'] == 3 and resumed['cursor'] == 6
        assert resumed['state']['dmd_counters']['value']['counters'].tolist() == [3, 3, 15]
        if 'student_ema' in resumed['state']:
            assert resumed['state']['student_ema']['value']['schema_version'] == 'distillschedule.parameter_shadows.v1'
        metrics = [json.loads(line) for line in (root/'resumed/ranks'/f'rank-{rank:05d}'/'metrics.jsonl').read_text().splitlines()]
        assert [row['sample_id'] for row in metrics] == [data.sample_id(2 + rank), data.sample_id(4 + rank)]
        assert all(row['execution']['effects']['fsdp_shard_boundaries'] == _boundaries() for row in metrics)
    manifest = json.loads((checkpoint/'manifest.json').read_text())
    assert set(manifest['files']) == {'rank-00000.pt', 'rank-00001.pt'} and (checkpoint/'_SUCCESS').is_file()
    fresh_cli(config_path, root/'export.log', 2, 'export', config_path, '--checkpoint', checkpoint, '--output', root/'export')
    import os
    import subprocess
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2',
        str(Path(__file__).resolve()), 'output', '--config', str(config_path), '--checkpoint', str(checkpoint), '--output', str(root)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, env=dict(os.environ))
    (root/'owned-generation.log').write_text(result.stdout + '\n' + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    from distillschedule.training.inference import read_export
    _, metadata = read_export(root/'export')
    inference = dict(schema_version='distillschedule.training.v1',
        model={key: metadata['model'][key] for key in ('factory', 'parameters')}, algorithm=dict(name='dmd'),
        data=dict(prepared=config.data['prepared']), execution=dict(mode='reference'),
        trainer=dict(device='cpu', cycles=1, seed=config.trainer['seed'], output_dir=str(root/'unused')))
    _write(root/'inference.json', inference)
    asset_root = Path(config.model['parameters']['assets']).parent
    hidden = asset_root.with_name(asset_root.name + '-retired')
    asset_root.rename(hidden)
    try:
        generated = _single_cli(root, 'portable-current-student', 'generate', root/'inference.json',
            '--weights', root/'export', '--prepared', config.data['prepared'], '--output', root/'generated', '--output-kind', 'audio_video')
        artifact = torch.load(generated['artifact'], weights_only=True)
        owned = torch.load(root/'generation-0.pt', weights_only=True)
        _close(owned['values'], artifact)
        from distillschedule.training.plugins import load_callable
        from distillschedule.training.inference import restore_modules
        factory, _ = load_callable(metadata['model']['factory'])
        portable = factory(metadata['model']['parameters'])
        restore_modules(portable, root/'export', metadata)
        assert equal(owned['student'], cpu_tree(portable.state['student'].state_dict()))
        assert artifact['decoded']['video'].shape == (3, 8, 8, 3)
        assert artifact['decoded']['audio']['samples'].shape == (9840, 2)
    finally:
        hidden.rename(asset_root)
    shard = checkpoint/'rank-00001.pt'; hidden_shard = shard.with_suffix('.hidden')
    shard.rename(hidden_shard)
    try:
        fresh_cli(config_path, root/'missing-shard.log', 2, 'train', config_path, '--cycles', 3,
            '--resume', checkpoint, '--output', root/'must-not-exist', expected_error='shard')
        assert not (root/'must-not-exist').exists()
    finally:
        hidden_shard.rename(shard)
    precision = config.model['parameters'].get('compute_precision', 'fp32')
    changed = config.to_dict()
    changed['model']['parameters']['compute_precision'] = 'bf16-mixed' if precision == 'fp32' else 'fp32'
    changed_path = root/'wrong-precision.json'; _write(changed_path, changed)
    fresh_cli(changed_path, root/'wrong-precision.log', 2, 'train', changed_path, '--cycles', 3,
        '--resume', checkpoint, '--output', root/'wrong-precision-must-not-exist', expected_error='identity')
    assert not (root/'wrong-precision-must-not-exist').exists()
    _write(root/'proof.json', dict(world_size=2, all_rank_state_rng_data_order_exact=True,
        fresh_process_restore=True, current_student_native_av_export=True, missing_rank_shard_rejected=True,
        compute_precision=precision, checkpoint_precision_drift_rejected=True,
        quality='NOT_RUN', cuda='NOT_RUN', full22b='NOT_RUN'))


if __name__ == '__main__':
    import argparse
    deterministic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['oracle', 'output'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    if args.stage == 'oracle':
        oracle(args.config, args.output)
    else:
        output_probe(args.config, args.checkpoint, args.output)
