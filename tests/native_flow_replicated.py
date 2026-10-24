"""SYNTHETIC_TEST_ONLY native FLUX/Qwen replicated acceptance harness.

External launch (same entry works for Gloo2, CUDA2 or a parent-owned NCCL16):
  torchrun ... tests/native_flow_replicated.py oracle --config REQUEST --output ROOT
  torchrun ... tests/native_flow_replicated.py cli train REQUEST ...
  torchrun ... tests/native_flow_replicated.py cli export REQUEST ...
Single-host convenience: lifecycle --config REQUEST --output ROOT --ranks 2.
All CLI stages run in fresh processes; no assets are downloaded or roles cloned.
The oracle is sequential independent local B1 gradients averaged before clipping.
"""
from copy import deepcopy
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from distillschedule.training.checkpoint import cpu_tree, pack_state, rng_state
from distillschedule.training.config import load_training_config
from distillschedule.training import distributed
from distillschedule.training.session import construct, build_cycle
from distillschedule.workloads.best_known.registry import fingerprint
from test_training_end_to_end import equal
from test_training_flow_image_device import REPLICATED_MANUAL
from flow_replica_oracle import sequential_b1_oracle, observe_gradients


def deterministic():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # Set before any CUDA context; never alter the production execution policy.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _frozen(state):
    return cpu_tree({role: {name: parameter for name, parameter in state[role].named_parameters(remove_duplicate=False)
                           if not parameter.requires_grad} for role in ('student', 'fake', 'teacher', 'media')})


def _assert_native(components):
    from distillschedule.training.flow_image import ImageRole
    assert components.metadata['evidence_kind'] == 'SYNTHETIC_TEST_ONLY'
    assert all(isinstance(components.state[role], ImageRole) for role in ('student', 'fake', 'teacher'))
    assert components.state['student'].native.__class__.__module__.startswith('diffusers.')


def _close(expected, actual):
    # NCCL trees need not sum 16 FP32 values in Python's sequential order.
    # State/gradient tolerances are explicit; restart/aliases remain bitwise.
    from torch.utils._pytree import tree_flatten
    expected_leaves, expected_tree = tree_flatten(expected)
    actual_leaves, actual_tree = tree_flatten(actual)
    assert actual_tree == expected_tree
    for left, right in zip(expected_leaves, actual_leaves):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(right, left, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        else:
            assert type(left) is type(right) and left == right


ORACLE_CYCLES = 3
ORACLE_RTOL = 2e-5
ORACLE_ATOL = 2e-7


def _student_mean_norm(gradients):
    values = [value for value in gradients[0]['mean'].values() if value is not None]
    assert gradients[0]['role'] == 'student' and values
    return float(sum(value.double().square().sum() for value in values).sqrt())


def oracle(config_path, output):
    """External torchrun; independent native G1/F5 oracle for THREE cycles.

    Cycle zero may have a zero Student gradient because Fake equals Teacher.
    The original initialization is never perturbed: a later cycle MUST have a
    nonzero Student mean gradient, after the real preceding five Fake updates.
    """
    from torch.utils._pytree import tree_map
    from distillschedule.open_workload.execution import ExecutionBackend
    from distillschedule.open_workload.torch_backend import sharing_inventory, resolve_module, signature
    config = load_training_config(config_path)
    root = Path(output)
    with distributed.context(config):
        rank, world = distributed.rank_world()
        assert world >= 2
        distributed.primary(lambda: root.mkdir(parents=True, exist_ok=False), 'oracle output')
        # One independent full-world sequential oracle per process is modest for
        # these actual tiny fixtures; it uses no extra rank controller or clone.
        model, dataset, algorithm, identity = distributed.checked(lambda: construct(config), 'oracle construction')
        distributed.checked(lambda: _assert_native(model), 'actual native fixture')
        initial = pack_state(model.state)
        device = model.state['student'].device
        initial_rng, before_frozen = rng_state(device=device), _frozen(model.state)
        expected_cycles = []
        for cycle in range(ORACLE_CYCLES):
            def independent_cycle():
                assert len(dataset) >= world * ORACLE_CYCLES
                workload = build_cycle(config, model, dataset, algorithm, cycle=cycle, cursor=cycle * world + rank)
                indices = range(cycle * world, (cycle + 1) * world)
                feeds = [tree_map(lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, dataset.sample(index))
                         for index in indices]
                expected = sequential_b1_oracle(model.state, feeds, config.algorithm['parameters'], workload.seed)
                assert equal(before_frozen, _frozen(model.state)) and equal(initial_rng, rng_state(device=device))
                assert any(not equal(row['local'][0], row['local'][1]) for row in expected['gradients'])
                assert not equal(expected['stochastic'][0][3], expected['stochastic'][1][3])
                return dict(state=pack_state(model.state),
                    gradients=[{key: value for key, value in row.items() if key != 'local'} for row in expected['gradients']],
                    losses=expected['losses'], stochastic=expected['stochastic'],
                    sample_ids=[dataset.sample_id(index) for index in indices], seed=workload.seed)
            expected_cycles.append(distributed.checked(independent_cycle, f'independent B1 mean cycle {cycle}'))
        def check_nonzero():
            norms = [_student_mean_norm(row['gradients']) for row in expected_cycles]
            assert all(torch.isfinite(torch.tensor(value)) for value in norms)
            assert any(value > 0 for value in norms[1:]), 'later-cycle Student gradient must genuinely be nonzero'
            return norms
        student_norms = distributed.checked(check_nonzero, 'nonzero Student oracle witness')
        # CPU tensor snapshots only; release the live owner before reconstructing.
        del model, dataset, algorithm
        gc.collect()
        records, reference_cycles = [], []
        for mode in ('reference', 'manual'):
            request = replace(config, execution=dict(mode='reference') if mode == 'reference' else deepcopy(REPLICATED_MANUAL))
            model, dataset, algorithm, _ = distributed.checked(lambda: construct(request), 'runtime construction')
            def check_initial():
                _assert_native(model)
                assert equal(initial, pack_state(model.state)) and equal(initial_rng, rng_state(device=device))
            distributed.checked(check_initial, 'same initial native owner and RNG')
            session, versions, mode_cycles = None, None, []
            for cycle, expected in enumerate(expected_cycles):
                workload = distributed.checked(lambda: build_cycle(request, model, dataset, algorithm,
                    cycle=cycle, cursor=cycle * world + rank, versions=versions), 'runtime workload')
                backend = ExecutionBackend(workload)
                if session is None:
                    binding = backend.identity()
                    assert binding['replica_semantics']['forward_interface'] == 'private_methods'
                    assert binding['replica_semantics']['random_streams'] == 'rank_independent_v1'
                    option = (backend.inventory(1)['candidates'][0] if mode == 'reference'
                              else deepcopy(REPLICATED_MANUAL['implementation']))
                    pairs = sharing_inventory(workload)
                    assert pairs
                    session = backend.session(option, copy_state=False, persistent=True)
                    actual_grad, handles = observe_gradients(session)
                else:
                    distributed.checked(lambda: session.bind_cycle(workload, graph=backend.graph(option)), 'cycle rebind')
                before_rng, offset = rng_state(device=device), len(actual_grad)
                outputs = session.run()
                versions = session.receipt['dispatcher']['final_versions']
                state = pack_state(model.state)
                actual = dict(state=state, gradients=actual_grad[offset:], losses=cpu_tree(outputs))
                def verify_cycle():
                    assert workload.seed == expected['seed']
                    assert dataset.sample_id(cycle * world + rank) == expected['sample_ids'][rank]
                    _close(expected['state'], state)
                    _close(expected['gradients'], actual['gradients'])
                    _close(expected['losses'][rank], actual['losses'])
                    assert equal(before_rng, rng_state(device=device)), 'outer RNG changed'
                    assert equal(before_frozen, _frozen(model.state)), 'frozen logical parameter values changed'
                    assert model.state['dmd_counters'].tolist() == [cycle + 1, cycle + 1, 5 * (cycle + 1)]
                    trainables = []
                    for role, steps in (('student', cycle + 1), ('fake', 5 * (cycle + 1))):
                        parameters = {id(p) for p in model.state[role].parameters() if p.requires_grad}
                        owned = {id(p) for group in model.state[role + '_optimizer'].param_groups for p in group['params']}
                        assert parameters == owned and all(p.grad is None for p in model.state[role].parameters())
                        trainables.append(parameters)
                        assert model.state[role + '_scheduler'].last_epoch == steps
                        assert all(int(v['step']) == steps for v in model.state[role + '_optimizer'].state.values())
                    assert not trainables[0] & trainables[1]
                    aliases = 0
                    for first, second, _ in pairs:
                        a = resolve_module(session.state, first[0]).get_parameter(first[1])
                        b = resolve_module(session.state, second[0]).get_parameter(second[1])
                        assert (a is b) == (mode == 'manual') and not a.requires_grad and not b.requires_grad
                        aliases += a is b
                    assert session.effects['shared_parameter_aliases'] == aliases
                    assert session.effects['replica_state_consensus'] and session.effects['reduction_completed']
                    assert session.effects['lifecycle_installations'] == 1
                    if mode == 'manual':
                        assert equal(reference_cycles[cycle], actual), 'reference/manual must be bitwise identical'
                    return aliases
                aliases = distributed.checked(verify_cycle, f'native oracle comparison cycle {cycle}')
                if mode == 'reference':
                    reference_cycles.append(actual)
                distributed.checked(lambda: torch.save(dict(expected_gradients=expected['gradients'],
                    actual_gradients=actual['gradients'], expected_losses=expected['losses'][rank],
                    actual_losses=actual['losses'], stochastic=expected['stochastic']),
                    root/f'rank-{rank:05d}-{mode}-cycle-{cycle}.pt'), 'rank-native gradient evidence')
                mode_cycles.append(dict(cycle=cycle, seed=workload.seed, sample_id=expected['sample_ids'][rank],
                    global_sample_ids=expected['sample_ids'], student_mean_gradient_l2=student_norms[cycle],
                    final_state_sha256=fingerprint(signature(state)), effects=deepcopy(session.effects)))
            for handle in handles: handle.remove()
            records.append(dict(mode=mode, aliases=aliases, cycles=mode_cycles,
                semantic_identity=binding['replica_semantics'], initial_state_sha256=fingerprint(signature(initial)),
                oracle_rtol=ORACLE_RTOL, oracle_atol=ORACLE_ATOL, frozen_rng_unchanged=True,
                actual_classes={role: type(model.state[role].native).__qualname__ for role in ('student','fake','teacher')}))
            del model, dataset, algorithm, workload, backend, session, actual, outputs
            gc.collect()
        proof = dict(evidence_kind='SYNTHETIC_TEST_ONLY', rank=rank, world_size=world, device=str(device),
            torch_version=torch.__version__, family=identity['model_metadata']['family'], independent_b1_gradient_mean=True,
            average_before_clip_adam=True, oracle_cycles=ORACLE_CYCLES, later_student_gradient_nonzero=True,
            reference_manual_exact=True, records=records, quality='NOT_RUN', pretrained_capacity='NOT_RUN', performance='NOT_RUN')
        distributed.checked(lambda: _write(root/f'rank-{rank:05d}.json', proof), 'rank-native proof')
        rows = distributed.gather(proof)
        distributed.primary(lambda: _write(root/'proof.json', dict(evidence_kind='SYNTHETIC_TEST_ONLY', ranks=rows)), 'oracle proof')
    return root


def fresh_cli(config, output, ranks, *args, expected_error=None):
    """Fresh public CLI processes, with deterministic test settings only."""
    script = Path(__file__).resolve()
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(ranks),
               str(script), 'cli', *map(str, args)]
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=600)
    Path(output).write_text(result.stdout + '\n' + result.stderr)
    if expected_error:
        assert result.returncode != 0 and expected_error in result.stdout + result.stderr, result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr


def verify_restart(root, mode, world):
    """Read every rank shard; never infer success from rank zero alone."""
    folder = Path(root)/mode
    for rank in range(world):
        paths = [folder/name/'checkpoints/cycle-000003'/f'rank-{rank:05d}.pt' for name in ('resumed','whole')]
        a, b = (torch.load(path, weights_only=True, map_location='cpu') for path in paths)
        assert equal(a, b), (rank, mode, 'full checkpoint payload')
        assert a['cycle'] == 3 and a['cursor'] == 3 * world
        assert a['state']['dmd_counters']['value']['counters'].tolist() == [3,3,15]
        assert a['state']['student_scheduler']['value']['last_epoch'] == 3
        assert a['state']['fake_scheduler']['value']['last_epoch'] == 15
        rows = [json.loads(line) for line in (folder/'resumed'/f'ranks/rank-{rank:05d}/metrics.jsonl').read_text().splitlines()]
        assert [row['sample_id'] for row in rows] == [f'native-{cycle * world + rank}' for cycle in (1,2)]
        for row in rows:
            effects = row['execution']['effects']
            assert effects['lifecycle_installations'] == 1 and effects['replica_state_consensus']
            assert (effects['shared_parameter_aliases'] > 0) == (mode == 'manual')
            assert effects['checkpoint_calls'] == effects['recompute_calls'] == 0
    checkpoint = folder/'resumed/checkpoints/cycle-000003'
    manifest = json.loads((checkpoint/'manifest.json').read_text())
    assert set(manifest['files']) == {f'rank-{rank:05d}.pt' for rank in range(world)}
    assert manifest['topology']['world_size'] == world
    assert (checkpoint/'_SUCCESS').is_file()
    return checkpoint


def _single_cli(root, name, *args):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'cli', *map(str,args)],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, 'OMP_NUM_THREADS':'1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    (Path(root)/(name+'.log')).write_text(result.stdout+'\n'+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
    return json.loads(result.stdout)


def portable_output(config_path, root, mode):
    """A fresh export-only process encodes an unseen prompt with owned codecs."""
    root = Path(root); folder = root/mode
    from distillschedule.training.inference import read_export
    _, exported = read_export(folder/'export')
    assert set(exported['model']['modules']) == {'student','media'}
    assert 'assets' not in exported['model']['parameters']
    from distillschedule.data.export_inputs import SCHEMA
    source = dict(schema_version=SCHEMA, weights=str(folder/'export'), mode='prompts',
                  records=[dict(id='new-replica-prompt', prompt='a blue boat on a lake',
                                                     latent_shape=[1,4,4,4])])
    _write(folder/'new-prompt.json', source)
    _single_cli(folder, 'prepare-new-prompt', 'data','prepare',folder/'new-prompt.json','--output',folder/'new-prepared',
        '--preparer','distillschedule.data.export_inputs:prepare_inputs')
    prepared = folder/'new-prepared/manifest.json'
    # Export inference has its own explicit single-device owner, no groups or
    # training assets. CPU works for both CPU and GPU training artifacts.
    inference = dict(schema_version='distillschedule.training.v1', model=deepcopy(exported['model']),
        algorithm=dict(name='dmd'), data=dict(prepared=str(prepared)), execution=dict(mode='reference'),
        trainer=dict(cycles=1,seed=909,device='cpu',output_dir=str(folder/'unused')))
    inference['model'] = {key: inference['model'][key] for key in ('factory','parameters')}
    _write(folder/'inference.json', inference)
    result = _single_cli(folder, 'generate-new-prompt', 'generate',folder/'inference.json','--weights',folder/'export',
        '--prepared',prepared,'--output',folder/'generated','--output-kind','image')
    artifact = torch.load(result['artifact'], weights_only=True, map_location='cpu')
    assert artifact['decoded']['image'].shape == (8,8,3) and Path(result['image']['path']).is_file()
    # The selected export must carry CURRENT Student, not Teacher/base/EMA.
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.inference import restore_modules
    factory, _ = load_callable(exported['model']['factory'])
    components = factory(exported['model']['parameters'])
    restore_modules(components, folder/'export', exported)
    checkpoint = torch.load(folder/'resumed/checkpoints/cycle-000003/rank-00000.pt', weights_only=True, map_location='cpu')
    assert equal(cpu_tree(components.state['student'].state_dict()), checkpoint['state']['student']['value'])
    from distillschedule.data import PreparedDataset
    from distillschedule.training.session import _generate_values
    request = load_training_config(folder/'inference.json')
    expected = _generate_values(request, components, PreparedDataset(prepared).sample(0), {}, output_kind='image')
    assert equal(expected, artifact), 'fresh output must be exact current Student and owned codecs'
    return result


def request(config_path, output, device, mode='reference'):
    value = load_training_config(config_path).to_dict()
    value['data']['sharding'] = 'rank_strided'
    value['trainer'].update(device=device, distributed=dict(backend='nccl' if device == 'cuda' else 'gloo', timeout_seconds=180))
    value['execution'] = dict(mode='reference') if mode == 'reference' else deepcopy(REPLICATED_MANUAL)
    _write(output, value)
    return output


def lifecycle(config_path, output, ranks=2):
    """Single-host test driver only; external multi-node launch stays caller-owned."""
    config = load_training_config(config_path)
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    records = []
    for mode in ('reference','manual'):
        folder = root/mode; folder.mkdir()
        value = config.to_dict(); value['execution'] = dict(mode='reference') if mode=='reference' else deepcopy(REPLICATED_MANUAL)
        request = folder/'request.json'; _write(request,value)
        for name, cycles, resume in (('partial',1,None),('resumed',3,folder/'partial/checkpoints/cycle-000001'),('whole',3,None)):
            args = ['train',request,'--cycles',cycles,'--output',folder/name]
            if resume: args += ['--resume',resume]
            fresh_cli(request,folder/(name+'.log'),ranks,*args)
        checkpoint = verify_restart(root, mode, ranks)
        fresh_cli(request,folder/'export.log',ranks,'export',request,'--checkpoint',checkpoint,'--output',folder/'export')
        generated = portable_output(request,root,mode)
        # All file manifests must be complete even if only rank one is absent.
        shard = checkpoint/'rank-00001.pt'; hidden = shard.with_suffix('.hidden')
        shard.rename(hidden)
        try:
            fresh_cli(request,folder/'missing-shard.log',ranks,'train',request,'--cycles',3,
                '--resume',checkpoint,'--output',folder/'must-not-exist',expected_error='shard')
            assert not (folder/'must-not-exist').exists()
        finally:
            hidden.rename(shard)
        records.append(dict(mode=mode, checkpoint=str(checkpoint), generated=generated,
                            all_rank_state_rng_sample_order_exact=True, missing_shard_refused=True))
    for rank in range(ranks):
        payloads = [torch.load(root/mode/'resumed/checkpoints/cycle-000003'/f'rank-{rank:05d}.pt', weights_only=True,
                              map_location='cpu') for mode in ('reference','manual')]
        for key in ('state','rng','cycle','cursor','versions'):
            assert equal(payloads[0][key],payloads[1][key]), (rank,key)
    _write(root/'proof.json', dict(evidence_kind='SYNTHETIC_TEST_ONLY', world_size=ranks, records=records,
        current_student_new_prompt=True, quality='NOT_RUN', pretrained_capacity='NOT_RUN', performance='NOT_RUN'))
    return root


def main():
    deterministic()
    if len(sys.argv) > 1 and sys.argv[1] == 'cli':
        import runpy
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        runpy.run_module('distillschedule', run_name='__main__', alter_sys=True)
        return
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['request','oracle','lifecycle','verify','portable'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--ranks',type=int,default=2)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--mode',choices=['reference','manual'],default='reference')
    args = parser.parse_args()
    if args.stage == 'request': request(args.config,args.output,args.device,args.mode)
    elif args.stage == 'oracle': oracle(args.config,args.output)
    elif args.stage == 'lifecycle': lifecycle(args.config,args.output,args.ranks)
    else:
        for mode in ('reference','manual'):
            verify_restart(args.output,mode,args.ranks)
            if args.stage == 'portable': portable_output(args.config,args.output,mode)


if __name__ == '__main__':
    main()
