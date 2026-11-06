"""Fresh-process raw-media/train/restart/export and shared-planner acceptance."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from distillschedule.data import prepare
from distillschedule.examples.tiny_av import build_fixture
from distillschedule.training.config import TrainingConfig, load_training_config
from distillschedule.training.session import train


def run_cli(*args):
    result = subprocess.run([sys.executable, "-m", "distillschedule", *map(str, args)],
                            capture_output=True, text=True,
                            env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def equal(x, y):
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and torch.equal(x, y)
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(equal(value, y[key]) for key, value in x.items())
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(equal(a, b) for a, b in zip(x, y))
    return x == y


def fixture(tmp_path, samples=6):
    root = tmp_path / "media"
    build_fixture(root, samples=samples)
    prepare(root / "raw/manifest.json", root / "prepared")
    return root, load_training_config(root / "train.json")


def test_fresh_process_resume_matches_all_training_state_and_generates(tmp_path):
    root, config = fixture(tmp_path)
    # A real user plugin can configure stdout logging while it is imported.
    # Keep the actual model, algorithm, checkpoint and inference chain intact.
    plugin = root / 'noisy.py'
    plugin.write_text('''import logging
import sys
from distillschedule.examples import tiny_av
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.propagate = False
def diagnostic(stage):
    print(stage + " print")
    logger.warning(stage + " log")
    logging.warning(stage + " root log")
    logging.getLogger("preconfigured.child").warning(stage + " named log")
diagnostic("import")
def build_model(parameters):
    diagnostic("model")
    components = tiny_av.build_model(parameters)
    components.generate = generate
    components.decode = decode
    return components
def build_algorithm(*args):
    diagnostic("algorithm")
    return tiny_av.build_algorithm(*args)
def generate(*args):
    diagnostic("generate")
    return tiny_av.generate(*args)
def decode(*args):
    diagnostic("decode")
    return tiny_av.decode(*args)
''')
    raw = config.to_dict()
    raw['model']['factory'] = str(plugin) + ':build_model'
    raw['algorithm']['factory'] = str(plugin) + ':build_algorithm'
    (root / 'train.json').write_text(json.dumps(raw))

    def noisy_cli(*args):
        script = '''import io
import logging
import sys
from distillschedule.cli import main
stdout, stderr = sys.stdout, sys.stderr
handler = logging.StreamHandler(stdout)
logging.getLogger().addHandler(handler)
logger = logging.getLogger("preconfigured.child")
logger.propagate = False
named_handler = logging.StreamHandler(stdout)
logger.addHandler(named_handler)
unrelated = io.StringIO()
unrelated_handler = logging.StreamHandler(unrelated)
logger.addHandler(unrelated_handler)
code = main(sys.argv[1:])
assert sys.stdout is stdout and sys.stderr is stderr
assert handler.stream is stdout and named_handler.stream is stdout
assert unrelated_handler.stream is unrelated and not unrelated.closed
assert "model named log" in unrelated.getvalue()
raise SystemExit(code)
'''
        completed = subprocess.run([sys.executable, '-c', script, *map(str, args)],
                                   capture_output=True, text=True,
                                   env={**os.environ, 'OMP_NUM_THREADS': '1'})
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result = json.loads(completed.stdout)
        stages = ['import', 'model']
        if args[0] in ('train', 'evaluate', 'export'):
            stages.append('algorithm')
        if args[0] in ('evaluate', 'generate'):
            stages.extend(('generate', 'decode'))
        assert all(stage + ' ' + kind in completed.stderr for stage in stages
                   for kind in ('print', 'log', 'root log', 'named log'))
        return result

    assert run_cli("doctor", "--config", root / "train.json")["model_imported"] is False
    first = noisy_cli("train", root / "train.json")
    checkpoint = Path(first["checkpoint"])
    resumed = noisy_cli("train", root / "train.json", "--resume", checkpoint,
                      "--cycles", 6, "--output", root / "resumed")
    uninterrupted = noisy_cli("train", root / "train.json", "--cycles", 6,
                            "--output", root / "uninterrupted")
    a = torch.load(Path(resumed["checkpoint"]) / "state.pt", weights_only=True)
    b = torch.load(Path(uninterrupted["checkpoint"]) / "state.pt", weights_only=True)
    for key in ("state", "rng", "cycle", "cursor", "versions", "policy"):
        assert equal(a[key], b[key]), key
    assert resumed["resume_cycle"] == 4 and resumed["cycles_this_process"] == 2
    assert resumed["final_versions"]["student"] == 6
    assert resumed["final_versions"]["fake"] == 12
    assert resumed["final_versions"]["teacher"] == 0
    generated = noisy_cli("evaluate", root / "train.json", "--checkpoint", checkpoint,
                        "--output", root / "generated")
    artifact = torch.load(generated["artifact"], weights_only=True)
    assert artifact["decoded"]["video"].dtype == torch.uint8
    exported = noisy_cli("export", root / "train.json", "--checkpoint", checkpoint,
                       "--output", root / "exported")
    assert exported["kind"] == "INFERENCE_WEIGHTS_NOT_RESUMABLE_CHECKPOINT"
    assert torch.load(root / "exported/weights.pt", weights_only=True)["student"]
    portable = noisy_cli('generate', root / 'train.json', '--weights', root / 'exported',
                        '--prepared', root / 'prepared/manifest.json', '--output', root / 'portable')
    assert torch.load(portable['artifact'], weights_only=True)['decoded']['video'].dtype == torch.uint8
    rows = [json.loads(row) for row in (root / "resumed/metrics.jsonl").read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == ["sample-000004", "sample-000005"]


def test_measured_auto_plan_is_executed_and_reused_across_real_updates(tmp_path):
    root, config = fixture(tmp_path, samples=4)
    raw = config.to_dict()
    raw["trainer"].update(cycles=3, output_dir=str(root / "auto"))
    raw["execution"] = {"mode": "auto", "memory_limit_bytes": 100_000_000,
                        "max_candidates": 3, "samples": 1, "max_program_runs": 8}
    automatic = train(TrainingConfig.from_dict(raw, source=config.source))
    assert automatic["policy"]["plan_sha256"]
    assert automatic["policy"]["initial_reference_gate"]["status"] == "PASS"
    assert automatic["final_versions"]["fake"] == 6
    plan_path = root / "auto/planning/selected-plan.json"
    document = json.loads(plan_path.read_text())
    rows = [json.loads(row) for row in (root / "auto/metrics.jsonl").read_text().splitlines()]
    assert rows[1]["execution"]["dispatcher"]["executed_operations"] == document["selected"]["order"]
    # A saved plan from this same initial state also drives a fresh real run.
    raw["trainer"]["output_dir"] = str(root / "saved-plan")
    raw["execution"] = {"mode": "plan", "plan": str(plan_path)}
    saved = train(TrainingConfig.from_dict(raw, source=config.source))
    assert saved["policy"]["implementation"] == document["selected"]["implementation"]
    assert saved["policy"]["plan_sha256"] == document["plan_sha256"]
    assert saved["final_versions"] == automatic["final_versions"]


def test_unknown_model_factory_without_solver_edits(tmp_path):
    root, config = fixture(tmp_path, samples=4)
    template = Path(__file__).resolve().parents[1] / "examples/framework/newmodel.py"
    raw = config.to_dict()
    raw["model"]["factory"] = str(template) + ":build_model"
    raw["trainer"].update(cycles=2, output_dir=str(root / "new-model"))
    result = train(TrainingConfig.from_dict(raw, source=config.source))
    assert result["completed_cycles"] == 2
    assert result["final_versions"]["student"] == 2
    assert result["identity"]["model"]["reference"].endswith("newmodel.py:build_model")


def test_invalid_resume_identity_rejects_without_creating_output(tmp_path):
    root, config = fixture(tmp_path, samples=4)
    completed = train(config)
    raw = config.to_dict()
    raw["trainer"].update(seed=99, cycles=5, resume_from=completed["checkpoint"],
                          output_dir=str(root / "wrong"))
    # Supply one extra sample bound to this modified request first, so failure
    # is checkpoint identity rather than input exhaustion.
    raw["trainer"]["cycles"] = 4
    with pytest.raises(ValueError, match="identity"):
        train(TrainingConfig.from_dict(raw, source=config.source))
    assert not (root / "wrong").exists()


@pytest.mark.parametrize('project', [False, True])
def test_export_releases_loaded_checkpoint_before_staging_and_restores_weights(tmp_path, monkeypatch, project):
    import gc
    import weakref
    from distillschedule.examples.tiny_av import build_model
    from distillschedule.training import inference, session
    root, config = fixture(tmp_path)
    trained = train(config)
    original_load, original_publish = session._load_for_output, inference.publish_export
    source_refs, loaded = [], []

    def projection(components, parameters):
        assert source_refs and all(ref() is None for ref in source_refs)
        return session.InferenceProjection(components, config.model['factory'], parameters)

    def load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        # Observe the real strict loader's module payload, not synthetic state.
        source_refs.extend(weakref.ref(tensor) for row in result[2]['state'].values()
                           if row['kind'] == 'module' for tensor in row['value'].values())
        loaded.append(result[0])
        if project:
            result[0].inference_projection = projection
        return result

    def publish(*args, **kwargs):
        gc.collect()
        assert source_refs and all(ref() is None for ref in source_refs)
        # Optimizer slots intentionally installed into live training owners are
        # legitimate state, not disposable checkpoint module copies.
        optimizer = loaded[0].state['student_optimizer']
        assert optimizer.state and optimizer.param_groups[0]['params'][0] in optimizer.state
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(session, '_load_for_output', load)
    monkeypatch.setattr(inference, 'publish_export', publish)
    exported = session.export(config, trained['checkpoint'], root / 'exported')
    path, manifest = inference.read_export(root / 'exported')
    assert manifest == exported and manifest['schema_version'] == 'distillschedule.inference_export.v1'
    assert manifest['checkpoint_cycle'] == 4
    weights = torch.load(path / 'weights.pt', weights_only=True)
    components = build_model({})
    inference.restore_modules(components, path, manifest)
    for name, module in inference.modules(components.state).items():
        assert equal(weights[name], module.state_dict())
        assert equal(weights[name], loaded[0].state[name].state_dict())
    with torch.no_grad():
        loaded[0].state['student'].layers[0].weight.add_(1)
    assert equal(weights, torch.load(path / 'weights.pt', weights_only=True))
    assert not equal(weights['student'], loaded[0].state['student'].state_dict())


def test_generation_hashes_each_weight_file_once_and_loads_that_descriptor(tmp_path, monkeypatch):
    import hashlib
    from distillschedule.training import session
    root, config = fixture(tmp_path)
    trained = train(config)
    session.export(config, trained['checkpoint'], root / 'weights')
    expected = (root / 'weights/weights.pt').read_bytes()
    weight_inode = (root / 'weights/weights.pt').stat().st_ino
    original_sha, original_load = hashlib.sha256, torch.load
    hashes, loaded = [], []

    class Hash:
        def __init__(self, *args, **kwargs):
            self.actual = original_sha(*args, **kwargs)
            self.seen = bytearray(args[0] if args else b'')

        def update(self, value):
            self.seen.extend(value)
            self.actual.update(value)

        def hexdigest(self):
            if self.seen == expected:
                hashes.append(len(self.seen))
            return self.actual.hexdigest()

    def load(value, *args, **kwargs):
        if hasattr(value, 'read') and os.fstat(value.fileno()).st_ino == weight_inode:
            loaded.append(value)
            assert value.tell() == 0 and not value.closed
        return original_load(value, *args, **kwargs)

    monkeypatch.setattr(hashlib, 'sha256', Hash)
    monkeypatch.setattr(torch, 'load', load)
    result = session.generate_from_export(config, root / 'weights', root / 'output',
                                          prepared=root / 'prepared/manifest.json')
    assert result['status'] == 'GENERATED'
    assert hashes == [len(expected)], 'full weights were hashed more than once per generation'
    assert len(loaded) == 1 and loaded[0].closed


def test_named_role_export_reuses_exact_shared_storage_snapshot(tmp_path):
    from distillschedule.training.inference import publish_export
    from distillschedule.training.plugins import load_callable
    from distillschedule.examples.tiny_av import build_model
    components = build_model({})
    components.state = {'student': components.state['student'], 'teacher': components.state['student']}
    _, source = load_callable('distillschedule.examples.tiny_av:build_model')
    publish_export(components, tmp_path / 'shared', model={
        'factory': source['reference'], 'source': source, 'parameters': {},
        'metadata': components.metadata}, identity={'model': source, 'model_metadata': components.metadata}, cycle=0)
    weights = torch.load(tmp_path / 'shared/weights.pt', weights_only=True)
    for name, tensor in weights['student'].items():
        assert equal(tensor, weights['teacher'][name])
        assert tensor.untyped_storage().data_ptr() == weights['teacher'][name].untyped_storage().data_ptr()
    with torch.no_grad():
        components.state['student'].layers[0].weight.add_(1)
    assert equal(weights, torch.load(tmp_path / 'shared/weights.pt', weights_only=True))


@pytest.mark.parametrize('damage', ['weights', 'manifest', 'marker', 'symlink', 'fifo', 'extra', 'escape'])
def test_export_integrity_fails_before_model_construction(tmp_path, monkeypatch, damage):
    import hashlib
    from distillschedule.training import session
    root, config = fixture(tmp_path)
    trained = train(config)
    target = root / 'weights'
    session.export(config, trained['checkpoint'], target)
    weights, manifest = target / 'weights.pt', target / 'export.json'
    if damage in ('weights', 'manifest', 'marker'):
        path = {'weights': weights, 'manifest': manifest, 'marker': target / '_SUCCESS'}[damage]
        path.write_bytes(path.read_bytes() + b'corrupt')
    elif damage in ('symlink', 'fifo'):
        weights.unlink()
        if damage == 'symlink':
            weights.symlink_to(root / 'train.json')
        else:
            os.mkfifo(weights)
    elif damage == 'extra':
        (target / 'unlisted').write_text('changed')
    else:
        value = json.loads(manifest.read_text())
        value['files']['../outside'] = value['files'].pop('weights.pt')
        manifest.write_text(json.dumps(value))
        (target / '_SUCCESS').write_text(hashlib.sha256(manifest.read_bytes()).hexdigest())

    def forbidden(*args, **kwargs):
        raise AssertionError('model imported before export integrity admission')

    monkeypatch.setattr(session, '_construct_model', forbidden)
    with pytest.raises((ValueError, OSError)):
        session.generate_from_export(config, target, root / 'output', prepared=root / 'prepared/manifest.json')
    assert not (root / 'output').exists()


@pytest.mark.parametrize('stage', ['construction', 'loading', 'generation'])
@pytest.mark.parametrize('change', ['replace', 'same_size_restored_mtime', 'manifest', 'extra'])
def test_generation_rejects_export_changes_and_closes_descriptors(tmp_path, monkeypatch, stage, change):
    from distillschedule.training import inference, session
    root, config = fixture(tmp_path)
    trained = train(config)
    target = root / 'weights'
    session.export(config, trained['checkpoint'], target)
    streams = []
    original_open = inference.VerifiedExport._open

    def opened(self, *args):
        stream = original_open(self, *args)
        streams.append(stream)
        return stream

    def mutate():
        path = target / 'weights.pt'
        if change == 'replace':
            replacement = root / 'replacement'
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        elif change == 'same_size_restored_mtime':
            before = path.stat()
            with path.open('r+b') as stream:
                # A byte-preserving write still changes ctime and must be seen.
                stream.write(path.read_bytes()[:1])
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        elif change == 'manifest':
            (target / 'export.json').write_bytes((target / 'export.json').read_bytes() + b' ')
        else:
            (target / 'new-file').write_text('unexpected')

    if stage == 'construction':
        original = session._construct_model

        def construct(*args, **kwargs):
            result = original(*args, **kwargs)
            mutate()
            return result

        monkeypatch.setattr(session, '_construct_model', construct)
    elif stage == 'loading':
        original = torch.load

        def load(value, *args, **kwargs):
            result = original(value, *args, **kwargs)
            if hasattr(value, 'read') and isinstance(value.name, int):
                mutate()
            return result

        monkeypatch.setattr(torch, 'load', load)
    else:
        original = session._generate_values

        def generate(*args, **kwargs):
            result = original(*args, **kwargs)
            mutate()
            return result

        monkeypatch.setattr(session, '_generate_values', generate)
    monkeypatch.setattr(inference.VerifiedExport, '_open', opened)
    with pytest.raises(ValueError, match='export'):
        session.generate_from_export(config, target, root / 'output', prepared=root / 'prepared/manifest.json')
    assert streams and all(stream.closed for stream in streams)
    assert not (root / 'output').exists()


def test_standalone_loader_rejects_forged_manifest_and_keeps_custom_callback_contract(tmp_path):
    from distillschedule.examples.tiny_av import build_model
    from distillschedule.training import inference
    from distillschedule.training.plugins import load_callable
    components = build_model({})
    _, source = load_callable('distillschedule.examples.tiny_av:build_model')
    model = {'factory': source['reference'], 'source': source, 'parameters': {}, 'metadata': components.metadata}
    calls = []

    def publish(state, directory):
        (directory / 'nested').mkdir()
        (directory / 'nested/custom.txt').write_text('weights')
        return {'custom': 1}

    def load(state, directory, metadata):
        calls.append((state is components.state, directory, metadata))
        assert (directory / 'nested/custom.txt').read_text() == 'weights'

    components.export, components.load_export = publish, load
    inference.publish_export(components, tmp_path / 'custom', model=model,
                             identity={'model': source, 'model_metadata': components.metadata}, cycle=0)
    root, manifest = inference.read_export(tmp_path / 'custom')
    inference.restore_modules(components, root, manifest)
    assert calls == [(True, root, {'custom': 1})]
    forged = {**manifest, 'checkpoint_cycle': 1}
    with pytest.raises(ValueError, match='manifest differs'):
        inference.restore_modules(components, root, forged)
    assert len(calls) == 1

    def changing_metadata(state, directory, metadata):
        metadata['custom'] = 2

    components.load_export = changing_metadata
    with pytest.raises(ValueError, match='manifest changed in memory'):
        inference.restore_modules(components, root, manifest)

    def mutating_load(state, directory, metadata):
        (directory / 'nested/custom.txt').write_text('changed')

    components.load_export = mutating_load
    with pytest.raises(ValueError, match='export file changed'):
        inference.restore_modules(components, root, manifest)


def _distributed_export_capacity_worker(rank, rendezvous, destination):
    from datetime import timedelta
    from types import SimpleNamespace
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard
    from distillschedule.open_workload import torch_backend
    from distillschedule.training import inference, session, tensor_state
    from distillschedule.training.adapters import ModelComponents
    from distillschedule.training.plugins import load_callable
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=40))
    original_snapshot, original_full = tensor_state._snapshot_tensor, torch_backend.full_tensor
    original_load, original_publish = session._load_for_output, inference.publish_export
    try:
        mesh = init_device_mesh('cpu', (2,))
        dense = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        module = torch.nn.Linear(3, 4, bias=False)
        module.weight = torch.nn.Parameter(distribute_tensor(dense, mesh, [Shard(0)]))
        components = ModelComponents(state={'model': module})
        _, source = load_callable('distillschedule.examples.tiny_av:build_model')
        model = {'factory': source['reference'], 'source': source, 'parameters': {}, 'metadata': {}}
        identity = {'model': source, 'model_metadata': {}}
        counts = {'materializations': 0, 'snapshots': 0}

        def snapshot(*args, **kwargs):
            assert rank == 0, 'nonprimary retained a full CPU model snapshot'
            counts['snapshots'] += 1
            return original_snapshot(*args, **kwargs)

        def full(value):
            counts['materializations'] += 1
            return original_full(value)

        tensor_state._snapshot_tensor, torch_backend.full_tensor = snapshot, full
        inference.publish_export(components, Path(destination) / 'sharded', model=model,
                                 identity=identity, cycle=0, primary_only=True)
        assert counts == {'materializations': 1, 'snapshots': int(rank == 0)}

        # Exercise the public export's policy binding with actual DTensor state;
        # checkpoint reading itself is covered by the real lifecycle tests above.
        os.environ.update(WORLD_SIZE='2', RANK=str(rank))
        config = SimpleNamespace(trainer={'device': 'cpu'}, data={'sharding': 'rank_strided'},
                                 model={'factory': source['reference']})
        session._load_for_output = lambda *args: (components, None, {'cycle': 3}, identity, None)

        def publish(*args, **kwargs):
            assert kwargs['primary_only'] is True
            return original_publish(*args, **kwargs)

        inference.publish_export = publish
        session.export(config, 'unused-isolated-loader', Path(destination) / 'public')
        assert counts == {'materializations': 2, 'snapshots': 2 * int(rank == 0)}
        inference.publish_export = original_publish

        # A complete primary-owned replica must NOT enter a tensor collective
        # inside rank-zero publication; nonprimary has no role model at all.
        replica = ModelComponents({'model': torch.nn.Linear(3, 4, bias=False)}) if rank == 0 else None
        inference.publish_export(replica, Path(destination) / 'replica', model=model,
                                 identity=identity, cycle=0, primary_replica=True)
        assert counts['snapshots'] == 3 * int(rank == 0)
        assert counts['materializations'] == 2 + int(rank == 0)

        def failed_snapshot(*args, **kwargs):
            assert rank == 0
            raise MemoryError('test export staging capacity')

        tensor_state._snapshot_tensor = failed_snapshot
        with pytest.raises(ValueError, match='test export staging capacity'):
            inference.publish_export(components, Path(destination) / 'failed', model=model,
                                     identity=identity, cycle=0, primary_only=True)
        assert not (Path(destination) / 'failed/_SUCCESS').exists()
        (Path(destination) / f'rank-{rank}.json').write_text(json.dumps(counts))
    finally:
        tensor_state._snapshot_tensor, torch_backend.full_tensor = original_snapshot, original_full
        session._load_for_output, inference.publish_export = original_load, original_publish
        dist.destroy_process_group()


def test_distributed_export_gathers_on_all_owners_but_retains_only_primary(tmp_path):
    torch.multiprocessing.spawn(_distributed_export_capacity_worker,
                                args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    weights = torch.load(tmp_path / 'sharded/weights.pt', weights_only=True)
    assert torch.equal(weights['model']['weight'], torch.arange(12, dtype=torch.float32).reshape(4, 3))
    rows = [json.loads((tmp_path / f'rank-{rank}.json').read_text()) for rank in range(2)]
    assert rows[0]['snapshots'] == 3 and rows[1]['snapshots'] == 0


def _export_input_fixture(tmp_path):
    root, config = fixture(tmp_path)
    plugin = root / 'inference_inputs.py'
    plugin.write_text('''import hashlib
import os
import random
from pathlib import Path
import torch
from distillschedule.examples.tiny_av import build_model as original

def build(parameters):
    value = original(parameters)
    value.encode_input = encode_input
    def close():
        marker = os.environ.get('DMD_TEST_EXPORT_CLOSE')
        if marker:
            Path(marker).write_text('closed')
            torch.rand(1)
            random.random()
            action = os.environ.get('DMD_TEST_EXPORT_CLOSE_ACTION')
            if action == 'error':
                raise RuntimeError('test close failed')
            if action == 'mutate':
                target = Path(marker).parent / 'weights/weights.pt'
                target.write_bytes(target.read_bytes() + b'changed-on-close')
            if action == 'code':
                target = Path(__file__)
                target.write_text(target.read_text() + '\\n# changed-on-close\\n')
    value.close = close
    return value

def encode_input(state, record, parameters):
    text = record['text']
    if record.get('mutate_export'):
        path = Path(parameters['source_dir']) / 'weights/weights.pt'
        path.write_bytes(path.read_bytes() + b'changed')
    feature = next(state['student'].parameters()).detach().reshape(-1)[:4].reshape(1,4,1,1,1)
    return {'features': {'latent': feature.expand(1,4,2,4,4).clone(), 'condition': torch.zeros(1,4,1,1,1)},
            'metadata': {'native': 'actual-restored-model-input-fixture'},
            'sources': {'text': {'kind': 'inline_text', 'sha256': hashlib.sha256(text.encode()).hexdigest()}}}
''')
    raw = config.to_dict()
    raw['model']['factory'] = str(plugin) + ':build'
    config = TrainingConfig.from_dict(raw, source=config.source)
    trained = train(config)
    from distillschedule.training.session import export
    export(config, trained['checkpoint'], root / 'weights')
    request = root / 'inference-inputs.json'
    request.write_text(json.dumps({'schema_version': 'distillschedule.export_inputs.v1',
        'weights': 'weights', 'mode': 'text', 'records': [
            {'id': 'new-prompt-0', 'text': 'new prompt zero'}, {'id': 'new-prompt-1', 'text': 'new prompt one'}]}))
    return root, config, request


def test_export_input_preparation_hashes_weights_once_across_all_rows_and_publication(tmp_path, monkeypatch):
    import hashlib
    from distillschedule.data import PreparedDataset
    from distillschedule.data.export_inputs import prepare_inputs
    root, config, request = _export_input_fixture(tmp_path)
    expected = (root / 'weights/weights.pt').read_bytes()
    original = hashlib.sha256
    hashes = []

    class Hash:
        def __init__(self, *args, **kwargs):
            self.actual, self.seen = original(*args, **kwargs), bytearray(args[0] if args else b'')

        def update(self, value):
            self.actual.update(value)
            self.seen.extend(value)

        def hexdigest(self):
            if self.seen == expected:
                hashes.append(len(self.seen))
            return self.actual.hexdigest()

    monkeypatch.setattr(hashlib, 'sha256', Hash)
    path = prepare_inputs(request, root / 'new-inputs')
    assert hashes == [len(expected)], 'preparation rehashed export contents per row or during publication'
    dataset = PreparedDataset(path)
    assert [dataset.sample_id(index) for index in range(len(dataset))] == ['new-prompt-0', 'new-prompt-1']
    from distillschedule.training.session import generate_from_export
    result = generate_from_export(config, root / 'weights', root / 'new-output', prepared=path)
    assert result['status'] == 'GENERATED' and hashes == [len(expected)] * 2


def test_export_input_preparation_rejects_mutation_and_closes_admission(tmp_path, monkeypatch):
    from distillschedule.data.export_inputs import prepare_inputs
    from distillschedule.training.inference import VerifiedExport
    root, _, request = _export_input_fixture(tmp_path)
    value = json.loads(request.read_text())
    value['records'][1]['mutate_export'] = True
    request.write_text(json.dumps(value))
    streams, original = [], VerifiedExport._open

    def opened(*args, **kwargs):
        stream = original(*args, **kwargs)
        streams.append(stream)
        return stream

    monkeypatch.setattr(VerifiedExport, '_open', opened)
    with pytest.raises(ValueError, match='export'):
        prepare_inputs(request, root / 'rejected-inputs')
    assert streams and all(stream.closed for stream in streams)
    assert not (root / 'rejected-inputs').exists()
    assert not (root / '.rejected-inputs.lock').exists()


@pytest.mark.parametrize('action', ['success', 'error', 'mutate', 'code'])
def test_export_input_provider_cleanup_finishes_before_publication_and_restores_rng(tmp_path, monkeypatch, action):
    from distillschedule.data.export_inputs import prepare_inputs
    from distillschedule.training.checkpoint import rng_state
    from distillschedule.training.inference import VerifiedExport
    root, _, request = _export_input_fixture(tmp_path)
    marker = root / 'close-marker'
    monkeypatch.setenv('DMD_TEST_EXPORT_CLOSE', str(marker))
    monkeypatch.setenv('DMD_TEST_EXPORT_CLOSE_ACTION', action)
    before = rng_state(device='cpu')
    streams, original = [], VerifiedExport._open

    def opened(*args, **kwargs):
        stream = original(*args, **kwargs)
        streams.append(stream)
        return stream

    monkeypatch.setattr(VerifiedExport, '_open', opened)
    if action == 'success':
        prepare_inputs(request, root / 'closed-inputs')
        assert (root / 'closed-inputs/manifest.json').is_file()
    else:
        expected = RuntimeError if action == 'error' else ValueError
        with pytest.raises(expected, match='close failed|export file changed|source hash changed'):
            prepare_inputs(request, root / 'closed-inputs')
        assert not (root / 'closed-inputs').exists()
        assert not (root / '.closed-inputs.lock').exists()
    assert marker.read_text() == 'closed'
    assert streams and all(stream.closed for stream in streams)
    assert equal(before, rng_state(device='cpu'))


def test_native_wan_output_uses_the_same_primary_only_weight_staging(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from distillschedule.training import inference, wan_native_inference as native
    from distillschedule.training.adapters import ModelComponents

    components = ModelComponents({'generator': torch.nn.Linear(3, 4)}, metadata={'fixture': True})
    monkeypatch.setattr(native, 'project_current_generator', lambda *args, **kwargs:
                        SimpleNamespace(components=components, parameters={}))
    monkeypatch.setattr(native, 'standalone_config', lambda *args: {})
    original, staging = inference.publish_export, []

    def publish(*args, **kwargs):
        staging.append(kwargs.get('primary_only', False))
        return original(*args, **kwargs)

    monkeypatch.setattr(inference, 'publish_export', publish)
    request = native.NativeOutputRequest({}, tmp_path / 'native-weights')
    result = request.publish(None, manifest={'cycle': 2}, identity={'fixture': True})
    assert staging == [True]
    assert result['checkpoint_cycle'] == 2 and (request.output / '_SUCCESS').is_file()
    weights = torch.load(request.output / 'weights.pt', weights_only=True)
    assert equal(weights['generator'], components.state['generator'].state_dict())
