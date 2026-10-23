import json

import pytest
import torch

from distillschedule.data import prepare
from distillschedule.examples.tiny_av import build_fixture


def test_independent_stream_sampling_and_complete_cycle_resume(tmp_path):
    from distillschedule.training.data_streams import InputStreams
    for name, count in (('generator', 6), ('critic', 12)):
        root = tmp_path / name
        build_fixture(root, samples=count)
        prepare(root / 'raw/manifest.json', root / 'prepared')
    spec = {name: {'prepared': str(tmp_path / name / 'prepared/manifest.json'),
                   'samples_per_cycle': count}
            for name, count in (('generator', 1), ('critic', 2))}
    data = InputStreams(spec, seed=19, world_size=2)
    data.admit_cycles(3)
    before = torch.get_rng_state().clone()
    assert data.sample_id(0) == {'generator': ['sample-000000'],
                                 'critic': ['sample-000000', 'sample-000002']}
    assert data.sample_id(1)['critic'] == ['sample-000001', 'sample-000003']
    assert data.sample_id(2)['critic'] == ['sample-000004', 'sample-000006']
    assert len(data.cycle_sample(2)['critic']) == 2
    progress = data.progress(4)
    assert progress['critic']['cursor'] == 8
    restored = InputStreams(spec, seed=19, world_size=2)
    restored.validate_data_progress(progress, 4)
    assert restored.sample_id(4) == data.sample_id(4)
    assert torch.equal(before, torch.get_rng_state())
    changed = json.loads(json.dumps(progress))
    changed['critic']['cursor'] -= 1
    with pytest.raises(ValueError, match='stream.*progress'):
        restored.validate_data_progress(changed, 4)
    with pytest.raises(ValueError, match='exceeds'):
        data.admit_cycles(4)


def test_stream_epoch_shuffle_uses_independent_reproducible_sampling(tmp_path):
    from distillschedule.training.data_streams import InputStreams
    build_fixture(tmp_path / 'media', samples=4)
    prepare(tmp_path / 'media/raw/manifest.json', tmp_path / 'prepared')
    spec = {name: {'prepared': str(tmp_path / 'prepared/manifest.json'),
                   'sampling': {'kind': 'epochs', 'shuffle': True}}
            for name in ('generator', 'critic')}
    data = InputStreams(spec, seed=9)
    data.admit_cycles(30)
    order = [data.sample_id(i) for i in range(12)]
    assert order == [InputStreams(spec, seed=9).sample_id(i) for i in range(12)]
    assert any(row['generator'] != row['critic'] for row in order)


def test_public_multistream_training_new_process_resume_matches_all_state(tmp_path):
    from pathlib import Path
    from test_training_end_to_end import run_cli, equal
    root = tmp_path / 'media'
    build_fixture(root, samples=6)
    prepare(root / 'raw/manifest.json', root / 'prepared')
    workflow = tmp_path / 'workflow.py'
    workflow.write_text('''from distillschedule.examples.tiny_av import build_algorithm
def build(components, feeds, parameters, cycle, seed):
    inputs = dict(feeds['generator'][0])
    inputs['condition'] = feeds['critic'][0]['condition']
    return build_algorithm(components, inputs, parameters, cycle, seed)
''')
    config = json.loads((root / 'train.json').read_text())
    config['data'] = {'streams': {name: {'prepared': str(root / 'prepared/manifest.json'),
                                        'sampling': {'kind': 'epochs', 'shuffle': True}}
                                  for name in ('generator', 'critic')}}
    config['algorithm']['factory'] = str(workflow) + ':build'
    config['trainer'].update(cycles=2, output_dir=str(tmp_path / 'first'))
    config_path = root / 'streams.json'
    config_path.write_text(json.dumps(config))
    first = run_cli('train', config_path)
    resumed = run_cli('train', config_path, '--resume', first['checkpoint'], '--cycles', 4,
                      '--output', tmp_path / 'resumed')
    whole = run_cli('train', config_path, '--cycles', 4, '--output', tmp_path / 'whole')
    a = torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(whole['checkpoint']) / 'state.pt', weights_only=True)
    assert equal(a, b)
    manifest = json.loads((Path(resumed['checkpoint']) / 'manifest.json').read_text())
    assert manifest['data_progress']['generator']['cursor'] == 4
    assert manifest['data_progress']['critic']['cursor'] == 4


def test_multistream_evaluation_uses_physical_generator_record_not_cycle_cursor(tmp_path):
    from test_training_end_to_end import run_cli
    root = tmp_path / 'media'
    build_fixture(root, samples=1)
    prepare(root / 'raw/manifest.json', root / 'prepared')
    workflow = tmp_path / 'workflow.py'
    workflow.write_text('''from distillschedule.examples.tiny_av import build_algorithm
def build(components, feeds, parameters, cycle, seed):
    return build_algorithm(components, feeds['generator'][0], parameters, cycle, seed)
''')
    config = json.loads((root / 'train.json').read_text())
    config['data'] = {'streams': {'generator': {
        'prepared': str(root / 'prepared/manifest.json'), 'samples_per_cycle': 2,
        'sampling': {'kind': 'epochs', 'shuffle': False}}}}
    config['algorithm']['factory'] = str(workflow) + ':build'
    config['trainer'].update(cycles=1, output_dir=str(tmp_path / 'first'))
    path = root / 'streams.json'; path.write_text(json.dumps(config))
    result = run_cli('train', path)
    evaluated = run_cli('evaluate', path, '--checkpoint', result['checkpoint'],
                        '--output', tmp_path / 'evaluated', '--sample-index', 0)
    assert evaluated['sample_id'] == 'sample-000000'


def _multistream_fixture(tmp_path):
    """Real prepared media and two independently consumed stream records."""
    root = tmp_path / 'media'
    build_fixture(root, samples=12)
    prepare(root / 'raw/manifest.json', root / 'prepared')
    workflow = tmp_path / 'multistream-workflow.py'
    workflow.write_text('''from distillschedule.examples.tiny_av import build_algorithm
def build(components, feeds, parameters, cycle, seed):
    inputs = dict(feeds['generator'][0])
    inputs['condition'] = (feeds['critic'][0]['condition'] + feeds['critic'][1]['condition']) / 2
    return build_algorithm(components, inputs, parameters, cycle, seed)
''')
    raw = json.loads((root / 'train.json').read_text())
    raw['data'] = {'streams': {name: {
        'prepared': str(root / 'prepared/manifest.json'), 'samples_per_cycle': count}
        for name, count in (('generator', 1), ('critic', 2))}}
    raw['algorithm']['factory'] = str(workflow) + ':build'
    path = root / 'multistream.json'
    path.write_text(json.dumps(raw))
    return root, path


def test_two_rank_unequal_stream_counts_resume_all_state_and_sample_order(tmp_path):
    from test_training_distributed import _launch
    from test_training_end_to_end import equal
    _, config = _multistream_fixture(tmp_path)
    first = _launch(tmp_path, config, 'first', 1, 'post_backward')
    resumed = _launch(tmp_path, config, 'resumed', 3, 'post_backward',
                      first / 'checkpoints/cycle-000001')
    whole = _launch(tmp_path, config, 'whole', 3, 'post_backward')
    for rank in range(2):
        relative = f'checkpoints/cycle-000003/rank-{rank:05d}.pt'
        assert equal(torch.load(resumed / relative, weights_only=True),
                     torch.load(whole / relative, weights_only=True))
        rows = [json.loads(line) for line in
                (resumed / f'ranks/rank-{rank:05d}/metrics.jsonl').read_text().splitlines()]
        for row, cycle in zip(rows, (1, 2)):
            assert row['sample_id'] == {
                'generator': [f'sample-{cycle * 2 + rank:06d}'],
                'critic': [f'sample-{cycle * 4 + rank:06d}',
                           f'sample-{cycle * 4 + 2 + rank:06d}']}
            assert row['data_progress']['generator']['cursor'] == (cycle + 1) * 2
            assert row['data_progress']['critic']['cursor'] == (cycle + 1) * 4
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('corruption', ['missing', 'inconsistent', 'shard_disagreement'])
@pytest.mark.parametrize('operation', ['train', 'evaluate'])
def test_public_multistream_checkpoint_rejects_corrupt_progress_before_output(tmp_path, corruption, operation):
    from pathlib import Path
    from test_training_end_to_end import run_cli
    from distillschedule.workloads.best_known.registry import file_sha256
    import os
    import subprocess
    import sys
    root, config = _multistream_fixture(tmp_path)
    trained = run_cli('train', config, '--cycles', 1, '--output', root / 'first')
    checkpoint = Path(trained['checkpoint'])
    payload = torch.load(checkpoint / 'state.pt', weights_only=True)
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    if corruption == 'missing':
        payload.pop('data_progress')
        manifest.pop('data_progress')
    else:
        payload['data_progress']['critic']['cursor'] += 1
        if corruption == 'inconsistent':
            manifest['data_progress'] = payload['data_progress']
    torch.save(payload, checkpoint / 'state.pt')
    manifest['files']['state.pt'] = file_sha256(checkpoint / 'state.pt')
    (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
    (checkpoint / '_SUCCESS').write_text(file_sha256(checkpoint / 'manifest.json') + '\n')
    before = {p.name: file_sha256(p) for p in checkpoint.iterdir() if p.is_file()}
    output = root / 'rejected'
    arguments = (['--resume', str(checkpoint), '--cycles', '2'] if operation == 'train'
                 else ['--checkpoint', str(checkpoint)])
    result = subprocess.run([sys.executable, '-m', 'distillschedule', operation, str(config),
                             *arguments, '--output', str(output)],
                            text=True, capture_output=True, env=os.environ.copy(), timeout=90)
    assert result.returncode != 0
    assert 'progress' in result.stderr, result.stdout + result.stderr
    assert not output.exists()
    assert before == {p.name: file_sha256(p) for p in checkpoint.iterdir() if p.is_file()}
