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
