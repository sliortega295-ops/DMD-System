"""Independent W2 source/rank/control/freshness admission, no numerical stand-in."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import torch

from test_training_helios_nogan_public_cpu import fixture_assets as fixture_assets
from test_training_helios_nogan_bootstrap_cpu import _write,_pin
from test_training_helios_public_cpu import STACK_AVAILABLE
from distillschedule.data import PreparedDataset
from distillschedule.data import helios_nogan_reference_inputs as inputs
from distillschedule.training.helios_inputs import HeliosCohortTask, MAX_COHORTS
from distillschedule.workloads.best_known.registry import fingerprint

if os.environ.get('HELIOS_NOGAN_PUBLIC_REQUIRE_STACK') == '1' and not STACK_AVAILABLE:
    raise RuntimeError('required full W2 CPU stack missing: Diffusers0.37.1/PEFT0.17.1')
pytestmark = pytest.mark.skipif(not STACK_AVAILABLE,
    reason='separate full original Diffusers0.37.1/PEFT0.17.1 W2 CPU proof')


def _document(feeds):
    return json.loads(bytes(feeds['control'][:feeds['control_length'].item()].tolist()))


def _encode(feeds,document):
    raw = json.dumps(document,sort_keys=True).encode()
    feeds = deepcopy(feeds)
    feeds['control'].zero_(); feeds['control'][:len(raw)] = torch.tensor(list(raw),dtype=torch.uint8)
    feeds['control_length'][0] = len(raw)
    return feeds


def test_original_all16_controls_rank0_payload_and_step_conditioning(fixture_assets):
    dataset = PreparedDataset(fixture_assets['prepared'])
    seen = []
    for index in range(3):
        trace = inputs.decode(dataset.sample(index),HeliosCohortTask(index))
        trace.validate_cpu_reference()
        assert len(trace._control) == 16 and set(trace._loaded) == {0}
        assert set(inputs.content_identity(trace.reference_source)) == {
            f'step/{step}/sample/{name}' for step in trace.task.source_steps for name in ('text_embedding','attention_mask')}
        entry = trace.load_generator_entry(rank=0,device='cpu')
        assert entry.tensors['entry_noise'].shape == (1,16,9,48,80)
        assert entry.tensors['text_embedding'].shape == (1,512,4096)
        assert inputs.cpu_telemetry('cpu',trace) is True
        assert inputs.cpu_telemetry('cpu',entry.operation_trace) is True
        with pytest.raises(ValueError,match='source rank0'):
            trace.load_generator_entry(rank=1,device='cpu')
        seen.append(inputs.conditioning_fingerprint(trace.reference_document['content']))
    assert len(set(seen)) == 3


def test_conditioning_freshness_does_not_confuse_changed_step_names_with_new_bytes(fixture_assets):
    feeds = PreparedDataset(fixture_assets['prepared']).sample(0)
    content = _document(feeds)['content']
    copied = {'step/'+str(int(key.split('/')[1])+5)+'/'+key.split('/',2)[2]:value for key,value in content.items()}
    assert content != copied and inputs.conditioning_fingerprint(content) == inputs.conditioning_fingerprint(copied)


@pytest.mark.parametrize('change', ['scope','seed','source','task','rank_control','source_bytes','tensor','padding'])
def test_control_seed_source_task_bytes_and_unqualified_cpu_refuse_before_write(fixture_assets,change):
    feeds = PreparedDataset(fixture_assets['prepared']).sample(0)
    doc = _document(feeds)
    if change == 'scope':
        doc['scope'] = 'CPU1_RANK0_PER_SAMPLE_ORIGINAL_W1_NOT_WORLD16'
    elif change == 'seed':
        doc['sources']['seed'] += 1
    elif change == 'source':
        doc['source'] = '0'*64
    elif change == 'task':
        doc['task']['index'] = 1
    elif change == 'rank_control':
        doc['manifests'][15]['contract']['full_latent_shape'][-1] = 16
    elif change == 'source_bytes':
        doc['source_hashes'][1] = doc['source_hashes'][0]
    elif change == 'tensor':
        feeds['tensors'][0] = feeds['tensors'][0].clone(); feeds['tensors'][0].flatten()[0] += 1
    elif change == 'padding':
        feeds['control'][-1] = 1
    if change not in ('tensor','padding'):
        feeds = _encode(feeds,doc)
    with pytest.raises((ValueError,AssertionError,TypeError)):
        inputs.decode(feeds,HeliosCohortTask(0))


@pytest.mark.parametrize('count',[1,MAX_COHORTS + 1])
def test_finite_cohort_capacity_rejects_before_publication(fixture_assets,tmp_path,count):
    source = json.loads((Path(fixture_assets['root'])/'training-inputs/source.json').read_text())
    stream = json.loads(Path(source['stream']['path']).read_text())
    for key in ('sources','inventory'):
        stream['identity'][key] = (stream['identity'][key]*(MAX_COHORTS + 1))[:count]
    stream['cohorts'] = (stream['cohorts']*(MAX_COHORTS + 1))[:count]
    stream.pop('stream_sha256'); stream['stream_sha256'] = fingerprint(stream)
    source['stream'] = _pin(_write(tmp_path/'stream.json',stream))
    path = _write(tmp_path/'source.json',source)
    with pytest.raises(ValueError,match='cycles/source'):
        inputs.prepare_inputs(path,tmp_path/'out')
    assert not (tmp_path/'out').exists()



def test_duplicate_cohorts_inside_capacity_still_refuse_before_publication(fixture_assets,tmp_path):
    source = json.loads((Path(fixture_assets['root'])/'training-inputs/source.json').read_text())
    stream = json.loads(Path(source['stream']['path']).read_text())
    # A legal count must not disguise reused feature paths as fresh data.
    for key in ('sources','inventory'):
        stream['identity'][key] = [stream['identity'][key][0]] * 3
    stream['cohorts'] = [stream['cohorts'][0]] * 3
    stream.pop('stream_sha256'); stream['stream_sha256'] = fingerprint(stream)
    source['stream'] = _pin(_write(tmp_path/'stream.json',stream))
    with pytest.raises(ValueError,match='globally distinct feature paths'):
        inputs.prepare_inputs(_write(tmp_path/'source.json',source),tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_reordered_stream_refuses_and_cpu_gate_is_closed(fixture_assets,tmp_path):
    source = json.loads((Path(fixture_assets['root'])/'training-inputs/source.json').read_text())
    stream = json.loads(Path(source['stream']['path']).read_text())
    stream['cohorts'][0],stream['cohorts'][1] = stream['cohorts'][1],stream['cohorts'][0]
    stream.pop('stream_sha256'); stream['stream_sha256'] = fingerprint(stream)
    source['stream'] = _pin(_write(tmp_path/'stream.json',stream))
    with pytest.raises(ValueError,match='cohort order/path/task'):
        inputs.prepare_inputs(_write(tmp_path/'source.json',source),tmp_path/'out')
    from types import SimpleNamespace
    from distillschedule.data.wan_reference_inputs import cpu_telemetry
    with pytest.raises(ValueError,match='original W1 reference trace'):
        cpu_telemetry('cpu',SimpleNamespace(validate_cpu_reference=lambda:None))
    with pytest.raises(ValueError,match='original W2 reference trace'):
        inputs.cpu_telemetry('cpu',SimpleNamespace(validate_cpu_reference=lambda:None))
    assert inputs.cpu_telemetry('cuda',object()) is False


def test_cpu_topology_and_cli_dispatch_are_explicit(fixture_assets,tmp_path,monkeypatch,capsys):
    feeds = PreparedDataset(fixture_assets['prepared']).sample(0)
    monkeypatch.setenv('WORLD_SIZE','2')
    with pytest.raises(ValueError,match='CPU1/rank0'):
        inputs.decode(feeds,HeliosCohortTask(0))
    monkeypatch.setenv('WORLD_SIZE','1')
    from distillschedule.data.cli import main
    assert main(['prepare-helios-nogan-reference','--sources',str(tmp_path/'missing.json'),'--output',str(tmp_path/'out')]) == 2
    assert 'ERROR' in capsys.readouterr().err
