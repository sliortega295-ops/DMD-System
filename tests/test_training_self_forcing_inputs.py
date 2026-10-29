"""Actual optional native T5/tokenizer and strict embedding preparation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from test_training_self_forcing_cpu import source_root, fixture_assets
from test_training_end_to_end import equal


@pytest.fixture
def text_assets(tmp_path, source_root):
    import sentencepiece as spm
    from transformers import T5Tokenizer
    from distillschedule.training.self_forcing import original
    from distillschedule.workloads.best_known.registry import file_sha256
    api = original(source_root)
    root = tmp_path / 'native-local-text'; root.mkdir()
    corpus = root / 'corpus.txt'
    corpus.write_text('hello world\na different world\nnew actual prompt\nhello other word\n')
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(root / 'spiece'),
        vocab_size=32, model_type='char', pad_id=0, unk_id=1, bos_id=-1, eos_id=2,
        hard_vocab_limit=False, minloglevel=2)
    tokenizer = T5Tokenizer(vocab_file=str(root / 'spiece.model'), extra_ids=0, model_max_length=8)
    tokenizer.save_pretrained(root / 'tokenizer')
    config = dict(vocab=tokenizer.vocab_size, dim=16, dim_attn=16, dim_ffn=32,
                  num_heads=2, num_layers=1, num_buckets=8, shared_pos=True, dropout=0.)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(765)
        encoder = api.t5.T5Encoder(**config).eval()
        torch.save(encoder.state_dict(), root / 't5.pt')
    spec = dict(source_root=str(source_root), config=config,
        weights=dict(path='t5.pt', sha256=file_sha256(root / 't5.pt')), tokenizer='tokenizer', text_len=8,
        tokenizer_use_fast=False)
    return root, spec


@pytest.fixture
def bf16_text_assets(text_assets):
    """Tiny original-class BF16 state, never a converted pretrained asset."""
    from distillschedule.training.self_forcing import original
    from distillschedule.workloads.best_known.registry import file_sha256
    root, spec = text_assets
    spec = deepcopy(spec)
    api = original(spec['source_root'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(765)
        encoder = api.t5.T5Encoder(**spec['config']).to(dtype=torch.bfloat16).eval()
        torch.save(encoder.state_dict(), root / 't5-bfloat16.pt')
    spec['weights'] = dict(path='t5-bfloat16.pt', sha256=file_sha256(root / 't5-bfloat16.pt'))
    spec['precision'] = dict(weights='bfloat16', output='float32')
    return root, spec


def test_prompt_only_is_original_local_native_t5_not_a_synthetic_conditioner(text_assets):
    from distillschedule.training.self_forcing import original
    from distillschedule.data.self_forcing_inputs import build_local_text_encoder, encode_text, prepare_inputs, SCHEMA
    from distillschedule.data import PreparedDataset
    root, spec = text_assets
    api = original(spec['source_root'])
    encoder, documents = build_local_text_encoder(spec, base_dir=root)
    assert type(encoder.text_encoder) is api.t5.T5Encoder
    assert encoder.forward.__func__ is api.wrapper.WanTextEncoder.forward
    assert encoder.device == torch.device('cpu')
    assert all(not p.requires_grad for p in encoder.parameters())
    row = encode_text([' hello   world ', 'new actual prompt'], spec, base_dir=root)
    ids, mask = encoder.tokenizer([' hello   world ', 'new actual prompt'], return_mask=True, add_special_tokens=True)
    with torch.no_grad():
        expected = encoder.text_encoder(ids, mask)
        for tensor, length in zip(expected, mask.gt(0).sum(1).long()):
            tensor[length:] = 0.
    assert equal(row, {'prompt_embeds': expected})
    source = root / 'prompts.json'
    source.write_text(json.dumps(dict(schema_version=SCHEMA, mode='prompts', encoder=spec,
        records=[dict(id='actual-t5-prompt', prompt=[' hello   world ', 'new actual prompt'])])))
    prepared = prepare_inputs(source, root / 'prepared')
    dataset = PreparedDataset(prepared)
    assert equal(dataset.sample(0), row)
    assert dataset.validate(sources=True)['status'] == 'PASS'
    assert dataset.manifest['metadata']['encoder_runtime']['torch']
    assert len(documents) >= 24  # source lock + local checkpoint/tokenizer
    # The same exact native T5 produces a pinned negative-prompt condition.
    negative = encode_text('', spec, base_dir=root)['prompt_embeds']
    assert negative.shape == (1, 8, 16) and negative.dtype == torch.float32


def test_explicit_bf16_native_t5_widens_only_the_actual_upstream_output(bf16_text_assets):
    from types import SimpleNamespace
    from distillschedule.training.self_forcing import original
    from distillschedule.data.self_forcing_inputs import build_local_text_encoder, encode_text, encode_record
    from distillschedule.workloads.best_known.registry import file_sha256
    root, spec = bf16_text_assets
    api = original(spec['source_root'])
    weights = torch.load(root / spec['weights']['path'], weights_only=True)
    encoder, documents = build_local_text_encoder(spec, base_dir=root)
    assert type(encoder.text_encoder) is api.t5.T5Encoder
    assert encoder.forward.__func__ is api.wrapper.WanTextEncoder.forward
    assert all(value.dtype == torch.bfloat16 for value in encoder.text_encoder.state_dict().values())
    assert equal(encoder.text_encoder.state_dict(), weights)
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    # The oracle is a separate original T5 plus the exact original wrapper body.
    oracle = api.t5.T5Encoder(**spec['config']).to(dtype=torch.bfloat16).eval().requires_grad_(False)
    oracle.load_state_dict(weights, strict=True)
    prompts = [' hello   world ', 'new actual prompt']
    with torch.no_grad():
        native = api.wrapper.WanTextEncoder.forward(SimpleNamespace(
            text_encoder=oracle, tokenizer=encoder.tokenizer, device=torch.device('cpu')), text_prompts=prompts)
        actual = encoder(text_prompts=prompts)
    assert native['prompt_embeds'].dtype == torch.bfloat16
    assert equal(actual, native)
    widened = native['prompt_embeds'].float()
    row = encode_text(prompts, spec, base_dir=root)
    assert row['prompt_embeds'].dtype == torch.float32
    assert torch.equal(row['prompt_embeds'].view(torch.int32), widened.view(torch.int32))
    parameters = dict(mode='prompts', source_dir=str(root), encoder=spec)
    for cached in (None, encoder):
        record = encode_record(dict(id='native-bf16', prompt=prompts), parameters, encoder=cached)
        assert torch.equal(record['prompt_embeds'].view(torch.int32), widened.view(torch.int32))
    negative = encode_text('', spec, base_dir=root)['prompt_embeds']
    assert negative.shape == (1, 8, 16) and negative.dtype == torch.float32
    assert next(item for item in documents if item['path'] == str(root / spec['weights']['path']))['sha256'] == spec['weights']['sha256']
    assert file_sha256(root / spec['weights']['path']) == spec['weights']['sha256']
    assert not torch.cuda.is_initialized()


def test_prepared_bf16_native_prompt_records_keep_fp32_layout_and_precision_provenance(bf16_text_assets):
    from distillschedule.data.self_forcing_inputs import encode_text, prepare_inputs, SCHEMA
    from distillschedule.data import PreparedDataset
    from distillschedule.workloads.best_known.registry import fingerprint
    root, spec = bf16_text_assets
    prompts = ['hello world', 'new actual prompt']
    source = root / 'bf16-prompts.json'
    source.write_text(json.dumps(dict(schema_version=SCHEMA, mode='prompts', encoder=spec,
        records=[dict(id='bf16-native-prompt', prompt=prompts)])))
    dataset = PreparedDataset(prepare_inputs(source, root / 'bf16-prepared'))
    expected = encode_text(prompts, spec, base_dir=root)
    assert equal(dataset.sample(0), expected)
    assert dataset.validate(sources=True)['status'] == 'PASS'
    assert dataset.manifest['metadata']['layout'] == 'EXACT_NATIVE_FP32_BLC'
    identity = dataset.manifest['provenance']['encoder']
    assert identity['parameters']['encoder'] == spec
    assert identity['parameters']['encoder']['precision'] == dict(weights='bfloat16', output='float32')
    assert identity['parameters_sha256'] == fingerprint(identity['parameters'])
    assert dataset.manifest['metadata']['encoder_runtime']['torch']
    # The dtype declaration cannot hide changed source bytes after publication.
    checkpoint = root / spec['weights']['path']
    with checkpoint.open('ab') as stream:
        stream.write(b'changed-after-publish')
    with pytest.raises(ValueError, match='source hash changed'):
        dataset.validate(sources=True)


def test_absent_and_explicit_fp32_precision_preserve_original_native_bitwise_path(text_assets):
    from distillschedule.data.self_forcing_inputs import build_local_text_encoder, encode_text, prepare_inputs, SCHEMA
    from distillschedule.data import PreparedDataset
    root, spec = text_assets
    declared = deepcopy(spec)
    declared['precision'] = dict(weights='float32', output='float32')
    encoder, _ = build_local_text_encoder(spec, base_dir=root)
    prompts = [' hello   world ', 'new actual prompt']
    with torch.no_grad():
        original = encoder(text_prompts=prompts)
    assert all(value.dtype == torch.float32 for value in encoder.text_encoder.state_dict().values())
    for candidate in (spec, declared):
        row = encode_text(prompts, candidate, base_dir=root)
        assert torch.equal(row['prompt_embeds'].view(torch.int32), original['prompt_embeds'].view(torch.int32))
    source = root / 'default-prompts.json'
    source.write_text(json.dumps(dict(schema_version=SCHEMA, mode='prompts', encoder=spec,
        records=[dict(id='default-native-prompt', prompt=prompts)])))
    dataset = PreparedDataset(prepare_inputs(source, root / 'default-prepared'))
    assert 'precision' not in dataset.manifest['provenance']['encoder']['parameters']['encoder']
    assert equal(dataset.sample(0), original)


@pytest.mark.parametrize('precision', [None, 'bfloat16', {}, {'weights': 'bfloat16'},
    {'output': 'float32'}, {'weights': 'float16', 'output': 'float32'},
    {'weights': 'bfloat16', 'output': 'bfloat16'}, {'weights': 'float32', 'output': 'float16'},
    {'weights': True, 'output': 'float32'},
    {'weights': 'bfloat16', 'output': 'float32', 'normalize': True}])
def test_native_text_precision_declaration_fails_closed(text_assets, precision):
    from distillschedule.data.self_forcing_inputs import prepare_inputs, SCHEMA
    root, spec = text_assets
    spec = deepcopy(spec)
    spec['precision'] = precision
    source = root / 'bad-precision.json'
    source.write_text(json.dumps(dict(schema_version=SCHEMA, mode='prompts', encoder=spec,
        records=[dict(id='prompt', prompt='hello world')])))
    with pytest.raises(ValueError, match='precision'):
        prepare_inputs(source, root / 'must-not-publish-precision')
    assert not (root / 'must-not-publish-precision').exists()


@pytest.mark.parametrize('case', ['undeclared_bf16', 'declared_fp32_bf16', 'declared_bf16_fp32', 'mixed_dtype',
    'nonfinite', 'sparse', 'missing_key', 'wrong_shape', 'wrong_hash', 'unknown_encoder', 'tokenizer_escape'])
def test_explicit_native_text_precision_preserves_checkpoint_and_local_asset_gates(bf16_text_assets, case):
    from distillschedule.data.self_forcing_inputs import prepare_inputs, SCHEMA
    from distillschedule.workloads.best_known.registry import file_sha256
    root, spec = bf16_text_assets
    spec = deepcopy(spec)
    checkpoint = root / spec['weights']['path']
    if case == 'undeclared_bf16':
        spec.pop('precision')
    elif case == 'declared_fp32_bf16':
        spec['precision']['weights'] = 'float32'
    elif case == 'declared_bf16_fp32':
        spec['weights'] = dict(path='t5.pt', sha256=file_sha256(root / 't5.pt'))
    elif case in ('mixed_dtype', 'nonfinite', 'sparse', 'missing_key', 'wrong_shape'):
        weights = torch.load(checkpoint, weights_only=True)
        name = next(iter(weights))
        if case == 'mixed_dtype':
            weights[name] = weights[name].float()
        elif case == 'nonfinite':
            weights[name].view(-1)[0] = float('nan')
        elif case == 'sparse':
            weights[name] = weights[name].to_sparse()
        elif case == 'missing_key':
            weights.pop(name)
        else:
            weights[name] = weights[name].reshape(-1)[:1]
        torch.save(weights, checkpoint)
        spec['weights']['sha256'] = file_sha256(checkpoint)
    elif case == 'wrong_hash':
        spec['weights']['sha256'] = '0' * 64
    elif case == 'unknown_encoder':
        spec['download_model'] = True
    else:
        outside = root.parent / 'outside-tokenizer'
        outside.mkdir()
        (outside / 'tokenizer_config.json').write_text('{}')
        (root / 'escaped-tokenizer').symlink_to(outside, target_is_directory=True)
        spec['tokenizer'] = 'escaped-tokenizer'
    source = root / 'bad-bf16.json'
    source.write_text(json.dumps(dict(schema_version=SCHEMA, mode='prompts', encoder=spec,
        records=[dict(id='prompt', prompt='hello world')])))
    with pytest.raises(ValueError):
        prepare_inputs(source, root / 'must-not-publish-bf16')
    assert not (root / 'must-not-publish-bf16').exists()


@pytest.mark.parametrize('corruption', ['weights_hash', 'weights_dtype', 'missing_local_tokenizer', 'unknown_encoder'])
def test_native_text_assets_fail_closed_no_download_or_partial_publish(text_assets, corruption):
    from distillschedule.data.self_forcing_inputs import prepare_inputs, SCHEMA
    from distillschedule.workloads.best_known.registry import file_sha256
    root, spec = text_assets
    spec = deepcopy(spec)
    if corruption == 'weights_hash':
        spec['weights']['sha256'] = '0' * 64
    elif corruption == 'weights_dtype':
        value = torch.load(root / 't5.pt', weights_only=True)
        value = {name: tensor.double() for name, tensor in value.items()}
        torch.save(value, root / 't5.pt')
        spec['weights']['sha256'] = file_sha256(root / 't5.pt')
    elif corruption == 'missing_local_tokenizer':
        spec['tokenizer'] = 'NO_LOCAL_TOKENIZER'
    else:
        spec['download_model'] = True
    source = root / 'bad.json'
    source.write_text(json.dumps(dict(schema_version=SCHEMA, mode='prompts', encoder=spec,
        records=[dict(id='prompt', prompt='hello world')])))
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_inputs(source, root / 'must-not-publish')
    assert not (root / 'must-not-publish').exists()


@pytest.mark.parametrize('corruption', ['nan', 'int', 'bfloat16', 'layout', 'gradient', 'field', 'shape'])
def test_preencoded_native_condition_rejection(corruption):
    from distillschedule.data.self_forcing_inputs import validate_row
    row = {'prompt_embeds': torch.zeros(1, 8, 16)}
    if corruption == 'nan':
        row['prompt_embeds'][0, 0, 0] = float('nan')
    elif corruption == 'int':
        row['prompt_embeds'] = row['prompt_embeds'].long()
    elif corruption == 'bfloat16':
        row['prompt_embeds'] = row['prompt_embeds'].bfloat16()
    elif corruption == 'layout':
        row['prompt_embeds'] = row['prompt_embeds'].to_sparse()
    elif corruption == 'gradient':
        row['prompt_embeds'].requires_grad_(True)
    elif corruption == 'field':
        row['fake_pixels'] = torch.zeros(1)
    else:
        row['prompt_embeds'] = torch.zeros(1, 16, 8)
    with pytest.raises(ValueError):
        validate_row(row, shape=(1, 8, 16))


def test_raw_native_t5_prompt_streams_reach_real_training_and_video(text_assets, fixture_assets, tmp_path):
    """Join the actual offline encoder to the same public native training path."""
    from distillschedule.data.self_forcing_inputs import encode_text, prepare_inputs, SCHEMA
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import train, evaluate
    from distillschedule.workloads.best_known.registry import file_sha256
    root, encoder = text_assets
    negative = encode_text('', encoder, base_dir=root)['prompt_embeds']
    assets_path = Path(fixture_assets['assets'])/'assets.json'
    assets = json.loads(assets_path.read_text())
    negative_path = assets_path.parent/assets['negative_prompt_embeds']['path']
    torch.save(negative, negative_path)
    assets['negative_prompt_embeds']['sha256'] = file_sha256(negative_path)
    assets_path.write_text(json.dumps(assets))
    config_path = Path(fixture_assets['config'])
    config = json.loads(config_path.read_text())
    config['model']['parameters']['settings']['negative_prompt'] = ''
    config['data']['streams'] = {}
    for role, count in (('generator', 1), ('critic', 5)):
        source = root/(role+'-raw.json')
        source.write_text(json.dumps(dict(schema_version=SCHEMA, mode='prompts', encoder=encoder,
            records=[dict(id=f'{role}-raw-{index}', prompt=('hello world' if role == 'generator'
                else 'a different world')+f' {index}') for index in range(count)])))
        prepared = prepare_inputs(source, root/(role+'-prepared'))
        config['data']['streams'][role] = dict(prepared=str(prepared), samples_per_cycle=count)
    config['trainer'].update(cycles=1, output_dir=str(tmp_path/'actual-text-run'))
    config_path.write_text(json.dumps(config))
    request = load_training_config(config_path)
    result = train(request)
    assert result['final_versions']['native_cycle'] == 1
    generated = evaluate(request, result['checkpoint'], tmp_path/'actual-text-video')
    assert generated['video']['shape'] == [81, 32, 32, 3]
    assert len(list(Path(generated['video']['frames']).glob('*.png'))) == 81
    assert not torch.cuda.is_initialized()
