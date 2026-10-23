"""Actual optional native T5/tokenizer and strict embedding preparation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from test_training_self_forcing_cpu import source_root
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


@pytest.mark.parametrize('corruption', ['nan', 'int', 'layout', 'gradient', 'field', 'shape'])
def test_preencoded_native_condition_rejection(corruption):
    from distillschedule.data.self_forcing_inputs import validate_row
    row = {'prompt_embeds': torch.zeros(1, 8, 16)}
    if corruption == 'nan':
        row['prompt_embeds'][0, 0, 0] = float('nan')
    elif corruption == 'int':
        row['prompt_embeds'] = row['prompt_embeds'].long()
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
