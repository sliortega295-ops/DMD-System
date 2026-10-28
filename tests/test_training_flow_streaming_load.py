"""SYNTHETIC_TEST_ONLY: real upstream T5 safe weights and bounded shard lifetime."""
import gc
import json
import os
import weakref

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('DMD_MODERN_FLOW_CPU') != '1',
    reason='requires the existing pinned modern-flow CPU environment')


def _checkpoint(path, dtype):
    import torch
    from transformers import T5Config, T5EncoderModel
    from safetensors.torch import load_file, save_file

    torch.manual_seed(104)
    native = T5EncoderModel(T5Config(vocab_size=32, d_model=16, d_ff=32,
        d_kv=8, num_layers=1, num_heads=2, dropout_rate=0.)).float().eval()
    native.to(dtype=dtype)
    native.save_pretrained(path, safe_serialization=True, max_shard_size='1KB')
    # Exercise the same tied Parameter supplied in two different files.
    index_path = next(path.glob('*.safetensors.index.json'))
    index = json.loads(index_path.read_text())
    first = index['weight_map']['shared.weight']
    other = next(p for p in sorted(path.glob('*.safetensors')) if p.name != first)
    payload = load_file(other)
    payload['encoder.embed_tokens.weight'] = native.shared.weight.detach().clone()
    save_file(payload, other)
    index['weight_map']['encoder.embed_tokens.weight'] = other.name
    index_path.write_text(json.dumps(index))
    return native.float()


@pytest.mark.parametrize('dtype_name', ['float32', 'bfloat16'])
def test_native_loading_releases_each_shard_before_reading_the_next(tmp_path, monkeypatch, dtype_name):
    import torch
    from safetensors import torch as safe_torch
    from distillschedule.training.flow_image import load_native

    native = _checkpoint(tmp_path, getattr(torch, dtype_name))
    original = safe_torch.load_file
    previous = []
    loaded_files = []

    def observe(path, *args, **kwargs):
        gc.collect()
        assert all(ref() is None for ref in previous), 'previous checkpoint shard is still retained'
        payload = original(path, *args, **kwargs)
        previous[:] = [weakref.ref(value) for value in payload.values()]
        loaded_files.append(path)
        return payload

    monkeypatch.setattr(safe_torch, 'load_file', observe)
    loaded, _ = load_native(type(native), tmp_path, 100_000,
        dtype_policy='bf16_or_fp32_to_fp32' if dtype_name == 'bfloat16' else 'strict_fp32')
    gc.collect()
    assert all(ref() is None for ref in previous), 'final shard leaked into the live native model'
    assert len(loaded_files) == len(list(tmp_path.glob('*.safetensors'))) > 1
    assert loaded.shared.weight is loaded.encoder.embed_tokens.weight
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in native.state_dict().items())
    inputs = dict(input_ids=torch.tensor([[3, 4, 5, 0]]), attention_mask=torch.tensor([[1, 1, 1, 0]]))
    expected = native(**inputs).last_hidden_state
    actual = loaded(**inputs).last_hidden_state
    assert torch.equal(expected, actual)
    expected.square().sum().backward()
    actual.square().sum().backward()
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(native.parameters(), loaded.parameters(), strict=True))
    assert not torch.cuda.is_initialized()


def test_streaming_native_constructor_keeps_success_rng_and_rejects_nonfinite_without_consuming_rng(tmp_path):
    import torch
    from safetensors.torch import load_file, save_file
    from distillschedule.training.flow_image import architecture, load_native, shell

    native = _checkpoint(tmp_path, torch.float32)
    cls = type(native)
    raw = json.loads((tmp_path/'config.json').read_text())
    torch.manual_seed(209)
    shell(cls, raw).float()
    with torch.device('cpu'):
        cls(architecture(cls, raw)).float()
    expected = torch.get_rng_state().clone()
    torch.manual_seed(209)
    load_native(cls, tmp_path, 100_000)
    assert torch.equal(torch.get_rng_state(), expected)

    final = sorted(tmp_path.glob('*.safetensors'))[-1]
    payload = load_file(final)
    key = next(iter(payload))
    payload[key] = payload[key].clone().fill_(float('nan'))
    save_file(payload, final)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='nonfinite'):
        load_native(cls, tmp_path, 100_000)
    assert torch.equal(torch.get_rng_state(), before)
    assert not torch.cuda.is_initialized()
