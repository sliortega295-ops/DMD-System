"""SYNTHETIC_TEST_ONLY: tiny native models, source-derived public config ABI.

Metadata provenance (no pretrained tensors or full model construction):
FLUX.2-klein-base-4B a3b4f4849157f664bdbc776fd7453c2783562f4d;
Qwen-Image 75e0b4be04f60ec59a75f475837eced720f823b6;
Wan2.2-TI2V-5B-Diffusers b8fff7315c768468a5333511427288870b2e9635;
HunyuanVideo-1.5-Diffusers-480p_t2v 286be7ce72277246578a3e3cc2487e95ddae5bcf.
All architecture sizes/token IDs below are deliberately reduced for CPU tests.
"""
from copy import deepcopy
import json
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('DMD_MODERN_FLOW_CPU') != '1',
    reason='opt in to the isolated modern-flow pinned CPU stack with DMD_MODERN_FLOW_CPU=1')


def legacy_case(name):
    import transformers
    text = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        attention_dropout=0., use_sliding_window=False, sliding_window=None,
        rope_theta=1000000., rope_scaling=dict(type='default', rope_type='default', mrope_section=[1, 1, 2]),
        bos_token_id=1, eos_token_id=2, vision_token_id=29, tie_word_embeddings=False)
    if name == 'flux':
        text.pop('vision_token_id'); text['rope_scaling'] = None
        text.update(model_type='qwen3', head_dim=8, tie_word_embeddings=True)
        return transformers.Qwen3ForCausalLM, text
    if name == 'hunyuan':
        return transformers.Qwen2_5_VLTextModel, dict(text, model_type='qwen2_5_vl_text')
    if name == 'qwen':
        text.pop('tie_word_embeddings')
        return transformers.Qwen2_5_VLForConditionalGeneration, dict(text, model_type='qwen2_5_vl',
            image_token_id=30, video_token_id=31, vision_start_token_id=27, vision_end_token_id=28,
            tie_word_embeddings=False, vision_config=dict(model_type='qwen2_5_vl', depth=1,
                hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16,
                in_channels=3, in_chans=3, patch_size=2, spatial_patch_size=2,
                temporal_patch_size=1, spatial_merge_size=1, fullatt_block_indexes=[0]))
    config = dict(model_type='umt5' if name == 'wan22' else 't5', vocab_size=32, d_model=16,
        d_ff=32, d_kv=8, num_layers=1, num_decoder_layers=1, num_heads=2,
        dropout_rate=0., feed_forward_proj='gated-gelu', dense_act_fn='gelu_new', is_gated_act=True,
        decoder_start_token_id=0, tie_word_embeddings=False, use_cache=False,
        tokenizer_class='T5Tokenizer' if name == 'wan22' else 'ByT5Tokenizer')
    if name == 'wan22':
        config.update(output_past=True, scalable_attention=True)
    else:
        config.update(gradient_checkpointing=False)
    return getattr(transformers, 'UMT5EncoderModel' if name == 'wan22' else 'T5EncoderModel'), config


@pytest.mark.parametrize('name', ['flux', 'qwen', 'wan22', 'hunyuan', 'hunyuan_t5'])
def test_source_derived_legacy_config_uses_exact_native_parser_without_mutating_raw(name):
    from distillschedule.training.flow_image import architecture
    cls, raw = legacy_case(name)
    original = deepcopy(raw)
    expected = cls.config_class.from_dict(deepcopy(raw))
    actual = architecture(cls, raw)
    assert actual.to_dict() == expected.to_dict()
    assert raw == original
    if name == 'hunyuan_t5':
        assert actual.tie_word_embeddings is True and actual.scale_decoder_outputs is False


@pytest.mark.parametrize('sharded', [False, True])
@pytest.mark.parametrize('classname', ['T5EncoderModel', 'UMT5EncoderModel'])
def test_native_safe_save_omitted_identical_parameter_alias_strict_loads_exactly(tmp_path, classname, sharded):
    import torch
    import transformers
    from safetensors.torch import load_file
    from distillschedule.training.flow_image import load_native
    cls = getattr(transformers, classname)
    config = cls.config_class(vocab_size=32, d_model=16, d_ff=32, d_kv=8,
        num_layers=1, num_heads=2, dropout_rate=0.)
    native = cls(config).float().eval()
    native.save_pretrained(tmp_path, safe_serialization=True, max_shard_size='1KB' if sharded else '5GB')
    payload = {key: value for path in tmp_path.glob('*.safetensors') for key, value in load_file(path).items()}
    assert 'shared.weight' in payload and 'encoder.embed_tokens.weight' not in payload
    actual, raw = load_native(cls, tmp_path, 100_000)
    assert raw == json.loads((tmp_path/'config.json').read_text())
    assert actual.shared.weight is actual.encoder.embed_tokens.weight
    assert len(list(actual.parameters())) < len(list(actual.named_parameters(remove_duplicate=False)))
    assert all(torch.equal(value, actual.state_dict()[key]) for key, value in native.state_dict().items())
    inputs = dict(input_ids=torch.tensor([[3, 4, 5, 0]]), attention_mask=torch.tensor([[1, 1, 1, 0]]))
    expected = native(**inputs).last_hidden_state
    output = actual(**inputs).last_hidden_state
    assert torch.equal(expected, output)
    expected.square().sum().backward(); output.square().sum().backward()
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(native.parameters(), actual.parameters(), strict=True))
    assert not torch.cuda.is_initialized()


def native_output(model):
    import torch
    result = model(input_ids=torch.tensor([[3, 4, 5, 0]]), attention_mask=torch.tensor([[1, 1, 1, 0]]))
    return result.last_hidden_state if hasattr(result, 'last_hidden_state') else result.logits


@pytest.mark.parametrize('name', ['flux', 'qwen', 'wan22', 'hunyuan', 'hunyuan_t5'])
def test_legacy_native_effective_config_forward_gradient_raw_restore_and_safe_reload(name, tmp_path):
    import hashlib
    import torch
    from distillschedule.training.flow_image import architecture, load_native, shell
    cls, raw = legacy_case(name)
    effective = cls.config_class.from_dict(deepcopy(raw))
    native = cls(deepcopy(effective)).float().eval()
    native.save_pretrained(tmp_path, safe_serialization=True)
    # Save the original source intent, not native Config.to_dict() normalization.
    document = json.dumps(raw).encode()
    (tmp_path/'config.json').write_bytes(document)
    loaded, retained = load_native(cls, tmp_path, 100_000)
    assert retained == raw and (tmp_path/'config.json').read_bytes() == document
    assert hashlib.sha256(document).hexdigest() == hashlib.sha256((tmp_path/'config.json').read_bytes()).hexdigest()
    assert architecture(cls, retained).to_dict() == effective.to_dict()
    if name == 'hunyuan_t5':
        assert loaded.config.scale_decoder_outputs is False
        assert shell(cls, retained).config.scale_decoder_outputs is False
        assert architecture(cls, retained).scale_decoder_outputs is False
    expected, actual = native_output(native), native_output(loaded)
    assert torch.equal(expected, actual)
    expected.square().sum().backward(); actual.square().sum().backward()
    assert all((a.grad is None and b.grad is None) or torch.equal(a.grad, b.grad)
               for a, b in zip(native.parameters(), loaded.parameters(), strict=True))
    assert any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in loaded.parameters())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('name', ['flux', 'qwen', 'hunyuan'])
def test_legacy_rope_and_flat_fields_match_corresponding_modern_native_configuration(name):
    import torch
    from distillschedule.training.flow_image import architecture
    cls, raw = legacy_case(name)
    parsed = architecture(cls, raw)
    modern = deepcopy(raw)
    if name == 'qwen':
        # Only Qwen config normalization is used in this comparison fixture;
        # production and T5 reconstruction always retain raw source documents.
        text = parsed.text_config.to_dict()
        modern = {key: value for key, value in raw.items() if key not in text and key not in ('rope_theta', 'rope_scaling')}
        modern.update(model_type='qwen2_5_vl', text_config=text, tie_word_embeddings=False)
    else:
        modern.pop('rope_scaling'); modern.pop('rope_theta')
        modern['rope_parameters'] = deepcopy(parsed.rope_parameters)
    effective = architecture(cls, modern)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101); old_model = cls(parsed).float().eval()
        torch.manual_seed(101); new_model = cls(effective).float().eval()
    a, b = native_output(old_model), native_output(new_model)
    assert torch.equal(a, b)
    a.square().sum().backward(); b.square().sum().backward()
    assert all((x.grad is None and y.grad is None) or torch.equal(x.grad, y.grad)
               for x, y in zip(old_model.parameters(), new_model.parameters(), strict=True))


@pytest.mark.parametrize('name', ['flux', 'qwen', 'wan22', 'hunyuan', 'hunyuan_t5'])
def test_unknown_native_legacy_fields_refuse_before_any_payload(name, tmp_path, monkeypatch):
    from safetensors import torch as safe_torch
    from distillschedule.training.flow_image import load_native
    cls, raw = legacy_case(name)
    raw['hidden_szie'] = 16
    (tmp_path/'config.json').write_text(json.dumps(raw))
    def forbidden(*args, **kwargs):
        raise AssertionError('payload loaded before config admission')
    monkeypatch.setattr(safe_torch, 'load_file', forbidden)
    with pytest.raises(ValueError, match='unknown native text config fields: hidden_szie'):
        load_native(cls, tmp_path, 100_000)


@pytest.mark.parametrize('name,key,value', [
    ('wan22', 'output_past', False), ('wan22', 'scalable_attention', False),
    ('wan22', 'tokenizer_class', 'UnknownTokenizer'), ('hunyuan_t5', 'gradient_checkpointing', True),
    ('hunyuan_t5', 'gradient_checkpointing', 0), ('hunyuan_t5', 'decoder_start_token_id', 1),
    ('hunyuan_t5', 'tokenizer_class', 'T5Tokenizer'), ('hunyuan', 'tie_word_embeddings', True),
    ('hunyuan', 'vision_token_id', '29'),
])
def test_unaudited_legacy_metadata_values_are_not_silently_ignored(name, key, value):
    from distillschedule.training.flow_image import architecture
    cls, raw = legacy_case(name); raw[key] = value
    with pytest.raises(ValueError, match='unsupported|invalid'):
        architecture(cls, raw)


@pytest.mark.parametrize('kind', ['rope', 'theta', 'type', 'unknown-rope', 'dtype', 'width', 'nested-rope', 'vision', 'nested-unknown'])
def test_conflicting_or_unknown_alias_forms_refuse_before_payload(kind, tmp_path, monkeypatch):
    from safetensors import torch as safe_torch
    from distillschedule.training.flow_image import architecture, load_native
    cls, raw = legacy_case('qwen' if kind in ('width', 'nested-rope', 'vision', 'nested-unknown') else 'flux')
    if kind == 'rope': raw['rope_scaling'] = dict(rope_type='linear', factor=2.); raw['rope_parameters'] = dict(rope_type='default')
    elif kind == 'theta': raw['rope_parameters'] = dict(rope_type='default', rope_theta=123.)
    elif kind == 'type': raw['rope_scaling'] = dict(type='linear', rope_type='default')
    elif kind == 'unknown-rope': raw['rope_scaling'] = dict(rope_type='default', rope_thtea=123.)
    elif kind == 'dtype': raw.update(dtype='float32', torch_dtype='bfloat16')
    elif kind == 'vision': raw['vision_config']['in_chans'] = 4
    else:
        raw['text_config'] = architecture(cls, raw).text_config.to_dict()
        if kind == 'width': raw['text_config']['hidden_size'] = 32
        elif kind == 'nested-rope': raw['text_config']['rope_parameters']['rope_theta'] = 123.
        else: raw['text_config']['ignored_typo'] = True
    (tmp_path/'config.json').write_text(json.dumps(raw))
    def forbidden(*args, **kwargs): raise AssertionError('payload loaded before alias admission')
    monkeypatch.setattr(safe_torch, 'load_file', forbidden)
    with pytest.raises(ValueError, match='conflicting|unknown'):
        load_native(cls, tmp_path, 100_000)


def test_flat_nested_qwen_effective_window_and_family_width_admission():
    from distillschedule.training import qwen_image
    from distillschedule.training.flow_image import architecture
    cls, flat = legacy_case('qwen')
    nested = deepcopy(flat); nested['text_config'] = architecture(cls, flat).text_config.to_dict()
    nested['sliding_window'] = 32768  # native normalization makes this inert when disabled
    nested['transformers_version'] = '4.53.1'  # parent metadata need not occur in child
    assert architecture(cls, nested).text_config.sliding_window is None
    media = dict(text_encoder=flat, text_max_length=64, vae=dict(z_dim=4, latents_mean=[0.]*4, latents_std=[1.]*4))
    transformer = dict(_class_name='QwenImageTransformer2DModel', patch_size=2, in_channels=16,
        out_channels=4, joint_attention_dim=16, guidance_embeds=False, zero_cond_t=False,
        use_additional_t_cond=False, use_layer3d_rope=False)
    qwen_image.validate_architecture(transformer, media)
    transformer['joint_attention_dim'] = 32
    with pytest.raises(ValueError, match='matched VAE/VL widths'):
        qwen_image.validate_architecture(transformer, media)


def save_t5(tmp_path, *, sharded=False):
    from transformers import T5Config, T5EncoderModel
    model = T5EncoderModel(T5Config(vocab_size=32, d_model=16, d_ff=32, d_kv=8,
        num_layers=1, num_heads=2, dropout_rate=0.)).float().eval()
    model.save_pretrained(tmp_path, safe_serialization=True, max_shard_size='1KB' if sharded else '5GB')
    return model


@pytest.mark.parametrize('kind', ['only-other-alias', 'both-equal', 'contradictory', 'missing-group',
    'missing-independent', 'shape', 'integer', 'fp16', 'nonfinite', 'unknown'])
def test_alias_completion_requires_identical_parameter_slots_and_strict_tensors(tmp_path, kind):
    import torch
    from safetensors.torch import load_file, save_file
    from distillschedule.training.flow_image import load_native
    native = save_t5(tmp_path); path = next(tmp_path.glob('*.safetensors')); state = load_file(path)
    value = state['shared.weight']
    if kind == 'only-other-alias': state['encoder.embed_tokens.weight'] = state.pop('shared.weight')
    elif kind == 'both-equal': state['encoder.embed_tokens.weight'] = value.clone()
    elif kind == 'contradictory': state['encoder.embed_tokens.weight'] = value + 1
    elif kind == 'missing-group': del state['shared.weight']
    elif kind == 'missing-independent': del state[next(key for key in state if key != 'shared.weight')]
    elif kind == 'shape': state['encoder.embed_tokens.weight'] = torch.zeros(1)
    elif kind == 'integer': state['encoder.embed_tokens.weight'] = value.to(torch.int64)
    elif kind == 'fp16': state['encoder.embed_tokens.weight'] = value.half()
    elif kind == 'nonfinite': state['encoder.embed_tokens.weight'] = value.clone().fill_(float('nan'))
    else: state['unknown.weight'] = value.clone()
    save_file(state, path)
    if kind in ('only-other-alias', 'both-equal'):
        loaded, _ = load_native(type(native), tmp_path, 100_000)
        assert loaded.shared.weight is loaded.encoder.embed_tokens.weight
        assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in native.state_dict().items())
    else:
        with pytest.raises(ValueError, match='incomplete|contradictory|shape/dtype|nonfinite|unexpected'):
            load_native(type(native), tmp_path, 100_000)


@pytest.mark.parametrize('classname', ['T5EncoderModel', 'UMT5EncoderModel'])
def test_tied_safe_storage_keeps_explicit_bf16_policy_not_runtime_precision(tmp_path, classname):
    import torch
    import transformers
    from distillschedule.training.flow_image import load_native
    cls = getattr(transformers, classname)
    native = cls(cls.config_class(vocab_size=32, d_model=16, d_ff=32, d_kv=8,
        num_layers=1, num_heads=2, dropout_rate=0.)).bfloat16().eval()
    native.save_pretrained(tmp_path, safe_serialization=True)
    with pytest.raises(ValueError, match='dtype|FP32'): load_native(cls, tmp_path, 100_000)
    loaded, _ = load_native(cls, tmp_path, 100_000, dtype_policy='bf16_or_fp32_to_fp32')
    assert loaded.shared.weight is loaded.encoder.embed_tokens.weight
    assert all(value.dtype == torch.float32 for value in loaded.parameters())
    assert all(torch.equal(value.float(), loaded.state_dict()[key]) for key, value in native.state_dict().items())


@pytest.mark.parametrize('kind', ['wrong-shard', 'missing-file', 'unindexed-alias', 'indexed-missing-alias', 'duplicate-json-key'])
def test_alias_shard_index_remains_exact_and_header_failures_precede_payload(tmp_path, kind, monkeypatch):
    from safetensors import torch as safe_torch
    from distillschedule.training.flow_image import load_native
    native = save_t5(tmp_path, sharded=True)
    index_path = next(tmp_path.glob('*.safetensors.index.json')); index = json.loads(index_path.read_text())
    weights = sorted(tmp_path.glob('*.safetensors')); assert len(weights) > 1
    key, filename = next(iter(index['weight_map'].items()))
    if kind == 'wrong-shard': index['weight_map'][key] = next(path.name for path in weights if path.name != filename)
    elif kind == 'missing-file': weights[0].unlink()
    elif kind == 'indexed-missing-alias': index['weight_map']['encoder.embed_tokens.weight'] = filename
    elif kind == 'unindexed-alias':
        path = tmp_path/index['weight_map']['shared.weight']; state = safe_torch.load_file(path)
        state['encoder.embed_tokens.weight'] = state['shared.weight'].clone(); safe_torch.save_file(state, path)
    index_path.write_text(json.dumps(index))
    if kind == 'duplicate-json-key':
        index_path.write_text('{"weight_map":{},"weight_map":{}}')
    def forbidden(*args, **kwargs): raise AssertionError('payload loaded before complete header/index admission')
    monkeypatch.setattr(safe_torch, 'load_file', forbidden)
    with pytest.raises(ValueError, match='shard|index|incomplete|duplicate JSON'):
        load_native(type(native), tmp_path, 100_000)


def test_missing_independent_same_shape_equal_weight_is_not_filled(tmp_path):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from safetensors.torch import load_file, save_file
    from distillschedule.training.flow_image import load_native
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=False))
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    with torch.no_grad(): model.lm_head.weight.copy_(model.model.embed_tokens.weight)
    model.save_pretrained(tmp_path, safe_serialization=True)
    path = next(tmp_path.glob('*.safetensors')); payload = load_file(path)
    del payload['lm_head.weight']; save_file(payload, path)
    with pytest.raises(ValueError, match='incomplete native weights'):
        load_native(type(model), tmp_path, 100_000)


def test_duplicate_native_renamed_keys_refuse_before_payload(tmp_path, monkeypatch):
    from safetensors import torch as safe_torch
    from distillschedule.training.flow_image import architecture, load_native
    cls, raw = legacy_case('qwen')
    native = cls(architecture(cls, raw)).float().eval()
    native.save_pretrained(tmp_path, safe_serialization=True)
    path = next(tmp_path.glob('*.safetensors')); payload = safe_torch.load_file(path)
    # Actual TF5 save uses old model.* names, while the actual constructor has
    # model.language_model.* slots. Both spelling the same key is a duplicate.
    payload['model.language_model.embed_tokens.weight'] = payload['model.embed_tokens.weight'].clone()
    safe_torch.save_file(payload, path)
    def forbidden(*args, **kwargs): raise AssertionError('duplicate key reached payload loading')
    monkeypatch.setattr(safe_torch, 'load_file', forbidden)
    with pytest.raises(ValueError, match='duplicate'):
        load_native(cls, tmp_path, 100_000)


@pytest.mark.parametrize('kind', ['shape', 'integer', 'fp16'])
def test_malformed_alias_headers_refuse_before_payload(tmp_path, kind, monkeypatch):
    import torch
    from safetensors import torch as safe_torch
    from distillschedule.training.flow_image import load_native
    native = save_t5(tmp_path); path = next(tmp_path.glob('*.safetensors')); payload = safe_torch.load_file(path)
    payload['encoder.embed_tokens.weight'] = (torch.zeros(1) if kind == 'shape' else
        payload['shared.weight'].to(torch.int64 if kind == 'integer' else torch.float16))
    safe_torch.save_file(payload, path)
    def forbidden(*args, **kwargs): raise AssertionError('malformed alias header reached payload loading')
    monkeypatch.setattr(safe_torch, 'load_file', forbidden)
    with pytest.raises(ValueError, match='shape/dtype'):
        load_native(type(native), tmp_path, 100_000)


@pytest.mark.parametrize('contradictory', [False, True])
def test_supplied_aliases_in_different_indexed_shards_must_agree(tmp_path, contradictory):
    import torch
    from safetensors.torch import load_file, save_file
    from distillschedule.training.flow_image import load_native
    native = save_t5(tmp_path, sharded=True)
    index_path = next(tmp_path.glob('*.safetensors.index.json')); index = json.loads(index_path.read_text())
    source = tmp_path/index['weight_map']['shared.weight']; value = load_file(source)['shared.weight']
    target = next(path for path in tmp_path.glob('*.safetensors') if path != source)
    payload = load_file(target); payload['encoder.embed_tokens.weight'] = value + 1 if contradictory else value.clone()
    save_file(payload, target)
    index['weight_map']['encoder.embed_tokens.weight'] = target.name; index_path.write_text(json.dumps(index))
    if contradictory:
        with pytest.raises(ValueError, match='contradictory'):
            load_native(type(native), tmp_path, 100_000)
    else:
        loaded, _ = load_native(type(native), tmp_path, 100_000)
        assert loaded.shared.weight is loaded.encoder.embed_tokens.weight
        assert torch.equal(loaded.shared.weight, native.shared.weight)


def test_text_config_and_alias_source_semantics_are_runtime_bound():
    import inspect
    from pathlib import Path
    from distillschedule.training import qwen_image, flow_text_config
    from distillschedule.training.flow_image import runtime_identity
    from distillschedule.workloads.best_known.registry import file_sha256
    runtime = runtime_identity(qwen_image)
    config_class = qwen_image.classes()[2].config_class
    for cls in (config_class, *config_class.sub_configs.values()):
        assert runtime['source_files'][cls.__module__+'.'+cls.__name__] == file_sha256(Path(inspect.getfile(cls)))
    assert runtime['source_files'][flow_text_config.__name__] == file_sha256(Path(flow_text_config.__file__))
    assert all(module in runtime['source_files'] for module in (
        'transformers.configuration_utils', 'transformers.modeling_rope_utils', 'transformers.modeling_utils'))
