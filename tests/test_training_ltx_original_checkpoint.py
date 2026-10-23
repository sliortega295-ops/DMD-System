"""Original LTX safetensors metadata/prefixes use the actual pinned loader stack."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope='module')
def native_checkpoint(tmp_path_factory):
    source = os.environ.get('LTX_SOURCE_ROOT')
    if not source:
        pytest.skip('set LTX_SOURCE_ROOT for optional original-checkpoint import acceptance')
    from distillschedule.training.ltx_source import original
    from distillschedule.training.plugins import load_callable
    from safetensors.torch import save_file
    api = original(source)
    factory, _ = load_callable(str(Path(__file__).resolve().parents[1] /
        'recipes/ltx25/create_cpu_fixture.py') + ':tiny_architectures')
    architecture, _ = factory()
    config = deepcopy(architecture)
    config.pop('model_type'); config.pop('double_precision_rope')
    config.update(caption_proj_before_connector=True, frequencies_precision='float32',
        dropout=0., attention_bias=True, num_vector_embeds=None, activation_fn='gelu-approximate',
        num_embeds_ada_norm=1000, use_linear_projection=False, only_cross_attention=False,
        cross_attention_norm=True, double_self_attention=False, upcast_attention=False,
        standardization_norm='rms_norm', norm_elementwise_affine=False, qk_norm='rms_norm',
        positional_embedding_type='rope', use_audio_video_cross_attention=True, share_ff=False,
        av_cross_ada_norm=True, norm_eps=1e-6, positional_embedding_theta=10000.,
        positional_embedding_max_pos=[20, 2048, 2048], audio_positional_embedding_max_pos=[20],
        timestep_scale_multiplier=1000, av_ca_timestep_scale_multiplier=1, apply_gated_attention=False)
    metadata = dict(config=json.dumps({'transformer': config}), model_version='2.5',
                    license='LTX-2.x Community License; RANDOM fixture weights, not pretrained')
    configurator = api.source.import_module('ltx_core.model.transformer.model_configurator')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(731)
        native = configurator.LTXModelConfigurator.from_metadata({'config': {'transformer': config}})
        for name, p in native.named_parameters():
            if 'scale_shift_table' in name:
                torch.nn.init.normal_(p, std=.02)
    state = {'model.diffusion_model.' + name: value.detach().clone().contiguous()
             for name, value in native.state_dict().items()}
    root = tmp_path_factory.mktemp('original-ltx')
    path = root / 'original-format.safetensors'; save_file(state, path, metadata=metadata)
    return api, native, path, state, metadata


def test_actual_original_loader_configurator_keyops_and_builder_execute(native_checkpoint, monkeypatch):
    from distillschedule.training.ltx import load_original_transformer
    from distillschedule.training.ltx_media import file_sha256
    api, reference, path, _, _ = native_checkpoint
    builder_api = api.source.import_module('ltx_core.loader.single_gpu_model_builder')
    actual_build = builder_api.SingleGPUModelBuilder.build
    calls = []
    def observed(self, *args, **kwargs):
        calls.append((type(self).__module__, self.model_sd_ops.name, kwargs))
        return actual_build(self, *args, **kwargs)
    monkeypatch.setattr(builder_api.SingleGPUModelBuilder, 'build', observed)
    loaded, architecture, receipt = load_original_transformer(api.source, path, file_sha256(path),
        max_parameters=200_000, dtype_policy='strict_fp32')
    assert type(loaded) is api.model.LTXModel
    assert calls and calls[0][1] == 'LTXV_MODEL_COMFY_PREFIX_MAP'
    assert calls[0][2]['device'] == torch.device('cpu')
    assert receipt['loader'].endswith('SafetensorsModelStateDictLoader')
    assert receipt['configurator'].endswith('LTXModelConfigurator')
    assert receipt['builder'].endswith('SingleGPUModelBuilder')
    assert receipt['loras'] == [] and receipt['strict_selected_key_count'] == len(reference.state_dict())
    assert architecture['num_layers'] == 1 and architecture['attention_head_dim'] == 24
    for name, value in reference.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name
    attention_api = api.source.import_module('ltx_core.model.transformer.attention')
    for child in loaded.modules():
        if isinstance(child, attention_api.Attention):
            assert child.attention_function.label == child.masked_attention_function.label == 'SDPA[MATH]'
    assert torch.cuda.is_initialized() is False


@pytest.mark.parametrize('case', ['empty', 'missing', 'shape', 'nonfinite', 'dtype', 'metadata', 'budget', 'hash'])
def test_incomplete_or_changed_original_checkpoint_refused(native_checkpoint, tmp_path, case):
    from distillschedule.training.ltx import load_original_transformer
    from distillschedule.training.ltx_media import file_sha256
    from safetensors.torch import save_file
    api, _, _, state, metadata = native_checkpoint
    state = {name: value.clone() for name, value in state.items()}; metadata = dict(metadata)
    first = next(iter(state))
    if case == 'empty':
        state = {}
    elif case == 'missing':
        state.pop(first)
    elif case == 'shape':
        state[first] = state[first].reshape(-1)[:1]
    elif case == 'nonfinite':
        state[first].fill_(float('nan'))
    elif case == 'dtype':
        state[first] = state[first].double()
    elif case == 'metadata':
        metadata['config'] = json.dumps({'transformer': {}})
    path = tmp_path / (case + '.safetensors'); save_file(state, path, metadata=metadata)
    with pytest.raises(ValueError):
        load_original_transformer(api.source, path, '0' * 64 if case == 'hash' else file_sha256(path),
            max_parameters=2 if case == 'budget' else 200_000, dtype_policy='strict_fp32')


def test_original_native_precision_cast_is_explicit_and_returned_meta_is_refused(native_checkpoint, tmp_path, monkeypatch):
    from distillschedule.training.ltx import load_original_transformer
    from distillschedule.training.ltx_media import file_sha256
    from safetensors.torch import save_file
    api, _, path, state, metadata = native_checkpoint
    bf16 = {name: value.bfloat16() for name, value in state.items()}
    file = tmp_path / 'native-bf16.safetensors'; save_file(bf16, file, metadata=metadata)
    with pytest.raises(ValueError):
        load_original_transformer(api.source, file, file_sha256(file), max_parameters=200_000, dtype_policy='strict_fp32')
    loaded, _, receipt = load_original_transformer(api.source, file, file_sha256(file),
        max_parameters=200_000, dtype_policy='cast_native_float_to_fp32')
    assert receipt['dtype_policy'] == 'cast_native_float_to_fp32'
    for name, value in loaded.state_dict().items():
        assert value.dtype == torch.float32 and torch.equal(value, bf16['model.diffusion_model.' + name].float())
    builder = api.source.import_module('ltx_core.loader.single_gpu_model_builder').SingleGPUModelBuilder
    monkeypatch.setattr(builder, 'build', lambda self, **kwargs: self.meta_model(self.model_metadata(), self.module_ops))
    with pytest.raises(ValueError, match='meta|uninitialized'):
        load_original_transformer(api.source, path, file_sha256(path), max_parameters=200_000, dtype_policy='strict_fp32')


def test_original_format_roles_reach_common_dmd_and_current_generator_projection(native_checkpoint, tmp_path):
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.ltx import build_inference_model, project_current_generator
    from distillschedule.training.ltx_media import file_sha256
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import construct, build_cycle
    from distillschedule.open_workload.torch_backend import Session
    api, _, checkpoint, _, _ = native_checkpoint
    create, _ = load_callable(str(Path(__file__).resolve().parents[1] /
        'recipes/ltx25/create_cpu_fixture.py') + ':build_fixture')
    files = create(tmp_path / 'fixture', api.source.root, cycles=1)
    bundle = Path(files['assets']); target = bundle / 'original-format.safetensors'
    import shutil
    shutil.copyfile(checkpoint, target)
    manifest_path = bundle / 'assets.json'
    manifest = json.loads(manifest_path.read_text())
    for role in ('student', 'fake', 'teacher'):
        manifest['roles'][role] = dict(format='ltx_original_safetensors', checkpoint=dict(
            path=target.name, sha256=file_sha256(target)), cpu_max_parameters=200_000, dtype_policy='strict_fp32')
    manifest_path.write_text(json.dumps(manifest))
    config = load_training_config(files['config'])
    components, dataset, algorithm, _ = construct(config)
    assert all(row['format'] == 'ORIGINAL_LTX_2_5_SAFETENSORS'
               for row in components.metadata['asset_identity']['roles'].values())
    workload = build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
    session = Session(workload, {'backend': 'torch-callable-resident-v1',
        'share_immutable': False, 'checkpoint_bindings': []}, copy_state=False)
    session.run()
    assert components.state['dmd_counters'].tolist() == [1, 1, 5]
    projection = project_current_generator(components, config.model['parameters'])
    inference = build_inference_model(dict(projection.parameters, device='cpu'))
    for name, module in inference.state.items():
        module.load_state_dict(projection.components.state[name].state_dict(), strict=True)
    torch.manual_seed(919)
    with torch.no_grad():
        expected = components.generate(components.state, dataset.sample(0), {})
        torch.manual_seed(919)
        actual = inference.generate(inference.state, dataset.sample(0), {})
    assert torch.equal(expected['video_latents'], actual['video_latents'])
    assert torch.equal(expected['audio_latents'], actual['audio_latents'])
