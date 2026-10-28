"""Original LTX safetensors metadata/prefixes use the actual pinned loader stack."""
from copy import deepcopy
import json
import os
from pathlib import Path
import weakref

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
    calls = []; builder_storage = {}; builder_parameters = {}
    def observed(self, *args, **kwargs):
        calls.append((type(self).__module__, self.model_sd_ops.name, kwargs))
        model = actual_build(self, *args, **kwargs)
        builder_storage.update({name: value.data_ptr() for name, value in model.state_dict().items()})
        builder_parameters.update({name: id(value) for name, value in model.named_parameters(remove_duplicate=False)})
        return model
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
        assert loaded.state_dict()[name].data_ptr() != builder_storage[name], 'own storage, not safetensors views'
    assert {name: id(value) for name, value in loaded.named_parameters(remove_duplicate=False)} == builder_parameters
    assert receipt['storage_ownership'] == 'INDEPENDENT_TORCH_STORAGE_PRESERVE_PARAMETER_OBJECTS'
    attention_api = api.source.import_module('ltx_core.model.transformer.attention')
    for child in loaded.modules():
        if isinstance(child, attention_api.Attention):
            assert child.attention_function.label == child.masked_attention_function.label == 'SDPA[MATH]'
    assert torch.cuda.is_initialized() is False


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_original_native_builder_reuses_one_validated_payload(native_checkpoint, tmp_path, monkeypatch, dtype):
    from distillschedule.training.ltx import load_original_transformer
    from distillschedule.training.ltx_media import file_sha256
    from safetensors.torch import save_file
    api, reference, _, state, metadata = native_checkpoint
    checkpoint = tmp_path / 'one-payload.safetensors'
    save_file({name: value.to(dtype) for name, value in state.items()}, checkpoint, metadata=metadata)
    digest = file_sha256(checkpoint)
    loader = api.source.import_module('ltx_core.loader.sft_loader').SafetensorsModelStateDictLoader
    builder = api.source.import_module('ltx_core.loader.single_gpu_model_builder').SingleGPUModelBuilder
    original_load, original_build = loader.load, builder.build
    loads = []; registries = []
    def load(self, *args, **kwargs):
        loads.append((args, kwargs))
        return original_load(self, *args, **kwargs)
    def build(self, *args, **kwargs):
        registries.append(self.registry)
        return original_build(self, *args, **kwargs)
    monkeypatch.setattr(loader, 'load', load)
    monkeypatch.setattr(builder, 'build', build)
    loaded, _, _ = load_original_transformer(api.source, checkpoint, digest,
        max_parameters=200_000, dtype_policy='strict_fp32' if dtype == torch.float32 else 'cast_native_float_to_fp32')
    assert len(loads) == 1, 'validation and actual native build must share a single loader payload'
    assert len(registries) == 1 and registries[0].get([str(checkpoint)],
        api.source.import_module('ltx_core.model.transformer.model_configurator').LTXV_MODEL_COMFY_RENAMING_MAP) is None
    for name, expected in reference.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], expected.to(dtype).float())
    with torch.no_grad():
        next(loaded.parameters()).add_(1)
    assert file_sha256(checkpoint) == digest, 'training must not mutate checkpoint-backed storage'
    assert not torch.cuda.is_initialized()


def test_original_validation_does_not_retain_a_model_sized_cast_copy(native_checkpoint, tmp_path, monkeypatch):
    from distillschedule.training.ltx import load_original_transformer
    from distillschedule.training.ltx_media import file_sha256
    from safetensors.torch import save_file
    api, _, _, state, metadata = native_checkpoint
    checkpoint = tmp_path / 'cast-lifetime.safetensors'
    save_file({name: value.bfloat16() for name, value in state.items()}, checkpoint, metadata=metadata)
    builder = api.source.import_module('ltx_core.loader.single_gpu_model_builder').SingleGPUModelBuilder
    original_float, original_build = torch.Tensor.float, builder.build
    cast_refs = []; build_calls = []
    def cast(tensor, *args, **kwargs):
        value = original_float(tensor, *args, **kwargs)
        if tensor.dtype == torch.bfloat16:
            assert not any(ref() is not None for ref in cast_refs), 'only one validation cast may be live'
            cast_refs.append(weakref.ref(value))
        return value
    def build(self, *args, **kwargs):
        build_calls.append(1)
        assert not any(ref() is not None for ref in cast_refs), 'validation casts must be released before native build'
        return original_build(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'float', cast)
    monkeypatch.setattr(builder, 'build', build)
    load_original_transformer(api.source, checkpoint, file_sha256(checkpoint),
        max_parameters=200_000, dtype_policy='cast_native_float_to_fp32')
    assert build_calls == [1] and cast_refs
    assert all(ref() is None for ref in cast_refs)


@pytest.mark.parametrize('failure', ['weight', 'raise'])
def test_original_builder_failure_releases_cached_payload(native_checkpoint, monkeypatch, failure):
    from distillschedule.training.ltx import load_original_transformer
    from distillschedule.training.ltx_media import file_sha256
    api, _, checkpoint, _, _ = native_checkpoint
    builder = api.source.import_module('ltx_core.loader.single_gpu_model_builder').SingleGPUModelBuilder
    operations = api.source.import_module('ltx_core.model.transformer.model_configurator').LTXV_MODEL_COMFY_RENAMING_MAP
    original_build = builder.build
    registries = []
    def broken(self, *args, **kwargs):
        registries.append(self.registry)
        if failure == 'raise':
            raise RuntimeError('native build fixture failed')
        model = original_build(self, *args, **kwargs)
        with torch.no_grad():
            next(model.parameters()).add_(1)
        return model
    monkeypatch.setattr(builder, 'build', broken)
    with pytest.raises((ValueError, RuntimeError), match='exact selected weight|native build fixture failed'):
        load_original_transformer(api.source, checkpoint, file_sha256(checkpoint),
            max_parameters=200_000, dtype_policy='strict_fp32')
    assert len(registries) == 1
    assert registries[0].get([str(checkpoint)], operations) is None


def test_original_architecture_rejected_before_native_materialization_if_not_reconstructable(
        native_checkpoint, tmp_path, monkeypatch):
    from distillschedule.training.ltx import load_original_transformer
    from distillschedule.training.ltx_media import file_sha256
    from safetensors.torch import save_file
    api, _, _, _, metadata = native_checkpoint
    config = json.loads(metadata['config'])['transformer']; config['num_layers'] = 17
    configurator = api.source.import_module('ltx_core.model.transformer.model_configurator')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(732)
        native = configurator.LTXModelConfigurator.from_metadata({'config': {'transformer': config}})
        for name, parameter in native.named_parameters():
            if 'scale_shift_table' in name:
                torch.nn.init.normal_(parameter, std=.02)
    count = sum(parameter.numel() for parameter in native.parameters())
    assert count < 3_000_000, 'the original-format rejection fixture must remain CPU-small'
    state = {'model.diffusion_model.' + name: value.detach().contiguous()
             for name, value in native.state_dict().items()}
    path = tmp_path / 'seventeen-narrow-layers.safetensors'
    save_file(state, path, metadata=dict(metadata, config=json.dumps({'transformer': config})))
    builder = api.source.import_module('ltx_core.loader.single_gpu_model_builder').SingleGPUModelBuilder
    actual_meta, actual_build = builder.meta_model, builder.build
    calls = []
    def meta(self, *args, **kwargs):
        calls.append('meta_model'); return actual_meta(self, *args, **kwargs)
    def build(self, *args, **kwargs):
        calls.append('build'); return actual_build(self, *args, **kwargs)
    monkeypatch.setattr(builder, 'meta_model', meta)
    monkeypatch.setattr(builder, 'build', build)
    with pytest.raises(ValueError, match='bounded explicit native architecture'):
        load_original_transformer(api.source, path, file_sha256(path),
            max_parameters=count + 1, dtype_policy='strict_fp32')
    assert not calls, 'reject before even the native meta-model sizing/materialization stage'


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
