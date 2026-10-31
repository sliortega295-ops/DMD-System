"""Actual tiny original G/PEFT/VAE -> import CLI -> independent generation.

Fixture values mark an OLD Generator without executing training. No mocked
model, optimizer, sampler, export loader or decoder may stand in for upstream.
"""
from copy import deepcopy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch


def stack_available():
    try:
        return all(importlib.util.find_spec(n) is not None for n in ('peft', 'diffusers', 'omegaconf'))
    except importlib.metadata.PackageNotFoundError:
        return False


AVAILABLE = stack_available()
if os.environ.get('R217_REQUIRE_UPSTREAM') == '1' and not AVAILABLE:
    raise RuntimeError('R217 actual upstream requires the existing PEFT/Diffusers/OmegaConf stack; no substitute')
pytestmark = pytest.mark.skipif(not AVAILABLE, reason='actual existing upstream stack unavailable; no mocked substitute')


def upstream_fixture(root, *, missing_pyramid=False, raw_wan_keys=False, rope_max_seq_len=None, num_layers=1):
    from peft import LoraConfig
    from safetensors.torch import save_file
    from test_training_wan_native_outputs import codec_request, pinned
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.training import wan_generator_import as api, wan_native_inference as native
    from runtime.critic_pool_helios import TRANSFORMER_ADDITIONAL_KWARGS, _all_linear_targets, _materialize_missing_pyramid_patches
    root.mkdir(parents=True)
    task = OriginalWanTask(video_height=128, video_width=128, video_frames=33, first_step=0,
        lora_rank=1, stage2_inference_steps=(1,1,1))
    cls = load_reference_model_class(backend='torch-sdpa')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(283)
        g = cls(patch_size=(1,2,2), num_attention_heads=2, attention_head_dim=16,
            in_channels=16, out_channels=16, text_dim=4096, freq_dim=16,
            ffn_dim=64, num_layers=num_layers, rope_dim=(4,6,6), is_use_gan=False,
            is_use_gan_final=False, is_use_gan_hooks=False, **TRANSFORMER_ADDITIONAL_KWARGS)
        with torch.no_grad(): g.proj_out.weight.normal_(0, .02)
        # Independent OLD producer: save raw FP32 base before any loader or
        # precision conversion. Never mirror the importer's dtype ordering.
        g.save_config(root)
        if rope_max_seq_len is not None:
            config_path = root/'config.json'
            raw = json.loads(config_path.read_text())
            raw['rope_max_seq_len'] = rope_max_seq_len
            config_path.write_text(json.dumps(raw)+'\n')
        base = {name: t for name, t in g.state_dict().items()
                if not missing_pyramid or not name.startswith(('patch_short.', 'patch_mid.', 'patch_long.'))}
        if raw_wan_keys:
            base['scale_shift_table'] = base.pop('norm_out.scale_shift_table')
        base_path = root/'base.safetensors'
        save_file(base, base_path, metadata=dict(evidence_kind=api.FIXTURE))
        g = cls.from_pretrained(str(root), low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
            device_map='cpu', max_workers=1, transformer_additional_kwargs=deepcopy(TRANSFORMER_ADDITIONAL_KWARGS))
        if missing_pyramid:
            # This asserts the original loader really took its meta path; a
            # permissive fallback/random constructor cannot become the golden.
            assert all(t.device.type == 'meta' and t.dtype == torch.float32
                for name in ('patch_short','patch_mid','patch_long') for t in getattr(g,name).parameters())
        materialized = _materialize_missing_pyramid_patches(g, api.PYRAMID_SEED)
        assert materialized == (['patch_short','patch_mid','patch_long'] if missing_pyramid else [])
        targets = sorted(set(_all_linear_targets(g)+['patch_short','patch_mid','patch_long']))
        g.requires_grad_(False)
        g.add_adapter(LoraConfig(r=1, lora_alpha=1., lora_dropout=0., init_lora_weights='gaussian',
            target_modules=targets, exclude_modules=['down','up']))
        for name, t in g.named_parameters():
            if 'lora_' in name:
                t.data = t.data.float()
                if 'lora_B' in name:
                    with torch.no_grad(): t.fill_(.031)
    trained = {name: t.detach().contiguous() for name, t in g.named_parameters() if t.requires_grad}
    payload = root/'final_generator_trainables.safetensors'
    save_file(trained, payload, metadata=dict(component='generator', purpose='cross-placement-final-state-parity'))
    output_request = codec_request(root/'codec')
    codec, _ = native._load_codec(output_request['codec'], output_request['budget'], api.FIXTURE)
    if num_layers >= 3:
        from peft.tuners.tuners_utils import BaseTunerLayer
        installed = sorted(name for name, layer in g.named_modules()
            if isinstance(layer, BaseTunerLayer) and 'default' in layer.lora_A and 'default' in layer.lora_B)
        assert installed == targets
        assert len(g.peft_config['default'].target_modules) < len(targets)
    projection = native.project_current_generator(NS(g=NS(module=g), task=task), codec, output_request, world_size=1)
    receipt = dict(status='COMPLETED_ORIGINAL_WAN_WORK', rank=0, finite=True,
        identity=dict(source_head='a'*40, source_fingerprint='b'*64, assets=dict(model_files={
            'config.json': pinned(root/'config.json')['sha256'], 'base.safetensors': pinned(base_path)['sha256']})),
        task=task.to_dict(), initial_versions=dict(generator=0, critic=0, ema=0),
        final_versions=dict(generator=2, critic=10, ema=10),
        cycles=[dict(cycle=i, versions=dict(generator=i, critic=5*i, ema=5*i)) for i in (1,2)],
        numerical_comparison='REQUIRES_MATCHED_REFERENCE', quality_validation='NOT_RUN',
        measurement_scope='ACTUAL_UPSTREAM_TINY_CONSTRUCTED_VALUES_NOT_REAL_TRAINING',
        state_evidence=dict(generator=dict(pinned(payload), writer_rank=0, status='PASS', tensor_count=len(trained))))
    receipt_path = root/'rank_00.json'; receipt_path.write_text(json.dumps(receipt)+'\n')
    from distillschedule.workloads.best_known.registry import HELIOS_SHA
    request = dict(schema_version=api.SCHEMA, kind=api.KIND, source=api.source_identity(), parameters=projection.parameters,
        base=dict(config=pinned(root/'config.json'), weights=[pinned(base_path)], indexes=[],
            key_policy=(dict(kind=api.WAN_KEYS, source_sha256=HELIOS_SHA) if raw_wan_keys else dict(kind=api.NATIVE_KEYS)),
            pyramid=(dict(kind='SOURCE_SEEDED_WAN_MISSING_PATCHES', seed=api.PYRAMID_SEED)
                if missing_pyramid else dict(kind='COMPLETE_SAFE_BASE'))),
        trainables=dict(payload=pinned(payload), parameter_names=list(trained)), codec=output_request['codec'],
        origin=dict(contract=api.ORIGIN_CONTRACT, receipt=pinned(receipt_path),
                    source_head='a'*40, source_fingerprint='b'*64))
    path = root/'initialization.json'; path.write_text(json.dumps(request)+'\n')
    return path, g, codec, request


@pytest.mark.parametrize('missing_pyramid,raw_wan_keys', [(False,False), (True,False), (True,True)])
def test_actual_upstream_import_cli_export_and_new_process_new_prompt(tmp_path, monkeypatch, capsys, missing_pyramid, raw_wan_keys):
    import weakref
    from safetensors.torch import save_file
    from test_training_wan_native_outputs import pinned
    from distillschedule.cli import main
    from distillschedule.data import import_preencoded
    from distillschedule.training.checkpoint import rng_state
    from distillschedule.training import wan_generator_import as api
    from test_training_end_to_end import equal
    root = tmp_path; old = root/'OLD-source'
    path, g, codec, request = upstream_fixture(old, missing_pyramid=missing_pyramid, raw_wan_keys=raw_wan_keys)
    current = deepcopy(g.state_dict()); rng = rng_state()
    # Admission owns a full copy of all trainables. Once copied into the
    # live model, it must not survive into the export serialization peak.
    staged_refs = []
    stage = api.stage_trainables
    def tracked_stage(*args, **kwargs):
        values = stage(*args, **kwargs)
        staged_refs.extend(weakref.ref(tensor) for tensor in values.values())
        return values
    monkeypatch.setattr(api, 'stage_trainables', tracked_stage)
    from distillschedule.training import inference
    publish = inference.publish_export
    def tracked_publish(*args, **kwargs):
        assert staged_refs and all(ref() is None for ref in staged_refs)
        return publish(*args, **kwargs)
    monkeypatch.setattr(inference, 'publish_export', tracked_publish)
    def no_training(*a, **kw): raise AssertionError('initialization attempted optimizer construction')
    monkeypatch.setattr(torch.optim.Adam, '__init__', no_training)
    monkeypatch.setattr(torch.optim.AdamW, '__init__', no_training)
    portable = root/'portable'
    capsys.readouterr()  # original from_pretrained producer diagnostics are not CLI JSON
    assert main(['import-weights', str(path), '--output', str(portable)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['training_cycles_executed'] == 0 and result['checkpoint_cycle'] == 0
    assert result['identity']['weights_initialization']['origin_cycle'] == 2
    assert result['identity']['weights_initialization']['origin_source_head'] == 'a'*40
    assert equal(rng, rng_state()) and all(torch.equal(current[n], t) for n, t in g.state_dict().items())
    weights = torch.load(portable/'weights.pt', map_location='cpu', weights_only=True)
    assert set(weights) == {'generator', 'codec'}
    assert all(torch.equal(weights['generator']['model.'+n], t) for n, t in current.items())
    assert all(torch.equal(weights['codec'][n], t) for n, t in codec.state_dict().items())
    new = root/'NEW-prompt'; new.mkdir()
    torch.save(dict(prompt_embeds=torch.randn((1,2,4096), generator=torch.Generator().manual_seed(981)).bfloat16()), new/'prompt.pt')
    (new/'inputs.json').write_text(json.dumps(dict(schema_version='distillschedule.data.preencoded.v1',
        records=[dict(id='NEW-initialized-W1-prompt', features=dict(path='prompt.pt', sha256=pinned(new/'prompt.pt')['sha256']))],
        metadata=dict(evidence_kind=request['parameters']['evidence_kind']))))
    prepared = import_preencoded(new/'inputs.json', new/'prepared')
    shutil.rmtree(old)
    script = '''import os,sys,torch
if os.environ.get('R217_ROOT'):
 from overlay import activate
 activate()
from pathlib import Path
root=Path(sys.argv[1]);old=str(root/'OLD-source')
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):assert old not in str(args[0]),'reopened OLD source'
sys.addaudithook(audit)
def refuse(*a,**kw):raise AssertionError('inference constructed optimizer')
torch.optim.Adam.__init__=torch.optim.AdamW.__init__=refuse
torch.set_num_threads(1)
from distillschedule.cli import main
assert main(['generate',str(root/'portable.inference.json'),'--weights',str(root/'portable'),
 '--prepared',str(root/'NEW-prompt/prepared/manifest.json'),'--output-kind','video','--output',str(root/'video')])==0
value=torch.load(root/'video/generation.pt',map_location='cpu',weights_only=True)
assert value['decoded']['video'].shape==(33,128,128,3)
assert not torch.cuda.is_initialized()
print('ACTUAL_TINY_UPSTREAM_INITIALIZATION_EXPORT_NEW_PROMPT_VAE_PASS_NOT_GPU_OR_QUALITY')
'''
    env = dict(os.environ, WORLD_SIZE='1', RANK='0', LOCAL_RANK='0', CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([sys.executable, '-B', '-c', script, str(root)], env=env,
        text=True, capture_output=True, timeout=300)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    assert 'ACTUAL_TINY_UPSTREAM_INITIALIZATION_EXPORT_NEW_PROMPT_VAE_PASS' in result.stdout
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('budget_name', ['max_export_bytes_per_rank', 'max_export_bytes_all_ranks'])
def test_actual_upstream_low_export_budget_refuses_before_publication(tmp_path, capsys, budget_name):
    from distillschedule.cli import main
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    path, g, _, request = upstream_fixture(tmp_path/'OLD-source')
    current = deepcopy(g.state_dict()); rng = rng_state()
    request['parameters']['budget'][budget_name] = 1
    path.write_text(json.dumps(request)+'\n')
    output = tmp_path/'portable'
    assert main(['import-weights', str(path), '--output', str(output)]) == 2
    assert 'export and transient host capacity' in capsys.readouterr().err
    assert not output.exists() and not output.with_name(output.name+'.inference.json').exists()
    assert equal(rng, rng_state()) and all(torch.equal(current[n], t) for n, t in g.state_dict().items())


def test_actual_from_config_retains_rope_metadata_without_changing_constructor_state_or_dense_output(tmp_path):
    import inspect
    from runtime.critic_pool_helios import TRANSFORMER_ADDITIONAL_KWARGS
    from distillschedule.training.helios_inference import _json as plain
    path, g, _, request = upstream_fixture(tmp_path/'control')
    raw = json.loads((path.parent/'config.json').read_text())
    cls = type(g)
    signature = set(inspect.signature(cls.__init__).parameters)-{'self'}
    assert 'rope_max_seq_len' not in signature and 'rope_max_seq_len' not in raw
    assert set(request['parameters']['architecture']) == signature
    models = []
    for extra in ({}, {'rope_max_seq_len': 1}, {'rope_max_seq_len': 4096}):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(283)
            model = cls.from_config(dict(raw, **extra), **deepcopy(TRANSFORMER_ADDITIONAL_KWARGS))
            with torch.no_grad():
                model.proj_out.weight.normal_(0, .02)
        assert plain({n: model.config[n] for n in signature}) == request['parameters']['architecture']
        if extra:
            assert model.config['rope_max_seq_len'] == extra['rope_max_seq_len']
        models.append(model.eval())
    expected = models[0].state_dict()
    for model in models[1:]:
        assert model.state_dict().keys() == expected.keys()
        assert all(torch.equal(t, model.state_dict()[n]) for n, t in expected.items())
        assert all(torch.equal(t, dict(model.named_buffers())[n]) for n, t in models[0].named_buffers())
    rng = torch.Generator().manual_seed(981)
    inputs = dict(hidden_states=torch.randn((1,16,2,4,4), generator=rng), timestep=torch.tensor([500.]),
        encoder_hidden_states=torch.randn((1,2,4096), generator=rng), return_dict=False)
    outputs, calls = [], []
    for model in models:
        handles = [layer.register_forward_hook(lambda *args: calls.append(True))
            for layer in (model.blocks[0].attn1, model.blocks[0].attn2)]
        try:
            with torch.no_grad():
                outputs.append(model(**inputs)[0])
        finally:
            for handle in handles:
                handle.remove()
    assert len(calls) == 6 and torch.count_nonzero(outputs[0]) > 0
    assert all(torch.equal(outputs[0], value) for value in outputs[1:])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('field', ['rope_max_seq_len', 'unknown_native_config_field'])
def test_actual_upstream_nonconstructor_config_projection_and_public_import_are_exact(tmp_path, capsys, field):
    import inspect
    from test_training_wan_native_outputs import pinned
    from distillschedule.cli import main
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.training import wan_native_inference as native
    path, g, codec, request = upstream_fixture(tmp_path/'OLD-source',
        rope_max_seq_len=1024 if field == 'rope_max_seq_len' else None)
    current = deepcopy(g.state_dict())
    signature = set(inspect.signature(type(g).__init__).parameters)-{'self'}
    assert set(request['parameters']['architecture']) == signature and field not in signature
    if field == 'unknown_native_config_field':
        g.register_to_config(**{field: 1024})
        p = request['parameters']
        output_request = dict(prompt_shape=p['geometry']['prompt_shape'], fps=p['geometry']['fps'],
            budget=p['budget'], seed=p['seed'], evidence_kind=p['evidence_kind'], codec=request['codec'])
        with pytest.raises(ValueError, match='complete original Generator constructor required'):
            native.project_current_generator(NS(g=NS(module=g), task=OriginalWanTask.from_dict(p['task'])),
                codec, output_request, world_size=1)
        config_path = Path(request['base']['config']['path'])
        raw = json.loads(config_path.read_text()); raw[field] = 1024
        config_path.write_text(json.dumps(raw)+'\n')
        request['base']['config'] = pinned(config_path)
        receipt_path = Path(request['origin']['receipt']['path'])
        receipt = json.loads(receipt_path.read_text())
        receipt['identity']['assets']['model_files']['config.json'] = pinned(config_path)['sha256']
        receipt_path.write_text(json.dumps(receipt)+'\n')
        request['origin']['receipt'] = pinned(receipt_path)
        path.write_text(json.dumps(request)+'\n')
    raw_pin = deepcopy(request['base']['config'])
    raw_bytes = Path(raw_pin['path']).read_bytes()
    portable = tmp_path/'portable'
    capsys.readouterr()
    result = main(['import-weights', str(path), '--output', str(portable)])
    captured = capsys.readouterr()
    if field == 'unknown_native_config_field':
        assert result == 2 and 'unknown pinned base constructor field' in captured.err
        assert not portable.exists() and not portable.with_name(portable.name+'.inference.json').exists()
    else:
        assert result == 0 and g.config[field] == 1024
        manifest = json.loads(captured.out)
        provenance = manifest['identity']['weights_initialization']
        assert provenance['base_non_constructor_config'] == {'rope_max_seq_len': 1024}
        assert provenance['base'][0] == dict(name='config.json', bytes=raw_pin['bytes'], sha256=raw_pin['sha256'])
        assert manifest['model']['parameters']['architecture'] == request['parameters']['architecture']
        assert manifest['model']['metadata'] == native._metadata(request['parameters'])
        weights = torch.load(portable/'weights.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(weights['generator']['model.'+n], t) for n, t in current.items())
        assert manifest['training_cycles_executed'] == 0
    assert Path(raw_pin['path']).read_bytes() == raw_bytes and pinned(Path(raw_pin['path'])) == raw_pin
    assert all(torch.equal(current[n], t) for n, t in g.state_dict().items())
    assert not torch.cuda.is_initialized()


def test_actual_upstream_compacted_peft_targets_project_import_and_strict_reload_are_exact(tmp_path, capsys):
    from peft.tuners.tuners_utils import BaseTunerLayer
    from distillschedule.cli import main
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.training import wan_native_inference as native
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    path, g, codec, request = upstream_fixture(tmp_path/'OLD-source', num_layers=3)
    installed = sorted(name for name, layer in g.named_modules()
        if isinstance(layer, BaseTunerLayer) and 'default' in layer.lora_A and 'default' in layer.lora_B)
    # Three complete original blocks cross PEFT's target-compaction threshold;
    # unlike the one-block fixture, the config is no longer the layer inventory.
    assert len(installed) == 39
    assert len(g.peft_config['default'].target_modules) < len(installed)
    assert sorted(g.peft_config['default'].target_modules) != installed
    assert request['parameters']['peft']['target_modules'] == installed
    current, codec_current = deepcopy(g.state_dict()), deepcopy(codec.state_dict())
    owners = {name: id(t) for name, t in g.named_parameters()}
    modes = {name: layer.training for name, layer in g.named_modules()}
    trainability = {name: t.requires_grad for name, t in g.named_parameters()}
    peft_config = deepcopy(g.peft_config['default'].to_dict())
    p = request['parameters']; rng = rng_state()
    output_request = dict(prompt_shape=p['geometry']['prompt_shape'], fps=p['geometry']['fps'],
        budget=p['budget'], seed=p['seed'], evidence_kind=p['evidence_kind'], codec=request['codec'])
    projected = native.project_current_generator(NS(g=NS(module=g), task=OriginalWanTask.from_dict(p['task'])),
        codec, output_request, world_size=1)
    assert projected.parameters == p
    portable = tmp_path/'portable'
    capsys.readouterr()
    assert main(['import-weights', str(path), '--output', str(portable)]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest['training_cycles_executed'] == manifest['checkpoint_cycle'] == 0
    assert manifest['model']['parameters']['peft']['target_modules'] == installed
    weights = torch.load(portable/'weights.pt', map_location='cpu', weights_only=True)
    expected = {'model.'+n: t for n, t in current.items()}
    assert weights['generator'].keys() == expected.keys()
    assert all(torch.equal(weights['generator'][n], t) for n, t in expected.items())
    assert weights['codec'].keys() == codec_current.keys()
    assert all(torch.equal(weights['codec'][n], t) for n, t in codec_current.items())
    assert equal(rng, rng_state())
    # A fresh public inference factory must consume the portable FULL state,
    # not reopen the old trainables or relax missing/unexpected-key handling.
    fresh = native.build_inference_model(manifest['model']['parameters'])
    for name in ('generator', 'codec'):
        result = fresh.state[name].load_state_dict(weights[name], strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        assert all(torch.equal(t, fresh.state[name].state_dict()[n]) for n, t in weights[name].items())
    assert all(torch.equal(current[n], t) for n, t in g.state_dict().items())
    assert all(torch.equal(codec_current[n], t) for n, t in codec.state_dict().items())
    assert owners == {name: id(t) for name, t in g.named_parameters()}
    assert modes == {name: layer.training for name, layer in g.named_modules()}
    assert trainability == {name: t.requires_grad for name, t in g.named_parameters()}
    assert peft_config == g.peft_config['default'].to_dict()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault', ['missing_adapter', 'extra_adapter', 'wrong_default', 'merged'])
def test_actual_upstream_compacted_peft_projection_still_refuses_invalid_adapter_inventory(tmp_path, fault):
    from peft import LoraConfig
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.training import wan_native_inference as native
    path, g, codec, request = upstream_fixture(tmp_path/'OLD-source', num_layers=3)
    layer = g.blocks[0].attn1.to_q
    if fault == 'missing_adapter':
        layer.delete_adapter('default')
    elif fault == 'extra_adapter':
        g.extra_adapter = deepcopy(layer)
    elif fault == 'wrong_default':
        g.add_adapter(LoraConfig(r=1, lora_alpha=1., lora_dropout=0., init_lora_weights='gaussian',
            target_modules=['blocks.0.attn1.to_q']), adapter_name='other')
    else:
        layer.merge()
    current = deepcopy(g.state_dict()); peft_config = deepcopy(g.peft_config)
    p = request['parameters']
    output_request = dict(prompt_shape=p['geometry']['prompt_shape'], fps=p['geometry']['fps'],
        budget=p['budget'], seed=p['seed'], evidence_kind=p['evidence_kind'], codec=request['codec'])
    with pytest.raises(ValueError, match='full original LoRA/pyramid targets changed|CURRENT unmerged default Generator adapters required'):
        native.project_current_generator(NS(g=NS(module=g), task=OriginalWanTask.from_dict(p['task'])),
            codec, output_request, world_size=1)
    assert g.state_dict().keys() == current.keys()
    assert all(torch.equal(current[n], t) for n, t in g.state_dict().items())
    assert g.peft_config == peft_config
    assert not torch.cuda.is_initialized()
