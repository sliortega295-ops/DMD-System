"""Native W2 output ownership, not pretrained/world16 CUDA qualification."""
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from test_training_helios_stream import config


def test_terminal_output_does_not_weaken_training_resume():
    from distillschedule.training.helios_lifecycle import validate_progress, expected_versions
    binding = dict(stream=dict(cohorts=[{}] * 100), checkpoint_policy={'binding': 'same'})
    header = dict(policy=binding['checkpoint_policy'], cycle=100, cursor=100, versions=expected_versions(100))
    with pytest.raises(ValueError, match='incomplete-stream'):
        validate_progress(header, binding)
    output = {**binding, 'native_output': object()}
    validate_progress(header, output)
    for changes in ({'cycle': 101, 'cursor': 101, 'versions': expected_versions(101)},
                    {'cycle': 0, 'cursor': 0, 'versions': expected_versions(0)},
                    {'versions': expected_versions(99)}, {'cursor': 99}, {'policy': {}}):
        with pytest.raises(ValueError):
            validate_progress({**header, **changes}, output)


def test_public_export_enters_original_worker_with_complete_request(tmp_path, monkeypatch):
    from distillschedule.training import native, helios_worker
    from distillschedule.training.config import TrainingConfig
    raw = config(tmp_path, cycles=100).to_dict()
    raw['trainer']['stop_after_cycle'] = 50
    raw['trainer']['export'] = {'schema_version': 'distillschedule.helios_native_output_request.v1'}
    raw['trainer']['resume_from'] = str(tmp_path / 'stale-checkpoint')
    conf = TrainingConfig.from_dict(raw, source=tmp_path / 'training.json')
    checkpoint, output = tmp_path / 'actual-checkpoint', tmp_path / 'portable'
    seen = []
    def admit(clean):
        assert 'stop_after_cycle' not in clean.trainer
        assert 'export' not in clean.trainer and clean.trainer['checkpoint_every'] == 0
        assert clean.trainer['resume_from'] == str(checkpoint)
        assert clean.trainer['cycles'] == 100 and clean.execution == conf.execution
        assert clean.data == conf.data and Path(clean.trainer['output_dir']) != output
        seen.append('admit')
        return 'W2', dict(identity=dict(dirty=False)), {}, {'test': 'synthetic admission only'}
    monkeypatch.setattr(native, '_admit', admit)
    def run(clean, *, admit=None, native_output=None):
        assert callable(admit) and native_output is not None
        assert admit()[0] == 'W2'
        assert native_output.value == raw['trainer']['export']
        seen.append('original-worker-entry')
        native_output.result = dict(status='EXPORTED', checkpoint_cycle=100, test='SYNTHETIC_TRANSPORT_ONLY')
        return 0
    monkeypatch.setattr(helios_worker, 'run_reference', run)
    result = native.export_native(conf, checkpoint, output)
    assert result['checkpoint_cycle'] == 100 and seen == ['admit', 'original-worker-entry']


def _marker_engines(current=False):
    """Real native state owners; cycle counters are synthetic checkpoint markers."""
    import torch
    from test_training_native_state import _engine
    g, c = _engine(), _engine()
    for e, step in ((g, 102 if current else 2), (c, 510 if current else 10)):
        e.global_steps, e.global_samples = step, step * e.train_batch_size()
        e.lr_scheduler.last_epoch, e.lr_scheduler._step_count = step, step + 1
        for p in e.module.parameters():
            if p.requires_grad:
                if current:
                    with torch.no_grad():
                        p.fill_(.031)
                e.optimizer.state[p] = dict(step=torch.tensor(float(step)), exp_avg=torch.full_like(p, .017),
                                            exp_avg_sq=torch.full_like(p, .021))
    names = {n: p for n, p in g.module.named_parameters() if p.requires_grad}
    ema = NS(_names=list(names), _parameters=list(names.values()),
        _shadows=[p.detach().clone() + .07 for p in names.values()], optimization_step=510 if current else 10,
        decay=.9, min_decay=0., update_after_step=0, cur_decay_value=None, device='cpu')
    return g, c, ema


def _native_parser_args(root, *, variant='no-gan'):
    """The real native parser is flat; its rollout constructs fixed shift Args."""
    import argparse
    from distillschedule.training import helios_worker as worker
    from distillschedule.training.helios_reference_assets import SCHEMA as ASSETS
    roles = ('generator_model', 'generator_precast', 'score_model', 'score_precast',
             'logical_checkpoint', 'negative_prompt')
    assets = dict(schema_version=ASSETS, assets={role: dict(path=str(root / role)) for role in roles})
    argv = worker.reference_argv(assets, root / 'SYNTHETIC_TEST_ONLY_PARSER',
        stream=dict(cohorts=[dict(trace_bundle=str(root / 'trace'), trace_hash='2' * 64)]),
        logical_hash='1' * 64)
    argv[argv.index('--variant') + 1] = variant
    args = worker._backend().impl._parser().parse_args(argv)
    assert type(args) is argparse.Namespace and args.variant == variant
    assert not hasattr(args, 'training_config')
    assert not any(hasattr(args, key) for key in ('base_seq_len', 'max_seq_len', 'base_shift', 'max_shift'))
    return args


def test_native_flat_parser_has_no_sampler_config_and_fixed_rollout_controls_match():
    import ast
    from distillschedule.training import helios_native_inference as output
    from distillschedule.training.helios_inference import SHIFT_KEYS
    from distillschedule.workloads.best_known.registry import repository_root
    root = repository_root()
    _native_parser_args(root)
    for relative in ('runtime/gst_pool_helios.py', 'runtime/static_fsdp2_p0_helios.py',
                     'runtime/critic_pool_helios.py'):
        tree = ast.parse((root / relative).read_text())
        controls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == 'SimpleNamespace'
                    and {kw.arg for kw in node.keywords} >= set(SHIFT_KEYS)]
        assert len(controls) == 1
        actual = {kw.arg: ast.literal_eval(kw.value) for kw in controls[0].keywords if kw.arg in SHIFT_KEYS}
        assert actual == output.video.NATIVE['args']


def _owned_export(tmp_path, monkeypatch, *, optimized, engines_factory, request, callback_type, task=None):
    """Actual public/worker/lifecycle/native-state chain; CUDA/asset intake injected.

    The bootstrap/input/transport identities are explicitly synthetic. No
    training cycle is run merely to produce an output-consumer checkpoint.
    """
    from contextlib import contextmanager
    import torch
    from distillschedule.training import native, helios_worker as worker, helios_lifecycle as lifecycle
    from distillschedule.training import helios_native_inference as output
    from distillschedule.training import helios_safe_stream as safe
    from distillschedule.training.native_state import engine_state
    from distillschedule.training.checkpoint import save_checkpoint, pack_state, rng_state
    from distillschedule.training.helios_reference_assets import SCHEMA as ASSETS
    from distillschedule.training.config import TrainingConfig
    from distillschedule.search.helios_request import OriginalHeliosRequest
    from test_training_end_to_end import equal
    task = task or OriginalHeliosRequest(stage2_inference_steps=(1, 1, 1))
    stream = dict(stream_sha256='1' * 64,
        identity=dict(source_fingerprint='SYNTHETIC_ADMISSION_ONLY', request=task.to_dict()),
        cohorts=[dict(trace_hash='2' * 64) for _ in range(100)])
    assets = dict(schema_version=ASSETS, assets={'scope': 'SYNTHETIC_BOOTSTRAP_NOT_NATIVE_CAPACITY'})
    selected = safe.descriptor() if optimized else dict(scope='SYNTHETIC_REFERENCE_POLICY')
    identity = dict(schema_version='distillschedule.helios_safe_stream_identity.v1' if optimized
                    else 'distillschedule.helios_training_identity.v1', source_head='3' * 40,
        source_fingerprint=stream['identity']['source_fingerprint'], stream_sha256=stream['stream_sha256'],
        assets=assets['assets'], bootstrap_logical_hash='4' * 64, bootstrap_trace_hash='2' * 64)
    identity['execution_policy' if optimized else 'reference_policy'] = selected
    old = tmp_path / 'old-training'; old.mkdir()
    g, c, ema = engines_factory(current=True)
    state = engine_state(dict(generator=g, critic=c), ema=ema)
    before = pack_state(state)
    policy = lifecycle._checkpoint_policy(identity, optimized=optimized)
    checkpoint = save_checkpoint(old / 'checkpoints/cycle-000100', state, identity=identity, cycle=100, cursor=100,
        versions=lifecycle.expected_versions(100), policy=policy)
    saved_rng = rng_state(device='cpu')
    raw = config(tmp_path, cycles=100, mode='manual' if optimized else 'reference').to_dict()
    if optimized:
        raw['execution']['preset'] = safe.PRESET_ID
    raw['trainer']['export'] = request
    conf = TrainingConfig.from_dict(raw, source=tmp_path / 'training.json')
    admitted = dict(stream=stream, assets=assets, selected=selected,
                    bootstrap=dict(logical_manifest_hash='4' * 64), argv=[])
    if optimized:
        admitted['implementation'] = safe.implementation()
    monkeypatch.setattr(native, '_admit', lambda clean: ('W2', dict(identity=dict(dirty=False, head='3' * 40)), {}, admitted))
    @contextmanager
    def transport():
        # NOT real world16: source/asset/outer CUDA transport are injected;
        # actual native restore and callback scopes below are not replaced.
        yield dict(metadata_group=None, owns_process_group=True, scope='SYNTHETIC_TRANSPORT_ONLY')
    monkeypatch.setattr(worker, 'reference_process_group', transport)
    monkeypatch.setattr(safe, 'validate_environment', lambda **kw: None)
    monkeypatch.setenv('RANK', '0')
    args = _native_parser_args(tmp_path)
    observed = []
    def main():
        binding = lifecycle.current_training()
        assert binding['native_output'] is not None and binding['config'].trainer['checkpoint_every'] == 0
        assert binding['identity'] == identity
        binding['checkpoint_policy'] = policy
        # In production these come from original source/asset/build admission.
        if optimized:
            binding['effective_configuration'] = dict(scope='SYNTHETIC_INSTALLATION_ONLY')
        a, b, shadow = engines_factory(current=False)
        ids = [[id(p) for p in e.module.parameters()] for e in (a, b)]
        result = {}
        lifecycle.execute_cycles(p0=NS(), args=args, policy=NS(), generator=a, critic=b, ema=shadow,
            diagnostics=None, negative=None, rank=0, device='cpu', binding=binding, result=result)
        assert equal(before, pack_state(engine_state(dict(generator=a, critic=b), ema=shadow)))
        assert equal(saved_rng, rng_state(device='cpu'))
        assert ids == [[id(p) for p in e.module.parameters()] for e in (a, b)]
        assert result['output_only'] == dict(cycles=0, capture=0, optimizer_updates=0, ema_updates=0, merge_count=0)
        assert result['completed_cycles'] == 0 and result['resume_cycle'] == 100
        assert result['final_versions'] == lifecycle.expected_versions(100)
        observed.append(result)
        return 0
    monkeypatch.setattr(worker, '_backend', lambda: NS(main=main))
    monkeypatch.setattr(output, 'NativeOutputRequest', callback_type)
    import distillschedule.runtime.helios_planner_cycle as original
    monkeypatch.setattr(original, '_prepare_cycle', lambda **kw: pytest.fail('output prepared a training cycle'))
    monkeypatch.setattr(lifecycle, 'load_cohort', lambda *a, **kw: pytest.fail('output consumed a training cohort'))
    result = native.export_native(conf, checkpoint, tmp_path / 'portable')
    assert len(observed) == 1 and lifecycle.current_training() is None
    return result, observed[0], old


@pytest.mark.parametrize('optimized', [False, True], ids=['reference', 'safe_stream'])
@pytest.mark.parametrize('aligned', [False, True], ids=['normal','explicit-aligned'])
def test_actual_native_restore_output_boundary_preserves_all_owned_state(tmp_path, monkeypatch, optimized, aligned):
    from distillschedule.search.helios_request import OriginalHeliosRequest
    task = OriginalHeliosRequest(stage_schedule_policy='benchmark_v4_strict_global') if aligned else None
    calls = []
    class Output:
        def __init__(self, value, output):
            self.result = None
        def admit(self):
            calls.append('admit-before-build')
        def publish(self, view, *, manifest, identity):
            assert manifest['cycle'] == 100 and view.generator.global_steps == 102
            if aligned:
                assert view.request == task.to_dict()
            assert calls == ['admit-before-build']
            calls.append('publish-after-restore')
            self.result = dict(status='EXPORTED', test='SYNTHETIC_CALLBACK_REAL_NATIVE_RESTORE')
            return self.result
    result, receipt, old = _owned_export(tmp_path, monkeypatch, optimized=optimized,
        engines_factory=_marker_engines, request={}, callback_type=Output,task=task)
    assert result['status'] == 'EXPORTED' and calls == ['admit-before-build', 'publish-after-restore']
    assert receipt['output_only']['optimizer_updates'] == 0


def output_budget():
    return dict(max_parameters=20_000_000, max_loaded_bytes=2**31, max_primary_export_bytes=2**31,
        max_transient_tensor_bytes=2**28, max_activation_elements=100_000_000,
        max_attention_elements=512_000_000, host_reserve_bytes=2**20, device_reserve_bytes=2**20)


@pytest.mark.parametrize('change', ['source', 'runtime', 'schema', 'precision', 'gan', 'wrong_family',
                                  'world8', 'geometry', 'prompt_dtype_shape', 'budget', 'noise'])
def test_portable_w2_metadata_refuses_before_any_model_constructor(monkeypatch, change):
    from distillschedule.training import helios_native_inference as output
    from distillschedule.search.helios_request import OriginalHeliosRequest
    task = OriginalHeliosRequest()
    p = dict(schema_version=output.SCHEMA, precision=output.PRECISION,
        source={'test': 'SYNTHETIC_SOURCE_BINDING'}, runtime={}, architecture={}, peft={'r': 1},
        task=task.to_dict(), native={**output.video.NATIVE, 'rollout': output.video._rollout(task)},
        noise_protocol=output.video.noise_protocol(), geometry=dict(noise_shape=list(task.latent_shape),
            prompt_shape=[1, 2, 4096], video_shape=[33, 384, 640, 3], fps=16), budget=output_budget(),
        seed=721, evidence_kind=output.video.FIXTURE, codec={'evidence_kind': output.video.FIXTURE})
    monkeypatch.setattr(output, 'inference_source', lambda: {'test': 'SYNTHETIC_SOURCE_BINDING'})
    monkeypatch.setattr(output.video, 'runtime_identity', lambda: {})
    def forbidden(*args, **kwargs):
        pytest.fail('invalid metadata reached a model or codec constructor')
    monkeypatch.setattr(output.video, '_generator', forbidden)
    monkeypatch.setattr(output.video, '_admit_codec', forbidden)
    if change in ('source', 'runtime'):
        p[change] = {'changed': True}
    elif change in ('schema', 'precision'):
        p['schema_version' if change == 'schema' else change] = 'wrong'
    elif change == 'gan':
        p['task']['variant'] = 'gan'
    elif change == 'wrong_family':
        p['task']['model_family'] = 'wan1.3b'
    elif change == 'world8':
        p['task']['world_size'] = 8
    elif change == 'geometry':
        p['geometry']['noise_shape'][-1] -= 1
    elif change == 'prompt_dtype_shape':
        p['geometry']['prompt_shape'][2] = 32
    elif change == 'budget':
        p['budget']['max_primary_export_bytes'] = 0
    else:
        p['noise_protocol']['schema_version'] = 'OLD_CHOLESKY'
    with pytest.raises(ValueError):
        output._probe(p)


def test_host_capacity_accounts_for_actual_parent_limit_and_preserves_reserve(tmp_path, monkeypatch):
    from distillschedule.training import helios_native_inference as output
    from distillschedule.search import resources
    child, parent = tmp_path / 'child', tmp_path / 'parent'
    child.mkdir(); parent.mkdir()
    (child / 'memory.current').write_text('100')
    (parent / 'memory.current').write_text('800')
    monkeypatch.setattr(resources, '_capacity_observation', lambda: (child, dict(cgroup_limits=[
        dict(path=str(child), limit_bytes=1000), dict(path=str(parent), limit_bytes=1100)])))
    monkeypatch.setattr(output.os, 'sysconf', lambda name: 1000 if name == 'SC_AVPHYS_PAGES' else 1)
    assert output._host_admit(200, 100)['available_bytes'] == 300
    with pytest.raises(ValueError, match='host/cgroup'):
        output._host_admit(201, 100)


def test_owned_device_capacity_never_queries_an_unowned_device(monkeypatch):
    from distillschedule.training import helios_native_inference as output
    import torch
    calls = []
    def memory(device):
        calls.append(torch.device(device)); return 200, 300
    monkeypatch.setattr(torch.cuda, 'mem_get_info', memory)
    output._device_admit('cpu', 1000, 1000)
    output._device_admit('cuda:1', 100, 100)
    with pytest.raises(ValueError, match='owned-device'):
        output._device_admit('cuda:1', 101, 100)
    assert calls == [torch.device('cuda:1'), torch.device('cuda:1')]


def test_codec_host_rejection_precedes_real_allocation(tmp_path, monkeypatch):
    from distillschedule.training import helios_native_inference as output
    request = dict(schema_version=output.EXPORT_SCHEMA, budget=output_budget(), seed=721,
        fps=16, prompt_shape=[1, 2, 4096], evidence_kind=output.video.FIXTURE,
        codec=dict(weights=[dict(bytes=1024)]))
    def reject(*args):
        raise ValueError('injected actual cgroup headroom refusal')
    monkeypatch.setattr(output, '_host_admit', reject)
    monkeypatch.setattr(output.video, '_load_codec', lambda *args: pytest.fail('allocated before host admission'))
    with pytest.raises(ValueError, match='cgroup headroom'):
        output.NativeOutputRequest(request, tmp_path / 'portable').admit()


@pytest.mark.parametrize('pin', [None, 'not-a-pin', 4, {'bytes': '1024'}, {'bytes': 0}])
def test_malformed_codec_pin_is_a_contract_error_before_allocation(tmp_path, monkeypatch, pin):
    from distillschedule.training import helios_native_inference as output
    request = dict(schema_version=output.EXPORT_SCHEMA, budget=output_budget(), seed=721,
        fps=16, prompt_shape=[1, 2, 4096], evidence_kind=output.video.FIXTURE,
        codec=dict(weights=[pin]))
    monkeypatch.setattr(output.video, '_load_codec', lambda *args: pytest.fail('invalid pin reached codec allocation'))
    with pytest.raises(ValueError, match='pinned bounded codec shards'):
        output.NativeOutputRequest(request, tmp_path / 'portable').admit()


def test_sidecar_failure_does_not_fabricate_config_or_discard_committed_weights(tmp_path, monkeypatch):
    import torch
    from distillschedule.training import helios_native_inference as output
    from distillschedule.training.adapters import InferenceProjection, ModelComponents
    from distillschedule.training.inference import read_export
    components = ModelComponents(state=dict(generator=torch.nn.Linear(2, 2)),
                                 metadata=dict(evidence_kind='SYNTHETIC_PROJECTION_ONLY'))
    monkeypatch.setattr(output, '_project', lambda *args: InferenceProjection(components, output.FACTORY,
                         {'test': 'SYNTHETIC_PROJECTION_NOT_LOADABLE_NATIVE_MODEL', 'seed': 721}))
    def fail(*args):
        raise OSError('injected companion config failure')
    monkeypatch.setattr(output.os, 'link', fail)
    callback = output.NativeOutputRequest({}, tmp_path / 'portable')
    with pytest.raises(OSError, match='companion config failure'):
        callback.publish(NS(), manifest=dict(cycle=100), identity={'test': 'SYNTHETIC_TRAINING_IDENTITY'})
    root, manifest = read_export(tmp_path / 'portable')
    assert manifest['status'] == 'EXPORTED' and (root / 'weights.pt').is_file()
    assert callback.result is None and not callback.config_path.exists()
    assert not list(tmp_path.glob('.portable.inference.json.tmp-*'))


def _compressed_original_projection(tmp_path, variant):
    """Real two-block class/LoRA256: PEFT's >=20-target compression, no training."""
    from copy import deepcopy
    import inspect
    import torch
    from test_training_helios_public_cpu import STACK_AVAILABLE
    if not STACK_AVAILABLE:
        pytest.skip('original optional Helios CPU stack required')
    from peft import LoraConfig
    from peft.tuners.tuners_utils import BaseTunerLayer, MIN_TARGET_MODULES_FOR_OPTIMIZATION
    from test_training_wan_native_outputs import codec_request
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from distillschedule.search.helios_request import OriginalHeliosRequest
    from distillschedule.training import helios_native_inference as common
    from distillschedule.training import helios_gan_native_inference as gan
    from runtime.critic_pool_helios import TRANSFORMER_ADDITIONAL_KWARGS, _all_linear_targets
    module = gan if variant == 'gan' else common
    cls = load_reference_model_class(backend='torch-sdpa')
    architecture = common.video.plain({k: p.default for k, p in inspect.signature(cls.__init__).parameters.items()
                                      if k != 'self'})
    architecture.update(num_attention_heads=1, attention_head_dim=16, freq_dim=16, ffn_dim=32,
                        num_layers=2, rope_dim=[4, 6, 6], **TRANSFORMER_ADDITIONAL_KWARGS)
    g = cls(**architecture)
    for name, value in g.named_parameters():
        value.data = value.data.to(torch.float32 if any(k in name for k in cls._keep_in_fp32_modules)
                                   else torch.bfloat16)
    targets = sorted(set(_all_linear_targets(g) + ['patch_short', 'patch_mid', 'patch_long']))
    assert len(targets) == 29 and len(targets) >= MIN_TARGET_MODULES_FOR_OPTIMIZATION
    g.requires_grad_(False)
    g.add_adapter(LoraConfig(r=256, lora_alpha=256., lora_dropout=0., init_lora_weights='gaussian',
                            target_modules=targets, exclude_modules=['down', 'up']))
    for name, value in g.named_parameters():
        if 'lora_' in name:
            value.data = value.data.float()
    actual = sorted(name for name, layer in g.named_modules() if isinstance(layer, BaseTunerLayer)
                    and 'default' in getattr(layer, 'lora_A', {}) and 'default' in getattr(layer, 'lora_B', {}))
    assert actual == targets and len(g.peft_config['default'].target_modules) < len(targets)
    g.enable_gradient_checkpointing(); g.train()
    # Materialize real optimizer state independently of a model forward/update.
    optimizer = torch.optim.AdamW([value for value in g.parameters() if value.requires_grad], lr=.001)
    for value in g.parameters():
        if value.requires_grad:
            optimizer.state[value] = dict(step=torch.tensor(2.), exp_avg=torch.full_like(value, .017),
                                          exp_avg_sq=torch.full_like(value, .021))
            value.grad = torch.full_like(value, .009)
    request = codec_request(tmp_path / 'codec')
    request.update(schema_version=module.EXPORT_SCHEMA, budget=output_budget())
    codec, _ = common.video._load_codec(request['codec'], request['budget'], request['evidence_kind'])
    callback = module.NativeOutputRequest(request, tmp_path / 'portable')
    callback.codec = codec
    view = NS(generator=NS(module=g), request=OriginalHeliosRequest(variant=variant).to_dict())
    return NS(model=g, optimizer=optimizer, callback=callback, view=view, targets=targets, module=module,
              original_config=deepcopy(g.peft_config['default'].to_dict()))


def _projection_owned_snapshot(case):
    from copy import deepcopy
    from distillschedule.training.checkpoint import rng_state
    g = case.model
    return dict(parameters={name: (id(value), value._version, value.requires_grad, value.detach().clone(),
                                   None if value.grad is None else value.grad.detach().clone())
                            for name, value in g.named_parameters()},
                buffers={name: value.detach().clone() for name, value in g.named_buffers()},
                flags=[(name, module.training) for name, module in g.named_modules()],
                checkpoint=g.gradient_checkpointing, peft=deepcopy(g.peft_config['default'].to_dict()),
                optimizer=deepcopy(case.optimizer.state_dict()),
                optimizer_parameters=[[id(value) for value in group['params']] for group in case.optimizer.param_groups],
                rng=rng_state(device='cpu'))


@pytest.mark.parametrize('variant', ['no-gan', 'gan'], ids=['W2', 'W3'])
def test_native_projection_uses_actual_installed_targets_after_peft_compression(tmp_path, variant):
    from test_training_end_to_end import equal
    from distillschedule.training import helios_native_inference as common
    case = _compressed_original_projection(tmp_path, variant)
    before = _projection_owned_snapshot(case)
    projection = case.callback.project(case.view)
    assert equal(before, _projection_owned_snapshot(case))
    assert case.model.peft_config['default'].to_dict() == case.original_config
    assert projection.parameters['peft']['target_modules'] == case.targets
    assert projection.parameters['source'] == case.module.inference_source()
    assert projection.parameters['schema_version'] == case.module.SCHEMA and projection.factory == case.module.FACTORY
    assert projection.components.state['generator'].model is case.model
    if variant == 'gan':
        assert projection.parameters['source'] != common.inference_source()


@pytest.mark.parametrize('variant', ['no-gan', 'gan'], ids=['W2', 'W3'])
@pytest.mark.parametrize('fault', ['missing-A', 'missing-B', 'extra-target', 'dtype', 'merged'])
def test_native_compressed_targets_keep_exact_adapter_and_state_refusals(tmp_path, variant, fault):
    from copy import deepcopy
    import torch
    from peft.tuners.tuners_utils import BaseTunerLayer
    from test_training_end_to_end import equal
    case = _compressed_original_projection(tmp_path, variant)
    layer = next(layer for layer in case.model.modules() if isinstance(layer, BaseTunerLayer))
    if fault == 'missing-A':
        layer.lora_A.pop('default')
    elif fault == 'missing-B':
        layer.lora_B.pop('default')
    elif fault == 'extra-target':
        case.model.add_module('unexpected_adapter', deepcopy(layer))
    elif fault == 'dtype':
        layer.lora_A['default'].weight.data = layer.lora_A['default'].weight.data.to(torch.bfloat16)
    else:
        layer.merged_adapters.append('default')
    before = _projection_owned_snapshot(case)
    with pytest.raises(ValueError, match='LoRA/pyramid targets|state/dtype layout|CURRENT unmerged default'):
        case.callback.project(case.view)
    assert equal(before, _projection_owned_snapshot(case))
