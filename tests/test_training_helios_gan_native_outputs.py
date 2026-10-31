"""W3 output owner tests, no native CUDA/kernel/quality qualification."""
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from distillschedule.training.config import TrainingConfig


def config(root):
    return TrainingConfig.from_dict(dict(schema_version='distillschedule.training.v1',
        model=dict(workload='W3'), algorithm=dict(name='original_helios_gan'),
        data=dict(input_kind='original_helios_gan_cohort_stream.v1', prepared='stream.json'),
        execution=dict(adapter='native', mode='manual', task='task.json', assets='assets.json',
                       preset='helios_gan.fa3_down40_nomerge_stream.v1'),
        trainer=dict(cycles=100, checkpoint_every=50, output_dir='run')), source=root / 'training.json')


def test_gan_terminal_export_preserves_physical_and_logical_clocks():
    from distillschedule.training.helios_gan_training import validate_progress, expected_versions, logical_versions
    binding = dict(stream=dict(cohorts=[{}] * 100), checkpoint_policy={'binding': 'same'})
    header = dict(policy=binding['checkpoint_policy'], cycle=100, cursor=100, versions=expected_versions(100))
    assert header['versions'] == dict(generator=102, critic=510, ema=900)
    assert logical_versions(100) == dict(generator=300, critic=1500, ema=900)
    with pytest.raises(ValueError, match='incomplete-stream'):
        validate_progress(header, binding)
    validate_progress(header, {**binding, 'native_output': object()})
    for changes in ({'cycle': 101, 'cursor': 101}, {'cycle': 0, 'cursor': 0},
                    {'versions': logical_versions(100)}, {'cursor': 99}, {'policy': {}}, {'cycle': True}):
        with pytest.raises(ValueError):
            validate_progress({**header, **changes}, {**binding, 'native_output': object()})


def test_public_gan_export_uses_its_original_full_gan_owner(tmp_path, monkeypatch):
    from distillschedule.training import native, helios_gan_training as gan
    raw = config(tmp_path).to_dict()
    raw['trainer']['stop_after_cycle'] = 50
    raw['trainer']['export'] = {'schema_version': 'distillschedule.helios_gan_native_output_request.v1'}
    conf = TrainingConfig.from_dict(raw, source=tmp_path / 'training.json')
    checkpoint, output = tmp_path / 'checkpoint', tmp_path / 'portable'
    seen = []
    def admit(clean):
        assert clean.model == conf.model and clean.execution == conf.execution and clean.data == conf.data
        assert 'stop_after_cycle' not in clean.trainer
        assert 'export' not in clean.trainer and clean.trainer['checkpoint_every'] == 0
        assert clean.trainer['resume_from'] == str(checkpoint)
        seen.append('source-admit')
        return 'W3', dict(identity=dict(dirty=False)), {}, {}
    monkeypatch.setattr(native, '_admit', admit)
    def run(clean, *, admit, native_output):
        assert admit()[0] == 'W3'
        assert native_output.export_schema == 'distillschedule.helios_gan_native_output_request.v1'
        seen.append('original-GAN-owner')
        native_output.result = dict(status='EXPORTED', scope='SYNTHETIC_OWNER_DISPATCH_ONLY')
        return 0
    monkeypatch.setattr(gan, 'run_training', run)
    assert native.export_native(conf, checkpoint, output)['status'] == 'EXPORTED'
    assert seen == ['source-admit', 'original-GAN-owner']


def with_gan_head(factory, *, current):
    """Real named GAN-head/Adam state; counters are explicit synthetic markers."""
    import torch
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
    g, old, ema = factory(current=current)
    old.module.add_module('gan_head', torch.nn.Linear(1, 1))
    parameters = [p for p in old.module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.9)
    step = 510 if current else 10
    c = FSDP2EngineAdapter(old.module, optimizer, scheduler, physical_step=step, global_batch=old.train_batch_size())
    scheduler.last_epoch, scheduler._step_count = step, step + 1
    for p in parameters:
        if current:
            with torch.no_grad():
                p.fill_(.037)
        optimizer.state[p] = dict(step=torch.tensor(float(step)), exp_avg=torch.full_like(p, .017),
                                   exp_avg_sq=torch.full_like(p, .021))
    ema.optimization_step = 900 if current else 800
    return g, c, ema


def _owned_gan_export(root, monkeypatch, *, engines_factory, request, callback_type, task=None):
    """Actual public/GAN owner/complete state restore; outer CUDA intake injected."""
    from contextlib import contextmanager
    from distillschedule.training import native, helios_worker as worker, helios_lifecycle as lifecycle
    from distillschedule.training import helios_gan_training as gan, helios_gan_inputs as inputs
    from distillschedule.training import helios_gan_native_inference as output
    from distillschedule.training.native_state import engine_state
    from distillschedule.training.checkpoint import save_checkpoint, pack_state, rng_state
    from test_training_end_to_end import equal
    task = task or inputs.REQUEST
    stream = dict(stream_sha256='1' * 64, identity=dict(source_fingerprint='SYNTHETIC_INTAKE', request=task.to_dict()),
                  cohorts=[{} for _ in range(100)])
    assets = dict(assets={'scope': 'SYNTHETIC_BOOTSTRAP_NOT_GAN_OR_GPU_QUALIFICATION'})
    admitted = dict(stream=stream, assets=assets, selected=gan.descriptor(), implementation=gan.implementation(),
                    bootstrap=dict(logical_manifest_hash='2' * 64), argv=[])
    identity = dict(schema_version='distillschedule.helios_gan_training_identity.v1', source_head='3' * 40,
        source_fingerprint=stream['identity']['source_fingerprint'], stream_sha256=stream['stream_sha256'],
        assets=assets['assets'], execution_policy=admitted['selected'], bootstrap_logical_hash='2' * 64)
    old = root / 'old-training'; old.mkdir()
    g, c, ema = engines_factory(current=True)
    state = engine_state(dict(generator=g, critic=c), ema=ema)
    before, policy = pack_state(state), gan._checkpoint_policy(identity)
    assert any('gan_head' in n for n in before['critic.weights']['value']['tensors'])
    checkpoint = save_checkpoint(old / 'checkpoints/cycle-000100', state, identity=identity, cycle=100, cursor=100,
        versions=gan.expected_versions(100), policy=policy)
    saved_rng = rng_state(device='cpu')
    raw = config(root).to_dict(); raw['trainer']['export'] = request
    conf = TrainingConfig.from_dict(raw, source=root / 'training.json')
    monkeypatch.setattr(native, '_admit', lambda clean: ('W3', dict(identity=dict(dirty=False, head='3' * 40)), {}, admitted))
    @contextmanager
    def transport(label):
        assert label == 'W3'
        yield dict(metadata_group=None, owns_process_group=True, scope='SYNTHETIC_OUTER_TRANSPORT_ONLY')
    monkeypatch.setattr(gan.distributed, 'world16_reference_process_group', transport)
    monkeypatch.setattr(gan, 'validate_environment', lambda: None)
    monkeypatch.setenv('RANK', '0')
    observed = []
    class NoCycle:
        def __getattr__(self, name):
            return lambda *a, **kw: pytest.fail('export called GAN training/cache lifecycle: ' + name)
    def main():
        binding = lifecycle.current_training()
        assert binding['gan_stream'] is True and binding['identity'] == identity
        binding['gan_core'], binding['checkpoint_policy'] = NoCycle(), policy
        a, b, shadow = engines_factory(current=False)
        ids = [[id(p) for p in e.module.parameters()] for e in (a, b)]
        modes = [[(n, m.training) for n, m in e.module.named_modules()] for e in (a, b)]
        flags = [[p.requires_grad for p in e.module.parameters()] for e in (a, b)]
        args = NS(training_config=NS(base_seq_len=256, max_seq_len=4096, base_shift=.5, max_shift=1.15))
        result = {}
        gan.execute_cycles(p0=NS(), args=args, policy=NS(), result=result, rank=0, device='cpu',
            generator=a, critic=b, ema=shadow, diagnostics=None, negative=None, cfg=None,
            generator_free_memory=None, binding=binding)
        assert equal(before, pack_state(engine_state(dict(generator=a, critic=b), ema=shadow)))
        assert equal(saved_rng, rng_state(device='cpu'))
        assert ids == [[id(p) for p in e.module.parameters()] for e in (a, b)]
        assert modes == [[(n, m.training) for n, m in e.module.named_modules()] for e in (a, b)]
        assert flags == [[p.requires_grad for p in e.module.parameters()] for e in (a, b)]
        assert result['output_only'] == dict(cycles=0, capture=0, optimizer_updates=0, ema_updates=0, merge_count=0)
        assert result['final_versions'] == gan.expected_versions(100)
        observed.append(result)
        return 0
    monkeypatch.setattr(worker, '_backend', lambda: NS(main=main))
    monkeypatch.setattr(output, 'NativeOutputRequest', callback_type)
    import distillschedule.runtime.helios_planner_cycle as original
    monkeypatch.setattr(original, '_prepare_cycle', lambda **kw: pytest.fail('export prepared a training cycle'))
    monkeypatch.setattr(inputs, 'load_cohort', lambda *a, **kw: pytest.fail('export consumed a GAN cohort'))
    result = native.export_native(conf, checkpoint, root / 'portable')
    assert len(observed) == 1 and lifecycle.current_training() is None
    return result, observed[0], old


@pytest.mark.parametrize('aligned', [False, True], ids=['normal','explicit-aligned'])
def test_full_native_gan_state_and_heads_restored_without_training(tmp_path, monkeypatch, aligned):
    from test_training_helios_native_outputs import _marker_engines
    from distillschedule.search.helios_request import OriginalHeliosRequest
    task = OriginalHeliosRequest(variant='gan',stage_schedule_policy='benchmark_v4_strict_global') if aligned else None
    calls = []
    class Output:
        def __init__(self, value, output):
            self.result = None
        def admit(self):
            calls.append('admit')
        def publish(self, view, *, manifest, identity):
            assert view.request['variant'] == 'gan'
            if aligned:
                assert view.request == task.to_dict()
            assert view.generator.global_steps == 102 and manifest['versions']['ema'] == 900
            assert calls == ['admit']
            calls.append('publish')
            self.result = dict(status='EXPORTED', test='SYNTHETIC_CALLBACK_REAL_FULL_GAN_STATE_RESTORE')
            return self.result
    factory = lambda **kw: with_gan_head(_marker_engines, **kw)
    result, receipt, _ = _owned_gan_export(tmp_path, monkeypatch, engines_factory=factory,
                                         request={}, callback_type=Output,task=task)
    assert result['status'] == 'EXPORTED' and calls == ['admit', 'publish']
    assert receipt['completed_cycles'] == 0 and receipt['resume_cycle'] == 100


@pytest.mark.parametrize('change', ['nogan_schema', 'wan_schema', 'nogan_task', 'source', 'runtime', 'noise', 'geometry'])
def test_gan_portable_identity_never_aliases_nogan_or_wan(monkeypatch, change):
    from distillschedule.training import helios_gan_native_inference as output
    from distillschedule.training.helios_gan_inputs import REQUEST
    from test_training_helios_native_outputs import output_budget
    video = output.common.video
    p = dict(schema_version=output.SCHEMA, precision=output.common.PRECISION,
        source={'test': 'SYNTHETIC_GAN_SOURCE'}, runtime={}, architecture={}, peft={'r': 1}, task=REQUEST.to_dict(),
        native={**video.NATIVE, 'rollout': video._rollout(REQUEST)}, noise_protocol=video.noise_protocol(),
        geometry=dict(noise_shape=list(REQUEST.latent_shape), prompt_shape=[1, 2, 4096],
                      video_shape=[33, 384, 640, 3], fps=16),
        budget=output_budget(), seed=721, evidence_kind=video.FIXTURE, codec={'evidence_kind': video.FIXTURE})
    monkeypatch.setattr(output, 'inference_source', lambda: {'test': 'SYNTHETIC_GAN_SOURCE'})
    monkeypatch.setattr(video, 'runtime_identity', lambda: {})
    monkeypatch.setattr(video, '_generator', lambda *a, **kw: pytest.fail('invalid identity reached constructor'))
    monkeypatch.setattr(video, '_admit_codec', lambda *a, **kw: pytest.fail('invalid identity reached codec'))
    if change in ('nogan_schema', 'wan_schema'):
        p['schema_version'] = output.common.SCHEMA if change == 'nogan_schema' else video.SCHEMA
    elif change == 'nogan_task':
        p['task']['variant'] = 'no-gan'
    elif change in ('source', 'runtime'):
        p[change] = {'changed': True}
    elif change == 'noise':
        p['noise_protocol']['schema_version'] = 'OLD_CHOLESKY'
    else:
        p['geometry']['noise_shape'][-1] -= 1
    with pytest.raises(ValueError):
        output._probe(p)


def test_gan_request_rejects_w2_request_before_codec_loading(tmp_path, monkeypatch):
    from distillschedule.training import helios_gan_native_inference as output
    monkeypatch.setattr(output.common.video, '_load_codec', lambda *a: pytest.fail('W2 request allocated W3 codec'))
    request = {'schema_version': output.common.EXPORT_SCHEMA, 'seed': 721, 'prompt_shape': [1, 2, 4096],
               'fps': 16, 'budget': {}, 'codec': {}, 'evidence_kind': output.common.video.FIXTURE}
    with pytest.raises(ValueError, match='trainer.export'):
        output.NativeOutputRequest(request, tmp_path / 'portable').admit()


def test_merged_current_generator_refused_before_projection_or_capacity():
    from distillschedule.training import helios_gan_native_inference as output
    g = NS(peft_config={'default': None}, modules=lambda: [NS(merged=True)])
    view = NS(generator=NS(module=g))
    callback = output.NativeOutputRequest({}, 'unused')
    with pytest.raises(ValueError, match='unmerged'):
        callback.project(view)
