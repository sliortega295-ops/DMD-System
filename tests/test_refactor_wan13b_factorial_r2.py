"""R2 CPU control-flow checks; fake device/model/math are SYNTHETIC_TEST_ONLY."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from distillschedule.runtime import wan13b_factorial_r2 as r2

ROOT = Path(__file__).resolve().parents[1]
HEAD = '2' * 40


def configuration(cell='baseline'):
    return r2.make_configuration(cell, expected_head=HEAD,
                                 source_hashes=r2.source_hashes(ROOT))


@pytest.mark.parametrize('cell,placement,b', [
    ('baseline', 'world16-fsdp', False),
    ('only_A', 'resident-async128-tree-chunk2048', False),
    ('full', 'resident-async128-tree-chunk2048', True),
])
def test_exact_lowering_and_independent_runtime_policy(cell, placement, b):
    cfg = configuration(cell)
    r2.validate_configuration(cfg, expected_head=HEAD)
    assert cfg['placement'] == placement
    assert cfg['runtime_policy'] == {
        'clean_runtime': b, 'runtime_constants_cache': b,
        'resident_clean_model_state': b,
    }
    assert cfg['cycles'] == {'count': 4, 'warmup': [1], 'measurement': [2, 3, 4]}
    env = r2.lower_environment(cfg)
    assert env['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] == '1'
    assert env['DISTILLSCHEDULE_WAN13B_EMA_IMPLEMENTATION'] == 'reference'
    assert cfg['operator_flags']['score_cuda_graph'] is False
    assert cfg['operator_flags']['fused_input_lora_a'] is False
    assert cfg['operator_flags']['generator_rollout_merged_shadow'] is False
    assert sum(cfg['operator_flags'].values()) == (6 if b else 0)


def test_only_b_stays_blocked():
    with pytest.raises(ValueError, match='only_B.*BLOCKED'):
        configuration('only_B')


@pytest.mark.parametrize('mutate', [
    lambda c: c['geometry'].__setitem__('video_height', 368),
    lambda c: c['matched_contract'].__setitem__('global_batch', 8),
    lambda c: c['matched_contract'].__setitem__('seed', 42),
    lambda c: c['updates_per_cycle'].__setitem__('critic', 4),
    lambda c: c['cycles'].__setitem__('count', 2),
    lambda c: c['operator_flags'].__setitem__('score_cuda_graph', True),
    lambda c: c['source'].__setitem__('audited_parent', '3' * 40),
    lambda c: c['runtime_policy'].__setitem__('clean_runtime', True),
])
def test_fail_closed_configuration(mutate):
    cfg = configuration()
    mutate(cfg)
    with pytest.raises(ValueError):
        r2.validate_configuration(cfg, expected_head=HEAD)


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def payload(rank, slot):
    return r2.digest({'SYNTHETIC_TEST_ONLY': True, 'rank': rank, 'slot': slot})


def synthetic_gather(observation):
    rows = []
    for rank in range(16):
        row = copy.deepcopy(observation)
        row['rank'] = rank
        census = row['census']
        values = [payload(rank, slot) for slot in range(6)]
        census['sample_payload_identities'] = values
        census['generator_trace_payload_identity'] = values[0]
        census['critic_trace_payload_identities'] = values[1:]
        census['sample_occurrence_identities'] = [f'cycle=1/slot={slot}/payload={v}' for slot, v in enumerate(values)]
        rows.append(row)
    return rows


def cpu_chain(monkeypatch, cell, rank=0, *, gather=synthetic_gather, drift=None):
    """Actual builders, p0 handlers, graph control, cycle and collector; CPU seams.

    Only optional YAML imports, CUDA/collectives, model payload/placement,
    kernel installation/capture and model-specific rollout/Score math are fake.
    No production cycle, p0 update handler or collector is replaced.
    """
    import contextlib
    import types
    import torch
    from distillschedule.runtime import wan13b_rollout_cuda_graph as graph
    from distillschedule.runtime import wan13b_critic_grad_graph as critic_graph
    from distillschedule.runtime import wan13b_norm_policy as norm
    from distillschedule.runtime.fsdp2_static import ema_update_reference

    cfg = configuration(cell)
    torch.manual_seed(r2.MATCHED['seed'] + rank)
    for key, value in r2.lower_environment(cfg).items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('NCCL_ALGO', raising=False)
    monkeypatch.delenv('DISTILLSCHEDULE_WAN13B_COMPILE_CACHE_ROOT', raising=False)
    if cell != 'baseline':
        monkeypatch.setenv('NCCL_ALGO', 'Tree')
    # An unused configuration loader is the only missing optional import.
    monkeypatch.setitem(sys.modules, 'omegaconf', types.SimpleNamespace(OmegaConf=types.SimpleNamespace()))
    worker = load_script('w1_r2_cpu_worker', 'scripts/distillschedule/probe_wan13b_factorial_r2_world16.py')
    legacy = worker.load_legacy()
    legacy._load_native_runtime()
    fair, base, p0 = legacy.fair, legacy.base, legacy.p0
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(torch.distributed, 'barrier', lambda: None)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: rank)
    monkeypatch.setattr(base, '_memory', lambda d: dict(allocated_bytes=0, reserved_bytes=0,
                        max_allocated_bytes=0, max_reserved_bytes=0, driver_free_bytes=1, driver_total_bytes=1))
    monkeypatch.setattr(base, 'trim_host_allocator', lambda: False)
    monkeypatch.setattr(base, 'suspend_transformers_zero3_init', contextlib.nullcontext)
    monkeypatch.setattr(base, 'configure_fsdp2_checkpointing', lambda m: None)
    events, models, norm_receipts = [], {}, {}

    class Model(torch.nn.Module):
        def __init__(self, role):
            super().__init__()
            self.role, self.suppress_hooks = role, False
            self.lora_weight = torch.nn.Parameter(torch.tensor(0.1))
            self.frozen = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.bfloat16), requires_grad=False)
            self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(30)])
            for block in self.blocks:
                block._distillschedule_modulation_chunk_tokens = 512
            self.evidence = {}

        def enable_adapters(self):
            self.adapter_enabled = True

        def disable_adapters(self):
            self.adapter_enabled = False

        def forward(self, hidden_states, **kwargs):
            value = hidden_states.float() * (self.frozen.float() + self.lora_weight)
            if not self.suppress_hooks:
                ep = self.evidence.get('fused', {})
                if ep.get('enabled') and not torch.is_grad_enabled():
                    ep['model_call_count'] += 1
                    ep['fused_invocation_count'] += 305
                    ep['reference_fallback_count'] += 4
                    ep['reference_fallback_reasons'] = {'BASE_DTYPE_torch.float32': ep['reference_fallback_count']}
                grad = self.evidence.get('grad', {})
                if grad.get('enabled') and torch.is_grad_enabled():
                    grad['fused_invocation_count'] += 1
                nr = norm_receipts.get(self.role, {})
                if nr.get('enabled'):
                    nr['fused_call_count'] += 1
                    nr['layernorm_fused_call_count'] += 1
                    if value.requires_grad:
                        value.register_hook(lambda g: (nr.__setitem__('backward_kernel_call_count', nr['backward_kernel_call_count'] + 1), g)[1])
                cast = self.evidence.get('cast', {})
                if cast.get('enabled') and not torch.is_grad_enabled():
                    cast['cache_hit_count'] += 1
                    cast['cache_miss_count'] += 1
                    cast['avoided_destination_write_bytes'] += 4
            return (value,)

    def build_model(**kwargs):
        model = Model(kwargs['role'])
        models[kwargs['role']] = model
        return model, {'block_count': 30, 'chunked_residual_storage': 'cuda_out_of_place',
                       'chunked_residual_runtime': {'requested_storage': 'cuda_out_of_place',
                        'storage_capable': True, 'transformer_source_sha256': r2.HELIOS_SHA256}}

    monkeypatch.setattr(base, 'build_helios_pool_model', build_model)

    def fake_fsdp(model):
        return dict(backend='FSDP2_COMPOSABLE_FULL_SHARD', reshard_after_forward=True,
                    trainable_parameter_dtypes=['torch.float32'], frozen_parameter_dtypes=['torch.bfloat16'],
                    gradient_reduction='SUM_THEN_DIVIDE_WORLD_SIZE')
    monkeypatch.setattr(fair, 'apply_wan13b_fair_fsdp2_policy', fake_fsdp)

    def installer(kind, flag, *, generator_only=False):
        def install(model, *, role, **kwargs):
            requested = r2.lower_environment(cfg)['DISTILLSCHEDULE_WAN13B_' + flag.upper()] == '1'
            enabled = requested and (not generator_only or role == 'generator')
            receipt = dict(requested=requested, enabled=enabled,
                           disabled_reason='GENERATOR_ONLY' if requested and not enabled else 'NOT_REQUESTED',
                           model_call_count=0, fused_invocation_count=0, reference_fallback_count=0,
                           cache_capacity=1, cache_hit_count=0, cache_miss_count=0,
                           avoided_destination_write_bytes=0)
            model.evidence[kind] = receipt
            return receipt
        return install
    monkeypatch.setattr(fair, 'install_generator_nograd_lora_fused_epilogue',
                        installer('fused', 'generator_nograd_lora_fused_epilogue', generator_only=True))
    monkeypatch.setattr(fair, 'install_generator_nograd_lora_cast_cache',
                        installer('cast', 'generator_nograd_lora_cast_cache', generator_only=True))
    monkeypatch.setattr(fair, 'install_grad_lora_fused_epilogue', installer('grad', 'grad_lora_fused_epilogue'))

    def norm_install(*, role, **kwargs):
        enabled = cfg['operator_flags']['modulated_norm_fusion']
        nr = dict(enabled=enabled, requested=enabled, registered_width=1536,
                  fused_call_count=0, layernorm_fused_call_count=0,
                  backward_kernel_call_count=0, reference_fallback_count=0)
        norm_receipts[role] = nr
        return nr
    monkeypatch.setattr(norm, 'install', norm_install)
    monkeypatch.setattr(graph.RolloutCudaGraph, '_audit_source', lambda self: None)

    def cpu_capture(self, args, kwargs):
        sa, sk = graph.clone_tree(args), graph.clone_tree(kwargs)
        for _ in range(3):
            self.source(*sa, **sk)
            self.receipt['warmup_model_call_count'] += 1
        output = self.source(*sa, **sk)
        self.receipt['capture_model_call_count'] += 1
        source = self.source

        class Replay:
            def replay(self):
                source.suppress_hooks = True
                try:
                    output[0].copy_(source(*sa, **sk)[0])
                finally:
                    source.suppress_hooks = False
        return sa, sk, Replay(), output
    monkeypatch.setattr(graph.RolloutCudaGraph, '_capture', cpu_capture)
    monkeypatch.setattr(critic_graph.CriticGradGraph, '_audit_source', lambda self: None)

    def cpu_grad_capture(self, keys, values):
        self.receipt['capture_entries'] += 1
        self.receipt['cold_original_forward_calls'] += 4
        self.receipt['cold_original_backward_calls'] += 4
        return lambda *xs: self.original_forward(**dict(zip(keys, xs)), return_dict=False)[0]
    monkeypatch.setattr(critic_graph.CriticGradGraph, '_capture', cpu_grad_capture)

    class Engine:
        def __init__(self, model, step):
            self.module, self.global_steps = model, step
            self.optimizer = torch.optim.AdamW([model.lora_weight], lr=1e-4)
        def __call__(self, **kwargs):
            return self.module(**kwargs)
        def backward(self, loss):
            loss.backward()
        def step(self):
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_steps += 1

    def build_engine(*, module, physical_step, **kwargs):
        return Engine(module, physical_step), dict(optimizer_implementation='PYTORCH_DEFAULT',
                    gradient_sync='FSDP2_SUM_THEN_DIVIDE_REDUCE_SCATTER' if cell == 'baseline'
                    else 'WORLD16_GRAD_READY_ASYNC_SUM_THEN_DIVIDE_ALL_REDUCE', async_gradient_bucket_mb=None if cell == 'baseline' else 128)
    monkeypatch.setattr(base, 'build_fresh_engine', build_engine)

    class EMA:
        implementation = 'reference'
        def __init__(self, module, *, optimization_step, **kwargs):
            self.parameter, self.optimization_step = module.lora_weight, optimization_step
            self.shadow = self.parameter.detach().clone()
        def step(self):
            ema_update_reference(self.shadow, self.parameter.detach(), 0.9)
            self.optimization_step += 1
    monkeypatch.setattr(base, 'FreshTrainableEMA', EMA)

    parent = load_script('w1_r2_parent_fixture', 'tests/test_refactor_wan13b_two_cycle_timing.py')
    rollout_contract = parent._rollout_contract()

    class Trace:
        rank_control_validation = dict(status='PASS', semantics='official_helios_normal_rank0_broadcast_strict',
            cross_rank_call_count_parity=True, local_exception_entry_count=0, rank_count=16,
            operation_count=6, global_entry_count=96, operations=[
                dict(operation=row['operation'], rank_count=16, synchronized_exit=[v - 1 for v in row['inventory']['stage_call_count'].values()],
                     per_stage_call_counts=list(row['inventory']['stage_call_count'].values()),
                     logical_call_count=row['inventory']['call_count'], physical_call_count=row['inventory']['call_count'])
                for row in rollout_contract])
        def load_generator_entry(self, **kwargs):
            return types.SimpleNamespace(identity_sha256=payload(rank, 0))
        def load_critic_entry(self, *, step, **kwargs):
            return types.SimpleNamespace(pool_entry=types.SimpleNamespace(logical_step=step, identity_sha256=payload(rank, step - 9)))
    trace = Trace()

    def inventory(model, contract):
        stages, total, out = {}, 0, None
        for stage, count in contract['inventory']['stage_call_count'].items():
            stages[stage] = 0
            for _ in range(count):
                shape = cfg['geometry']['latent_spatial_shapes'][stage]
                out = model(hidden_states=torch.ones(1, 1, *shape))[0]
                stages[stage] += 1
                total += 1
        return out, dict(call_count=total, logical_call_count=total,
                         stage_call_count=stages, logical_stage_call_count=dict(stages), dummy_call_count=0)

    def g_rollout(*, engine, entry, clean_runtime, runtime_cache, **kwargs):
        events.append(('G', clean_runtime, runtime_cache is not None))
        # The pinned four-call G rollout has exactly one detached no-grad call.
        shape = cfg['geometry']['latent_spatial_shapes'][0]
        with torch.no_grad():
            engine.module(hidden_states=torch.ones(1, 1, *shape))
        out = engine.module(hidden_states=torch.ones(1, 1, 24, 40))[0]
        out = engine.module(hidden_states=out.detach())[0]
        out = engine.module(hidden_states=out.detach())[0]
        record = dict(rollout_contract[0]['inventory'], trace_identity_sha256=entry.identity_sha256)
        return out, torch.ones_like(out, dtype=torch.bool), record
    monkeypatch.setattr(p0, 'run_generator_update_trace_rollout', g_rollout)
    monkeypatch.setattr(p0, 'prepare_dmd_score_batch', lambda **kw: dict(generated_x0=kw['generated_x0'], encoder_hidden_states=torch.ones(1)))

    def score(model, batch, *, adapter_enabled, **kwargs):
        model.enable_adapters() if adapter_enabled else model.disable_adapters()
        with torch.no_grad():
            return model(hidden_states=batch['generated_x0'].detach())[0]
    monkeypatch.setattr(p0, 'score_forward', score)
    monkeypatch.setattr(p0, 'construct_dmd_gradient', lambda **kw: (torch.ones_like(kw['batch']['generated_x0']), {}))

    def c_rollout(*, model, loaded_entries, clean_runtime, runtime_cache, **kwargs):
        step = loaded_entries[0].pool_entry.logical_step
        events.append(('R', step, clean_runtime, runtime_cache is not None))
        if runtime_cache is not None:
            runtime_cache['SYNTHETIC_TEST_ONLY_schedule'] = True
        with torch.no_grad():
            out, record = inventory(model, rollout_contract[step - 9])
        record.update(member_trace_identities=[loaded_entries[0].pool_entry.identity_sha256], batch_size=1)
        return [out.detach()], record
    monkeypatch.setattr(p0, 'run_generator_trace_rollout_batch', c_rollout)

    def student_batch(*, generated_x0, loaded, **kwargs):
        events.append(('C', loaded.pool_entry.logical_step))
        batch = {key: torch.zeros_like(generated_x0) for key in (
            'hidden_states', 'timestep', 'encoder_hidden_states', 'indices_hidden_states',
            'indices_latents_history_short', 'indices_latents_history_mid', 'indices_latents_history_long',
            'latents_history_short', 'latents_history_mid', 'latents_history_long', 'generated_x0', 'critic_noise')}
        batch['hidden_states'] = torch.ones_like(generated_x0)
        return batch
    monkeypatch.setattr(p0, 'prepare_student_batch', student_batch)
    bootstrap = types.SimpleNamespace(manifest={'ema': {'optimization_step': 10, 'decay': 0.999, 'update_after_step': 0}},
        component=lambda role: dict(seed=20260903, optimizer_step=2 if role == 'generator' else 10),
        restore_trainables=lambda *a, **kw: {'status': 'SYNTHETIC_TEST_ONLY'})
    args = types.SimpleNamespace(run_id='SYNTHETIC_TEST_ONLY', expected_head=HEAD, placement=cfg['placement'],
                                 model_root=Path('/SYNTHETIC_TEST_ONLY'), factorial_configuration=cfg)
    legacy._configure_shared_mode_args(args)
    legacy._validate_arm_environment(args.placement)
    g, c, ema, build = base._build_engines(args=args, bootstrap=bootstrap, device=torch.device('cpu'))
    if drift:
        drift(cfg, build)
    row = worker.run_four_cycles(cfg=cfg, args=args, legacy=legacy, rank=rank, device=torch.device('cpu'),
              generator_engine=g, critic_engine=c, ema=ema, build=build, trace=trace, negative=torch.ones(1),
              barrier=lambda: None, synchronize=lambda: None, gather_c1=gather, synthetic_test_only=True)
    row.update(hostname=f'SYNTHETIC_TEST_ONLY-node{rank // 8}', local_rank=rank % 8)
    assert legacy.p0._run_nogan_generator_update_p0.__module__ == 'wan13b_pinned_p0_worker'
    assert legacy.fair._build_model_fair.__module__ == 'wan13b_two_cycle_fair_worker'
    return cfg, row, events, models


@pytest.mark.parametrize('cell', ['baseline', 'only_A', 'full'])
def test_real_production_chain_four_cycles_cpu_mock(monkeypatch, cell):
    cfg, row, events, models = cpu_chain(monkeypatch, cell)
    assert row['status'] == 'SYNTHETIC_TEST_ONLY'
    assert row['final_versions'] == {'generator': 6, 'critic': 30, 'ema': 30}
    expected_order = ['G'] + ['R'] * 5 + ['C'] * 5
    assert [e[0] for e in events] == expected_order * 4
    b = cell == 'full'
    assert all(e[-2:] == (b, b) for e in events if e[0] in ('G', 'R'))
    assert row['runtime_cache_created'] is b
    assert all(hasattr(m, '_distillschedule_resident_clean_state') is b for m in models.values())
    assert row['c1_binding']['rank_count'] == 16
    assert row['numerical_status'] == 'NOT_RUN'
    assert row['formal_timing_eligible'] is False
    if b:
        assert row['mechanism_receipts']['graph']['generator']['model_call_count'] == 108
        assert row['mechanism_receipts']['critic_graph']['critic']['backward_replay_calls'] == 20
        assert row['mechanism_receipts']['cast']['enabled'] is True


@pytest.fixture(params=['baseline', 'only_A', 'full'])
def observed_world(monkeypatch, request):
    rows = []
    for rank in range(16):
        with monkeypatch.context() as patch:
            cfg, row, _, _ = cpu_chain(patch, request.param, rank)
            rows.append(row)
    return cfg, rows


def test_world16_collector_real_chain_and_synthetic_never_native(observed_world):
    cfg, rows = observed_world
    collector = load_script('w1_r2_cpu_collector', 'scripts/distillschedule/summarize_wan13b_factorial_r2_world16.py')
    # Exercise the actual JSON producer/consumer boundary, including integer keys.
    rows = json.loads(json.dumps(rows))
    summary = collector.summarize_rows(rows, cfg, run_id='SYNTHETIC_TEST_ONLY', allow_synthetic_test_only=True)
    assert summary['errors'] == []
    assert summary['status'] == 'SYNTHETIC_TEST_ONLY'
    assert summary['C2_seconds'] == max(row['cycles'][1]['local_elapsed_seconds'] for row in rows)
    maxima = [max(row['cycles'][i]['local_elapsed_seconds'] for row in rows) for i in (1, 2, 3)]
    assert summary['C2_C4_mean_seconds'] == sum(maxima) / 3
    assert summary['native_data_status'] == 'NOT_RUN'
    assert summary['formal_timing_eligible'] is False
    refused = collector.summarize_rows(rows, cfg, run_id='SYNTHETIC_TEST_ONLY')
    assert refused['status'] == 'FAIL'
    assert refused['C2_seconds'] is None
    assert refused['C2_C4_mean_seconds'] is None


def test_collector_negative_evidence_bindings(monkeypatch):
    # All receipts originate in the real chain, not fabricated collector inputs.
    rows = []
    for rank in range(16):
        with monkeypatch.context() as patch:
            cfg, row, _, _ = cpu_chain(patch, 'only_A', rank)
            rows.append(row)
    collector = load_script('w1_r2_cpu_negative_collector', 'scripts/distillschedule/summarize_wan13b_factorial_r2_world16.py')
    mutations = {
        'wrong_source': lambda x: x[0].__setitem__('expected_head', '3' * 40),
        'wrong_arm': lambda x: x[0].__setitem__('cell', 'full'),
        'wrong_flags': lambda x: x[0]['lowered_runtime_environment'].__setitem__('DISTILLSCHEDULE_WAN13B_SCORE_CUDA_GRAPH', '1'),
        'wrong_hash': lambda x: x[0].__setitem__('configuration_fingerprint', 'f' * 64),
        'wrong_seed': lambda x: x[0]['matched_contract'].__setitem__('seed', 42),
        'wrong_runtime_seed': lambda x: x[0].__setitem__('rank_runtime_seed', 42),
        'wrong_batch': lambda x: x[0]['matched_contract'].__setitem__('global_batch', 8),
        'wrong_updates': lambda x: x[0]['cycle_versions'][3].__setitem__('critic', 29),
        'wrong_census': lambda x: x[0]['cycle_census'][2].__setitem__('critic_score_forwards', 4),
        'missing_rank': lambda x: x.pop(),
        'missing_cycle': lambda x: x[0]['cycles'].pop(),
        'wrong_cycle': lambda x: x[0]['cycles'][3].__setitem__('cycle', 2),
        'wrong_shape': lambda x: x[0]['build_evidence']['generator']['metadata']['modulation_chunk_runtime'].__setitem__('effective_tokens', [512]),
        'cycle_payload_permutation': lambda x: x[0]['cycle_census'][3]['sample_payload_identities'].reverse(),
        'unbound_c1': lambda x: x[0]['c1_binding']['observations'][0].__setitem__('configuration_fingerprint', '0' * 64),
        'unexpected_model_hotpath': lambda x: x[0]['build_evidence']['generator'].__setitem__('resident_clean_model_state_initial', {'enabled': True}),
        'unexpected_cache': lambda x: x[0].__setitem__('runtime_cache_created', True),
        'invalid_timer': lambda x: x[0]['cycles'][2].__setitem__('local_elapsed_seconds', float('nan')),
    }
    for label, mutate in mutations.items():
        altered = copy.deepcopy(rows)
        mutate(altered)
        summary = collector.summarize_rows(altered, cfg, run_id='SYNTHETIC_TEST_ONLY', allow_synthetic_test_only=True)
        assert summary['status'] == 'FAIL', label
        assert summary['C2_seconds'] is None, label
        assert summary['C2_C4_mean_seconds'] is None, label


def test_c1_failure_stops_before_c2(monkeypatch):
    def missing_rank(observation):
        assert observation['versions'] == {'generator': 3, 'critic': 15, 'ema': 15}
        return synthetic_gather(observation)[:-1]
    with pytest.raises(AssertionError, match='every ordered world16 rank'):
        cpu_chain(monkeypatch, 'full', gather=missing_rank)


def test_observed_mechanism_not_config_checkbox(monkeypatch):
    def wrong_reducer(cfg, build):
        build['generator']['engine']['async_gradient_bucket_mb'] = 32
    with pytest.raises(AssertionError, match='async128'):
        cpu_chain(monkeypatch, 'only_A', drift=wrong_reducer)


def test_native_source_and_hash_bound_before_runtime(tmp_path):
    cfg = configuration('full')
    path = tmp_path / 'configuration.json'
    path.write_text(json.dumps(cfg))
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert r2.load_configuration(path, expected_hash=sha, expected_head=HEAD, root=ROOT, verify_git=False) == cfg
    with pytest.raises(ValueError, match='configuration SHA-256'):
        r2.load_configuration(path, expected_hash='0' * 64, expected_head=HEAD)
    with pytest.raises(ValueError, match='configuration/source/arm'):
        r2.load_configuration(path, expected_hash=sha, expected_head='3' * 40)
    cfg['source']['files_sha256'][r2.SOURCE_PATHS[0]] = '0' * 64
    path.write_text(json.dumps(cfg))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='executable/source hash'):
        r2.load_configuration(path, expected_hash=sha, expected_head=HEAD, root=ROOT, verify_git=False)
    # No rewriting a clean-source gate to accept our intentionally uncommitted work.
    cfg = configuration()
    path.write_text(json.dumps(cfg))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='HEAD/clean state'):
        r2.load_configuration(path, expected_hash=sha, expected_head=HEAD, root=ROOT)


def test_old_defaults_and_runtime_flag_refusal(monkeypatch):
    import types
    worker = load_script('w1_r2_defaults_worker', 'scripts/distillschedule/probe_wan13b_factorial_r2_world16.py')
    legacy = worker.load_legacy()
    assert legacy.TIMED_CYCLES == 2
    assert legacy._clean_runtime(r2.RESIDENT) is True
    assert legacy._clean_runtime(r2.FSDP) is False
    assert r2.builder_clean_model_state(types.SimpleNamespace(), legacy_requested=True) is True
    assert r2.builder_clean_model_state(types.SimpleNamespace(), legacy_requested=False) is False
    cfg = configuration('only_A')
    env = r2.lower_environment(cfg)
    env['NCCL_ALGO'] = 'Tree'
    r2.assert_runtime_environment(cfg, env)
    env['DISTILLSCHEDULE_WAN13B_MODULATED_NORM_FUSION'] = '1'
    with pytest.raises(ValueError, match='lowered runtime flag'):
        r2.assert_runtime_environment(cfg, env)


def test_controller_lowers_config_and_keeps_original_guards(tmp_path):
    import hashlib
    import subprocess
    import os
    cfg = configuration('full')
    path = tmp_path / 'full.json'
    path.write_text(json.dumps(cfg))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    script = ROOT / 'scripts/distillschedule/run_wan13b_fair_world16.py'
    common = [sys.executable, str(script), '--run-id', 'r2-cpu-dry-run', '--mode', 'four-cycle-factorial-r2',
              '--placement', cfg['placement'], '--expected-head', HEAD, '--model-root', '/SYNTHETIC_TEST_ONLY',
              '--bootstrap', '/SYNTHETIC_TEST_ONLY', '--bootstrap-hash', r2.MATCHED['bootstrap_hash'],
              '--factorial-config', str(path), '--factorial-config-sha256', sha,
              '--master-addr', '192.0.2.1', '--dry-run']
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(ROOT / 'src')}
    result = subprocess.run(common, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert 'launch_wan13b_fair_world16_node.sh' in result.stdout
    for name, value in r2.lower_environment(cfg).items():
        if name.endswith('EMA_IMPLEMENTATION'):
            continue  # The unchanged node launcher binds reference EMA.
        assert name + '=' + value in result.stdout
    rejected = subprocess.run(common + ['--score-cuda-graph'], capture_output=True, text=True, env=env)
    assert rejected.returncode != 0
    assert 'unregistered CLI mechanism' in rejected.stderr
    subprocess.run(['bash', '-n', str(ROOT / 'scripts/distillschedule/launch_wan13b_fair_world16_node.sh')], check=True)


def test_versioned_launcher_preflight_before_lock_and_training(tmp_path):
    """Synthetic executables: real shell routing, zero CUDA/GPU/SSH invocation."""
    import hashlib
    import os
    import subprocess
    cfg = configuration('only_A')
    config_path = tmp_path / 'only_a.json'
    config_path.write_text(json.dumps(cfg))
    sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    bin_dir = tmp_path / 'synthetic/bin'
    bin_dir.mkdir(parents=True)
    (bin_dir / 'python').write_text(f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
    (bin_dir / 'python').chmod(0o755)
    marker = tmp_path / 'training_was_started'
    for name, content in {
        'git': f'#!/bin/bash\ncase " $* " in *" rev-parse HEAD "*) echo {HEAD};; esac\nexit 0\n',
        'nvidia-smi': '#!/bin/bash\necho SYNTHETIC_TEST_ONLY_FOREIGN_PID\n',
        'torchrun': f'#!/bin/bash\ntouch "{marker}"\nexit 99\n',
    }.items():
        path = bin_dir / name
        path.write_text(content)
        path.chmod(0o755)
    helios = tmp_path / 'helios'
    (helios / 'helios/modules').mkdir(parents=True)
    transformer = helios / 'helios/modules/transformer_helios.py'
    transformer.write_text('# SYNTHETIC_TEST_ONLY\n')
    asset = tmp_path / 'synthetic-assets'
    asset.mkdir()
    run_root = tmp_path / 'run'
    lock = tmp_path / 'synthetic-node.lock'
    env = {**os.environ, **r2.lower_environment(cfg),
           'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'],
           'DMD_SYSTEM_ROOT': str(ROOT), 'HELIOS_RUNTIME_ROOT': str(helios),
           'DISTILLSCHEDULE_PYENV': str(bin_dir.parent),
           'DISTILLSCHEDULE_ASSETS': str(asset), 'DISTILLSCHEDULE_WAN13B_MODEL_ROOT': str(asset),
           'DISTILLSCHEDULE_WAN13B_BOOTSTRAP': str(asset), 'DISTILLSCHEDULE_WAN13B_BOOTSTRAP_HASH': r2.MATCHED['bootstrap_hash'],
           'DISTILLSCHEDULE_TRACE_BUNDLE': str(asset), 'DISTILLSCHEDULE_NEGATIVE_PROMPT_ASSET': str(asset),
           'DISTILLSCHEDULE_EXPECTED_HELIOS_TRANSFORMER_SHA256': hashlib.sha256(transformer.read_bytes()).hexdigest(),
           'DISTILLSCHEDULE_WAN13B_FACTORIAL_CONFIG': str(config_path),
           'DISTILLSCHEDULE_WAN13B_FACTORIAL_CONFIG_SHA256': '0' * 64,
           'DISTILLSCHEDULE_RUN_ROOT': str(run_root), 'DISTILLSCHEDULE_NODE_LOCK': str(lock),
           'PYTHONDONTWRITEBYTECODE': '1'}
    command = ['bash', str(ROOT / 'scripts/distillschedule/launch_wan13b_fair_world16_node.sh'),
               '0', 'SYNTHETIC_TEST_ONLY', 'four-cycle-factorial-r2', HEAD, '192.0.2.1', '29999', cfg['placement']]
    wrong_hash = subprocess.run(command, capture_output=True, text=True, env=env)
    assert wrong_hash.returncode != 0
    assert 'configuration SHA-256 changed' in wrong_hash.stderr
    assert not lock.exists()
    assert not marker.exists()
    env['DISTILLSCHEDULE_WAN13B_FACTORIAL_CONFIG_SHA256'] = sha
    valid_preflight = subprocess.run(command, capture_output=True, text=True, env=env)
    assert valid_preflight.returncode == 81, valid_preflight.stderr
    assert 'gpu_processes_present' in valid_preflight.stderr
    assert lock.exists()  # The unchanged allocation lock follows validated config.
    assert not marker.exists()


def test_complete_full_bundle_requires_each_actual_mechanism(monkeypatch):
    cfg, row, _, _ = cpu_chain(monkeypatch, 'full')
    worker = load_script('w1_r2_full_mechanism_worker', 'scripts/distillschedule/probe_wan13b_factorial_r2_world16.py')
    legacy = worker.load_legacy()
    mutations = [
        ('generator', 'generator_nograd_lora_fused_epilogue', 'enabled', False),
        ('critic', 'grad_lora_fused_epilogue', 'fused_invocation_count', 0),
        ('generator', 'modulated_norm_fusion', 'backward_kernel_call_count', 0),
        ('generator', 'generator_rollout_cuda_graph', 'model_call_count', 107),
        ('critic', 'critic_grad_cuda_graph', 'backward_replay_calls', 19),
        ('generator', 'generator_nograd_lora_cast_cache', 'cache_hit_count', 0),
    ]
    for role, mechanism, field, value in mutations:
        build = copy.deepcopy(row['build_evidence'])
        build[role]['metadata'][mechanism][field] = value
        with pytest.raises(AssertionError):
            r2.validate_mechanisms(legacy, build, cfg, cycles=4)


def test_controller_separates_local_and_remote_config_paths_for_real_transport(tmp_path, monkeypatch):
    """Local config is hashed locally; native worker AND collector read remote copy."""
    import hashlib
    import subprocess
    c = load_script('w1_r2_distinct_config_controller', 'scripts/distillschedule/run_wan13b_fair_world16.py')
    cfg = configuration('baseline')
    local = tmp_path / 'baseline.json'; local.write_text(json.dumps(cfg))
    sha = hashlib.sha256(local.read_bytes()).hexdigest()
    remote = '/share/project/shuchenweng/yongyan_liu/research_artifacts/R2_CPU_TEST/baseline.json'
    args = c.parser().parse_args(['--run-id','CPU_TEST_ONLY','--mode','four-cycle-factorial-r2',
        '--placement','world16-fsdp','--expected-head',HEAD,'--model-root','/CPU_TEST_ONLY',
        '--bootstrap','/CPU_TEST_ONLY','--bootstrap-hash',r2.MATCHED['bootstrap_hash'],
        '--factorial-config',str(local),'--remote-factorial-config',remote,'--factorial-config-sha256',sha,
        '--master-addr','192.0.2.1'])
    command = ' '.join(c._command(args,0,'192.0.2.1'))
    assert 'DISTILLSCHEDULE_WAN13B_FACTORIAL_CONFIG='+remote in command
    assert 'DISTILLSCHEDULE_WAN13B_FACTORIAL_CONFIG='+str(local) not in command
    observed=[]
    monkeypatch.setattr(c.subprocess,'run',lambda command,**kw:(observed.append(command)or type('P',(),{'returncode':0})()))
    monkeypatch.setattr(c.subprocess,'check_output',lambda *a,**kw:b'{"status":"PASS","evidence_kind":"SYNTHETIC_TEST_ONLY"}')
    c._summarize(args,tmp_path/'local_output')
    text=' '.join(observed[0])
    assert '--factorial-config '+remote in text and '--factorial-config '+str(local) not in text
    assert '--factorial-config-sha256 '+sha in text
