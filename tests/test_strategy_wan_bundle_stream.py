"""Fresh incumbent lifecycle: CPU protocol/state evidence, not CUDA proof."""
from dataclasses import replace
from pathlib import Path

import pytest

from distillschedule.search.wan_candidate_bundle import INCUMBENT, WanCandidateBundle
from distillschedule.search.wan_implementation import WanImplementation
from distillschedule.strategies.wan_strategy import WanStrategy, strategy_environment
from distillschedule.strategies.wan_task import OriginalWanTask


def stream_bundle():
    return WanCandidateBundle.parse('incumbent21a-stream-v2')


def test_versioned_bundle_keeps_legacy_pins_and_roundtrips_actual_environment():
    bundle = stream_bundle()
    assert bundle.continuous and bundle.enabled and bundle != INCUMBENT
    assert WanCandidateBundle.from_dict(bundle.to_dict()) == bundle
    strategy = WanStrategy('resident', 2048, True, 5, bundle)
    assert WanImplementation.parse(strategy.backend.id) == strategy.backend
    for step in (0, 10, 15, 25):
        strategy.validate_for(replace(OriginalWanTask(), first_step=step))
    with pytest.raises(ValueError, match='exact original'):
        INCUMBENT.validate_task(replace(OriginalWanTask(), first_step=15))
    with pytest.raises(ValueError):
        strategy.validate_for(replace(OriginalWanTask(), lora_rank=64))
    with strategy_environment(strategy):
        from distillschedule.search.wan_implementation import current_implementation
        assert current_implementation() == strategy.backend


def test_new_bootstrap_clock_contract_is_not_an_unchecked_historical_relabel():
    bundle = stream_bundle()
    for g in (0, 2, 7):
        bundle.validate_versions(dict(generator=g, critic=5*g, ema=5*g))
    for clocks in (dict(generator=3, critic=10, ema=10), dict(generator=2, critic=10, ema=0)):
        with pytest.raises(ValueError):
            bundle.validate_versions(clocks)
    with pytest.raises(ValueError, match='G2'):
        INCUMBENT.validate_versions(dict(generator=3, critic=15, ema=15))


def test_new_preset_is_research_only_and_retains_all_incumbent_mechanisms():
    from distillschedule.training.native_presets import describe_preset, preset_implementation
    old = describe_preset('W1', 'wan_ar.incumbent21a.v1')
    new = describe_preset('W1', 'wan_ar.incumbent21a_stream.v2')
    assert new['mechanisms'] == old['mechanisms']
    assert new['continuous_training'] and new['cycles']['max'] == 100
    assert new['gpu_execution'] == 'NOT_RUN' and new['numerical_status'] == 'REVIEW'
    assert old['continuous_training'] is False and old['cycles'] == 2
    assert preset_implementation('W1', new['id']).candidate_bundle == stream_bundle()


def test_rollout_cold_resume_uses_restored_next_version_not_g3(monkeypatch):
    torch = pytest.importorskip('torch')
    from distillschedule.runtime.wan13b_rollout_cuda_graph import RolloutCudaGraph, clone_tree
    monkeypatch.setattr(RolloutCudaGraph, '_audit_source', lambda self: None)

    class Model(torch.nn.Linear):
        def forward(self, hidden_states):
            return super().forward(hidden_states)

    model = Model(2, 2)
    controller = RolloutCudaGraph(model)

    def capture(self, args, kwargs):
        from types import SimpleNamespace
        a, k = clone_tree(args), clone_tree(kwargs)
        out = model(*a, **k).clone()
        self.receipt['warmup_model_call_count'] += 3
        self.receipt['capture_model_call_count'] += 1
        return a, k, SimpleNamespace(replay=lambda: out.copy_(model(*a, **k))), out

    monkeypatch.setattr(RolloutCudaGraph, '_capture', capture)
    controller.bind_stream_cycle(version=8, logical_steps=tuple(range(35, 40)), source='a'*64)
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        controller.begin_rollout(version=8, logical_steps=(35,))
        first = controller(hidden_states=torch.ones(1, 2))
        changed = controller(hidden_states=torch.full((1, 2), 2.))
        assert torch.equal(first, model(hidden_states=torch.ones(1, 2)))
        assert torch.equal(changed, model(hidden_states=torch.full((1, 2), 2.)))
        controller.end_rollout()
    assert torch.equal(rng, torch.get_rng_state())
    controller.bind_stream_cycle(version=9, logical_steps=tuple(range(40, 45)), source='a'*64)
    with torch.no_grad():
        controller.begin_rollout(version=9, logical_steps=(40,))
        with pytest.raises(ValueError, match='signature after'):
            controller(hidden_states=torch.ones(2, 2))
        controller.end_rollout()
    with pytest.raises(ValueError, match='source'):
        controller.bind_stream_cycle(version=10, logical_steps=tuple(range(45, 50)), source='b'*64)


def test_continuation_requires_a_fresh_stream_not_legacy_replay(monkeypatch):
    from distillschedule.strategies.wan_worker import run_worker, OriginalWanAssets
    for k, v in dict(WORLD_SIZE=16, LOCAL_WORLD_SIZE=8, RANK=0, LOCAL_RANK=0).items():
        monkeypatch.setenv(k, str(v))
    assets = OriginalWanAssets('unused', 'unused', 'a'*64, 'unused', 'b'*64, 'unused')
    strategy = WanStrategy('resident', 2048, True, 5, stream_bundle())
    with pytest.raises(ValueError, match='fresh'):
        run_worker(OriginalWanTask(), assets, output='NEVER_CREATED', mode='manual', strategy=strategy)


@pytest.fixture
def native_cpu_bundle(monkeypatch):
    """Original session/dispatcher + real CPU Adam/autograd/cache controllers.

    Only CUDA audits/capture drivers and trace file IO are replaced. The toy
    one-parameter math is NOT trained Wan, CUDA capture or Triton parity proof.
    """
    import random
    from types import SimpleNamespace as NS
    import torch
    import numpy as np
    from distillschedule.runtime.wan13b_rollout_cuda_graph import RolloutCudaGraph, clone_tree
    from distillschedule.runtime.wan13b_critic_grad_graph import CriticGradGraph
    from distillschedule.runtime.wan13b_lora_cast import install_generator_nograd_lora_cast_cache
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
    from distillschedule.strategies.wan_bridge import OriginalWanSession
    from distillschedule.strategies.wan_planning import source_fingerprint
    from distillschedule.strategies.wan_stream import task_at
    from distillschedule.runtime.fsdp2_p0 import expected_trace_rollout_call_inventory
    from distillschedule.search.wan_candidate_bundle import ASSETS
    import distillschedule.strategies.wan_trace as traces

    monkeypatch.setattr(RolloutCudaGraph, '_audit_source', lambda self: None)
    monkeypatch.setattr(CriticGradGraph, '_audit_source', lambda self: None)

    def rollout_capture(self, args, kwargs):
        a, k = clone_tree(args), clone_tree(kwargs)
        output = self.source(*a, **k)[0].clone()
        self.receipt['warmup_model_call_count'] += 3
        self.receipt['capture_model_call_count'] += 1
        return a, k, NS(replay=lambda: output.copy_(self.source(*a, **k)[0])), (output,)

    def critic_capture(self, keys, values):
        self.receipt['capture_entries'] += 1
        self.receipt['cold_original_forward_calls'] += 4
        self.receipt['cold_original_backward_calls'] += 4
        return lambda *xs: self.original_forward(**dict(zip(keys, xs)), return_dict=False)[0]

    monkeypatch.setattr(RolloutCudaGraph, '_capture', rollout_capture)
    monkeypatch.setattr(CriticGradGraph, '_capture', critic_capture)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.tensor(.25))
            self._distillschedule_resident_clean_state = NS(training=True, adapter_enabled=True)

        def _cast_input_dtype(self, value, dtype):
            return value.to(dtype)

        def forward(self, hidden_states, return_dict=False):
            x = self._cast_input_dtype(hidden_states, torch.float32)
            y = self._cast_input_dtype(hidden_states, torch.float32)
            return ((x+y)*self.lora_weight*.5,)

    task = OriginalWanTask(first_step=0)
    bundle = stream_bundle()
    strategy = WanStrategy('resident', 2048, True, 5, bundle)
    stream = dict(stream_sha256='a'*64, identity=dict(source_fingerprint=source_fingerprint()),
        cohorts=[dict(task=task_at(task, i).to_dict(), trace_bundle=str(i), trace_hash=f'{i+1:064x}') for i in range(3)])

    def trace_for(task, path, expected_hash):
        rows = []
        for phase, steps in (('generator', task.source_steps[:1]), ('critic', task.source_steps)):
            for step in steps:
                counts = [1+(step+stage)%2 for stage in range(3)]
                rows.append(dict(operation=f'step/{step}/{phase}/rollout', rank_count=16,
                    synchronized_exit=[x-1 for x in counts], per_stage_call_counts=counts,
                    logical_call_count=sum(counts), physical_call_count=sum(counts)))
        return NS(variant='no-gan', logical_steps=task.source_steps, bundle_hash=expected_hash,
            rank_control_validation=dict(status='PASS', semantics='official_helios_normal_rank0_broadcast_strict',
                cross_rank_call_count_parity=True, local_exception_entry_count=0, rank_count=16,
                operation_count=len(rows), global_entry_count=16*len(rows), operations=rows))

    monkeypatch.setattr(traces, 'load_original_trace', trace_for)

    def build(*, optimized=True, backend_only=False):
        torch.manual_seed(123); random.seed(321); np.random.seed(777)
        selected = strategy if optimized else WanStrategy('resident', 2048, False, 5)
        engines, metadata = [], {}
        for role in ('generator', 'critic'):
            model = Model()
            optimizer = torch.optim.Adam([model.lora_weight], lr=.002)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.9)
            engine = FSDP2EngineAdapter(model, optimizer, scheduler, physical_step=0, global_batch=1)
            engines.append(engine)
            if optimized:
                m = dict(candidate_bundle=bundle.to_dict(), candidate_bundle_sources=bundle.source_identity())
                m['generator_nograd_lora_cast_cache'] = install_generator_nograd_lora_cast_cache(
                    model, role=role, linear_type=Model)
                if role == 'generator':
                    controller = RolloutCudaGraph(model)
                    object.__setattr__(model, '_distillschedule_rollout_cuda_graph', controller)
                    m['generator_rollout_cuda_graph'] = controller.receipt
                else:
                    controller = CriticGradGraph(model, reference_checks=False)
                    object.__setattr__(model, '_distillschedule_critic_grad_cuda_graph', controller)
                    model.forward = controller.forward
                    m['critic_grad_cuda_graph'] = controller.receipt
                metadata[role] = dict(metadata=m)
        g, c = engines
        ema = NS(_names=['lora_weight'], _parameters=[g.module.lora_weight],
            _shadows=[g.module.lora_weight.detach().clone()], optimization_step=0,
            decay=.9, min_decay=0., update_after_step=0, cur_decay_value=None, device='cpu')
        seen = []

        def inventory(trace, phase, step):
            return expected_trace_rollout_call_inventory(trace.rank_control_validation,
                operation=f'step/{step}/{phase}/rollout', expected_rank_count=16)

        def gupdate(**kw):
            step = kw['trace'].logical_steps[0]
            x = torch.full((1,1,12,20), step/100 + 1)
            # Same RNG-consuming update in reference/uninterrupted/resumed.
            target = torch.rand(()) + random.random() + float(np.random.random())
            g.backward((g(hidden_states=x)[0].mean()-target).square()); g.step()
            seen.append(('g', step))
            return dict(physical_score_forwards=3, rollout=inventory(kw['trace'], 'generator', step))

        def rollout(**kw):
            step = kw['steps'][0]
            inv = inventory(kw['trace'], 'critic', step)
            graph = getattr(g.module, '_distillschedule_rollout_cuda_graph', None)
            if graph is not None:
                graph.begin_rollout(version=g.global_steps, logical_steps=(step,))
            try:
                with torch.no_grad():
                    for stage, count in inv['stage_call_count'].items():
                        for _ in range(count):
                            x = torch.full((1,1,*task.stage_spatial_shapes[stage]), step/100+1, dtype=torch.bfloat16)
                            output = (graph if graph is not None else g.module)(hidden_states=x)[0]
                kw['entry_outputs'][step] = NS(tensors={})
                seen.append(('r', step))
                return {step: output}, [dict(inv, batch_size=1)], {}
            finally:
                if graph is not None:
                    graph.end_rollout()

        def critic(**kw):
            step = kw['step']
            x = kw['x0'].mean().detach().expand(1,1,*task.stage_spatial_shapes[step%3]).contiguous()
            c.backward((c(hidden_states=x, return_dict=False)[0].mean()-.5).square()); c.step()
            seen.append(('c', step))
            return dict(logical_step=step, score_forwards=1)

        def estep():
            ema._shadows[0].mul_(.9).add_(g.module.lora_weight.detach(), alpha=.1)
            ema.optimization_step += 1; ema.cur_decay_value = .9
        ema.step = estep
        helpers = NS(_run_nogan_generator_update_p0=gupdate, _precompute_critic_rollouts=rollout,
                     _run_nogan_critic_precomputed=critic)
        fair = NS(base=NS(_build_engines=lambda **k: (g,c,ema,metadata), p0=helpers))
        if backend_only:
            # Let the worker own real OriginalWanSession construction while
            # reusing this fixture's scalar engine/math and capture adapters.
            return NS(fair=fair, g=g, c=c, ema=ema, build=metadata, seen=seen)
        with strategy_environment(selected):
            session = OriginalWanSession(task=task, strategy=selected,
                fair=fair,
                bootstrap=NS(manifest=dict(lora_rank=256)), model_root='CPU_SCALAR_NOT_WAN', device='cpu', rank=0,
                trace=trace_for(task, '0', stream['cohorts'][0]['trace_hash']), negative=None, diagnostics=None,
                asset_identity=ASSETS if optimized else None)
        return session, seen

    return NS(build=build, task=task, stream=stream, strategy=strategy)


def _run_cpu_bundle(f, session, *, start=0, after=None):
    from types import SimpleNamespace as NS
    from distillschedule.strategies.wan_stream import bind_cohort
    from distillschedule.strategies.wan_worker import execute_cycles
    from distillschedule.strategies.wan_evidence import validate_cycle
    order = ('g', *(f'r{i}' for i in range(5)), *(k for i in range(5) for k in (f'c{i}', f'e{i}')))
    return execute_cycles(session, cycles=3, start_cycle=start, mode='run',
        torch=NS(cuda=NS(synchronize=lambda d: None)), distributed=NS(barrier=lambda: None),
        before_cycle=lambda cycle, current: bind_cohort(current, f.task, f.stream, cycle-1, order),
        after_cycle=after, validate_result=lambda raw: validate_cycle(session.task, session.trace, raw))


def test_actual_dispatch_graph_cast_adam_rng_and_cold_restart_match_reference(native_cpu_bundle, tmp_path):
    import torch
    from distillschedule.strategies.wan_stream import restore_stream_cursor
    from distillschedule.runtime.wan_bundle_lifecycle import validate_stream_receipts
    from distillschedule.training.native_state import engine_state
    from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint, pack_state, rng_state
    from test_training_end_to_end import equal
    f = native_cpu_bundle
    identity = dict(stream=f.stream['stream_sha256'], source=f.stream['identity']['source_fingerprint'],
                    task=f.task.to_dict(), strategy=f.strategy.to_dict())
    reference, _ = f.build(optimized=False)
    _run_cpu_bundle(f, reference)
    expected = pack_state(engine_state({'generator': reference.g, 'critic': reference.c}, ema=reference.ema))
    expected_rng = rng_state()
    with strategy_environment(f.strategy):
        session, seen = f.build()
        state = engine_state({'generator': session.g, 'critic': session.c}, ema=session.ema)
        owners = [id(x) for e in (session.g,session.c) for x in (e,e.optimizer,e.lr_scheduler,*e.module.parameters())]

        def save(cycle, current, record):
            current.bundle_lifecycle.validate_checkpoint_boundary()
            if cycle == 1:
                save_checkpoint(tmp_path/'restart', state, identity=identity,
                                cycle=cycle, cursor=cycle, versions=current.versions())

        result = _run_cpu_bundle(f, session, after=save)
        assert owners == [id(x) for e in (session.g,session.c) for x in (e,e.optimizer,e.lr_scheduler,*e.module.parameters())]
        assert equal(expected, pack_state(state)) and equal(expected_rng, rng_state())
        assert [s for role,s in seen if role=='c'] == list(range(15))
        assert [r['cohort']['cohort_index'] for r in result['cycles']] == [0,1,2]
        assert len({r['cohort']['trace_hash'] for r in result['cycles']}) == 3
        receipt = validate_stream_receipts(session.build, cycles=3)
        assert [r['rollout_calls'] for r in receipt['cycles']] == [22,23,22]
        assert session.bundle_lifecycle.rollout.receipt['cache_entry_count'] == 3
        assert session.bundle_lifecycle.critic.receipt['capture_entries'] == 3
        cast = session.g.module._distillschedule_generator_nograd_lora_cast_cache
        assert cast['cache_hit_count'] > 0 and cast['cache_miss_count'] > 0
        # The complete gate's Triton norm/epilogue fields are existing receipt
        # fixtures, NOT CPU evidence of those CUDA kernels executing.
        from copy import deepcopy
        from test_refactor_wan_incumbent_bundle import _executed_receipts
        from distillschedule.runtime.wan_incumbent_evidence import validate_bundle_build
        coupled = _executed_receipts()
        for role in ('generator','critic'):
            coupled[role]['metadata'].update(session.build[role]['metadata'])
        assert validate_bundle_build(stream_bundle(), coupled, executed=True, cycles=3)['actual_compute_observed']
        for role, mechanism, field in (
            ('generator','generator_rollout_cuda_graph','graph_replay_model_call_count'),
            ('critic','critic_grad_cuda_graph','backward_replay_calls'),
            ('generator','generator_nograd_lora_cast_cache','cache_hit_count'),
            ('generator','grad_lora_fused_epilogue','fused_invocation_count')):
            bad = deepcopy(coupled); bad[role]['metadata'][mechanism][field] = 0
            with pytest.raises((ValueError, AssertionError)):
                validate_bundle_build(stream_bundle(), bad, executed=True, cycles=3)
        resumed, resumed_seen = f.build()
        resumed_state = engine_state({'generator': resumed.g, 'critic': resumed.c}, ema=resumed.ema)
        initial_restored = pack_state(resumed_state)
        with pytest.raises(ValueError, match='identity'):
            restore_checkpoint(tmp_path/'restart', resumed_state, identity=dict(identity, source='stale-source'))
        assert equal(initial_restored, pack_state(resumed_state))
        restored = restore_checkpoint(tmp_path/'restart', resumed_state, identity=identity)
        assert not resumed.bundle_lifecycle.rollout._cache and not resumed.bundle_lifecycle.critic.cache
        restore_stream_cursor(resumed, f.task, f.stream, completed_cycles=restored['cycle'])
        _run_cpu_bundle(f, resumed, start=restored['cycle'])
        assert equal(expected, pack_state(resumed_state)) and equal(expected_rng, rng_state())
        assert [s for role,s in resumed_seen if role=='c'] == list(range(5,15))
        validate_stream_receipts(resumed.build, cycles=2)
        assert resumed.bundle_lifecycle.rollout.receipt['stream_lifecycle']['first_version'] == 2
        assert resumed.bundle_lifecycle.critic.receipt['stream_lifecycle']['first_version'] == 5

    # A truly fresh Python process reuses the exact same explicit CPU fixture
    # and original execution helper, not another trainer or serialized Graph.
    import json
    import os
    import subprocess
    import sys
    script = '''
import json, sys, torch
from _pytest.monkeypatch import MonkeyPatch
from test_strategy_wan_bundle_stream import native_cpu_bundle, _run_cpu_bundle
from distillschedule.strategies.wan_strategy import strategy_environment
from distillschedule.strategies.wan_stream import restore_stream_cursor
from distillschedule.training.native_state import engine_state
from distillschedule.training.checkpoint import restore_checkpoint, pack_state, rng_state
from distillschedule.runtime.wan_bundle_lifecycle import validate_stream_receipts
patch = MonkeyPatch()
f = native_cpu_bundle.__wrapped__(patch)
with strategy_environment(f.strategy):
    session, seen = f.build()
    state = engine_state({'generator': session.g, 'critic': session.c}, ema=session.ema)
    saved = restore_checkpoint(sys.argv[1], state, identity=json.loads(sys.argv[3]))
    assert not session.bundle_lifecycle.rollout._cache and not session.bundle_lifecycle.critic.cache
    restore_stream_cursor(session, f.task, f.stream, completed_cycles=saved['cycle'])
    _run_cpu_bundle(f, session, start=saved['cycle'])
    session.bundle_lifecycle.validate_checkpoint_boundary()
    receipt = validate_stream_receipts(session.build, cycles=2)
    torch.save(dict(state=pack_state(state), rng=rng_state(), seen=seen,
                    first_g=session.bundle_lifecycle.rollout.receipt['stream_lifecycle']['first_version'],
                    first_c=session.bundle_lifecycle.critic.receipt['stream_lifecycle']['first_version']), sys.argv[2])
'''
    # pytest adds tests/ to this interpreter's sys.path, not to its children's
    # environment. The fresh-process proof must locate its explicit fixture.
    paths = [str(Path(__file__).resolve().parent)]
    if os.environ.get('PYTHONPATH'):
        paths.append(os.environ['PYTHONPATH'])
    child_env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths))
    completed = subprocess.run([sys.executable, '-c', script, str(tmp_path/'restart'),
        str(tmp_path/'child.pt'), json.dumps(identity)], env=child_env, text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    child = torch.load(tmp_path/'child.pt', weights_only=False)
    assert equal(expected, child['state']) and equal(expected_rng, child['rng'])
    assert [s for role,s in child['seen'] if role=='c'] == list(range(5,15))
    assert child['first_g'] == 2 and child['first_c'] == 5


@pytest.mark.parametrize('pending', ['cycle', 'backward', 'gradient', 'cast', 'communication', 'address', 'source'])
def test_actual_lifecycle_rejects_unsafe_checkpoint_or_next_cohort(native_cpu_bundle, monkeypatch, pending):
    import torch
    from types import SimpleNamespace as NS
    from distillschedule.strategies.wan_stream import bind_cohort
    f = native_cpu_bundle
    with strategy_environment(f.strategy):
        session, _ = f.build()
        lifecycle = session.bundle_lifecycle
        if pending == 'cycle':
            bind_cohort(session, f.task, f.stream, 0, ('g', *(f'r{i}' for i in range(5)), *(f'c{i}' for i in range(5))))
        elif pending == 'backward':
            lifecycle.critic.receipt['replay_calls'] = 1
        elif pending == 'gradient':
            session.g.module.lora_weight.grad = torch.ones_like(session.g.module.lora_weight)
        elif pending == 'cast':
            def active():
                raise ValueError('active cast cache')
            lifecycle.cast_quiescence = active
        elif pending == 'communication':
            session.c._buckets_by_index = [NS(work=NS(is_completed=lambda: False))]
        elif pending == 'address':
            session.g.module.lora_weight = torch.nn.Parameter(session.g.module.lora_weight.clone())
        else:
            import distillschedule.strategies.wan_planning as planning
            monkeypatch.setattr(planning, 'source_fingerprint', lambda: 'stale-source')
            with pytest.raises(ValueError, match='source'):
                bind_cohort(session, f.task, f.stream, 0,
                            ('g', *(f'r{i}' for i in range(5)), *(f'c{i}' for i in range(5))))
            return
        with pytest.raises(ValueError, match='cycle|pending|gradient|cast|communication|addresses'):
            lifecycle.validate_checkpoint_boundary()


def test_controller_rejects_rng_consuming_capture_and_later_critic_shape(native_cpu_bundle, monkeypatch):
    import torch
    from distillschedule.strategies.wan_stream import bind_cohort
    f = native_cpu_bundle
    with strategy_environment(f.strategy):
        session, _ = f.build()
        g, c = session.bundle_lifecycle.rollout, session.bundle_lifecycle.critic
        bind_cohort(session, f.task, f.stream, 0,
                    ('g', *(f'r{i}' for i in range(5)), *(f'c{i}' for i in range(5))))
        capture = g._capture
        def stochastic(*args, **kwargs):
            torch.rand(())
            return capture(*args, **kwargs)
        monkeypatch.setattr(g, '_capture', stochastic)
        g.begin_rollout(version=1, logical_steps=(0,))
        with torch.no_grad(), pytest.raises(ValueError, match='training RNG'):
            g(hidden_states=torch.ones(1,1,12,20))
        g.end_rollout()
        assert not g._cache
        # First cycle may encounter only one of the task's three legal shapes.
        # A genuinely new second-cycle shape must NOT silently fall back or
        # allocate more captures. This is an explicit bounded-support limit.
        for _ in range(5):
            session.c.backward(session.c(hidden_states=torch.ones(1,1,12,20), return_dict=False)[0].mean())
            session.c.step()
        c.bind_stream_cycle(version=5, logical_steps=tuple(range(5,10)), source=session.bundle_lifecycle.source)
        with pytest.raises(ValueError, match='signature after cold'):
            session.c(hidden_states=torch.ones(1,1,24,40), return_dict=False)


def test_v2_assets_require_original_model_negative_and_cold_new_bootstrap():
    from types import SimpleNamespace as NS
    from copy import deepcopy
    from distillschedule.search.wan_candidate_bundle import ASSETS
    bundle = stream_bundle()
    identity = dict(ASSETS, bootstrap_hash='a'*64, trace_bundle_hash='b'*64, trace_binding_hash='c'*64)
    bundle.validate_assets(identity)
    for field in ('model_identity_sha256', 'negative_manifest_hash', 'bootstrap_hash'):
        with pytest.raises(ValueError):
            bundle.validate_assets(dict(identity, **{field: '0'*64}))
    manifest = dict(model_identity_sha256=ASSETS['model_identity_sha256'],
        components={r: dict(optimizer_step=0, scheduler_step=0) for r in ('generator','critic')},
        ema=dict(optimization_step=0, decay=.999, update_after_step=0))
    bootstrap = NS(manifest=manifest, manifest_hash='a'*64)
    bundle.validate_bootstrap(bootstrap, OriginalWanTask(first_step=0))
    changed = deepcopy(bootstrap)
    changed.manifest['components']['generator'].update(optimizer_step=2, scheduler_step=2)
    changed.manifest['components']['critic'].update(optimizer_step=10, scheduler_step=10)
    changed.manifest['ema']['optimization_step'] = 10
    with pytest.raises(ValueError, match='must be cold'):
        bundle.validate_bootstrap(changed, OriginalWanTask())
    changed.manifest_hash = ASSETS['bootstrap_hash']
    bundle.validate_bootstrap(changed, OriginalWanTask())
    with pytest.raises(ValueError, match='first input cohort'):
        bundle.validate_bootstrap(changed, OriginalWanTask(first_step=15))


def test_new_bootstrap_uses_actual_native_payload_loader_and_hash_check(tmp_path):
    import json
    import torch
    from safetensors.torch import save_file
    from distillschedule.runtime.wan13b_upper_bound import Wan13BBootstrap
    from distillschedule.search.wan_candidate_bundle import ASSETS
    from distillschedule.workloads.best_known.registry import fingerprint, file_sha256
    model = torch.nn.Module()
    model.register_parameter('lora_weight', torch.nn.Parameter(torch.zeros(2)))
    payload = tmp_path/'weights.safetensors'
    save_file({'lora_weight': torch.ones(2)}, payload)
    manifest = dict(schema_version='distillschedule.wan13b_bootstrap.v1', block_count=30, lora_rank=256,
        model_identity_sha256=ASSETS['model_identity_sha256'],
        components={r: dict(trainable_payload=payload.name, sha256=file_sha256(payload),
            parameter_names=['lora_weight'], optimizer_step=0, scheduler_step=0) for r in ('generator','critic')},
        ema=dict(optimization_step=0, decay=.999, update_after_step=0))
    manifest['manifest_hash'] = fingerprint(manifest)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    bootstrap = Wan13BBootstrap.load(tmp_path, expected_hash=manifest['manifest_hash'])
    stream_bundle().validate_bootstrap(bootstrap, OriginalWanTask(first_step=0))
    bootstrap.restore_trainables(model, component='generator')
    assert torch.equal(model.lora_weight, torch.ones(2))
    save_file({'lora_weight': torch.full((2,), 2.)}, payload)
    with pytest.raises(ValueError, match='payload hash changed'):
        Wan13BBootstrap.load(tmp_path, expected_hash=manifest['manifest_hash'])


def test_stream_bundle_import_and_description_do_not_import_torch_or_cuda():
    import os
    import subprocess
    import sys
    script = """
import sys
from distillschedule.runtime import wan_bundle_lifecycle
from distillschedule.training.native_presets import describe_preset, preset_implementation
assert describe_preset('W1', 'wan_ar.incumbent21a_stream.v2')['gpu_execution'] == 'NOT_RUN'
assert preset_implementation('W1', 'wan_ar.incumbent21a_stream.v2').candidate_bundle.continuous
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-c', script], env=os.environ, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
