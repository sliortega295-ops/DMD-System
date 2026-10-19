"""SYNTHETIC_TEST_ONLY: real config/control flow/writers/collector, no CUDA/math."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from distillschedule.runtime import livewan_crossscale_worker as w
from distillschedule.runtime import livewan_system_factorial_r2 as r2
from distillschedule.runtime import livewan_system_factorial_collect as collector
from test_livewan_system_factorial_r2 import ROOT, HEAD, env


class CpuCudaEvents:
    """Deterministic endpoint clock only, NOT a performance simulation."""
    def __init__(self):
        self.clock = 0
        self.cuda = self
    def Event(self, **kw):
        owner = self
        class Event:
            def record(self, *args):
                self.value = owner.clock
            def elapsed_time(self, other):
                return other.value - self.value
        return Event()
    def current_stream(self):
        return None


def trace_fixture(tmp_path):
    # Exercise the unchanged real trace producer and parser, not a fabricated
    # already-parsed FormalTrace. Samples remain explicit CPU-only placeholders.
    from unittest.mock import patch
    import hashlib
    source = tmp_path / 'SYNTHETIC_TEST_ONLY.source.json'
    output = tmp_path / 'SYNTHETIC_TEST_ONLY.continuous.json'
    prompts = tmp_path / 'SYNTHETIC_TEST_ONLY.prompts'
    prompts.write_text('SYNTHETIC_TEST_ONLY')
    rows = [dict(logical_step=step, logical_slot=slot, dataset_index=slot,
        sample_path='SYNTHETIC_TEST_ONLY', sample_sha256=None,
        operation_key=f'step={step}/slot={slot}', world_frames=3, rope_gap=0,
        seeds={k: step*16+slot for k in ('rollout', 'generator_dmd_t', 'generator_dmd_noise',
                                      'generator_grad_blocks', 'critic_t', 'critic_noise')},
        generator_update=step == 0, critic_update=True) for step in range(5) for slot in range(16)]
    source.write_text(json.dumps(dict(livewan_commit=w.LIVEWAN_COMMIT, wan21_commit=w.WAN21_COMMIT,
        seed=r2.SEED, world_size=16, global_batch=16, steps=5, source_sample_count=80, rows=rows,
        prompts=dict(path=str(prompts), sha256=hashlib.sha256(prompts.read_bytes()).hexdigest()))))
    spec = spec_from_file_location('r2_cpu_trace_builder', ROOT/'scripts/distillschedule/build_livewan_continuous_trace.py')
    builder = module_from_spec(spec); spec.loader.exec_module(builder)
    with patch('sys.argv', ['CPU_ONLY', '--source-trace', str(source), '--output', str(output), '--cycles', '4']):
        assert builder.main() == 0
    return w.FormalTrace.load(output, expected_cycles=4)


def execute_mock_rank(monkeypatch, *, plan, trace, rank, arm):
    """Execute the original four-cycle scheduling wrapper; stub only leaf math/transport."""
    mode = 'baseline' if arm == 'baseline' else 'pipeline'
    clock = CpuCudaEvents()
    start = clock.Event(enable_timing=True); start.record()
    inv, ev, versions, records = w.CallInventory(), w.RunEvidence(), dict(generator=0, critic=0, ema=0), []
    calls = []
    def count(role, grad, checkpoint, n):
        for _ in range(n):
            inv.add(role, grad, checkpoint)
    def count_math(**kw):
        cycle = kw['step_offset'] // 5
        calls.append(kw)
        slots = (rank,) if mode == 'baseline' else w._pipeline_8x8_slot_pairs(plan)[rank % 8]
        if mode == 'baseline' or rank < 8:
            for step in range(5):
                for slot in slots:
                    row = trace.row(cycle * 5 + step, slot)
                    # Incoming lookahead already executed after the previous endpoint.
                    if mode != 'baseline' and cycle and step == 0 and slot == slots[0]:
                        continue
                    count('student', False, False, 1 + row.new_blocks * 5)
            count('student', True, True, 3)
            versions['generator'] += 1
        if mode == 'baseline':
            count('teacher', False, False, 1); count('critic', False, False, 1)
            count('critic', True, True, 5); versions['critic'] += 5
        elif rank >= 8:
            # Logical Score B1-equivalent counts are independent of physical B2 calls.
            for key, n in [('student/grad/checkpoint', 3), ('teacher/nograd/plain', 2),
                           ('critic/nograd/plain', 2), ('critic/grad/checkpoint', 10)]:
                # Student gradients execute on G, not Score. Only its missing second
                # logical sample is counted here, matching the production B2 census.
                if key.startswith('student'):
                    continue
                ev.add_logical_score_calls(key, n)
            count('teacher', False, False, 2); count('critic', False, False, 2)
            count('critic', True, True, 5)
            versions['critic'] += 5
        clock.clock += 1000 + cycle * 100
    # G: production has 3 grad calls per logical slot => two slots per lane.
    # Its actual G math handles B1 twice, so model invocation stubs must count 6.
    original_math = count_math
    def math_with_g_grad(**kw):
        original_math(**kw)
        if mode == 'pipeline' and rank < 8:
            count('student', True, True, 3)
    monkeypatch.setattr(w, '_run_baseline', math_with_g_grad)
    monkeypatch.setattr(w, '_run_pipeline_8x8', math_with_g_grad)
    if mode == 'baseline':
        r2.run_baseline_cycles(w, torch=clock, rank=rank, trace=trace, loader=None, bundle=None,
            streamers=(), score_context=None, evidence=ev, versions=versions, device=None,
            inventory=inv, cycle_endpoint_records=records)
    else:
        def prepare(torch, **kw):
            row = kw['row']; count('student', False, False, 1 + row.new_blocks * 5)
            return NS(x0='SYNTHETIC_TEST_ONLY')
        monkeypatch.setattr(w, 'prepare_rollout', prepare)
        monkeypatch.setattr(w, '_record_output', lambda *args, **kw: None)
        sync_versions = []
        hot = NS(begin_source=lambda **kw: sync_versions.append(kw['version']), finish=lambda: None,
                 receipt=lambda: dict(execution_status='PASS', events=[dict(version=x) for x in sync_versions]))
        transport = NS(wait_staged=lambda **kw: None, enqueue_send=lambda **kw: None,
                       finish=lambda: None, receipt=lambda: dict(execution_status='PASS', cycles=4,
                                                                  cross_cycle_lookahead=True))
        w._run_pipeline_8x8_continuous(torch=clock, rank=rank, trace=trace, loader=None,
            bundle=NS(student=None), streamers=(None, None), score_context=None,
            run_kind=r2.RUN_KIND, evidence=ev, versions=versions, device=None,
            runahead_host_transport=transport, critic_shadow_hot_sync=hot,
            slot_pairs_by_lane=w._pipeline_8x8_slot_pairs(plan),
            full_context_lens_elision=arm == 'full', critic_step1_split_overlap=False,
            critic_b1_attribution=False, inventory=inv, cycle_endpoint_records=records,
            factorial_timing=True, explicit_full_context=arm in r2.A_ARMS)
        assert all(x['rollout_runahead'] and x['split_dmd_shadow'] and x['critic_b2_only']
                   and not x['finalize_host_transport']
                   and x['explicit_full_context'] == (arm in r2.A_ARMS) for x in calls)
        # The mock hot-sync source exists only in Score; G receives inside the math leaf.
        if rank < 8:
            ev.critic_shadow_hot_sync['events'] = [dict(version=x) for x in (5, 10, 15)]
    timing = r2.finish_timing(torch=clock, start_event=start, records=records, plan=plan,
        mode=mode, rank=rank, evidence_kind='SYNTHETIC_TEST_ONLY')
    assert len(calls) == 4 and [x['step_offset'] for x in calls] == [0, 5, 10, 15]
    return timing, inv, ev


def produce_run(monkeypatch, env, arm):
    trace = trace_fixture(env)
    monkeypatch.setenv('LIVEWAN_OPERATION_TRACE', str(trace.path))
    monkeypatch.setenv('DISTILLSCHEDULE_OUTPUT_DIR', str(env))
    p = r2.make_plan(ROOT, arm=arm, source_head=HEAD)
    w._validate_plan_identity(p, mode='baseline' if arm == 'baseline' else 'pipeline')
    # Exercise real local-rank guard and real atomic plan writer on both nodes.
    for node in (0, 1):
        assert w._write_effective_plan_for_run(plan=p, output_dir=env, run_kind=r2.RUN_KIND,
                                              local_rank=0, node_rank=node)
        before = (env/f'effective_plan.node{node}.json').read_bytes()
        for local_rank in range(1, 8):
            assert not w._write_effective_plan_for_run(plan=p, output_dir=env,
                run_kind=r2.RUN_KIND, local_rank=local_rank, node_rank=node)
        assert before == (env/f'effective_plan.node{node}.json').read_bytes()
    for rank in range(16):
        timing, inv, ev = execute_mock_rank(monkeypatch, plan=p, trace=trace, rank=rank, arm=arm)
        is_g = rank < 8; baseline = arm == 'baseline'
        logical = dict(inv.counts); logical.update(ev.logical_score_calls)
        row = dict(rank=rank, run_id=p['evidence']['run_id'], status='PASS', run_kind=r2.RUN_KIND,
            mode='baseline' if baseline else 'pipeline', head=HEAD,
            trace_sha256=trace.sha256, bootstrap_sha256='3'*64, source_receipt_sha256='4'*64,
            effective_plan_sha256=r2.digest(p), profiled=False,
            system_factorial_timing=timing, final_versions=dict(generator=4, critic=20, ema=0),
            logical_generator_updates=4, logical_critic_updates=20,
            physical_generator_commits=4 if baseline or is_g else 0,
            physical_critic_commits=20 if baseline or not is_g else 0,
            system_factorial_c1_finite=dict(finite=True, count=6 if baseline else 1 if is_g else 5),
            generator_state_metrics=dict(test_only=0.0), critic_state_metrics=dict(test_only=0.0),
            cgroup_oom_before=dict(oom=0, oom_kill=0), cgroup_oom_after=dict(oom=0, oom_kill=0),
            physical_call_inventory=inv.counts, logical_call_inventory=logical,
            continuous_pipeline=ev.continuous_pipeline, critic_shadow_hot_sync=ev.critic_shadow_hot_sync,
            generator_runahead_transport=ev.generator_runahead_transport,
            peak_memory=dict(max_allocated_bytes=0, max_reserved_bytes=0, driver_free_bytes=0))
        if arm in r2.A_ARMS and not is_g:
            calls = inv.counts['critic/grad/checkpoint'] * 30
            row['system_factorial_recompute'] = dict(checkpoint_calls=calls, function_entries=2*calls,
                replay_entries=calls, cache_reads=0, saved_tensor_policy='ORIGINAL', rng_policy='ORIGINAL')
        if arm == r2.CAPACITY_ARM:
            row['build']={'capacity_offload':dict(r2.CAPACITY_POLICY,
                execution_status='PASS' if is_g else 'NOT_APPLICABLE_SCORE_RANK')}
            if is_g:
                n=r2.CAPACITY_POLICY['generator_numel']
                row['build']['student']={'cpu_optimizer':dict(name='replicated_cpu_foreach_adamw',
                    state_device='cpu',execution_parameter_device=f'cuda:{rank%8}',steps=4,
                    moment_bytes=8*n,cpu_parameter_mirror_bytes=4*n,cpu_gradient_buffer_bytes=4*n,
                    pinned_parameter_mirror_bytes=4*n,pinned_gradient_buffer_bytes=4*n,
                    cpu_consumption_fences=4,transfers_and_cpu_update_inside_timer=True,
                    gradient_mean_and_clip_unchanged=True,original_parameter_dtype_preserved=True,
                    gradient_d2h_bytes=16*n,updated_parameter_h2d_bytes=16*n,
                    gradient_d2h_events=3300,updated_parameter_h2d_events=3300)}
            else:row['build']['critic']={'optimizer':{'state_device':'cuda'}}
        w._write_rank_receipt(output_dir=env, rank=rank, value=row)
    return trace


@pytest.mark.parametrize('arm', ['baseline', 'only_A'])
def test_real_cpu_producer_loop_writer_collector_rejects_native_claim(monkeypatch, env, arm):
    trace = produce_run(monkeypatch, env, arm)
    with pytest.raises(ValueError, match='CPU mock'):
        collector.collect(env, arm=arm, expected_head=HEAD, trace=trace)
    s = collector.collect(env, arm=arm, expected_head=HEAD, trace=trace, allow_synthetic=True)
    assert s['status'] == 'PASS_CPU_MOCK_ONLY'
    assert s['cycle_2_seconds'] is None and not s['native_descriptive_timing_eligible']
    assert s['C2_C4_mean_seconds'] is None
    assert s['cpu_mock_endpoint_arithmetic'] == pytest.approx([1, 1.1, 1.2, 1.3])


@pytest.mark.parametrize('change', ['missing_rank', 'missing_cycle', 'hash', 'node_plan', 'head',
                                    'arm', 'version', 'census', 'recompute', 'oom', 'finite'])
def test_real_chain_rejects_invalid_observations(monkeypatch, env, change):
    trace = produce_run(monkeypatch, env, 'only_A')
    path = env/'receipts/rank_008.json'; row = json.loads(path.read_text())
    if change == 'missing_rank': path.unlink()
    elif change == 'missing_cycle': row['system_factorial_timing']['cycles'].pop()
    elif change == 'hash': row['effective_plan_sha256'] = '0'*64
    elif change == 'head': row['head'] = 'b'*40
    elif change == 'arm': row['system_factorial_timing']['arm'] = 'full'
    elif change == 'version': row['system_factorial_timing']['cycles'][1]['local_versions']['critic'] = 11
    elif change == 'census': row['system_factorial_timing']['cycles'][1]['observed_call_deltas']['physical_call_inventory']['critic/grad/checkpoint'] -= 1
    elif change == 'recompute': row['system_factorial_recompute']['cache_reads'] = 1
    elif change == 'oom': row['cgroup_oom_after']['oom'] = 1
    elif change == 'finite': row['system_factorial_c1_finite']['finite'] = False
    elif change == 'node_plan':
        p = json.loads((env/'effective_plan.node1.json').read_text()); p.pop('effective_plan_sha256')
        p['metadata'][r2.KEY]['arm'] = 'full'
        w.atomic_json(env/'effective_plan.node1.json', dict(p, effective_plan_sha256=r2.digest(p)))
    if change != 'missing_rank': w.atomic_json(path, row)
    try:
        s = collector.collect(env, arm='only_A', expected_head=HEAD, trace=trace, allow_synthetic=True)
    except (ValueError, FileNotFoundError):
        pass
    else:
        assert s['status'] == 'FAIL'


@pytest.mark.parametrize('rank', [0, 8])
def test_complete_full_control_flow_uses_canonical_B_and_depth1(monkeypatch, env, rank):
    p = r2.make_plan(ROOT, arm='full', source_head=HEAD)
    timing, _, ev = execute_mock_rank(monkeypatch, plan=p, trace=trace_fixture(env), rank=rank, arm='full')
    assert len(timing['cycles']) == 4 and ev.continuous_pipeline['lookahead_depth'] == 1
    assert w._critic_attention_checkpoint_enabled(p, mode='pipeline')
    assert w._generator_rollout_cast_cache_enabled(p, mode='pipeline')


def test_real_worker_calls_writer_before_models_and_timer():
    source = inspect.getsource(w.run_worker)
    assert source.index('_write_effective_plan_for_run(') < source.index('build_models(')
    assert source.index('_write_effective_plan_for_run(') < source.index('timing_start_ns =')


def controller():
    spec = spec_from_file_location('r2_native_controller_cpu_test', ROOT/'scripts/distillschedule/run_livewan_crossscale_world16.py')
    module = module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def controller_args(arm):
    args = ['--hosts', 'CPU_NODE_0,CPU_NODE_1', '--run-id', 'SYNTHETIC_TEST_ONLY',
            '--run-kind', r2.RUN_KIND, '--system-factorial-arm', arm, '--expected-head', HEAD,
            '--execution-mode', 'baseline' if arm == 'baseline' else 'pipeline',
            '--pipeline-variant', 'b1' if arm == 'baseline' else 'c51-continuous-b16-generator-fused-rms',
            '--baseline-placement', 'world16-fsdp' if arm == 'baseline' else 'legacy-hybrid',
            '--student-rollout-padding', 'max-noop' if arm == 'baseline' else 'off',
            '--master-addr', '192.0.2.1', '--dry-run']
    for flag in ['remote-root', 'remote-output-root', 'student-checkpoint', 'critic-checkpoint',
                 'teacher-checkpoint', 'trace', 'bootstrap', 'source-root', 'source-receipt', 'python-overlay']:
        args += ['--'+flag, '/SYNTHETIC_TEST_ONLY/'+flag]
    return args


@pytest.mark.parametrize('arm', ['baseline', 'only_A', 'full'])
def test_production_controller_node_and_collector_bindings_without_ssh(monkeypatch, env, arm):
    import subprocess
    import shlex
    c = controller(); args = c.parser().parse_args(controller_args(arm))
    # Production command lowering only. Any attempted subprocess/SSH fails this CPU test.
    monkeypatch.setattr(c.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected external process'))
    for node in (0, 1):
        text = ' '.join(c.command(args, node, '192.0.2.1'))
        assert f'LIVEWAN_SYSTEM_FACTORIAL_ARM={arm}' in text
        assert 'LIVEWAN_SYSTEM_FACTORIAL_SEED=20260830' in text
        assert 'system-factorial-timing' in text
    # Execute the REAL extracted node-header arm branch; not a hand-reimplementation.
    source = (ROOT/'scripts/distillschedule/launch_livewan_crossscale_world16_node.sh').read_text()
    start = source.index('if [[ "${RUN_KIND}" == system-factorial-timing ]]')
    branch = source[start:source.index('\nfi', start)+3]
    exports = dict(RUN_KIND=r2.RUN_KIND, LIVEWAN_SYSTEM_FACTORIAL_SEED='20260830',
        LIVEWAN_SYSTEM_FACTORIAL_ARM=arm, PROFILE_TWO_CYCLES='0', EXECUTION_MODE=args.execution_mode,
        PIPELINE_VARIANT=args.pipeline_variant, LIVEWAN_BASELINE_PLACEMENT=args.baseline_placement,
        LIVEWAN_STUDENT_ROLLOUT_PADDING=args.student_rollout_padding)
    import os
    # Restore process function only for bash's non-GPU, non-network conditional.
    monkeypatch.undo()
    result = subprocess.run(['bash', '-c', 'set -eu;\n'+branch], env=dict(os.environ, **exports), capture_output=True)
    assert result.returncode == 0, result.stderr
    exports['LIVEWAN_SYSTEM_FACTORIAL_ARM'] = 'only_B'
    result = subprocess.run(['bash', '-c', 'set -eu;\n'+branch], env=dict(os.environ, **exports), capture_output=True)
    assert result.returncode == 2 and b'BLOCKED' in result.stderr


def test_collector_full_routes_actual_counters_through_original_validators(monkeypatch):
    import importlib
    received = []
    def probe(*args, **kwargs):
        received.append(kwargs)
        return []
    validators = {
        'livewan_score_fa3': ['backend_receipt_errors', 'runtime_receipt_errors'],
        'livewan_score_rmsnorm': ['runtime_receipt_errors'],
        'livewan_score_gate': ['runtime_receipt_errors'],
        'livewan_score_layernorm': ['runtime_receipt_errors'],
        'livewan_rollout_cast_cache': ['runtime_receipt_errors'],
        'livewan_generator_rms_epilogue': ['runtime_receipt_errors'],
        'livewan_generator_modulation': ['runtime_receipt_errors'],
        'livewan_generator_gate': ['runtime_receipt_errors'],
        'livewan_generator_splitkv': ['runtime_receipt_errors'],
    }
    for name, attrs in validators.items():
        m = importlib.import_module('distillschedule.runtime.'+name)
        for attr in attrs:
            monkeypatch.setattr(m, attr, probe)
    collector.mechanism_errors(dict(physical_call_inventory={'critic/grad/checkpoint': 20},
        generator_attention_backend=dict(invocation_count=7, readonly_splitkv={})), arm='full', rank=8)
    assert len(received) == 10
    assert any(x.get('expected_calls') == 600 for x in received)
    assert sum(x.get('cycles') == 4 for x in received) == 4
    assert any(x.get('total_calls') == 7 for x in received)


def test_collector_fresh_process_gets_producer_render_inputs_not_inherited_env(monkeypatch, env):
    c = controller(); args = c.parser().parse_args(controller_args('full'))
    commands = []
    monkeypatch.setattr(c.subprocess, 'run', lambda command, **kw: (commands.append(command) or NS(returncode=0)))
    # This result is ONLY the controller transport seam, not a native collector PASS.
    monkeypatch.setattr(c, 'read_remote_file', lambda *a, **kw: b'{"status":"PASS","evidence_kind":"SYNTHETIC_TEST_ONLY"}')
    c.summarize(args, env/'local')
    text = ' '.join(commands[0])
    for key, value in dict(DISTILLSCHEDULE_RUN_ID=args.run_id,
        DISTILLSCHEDULE_OUTPUT_DIR=args.remote_output_root/args.run_id,
        DISTILLSCHEDULE_PYTHON=args.pyenv/'bin/python', DMD_SYSTEM_ROOT=args.remote_root,
        LIVEWAN_STUDENT_CHECKPOINT=args.student_checkpoint, LIVEWAN_CRITIC_CHECKPOINT=args.critic_checkpoint,
        LIVEWAN_TEACHER_CHECKPOINT=args.teacher_checkpoint, LIVEWAN_OPERATION_TRACE=args.trace,
        LIVEWAN_SYSTEMS_BOOTSTRAP=args.bootstrap).items():
        assert f'{key}={value}' in text
    assert 'summarize_livewan_system_factorial_r2.py' in text and '--arm full' in text
