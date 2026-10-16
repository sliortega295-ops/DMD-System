"""CPU-only conditional ablation, not a substitute for world16 validation."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillschedule.core.config import load_plan
from distillschedule.runtime import livewan_crossscale_worker as worker
from distillschedule.runtime import livewan_replay_ablation as ablation
from distillschedule.runtime.livewan_score_fa3 import SAC_CONTRACT

ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / 'configs/distillschedule/plans'


@pytest.fixture
def plans(monkeypatch):
    for key in ('LIVEWAN_STUDENT_CHECKPOINT', 'LIVEWAN_CRITIC_CHECKPOINT',
                'LIVEWAN_TEACHER_CHECKPOINT', 'LIVEWAN_OPERATION_TRACE',
                'LIVEWAN_SYSTEMS_BOOTSTRAP', 'DISTILLSCHEDULE_OUTPUT_DIR'):
        monkeypatch.setenv(key, '/unread/' + key)
    return (load_plan(PLANS / ablation.PARENT_FILE).plan.to_dict(),
            load_plan(PLANS / ablation.PLAN_FILE).plan.to_dict())


def test_one_factor_and_full_effective_contract(plans):
    control, candidate = plans
    assert not ablation.enabled(control, mode='pipeline')
    assert ablation.enabled(candidate, mode='pipeline')
    assert candidate['metadata']['critic_attention_checkpoint'] == ablation.RECOMPUTE_CONTRACT
    normalized = ablation.parent_plan(candidate, mode='pipeline')
    assert normalized == control
    worker._validate_plan_identity(control, mode='pipeline')
    worker._validate_plan_identity(candidate, mode='pipeline')
    assert candidate['workload'] == control['workload']
    assert candidate['rank_groups'] == control['rank_groups']
    for lowering in (lambda p: worker._apply_trajectory_contract(p, cycles=2, mode='pipeline'),
                     lambda p: worker._apply_two_cycle_timing_contract(p, mode='pipeline')):
        a, b = copy.deepcopy(control), copy.deepcopy(candidate)
        lowering(a); lowering(b)
        worker._validate_plan_identity(b, mode='pipeline')
        assert ablation.parent_plan(b, mode='pipeline') == a
        assert worker._effective_plan_sha256(a) != worker._effective_plan_sha256(b)


@pytest.mark.parametrize('field,value', [
    ('generator_cross_attention_kv_cache', None), ('generator_rollout_cast_cache', None),
    ('checkpoint_composition', None), ('score_context_prefetch', True),
    ('critic_checkpoint_rng_elision', {'enabled': True}),
    ('critic_ffn_down_checkpoint', {'enabled': True}),
])
def test_no_unregistered_parent_changes(plans, field, value):
    _, candidate = plans
    candidate['metadata'][field] = value
    with pytest.raises(RuntimeError):
        worker._validate_plan_identity(candidate, mode='pipeline')


def test_original_canonical_refusals_are_unchanged(plans):
    control, candidate = plans
    for field in ('critic_attention_checkpoint', 'generator_rollout_cast_cache',
                  'generator_cross_attention_kv_cache'):
        bad = copy.deepcopy(control); bad['metadata'].pop(field, None)
        with pytest.raises(RuntimeError):
            worker._validate_plan_identity(bad, mode='pipeline')
    with pytest.raises(RuntimeError):
        ablation.enabled(candidate, mode='baseline')
    candidate['metadata'].pop('checkpoint_replay_ablation')
    with pytest.raises(RuntimeError):
        worker._validate_plan_identity(candidate, mode='pipeline')


def test_workload_or_parent_drift_rejected(plans):
    _, original = plans
    for change in (lambda p: p['workload']['shape'].__setitem__('batch_size', 8),
                   lambda p: p.__setitem__('id', 'pretend-c51'),
                   lambda p: p['metadata']['checkpoint_replay_ablation'].__setitem__('preserve_rng_state', False)):
        bad = copy.deepcopy(original); change(bad)
        with pytest.raises(RuntimeError):
            ablation.enabled(bad, mode='pipeline')


@pytest.mark.parametrize('check_pinned_census', [False, True])
def test_real_checkpoint_forward_backward_recompute_and_rng(plans, monkeypatch, check_pinned_census, record_property):
    torch = pytest.importorskip('torch')
    import torch.utils.checkpoint as checkpoint
    from distillschedule.runtime import livewan_score_fa3 as fa3
    assert not torch.cuda.is_initialized()
    if check_pinned_census and not torch.__version__.startswith('2.10.'):
        pytest.skip('Production SAC replay census requires the existing pinned Torch 2.10; not inferred from newer CPU API')
    counts = {'calls': 0}
    op_label = 'attention_census' if check_pinned_census else 'attention_mechanism'
    @torch.library.custom_op('livewan_ablation_cpu::' + op_label, mutates_args=())
    def attention(x: torch.Tensor) -> torch.Tensor:
        counts['calls'] += 1
        return x.sin()
    @attention.register_fake
    def _(x):
        return torch.empty_like(x)
    def setup(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])
    def backward(ctx, grad):
        return grad * ctx.saved_tensors[0].cos()
    attention.register_autograd(backward, setup_context=setup)
    op = getattr(torch.ops.livewan_ablation_cpu, op_label).default
    identity = {'interface_sha256': fa3.INTERFACE_SHA256, 'operator': str(op)}
    monkeypatch.setattr(fa3, 'discover_flash_forward', lambda _: (op, identity))
    monkeypatch.setattr(ablation, 'discover_flash_forward', lambda _: (op, identity))
    # CPU fixture substitutes only CUDA metadata/discovery, never policy/checkpoint/autograd.
    metadata = lambda t, args, kw: (((2, 19320, 12, 128), 'fixture'),) * 3
    monkeypatch.setattr(fa3, 'attention_metadata', metadata)
    monkeypatch.setattr(ablation, 'attention_metadata', metadata)
    monkeypatch.setattr(fa3, 'load_flash_attn3', lambda: (SimpleNamespace(), None))
    control, candidate = plans
    results = []
    for plan in (control, candidate):
        torch.manual_seed(2718)
        x = torch.linspace(-1, 1, 48, requires_grad=True)
        proxy = worker._build_critic_fa3_checkpoint_proxy(plan, checkpoint, torch)
        assert isinstance(proxy, ablation.CriticFA3RecomputeProxy) == ablation.enabled(plan, mode='pipeline')
        optimizer = torch.optim.AdamW([x], lr=1e-3)
        before = counts['calls']
        def block(x):
            return (attention(x) * torch.rand_like(x)).square()
        y0 = proxy.checkpoint(block, x, use_reentrant=False)
        y1 = proxy.checkpoint(block, x + 0.5, use_reentrant=False)
        (y0.sum() + y1.sum()).backward()
        optimizer.step()
        assert len(proxy.contexts.checkpoints) == 2
        assert all(len(row['original']) == 1 for row in proxy.contexts.checkpoints)
        results.append((torch.stack((y0.detach(), y1.detach())), x.grad.clone(), torch.get_rng_state().clone(),
                        counts['calls'] - before, proxy.contexts.receipt(), x.detach().clone(),
                        optimizer.state[x]['exp_avg'].clone(), optimizer.state[x]['exp_avg_sq'].clone()))
    a, b = results
    assert all(torch.equal(a[i], b[i]) for i in (0, 1, 2, 5, 6, 7))
    assert (a[3], b[3]) == (2, 4)
    if check_pinned_census:
        assert a[4]['replay_fa_cache_reads'] == 2
        assert b[4]['replay_fa_cache_reads'] == 0
        assert b[4]['replay_fa_recompute_calls'] == 2
    assert b[4]['contract_failures'] == 0
    record_property('torch_version', torch.__version__)
    record_property('control_actual_attention_executions', a[3])
    record_property('candidate_actual_attention_executions', b[3])
    record_property('independent_checkpoint_scopes', 2)
    record_property('forward_gradient_rng_adam_state_equal', True)
    record_property('pinned_210_replay_census_validated', check_pinned_census)
    record_property('cuda_initialized', torch.cuda.is_initialized())
    assert not torch.cuda.is_initialized()


def test_receipt_cannot_claim_recompute_from_configuration():
    r = dict(name=ablation.RECOMPUTE_CONTRACT['name'],
             source={'interface_sha256': ablation.INTERFACE_SHA256, 'operator': ablation.OP_NAME.replace('::', '.') + '.default'},
             execution_status='PASS', production_installed=True, self_attention_only=True,
             contract_failures=0, checkpoint_calls=300, original_fa_calls=300,
             replay_fa_cache_reads=0, replay_fa_recompute_calls=300,
             expected_checkpoint_calls=300, allow_cache_entry_mutation=False,
             cpu_offload=False, rng_preservation='UNCHANGED_CHECKPOINT_DEFAULT_TRUE')
    assert not ablation.runtime_receipt_errors(r, rank=8, expected_calls=300)
    for k, v in [('replay_fa_cache_reads', 300), ('replay_fa_recompute_calls', 0),
                 ('contract_failures', 1), ('checkpoint_calls', 299), ('cpu_offload', True)]:
        assert ablation.runtime_receipt_errors(dict(r, **{k:v}), rank=8, expected_calls=300)


def _script(name):
    spec = importlib.util.spec_from_file_location('ablation_test_' + name.replace('.', '_'), ROOT / 'scripts/distillschedule' / name)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('kind', ['trajectory', 'two-cycle-timing'])
def test_controller_node_route_selects_actual_new_plan(tmp_path, kind):
    import os, subprocess, sys
    args = [sys.executable, str(ROOT/'scripts/distillschedule/run_livewan_crossscale_world16.py'),
            '--hosts', 'node0,node1', '--run-id', 'cpu-c51-ablation', '--run-kind', kind,
            '--execution-mode', 'pipeline', '--pipeline-variant', ablation.VARIANT,
            '--expected-head', 'a'*40, '--master-addr', '127.0.0.1', '--dry-run',
            '--local-evidence-root', str(tmp_path/'evidence')]
    for key in ('remote-root', 'remote-output-root', 'student-checkpoint', 'critic-checkpoint',
                'teacher-checkpoint', 'trace', 'bootstrap', 'source-root', 'source-receipt', 'python-overlay'):
        args += ['--'+key, str(tmp_path/key)]
    if kind == 'trajectory':
        args += ['--trajectory-cycles', '2', '--trajectory-arm-id', 'noreplay', '--trajectory-seed', '17']
    else:
        args += ['--two-cycle-arm-id', 'noreplay', '--two-cycle-seed', '17',
                 '--correctness-receipt', str(tmp_path/'gate.json'), '--correctness-sha256', 'b'*64]
    p = subprocess.run(args, check=True, capture_output=True, text=True, timeout=15)
    assert all(ablation.VARIANT in ' '.join(c) for c in json.loads(p.stdout)['commands'])
    shell = (ROOT/'scripts/distillschedule/launch_livewan_crossscale_world16_node.sh').read_text()
    # Run the actual shell through plan selection, stopping before all executable,
    # asset, Git, lock and GPU work. This exercises route guards, not a string match.
    prefix = shell[:shell.index('for executable in ')] + '\nprintf "CPU_SELECTED_PLAN=%s\\n" "$PLAN"\n'
    env = dict(os.environ, DMD_SYSTEM_ROOT=str(ROOT), LIVEWAN_PIPELINE_VARIANT=ablation.VARIANT,
               LIVEWAN_PYTHON_OVERLAY='/unread/overlay', DISTILLSCHEDULE_OUTPUT_DIR='/unread/output',
               LIVEWAN_WAN21_SOURCE='/unread/source', LIVEWAN_SOURCE_RECEIPT='/unread/receipt',
               LIVEWAN_TRAJECTORY_CYCLES='2', LIVEWAN_TRAJECTORY_SEED='17',
               LIVEWAN_TRAJECTORY_ARM_ID='noreplay', LIVEWAN_TWO_CYCLE_SEED='17',
               LIVEWAN_TWO_CYCLE_ARM_ID='noreplay')
    command = ['bash', '-c', prefix, 'cpu-route', '0', 'cpu-route', kind, 'pipeline', 'a'*40, '127.0.0.1', '29601']
    result = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    assert 'CPU_SELECTED_PLAN=' + str(PLANS/ablation.PLAN_FILE) in result.stdout
    if kind == 'trajectory':
        env['LIVEWAN_TRAJECTORY_CYCLES'] = '1'
        assert subprocess.run(command, env=env, capture_output=True).returncode != 0


def test_conditional_pair_comparator_gate_and_worker_loading(tmp_path, monkeypatch):
    """Synthetic CPU receipts exercise real comparison/binding, not GPU evidence."""
    import hashlib, subprocess, sys
    from distillschedule.evidence.trajectory_coarse import AppendOnlyTrajectoryWriter, TrajectoryCycleReceipt
    fixture_spec = importlib.util.spec_from_file_location('timing_test_fixture', ROOT/'tests/test_refactor_livewan_two_cycle_timing.py')
    fixture = importlib.util.module_from_spec(fixture_spec); fixture_spec.loader.exec_module(fixture)
    summaries = []
    for label, variant in [('control', ablation.PARENT_VARIANT), ('candidate', ablation.VARIANT)]:
        run = tmp_path/label; run.mkdir(); summary = run/'summary.json'
        fixture._trajectory(summary, mode='pipeline', variant=variant, run_id=label)
        value = json.loads(summary.read_text()); value['arm_id'] = label
        for cycle, row in enumerate(value['numeric_state_anchors'], 1):
            p = Path(row['path']); anchor = json.loads(p.read_text()); anchor.pop('anchor_sha256')
            from distillschedule.evidence.trajectory_state_sketch import SCHEMA_VERSION
            anchor['components'] = {component: dict(schema_version=SCHEMA_VERSION,
                coordinate_schema='canonical_tensor_name/global_flat_index/value_fp32',
                inventory_tensor_count=1, inventory_sha256='9'*64, tensor_count=1,
                coordinate_count=1, samples_per_tensor=1, max_tensors=1,
                coordinates=[dict(tensor='lora', flat_index=0, value_fp32=1.0)])
                for component in ('generator_trainable','generator_adam','critic_trainable','critic_adam')}
            digest = hashlib.sha256(json.dumps(anchor, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            p.write_text(json.dumps(dict(anchor, anchor_sha256=digest))); row['sha256'] = digest
        summary.write_text(json.dumps(value)); summaries.append(summary)
        for node in (0, 1):
            p=run/'launcher'/f'node_{node}'/'launch_manifest.txt'; p.parent.mkdir(parents=True)
            p.write_text(f'head={fixture.HEAD}\nrun_kind=trajectory\ntrajectory_cycles=2\nexecution_mode=pipeline\npipeline_variant={variant}\nclaim_boundary=DIAGNOSTIC_ONLY\nrun_id={label}\ntrajectory_arm_id={label}\ntrajectory_seed=17\n')
        for rank in range(16):
            writer=AppendOnlyTrajectoryWriter(run/'trajectory'/f'rank_{rank:03d}.jsonl',run_id=label,rank=rank)
            for cycle in (1, 2):
                samples=tuple(f'{cycle}/sample/{rank}/{i}' for i in range(5))
                writer.append(TrajectoryCycleReceipt(run_id=label,workload='livewan',variant=label,
                    rank=rank,world_size=16,seed=17,cycle=cycle,expected_total_cycles=2,logical_step=(cycle-1)*5,
                    matched_contract={'test_contract':'same'},implementation={'execution_mode':'pipeline'},
                    sample_identities=samples,rng_identities={k:'b'*64 for k in samples},
                    versions={'generator':cycle,'critic':cycle*5,'ema':0},
                    cycle_updates={'generator':1,'critic':5,'ema':0},
                    cumulative_updates={'generator':cycle,'critic':cycle*5,'ema':0},
                    call_census={'application/student':20},losses={'generator_mean':1.,'critic_mean':2.},
                    gradient_norms={'generator_mean':3.,'critic_mean':4.},
                    tensor_contracts={'latent':{'shape':[1,16,21,46,80],'dtype':'torch.bfloat16'}},
                    state_fingerprints={'generator':'g','critic':'c'},finite=True))
    comparison=tmp_path/'comparison.json'; gate=tmp_path/'gate.json'
    compare_cmd=[sys.executable,str(ROOT/'scripts/distillschedule/compare_livewan_two_cycle_trajectories.py'),
                 '--baseline-run-root',str(summaries[0].parent),'--pipeline-run-root',str(summaries[1].parent),
                 '--baseline-summary',str(summaries[0]),'--pipeline-summary',str(summaries[1]),
                 '--expected-head',fixture.HEAD,'--reference-variant',ablation.PARENT_VARIANT,
                 '--pipeline-variant',ablation.VARIANT,'--output',str(comparison)]
    subprocess.run(compare_cmd,check=True,capture_output=True,text=True)
    builder=[sys.executable,str(ROOT/'scripts/distillschedule/build_livewan_two_cycle_timing_gate.py'),
             '--baseline-trajectory-summary',str(summaries[0]),'--pipeline-trajectory-summary',str(summaries[1]),
             '--trajectory-comparison',str(comparison),'--expected-head',fixture.HEAD,
             '--reference-variant',ablation.PARENT_VARIANT,'--pipeline-variant',ablation.VARIANT,'--output',str(gate)]
    subprocess.run(builder,check=True,capture_output=True,text=True)
    value=json.loads(gate.read_text())
    assert value['schema_version']=='distillschedule.livewan_conditional_replay_timing_gate.v1'
    assert value['numerical_review']['extreme_thresholds'] == _script('build_livewan_two_cycle_timing_gate.py').EXTREME_NUMERIC_THRESHOLDS
    digest=hashlib.sha256(gate.read_bytes()).hexdigest()
    arguments=dict(path=gate,expected_sha256=digest,expected_head=fixture.HEAD,mode='pipeline',seed=17,
                   trace=SimpleNamespace(sha256=fixture.IDENTITY['trace_sha256']),
                   bootstrap=SimpleNamespace(sha256=fixture.IDENTITY['bootstrap_sha256']),
                   source_receipt={'source_receipt_sha256':fixture.IDENTITY['source_receipt_sha256']})
    for arm, variant in [('control',ablation.PARENT_VARIANT),('candidate',ablation.VARIANT)]:
        assert worker._load_two_cycle_timing_gate(**arguments,pipeline_variant=variant,arm_id=arm)['status']=='PASS'
    with pytest.raises(RuntimeError):
        worker._load_two_cycle_timing_gate(**arguments,pipeline_variant='c34-continuous-b16-residual-layout',arm_id='control')
    with pytest.raises(RuntimeError):
        worker._load_two_cycle_timing_gate(**dict(arguments,expected_head='b'*40),pipeline_variant=ablation.VARIANT,arm_id='candidate')
    broken=json.loads(comparison.read_text()); broken['state_comparisons'][0]['normalized_l2']=0.6
    comparison.write_text(json.dumps(broken))
    assert subprocess.run(builder,capture_output=True).returncode != 0


def test_direct_worker_cannot_use_unqualified_run_kind(plans, tmp_path):
    _, plan = plans
    path = tmp_path / 'effective.json'
    path.write_text(json.dumps(plan))
    for kind in ('capacity', 'correctness', 'timing', 'profile', 'validate'):
        with pytest.raises(RuntimeError, match='qualification/native two-cycle'):
            worker.run_worker(plan_path=path, mode='pipeline', run_kind=kind, repo_root=ROOT)


def test_metadata_replay_order_audit_remains_fail_closed():
    torch = pytest.importorskip('torch')
    op = torch.ops.aten.sin.default
    def metadata(torch, args, kwargs):
        return args
    contexts = ablation.RecomputeAttentionContexts(torch, op, {}, metadata_reader=metadata)
    forward, replay = contexts()
    # Also exercise the shared ordered ledger independently of Torch's API version.
    policy = forward.policy_fn
    policy(SimpleNamespace(is_recompute=False), op, 'sample-a')
    policy(SimpleNamespace(is_recompute=True), op, 'sample-a')
    with pytest.raises(RuntimeError, match='identity changed'):
        policy(SimpleNamespace(is_recompute=True), op, 'sample-b')
    receipt = contexts.receipt()
    assert receipt['replay_fa_recompute_calls'] == 1
    assert receipt['contract_failures'] == 1
    assert receipt['replay_fa_cache_reads'] == 0


def test_frozen_parent_dependency_digest_cannot_drift(plans, monkeypatch):
    _, candidate = plans
    monkeypatch.setattr(ablation, 'PARENT_PLAN_SOURCES_SHA256', '0' * 64)
    with pytest.raises(RuntimeError, match='frozen parent plan/include source changed'):
        ablation.enabled(candidate, mode='pipeline')
