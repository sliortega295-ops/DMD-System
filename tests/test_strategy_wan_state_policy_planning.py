"""SYNTHETIC_TEST_ONLY: policy binding, not GPU timings or qualification."""
from dataclasses import replace
import copy
import json
from types import SimpleNamespace

import pytest

from distillschedule.search.wan_candidate_bundle import INCUMBENT
from distillschedule.strategies.wan_calibration import component
from distillschedule.strategies.wan_planning import plan
from distillschedule.strategies.wan_strategy import WanStrategy, implementations
from distillschedule.strategies.wan_task import OriginalWanTask
from test_strategy_wan_planning import collection


def reference():
    return WanStrategy('resident', 2048, True, candidate_bundle=INCUMBENT)


def retained():
    return replace(reference(), retained_critic_blocks=(26, 27, 28, 29), retention_execution='planner-retain4.v1')


def test_legacy_compute_key_unchanged_and_state_policy_not_a_schedule_window():
    plain, child = reference(), retained()
    assert plain.compute_key == plain.backend.id
    assert child.compute_key != plain.compute_key
    assert replace(child, rollout_window=5).compute_key == child.compute_key
    assert replace(child, rollout_graph_pool_policy='shared-scratch.v1').compute_key != child.compute_key


@pytest.mark.parametrize('action', ['g', 'r0', 'c0', 'e0'])
def test_component_cost_cannot_alias_state_lifetimes_or_graph_scratch(action):
    task = OriginalWanTask()
    work = {action: {'SYNTHETIC_TEST_ONLY': True}}
    a = component(task, action, reference(), cycle=2, work=work)
    b = component(task, action, retained(), cycle=2, work=work)
    assert a.digest != b.digest
    assert 'state_policy' not in a.context
    assert b.context['state_policy'] == retained().compute_policy


@pytest.mark.parametrize('retained_duration, expected_retained', [(30, True), (110, False)])
def test_same_backend_two_state_policies_search_actual_supplied_costs(tmp_path, retained_duration, expected_retained):
    task = OriginalWanTask()
    path = collection(tmp_path, task, [(reference(), 80, 100, 20, 500),
                                      (retained(), retained_duration, 100, 45, 500)])
    result = plan(task, [path])
    assert len(result['ranked']) == 2
    selected = WanStrategy.from_dict(result['selected']['strategy'])
    assert bool(selected.retained_critic_blocks) is expected_retained
    assert result['evidence_scope'] == 'CPU_FIXTURE_ONLY'
    assert not result['claimable_speedup']


def test_memory_bound_rejects_fast_retention_instead_of_always_choosing_it(tmp_path):
    task = OriginalWanTask()
    path = collection(tmp_path, task, [(reference(), 80, 100, 20, 200),
                                      (retained(), 30, 100, 120, 200)])
    result = plan(task, [path])
    assert not WanStrategy.from_dict(result['selected']['strategy']).retained_critic_blocks
    assert len(result['rejected']) == 1


def test_unmeasured_state_policy_cannot_borrow_same_backend_calibration():
    from distillschedule.strategies.wan_planning import calibrated_variant
    variants = [(reference(), {'synthetic': True}, object())]
    with pytest.raises(ValueError, match='compute/state policy'):
        calibrated_variant(retained(), variants)
    with pytest.raises(ValueError, match='compute/state policy'):
        calibrated_variant(reference(), variants * 2)


def test_retention_inventory_opt_in_and_only_original_qualified_block_set():
    task = OriginalWanTask()
    ordinary = implementations(task, triton_available=True, incumbent_bundle_available=True)
    assert not any(s.retained_critic_blocks for s in ordinary)
    expanded = implementations(task, triton_available=True, incumbent_bundle_available=True,
                               critic_retention_available=True)
    extras = [s for s in expanded if s.retained_critic_blocks]
    assert len(extras) == 1 and extras[0].retained_critic_blocks == (26, 27, 28, 29)
    assert extras[0].retention_policy == 'retain4.v1'
    with pytest.raises(ValueError, match='incumbent'):
        implementations(task, triton_available=True, critic_retention_available=True)


@pytest.mark.parametrize('bad', [(29,), tuple(range(24, 30)), ()])
def test_versioned_planner_permission_does_not_open_other_manual_experiments(bad):
    with pytest.raises(ValueError, match='retention'):
        replace(reference(), retained_critic_blocks=bad, retention_execution='planner-retain4.v1')


@pytest.mark.parametrize('tamper', ['retention', 'scratch'])
def test_rehashed_plan_cannot_substitute_unmeasured_state_policy(tmp_path, monkeypatch, tamper):
    import distillschedule.strategies.wan_planning as planning
    from distillschedule.workloads.best_known.registry import fingerprint, write_new_json
    task = OriginalWanTask()
    strategy = reference() if tamper == 'retention' else retained()
    path = collection(tmp_path, task, [(strategy, 80, 100, 20, 500)])
    result = planning.plan(task, [path])
    doc, variants = planning.load_collections(task, [path])
    # Explicit test-only boundary injection reaches the deeper production
    # validator. No fixture is admitted by the unmocked MEASURED/source gate.
    monkeypatch.setattr(planning, 'load_collections', lambda *args: (doc, variants))
    result['evidence_scope'] = 'MEASURED'
    result['test_evidence_kind'] = 'SYNTHETIC_TEST_ONLY_NOT_NATIVE_EVIDENCE'
    selected = WanStrategy.from_dict(result['selected']['strategy'])
    selected = replace(selected, retained_critic_blocks=(26, 27, 28, 29), retention_execution='planner-retain4.v1') if tamper == 'retention' else replace(selected, rollout_graph_pool_policy='shared-scratch.v1')
    result['selected']['strategy'] = selected.to_dict()
    unsigned = dict(result); unsigned.pop('plan_sha256')
    result['plan_sha256'] = fingerprint(unsigned)
    target = tmp_path/'SYNTHETIC_TEST_ONLY_REHASHED_PLAN.json'
    write_new_json(target, result)
    with pytest.raises(ValueError, match='compute/state policy'):
        planning.validate_plan(task, target)


def test_selected_policy_reaches_real_checkpoint_dispatch_and_cpu_backward(tmp_path):
    """Actual original Session -> real wrapper -> checkpoint, on tiny CPU tensors."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from distillschedule.strategies.wan_bridge import OriginalWanSession
    from distillschedule.strategies.wan_strategy import strategy_environment
    from distillschedule.search.wan_candidate_bundle import ASSETS
    task = OriginalWanTask()
    path = collection(tmp_path, task, [(reference(), 80, 100, 20, 500),
                                      (retained(), 40, 100, 45, 500)])
    selected = WanStrategy.from_dict(plan(task, [path])['selected']['strategy'])
    assert selected.retention_execution == 'planner-retain4.v1'

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear = torch.nn.Linear(4, 4); self.calls = 0
        def forward(self, x):
            self.calls += 1
            return self.linear(x).sin()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.blocks = torch.nn.ModuleList(Block() for _ in range(30))
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = lambda block, *a, **k: checkpoint(block, *a, use_reentrant=False, **k)
        def forward(self, x):
            for block in self.blocks: x = self._gradient_checkpointing_func(block, x)
            return x

    with torch.random.fork_rng():
        torch.manual_seed(71)
        g = Model(); control = Model(); child = copy.deepcopy(control)
    build = {role: {'metadata': {}} for role in ('generator', 'critic')}
    fair = SimpleNamespace(base=SimpleNamespace(_build_engines=lambda **kw:
        (SimpleNamespace(module=g, global_steps=2), SimpleNamespace(module=child, global_steps=10),
         SimpleNamespace(optimization_step=10), build)))
    with strategy_environment(selected):
        session = OriginalWanSession(task=task, strategy=selected, fair=fair,
            bootstrap=SimpleNamespace(manifest={'lora_rank':task.lora_rank}), model_root='SYNTHETIC_TEST_ONLY',
            device='cpu', rank=0, trace=SimpleNamespace(variant='no-gan', logical_steps=task.source_steps),
            negative=None, diagnostics=None, asset_identity=ASSETS)
    x = torch.arange(12, dtype=torch.float32).view(3, 4)/8
    before = torch.get_rng_state()
    a = control(x); a.square().sum().backward()
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    b = session.c.module(x); b.square().sum().backward()
    assert torch.equal(a, b) and torch.equal(after, torch.get_rng_state())
    for p, q in zip(control.parameters(), child.parameters()): assert torch.equal(p.grad, q.grad)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-4) for m in (control, child)]
    for opt in optimizers: opt.step()
    for p, q in zip(control.parameters(), child.parameters()): assert torch.equal(p, q)
    assert [b.calls for b in control.blocks] == [2]*30
    assert [b.calls for b in child.blocks] == [2]*26 + [1]*4
    receipt = build['critic']['metadata']['checkpoint_block_retention']
    assert receipt['retained_forward_calls'] == 4 and receipt['checkpoint_forward_calls'] == 26
    assert not build['generator']['metadata']['checkpoint_block_retention']['enabled']
    assert not torch.cuda.is_initialized()


def test_default_manual_only_four_blocks_remain_rejected_before_cuda(monkeypatch, tmp_path):
    from distillschedule.strategies.wan_worker import run_worker, OriginalWanAssets
    for key, value in {'WORLD_SIZE':'16','RANK':'0','LOCAL_RANK':'0','LOCAL_WORLD_SIZE':'8'}.items():
        monkeypatch.setenv(key, value)
    manual = replace(retained(), retention_execution='manual-only.v1')
    assets = OriginalWanAssets('/dummy/model','/dummy/bootstrap','a'*64,'/dummy/trace','b'*64,'/dummy/negative')
    with pytest.raises(ValueError, match='explicit manual candidate'):
        run_worker(OriginalWanTask(), assets, output=tmp_path/'must_not_exist', mode='calibrate', strategy=manual)
    assert not (tmp_path/'must_not_exist').exists()


def test_real_cli_lists_optin_without_launching_and_preserves_default(tmp_path, monkeypatch):
    from distillschedule.strategies import wan_cli
    from distillschedule.workloads.best_known.registry import write_new_json
    monkeypatch.setattr(wan_cli.importlib.metadata, 'version', lambda name: 'SYNTHETIC_CAPABILITY_FIXTURE_ONLY')
    task = tmp_path/'task.json'; write_new_json(task, OriginalWanTask().to_dict())
    ordinary, expanded = tmp_path/'ordinary.json', tmp_path/'expanded.json'
    assert wan_cli.main(['validate','--task',str(task),'--output',str(ordinary)]) == 0
    assert not any(s.get('retained_critic_blocks') for s in json.loads(ordinary.read_text())['implementations'])
    assert wan_cli.main(['validate','--task',str(task),'--output',str(expanded),
                         '--include-incumbent-bundle','--include-critic-retention']) == 0
    doc = json.loads(expanded.read_text())
    assert sum(bool(s.get('retained_critic_blocks')) for s in doc['implementations']) == 1
    assert doc['opt_in_compute_state_policy']['version'] == 'planner-retain4.v1'
    assert doc['gpu_execution'] == 'NOT_RUN' and not doc['claimable_speedup']
