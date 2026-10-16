"""CPU fixtures for wiring/negative gates; NOT CUDA validation or measured cost."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillschedule.search.h3_implementation import H3, H3Implementation, implementations
from distillschedule.search.h3_request import H3Request
from distillschedule.search.contracts import semantic_contract
from distillschedule.search.cost_model import CostModel
from distillschedule.search.execution import NativeTrialRunner, resolve_candidate, validated_profile
from distillschedule.search.h3_native import summarize_native_two_cycle, compare_native_two_cycle
from distillschedule.search.model_adapters import SUPPORTED_WORKLOADS
from distillschedule.search.model_search import search_plans
from distillschedule.search.native_macro_adapter import lower_prediction
from distillschedule.search.workflow_cli import candidate_inventory, main
from distillschedule.workloads.best_known import registry
from test_refactor_h3_search_native import HEAD, _receipts
from test_refactor_h3_planner_cycle_observer import FakeCuda, trainer_contract

LOW = H3Request(480, 832)


@pytest.mark.parametrize("height,width,frames", [(768, 1344, 124), (480, 832, 124),
                                               (768, 1344, 56), (480, 832, 56)])
@pytest.mark.parametrize("policy", ["cpu_staged_trim", "cuda_no_trim"])
def test_request_cache_roundtrip_lowering_and_default_preservation(height, width, frames, policy):
    task = H3Request(height, width, frames)
    canonical = registry.load_profile(H3)
    before = deepcopy(canonical)
    assert H3Request.parse(task.id) == H3Request.from_dict(task.to_dict()) == task
    assert H3Request().profile() == canonical
    impl = H3Implementation(cache_policy=policy, request=task)
    assert H3Implementation.parse(impl.id) == impl
    assert H3Implementation.from_assignment(impl.assignment) == impl
    profile = impl.lower_profile()
    assert registry.load_profile(H3) == before
    assert profile['contract']['resolution'] == [task.height, task.width]
    assert profile['contract']['frames'] == task.frames
    assert profile['runtime']['model_config']['training']['dmd']['height'] == task.height
    assert profile['runtime']['model_config']['training']['dmd']['width'] == task.width
    assert profile['runtime']['model_config']['training']['dmd']['num_frames'] == task.frames
    assert semantic_contract(profile) == semantic_contract(task.profile())
    assert profile['semantics'] == 'approximate_grid1000'
    assert profile['hardware'] == canonical['hardware']
    runtime = profile['runtime']
    assert runtime['model_config']['training']['a100']['fa3_replay_cache']['storage'] == impl.storage
    assert runtime['environment']['H3_FA3_REPLAY_CACHE_TRIM_BEFORE_BACKWARD'] == str(int(impl.trim_before_backward))
    impl.receipt_binding(runtime['model_config'], runtime['environment'])
    if task != H3Request():
        assert profile['contract']['request'] == task.to_dict()
        assert 'seconds' not in profile['historical']
        assert semantic_contract(profile) != semantic_contract(canonical)
    assert H3Implementation().id == 'joint-h3v1-d2-n128'
    assert H3Implementation(2, 128, 3).id == 'joint-h3v2-d2-n128-w3'


@pytest.mark.parametrize("change", [dict(height=481), dict(width=1344), dict(frames=125),
                                     dict(frames=60), dict(frames=90), dict(frames=True), dict(frames=56.0),
                                     dict(height=True), dict(schema_version='unknown'), dict(fake_update_ratio=4),
                                     dict(world_size=8)])
def test_request_unknown_semantics_fail_closed(change):
    with pytest.raises(ValueError):
        H3Request.from_dict({**LOW.to_dict(), **change})


def test_inactive_cuda_prefetch_is_not_an_invented_candidate():
    values = implementations(request=LOW, explicit_cache=True)
    assert len(values) == 6
    assert all(i.prefetch_depth == 2 for i in values if i.storage == 'cuda')
    with pytest.raises(ValueError, match='inactive depth2'):
        H3Implementation(1, cache_policy='cuda_no_trim', request=LOW)
    with pytest.raises(ValueError, match='cache policy'):
        H3Implementation(cache_policy='cuda_with_trim', request=LOW)
    with pytest.raises(ValueError, match='explicit v3'):
        H3Implementation(request=LOW)


@pytest.mark.parametrize('frames', [124, 56])
@pytest.mark.parametrize('change', ['environment', 'shape', 'frames', 'world_size', 'config_storage', 'config_trim'])
def test_selected_fields_cannot_be_ignored_by_actual_config(change, frames):
    impl = H3Implementation(cache_policy='cuda_no_trim', request=replace(LOW, frames=frames))
    runtime = impl.lower_profile()['runtime']
    config, env = deepcopy(runtime['model_config']), dict(runtime['environment'])
    cache = config['training']['a100']['fa3_replay_cache']
    if change == 'environment': env['H3_FA3_REPLAY_CACHE_STORAGE'] = 'cpu_staged'
    if change == 'shape': config['training']['dmd']['width'] = 1344
    if change == 'frames': config['training']['dmd']['num_frames'] = 124 if frames == 56 else 56
    if change == 'world_size':
        config['distributed']['fsdp2']['size'] = 8
        config['training']['a100']['matched_compute']['expected_world_size'] = 8
    if change == 'config_storage': cache['storage'] = 'cpu_staged'
    if change == 'config_trim': cache['trim_before_backward'] = True
    with pytest.raises(ValueError):
        impl.receipt_binding(config, env)


def _cuda_observer_case(monkeypatch, *, run_kind='correctness'):
    from distillschedule.backends.training.minimax_h3.planner_cycle_observer import H3PlannerCycleObserver
    impl = H3Implementation(cache_policy='cuda_no_trim', request=LOW)
    runtime = impl.lower_profile()['runtime']
    # The real dispatcher supplies the run kind separately from profile env.
    # A CPU fixture must not inherit another workload's mode/action controls.
    monkeypatch.setenv('DISTILLSCHEDULE_RUN_KIND', run_kind)
    monkeypatch.delenv('H3_PLANNER_ROLLOUT_WINDOW', raising=False)
    monkeypatch.delenv('H3_PLANNER_EXECUTION_MODE', raising=False)
    for k, v in runtime['environment'].items(): monkeypatch.setenv(k, v)
    trainer = trainer_contract(config=runtime['model_config'], fa3_replay_cache_storage='cuda',
                               fa3_replay_cache_trim_before_backward=False)
    observer = H3PlannerCycleObserver(torch=SimpleNamespace(cuda=FakeCuda()), rank=0, source_head=HEAD)
    return observer, trainer


@pytest.mark.parametrize('inherited_mode', [None, '', 'native', 'two-cycle-timing',
                                           'calibration', 'diagnostic-timing', 'correctness'])
def test_actual_native_registration_is_checked_not_only_environment(monkeypatch, inherited_mode):
    if inherited_mode is None:
        monkeypatch.delenv('DISTILLSCHEDULE_RUN_KIND', raising=False)
    else:
        monkeypatch.setenv('DISTILLSCHEDULE_RUN_KIND', inherited_mode)
    monkeypatch.setenv('H3_PLANNER_ROLLOUT_WINDOW', '3')
    monkeypatch.setenv('H3_PLANNER_EXECUTION_MODE', 'old-experiment-mode')
    observer, trainer = _cuda_observer_case(monkeypatch)
    assert observer.run_kind == 'correctness'
    observer.validate_start(trainer, current_iter=0, resume_ckpt_path=None)
    trainer.fa3_replay_cache_registration = SimpleNamespace(enabled=True, storage='cpu_staged',
        block_indices=tuple(range(50)), max_d2h_inflight=2, trim_before_backward=True)
    with pytest.raises(ValueError, match='registration changed'): observer.after_setup(trainer)
    trainer.fa3_replay_cache_registration.storage = 'cuda'
    trainer.fa3_replay_cache_registration.trim_before_backward = False
    trainer.grid_replay_registration = SimpleNamespace(stats={})
    observer.after_setup(trainer)


@pytest.mark.parametrize('mode', ['', 'native', 'two-cycle-timing', 'old-experiment-mode'])
def test_real_observer_still_rejects_unsupported_run_modes(monkeypatch, mode):
    observer, trainer = _cuda_observer_case(monkeypatch, run_kind=mode)
    with pytest.raises(ValueError, match='unsupported H3 native mode'):
        observer.validate_start(trainer, current_iter=0, resume_ckpt_path=None)
    assert not observer.torch.cuda.events, 'invalid mode must reject before setup'


@pytest.mark.parametrize('change', ['missing', 'storage', 'trim', 'depth', 'blocks', 'cpu_transfer', 'unused', 'rank'])
def test_receipt_selected_cache_execution_and_all_ranks_fail_closed(change):
    impl = H3Implementation(cache_policy='cuda_no_trim', request=LOW)
    rows = _receipts(impl)
    assert summarize_native_two_cycle(rows, expected_head=HEAD, candidate_id=impl.id)['status'] == 'PASS'
    row = rows[-1]['cycles'][-1]
    actual = row['fa3_replay_cache_registration']
    if change == 'missing': row.pop('fa3_replay_cache_registration')
    if change == 'storage': actual['storage'] = 'cpu_staged'
    if change == 'trim': actual['trim_before_backward'] = True
    if change == 'depth': actual['max_d2h_inflight'] = 1
    if change == 'blocks': actual['block_indices'].pop()
    if change == 'cpu_transfer': actual['stats']['cpu_d2h_bytes'] = 1
    if change == 'unused': actual['stats']['compact_forward_impl_calls'] = 0
    if change == 'rank': rows.pop()
    with pytest.raises(ValueError): summarize_native_two_cycle(rows, expected_head=HEAD, candidate_id=impl.id)


def test_shape_and_storage_costs_and_comparisons_are_not_reused():
    from distillschedule.search.native_macro_adapter import native_graph, cost_key
    from distillschedule.search.h3_action_costs import action_graph, component_contract
    cpu = H3Implementation(cache_policy='cpu_staged_trim', request=LOW)
    gpu = replace(cpu, cache_policy='cuda_no_trim')
    large = replace(cpu, request=H3Request())
    short = replace(cpu, request=replace(LOW, frames=56))
    short_large = replace(large, request=H3Request(768, 1344, 56))
    values = (cpu, gpu, large, short, short_large)
    assert len({i.id for i in values}) == 5
    assert len({tuple(sorted(i.assignment.items())) for i in values}) == 5
    op = native_graph(H3).operations[0]
    assert len({cost_key(H3, op, i) for i in values}) == 5
    op = action_graph().operations[0]
    work = dict(physical_batch=1, logical_batch=16, fixture='not measured')
    assert len({component_contract(op, replace(i, rollout_window=3), work).digest for i in values}) == 5
    with pytest.raises(ValueError, match='requests/shapes'):
        compare_native_two_cycle(_receipts(cpu), _receipts(large), expected_head=HEAD)
    with pytest.raises(ValueError, match='requests/shapes'):
        compare_native_two_cycle(_receipts(cpu), _receipts(short), expected_head=HEAD)


@pytest.mark.parametrize('workload', sorted(SUPPORTED_WORKLOADS))
def test_uniform_inventory_resolves_existing_family_bindings(workload):
    from distillschedule.search.registry import get_candidate
    result = candidate_inventory(workload)
    assert result['gpu_execution'] == 'NOT_RUN' and result['claimable_speedup'] is False
    assert result['candidates']
    for row in result['candidates']:
        assert get_candidate(workload, row['id']).workload == workload
    if workload != H3:
        with pytest.raises(ValueError, match='only for H3'): candidate_inventory(workload, request=LOW)


def test_unified_calibrate_default_is_source_only_and_preserves_request(tmp_path, monkeypatch):
    monkeypatch.setattr(NativeTrialRunner, '_rpc', lambda *a, **k: pytest.fail('no remote/native work allowed'))
    cpu = H3Implementation(cache_policy='cpu_staged_trim', request=LOW)
    gpu = replace(cpu, cache_policy='cuda_no_trim')
    output = tmp_path / 'intent'
    assert main(['calibrate', '--workload', H3, '--candidate', cpu.id, gpu.id,
        '--expected-head', registry.git_head(registry.repository_root()), '--run-id', 'cpu-only',
        '--output', str(output)]) == 0
    result = json.loads((output/'intent.json').read_text())
    assert result['gpu_execution'] == 'NOT_RUN'
    assert result['source_only_plans'][1]['fully_resolved_profile']['contract']['resolution'] == [480,832]
    assert main(['calibrate', '--workload', H3, '--candidate', cpu.id, replace(gpu,request=H3Request()).id,
        '--expected-head', registry.git_head(registry.repository_root()), '--run-id', 'mismatch',
        '--output', str(tmp_path/'invalid')]) == 2
    assert not (tmp_path/'invalid').exists()


def _joined_calibration(tmp_path, monkeypatch):
    import test_refactor_native_macro_calibration as fixtures
    from test_refactor_native_macro_calibration import _fixture, _assemble
    from distillschedule.search import native_macro_adapter as adapter
    from distillschedule.search.native_macro_calibration import assemble_calibrations
    cpu = H3Implementation(cache_policy='cpu_staged_trim', request=LOW)
    gpu = replace(cpu, cache_policy='cuda_no_trim')
    paths=[]
    duration = fixtures._duration
    for index, impl in enumerate((cpu, gpu)):
        base=tmp_path/str(index); base.mkdir()
        # Synthetic CPU fixtures deliberately favor CUDA so the entire selected
        # path is exercised; these are not historical or measured GPU costs.
        with monkeypatch.context() as synthetic:
            synthetic.setattr(fixtures, '_duration', lambda i, r: (2 if impl == cpu else 1) * duration(i, r))
            f=_fixture(base,monkeypatch,H3,impl)
        result=_assemble(f)
        path=base/'result.json'; path.write_text(json.dumps(result)); paths.append(path)
    monkeypatch.setattr(registry, 'git_head', lambda _: HEAD)
    monkeypatch.setattr(adapter, 'git_head', lambda _: HEAD)
    from distillschedule.search import execution
    monkeypatch.setattr(execution, 'git_head', lambda _: HEAD)
    problem,costs=assemble_calibrations(paths,output=tmp_path/'joined',expected_head=HEAD)
    return cpu,gpu,problem,costs,paths


def test_cpu_fixture_calibration_search_serialization_and_native_dispatch_binding(tmp_path,monkeypatch):
    from distillschedule.search.execution import run_plan
    from distillschedule.workloads.best_known import dispatch
    cpu,gpu,problem,costs,_=_joined_calibration(tmp_path,monkeypatch)
    result=search_plans(problem,costs,exhaustive=True,max_proposals=16)
    assert len(result['ranked']) == 2 and result['rejections']
    assert {row['status'] for row in result['rejections']} <= {'UNCALIBRATED','REJECTED_CONTRACT'}
    # Retain the CUDA binding rule but remove its observations: CPU costs may
    # not fill it, even though the workload source/shape/assets still match.
    from distillschedule.search.cost_model import MissingCalibration
    from distillschedule.search.simulator import simulate
    from distillschedule.search.native_macro_adapter import cost_key
    gpu_keys={cost_key(H3,op,gpu) for op in problem.graph.operations}
    partial=CostModel(costs.binding,tuple(r for r in costs.records if r.key not in gpu_keys))
    with pytest.raises(MissingCalibration): simulate(problem,gpu.assignment,partial)
    assert result['claimable_speedup'] is False
    plan=lower_prediction(result,problem,costs,run_id='cpu-fixture-selected')
    saved=json.loads(json.dumps(plan))
    selected=H3Implementation.parse(saved['search_candidate']['id'])
    assert selected == gpu and selected.request==LOW
    assert saved['fully_resolved_profile']['runtime']['environment']['H3_FA3_REPLAY_CACHE_STORAGE']==selected.storage
    calls=[]
    monkeypatch.setattr(dispatch,'run_resolved',lambda receipt,**kw: calls.append(receipt) or 0)
    assert run_plan(saved,dry_run=True)==0
    assert calls[0]['search_candidate']['id']==selected.id
    saved['fully_resolved_profile']['runtime']['environment']['H3_FA3_REPLAY_CACHE_STORAGE']='ignored'
    with pytest.raises(ValueError,match='serialized profile'): run_plan(saved,dry_run=True)
    assert len(calls)==1


def test_joined_cost_sources_are_rechecked_after_serialization(tmp_path,monkeypatch):
    from distillschedule.search.native_macro_adapter import validate_native_problem
    _,_,problem,_,paths=_joined_calibration(tmp_path,monkeypatch)
    result=json.loads(paths[0].read_text())
    provenance=json.loads(Path(result['provenance']['path']).read_text())
    source=Path(provenance['native_sources'][0]['path'])
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError,match='source receipt changed'): validate_native_problem(problem)


def test_rehashed_substitute_cost_cannot_replace_selected_native_calibration(tmp_path,monkeypatch):
    _,_,problem,costs,_=_joined_calibration(tmp_path,monkeypatch)
    first=costs.records[0]
    raw=json.loads(Path(first.evidence['source_receipt']).read_text())
    raw['samples_ns']=[n+100 for n in raw['samples_ns']]
    path=tmp_path/'fabricated-cost.json'; path.write_text(json.dumps(raw))
    replacement=replace(first,samples_ns=tuple(raw['samples_ns']),evidence={**first.evidence,
        'source_receipt':str(path), 'source_receipt_sha256':registry.file_sha256(path)})
    changed=replace(costs,records=(replacement,*costs.records[1:]))
    changed.verify_receipts()  # Self-consistency alone is deliberately insufficient.
    result=search_plans(problem,changed,exhaustive=True,max_proposals=16)
    with pytest.raises(ValueError,match='assembled source observations'):
        lower_prediction(result,problem,changed,run_id='must-not-launch')


@pytest.mark.parametrize('workload', sorted(SUPPORTED_WORKLOADS))
def test_selected_plan_cannot_cross_dispatch_to_another_workload(workload):
    other = next(w for w in sorted(SUPPORTED_WORKLOADS) if w != workload)
    receipt = dict(profile_id=workload, model_plan={'problem':{'workload':other}})
    with pytest.raises(ValueError, match='dispatch workload differ'):
        validated_profile(receipt, root=registry.repository_root())
