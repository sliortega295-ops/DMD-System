"""Synthetic capacity receipts exercise a CPU controller; never GPU capacity proof."""
from dataclasses import replace
import json
import pytest
from distillschedule.search import model_validation
from distillschedule.search.cost_model import CostModel
from distillschedule.search.model_search import search_plans
from distillschedule.search.runtime_identity import observed_binding
from distillschedule.search.wan_implementation import WanImplementation
from distillschedule.search.wan_model_adapter import cost_key
from distillschedule.workloads.best_known.registry import file_sha256, git_head, repository_root
from test_refactor_model_validation import fake_observations, measurement
from test_refactor_wan_model_adapter import calibration_fixture

def rejected_anchor_inputs(tmp_path, *, baseline=False):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    binding = observed_binding(fake_observations(), source_head=git_head(repository_root()))
    problem, costs = calibration_fixture(inputs, binding)
    impl = WanImplementation('fsdp', 512, False) if baseline else WanImplementation('resident', 2048, True)
    keys = {cost_key(op, impl) for op in problem.graph.operations}
    records = []
    for record in costs.records:
        if record.key in keys:
            evidence = dict(record.evidence)
            path = inputs / (record.key + '.json')
            raw = json.loads(path.read_text())
            raw['workspace_bytes_per_rank'] = 2000
            path.write_text(json.dumps(raw))
            evidence['source_receipt_sha256'] = file_sha256(path)
            record = replace(record, workspace_bytes_per_rank=2000, evidence=evidence)
        records.append(record)
    costs = CostModel(costs.binding, tuple(records))
    prediction = search_plans(problem, costs, exhaustive=True, max_proposals=12)
    return (problem, costs, prediction, impl.id)

def test_capacity_rejected_incumbent_is_not_launched_or_given_old_timing(tmp_path):
    p, c, pred, rejected = rejected_anchor_inputs(tmp_path)
    s = model_validation.ModelValidationSession(problem=p, costs=c, prediction=pred, output=tmp_path / 'validation', max_trials=3)
    calls = []

    def execute(candidate, phase, out):
        calls.append((candidate.id, phase))
        assert candidate.id != rejected
        return measurement(s, candidate)
    result = s.run(execute)
    assert [phase for _, phase in calls] == ['baseline', 'confirmation']
    assert result['status'] == 'COMPLETE_NATIVE_VALIDATION'
    assert result['timing_attempts'] == 2
    assert result['incumbent'] is None
    assert result['improvement_over_incumbent'] is None
    assert result['rejected_anchors'][rejected]['gpu_execution'] == 'NOT_RUN'
    assert result['rejected_anchors'][rejected]['capacity_failure_observed'] is False
    assert result['capacity_failures'] == 0
    assert result['strict_winner'] is None
    assert result['cycle2_speedup_diagnostic'] == 1
    assert s.run(lambda *a: pytest.fail('completed session reran')) == result
    resumed = model_validation.ModelValidationSession(problem=p, costs=c, prediction=pred, output=tmp_path / 'validation', max_trials=3, resume=True)
    assert resumed.run(lambda *a: pytest.fail('resume launched rejected incumbent')) == result
    with pytest.raises(ValueError, match='rejected by prediction'):
        s._trial(rejected, 'incumbent', execute)

def test_rejected_baseline_still_blocks_before_any_native_launch(tmp_path):
    p, c, pred, _ = rejected_anchor_inputs(tmp_path, baseline=True)
    with pytest.raises(ValueError, match='baseline/proposal'):
        model_validation.ModelValidationSession(problem=p, costs=c, prediction=pred, output=tmp_path / 'validation', max_trials=3)
    assert not (tmp_path / 'validation').exists()

def test_rejected_incumbent_does_not_expand_new_candidate_budget(tmp_path):
    p, c, pred, rejected = rejected_anchor_inputs(tmp_path)
    s = model_validation.ModelValidationSession(problem=p, costs=c, prediction=pred, output=tmp_path / 'validation', max_trials=4)
    calls = []

    def execute(candidate, phase, out):
        calls.append((candidate.id, phase))
        assert candidate.id != rejected
        return measurement(s, candidate, seconds=5 if phase == 'candidate' else 10)
    r = s.run(execute)
    assert [phase for _, phase in calls] == ['baseline', 'candidate', 'confirmation']
    assert r['new_candidate_attempts'] == 1
    assert r['timing_attempts'] == 3
    assert r['improvement_over_incumbent'] is None


def test_same_baseline_timing_jitter_cannot_become_strict_winner(tmp_path):
    problem, costs, prediction, _ = rejected_anchor_inputs(tmp_path)
    controller = model_validation.ModelValidationSession(
        problem=problem, costs=costs, prediction=prediction,
        output=tmp_path / 'validation', max_trials=3)
    result = controller.run(lambda candidate, phase, _: measurement(
        controller, candidate, numerical='PASS', seconds=9 if phase == 'confirmation' else 10))
    assert result['baseline']['candidate_id'] == result['confirmed_candidate']['candidate_id']
    assert result['strict_winner'] is None


@pytest.mark.parametrize('entry', ['prepare_reference', 'trajectory', 'timing', 'calibration'])
def test_native_runner_blocks_rejected_peer_before_any_gpu_path(tmp_path, monkeypatch, entry):
    from distillschedule.search import execution
    from distillschedule.workloads.best_known.registry import fingerprint
    problem, costs, prediction, rejected = rejected_anchor_inputs(tmp_path)
    controller = model_validation.ModelValidationSession(
        problem=problem, costs=costs, prediction=prediction,
        output=tmp_path / 'validation', max_trials=3)
    assets = {'assets': {'fixture': 'CPU_ONLY'}}
    identity = {**controller.identity, 'asset_fingerprint': fingerprint(assets['assets'])}
    runner = execution.NativeTrialRunner(
        workload=problem.workload, identity=identity, assets=assets, output=tmp_path,
        run_prefix='CPU_ONLY', reference_id=controller.seeds[0], bootstrap_peer=rejected,
        admitted_candidate_ids=set(controller.candidates))
    monkeypatch.setattr(runner, '_rpc', lambda *a, **k: pytest.fail('unexpected RPC/GPU preparation'))
    monkeypatch.setattr(execution, 'run_plan', lambda *a, **k: pytest.fail('unexpected GPU launch'))
    candidate = runner.bootstrap_peer
    with pytest.raises(ValueError, match='rejected by prediction'):
        if entry == 'prepare_reference':
            runner._prepare(runner.reference)
        elif entry == 'trajectory':
            runner._trajectory(candidate)
        elif entry == 'timing':
            runner(candidate, 'incumbent', tmp_path / 'timing')
        else:
            runner.calibrate(candidate, directory=tmp_path / 'calibration')
    assert not (tmp_path / 'preparation').exists()


def test_native_runner_can_reuse_bound_preparation_without_launching_rejected_peer(tmp_path, monkeypatch):
    from distillschedule.search import execution
    from distillschedule.workloads.best_known.registry import fingerprint
    problem, costs, prediction, rejected = rejected_anchor_inputs(tmp_path)
    controller = model_validation.ModelValidationSession(
        problem=problem, costs=costs, prediction=prediction,
        output=tmp_path / 'validation', max_trials=3)
    assets = {'assets': {'fixture': 'CPU_ONLY'}}
    identity = {**controller.identity, 'asset_fingerprint': fingerprint(assets['assets'])}
    runner = execution.NativeTrialRunner(
        workload=problem.workload, identity=identity, assets=assets, output=tmp_path,
        run_prefix='CPU_ONLY', reference_id=controller.seeds[0], bootstrap_peer=rejected,
        admitted_candidate_ids=set(controller.candidates))
    ready = {'source_head': identity['source_head'], 'candidate_id': runner.reference.id, 'assets': assets}
    folder = tmp_path / 'preparation'
    folder.mkdir()
    path = folder / (runner.reference.id + '-ready.json')
    path.write_text(json.dumps(ready))
    monkeypatch.setattr(runner, '_rpc', lambda *a, **k: pytest.fail('unexpected preparation launch'))
    assert runner._prepare(runner.reference) == ready
    runner.prepared.clear()
    ready['source_head'] = 'wrong-source'
    path.write_text(json.dumps(ready))
    with pytest.raises(ValueError, match='source/candidate'):
        runner._prepare(runner.reference)


def test_public_cli_passes_admission_and_chooses_a_feasible_bootstrap_peer(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from distillschedule.search import model_cli, execution
    from distillschedule.workloads.best_known import registry
    problem, costs, _, rejected = rejected_anchor_inputs(tmp_path)
    cost_path = tmp_path / 'costs.json'
    cost_path.write_text(json.dumps(costs.to_dict()))
    assets = {'assets': {'fixture': 'CPU_ONLY'}}
    original_fingerprint = model_cli.fingerprint
    monkeypatch.setattr(model_cli, 'fingerprint', lambda value:
        problem.binding['asset_fingerprint'] if value == assets['assets'] else original_fingerprint(value))
    monkeypatch.setattr(registry, 'load_assets', lambda *a: assets)
    captured = {}

    class CPUReceiptProducer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.identity = kwargs['identity']

        def __call__(self, candidate, phase, directory):
            assert candidate.id in captured['admitted_candidate_ids']
            return measurement(self, candidate)

    monkeypatch.setattr(execution, 'NativeTrialRunner', CPUReceiptProducer)
    args = SimpleNamespace(workload=problem.workload, execute_native=True, validation_dir=tmp_path / 'validation',
        assets_config=tmp_path / 'assets.json', resume=False, dry_run=False, emit_native_plan=None,
        historical_profile_prior=False, max_trials=3, problem=None,
        planning_layout=tmp_path / 'inputs/layout.json', cost_model=cost_path,
        search_seed=1, max_proposals=12, search_timeout=120, exhaustive=True, run_id='CPU_ONLY')
    result = model_cli.model_based_search(args)
    assert rejected not in captured['admitted_candidate_ids']
    assert captured['bootstrap_peer'] in captured['admitted_candidate_ids']
    assert captured['bootstrap_peer'] != captured['reference_id']
    assert result['incumbent'] is None
