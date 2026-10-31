"""Opt-in tiny-native CUDA shared planning; no public/pretrained claim.

Reuse the R17 qualified native/reference fixture and keep CUDA owners serial.
Only the root-owned Torch2.10 lane runs this file. CPU collection is NOT_RUN.
"""
import inspect
import json
from pathlib import Path

import pytest
import torch

from distillschedule.open_workload import planning
from distillschedule.open_workload.execution import ExecutionBackend
from distillschedule.open_workload.resident_trial import _digest
from distillschedule.open_workload.torch_backend import Session
from distillschedule.training.checkpoint import cpu_tree, rng_state
from distillschedule.training.session import construct, build_cycle
from distillschedule.workloads.best_known.registry import file_sha256, write_new_json
from test_resident_trial_gpu import (
    native_case, budget, pointers, physical, public_refused, _uninterrupted, BASELINE,
)
from test_training_end_to_end import equal
from cuda_test_policy import cuda_bitwise_policy

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(),
    reason='actual CUDA resident planning NOT_RUN: CUDA-capable Torch/device required'),
    pytest.mark.usefixtures('cuda_bitwise_policy')]


@pytest.mark.parametrize('warmed', [False, True])
def test_native_cuda_shared_calibration_search_saved_plan_adopts_once(native_case, warmed, monkeypatch, tmp_path):
    family, config = native_case; first = int(warmed)
    # This reference path already has the independent pinned-source oracle in
    # R17. It is executed here again, not substituted by a historical result.
    initial, expected, expected_outputs, _ = _uninterrupted(config, first + 2)
    components, data, algorithm, _ = construct(config)
    owner = components.state['native_cycle']; owner.load_state_dict(initial)
    if warmed:
        warm = build_cycle(config, components, data, algorithm, cycle=0, cursor=0)
        owner.run(warm.feeds, cycle=0)
    workload = build_cycle(config, components, data, algorithm, cycle=first, cursor=first)
    before = cpu_tree(owner.state_dict()); original = pointers(owner); flags = physical(owner)
    outer = rng_state(device=owner.device); feeds = _digest(workload.feeds)
    selected_device = torch.cuda.current_device()
    other = next((i for i in range(torch.cuda.device_count()) if i != owner.device.index), None)
    other_rng = None if other is None else torch.cuda.get_rng_state(other).clone()
    public_refused(owner, family)

    def unchanged():
        assert equal(before, cpu_tree(owner.state_dict())) and pointers(owner) == original
        assert equal(outer, rng_state(device=owner.device)) and _digest(workload.feeds) == feeds
        assert physical(owner) == flags and all(p.grad is None for p in owner.model.parameters())
        assert not hasattr(owner, '_resident_trial_active')
        assert torch.cuda.current_device() == selected_device
        if other is not None: assert torch.equal(other_rng, torch.cuda.get_rng_state(other))

    creations = []; resets = []
    initialize = Session.__init__; reset_peak = torch.cuda.reset_peak_memory_stats
    def no_clone(self, workload, implementation, **kwargs):
        assert kwargs.get('copy_state') is False, 'resident planning attempted a CUDA owner clone'
        creations.append(dict(implementation=implementation, copy_state=False))
        initialize(self, workload, implementation, **kwargs)
    def phase_reset(device):
        frame = inspect.currentframe().f_back
        assert frame.f_globals['__name__'] == 'distillschedule.open_workload.resident_trial'
        assert frame.f_code.co_name in ('_admit_cuda', '_begin_cuda'), 'nested measurement erased phase peak'
        resets.append(frame.f_code.co_name)
        reset_peak(device)
    monkeypatch.setattr(Session, '__init__', no_clone)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', phase_reset)

    manifest = planning.calibrate(workload, tmp_path/'costs', samples=1, max_candidates=2,
        max_program_runs=3, resident_budget=budget())
    unchanged()
    assert manifest['budget']['actual'] == 3 and len(manifest['variants']) == 2
    assert all(row['reference_gate']['status'] == 'PASS' and row['native_capability_evidence']['active']
               for row in manifest['variants'])
    assert manifest['resident_trial']['closed'] and not manifest['resident_trial']['adopted']
    assert manifest['hardware_probe']['status'] == 'NOT_RUN_RESIDENT_BUDGET'
    measured = []
    for variant in manifest['variants']:
        costs = json.loads(Path(variant['cost_model']).read_text())
        for record in costs['records']:
            raw = json.loads(Path(record['evidence']['source_receipt']).read_text())
            for sample in raw['rank_observations']:
                for observation in sample:
                    if 'cuda_event_ns' not in observation: continue  # zero-cost finish barrier
                    assert observation['capacity_measurement'] == 'resident_phase_high_water_bound'
                    assert observation['allocator_peak_counter_reset'] is False
                    measured.append(observation)
    assert len(measured) == 2 * len(workload.program.calls)
    document = planning.plan(tmp_path/'costs/calibration.json', memory_limit_bytes=2**31)
    write_new_json(tmp_path/'plan.json', document)
    assert document['search']['ranked'] and document['resident_budget'] == manifest['resident_budget']

    publish = planning._publish_resident_success
    def failed_publication(path, value):
        if path.name == 'execution.json':
            assert owner.cycle == first + 1 and value['resident_trial']['adopted']
            assert not owner._resident_trial_active.receipt['adopted'], 'adoption preceded publication'
            raise OSError('injected saved-plan publication failure')
        return publish(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(planning, '_publish_resident_success', failed_publication)
        with pytest.raises(OSError, match='publication failure'):
            planning.run(workload, tmp_path/'plan.json', tmp_path/'failed-run',
                         persistent=True, resident_budget=budget())
    unchanged()
    assert not (tmp_path/'failed-run/execution.json').exists()
    failure = json.loads((tmp_path/'failed-run/failure.json').read_text())
    assert failure['resident_trial']['closed'] and not failure['resident_trial']['adopted']
    assert failure['reference_gate']['status'] == 'PASS'
    assert all(row['active'] for row in failure['native_capability_evidence'])
    failed_phases = [row['phase'] for row in failure['resident_trial']['cuda_observations']]
    assert 'failed' in failed_phases and failed_phases[-1] == 'rollback'

    result, session, output = planning.run(workload, tmp_path/'plan.json', tmp_path/'run',
                                         persistent=True, resident_budget=budget())
    assert session.state['native_cycle'] is owner and owner.cycle == first + 1
    assert equal(cpu_tree(owner.state_dict()), expected[first])
    assert equal(cpu_tree(next(iter(output.values()))), expected_outputs[first])
    assert result['reference_gate']['status'] == 'PASS'
    assert result['selected_implementation'] == document['selected']['implementation']
    assert result['executed_order'] == document['selected']['order']
    assert all(row['active'] for row in result['native_capability_evidence'])
    assert result['resident_trial']['adopted'] and result['resident_trial']['closed']
    assert json.loads((tmp_path/'run/execution.json').read_text()) == result
    assert equal(outer, rng_state(device=owner.device)) and _digest(workload.feeds) == feeds
    current = pointers(owner)
    for key in ('model', 'parameters', 'optimizers'): assert current[key] == original[key]
    if warmed: assert current['slots'] == original['slots']
    assert torch.cuda.current_device() == selected_device
    if other is not None: assert torch.equal(other_rng, torch.cuda.get_rng_state(other))
    public_refused(owner, family)
    assert len(creations) == 7, 'three calibration plus two paired saved-plan attempts'

    # No public optimized continuation is implied; the normal reference owner
    # must still be usable after exactly one adopted private cycle.
    continuation = build_cycle(config, components, data, algorithm, cycle=first + 1, cursor=first + 1)
    continued = Session(continuation, BASELINE, copy_state=False).run()
    assert equal(cpu_tree(owner.state_dict()), expected[first + 1])
    assert equal(cpu_tree(next(iter(continued.values()))), expected_outputs[first + 1])
    proof = dict(schema_version='distillschedule.resident_planning_gpu_proof.v1',
        family=family, warmed=warmed, evidence='ACTUAL_RANDOM_TINY_NATIVE_CUDA_NOT_PRETRAINED',
        source=owner.source.identity, framework_source=planning.source_identity(),
        plan_sha256=document['plan_sha256'], calibration_sha256=file_sha256(tmp_path/'costs/calibration.json'),
        execution_sha256=file_sha256(tmp_path/'run/execution.json'),
        calibration_purity='EXACT_STATE_ADAM_EMA_FEED_NATIVE_AND_OUTER_RNG',
        current_native_reference='EXACT_PASS', independent_source_oracle='R17_QUALIFIED_NOT_RERUN_HERE',
        shared_search='MEASURED_COSTS_SHARED_DP_SAVED_PLAN',
        selected_implementation=result['selected_implementation'], selected_effects=result['native_capability_evidence'],
        calibration_numeric_and_effect_gates='PASS_BOTH_NATIVE_OPTIONS',
        saved_plan_numeric_gate=result['reference_gate'], accepted_cycles=1, normal_continued_cycles=1,
        owner_clones=0, nested_peak_resets=0, phase_reset_count=len(resets),
        measured_capacity_scope='NON_RESETTING_RESIDENT_PHASE_HIGH_WATER_BOUND',
        failed_publication='EXACT_ROLLBACK_NO_SUCCESS_RECEIPT', public_cuda='REFUSED',
        failed_publication_observations=failure['resident_trial']['cuda_observations'],
        unowned_rng='NOT_VISIBLE' if other is None else 'UNCHANGED',
        calibration_observations=manifest['resident_trial']['cuda_observations'],
        execution_observations=result['resident_trial']['cuda_observations'])
    path = tmp_path/'resident-planning-proof.json'; write_new_json(path, proof)
    print('CUDA_RESIDENT_PLANNING_PROOF ' + json.dumps(dict(path=str(path), sha256=file_sha256(path),
        family=family,warmed=warmed,status='PASS_PRIVATE_TINY_NATIVE_PLANNING'),sort_keys=True))
