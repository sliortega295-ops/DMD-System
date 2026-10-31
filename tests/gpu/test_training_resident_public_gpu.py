"""Public budgeted training on actual tiny native CUDA models.

Opt-in single-owner qualification, not pretrained performance or quality. Use
the original pinned target environment and root-owned serial GPU schedule.
"""
from dataclasses import replace
import gc
import json
from pathlib import Path

import pytest
import torch

from distillschedule.training import session
from distillschedule.training.policy import CyclePolicy
from distillschedule.training.resident import active
from distillschedule.training.checkpoint import cpu_tree
from distillschedule.open_workload.torch_backend import Session
from test_resident_trial_gpu import native_case, budget, _uninterrupted, pointers
from test_training_resident_lifecycle import (
    implementation, test_native_public_manual_fresh_process_checkpoint_resume as manual_restart,
    test_public_auto_saved_plan_and_resume_never_search_again as saved_restart,
)
from test_training_end_to_end import equal
from cuda_test_policy import cuda_bitwise_policy, run_cli

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(),
    reason='actual public resident CUDA lifecycle NOT_RUN: CUDA device required'),
    pytest.mark.usefixtures('cuda_bitwise_policy')]


def public(config):
    limits = dict(schema_version='distillschedule.training_resident.v1',
        construction_limit_bytes=128 << 20, host_staging_limit_bytes=128 << 20,
        trial=vars(budget()))
    return replace(config, execution=dict(config.execution, resident_budget=limits))


def test_actual_public_auto_keeps_native_owner_and_new_inputs(native_case, tmp_path, monkeypatch):
    family, config = native_case
    _, expected, _, _ = _uninterrupted(config, 3)
    other = next((i for i in range(torch.cuda.device_count())
                  if i != torch.device(config.trainer['device']).index), None)
    other_rng = None if other is None else torch.cuda.get_rng_state(other).clone()
    config = public(config)
    config = replace(config, execution=dict(config.execution, mode='auto', memory_limit_bytes=2 << 30,
        max_candidates=2, samples=1, max_program_runs=3))
    components, data, algorithm, _ = session.construct(config)
    owner = components.state['native_cycle']; before = pointers(owner)
    assert active(owner) and components.construction_receipt['observed_managed_storage']['device_storage_bytes'] > 0
    policy = CyclePolicy(config.execution); versions = None
    output = tmp_path/'auto'; output.mkdir()
    created = []; initialize = Session.__init__
    def no_clone(self, workload, implementation, **kwargs):
        assert kwargs.get('copy_state') is False
        created.append(implementation)
        initialize(self, workload, implementation, **kwargs)
    monkeypatch.setattr(Session, '__init__', no_clone)
    try:
        records = []
        for cycle in range(3):
            workload = session.build_cycle(config, components, data, algorithm,
                cycle=cycle, cursor=cycle, versions=versions)
            current, _ = policy.run(workload, output)
            assert current.state['native_cycle'] is owner and owner.cycle == cycle + 1
            assert equal(cpu_tree(owner.state_dict()), expected[cycle])
            assert pointers(owner)['parameters'] == before['parameters']
            assert active(owner) and not hasattr(owner, '_resident_trial_active')
            versions = current.receipt['dispatcher']['final_versions']
            records.append(dict(cycle=cycle+1, final_versions=versions,
                selected=current.implementation,
                effects=current.receipt.get('native_capability_evidence')))
        if other_rng is not None:
            assert torch.equal(other_rng, torch.cuda.get_rng_state(other))
        assert policy.bound['resident_budget'] == config.execution['resident_budget']
        assert json.loads((output/'plan-admission/execution.json').read_text())['resident_trial']['adopted']
        proof = dict(family=family, actual_cycles=3, exact_native_state=True,
            owner_clones=0, session_constructions=len(created), unowned_rng='UNCHANGED',
            construction=components.construction_receipt, cycles=records,
            shared_search=True, selected_plan=str(output/'planning/selected-plan.json'))
    finally:
        components.close()
    assert not active(owner)
    (tmp_path/'public-auto-proof.json').write_text(json.dumps(proof, indent=2)+'\n')
    del owner, current, workload, components, policy
    gc.collect()


def test_actual_public_manual_cli_resume_export(native_case, tmp_path, monkeypatch):
    family, config = native_case
    # Reuse the exact public CPU assertion chain, substituting only the existing
    # deterministic test subprocess wrapper; no production classes are patched.
    import test_training_end_to_end
    monkeypatch.setattr(test_training_end_to_end, 'run_cli', run_cli)
    manual_restart((family, public(config)), tmp_path)
    rows = [json.loads(line) for line in (tmp_path/'resumed/metrics.jsonl').read_text().splitlines()]
    assert rows[0]['execution']['resident_trial']['adopted']
    assert rows[1]['execution']['resident_lifecycle']['last_continuation']['peak_allocated_bytes'] > 0
    (tmp_path/'public-manual-proof.json').write_text(json.dumps(dict(family=family,
        actual_cycles=3, same_source_fresh_process_exact_state=True,
        optimizer_scheduler_ema_rng_data_and_clocks='EXACT', current_student_export_decoded=True,
        resumed_cycles=2, admission=json.loads((tmp_path/'resumed/resume-owner-admission.json').read_text())),
        indent=2)+'\n')
    gc.collect()


@pytest.mark.parametrize('native_case', ['sd15'], indirect=True)
def test_actual_public_saved_plan_resume(native_case, tmp_path, monkeypatch):
    import test_training_end_to_end
    monkeypatch.setattr(test_training_end_to_end, 'run_cli', run_cli)
    family, config = native_case
    saved_restart((family, public(config)), tmp_path, monkeypatch)
    (tmp_path/'public-saved-proof.json').write_text(json.dumps(dict(family=family,
        auto_and_saved_plan_training_state='EXACT', fresh_process_resume='EXACT',
        no_implicit_research=True, tampered_plan='REJECTED'))+'\n')
