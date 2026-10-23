"""Bounded stateful native options, without replaying optimizer macrocalls."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from distillschedule.open_workload.checkpoint_policy import native_effect_evidence
from distillschedule.open_workload.planning import calibrate
from distillschedule.open_workload.torch_backend import Workload, Session, candidates, signature
from distillschedule.training.policy import CyclePolicy, cycle_structure
from distillschedule.workloads.program import Call, Program


class NativeOwner:
    descriptor_version = 1
    exercise = True

    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor([.2, -.3]))
        self.adam = torch.optim.Adam([self.weight], lr=.001)
        self.clock = 0
        self.enabled = False
        self.effects = None

    def state_dict(self):
        return dict(weight=self.weight.detach().clone(), adam=deepcopy(self.adam.state_dict()), clock=self.clock)

    def load_state_dict(self, value):
        with torch.no_grad():
            self.weight.copy_(value['weight'])
        self.adam.load_state_dict(value['adam']); self.clock = value['clock']

    def execution_options(self):
        return dict(schema_version='distillschedule.owner_execution.v1',
                    baseline={'gradient_checkpointing': False},
                    options=[{'gradient_checkpointing': False}, {'gradient_checkpointing': True}],
                    effect=dict(kind='native_checkpoint', enabled_key='gradient_checkpointing', components=['native']),
                    identity={'version': self.descriptor_version})

    @contextmanager
    def execution_scope(self, option, effects):
        old = self.enabled
        self.enabled = option['gradient_checkpointing']; self.effects = effects
        effects['components'] = {'native': dict(checkpoint_requests=0, recompute_calls=0, disabled_verified=False)}
        try:
            yield
        finally:
            effects['components']['native']['disabled_verified'] = not self.enabled
            self.enabled = old; self.effects = None

    def run(self, x):
        def forward(value):
            return (value * self.weight).sin().square().sum()
        if self.enabled and self.exercise:
            row = self.effects['components']['native']; row['checkpoint_requests'] += 1
            count = 0
            def counted(value):
                nonlocal count
                count += 1
                if count > 1:
                    row['recompute_calls'] += 1
                return forward(value)
            loss = torch.utils.checkpoint.checkpoint(counted, x, use_reentrant=False)
        else:
            loss = forward(x)
        loss.backward(); self.adam.step(); self.adam.zero_grad(set_to_none=True); self.clock += 1
        return loss.detach()


class MalformedOwner(NativeOwner):
    def execution_options(self):
        value = super().execution_options()
        if self.fault == 'schema': value['schema_version'] = 'invented'
        elif self.fault == 'baseline': value['baseline'] = []
        elif self.fault == 'boolean': value['options'][1]['gradient_checkpointing'] = 1
        elif self.fault == 'budget': value['options'] *= 17
        elif self.fault == 'effect': value['effect']['enabled_key'] = []
        elif self.fault == 'identity': value['identity'] = {}
        elif self.fault == 'opaque': value['identity']['callback'] = object()
        elif self.fault == 'rng': torch.rand(1)
        return value


class ChangedMathOwner(NativeOwner):
    def run(self, x):
        return super().run(x) + (1 if self.enabled else 0)


class MustNotRunOwner(NativeOwner):
    def run(self, x):
        raise AssertionError('invalid selection executed a native update')


class BaselineOnlyOwner(NativeOwner):
    def execution_options(self):
        value = super().execution_options(); value['options'] = [value['baseline']]
        return value


def native_call(ctx):
    return ctx.state['owner'].run(ctx.inputs['x'])


def build(owner=None):
    return Workload(Program((Call('native', 'native', ('x',), ('owner',), ('owner',),
                                 grad='external_grad', replay_safe=False),), {'owner': 0}, ('x',), ('native',)),
                    {'native': native_call}, {'owner': owner or NativeOwner()}, {'x': torch.tensor([.7, -.5])})


def optimized():
    return dict(backend='torch-callable-resident-v1', share_immutable=False, checkpoint_bindings=[],
                owner_execution={'owner': {'gradient_checkpointing': True}})


def test_optional_owner_inventory_identity_and_effects():
    workload = build(); before = signature(workload.state)
    inventory = candidates(workload)
    assert inventory['total_candidates'] == 2
    assert 'owner_execution' not in inventory['candidates'][0]
    assert inventory['candidates'][1] == optimized()
    assert signature(workload.state) == before
    identity = workload.identity; structure = cycle_structure(workload)
    session = Session(workload, optimized()); session.run()
    assert native_effect_evidence(optimized(), session.effects)['active']
    assert not session.state['owner'].enabled and not workload.state['owner'].clock
    workload.state['owner'].descriptor_version += 1
    assert workload.identity != identity and cycle_structure(workload) != structure


def test_new_owner_manual_is_paired_before_live_adoption(tmp_path):
    workload = build(); original = signature(workload.state)
    policy = CyclePolicy({'mode': 'manual', 'implementation': optimized()})
    session, _ = policy.run(workload, tmp_path)
    assert policy.bound['initial_reference_gate']['status'] == 'PASS'
    assert session.state['owner'] is not workload.state['owner']
    assert signature(workload.state) == original and session.state['owner'].clock == 1


def test_unexercised_owner_is_pruned_and_manual_does_not_commit(tmp_path):
    owner = NativeOwner(); owner.exercise = False; workload = build(owner)
    before = signature(workload.state)
    manifest = calibrate(workload, tmp_path / 'calibration', samples=1, max_candidates=2, max_program_runs=3)
    assert len(manifest['variants']) == 1 and len(manifest['failed']) == 1
    with pytest.raises(ValueError, match='capability.*exercised'):
        CyclePolicy({'mode': 'manual', 'implementation': optimized()}).run(workload, tmp_path)
    assert signature(workload.state) == before


@pytest.mark.parametrize('option', [dict(unknown={'gradient_checkpointing': True}),
                                  dict(owner={'gradient_checkpointing': 1}),
                                  dict(owner={'wrong': True}), {}, []])
def test_bad_owner_options_reject_before_execution(option):
    workload = build(); before = signature(workload.state)
    with pytest.raises(ValueError):
        Session(workload, dict(optimized(), owner_execution=option), copy_state=False)
    assert signature(workload.state) == before


def test_partial_provider_rejected_and_legacy_no_provider_unchanged():
    owner = NativeOwner(); owner.execution_scope = None
    with pytest.raises(ValueError, match='owner'):
        candidates(build(owner))
    from test_open_workload_execution import build as legacy
    workload = legacy(n=1)
    assert 'owner_execution' not in workload.identity
    assert all('owner_execution' not in option for option in candidates(workload)['candidates'])


def test_owner_effect_gate_composes_nested_selective_policy():
    implementation = {'local': dict(optimized(), checkpoint_policy='save_matmul')}
    effects = {'replica_effects': [dict(selective_save_requests=2, selective_native_cache_bytes=8)]}
    assert not native_effect_evidence(implementation, effects)['active']


@pytest.mark.parametrize('fault', ['schema', 'baseline', 'boolean', 'budget', 'effect', 'identity', 'opaque', 'rng'])
def test_malformed_descriptor_fails_before_native_execution(fault):
    owner = MalformedOwner(); owner.fault = fault; workload = build(owner)
    before = signature(workload.state); rng = torch.random.get_rng_state()
    with pytest.raises(ValueError, match='owner'):
        candidates(workload)
    assert signature(workload.state) == before and torch.equal(torch.random.get_rng_state(), rng)


def test_owner_numerical_failure_does_not_adopt_or_commit(tmp_path):
    workload = build(ChangedMathOwner()); before = signature(workload.state)
    policy = CyclePolicy({'mode': 'manual', 'implementation': optimized()})
    with pytest.raises(ValueError, match='numerical admission'):
        policy.run(workload, tmp_path)
    assert policy.bound is None and policy.session is None and signature(workload.state) == before
    assert not (tmp_path / 'execution.json').exists()


def test_descriptor_and_source_change_reject_saved_plan_and_resume(tmp_path):
    from distillschedule.open_workload.planning import plan, run
    from distillschedule.workloads.best_known.registry import write_new_json
    workload = build(); calibrate(workload, tmp_path / 'cal', samples=1, max_candidates=2, max_program_runs=3)
    document = plan(tmp_path / 'cal/calibration.json', memory_limit_bytes=100_000)
    write_new_json(tmp_path / 'plan.json', document)
    policy = CyclePolicy({'mode': 'manual', 'implementation': optimized()}); session, _ = policy.run(workload, tmp_path)
    workload.state['owner'].descriptor_version += 1
    before = signature(workload.state)
    with pytest.raises(ValueError, match='changed'):
        run(workload, tmp_path / 'plan.json', tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists() and signature(workload.state) == before
    resumed = CyclePolicy(policy.execution); resumed.restore(policy.bound)
    with pytest.raises(ValueError, match='changed'):
        resumed.prepare_resume(workload)
    assert signature(workload.state) == before


def test_first_postrestore_owner_update_is_paired_once(tmp_path):
    policy = CyclePolicy({'mode': 'manual', 'implementation': optimized()})
    session, _ = policy.run(build(), tmp_path)
    workload = replace(build(), state=session.state,
                       program=replace(session.workload.program, initial_versions={'owner': 1}))
    resumed = CyclePolicy(policy.execution); resumed.restore(policy.bound); resumed.prepare_resume(workload)
    before = signature(workload.state)
    new, _ = resumed.run(workload, tmp_path)
    assert new.receipt['owner_resume_reference_admission'] and new.receipt['reference_gate']['status'] == 'PASS'
    assert signature(workload.state) == before and new.state['owner'].clock == 2
    workload = replace(workload, state=new.state, program=replace(workload.program, initial_versions={'owner': 2}))
    steady, _ = resumed.run(workload, tmp_path)
    assert steady is new and steady.state['owner'].clock == 3
    assert 'owner_resume_reference_admission' not in steady.receipt


def test_saved_plan_replay_rejects_new_no_effect_and_changed_evidence(tmp_path):
    import json
    from distillschedule.open_workload.planning import plan, run, load_calibration
    from distillschedule.workloads.best_known.registry import fingerprint, write_new_json
    workload = build(); manifest = calibrate(workload, tmp_path / 'cal', samples=1,
                                             max_candidates=2, max_program_runs=3)
    document = plan(tmp_path / 'cal/calibration.json', memory_limit_bytes=100_000)
    document['selected'] = next(row for row in document['search']['ranked'] if row['implementation'] == optimized())
    document.pop('plan_sha256'); document['plan_sha256'] = fingerprint(document)
    write_new_json(tmp_path / 'plan.json', document)
    workload.state['owner'].exercise = False; before = signature(workload.state)
    with pytest.raises(ValueError, match='capability.*exercised'):
        run(workload, tmp_path / 'plan.json', tmp_path / 'no-effect')
    assert signature(workload.state) == before and not (tmp_path / 'no-effect/execution.json').exists()
    assert (tmp_path / 'no-effect/failure.json').is_file()
    row = next(item for item in manifest['variants'] if item['implementation'] == optimized())
    row['effects']['owner_execution']['owner']['components']['native']['checkpoint_requests'] = 0
    (tmp_path / 'cal/calibration.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unobserved or changed native'):
        load_calibration(tmp_path / 'cal/calibration.json')


def test_resume_rejects_incompatible_owner_option_before_state_mutation(tmp_path):
    policy = CyclePolicy({'mode': 'manual', 'implementation': optimized()})
    session, _ = policy.run(build(), tmp_path)
    workload = replace(build(), state=session.state,
                       program=replace(session.workload.program, initial_versions={'owner': 1}))
    before = signature(workload.state)
    changed = deepcopy(policy.execution); changed['implementation']['owner_execution']['owner']['gradient_checkpointing'] = False
    resumed = CyclePolicy(changed); resumed.restore(policy.bound)
    with pytest.raises(ValueError, match='owner execution policy changed'):
        resumed.prepare_resume(workload)
    assert signature(workload.state) == before and resumed.session is None


def test_bad_manual_owner_selection_rejects_before_any_reference_update(tmp_path):
    workload = build(MustNotRunOwner())
    implementation = optimized(); implementation['owner_execution'] = {'missing': {'gradient_checkpointing': True}}
    with pytest.raises(ValueError, match='owner'):
        CyclePolicy(dict(mode='manual', implementation=implementation)).run(workload, tmp_path)


def test_baseline_only_provider_preserves_default_and_rejects_override():
    workload = build(BaselineOnlyOwner()); inventory = candidates(workload)
    assert inventory['total_candidates'] == len(inventory['candidates']) == 1
    assert 'stateful_owner_execution' not in inventory['capabilities']
    Session(workload, inventory['candidates'][0]).run()
    implementation = optimized(); implementation['owner_execution']['owner']['gradient_checkpointing'] = False
    before = signature(workload.state)
    with pytest.raises(ValueError, match='owner'):
        Session(workload, implementation, copy_state=False)
    assert signature(workload.state) == before
