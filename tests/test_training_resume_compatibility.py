"""Minor framework fixes preserve complete-cycle state, not just weights."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from distillschedule.training.config import TrainingConfig
from distillschedule.training.session import train
from test_training_end_to_end import fixture, equal


def test_framework_and_planning_only_changes_resume_the_real_owned_state(tmp_path, monkeypatch):
    from distillschedule.training import session
    from distillschedule.open_workload import planning
    root, config = fixture(tmp_path, samples=6)
    raw = config.to_dict(); raw['trainer'].update(cycles=1, output_dir=str(root / 'first'))
    first = train(TrainingConfig.from_dict(raw, source=config.source))
    old_identity = first['identity']
    old_policy = first['policy']
    monkeypatch.setattr(session, '_source', lambda: 'a' * 64)
    monkeypatch.setattr(planning, 'source_identity', lambda: 'b' * 64)
    raw['trainer'].update(cycles=4, resume_from=first['checkpoint'], output_dir=str(root / 'resumed'))
    resumed = train(TrainingConfig.from_dict(raw, source=config.source))
    raw['trainer'].pop('resume_from'); raw['trainer']['output_dir'] = str(root / 'whole')
    whole = train(TrainingConfig.from_dict(raw, source=config.source))
    a = torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(whole['checkpoint']) / 'state.pt', weights_only=True)
    for key in ('state', 'rng', 'versions', 'cycle', 'cursor'):
        assert equal(a[key], b[key]), key
    assert resumed['identity']['framework'] != old_identity['framework']
    assert resumed['policy'] == old_policy  # Original source provenance is retained.


@pytest.mark.parametrize('field', ['config', 'dataset', 'model', 'algorithm', 'model_metadata', 'resume_contract'])
def test_minor_fix_compatibility_does_not_relax_model_or_training_identity(field):
    from distillschedule.training.compatibility import same_training_identity, TRAINING_CONTRACT
    original = dict(framework='a' * 64, config='config', dataset='dataset', model={},
                    algorithm={}, model_metadata={}, resume_contract=TRAINING_CONTRACT)
    current = deepcopy(original); current['framework'] = 'b' * 64
    assert same_training_identity(original, current)
    current[field] = 'changed'
    assert not same_training_identity(original, current)


def test_unversioned_unknown_source_is_not_implicitly_compatible():
    from distillschedule.training.compatibility import same_training_identity, execution_source_compatible
    assert not same_training_identity({'framework': 'a' * 64}, {'framework': 'b' * 64})
    assert not execution_source_compatible({'source': 'a' * 64}, 'b' * 64)


@pytest.mark.parametrize('source', [[], {}, None, 4, 'bad-hash'])
def test_malformed_policy_source_returns_incompatible(source):
    from distillschedule.training.compatibility import execution_source_compatible
    assert execution_source_compatible({'source': source}, 'b' * 64) is False


def test_old_release_contract_is_not_promoted_by_a_future_epoch(monkeypatch):
    from distillschedule.training import compatibility as c
    old_training = 'b3a92414c8e360ea9f231962e67a0fdf65ec4abaed12d929b8c5a5c9d692e69a'
    old_execution = 'fe1678279c8af3d8e41a890e898be99e5a448f4df91fa5b1df46df20dba20ccb'
    assert c._LEGACY_TRAINING[old_training] == 'distillschedule.training_resume.v1'
    assert c._LEGACY_EXECUTION[old_execution] == 'distillschedule.program_resume.v1'
    monkeypatch.setattr(c, 'TRAINING_CONTRACT', 'distillschedule.training_resume.v2')
    monkeypatch.setattr(c, 'EXECUTION_CONTRACT', 'distillschedule.program_resume.v2')
    assert not c.same_training_identity({'framework': old_training},
        {'framework': 'b' * 64, 'resume_contract': c.TRAINING_CONTRACT})
    assert not c.execution_source_compatible({'source': old_execution}, 'b' * 64)
    assert not c.execution_source_compatible({'source': old_execution,
        'resume_contract': 'distillschedule.program_resume.v1'}, old_execution)


def test_runtime_cannot_change_again_mid_process_after_resume(tmp_path, monkeypatch):
    from distillschedule.examples.tiny_av import build_algorithm, build_model
    from distillschedule.open_workload import planning
    from distillschedule.training.policy import CyclePolicy

    def build():
        return build_algorithm(build_model({}), {
            'latent': torch.zeros(1, 4, 1, 2, 2),
            'condition': torch.zeros(1, 4, 1, 1, 1),
        }, {'fake_updates': 1}, cycle=0, seed=7)

    policy = CyclePolicy({'mode': 'reference'})
    policy.run(build(), tmp_path)
    restored = CyclePolicy({'mode': 'reference'}); restored.restore(policy.bound)
    monkeypatch.setattr(planning, 'source_identity', lambda: 'a' * 64)
    restored.prepare_resume(build())
    monkeypatch.setattr(planning, 'source_identity', lambda: 'b' * 64)
    with pytest.raises(ValueError, match='source'):
        restored.run(build(), tmp_path)
