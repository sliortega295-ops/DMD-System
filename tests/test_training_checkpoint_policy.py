"""Public checkpoint cadence, retention and complete-cycle timing on CPU."""
import json
from pathlib import Path

import pytest
import torch

from distillschedule.training.config import TrainingConfig
from distillschedule.training.session import train
from test_training_end_to_end import fixture, equal


def test_default_is_thirty_minutes_and_final_cycle():
    from distillschedule.training.checkpoint import CheckpointPolicy
    policy = CheckpointPolicy.from_trainer({})
    assert policy.keep_last == 3
    assert not policy.due(cycle=1, total=100, elapsed=1799.)
    assert policy.due(cycle=2, total=100, elapsed=1800.)
    assert policy.due(cycle=100, total=100, elapsed=1.)
    explicit = CheckpointPolicy.from_trainer({'checkpoint_every': 2})
    assert not explicit.due(cycle=1, total=4, elapsed=9999.)
    assert explicit.due(cycle=2, total=4, elapsed=1.)
    assert not CheckpointPolicy.from_trainer({'checkpoint_every': 0}).due(
        cycle=4, total=4, elapsed=9999.)


def test_checkpoint_settings_are_configurable_but_not_mathematical_identity(tmp_path):
    from test_training_config import request
    raw = request()
    original = TrainingConfig.from_dict(raw, source=tmp_path / 'config.json')
    raw['trainer'].update(checkpoint_interval_seconds=60., checkpoint_keep_last=5)
    configured = TrainingConfig.from_dict(raw, source=original.source)
    assert configured.identity == original.identity


@pytest.mark.parametrize('field,value', [
    ('checkpoint_interval_seconds', 0), ('checkpoint_interval_seconds', float('inf')),
    ('checkpoint_interval_seconds', True), ('checkpoint_keep_last', -1),
    ('checkpoint_keep_last', True), ('checkpoint_keep_last', 1.5),
])
def test_invalid_checkpoint_settings_fail_before_construction(tmp_path, field, value):
    from test_training_config import request
    raw = request(); raw['trainer'][field] = value
    with pytest.raises(ValueError, match='checkpoint'):
        TrainingConfig.from_dict(raw, source=tmp_path / 'config.json')


def test_retention_keeps_latest_three_and_never_touches_unpublished_or_foreign(tmp_path, monkeypatch):
    from distillschedule.training.checkpoint import save_checkpoint, prune_checkpoints
    root = tmp_path / 'checkpoints'
    state = {'weight': torch.tensor([2.])}
    for cycle in range(1, 6):
        save_checkpoint(root / f'cycle-{cycle:06d}', state, identity={'run': 'ours'},
                        cycle=cycle, cursor=cycle, versions={})
    save_checkpoint(root / 'cycle-000000', state, identity={'run': 'other'},
                    cycle=0, cursor=0, versions={})
    (root / 'cycle-000006').mkdir()  # Incomplete: not eligible for deletion.
    (root / '.cycle-000007.pending').mkdir()
    external = tmp_path / 'outside'; external.mkdir()
    (root / 'cycle-000008').symlink_to(external, target_is_directory=True)
    from distillschedule.training import distributed
    def forbidden():
        raise AssertionError('rank-zero retention entered collective topology discovery')
    monkeypatch.setattr(distributed, 'topology', forbidden)
    removed = prune_checkpoints(root, identity={'run': 'ours'}, keep_last=3)
    assert [Path(p).name for p in removed] == ['cycle-000001', 'cycle-000002']
    assert all((root / f'cycle-{n:06d}').is_dir() for n in [0, 3, 4, 5, 6])
    assert external.exists() and (root / 'cycle-000008').is_symlink()
    assert (root / '.cycle-000007.pending').exists()


def test_default_run_saves_final_only_and_explicit_retention_preserves_training(tmp_path):
    root, config = fixture(tmp_path, samples=6)
    raw = config.to_dict()
    raw['trainer'].pop('checkpoint_every')
    raw['trainer'].update(cycles=5, output_dir=str(root / 'default'))
    default = train(TrainingConfig.from_dict(raw, source=config.source))
    assert sorted(p.name for p in (root / 'default/checkpoints').iterdir()) == ['cycle-000005']
    raw['trainer'].update(checkpoint_every=1, output_dir=str(root / 'frequent'))
    frequent = train(TrainingConfig.from_dict(raw, source=config.source))
    assert sorted(p.name for p in (root / 'frequent/checkpoints').iterdir()) == [
        'cycle-000003', 'cycle-000004', 'cycle-000005']
    a = torch.load(Path(default['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(frequent['checkpoint']) / 'state.pt', weights_only=True)
    assert equal(a, b)
    rows = [json.loads(line) for line in (root / 'frequent/metrics.jsonl').read_text().splitlines()]
    for row in rows:
        assert row['prepublication_wall_s'] >= row['wall_s'] >= 0
        assert row['cycle_timing_scope'] == 'CYCLE_PREPUBLICATION_BEFORE_METRICS_CHECKPOINT_AND_POLICY_PUBLICATION'
        assert 'checkpoint' not in row['timings_s'] and 'publication' not in row['timings_s']
        assert row['timings_s']['finite_check'] >= 0
    assert frequent['timings_s']['checkpoint'] > 0
    assert frequent['job_wall_s'] >= frequent['cycles_wall_s'] >= sum(r['prepublication_wall_s'] for r in rows)
    assert frequent['timings_s']['logging'] >= 0


def test_prepublication_metrics_keep_full_cycle_and_job_costs_in_summary(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from distillschedule.training import session

    root, config = fixture(tmp_path)
    raw = config.to_dict()
    raw['trainer'].update(cycles=2, checkpoint_every=1, checkpoint_keep_last=1, output_dir=str(root / 'timed'))
    clock = [0.]
    monkeypatch.setattr(session, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    original_run = session.CyclePolicy.run
    original_append, original_save = session._append_json, session.save_checkpoint
    original_prune, original_json = session.prune_checkpoints, session._json

    def run(*args, **kwargs):
        value = original_run(*args, **kwargs)
        clock[0] += 2.
        return value

    def append(path, value):
        original_append(path, value)
        if path.name == 'metrics.jsonl':
            clock[0] += 7.

    def save(*args, **kwargs):
        value = original_save(*args, **kwargs)
        clock[0] += 11.
        return value

    def prune(*args, **kwargs):
        value = original_prune(*args, **kwargs)
        clock[0] += 13.
        return value

    def publish(path, value):
        original_json(path, value)
        if path.name == 'effective-policy.json':
            clock[0] += 17.

    monkeypatch.setattr(session.CyclePolicy, 'run', run)
    monkeypatch.setattr(session, '_append_json', append)
    monkeypatch.setattr(session, 'save_checkpoint', save)
    monkeypatch.setattr(session, 'prune_checkpoints', prune)
    monkeypatch.setattr(session, '_json', publish)
    result = train(TrainingConfig.from_dict(raw, source=config.source))
    rows = [json.loads(line) for line in (root / 'timed/metrics.jsonl').read_text().splitlines()]
    assert [row['wall_s'] for row in rows] == [2., 2.]
    assert [row['prepublication_wall_s'] for row in rows] == [2., 2.]
    assert all(row['cycle_timing_scope'] == 'CYCLE_PREPUBLICATION_BEFORE_METRICS_CHECKPOINT_AND_POLICY_PUBLICATION'
               and 'cycle_wall_s' not in row and 'checkpoint' not in row['timings_s']
               and 'publication' not in row['timings_s'] for row in rows)
    assert result['timings_s']['execution'] == 4.
    assert result['timings_s']['logging'] == 14.
    assert result['timings_s']['checkpoint'] == 48.
    assert result['timings_s']['publication'] == 17.
    assert result['cycles_this_process'] == 2
    assert result['cycles_wall_s'] == result['job_wall_s'] == 83.
    assert result['cycles_timing_scope'] == 'ALL_COMPLETED_CYCLES_THROUGH_METRICS_CHECKPOINT_AND_POLICY_PUBLICATION'
    assert json.loads((root / 'timed/summary.json').read_text()) == result


def test_output_conflict_is_rejected_before_model_factory(tmp_path, monkeypatch):
    root, config = fixture(tmp_path)
    Path(config.trainer['output_dir']).mkdir()
    from distillschedule.training import session
    def forbidden(*a, **k):
        raise AssertionError('model construction reached')
    monkeypatch.setattr(session, '_construct_model', forbidden)
    with pytest.raises(FileExistsError):
        train(config)
