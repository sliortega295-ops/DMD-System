"""Real W1 lifecycle with scalar CPU math; NOT pretrained/CUDA/world16 proof.

The producer, lowerer, worker, dispatcher, checkpoint IO and receipt consumers
are real. Physical assets/CUDA/peer-rank transport are the existing explicit
SYNTHETIC_TEST_ONLY fixture. The clean-source bit is a public-wiring test shim,
never a source-cleanliness claim; worker source HEAD/fingerprint remain real.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from distillschedule.strategies import wan_worker
from distillschedule.training import native
from distillschedule.training.checkpoint import rng_state
from distillschedule.training.config import TrainingConfig
from distillschedule.workloads.best_known.registry import file_sha256
from test_strategy_wan_bundle_worker import cpu_worker, _state  # noqa: F401
from test_strategy_wan_bundle_stream import native_cpu_bundle  # noqa: F401
from test_training_end_to_end import equal
from test_training_native_presets import clean_source
from test_training_native_stream_horizon import configuration


def _rank_receipts(root, receipt):
    """Collector fixtures, explicitly not independently executed peer ranks."""
    root.mkdir()
    for rank in range(16):
        (root / f'rank_{rank:02d}.json').write_text(json.dumps(dict(receipt, rank=rank)))


def test_public_stop_and_fresh_process_resume_keep_horizon_state_rng_and_order(cpu_worker, monkeypatch):
    h = cpu_worker
    clean_source(monkeypatch)
    full = h.run('full', public=True)
    expected, expected_rng = _state(h.parts[-1]), rng_state()
    assert full['status'] == 'COMPLETED_ORIGINAL_WAN_WORK'
    before = len(h.begins)
    stopped = h.run('first', stop=1, public=True)
    assert stopped['status'] == 'STOPPED_ORIGINAL_WAN_AT_CHECKPOINT'
    assert stopped['training_segment'] == dict(horizon_cycles=3, start_cycle=0,
        requested_stop_cycle=1, completed_cycle=1, executed_cycles=1, complete_horizon=False)
    assert len(h.begins) == before + 1
    assert h.parts[-1].seen == h.parts[0].seen[:11]
    assert [row['cycle'] for row in stopped['cycles']] == [1]
    assert stopped['backend_effects']['candidate_bundle_effects']['actual_compute_observed']
    checkpoint = Path(stopped['training_restart']['latest_checkpoint'])
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    assert (checkpoint / '_SUCCESS').read_text().strip() == file_sha256(checkpoint / 'manifest.json')
    assert manifest['cycle'] == manifest['cursor'] == 1 and len(manifest['files']) == 16
    assert all(file_sha256(checkpoint / name) == sha for name, sha in manifest['files'].items())
    assert sorted(p.name for p in checkpoint.parent.iterdir()) == ['cycle-000001']
    for key in ('identity', 'strategy', 'task', 'stream_sha256'):
        assert stopped[key] == full[key]
    for key in ('checkpoint_identity', 'checkpoint_policy'):
        assert stopped['training_restart'][key] == full['training_restart'][key]
    assert not stopped['formal_speedup_eligible']
    assert json.loads((h.root / 'first/rank_00.json').read_text()) == json.loads(json.dumps(stopped))

    # A different interpreter constructs fresh owners, restores actual shards,
    # then returns through the same public producer/worker receipt interface.
    script = '''
import json, sys, torch
from pathlib import Path
from _pytest.monkeypatch import MonkeyPatch
from test_strategy_wan_bundle_stream import native_cpu_bundle
from test_strategy_wan_bundle_worker import cpu_worker, _state
from test_training_native_presets import clean_source
from distillschedule.training.checkpoint import rng_state
patch = MonkeyPatch()
try:
    f = native_cpu_bundle.__wrapped__(patch)
    h = cpu_worker.__wrapped__(f, patch, Path(sys.argv[1]))
    clean_source(patch)
    receipt = h.run('resumed', resume=Path(sys.argv[2]), public=True)
    torch.save(dict(state=_state(h.parts[-1]), rng=rng_state(), seen=h.parts[-1].seen,
        receipt=receipt, first=h.begins[0]), sys.argv[3])
finally:
    patch.undo()
'''
    completed = subprocess.run([sys.executable, '-c', script, str(h.root), str(checkpoint), str(h.root / 'child.pt')],
        env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(Path(__file__).parent),
            str(Path(__file__).resolve().parents[1] / 'src'), os.environ.get('PYTHONPATH', '')])},
        text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    child = torch.load(h.root / 'child.pt', weights_only=False)
    assert equal(expected, child['state']) and equal(expected_rng, child['rng'])
    assert h.parts[-1].seen + child['seen'] == h.parts[0].seen
    resumed = child['receipt']
    assert child['first']['cold'] and child['first']['index'] == 1
    assert resumed['status'] == 'COMPLETED_ORIGINAL_WAN_WORK'
    assert resumed['training_segment'] == dict(horizon_cycles=3, start_cycle=1,
        requested_stop_cycle=3, completed_cycle=3, executed_cycles=2, complete_horizon=True)
    assert resumed['final_versions'] == full['final_versions'] == dict(generator=3, critic=15, ema=15)
    assert [row['cycle'] for row in resumed['cycles']] == [2, 3]
    assert resumed['training_restart']['checkpoint_identity'] == stopped['training_restart']['checkpoint_identity']
    assert resumed['training_restart']['checkpoint_policy'] == stopped['training_restart']['checkpoint_policy']
    for actual, uninterrupted in zip(resumed['cycles'], full['cycles'][1:], strict=True):
        for key in ('cycle', 'versions', 'cohort', 'bundle_lifecycle', 'call_census', 'prepared',
                    'rollouts', 'generator', 'critic'):
            assert actual[key] == uninterrupted[key], key
    final = Path(resumed['training_restart']['latest_checkpoint'])
    assert json.loads((final / 'manifest.json').read_text())['cycle'] == 3
    payload = torch.load(final / 'rank-00000.pt', weights_only=True)
    assert equal(expected, payload['state']) and equal(expected_rng, payload['rng'])

    # Existing comparison rejects partial/unequal segments instead of
    # promoting either a graceful stop or final resumed segment to a full run.
    from distillschedule.strategies.wan_evidence import compare_runs
    for name, receipt in (('collected-full', full), ('collected-stop', stopped), ('collected-resume', resumed)):
        _rank_receipts(h.root / name, receipt)
    assert compare_runs(h.root / 'collected-full', h.root / 'collected-full')['status'] == 'PASS'
    for name in ('collected-stop', 'collected-resume'):
        with pytest.raises(ValueError, match='unmatched or incomplete'):
            compare_runs(h.root / 'collected-full', h.root / name)


@pytest.mark.parametrize('stop,every', [(0, 50), (101, 50), (True, 50), (50.0, 50), (50, 0), (49, 50)])
def test_public_invalid_stop_or_cadence_rejected(tmp_path, stop, every):
    raw = configuration(tmp_path, 'W1', optimized=True).to_dict()
    raw['trainer'].update(stop_after_cycle=stop, checkpoint_every=every)
    with pytest.raises(ValueError, match='stop_after_cycle.*checkpoint boundary'):
        native._options(TrainingConfig.from_dict(raw, source=tmp_path / 'training.json'), 'W1')


@pytest.mark.parametrize('optimized', [False, True])
def test_public_horizon100_stop50_does_not_shorten_identity(tmp_path, optimized):
    raw = configuration(tmp_path, 'W1', optimized=optimized).to_dict()
    raw['trainer']['stop_after_cycle'] = 50
    config = TrainingConfig.from_dict(raw, source=tmp_path / 'training.json')
    assert native._options(config, 'W1') == ('manual' if optimized else 'reference')
    assert config.trainer['cycles'] == 100 and config.trainer['stop_after_cycle'] == 50
    assert native.describe_native(config)['capabilities']['stop']['checkpoint_required']


def test_explicit_strategy_lowerer_and_existing_cli_forward_stop(cpu_worker, capsys):
    """Non-preset/CLI route uses the same production worker and checkpoint."""
    h = cpu_worker
    raw = h.configuration('cli', stop=1).to_dict()
    raw['execution'].pop('preset')
    strategy = h.root / 'strategy.json'
    strategy.write_text(json.dumps(h.f.strategy.to_dict()))
    raw['execution']['implementation'] = str(strategy)
    config = TrainingConfig.from_dict(raw, source=h.root / 'training.json')
    report = native.validate_native(config)
    args = report['native_argv']
    assert args[args.index('--stop-after-cycle') + 1] == '1'
    assert args[args.index('--checkpoint-every') + 1] == '1'
    assert '--cycles' not in args  # Full horizon comes from the untouched stream.
    h.transport.output = h.root / 'cli'
    from distillschedule.strategies.wan_cli import main
    assert main(args) == 0
    receipt = json.loads((h.root / 'cli/rank_00.json').read_text())
    assert receipt['status'] == 'STOPPED_ORIGINAL_WAN_AT_CHECKPOINT'
    assert receipt['training_segment']['horizon_cycles'] == 3
    assert receipt['training_segment']['executed_cycles'] == 1
    assert not h.transport.initialized


def test_shortening_horizon_still_rejects_the_same_stream_before_model(cpu_worker):
    h = cpu_worker
    raw = h.configuration('shortened', stop=1).to_dict()
    raw['trainer']['cycles'] = 2
    with pytest.raises(ValueError, match='stream source or requested cycles differ'):
        native.validate_native(TrainingConfig.from_dict(raw, source=h.root / 'training.json'))
    with pytest.raises(ValueError, match='complete fresh input sequence'):
        wan_worker.run_worker(h.f.task, h.assets, output=h.root / 'shortened-worker', mode='train',
            strategy=h.f.strategy, cycles=2, stream_path=h.root / 'stream/stream.json',
            checkpoint_every=1, stop_after_cycle=1)
    assert not h.parts and not h.begins


def test_terminal_explicit_stop_publishes_final_checkpoint_off_cadence(cpu_worker):
    h = cpu_worker
    receipt = h.run('terminal', stop=3, every=2)
    assert receipt['status'] == 'COMPLETED_ORIGINAL_WAN_WORK'
    assert receipt['training_segment']['complete_horizon']
    assert h.boundaries == [2, 3]
    assert Path(receipt['training_restart']['latest_checkpoint']).name == 'cycle-000003'


def test_export_discards_stop_control_without_changing_horizon(cpu_worker, monkeypatch):
    h = cpu_worker
    raw = h.configuration('first', stop=1).to_dict()
    raw['trainer']['export'] = {'scope': 'SYNTHETIC_TEST_ONLY_ADMISSION_BOUNDARY'}
    config = TrainingConfig.from_dict(raw, source=h.root / 'training.json')
    class OutputAdmissionReached(Exception):
        pass
    def admit(projected):
        assert 'stop_after_cycle' not in projected.trainer
        assert projected.trainer['cycles'] == 3 and projected.trainer['checkpoint_every'] == 0
        assert projected.data == config.data and projected.execution == config.execution
        raise OutputAdmissionReached
    monkeypatch.setattr(native, '_admit', admit)
    with pytest.raises(OutputAdmissionReached):
        native.export_native(config, h.root / 'checkpoint', h.root / 'export')
    assert not h.parts and not h.begins


def test_stop_does_not_widen_frozen_two_cycle_wan_or_helios(tmp_path):
    for workload in ('W1', 'W2'):
        raw = configuration(tmp_path, workload, optimized=True).to_dict()
        raw['trainer'].update(cycles=2, checkpoint_every=1, stop_after_cycle=1)
        if workload == 'W1':
            raw['execution']['preset'] = 'wan_ar.incumbent21a.v1'
            raw['trainer']['checkpoint_every'] = 0
            raw['data'] = {'replay_fixed_trace': True}
        else:
            # W2 fresh streams now support stop; its historical benchmark does not.
            raw['execution']['preset'] = 'helios_nogan.safe_ed8.v1'
            raw['trainer']['checkpoint_every'] = 0
            raw['data'] = {}
        with pytest.raises(ValueError, match='stop_after_cycle requires an explicit'):
            native._options(TrainingConfig.from_dict(raw, source=tmp_path / 'training.json'), workload)


@pytest.mark.parametrize('stop,every', [(0, 1), (4, 1), (True, 1), (1.0, 1), (1, 0), (1, 2)])
def test_worker_invalid_stop_rejected_before_allocation(cpu_worker, stop, every):
    h = cpu_worker
    with pytest.raises(ValueError, match='stop_after_cycle.*checkpoint boundary'):
        h.run('invalid-stop', stop=stop, every=every)
    assert not h.parts and not h.begins and not (h.root / 'invalid-stop').exists()


@pytest.mark.parametrize('stop', [1, 2])
def test_stop_not_after_resume_is_rejected_before_model_or_updates(cpu_worker, stop):
    h = cpu_worker
    stopped = h.run('first', stop=2)
    checkpoint = Path(stopped['training_restart']['latest_checkpoint'])
    before = (len(h.parts), len(h.begins))
    with pytest.raises(ValueError, match='stop_after_cycle.*resume'):
        h.run('invalid-resume', stop=stop, resume=checkpoint)
    assert (len(h.parts), len(h.begins)) == before
    assert not (h.root / 'invalid-resume/checkpoints').exists()


def test_missing_peer_shard_cannot_become_a_successful_stop(cpu_worker, monkeypatch):
    h = cpu_worker
    gather = dist.all_gather_object
    def missing_peer(rows, value, *, group=None):
        gather(rows, value, group=group)
        if isinstance(value, dict) and set(value) == {'rank-00000.pt'}:
            rows[15] = {}
    monkeypatch.setattr(dist, 'all_gather_object', missing_peer)
    with pytest.raises(ValueError, match='checkpoint shards incomplete'):
        h.run('failed-publication', stop=1)
    receipt = json.loads((h.root / 'failed-publication/rank_00.json').read_text())
    assert receipt['status'] == 'FAILED_ORIGINAL_WAN_WORK'
    assert receipt['training_restart']['latest_checkpoint'] is None
    assert [row['cycle'] for row in receipt['cycles']] == [1]
    assert not list((h.root / 'failed-publication').rglob('_SUCCESS'))
    assert len(h.begins) == 1


def test_rank_stop_mismatch_rejected_before_collective_checkpoint_branches():
    from types import SimpleNamespace as NS
    def gather(rows, value, *, group):
        rows[:] = [deepcopy(value) for _ in rows]
        rows[8]['controls']['stop_after_cycle'] = 100
    distributed = NS(all_gather_object=gather, get_backend=lambda group: 'gloo')
    with pytest.raises(ValueError, match='rank.*control'):
        wan_worker.admit_run_controls(distributed, object(), mode='train', cycles=100,
            checkpoint_every=50, resume_from=None, stop_after_cycle=50)


def test_legacy_fixed_trace_rejects_stop_before_any_output(tmp_path, monkeypatch):
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.strategies.wan_strategy import WanStrategy
    for key, value in dict(WORLD_SIZE=16, LOCAL_WORLD_SIZE=8, RANK=0, LOCAL_RANK=0).items():
        monkeypatch.setenv(key, str(value))
    assets = wan_worker.OriginalWanAssets('unused', 'unused', 'a'*64, 'unused', 'b'*64, 'unused')
    with pytest.raises(ValueError, match='stop_after_cycle.*fresh'):
        wan_worker.run_worker(OriginalWanTask(), assets, mode='manual', cycles=2,
            strategy=WanStrategy('fsdp', 512, False), stop_after_cycle=1, output=tmp_path / 'untouched')
    assert not (tmp_path / 'untouched').exists()
