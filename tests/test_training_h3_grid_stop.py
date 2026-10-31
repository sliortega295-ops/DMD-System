"""W5 durable stop controls; CPU lifecycle evidence, not pretrained H3 parity."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path

import pytest

from distillschedule.training.config import TrainingConfig
from distillschedule.training import native
from test_training_h3_grid import training


def test_public_horizon50_stop25_keeps_original_identity(tmp_path):
    from distillschedule.training.h3_grid_training import training_identity
    config = training(tmp_path, cycles=50, checkpoint_every=25, stop_after_cycle=25)
    before = config.to_dict()
    assert native._options(config, "W5") == "reference"
    assert config.to_dict() == before and config.trainer["cycles"] == 50
    stop = native.describe_native(config)["capabilities"]["stop"]
    assert stop["checkpoint_required"] and stop["immutable_horizon"]
    assets = {"native_config": {"sha256": "SYNTHETIC_IDENTITY_ONLY"}}
    inputs = {"inventory_sha256": "SYNTHETIC_IDENTITY_ONLY"}
    assert training_identity("source", assets, inputs, config.trainer["cycles"]) == training_identity(
        "source", assets, inputs, training(tmp_path, cycles=50).trainer["cycles"])


@pytest.mark.parametrize("stop,every", [(0, 25), (51, 25), (True, 25), (25.0, 25),
    ("25", 25), (None, 25), (25, 0), (24, 25)])
def test_invalid_stop_is_refused_before_assets(tmp_path, stop, every):
    with pytest.raises(ValueError, match="stop_after_cycle.*checkpoint boundary"):
        native._options(training(tmp_path, cycles=50, checkpoint_every=every, stop_after_cycle=stop), "W5")


def test_stop_does_not_admit_optimized_or_frozen_w5(tmp_path):
    raw = training(tmp_path, stop_after_cycle=1).to_dict()
    raw["execution"].update(mode="plan", plan="failed-optimized-plan.json")
    with pytest.raises(ValueError, match="reference"):
        native._options(TrainingConfig.from_dict(raw, source=tmp_path / "request.json"), "W5")
    raw["data"] = {"replay_fixed_trace": True}
    raw["trainer"].pop("seed")
    with pytest.raises(ValueError, match="stop_after_cycle requires an explicit"):
        native._options(TrainingConfig.from_dict(raw, source=tmp_path / "request.json"), "W5")


class _BoundaryClock:
    """Control-only fixture, not an H3 optimizer or numerical state substitute."""
    def __init__(self, lifecycle):
        self.lifecycle = lifecycle

    def state_dict(self):
        return {"cycle": self.lifecycle.completed}

    def load_state_dict(self, value):
        assert value == self.state_dict()

    def validate_checkpoint_boundary(self):
        assert self.lifecycle.cursor.cursor == 6 * self.lifecycle.completed


def _loop(tmp_path, *, stop=None, every=1):
    """Reuse the existing source-extracted native-loop control fixture.

    The real lifecycle and durable writer run; setup, math and CUDA are the
    explicitly synthetic existing fixture. Actual upstream tests stay separate.
    """
    from test_refactor_h3_planner_cycle_observer import _native_fixture
    from distillschedule.training.h3_grid_training import GridTrainingLifecycle
    config = training(tmp_path, checkpoint_every=every, **({"stop_after_cycle": stop} if stop is not None else {}))
    fixture = _native_fixture(observed=False)
    trainer = fixture.trainer
    trainer.max_train_iters = 3
    trainer.save_every_iters = every
    trainer.output_train_dir = config.trainer["output_dir"]
    lifecycle = GridTrainingLifecycle(config, {}, {"scope": "CPU_LOOP_CONTROL_ONLY", "horizon": 3})
    class Cursor:
        cursor, epoch = 0, 0
        def __next__(self):
            value = {"meta": {"id": self.cursor}}
            self.cursor += 1
            return value
    def after_setup(owner):
        assert owner is trainer
        lifecycle.cursor = Cursor()
        lifecycle.state = {"clocks": _BoundaryClock(lifecycle)}
        return lifecycle.cursor
    lifecycle.after_setup = after_setup
    trainer.grid_training_lifecycle = lifecycle
    return fixture, lifecycle


@pytest.mark.parametrize("stop,every,expected,saved", [
    (1, 1, 1, [1]), (3, 2, 3, [2, 3]), (None, 2, 3, [2]), (None, 0, 3, [])])
def test_real_loop_seam_stops_only_after_durable_publication_and_defaults_unchanged(
        tmp_path, stop, every, expected, saved):
    import torch
    from distillschedule.workloads.best_known.registry import file_sha256
    fixture, life = _loop(tmp_path, stop=stop, every=every)
    fixture.trainer.train()
    assert life.completed == expected and life.cursor.cursor == expected * 6
    assert fixture.actual_updates == {"student": expected, "fake": expected * 5}
    assert fixture.trainer.max_train_iters == life.identity["horizon"] == 3
    root = Path(fixture.trainer.output_train_dir)
    assert sorted(p.name for p in (root / "checkpoints").glob("cycle-*")) == [f"cycle-{i:06d}" for i in saved]
    for cycle in saved:
        checkpoint = root / "checkpoints" / f"cycle-{cycle:06d}"
        assert (checkpoint / "_SUCCESS").read_text().strip() == file_sha256(checkpoint / "manifest.json")
        header = json.loads((checkpoint / "manifest.json").read_text())
        assert header["cycle"] == cycle and header["cursor"] == cycle * 6
        assert all(file_sha256(checkpoint / name) == sha for name, sha in header["files"].items())
    receipt = json.loads((root / "ranks/rank-00000.json").read_text())
    assert receipt["status"] == ("STOPPED_AT_CHECKPOINT" if expected < 3 else "COMPLETED")
    assert receipt["training_segment"] == dict(horizon_cycles=3, start_cycle=0,
        requested_stop_cycle=expected, completed_cycle=expected, executed_cycles=expected,
        complete_horizon=expected == 3)
    assert receipt["known_optimized_backward_gate"] == "FAIL_UNCHANGED"
    assert receipt["numerical_cross_reference"] == "NOT_RUN" and not receipt["claimable_speedup"]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("failure", ["raises", "unpublished"])
def test_checkpoint_failure_cannot_return_a_successful_stop(tmp_path, monkeypatch, failure):
    from distillschedule.training import checkpoint
    fixture, life = _loop(tmp_path, stop=1)
    def fail(*args, **kwargs):
        if failure == "raises":
            raise OSError("injected incomplete checkpoint publication")
        return None
    monkeypatch.setattr(checkpoint, "save_checkpoint", fail)
    with pytest.raises((OSError, ValueError), match="publication|published complete checkpoint"):
        fixture.trainer.train()
    assert life.completed == 1 and fixture.actual_updates == {"student": 1, "fake": 5}
    assert not list(tmp_path.rglob("_SUCCESS")) and not list(tmp_path.rglob("rank-00000.json"))


@pytest.mark.parametrize("stop", [1, 2])
def test_stale_stop_is_refused_by_pre_model_checkpoint_header(tmp_path, stop):
    from distillschedule.training.h3_grid_training import GridTrainingLifecycle
    fixture, life = _loop(tmp_path, stop=2)
    fixture.trainer.train()
    path = Path(fixture.trainer.output_train_dir) / "checkpoints/cycle-000002"
    config = training(tmp_path, stop_after_cycle=stop, resume_from=str(path))
    resumed = GridTrainingLifecycle(config, {}, deepcopy(life.identity))
    with pytest.raises(ValueError, match="stop_after_cycle.*resume"):
        resumed.checkpoint_header()
    assert resumed.state is None and resumed.cursor is None


def _stop_rank(rank, root, scenario):
    import torch
    import torch.distributed as dist
    from distillschedule.training.h3_grid_training import collective_admission
    root = Path(root)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        if scenario == "explicitness":
            controls = {"stop_after_cycle": 3} if rank == 1 else {}
        else:
            controls = {"stop_after_cycle": 1 if rank == 0 else (2 if scenario == "mismatch" else 0)}
        config = training(root, **controls)
        def admitted():
            return "W5", {"identity": {"head": "CONTROL_FIXTURE_ONLY", "dirty": False}}, {}, {
                "assets": {"native_config": {"sha256": "CONTROL_FIXTURE_ONLY"}},
                "inputs": {"inventory_sha256": "CONTROL_FIXTURE_ONLY"}}
        with pytest.raises(ValueError, match="rank failures|ranks disagree.*stop_after_cycle") as error:
            collective_admission(config, admitted)
        (root / f"rank-{rank}.json").write_text(json.dumps({"error": str(error.value),
            "cuda_initialized": torch.cuda.is_initialized()}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scenario", ["mismatch", "invalid", "explicitness"])
def test_real_gloo_stop_disagreement_rejected_before_owner_construction(tmp_path, scenario):
    import torch.multiprocessing as mp
    mp.spawn(_stop_rank, args=(str(tmp_path), scenario), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert rows[0] == rows[1] and rows[0]["cuda_initialized"] is False


def _checkpoint_rank(rank, root, failure):
    import os
    import torch
    import torch.distributed as dist
    from distillschedule.workloads.best_known.registry import file_sha256
    root = Path(root)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2", GROUP_RANK="0")
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    save = torch.save
    try:
        fixture, life = _loop(root, stop=1)
        if failure and rank == 1:
            def fail(*args, **kwargs):
                raise OSError("rank1 checkpoint shard write failure")
            torch.save = fail
        if failure:
            with pytest.raises(ValueError, match="rank1 checkpoint shard write failure") as error:
                fixture.trainer.train()
            message = str(error.value)
            assert not list(root.rglob("_SUCCESS")) and not (root / "run/ranks").exists()
        else:
            fixture.trainer.train()
            message = None
            path = root / "run/checkpoints/cycle-000001"
            header = json.loads((path / "manifest.json").read_text())
            assert (path / "_SUCCESS").read_text().strip() == file_sha256(path / "manifest.json")
            assert set(header["files"]) == {"rank-00000.pt", "rank-00001.pt"}
            assert all(file_sha256(path / name) == sha for name, sha in header["files"].items())
            receipt = json.loads((root / "run/ranks" / f"rank-{rank:05d}.json").read_text())
            assert receipt["status"] == "STOPPED_AT_CHECKPOINT" and receipt["latest_checkpoint"] == str(path)
        assert life.completed == 1 and life.cursor.cursor == 6
        assert fixture.actual_updates == {"student": 1, "fake": 5}
        (root / f"rank-{rank}.json").write_text(json.dumps({"error": message,
            "cuda_initialized": torch.cuda.is_initialized()}))
    finally:
        torch.save = save
        dist.destroy_process_group()


@pytest.mark.parametrize("failure", [False, True])
def test_real_gloo_loop_stop_waits_for_all_rank_checkpoint_publication(tmp_path, failure):
    import torch.multiprocessing as mp
    mp.spawn(_checkpoint_rank, args=(str(tmp_path), failure), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert rows[0] == rows[1] and rows[0]["cuda_initialized"] is False
