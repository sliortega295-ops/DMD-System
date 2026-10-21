"""Real Gloo training/restart, not simulated ranks or a GPU performance claim."""
from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillschedule.data import prepare
from distillschedule.examples.tiny_av import build_fixture
from distillschedule.training.config import TrainingConfig, load_training_config
from test_training_end_to_end import equal


def _worker(rank, rendezvous, config_path, output, cycles, reduction, resume, world=2):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE=str(world), RANK=str(rank), LOCAL_RANK=str(rank % 2),
                      LOCAL_WORLD_SIZE="2", GROUP_RANK=str(rank // 2))
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=world, timeout=timedelta(seconds=90))
    try:
        from distillschedule.training.session import train
        config = load_training_config(config_path)
        raw = config.to_dict()
        raw["data"]["sharding"] = "rank_strided"
        raw["trainer"].update(cycles=cycles, output_dir=output)
        raw["execution"] = {"mode": "manual", "implementation": {
            "backend": "torch-callable-replicated-v1", "reduction": reduction,
            "bucket_cap_mb": 1 if reduction == "ddp_overlap" else 0,
            "local": {"backend": "torch-callable-resident-v1", "share_immutable": False,
                      "checkpoint_bindings": []}}}
        if reduction.startswith("hsdp2"):
            raw["execution"]["implementation"]["node_groups"] = [[0, 1], [2, 3]]
        if reduction == "auto":
            raw["execution"] = {"mode": "auto", "memory_limit_bytes": 10**8,
                                "max_candidates": 2, "samples": 1, "max_program_runs": 6}
        if resume:
            raw["trainer"]["resume_from"] = resume
        result = train(TrainingConfig.from_dict(raw, source=config.source))
        assert result["completed_cycles"] == cycles
        assert result["world_size"] == world
        assert result["final_versions"]["student"] == cycles
        assert result["final_versions"]["fake"] == cycles * 2
        assert result["rank_count"] == world
        if reduction != "auto":
            assert result["policy"]["implementation"]["reduction"] == reduction
        else:
            assert result["policy"]["initial_reference_gate"]["status"] == "PASS"
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _launch(tmp_path, config, name, cycles, reduction, resume=None, world=2):
    output = tmp_path / name
    mp.spawn(_worker, args=(str(tmp_path / (name + "-rdzv")), str(config), str(output),
                            cycles, reduction, str(resume) if resume else None, world), nprocs=world, join=True)
    return output


@pytest.mark.parametrize("reduction", ["post_backward", "ddp_overlap", "fsdp2_reshard"])
def test_real_distributed_continuation_preserves_every_shard_and_sample(tmp_path, reduction):
    if reduction.startswith("fsdp2") and tuple(map(int, torch.__version__.split(".")[:2])) < (2, 10):
        pytest.skip("CPU FSDP2 requires Torch >= 2.10")
    media = tmp_path / "media"
    build_fixture(media, samples=8)
    prepare(media / "raw/manifest.json", media / "prepared")
    config = media / "train.json"
    first = _launch(tmp_path, config, "first", 2, reduction)
    checkpoint = first / "checkpoints/cycle-000002"
    resumed = _launch(tmp_path, config, "resumed", 4, reduction, checkpoint)
    whole = _launch(tmp_path, config, "whole", 4, reduction)
    for rank in range(2):
        a = torch.load(resumed / f"checkpoints/cycle-000004/rank-{rank:05d}.pt", weights_only=True)
        b = torch.load(whole / f"checkpoints/cycle-000004/rank-{rank:05d}.pt", weights_only=True)
        for key in ("state", "rng", "cycle", "cursor", "versions", "policy"):
            assert equal(a[key], b[key]), (rank, key)
        rows = [json.loads(line) for line in (resumed / f"ranks/rank-{rank:05d}/metrics.jsonl").read_text().splitlines()]
        assert [row["sample_id"] for row in rows] == [f"sample-{cycle * 2 + rank:06d}" for cycle in (2, 3)]
        assert all(row["execution"]["effects"]["lifecycle_installations"] == 1 for row in rows)
    manifest = json.loads((resumed / "checkpoints/cycle-000004/manifest.json").read_text())
    assert manifest["topology"]["world_size"] == 2
    assert set(manifest["files"]) == {"rank-00000.pt", "rank-00001.pt"}
    # A directory with only one rank's data is never a valid checkpoint.
    (resumed / "checkpoints/cycle-000004/rank-00001.pt").unlink()
    from distillschedule.training.checkpoint import read_checkpoint
    with pytest.raises((ValueError, FileNotFoundError), match="shard|topology"):
        read_checkpoint(resumed / "checkpoints/cycle-000004", identity=manifest["identity"])


@pytest.mark.parametrize("reduction,world", [("hsdp2_reshard", 4), ("auto", 2)])
def test_hsdp_and_measured_auto_keep_the_selected_layout_across_restart(tmp_path, reduction, world):
    media = tmp_path / "media"
    build_fixture(media, samples=world * 3)
    prepare(media / "raw/manifest.json", media / "prepared")
    first = _launch(tmp_path, media / "train.json", "first", 2, reduction, world=world)
    resumed = _launch(tmp_path, media / "train.json", "resumed", 3, reduction,
                       first / "checkpoints/cycle-000002", world=world)
    whole = _launch(tmp_path, media / "train.json", "whole", 3, reduction, world=world)
    for rank in range(world):
        a = torch.load(resumed / f"checkpoints/cycle-000003/rank-{rank:05d}.pt", weights_only=True)
        b = torch.load(whole / f"checkpoints/cycle-000003/rank-{rank:05d}.pt", weights_only=True)
        for key in ("state", "rng", "cycle", "cursor", "versions"):
            assert equal(a[key], b[key]), (rank, key)
    if reduction.startswith("hsdp2"):
        assert equal(a["policy"], b["policy"])
    else:
        # Independent calibrations have different absolute paths/timings, but
        # a restart must use its original saved selection without recalibration.
        assert not (resumed / "planning").exists()
        assert a["policy"]["plan_sha256"] == json.loads((first / "summary.json").read_text())["policy"]["plan_sha256"]


def _rejection_worker(rank, rendezvous, output):
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE="2", GROUP_RANK="0")
    torch.set_num_threads(1)
    torch.manual_seed(13)
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        from copy import deepcopy
        from distillschedule.training.checkpoint import pack_state, unpack_state, save_checkpoint
        from distillschedule.open_workload.replicated import control_plane, gather, broadcast
        metadata = dist.new_group(backend="gloo")
        try:
            with control_plane(metadata):
                assert gather(rank) == [0, 1]
                assert broadcast("rank0" if rank == 0 else None) == "rank0"
                assert dist.get_backend() == "gloo"
        finally:
            dist.destroy_process_group(metadata)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        state = {"model": model, "optimizer": optimizer}
        before = deepcopy(model.state_dict())
        payload = pack_state(state)
        if rank == 1:
            payload["optimizer"]["parameters"][0].reverse()
        with pytest.raises(ValueError, match="ownership"):
            unpack_state(payload, state)
        assert equal(before, model.state_dict())
        real_save = torch.save
        if rank == 1:
            def fail_write(*args, **kwargs):
                raise OSError("injected rank1 disk failure")
            torch.save = fail_write
        try:
            with pytest.raises(ValueError, match="rank1 disk failure"):
                save_checkpoint(Path(output) / "never-complete", state, identity={"test": "CPU_ONLY"},
                                 cycle=1, cursor=2, versions={"model": 1, "optimizer": 1})
        finally:
            torch.save = real_save
        dist.barrier()
        assert not (Path(output) / "never-complete").exists()
        assert not list(Path(output).glob(".never-complete.tmp-*/_SUCCESS"))
    finally:
        dist.destroy_process_group()


def test_rank_local_rejection_is_collective_and_cannot_publish_partial_checkpoint(tmp_path):
    mp.spawn(_rejection_worker, args=(str(tmp_path / "rdzv"), str(tmp_path)), nprocs=2, join=True)


def test_cuda_rng_checkpoint_queries_only_the_owned_device(monkeypatch):
    from distillschedule.training.checkpoint import rng_state, restore_rng
    expected = torch.arange(8, dtype=torch.uint8)
    observed = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: expected.clone() if device == 3 else pytest.fail("foreign GPU RNG"))
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("initializes unrelated GPU contexts"))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value, device: observed.append((value.clone(), device)))
    state = rng_state()
    restore_rng(state)
    assert len(observed) == 1 and torch.equal(observed[0][0], expected) and observed[0][1] == 3
