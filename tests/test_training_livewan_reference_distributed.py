"""Real two-rank Gloo preflight, NOT world16 NCCL or native CUDA proof."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_training_livewan_reference import (  # actual original-source/native-input fixtures
    sources as sources, prepared as prepared, public as public,
)


def _preflight_worker(rank, rendezvous, raw, source, root, case):
    from distillschedule.training import native, livewan_training as training
    from distillschedule.training.config import TrainingConfig
    from distillschedule.open_workload.replicated import control_plane
    from distillschedule.workloads.best_known.registry import fingerprint
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2", GROUP_RANK="0")
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    groups, reached = [], []
    # Explicit DISPATCH-ONLY facade: physical allocation is unavailable. All
    # config/source/manifest I/O, Gloo exchanges, ownership and failures are real.
    @contextmanager
    def cpu_plane():
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=60)); groups.append(group)
        try:
            with control_plane(group):
                yield {"scope": "CPU_GLOO_DISPATCH_ONLY_NOT_WORLD16"}
        finally:
            dist.destroy_process_group(group)
    training.reference_process_group = cpu_plane
    observed = deepcopy(source); observed["identity"]["dirty"] = False
    if case == "source_disagreement" and rank == 1: observed["identity"]["head"] = "f" * 40
    native._source = lambda conf: deepcopy(observed)  # NO claim of clean committed source qualification
    def model_boundary(*args):
        reached.append("DISPATCH_ONLY_MODEL_BOUNDARY")
        raise RuntimeError("CPU_DISPATCH_ONLY_STOP_BEFORE_ANY_MODEL")
    training._run = model_boundary
    local = deepcopy(raw)
    output = Path(local["trainer"]["output_dir"])
    root = Path(root)
    if rank == 1:
        if case == "missing_assets": local["execution"]["assets"] = str(root / "missing-assets.json")
        elif case == "missing_inputs": local["data"]["prepared"] = str(root / "missing-inputs.json")
        elif case == "changed_inputs":
            value = json.loads(Path(local["data"]["prepared"]).read_text())
            value["rows"][0]["seeds"]["rollout"] += 1
            value["input_sha256"] = fingerprint({k: v for k, v in value.items() if k != "input_sha256"})
            path = root / "changed-inputs.json"; path.write_text(json.dumps(value)); local["data"]["prepared"] = str(path)
        elif case == "control_disagreement": local["trainer"]["checkpoint_every"] = 2
        elif case == "environment": os.environ["LIVEWAN_TRAJECTORY_CYCLES"] = "3"
    dist.barrier()
    conf = TrainingConfig.from_dict(local, source=root / f"rank-{rank}.json")
    error = None
    try:
        native.run_native(conf)
    except Exception as exc:
        error = str(exc)
    reports = [None, None]
    dist.all_gather_object(reports, {"error": error, "reached": len(reached), "groups": len(groups)})
    assert reports[0]["error"] == reports[1]["error"] and reports[0]["error"], reports
    if case == "dispatch_boundary":
        assert all(r["reached"] == 1 and "CPU_DISPATCH_ONLY_STOP_BEFORE_ANY_MODEL" in r["error"] for r in reports), reports
        receipt = json.loads((output / "ranks" / f"rank-{rank:05d}.json").read_text())
        assert receipt["status"] == "FAIL" and not receipt["claimable_speedup"]
    else:
        assert all(r["reached"] == 0 and r["groups"] == 1 for r in reports), reports
        if case == "old_output":
            assert (output / "prior.bin").read_bytes() == b"unowned evidence\x00\xff"
            assert not (output / ".reference-owner").exists()
        else:
            assert not output.exists(), "failed preflight acquired an output or published false success"
        if case in {"missing_assets", "missing_inputs", "changed_inputs", "environment"}:
            assert "rank failures [None," in error, reports
    # Peers really returned from all gates; caller Gloo remains usable and is
    # not destroyed by the lifecycle's independent metadata-plane ownership.
    value = torch.tensor(rank + 1); dist.all_reduce(value)
    assert value.item() == 3 and not torch.cuda.is_initialized()
    (root / f"{case}-{rank}.json").write_text(json.dumps(reports[rank]))
    dist.destroy_process_group()


@pytest.mark.parametrize("case", ["missing_assets", "missing_inputs", "changed_inputs", "source_disagreement",
                                  "control_disagreement", "environment", "old_output", "dispatch_boundary"])
def test_public_rank_local_preflight_failure_never_hangs_peer_or_constructs_model(public, tmp_path, case):
    from distillschedule.training import native
    conf, _, _ = public
    if case == "old_output":
        output = Path(conf.trainer["output_dir"]); output.mkdir()
        (output / "prior.bin").write_bytes(b"unowned evidence\x00\xff")
    mp.spawn(_preflight_worker, args=(str(tmp_path / "rendezvous"), conf.to_dict(), native._source(conf),
                                     str(tmp_path), case), nprocs=2, join=True)


def test_reference_process_owner_rejects_caller_group_without_destroying_it(tmp_path, monkeypatch):
    from distillschedule.training.livewan_training import reference_process_group
    dist.init_process_group("gloo", init_method="file://" + str(tmp_path / "group"), rank=0, world_size=1)
    try:
        for name, value in {"WORLD_SIZE": "16", "RANK": "0", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "8", "GROUP_RANK": "0"}.items():
            monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match="do not enter on an embedding group"):
            with reference_process_group():
                raise AssertionError("borrowed caller group")
        assert dist.is_initialized()
    finally:
        dist.destroy_process_group()
