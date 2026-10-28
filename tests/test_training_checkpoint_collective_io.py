"""Real Gloo restore: distributed file hashing, complete admission, local load."""
from collections import Counter
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillschedule.training import checkpoint as cp
from test_training_end_to_end import equal


class _NativeFiles:
    """Opaque native sidecar inventory; the common publisher does not map it."""
    def __init__(self, rank, world):
        self.rank, self.world = rank, world
        self.loads = 0
        self.owner = {"scope": "CPU_TEST_ONLY_OPAQUE_NATIVE_FILES"}

    def validate_checkpoint_boundary(self):
        pass

    def save(self, path):
        path.mkdir(exist_ok=True)
        name = f"opaque-{self.rank}.bin"
        (path / name).write_bytes(f"native-{self.rank}".encode())
        return {"owner": self.owner, "files": {"native/" + name: cp._sha(path / name)}}

    def validate_inventory(self, path, owner):
        assert owner == self.owner
        assert {item.name for item in path.iterdir()} == {f"opaque-{rank}.bin" for rank in range(self.world)}

    def validate_restore(self, path, owner):
        self.validate_inventory(path, owner)

    def load(self, path, owner):
        self.validate_inventory(path, owner)
        self.loads += 1


def _init(rank, world, rendezvous):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE=str(world), RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE=str(world), GROUP_RANK="0")
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=world, timeout=timedelta(seconds=30))


def _state(rank):
    torch.manual_seed(71 + rank)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    model(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"model": model, "optimizer": optimizer}


def _save(path, state, *, native=None):
    return cp.save_checkpoint(path, state, identity={"source": "collective-io-test"},
        cycle=3, cursor=3 * cp.distributed.rank_world()[1], versions={"model": 3},
        policy={"scope": "CPU_REAL_CHECKPOINT"},
        data_progress={"stream": {"cursor": 3}}, native_checkpoint=native)


def _valid_worker(rank, world, rendezvous, output, with_native):
    _init(rank, world, rendezvous)
    root = Path(output)
    try:
        original = _state(rank)
        expected = deepcopy(cp.pack_state(original))
        native = _NativeFiles(rank, world) if with_native else None
        path = _save(root / "checkpoint", original, native=native)
        hashes, loads = [], []
        actual_sha, actual_load = cp._sha, torch.load

        def counted_sha(path):
            hashes.append(str(Path(path).relative_to(root / "checkpoint")))
            return actual_sha(path)

        def counted_load(path, *args, **kwargs):
            loads.append(Path(path).name)
            return actual_load(path, *args, **kwargs)

        cp._sha, torch.load = counted_sha, counted_load
        identity = {"source": "collective-io-test"}
        try:
            # Public direct reads retain their complete all-shard validation.
            public = cp.read_checkpoint(path, identity=identity)
            expected_files = {f"rank-{peer:05d}.pt" for peer in range(world)}
            assert Counter(name for name in hashes if name.endswith(".pt")) == Counter(expected_files)
            hashes.clear(); loads.clear()
            target = _state(rank + 20)
            parameter_ids = [id(parameter) for parameter in target["model"].parameters()]
            payload = cp.restore_checkpoint(path, target, identity=identity, native_checkpoint=native)
        finally:
            cp._sha, torch.load = actual_sha, actual_load
        assert equal(payload, public)
        assert equal(cp.pack_state(target), expected)
        assert equal(cp.rng_state(), payload["rng"])
        assert parameter_ids == [id(parameter) for parameter in target["model"].parameters()]
        assert hashes.count(f"rank-{rank:05d}.pt") == 1
        assert [name for name in hashes if name.endswith(".pt")] == [f"rank-{rank:05d}.pt"]
        assert loads == [f"rank-{rank:05d}.pt"]
        if with_native:
            assert Counter(name for name in hashes if name.startswith("native/")) == Counter(
                f"native/opaque-{peer}.bin" for peer in range(world))
            assert native.loads == 1
        records = cp.distributed.gather({"rank": rank, "hashes": hashes, "loads": loads})
        assert Counter(name for row in records for name in row["hashes"] if name.endswith(".pt")) == Counter(expected_files)
        (root / f"valid-rank-{rank}.json").write_text(json.dumps(records))
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,with_native", [(2, False), (4, False), (2, True)])
def test_collective_restore_hashes_each_common_shard_once_and_restores_exactly(tmp_path, world, with_native):
    mp.spawn(_valid_worker, args=(world, str(tmp_path / "rdzv"), str(tmp_path), with_native),
             nprocs=world, join=True)


def _rejection_worker(rank, rendezvous, output):
    _init(rank, 2, rendezvous)
    root = Path(output)
    try:
        path = _save(root / "checkpoint", _state(rank))
        actual_sha, actual_load, actual_gather = cp._sha, torch.load, cp.distributed.gather
        manifest_bytes = (path / "manifest.json").read_bytes()
        marker_bytes = (path / "_SUCCESS").read_bytes()
        peer_bytes = (path / "rank-00001.pt").read_bytes()
        alias = root / "alias"
        if rank == 0:
            shutil.copytree(path, alias)
        dist.barrier()
        cases = []
        for fault in ("corrupt-peer", "missing-peer", "manifest-disagreement", "local-hash-error",
                      "local-load-error", "payload-progress", "coverage-missing", "coverage-duplicate",
                      "coverage-wrong-rank"):
            if rank == 0:
                (path / "rank-00001.pt").write_bytes(peer_bytes)
                (path / "manifest.json").write_bytes(manifest_bytes)
                (path / "_SUCCESS").write_bytes(marker_bytes)
                if fault == "corrupt-peer":
                    (path / "rank-00001.pt").write_bytes(b"corrupt checkpoint")
                elif fault == "missing-peer":
                    (path / "rank-00001.pt").unlink()
                elif fault == "manifest-disagreement":
                    document = json.loads(manifest_bytes)
                    document["policy"] = {"changed": True}
                    (alias / "manifest.json").write_text(json.dumps(document))
                    (alias / "_SUCCESS").write_text(actual_sha(alias / "manifest.json") + "\n")
                elif fault == "payload-progress":
                    payload = actual_load(path / "rank-00001.pt", weights_only=True)
                    payload["cycle"] += 1
                    torch.save(payload, path / "rank-00001.pt")
                    document = json.loads(manifest_bytes)
                    document["files"]["rank-00001.pt"] = actual_sha(path / "rank-00001.pt")
                    (path / "manifest.json").write_text(json.dumps(document))
                    (path / "_SUCCESS").write_text(actual_sha(path / "manifest.json") + "\n")
            dist.barrier()
            state = _state(rank + 40)
            before, rng = deepcopy(cp.pack_state(state)), cp.rng_state()
            loads = []

            def fail_sha(path):
                if fault == "local-hash-error" and rank == 1 and Path(path).name == "rank-00001.pt":
                    raise OSError("injected local hash read failure")
                return actual_sha(path)

            def fail_load(path, *args, **kwargs):
                loads.append(Path(path).name)
                if fault == "local-load-error" and rank == 1:
                    raise OSError("injected local payload read failure")
                return actual_load(path, *args, **kwargs)

            def wrong_coverage(value):
                gathered = actual_gather(value)
                if isinstance(value, dict) and set(value) == {"rank", "filename", "sha256"}:
                    if fault == "coverage-missing":
                        return gathered[:1]
                    if fault == "coverage-duplicate":
                        return [gathered[0], deepcopy(gathered[0])]
                    if fault == "coverage-wrong-rank":
                        gathered[1]["rank"] = 0
                return gathered

            cp._sha, torch.load, cp.distributed.gather = fail_sha, fail_load, wrong_coverage
            try:
                with pytest.raises(ValueError) as rejected:
                    cp.restore_checkpoint(alias if fault == "manifest-disagreement" and rank == 1 else path,
                                          state, identity={"source": "collective-io-test"})
            finally:
                cp._sha, torch.load, cp.distributed.gather = actual_sha, actual_load, actual_gather
            errors = actual_gather(str(rejected.value))
            assert errors[0] == errors[1], (fault, errors)
            assert equal(cp.pack_state(state), before), fault
            assert equal(cp.rng_state(), rng), fault
            if fault not in ("local-load-error", "payload-progress"):
                assert not loads, (fault, loads)
            value = torch.tensor(rank + 1)
            dist.all_reduce(value)
            assert value.item() == 3, "restore rejection damaged the caller process group"
            cases.append(fault)
        (root / f"rejections-rank-{rank}.json").write_text(json.dumps(cases))
    finally:
        dist.destroy_process_group()


def test_any_rank_rejection_precedes_state_or_rng_mutation(tmp_path):
    mp.spawn(_rejection_worker, args=(str(tmp_path / "rdzv"), str(tmp_path)), nprocs=2, join=True)
    assert json.loads((tmp_path / "rejections-rank-0.json").read_text()) == json.loads(
        (tmp_path / "rejections-rank-1.json").read_text())
