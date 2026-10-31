"""Real Gloo restore: distributed file hashing, complete admission, local load."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import sys
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillschedule.training import checkpoint as cp
from test_training_end_to_end import equal


@pytest.fixture(autouse=True)
def _checkpoint_io_ci_trace(monkeypatch):
    # Only these workers need the diagnostic control. Keeping it out of the
    # shared CI environment preserves other native providers' strict admission.
    if os.environ.get("GITHUB_ACTIONS") == "true":
        monkeypatch.setenv("DISTILLSCHEDULE_CHECKPOINT_IO_TRACE", "1")


@pytest.mark.parametrize("ci", [False, True])
def test_checkpoint_io_trace_is_scoped_and_restores_the_parent_environment(monkeypatch, ci):
    monkeypatch.delenv("DISTILLSCHEDULE_CHECKPOINT_IO_TRACE", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true" if ci else "false")
    before = dict(os.environ)
    with pytest.MonkeyPatch.context() as scoped:
        _checkpoint_io_ci_trace.__wrapped__(scoped)
        assert os.environ.get("DISTILLSCHEDULE_CHECKPOINT_IO_TRACE") == ("1" if ci else None)
    assert dict(os.environ) == before


def _checkpoint_io_resources():
    def read(path):
        try:
            with Path(path).open() as stream:
                value = stream.read(4097)
            return value if len(value) <= 4096 else {"error": "metadata exceeds 4096 characters"}
        except OSError as error:
            return {"error": type(error).__name__ + ": " + str(error)}

    status = read("/proc/self/status")
    process = {key: value.strip() for line in status.splitlines()
               if ":" in line for key, value in [line.split(":", 1)]
               if key in ("Threads", "VmRSS", "VmSize")} if isinstance(status, str) else status
    membership = read("/proc/self/cgroup")
    root = Path("/sys/fs/cgroup")
    cgroup = {"membership": membership, "mount_root": str(root)}
    paths = {"mount_root": root}
    if isinstance(membership, str):
        unified = [line[3:] for line in membership.splitlines() if line.startswith("0::")]
        if len(unified) == 1:
            current = (root / unified[0].lstrip("/")).resolve()
            if current.is_relative_to(root):
                paths["current"] = current
            else:
                cgroup["current_error"] = "namespace membership is outside visible cgroup mount"
    for scope, path in paths.items():
        cgroup[scope] = {"path": str(path), "files": {name: read(path / name) for name in (
            "pids.current", "pids.max", "pids.events", "memory.current", "memory.max", "memory.events")}}
    try:
        import resource
        limits = {name: list(resource.getrlimit(getattr(resource, name)))
                  for name in ("RLIMIT_NPROC", "RLIMIT_NOFILE")}
    except (ImportError, OSError, AttributeError) as error:
        limits = {"error": type(error).__name__ + ": " + str(error)}
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as error:
        affinity = {"error": type(error).__name__ + ": " + str(error)}
    return {"process_status": process, "cgroup": cgroup, "limits": limits, "affinity": affinity,
            "torch_threads": {"intra": torch.get_num_threads(), "interop": torch.get_num_interop_threads()},
            "CUDA_initialized": torch.cuda.is_initialized()}


def _checkpoint_io_event(rank, world, phase, event, error=None):
    if os.environ.get("DISTILLSCHEDULE_CHECKPOINT_IO_TRACE") != "1":
        return
    try:
        resources = _checkpoint_io_resources()
    except Exception as failure:
        resources = {"diagnostic_error": type(failure).__name__ + ": " + str(failure)}
    row = {"schema_version": "distillschedule.test_checkpoint_io_phase.v1", "rank": rank,
           "world_size": world, "pid": os.getpid(), "monotonic_ns": time.monotonic_ns(),
           "phase": phase, "event": event, "resources": resources}
    if error is not None:
        row["error"] = type(error).__name__ + ": " + str(error)
    try:
        sys.stderr.write("CHECKPOINT_IO_PHASE " + json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass  # Diagnostics must not replace the worker's actual result or exception.


@contextmanager
def _checkpoint_io_phase(rank, world, phase):
    _checkpoint_io_event(rank, world, phase, "ENTER")
    try:
        yield
    except BaseException as error:
        _checkpoint_io_event(rank, world, phase, "ERROR", error)
        raise
    else:
        _checkpoint_io_event(rank, world, phase, "RETURN")


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
    _checkpoint_io_event(rank, world, "worker", "ENTER")
    with _checkpoint_io_phase(rank, world, "init"):
        _init(rank, world, rendezvous)
    root = Path(output)
    try:
        with _checkpoint_io_phase(rank, world, "state.original"):
            original = _state(rank)
        expected = deepcopy(cp.pack_state(original))
        native = _NativeFiles(rank, world) if with_native else None
        with _checkpoint_io_phase(rank, world, "save"):
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
            with _checkpoint_io_phase(rank, world, "read"):
                public = cp.read_checkpoint(path, identity=identity)
            expected_files = {f"rank-{peer:05d}.pt" for peer in range(world)}
            assert Counter(name for name in hashes if name.endswith(".pt")) == Counter(expected_files)
            hashes.clear(); loads.clear()
            with _checkpoint_io_phase(rank, world, "state.target"):
                target = _state(rank + 20)
            parameter_ids = [id(parameter) for parameter in target["model"].parameters()]
            with _checkpoint_io_phase(rank, world, "restore"):
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
        with _checkpoint_io_phase(rank, world, "gather"):
            records = cp.distributed.gather({"rank": rank, "hashes": hashes, "loads": loads})
        assert Counter(name for row in records for name in row["hashes"] if name.endswith(".pt")) == Counter(expected_files)
        (root / f"valid-rank-{rank}.json").write_text(json.dumps(records))
        assert not torch.cuda.is_initialized()
    finally:
        with _checkpoint_io_phase(rank, world, "destroy"):
            dist.destroy_process_group()
    _checkpoint_io_event(rank, world, "worker", "RETURN")


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


def _output_config(source, output, reduction="post_backward"):
    from distillschedule.training.config import TrainingConfig, load_training_config

    config = load_training_config(source)
    raw = config.to_dict()
    raw["data"]["sharding"] = "rank_strided"
    raw["trainer"].update(cycles=1, checkpoint_every=1, output_dir=str(output))
    raw["execution"] = {"mode": "manual", "implementation": {
        "backend": "torch-callable-replicated-v1", "reduction": reduction,
        "bucket_cap_mb": 0, "local": {"backend": "torch-callable-resident-v1",
            "share_immutable": False, "checkpoint_bindings": []}}}
    return TrainingConfig.from_dict(raw, source=config.source)


def _output_worker(rank, world, rendezvous, output, source, with_native, reduction):
    from distillschedule.training import session

    _init(rank, world, rendezvous)
    root = Path(output)
    try:
        config = _output_config(source, root / "training", reduction)
        with pytest.MonkeyPatch.context() as patch:
            actual_save = session.save_checkpoint
            if with_native:
                def save_native(*args, **kwargs):
                    return actual_save(*args, **kwargs, native_checkpoint=_NativeFiles(rank, world))
                patch.setattr(session, "save_checkpoint", save_native)
            trained = session.train(config)
        path = Path(trained["checkpoint"])
        expected = torch.load(path / f"rank-{rank:05d}.pt", weights_only=True)
        records = []
        for operation in ("export", "evaluate"):
            hashes, loads, mutations = [], [], []
            actual_sha, actual_load, actual_unpack = cp._sha, torch.load, session.unpack_state

            def counted_sha(filename):
                filename = Path(filename)
                if filename.is_relative_to(path):
                    hashes.append(str(filename.relative_to(path)))
                return actual_sha(filename)

            def counted_load(filename, *args, **kwargs):
                if isinstance(filename, (str, Path)) and Path(filename).is_relative_to(path):
                    loads.append(Path(filename).name)
                return actual_load(filename, *args, **kwargs)

            def checked_unpack(packed, state, **kwargs):
                assert equal(packed, expected["state"])
                parameter_ids = [id(p) for value in state.values() if isinstance(value, torch.nn.Module)
                                 for p in value.parameters()]
                actual_unpack(packed, state, **kwargs)
                assert equal(cp.pack_state(state), expected["state"])
                assert parameter_ids == [id(p) for value in state.values() if isinstance(value, torch.nn.Module)
                                         for p in value.parameters()]
                mutations.append("exact_saved_state_restored")

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(cp, "_sha", counted_sha)
                patch.setattr(torch, "load", counted_load)
                patch.setattr(session, "unpack_state", checked_unpack)
                result = getattr(session, operation)(config, path, root / operation)
            assert result["checkpoint_cycle"] == 1
            assert mutations == ["exact_saved_state_restored"]
            assert loads == [f"rank-{rank:05d}.pt"]
            assert [name for name in hashes if name.endswith(".pt")] == [f"rank-{rank:05d}.pt"]
            if with_native:
                assert Counter(name for name in hashes if name.startswith("native/")) == Counter(
                    f"native/opaque-{peer}.bin" for peer in range(world))
            rows = cp.distributed.gather({"rank": rank, "operation": operation,
                                          "hashes": hashes, "loads": loads})
            assert Counter(name for row in rows for name in row["hashes"] if name.endswith(".pt")) == Counter(
                f"rank-{peer:05d}.pt" for peer in range(world))
            records.extend(rows)
        assert not torch.cuda.is_initialized()
        (root / f"output-rank-{rank}.json").write_text(json.dumps(records))
    finally:
        dist.destroy_process_group()


def _output_fixture(tmp_path, world):
    from distillschedule.data import prepare
    from distillschedule.examples.tiny_av import build_fixture

    media = tmp_path / "media"
    build_fixture(media, samples=world * 2)
    prepare(media / "raw/manifest.json", media / "prepared")
    return media / "train.json"


@pytest.mark.parametrize("world,with_native,reduction", [
    (2, False, "post_backward"), (4, False, "post_backward"), (2, True, "post_backward"),
    (2, False, "fsdp2_reshard"), (2, False, "fsdp2_keep_forward"),
])
def test_public_output_collectively_hashes_each_common_shard_once(tmp_path, world, with_native, reduction):
    if reduction.startswith("fsdp2") and tuple(map(int, torch.__version__.split(".")[:2])) < (2, 10):
        pytest.skip("CPU FSDP2 requires Torch >= 2.10")
    source = _output_fixture(tmp_path, world)
    mp.spawn(_output_worker, args=(world, str(tmp_path / "rdzv"), str(tmp_path), str(source), with_native, reduction),
             nprocs=world, join=True)


def _output_rejection_worker(rank, rendezvous, output, source):
    from distillschedule.training import session

    _init(rank, 2, rendezvous)
    root = Path(output)
    try:
        config = _output_config(source, root / "training")
        actual_save = session.save_checkpoint
        with pytest.MonkeyPatch.context() as patch:
            def save_native(*args, **kwargs):
                return actual_save(*args, **kwargs, native_checkpoint=_NativeFiles(rank, 2))
            patch.setattr(session, "save_checkpoint", save_native)
            trained = session.train(config)
        original = Path(trained["checkpoint"])
        cases = []
        for fault in ("corrupt-peer", "missing-peer", "manifest-disagreement", "native-corrupt", "coverage-missing"):
            path, alias = root / fault, root / (fault + "-alias")
            if rank == 0:
                shutil.copytree(original, path)
                if fault == "corrupt-peer":
                    (path / "rank-00001.pt").write_bytes(b"corrupt checkpoint")
                elif fault == "missing-peer":
                    (path / "rank-00001.pt").unlink()
                elif fault == "native-corrupt":
                    (path / "native/opaque-1.bin").write_bytes(b"corrupt native sidecar")
                elif fault == "manifest-disagreement":
                    shutil.copytree(path, alias)
                    manifest = json.loads((alias / "manifest.json").read_text())
                    manifest["versions"]["student"] += 1
                    (alias / "manifest.json").write_text(json.dumps(manifest))
                    (alias / "_SUCCESS").write_text(cp._sha(alias / "manifest.json") + "\n")
            dist.barrier()
            loads, providers = [], []
            actual_load, actual_construct, actual_gather = torch.load, session._checked_construct, cp.distributed.gather

            def counted_load(filename, *args, **kwargs):
                if isinstance(filename, (str, Path)) and Path(filename).is_relative_to(path):
                    loads.append(str(filename))
                return actual_load(filename, *args, **kwargs)

            def tracked_construct(*args, **kwargs):
                values = actual_construct(*args, **kwargs)
                providers.append((values[0], deepcopy(cp.pack_state(values[0].state))))
                return values

            def wrong_coverage(value):
                rows = actual_gather(value)
                if fault == "coverage-missing" and isinstance(value, dict) and set(value) == {"rank", "filename", "sha256"}:
                    return rows[:1]
                return rows

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(torch, "load", counted_load)
                patch.setattr(session, "_checked_construct", tracked_construct)
                patch.setattr(cp.distributed, "gather", wrong_coverage)
                patch.setattr(session, "unpack_state", lambda *args, **kwargs: pytest.fail("state mutation preceded admission"))
                with pytest.raises(ValueError) as rejected:
                    session.export(config, alias if fault == "manifest-disagreement" and rank == 1 else path,
                                   root / (fault + "-must-not-export"))
            errors = actual_gather(str(rejected.value))
            assert errors[0] == errors[1], (fault, errors)
            assert not loads, (fault, loads)
            assert not (root / (fault + "-must-not-export")).exists()
            assert all(equal(cp.pack_state(components.state), before) for components, before in providers)
            probe = torch.tensor(rank + 1)
            dist.all_reduce(probe)
            assert probe.item() == 3
            cases.append(fault)
        assert not torch.cuda.is_initialized()
        (root / f"output-rejections-rank-{rank}.json").write_text(json.dumps(cases))
    finally:
        dist.destroy_process_group()


def test_public_output_rejects_before_payload_load_or_state_mutation(tmp_path):
    source = _output_fixture(tmp_path, 2)
    mp.spawn(_output_rejection_worker, args=(str(tmp_path / "rdzv"), str(tmp_path), str(source)),
             nprocs=2, join=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_checkpoint_io_phase_trace_is_opt_in_and_rng_neutral(monkeypatch, capfd, enabled):
    import sys

    flag = "DISTILLSCHEDULE_CHECKPOINT_IO_TRACE"
    if enabled:
        monkeypatch.setenv(flag, "1")
    else:
        monkeypatch.delenv(flag, raising=False)
    calls = []
    def resources():
        calls.append("read")
        return {"scope": "SYNTHETIC_DIAGNOSTIC_TEST_ONLY"}
    monkeypatch.setattr(sys.modules[__name__], "_checkpoint_io_resources", resources, raising=False)
    before = cp.rng_state(device="cpu")
    threads = torch.get_num_threads(), torch.get_num_interop_threads()
    with _checkpoint_io_phase(2, 4, "state"):
        value = 7
    assert value == 7 and equal(cp.rng_state(device="cpu"), before)
    assert (torch.get_num_threads(), torch.get_num_interop_threads()) == threads
    output = capfd.readouterr()
    assert not output.out
    if enabled:
        rows = [json.loads(line.split(" ", 1)[1]) for line in output.err.splitlines()]
        assert [row["event"] for row in rows] == ["ENTER", "RETURN"]
        assert all(row["phase"] == "state" and row["rank"] == 2 and row["world_size"] == 4
                   and row["pid"] == os.getpid() for row in rows)
        assert calls == ["read", "read"]
    else:
        assert output.err == "" and calls == []


def test_checkpoint_io_phase_trace_preserves_original_exception(monkeypatch, capfd):
    monkeypatch.setenv("DISTILLSCHEDULE_CHECKPOINT_IO_TRACE", "1")
    failure = RuntimeError("original worker failure")
    before = cp.rng_state(device="cpu")
    with pytest.raises(RuntimeError) as raised:
        with _checkpoint_io_phase(1, 4, "restore"):
            raise failure
    assert raised.value is failure and equal(cp.rng_state(device="cpu"), before)
    rows = [json.loads(line.split(" ", 1)[1]) for line in capfd.readouterr().err.splitlines()]
    assert [row["event"] for row in rows] == ["ENTER", "ERROR"]
    assert rows[-1]["error"] == "RuntimeError: original worker failure"


def test_checkpoint_io_phase_resources_are_read_only_metadata():
    before = cp.rng_state(device="cpu")
    threads = torch.get_num_threads(), torch.get_num_interop_threads()
    resources = _checkpoint_io_resources()
    json.dumps(resources, allow_nan=False)
    assert "process_status" in resources and "cgroup" in resources and "limits" in resources
    assert resources["torch_threads"] == {"intra": threads[0], "interop": threads[1]}
    assert resources["CUDA_initialized"] is False
    assert equal(cp.rng_state(device="cpu"), before)
    assert (torch.get_num_threads(), torch.get_num_interop_threads()) == threads
