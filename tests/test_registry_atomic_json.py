"""Local CPU publication contracts; all receipt data is SYNTHETIC_TEST_ONLY.

These tests do not attest shared-filesystem cache coherence or native training.
"""
from copy import deepcopy
import errno
import json
import multiprocessing
import os
from pathlib import Path
import queue
import random
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from distillschedule.workloads.best_known import registry


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _writer(path, index, barrier, results):
    """Independent CPU processes call the actual public writer, not a double."""
    try:
        barrier.wait(timeout=20)
        registry.write_new_json(Path(path), {"writer": index, "payload": [str(index) * 60] * 400})
        results.put((index, "created"))
    except FileExistsError:
        results.put((index, "exists"))
    except BaseException as exc:
        results.put((index, repr(exc)))


def test_bytes_return_value_and_parent_creation_are_unchanged(tmp_path):
    target = tmp_path / "new" / "rank-00008.json"
    value = {"unicode": "完整收尾", "nested": {"z": None, "a": [True, 1, 1.25]}, "padding": "x" * 24000}
    assert registry.write_new_json(target, value) is None
    assert target.read_bytes() == _bytes(value)
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), object()])
def test_serialization_failure_never_exposes_a_final_or_leaves_a_temporary(tmp_path, bad):
    target = tmp_path / "rank.json"
    with pytest.raises((ValueError, TypeError)):
        registry.write_new_json(target, {"a_written_first": "x" * 24000, "z_invalid": bad})
    assert not os.path.lexists(target)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling_symlink", "directory"])
def test_existing_targets_are_refused_without_touching_their_contents(tmp_path, kind):
    target, original = tmp_path / "rank.json", tmp_path / "original.json"
    original.write_bytes(b"IMMUTABLE_ORIGINAL\n")
    if kind == "file":
        target.write_bytes(b"PRIOR_FAILURE_MUST_STAY\n")
    elif kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(original if kind == "symlink" else tmp_path / "missing.json")
    before = target.lstat()
    # As with open('x'), an existing name wins even over an invalid new value.
    with pytest.raises(FileExistsError) as error:
        registry.write_new_json(target, {"invalid": object()})
    assert error.value.errno == errno.EEXIST
    assert error.value.filename == str(target)
    after = target.lstat()
    assert (after.st_ino, after.st_mode, after.st_size) == (before.st_ino, before.st_mode, before.st_size)
    assert original.read_bytes() == b"IMMUTABLE_ORIGINAL\n"
    if kind == "file":
        assert target.read_bytes() == b"PRIOR_FAILURE_MUST_STAY\n"
    assert set(tmp_path.iterdir()) == {target, original}


def test_final_name_is_invisible_while_real_encoder_has_written_a_prefix(tmp_path, monkeypatch):
    target = tmp_path / "rank-00008.json"
    value = {"payload": ["SYNTHETIC_TEST_ONLY" * 4] * 400, "rank": 8}
    paused, release = threading.Event(), threading.Event()
    outcomes = queue.Queue()
    original = json.JSONEncoder.iterencode

    def iterencode(self, *args, **kwargs):
        emitted = 0
        for chunk in original(self, *args, **kwargs):
            yield chunk  # Real json.dump writes this chunk before requesting another.
            emitted += len(chunk.encode("utf-8"))
            if emitted >= 10000 and not paused.is_set():
                paused.set()
                if not release.wait(10):
                    raise TimeoutError("test reader did not release the serializer")

    def write():
        try:
            registry.write_new_json(target, value)
            outcomes.put(None)
        except BaseException as exc:
            outcomes.put(exc)

    monkeypatch.setattr(json.JSONEncoder, "iterencode", iterencode)
    thread = threading.Thread(target=write, daemon=True)
    thread.start()
    try:
        assert paused.wait(10)
        visible = target.read_bytes() if target.exists() else None
        assert visible is None, f"partial final name visible: {len(visible)} of {len(_bytes(value))} bytes"
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and outcomes.get_nowait() is None
    assert target.read_bytes() == _bytes(value)
    assert list(tmp_path.iterdir()) == [target]


def test_real_process_writers_publish_exactly_one_complete_winner(tmp_path):
    target = tmp_path / "rank.json"
    context = multiprocessing.get_context("spawn")
    barrier, results = context.Barrier(7), context.Queue()
    processes = [context.Process(target=_writer, args=(str(target), i, barrier, results)) for i in range(6)]
    observations = []
    try:
        for process in processes:
            process.start()
        barrier.wait(timeout=20)
        deadline = time.monotonic() + 20
        while any(process.is_alive() for process in processes):
            assert time.monotonic() < deadline
            if target.exists():
                raw = target.read_bytes()
                value = json.loads(raw)
                assert raw == _bytes(value)
                observations.append(value["writer"])
            time.sleep(0.001)
        returned = [results.get(timeout=5) for _ in processes]
        for process in processes:
            process.join(5)
            assert process.exitcode == 0
        winners = [i for i, status in returned if status == "created"]
        assert len(winners) == 1 and sum(status == "exists" for _, status in returned) == 5
        winner = winners[0]
        assert target.read_bytes() == _bytes({"writer": winner, "payload": [str(winner) * 60] * 400})
        assert all(i == winner for i in observations)
        assert list(tmp_path.iterdir()) == [target]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()  # Only this test's own bounded CPU children.
            if process.pid is not None:
                process.join(5)
        results.close()
        results.join_thread()


def test_complete_file_fsync_precedes_link_and_directory_fsync_follows_cleanup(tmp_path, monkeypatch):
    target, value = tmp_path / "rank.json", {"rank": 8, "small_buffered_value": "complete"}
    expected, events = _bytes(value), []
    fsync, link = os.fsync, os.link

    def observe_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            assert events == ["file_fsync", "exclusive_link"]
            assert list(tmp_path.iterdir()) == [target]
            assert target.read_bytes() == expected
            events.append("directory_fsync")
        else:
            assert not target.exists()
            temporary, = tmp_path.iterdir()
            assert temporary.read_bytes() == expected  # Includes flush before fsync.
            events.append("file_fsync")
        fsync(fd)

    def observe_link(source, destination):
        assert events == ["file_fsync"]
        assert Path(source).parent == Path(destination).parent == tmp_path
        assert not target.exists() and Path(source).read_bytes() == expected
        result = link(source, destination)
        events.append("exclusive_link")
        return result

    monkeypatch.setattr(os, "fsync", observe_fsync)
    monkeypatch.setattr(os, "link", observe_link)
    registry.write_new_json(target, value)
    assert events == ["file_fsync", "exclusive_link", "directory_fsync"]


@pytest.mark.parametrize("stage", ["file_fsync", "link"])
def test_prepublication_io_failures_clean_only_owned_temporary(tmp_path, monkeypatch, stage):
    target = tmp_path / "rank.json"
    other = tmp_path / ".another-writer.pending"
    other.write_bytes(b"OTHER_WRITER")

    def fail(*args, **kwargs):
        raise OSError(errno.EIO, "injected " + stage)

    monkeypatch.setattr(os, "fsync" if stage == "file_fsync" else "link", fail)
    with pytest.raises(OSError, match="injected " + stage):
        registry.write_new_json(target, {"rank": 8})
    assert not target.exists() and other.read_bytes() == b"OTHER_WRITER"
    assert list(tmp_path.iterdir()) == [other]


def test_directory_fsync_failure_retains_complete_published_file_and_raises(tmp_path, monkeypatch):
    target, value = tmp_path / "rank.json", {"rank": 8}
    original = os.fsync

    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "injected directory fsync")
        original(fd)

    monkeypatch.setattr(os, "fsync", fail_directory)
    with pytest.raises(OSError, match="injected directory fsync"):
        registry.write_new_json(target, value)
    # Publication cannot be rolled back after readers may have observed it.
    assert target.read_bytes() == _bytes(value)
    assert list(tmp_path.iterdir()) == [target]


def test_racing_destination_and_temporary_collision_never_remove_another_writer(tmp_path, monkeypatch):
    target = tmp_path / "rank.json"
    original = os.link

    def racing_link(source, destination):
        target.write_bytes(b"OTHER_WINNER\n")
        return original(source, destination)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError) as error:
        registry.write_new_json(target, {"rank": 8})
    assert error.value.errno == errno.EEXIST
    assert error.value.filename == str(target)
    assert error.value.filename2 is None
    assert target.read_bytes() == b"OTHER_WINNER\n"
    assert list(tmp_path.iterdir()) == [target]
    target.unlink()
    collision = tmp_path / ".distillschedule-json-collision.tmp"
    collision.write_bytes(b"OTHER_TEMPORARY\n")
    monkeypatch.setattr(registry.uuid, "uuid4", lambda: SimpleNamespace(hex="collision"))
    with pytest.raises(FileExistsError):
        registry.write_new_json(target, {"rank": 8})
    assert collision.read_bytes() == b"OTHER_TEMPORARY\n"
    assert not target.exists() and list(tmp_path.iterdir()) == [collision]


@pytest.mark.parametrize("mask", [0o022, 0o027, 0o077])
def test_original_open_x_permissions_follow_the_callers_umask(tmp_path, mask):
    target = tmp_path / ("r" * 240 + ".json")
    previous = os.umask(mask)
    try:
        registry.write_new_json(target, {"rank": 8})
    finally:
        os.umask(previous)
    assert stat.S_IMODE(target.stat().st_mode) == 0o666 & ~mask
    assert list(tmp_path.iterdir()) == [target]


def test_public_writer_preserves_python_numpy_torch_rng_and_never_initializes_cuda(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cuda_initialized = torch.cuda.is_initialized()
    for i in range(3):
        registry.write_new_json(tmp_path / f"rank-{i}.json", {"rank": i})
    with pytest.raises(ValueError):
        registry.write_new_json(tmp_path / "invalid.json", {"nan": float("nan")})
    assert random.getstate() == python_state
    now = np.random.get_state()
    assert now[0] == numpy_state[0] and np.array_equal(now[1], numpy_state[1]) and now[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.cuda.is_initialized() == cuda_initialized is False


def test_cold_writer_import_needs_no_tensor_or_model_runtime(tmp_path):
    code = """import random,sys
from pathlib import Path
before=random.getstate()
from distillschedule.workloads.best_known.registry import write_new_json
write_new_json(Path(sys.argv[1]), {'rank': 8})
assert random.getstate()==before
assert not any(name in sys.modules for name in ('torch','numpy','transformers','diffusers'))
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "rank.json")], check=True, timeout=20)


def test_original_w6_all_rank_receipts_still_feed_the_original_public_consumer(tmp_path):
    from distillschedule.strategies.h3_ar_task import H3ARTask
    from distillschedule.training import h3_ar_inputs as inputs
    from distillschedule.training.h3_ar_training import validate_joint

    # Metadata-only fixtures; no .pt payload is loaded and no native model runs.
    task, sources = H3ARTask(), []
    for rank in range(16):
        source = tmp_path / f"SYNTHETIC_TEST_ONLY-source-{rank}"
        source.write_bytes(f"SYNTHETIC_TEST_ONLY:{rank}".encode())
        pin = dict(path=str(source), sha256=registry.file_sha256(source))
        sources.append(dict(rank=rank, record_id=f"SYNTHETIC_TEST_ONLY-{rank}", **pin,
                            sources={"original_cached_record": pin}, metadata=inputs._metadata(task, rank)))
    document = inputs._epoch_document(sources, cycles=50, seed=42)
    prepared = tmp_path / "prepared.json"
    registry.write_new_json(prepared, document)
    admitted = inputs.input_metadata(prepared, cycles=50, seed=42, verify_files=True)
    output = tmp_path / "receipts"
    joint = []
    for rank in range(16):
        cursor = inputs.H3ARInputStream(prepared, rank=rank, cycles=50, seed=42, completed=49)
        row = dict(rank=rank, cycle=50, versions=dict(student=50, fake=50, ema=0), native_global_steps=100,
                   input_identity=inputs.source_identity(cursor.record(50)), evidence_kind="SYNTHETIC_TEST_ONLY", padding="")
        # Exercise the real 24KB boundary without copying any historical receipt.
        row["padding"] = "x" * (24077 + rank % 2 - len(_bytes(row)))
        path = output / f"rank-{rank:05d}.json"
        registry.write_new_json(path, row)
        assert path.stat().st_size == 24077 + rank % 2
        consumed = registry._object(path)
        assert consumed == row and path.read_bytes() == _bytes(row)
        joint.append({k: consumed[k] for k in ("rank", "cycle", "versions", "native_global_steps", "input_identity")})
    validate_joint(joint, cycle=50, inputs=admitted)
    assert admitted["source_records"][6]["metadata"]["input_shapes"]["audio"][-1] == 94
    assert len(list(output.iterdir())) == 16
    with pytest.raises(ValueError, match="sixteen"):
        validate_joint(joint[:-1], cycle=50, inputs=admitted)
    changed = deepcopy(joint)
    changed[8]["native_global_steps"] = 99
    with pytest.raises(ValueError, match="clock"):
        validate_joint(changed, cycle=50, inputs=admitted)
