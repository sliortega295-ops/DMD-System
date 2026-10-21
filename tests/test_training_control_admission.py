"""CPU_TEST_ONLY: real Gloo entrypoint admission, not GPU qualification."""
import hashlib
import json
import os
import shutil
import socket
import time
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_training_end_to_end import equal

from distillschedule.data import prepare
from distillschedule.examples.tiny_av import build_fixture
from distillschedule.training.config import TrainingConfig, load_training_config


def construction_sentinel(parameters):
    """Stop old production code safely, before any model or update exists."""
    marker = Path(parameters["marker_dir"]) / f"factory-{os.environ['RANK']}.txt"
    marker.write_text("CPU_TEST_ONLY: factory reached\n")
    raise AssertionError("CPU_TEST_ONLY: factory reached before training control admission")


def _worker(rank, port, source, requests, results):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE="2", GROUP_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    from distillschedule.training.session import train
    config = TrainingConfig.from_dict(requests[rank], source=source)
    result, error = None, None
    try:
        # Let the real public context own initialization, metadata and teardown.
        result = train(config)
    except ValueError as exc:
        error = f"{type(exc).__name__}: {exc}"
    assert not dist.is_initialized(), "training context leaked its owned group"
    Path(results, f"rank-{rank}.json").write_text(json.dumps({
        "status": "CPU_TEST_ONLY", "error": error, "result": result,
        "factory_entered": Path(results, f"factory-{rank}.txt").exists(),
        "output_exists": Path(config.trainer["output_dir"]).exists(),
    }))


def _run_ranks(root, config, requests):
    root.mkdir()
    # This is a local test rendezvous, never a cluster launcher or GPU request.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    processes = mp.spawn(_worker, args=(port, str(config.source), requests, str(root)),
                         nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not processes.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("CPU_TEST_ONLY Gloo admission exceeded the bounded 60s test deadline")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
        for process in processes.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    return [json.loads((root / f"rank-{rank}.json").read_text()) for rank in range(2)]


def _fixture(root):
    build_fixture(root, samples=6)
    prepare(root / "raw/manifest.json", root / "prepared")
    config = load_training_config(root / "train.json")
    raw = config.to_dict()
    raw["data"]["sharding"] = "rank_strided"
    raw["trainer"].update(cycles=2, checkpoint_every=1,
                          distributed={"backend": "gloo", "timeout_seconds": 20})
    return config, raw


def _synthetic_manifest(path, *, cycle=1):
    """Only completed-metadata admission; this is NOT a usable checkpoint."""
    path.mkdir()
    payload = json.dumps({"status": "CPU_TEST_ONLY", "cycle": cycle}).encode()
    (path / "manifest.json").write_bytes(payload)
    (path / "_SUCCESS").write_text(hashlib.sha256(payload).hexdigest() + "\n")
    return str(path)


@pytest.mark.parametrize("case", [
    "cycles", "checkpoint_every", "resume_presence", "manifest_contents", "manifest_bytes", "missing_manifest",
    "unpublished_manifest", "changed_success", "unreadable_manifest", "output_root",
])
def test_control_disagreement_rejects_collectively_before_factory_or_output(tmp_path, case):
    config, raw = _fixture(tmp_path / "media")
    results = tmp_path / "results"
    raw["model"] = {"factory": str(Path(__file__).resolve()) + ":construction_sentinel",
                    "parameters": {"marker_dir": str(results)}}
    raw["trainer"]["output_dir"] = str(tmp_path / "never-created")
    requests = [deepcopy(raw), deepcopy(raw)]
    if case == "cycles":
        requests[1]["trainer"]["cycles"] = 3
    elif case == "checkpoint_every":
        requests[1]["trainer"]["checkpoint_every"] = 0
    elif case == "output_root":
        requests[1]["trainer"]["output_dir"] = str(tmp_path / "also-never-created")
    else:
        first = _synthetic_manifest(tmp_path / "checkpoint-a")
        second = _synthetic_manifest(tmp_path / "checkpoint-b", cycle=2 if case == "manifest_contents" else 1)
        requests[0]["trainer"]["resume_from"] = first
        requests[1]["trainer"]["resume_from"] = second
        if case == "resume_presence":
            del requests[1]["trainer"]["resume_from"]
        elif case == "missing_manifest":
            Path(second, "manifest.json").unlink()
        elif case == "manifest_bytes":
            payload = Path(second, "manifest.json").read_bytes() + b"\n"
            Path(second, "manifest.json").write_bytes(payload)
            Path(second, "_SUCCESS").write_text(hashlib.sha256(payload).hexdigest() + "\n")
        elif case == "unpublished_manifest":
            Path(second, "_SUCCESS").unlink()
        elif case == "changed_success":
            Path(second, "_SUCCESS").write_text("CPU_TEST_ONLY: stale publication hash\n")
        elif case == "unreadable_manifest":
            Path(second, "manifest.json").unlink()
            Path(second, "manifest.json").mkdir()
    # These deliberately mutable controls must NOT change restart identity.
    assert TrainingConfig.from_dict(requests[0], source=config.source).identity == TrainingConfig.from_dict(
        requests[1], source=config.source).identity
    rows = _run_ranks(results, config, requests)
    assert all(row["error"] and "training run/restart control" in row["error"] for row in rows), rows
    assert rows[0]["error"] == rows[1]["error"], rows
    if case in {"missing_manifest", "unpublished_manifest", "changed_success", "unreadable_manifest"}:
        assert "rank failures [None," in rows[0]["error"], rows
    else:
        assert "ranks disagree" in rows[0]["error"], rows
    assert all(not row["factory_entered"] and not row["output_exists"] for row in rows), rows


def test_byte_identical_resume_mount_aliases_run_the_real_training_lifecycle(tmp_path):
    config, raw = _fixture(tmp_path / "media")
    first = deepcopy(raw)
    first["trainer"].update(cycles=1, output_dir=str(tmp_path / "first"))
    rows = _run_ranks(tmp_path / "first-workers", config, [first, deepcopy(first)])
    assert all(row["error"] is None for row in rows), rows
    checkpoint = Path(rows[0]["result"]["checkpoint"])
    alias = tmp_path / "explicit-other-mount"
    shutil.copytree(checkpoint, alias)
    assert alias.resolve() != checkpoint.resolve()
    assert (alias / "manifest.json").read_bytes() == (checkpoint / "manifest.json").read_bytes()
    requests = [deepcopy(raw), deepcopy(raw)]
    for rank, request in enumerate(requests):
        request["trainer"].update(output_dir=str(tmp_path / "resumed"),
                                  resume_from=str(checkpoint if rank == 0 else alias))
    # Default cadence 1 and an explicitly supplied 1 select the same branches.
    del requests[1]["trainer"]["checkpoint_every"]
    resumed = _run_ranks(tmp_path / "resume-workers", config, requests)
    assert all(row["error"] is None for row in resumed), resumed
    whole = deepcopy(raw)
    whole["trainer"]["output_dir"] = str(tmp_path / "whole")
    uninterrupted = _run_ranks(tmp_path / "whole-workers", config, [whole, deepcopy(whole)])
    assert all(row["error"] is None for row in uninterrupted), uninterrupted
    for rank in range(2):
        result = resumed[rank]["result"]
        assert result["resume_cycle"] == result["cycles_this_process"] == 1
        assert result["completed_cycles"] == result["final_versions"]["student"] == 2
        assert result["final_versions"]["fake"] == 4
        assert result["policy"]["initial_reference_gate"] == {"status": "NOT_RUN", "reason": "explicit user strategy"}
        assert result["gpu_execution"] == "NOT_RUN"
        actual = torch.load(Path(result["checkpoint"]) / f"rank-{rank:05d}.pt", weights_only=True)
        expected = torch.load(Path(uninterrupted[rank]["result"]["checkpoint"]) / f"rank-{rank:05d}.pt",
                              weights_only=True)
        for key in ("state", "rng", "cycle", "cursor", "versions", "policy"):
            assert equal(actual[key], expected[key]), (rank, key)
        metrics = Path(requests[rank]["trainer"]["output_dir"]) / f"ranks/rank-{rank:05d}/metrics.jsonl"
        row = json.loads(metrics.read_text())
        assert row["sample_id"] == f"sample-{2 + rank:06d}"
        assert row["execution"]["effects"]["lifecycle_installations"] == 1
