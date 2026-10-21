"""CPU_TEST_ONLY: metadata and resume failures with real caller-owned Gloo."""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist

from distillschedule.examples.tiny_av import build_algorithm
from distillschedule.training.config import TrainingConfig
from test_training_control_admission import _fixture
from test_training_policy_continuation import _run_workers


def resume_algorithm(components, feeds, parameters, cycle, seed):
    """A real adapter whose local input admission can fail on restart."""
    if dist.get_rank() == 1 and Path(parameters["failure_flag"]).exists():
        raise OSError("injected rank1 resumed cycle construction failure")
    return build_algorithm(components, feeds, parameters, cycle, seed)


class _FailingStream:
    def __init__(self, stream, fault):
        self.stream, self.fault = stream, fault

    def __enter__(self):
        self.stream.__enter__()
        return self

    def write(self, value):
        if self.fault == "write":
            raise OSError("injected rank1 metrics write failure")
        return self.stream.write(value)

    def __exit__(self, *args):
        self.stream.__exit__(*args)
        if self.fault == "close":
            raise OSError("injected rank1 metrics close failure")


def _failure_worker(rank, rendezvous, output, source, raw, fault, fail_report):
    from distillschedule.training import session

    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE="2", GROUP_RANK="0")
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=10))
    root = Path(output)
    try:
        checkpoint = None
        if fault == "resume":
            raw["algorithm"]["factory"] = str(Path(__file__).resolve()) + ":resume_algorithm"
            raw["algorithm"]["parameters"]["failure_flag"] = str(root / "fail-resume")
            first = deepcopy(raw)
            first["trainer"].update(cycles=1, output_dir=str(root / "first"))
            checkpoint = session.train(TrainingConfig.from_dict(first, source=source))["checkpoint"]
            if rank == 0:
                (root / "fail-resume").touch()
            dist.barrier()
            raw["trainer"].update(cycles=2, resume_from=checkpoint)
        config = TrainingConfig.from_dict(raw, source=source)
        report_attempts = []
        with pytest.MonkeyPatch.context() as patch:
            original_open, original_write_text = Path.open, Path.write_text
            if rank == 1 and fault in {"open", "write", "close"}:
                def open_with_fault(path, *args, **kwargs):
                    if path.name == "metrics.jsonl":
                        if fault == "open":
                            raise OSError("injected rank1 metrics open failure")
                        return _FailingStream(original_open(path, *args, **kwargs), fault)
                    return original_open(path, *args, **kwargs)
                patch.setattr(Path, "open", open_with_fault)
            elif rank == 1 and fault == "serialize":
                original_metrics = session._metrics
                patch.setattr(session, "_metrics", lambda values: {
                    **original_metrics(values), "non_json_value": object()})
            if fail_report:
                def write_text_with_fault(path, *args, **kwargs):
                    if path.name == "failure.json.tmp":
                        report_attempts.append(str(path))
                        raise OSError("injected secondary failure report error")
                    return original_write_text(path, *args, **kwargs)
                patch.setattr(Path, "write_text", write_text_with_fault)
            error = result = None
            started = time.monotonic()
            try:
                result = session.train(config)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
        # Rendezvous outside the process group first: a recovery collective must
        # not accidentally match a peer stranded in training's metadata gather.
        (root / f"observed-{rank}.json").write_text(json.dumps({"error": error, "elapsed": elapsed}))
        deadline = time.monotonic() + 15
        while not (root / f"observed-{1-rank}.json").exists():
            if time.monotonic() >= deadline:
                raise AssertionError("peer did not leave training within the bounded deadline")
            time.sleep(.02)
        group_initialized = dist.is_initialized()
        probe, group_error = torch.tensor(rank + 1), None
        try:
            dist.all_reduce(probe)
        except Exception as exc:
            group_error = f"{type(exc).__name__}: {exc}"
        (root / f"rank-{rank}.json").write_text(json.dumps({
            "status": "CPU_TEST_ONLY", "error": error, "elapsed": elapsed, "result": result,
            "group_initialized": group_initialized, "group_error": group_error, "probe": probe.item(),
            "failure_report_attempts": report_attempts, "prior_checkpoint": checkpoint}))
    finally:
        dist.destroy_process_group()


def _assert_common_failure(rows, message):
    assert all(row["error"] and message in row["error"] for row in rows), rows
    assert rows[0]["error"] == rows[1]["error"], rows
    assert all(row["elapsed"] < 5 and row["result"] is None for row in rows), rows
    assert all(row["group_initialized"] and row["group_error"] is None and row["probe"] == 3
               for row in rows), rows


@pytest.mark.parametrize("every,fault,fail_report", [
    (1, "open", False), (0, "open", False), (1, "write", False),
    (1, "close", False), (1, "serialize", False), (1, "open", True),
])
def test_rank_local_metrics_failure_precedes_checkpoint_and_preserves_embedding_group(
        tmp_path, every, fault, fail_report):
    config, raw = _fixture(tmp_path / "media")
    output = tmp_path / "failed"
    raw["trainer"].update(cycles=1, checkpoint_every=every, output_dir=str(output))
    rows = _run_workers(tmp_path / "workers", _failure_worker, str(config.source), raw, fault, fail_report)
    message = "not JSON serializable" if fault == "serialize" else f"injected rank1 metrics {fault} failure"
    _assert_common_failure(rows, message)
    assert "training metrics publication" in rows[0]["error"], rows
    assert "secondary failure report" not in rows[0]["error"], rows
    assert not (output / "checkpoints").exists()
    assert not (output / "summary.json").exists()
    assert not (output / "effective-policy.json").exists()
    assert not list(output.rglob("_SUCCESS"))
    if fail_report:
        assert all(len(row["failure_report_attempts"]) == 1 for row in rows), rows
        assert not list(output.rglob("failure.json"))
    else:
        for rank in range(2):
            report = json.loads((output / f"ranks/rank-{rank:05d}/failure.json").read_text())
            assert report["error"] in rows[rank]["error"]
            assert report["completed_cycles_this_process"] == 1
            assert report["last_complete_checkpoint"] is None


def test_rank_local_initial_resume_construction_failure_precedes_collectives_and_output(tmp_path):
    config, raw = _fixture(tmp_path / "media")
    output = tmp_path / "failed-resume"
    raw["trainer"]["output_dir"] = str(output)
    rows = _run_workers(tmp_path / "workers", _failure_worker, str(config.source), raw, "resume", False)
    _assert_common_failure(rows, "injected rank1 resumed cycle construction failure")
    assert "resume cycle construction" in rows[0]["error"], rows
    assert not output.exists()
    checkpoint = Path(rows[0]["prior_checkpoint"])
    assert (checkpoint / "_SUCCESS").is_file()
    assert json.loads((checkpoint.parent.parent / "summary.json").read_text())["status"] == "COMPLETE"
    assert not (checkpoint.parent / "cycle-000002").exists()


@pytest.mark.parametrize("metadata", ["metrics.jsonl", "validation.jsonl"])
def test_secondary_failure_report_error_preserves_original_local_metadata_error(tmp_path, monkeypatch, metadata):
    from distillschedule.training.session import train
    from test_training_end_to_end import fixture
    from test_training_validation import validation_request

    root, config = fixture(tmp_path)
    output = root / "failed"
    raw = validation_request(config, output)
    raw["trainer"]["cycles"] = 1
    primary = OSError("injected local " + metadata + " failure")
    original_open, original_write_text = Path.open, Path.write_text
    reports = []

    def failing_open(path, *args, **kwargs):
        if path.name == metadata:
            raise primary
        return original_open(path, *args, **kwargs)

    def failing_report(path, *args, **kwargs):
        if path.name == "failure.json.tmp":
            reports.append(str(path))
            raise OSError("injected secondary failure report error")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    monkeypatch.setattr(Path, "write_text", failing_report)
    with pytest.raises(OSError) as caught:
        train(TrainingConfig.from_dict(raw, source=config.source))
    assert caught.value is primary
    assert len(reports) == 1
    assert not (output / "checkpoints").exists()
    assert not (output / "summary.json").exists()
    assert not list(output.rglob("_SUCCESS"))
