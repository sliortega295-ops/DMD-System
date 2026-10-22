"""Read-only host loss collection for atomic original-native operations."""
from copy import deepcopy
from datetime import timedelta
import hashlib
from pathlib import Path
import json
import os
import time

import pytest
import torch

from distillschedule.training.adapters import ModelComponents


def test_optional_host_metrics_keep_existing_scalar_metrics():
    from distillschedule.training.session import _training_metrics
    outputs = {"loss": torch.tensor(2.0), "native": {"loss": 3.5}}
    plain = ModelComponents(state={"value": 0}).validate()
    assert _training_metrics(plain, outputs) == {"loss": 2.0}
    model = ModelComponents(state={"value": 0}, collect_metrics=lambda rows: {"native/loss": rows["native"]["loss"]})
    assert _training_metrics(model.validate(), outputs) == {"loss": 2.0, "native/loss": 3.5}


@pytest.mark.parametrize("extra", [None, [], {"loss": 3.0}, {"": 1.0}, {3: 1.0},
                                   {"native": True}, {"native": 1}, {"native": torch.tensor(1.)},
                                   {"native": float("nan")}, {"native": float("inf")}])
def test_invalid_host_metrics_reject_without_tensor_coercion(extra):
    from distillschedule.training.session import _training_metrics
    model = ModelComponents(state={"value": 0}, collect_metrics=lambda rows: extra)
    with pytest.raises(ValueError, match="metric"):
        _training_metrics(model, {"loss": torch.tensor(2.)})


def test_metric_callback_must_be_callable():
    with pytest.raises(ValueError, match="collect_metrics must be callable"):
        ModelComponents(state={"value": 0}, collect_metrics=True).validate()


def test_real_training_host_metrics_do_not_change_updates_rng_or_data(tmp_path, monkeypatch):
    from distillschedule.data import prepare
    from distillschedule.examples.tiny_av import build_fixture
    from distillschedule.training.config import load_training_config, TrainingConfig
    from distillschedule.training import session
    from test_training_end_to_end import equal

    root = tmp_path / "media"
    build_fixture(root, samples=2)
    prepare(root / "raw/manifest.json", root / "prepared")
    loaded = load_training_config(root / "train.json")
    raw = loaded.to_dict()
    raw["trainer"]["cycles"] = 2
    config = TrainingConfig.from_dict(raw, source=loaded.source)
    control = session.train(config)
    original = session._construct_model
    seen = []

    def factory(*args, **kwargs):
        components, identity = original(*args, **kwargs)

        def collect(outputs):
            losses = [value for value in outputs.values() if isinstance(value, torch.Tensor) and value.numel() == 1]
            assert losses
            value = float(losses[0].detach())
            seen.append(value)
            return {"adapter/first_loss": value}

        components.collect_metrics = collect
        return components, identity

    monkeypatch.setattr(session, "_construct_model", factory)
    raw = deepcopy(config.to_dict())
    raw["trainer"]["output_dir"] = str(tmp_path / "with-metrics")
    candidate = session.train(TrainingConfig.from_dict(raw, source=config.source))
    a = torch.load(Path(control["checkpoint"]) / "state.pt", weights_only=True)
    b = torch.load(Path(candidate["checkpoint"]) / "state.pt", weights_only=True)
    for key in ("state", "rng", "cursor", "cycle", "versions", "policy"):
        assert equal(a[key], b[key]), key
    rows = [json.loads(line) for line in (tmp_path / "with-metrics/metrics.jsonl").read_text().splitlines()]
    assert len(rows) == len(seen) == 2
    assert [row["metrics"]["adapter/first_loss"] for row in rows] == seen
    assert [row["sample_id"] for row in rows] == ["sample-000000", "sample-000001"]
    assert not torch.cuda.is_initialized()


def _metric_worker(rank, rendezvous, output, config_path, fault, failure_cycle):
    import torch.distributed as dist
    from distillschedule.training import session
    from distillschedule.training.config import TrainingConfig, load_training_config
    from test_training_end_to_end import equal

    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2", GROUP_RANK="0")
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=10))
    root = Path(output)
    try:
        loaded = load_training_config(config_path)
        raw = loaded.to_dict()
        raw["data"]["sharding"] = "rank_strided"
        raw["trainer"].update(cycles=2, checkpoint_every=1, output_dir=str(root / "candidate"))
        raw["execution"] = {"mode": "manual", "implementation": {
            "backend": "torch-callable-replicated-v1", "reduction": "post_backward", "bucket_cap_mb": 0,
            "local": {"backend": "torch-callable-resident-v1", "share_immutable": False,
                      "checkpoint_bindings": []}}}
        control = None
        if fault == "valid":
            baseline = deepcopy(raw)
            baseline["trainer"]["output_dir"] = str(root / "control")
            control = session.train(TrainingConfig.from_dict(baseline, source=loaded.source))
        original_construct = session._construct_model
        original_append, original_save = session._append_json, session.save_checkpoint
        seen, published, saved = [], [], []
        checkpoint_before = None

        def checkpoint_hashes():
            checkpoint = root / "candidate/checkpoints/cycle-000001"
            return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(checkpoint.glob("*")) if path.is_file()}

        def construct(*args, **kwargs):
            components, identity = original_construct(*args, **kwargs)
            def collect(outputs):
                nonlocal checkpoint_before
                losses = [value for value in outputs.values() if isinstance(value, torch.Tensor) and value.numel() == 1]
                assert losses and all(torch.isfinite(value).all() for value in losses)
                value = float(losses[0].detach())
                seen.append(value)
                if fault != "valid" and len(seen) == failure_cycle:
                    checkpoint_before = checkpoint_hashes()
                    if rank == 1:
                        if fault == "exception":
                            raise ValueError("rank-one metric callback failed")
                        return {"adapter/first_loss": float("nan")}
                return {"adapter/first_loss": value}
            components.collect_metrics = collect
            return components, identity

        def append(path, value):
            if Path(path).name == "metrics.jsonl":
                published.append(value["cycle"])
            return original_append(path, value)

        def save(*args, **kwargs):
            saved.append(kwargs["cycle"])
            return original_save(*args, **kwargs)

        session._construct_model, session._append_json, session.save_checkpoint = construct, append, save
        error, result = None, None
        started = time.monotonic()
        try:
            result = session.train(TrainingConfig.from_dict(raw, source=loaded.source))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            session._construct_model, session._append_json, session.save_checkpoint = (
                original_construct, original_append, original_save)
        elapsed = time.monotonic() - started
        state_equal = None
        if control is not None and result is not None:
            a = torch.load(Path(control["checkpoint"]) / f"rank-{rank:05d}.pt", weights_only=True)
            b = torch.load(Path(result["checkpoint"]) / f"rank-{rank:05d}.pt", weights_only=True)
            state_equal = {key: equal(a[key], b[key]) for key in ("state", "rng", "cursor", "cycle", "versions", "policy")}
        # A probe cannot run until both ranks have left train: otherwise it may
        # accidentally match the broken implementation's publication collective.
        (root / f"observed-{rank}.json").write_text(json.dumps({"error": error, "elapsed": elapsed}))
        deadline = time.monotonic() + 15
        while not (root / f"observed-{1-rank}.json").exists():
            if time.monotonic() >= deadline:
                raise AssertionError("peer did not leave public training within the bounded deadline")
            time.sleep(.02)
        group_error = None
        probe = torch.tensor(rank + 1)
        try:
            dist.all_reduce(probe)
        except Exception as exc:
            group_error = f"{type(exc).__name__}: {exc}"
        (root / f"rank-{rank}.json").write_text(json.dumps({
            "error": error, "elapsed": elapsed, "group_error": group_error, "probe": probe.item(),
            "group_initialized": dist.is_initialized(), "state_equal": state_equal,
            "published": published, "saved": saved, "seen": seen,
            "checkpoint_before": checkpoint_before, "checkpoint_after": checkpoint_hashes(),
            "cuda_initialized": torch.cuda.is_initialized()}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fault", ["exception", "nonfinite"])
@pytest.mark.parametrize("failure_cycle", [1, 2])
def test_rank_local_host_metric_failure_is_coordinated_before_progress(tmp_path, fault, failure_cycle):
    from distillschedule.data import prepare
    from distillschedule.examples.tiny_av import build_fixture
    from test_training_policy_continuation import _run_workers

    media = tmp_path / "media"
    build_fixture(media, samples=4)
    prepare(media / "raw/manifest.json", media / "prepared")
    root = tmp_path / "workers"
    rows = _run_workers(root, _metric_worker, str(media / "train.json"), fault, failure_cycle)
    assert all(row["error"] and "training metric collection" in row["error"] for row in rows), rows
    assert rows[0]["error"] == rows[1]["error"], rows
    expected = "rank-one metric callback failed" if fault == "exception" else "named finite Python floats"
    assert expected in rows[0]["error"]
    for rank, row in enumerate(rows):
        # The bounded worker/group deadlines detect hangs; shared CI load is
        # not a performance contract for this correctness regression.
        assert row["group_initialized"]
        assert row["group_error"] is None and row["probe"] == 3
        assert row["published"] == row["saved"] == list(range(1, failure_cycle))
        assert len(row["seen"]) == failure_cycle and not row["cuda_initialized"]
        assert row["checkpoint_before"] == row["checkpoint_after"]
        rank_root = root / f"candidate/ranks/rank-{rank:05d}"
        failure = json.loads((rank_root / "failure.json").read_text())
        assert failure["cursor"] == (failure_cycle - 1) * 2
        assert failure["completed_cycles_this_process"] == failure_cycle - 1
        previous = root / "candidate/checkpoints/cycle-000001"
        assert failure["last_complete_checkpoint"] == (str(previous) if failure_cycle == 2 else None)
        metric_rows = [json.loads(line) for line in (rank_root / "metrics.jsonl").read_text().splitlines()] if (
            rank_root / "metrics.jsonl").exists() else []
        assert [entry["cycle"] for entry in metric_rows] == list(range(1, failure_cycle))
    assert not (root / f"candidate/checkpoints/cycle-{failure_cycle:06d}").exists()
    assert not (root / "candidate/summary.json").exists()
    if failure_cycle == 2:
        assert (root / "candidate/checkpoints/cycle-000001/_SUCCESS").is_file()


def test_two_rank_host_metrics_preserve_real_updates_rng_and_samples(tmp_path):
    from distillschedule.data import prepare
    from distillschedule.examples.tiny_av import build_fixture
    from test_training_policy_continuation import _run_workers

    media = tmp_path / "media"
    build_fixture(media, samples=4)
    prepare(media / "raw/manifest.json", media / "prepared")
    root = tmp_path / "workers"
    rows = _run_workers(root, _metric_worker, str(media / "train.json"), "valid", None)
    for rank, row in enumerate(rows):
        assert row["error"] is None and all(row["state_equal"].values()), row
        assert row["group_initialized"] and row["group_error"] is None and row["probe"] == 3
        assert row["published"] == row["saved"] == [1, 2] and not row["cuda_initialized"]
        metrics = [json.loads(line) for line in (root / f"candidate/ranks/rank-{rank:05d}/metrics.jsonl").read_text().splitlines()]
        baseline = [json.loads(line) for line in (root / f"control/ranks/rank-{rank:05d}/metrics.jsonl").read_text().splitlines()]
        assert [entry["sample_id"] for entry in metrics] == [f"sample-{cycle * 2 + rank:06d}" for cycle in range(2)]
        assert [entry["metrics"]["adapter/first_loss"] for entry in metrics] == row["seen"]
        assert [{key: value for key, value in entry["metrics"].items() if key != "adapter/first_loss"}
                for entry in metrics] == [entry["metrics"] for entry in baseline]
