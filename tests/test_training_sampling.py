"""Explicit Program epochs over actual prepared media; CPU functional evidence only."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.multiprocessing as mp

from distillschedule.data import PreparedDataset
from distillschedule.training.checkpoint import rng_state
from distillschedule.training.config import TrainingConfig, digest
from distillschedule.training.sampling import ProgramSampling
from distillschedule.training.session import evaluate, export, generate_from_export, train
from distillschedule.workloads.best_known.registry import file_sha256

from test_training_config import request
from test_training_distributed import _launch
from test_training_end_to_end import equal, fixture, run_cli


def _rows(root, rank=None):
    path = Path(root) if rank is None else Path(root) / f"ranks/rank-{rank:05d}"
    return [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]


def test_epoch_policy_parses_and_doctor_does_not_import_torch(tmp_path):
    raw = request()
    raw["data"]["sampling"] = {"kind": "epochs", "shuffle": True}
    path = tmp_path / "train.json"
    path.write_text(json.dumps(raw))
    result = subprocess.run([sys.executable, "-c",
        "import json, sys; from distillschedule.training.config import load_training_config; "
        "from distillschedule.training.cli import doctor; config = load_training_config(sys.argv[1]); "
        "assert config.data['sampling'] == {'kind': 'epochs', 'shuffle': True}; "
        "print(json.dumps(doctor(config))); assert 'torch' not in sys.modules", str(path)],
        env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["model_imported"] is False and report["cuda_initialized"] is False


def test_default_finite_identity_exhaustion_and_explicit_unshuffled_epochs(tmp_path):
    config = TrainingConfig.from_dict(request(), source=tmp_path / "train.json")
    assert "sampling" not in config.data and "sampling" not in config.to_dict()["data"]
    legacy_identity = config.to_dict()
    legacy_identity["trainer"] = {key: value for key, value in legacy_identity["trainer"].items()
                                  if key not in ("output_dir", "resume_from", "cycles", "checkpoint_every")}
    assert config.identity == digest(legacy_identity)
    finite = ProgramSampling(size=5, seed=7, world_size=2)
    finite.admit_cycles(2)  # An unused finite tail remains untouched, as before.
    assert [finite.position(i).sample_index for i in range(5)] == list(range(5))
    finite.validate_progress(2, 4)
    with pytest.raises(ValueError, match="exhausted"):
        finite.position(5)
    with pytest.raises(ValueError, match="exceeds prepared samples"):
        finite.admit_cycles(3)
    with pytest.raises(ValueError, match="cursor exceeds"):
        finite.validate_progress(3, 6)
    for policy in ({"kind": "finite", "shuffle": False}, {"kind": "epochs", "shuffle": False}):
        raw = config.to_dict()
        raw["data"]["sampling"] = policy
        explicit = TrainingConfig.from_dict(raw, source=config.source)
        assert explicit.data["sampling"] == policy and explicit.identity != config.identity
    epochs = ProgramSampling({"kind": "epochs", "shuffle": False}, size=6, seed=7, world_size=2)
    epochs.admit_cycles(8)  # Partial final epochs are allowed, not partial cohorts.
    epochs.validate_progress(8, 16)
    assert [epochs.position(i).sample_index for i in range(16)] == [i % 6 for i in range(16)]
    assert epochs.position(7).epoch == 1 and epochs.position(7).epoch_offset == 1


@pytest.mark.parametrize("policy", [None, {}, [], "epochs", True,
    {"kind": "epochs"}, {"shuffle": False}, {"kind": "unknown", "shuffle": False},
    {"kind": True, "shuffle": False}, {"kind": "epochs", "shuffle": 1},
    {"kind": "epochs", "shuffle": "false"}, {"kind": "finite", "shuffle": True},
    {"kind": "epochs", "shuffle": False, "seed": 4},
    {"kind": "epochs", "shuffle": False, "algorithm": "future-v2"}])
def test_sampling_policy_is_strict_and_never_guesses(tmp_path, policy):
    raw = request()
    raw["data"]["sampling"] = policy
    with pytest.raises(ValueError, match="sampling"):
        TrainingConfig.from_dict(raw, source=tmp_path / "train.json")


@pytest.mark.parametrize("workload", [f"W{i}" for i in range(1, 7)])
def test_all_native_input_contracts_reject_program_sampling(tmp_path, workload):
    raw = request()
    raw["model"] = {"workload": workload}
    raw["execution"] = {"adapter": "native", "mode": "reference"}
    raw["data"]["sampling"] = {"kind": "epochs", "shuffle": False}
    with pytest.raises(ValueError, match="Program adapter"):
        TrainingConfig.from_dict(raw, source=tmp_path / "train.json")


def test_stable_full_epoch_permutations_disjoint_rank_cohorts_and_no_rng_consumption():
    before = rng_state()
    policy = {"kind": "epochs", "shuffle": True}
    sampling = ProgramSampling(policy, size=12, seed=7, world_size=3)
    orders = [[sampling.position(epoch * 12 + offset).sample_index for offset in range(12)]
              for epoch in range(3)]
    assert orders[0] == [10, 4, 3, 7, 5, 11, 2, 6, 0, 8, 1, 9]  # sha256-sort-v1 vector
    assert orders[0] != orders[1] != orders[2]
    for order in orders:
        assert sorted(order) == list(range(12))
        owners = [set(order[rank::3]) for rank in range(3)]
        assert all(len(owner) == 4 for owner in owners)
        assert all(owners[a].isdisjoint(owners[b]) for a in range(3) for b in range(a + 1, 3))
    restarted = ProgramSampling(policy, size=12, seed=7, world_size=3)
    assert [restarted.position(i).sample_index for i in range(11, 30)] == [
        orders[i // 12][i % 12] for i in range(11, 30)]
    # Returning to an earlier epoch recomputes its order, not a growing cache.
    assert [sampling.position(i).sample_index for i in range(12)] == orders[0]
    other_seed = ProgramSampling(policy, size=12, seed=8, world_size=3)
    assert [other_seed.position(i).sample_index for i in range(12)] != orders[0]
    assert equal(before, rng_state())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("changes", [{"size": True}, {"size": 0}, {"size": -1}, {"size": 2.5},
    {"seed": True}, {"seed": -1}, {"seed": 2**63}, {"world_size": True}, {"world_size": 0},
    {"world_size": -1}, {"size": 5, "world_size": 2}, {"size": 1, "world_size": 2}])
def test_sampling_sizes_seed_and_actual_world_fail_closed(changes):
    with pytest.raises(ValueError):
        ProgramSampling({"kind": "epochs", "shuffle": True}, **{"size": 6, "seed": 7, **changes})


@pytest.mark.parametrize("cursor", [True, False, -1, 1.0, "1", None])
def test_sampling_cursor_does_not_accept_bool_negative_or_coercion(cursor):
    sampling = ProgramSampling({"kind": "epochs", "shuffle": True}, size=4, seed=7)
    with pytest.raises(ValueError, match="cursor"):
        sampling.position(cursor)


@pytest.mark.parametrize("cycle,cursor", [(True, 2), (1, True), (0, False), (-1, -2),
    (1, 3), (1.0, 2), (1, 2.0), (None, 0)])
def test_saved_cursor_must_be_total_complete_cohorts(cycle, cursor):
    sampling = ProgramSampling({"kind": "epochs", "shuffle": True}, size=4, seed=7, world_size=2)
    with pytest.raises(ValueError, match="cursor"):
        sampling.validate_progress(cycle, cursor)


def test_fresh_process_shuffled_epoch_resume_preserves_all_state_and_samples(tmp_path):
    root, config = fixture(tmp_path, samples=4)
    raw = config.to_dict()
    raw["data"]["sampling"] = {"kind": "epochs", "shuffle": True}
    path = root / "epochs.json"
    path.write_text(json.dumps(raw))
    first = run_cli("train", path, "--cycles", 3, "--output", root / "first")
    resumed = run_cli("train", path, "--resume", first["checkpoint"],
                      "--cycles", 9, "--output", root / "resumed")
    whole = run_cli("train", path, "--cycles", 9, "--output", root / "whole")
    a = torch.load(Path(resumed["checkpoint"]) / "state.pt", weights_only=True)
    b = torch.load(Path(whole["checkpoint"]) / "state.pt", weights_only=True)
    for key in ("state", "rng", "cycle", "cursor", "versions", "policy"):
        assert equal(a[key], b[key]), key
    assert a["cycle"] == a["cursor"] == 9
    assert resumed["resume_cycle"] == 3 and resumed["cycles_this_process"] == 6
    assert resumed["final_versions"]["student"] == 9
    assert resumed["final_versions"]["fake"] == 18
    split_rows = _rows(root / "first") + _rows(root / "resumed")
    whole_rows = _rows(root / "whole")
    fields = ("cycle", "cursor", "sample_id", "sample_index", "epoch", "epoch_offset", "versions")
    assert [[row[key] for key in fields] for row in split_rows] == [
        [row[key] for key in fields] for row in whole_rows]
    for start in (0, 4):
        epoch = whole_rows[start:start + 4]
        assert sorted(row["sample_index"] for row in epoch) == list(range(4))
        assert [row["epoch_offset"] for row in epoch] == list(range(4))
        assert {row["epoch"] for row in epoch} == {start // 4}
    assert all(row["sample_id"] == f"sample-{row['sample_index']:06d}" for row in whole_rows)
    assert whole["sampling"]["kind"] == "epochs" and whole["sampling"]["shuffle"] is True
    assert whole["sampling"]["algorithm"] == "sha256-sort-v1"
    # This is real TinyAV autograd and Adam state, never pretrained/GPU evidence.
    for name in ("student_optimizer", "fake_optimizer"):
        slots = a["state"][name]["value"]["state"]
        assert slots and all(value["exp_avg"].count_nonzero() for value in slots.values())


def test_default_training_still_rejects_exhaustion_without_output(tmp_path):
    root, config = fixture(tmp_path, samples=3)
    with pytest.raises(ValueError, match="exceeds prepared samples"):
        train(config)
    assert not (root / "run").exists()


def test_epoch_restart_refuses_policy_seed_and_dataset_drift(tmp_path):
    root, config = fixture(tmp_path, samples=4)
    raw = config.to_dict()
    raw["data"]["sampling"] = {"kind": "epochs", "shuffle": True}
    raw["trainer"].update(cycles=3, output_dir=str(root / "first"))
    config = TrainingConfig.from_dict(raw, source=config.source)
    first = train(config)
    for change in ("policy", "seed", "dataset"):
        altered = config.to_dict()
        output = root / ("wrong-" + change)
        altered["trainer"].update(cycles=5, resume_from=first["checkpoint"], output_dir=str(output))
        if change == "policy":
            altered["data"]["sampling"]["shuffle"] = False
        elif change == "seed":
            altered["trainer"]["seed"] = 8
        else:
            manifest_path = root / "prepared/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["metadata"]["identity_drift"] = "must reject"
            manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="identity"):
            train(TrainingConfig.from_dict(altered, source=config.source))
        assert not output.exists()


def test_malformed_checkpoint_sampling_progress_rejects_before_layout_or_output(tmp_path, monkeypatch):
    from distillschedule.training.policy import CyclePolicy
    root, config = fixture(tmp_path, samples=2)
    raw = config.to_dict()
    raw["data"]["sampling"] = {"kind": "epochs", "shuffle": False}
    raw["trainer"].update(cycles=1, output_dir=str(root / "first"))
    config = TrainingConfig.from_dict(raw, source=config.source)
    first = train(config)
    checkpoint = Path(first["checkpoint"])
    manifest_path = checkpoint / "manifest.json"
    original = json.loads(manifest_path.read_text())

    def must_not_install(*args, **kwargs):
        pytest.fail("malformed sampling progress reached physical layout restoration")

    monkeypatch.setattr(CyclePolicy, "prepare_resume", must_not_install)
    for index, (cycle, cursor) in enumerate(((True, 1), (1, True), (1, 2), (1, -1), (1, "1"))):
        manifest_path.write_text(json.dumps({**original, "cycle": cycle, "cursor": cursor}))
        (checkpoint / "_SUCCESS").write_text(file_sha256(manifest_path) + "\n")
        altered = config.to_dict()
        altered["trainer"].update(cycles=3, resume_from=str(checkpoint), output_dir=str(root / f"wrong-{index}"))
        with pytest.raises(ValueError, match="cursor"):
            train(TrainingConfig.from_dict(altered, source=config.source))
        with pytest.raises(ValueError, match="cursor"):
            evaluate(config, checkpoint, root / f"wrong-output-{index}")
        assert not (root / f"wrong-{index}").exists()
        assert not (root / f"wrong-output-{index}").exists()


def test_epoch_auto_saved_plan_and_output_use_the_correct_physical_inputs(tmp_path):
    root, config = fixture(tmp_path, samples=4)
    raw = config.to_dict()
    raw["data"]["sampling"] = {"kind": "epochs", "shuffle": True}
    raw["trainer"].update(cycles=3, output_dir=str(root / "auto"))
    raw["execution"] = {"mode": "auto", "memory_limit_bytes": 100_000_000,
                        "max_candidates": 3, "samples": 1, "max_program_runs": 8}
    automatic = train(TrainingConfig.from_dict(raw, source=config.source))
    assert _rows(root / "auto")[0]["sample_index"] == 3  # Not manifest index 0.
    assert automatic["policy"]["initial_reference_gate"]["status"] == "PASS"
    raw["trainer"].update(cycles=5, resume_from=automatic["checkpoint"], output_dir=str(root / "resumed"))
    resumed = train(TrainingConfig.from_dict(raw, source=config.source))
    assert not (root / "resumed/planning").exists()
    raw["trainer"].pop("resume_from")
    raw["trainer"]["output_dir"] = str(root / "saved")
    raw["execution"] = {"mode": "plan", "plan": str(root / "auto/planning/selected-plan.json")}
    saved_config = TrainingConfig.from_dict(raw, source=config.source)
    saved = train(saved_config)
    a = torch.load(Path(resumed["checkpoint"]) / "state.pt", weights_only=True)
    b = torch.load(Path(saved["checkpoint"]) / "state.pt", weights_only=True)
    for key in ("state", "rng", "cycle", "cursor", "versions"):
        assert equal(a[key], b[key]), key
    assert saved["policy"]["initial_reference_gate"]["status"] == "PASS"
    generated = evaluate(saved_config, saved["checkpoint"], root / "evaluated", sample_index=1)
    assert generated["sample_index"] == 1 and generated["sample_id"] == "sample-000001"
    exported = export(saved_config, saved["checkpoint"], root / "exported")
    assert exported["kind"] == "INFERENCE_WEIGHTS_NOT_RESUMABLE_CHECKPOINT"
    portable = generate_from_export(saved_config, root / "exported", root / "portable",
                                    prepared=saved_config.data["prepared"], sample_index=1)
    assert portable["sample_index"] == 1 and portable["sample_id"] == "sample-000001"
    artifact = torch.load(generated["artifact"], weights_only=True)
    portable_artifact = torch.load(portable["artifact"], weights_only=True)
    assert equal(artifact, portable_artifact)
    # Independently feed physical manifest record 1 to the restored Student.
    from distillschedule.examples.tiny_av import build_model
    model = build_model({**config.model["parameters"], "seed": config.trainer["seed"]})
    model.state["student"].load_state_dict(b["state"]["student"]["value"])
    feeds = PreparedDataset(saved_config.data["prepared"]).sample(1)
    with torch.no_grad():
        expected = model.generate(model.state, feeds, {})
    assert equal(artifact["generated"], expected)


def test_real_gloo_epoch_continuation_preserves_every_shard_cursor_and_rank_order(tmp_path):
    root, config = fixture(tmp_path, samples=4)
    raw = config.to_dict()
    raw["data"]["sampling"] = {"kind": "epochs", "shuffle": True}
    path = root / "epochs.json"
    path.write_text(json.dumps(raw))
    first = _launch(tmp_path, path, "first", 1, "post_backward")
    resumed = _launch(tmp_path, path, "resumed", 5, "post_backward", first / "checkpoints/cycle-000001")
    whole = _launch(tmp_path, path, "whole", 5, "post_backward")
    rank_rows = []
    fields = ("cycle", "cursor", "sample_id", "sample_index", "epoch", "epoch_offset", "versions")
    for rank in range(2):
        a = torch.load(resumed / f"checkpoints/cycle-000005/rank-{rank:05d}.pt", weights_only=True)
        b = torch.load(whole / f"checkpoints/cycle-000005/rank-{rank:05d}.pt", weights_only=True)
        for key in ("state", "rng", "cycle", "cursor", "versions", "policy"):
            assert equal(a[key], b[key]), (rank, key)
        assert a["cycle"] == 5 and a["cursor"] == 10
        split = _rows(first, rank) + _rows(resumed, rank)
        rows = _rows(whole, rank)
        rank_rows.append(rows)
        assert [[row[key] for key in fields] for row in split] == [[row[key] for key in fields] for row in rows]
        assert [row["epoch_offset"] for row in rows] == [rank, 2 + rank, rank, 2 + rank, rank]
        assert [row["epoch"] for row in rows] == [0, 0, 1, 1, 2]
        assert all(row["sample_id"] == f"sample-{row['sample_index']:06d}" for row in rows)
        assert all(row["execution"]["effects"]["lifecycle_installations"] == 1 for row in rows)
    for start in (0, 2):
        owners = [{row["sample_index"] for row in rows[start:start + 2]} for rows in rank_rows]
        assert owners[0].isdisjoint(owners[1]) and owners[0] | owners[1] == set(range(4))
    summary = json.loads((resumed / "summary.json").read_text())
    assert summary["world_size"] == summary["rank_count"] == 2
    assert summary["sampling"]["algorithm"] == "sha256-sort-v1"


def test_actual_two_rank_epoch_tail_rejects_without_padding_dropping_or_output(tmp_path):
    root, config = fixture(tmp_path, samples=3)
    raw = config.to_dict()
    raw["data"]["sampling"] = {"kind": "epochs", "shuffle": False}
    path = root / "epochs.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(mp.ProcessRaisedException, match="divisible by actual world_size"):
        _launch(tmp_path, path, "rejected-tail", 4, "post_backward")
    assert not (tmp_path / "rejected-tail").exists()
