"""Real CPU native-format fixtures, NOT pretrained quality or CUDA evidence.

The original pinned TeacherLatents, production InputLoader, C51 pool lowering,
and checkpoint owners are exercised. No teacher archives are copied to invent
fresh records, and the existing native trainer/math are not replaced.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest
import torch

from test_training_livewan_inputs import original_source, sample
from distillschedule.data.livewan_inputs import C51_KIND, prepare_c51_inputs
from distillschedule.runtime.livewan_crossscale_worker import InputLoader
from distillschedule.workloads.best_known.registry import fingerprint

EPOCH_POLICY = "distillschedule.livewan_teacher_epochs.v1"
EVIDENCE = "CPU_NATIVE_SHAPE_FIXTURE_NOT_PRETRAINED"


def _fresh_process_environment():
    # A child Python process does not inherit pytest's tests/ sys.path entry.
    root = Path(__file__).resolve().parents[1]
    paths = [str(root / 'tests'), str(root / 'src')]
    if os.environ.get('PYTHONPATH'):
        paths.append(os.environ['PYTHONPATH'])
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                PYTHONPATH=os.pathsep.join(paths))


@pytest.fixture(scope="module")
def finite_pool(tmp_path_factory):
    root = tmp_path_factory.mktemp("livewan-finite-epoch-cpu-fixture")
    data = root / "teacher"
    data.mkdir()
    prompts = root / "prompts.pt"
    torch.save(dict(pos=torch.tensor([.25, .5], dtype=torch.float16).view(2, 1, 1).expand(2, 512, 4096),
                    neg=torch.tensor(-.25, dtype=torch.float16).expand(1, 512, 4096),
                    prompts=["CPU epoch fixture A", "CPU epoch fixture B"], neg_prompt="CPU fixture"), prompts)
    for index in range(80):
        torch.save(sample(index), data / f"sample-{index:04d}.pt")
    assert sum(p.stat().st_size for p in data.glob("*.pt")) < 500_000
    return data, prompts


def prepare_epoch(data, prompts, path, source, *, cycles=100, seed=31415):
    return prepare_c51_inputs(data, prompts, path, source_root=source, cycles=cycles, seed=seed,
                              evidence_kind=EVIDENCE, input_policy=EPOCH_POLICY)


def test_old_v1_still_rejects_80_records_for_horizon100(original_source, finite_pool, tmp_path):
    data, prompts = finite_pool
    with pytest.raises(ValueError, match="exactly 80 distinct.*no padding/replay"):
        prepare_c51_inputs(data, prompts, tmp_path / "v1.json", source_root=original_source,
                           cycles=100, seed=31415, evidence_kind=EVIDENCE)
    assert not (tmp_path / "v1.json").exists()


def test_opt_in_epoch_pool_reaches_actual_native_owner(original_source, finite_pool, tmp_path):
    data, prompts = finite_pool
    value = prepare_epoch(data, prompts, tmp_path / "epochs.json", original_source)
    assert len(value["records"]) == 80 and len(value["rows"]) == 8000
    loader = InputLoader(torch=torch, input_kind=C51_KIND, prepared=tmp_path / "epochs.json")
    assert type(loader.inputs.dataset).__name__ == "TeacherLatents"
    assert Path(loader.inputs.dataset_source.__file__).resolve() == original_source / "wanstreamer/data.py"
    first, repeated = loader.inputs.row(0, 0), loader.inputs.row(5, 0)
    assert first.record_id == repeated.record_id and first.operation_key != repeated.operation_key
    a, b = loader.load(first), loader.load(repeated)
    assert all(torch.equal(x, y) for x, y in zip(a[:2], b[:2]))
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope="module")
def epoch_stream(original_source, finite_pool, tmp_path_factory):
    path = tmp_path_factory.mktemp("livewan-epoch-manifest") / "epochs.json"
    value = prepare_epoch(*finite_pool, path, original_source)
    return path, value


@pytest.fixture(scope="module")
def epoch_loader(epoch_stream):
    return InputLoader(torch=torch, input_kind=C51_KIND, prepared=epoch_stream[0])


def test_all_100_cycles_have_80_distinct_records_and_fresh_operation_rng(epoch_loader, epoch_stream):
    from distillschedule.data.livewan_inputs import STREAMS, _derived_seed, CONTRACT
    from distillschedule.training.livewan_inputs import EpochTeacherRow
    value = epoch_stream[1]
    rows = epoch_loader.inputs.rows
    assert value["contract"] == CONTRACT
    assert value["input_policy"]["draw_count"] == 8000 and value["input_policy"]["epoch_count"] == 100
    assert len({row.draw_id for row in rows}) == len({row.operation_key for row in rows}) == 8000
    assert len({row.cohort_id for row in rows}) == 100
    assert all(type(row) is EpochTeacherRow for row in rows)
    assert all(len({row.seeds[name] for row in rows}) == 8000 for name in STREAMS)
    assert len({_derived_seed(31415, row.operation_key, "control") for row in rows}) == 8000
    for cycle in range(100):
        cohort = rows[80*cycle:80*(cycle+1)]
        assert len({row.record_id for row in cohort}) == len({row.sample_path for row in cohort}) == 80
        assert [row.draw_index for row in cohort] == list(range(80*cycle, 80*(cycle+1)))
        assert all(row.epoch_index == cycle and row.epoch_offset == row.draw_index % 80 for row in cohort)
        for row in cohort:
            assert row.seeds == {name: _derived_seed(31415, row.operation_key, name) for name in STREAMS}
            control = random.Random(_derived_seed(31415, row.operation_key, "control"))
            assert row.world_frames == (3, 6, 9)[control.randrange(3)]
            assert row.rope_gap == (control.randint(1, 240) if control.random() < .7 else 0)
    assert epoch_loader.inputs.source_reuse_across_cycles is True
    assert len(list(Path(value["data_root"]).glob("*.pt"))) == 80  # never materialize 8000 fake-fresh archives


def native_rng_probe(row):
    """Existing worker RNG leaves plus its exact rollout/grad-block generator calls.

    Tiny CPU draws verify stream ownership only, not full native rollout math.
    """
    from distillschedule.runtime.livewan_crossscale_worker import _randn, _sample_flow_t, GRAD_BLOCKS
    return dict(
        rollout=torch.randn((8,), generator=torch.Generator().manual_seed(row.seeds["rollout"])),
        generator_dmd_t=_sample_flow_t(torch, seed=row.seeds["generator_dmd_t"], device="cpu"),
        generator_dmd_noise=_randn(torch, (8,), seed=row.seeds["generator_dmd_noise"], device="cpu"),
        generator_grad_blocks=torch.randperm(row.new_blocks, generator=torch.Generator().manual_seed(
            row.seeds["generator_grad_blocks"]))[:min(GRAD_BLOCKS, row.new_blocks)],
        critic_t=_sample_flow_t(torch, seed=row.seeds["critic_t"], device="cpu"),
        critic_noise=_randn(torch, (8,), seed=row.seeds["critic_noise"], device="cpu"))


def test_repeated_record_gets_new_actual_rng_draws_without_training_rng(epoch_loader):
    from test_training_livewan_state import equal
    random.seed(401); torch.manual_seed(402)
    py_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    first, second = epoch_loader.inputs.row(0, 0), epoch_loader.inputs.row(5, 0)
    a, b = native_rng_probe(first), native_rng_probe(second)
    assert first.record_id == second.record_id and not equal(a, b)
    assert all(not torch.equal(a[key], b[key]) for key in ("rollout", "generator_dmd_noise", "critic_noise"))
    assert equal(b, native_rng_probe(second))
    assert py_rng == random.getstate() and torch.equal(torch_rng, torch.get_rng_state())


def test_non_cycle_multiple_pool_preserves_epochs_without_dropping_or_within_cycle_reuse(
        original_source, finite_pool, tmp_path):
    data, prompts = finite_pool
    local = tmp_path / "teacher"
    shutil.copytree(data, local)
    torch.save(sample(80), local / "sample-0080.pt")
    value = prepare_epoch(local, prompts, tmp_path / "epochs.json", original_source, cycles=3)
    rows = value["rows"]
    assert len(value["records"]) == 81 and len(rows) == 240
    assert value["input_policy"]["epoch_count"] == 3
    assert {r["dataset_index"] for r in rows[:81]} == set(range(81))
    assert rows[0]["record_id"] == rows[81]["record_id"] and rows[81]["epoch_index"] == 1
    assert all(len({r["record_id"] for r in rows[80*c:80*(c+1)]}) == 80 for c in range(3))


@pytest.mark.parametrize("damage", ["insufficient", "duplicate_archive", "duplicate_latent"])
def test_epoch_pool_cannot_fake_distinct_records(original_source, finite_pool, tmp_path, damage):
    local = tmp_path / "teacher"
    shutil.copytree(finite_pool[0], local)
    target = local / "sample-0079.pt"
    if damage == "insufficient":
        target.unlink()
    elif damage == "duplicate_archive":
        shutil.copyfile(local / "sample-0078.pt", target)
    else:
        record = sample(79); record["latents"] = sample(78)["latents"].clone()
        torch.save(record, target)
    with pytest.raises(ValueError, match="at least 80|duplicate/copied/replayed"):
        prepare_epoch(local, finite_pool[1], tmp_path / "bad.json", original_source)
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize("damage", ["policy", "policy_order", "draw_count", "epoch_count", "record_count", "horizon", "seed",
                                    "source", "record_hash", "record_identity", "draw_index", "draw_identity", "epoch",
                                    "epoch_offset", "cohort", "operation_seed", "within_cycle_reuse", "draw_type"])
def test_epoch_manifest_drift_refused_by_c51_metadata_and_actual_consumer(epoch_stream, tmp_path, damage):
    from distillschedule.training.livewan_c51_training import input_metadata
    value = deepcopy(epoch_stream[1])
    if damage == "policy": value["input_policy"]["schema_version"] += ".unknown"
    elif damage == "policy_order": value["input_policy"]["record_order"].reverse()
    elif damage in {"draw_count", "epoch_count", "record_count"}: value["input_policy"][damage] -= 1
    elif damage == "horizon": value["cycles"] -= 1
    elif damage == "seed": value["seed"] += 1
    elif damage == "source": value["source"]["head"] = "0"*40
    elif damage == "record_hash": value["records"][0]["sha256"] = "0"*64
    elif damage == "record_identity": value["records"][0]["record_id"] += "changed"
    elif damage == "draw_index": value["rows"][0]["draw_index"] += 1
    elif damage == "draw_identity": value["rows"][0]["draw_id"] += "changed"
    elif damage == "epoch": value["rows"][0]["epoch_index"] += 1
    elif damage == "epoch_offset": value["rows"][0]["epoch_offset"] += 1
    elif damage == "cohort": value["rows"][0]["cohort_id"] += "changed"
    elif damage == "operation_seed": value["rows"][0]["seeds"]["rollout"] += 1
    elif damage == "draw_type": value["rows"][0]["draw_index"] = False
    else:
        # Preserve the logical address and fresh operation seeds, replay only a
        # dataset record inside one cycle. Even a hash-valid manifest rejects.
        first, repeated = value["rows"][0], value["rows"][1]
        for key in ("dataset_index", "record_id", "sample_path", "sample_sha256", "prompt_idx"):
            repeated[key] = first[key]
    value["input_sha256"] = fingerprint({k: v for k, v in value.items() if k != "input_sha256"})
    path = tmp_path / "changed.json"; path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        input_metadata(path, cycles=100, seed=31415, source=epoch_stream[1]["source"])
    with pytest.raises(ValueError):
        InputLoader(torch=torch, input_kind=C51_KIND, prepared=path)


def test_actual_c51_metadata_lowering_and_joint_census_use_epoch_draws(epoch_stream, epoch_loader):
    from distillschedule.training.livewan_c51_training import input_metadata, make_plan, validate_joint
    from distillschedule.training.livewan_c51_state import PoolInputs, physical_versions
    from distillschedule.runtime.livewan_crossscale_worker import _pipeline_8x8_slot_pairs
    from distillschedule.runtime.livewan_c51_bundle import validate_plan
    assert input_metadata(epoch_stream[0], cycles=100, seed=31415, source=epoch_stream[1]["source"]) == epoch_stream[1]
    plan = make_plan(dict(native=dict(models={r: dict(root="/explicit/"+r) for r in ("student", "critic", "teacher")})), "/unused")
    assert validate_plan(plan, mode="pipeline")
    pairs = _pipeline_8x8_slot_pairs(plan)
    observations = []
    for rank in range(16):
        pool = PoolInputs(epoch_loader, rank=rank, pairs=pairs)
        rows = pool.expected(49)
        assert all(row["epoch_index"] == 49 and row["draw_index"] // 80 == 49 for row in rows)
        observations.append(dict(rank=rank, cycle=50, versions=physical_versions(rank, 50),
                                 slots=list(pool.slots), rows=rows, quiescent=True))
    assert validate_joint(observations, cycle=50, pairs=pairs)["cursor"] == 250
    broken = deepcopy(observations)
    for rank in (1, 9): broken[rank]["rows"][0]["record_id"] = broken[0]["rows"][0]["record_id"]
    with pytest.raises(ValueError, match="incomplete/replayed"):
        validate_joint(broken, cycle=50, pairs=pairs)


@pytest.mark.parametrize("rank", [0, 8], ids=["Generator", "Score"])
def test_epoch_receipt_requires_declared_reuse_policy_not_relaxed_v1_gate(epoch_stream, epoch_loader, rank):
    from test_training_livewan_c51_continuation import pool, continuation_runtime_fixture
    from distillschedule.training.livewan_c51_training import input_marker
    from distillschedule.runtime.livewan_c51_bundle import continuation_receipt_errors
    cursor = pool(epoch_stream, rank)
    value = continuation_runtime_fixture(epoch_loader.inputs, cursor.pairs, rank=rank, start=49, end=51)
    assert continuation_receipt_errors(value, rank=rank, inputs=epoch_loader.inputs, pairs=cursor.pairs)
    value["continuation"].update(input_marker(epoch_loader.inputs))
    assert continuation_receipt_errors(value, rank=rank, inputs=epoch_loader.inputs, pairs=cursor.pairs) == []
    for key, bad in (("source_reuse_across_cycles", False), ("source_reuse_across_cycles", 1),
                     ("input_policy", "unversioned"), ("input_policy_sha256", "0"*64)):
        broken = deepcopy(value); broken["continuation"][key] = bad
        assert continuation_receipt_errors(broken, rank=rank, inputs=epoch_loader.inputs, pairs=cursor.pairs), key


@pytest.mark.parametrize("role", ["generator", "critic"])
def test_actual_named_owner_epoch_checkpoint_new_process_resume(original_source, finite_pool, tmp_path, role):
    from test_training_livewan_c51_continuation import cpu_owner_chain
    from test_training_livewan_state import equal
    path = tmp_path / "epochs4.json"
    prepare_epoch(*finite_pool, path, original_source, cycles=4)
    full = cpu_owner_chain(path, original_source, tmp_path / "full", role=role)
    cpu_owner_chain(path, original_source, tmp_path / "first", role=role, stop=2)
    command = [sys.executable, "-c", (
        "import sys; from test_training_livewan_c51_continuation import cpu_owner_chain; "
        "cpu_owner_chain(sys.argv[1],sys.argv[2],sys.argv[3],role=sys.argv[4],resume=sys.argv[5])"),
        str(path), str(original_source), str(tmp_path / "resumed"), role, str(tmp_path / "first/midpoint")]
    result = subprocess.run(command, env=_fresh_process_environment(),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    resumed = torch.load(tmp_path / "resumed/result.pt", weights_only=True)
    assert equal(full["state"], resumed["state"]) and equal(full["rng"], resumed["rng"])
    assert equal(full["outputs"][2 if role == "generator" else 10:], resumed["outputs"])


def epoch_cursor_resume_probe(manifest, checkpoint, output):
    """Fresh process, real C51 data owners/checkpoint, no model or CUDA math."""
    from test_training_livewan_c51_continuation import pool, consume
    from distillschedule.training.checkpoint import restore_checkpoint, rng_state
    g, score = pool((Path(manifest), None), 0), pool((Path(manifest), None), 8)
    identity = dict(scope="CPU_EPOCH_DATA_ONLY_NOT_DMD_OR_WORLD16", input_sha256=g.inputs.input_sha256, horizon=100)
    # Observe the actual original dataset; never substitute data or model math.
    cls, reads = type(g.inputs.dataset), []
    original = cls.__getitem__
    def observed(self, index):
        reads.append(index)
        return original(self, index)
    cls.__getitem__ = observed
    random.seed(999); torch.manual_seed(999)
    restore_checkpoint(checkpoint, {"generator_data": g, "score_data": score}, identity=identity)
    assert not reads and g.cycle == score.cycle == 50
    before = rng_state()
    row = g.inputs.row(250, g.slots[0])
    samples = native_rng_probe(row)
    g_rows, s_rows = consume(g, 50), consume(score, 50)
    assert g_rows == s_rows and len(reads) == 10
    from test_training_livewan_state import equal
    assert equal(before, rng_state())
    torch.save(dict(rows=g_rows, samples=samples, rng=before, cursor=g.state_dict()), output)


def test_horizon100_real_epoch_cursor_checkpoint50_new_process_restore(epoch_stream, tmp_path):
    from test_training_livewan_c51_continuation import pool, consume
    from test_training_livewan_state import equal
    from distillschedule.training.checkpoint import save_checkpoint, rng_state, pack_state, unpack_state
    g, score = pool(epoch_stream, 0), pool(epoch_stream, 8)
    random.seed(401); torch.manual_seed(402)
    before = rng_state()
    for cycle in range(50):
        assert consume(g, cycle) == consume(score, cycle)
    assert g.cycle == score.cycle == 50 and equal(before, rng_state())
    identity = dict(scope="CPU_EPOCH_DATA_ONLY_NOT_DMD_OR_WORLD16", input_sha256=g.inputs.input_sha256, horizon=100)
    checkpoint = tmp_path / "cycle-000050"
    save_checkpoint(checkpoint, {"generator_data": g, "score_data": score}, identity=identity, cycle=50,
                    cursor=250, versions=dict(generator=50, critic=250, ema=0), policy=dict(scope="CPU_DATA_ONLY"))
    packed = pack_state({"data": g})
    fresh = pool(epoch_stream, 0)
    for key, bad in (("input_sha256", "0"*64), ("cursor", 245), ("cycle", 49), ("next_keys", []), ("history", [])):
        changed = deepcopy(packed); changed["data"]["value"][key] = bad
        with pytest.raises(ValueError): unpack_state(changed, {"data": fresh})
        assert fresh.cycle == 0
    row = g.inputs.row(250, g.slots[0])
    expected_samples = native_rng_probe(row)
    expected_rows = consume(g, 50)
    assert expected_rows[0]["draw_index"] == 4000 and expected_rows[0]["epoch_index"] == 50
    output = tmp_path / "restored.pt"
    run = subprocess.run([sys.executable, "-c", (
        "import sys; from test_training_livewan_epoch_inputs import epoch_cursor_resume_probe; "
        "epoch_cursor_resume_probe(*sys.argv[1:])"), str(epoch_stream[0]), str(checkpoint), str(output)],
        env=_fresh_process_environment(),
        capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout+run.stderr
    restored = torch.load(output, weights_only=True)
    assert restored["rows"] == expected_rows and restored["cursor"] == g.state_dict()
    assert equal(restored["samples"], expected_samples) and equal(restored["rng"], before)
    assert not torch.cuda.is_initialized()


def test_public_preparation_opt_in_reaches_same_c51_loader(original_source, finite_pool, tmp_path, capsys):
    from distillschedule.data.cli import main
    output = tmp_path / "epochs.json"
    assert main(["prepare-livewan-c51-inputs", "--data-root", str(finite_pool[0]), "--prompts", str(finite_pool[1]),
                 "--source-root", str(original_source), "--cycles", "100", "--seed", "31415", "--output", str(output),
                 "--input-policy", EPOCH_POLICY, "--evidence-kind", EVIDENCE]) == 0
    result = json.loads(capsys.readouterr().out)
    loader = InputLoader(torch=torch, input_kind=C51_KIND, prepared=output)
    assert result["input_policy"]["schema_version"] == EPOCH_POLICY
    assert result["input_sha256"] == loader.inputs.input_sha256
    assert loader.load(loader.inputs.row(495, 0))[2] == loader.inputs.row(495, 0).prompt_idx


def test_epoch_policy_does_not_change_legacy_input_or_public_training_config(epoch_stream, tmp_path):
    from distillschedule.data.livewan_inputs import prepare_inputs, KIND
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.native import _options
    with pytest.raises(ValueError, match="requires explicit C51"):
        prepare_inputs("/missing", "/missing", tmp_path / "invalid.json", source_root="/missing", cycles=2, seed=1,
                       input_policy=EPOCH_POLICY)
    raw = dict(schema_version="distillschedule.training.v1", model=dict(workload="W4"),
        algorithm=dict(name="original_self_forcing_c51"), data=dict(input_kind=C51_KIND, prepared=str(epoch_stream[0])),
        execution=dict(adapter="native", mode="manual", task="/explicit/task.json", assets="/explicit/assets.json",
                       capacity_policy="/explicit/capacity.json", preset="self_forcing.c51_stream.v1"),
        trainer=dict(cycles=100, seed=31415, checkpoint_every=50, output_dir="/explicit/output"))
    assert _options(TrainingConfig.from_dict(raw, source=tmp_path / "training.json"), "W4") == "manual"
    raw["data"]["input_kind"] = KIND
    with pytest.raises(ValueError):
        _options(TrainingConfig.from_dict(raw, source=tmp_path / "training.json"), "W4")


@pytest.mark.parametrize("change", ["horizon", "seed", "policy"])
def test_real_cursor_refuses_different_prepared_horizon_seed_or_policy(
        original_source, finite_pool, epoch_stream, tmp_path, change):
    from test_training_livewan_c51_continuation import pool
    if change == "policy":
        first, second = tmp_path / "epoch-one.json", tmp_path / "fresh-one.json"
        prepare_epoch(*finite_pool, first, original_source, cycles=1)
        prepare_c51_inputs(*finite_pool, second, source_root=original_source, cycles=1, seed=31415, evidence_kind=EVIDENCE)
    else:
        first, second = epoch_stream[0], tmp_path / "changed.json"
        prepare_epoch(*finite_pool, second, original_source, cycles=99 if change == "horizon" else 100,
                      seed=31416 if change == "seed" else 31415)
    current, changed = pool((first, None)), pool((second, None))
    before = changed.state_dict()
    with pytest.raises(ValueError, match="source/input/rank/slot/wave/history/cursor"):
        changed.load_state_dict(current.state_dict())
    assert changed.state_dict() == before


def test_epoch_native_record_hash_failure_does_not_advance_actual_pool(original_source, finite_pool, tmp_path):
    from test_training_livewan_c51_continuation import pool
    from dataclasses import replace
    data = tmp_path / "teacher"; shutil.copytree(finite_pool[0], data)
    manifest = tmp_path / "epochs.json"
    prepare_epoch(data, finite_pool[1], manifest, original_source, cycles=2)
    cursor = pool((manifest, None))
    before = cursor.state_dict()
    row = cursor.inputs.row(0, cursor.slots[0])
    with pytest.raises(ValueError, match="row identity|lane sample"):
        cursor.load(replace(row, draw_id="replayed-draw"))
    Path(row.sample_path).write_bytes(Path(row.sample_path).read_bytes()+b"changed archive bytes")
    with pytest.raises(ValueError, match="record hash changed"):
        cursor.load(row)
    assert cursor.state_dict() == before and not cursor.pending


def test_epoch_policy_reaches_actual_public_dispatch_and_continuous_worker_lowering(
        original_source, finite_pool, tmp_path, monkeypatch, record_property):
    from test_training_livewan_c51_continuation import cpu_continuation_chain
    from distillschedule.runtime import livewan_crossscale_worker as worker
    manifest = tmp_path / "epochs4.json"
    value = prepare_epoch(*finite_pool, manifest, original_source, cycles=4)
    original, observations = worker._run_pipeline_8x8_continuous, []
    def observed(**kwargs):
        result = original(**kwargs)
        observations.append(deepcopy(kwargs["evidence"].continuous_pipeline))
        return result
    monkeypatch.setattr(worker, "_run_pipeline_8x8_continuous", observed)
    result = cpu_continuation_chain(manifest, original_source, tmp_path / "stopped", stop=2)
    assert result["completed"] == 2 and (tmp_path / "stopped/checkpoints/cycle-000002/_SUCCESS").is_file()
    assert not any(row[0] == "send" and row[1] >= 10 for row in result["timeline"])
    assert len(observations) == 1
    marker = observations[0]
    assert marker["source_reuse_across_cycles"] is True and marker["input_policy"] == EPOCH_POLICY
    assert marker["input_policy_sha256"] == fingerprint(value["input_policy"])
    assert marker["training_quality_claimable"] is False and marker["formal_timing_eligible"] is False
    record_property("scope", "REAL_CPU_PUBLIC_DISPATCH_DATA_OWNER_LOOP_CHECKPOINT; EXPLICIT_CUDA_GLOO_NATIVE_MATH_STANDINS; NOT_GPU")
