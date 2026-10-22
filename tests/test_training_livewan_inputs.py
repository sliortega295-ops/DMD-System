"""CPU fixtures with native shapes, NOT pretrained LiveWan quality evidence.

The opt-in source is the real licensed gitlink, imported directly; neither the
upstream TeacherLatents nor the production InputLoader is replaced by a fake.
Expanded tensors keep these metadata/ownership fixtures small on disk.
"""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

import pytest
import torch

from distillschedule.runtime.livewan_crossscale_worker import FormalTrace, InputLoader, OperationTraceRow
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint


KIND = "original_livewan_teacher_cohorts.v1"
PIN = "1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5"


@pytest.fixture(scope="module")
def original_source():
    value = os.environ.get("DMD_LIVEWAN_TEST_UPSTREAM")
    if not value:
        pytest.skip("real pinned LiveWan source requires DMD_LIVEWAN_TEST_UPSTREAM")
    root = Path(value).resolve()
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == PIN
    return root


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("livewan-native-shape-fixture")
    data = root / "teacher"
    data.mkdir()
    prompts = root / "prompts.pt"
    torch.save({"pos": torch.tensor([.25, .5], dtype=torch.float16).view(2, 1, 1).expand(2, 512, 4096),
                "neg": torch.tensor(-.25, dtype=torch.float16).expand(1, 512, 4096),
                "prompts": ["CPU fixture prompt A", "CPU fixture prompt B"],
                "neg_prompt": "CPU fixture negative"}, prompts)
    for index in range(160):
        torch.save(sample(index), data / f"sample-{index:04d}.pt")
    assert sum(p.stat().st_size for p in data.glob("*.pt")) < 1_000_000
    return data, prompts


def sample(index):
    return {"latents": torch.tensor(index / 128, dtype=torch.float16).expand(16, 21, 46, 80),
            "prompt_idx": index % 2, "seed": index, "size": "640*368", "frames": 81,
            "steps": 40, "cfg": 5.0}


@pytest.mark.parametrize("layout", ["contiguous", "strided", "expanded"])
def test_native_tensor_hash_matches_original_byte_serialization(layout):
    import hashlib
    from distillschedule.data.livewan_inputs import _tensor_hash
    value = torch.arange(24, dtype=torch.float16).reshape(4, 6)
    if layout == "strided":
        value = value[:, ::2]
    elif layout == "expanded":
        value = value[:1].expand(4, 6)
    expected = hashlib.sha256()
    expected.update(fingerprint({"shape": list(value.shape), "dtype": str(value.dtype)}).encode())
    expected.update(value.detach().contiguous().view(torch.uint8).numpy().tobytes())
    assert _tensor_hash(value) == expected.hexdigest()


@pytest.fixture
def local_corpus(corpus, tmp_path):
    data, prompts = corpus
    target = tmp_path / "teacher"
    shutil.copytree(data, target)
    prompt_target = tmp_path / "prompts.pt"
    shutil.copyfile(prompts, prompt_target)
    return target, prompt_target


def prepare(data, prompts, output, source):
    from distillschedule.data.livewan_inputs import prepare_inputs
    return prepare_inputs(data, prompts, output, source_root=source, cycles=2, seed=20260830,
                          evidence_kind="CPU_NATIVE_SHAPE_FIXTURE_NOT_PRETRAINED")


def original_dataset(source, data, prompts):
    # Direct import of the actual pinned file, without sys.path/package stubs.
    spec = importlib.util.spec_from_file_location("livewan_original_data_baseline", source / "wanstreamer/data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TeacherLatents(str(data), str(prompts), frames=None)


def test_real_source_baseline_has_independent_record_and_prompt_indices(original_source, corpus):
    data, prompts = corpus
    dataset = original_dataset(original_source, data, prompts)
    latent, context, prompt = dataset[3]
    assert len(dataset) == 160 and prompt == 1
    assert tuple(latent.shape) == (16, 21, 46, 80) and tuple(context.shape) == (512, 4096)
    assert latent.dtype == context.dtype == torch.float32
    assert latent[0, 0, 0, 0].item() == 3 / 128
    assert context[0, 0].item() == .5


def test_historical_loader_control_still_uses_trace_prompt_identity(original_source, corpus, tmp_path):
    data, prompts = corpus
    path = data / "sample-0003.pt"
    row = OperationTraceRow(0, 0, 3, str(path), file_sha256(path), "control", 9, 0, {}, True, True)
    trace = FormalTrace(tmp_path / "trace.json", "a" * 64, prompts, file_sha256(prompts), (row,))
    loader = InputLoader(torch=torch, trace=trace)
    with pytest.raises(RuntimeError, match="prompt identity"):
        loader.load(row)
    latent, context, pi = loader.load(replace(row, dataset_index=1))
    assert pi == 1 and latent[0, 0, 0, 0].item() == 3 / 128 and context[0, 0].item() == .5


def test_actual_original_and_real_input_loader_roundtrip(original_source, corpus, tmp_path):
    from distillschedule.training.livewan_inputs import NativeTeacherCursor
    data, prompts = corpus
    path = tmp_path / "inputs.json"
    manifest = prepare(data, prompts, path, original_source)
    loader = InputLoader(torch=torch, input_kind=KIND, prepared=path)
    assert type(loader.inputs.dataset).__name__ == "TeacherLatents"
    assert Path(loader.inputs.dataset_source.__file__).resolve() == original_source / "wanstreamer/data.py"
    assert manifest["source"]["head"] == PIN and manifest["source"]["license"] == "Apache-2.0"
    assert manifest["claim_boundary"]["training_lifecycle"] == "NOT_IMPLEMENTED"
    row = next(r for r in loader.inputs.rows if r.dataset_index == 3)
    assert row.prompt_idx == 1 and row.record_id != loader.inputs.records[1]["record_id"]
    actual = loader.load(row)
    expected = original_dataset(original_source, data, prompts)[3]
    assert actual[2] == expected[2] == 1
    assert all(torch.equal(a, b) for a, b in zip(actual[:2], expected[:2]))
    assert torch.equal(loader.context_from_trace(row, device="cpu")[0], expected[1])
    assert tuple(loader.negative.shape) == (1, 512, 4096)
    assert torch.equal(loader.negative, torch.load(prompts, weights_only=True)["neg"].float())
    cursor = NativeTeacherCursor(loader, rank=row.logical_slot)
    cursor.seek(row.cycle_index)
    assert cursor.peek().cycle_index == row.cycle_index
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mutation", ["missing", "changed", "geometry", "dtype", "prompt", "metadata", "duplicate", "aliased", "copied", "incomplete"])
def test_refuses_nonfresh_or_invalid_native_records(original_source, local_corpus, tmp_path, mutation):
    data, prompts = local_corpus
    target = data / "sample-0003.pt"
    value = sample(3)
    if mutation == "missing":
        target.unlink()
    elif mutation == "changed":
        # A valid initial manifest must not accept a later byte change.
        path = tmp_path / "inputs.json"
        prepare(data, prompts, path, original_source)
        value["cfg"] = 6.0
        torch.save(value, target)
        with pytest.raises(ValueError, match="changed|identity|hash"):
            InputLoader(torch=torch, input_kind=KIND, prepared=path)
        return
    elif mutation == "geometry":
        value["latents"] = torch.tensor(.1, dtype=torch.float16).expand(16, 22, 46, 80)
    elif mutation == "dtype":
        value["latents"] = value["latents"].float()
    elif mutation == "prompt":
        value["prompt_idx"] = 2
    elif mutation == "metadata":
        value["frames"] = 69
    elif mutation == "duplicate":
        shutil.copyfile(data / "sample-0002.pt", target)
    elif mutation == "aliased":
        target.unlink()
        target.symlink_to(data / "sample-0002.pt")
    elif mutation == "copied":
        # Reserializing with a distinct filename/metadata changes archive SHA,
        # not the latent identity; this must not earn a fresh dataset record.
        value["latents"] = sample(2)["latents"].clone()
        value["seed"] = 98765
    elif mutation == "incomplete":
        (data / "sample-0159.pt").unlink()
    if mutation not in {"missing", "duplicate", "aliased", "incomplete"}:
        torch.save(value, target)
    with pytest.raises(ValueError, match="record|geometry|dtype|prompt|metadata|fresh|cohort"):
        prepare(data, prompts, tmp_path / "invalid.json", original_source)
    assert not (tmp_path / "invalid.json").exists()


@pytest.mark.parametrize("mutation", ["positive_geometry", "negative_dtype", "prompt_text_count"])
def test_original_prompt_bundle_is_exact_not_repaired(original_source, local_corpus, tmp_path, mutation):
    data, prompts = local_corpus
    value = torch.load(prompts, weights_only=True)
    if mutation == "positive_geometry":
        value["pos"] = torch.tensor(.25, dtype=torch.float16).expand(2, 511, 4096)
    elif mutation == "negative_dtype":
        value["neg"] = value["neg"].float()
    else:
        value["prompts"] = value["prompts"][:1]
    torch.save(value, prompts)
    with pytest.raises(ValueError, match="prompt.*geometry/dtype"):
        prepare(data, prompts, tmp_path / "invalid.json", original_source)


def test_missing_actual_pinned_source_does_not_publish_an_input_manifest(corpus, tmp_path):
    data, prompts = corpus
    with pytest.raises(ValueError, match="source"):
        prepare(data, prompts, tmp_path / "invalid.json", tmp_path / "unpopulated-livewan")
    assert not (tmp_path / "invalid.json").exists()


@pytest.mark.parametrize("mutation", ["source_head", "source_hash", "prompt_hash", "prompt_binding", "row", "manifest_hash"])
def test_source_prompt_operation_and_manifest_bindings_fail_closed(original_source, corpus, tmp_path, mutation):
    data, prompts = corpus
    path = tmp_path / "inputs.json"
    doc = prepare(data, prompts, path, original_source)
    if mutation == "source_head":
        doc["source"]["head"] = "0" * 40
    elif mutation == "source_hash":
        doc["source"]["files"]["wanstreamer/data.py"] = "0" * 64
    elif mutation == "prompt_hash":
        doc["prompts"]["sha256"] = "0" * 64
    elif mutation == "prompt_binding":
        doc["records"][3]["prompt_idx"] = 0
    elif mutation == "row":
        doc["rows"][0]["seeds"]["critic_noise"] += 1
    else:
        doc["input_sha256"] = "0" * 64
    if mutation != "manifest_hash":
        doc["input_sha256"] = fingerprint({k: v for k, v in doc.items() if k != "input_sha256"})
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="identity|changed|hash|source"):
        InputLoader(torch=torch, input_kind=KIND, prepared=path)


def test_next_native_read_rechecks_bytes_and_row_identity(original_source, local_corpus, tmp_path):
    data, prompts = local_corpus
    path = tmp_path / "inputs.json"
    prepare(data, prompts, path, original_source)
    loader = InputLoader(torch=torch, input_kind=KIND, prepared=path)
    row = loader.inputs.rows[0]
    with pytest.raises(ValueError, match="row identity"):
        loader.load(replace(row, prompt_idx=1 - row.prompt_idx))
    target = Path(row.sample_path)
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash|changed"):
        loader.load(row)


def test_manifest_bytes_cannot_change_during_original_reinventory(original_source, corpus, tmp_path, monkeypatch):
    from distillschedule.training import livewan_inputs
    data, prompts = corpus
    path = tmp_path / "inputs.json"
    prepare(data, prompts, path, original_source)
    inventory = livewan_inputs._inventory
    def observed(*args, **kwargs):
        result = inventory(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(livewan_inputs, "_inventory", observed)
    with pytest.raises(ValueError, match="manifest hash changed during"):
        InputLoader(torch=torch, input_kind=KIND, prepared=path)


def test_complete_cohorts_rank_cursor_checkpoint_seek_and_rng(original_source, corpus, tmp_path):
    from distillschedule.training.checkpoint import pack_state, unpack_state
    from distillschedule.training.livewan_inputs import NativeTeacherCursor
    from tools.assets.build_livewan_crossscale_trace import derived_seed
    data, prompts = corpus
    path = tmp_path / "inputs.json"
    random.seed(401)
    torch.manual_seed(402)
    py_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    manifest = prepare(data, prompts, path, original_source)
    loader = InputLoader(torch=torch, input_kind=KIND, prepared=path)
    rows = loader.inputs.rows
    assert len(rows) == 160 and len({r.record_id for r in rows}) == 160
    assert {r.record_id for r in rows[:80]}.isdisjoint({r.record_id for r in rows[80:]})
    assert [(r.logical_step, r.logical_slot) for r in rows] == [(s, k) for s in range(10) for k in range(16)]
    assert all(r.generator_update == (r.cycle_step == 0) and r.critic_update for r in rows)
    assert all(r.seeds[name] == derived_seed(20260830, r.operation_key, name) for r in rows for name in r.seeds)
    rank_rows = []
    for rank in range(16):
        cursor = NativeTeacherCursor(loader, rank=rank)
        rank_rows.append(cursor.peek().record_id)
        for _ in range(5):
            row, (latent, context, prompt) = next(cursor)
            assert row.logical_slot == rank and prompt == row.prompt_idx
        state = pack_state({"data": cursor})
        following = cursor.peek()
        restored = NativeTeacherCursor(loader, rank=rank)
        unpack_state(state, {"data": restored})
        assert restored.peek() == following and restored.state_dict() == state["data"]["value"]
        uninterrupted, resumed = next(cursor), next(restored)
        assert uninterrupted[0] == resumed[0] and uninterrupted[1][2] == resumed[1][2]
        assert all(torch.equal(a, b) for a, b in zip(uninterrupted[1][:2], resumed[1][:2]))
        restored.seek(2)
        with pytest.raises(StopIteration):
            next(restored)
        with pytest.raises(ValueError, match="cursor|rank|identity"):
            NativeTeacherCursor(loader, rank=(rank + 1) % 16).load_state_dict(state["data"]["value"])
    assert len(set(rank_rows)) == 16
    cursor = NativeTeacherCursor(loader, rank=0)
    next(cursor)
    with pytest.raises(ValueError, match="complete.*cycle"):
        cursor.state_dict()
    assert random.getstate() == py_rng and torch.equal(torch.get_rng_state(), torch_rng)
    assert manifest["contract"]["ema_updates"] == 0


def test_cursor_state_admission_is_nonmutating_and_identity_bound(original_source, corpus, tmp_path):
    from distillschedule.training.checkpoint import pack_state, unpack_state
    from distillschedule.training.livewan_inputs import NativeTeacherCursor
    data, prompts = corpus
    path = tmp_path / "inputs.json"
    prepare(data, prompts, path, original_source)
    loader = InputLoader(torch=torch, input_kind=KIND, prepared=path)
    cursor = NativeTeacherCursor(loader, rank=0)
    for key, bad in (("input_sha256", "0" * 64), ("world_size", 8), ("world_size", 16.0),
                     ("rank", False), ("cycle", 0.0), ("cycle", 1),
                     ("cursor", 1), ("next_operation_key", "replayed-operation")):
        packed = pack_state({"data": cursor})
        packed["data"]["value"][key] = bad
        with pytest.raises(ValueError, match="cursor|identity"):
            unpack_state(packed, {"data": cursor})
        assert cursor.cursor == 0


def test_cursor_does_not_advance_on_failed_native_consumption(original_source, local_corpus, tmp_path):
    from distillschedule.training.livewan_inputs import NativeTeacherCursor
    data, prompts = local_corpus
    path = tmp_path / "inputs.json"
    prepare(data, prompts, path, original_source)
    loader = InputLoader(torch=torch, input_kind=KIND, prepared=path)
    cursor = NativeTeacherCursor(loader, rank=0)
    Path(cursor.peek().sample_path).unlink()
    with pytest.raises(ValueError, match="record hash changed"):
        next(cursor)
    assert cursor.cursor == 0


def test_explicit_input_route_does_not_fall_back_to_historical_trace(original_source, corpus, tmp_path):
    data, prompts = corpus
    trace = FormalTrace(tmp_path / "trace.json", "a" * 64, prompts, file_sha256(prompts), ())
    for kwargs in ({"trace": trace, "input_kind": KIND, "prepared": "unused.json"},
                   {"input_kind": "unknown", "prepared": "unused.json"},
                   {"prepared": "unused.json"}):
        with pytest.raises(RuntimeError, match="explicit input_kind"):
            InputLoader(torch=torch, **kwargs)


def test_historical_default_prefix_behavior_is_unchanged(corpus, tmp_path):
    _, prompts = corpus
    path = tmp_path / "long-historical.pt"
    torch.save({"latents": torch.tensor(.25).expand(16, 22, 46, 80), "prompt_idx": 0}, path)
    row = OperationTraceRow(0, 0, 0, str(path), None, "historical", 9, 0, {}, True, True)
    trace = FormalTrace(tmp_path / "trace.json", "a" * 64, prompts, file_sha256(prompts), (row,))
    assert tuple(InputLoader(torch=torch, trace=trace).load(row)[0].shape) == (16, 21, 46, 80)


class UnsafeArchive:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.system, (f"touch {self.marker}",)


def test_native_original_read_uses_admitted_path_after_symlink_retarget(original_source, local_corpus, tmp_path):
    data, prompts = local_corpus
    alias = data / "sample-0003.pt"
    admitted = tmp_path / "admitted-record.pt"
    alias.rename(admitted)
    alias.symlink_to(admitted)
    manifest = tmp_path / "inputs.json"
    prepare(data, prompts, manifest, original_source)
    loader = InputLoader(torch=torch, input_kind=KIND, prepared=manifest)
    row = next(r for r in loader.inputs.rows if r.dataset_index == 3)
    marker = tmp_path / "UNADMITTED_ARCHIVE_EXECUTED"
    replacement = tmp_path / "unadmitted.pt"
    torch.save(UnsafeArchive(marker), replacement)
    alias.unlink()
    alias.symlink_to(replacement)
    try:
        latent, _, prompt = loader.load(row)
    except (TypeError, ValueError):
        assert not marker.exists(), "unadmitted alias target was unpickled"
        raise
    assert not marker.exists()
    assert prompt == 1 and latent[0, 0, 0, 0].item() == 3 / 128
    assert isinstance(loader.inputs.dataset.files, tuple)


def test_failed_manifest_write_leaves_no_partial_output_and_can_retry(original_source, corpus, tmp_path, monkeypatch):
    data, prompts = corpus
    output = tmp_path / "inputs.json"
    def interrupted(value, handle, **kwargs):
        handle.write('{"partial":')
        raise OSError("injected disk write failure")
    with monkeypatch.context() as patch:
        patch.setattr(json, "dump", interrupted)
        with pytest.raises(OSError, match="injected disk"):
            prepare(data, prompts, output, original_source)
    assert not output.exists(), "partial final manifest prevents a clean retry"
    assert not list(tmp_path.glob(".inputs.json.*"))
    value = prepare(data, prompts, output, original_source)
    assert json.loads(output.read_text()) == value
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(data, prompts, output, original_source)
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(".inputs.json.*"))


def test_public_original_teacher_preparation_reaches_real_native_loader(original_source, corpus, tmp_path, capsys):
    from distillschedule.data.cli import main
    data, prompts = corpus
    output = tmp_path / "inputs.json"
    assert main(["prepare-livewan-inputs", "--data-root", str(data), "--prompts", str(prompts),
                 "--source-root", str(original_source), "--cycles", "2", "--seed", "20260830",
                 "--evidence-kind", "CPU_NATIVE_SHAPE_FIXTURE_NOT_PRETRAINED",
                 "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    loader = InputLoader(torch=torch, input_kind=KIND, prepared=output)
    assert result["input_sha256"] == loader.inputs.input_sha256
    assert loader.load(loader.inputs.rows[0])[2] == loader.inputs.rows[0].prompt_idx
    assert not torch.cuda.is_initialized()


def test_single_complete_cohort_preserves_native_keys_rng_and_exhaustion(original_source, local_corpus, tmp_path):
    from distillschedule.data.livewan_inputs import prepare_inputs
    from distillschedule.training.livewan_inputs import NativeTeacherCursor
    from tools.assets.build_livewan_crossscale_trace import derived_seed
    data, prompts = local_corpus
    for path in sorted(data.glob("*.pt"))[80:]:
        path.unlink()
    output = tmp_path / "one-cycle.json"
    rng, py_rng = torch.get_rng_state().clone(), random.getstate()
    value = prepare_inputs(data, prompts, output, source_root=original_source, cycles=1, seed=20260830,
                           evidence_kind="CPU_NATIVE_SHAPE_FIXTURE_NOT_PRETRAINED")
    loader = InputLoader(torch=torch, input_kind=KIND, prepared=output)
    assert len(value["rows"]) == 80
    for row in loader.inputs.rows:
        assert row.operation_key == f"step={row.logical_step}/slot={row.logical_slot}"
        assert all(row.seeds[key] == derived_seed(20260830, row.operation_key, key) for key in row.seeds)
    cursor = NativeTeacherCursor(loader, rank=3)
    checks = []
    unchanged = loader.inputs._unchanged
    def observed():
        checks.append(True)
        unchanged()
    loader.inputs._unchanged = observed
    assert [next(cursor)[0].logical_step for _ in range(5)] == list(range(5))
    assert len(checks) == 10  # each original read remains bracketed; no extra cursor pre-scan
    assert cursor.state_dict()["next_operation_key"] is None
    with pytest.raises(StopIteration):
        next(cursor)
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()


def test_original_owner_ci_covers_source_pins_and_historical_context():
    import fnmatch
    import yaml
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.load((root / ".github/workflows/livewan-native-inputs-cpu.yml").read_text(), Loader=yaml.BaseLoader)
    patterns = workflow["on"]["pull_request"]["paths"]
    for changed in ("third_party/livewan", ".gitmodules", "src/distillschedule/workloads/livewan_crossscale.py",
                    "src/distillschedule/search/livewan_request.py", "src/distillschedule/workloads/best_known/registry.py",
                    "src/distillschedule/data/livewan_inputs.py", "tests/test_refactor_livewan_context_staging.py",
                    "tests/test_training_environment.py"):
        assert any(fnmatch.fnmatchcase(changed, pattern) for pattern in patterns), changed
    runs = "\n".join(step.get("run", "") for step in workflow["jobs"]["original-data-owner"]["steps"])
    assert "tests/test_refactor_livewan_context_staging.py" in runs
    assert "tests/test_training_environment.py" in runs
    for name in (
        "test_real_detached_projector_first_boundary_save_new_owner_rng_continuation",
        "test_real_driver_observer_producer_feeds_owned_binding_and_capacity_consumer",
        "test_rank_local_owned_uuid_probe_failure_returns_to_real_gloo_peer_before_payload",
        "test_token_rotation_does_not_change_checkpoint_identity[livewan_reference_assets]",
        "test_dispatch_only_outer_lifecycle_reuses_actual_baseline_and_real_loads[False]",
        "test_dispatch_only_outer_lifecycle_reuses_actual_baseline_and_real_loads[True]",
    ):
        assert name in runs


@pytest.mark.parametrize("archive", ["record", "prompts"])
def test_malicious_archive_refused_before_original_unsafe_loading(original_source, local_corpus, tmp_path, archive, monkeypatch):
    from distillschedule.data import livewan_inputs
    data, prompts = local_corpus
    marker = tmp_path / "MUST_NOT_EXIST"
    target = prompts if archive == "prompts" else data / "sample-0000.pt"
    torch.save(UnsafeArchive(marker), target)
    entered = []
    original = livewan_inputs._original_dataset
    def observed(*args, **kwargs):
        entered.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(livewan_inputs, "_original_dataset", observed)
    with pytest.raises(ValueError, match="safely|restricted"):
        prepare(data, prompts, tmp_path / "invalid.json", original_source)
    assert not marker.exists() and not entered


def test_fresh_input_path_does_not_publish_an_unimplemented_w4_training_lifecycle(tmp_path):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.native import validate_native
    config = TrainingConfig.from_dict({"schema_version": "distillschedule.training.v1",
        "model": {"workload": "W4"}, "algorithm": {},
        "data": {"input_kind": KIND, "prepared": "inputs.json"},
        "execution": {"adapter": "native", "mode": "plan", "task": "task.json", "assets": "assets.json", "plan": "plan.json"},
        "trainer": {"cycles": 2, "checkpoint_every": 0, "output_dir": "run"}}, source=tmp_path / "training.json")
    with pytest.raises(ValueError, match="W4 fresh.*lifecycle"):
        validate_native(config)
