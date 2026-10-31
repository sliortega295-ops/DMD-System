"""Opt-in original W5 lifecycle; CPU evidence is not pretrained H3 parity."""
import json
import os
from pathlib import Path

import pytest

from distillschedule.training.config import TrainingConfig
from distillschedule.training.native import describe_native, validate_native


INPUT = "original_h3_grid_latent_dataset.v1"


@pytest.fixture
def original_source(monkeypatch):
    root = os.environ.get("DMD_H3_TEST_UPSTREAM")
    if not root:
        pytest.skip("real pinned LightX2V source requires DMD_H3_TEST_UPSTREAM")
    root = Path(root).resolve()
    assert (root / "lightx2v_train/lightx2v_train/data/video_dataset.py").is_file()
    monkeypatch.syspath_prepend(str(root / "lightx2v_train"))
    return root


def test_actual_loop_import_does_not_require_disabled_optional_triton(original_source):
    import torch
    from distillschedule.backends.training.minimax_h3.trainer_loop import H3A100LoopMixin
    assert H3A100LoopMixin.__module__ == "distillschedule.backends.training.minimax_h3.trainer_loop"
    assert torch.cuda.is_initialized() is False


def test_actual_grid_registration_defers_triton_to_its_original_setup(original_source):
    import torch
    from distillschedule.backends.training.minimax_h3 import grid_adaln
    from distillschedule.backends.training.minimax_h3.trainer import MiniMaxH3A100DmdTrainer
    assert grid_adaln.DEFAULT_GRID_SIZE == 1000
    assert MiniMaxH3A100DmdTrainer._sample_renoise_sigmas.__name__ == "sample_grid_sigmas"
    assert torch.cuda.is_initialized() is False


def training(tmp_path, **trainer):
    return TrainingConfig.from_dict({
        "schema_version": "distillschedule.training.v1",
        "model": {"workload": "W5"}, "algorithm": {},
        "data": {"input_kind": INPUT, "prepared": "latent-inventory.json"},
        "execution": {"adapter": "native", "mode": "reference",
                      "task": "task.json", "assets": "assets.json"},
        "trainer": {"cycles": 3, "seed": 20260817, "checkpoint_every": 1,
                    "output_dir": "run", **trainer},
    }, source=tmp_path / "training.json")


def test_original_grid_lifecycle_is_explicit_and_described_without_model_import(tmp_path):
    result = describe_native(training(tmp_path))
    assert result["capabilities"]["modes"] == ["reference"]
    assert result["capabilities"]["cycles"]["fresh_inputs"] == INPUT
    assert result["capabilities"]["checkpoint"]["supported"] is True
    assert result["capabilities"]["resume"]["supported"] is True
    assert result["claimable_speedup"] is False
    assert result["gpu_execution"] == "NOT_RUN"


def test_original_grid_options_are_not_silently_rejected_as_benchmark_options(tmp_path):
    # The new lifecycle admits the controls, then checks the explicit native
    # config/source/input files. It must not drop seed/cycles/checkpoint knobs.
    with pytest.raises(ValueError, match="missing execution.task"):
        validate_native(training(tmp_path))


def test_original_grid_does_not_accept_a_selected_optimized_plan(tmp_path):
    conf = training(tmp_path).to_dict()
    conf["execution"].update(mode="plan", plan="failed-optimized-plan.json")
    with pytest.raises(ValueError, match="reference"):
        validate_native(TrainingConfig.from_dict(conf, source=tmp_path / "training.json"))


@pytest.fixture
def cached_data(tmp_path):
    import torch
    rows = []
    for index in range(288):
        path = tmp_path / f"condition-{index:04d}.pt"
        torch.save({"positive": {"prompt_embeds": torch.full((1, 2, 3), float(index)),
                                  "text_token_tags": torch.tensor([0, 1])}}, path)
        rows.append({"id": f"source-{index:04d}", "condition_path": str(path)})
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text("".join(json.dumps(row) + "\n" for row in rows))
    native = {"data": {"train": {"name": "latent_dataset", "data_path": str(metadata),
        "batch_size": 1, "dataset_repeat": 1, "shuffle": True, "drop_last": True,
        "num_workers": 0, "pin_memory": False}}, "training": {"max_train_iters": 3}}
    config = tmp_path / "native.json"
    config.write_text(json.dumps(native))
    return config, native, rows


def real_loader(native, monkeypatch):
    # Real source dataset + real B1/world16 DistributedSampler and DataLoader.
    # No fake process group is advertised as a native world16 launch.
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from lightx2v_train.data.video_dataset import LatentDataset
    dataset = LatentDataset(native["data"]["train"]["data_path"])
    rank = dist.get_rank() if dist.is_initialized() else 0
    sampler = DistributedSampler(dataset, num_replicas=16, rank=rank, shuffle=True, drop_last=True)
    return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, pin_memory=False)


@pytest.mark.parametrize("path_kind", ["absolute", "parent_relative", "symlink"])
def test_original_dataloader_resume_seek_preserves_restored_torch_rng(
        tmp_path, original_source, cached_data, monkeypatch, path_kind):
    import torch
    from distillschedule.training.h3_grid_inputs import NativeLatentCursor, prepare_inputs, load_inputs
    config_path, native, rows = cached_data
    metadata = Path(native["data"]["train"]["data_path"])
    if path_kind == "parent_relative":
        metadata = tmp_path / "metadata" / "samples.jsonl"
        metadata.parent.mkdir()
        for row in rows:
            row["condition_path"] = "../" + Path(row["condition_path"]).name
    elif path_kind == "symlink":
        links = tmp_path / "condition-links"
        links.mkdir()
        for row in rows:
            target = Path(row["condition_path"])
            alias = links / target.name
            alias.symlink_to(target)
            row["condition_path"] = str(alias)
    metadata.write_text("".join(json.dumps(row) + "\n" for row in rows))
    native["data"]["train"]["data_path"] = str(metadata)
    config_path.write_text(json.dumps(native))
    metadata_bytes = metadata.read_bytes()
    inventory = prepare_inputs(config_path, tmp_path / "inputs.json")
    assert load_inputs(tmp_path / "inputs.json", config_path=config_path, cycles=3) == inventory
    loader = real_loader(native, monkeypatch)
    order = list(loader.sampler)
    torch.manual_seed(401)
    original_iterator = iter(real_loader(native, monkeypatch))
    original_rng = torch.get_rng_state().clone()
    original_sample = next(original_iterator)
    torch.manual_seed(401)
    fresh = NativeLatentCursor(loader, inventory, cycle=0, resume=False)
    assert torch.equal(torch.get_rng_state(), original_rng)
    first = next(fresh)
    assert first["meta"] == original_sample["meta"]
    prefix = [first["meta"]["id"][0]] + [next(fresh)["meta"]["id"][0] for _ in range(5)]
    assert prefix == [rows[i]["id"] for i in order[:6]]
    saved_rng = torch.get_rng_state().clone()
    following = next(fresh)
    torch.set_rng_state(saved_rng)
    resumed = NativeLatentCursor(real_loader(native, monkeypatch), inventory, cycle=1, resume=True)
    assert torch.equal(torch.get_rng_state(), saved_rng)
    assert next(resumed)["meta"] == following["meta"]
    assert resumed.cursor == 7 and resumed.epoch == 0
    # Canonical comparison must not rewrite the metadata or the native sample.
    native_path = Path(rows[order[6]]["condition_path"])
    if not native_path.is_absolute():
        native_path = metadata.parent / native_path
    assert following["meta"]["condition_path"] == [str(native_path)]
    assert metadata.read_bytes() == metadata_bytes
    assert load_inputs(tmp_path / "inputs.json", config_path=config_path, cycles=3) == inventory
    assert not torch.cuda.is_initialized()


def test_native_cursor_resume_seek_does_not_nest_collective_admission(
        tmp_path, original_source, cached_data, monkeypatch):
    from distillschedule.training import distributed
    from distillschedule.training.h3_grid_inputs import NativeLatentCursor, prepare_inputs
    config_path, native, _ = cached_data
    inventory = prepare_inputs(config_path, tmp_path / "inputs.json")
    checked = distributed.checked
    labels = []
    active = False
    def observed(fn, label):
        nonlocal active
        assert not active, "constructor resume seeking nested a collective check"
        labels.append(label)
        active = True
        try:
            return checked(fn, label)
        finally:
            active = False
    monkeypatch.setattr(distributed, "checked", observed)
    cursor = distributed.checked(lambda: NativeLatentCursor(
        real_loader(native, None), inventory, cycle=1, resume=True), "constructor admission")
    assert labels == ["constructor admission"] and cursor.cursor == 6
    next(cursor)
    assert len(labels) == 2 and cursor.cursor == 7


def test_native_next_sample_rechecks_pinned_payload_before_consumption(
        tmp_path, original_source, cached_data):
    import torch
    from distillschedule.training.h3_grid_inputs import NativeLatentCursor, prepare_inputs
    config_path, native, _ = cached_data
    inventory = prepare_inputs(config_path, tmp_path / "inputs.json")
    cursor = NativeLatentCursor(real_loader(native, None), inventory, cycle=0, resume=False)
    expected = inventory["rows"][cursor.order[0]]
    torch.save({"positive": {"prompt_embeds": torch.full((1, 2, 3), 9999.),
                              "text_token_tags": torch.ones(2, dtype=torch.long)}}, expected["condition_path"])
    with pytest.raises(ValueError, match="cached condition changed"):
        next(cursor)
    assert cursor.cursor == 0


@pytest.mark.parametrize("mutation", ["short", "duplicate", "missing", "changed", "repeat", "workers"])
def test_fresh_native_inventory_rejects_replay_or_changed_data(tmp_path, cached_data, mutation):
    from distillschedule.training.h3_grid_inputs import prepare_inputs, load_inputs
    config_path, native, rows = cached_data
    inventory_path = tmp_path / "inputs.json"
    prepare_inputs(config_path, inventory_path)
    metadata = Path(native["data"]["train"]["data_path"])
    if mutation == "short":
        metadata.write_text("".join(json.dumps(row) + "\n" for row in rows[:96]))
    elif mutation == "duplicate":
        rows[-1]["condition_path"] = rows[0]["condition_path"]
        metadata.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif mutation == "missing":
        Path(rows[7]["condition_path"]).unlink()
    elif mutation == "changed":
        Path(rows[7]["condition_path"]).write_bytes(b"changed cached condition")
    else:
        native["data"]["train"]["dataset_repeat" if mutation == "repeat" else "num_workers"] = 2
        config_path.write_text(json.dumps(native))
    with pytest.raises(ValueError):
        load_inputs(inventory_path, config_path=config_path, cycles=3)


def tiny_original_trainer(tmp_path, native, inventory, public, *, resume=None):
    """Substitute only pretrained DiT/AdaLN/FSDP construction at the model edge.

    Everything sampled, rolled out, renoised, scored, differentiated, clipped,
    optimized and scheduled is executed by the real original source methods.
    Tiny geometry/parameters are an explicit CPU fixture, NOT H3 parity.
    """
    from contextlib import nullcontext
    import torch
    from distillschedule.backends.training.minimax_h3 import grid_adaln
    from distillschedule.backends.training.minimax_h3.trainer import MiniMaxH3A100DmdTrainer
    from distillschedule.backends.training.minimax_h3.model import MiniMaxH3A100Model
    from distillschedule.training.h3_grid_training import GridTrainingLifecycle
    assert grid_adaln.DEFAULT_GRID_SIZE == 1000

    class TinyDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_adapters = torch.nn.ParameterDict({
                "student": torch.nn.Parameter(torch.tensor([.04, .02])),
                "fake": torch.nn.Parameter(torch.tensor([.01, .03]))})
            self.register_buffer("base", torch.tensor(.005))
            self.active_adapter = "student"

        def set_adapter(self, role):
            self.active_adapter = role

        def enable_adapters(self):
            pass

        def disable_adapters(self):
            self.active_adapter = None

        def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states, **kwargs):
            weight = self.base + (0 if self.active_adapter is None else self.lora_adapters[self.active_adapter].mean())
            condition = encoder_hidden_states.float().mean() / 1000
            return hidden_states * weight + condition, audio_hidden_states * weight + condition

    class TinyModel(MiniMaxH3A100Model):
        def load_components(self, **kwargs):
            self.transformer = TinyDenoiser()
            self.patch_size, self.vae_spatial_scale_factor = (1, 2, 2), 16
            self.video_latent_channels = self.audio_latent_channels = 2
            self.use_autocast = False
            self._mark_all_adapters_trainable()
            self._activate_role("student")

        def precompute_adaln(self, *args, **kwargs):
            pass  # no giant pretrained AdaLN exists in this tiny DiT fixture

        def adaln_scope(self, *args, **kwargs):
            return nullcontext()

        def drop_adaln_key(self, *args, **kwargs):
            pass

        def log_adaln_stats(self):
            pass

    class TinyTrainer(MiniMaxH3A100DmdTrainer):
        def setup(self, resume_ckpt_path=None):
            assert resume_ckpt_path is None  # never feed v2 to the legacy loader
            self.trainable_params = self.shared_model.role_parameters("student")
            self.fake_trainable_params = self.shared_model.role_parameters("fake")
            # The actual original builders, not a substitute optimizer/update.
            self.optimizer = self._build_optimizer(self.trainable_params)
            self.fake_optimizer = self._build_optimizer(self.fake_trainable_params,
                self.config["training"]["fake"]["optimizer"])
            self.lr_scheduler = self._build_lr_scheduler(self.optimizer)
            self.fake_lr_scheduler = self._build_lr_scheduler(self.fake_optimizer,
                num_warmup_steps=0, num_training_steps=self.max_train_iters * 5)
            self.fake_model = self.teacher_model = self.model

        def save_checkpoint_a100(self, iteration):
            raise AssertionError("historical cursorless checkpoint path must not run")

    # Original constructor validates the protocol and installs actual Grid1000
    # sampling. Only the unavailable CUDA/FSDP/DiT edge is replaced above.
    trainer = TinyTrainer(native)
    import torch.distributed as dist
    world = dist.get_world_size() if dist.is_initialized() else 1
    trainer.matched_expected_world_size = world  # local CPU call-census oracle only
    trainer.matched_cycle_census.expected_world_size = world
    model = TinyModel(native)
    model.load_components()
    trainer.set_model(model)
    trainer.set_data(real_loader(native, None), None)
    raw = public.to_dict()
    if resume:
        raw["trainer"]["resume_from"] = str(resume)
    public = TrainingConfig.from_dict(raw, source=public.source)
    identity = {"schema_version": "CPU_REAL_ORIGINAL_LOOP_ONLY", "native": "TINY_MODEL_EDGE", "horizon": 3}
    lifecycle = GridTrainingLifecycle(public, {"inputs": inventory}, identity)
    trainer.grid_training_lifecycle = lifecycle
    return trainer, lifecycle


def tiny_config(tmp_path, cached_data):
    from copy import deepcopy
    from distillschedule.workloads.best_known.registry import load_profile
    from distillschedule.training.h3_grid_inputs import prepare_inputs
    path, small, rows = cached_data
    native = deepcopy(load_profile("minimax_h3_grid1000_best")["runtime"]["model_config"])
    native["data"] = small["data"]
    native["training"].update(max_train_iters=3, save_every_iters=1, output_dir=str(tmp_path / "run"))
    native["training"]["dmd"].update(height=32, width=32, num_frames=5)
    native["training"]["a100"]["critic_rollout_reorder"]["enabled"] = False
    native["training"]["a100"]["activation_policy"]["name"] = "none"
    for key in ("block_pointwise_fusion", "rotary_fusion", "swiglu_fusion", "lora_scale1_elision"):
        native["training"]["a100"][key] = {k: False for k in native["training"]["a100"][key]}
    native["training"]["a100"]["fa3_nograd_splits"] = {"num_splits": 1, "grad_num_splits": 1}
    native["training"]["a100"]["fa3_replay_cache"] = {"enabled": False, "blocks": [], "storage": "cuda",
        "max_d2h_inflight": 2, "trim_before_backward": False}
    native["training"]["a100"]["adaln_grid"]["manifest"] = str(tmp_path / "UNUSED_TINY_MODEL_EDGE.json")
    # Text-only conditions required by the actual H3 original packing helper.
    import torch
    for index, row in enumerate(rows):
        torch.save({"positive": {"prompt_embeds": torch.full((1, 2, 3), float(index)),
                                  "text_token_tags": torch.ones(2, dtype=torch.long)}}, row["condition_path"])
    path.write_text(json.dumps(native))
    inventory = prepare_inputs(path, tmp_path / "tiny-inputs.json")
    return native, inventory, training(tmp_path)


def test_real_original_grid_loop_three_cycles_exact_interrupted_continuation(
        tmp_path, original_source, cached_data, monkeypatch):
    import random
    import numpy as np
    import torch
    from distillschedule.training.checkpoint import rng_state, pack_state, restore_rng
    from test_training_end_to_end import equal
    native, inventory, public = tiny_config(tmp_path, cached_data)
    random.seed(19); np.random.seed(19); torch.manual_seed(19)
    initial_rng = rng_state()
    control, life = tiny_original_trainer(tmp_path, native, inventory, public)
    control.train()
    expected = pack_state(life.state)
    expected_rng = rng_state()
    expected_order = life.cursor.order[:18]
    # Run with the SAME immutable target/horizon, interrupt after a published
    # complete-cycle checkpoint, then reconstruct real owners and continue.
    import shutil
    shutil.rmtree(tmp_path / "run")
    restore_rng(initial_rng)
    stopped = training(tmp_path, stop_after_cycle=1)
    interrupted, partial = tiny_original_trainer(tmp_path, native, inventory, stopped)
    interrupted.train()
    assert partial.completed == 1 and partial.cursor.cursor == 6
    assert interrupted.max_train_iters == control.max_train_iters == 3
    checkpoint = tmp_path / "run/checkpoints/cycle-000001"
    stale, _ = tiny_original_trainer(tmp_path, native, inventory, stopped, resume=checkpoint)
    monkeypatch.setattr(stale, "setup", lambda **kw: pytest.fail("stale stop reached setup"))
    with pytest.raises(ValueError, match="stop_after_cycle.*resume"):
        stale.train()
    # The new process must keep the original immutable target and scheduler,
    # restore complete native state, and neither repeat nor drop a sample.
    import subprocess
    import sys
    (tmp_path / "resume-public.json").write_text(json.dumps(public.to_dict()))
    script = r'''
import json, pathlib, sys, torch
root, source, tests = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(source / 'lightx2v_train'))
sys.path.insert(0, str(tests))
from test_training_h3_grid import tiny_original_trainer
from distillschedule.training.config import load_training_config
from distillschedule.training.checkpoint import pack_state, rng_state
native = json.loads((root / 'native.json').read_text())
inputs = json.loads((root / 'tiny-inputs.json').read_text())
public = load_training_config(root / 'resume-public.json')
owner, life = tiny_original_trainer(root, native, inputs, public,
    resume=root / 'run/checkpoints/cycle-000001')
parameter_ids = [id(p) for p in owner.model.denoiser_module().parameters()]
owner.train()
assert parameter_ids == [id(p) for p in owner.model.denoiser_module().parameters()]
assert life.completed == 3 and life.cursor.cursor == 18 and owner.max_train_iters == 3
assert not torch.cuda.is_initialized()
torch.save(dict(state=pack_state(life.state), rng=rng_state(), order=life.cursor.order[:18]), root / 'resumed.pt')
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(original_source),
        str(Path(__file__).parent.resolve())], text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    continued = torch.load(tmp_path / "resumed.pt", weights_only=True)
    assert continued["order"] == expected_order
    assert equal(expected, continued["state"])
    assert equal(expected_rng, continued["rng"])
    assert torch.cuda.is_initialized() is False


@pytest.mark.parametrize("kind", ["optimizer_policy", "optimizer_clock", "moment_dtype", "clock", "scheduler", "state", "cursor", "missing_shard", "legacy"])
def test_original_resume_rejects_malformed_state_before_trainable_mutation(
        tmp_path, original_source, cached_data, monkeypatch, kind):
    from copy import deepcopy
    import torch
    from distillschedule.workloads.best_known.registry import file_sha256
    from test_training_end_to_end import equal
    native, inventory, public = tiny_config(tmp_path, cached_data)
    trainer, partial = tiny_original_trainer(tmp_path, native, inventory, public)
    after = partial.after_cycle
    def stop(trainer, cycle):
        after(trainer, cycle)
        raise InterruptedError("published complete boundary")
    monkeypatch.setattr(partial, "after_cycle", stop)
    with pytest.raises(InterruptedError):
        trainer.train()
    checkpoint = tmp_path / "run/checkpoints/cycle-000001"
    shard = checkpoint / "state.pt"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    payload = torch.load(shard, weights_only=True)
    if kind == "optimizer_policy":
        payload["state"]["student.optimizer"]["value"]["optimizer"]["param_groups"][0]["lr"] = .03
    elif kind == "optimizer_clock":
        payload["state"]["fake.optimizer"]["value"]["optimizer"]["state"][0]["step"] += 1
    elif kind == "moment_dtype":
        slots = payload["state"]["student.optimizer"]["value"]["optimizer"]["state"][0]
        slots["exp_avg"] = slots["exp_avg"].double()
    elif kind == "clock":
        payload["state"]["clocks"]["value"]["score_serial"] += 1
    elif kind == "scheduler":
        payload["state"]["fake.scheduler"]["value"]["last_epoch"] -= 1
    elif kind == "state":
        payload["state"].pop("fake.weights")
    elif kind == "cursor":
        manifest["cursor"] = payload["cursor"] = 7
    elif kind == "legacy":
        manifest["schema_version"] = "h3-a100-shared-backbone-v1"
    torch.save(payload, shard)
    manifest["files"]["state.pt"] = file_sha256(shard)
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    (checkpoint / "_SUCCESS").write_text(file_sha256(checkpoint / "manifest.json") + "\n")
    if kind == "missing_shard":
        shard.unlink()
    resumed, _ = tiny_original_trainer(tmp_path, native, inventory, public, resume=checkpoint)
    before = deepcopy(resumed.model.denoiser_module().state_dict())
    with pytest.raises(ValueError):
        resumed.train()
    assert equal(before, resumed.model.denoiser_module().state_dict())


def test_worker_rejects_renamed_fixed_trace_payloads_as_fresh_rows(tmp_path, original_source, cached_data):
    import torch
    from distillschedule.training.h3_grid_inputs import prepare_inputs, validate_payloads
    native, inventory, _ = tiny_config(tmp_path, cached_data)
    rows = inventory["rows"]
    # A renamed .pt has different serialization bytes yet identical original
    # cached condition tensors. Metadata identity alone is not freshness proof.
    torch.save(torch.load(rows[0]["condition_path"], weights_only=True), rows[-1]["condition_path"])
    changed = prepare_inputs(cached_data[0], tmp_path / "renamed-inputs.json")
    with pytest.raises(ValueError, match="duplicate cached condition"):
        validate_payloads(changed)


def test_original_cached_conditions_are_safely_admitted_before_upstream_unpickling(
        tmp_path, original_source, cached_data, monkeypatch):
    import torch
    from distillschedule.training.checkpoint import rng_state
    from distillschedule.training.h3_grid_inputs import validate_payloads
    from test_training_end_to_end import equal
    _, inventory, _ = tiny_config(tmp_path, cached_data)
    metadata = Path(inventory["metadata"]["path"]).read_bytes()
    before = rng_state()
    real_load = torch.load
    calls = []
    def observed(path, *args, **kwargs):
        calls.append((str(Path(path).resolve()), kwargs.get("map_location"), kwargs.get("weights_only")))
        return real_load(path, *args, **kwargs)
    monkeypatch.setattr(torch, "load", observed)
    assert len(validate_payloads(inventory)) == 64
    paths = [row["condition_path"] for row in inventory["rows"]]
    assert calls == [(path, "cpu", True) for path in paths] + [(path, "cpu", False) for path in paths]
    assert equal(before, rng_state())
    assert Path(inventory["metadata"]["path"]).read_bytes() == metadata
    assert not torch.cuda.is_initialized()


def test_original_cached_condition_rejects_executable_pickle_before_any_native_read(
        tmp_path, original_source, cached_data, monkeypatch):
    import pickle
    import torch
    from distillschedule.training.h3_grid_inputs import prepare_inputs, validate_payloads
    from lightx2v_train.data.video_dataset import LatentDataset
    _, inventory, _ = tiny_config(tmp_path, cached_data)
    marker = tmp_path / "UNSAFE_CONDITION_EXECUTED"
    class ExecutableCondition:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",)
    # The last archive is content-pinned just like an ordinary supplied payload.
    # Keeping valid tensors at its positive key makes an unrestricted native
    # load otherwise succeed and proves the extra object cannot be ignored.
    path = inventory["rows"][-1]["condition_path"]
    value = torch.load(path, weights_only=True)
    value["unsupported"] = ExecutableCondition()
    torch.save(value, path)
    inventory = prepare_inputs(cached_data[0], tmp_path / "executable-inputs.json")
    real_getitem = LatentDataset.__getitem__
    reads = []
    def observed(dataset, index):
        reads.append(index)
        return real_getitem(dataset, index)
    monkeypatch.setattr(LatentDataset, "__getitem__", observed)
    error = None
    try:
        validate_payloads(inventory)
    except Exception as caught:
        error = caught
    assert not marker.exists(), "the supplied condition executed before admission"
    assert reads == [], "upstream unpickled conditions before restricted archive admission"
    assert isinstance(error, pickle.UnpicklingError) and "Unsupported global" in str(error)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("kind", ["dtype", "shape", "tags", "positive"])
def test_original_cached_condition_payload_gate_rejects_malformed_tensors(
        tmp_path, original_source, cached_data, kind):
    import torch
    from distillschedule.training.h3_grid_inputs import prepare_inputs, validate_payloads
    _, inventory, _ = tiny_config(tmp_path, cached_data)
    row = inventory["rows"][0]
    value = torch.load(row["condition_path"], weights_only=True)
    if kind == "dtype": value["positive"]["prompt_embeds"] = value["positive"]["prompt_embeds"].double()
    elif kind == "shape": value["positive"]["prompt_embeds"] = torch.zeros(2, 2, 3)
    elif kind == "tags": value["positive"]["text_token_tags"][0] = 0
    else: value = {"positive": {"wrong": torch.ones(2)}}
    torch.save(value, row["condition_path"])
    inventory = prepare_inputs(cached_data[0], tmp_path / "malformed-inputs.json")
    with pytest.raises(ValueError, match="condition"):
        validate_payloads(inventory)


@pytest.mark.parametrize("policy", ["none", "checkpoint_boundary_cpu"])
def test_resolved_original_reference_config_is_validated_without_overrides(
        tmp_path, cached_data, policy):
    from copy import deepcopy
    from distillschedule.training.h3_grid_training import reference_config
    from distillschedule.search.h3_request import H3Request
    native, _, public = tiny_config(tmp_path, cached_data)
    task = H3Request(480, 832, 56)
    (tmp_path / "task.json").write_text(json.dumps(task.to_dict()))
    native["training"]["dmd"].update(height=480, width=832, num_frames=56)
    native["training"]["a100"]["activation_policy"]["name"] = policy
    before = deepcopy(native)
    request, observed = reference_config(native, public)
    assert request == task and observed["name"] == policy and native == before


@pytest.mark.parametrize("field", ["target", "seed", "save", "output", "dtype", "lora", "steps", "fake_ratio",
    "sigma", "fusion", "split", "replay", "scheduler", "world", "auto_resume", "data_default"])
def test_reference_config_rejects_silent_control_or_math_changes(tmp_path, cached_data, field):
    from distillschedule.training.h3_grid_training import reference_config
    from distillschedule.search.h3_request import H3Request
    native, _, public = tiny_config(tmp_path, cached_data)
    (tmp_path / "task.json").write_text(json.dumps(H3Request(480, 832, 56).to_dict()))
    native["training"]["dmd"].update(height=480, width=832, num_frames=56)
    t = native["training"]
    if field in ("target", "seed", "save", "output"):
        key = {"target": "max_train_iters", "seed": "seed", "save": "save_every_iters", "output": "output_dir"}[field]
        t[key] = "elsewhere" if field == "output" else t[key] + 1
    elif field == "dtype": native["model"]["transformer_param_dtype"] = "fp32"
    elif field == "lora": t["student"]["lora"]["rank"] = 64
    elif field == "steps": t["dmd"]["num_inference_steps"] = 3
    elif field == "fake_ratio": t["dmd"]["fake_update_ratio"] = 4
    elif field == "sigma": t["dmd"]["renoise_sigma_min"] = .03
    elif field == "fusion": t["a100"]["block_pointwise_fusion"]["enabled"] = True
    elif field == "split": t["a100"]["fa3_nograd_splits"]["grad_num_splits"] = 2
    elif field == "replay": t["a100"]["fa3_replay_cache"]["enabled"] = True
    elif field == "scheduler": t["lr_scheduler"] = "cosine"
    elif field == "world": native["distributed"]["fsdp2"]["size"] = 8
    elif field == "auto_resume": native["resume"]["auto_resume"] = True
    else: native["data"]["train"].pop("num_workers")
    with pytest.raises(ValueError):
        reference_config(native, public)


def test_real_upstream_source_gate_and_missing_or_wrong_source_rejection(tmp_path, original_source):
    from distillschedule.training.h3_grid_training import _source, LIGHTX2V
    assert _source({"path": str(original_source), "head": LIGHTX2V}, LIGHTX2V, "LightX2V") == original_source
    with pytest.raises(ValueError, match="source identity"):
        _source({"path": str(original_source), "head": "f" * 40}, LIGHTX2V, "LightX2V")
    with pytest.raises(ValueError, match="missing absolute"):
        _source({"path": str(tmp_path / "missing"), "head": LIGHTX2V}, LIGHTX2V, "LightX2V")


def test_cpu_descriptor_and_admission_modules_stay_lazy_in_fresh_interpreter(tmp_path):
    import subprocess
    import sys
    program = """
import sys
from distillschedule.training.h3_grid_training import resume_metadata, reference_config
from distillschedule.training.h3_grid_inputs import load_inputs
assert not any(name in sys.modules for name in ('torch', 'diffusers', 'lightx2v_train', 'peft', 'triton'))
"""
    subprocess.run([sys.executable, "-c", program], check=True, timeout=30)


def test_native_entry_does_not_borrow_or_destroy_an_embedding_group(tmp_path, monkeypatch):
    """A real CPU caller group must reject before the CUDA/native owner edge."""
    import torch
    import torch.distributed as dist
    from distillschedule.training.h3_grid_training import run_reference
    for key, value in {"WORLD_SIZE": "16", "RANK": "0", "LOCAL_RANK": "0",
                       "LOCAL_WORLD_SIZE": "8", "GROUP_RANK": "0"}.items():
        monkeypatch.setenv(key, value)
    dist.init_process_group("gloo", init_method=(tmp_path / "embedding-rendezvous").as_uri(),
                            rank=0, world_size=1)
    try:
        with pytest.raises(ValueError, match="owns its process group"):
            run_reference(training(tmp_path), admit=lambda: pytest.fail("admission must not run"))
        tensor = torch.tensor(3)
        dist.all_reduce(tensor)
        assert tensor.item() == 3 and not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_actual_original_entrypoint_setup_rejects_non_native_sampler_before_updates(
        tmp_path, original_source, cached_data):
    """Real entry/config/data/factory/setup chain; only DiT/FSDP edge replaced.

    The actual upstream builder correctly returns a world1 RandomSampler in
    this CPU process. The lifecycle MUST reject that rather than pretending a
    native world16 run occurred. Isolate original registration patches.
    """
    import subprocess
    import sys
    from distillschedule.training.h3_grid_inputs import prepare_inputs
    from distillschedule.search.h3_request import H3Request
    native, _, public = tiny_config(tmp_path, cached_data)
    native["training"]["dmd"].update(height=480, width=832, num_frames=56)
    cached_data[0].write_text(json.dumps(native))
    (tmp_path / "task.json").write_text(json.dumps(H3Request(480, 832, 56).to_dict()))
    prepare_inputs(cached_data[0], tmp_path / "entry-inputs.json")
    (tmp_path / "entry-public.json").write_text(json.dumps(public.to_dict()))
    script = r'''
import json, pathlib, sys, torch
from pytest import MonkeyPatch
root, source, tests = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(source / 'lightx2v_train'))
sys.path.insert(0, str(tests))
from test_training_h3_grid import tiny_original_trainer
from distillschedule.training.config import load_training_config
from distillschedule.training.h3_grid_training import GridTrainingLifecycle, training_scope
from distillschedule.workloads.best_known.registry import file_sha256
native = json.loads((root / 'native.json').read_text())
inputs = json.loads((root / 'entry-inputs.json').read_text())
public = load_training_config(root / 'entry-public.json')
raw = public.to_dict(); raw['data']['prepared'] = str(root / 'entry-inputs.json')
from distillschedule.training.config import TrainingConfig
public = TrainingConfig.from_dict(raw, source=public.source)
tiny, _ = tiny_original_trainer(root, native, inputs, public)
from distillschedule.backends.training.minimax_h3 import grid_entrypoint as entry
from distillschedule.backends.training.minimax_h3.model import MiniMaxH3A100Model
from distillschedule.backends.training.minimax_h3.trainer import MiniMaxH3A100DmdTrainer
admitted = {'native_config': native, 'inputs': inputs, 'assets': {'native_config': {
    'path': str(root / 'native.json'), 'sha256': file_sha256(root / 'native.json')}}}
life = GridTrainingLifecycle(public, admitted, {'scope': 'CPU_ENTRY_CHAIN_NOT_NATIVE_ADMISSION'})
with MonkeyPatch.context() as patch:
    patch.setattr(MiniMaxH3A100Model, 'load_components', type(tiny.model).load_components)
    patch.setattr(MiniMaxH3A100DmdTrainer, 'setup', type(tiny).setup)
    sys.argv = ['original-entry-cpu-boundary-test', '--config', str(root / 'native.json')]
    try:
        with training_scope(life):
            entry.main()
    except ValueError as error:
        assert 'actual original deterministic LatentDataset/DataLoader owners required' in str(error)
    else:
        raise AssertionError('world1 original sampler was silently admitted')
assert not (root / 'run/checkpoints').exists()
assert not torch.cuda.is_initialized()
'''
    subprocess.run([sys.executable, "-c", script, str(tmp_path), str(original_source),
                    str(Path(__file__).parent)], check=True, timeout=60)


@pytest.mark.parametrize("mutation", [None, "legacy", "source", "target", "cursor", "clock", "missing_shard", "hash", "topology", "torch"])
def test_lazy_cpu_all_rank_checkpoint_metadata_gate(tmp_path, mutation):
    from distillschedule.training.h3_grid_training import resume_metadata, POLICY
    from distillschedule.training.checkpoint import SCHEMA as CHECKPOINT
    from distillschedule.workloads.best_known.registry import file_sha256
    identity = {"scope": "CPU_METADATA_FIXTURE_NOT_NATIVE_TENSOR_STATE"}
    policy = {"schema_version": POLICY, "identity_sha256": "cpu-fixture"}
    files = {}
    for rank in range(16):
        path = tmp_path / f"rank-{rank:05d}.pt"
        path.write_bytes(f"CPU metadata fixture rank{rank}; not pretrained tensor state".encode())
        files[path.name] = file_sha256(path)
    manifest = {"schema_version": CHECKPOINT, "kind": "RESUMABLE_COMPLETE_CYCLE", "identity": identity,
        "cycle": 1, "cursor": 6, "versions": {"student": 1, "fake": 5, "ema": 0}, "policy": policy,
        "topology": {"world_size": 16, "backend": "nccl", "node_groups": [list(range(8)), list(range(8, 16))]},
        "torch": "EXPLICIT_CPU_METADATA_ONLY", "files": files}
    if mutation == "legacy": manifest["schema_version"] = "h3-a100-shared-backbone-v1"
    elif mutation == "source": manifest["identity"] = {"wrong": "source"}
    elif mutation == "target": manifest["cycle"], manifest["cursor"] = 3, 18
    elif mutation == "cursor": manifest["cursor"] = 7
    elif mutation == "clock": manifest["versions"]["student"] = True
    elif mutation == "missing_shard": (tmp_path / "rank-00007.pt").unlink()
    elif mutation == "hash": (tmp_path / "rank-00007.pt").write_bytes(b"changed")
    elif mutation == "topology": manifest["topology"]["node_groups"] = [list(range(16))]
    elif mutation == "torch": manifest["torch"] = "CHANGED_RUNTIME"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "_SUCCESS").write_text(file_sha256(tmp_path / "manifest.json") + "\n")
    if mutation:
        with pytest.raises(ValueError):
            resume_metadata(tmp_path, identity=identity, policy=policy, target=3, torch_version="EXPLICIT_CPU_METADATA_ONLY")
    else:
        assert resume_metadata(tmp_path, identity=identity, policy=policy, target=3,
                               torch_version="EXPLICIT_CPU_METADATA_ONLY") == manifest


def identity_only_assets(tmp_path):
    model = tmp_path / "converted/transformer"; model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"_class_name": "MiniMaxH3Transformer3DModel", "num_refiner_layers": 2}))
    weight = model / "model.safetensors"; weight.write_bytes(b"CPU identity fixture, not pretrained weights")
    grid = {"schema_version": "h3_a100.adaln_grid.v1", "grid_size": 1000, "num_blocks": 50,
            "rows_per_entry": 6, "modulation_chunks": 6, "num_entries": 1004, "num_rollout_entries": 4,
            "hidden_size": 1, "renoise_sigma_min": .02, "renoise_sigma_max": .98,
            "video_shift": 6., "audio_shift": 3., "binary_file": "grid.bin"}
    grid_path = tmp_path / "grid.json"; grid_path.write_text(json.dumps(grid))
    binary = tmp_path / "grid.bin"
    with binary.open("wb") as handle: handle.truncate(1004 * 50 * 6 * 6 * 2)
    kernel = tmp_path / "kernel"; kernel.mkdir()
    (kernel / "fixture.so").write_bytes(b"CPU identity fixture, not executable CUDA")
    config = {"model": {"pretrained_model_name_or_path": str(model.parent)},
              "training": {"a100": {"adaln_grid": {"manifest": str(grid_path)}}}}
    env = {"H3_FLASH_ATTN3_LOCAL_PATH": str(kernel)}
    return config, env, model, weight, grid, grid_path, binary, kernel


@pytest.mark.parametrize("mutation", [None, "model", "grid", "binary", "kernel", "index"])
def test_frozen_asset_inventory_hashes_real_files_and_rejects_missing_payloads(tmp_path, mutation):
    from distillschedule.training.h3_grid_training import immutable_files
    config, env, model, weight, grid, grid_path, binary, kernel = identity_only_assets(tmp_path)
    expected = immutable_files(config, env)
    if mutation == "model":
        weight.write_bytes(b"different identity fixture")
        assert immutable_files(config, env) != expected
    elif mutation:
        if mutation == "grid": grid["video_shift"] = 12.; grid_path.write_text(json.dumps(grid))
        elif mutation == "binary": binary.write_bytes(b"truncated")
        elif mutation == "kernel": (kernel / "fixture.so").unlink()
        else: (model / "weights.index.json").write_text(json.dumps({"weight_map": {"p": "missing.safetensors"}}))
        with pytest.raises(ValueError): immutable_files(config, env)
    else:
        assert set(expected) == {str(p) for p in (model / "config.json", weight, grid_path, binary, kernel / "fixture.so")}


@pytest.mark.parametrize("mutation", [None, "source", "config", "input", "asset", "legacy_resume", "environment"])
def test_full_lazy_cpu_admission_with_real_pinned_sources_and_identity_only_assets(
        tmp_path, original_source, cached_data, mutation):
    """Real Git/source/config/input/hash gates; no pretend installed0.40/GPU.

    The tiny model/compiled-kernel/table files prove identity handling ONLY.
    Their expected runtime versions are declarations, not installed/runtime
    qualification. The actual0.34 CPU Diffusers environment stays untouched.
    """
    import importlib.metadata
    installed_diffusers = importlib.metadata.version("diffusers")
    diffusers = os.environ.get("DMD_H3_TEST_DIFFUSERS_SOURCE")
    if not diffusers:
        pytest.skip("real pinned Diffusers source requires DMD_H3_TEST_DIFFUSERS_SOURCE; not a runtime import shim")
    from distillschedule.training.h3_grid_training import ASSETS, LIGHTX2V, DIFFUSERS, DEPENDENCIES, immutable_files, environment_snapshot
    from distillschedule.training.h3_grid_inputs import prepare_inputs
    from distillschedule.training.native import run_native
    from distillschedule.workloads.best_known.registry import file_sha256
    from distillschedule.search.h3_request import H3Request
    native, _, public = tiny_config(tmp_path, cached_data)
    fixture, env, _, weight, _, grid_path, _, _ = identity_only_assets(tmp_path / "identity-only")
    native["model"]["pretrained_model_name_or_path"] = fixture["model"]["pretrained_model_name_or_path"]
    native["training"]["a100"]["adaln_grid"]["manifest"] = str(grid_path)
    native["training"]["dmd"].update(height=480, width=832, num_frames=56)
    cached_data[0].write_text(json.dumps(native))
    (tmp_path / "task.json").write_text(json.dumps(H3Request(480, 832, 56).to_dict()))
    prepare_inputs(cached_data[0], tmp_path / "full-inputs.json")
    env = {**environment_snapshot(os.environ), **env, "H3_ADALN_GRID_MANIFEST": str(grid_path)}
    # Missing distributions get an explicit synthetic expected version. CPU
    # admission does not represent this as runtime validation; train rejects it.
    expected_versions = {}
    for name in DEPENDENCIES:
        try: expected_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: expected_versions[name] = "0.0.0"
    expected_versions["diffusers"] = "0.40.0.dev0"
    assets = {"schema_version": ASSETS, "lightx2v": {"path": str(original_source), "head": LIGHTX2V},
        "diffusers": {"path": str(Path(diffusers).resolve()), "head": DIFFUSERS},
        "native_config": {"path": str(cached_data[0]), "sha256": file_sha256(cached_data[0])},
        "dependency_versions": expected_versions, "environment": env, "immutable_files": immutable_files(native, env)}
    raw = public.to_dict(); raw["data"]["prepared"] = str(tmp_path / "full-inputs.json")
    if mutation == "source": assets["diffusers"]["head"] = "f" * 40
    elif mutation == "config": native["training"]["seed"] += 1; cached_data[0].write_text(json.dumps(native))
    elif mutation == "input": Path(cached_data[2][0]["condition_path"]).unlink()
    elif mutation == "asset": weight.write_bytes(b"changed identity-only weights")
    elif mutation == "environment": assets["environment"]["H3_PLANNER_TWO_CYCLE"] = "1"
    elif mutation == "legacy_resume":
        legacy = tmp_path / "legacy"; legacy.mkdir()
        (legacy / "_SUCCESS").write_text("ok\n"); (legacy / "trainer_state.pt").write_text("cursorless legacy")
        raw["trainer"]["resume_from"] = str(legacy)
    (tmp_path / "assets.json").write_text(json.dumps(assets))
    public = TrainingConfig.from_dict(raw, source=public.source)
    if mutation:
        with pytest.raises(ValueError): validate_native(public)
    else:
        result = validate_native(public)
        assert result["status"] == "ADMITTED_CPU_NATIVE_BINDING_GPU_NOT_RUN"
        assert result["selected"]["activation_policy"]["name"] == "none"
        assert result["lifecycle"]["known_optimized_backward_gate"] == "FAIL_UNCHANGED"
        assert result["lifecycle"]["fixed_scheduler_horizon"] == 3
        assert "NUMERICAL_QUALIFICATION_NOT_RUN" in result["validation_scope"]
        assert run_native(public, dry_run=True)["status"] == "DRY_RUN_NATIVE_BINDING_NOT_EXECUTED"
        assert importlib.metadata.version("diffusers") == installed_diffusers
        public_path = tmp_path / "full-public.json"
        public_path.write_text(json.dumps(public.to_dict()))
        import subprocess
        import sys
        code = """
import sys
from distillschedule.training.config import load_training_config
from distillschedule.training.native import validate_native, run_native
config = load_training_config(sys.argv[1])
assert validate_native(config)['gpu_execution'] == 'NOT_RUN'
assert run_native(config, dry_run=True)['status'] == 'DRY_RUN_NATIVE_BINDING_NOT_EXECUTED'
assert not any(name in sys.modules for name in ('torch', 'diffusers', 'lightx2v_train', 'peft', 'triton'))
"""
        subprocess.run([sys.executable, "-c", code, str(public_path)], check=True, timeout=60)
