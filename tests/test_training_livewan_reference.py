"""W4 reference contracts; CPU owners are NOT pretrained/CUDA parity proof."""
import json
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

from distillschedule.training.config import TrainingConfig


def config(root, **trainer):
    return TrainingConfig.from_dict({
        "schema_version": "distillschedule.training.v1", "model": {"workload": "W4"},
        "algorithm": {}, "data": {"input_kind": "original_livewan_teacher_cohorts.v1",
                                     "prepared": str(root / "inputs.json")},
        "execution": {"adapter": "native", "mode": "reference", "task": str(root / "task.json"),
                      "assets": str(root / "assets.json"), "capacity_policy": str(root / "capacity.json")},
        "trainer": {"cycles": 3, "checkpoint_every": 1, "seed": 71,
                    "output_dir": str(root / "run"), **trainer}}, source=root / "training.json")


def test_fresh_reference_is_separately_described_and_not_c51(tmp_path):
    from distillschedule.training.native import describe_native, _options
    conf = config(tmp_path)
    assert _options(conf, "W4") == "reference"
    description = describe_native(conf)
    assert description["capabilities"]["modes"] == ["reference"]
    assert description["capabilities"]["cycles"]["min"] == 2
    assert description["capabilities"]["checkpoint"]["supported"] is True
    assert description["capabilities"]["resume"]["supported"] is True
    assert description["capabilities"]["bootstrap"]["versions"] == {"generator": 0, "critic": 0, "ema": 0}
    assert "replicated" in description["capabilities"]["placement"]["critic"]
    assert description["claimable_speedup"] is False


def test_reference_discovery_is_lazy(tmp_path):
    script = """
import sys
from pathlib import Path
from distillschedule.training.config import TrainingConfig
from distillschedule.training.native import describe_native
raw = __import__('json').loads(sys.argv[1])
result = describe_native(TrainingConfig.from_dict(raw, source=Path(sys.argv[2])))
assert result['capabilities']['checkpoint']['supported']
assert 'torch' not in sys.modules
assert not any(name.startswith(('wan.', 'wanstreamer.')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script, json.dumps(config(tmp_path).to_dict()),
                    str(tmp_path / "config.json")], check=True)


def test_reference_asset_owner_is_not_historical_bootstrap():
    from distillschedule.training.livewan_reference_assets import ReferenceAssets
    from distillschedule.runtime.livewan_crossscale_worker import Bootstrap
    assert ReferenceAssets is not Bootstrap and not issubclass(ReferenceAssets, Bootstrap)


def test_reference_state_factory_does_not_replace_resident_factory():
    from distillschedule.training.livewan_state import reference_model_bundle_state, model_bundle_state
    assert reference_model_bundle_state is not model_bundle_state


def test_public_reference_recipe_preserves_selected_c51_and_explicit_contract():
    from distillschedule.training.native import _options, describe_native, validate_native
    from distillschedule.workloads.best_known.registry import repository_root
    root = repository_root() / "recipes/self_forcing"
    reference = TrainingConfig.from_dict(json.loads((root / "training.reference.json").read_text()),
                                         source=root / "training.reference.json")
    selected = TrainingConfig.from_dict(json.loads((root / "training.json").read_text()),
                                        source=root / "training.json")
    assert _options(reference, "W4") == "reference" and _options(selected, "W4") == "plan"
    assert reference.trainer["cycles"] == 2
    assert reference.trainer["checkpoint_every"] == 1 and reference.trainer["seed"] == 20260830
    assert describe_native(reference)["capabilities"]["checkpoint"]["supported"] is True
    assert describe_native(selected)["capabilities"]["checkpoint"]["supported"] is False
    with pytest.raises(ValueError, match="REPLACE_WITH|placeholder"):
        validate_native(reference)


@pytest.fixture(scope="module")
def sources():
    names = ("DMD_LIVEWAN_TEST_UPSTREAM", "DMD_WAN21_TEST_UPSTREAM")
    if not all(os.environ.get(k) for k in names):
        pytest.skip("real original source owners require explicit LiveWan/Wan21 test paths")
    from distillschedule.training.livewan_reference_assets import source_identity
    return {key: source_identity(os.environ[name], project=key) for key, name in zip(("livewan", "wan21"), names)}


def capacity_document():
    from distillschedule.training.livewan_training import CAPACITY
    return {"schema_version": CAPACITY, "gpu_metric": "peak_allocated_bytes",
            "memory_limits": {str(r): 70 << 30 for r in range(16)}, "host_limits": {"0": 800 << 30, "1": 800 << 30},
            "gpu_reserve_bytes": 2 << 30, "host_reserve_bytes": 64 << 30}


def asset_document(root, sources):
    """Real safetensors bytes, deliberately TINY inventories, not Wan weights."""
    import torch
    from safetensors.torch import save_file
    from distillschedule.training.livewan_reference_assets import SCHEMA, BACKEND, DEPENDENCIES, frozen_inventory, environment_snapshot
    models = {}
    for role in ("student", "critic", "teacher"):
        directory = root / role; directory.mkdir()
        dims = (5120, 13824, 40, 40) if role == "teacher" else (1536, 8960, 12, 30)
        cfg = {**dict(zip(("dim", "ffn_dim", "num_heads", "num_layers"), dims)), "freq_dim": 256,
               "in_dim": 16, "out_dim": 16, "patch_size": [1, 2, 2], "text_len": 512,
               "qk_norm": True, "cross_attn_norm": True, "eps": 1e-6, "window_size": [-1, -1]}
        (directory / "config.json").write_text(json.dumps(cfg))
        save_file({"tiny_fixture_weight": torch.arange(12.).reshape(3, 4)}, directory / "diffusion_pytorch_model.safetensors")
        models[role] = frozen_inventory(directory)
    return {"schema_version": SCHEMA, "sources": deepcopy(sources), "models": models,
            "initial_student": {"kind": "base_weights"}, "critic_lora_seed": 72,
            "initial_versions": {"generator": 0, "critic": 0, "ema": 0}, "optimizer_steps": {"generator": 0, "critic": 0},
            "runtime": {"dependency_versions": {name: "CPU_FIXTURE_NOT_RUNTIME_ADMITTED" for name in DEPENDENCIES},
                        "environment": environment_snapshot(os.environ), "backend": deepcopy(BACKEND)}}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, sources):
    import torch
    from distillschedule.data.livewan_inputs import prepare_inputs
    root = tmp_path_factory.mktemp("w4-reference-native-inputs")
    data = root / "teacher-data"; data.mkdir()
    prompts = root / "prompts.pt"
    torch.save({"pos": torch.tensor([.25, .5], dtype=torch.float16).view(2, 1, 1).expand(2, 512, 4096),
                "neg": torch.tensor(-.25, dtype=torch.float16).expand(1, 512, 4096),
                "prompts": ["CPU A", "CPU B"], "neg_prompt": "CPU negative"}, prompts)
    for i in range(240):
        torch.save({"latents": torch.tensor(i / 256, dtype=torch.float16).expand(16, 21, 46, 80),
                    "prompt_idx": i % 2, "seed": i, "size": "640*368", "frames": 81, "steps": 40, "cfg": 5.},
                   data / f"sample-{i:04d}.pt")
    inputs = root / "inputs.json"
    value = prepare_inputs(data, prompts, inputs, source_root=sources["livewan"]["root"], cycles=3, seed=71,
                           evidence_kind="CPU_NATIVE_SHAPE_FIXTURE_NOT_PRETRAINED")
    return inputs, value


@pytest.fixture
def public(tmp_path, sources, prepared):
    from distillschedule.search.livewan_request import LiveWanRequest
    path, inputs = prepared
    assets = asset_document(tmp_path, sources)
    for name, value in (("task", LiveWanRequest().to_dict()), ("assets", assets), ("capacity", capacity_document())):
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    raw = config(tmp_path).to_dict(); raw["data"]["prepared"] = str(path)
    return TrainingConfig.from_dict(raw, source=tmp_path / "training.json"), assets, inputs


def test_public_lazy_validation_never_reads_model_payloads_or_imports_torch(public):
    conf, _, _ = public
    script = """
import sys, builtins
from pathlib import Path
from distillschedule.training.config import TrainingConfig
from distillschedule.training.native import validate_native
old = __import__('io').open
def guarded(path, *a, **k):
    assert not str(path).endswith('.safetensors'), 'lazy validator opened model weights'
    return old(path, *a, **k)
__import__('io').open = guarded
raw = __import__('json').loads(sys.argv[1])
report = validate_native(TrainingConfig.from_dict(raw, source=Path(sys.argv[2])))
assert report['capabilities']['checkpoint']['supported']
assert 'torch' not in sys.modules
assert 'WEIGHT_BYTES_RUNTIME_CAPACITY_AND_CUDA_MATH_NOT_RUN' in report['validation_scope']
"""
    subprocess.run([sys.executable, "-c", script, json.dumps(conf.to_dict()), str(conf.source)], check=True)


@pytest.mark.parametrize("mutation", ["unknown", "missing", "hash", "head", "shape", "seed", "versions", "steps", "environment"])
def test_reference_assets_fail_closed(public, mutation):
    from distillschedule.training.livewan_reference_assets import asset_config, ReferenceAssets
    _, assets, _ = public
    value = deepcopy(assets)
    if mutation == "unknown": value["callback"] = "evil:function"
    elif mutation == "missing": (Path(value["models"]["critic"]["root"]) / "diffusion_pytorch_model.safetensors").unlink()
    elif mutation == "hash":
        path = Path(value["models"]["student"]["root"]) / "diffusion_pytorch_model.safetensors"
        blob = path.read_bytes(); path.write_bytes(blob[:-1] + bytes([blob[-1] ^ 1]))
    elif mutation == "head": value["sources"]["wan21"]["head"] = "0" * 40
    elif mutation == "shape":
        path = Path(value["models"]["teacher"]["root"]) / "config.json"
        cfg = json.loads(path.read_text()); cfg["num_layers"] = 30; path.write_text(json.dumps(cfg))
    elif mutation == "seed": value["critic_lora_seed"] = True
    elif mutation == "versions": value["initial_versions"]["generator"] = 1
    elif mutation == "steps": value["optimizer_steps"]["critic"] = 10
    else: value["runtime"]["environment"]["LIVEWAN_TRAJECTORY_CYCLES"] = "3"
    with pytest.raises((ValueError, FileNotFoundError)):
        ReferenceAssets.load(value) if mutation == "hash" else asset_config(value)


def test_verified_reference_owner_derives_only_actual_native_inputs(public):
    from distillschedule.training.livewan_reference_assets import ReferenceAssets
    _, value, _ = public
    owner = ReferenceAssets.load(value)
    assert owner.versions == {"generator": 0, "critic": 0, "ema": 0}
    assert "systems_only" not in repr(owner.value) and "trace" not in owner.value
    assert owner.value["student_base"] == value["models"]["student"]
    assert owner.native_plan()["metadata"] == {}
    assert owner.native_plan()["workload"]["teacher"]["checkpoint"] == value["models"]["teacher"]["root"]


class UnsafeStage1:
    pass


@pytest.mark.parametrize("mutation", [None, "unsafe", "missing", "shape", "drift", "allowlist"])
def test_stage1_is_restricted_before_original_reader_and_detects_drift(public, mutation):
    import torch
    from distillschedule.training.livewan_reference_assets import ReferenceAssets
    from distillschedule.runtime.livewan_crossscale_worker import _load_student_state
    from distillschedule.workloads.best_known.registry import file_sha256
    conf, assets, _ = public
    path = conf.source.parent / "stage1.pt"
    state = {"tiny_fixture_weight": torch.arange(12.).reshape(3, 4)}
    if mutation in {"unsafe", "allowlist"}: state = {"model": state, "unsafe": UnsafeStage1()}
    elif mutation == "missing": state = {}
    elif mutation == "shape": state["tiny_fixture_weight"] = torch.ones(4)
    torch.save(state, path)
    assets["initial_student"] = {"kind": "stage1_state", "path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
    if mutation in {"unsafe", "missing", "shape", "allowlist"}:
        prior = torch.serialization.get_safe_globals()
        if mutation == "allowlist": torch.serialization.add_safe_globals([UnsafeStage1])
        try:
            with pytest.raises(ValueError, match="Stage1|globals"):
                ReferenceAssets.load(assets)
        finally:
            torch.serialization.clear_safe_globals()
            torch.serialization.add_safe_globals(prior)
        return
    owner = ReferenceAssets.load(assets)
    model = torch.nn.Module(); model.register_parameter("tiny_fixture_weight", torch.nn.Parameter(torch.zeros(3, 4)))
    if mutation == "drift":
        torch.save({"tiny_fixture_weight": torch.ones(3, 4)}, path)
        with pytest.raises(RuntimeError, match="SHA"):
            _load_student_state(torch, model, owner)
        assert not model.tiny_fixture_weight.any()
    else:
        receipt = _load_student_state(torch, model, owner)
        assert receipt["kind"] == "stage1_state" and torch.equal(model.tiny_fixture_weight, state["tiny_fixture_weight"])


@pytest.mark.parametrize("change", ["plan", "manual", "replay", "seed", "horizon", "capacity", "unknown"])
def test_public_rejects_reference_knobs_before_construction(public, change):
    from distillschedule.training.native import validate_native
    conf, _, _ = public
    raw = conf.to_dict()
    if change == "plan": raw["execution"].update(mode="plan", plan="winner.json")
    elif change == "manual": raw["execution"].update(mode="manual", implementation="overrides.json")
    elif change == "replay": raw["data"]["replay_fixed_trace"] = True
    elif change == "seed": raw["trainer"]["seed"] += 1
    elif change == "horizon": raw["trainer"]["cycles"] = 2
    elif change == "capacity": del raw["execution"]["capacity_policy"]
    else: raw["trainer"]["crop_frames"] = 69
    with pytest.raises(ValueError):
        validate_native(TrainingConfig.from_dict(raw, source=conf.source))


def test_actual_loader_cursor_consumes_once_and_only_after_success(prepared, monkeypatch):
    import torch
    from distillschedule.runtime.livewan_crossscale_worker import InputLoader
    from distillschedule.training.livewan_inputs import NativeTeacherCursor
    from distillschedule.training.livewan_training import CursorLoader
    path, _ = prepared
    loader = InputLoader(torch=torch, input_kind="original_livewan_teacher_cohorts.v1", prepared=path)
    cursor = NativeTeacherCursor(loader, rank=7)
    consuming = CursorLoader(loader, cursor)
    original = loader.inputs.dataset.__class__.__getitem__
    calls = []
    def witnessed(dataset, index):
        calls.append(index)
        return original(dataset, index)
    monkeypatch.setattr(loader.inputs.dataset.__class__, "__getitem__", witnessed)
    for step in range(5):
        row = consuming.row(step, 7)
        assert row is cursor.peek() and row.logical_step == step
        sample = consuming.load(row)
        assert sample[2] == row.prompt_idx and cursor.cursor == step + 1
        with pytest.raises(ValueError, match="repeated/skipped"):
            consuming.load(row)
    assert len(calls) == len(consuming.consumed) == 5 and cursor.state_dict()["cycle"] == 1
    row = cursor.peek()
    def fail(*args): raise OSError("actual native load failure")
    monkeypatch.setattr(loader, "load", fail)
    with pytest.raises(OSError, match="actual native load"):
        consuming.load(row)
    assert cursor.cursor == 5
    with pytest.raises(ValueError, match="finite"):
        consuming.row(15, 7)
    assert not torch.cuda.is_initialized()


def _tiny_reference_bundle(sources, *, mesh):
    """Actual CPU DTensors and original LoRA/Adam, NOT native Wan/FSDP math."""
    import torch
    from torch.distributed.tensor import distribute_tensor, Shard
    from distillschedule.runtime.livewan_crossscale_worker import ModelBundle, materialize_adam_state, _bind_trajectory_optimizer_parameter_names
    from distillschedule.training.livewan_state import _source, _reference_lora
    original = _reference_lora(_source(sources["livewan"]["root"]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(72)
        generator = torch.nn.Linear(4, 4, bias=False)
        generator.weight = torch.nn.Parameter(distribute_tensor(generator.weight.detach(), mesh, [Shard(0)]))
        critic = torch.nn.Sequential(original(torch.nn.Linear(4, 4).to(torch.bfloat16), rank=32))
        critic.eval()
        teacher = torch.nn.Linear(4, 4, bias=False).to(torch.bfloat16)
        teacher.weight = torch.nn.Parameter(distribute_tensor(teacher.weight.detach(), mesh, [Shard(0)]), requires_grad=False)
        teacher.eval()
    optg = torch.optim.AdamW(generator.parameters(), lr=2e-6, betas=(0., .99), weight_decay=0., eps=1e-8, foreach=True)
    lora = [p for p in critic.parameters() if p.requires_grad]
    optc = torch.optim.AdamW(lora, lr=4e-5, betas=(0., .99), weight_decay=0., eps=1e-8, foreach=True)
    gnames = _bind_trajectory_optimizer_parameter_names(named_parameters=list(generator.named_parameters()), optimizer=optg)
    cnames = _bind_trajectory_optimizer_parameter_names(named_parameters=list(critic.named_parameters()), optimizer=optc)
    ranks = list(range(mesh.size()))
    metadata = {"generator_ranks": ranks, "score_ranks": ranks, "critic_ranks": ranks,
        "student": {"placement": "world16_fsdp2_keep_unsharded", "optimizer_impl": "foreach_adamw",
                    "state": {"kind": "base_weights", "sha256": "a" * 64}, "optimizer": materialize_adam_state(torch, optg)},
        "critic": {"placement": "gpu_resident_replica", "base_sha256": "b" * 64, "lora_rank": 32,
                   "lora_modules": 1, "lora_numel": sum(p.numel() for p in lora), "optimizer": materialize_adam_state(torch, optc)},
        "teacher": {"placement": "fsdp2_group16_reshard", "mesh_ranks": ranks}}
    return ModelBundle(generator, critic, None, teacher, optg, optc, gnames, cnames, lora, [], None,
                       mesh.get_group(), mesh.get_group(), mesh.get_group(), None, None, None, None, (), (),
                       mesh.size(), mesh.size(), mesh.size(), metadata)


def _tiny_state(bundle, sources, actual, restore=None):
    from distillschedule.training.livewan_state import _source, _reference_lora, _ReferenceRole, _ReferenceWeights, _ReferenceOptimizer
    source = _source(sources["livewan"]["root"])
    original = _reference_lora(source)
    result = {}
    for name in ("generator", "critic"):
        owner = _ReferenceRole(bundle, name, source, original, actual, restore)
        result.update({name + ".weights": _ReferenceWeights(owner), name + ".optimizer": _ReferenceOptimizer(owner)})
    return result


def _tiny_update(bundle, actual):
    """Original LoRA/Adam real backward + DTensor Adam; NOT DMD loss."""
    import torch
    import numpy as np
    from distillschedule.training.livewan_training import quiesce
    x = torch.randn(2, 4).to(torch.bfloat16)
    target = torch.randn(2, 4) + random.random() + float(np.random.random())
    for _ in range(5):
        bundle.critic_optimizer.zero_grad(set_to_none=True)
        (bundle.critic(x).float() - target).square().mean().backward()
        bundle.critic_optimizer.step(); actual["critic"] += 1
    for group in bundle.generator_optimizer.param_groups:
        for p in group["params"]:
            p.grad = torch.ones_like(p) * float(target.mean())
    bundle.generator_optimizer.step(); actual["generator"] += 1
    quiesce(bundle)
    return x, target


def _equal(a, b):
    import torch
    if isinstance(a, torch.Tensor): return type(b) is type(a) and a.dtype == b.dtype and torch.equal(a, b)
    if type(a) is dict: return type(b) is dict and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if type(a) in (tuple, list): return type(b) is type(a) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def test_real_cpu_dtensor_and_original_lora_adam_rng_in_place_continuation(tmp_path, sources):
    import torch
    import torch.distributed as dist
    import numpy as np
    from torch.distributed.device_mesh import DeviceMesh
    from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint, pack_state, rng_state
    from distillschedule.training.livewan_state import reference_model_bundle_state, model_bundle_state
    actual = {"generator": 0, "critic": 0, "ema": 0}
    dist.init_process_group("gloo", init_method="file://" + str(tmp_path / "mesh"), rank=0, world_size=1)
    try:
        mesh = DeviceMesh("cpu", [0], mesh_dim_names=("fsdp",))
        bundle = _tiny_reference_bundle(sources, mesh=mesh)
        state = _tiny_state(bundle, sources, actual)
        with pytest.raises(ValueError, match="world16"):
            reference_model_bundle_state(bundle, source_root=sources["livewan"]["root"], versions=actual)
        with pytest.raises(ValueError, match="one actual split-pool"):
            model_bundle_state(bundle, source_root=sources["livewan"]["root"])
        random.seed(19); np.random.seed(20); torch.manual_seed(21)
        _tiny_update(bundle, actual)
        checkpoint = save_checkpoint(tmp_path / "checkpoint", state, identity={"scope": "CPU_REAL_DTENSOR_LORA_NOT_W4"},
                                     cycle=1, cursor=5, versions=actual)
        saved = pack_state(state)
        assert saved["generator.weights"]["value"]["tensors"]["weight"]["_tensor_kind"] == "distillschedule.dtensor_local.v1"
        draws = [_tiny_update(bundle, actual) for _ in range(2)]
        expected, expected_rng = pack_state(state), rng_state()
        resumed_actual = {"generator": 0, "critic": 0, "ema": 0}
        resumed_bundle = _tiny_reference_bundle(sources, mesh=mesh)
        resumed_state = _tiny_state(resumed_bundle, sources, resumed_actual, restore={"generator": 1, "critic": 5, "ema": 0})
        ids = [[id(p) for g in opt.param_groups for p in g["params"]] for opt in
               (resumed_bundle.generator_optimizer, resumed_bundle.critic_optimizer)]
        restored = restore_checkpoint(checkpoint, resumed_state, identity={"scope": "CPU_REAL_DTENSOR_LORA_NOT_W4"})
        resumed_actual.update(restored["versions"])
        assert _equal(saved, pack_state(resumed_state))
        for expected_draw in draws:
            assert _equal(expected_draw, _tiny_update(resumed_bundle, resumed_actual))
        assert _equal(expected, pack_state(resumed_state)) and _equal(expected_rng, rng_state())
        assert ids == [[id(p) for g in opt.param_groups for p in g["params"]] for opt in
                       (resumed_bundle.generator_optimizer, resumed_bundle.critic_optimizer)]
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("change", ["moment_dtype", "moment_shape", "moment_nan", "step", "names", "missing", "weight_layout"])
def test_reference_raw_state_rejected_before_any_mutation(tmp_path, sources, change):
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from distillschedule.training.checkpoint import pack_state, unpack_state, rng_state
    dist.init_process_group("gloo", init_method="file://" + str(tmp_path / "mesh"), rank=0, world_size=1)
    try:
        mesh = DeviceMesh("cpu", [0], mesh_dim_names=("fsdp",))
        actual = {"generator": 0, "critic": 0, "ema": 0}
        bundle = _tiny_reference_bundle(sources, mesh=mesh)
        state = _tiny_state(bundle, sources, actual)
        _tiny_update(bundle, actual)
        before = pack_state(state); rng_before = rng_state()
        bad = deepcopy(before)
        optimizer = bad["critic.optimizer"]["value"]
        slot = next(iter(optimizer["optimizer"]["state"].values()))
        if change == "moment_dtype": slot["exp_avg"] = slot["exp_avg"].half()
        elif change == "moment_shape": slot["exp_avg"] = slot["exp_avg"][:1]
        elif change == "moment_nan": slot["exp_avg"].fill_(float("nan"))
        elif change == "step": slot["step"].fill_(4)
        elif change == "names": optimizer["parameters"][0].reverse()
        elif change == "missing": del bad["critic.optimizer"]
        else: bad["generator.weights"]["value"]["tensors"]["weight"]["layout"]["placements"] = [{"kind": "replicate"}]
        with pytest.raises(ValueError): unpack_state(bad, state)
        assert _equal(before, pack_state(state)) and _equal(rng_before, rng_state())
    finally:
        dist.destroy_process_group()


def test_hash_valid_wrong_adam_step_checkpoint_rejects_before_weights_rng_mutation(tmp_path, sources):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from distillschedule.training.checkpoint import pack_state, save_checkpoint, restore_checkpoint, rng_state
    from distillschedule.workloads.best_known.registry import file_sha256
    dist.init_process_group("gloo", init_method="file://" + str(tmp_path / "mesh"), rank=0, world_size=1)
    try:
        mesh = DeviceMesh("cpu", [0], mesh_dim_names=("fsdp",))
        actual = {"generator": 0, "critic": 0, "ema": 0}
        bundle = _tiny_reference_bundle(sources, mesh=mesh); state = _tiny_state(bundle, sources, actual)
        _tiny_update(bundle, actual)
        root = save_checkpoint(tmp_path / "checkpoint", state, identity={"scope": "CPU_WRONG_STEP_ADMISSION"},
                               cycle=1, cursor=5, versions=actual)
        payload = torch.load(root / "state.pt", weights_only=True)
        slots = payload["state"]["critic.optimizer"]["value"]["optimizer"]["state"]
        next(iter(slots.values()))["step"].fill_(4)  # header remains G1/C5/EMA0
        torch.save(payload, root / "state.pt")
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["files"]["state.pt"] = file_sha256(root / "state.pt")
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "_SUCCESS").write_text(file_sha256(root / "manifest.json") + "\n")
        resumed_actual = {"generator": 0, "critic": 0, "ema": 0}
        resumed_bundle = _tiny_reference_bundle(sources, mesh=mesh)
        resumed = _tiny_state(resumed_bundle, sources, resumed_actual, restore=manifest["versions"])
        before, rng_before = pack_state(resumed), rng_state()
        with pytest.raises(ValueError, match="step/version"):
            restore_checkpoint(root, resumed, identity=manifest["identity"])
        assert _equal(before, pack_state(resumed)) and _equal(rng_before, rng_state())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("change", ["incomplete", "head", "horizon", "rank", "hash", "cursor", "versions", "finished"])
def test_reference_checkpoint_metadata_rejects_partial_or_changed_identity_before_model(public, change):
    from distillschedule.training.livewan_training import training_identity, checkpoint_policy, resume_metadata, TOPOLOGY
    from distillschedule.training.checkpoint import SCHEMA
    from distillschedule.workloads.best_known.registry import file_sha256
    conf, assets, inputs = public
    identity = training_identity("a" * 40, assets, inputs, conf, capacity_document())
    root = conf.source.parent / "resume"; root.mkdir()
    files = {}
    for rank in range(16):
        path = root / f"rank-{rank:05d}.pt"; path.write_bytes(b"CPU_METADATA_ONLY_NOT_A_TENSOR_SHARD")
        files[path.name] = file_sha256(path)
    header = {"schema_version": SCHEMA, "kind": "RESUMABLE_COMPLETE_CYCLE", "identity": deepcopy(identity),
              "cycle": 1, "cursor": 5, "versions": {"generator": 1, "critic": 5, "ema": 0},
              "policy": checkpoint_policy(identity), "topology": deepcopy(TOPOLOGY), "torch": "CPU_METADATA", "files": files}
    if change == "head": header["identity"]["source_head"] = "b" * 40
    elif change == "horizon": header["identity"]["horizon"] = 2
    elif change == "rank": (root / "rank-00015.pt").unlink()
    elif change == "hash": (root / "rank-00015.pt").write_bytes(b"changed")
    elif change == "cursor": header["cursor"] = 10
    elif change == "versions": header["versions"]["critic"] = 4
    elif change == "finished": header.update(cycle=3, cursor=15, versions={"generator": 3, "critic": 15, "ema": 0})
    (root / "manifest.json").write_text(json.dumps(header))
    if change != "incomplete": (root / "_SUCCESS").write_text(file_sha256(root / "manifest.json") + "\n")
    with pytest.raises(ValueError):
        resume_metadata(root, identity=identity, target=3, torch_version="CPU_METADATA", verify_payloads=True)


def test_saved_cursor_rank_and_versions_cross_state_admission_is_premutation(tmp_path, sources, prepared):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from distillschedule.runtime.livewan_crossscale_worker import InputLoader
    from distillschedule.training.livewan_inputs import NativeTeacherCursor
    from distillschedule.training.livewan_training import CursorLoader, _CursorState, _VersionsState
    from distillschedule.training.checkpoint import pack_state, unpack_state
    loader = InputLoader(torch=torch, input_kind="original_livewan_teacher_cohorts.v1", prepared=prepared[0])
    cursor = NativeTeacherCursor(loader, rank=0); consuming = CursorLoader(loader, cursor)
    for step in range(5): consuming.load(consuming.row(step, 0))
    dist.init_process_group("gloo", init_method="file://" + str(tmp_path / "mesh"), rank=0, world_size=1)
    try:
        mesh = DeviceMesh("cpu", [0], mesh_dim_names=("fsdp",))
        actual = {"generator": 0, "critic": 0, "ema": 0}
        bundle = _tiny_reference_bundle(sources, mesh=mesh); _tiny_update(bundle, actual)
        header = {"cursor": 5, "versions": dict(actual)}
        state = _tiny_state(bundle, sources, actual, restore=header["versions"])
        state.update({"data.cursor": _CursorState(cursor, header), "commits": _VersionsState(actual, cursor, header)})
        before = pack_state(state)
        for field, value in (("rank", 1), ("cursor", 10), ("next_operation_key", "forged")):
            bad = deepcopy(before); bad["data.cursor"]["value"][field] = value
            with pytest.raises(ValueError): unpack_state(bad, state)
            assert _equal(before, pack_state(state)) and cursor.cursor == 5
        bad = deepcopy(before); bad["commits"]["value"]["generator"] = 2
        with pytest.raises(ValueError): unpack_state(bad, state)
        assert _equal(before, pack_state(state)) and actual == header["versions"]
    finally:
        dist.destroy_process_group()


def test_dispatch_only_outer_lifecycle_reuses_actual_baseline_and_real_loads(public, sources, monkeypatch):
    """DISPATCH ONLY: real helper loop/input/LoRA/Adam, synthetic unavailable Wan math.

    This deliberately does NOT qualify original CUDA forward/backward, world16,
    FlashAttention, capacity, or geometry installation. The unavailable resource
    and model-math facades are confined to this test, not the public route.
    """
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace
    from torch.distributed.device_mesh import DeviceMesh
    from distillschedule.runtime import livewan_crossscale_worker as worker, livewan_geometry, livewan_cuda_setup
    from distillschedule.search.livewan_request import LiveWanRequest
    from distillschedule.training import livewan_training as training, livewan_state
    conf, assets, inputs = public
    admitted = training.admit_reference(conf, LiveWanRequest().to_dict(), assets,
        {"capacity_policy": conf.execution["capacity_policy"]})
    identity = training.training_identity("CPU_DISPATCH_ONLY", assets, inputs, conf, capacity_document())
    monkeypatch.setenv("WORLD_SIZE", "1"); monkeypatch.setenv("LOCAL_RANK", "0")
    dist.init_process_group("gloo", init_method="file://" + str(conf.source.parent / "mesh"), rank=0, world_size=1)
    try:
        mesh = DeviceMesh("cpu", [0], mesh_dim_names=("fsdp",))
        events = []
        owner = None
        bundle = _tiny_reference_bundle(sources, mesh=mesh)
        def build_models(**kwargs):
            nonlocal owner
            owner = kwargs["bootstrap"]
            assert kwargs["mode"] == "baseline" and kwargs["plan"] == owner.native_plan()
            events.append("build_models")
            return bundle
        monkeypatch.setattr(worker, "build_models", build_models)
        monkeypatch.setattr(worker, "_build_streamers", lambda *a, **k: events.append("streamers") or [object()])
        monkeypatch.setattr(worker, "build_score_context", lambda *a, **k: events.append("score_context") or object())
        original = worker._run_baseline
        def baseline(**kwargs):
            events.append(("actual_baseline", kwargs["step_offset"]))
            assert kwargs["run_kind"] == "trajectory" and len(kwargs["streamers"]) == 1
            return original(**kwargs)
        monkeypatch.setattr(worker, "_run_baseline", baseline)
        def forbidden(*a, **k): raise AssertionError("entered benchmark/pipeline writer instead of reference")
        for name in ("run_worker", "_run_baseline_trajectory", "_append_trajectory_cycle_receipt", "_trajectory_state_anchor"):
            monkeypatch.setattr(worker, name, forbidden)
        # These are explicit synthetic Wan-math surfaces; InputLoader and the
        # baseline's real optimizer/clip/reduction/commit code remain unchanged.
        def rollout(torch, *, row, loader, **kwargs):
            latent, context, pi = loader.load(row)
            assert pi == row.prompt_idx
            return SimpleNamespace(row=row, x0=latent, raw_context=context, context_lens=None)
        def generator_backward(torch, **kwargs):
            for parameter in bundle.generator_optimizer.param_groups[0]["params"]:
                parameter.grad = torch.ones_like(parameter) * .01
            return [torch.tensor(.1), torch.tensor(.2), torch.tensor(.3)]
        def critic_backward(torch, **kwargs):
            x = torch.tensor([[.1, .2, .3, .4]], dtype=torch.bfloat16)
            loss = bundle.critic(x).float().square().mean()
            loss.backward()
            return loss.detach()
        monkeypatch.setattr(worker, "prepare_rollout", rollout)
        monkeypatch.setattr(worker, "prepare_score_sample_context", lambda *a, **k: None)
        monkeypatch.setattr(worker, "dmd_gradient_for_sample", lambda *a, **k: None)
        monkeypatch.setattr(worker, "generator_backward_for_sample", generator_backward)
        monkeypatch.setattr(worker, "critic_backward_for_sample", critic_backward)
        monkeypatch.setattr(training, "_backend", lambda admitted: worker)
        monkeypatch.setattr(training, "_admit_model_layout", lambda *a: {"scope": "CPU_DISPATCH_ONLY_NOT_NATIVE_MODEL_LAYOUT"})
        monkeypatch.setattr(training, "admit_runtime", lambda assets: {"scope": "CPU_DISPATCH_ONLY"})
        monkeypatch.setattr(training, "_capacity", lambda *a, **k: {"oom": 0, "oom_kill": 0})
        monkeypatch.setattr(livewan_geometry, "install_geometry", lambda *a, **k: events.append("geometry_dispatch_only"))
        monkeypatch.setattr(livewan_cuda_setup, "materialize_blas_handles", lambda *a, **k: None)
        monkeypatch.setattr(livewan_state, "reference_model_bundle_state", lambda bundle, *, versions, restore_versions, **k:
                            _tiny_state(bundle, sources, versions, restore_versions))
        output = Path(conf.trainer["output_dir"])
        token = training._reserve_output(output)
        assert training._run(conf, admitted, identity, output, token) == 0
        assert events.count("build_models") == events.count("streamers") == events.count("score_context") == 1
        assert [e for e in events if isinstance(e, tuple)] == [("actual_baseline", 0), ("actual_baseline", 5), ("actual_baseline", 10)]
        receipt = json.loads((output / "ranks/rank-00000.json").read_text())
        assert receipt["status"] == "COMPLETED" and receipt["next_input_cursor"] == 15
        assert receipt["versions"] == {"generator": 3, "critic": 15, "ema": 0} and len(receipt["metrics"]) == 3
        assert len({r["record_id"] for metric in receipt["metrics"] for r in metric["inputs"]}) == 15
        assert len(list((output / "checkpoints").glob("cycle-*/_SUCCESS"))) == 3
        assert not receipt["claimable_speedup"] and not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_actual_original_wan_constructor_shapes_on_meta_only(sources):
    """Actual original model metadata, no weights, forward, FlashAttention or GPU.

    File-spec namespace loading avoids Wan's unrelated eager video/tokenizer
    application imports. Model and relative attention source remain ORIGINAL.
    This is not evidence that the full native package/runtime is installed.
    """
    import ast
    import math
    from types import ModuleType, SimpleNamespace
    import torch
    from distillschedule.runtime import livewan_crossscale_worker as worker
    from distillschedule.training.livewan_training import _meta_layouts
    root = Path(sources["wan21"]["root"]) / "wan"
    prefix = "_distillschedule_w4_meta_original_wan"
    for name, directory in ((prefix, root), (prefix + ".modules", root / "modules")):
        module = ModuleType(name); module.__path__ = [str(directory)]
        sys.modules[name] = module
    try:
        path = root / "modules/model.py"
        spec = importlib.util.spec_from_file_location(prefix + ".modules.model", path)
        model = importlib.util.module_from_spec(spec); sys.modules[spec.name] = model; spec.loader.exec_module(model)
        # Literal config attributes are read from the pinned ORIGINAL file;
        # no alternate constructor or manually reimplemented architecture.
        tree = ast.parse((root / "configs/wan_t2v_1_3B.py").read_text())
        fields = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "t2v_1_3B":
                    fields[target.attr] = ast.literal_eval(node.value)
        before = torch.get_rng_state().clone()
        layouts = _meta_layouts(worker, WanModel=model.WanModel, config=SimpleNamespace(**fields))
        assert layouts["1.3B"]["patch_embedding.weight"] == [1536, 16, 1, 2, 2]
        assert layouts["1.3B"]["blocks.29.ffn.0.weight"] == [8960, 1536]
        assert layouts["14B"]["blocks.39.ffn.0.weight"] == [13824, 5120]
        assert layouts["14B"]["head.head.weight"] == [64, 5120]
        assert sum(math.prod(s) for s in layouts["14B"].values()) > 14_000_000_000
        assert torch.equal(before, torch.get_rng_state()) and not torch.cuda.is_initialized()
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]


@pytest.mark.parametrize("mutation", [None, "missing", "shape", "dtype"])
def test_real_safetensors_header_gate_is_preallocation_and_hash_valid(public, mutation):
    import torch
    from safetensors.torch import save_file
    from distillschedule.training.livewan_reference_assets import admit_native_layouts, frozen_inventory
    _, assets, _ = public
    layouts = {size: {"tiny_fixture_weight": [3, 4]} for size in ("1.3B", "14B")}
    if mutation:
        root = Path(assets["models"]["teacher"]["root"])
        tensor = torch.ones(3, 4)
        if mutation == "shape": tensor = torch.ones(4, 3)
        elif mutation == "dtype": tensor = tensor.long()
        save_file({("wrong_key" if mutation == "missing" else "tiny_fixture_weight"): tensor},
                  root / "diffusion_pytorch_model.safetensors")
        assets["models"]["teacher"] = frozen_inventory(root)  # not a stale hash failure
        with pytest.raises(ValueError, match="keys/shapes|dtype"):
            admit_native_layouts(assets, layouts)
    else:
        assert set(admit_native_layouts(assets, layouts)) == {"1.3B", "14B"}


@pytest.mark.parametrize("mutation", [None, "available", "reserve", "allocator", "oom", "common_failure"])
def test_reference_capacity_wrapper_preserves_exact_budget_reserve_and_oom_policy(mutation):
    """CPU unit gate only: telemetry values are explicit facades, NOT capacity."""
    from types import SimpleNamespace
    from distillschedule.training.livewan_training import _capacity
    policy = capacity_document(); prior = deepcopy(policy)
    row = {"gpus": [{"device": rank, "free_bytes": 75 << 30} for rank in range(8)],
           "cgroup_events": {"oom": 0, "oom_kill": 0}}
    observer = SimpleNamespace(rows=[row], errors=[], sample=lambda: None)
    initial = None
    if mutation == "available": row["gpus"][7]["free_bytes"] = 71 << 30
    elif mutation == "reserve": row["gpus"][7]["free_bytes"] = 1 << 30; initial = {"oom": 0, "oom_kill": 0}
    elif mutation == "oom": initial = {"oom": 0, "oom_kill": 0}; row["cgroup_events"]["oom"] = 1
    elif mutation == "common_failure": observer.errors.append("explicit common host/driver gate failure")
    facade = SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 71 << 30 if mutation == "allocator" else 0))
    if mutation:
        with pytest.raises(ValueError): _capacity(facade, observer, policy, rank=0, initial=initial)
    else:
        assert _capacity(facade, observer, policy, rank=0) == {"oom": 0, "oom_kill": 0}
    assert policy == prior


def test_reference_failure_receipt_preserves_late_unowned_and_prior_owned_progress(tmp_path):
    from distillschedule.training.livewan_training import _receipt
    output = tmp_path / "owned"; (output / "ranks").mkdir(parents=True)
    (output / ".reference-owner").write_text("this-invocation\n")
    path = output / "ranks/rank-00000.json"
    sentinel = b'{"status":"UNOWNED_OLD_EVIDENCE"}'
    path.write_bytes(sentinel)
    with pytest.raises(ValueError, match="preserved"):
        _receipt(output, "this-invocation", {"status": "FAIL"})
    assert path.read_bytes() == sentinel
    path.unlink()  # fixture only; production never deletes an unowned receipt
    _receipt(output, "this-invocation", {"status": "PARTIAL", "completed_cycles": 2})
    _receipt(output, "this-invocation", {"status": "FAIL", "error": "native boundary failed"})
    value = json.loads(path.read_text())
    assert value["last_complete_progress"]["completed_cycles"] == 2
    assert value["error"] == "native boundary failed"
