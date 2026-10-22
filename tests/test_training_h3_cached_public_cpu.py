"""Actual original tiny AV models through public training, not quality evidence."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import random

import numpy as np

import pytest
import torch
from safetensors.torch import save_file

from test_h3_cached_original_cpu import original, tiny_architecture, sample, equal_tree


def json_file(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return path


def pin(path):
    from distillschedule.workloads.best_known.registry import file_sha256
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=file_sha256(path))


def make_case(root, *, gan_start=2, count=4, packed=False):
    from distillschedule.reference import diffsynth_h3_cached as provider
    from distillschedule.training.h3_cached_assets import SCHEMA, FIXTURE, runtime_identity, PRECISION, BACKEND
    from distillschedule.data.h3_cached_inputs import SCHEMA as INPUT_SCHEMA
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    api = original()
    api.training.seed_all(921)
    architecture = tiny_architecture()
    base = api.model.MiniMaxH3DiT(**architecture)
    weights = root / "full-base.safetensors"
    save_file(base.state_dict(), str(weights), metadata={"evidence_kind": FIXTURE})
    architecture_file = json_file(root / "architecture.json", architecture)
    config = asdict(api.dmd2.DMD2Config(critic_updates=1, hidden_channels=8, num_layers=2,
                                       gan_start_step=gan_start))
    geometry = dict(video_shape=[1, 24, 2, 4, 4], audio_shape=[2, 32, 3],
                    prompt_shape=[3, 5120], reference_video_shape=[1, 24, 3, 4, 4],
                    reference_audio_shape=[2, 32, 5])
    assets = dict(schema_version=SCHEMA, evidence_kind=FIXTURE, backend=BACKEND, precision=PRECISION,
                  protocol="CANONICAL_GAN500" if gan_start == 500 else "GAN_START_2_TEST_FIXTURE",
                  source=provider.source_identity(), runtime=runtime_identity(),
                  base=dict(config=pin(architecture_file), weights=[pin(weights)]),
                  dmd_config=config, geometry=geometry,
                  budget=dict(max_parameters=2_000_000, max_loaded_bytes=32_000_000,
                              max_cycles=count, max_input_elements=100_000))
    asset_file = json_file(root / "assets.json", assets)
    records = []
    for index in range(count):
        raw = sample(index)
        if packed:
            parsed, _ = api.dmd2.prepare_batch(raw)
            raw = (dict(input_latents=raw["input_latents"], audio_input_latents=raw["audio_input_latents"],
                        ref_visual_anchor=parsed["ref_visual_anchor"], ref_audio_anchor=parsed["ref_audio_anchor"],
                        imgvid_cond_noise_aug=1.0, audio_cond_noise_aug=1.0),
                   dict(prompt_embeds=raw["prompt_embeds"], packed=parsed["packed"]), {})
        path = root / f"cache-{index}.pth"
        torch.save(raw, path)
        records.append(dict(id=f"sample-{index}", cache=pin(path)))
    source_file = json_file(root / "inputs.json", dict(schema_version=INPUT_SCHEMA,
        evidence_kind=FIXTURE, source=provider.source_identity(), precision=PRECISION,
        geometry=geometry, records=records))
    return asset_file, source_file


def request(root, assets, prepared, cycles, *, resume=None, **overrides):
    from distillschedule.training.config import TrainingConfig
    value = dict(schema_version="distillschedule.training.v1",
        model=dict(factory="distillschedule.training.h3_cached_public:build_model",
                   parameters=dict(asset_manifest=str(assets))),
        data=dict(prepared=str(prepared), sampling=dict(kind="finite", shuffle=False)),
        algorithm=dict(factory="distillschedule.training.h3_cached_public:build_algorithm"),
        execution=dict(adapter="program", mode="reference"),
        trainer=dict(cycles=cycles, seed=777, device="cpu", checkpoint_every=1,
                     output_dir=str(root)))
    if resume:
        value["trainer"]["resume_from"] = str(resume)
    for section, values in overrides.items():
        value[section].update(values)
    return TrainingConfig.from_dict(value, source=root.parent / "training.json")


def read_owner(checkpoint, identity):
    from distillschedule.training.checkpoint import read_checkpoint
    from distillschedule.training.tensor_state import decode
    payload = read_checkpoint(checkpoint, identity=identity)
    return payload, decode(payload["state"]["native_update"]["value"], {})


def test_public_actual_update_and_one_nonreplayable_call(tmp_path):
    from distillschedule.data.h3_cached_inputs import prepare_inputs
    from distillschedule.training.session import construct, build_cycle, train
    assets, sources = make_case(tmp_path)
    prepared = prepare_inputs(sources, tmp_path / "prepared")
    config = request(tmp_path / "one", assets, prepared, 1)
    components, data, algorithm, _ = construct(config)
    owner = components.state["native_update"]
    assert set(components.state) == {"native_update"} and components.generate is None
    before = owner.state_dict()
    assert set(before["phase"]["roles"].values()) == {"teacher"}
    assert not any(before["phase"]["adapter_requires_grad"].values())
    assert all(before["phase"]["head_requires_grad"].values())
    frozen = {n: p.detach().clone() for n, p in owner.adapter.named_parameters() if ".adapters." not in n}
    inputs = data.sample(0)
    inputs_before = deepcopy(inputs)
    workload = build_cycle(config, components, data, algorithm, cycle=0, cursor=0)
    assert len(workload.program.calls) == 1
    call = workload.program.calls[0]
    assert call.grad == "external_grad" and not call.replay_safe
    assert tuple(call.writes) == ("native_update",)
    # The actual public train route owns the separate persistent model instance.
    result = train(config)
    payload, state = read_owner(result["checkpoint"], result["identity"])
    assert result["completed_cycles"] == result["cursor"] == 1
    assert result["final_versions"] == {"native_update": 1}
    assert state["generator_step"] == state["critic_step"] == 1
    assert state["last_metrics"]["gan_active"] is False
    recorded = json.loads((Path(config.trainer['output_dir'])/'metrics.jsonl').read_text())['metrics']
    assert recorded == {name: state['last_metrics'][field] for name, field in (
        ('generator/dmd_loss', 'dmd_loss'), ('generator/gan_loss', 'gan_loss'), ('critic/loss', 'critic_loss'))}
    assert set(state["phase"]["roles"].values()) == {"student"}
    for role in ("student", "critic"):
        slots = state["optimizers"][role]["state"]
        assert slots and any(torch.count_nonzero(row["exp_avg"]) for row in slots.values())
        assert all(row["step"].item() == 1 for row in slots.values())
    assert not any(key.startswith("head.") for key in state["optimizers"]["critic"]["state"])
    equal_tree(inputs_before, inputs)
    equal_tree(frozen, {n: p.detach() for n, p in owner.adapter.named_parameters() if ".adapters." not in n})
    assert "numpy" in payload["rng"] and "numpy" in state["rng"]
    assert not torch.cuda.is_initialized()
    assert original().dmd2.DMD2Config().gan_start_step == 500


def test_public_partial_critic_commit_poison_retains_previous_complete_cycle(tmp_path, monkeypatch):
    from distillschedule.data.h3_cached_inputs import prepare_inputs
    from distillschedule.training import session
    from distillschedule.training.checkpoint import cpu_tree, rng_state, save_checkpoint
    assets, sources = make_case(tmp_path, gan_start=500, count=2)
    prepared = prepare_inputs(sources, tmp_path / "prepared")
    root = tmp_path / "run"
    config = request(root, assets, prepared, 2)
    checkpoint = root / "checkpoints/cycle-000001"
    construct, owners, observed = session.construct, [], {}

    def capture_owner(*args, **kwargs):
        result = construct(*args, **kwargs)
        owners.append(result[0].state["native_update"])
        observed["constructed"] = result[:3]
        observed["caller_rng"] = cpu_tree(rng_state())
        return result

    monkeypatch.setattr(session, "construct", capture_owner)

    def trace_boundary(frame, event, arg):
        if (owners and frame.f_code is type(owners[0]).update.__code__
                and frame.f_locals.get("self") is owners[0] and frame.f_locals.get("cycle") == 1):
            if event == "call":
                observed["outer_rng"] = cpu_tree(rng_state())
                observed["complete"] = owners[0].state_dict()
                observed["row"] = deepcopy(frame.f_locals["row"])
                observed["checkpoint_bytes"] = {str(p.relative_to(checkpoint)): p.read_bytes()
                    for p in checkpoint.rglob("*") if p.is_file()}
            elif event == "return":
                # Observe the owner's finally BEFORE the Program RNG scope also
                # unwinds, so that scope cannot mask a missing native restore.
                observed["restored_outer_rng"] = cpu_tree(rng_state())
                observed["returned_flags"] = (owners[0]._busy, owners[0]._poisoned)
            return trace_boundary
        return None

    step_code = original().training.optimizer_step.__code__

    def fail_at_original_student_step(frame, event, arg):
        if (event == "call" and frame.f_code is step_code and owners
                and frame.f_locals.get("optimizers") is owners[0].optimizers
                and frame.f_locals.get("role") == "student" and owners[0].cycle == 1):
            owner = owners[0]
            observed["partial_clocks"] = (owner.cycle, owner.trainer.critic_step)
            observed["partial_optimizers"] = {role: owner._optimizer_value(role) for role in owner.optimizers}
            observed["partial_adapters"] = owner.api.training.adapter_state(owner.adapter)
            observed["native_rng"] = cpu_tree(rng_state())
            observed["busy_at_failure"] = owner._busy
            raise RuntimeError("explicit original student-step failure after critic commit")

    previous_trace, previous_profile = sys.gettrace(), sys.getprofile()
    try:
        # No source-qualified callable, optimizer or numerical hook is replaced.
        # The profile exception is at the actual original optimizer entry, after
        # its critic backward/clip/Adam commit but before the student backward.
        sys.settrace(trace_boundary)
        sys.setprofile(fail_at_original_student_step)
        with pytest.raises(RuntimeError, match="explicit original student-step failure"):
            session.train(config)
    finally:
        sys.setprofile(previous_profile)
        sys.settrace(previous_trace)

    assert len(owners) == 1
    owner = owners[0]
    complete = observed["complete"]
    assert complete["generator_step"] == complete["critic_step"] == 1
    assert owner.config.num_inference_steps == 8 and owner.config.critic_updates == 1
    assert owner.config.lora_rank == owner.config.lora_alpha == 64
    assert owner.config.gan_start_step == 500 and complete["last_metrics"]["gan_active"] is False
    assert observed["partial_clocks"] == (1, 2) == (owner.cycle, owner.trainer.critic_step)
    assert observed["busy_at_failure"] and observed["returned_flags"] == (False, True)
    assert owner._poisoned and not owner._busy
    equal_tree(observed["outer_rng"], observed["restored_outer_rng"])
    equal_tree(observed["caller_rng"], rng_state())
    equal_tree(observed["native_rng"], owner._rng)
    assert not torch.equal(complete["rng"]["torch"], owner._rng["torch"])
    equal_tree(complete["consumed"], owner._consumed)
    equal_tree(complete["last_metrics"], owner._last_metrics)
    partial = observed["partial_optimizers"]
    equal_tree(complete["optimizers"]["student"], partial["student"])
    for role in owner.optimizers:
        equal_tree(partial[role], owner._optimizer_value(role))
    critic_before, critic_after = complete["optimizers"]["critic"]["state"], partial["critic"]["state"]
    assert critic_before.keys() == critic_after.keys() and critic_after
    assert all(critic_before[name]["step"].item() == 1 and slot["step"].item() == 2
               for name, slot in critic_after.items())
    for moment in ("exp_avg", "exp_avg_sq"):
        assert any(not torch.equal(critic_before[name][moment], slot[moment])
                   for name, slot in critic_after.items())
    for name, value in complete["adapters"].items():
        if ".student." in name:
            equal_tree(value, observed["partial_adapters"][name])
    assert any(not torch.equal(value, observed["partial_adapters"][name])
               for name, value in complete["adapters"].items() if ".fake_score." in name)

    with pytest.raises(ValueError, match="poisoned"):
        session.build_cycle(config, *observed["constructed"], cycle=1, cursor=1)
    with pytest.raises(ValueError, match="poisoned"):
        owner.update(observed["row"], cycle=1)
    with pytest.raises(ValueError, match="poisoned"):
        owner.state_dict()
    with pytest.raises(ValueError, match="poisoned"):
        save_checkpoint(root / "checkpoints/cycle-000002", {"native_update": owner}, identity={},
                        cycle=2, cursor=2, versions={"native_update": 2})
    equal_tree(observed["caller_rng"], rng_state())
    failure = json.loads((root / "failure.json").read_text())
    assert failure["status"] == "FAILED" and "explicit original student-step failure" in failure["error"]
    assert failure["completed_cycles_this_process"] == failure["cursor"] == 1
    assert Path(failure["last_complete_checkpoint"]) == checkpoint
    assert observed["checkpoint_bytes"] and observed["checkpoint_bytes"] == {
        str(p.relative_to(checkpoint)): p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()}
    identity = json.loads((root / "identity.json").read_text())
    payload, saved = read_owner(checkpoint, identity)
    assert payload["cycle"] == payload["cursor"] == 1 and payload["versions"] == {"native_update": 1}
    equal_tree(complete, saved)
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["cycle"] == rows[0]["cursor"] == 1
    assert list((root / "checkpoints").iterdir()) == [checkpoint]
    assert not (root / "summary.json").exists()
    assert not torch.cuda.is_initialized()


def resume_worker(root):
    from distillschedule.training.session import train
    root = Path(root)
    summary = json.loads((root / "partial/summary.json").read_text())
    config = request(root / "resumed", root / "assets.json", root / "prepared/manifest.json", 3,
                     resume=summary["checkpoint"])
    result = train(config)
    json_file(root / "resume-process.json", dict(pid=os.getpid(), checkpoint=result["checkpoint"]))


@pytest.mark.parametrize("packed", [False, True])
def test_common_checkpoint_fresh_os_process_exact_gan_transition(tmp_path, packed):
    from distillschedule.data.h3_cached_inputs import prepare_inputs
    from distillschedule.training.session import train
    assets, sources = make_case(tmp_path, packed=packed)
    prepared = prepare_inputs(sources, tmp_path / "prepared")
    full = train(request(tmp_path / "full", assets, prepared, 3))
    partial = train(request(tmp_path / "partial", assets, prepared, 2))
    completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), str(tmp_path)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                 TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    process = json.loads((tmp_path / "resume-process.json").read_text())
    assert process["pid"] != os.getpid()
    resumed = json.loads((tmp_path / "resumed/summary.json").read_text())
    a, owner_a = read_owner(full["checkpoint"], full["identity"])
    b, owner_b = read_owner(resumed["checkpoint"], resumed["identity"])
    equal_tree(a, b)
    assert a["cycle"] == a["cursor"] == 3 and a["versions"] == {"native_update": 3}
    assert owner_a["last_metrics"]["gan_active"] is True
    for role, opt in owner_b["optimizers"].items():
        for name, slot in opt["state"].items():
            assert slot["step"].item() == (1 if name.startswith("head.") else 3)
            assert torch.isfinite(slot["exp_avg"]).all()
    assert partial["resume_cycle"] == 0 and resumed["resume_cycle"] == 2
    assert resumed["cycles_this_process"] == 1


def test_actual_original_direct_vs_public_exact_update_and_initialization(tmp_path):
    from distillschedule.data.h3_cached_inputs import prepare_inputs, decode_row
    from distillschedule.training.session import construct, build_cycle
    from distillschedule.training.policy import CyclePolicy
    from distillschedule.training.checkpoint import rng_state
    assets, sources = make_case(tmp_path, gan_start=500)
    prepared = prepare_inputs(sources, tmp_path / "prepared")
    config = request(tmp_path / "public", assets, prepared, 1)
    components, dataset, algorithm, _ = construct(config)
    owner = components.state["native_update"]
    before = owner.state_dict()
    api = original()
    teacher = tmp_path / "original-teacher"
    teacher.mkdir()
    shutil.copyfile(tmp_path / "full-base.safetensors", teacher / "model.safetensors")
    shutil.copyfile(tmp_path / "architecture.json", teacher / "config.json")
    api.training.seed_all(777)
    model, _, _ = api.training.load_teacher(teacher, "cpu", torch.float32)
    adapter = api.dmd2.H3Adapter(model, deepcopy(owner.config))
    head = api.dmd2.VelocityDiscriminator(adapter.config)
    optimizers, parameters = api.training.make_optimizers(adapter, head, adapter.config)
    def step(role, loss):
        api.training.optimizer_step(role, loss, optimizers, parameters, adapter.config)
    trainer = api.dmd2.DMD2Trainer(adapter, head, adapter.config, step)
    equal_tree(before["adapters"], api.training.adapter_state(adapter))
    equal_tree(before["heads"], head.state_dict())
    equal_tree(before["rng"], rng_state())
    frozen = {n: p.detach().clone() for n, p in owner.adapter.named_parameters() if ".adapters." not in n}
    row = decode_row(dataset.sample(0), absolute_step=0, geometry=owner.assets["geometry"])
    raw_before = deepcopy(row["raw"])
    inputs, negative = api.dmd2.prepare_batch(row["raw"])
    expected = trainer.update((inputs["input_latents"], inputs["audio_input_latents"]), inputs, negative)
    native_rng = rng_state()
    workload = build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
    CyclePolicy(config.execution).run(workload, tmp_path / "policy")
    actual = owner.state_dict()
    equal_tree(expected, actual["last_metrics"])
    equal_tree(api.training.adapter_state(adapter), actual["adapters"])
    equal_tree(head.state_dict(), actual["heads"])
    equal_tree(native_rng, actual["rng"])
    equal_tree(frozen, {n: p.detach() for n, p in owner.adapter.named_parameters() if ".adapters." not in n})
    equal_tree(raw_before, row["raw"])
    for role in optimizers:
        equal_tree(optimizers[role].state_dict(), owner.optimizers[role].state_dict())


def test_factory_consumes_advanced_caller_rng_without_reseeding(tmp_path):
    from distillschedule.training.h3_cached_public import build_model
    from distillschedule.training.h3_cached_assets import admit_assets
    from distillschedule.training.checkpoint import rng_state, restore_rng
    assets, _ = make_case(tmp_path)
    teacher = tmp_path / "original-teacher"
    teacher.mkdir()
    shutil.copyfile(tmp_path / "full-base.safetensors", teacher / "model.safetensors")
    shutil.copyfile(tmp_path / "architecture.json", teacher / "config.json")
    api = original()
    api.training.seed_all(777)
    random.random(); random.random()
    np.random.normal(size=7)
    torch.rand(17)
    advanced = rng_state()
    parameters = dict(asset_manifest=str(assets), device="cpu", seed=777)
    # Pin/config/meta inspection itself must not consume the caller's stream.
    admit_assets(parameters)
    equal_tree(advanced, rng_state())
    model, _, _ = api.training.load_teacher(teacher, "cpu", torch.float32)
    config = api.dmd2.DMD2Config(**json.loads(assets.read_text())["dmd_config"])
    adapter = api.dmd2.H3Adapter(model, config)
    head = api.dmd2.VelocityDiscriminator(config)
    api.training.make_optimizers(adapter, head, config)
    expected_rng = rng_state()
    restore_rng(advanced)
    owner = build_model(parameters).state["native_update"]
    equal_tree(expected_rng, rng_state())
    equal_tree(expected_rng, owner.state_dict()["rng"])
    equal_tree(api.training.adapter_state(adapter), owner.state_dict()["adapters"])
    equal_tree(head.state_dict(), owner.state_dict()["heads"])


@pytest.mark.parametrize("bad", ["scale", "public_binding", "hook", "predict", "optimizer_binding", "step_fn",
                                     "group_alias", "parameter_alias", "role", "grad", "modes", "identity", "defaults"])
def test_live_numeric_drift_refused_before_update(tmp_path, monkeypatch, bad):
    from distillschedule.data.h3_cached_inputs import prepare_inputs
    from distillschedule.training.session import construct, build_cycle
    from distillschedule.training import h3_cached_public as public
    assets, sources = make_case(tmp_path)
    prepared = prepare_inputs(sources, tmp_path / "prepared")
    config = request(tmp_path / "run", assets, prepared, 1)
    components, dataset, algorithm, _ = construct(config)
    owner = components.state["native_update"]
    if bad == "scale": monkeypatch.setattr(owner.adapter.layers[0][1], "scale", 0.5)
    if bad == "public_binding": monkeypatch.setattr(public, "native_update", lambda ctx: {})
    if bad == "hook": owner.adapter.register_forward_pre_hook(lambda *args: None)
    if bad == "predict": monkeypatch.setattr(owner.adapter, "predict", lambda *args: None)
    if bad == "optimizer_binding": monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *args: None)
    if bad == "step_fn": monkeypatch.setattr(owner.trainer, "step_fn", lambda *args: None)
    if bad == "group_alias": owner.optimizers["critic"].param_groups[0]["params"].append(owner.parameters["student"][0])
    if bad == "parameter_alias": owner.parameters["student"][0] = owner.parameters["critic"][0]
    if bad == "role": owner.adapter.layers[0][1].role = "fake_score"
    if bad == "grad": owner.parameters["student"][0].grad = torch.zeros_like(owner.parameters["student"][0])
    if bad == "modes": owner.adapter.train()
    if bad == "identity": owner.identity["scope"] = "changed"
    if bad == "defaults": owner.optimizers["critic"].defaults["eps"] = 0.5
    with pytest.raises(ValueError):
        build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
    assert owner.cycle == 0 and not (tmp_path / "run").exists()


@pytest.mark.parametrize("bad", ["missing_shard", "layout", "dtype", "nonfinite", "capacity", "config", "source", "runtime", "precision"])
def test_assets_strict_before_real_parameter_allocation(tmp_path, monkeypatch, bad):
    from distillschedule.training.h3_cached_public import build_model
    from distillschedule.training.h3_cached_assets import FIXTURE
    from safetensors.torch import load_file
    assets, _ = make_case(tmp_path)
    value = json.loads(assets.read_text())
    if bad in ("layout", "dtype", "nonfinite"):
        weights = load_file(str(tmp_path / "full-base.safetensors"))
        name = next(iter(weights))
        if bad == "layout": weights[name] = weights[name].reshape(-1)[:1]
        if bad == "dtype": weights[name] = weights[name].half()
        if bad == "nonfinite": weights[name].fill_(float("nan"))
        save_file(weights, str(tmp_path / "full-base.safetensors"), metadata=dict(evidence_kind=FIXTURE))
        value["base"]["weights"] = [pin(tmp_path / "full-base.safetensors")]
    if bad == "missing_shard": (tmp_path / "full-base.safetensors").unlink()
    if bad == "capacity": value["budget"]["max_parameters"] = 1
    if bad == "config": value["dmd_config"].pop("critic_updates")
    if bad == "source": value["source"]["manifest_sha256"] = "0" * 64
    if bad == "runtime": value["runtime"]["torch"]["version"] = "not-the-runtime"
    if bad == "precision": value["precision"] = "BF16"
    json_file(assets, value)
    register = torch.nn.Module.register_parameter
    allocated = []
    def observe(module, name, parameter):
        if parameter is not None and parameter.device.type != "meta":
            allocated.append(name)
        return register(module, name, parameter)
    monkeypatch.setattr(torch.nn.Module, "register_parameter", observe)
    with pytest.raises(ValueError):
        build_model(dict(asset_manifest=str(assets), device="cpu", seed=777))
    assert not allocated


def test_corrupt_common_restore_leaves_all_live_and_outer_rng_unchanged(tmp_path):
    from distillschedule.data.h3_cached_inputs import prepare_inputs
    from distillschedule.training.session import train, construct
    from distillschedule.training.checkpoint import restore_checkpoint, rng_state
    from distillschedule.workloads.best_known.registry import file_sha256
    assets, sources = make_case(tmp_path)
    prepared = prepare_inputs(sources, tmp_path / "prepared")
    result = train(request(tmp_path / "run", assets, prepared, 2))
    components, *_ = construct(request(tmp_path / "fresh", assets, prepared, 3))
    owner = components.state["native_update"]
    initial, outer = owner.state_dict(), rng_state()
    for corruption in ("marker", "missing_shard", "source", "data", "topology", "progress", "version", "slot_dtype"):
        target = tmp_path / ("corrupt-" + corruption)
        shutil.copytree(result["checkpoint"], target)
        manifest = json.loads((target / "manifest.json").read_text())
        payload = torch.load(target / "state.pt", weights_only=True)
        if corruption == "marker": (target / "_SUCCESS").unlink()
        elif corruption == "missing_shard": (target / "state.pt").unlink()
        elif corruption == "source": manifest["identity"]["framework"] = "0" * 64
        elif corruption == "data": payload["state"]["native_update"]["value"]["consumed"][0]["identity"] = "0" * 64
        elif corruption == "topology": manifest["topology"]["world_size"] = 2
        elif corruption == "progress": manifest["cursor"] = payload["cursor"] = 1
        elif corruption == "version": manifest["versions"] = payload["versions"] = {"native_update": 1}
        elif corruption == "slot_dtype":
            state = payload["state"]["native_update"]["value"]["optimizers"]["student"]["state"]
            slot = next(iter(state.values())); slot["exp_avg"] = slot["exp_avg"].half()
        if corruption not in ("marker", "missing_shard"):
            torch.save(payload, target / "state.pt")
            manifest["files"]["state.pt"] = file_sha256(target / "state.pt")
            json_file(target / "manifest.json", manifest)
            (target / "_SUCCESS").write_text(file_sha256(target / "manifest.json") + "\n")
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            restore_checkpoint(target, components.state, identity=result["identity"])
        equal_tree(initial, owner.state_dict())
        equal_tree(outer, rng_state())


def test_public_discovery_imports_no_torch_or_heavy_models():
    result = subprocess.run([sys.executable, "-B", "-c", """
import sys
import distillschedule.training.h3_cached_public
import distillschedule.training.h3_cached_assets
import distillschedule.data.h3_cached_inputs
assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)
assert not any(n.split('.')[0] in ('diffsynth','diffusers','peft','transformers') for n in sys.modules)
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bad", ["auto", "epochs", "distributed", "cuda", "algorithm"])
def test_unsupported_request_refused_before_output(tmp_path, bad):
    from distillschedule.data.h3_cached_inputs import prepare_inputs
    from distillschedule.training.session import train
    assets, sources = make_case(tmp_path)
    prepared = prepare_inputs(sources, tmp_path / "prepared")
    overrides = dict(auto=dict(execution=dict(mode="auto", memory_limit_bytes=100000)),
        epochs=dict(data=dict(sampling=dict(kind="epochs", epochs=1, shuffle=False))),
        distributed=dict(trainer=dict(distributed=dict(backend="gloo"))),
        cuda=dict(trainer=dict(device="cuda:0")), algorithm=dict(algorithm=dict(parameters={"steps": 1})))
    output = tmp_path / "refused"
    with pytest.raises((ValueError, RuntimeError)):
        train(request(output, assets, prepared, 1, **overrides[bad]))
    assert not output.exists()


def test_owner_tamper_stages_all_before_live_mutation(tmp_path):
    from distillschedule.data.h3_cached_inputs import prepare_inputs
    from distillschedule.training.session import train, construct
    from distillschedule.training.checkpoint import save_checkpoint
    assets, sources = make_case(tmp_path)
    prepared = prepare_inputs(sources, tmp_path / "prepared")
    result = train(request(tmp_path / "run", assets, prepared, 3))
    _, saved = read_owner(result["checkpoint"], result["identity"])
    components, *_ = construct(request(tmp_path / "fresh", assets, prepared, 3))
    owner = components.state["native_update"]
    initial = owner.state_dict()
    mutations = []
    for key in ("generator_step", "critic_step"):
        value = deepcopy(saved); value[key] -= 1; mutations.append(value)
    value = deepcopy(saved); next(iter(value["optimizers"]["critic"]["state"].values()))["step"].add_(1); mutations.append(value)
    value = deepcopy(saved); next(iter(value["adapters"].values())).fill_(float("nan")); mutations.append(value)
    value = deepcopy(saved); value["rng"]["torch"] = torch.ones(2); mutations.append(value)
    value = deepcopy(saved); value["consumed"][0]["identity"] = "0" * 64; mutations.append(value)
    value = deepcopy(saved); value["phase"]["roles"][next(iter(value["phase"]["roles"]))] = "teacher"; mutations.append(value)
    value = deepcopy(saved); value["optimizers"]["student"]["groups"][0]["lr"] *= 2; mutations.append(value)
    value = deepcopy(saved); value["optimizers"]["critic"]["defaults"]["eps"] = 0.5; mutations.append(value)
    value = deepcopy(saved); value["adapters"].pop(next(iter(value["adapters"]))); mutations.append(value)
    value = deepcopy(saved); key = next(iter(value["adapters"])); value["adapters"][key] = value["adapters"][key].half(); mutations.append(value)
    value = deepcopy(saved); key = next(iter(value["heads"])); value["heads"][key] = value["heads"][key].reshape(-1)[:1]; mutations.append(value)
    value = deepcopy(saved); value["config"]["audio_loss_weight"] = 0.25; mutations.append(value)
    value = deepcopy(saved); value["schedule"][0][1].add_(0.01); mutations.append(value)
    value = deepcopy(saved); key = next(iter(value["phase"]["training"])); value["phase"]["training"][key] = not value["phase"]["training"][key]; mutations.append(value)
    for value in mutations:
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            owner.load_state_dict(value)
        equal_tree(initial, owner.state_dict())
    owner.load_state_dict(saved)
    equal_tree(saved, owner.state_dict())
    for progress in (dict(cycle=2, cursor=3, versions={"native_update": 3}),
                     dict(cycle=3, cursor=2, versions={"native_update": 3}),
                     dict(cycle=3, cursor=3, versions={"native_update": 2})):
        with pytest.raises(ValueError, match="progress"):
            save_checkpoint(tmp_path / "bad-save", components.state, identity=result["identity"], **progress)
        assert not (tmp_path / "bad-save").exists()


if __name__ == "__main__":
    resume_worker(sys.argv[1])
