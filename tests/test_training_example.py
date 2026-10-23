"""CPU proof for an actual media encoder and real, tiny detached-score training.

This is not a pretrained quality, W1-W6 parity, GPU or performance test.
"""
from copy import deepcopy
import json
from pathlib import Path
import wave

import pytest
import torch

from distillschedule.examples.tiny_av import (
    QUALITY_STATUS, build_algorithm, build_fixture, build_model, decode, encode,
)
from distillschedule.open_workload.torch_backend import Session, candidates, compare
from distillschedule.training.config import load_training_config
from distillschedule.training.plugins import load_callable
from runtime.dmd_boundary import (
    add_noise, critic_denoising_loss, detached_dmd_surrogate_loss,
    dmd_input_gradient,
)


def media_record():
    return {"id": "real-media", "video": torch.arange(3 * 6 * 6 * 3).remainder(256)
            .to(torch.uint8).reshape(3, 6, 6, 3),
            "audio": {"samples": torch.linspace(-.7, .7, 80).reshape(40, 2),
                      "sample_rate": 8000},
            "text": "A small moving square with a tone.", "metadata": {}}


def copy_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (torch.nn.Module, torch.optim.Optimizer)):
        return copy_state(value.state_dict())
    if isinstance(value, dict):
        return {str(key): copy_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [copy_state(item) for item in value]
    return value


def independent_cycle(workload):
    """No builder hooks, Program calls or Session in the numerical oracle."""
    state, feeds = deepcopy((dict(workload.state), dict(workload.feeds)))
    student, fake, teacher = (state[key] for key in ("student", "fake", "teacher"))
    generated = student(feeds["latent"], feeds["condition"])
    sample = generated.detach()
    time = feeds["timestep"].float() / feeds["schedule_length"]
    with torch.no_grad():
        noisy = add_noise(sample, feeds["student_noise"], feeds["timestep"],
                          feeds["sigmas"], feeds["timesteps"])
        gradient, _ = dmd_input_gradient(
            estimated_clean=sample, noisy_sample=noisy, timestep=feeds["timestep"],
            sigmas=feeds["sigmas"], timesteps=feeds["timesteps"],
            fake_flow_cond=fake(noisy, feeds["condition"], time),
            real_flow_cond=teacher(noisy, feeds["condition"], time),
            real_flow_uncond=teacher(noisy, torch.zeros_like(feeds["condition"]), time),
            real_guidance_scale=float(feeds["guidance_scale"]),
        )
    loss = detached_dmd_surrogate_loss(generated, gradient)
    outputs = {"dmd.c0.student_loss": loss.detach()}
    loss.backward()
    # This must be a genuine differentiable Student path, not a detached scalar.
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in student.parameters())
    assert all(parameter.grad is None for role in (fake, teacher) for parameter in role.parameters())
    state["student_optimizer"].step()
    state["student_optimizer"].zero_grad(set_to_none=True)
    for index, epsilon in enumerate(feeds["fake_noise"]):
        with torch.no_grad():
            target = (student(feeds["latent"], feeds["condition"])
                      if workload.program.parameters["fake_sample_policy"] == "fresh_student"
                      else sample)
            noisy = add_noise(target, epsilon, feeds["timestep"], feeds["sigmas"], feeds["timesteps"])
        loss = critic_denoising_loss(fake(noisy, feeds["condition"], time), epsilon, target)
        outputs[f"dmd.c0.f{index}.loss"] = loss.detach()
        loss.backward()
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in fake.parameters())
        assert all(parameter.grad is None for parameter in teacher.parameters())
        state["fake_optimizer"].step()
        state["fake_optimizer"].zero_grad(set_to_none=True)
    return copy_state(state), copy_state(outputs)


def test_encoder_is_deterministic_and_uses_all_three_real_modalities():
    record = media_record()
    original = deepcopy(record)
    random_before = torch.random.get_rng_state().clone()
    feeds = encode(record, {})
    assert set(feeds) == {"latent", "condition"}
    assert feeds["latent"].shape == (1, 4, 2, 4, 4)
    assert feeds["condition"].shape == (1, 4, 1, 1, 1)
    assert all(value.dtype == torch.float32 and not value.requires_grad
               and torch.isfinite(value).all() for value in feeds.values())
    assert compare(feeds, encode(record, {}))["status"] == "PASS"
    assert torch.equal(random_before, torch.random.get_rng_state())
    assert compare(original, record)["status"] == "PASS"
    for modality in ("video", "audio", "text"):
        changed = deepcopy(record)
        if modality == "video":
            changed["video"] = torch.zeros_like(record["video"])
        elif modality == "audio":
            changed["audio"]["samples"].zero_()
        else:
            changed["text"] = "zzzzzzzz"
        assert compare(feeds, encode(changed, {}))["status"] != "PASS", modality


@pytest.mark.parametrize("change", [
    lambda record: record.update(video=torch.zeros(2, 4, 4, 3)),
    lambda record: record.update(video=torch.zeros(0, 4, 4, 3, dtype=torch.uint8)),
    lambda record: record["audio"].update(samples=torch.tensor([[float("nan")]])),
    lambda record: record["audio"].update(sample_rate=0),
    lambda record: record.update(text=None),
])
def test_encoder_rejects_invalid_decoded_media(change):
    record = media_record()
    change(record)
    with pytest.raises(ValueError):
        encode(record, {})


def test_model_factory_freezes_teacher_and_preserves_global_rng():
    before = torch.random.get_rng_state().clone()
    components = build_model({"device": "cpu", "seed": 7})
    assert components.validate() is components
    assert torch.equal(before, torch.random.get_rng_state())
    assert components.metadata["quality_status"] == QUALITY_STATUS
    assert components.metadata["pretrained"] is False
    assert components.immutable_modules == ("teacher",)
    for role in ("student", "fake"):
        assert isinstance(components.state[role + "_optimizer"], torch.optim.AdamW)
        assert all(parameter.requires_grad for parameter in components.state[role].parameters())
    assert not components.state["teacher"].training
    assert all(not parameter.requires_grad for parameter in components.state["teacher"].parameters())
    again = build_model({"device": "cpu", "seed": 7})
    assert compare(copy_state(components.state), copy_state(again.state))["status"] == "PASS"


@pytest.mark.parametrize("policy", ["reuse_student", "fresh_student"])
def test_real_autograd_and_adamw_updates_match_independent_dmd_cycle(policy):
    components = build_model({"device": "cpu", "seed": 7})
    parameters = {"fake_updates": 2, "fake_sample_policy": policy,
                  "schedule_steps": 16, "guidance_scale": 1.5}
    feeds = encode(media_record(), {})
    before_rng = torch.random.get_rng_state().clone()
    workload = build_algorithm(components, feeds, parameters, cycle=3, seed=7)
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert workload.program.parameters["cycles"] == 1
    assert workload.program.parameters["fake_updates"] == 2
    expected_state, expected_outputs = independent_cycle(workload)
    options = candidates(workload)["candidates"]
    full_checkpoint = next(option for option in options
                           if set(option["checkpoint_bindings"]) == {"student_forward", "fake_forward"})
    for option in (options[0], full_checkpoint):
        session = Session(workload, option)
        result = session.result_state(session.run())
        assert compare(expected_state, result["state"], rtol=2e-6, atol=2e-7)["status"] == "PASS"
        assert compare(expected_outputs, result["outputs"], rtol=2e-6, atol=2e-7)["status"] == "PASS"
        for role in ("student", "fake"):
            assert any(not torch.equal(value, result["state"][role][key])
                       for key, value in workload.state[role].state_dict().items())
            assert result["state"][role + "_optimizer"]["state"]
        assert compare(copy_state(workload.state["teacher"]), result["state"]["teacher"])["status"] == "PASS"
        assert all(value is None for gradients in result["grads"].values() for value in gradients.values())
        assert session.receipt["gpu_execution"] == "NOT_RUN"
    assert set(workload.feeds) - set(feeds) == {
        "student_noise", "fake_noise", "timestep", "sigmas", "timesteps", "guidance_scale", "schedule_length"}
    assert compare(feeds, {key: workload.feeds[key] for key in feeds})["status"] == "PASS"


def test_generation_and_decode_are_media_shaped_not_quality_claims():
    components = build_model({"device": "cpu", "seed": 7})
    feeds = encode(media_record(), {})
    with torch.no_grad():
        generated = components.generate(components.state, feeds, {})
    decoded = decode(generated, {})
    assert generated.shape == feeds["latent"].shape
    assert decoded["video"].shape == (2, 4, 4, 3)
    assert decoded["video"].dtype == torch.uint8
    assert decoded["audio_features"].shape == (2,)  # Features, not a fabricated reconstructed waveform.
    assert decoded["quality_status"] == QUALITY_STATUS


def test_fixture_builder_writes_actual_frames_wav_text_and_public_config(tmp_path):
    output = tmp_path / "tiny-av"
    files = build_fixture(output, samples=6)
    raw = json.loads(Path(files["raw_manifest"]).read_text())
    assert raw["schema_version"] == "distillschedule.data.raw.v1"
    assert raw["encoder"]["factory"] == "distillschedule.examples.tiny_av:encode"
    assert len(raw["records"]) == 6
    for record in raw["records"]:
        root = Path(files["raw_manifest"]).parent
        frames = sorted((root / record["video"]).glob("*.ppm"))
        assert len(frames) >= 2
        assert frames[0].read_bytes().startswith(b"P6\n8 8\n255\n")
        with wave.open(str(root / record["audio"]), "rb") as wav:
            assert wav.getframerate() == 8000 and wav.getnframes() > 0
            assert any(wav.readframes(wav.getnframes()))
        assert (root / record["text"]["path"]).read_text().strip()
    config = load_training_config(files["training_config"])
    assert config.data["prepared"] == str(output / "prepared" / "manifest.json")
    assert config.trainer["device"] == "cpu"
    assert config.trainer["cycles"] == 4 and config.trainer["checkpoint_every"] == 2
    assert "/home/" not in Path(files["training_config"]).read_text()
    with pytest.raises(FileExistsError):
        build_fixture(output, samples=6)


def test_external_newmodel_template_uses_the_same_open_workload_without_core_edits():
    source = Path(__file__).resolve().parents[1] / "examples" / "framework" / "newmodel.py"
    model_factory, _ = load_callable(str(source) + ":build_model")
    algorithm_factory, _ = load_callable(str(source) + ":build_algorithm")
    components = model_factory({"device": "cpu", "seed": 7})
    workload = algorithm_factory(components, encode(media_record(), {}),
                                 {"fake_updates": 1, "fake_sample_policy": "reuse_student"}, 0, 7)
    session = Session(workload, candidates(workload)["candidates"][0])
    session.run()
    assert session.receipt["status"] == "EXECUTED_OPEN_PROGRAM"
    assert components.metadata["external_template"] is True
